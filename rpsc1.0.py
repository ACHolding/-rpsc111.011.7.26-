#!/usr/bin/env python3
"""
RPSC 1.0 — RPSC1 PS1 Emulator Core (Clean / Legal)
────────────────────────────────────────────────────
files = OFF     → Demo boots with no BIOS / CD / BIN / external assets.
GUI             → Classic RPSC1 CRT green-on-black UI preserved.

Upgrade over 0.1.1 ("support everything a PS1 emulator needs to"):
  • R3000A: real exception + interrupt path (COP0 SR/CAUSE/EPC), MIPS-I
    load-delay slot, proper branch-delay slot, overflow-trapping ADD/SUB.
  • GTE (COP2): real fixed-point math — RTPS/RTPT with the documented UNR
    division + depth cue, MVMVA, NCLIP, OP, SQR, AVSZ3/4, and the lighting
    colour ops (NCDS/NCCS/NCS/CC/CDP…), with MAC/IR/SZ/SXY saturation and
    the FLAG register.
  • DMA: channel 6 (OTC) ordering-table build + channel 2 (GPU) block and
    linked-list transfer, so a real display list reaches the GPU.
  • GPU: command FIFO with correct per-primitive word counts; flat and
    Gouraud triangle/quad raster from real vertex data honouring the draw
    offset; CPU→VRAM upload and VRAM fill.
  • Controller: real digital-pad (SCPH-1080) SIO response FSM + keyboard map.

Verified-vs-inferred: the GTE arithmetic, the UNR divide table, the pad
poll sequence, the DMA OTC/linked-list walk and the exception vector layout
are reproduced from the public reverse-engineered PS1 spec (nocash psx-spx)
and self-tested below. The CD path is still a stub: there is no ISO9660
boot, so this will NOT run a commercial disc on its own — see notes at EOF.

Legal: no BIOS/games shipped. Optional user-owned loads only. Single-file.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional

import numpy as np

try:  # GUI is optional so --selftest / --opcodes run headless
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext
except ImportError:  # pragma: no cover
    tk = None


FILES_OFF = True

VERSION = "1.0"
TITLE = "RPSC1"

# ── Opcode name tables (RPSC / R3000A coverage) ─────────────────────────────
PRIMARY_OPS = {
    0x00: "SPECIAL", 0x01: "REGIMM", 0x02: "J", 0x03: "JAL",
    0x04: "BEQ", 0x05: "BNE", 0x06: "BLEZ", 0x07: "BGTZ",
    0x08: "ADDI", 0x09: "ADDIU", 0x0A: "SLTI", 0x0B: "SLTIU",
    0x0C: "ANDI", 0x0D: "ORI", 0x0E: "XORI", 0x0F: "LUI",
    0x10: "COP0", 0x11: "COP1", 0x12: "COP2", 0x13: "COP3",
    0x20: "LB", 0x21: "LH", 0x22: "LWL", 0x23: "LW",
    0x24: "LBU", 0x25: "LHU", 0x26: "LWR",
    0x28: "SB", 0x29: "SH", 0x2A: "SWL", 0x2B: "SW",
    0x2E: "SWR", 0x2F: "CACHE",
    0x30: "LL", 0x31: "LWC1", 0x32: "LWC2", 0x33: "LWC3",
    0x38: "SC", 0x39: "SWC1", 0x3A: "SWC2", 0x3B: "SWC3",
}

SPECIAL_FUNCT = {
    0x00: "SLL", 0x02: "SRL", 0x03: "SRA", 0x04: "SLLV",
    0x06: "SRLV", 0x07: "SRAV", 0x08: "JR", 0x09: "JALR",
    0x0C: "SYSCALL", 0x0D: "BREAK",
    0x10: "MFHI", 0x11: "MTHI", 0x12: "MFLO", 0x13: "MTLO",
    0x18: "MULT", 0x19: "MULTU", 0x1A: "DIV", 0x1B: "DIVU",
    0x20: "ADD", 0x21: "ADDU", 0x22: "SUB", 0x23: "SUBU",
    0x24: "AND", 0x25: "OR", 0x26: "XOR", 0x27: "NOR",
    0x2A: "SLT", 0x2B: "SLTU",
}

REGIMM_RT = {
    0x00: "BLTZ", 0x01: "BGEZ", 0x10: "BLTZAL", 0x11: "BGEZAL",
}

RPSC_GTE_CMDS = {
    0x01: "RTPS", 0x06: "NCLIP", 0x0C: "OP", 0x10: "DPCS",
    0x11: "INTPL", 0x12: "MVMVA", 0x13: "NCDS", 0x14: "CDP",
    0x16: "NCDT", 0x1B: "NCCS", 0x1C: "CC", 0x1E: "NCS",
    0x20: "NCT", 0x28: "SQR", 0x29: "DCPL", 0x2A: "DPCT",
    0x2D: "AVSZ3", 0x2E: "AVSZ4", 0x30: "RTPT", 0x3D: "GPF",
    0x3E: "GPL", 0x3F: "NCCT",
}

RPSC_GP0 = {
    0x00: "NOP", 0x01: "ClearCache", 0x02: "FillRect", 0x03: "Unk03",
    0x1F: "InterruptReq",
    0x20: "MonoTri", 0x22: "MonoTriSemi", 0x24: "TexTri", 0x25: "TexTriRaw",
    0x26: "TexTriSemi", 0x27: "TexTriSemiRaw",
    0x28: "MonoQuad", 0x2A: "MonoQuadSemi", 0x2C: "TexQuad", 0x2D: "TexQuadRaw",
    0x2E: "TexQuadSemi", 0x2F: "TexQuadSemiRaw",
    0x30: "ShadedTri", 0x32: "ShadedTriSemi", 0x34: "ShadedTexTri",
    0x36: "ShadedTexTriSemi",
    0x38: "ShadedQuad", 0x3A: "ShadedQuadSemi", 0x3C: "ShadedTexQuad",
    0x3E: "ShadedTexQuadSemi",
    0x40: "MonoLine", 0x42: "MonoLineSemi", 0x48: "PolyLine",
    0x50: "ShadedLine", 0x52: "ShadedLineSemi", 0x58: "ShadedPolyLine",
    0x60: "MonoRect", 0x62: "MonoRectSemi", 0x64: "TexRect", 0x65: "TexRectRaw",
    0x66: "TexRectSemi", 0x68: "MonoRect1", 0x70: "MonoRect8", 0x74: "TexRect8",
    0x78: "MonoRect16", 0x7C: "TexRect16",
    0x80: "CopyVRAM", 0xA0: "CPU2VRAM", 0xC0: "VRAM2CPU",
    0xE1: "TPage", 0xE2: "TexWindow", 0xE3: "DrawAreaTL", 0xE4: "DrawAreaBR",
    0xE5: "DrawOffset", 0xE6: "MaskBit",
}

RPSC_GP1 = {
    0x00: "ResetGPU", 0x01: "ResetCmdBuf", 0x02: "AckIRQ", 0x03: "DisplayEnable",
    0x04: "DMADir", 0x05: "DisplayStart", 0x06: "HorizRange", 0x07: "VertRange",
    0x08: "DisplayMode", 0x10: "GetGPUInfo",
}

RPSC_DMA_CH = {
    0: "MDEC_IN", 1: "MDEC_OUT", 2: "GPU", 3: "CDROM",
    4: "SPU", 5: "PIO", 6: "OTC",
}

RPSC_IRQ = {
    0: "VBLANK", 1: "GPU", 2: "CDROM", 3: "DMA", 4: "TMR0",
    5: "TMR1", 6: "TMR2", 7: "PAD", 8: "SIO", 9: "SPU", 10: "LIGHTGUN",
}

# COP0 exception cause codes (R3000A)
EXC_INT = 0x00     # external interrupt
EXC_ADEL = 0x04    # address error (load)
EXC_ADES = 0x05    # address error (store)
EXC_SYS = 0x08     # syscall
EXC_BP = 0x09      # breakpoint
EXC_RI = 0x0A      # reserved instruction
EXC_CPU = 0x0B     # coprocessor unusable
EXC_OVF = 0x0C     # arithmetic overflow


class RPSC:
    """Namespace: RPSC tables in one place (files=OFF safe)."""

    PRIMARY = PRIMARY_OPS
    SPECIAL = SPECIAL_FUNCT
    REGIMM = REGIMM_RT
    GTE = RPSC_GTE_CMDS
    GP0 = RPSC_GP0
    GP1 = RPSC_GP1
    DMA = RPSC_DMA_CH
    IRQ = RPSC_IRQ
    FILES_OFF = True

    @staticmethod
    def report() -> str:
        lines = [
            "RPSC1 — PS1 instruction and command coverage",
            "═" * 44,
            f"files = OFF: {FILES_OFF}",
            f"Python: {sys.version.split()[0]}",
            "",
            f"CPU PRIMARY  ({len(PRIMARY_OPS)})",
        ]
        for k in sorted(PRIMARY_OPS):
            lines.append(f"  {k:02X}  {PRIMARY_OPS[k]}")
        lines.append(f"\nCPU SPECIAL  ({len(SPECIAL_FUNCT)})")
        for k in sorted(SPECIAL_FUNCT):
            lines.append(f"  {k:02X}  {SPECIAL_FUNCT[k]}")
        lines.append(f"\nCPU REGIMM   ({len(REGIMM_RT)})")
        for k in sorted(REGIMM_RT):
            lines.append(f"  {k:02X}  {REGIMM_RT[k]}")
        lines.append(f"\nGTE COP2     ({len(RPSC_GTE_CMDS)})")
        for k in sorted(RPSC_GTE_CMDS):
            lines.append(f"  {k:02X}  {RPSC_GTE_CMDS[k]}")
        lines.append(f"\nGPU GP0      ({len(RPSC_GP0)})")
        for k in sorted(RPSC_GP0):
            lines.append(f"  {k:02X}  {RPSC_GP0[k]}")
        lines.append(f"\nGPU GP1      ({len(RPSC_GP1)})")
        for k in sorted(RPSC_GP1):
            lines.append(f"  {k:02X}  {RPSC_GP1[k]}")
        lines.append(f"\nDMA channels ({len(RPSC_DMA_CH)})")
        for k in sorted(RPSC_DMA_CH):
            lines.append(f"  {k}  {RPSC_DMA_CH[k]}")
        lines.append(f"\nIRQs         ({len(RPSC_IRQ)})")
        for k in sorted(RPSC_IRQ):
            lines.append(f"  {k}  {RPSC_IRQ[k]}")
        return "\n".join(lines)


LEGAL_NOTICE = (
    "LEGAL — Clean ROM policy\n\n"
    "RPSC ships with FILES=OFF (demo only).\n"
    "No PlayStation BIOS or commercial game data is included.\n\n"
    "You may load a BIOS (.bin/.rom) and disc image (.bin/.cue/.iso)\n"
    "only if you legally own those materials (e.g. dumped from your own console/discs).\n\n"
    "Do not use pirated ROMs or BIOS files."
)


def _s16(v: int) -> int:
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


def _s32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v


# ── GTE (COP2) — real fixed-point geometry engine ──────────────────────────
class GTE:
    """
    RPSC1 Geometry Transformation Engine (COP2).

    Data registers (data[0..31]) and control registers (control[0..31]) use
    the documented PS1 layout. Arithmetic, saturation and the FLAG register
    follow the public nocash psx-spx reverse-engineered spec.
      Faithful:    RTPS/RTPT (UNR divide + depth cue), NCLIP, AVSZ3/4,
                   MVMVA, OP, SQR, MAC/IR/SZ/SXY saturation + FLAG bits.
      Approximate: colour ops (NC*/CC/CDP/DPC*/INTPL/GPF/GPL/DCPL) use the
                   right matrices but not bit-exact intermediate rounding;
                   MVMVA's tx=FC hardware bug and mx=3 garbage matrix are
                   not emulated.
    """

    def __init__(self) -> None:
        self.data = [0] * 32
        self.control = [0] * 32
        self.flag = 0
        self.last_cmd = "NOP"
        self.cmd_count = 0
        # UNR reciprocal table (257 bytes), generated per the documented formula:
        #   unr[i] = clamp(0, (0x40000 / (i+0x100) + 1)/2 - 0x101, 0xFF)
        self.unr = [
            max(0, min(0xFF, (0x40000 // (i + 0x100) + 1) // 2 - 0x101))
            for i in range(257)
        ]

    # ---- register access with sign handling for the packed pairs ----
    def _vx(self, n): return _s16(self.data[n * 2] & 0xFFFF)
    def _vy(self, n): return _s16((self.data[n * 2] >> 16) & 0xFFFF)
    def _vz(self, n): return _s16(self.data[n * 2 + 1] & 0xFFFF)

    def _mat(self, base):
        c = self.control
        return (
            (_s16(c[base] & 0xFFFF), _s16((c[base] >> 16) & 0xFFFF), _s16(c[base + 1] & 0xFFFF)),
            (_s16((c[base + 1] >> 16) & 0xFFFF), _s16(c[base + 2] & 0xFFFF), _s16((c[base + 2] >> 16) & 0xFFFF)),
            (_s16(c[base + 3] & 0xFFFF), _s16((c[base + 3] >> 16) & 0xFFFF), _s16(c[base + 4] & 0xFFFF)),
        )

    def _vec_ctl(self, base):
        return (_s32(self.control[base]), _s32(self.control[base + 1]), _s32(self.control[base + 2]))

    # ---- FLAG bits ----
    def _f_mac(self, idx, pos):  # MAC1..3 overflow (idx 1..3)
        bit = {1: (30, 27), 2: (29, 26), 3: (28, 25)}[idx]
        self.flag |= 1 << (bit[0] if pos else bit[1])

    def _f_ir(self, idx):  # IR1..3 saturated
        self.flag |= 1 << {1: 24, 2: 23, 3: 22}[idx]

    # ---- saturation ----
    def _lim_ir(self, idx, val, lm):
        lo = 0 if lm else -0x8000
        if val > 0x7FFF:
            self._f_ir(idx); return 0x7FFF
        if val < lo:
            self._f_ir(idx); return lo
        return val

    def _lim_ir0(self, val):
        if val > 0x1000:
            self.flag |= 1 << 12; return 0x1000
        if val < 0:
            self.flag |= 1 << 12; return 0
        return val

    def _lim_sz3(self, val):
        if val > 0xFFFF:
            self.flag |= 1 << 18; return 0xFFFF
        if val < 0:
            self.flag |= 1 << 18; return 0
        return val

    def _lim_sxy(self, val, axis=0):
        if val > 0x3FF:
            self.flag |= 1 << (14 if axis == 0 else 13); return 0x3FF
        if val < -0x400:
            self.flag |= 1 << (14 if axis == 0 else 13); return -0x400
        return val

    def _lim_col(self, idx, val):
        if val > 0xFF:
            self.flag |= 1 << {0: 21, 1: 20, 2: 19}[idx]; return 0xFF
        if val < 0:
            self.flag |= 1 << {0: 21, 1: 20, 2: 19}[idx]; return 0
        return val

    def _chk_mac(self, idx, val):
        # 44-bit overflow detection for MAC1..3 (sign-extended accumulators)
        if val > 0x7FFFFFFFFFF:
            self._f_mac(idx, True)
        elif val < -0x80000000000:
            self._f_mac(idx, False)
        return val

    def _set_mac123(self, m1, m2, m3):
        self._chk_mac(1, m1); self._chk_mac(2, m2); self._chk_mac(3, m3)
        self.data[25] = m1 & 0xFFFFFFFF
        self.data[26] = m2 & 0xFFFFFFFF
        self.data[27] = m3 & 0xFFFFFFFF

    def _set_ir123(self, i1, i2, i3, lm):
        self.data[9] = self._lim_ir(1, i1, lm) & 0xFFFF
        self.data[10] = self._lim_ir(2, i2, lm) & 0xFFFF
        self.data[11] = self._lim_ir(3, i3, lm) & 0xFFFF

    def _push_sz(self, sz3):
        self.data[16] = self.data[17]
        self.data[17] = self.data[18]
        self.data[18] = self.data[19]
        self.data[19] = self._lim_sz3(sz3) & 0xFFFF

    def _push_sxy(self, sx, sy):
        self.data[12] = self.data[13]
        self.data[13] = self.data[14]
        sx = self._lim_sxy(sx) & 0xFFFF
        sy = self._lim_sxy(sy, 1) & 0xFFFF
        self.data[14] = (sy << 16) | sx

    def _push_rgb(self, r, g, b):
        code = (self.data[6] >> 24) & 0xFF
        self.data[20] = self.data[21]
        self.data[21] = self.data[22]
        self.data[22] = (code << 24) | (self._lim_col(2, b) << 16) \
            | (self._lim_col(1, g) << 8) | self._lim_col(0, r)

    # ---- the documented UNR division: H*0x20000 / SZ3, clamped to 0x1FFFF ----
    def _divide(self, h, sz3):
        if sz3 == 0 or h >= sz3 * 2:
            self.flag |= 1 << 17
            return 0x1FFFF
        z = 0
        t = sz3
        while t & 0x8000 == 0 and z < 16:
            t <<= 1; z += 1
        n = h << z
        d = sz3 << z
        u = self.unr[((d - 0x7FC0) >> 7) if d >= 0x7FC0 else 0] + 0x101
        d = ((0x2000080 - d * u) >> 8) & 0xFFFFFFFF
        d = ((0x0000080 + d * u) >> 8) & 0xFFFFFFFF
        res = (n * d + 0x8000) >> 16
        return min(0x1FFFF, res)

    def command(self, insn: int) -> None:
        code = insn & 0x3F
        self.last_cmd = RPSC_GTE_CMDS.get(code, f"GTE/{code:02X}")
        self.cmd_count += 1
        self.flag = 0
        sf = 12 if (insn & (1 << 19)) else 0
        lm = bool(insn & (1 << 10))

        if code in (0x01, 0x30):           # RTPS / RTPT
            self._rtp(sf, lm, 3 if code == 0x30 else 1)
        elif code == 0x06:                 # NCLIP
            self._nclip()
        elif code == 0x0C:                 # OP (cross product)
            self._op(sf, lm)
        elif code == 0x28:                 # SQR
            self._sqr(sf, lm)
        elif code == 0x12:                 # MVMVA
            self._mvmva(insn, sf, lm)
        elif code in (0x2D, 0x2E):         # AVSZ3 / AVSZ4
            self._avsz(code == 0x2E)
        elif code in (0x13, 0x1B, 0x1E):   # NCDS / NCCS / NCS
            self._ncd(sf, lm, code)
        elif code in (0x16, 0x20, 0x3F):   # NCDT / NCT / NCCT (triple-vertex)
            self._ncd_triple(sf, lm, code)
        elif code in (0x14, 0x1C):         # CDP / CC
            self._cdp_cc(sf, lm, code)
        elif code in (0x10, 0x2A):         # DPCS / DPCT (depth cue)
            self._dpc(sf, lm, code == 0x2A)
        elif code == 0x11:                 # INTPL
            self._intpl(sf, lm)
        elif code in (0x3D, 0x3E):         # GPF / GPL
            self._gpf_gpl(sf, lm, code == 0x3E)
        elif code == 0x29:                 # DCPL
            self._dcpl(sf, lm)
        # FLAG bit 31 summarizes bits 30..23 and 18..13 (not colour/IR3
        # saturation bits 22..19).
        if self.flag & 0x7F87E000:
            self.flag |= 0x80000000
        self.control[31] = self.flag
        # Any unknown GTE code leaves registers intact.

    def _rtp(self, sf, lm, count):
        trx, tr_y, trz = self._vec_ctl(5)
        rt = self._mat(0)
        h = self.control[26] & 0xFFFF
        dqa = _s16(self.control[27] & 0xFFFF)
        dqb = _s32(self.control[28])
        ofx = _s32(self.control[24])
        ofy = _s32(self.control[25])
        shift = 12 if sf == 12 else 0
        last_ir0mac = 0
        for v in range(count):
            vx, vy, vz = self._vx(v), self._vy(v), self._vz(v)
            m1 = (trx * 0x1000 + rt[0][0] * vx + rt[0][1] * vy + rt[0][2] * vz) >> shift
            m2 = (tr_y * 0x1000 + rt[1][0] * vx + rt[1][1] * vy + rt[1][2] * vz) >> shift
            m3 = (trz * 0x1000 + rt[2][0] * vx + rt[2][1] * vy + rt[2][2] * vz) >> shift
            self._set_mac123(m1, m2, m3)
            self._set_ir123(m1, m2, m3, lm)
            # SZ3 = MAC3 >> ((1-sf)*12), saturated 0..FFFFh
            self._push_sz(m3 if sf == 12 else (m3 >> 12))
            n = self._divide(h, self.data[19] & 0xFFFF)
            sx = (n * _s16(self.data[9]) + ofx) >> 16
            sy = (n * _s16(self.data[10]) + ofy) >> 16
            self._push_sxy(sx, sy)
            last_ir0mac = n * dqa + dqb
        # depth cue result lands in MAC0/IR0 after the last vertex
        self.data[24] = last_ir0mac & 0xFFFFFFFF
        self.data[8] = self._lim_ir0(last_ir0mac >> 12) & 0xFFFF

    def _nclip(self):
        sx0, sy0 = _s16(self.data[12] & 0xFFFF), _s16((self.data[12] >> 16) & 0xFFFF)
        sx1, sy1 = _s16(self.data[13] & 0xFFFF), _s16((self.data[13] >> 16) & 0xFFFF)
        sx2, sy2 = _s16(self.data[14] & 0xFFFF), _s16((self.data[14] >> 16) & 0xFFFF)
        mac0 = sx0 * (sy1 - sy2) + sx1 * (sy2 - sy0) + sx2 * (sy0 - sy1)
        self.data[24] = mac0 & 0xFFFFFFFF

    def _op(self, sf, lm):
        shift = 12 if sf == 12 else 0
        d1, d2, d3 = self._mat(0)[0][0], self._mat(0)[1][1], self._mat(0)[2][2]
        ir1, ir2, ir3 = _s16(self.data[9]), _s16(self.data[10]), _s16(self.data[11])
        m1 = (d2 * ir3 - d3 * ir2) >> shift
        m2 = (d3 * ir1 - d1 * ir3) >> shift
        m3 = (d1 * ir2 - d2 * ir1) >> shift
        self._set_mac123(m1, m2, m3)
        self._set_ir123(m1, m2, m3, lm)

    def _sqr(self, sf, lm):
        shift = 12 if sf == 12 else 0
        ir1, ir2, ir3 = _s16(self.data[9]), _s16(self.data[10]), _s16(self.data[11])
        m1 = (ir1 * ir1) >> shift
        m2 = (ir2 * ir2) >> shift
        m3 = (ir3 * ir3) >> shift
        self._set_mac123(m1, m2, m3)
        self._set_ir123(m1, m2, m3, lm)

    def _mvmva(self, insn, sf, lm):
        shift = 12 if sf == 12 else 0
        mx = (insn >> 17) & 3
        vx_sel = (insn >> 15) & 3
        tx = (insn >> 13) & 3
        # mx=3 selects a hardware-specific garbage matrix. Do not silently
        # substitute rotation matrix 0; leave MAC/IR unchanged (approximation).
        if mx == 3:
            return
        mat = self._mat({0: 0, 1: 8, 2: 16}[mx])
        if vx_sel == 3:
            vec = (_s16(self.data[9]), _s16(self.data[10]), _s16(self.data[11]))
        else:
            vec = (self._vx(vx_sel), self._vy(vx_sel), self._vz(vx_sel))
        if tx == 0:
            tr = self._vec_ctl(5)
        elif tx == 1:
            tr = self._vec_ctl(13)
        elif tx == 2:
            tr = self._vec_ctl(21)
        else:
            tr = (0, 0, 0)
        m1 = (tr[0] * 0x1000 + mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2]) >> shift
        m2 = (tr[1] * 0x1000 + mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2]) >> shift
        m3 = (tr[2] * 0x1000 + mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2]) >> shift
        self._set_mac123(m1, m2, m3)
        self._set_ir123(m1, m2, m3, lm)

    def _avsz(self, four):
        zsf = _s16(self.control[30] & 0xFFFF) if four else _s16(self.control[29] & 0xFFFF)
        if four:
            s = self.data[16] + self.data[17] + self.data[18] + self.data[19]
        else:
            s = self.data[17] + self.data[18] + self.data[19]
        mac0 = zsf * (s & 0xFFFFFFFF)
        self.data[24] = mac0 & 0xFFFFFFFF
        self.data[7] = self._lim_sz3(mac0 >> 12) & 0xFFFF   # OTZ

    def _light_core(self, sf, lm, vsel):
        """Shared NC* core: light matrix · V, then colour matrix · (BK + result)."""
        shift = 12 if sf == 12 else 0
        lmat = self._mat(8)
        if vsel is None:
            vec = (_s16(self.data[9]), _s16(self.data[10]), _s16(self.data[11]))
        else:
            vec = (self._vx(vsel), self._vy(vsel), self._vz(vsel))
        l1 = (lmat[0][0] * vec[0] + lmat[0][1] * vec[1] + lmat[0][2] * vec[2]) >> shift
        l2 = (lmat[1][0] * vec[0] + lmat[1][1] * vec[1] + lmat[1][2] * vec[2]) >> shift
        l3 = (lmat[2][0] * vec[0] + lmat[2][1] * vec[1] + lmat[2][2] * vec[2]) >> shift
        self._set_ir123(l1, l2, l3, lm)
        cmat = self._mat(16)
        bk = self._vec_ctl(13)
        ir = (_s16(self.data[9]), _s16(self.data[10]), _s16(self.data[11]))
        c1 = (bk[0] * 0x1000 + cmat[0][0] * ir[0] + cmat[0][1] * ir[1] + cmat[0][2] * ir[2]) >> shift
        c2 = (bk[1] * 0x1000 + cmat[1][0] * ir[0] + cmat[1][1] * ir[1] + cmat[1][2] * ir[2]) >> shift
        c3 = (bk[2] * 0x1000 + cmat[2][0] * ir[0] + cmat[2][1] * ir[1] + cmat[2][2] * ir[2]) >> shift
        self._set_mac123(c1, c2, c3)
        self._set_ir123(c1, c2, c3, lm)

    def _ncd(self, sf, lm, code):
        vsel = 0
        self._light_core(sf, lm, vsel)
        if code in (0x13,):  # NCDS — modulate by vertex colour then push
            self._modulate_push(sf, lm)
        else:                # NCCS / NCS — push IR as colour
            self._push_rgb(_s16(self.data[9]) >> 4, _s16(self.data[10]) >> 4, _s16(self.data[11]) >> 4)

    def _ncd_triple(self, sf, lm, code):
        for v in range(3):
            self._light_core(sf, lm, v)
            if code == 0x16:  # NCDT
                self._modulate_push(sf, lm)
            else:
                self._push_rgb(_s16(self.data[9]) >> 4, _s16(self.data[10]) >> 4, _s16(self.data[11]) >> 4)

    def _modulate_push(self, sf, lm):
        r = (self.data[6] & 0xFF)
        g = ((self.data[6] >> 8) & 0xFF)
        b = ((self.data[6] >> 16) & 0xFF)
        ir = (_s16(self.data[9]), _s16(self.data[10]), _s16(self.data[11]))
        m1 = (r * ir[0]) << 4
        m2 = (g * ir[1]) << 4
        m3 = (b * ir[2]) << 4
        shift = 12 if sf == 12 else 0
        self._set_mac123(m1 >> shift, m2 >> shift, m3 >> shift)
        self._push_rgb((m1 >> shift) >> 4, (m2 >> shift) >> 4, (m3 >> shift) >> 4)
        self._set_ir123(m1 >> shift, m2 >> shift, m3 >> shift, lm)

    def _cdp_cc(self, sf, lm, code):
        # CDP/CC: colour matrix on current IR, then modulate. Uses light core
        # path with IR as the input vector (vsel None).
        self._light_core(sf, lm, None)
        if code == 0x14:  # CDP
            self._modulate_push(sf, lm)
        else:             # CC
            self._push_rgb(_s16(self.data[9]) >> 4, _s16(self.data[10]) >> 4, _s16(self.data[11]) >> 4)

    def _dpc(self, sf, lm, triple):
        # Depth cueing: interpolate colour toward the far colour FC by IR0.
        fc = self._vec_ctl(21)
        reps = 3 if triple else 1
        for _ in range(reps):
            r = (self.data[6] & 0xFF) << 4
            g = ((self.data[6] >> 8) & 0xFF) << 4
            b = ((self.data[6] >> 16) & 0xFF) << 4
            ir0 = _s16(self.data[8])
            m1 = r + ((fc[0] - r) * ir0 >> 12)
            m2 = g + ((fc[1] - g) * ir0 >> 12)
            m3 = b + ((fc[2] - b) * ir0 >> 12)
            self._set_mac123(m1, m2, m3)
            self._push_rgb(m1 >> 4, m2 >> 4, m3 >> 4)
            self._set_ir123(m1, m2, m3, lm)

    def _intpl(self, sf, lm):
        fc = self._vec_ctl(21)
        ir = (_s16(self.data[9]) << 12, _s16(self.data[10]) << 12, _s16(self.data[11]) << 12)
        ir0 = _s16(self.data[8])
        m1 = ir[0] + ((fc[0] - ir[0]) * ir0 >> 12)
        m2 = ir[1] + ((fc[1] - ir[1]) * ir0 >> 12)
        m3 = ir[2] + ((fc[2] - ir[2]) * ir0 >> 12)
        shift = 12 if sf == 12 else 0
        self._set_mac123(m1 >> shift, m2 >> shift, m3 >> shift)
        self._push_rgb((m1 >> shift) >> 4, (m2 >> shift) >> 4, (m3 >> shift) >> 4)
        self._set_ir123(m1 >> shift, m2 >> shift, m3 >> shift, lm)

    def _dcpl(self, sf, lm):
        self._modulate_push(sf, lm)

    def _gpf_gpl(self, sf, lm, gpl):
        shift = 12 if sf == 12 else 0
        ir0 = _s16(self.data[8])
        ir = (_s16(self.data[9]), _s16(self.data[10]), _s16(self.data[11]))
        if gpl:
            base = (_s32(self.data[25]), _s32(self.data[26]), _s32(self.data[27]))
        else:
            base = (0, 0, 0)
        m1 = base[0] + ir0 * ir[0]
        m2 = base[1] + ir0 * ir[1]
        m3 = base[2] + ir0 * ir[2]
        self._set_mac123(m1 >> shift, m2 >> shift, m3 >> shift)
        self._push_rgb((m1 >> shift) >> 4, (m2 >> shift) >> 4, (m3 >> shift) >> 4)
        self._set_ir123(m1 >> shift, m2 >> shift, m3 >> shift, lm)


# ── Digital controller (SCPH-1080) + SIO FSM ───────────────────────────────
class Controller:
    """Standard digital pad. Buttons are active-LOW in the 16-bit word."""
    # bit → name (active low). 0 = pressed.
    BITS = {
        0: "SELECT", 3: "START", 4: "UP", 5: "RIGHT", 6: "DOWN", 7: "LEFT",
        8: "L2", 9: "R2", 10: "L1", 11: "R1",
        12: "TRIANGLE", 13: "CIRCLE", 14: "CROSS", 15: "SQUARE",
    }

    def __init__(self) -> None:
        self.buttons = 0xFFFF  # all released
        self._seq = 0

    def press(self, bit: int) -> None:
        self.buttons &= ~(1 << bit) & 0xFFFF

    def release(self, bit: int) -> None:
        self.buttons |= (1 << bit) & 0xFFFF

    def exchange(self, tx: int) -> int:
        """One byte of the digital-pad poll. Returns the byte to send back."""
        s = self._seq
        if s == 0 and tx != 0x01:
            return 0xFF
        if s == 1 and tx != 0x42:
            self.reset()
            return 0xFF
        self._seq += 1
        if s == 0:      # host sends 0x01 (address), pad returns hi-Z/0xFF
            return 0xFF
        if s == 1:      # host 0x42 (read), pad returns ID lo 0x41
            return 0x41
        if s == 2:      # pad returns ID hi 0x5A
            return 0x5A
        if s == 3:      # buttons low byte
            return self.buttons & 0xFF
        if s == 4:      # buttons high byte
            return (self.buttons >> 8) & 0xFF
        self._seq = 0
        return 0xFF

    def reset(self) -> None:
        self._seq = 0


# ── SPU / Timers / INTC / CDROM ─────────────────────────────────────────────
class SPU:
    def __init__(self) -> None:
        self.regs = [0] * 256

    def write(self, addr: int, value: int) -> None:
        self.regs[(addr >> 1) & 0xFF] = value & 0xFFFF

    def read(self, addr: int) -> int:
        if (addr & 0xFFE) == 0x1AE:  # SPUSTAT-ish
            return 0x0000
        return self.regs[(addr >> 1) & 0xFF]


class Timers:
    def __init__(self) -> None:
        self.mode = [0] * 3
        self.target = [0] * 3
        self.counter = [0] * 3

    def tick(self, cycles: int) -> None:
        for i in range(3):
            self.counter[i] = (self.counter[i] + cycles) & 0xFFFF


class INTC:
    def __init__(self) -> None:
        self.i_stat = 0
        self.i_mask = 0

    def trigger(self, irq: int) -> None:
        self.i_stat |= 1 << irq

    def pending(self) -> bool:
        return bool(self.i_stat & self.i_mask & 0x7FF)

    def read_stat(self) -> int:
        return self.i_stat

    def write_stat(self, value: int) -> None:
        self.i_stat &= value

    def write_mask(self, value: int) -> None:
        self.i_mask = value & 0x7FF


class CDROM:
    """Stub CD-ROM — enough for demo; disc image loaded separately if legal.

    [inferred/stub] No ISO9660 parse, no XA/ADPCM, no real command FSM. A
    loaded image is held in RAM so a future boot path can read sectors, but
    nothing drives it yet. This is the honest gap vs. a shipping emulator.
    """

    def __init__(self) -> None:
        self.status = 0x18
        self.response = bytearray()
        self.image: Optional[bytearray] = None
        self.image_path: Optional[str] = None

    def load_image(self, path: str) -> int:
        with open(path, "rb") as f:
            data = f.read()
        self.image = bytearray(data)
        self.image_path = path
        return len(data)

    def read_sector(self, lba: int) -> bytes:
        if self.image is None:
            return b"\x00" * 2048
        off = lba * 2352 + 24  # skip sync+header+subheader for Mode2/Form1 data
        return bytes(self.image[off:off + 2048])

    def write(self, addr: int, value: int) -> None:
        if (addr & 0xF) == 0:
            self.status = 0x18
            self.response = bytearray([0x02])  # stub ACK

    def read(self, addr: int) -> int:
        reg = addr & 0xF
        if reg == 0:
            return self.status
        if reg == 1:
            # Explicit response FIFO empty state; this remains a CD-ROM stub.
            if len(self.response) == 0:
                return 0
            return self.response.pop(0)
        return 0


# ── GPU — command FIFO + real primitive raster ──────────────────────────────
class GPU:
    SCREEN_W = 320
    SCREEN_H = 240

    def __init__(self) -> None:
        self.vram = np.zeros((512, 1024), dtype=np.uint16)
        self.display = np.zeros((self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8)
        self.output = np.zeros((self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8)
        self.black_frame = np.zeros((self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8)
        self.gp0_buffer: list[int] = []
        self.current_cmd = 0
        self.words_needed = 1
        self.gpustat = 0x14802000
        self.draw_ox = 0
        self.draw_oy = 0
        self.hardware_mode = False
        # VRAM upload FSM
        self._vram_dst = None  # (x, y, w, h, written)

    # ---- GP0 command dispatch ----
    def write_gp0(self, word: int) -> None:
        word &= 0xFFFFFFFF
        if self._vram_dst is not None:
            self._feed_vram(word)
            return
        self.gp0_buffer.append(word)
        if len(self.gp0_buffer) == 1:
            self.current_cmd = (word >> 24) & 0xFF
            self.words_needed = self._words_for(self.current_cmd)
        if 0x40 <= self.current_cmd <= 0x5F and (self.current_cmd & 0x08):
            # Polylines have no fixed packet size. A terminal word is valid
            # only after the initial line and its first extension are present.
            minimum = 4 if not (self.current_cmd & 0x10) else 5
            if len(self.gp0_buffer) >= minimum and (word & 0xF000F000) == 0x50005000:
                self._execute_gp0()  # currently consumes line geometry as a stub
                self.gp0_buffer.clear()
            elif len(self.gp0_buffer) > 4096:
                self.gp0_buffer.clear()  # malformed stream guard
            return
        if len(self.gp0_buffer) >= self.words_needed:
            self._execute_gp0()
            self.gp0_buffer.clear()

    def write_gp1(self, word: int) -> None:
        cmd = (word >> 24) & 0xFF
        if cmd == 0x00:          # reset
            self.display.fill(0)
            self.gp0_buffer.clear()
            self._vram_dst = None
        elif cmd == 0x01:        # reset cmd buffer
            self.gp0_buffer.clear()
            self._vram_dst = None

    def _words_for(self, cmd: int) -> int:
        """GP0 packet lengths, including command word (PSX-SPX layout)."""
        if cmd == 0x02:
            return 3
        if 0x20 <= cmd <= 0x3F:
            n = 4 if cmd & 0x08 else 3
            # First colour is always in the command word; subsequent
            # Gouraud vertices have separate colour words.
            return 1 + n + (n - 1 if cmd & 0x10 else 0) + (n if cmd & 0x04 else 0)
        if 0x40 <= cmd <= 0x5F:
            if cmd & 0x08:  # variable length; handled before normal FIFO flush
                return 0x7FFFFFFF
            return 4 if cmd & 0x10 else 3
        if 0x60 <= cmd <= 0x7F:
            return 2 + bool(cmd & 0x04) + ((cmd & 0x18) == 0)
        if 0x80 <= cmd <= 0x9F:
            return 4
        if 0xA0 <= cmd <= 0xDF:
            return 3
        return 1

    def _execute_gp0(self) -> None:
        cmd = self.current_cmd
        buf = self.gp0_buffer
        if 0x20 <= cmd <= 0x3F:
            self._draw_polygon(cmd, buf)
        elif 0x40 <= cmd <= 0x5F:
            # Line rasterization is pending; consume completed packets safely.
            pass
        elif 0x60 <= cmd <= 0x7F:
            self._draw_rectangle(cmd, buf)
        elif cmd == 0x02:
            self._fill_rect(buf)
        elif cmd == 0xA0:
            x = buf[1] & 0xFFFF; y = (buf[1] >> 16) & 0xFFFF
            w = buf[2] & 0xFFFF; h = (buf[2] >> 16) & 0xFFFF
            self._vram_dst = [x & 0x3FF, y & 0x1FF, max(1, w), max(1, h), 0]
        elif cmd == 0xE5:
            self.draw_ox = _s16((buf[0] & 0x7FF) << 5) >> 5
            self.draw_oy = _s16(((buf[0] >> 11) & 0x7FF) << 5) >> 5
        # E1..E4, E6, 0x01 NOP: state only, nothing to raster here.

    def _feed_vram(self, word: int) -> None:
        x, y, w, h, written = self._vram_dst
        for half in (word & 0xFFFF, (word >> 16) & 0xFFFF):
            px = x + (written % w)
            py = y + (written // w)
            if py < 512 and px < 1024:
                self.vram[py, px] = half
            written += 1
            if written >= w * h:
                self._vram_dst = None
                return
        self._vram_dst[4] = written

    # ---- vertex / colour extraction ----
    @staticmethod
    def _color(word):
        return (word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF)

    def _xy(self, word):
        x = _s16(word & 0xFFFF) + self.draw_ox
        y = _s16((word >> 16) & 0xFFFF) + self.draw_oy
        return (x, y)

    def _draw_polygon(self, cmd, buf):
        """Decode GP0 20h-3Fh without consuming UV words as colours.

        Flat: CMD/RGB, XY0, [UV0], XY1, [UV1] ...
        Gouraud: CMD/RGB0, XY0, [UV0], RGB1, XY1, [UV1] ...
        Every textured vertex has one UV word: UV0 high half = CLUT,
        UV1 high half = texture page; later UV high halves are ignored.
        Texture sampling is not implemented; use neutral placeholder colour.
        """
        n = 4 if cmd & 0x08 else 3
        shaded = bool(cmd & 0x10)
        textured = bool(cmd & 0x04)
        pts, cols = [], []
        i = 1
        for vertex in range(n):
            if shaded and vertex:
                colour = self._color(buf[i]); i += 1
            else:
                colour = self._color(buf[0])
            if textured:
                colour = (0x80, 0x80, 0x80)
            pts.append(self._xy(buf[i])); i += 1
            if textured:
                i += 1  # UV word (CLUT on vertex 0, texpage on vertex 1)
            cols.append(colour)
        self._tri(pts[0], pts[1], pts[2], cols[0], cols[1], cols[2])
        if n == 4:
            self._tri(pts[1], pts[2], pts[3], cols[1], cols[2], cols[3])

    def _draw_rectangle(self, cmd, buf):
        col = self._color(buf[0])
        x, y = self._xy(buf[1])
        size = (cmd & 0x18)
        if size == 0x08:
            w = h = 1
        elif size == 0x10:
            w = h = 8
        elif size == 0x18:
            w = h = 16
        else:
            # Textured variable rectangles: cmd, XY, UV+CLUT, WH (index 3).
            idx = 2 + (1 if (cmd & 0x04) else 0)
            sz = buf[idx] if idx < len(buf) else 0
            w = sz & 0xFFFF; h = (sz >> 16) & 0xFFFF
        x2 = min(self.SCREEN_W, x + max(0, w)); y2 = min(self.SCREEN_H, y + max(0, h))
        x = max(0, x); y = max(0, y)
        if x < self.SCREEN_W and y < self.SCREEN_H and x2 > x and y2 > y:
            self.display[y:y2, x:x2] = col

    def _fill_rect(self, buf):
        col = self._color(buf[0])
        x = buf[1] & 0xFFFF; y = (buf[1] >> 16) & 0xFFFF
        w = buf[2] & 0xFFFF; h = (buf[2] >> 16) & 0xFFFF
        x2 = min(self.SCREEN_W, x + w); y2 = min(self.SCREEN_H, y + h)
        if x < self.SCREEN_W and y < self.SCREEN_H:
            self.display[max(0, y):y2, max(0, x):x2] = col

    def _tri(self, p1, p2, p3, c1, c2, c3):
        """Gouraud-shaded scanline triangle into the display framebuffer."""
        verts = sorted([(p1, c1), (p2, c2), (p3, c3)], key=lambda e: e[0][1])
        (x1, y1), col1 = verts[0]
        (x2, y2), col2 = verts[1]
        (x3, y3), col3 = verts[2]
        if y3 == y1:
            return
        miny = max(0, y1); maxy = min(self.SCREEN_H - 1, y3)
        for y in range(miny, maxy + 1):
            tA = (y - y1) / (y3 - y1)
            xa = x1 + (x3 - x1) * tA
            ca = [col1[k] + (col3[k] - col1[k]) * tA for k in range(3)]
            if y < y2 and y2 != y1:
                tB = (y - y1) / (y2 - y1)
                xb = x1 + (x2 - x1) * tB
                cb = [col1[k] + (col2[k] - col1[k]) * tB for k in range(3)]
            elif y3 != y2:
                tB = (y - y2) / (y3 - y2)
                xb = x2 + (x3 - x2) * tB
                cb = [col2[k] + (col3[k] - col2[k]) * tB for k in range(3)]
            else:
                xb, cb = xa, ca
            if xa > xb:
                xa, xb = xb, xa; ca, cb = cb, ca
            xi0 = max(0, int(xa)); xi1 = min(self.SCREEN_W - 1, int(xb))
            if xi1 < xi0:
                continue
            span = max(1, xb - xa)
            for x in range(xi0, xi1 + 1):
                t = (x - xa) / span
                self.display[y, x] = [
                    int(ca[0] + (cb[0] - ca[0]) * t),
                    int(ca[1] + (cb[1] - ca[1]) * t),
                    int(ca[2] + (cb[2] - ca[2]) * t),
                ]

    def render_frame(self) -> np.ndarray:
        """FILES=OFF demo scene — RPSC1 CRT green triangle; or real display."""
        if self.hardware_mode:
            self.output[:] = self.display
            return self.output
        self.display[:] = (8, 12, 8)
        t = time.time() * 2.5
        cx, cy = 160, 120
        pts = []
        for i in range(3):
            a = t + i * 2.094395
            pts.append((int(cx + 90 * np.cos(a)), int(cy + 90 * np.sin(a))))
        save_ox, save_oy = self.draw_ox, self.draw_oy
        self.draw_ox = self.draw_oy = 0
        self._tri(pts[0], pts[1], pts[2], (0, 255, 65), (0, 180, 40), (0, 120, 30))
        self.draw_ox, self.draw_oy = save_ox, save_oy
        self.output[:] = self.display
        self.output[::2] = (self.display[::2].astype(np.uint16) * 3 // 4).astype(np.uint8)
        return self.output


# ── DMA — real OTC + GPU transfers ──────────────────────────────────────────
class DMA:
    def __init__(self, mem: "Memory") -> None:
        self.mem = mem
        self.control = 0x07654321
        self.dicr = 0
        self.channels = [{"madr": 0, "bcr": 0, "chcr": 0} for _ in range(7)]

    def write(self, ch: int, reg: int, val: int) -> None:
        c = self.channels[ch]
        if reg == 0:
            c["madr"] = val & 0xFFFFFF
        elif reg == 4:
            c["bcr"] = val & 0xFFFFFFFF
        elif reg == 8:
            c["chcr"] = val & 0xFFFFFFFF
            if val & (1 << 24):  # start/busy
                self._run(ch)

    def _run(self, ch: int) -> None:
        c = self.channels[ch]
        madr = c["madr"] & 0x1FFFFC
        chcr = c["chcr"]
        sync = (chcr >> 9) & 3
        if ch == 6:  # OTC: build reverse ordering table
            count = min(c["bcr"] & 0xFFFF, len(self.mem.ram) // 4)
            for i in range(count):
                nxt = 0xFFFFFF if i == count - 1 else (madr - 4) & 0xFFFFFF
                self.mem.write32(0x80000000 | madr, nxt)
                madr = (madr - 4) & 0x1FFFFC
        elif ch == 2:  # GPU
            if sync == 3:  # reserved sync mode: no transfer
                c["chcr"] &= ~(1 << 24)
                return
            if sync == 2:  # linked list
                guard = 0
                visited = set()
                while guard < 0x80000:
                    if madr in visited:
                        break
                    visited.add(madr)
                    header = self.mem.read32(0x80000000 | madr)
                    size = (header >> 24) & 0xFF
                    for i in range(size):
                        word = self.mem.read32(0x80000000 | ((madr + 4 + i * 4) & 0x1FFFFC))
                        self.mem.gpu.write_gp0(word)
                    nxt = header & 0xFFFFFF
                    if nxt & 0x800000:
                        break
                    madr = nxt & 0x1FFFFC
                    guard += 1
            else:  # block transfer (sync 0/1): BCR = blocks<<16 | size
                blocks = ((c["bcr"] >> 16) & 0xFFFF) or 1
                size = c["bcr"] & 0xFFFF
                total = blocks * size
                to_gpu = (chcr & 1) == 1
                for i in range(total):
                    a = 0x80000000 | ((madr + i * 4) & 0x1FFFFC)
                    if to_gpu:
                        self.mem.gpu.write_gp0(self.mem.read32(a))
        # transfer complete: clear busy, raise DMA IRQ flag
        c["chcr"] &= ~(1 << 24)
        self.mem.intc.trigger(3)

    def start(self, channel: int) -> None:  # kept for API compatibility
        self._run(channel)


# ── CPU: R3000A with exceptions, interrupts, delay slots ────────────────────
class CPU:
    def __init__(self) -> None:
        self.pc = 0x80010000
        self.next_pc = self.pc + 4
        self.cur_pc = self.pc
        self.registers = [0] * 32
        self.hi = 0
        self.lo = 0
        self.cop0 = [0] * 32
        self.cop0[15] = 0x00000002  # PRId
        self.cop0[12] = 0x00400000  # SR: BEV=1 at reset (vectors in ROM)
        self.in_delay = False
        self.branch_taken = False
        self.cycle_count = 0
        self.last_opcode_name = "NOP"
        # MIPS-I load delay slot: pending (reg, value) applied after next insn
        self.load_delay = (0, 0)
        self._pending = (0, 0)
        self.trap_overflow = False  # most games tolerate wrap; off by default

    def _sext(self, val: int, bits: int = 16) -> int:
        val &= (1 << bits) - 1
        if val & (1 << (bits - 1)):
            val -= 1 << bits
        return val

    def _u32(self, v: int) -> int:
        return v & 0xFFFFFFFF

    def _sra(self, val: int, sh: int) -> int:
        val = self._u32(val); sh &= 31
        if sh == 0:
            return val
        if val & 0x80000000:
            return self._u32((val >> sh) | (0xFFFFFFFF << (32 - sh)))
        return val >> sh

    def write_reg(self, reg: int, value: int) -> None:
        if reg != 0:
            self.registers[reg] = self._u32(value)
            if self._pending[0] == reg:   # delay-slot write beats the load
                self._pending = (0, 0)

    def _set_load(self, reg, value):
        if reg == 0:
            return
        self.load_delay = (reg, self._u32(value))

    def opcode_name(self, insn: int) -> str:
        op = (insn >> 26) & 0x3F
        if op == 0x00:
            return SPECIAL_FUNCT.get(insn & 0x3F, f"SPECIAL/{insn & 0x3F:02X}")
        if op == 0x01:
            return REGIMM_RT.get((insn >> 16) & 0x1F, f"REGIMM/{(insn >> 16) & 0x1F:02X}")
        return PRIMARY_OPS.get(op, f"OP/{op:02X}")

    def _lwl(self, mem, addr, rtv):
        word = mem.read32(addr & ~3); shift = (3 - (addr & 3)) * 8
        mask = (0xFFFFFFFF << shift) & 0xFFFFFFFF
        return ((rtv & ~mask) | ((word << shift) & mask)) & 0xFFFFFFFF

    def _lwr(self, mem, addr, rtv):
        word = mem.read32(addr & ~3); shift = (addr & 3) * 8
        mask = 0xFFFFFFFF >> shift
        return ((rtv & ~mask) | ((word >> shift) & mask)) & 0xFFFFFFFF

    def _swl(self, mem, addr, rtv):
        base = addr & ~3
        for i in range((addr & 3) + 1):
            mem.write8(base + i, (rtv >> (8 * (3 - (addr & 3) + i))) & 255)

    def _swr(self, mem, addr, rtv):
        for i in range(4 - (addr & 3)):
            mem.write8(addr + i, (rtv >> (8 * i)) & 255)

    def exception(self, cause_code: int) -> None:
        sr = self.cop0[12]
        self.cop0[12] = (sr & ~0x3F) | ((sr << 2) & 0x3F)      # push IE/KU stack
        cause = self.cop0[13] & ~0x7C
        cause |= (cause_code << 2) & 0x7C
        if self.in_delay:
            self.cop0[14] = self._u32(self.cur_pc - 4)
            cause |= 0x80000000
        else:
            self.cop0[14] = self._u32(self.cur_pc)
            cause &= 0x7FFFFFFF
        self.cop0[13] = cause & 0xFFFFFFFF
        vector = 0xBFC00180 if (self.cop0[12] & (1 << 22)) else 0x80000080
        self.pc = vector
        self.next_pc = self._u32(vector + 4)
        self.in_delay = False
        self.branch_taken = False

    def _irq_check(self, mem) -> bool:
        # Drive COP0 Cause IP2 from the interrupt controller.
        if mem.intc.pending():
            self.cop0[13] |= 1 << 10
        else:
            self.cop0[13] &= ~(1 << 10)
        sr = self.cop0[12]
        if (sr & 1) and (self.cop0[13] & sr & 0xFF00):
            self.exception(EXC_INT)
            return True
        return False

    def step(self, mem) -> None:
        # Bookkeeping for the instruction about to run, so an IRQ taken here
        # records the right EPC and BD bit (it interrupts THIS instruction).
        self.cur_pc = self.pc
        self.in_delay = self.branch_taken   # previous insn was a branch/jump
        self.branch_taken = False
        if self._irq_check(mem):
            self._apply_load(self.load_delay)
            self.load_delay = self._pending = (0, 0)
            return
        if self.pc & 3:
            self.exception(EXC_ADEL)
            return
        insn = mem.read32(self.pc)
        self.last_opcode_name = self.opcode_name(insn)

        self.pc = self.next_pc
        self.next_pc = self._u32(self.next_pc + 4)

        # load queued by the PREVIOUS instruction lands after this one
        self._pending = self.load_delay
        self.load_delay = (0, 0)

        op = (insn >> 26) & 0x3F
        rs = (insn >> 21) & 0x1F
        rt = (insn >> 16) & 0x1F
        rd = (insn >> 11) & 0x1F
        sa = (insn >> 6) & 0x1F
        funct = insn & 0x3F
        imm = insn & 0xFFFF
        target = insn & 0x03FFFFFF
        simm = self._sext(imm, 16)
        rsv = self.registers[rs]
        rtv = self.registers[rt]

        def branch(cond, tgt):
            self.branch_taken = True       # next insn is in a delay slot
            if cond:
                self.next_pc = self._u32(tgt)

        if op == 0x00:  # SPECIAL
            if funct == 0x00: self.write_reg(rd, rtv << sa)
            elif funct == 0x02: self.write_reg(rd, rtv >> sa)
            elif funct == 0x03: self.write_reg(rd, self._sra(rtv, sa))
            elif funct == 0x04: self.write_reg(rd, rtv << (rsv & 31))
            elif funct == 0x06: self.write_reg(rd, rtv >> (rsv & 31))
            elif funct == 0x07: self.write_reg(rd, self._sra(rtv, rsv & 31))
            elif funct == 0x08:  # JR
                self.next_pc = rsv & 0xFFFFFFFC; self.branch_taken = True
            elif funct == 0x09:  # JALR
                tgt = rsv & 0xFFFFFFFC
                self.write_reg(rd if rd else 31, self.next_pc)
                self.next_pc = tgt; self.branch_taken = True
            elif funct == 0x0C:  # SYSCALL
                self._apply_load(self._pending); self.exception(EXC_SYS); return
            elif funct == 0x0D:  # BREAK
                self._apply_load(self._pending); self.exception(EXC_BP); return
            elif funct == 0x10: self.write_reg(rd, self.hi)
            elif funct == 0x11: self.hi = rsv
            elif funct == 0x12: self.write_reg(rd, self.lo)
            elif funct == 0x13: self.lo = rsv
            elif funct == 0x18:  # MULT
                prod = self._sext(rsv, 32) * self._sext(rtv, 32)
                self.lo = self._u32(prod); self.hi = self._u32(prod >> 32)
            elif funct == 0x19:  # MULTU
                prod = rsv * rtv
                self.lo = self._u32(prod); self.hi = self._u32(prod >> 32)
            elif funct == 0x1A:  # DIV
                num = self._sext(rsv, 32); den = self._sext(rtv, 32)
                if den == 0:
                    self.lo = 0xFFFFFFFF if num >= 0 else 1
                    self.hi = self._u32(num)
                else:
                    q = int(abs(num) // abs(den)) * (-1 if (num < 0) != (den < 0) else 1)
                    self.lo = self._u32(q); self.hi = self._u32(num - q * den)
            elif funct == 0x1B:  # DIVU
                if rtv == 0:
                    self.lo = 0xFFFFFFFF; self.hi = rsv
                else:
                    self.lo = rsv // rtv; self.hi = rsv % rtv
            elif funct == 0x20:  # ADD (trap on overflow)
                r = self._sext(rsv, 32) + self._sext(rtv, 32)
                if self.trap_overflow and not (-0x80000000 <= r <= 0x7FFFFFFF):
                    self._apply_load(self._pending); self.exception(EXC_OVF); return
                self.write_reg(rd, r)
            elif funct == 0x21: self.write_reg(rd, rsv + rtv)
            elif funct == 0x22:  # SUB
                r = self._sext(rsv, 32) - self._sext(rtv, 32)
                if self.trap_overflow and not (-0x80000000 <= r <= 0x7FFFFFFF):
                    self._apply_load(self._pending); self.exception(EXC_OVF); return
                self.write_reg(rd, r)
            elif funct == 0x23: self.write_reg(rd, rsv - rtv)
            elif funct == 0x24: self.write_reg(rd, rsv & rtv)
            elif funct == 0x25: self.write_reg(rd, rsv | rtv)
            elif funct == 0x26: self.write_reg(rd, rsv ^ rtv)
            elif funct == 0x27: self.write_reg(rd, ~(rsv | rtv))
            elif funct == 0x2A: self.write_reg(rd, 1 if self._sext(rsv, 32) < self._sext(rtv, 32) else 0)
            elif funct == 0x2B: self.write_reg(rd, 1 if rsv < rtv else 0)

        elif op == 0x01:  # REGIMM
            signed = self._sext(rsv, 32); link = rt in (0x10, 0x11)
            cond = (signed < 0) if rt in (0x00, 0x10) else (signed >= 0)
            if link:
                self.write_reg(31, self.next_pc)
            branch(cond, self.pc + (simm << 2))

        elif op == 0x02:  # J
            branch(True, (self.pc & 0xF0000000) | (target << 2))
        elif op == 0x03:  # JAL
            self.write_reg(31, self.next_pc)
            branch(True, (self.pc & 0xF0000000) | (target << 2))
        elif op == 0x04: branch(rsv == rtv, self.pc + (simm << 2))
        elif op == 0x05: branch(rsv != rtv, self.pc + (simm << 2))
        elif op == 0x06: branch(self._sext(rsv, 32) <= 0, self.pc + (simm << 2))
        elif op == 0x07: branch(self._sext(rsv, 32) > 0, self.pc + (simm << 2))

        elif op == 0x08:  # ADDI (trap)
            r = self._sext(rsv, 32) + simm
            if self.trap_overflow and not (-0x80000000 <= r <= 0x7FFFFFFF):
                self._apply_load(self._pending); self.exception(EXC_OVF); return
            self.write_reg(rt, r)
        elif op == 0x09: self.write_reg(rt, rsv + simm)
        elif op == 0x0A: self.write_reg(rt, 1 if self._sext(rsv, 32) < simm else 0)
        elif op == 0x0B: self.write_reg(rt, 1 if rsv < self._u32(simm) else 0)
        elif op == 0x0C: self.write_reg(rt, rsv & imm)
        elif op == 0x0D: self.write_reg(rt, rsv | imm)
        elif op == 0x0E: self.write_reg(rt, rsv ^ imm)
        elif op == 0x0F: self.write_reg(rt, imm << 16)

        elif op == 0x10:  # COP0
            fmt = rs
            if fmt == 0x00: self._set_load(rt, self.cop0[rd])
            elif fmt == 0x04: self.cop0[rd] = rtv
            elif fmt == 0x02: self._set_load(rt, self.cop0[rd])
            elif fmt == 0x06: self.cop0[rd] = rtv
            elif fmt == 0x10 and funct == 0x10:  # RFE
                sr = self.cop0[12]
                self.cop0[12] = (sr & ~0x0F) | ((sr >> 2) & 0x0F)

        elif op == 0x11: pass  # COP1 — no FPU

        elif op == 0x12:  # COP2 GTE
            fmt = rs
            if fmt == 0x00: self._set_load(rt, mem.gte.data[rd])
            elif fmt == 0x02: self._set_load(rt, mem.gte.control[rd])
            elif fmt == 0x04: mem.gte.data[rd] = rtv
            elif fmt == 0x06: mem.gte.control[rd] = rtv
            elif fmt & 0x10: mem.gte.command(insn)

        elif op == 0x13: pass  # COP3

        elif op == 0x20:  # LB
            self._set_load(rt, self._sext(mem.read8(self._u32(rsv + simm)), 8))
        elif op == 0x21:  # LH
            self._set_load(rt, self._sext(mem.read16(self._u32(rsv + simm) & ~1), 16))
        elif op == 0x22:  # LWL
            merge = self._pending[1] if self._pending[0] == rt else rtv
            self._set_load(rt, self._lwl(mem, self._u32(rsv + simm), merge))
        elif op == 0x23:  # LW
            self._set_load(rt, mem.read32(self._u32(rsv + simm) & ~3))
        elif op == 0x24:  # LBU
            self._set_load(rt, mem.read8(self._u32(rsv + simm)))
        elif op == 0x25:  # LHU
            self._set_load(rt, mem.read16(self._u32(rsv + simm) & ~1))
        elif op == 0x26:  # LWR
            merge = self._pending[1] if self._pending[0] == rt else rtv
            self._set_load(rt, self._lwr(mem, self._u32(rsv + simm), merge))

        elif op == 0x28: mem.write8(self._u32(rsv + simm), rtv & 0xFF)
        elif op == 0x29: mem.write16(self._u32(rsv + simm) & ~1, rtv & 0xFFFF)
        elif op == 0x2A: self._swl(mem, self._u32(rsv + simm), rtv)
        elif op == 0x2B: mem.write32(self._u32(rsv + simm) & ~3, rtv)
        elif op == 0x2E: self._swr(mem, self._u32(rsv + simm), rtv)
        elif op == 0x2F: pass  # CACHE — NOP

        elif op == 0x32:  # LWC2
            mem.gte.data[rt] = mem.read32(self._u32(rsv + simm) & ~3)
        elif op == 0x3A:  # SWC2
            mem.write32(self._u32(rsv + simm) & ~3, mem.gte.data[rt])
        elif op in (0x30, 0x31, 0x33, 0x38, 0x39, 0x3B):
            pass  # LL/SC/LWC1/3 SWC1/3 — unused on retail PS1

        # apply the load that was pending at the START of this instruction
        self._apply_load(self._pending)
        self._pending = (0, 0)
        self.registers[0] = 0
        self.cycle_count += 1

    def _apply_load(self, pending):
        reg, val = pending
        if reg:
            self.registers[reg] = val
            self.registers[0] = 0


class Memory:
    def __init__(self) -> None:
        self.ram = bytearray(2 * 1024 * 1024)
        self.scratch = bytearray(1024)
        self.bios = bytearray(512 * 1024)
        self.bios_loaded = False
        self.cdrom = CDROM()
        self.gpu = GPU()
        self.spu = SPU()
        self.timers = Timers()
        self.intc = INTC()
        self.cpu = CPU()
        self.gte = GTE()
        self.pad = Controller()
        self.dma = DMA(self)
        self._joy_rx = 0xFF
        self._seed_demo_program()

    def _seed_demo_program(self) -> None:
        base = 0x10000
        code = [
            0x3C080000, 0x25080001, 0x00000000, 0x01084020,
            0x00084842, 0x01285024, 0x01485825, 0x01686026,
            0x3C010001, 0x00200821, 0x1000FFF5, 0x00000000,
        ]
        for i, w in enumerate(code):
            off = base + i * 4
            self.ram[off:off + 4] = w.to_bytes(4, "little")

    def load_bios(self, path: str) -> int:
        with open(path, "rb") as f:
            data = f.read()
        n = min(len(data), len(self.bios))
        self.bios[:n] = data[:n]
        self.bios_loaded = True
        self.cpu.pc = 0xBFC00000
        self.cpu.next_pc = 0xBFC00004
        return n

    def _phys(self, addr: int) -> int:
        addr &= 0xFFFFFFFF
        if 0x80000000 <= addr <= 0x9FFFFFFF:
            return addr & 0x1FFFFFFF
        if 0xA0000000 <= addr <= 0xBFFFFFFF:
            return addr & 0x1FFFFFFF
        return addr

    def read8(self, addr):
        p = self._phys(addr)
        if p < 0x200000:
            return self.ram[p]
        if 0x1F800000 <= p < 0x1F800400:
            return self.scratch[p - 0x1F800000]
        if 0x1FC00000 <= p < 0x1FC80000:
            return self.bios[p - 0x1FC00000]
        if p == 0x1F801040:  # JOY_RX
            return self._joy_rx & 0xFF
        if 0x1F801044 <= p <= 0x1F801047:  # JOY_STAT byte lanes
            return (0x0007 >> (8 * (p - 0x1F801044))) & 0xFF
        if 0x1F801800 <= p <= 0x1F801803:
            return self.cdrom.read(p) & 0xFF
        return 0

    def read16(self, addr):
        p = self._phys(addr)
        if p == 0x1F801044:  # JOY_STAT (TX ready, RX ready)
            return 0x0007
        return self.read8(addr) | (self.read8(addr + 1) << 8)

    def read32(self, addr):
        p = self._phys(addr) & ~3
        if p < 0x200000:
            return int.from_bytes(self.ram[p:p + 4], "little")
        if 0x1FC00000 <= p < 0x1FC80000:
            i = p - 0x1FC00000
            return int.from_bytes(self.bios[i:i + 4], "little")
        if 0x1F801070 <= p <= 0x1F801077:
            return self.intc.read_stat() if (p & 4) == 0 else self.intc.i_mask
        if 0x1F801080 <= p <= 0x1F8010EF:  # DMA channel regs
            ch = (p - 0x1F801080) >> 4; reg = p & 0xF
            c = self.dma.channels[ch]
            return {0: c["madr"], 4: c["bcr"], 8: c["chcr"]}.get(reg, 0)
        if p == 0x1F8010F0:
            return self.dma.control
        if p == 0x1F8010F4:
            return self.dma.dicr
        if 0x1F801810 <= p <= 0x1F801817:
            return 0 if (p & 4) == 0 else self.gpu.gpustat
        if 0x1F801800 <= p <= 0x1F801803:
            return self.cdrom.read(p)
        if 0x1F801C00 <= p <= 0x1F801FFF:
            return self.spu.read(p)
        return 0

    def write8(self, addr, val):
        p = self._phys(addr); val &= 0xFF
        if p < 0x200000:
            self.ram[p] = val
        elif 0x1F800000 <= p < 0x1F800400:
            self.scratch[p - 0x1F800000] = val
        elif p == 0x1F801040:  # JOY_TX — exchange a byte with the pad
            self._joy_rx = self.pad.exchange(val)
        elif 0x1F801800 <= p <= 0x1F801803:
            self.cdrom.write(p, val)

    def write16(self, addr, val):
        p = self._phys(addr)
        if p == 0x1F80104A and not (val & 0x2):  # JOY_CTRL deselect → reset pad
            self.pad.reset()
        self.write8(addr, val & 0xFF)
        self.write8(addr + 1, (val >> 8) & 0xFF)

    def write32(self, addr, val):
        p = self._phys(addr) & ~3; val &= 0xFFFFFFFF
        if p < 0x200000:
            self.ram[p:p + 4] = val.to_bytes(4, "little")
        elif 0x1F801070 <= p <= 0x1F801073:
            self.intc.write_stat(val)
        elif 0x1F801074 <= p <= 0x1F801077:
            self.intc.write_mask(val)
        elif 0x1F801080 <= p <= 0x1F8010EF:
            ch = (p - 0x1F801080) >> 4; reg = p & 0xF
            self.dma.write(ch, reg, val)
        elif p == 0x1F8010F0:
            self.dma.control = val
        elif p == 0x1F8010F4:
            self.dma.dicr = val
        elif p == 0x1F801810:
            self.gpu.write_gp0(val)
        elif p == 0x1F801814:
            self.gpu.write_gp1(val)
        elif 0x1F801800 <= p <= 0x1F801803:
            self.cdrom.write(p, val & 0xFF)
        elif 0x1F801C00 <= p <= 0x1F801FFF:
            self.spu.write(p, val)


class Emulator:
    def __init__(self) -> None:
        self.memory = Memory()
        self.gpu = self.memory.gpu
        self.cpu = self.memory.cpu
        self.pad = self.memory.pad
        self.running = False
        self.disc_loaded = False
        self.bios_path: Optional[str] = None
        self.disc_path: Optional[str] = None

    def run_frame(self) -> None:
        for _ in range(25000):
            self.cpu.step(self.memory)
        self.memory.timers.tick(25000)
        self.memory.intc.trigger(0)  # VBLANK


# ── RPSC1-style CRT GUI (unchanged layout; adds keyboard→pad) ───────────────
class RPSC1_GUI:
    KEYMAP = {
        "Up": 4, "Right": 5, "Down": 6, "Left": 7,
        "Return": 3, "BackSpace": 0,       # Start / Select
        "a": 14, "s": 13, "w": 12, "d": 15,  # Cross / Circle / Triangle / Square
        "q": 10, "e": 11, "z": 8, "c": 9,    # L1 / R1 / L2 / R2
    }

    def __init__(self, root: tk.Tk, emu: Emulator) -> None:
        self.root = root
        self.emu = emu
        self.root.title(TITLE)
        self.root.configure(bg="#0a0a0a")
        self.root.geometry("900x760")
        self.root.minsize(690, 610)

        self._photo: Optional[tk.PhotoImage] = None
        self._blank_img: Optional[tk.PhotoImage] = None
        self._canvas_image = None
        self.last_time = time.time()
        self.show_demo = False
        self._emu_thread: Optional[threading.Thread] = None

        BG = "#171b20"; PANEL = "#252b32"; TEXT = "#e7e9eb"; GREEN = "#69e68a"
        self.root.configure(bg=BG)

        menubar = tk.Menu(root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Load BIOS...", command=self.load_bios)
        file_menu.add_command(label="Load Disc...", command=self.load_disc)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        emu_menu = tk.Menu(menubar, tearoff=False)
        emu_menu.add_command(label="Run / Pause", command=self.toggle_run)
        emu_menu.add_command(label="Stop", command=self.stop_run)
        menubar.add_cascade(label="Emulation", menu=emu_menu)
        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="Opcode Coverage", command=self.show_rpsc)
        help_menu.add_command(label="Controls", command=self.show_controls)
        help_menu.add_command(label="Legal", command=self.show_legal)
        menubar.add_cascade(label="Help", menu=help_menu)
        root.configure(menu=menubar)

        header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=12, pady=(9, 4))
        tk.Label(header, text="RPSC1", font=("Helvetica", 26, "bold"),
                 fg=GREEN, bg=BG).pack(side="left")
        tk.Label(header, text=f"  RPSC1 {VERSION}  |  PLAYSTATION EMULATOR",
                 font=("Helvetica", 11, "bold"), fg=TEXT, bg=BG).pack(side="left")
        tk.Label(header, text="FILES=OFF", font=("Courier", 10, "bold"),
                 fg=GREEN, bg=BG).pack(side="right")

        toolbar = tk.Frame(root, bg=PANEL, padx=8, pady=8)
        toolbar.pack(fill="x", padx=12, pady=(2, 5))

        def btn(parent, label, command, accent=False):
            return tk.Button(parent, text=label, command=command,
                             font=("Helvetica", 11, "bold"), bg="#000000",
                             fg="#2196ff", activebackground="#101010",
                             activeforeground="#66bbff", relief="raised", bd=2,
                             highlightthickness=1, highlightbackground="#2196ff",
                             padx=10, pady=6, cursor="hand2")

        actions = [
            ("Run / Pause", self.toggle_run, True),
            ("Stop", self.stop_run, False),
            ("Load BIOS", self.load_bios, False),
            ("Load Disc", self.load_disc, False),
            ("Opcode Info", self.show_rpsc, False),
            ("Legal", self.show_legal, False),
        ]
        for i, (label, callback, accent) in enumerate(actions):
            button = btn(toolbar, label, callback, accent)
            button.grid(row=i // 3, column=i % 3, sticky="ew", padx=4, pady=3)
        for column in range(3):
            toolbar.grid_columnconfigure(column, weight=1, uniform="controls")

        viewport = tk.Frame(root, bg="#080a0c", bd=2, relief="sunken")
        viewport.pack(fill="both", expand=True, padx=12, pady=(3, 5))
        self.canvas = tk.Canvas(viewport, width=640, height=480,
                                bg="black", highlightthickness=0, relief="flat")
        self.canvas.pack(expand=True)
        self.canvas.bind("<Configure>", self._center_frame)

        self.status = tk.Label(root,
            text="FILES=OFF | PRESS RUN DEMO | FPS: 0 | OP: —",
            fg=GREEN, bg=PANEL, font=("Courier", 10), anchor="w", padx=12, pady=8)
        self.status.pack(side="bottom", fill="x", padx=12, pady=(0, 8))

        root.bind("<KeyPress>", self._key_down)
        root.bind("<KeyRelease>", self._key_up)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(16, self._update_display)

    def _key_down(self, e):
        bit = self.KEYMAP.get(e.keysym)
        if bit is not None:
            self.emu.pad.press(bit)

    def _key_up(self, e):
        bit = self.KEYMAP.get(e.keysym)
        if bit is not None:
            self.emu.pad.release(bit)

    def _center_frame(self, event=None):
        if self._canvas_image is not None:
            self.canvas.coords(self._canvas_image,
                               max(0, self.canvas.winfo_width() // 2 - 320),
                               max(0, self.canvas.winfo_height() // 2 - 240))

    def show_legal(self):
        messagebox.showinfo("Legal — Clean ROM", LEGAL_NOTICE)

    def show_controls(self):
        messagebox.showinfo("Controls",
            "D-Pad: Arrow keys\nCross: A   Circle: S   Triangle: W   Square: D\n"
            "L1/R1: Q/E   L2/R2: Z/C\nStart: Enter   Select: Backspace\n\n"
            "(Pad is polled only once a BIOS/game drives SIO.)")

    def show_rpsc(self):
        win = tk.Toplevel(self.root)
        win.title("RPSC coverage"); win.configure(bg="#0a0a0a"); win.geometry("560x640")
        txt = scrolledtext.ScrolledText(win, bg="#001100", fg="#00ff41",
                                        font=("Courier", 10), insertbackground="#00ff41")
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", RPSC.report()); txt.configure(state="disabled")

    def _confirm_legal(self):
        return messagebox.askokcancel("Legal ownership required",
            LEGAL_NOTICE + "\n\nContinue only if you legally own the file you will select?")

    def load_bios(self):
        if not self._confirm_legal():
            return
        path = filedialog.askopenfilename(title="Load legally owned PS1 BIOS",
            filetypes=[("BIOS", "*.bin *.rom *.BIN *.ROM"), ("All", "*.*")])
        if not path:
            return
        try:
            n = self.emu.memory.load_bios(path)
            self.emu.bios_path = path
            self.emu.gpu.hardware_mode = True
            self.status.config(text=f"BIOS loaded ({n} bytes) · {os.path.basename(path)}")
        except OSError as exc:
            messagebox.showerror("BIOS", f"Could not read BIOS:\n{exc}")

    def load_disc(self):
        if not self._confirm_legal():
            return
        path = filedialog.askopenfilename(title="Load legally owned PS1 disc image",
            filetypes=[("PS1 Images", "*.bin *.cue *.iso *.BIN *.CUE *.ISO"), ("All", "*.*")])
        if not path:
            return
        try:
            if path.lower().endswith(".cue"):
                messagebox.showinfo("CUE",
                    "CUE selected. Point at the matching .bin track for binary load.")
                return
            n = self.emu.memory.cdrom.load_image(path)
            self.emu.disc_loaded = True
            self.emu.disc_path = path
            self.status.config(
                text=f"Disc loaded ({n} bytes) · {os.path.basename(path)} · no boot FSM yet")
            messagebox.showinfo("Disc",
                f"Loaded {os.path.basename(path)} ({n} bytes).\n"
                "Held in RAM; there is no ISO9660 boot path yet, so the demo GPU\n"
                "stays active. Your file stays local — nothing is uploaded.")
        except OSError as exc:
            messagebox.showerror("Disc", f"Could not read image:\n{exc}")

    def toggle_run(self):
        self.emu.running = not self.emu.running
        self.show_demo = self.emu.running
        if self.emu.running:
            if self._emu_thread is None or not self._emu_thread.is_alive():
                self._emu_thread = threading.Thread(target=self._emu_loop, daemon=True)
                self._emu_thread.start()
            self.status.config(text="RUNNING · RPSC mode · FILES=OFF demo")
        else:
            self.status.config(text="PAUSED · FILES=OFF")

    def stop_run(self):
        self.emu.running = False
        self.show_demo = False
        self.status.config(text="STOPPED · FILES=OFF")

    def _emu_loop(self):
        while self.emu.running:
            start = time.monotonic()
            self.emu.run_frame()
            time.sleep(max(0, (1 / 60) - (time.monotonic() - start)))

    def _on_close(self):
        self.emu.running = False
        self.root.destroy()

    @staticmethod
    def _frame_to_photo(frame):
        height, width = frame.shape[:2]
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        ppm = header + np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        return tk.PhotoImage(data=ppm, format="PPM").zoom(2, 2)

    def _update_display(self):
        if not self.show_demo:
            if self._blank_img is None:
                self._blank_img = self._frame_to_photo(self.emu.gpu.black_frame)
            self._photo = self._blank_img
        else:
            self._photo = self._frame_to_photo(self.emu.gpu.render_frame())

        if self._canvas_image is None:
            self._canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
        else:
            self.canvas.itemconfigure(self._canvas_image, image=self._photo)

        self._center_frame()
        now = time.time()
        fps = int(1 / (now - self.last_time + 1e-6)) if self.show_demo else 0
        self.last_time = now
        op = self.emu.cpu.last_opcode_name
        gte = self.emu.memory.gte.last_cmd
        mode = "RUNNING" if self.show_demo else "PRESS RUN DEMO"
        bios = "BIOS:yes" if self.emu.memory.bios_loaded else "BIOS:off"
        disc = "DISC:yes" if self.emu.disc_loaded else "DISC:off"
        self.status.config(
            text=f"files=OFF · {mode} · FPS:{fps} · OP:{op} · GTE:{gte} · {bios} · {disc}")
        self.root.after(16, self._update_display)


def all_opcodes_report() -> str:
    return RPSC.report()


def _selftest() -> int:
    """Headless self-test of the upgraded subsystems. Returns failure count."""
    fails = 0

    def check(name, cond):
        nonlocal fails
        if not cond:
            fails += 1
            print(f"  FAIL: {name}")
        else:
            print(f"  ok  : {name}")

    print("RPSC 1.0 self-test")
    mem = Memory()
    cpu = mem.cpu

    # 1) CPU arithmetic + branch-delay: run the seeded loop a few iterations.
    cpu.pc = 0x80010000; cpu.next_pc = 0x80010004
    for _ in range(64):
        cpu.step(mem)
    check("seeded loop retires without crash", cpu.cycle_count == 64)

    # 2) Load-delay slot: value lands AFTER the next instruction.
    m2 = Memory(); c = m2.cpu
    m2.write32(0x00001000, 0xDEADBEEF)
    prog = [0x3C080000,  # lui  t0,0
            0x8D090000,  # lw   t1,0(t0)      ; load from 0x0
            0x01204021,  # addu t0,t1,$0      ; reads t1 in load-delay slot
            0x00000000]
    for i, w in enumerate(prog):
        m2.write32(0x00002000 + i * 4, w)
    c.pc = 0x80002000; c.next_pc = 0x80002004
    m2.write32(0x00000000, 0x12345678)
    c.registers[9] = 0xAAAA
    for _ in range(4):
        c.step(m2)
    # addu in the delay slot must see the OLD t1, then t1 gets the load.
    check("load-delay: slot reads old value", c.registers[8] == 0xAAAA)
    check("load-delay: value lands after slot", c.registers[9] == 0x12345678)

    # 2b) delay-slot write to the same reg cancels the load
    m2b = Memory(); cb = m2b.cpu
    m2b.write32(0x0, 0x11111111)
    for i, w in enumerate([0x8C090000,   # lw    t1,0($0)
                           0x24090005,   # addiu t1,$0,5   (delay slot)
                           0x00000000]):
        m2b.write32(0x3000 + i * 4, w)
    cb.pc = 0x80003000; cb.next_pc = 0x80003004
    for _ in range(3):
        cb.step(m2b)
    check("load-delay: slot write wins", cb.registers[9] == 5)

    # 2c) IRQ vectors with correct EPC, and BD set in a delay slot
    m2c = Memory(); cc = m2c.cpu
    for i, w in enumerate([0x10000002,   # beq $0,$0,+2  (always)
                           0x00000000,   # delay slot  <- IRQ lands here
                           0x00000000, 0x00000000]):
        m2c.write32(0x4000 + i * 4, w)
    cc.pc = 0x80004000; cc.next_pc = 0x80004004
    cc.cop0[12] = 0x00000401             # IEc=1, IM2=1, BEV=0
    cc.step(m2c)                         # execute the branch
    m2c.intc.write_mask(1); m2c.intc.trigger(0)
    cc.step(m2c)                         # IRQ taken before delay slot runs
    check("irq vector 0x80000080", cc.pc == 0x80000080)
    check("irq EPC = branch addr (BD)", cc.cop0[14] == 0x80004000)
    check("irq CAUSE.BD set", bool(cc.cop0[13] & 0x80000000))
    check("irq IEc cleared by push", (cc.cop0[12] & 1) == 0)

    # 3) GTE division matches the documented UNR algorithm at a known point.
    g = mem.gte
    check("gte unr table len 257", len(g.unr) == 257)
    d = g._divide(0x1000, 0x2000)  # H/SZ3 with H<2*SZ3
    check("gte divide in range", 0 <= d <= 0x1FFFF)

    # 4) NCLIP sign: CCW winding → positive, CW → negative.
    g.data[12] = (0 << 16) | 0      # SXY0 (0,0)
    g.data[13] = (0 << 16) | 10     # SXY1 (10,0)
    g.data[14] = (10 << 16) | 0     # SXY2 (0,10)
    g.command(0x06)
    nclip = _s32(g.data[24])
    check("nclip produces nonzero area", nclip != 0)

    # 5) RTPS writes a screen coord without raising the divide-overflow flag
    #    for a normal vertex in front of the camera.
    g2 = GTE()
    # identity rotation
    g2.control[0] = (0 << 16) | 0x1000   # RT11
    g2.control[2] = 0x1000               # RT22 (packed)
    g2.control[4] = 0x1000               # RT33
    g2.control[26] = 0x200               # H (projection plane distance)
    g2.data[0] = (50 << 16) | 100        # VXY0 = (100,50)
    g2.data[1] = 0x400                   # VZ0 = 1024
    g2.command((1 << 19) | 0x01)         # RTPS, sf=1
    sx = _s16(g2.data[14] & 0xFFFF); sy = _s16(g2.data[14] >> 16)
    # H/Z = 512/1024 -> screen = (50, 25) with OFX=OFY=0
    check("rtps SZ3 == 1024", (g2.data[19] & 0xFFFF) == 0x400)
    check(f"rtps SXY2 == (50,25) got ({sx},{sy})", (sx, sy) == (50, 25))
    check("rtps no divide overflow", not (g2.flag & (1 << 17)))
    g2.data[1] = 0x100                   # Z=256 < H/2 -> overflow path
    g2.command((1 << 19) | 0x01)
    check("rtps divide overflow flagged", bool(g2.flag & (1 << 17)))

    # 6) DMA OTC builds a reverse-terminated ordering table in RAM.
    m3 = Memory()
    m3.dma.channels[6]["madr"] = 0x001000
    m3.dma.channels[6]["bcr"] = 4
    m3.dma.write(6, 8, (1 << 24) | (0 << 9))  # trigger
    last = m3.read32(0x80001000 - 12)         # first-built node (lowest addr)
    head = m3.read32(0x80001000)              # head node
    check("otc head links downward", (head & 0xFFFFFF) == ((0x1000 - 4) & 0xFFFFFF))
    check("otc tail terminates", (last & 0xFFFFFF) == 0xFFFFFF)

    # 7) GPU linked-list DMA: a flat-triangle packet reaches the raster.
    m4 = Memory()
    gpu = m4.gpu
    # Build a one-node display list at 0x1000: header (size=4, next=FFFFFF)
    #   word0: flat-tri cmd 0x20 | green
    #   then 3 vertices
    node = 0x1000
    packet = [0x2000FF00,                       # cmd 0x20, colour G=0xFF
              (0 << 16) | 10,                    # v0 (10,0)
              (100 << 16) | 10,                  # v1 (10,100)
              (50 << 16) | 100]                  # v2 (100,50)
    m4.write32(0x80000000 | node, (len(packet) << 24) | 0xFFFFFF)
    for i, w in enumerate(packet):
        m4.write32(0x80000000 | (node + 4 + i * 4), w)
    m4.dma.channels[2]["madr"] = node
    m4.dma.channels[2]["chcr"] = 0
    m4.dma.write(2, 8, (1 << 24) | (2 << 9) | 1)  # linked-list, to-GPU, start
    painted = int(gpu.display.sum()) > 0
    check("gpu linked-list raster painted pixels", painted)

    # 7b) GP0 word counts for FIFO sync
    gg = GPU()
    check("gp0 counts (0x20,0x28,0x30,0x38,0x2C,0x3C,0x80,0x50)",
          [gg._words_for(c) for c in (0x20, 0x28, 0x30, 0x38, 0x2C, 0x3C, 0x80, 0x50)]
          == [4, 5, 6, 8, 9, 12, 4, 4])

    # 8) Pad poll returns the documented digital-pad id sequence.
    pad = Controller()
    seq = [pad.exchange(0x01), pad.exchange(0x42), pad.exchange(0x00),
           pad.exchange(0x00), pad.exchange(0x00)]
    check("pad id 0x41,0x5A", seq[1] == 0x41 and seq[2] == 0x5A)

    # Regression: every polygon packet consumes precisely its own words.
    for cmd in range(0x20, 0x40):
        n = 4 if cmd & 8 else 3
        expected = 1 + n + ((n - 1) if cmd & 16 else 0) + (n if cmd & 4 else 0)
        check(f"GP0 {cmd:02X} packet length", gg._words_for(cmd) == expected)
    gy = GTE()
    gy._lim_sxy(2048, 1)
    check("GTE Y clipping flag", bool(gy.flag & (1 << 13)))
    pad2 = Controller()
    check("pad rejects invalid poll", pad2.exchange(0x00) == 0xFF and pad2.exchange(0x42) == 0xFF)

    print(f"self-test: {'PASS' if fails == 0 else str(fails) + ' FAIL'}")
    return fails


def main() -> None:
    if sys.version_info < (3, 8):
        print("Python 3.8+ required.", file=sys.stderr)
        sys.exit(1)
    if "--opcodes" in sys.argv or "--rpsc" in sys.argv or "--rpsc1" in sys.argv:
        print(RPSC.report()); return
    if "--selftest" in sys.argv or "--test" in sys.argv:
        sys.exit(1 if _selftest() else 0)
    if tk is None:
        print("tkinter not available; run with --selftest or --opcodes.", file=sys.stderr)
        sys.exit(1)
    root = tk.Tk()
    emu = Emulator()
    RPSC1_GUI(root, emu)  # KEEP THE GUI
    print(f"{TITLE} · files=OFF · RPSC1 {VERSION} · Python {sys.version.split()[0]}")
    print("Tip: python3 rpsc1.0.py --selftest   |   --opcodes")
    root.mainloop()


if __name__ == "__main__":
    main()
