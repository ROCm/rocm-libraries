# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A1: rocKE port of the FlyDSL Qwen3.8-27B decode megakernel ``qwen_m8_replay`` (gfx950 / MI355X).

Reference kernel (MIT, Advanced Micro Devices, Inc.): ``atom/models/qwen38_27b/mono/kernels/verify_m8.py`` with the
replay-record transform ``gen_replay_records.py`` applied, https://github.com/proaditya/ATOM/tree/qwen38-megakernels
(commit 352229d6). "source" line numbers below refer to that transformed FlyDSL source.

One rocKE builder, one code object: FlyDSL's ABI (exactly one pointer-table argument, 65 rows x 45 i64 slots, endpoint
row 64), launch 256 x 256, 8 rows, LDS 159,792 B, no scratch, failure word ``native[289]``, every poll bounded.

    kernel = build_full(spec)     # spec in specs(); A1 is p2c_full_a1 (see build_a1.py)

Body (FlyDSL source line numbers; every stage an exact transliteration, op by op):
  census (237-269) -> prologue: embedding rows 0-7, residual0 / initial mixer zero, entry grid ``cmd*2048+1``
  (273-293) -> 64 layers (296-3034), D/D/D/A dispatch, epoch ``cmd*2048+2+12L-L//4``:
    layer-0 RMS | fold(L) (321-406 / 503-623) -> GDN: QKV + conv + BA (651-853), core / core tail / peer hand-off /
    finalize (855-1148, 2638-2686) | attention: QGKV (1697-1802), qk (Q/K norm + RoPE, new K/V, gate; 1804-1874),
    partials (2095-2327), all-256 native grid (98-137, 2872), reduce + mixer publication (2329-2410, 2876-2884) ->
    relay_mixer (2766-2792) -> output (1150-1270) -> post-norm owners (2958-2995) -> Up (1272-1553) -> relay
    (2716-2744) -> Down (1555-1688) -> layer 63: final Down grid, fold (1690-1695), grid ``cmd*2048+754``, final RMS
    rows 0-7, grid ``cmd*2048+755``, LM head stream -> fp32 logits (2503-2612, 3002-3025) -> layer-end control words
    (Int32 ``control+8`` / ``control+16+block``, 3026-3034).
Deviations from the reference kernel (memory ordering, see README.md): residual1 ``fc_od`` stores aux 17
(``sc0 sc1``) and write-through mixer zeros (``global_store_short ... sc0 sc1``).
Exact-order optimizations kept from the port: M3 = D3 (Down activation: one full-lane x4 load + row_ror:8 DPP),
S6 (GDN QKV epilogue operands with no producer issued before the fold wait), P1-fold (fold chunk / row-sum polls
issue the counter and failure-word loads under one ``vmcnt(0)``).
Knob ``a1``: one compiler-visible ``s_waitcnt vmcnt(0)`` after the attention Q loads, before the partial-tile loop
(LLVM then drops the in-loop Q guards that also waited for the K/V prefetch).
Scalar-predicate rule (compiler partial-EXEC hazard, see KNOWN_ISSUES.md): wave index, statuses (``st_load``) and
position / page / task bounds are SGPR values (readfirstlane); counter publications and the layer-end control words are
lane-0 asm blocks that narrow and restore EXEC inside the asm; no lane-dependent region ends a loop body.

Diagnostic twins (never timed): ``pin`` = dumps (output slice words endpoint cell 6, owners' mixer rows cell 7, core
row squares cell 9) + atomic order pinned (output K-slices in peer order, native 8192 + 16 tile; BA K-split tiles in
tile order, native 9728 + 16 tile); ``mxpin`` = mixer-order pin only. Every twin-only operation is under ``if PIN:`` /
``if MXPIN:``; with no knob the builder emits the base port kernel.

The intrinsic-call extension below adds generic LLVM intrinsic / instruction ops to rocKE's python lowering backend
(rocKE has no generic intrinsic op; this kernel lowers with backend="python").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from rocke.core import lower_llvm as LL
from rocke.core.ir import (
    BF16,
    F16,
    F32,
    I8,
    I16,
    I32,
    I64,
    IRBuilder,
    PtrType,
    Type,
    Value,
    VectorType,
)

# ================================================================ intrinsic-call extension
DECLS = {
    "p1c.buf.ld.v4i32": "declare <4 x i32> @llvm.amdgcn.raw.buffer.load.v4i32(<4 x i32>, i32, i32, i32 immarg)",
    "p1c.buf.ld.i32": "declare i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32>, i32, i32, i32 immarg)",
    "p1c.buf.ld.i8": "declare i8 @llvm.amdgcn.raw.buffer.load.i8(<4 x i32>, i32, i32, i32 immarg)",
    "p1c.buf.st.i8": "declare void @llvm.amdgcn.raw.buffer.store.i8(i8, <4 x i32>, i32, i32, i32 immarg)",
    "p1c.exp.f32": "declare float @llvm.exp.f32(float)",
    "p1c.maximum.f32": "declare float @llvm.maximum.f32(float, float)",
    "p1c.ds.swizzle": "declare i32 @llvm.amdgcn.ds.swizzle(i32, i32 immarg)",
    "p1c.permlane16.swap": "declare { i32, i32 } @llvm.amdgcn.permlane16.swap(i32, i32, i1 immarg, i1 immarg)",
    "p1c.ds.bpermute": "declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)",
    "p1c.mbcnt.lo": "declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)",
    "p1c.mbcnt.hi": "declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)",
    "p1c.mfma.scale": (
        "declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32>, "
        "<8 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32)"
    ),
    "p1c.buf.st.v4i32": "declare void @llvm.amdgcn.raw.buffer.store.v4i32(<4 x i32>, <4 x i32>, i32, i32, i32 immarg)",
    "p1c.rsq.f32": "declare float @llvm.amdgcn.rsq.f32(float)",
    "p1c.fma.f32": "declare float @llvm.fma.f32(float, float, float)",
    "p1c.fma.v8f32": "declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>)",
    "p1c.permlane32.swap": "declare { i32, i32 } @llvm.amdgcn.permlane32.swap(i32, i32, i1 immarg, i1 immarg)",
    # GDN core sum16: FlyDSL rocdl.update_dpp, row_shl with zero fill
    "p1c.update.dpp.f32": (
        "declare float @llvm.amdgcn.update.dpp.f32(float, float, i32 immarg, i32 immarg, i32 immarg, "
        "i1 immarg)"
    ),
    # attention qk / partials / reduce: FlyDSL's intrinsic and OCML forms
    "p1c.exp2.f32": "declare float @llvm.amdgcn.exp2.f32(float)",
    "p1c.mfma.bf16.32": (
        "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat>, <8 x bfloat>, "
        "<4 x float>, i32 immarg, i32 immarg, i32 immarg)"
    ),
    "p1c.mfma.bf16.1k": (
        "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16>, <4 x i16>, <4 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "p1c.ds.read.tr16": "declare <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3))",
    "p1c.buf.ld.i16": "declare i16 @llvm.amdgcn.raw.buffer.load.i16(<4 x i32>, i32, i32, i32 immarg)",
    "p1c.readlane": "declare i32 @llvm.amdgcn.readlane.i32(i32, i32)",
    "p1c.ocml.pow": "declare float @__ocml_pow_f32(float, float)",
    "p1c.ocml.cos": "declare float @__ocml_cos_f32(float)",
    "p1c.ocml.sin": "declare float @__ocml_sin_f32(float)",
    "p1c.ocml.exp2": "declare float @__ocml_exp2_f32(float)",
}
CALLEE = {
    "p1c.buf.ld.v4i32": "llvm.amdgcn.raw.buffer.load.v4i32",
    "p1c.buf.ld.i32": "llvm.amdgcn.raw.buffer.load.i32",
    "p1c.buf.ld.i8": "llvm.amdgcn.raw.buffer.load.i8",
    "p1c.buf.st.i8": "llvm.amdgcn.raw.buffer.store.i8",
    "p1c.exp.f32": "llvm.exp.f32",
    "p1c.maximum.f32": "llvm.maximum.f32",
    "p1c.ds.swizzle": "llvm.amdgcn.ds.swizzle",
    "p1c.permlane16.swap": "llvm.amdgcn.permlane16.swap",
    "p1c.ds.bpermute": "llvm.amdgcn.ds.bpermute",
    "p1c.mbcnt.lo": "llvm.amdgcn.mbcnt.lo",
    "p1c.mbcnt.hi": "llvm.amdgcn.mbcnt.hi",
    "p1c.mfma.scale": "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32",
    "p1c.buf.st.v4i32": "llvm.amdgcn.raw.buffer.store.v4i32",
    "p1c.rsq.f32": "llvm.amdgcn.rsq.f32",
    "p1c.fma.f32": "llvm.fma.f32",
    "p1c.fma.v8f32": "llvm.fma.v8f32",
    "p1c.permlane32.swap": "llvm.amdgcn.permlane32.swap",
    "p1c.update.dpp.f32": "llvm.amdgcn.update.dpp.f32",
    "p1c.exp2.f32": "llvm.amdgcn.exp2.f32",
    "p1c.mfma.bf16.32": "llvm.amdgcn.mfma.f32.16x16x32.bf16",
    "p1c.mfma.bf16.1k": "llvm.amdgcn.mfma.f32.16x16x16bf16.1k",
    "p1c.ds.read.tr16": "llvm.amdgcn.ds.read.tr16.b64.v4f16",
    "p1c.buf.ld.i16": "llvm.amdgcn.raw.buffer.load.i16",
    "p1c.readlane": "llvm.amdgcn.readlane.i32",
    "p1c.ocml.pow": "__ocml_pow_f32",
    "p1c.ocml.cos": "__ocml_cos_f32",
    "p1c.ocml.sin": "__ocml_sin_f32",
    "p1c.ocml.exp2": "__ocml_exp2_f32",
}


def _op_p1c_call(self, op) -> None:
    key = op.attrs["key"]
    self._need(key)
    imm = {int(k): v for k, v in op.attrs.get("imm", {}).items()}
    n = len(op.operands) + len(imm)
    it = iter(op.operands)
    args = [imm[i] if i in imm else self._operand_with_type(next(it)) for i in range(n)]
    call = f"call {op.attrs['ret']} @{CALLEE[key]}({', '.join(args)})"
    cur = self._current()
    if not op.results:
        cur.emit(f"  {call}")
    elif op.attrs["ret"].startswith("{"):
        tmp = self._fresh("p1cs")
        cur.emit(f"  {tmp} = {call}")
        for i, r in enumerate(op.results):
            cur.emit(f"  {r.name} = extractvalue {op.attrs['ret']} {tmp}, {i}")
    else:
        cur.emit(f"  {op.results[0].name} = {call}")


def _op_p1c_inst(self, op) -> None:
    t = op.attrs["text"].format(*[self._operand(v) for v in op.operands])
    self._current().emit(f"  {op.results[0].name} = {t}" if op.results else f"  {t}")


def install() -> None:
    for k, v in DECLS.items():
        LL._INTRINSIC_DECLS.setdefault(k, v)
    LL._Lowerer._op_p1c_call = _op_p1c_call
    LL._Lowerer._op_p1c_inst = _op_p1c_inst


install()


def call(
    b: IRBuilder,
    key: str,
    operands: Sequence[Value],
    ret: str,
    rtypes: Sequence[Type] = (),
    imm: dict | None = None,
    hint: str = "ic",
):
    op = b._op(
        "p1c.call",
        list(operands),
        list(rtypes),
        attrs={
            "key": key,
            "ret": ret,
            "imm": {str(k): v for k, v in (imm or {}).items()},
        },
        result_name_hint=hint,
    )
    return op.results


def inst(
    b: IRBuilder,
    text: str,
    operands: Sequence[Value],
    rtype: Type | None = None,
    hint: str = "pi",
):
    op = b._op(
        "p1c.inst",
        list(operands),
        [rtype] if rtype is not None else [],
        attrs={"text": text},
        result_name_hint=hint,
    )
    return op.results[0] if rtype is not None else None


V4I = VectorType(I32, 4)
V4F = VectorType(F32, 4)
PTR1 = PtrType(I8, "global")
PTR3 = PtrType(I8, "lds")


class Ext:
    """Typed wrappers (FlyDSL's intrinsic forms)."""

    def __init__(self, b: IRBuilder):
        self.b = b

    def buf_ld16(
        self, desc, off, aux
    ):  # llvm.amdgcn.raw.buffer.load.v4i32(desc, off, 0, aux)
        return call(
            self.b,
            "p1c.buf.ld.v4i32",
            [desc, off],
            "<4 x i32>",
            [V4I],
            {2: "i32 0", 3: f"i32 {aux}"},
            "bl16",
        )[0]

    def buf_ld4(self, desc, off, aux):
        return call(
            self.b,
            "p1c.buf.ld.i32",
            [desc, off],
            "i32",
            [I32],
            {2: "i32 0", 3: f"i32 {aux}"},
            "bl4",
        )[0]

    def buf_ld1(self, desc, off, aux):
        return call(
            self.b,
            "p1c.buf.ld.i8",
            [desc, off],
            "i8",
            [I8],
            {2: "i32 0", 3: f"i32 {aux}"},
            "bl1",
        )[0]

    def buf_st1(self, val_i8, desc, off, aux):
        call(
            self.b,
            "p1c.buf.st.i8",
            [val_i8, desc, off],
            "void",
            [],
            {3: "i32 0", 4: f"i32 {aux}"},
        )

    def exp(self, x):
        return call(self.b, "p1c.exp.f32", [x], "float", [F32], None, "exp")[0]

    def maximum(self, x, y):
        return call(self.b, "p1c.maximum.f32", [x, y], "float", [F32], None, "fmaxm")[0]

    def swizzle(self, x, pattern):
        return call(
            self.b, "p1c.ds.swizzle", [x], "i32", [I32], {1: f"i32 {pattern}"}, "swz"
        )[0]

    def permlane16_swap(self, x, y):
        return call(
            self.b,
            "p1c.permlane16.swap",
            [x, y],
            "{ i32, i32 }",
            [I32, I32],
            {2: "i1 false", 3: "i1 false"},
            "pl16",
        )

    def bpermute(self, addr, x):
        return call(self.b, "p1c.ds.bpermute", [addr, x], "i32", [I32], None, "bperm")[
            0
        ]

    def mbcnt_lo(self, m, c):
        return call(self.b, "p1c.mbcnt.lo", [m, c], "i32", [I32], None, "mbl")[0]

    def mbcnt_hi(self, m, c):
        return call(self.b, "p1c.mbcnt.hi", [m, c], "i32", [I32], None, "mbh")[0]

    def mfma_scale(self, a8, b8, acc, sa, sb, cbsz=0, blgp=4):
        return call(
            self.b,
            "p1c.mfma.scale",
            [a8, b8, acc, sa, sb],
            "<4 x float>",
            [V4F],
            {3: f"i32 {cbsz}", 4: f"i32 {blgp}", 5: "i32 0", 7: "i32 0"},
            "mfs",
        )[0]

    # ---- fold / QKV
    def buf_st16(
        self, val_v4i32, desc, off, aux
    ):  # llvm.amdgcn.raw.buffer.store.v4i32(v, desc, off, 0, aux)
        call(
            self.b,
            "p1c.buf.st.v4i32",
            [val_v4i32, desc, off],
            "void",
            [],
            {3: "i32 0", 4: f"i32 {aux}"},
        )

    def rsq(self, x):
        return call(self.b, "p1c.rsq.f32", [x], "float", [F32], None, "rsq")[0]

    def fma(self, a, b_, c):
        return call(self.b, "p1c.fma.f32", [a, b_, c], "float", [F32], None, "fma")[0]

    def fma8(self, a, b_, c):
        return call(
            self.b,
            "p1c.fma.v8f32",
            [a, b_, c],
            "<8 x float>",
            [VectorType(F32, 8)],
            None,
            "fma8",
        )[0]

    def permlane32_swap(self, x, y):
        return call(
            self.b,
            "p1c.permlane32.swap",
            [x, y],
            "{ i32, i32 }",
            [I32, I32],
            {2: "i1 false", 3: "i1 false"},
            "pl32",
        )

    # ---- GDN core
    def update_dpp_f32(
        self, x, ctrl
    ):  # llvm.amdgcn.update.dpp.f32(0.0, x, ctrl, 15, 15, true)
        return call(
            self.b,
            "p1c.update.dpp.f32",
            [x],
            "float",
            [F32],
            {
                0: "float 0.000000e+00",
                2: f"i32 {ctrl}",
                3: "i32 15",
                4: "i32 15",
                5: "i1 true",
            },
            "dpp",
        )[0]

    def fpext_bf16x8(self, v):  # <8 x bfloat> -> <8 x float>
        return inst(
            self.b,
            "fpext <8 x bfloat> {0} to <8 x float>",
            [v],
            VectorType(F32, 8),
            "fpx",
        )

    # FlyDSL addresses plain memory as typed pointers: one ``inttoptr`` of the base, then ``getelementptr <elem>``
    # (i32 index) per access, so LLVM's load/store vectorizer sees adjacent accesses (e.g. 4 conv taps -> one dwordx2).
    def ptr1(self, addr64):  # global base (a table entry) as ptr addrspace(1)
        return inst(
            self.b, "inttoptr i64 {0} to ptr addrspace(1)", [addr64], PTR1, "gp"
        )

    def ptr3(self, addr32):  # LDS base (the arena) as ptr addrspace(3)
        return inst(
            self.b, "inttoptr i32 {0} to ptr addrspace(3)", [addr32], PTR3, "lp"
        )

    def gep(self, p, elem: str, idx, lds=False):  # getelementptr <elem>, ptr p, i32 idx
        a = 3 if lds else 1
        return inst(
            self.b,
            f"getelementptr {elem}, ptr addrspace({a}) {{0}}, i32 {{1}}",
            [p, idx],
            PTR3 if lds else PTR1,
            "gep",
        )

    def load(self, p, ty: str, rtype: Type, align: int, lds=False):
        return inst(
            self.b,
            f"load {ty}, ptr addrspace({3 if lds else 1}) {{0}}, align {align}",
            [p],
            rtype,
            "ld",
        )

    def store(self, p, val, ty: str, align: int, lds=False):
        inst(
            self.b,
            f"store {ty} {{1}}, ptr addrspace({3 if lds else 1}) {{0}}, align {align}",
            [p, val],
        )

    def atomic_fadd_agent(
        self, p, val
    ):  # fx.atomic_add(f32, Monotonic, syncscope agent) on a typed pointer
        return inst(
            self.b,
            'atomicrmw fadd ptr addrspace(1) {0}, float {1} syncscope("agent") monotonic, align 4',
            [p, val],
            F32,
            "afa",
        )

    # ---- attention
    def exp2(
        self, x
    ):  # rocdl.exp2 (llvm.amdgcn.exp2.f32: v_exp_f32, no denormal scaling)
        return call(self.b, "p1c.exp2.f32", [x], "float", [F32], None, "ex2")[0]

    def mfma_bf16_32(
        self, a8, b8, acc
    ):  # rocdl.mfma_f32_16x16x32_bf16(<8 x bfloat>, <8 x bfloat>, <4 x float>)
        return call(
            self.b,
            "p1c.mfma.bf16.32",
            [a8, b8, acc],
            "<4 x float>",
            [V4F],
            {3: "i32 0", 4: "i32 0", 5: "i32 0"},
            "mqk",
        )[0]

    def mfma_bf16_1k(
        self, a4, b4, acc
    ):  # rocdl.mfma_f32_16x16x16bf16_1k(<4 x i16>, <4 x i16>, <4 x float>)
        return call(
            self.b,
            "p1c.mfma.bf16.1k",
            [a4, b4, acc],
            "<4 x float>",
            [V4F],
            {3: "i32 0", 4: "i32 0", 5: "i32 0"},
            "mpv",
        )[0]

    def ds_read_tr16(
        self, p3
    ):  # <4 x half> llvm.amdgcn.ds.read.tr16.b64.v4f16, returned as <4 x i16>
        h = call(
            self.b,
            "p1c.ds.read.tr16",
            [p3],
            "<4 x half>",
            [VectorType(F16, 4)],
            None,
            "tr16",
        )[0]
        return inst(
            self.b,
            "bitcast <4 x half> {0} to <4 x i16>",
            [h],
            VectorType(I16, 4),
            "tri",
        )

    def buf_ld2(
        self, desc, off, aux
    ):  # llvm.amdgcn.raw.buffer.load.i16(desc, off, 0, aux)
        return call(
            self.b,
            "p1c.buf.ld.i16",
            [desc, off],
            "i16",
            [I16],
            {2: "i32 0", 3: f"i32 {aux}"},
            "bl2",
        )[0]

    def readlane(self, v, idx):
        return call(self.b, "p1c.readlane", [v, idx], "i32", [I32], None, "rdl")[0]

    def ocml(
        self, name, *args
    ):  # FlyDSL fmath.powf / cos / sin / exp2 -> __ocml_<name>_f32 (linked at build time)
        return call(self.b, f"p1c.ocml.{name}", list(args), "float", [F32], None, name)[
            0
        ]


# ================================================================ the kernel
LDS_BYTES = 159744
CURRENT_WORD, CENSUS_WORD, CHUNK_WORD = 304, 3472, 3728
MAX_COMMAND_EPOCH = 1048575
DESC_W3 = 159744
DOWN_DELTA = (
    (3, 56),
    (6, 56),
    (9, 40),
    (12, 40),
    (15, 40),
    (18, 32),
    (21, 32),
    (24, 32),
    (27, 40),
    (30, 40),
    (33, 32),
    (36, 32),
    (39, 32),
    (42, 16),
    (45, 16),
    (48, 16),
    (49, -40),
    (50, -32),
    (52, -40),
    (53, -32),
    (55, -40),
    (56, -32),
    (58, -16),
    (59, -56),
    (61, -16),
    (62, -56),
    (64, -16),
    (65, -32),
    (67, -40),
    (68, -32),
    (70, -40),
    (71, -32),
)


def min_tree(ok: str, tmps: tuple, shifts) -> str:
    """Inline-asm text of FlyDSL's compiled cross-lane min tree ``ok = min(ok, ok.shuffle_xor(s, 64))`` (FlyDSL's ISA):
    s < 16: ds_swizzle (bitmask mode, xor s) + lgkmcnt(0) + v_min_i32; s = 16 / 32: copies, v_permlane16/32_swap,
    select(p0 == v, p1, p0) (v_cmp + v_cndmask), v_min_i32. ``tmps`` = (tA, tB) asm operand names.
    """
    ta, tb = tmps
    t = ""
    for s in shifts:
        if s < 16:
            t += f" ds_swizzle_b32 {ta}, {ok} offset:{(s << 10) | 0x1F}\n s_waitcnt lgkmcnt(0)\n v_min_i32 {ok}, {ok}, {ta}\n"
        else:
            t += (
                f" v_mov_b32 {ta}, {ok}\n v_mov_b32 {tb}, {ok}\n s_nop 1\n v_permlane{s}_swap_b32 {ta}, {tb}\n"
                f" v_cmp_eq_u32 vcc, {ta}, {ok}\n s_nop 1\n v_cndmask_b32 {ta}, {ta}, {tb}, vcc\n"
                f" v_min_i32 {ok}, {ok}, {ta}\n"
            )
    return t


@dataclass(frozen=True)
class KernelSpec:
    name: str = "p2c_full"
    symbol: str = "p2c_full"
    knobs: tuple = field(
        default_factory=tuple
    )  # () = production; ("pin",) = the diagnostic twin

    def has(self, k: str) -> bool:
        return k in self.knobs

    def as_dict(self) -> dict:
        return {"name": self.name, "symbol": self.symbol, "knobs": list(self.knobs)}


def build_full(spec: KernelSpec):
    """The p2c_full kernel (``spec.knobs`` empty) or its diagnostic pin twin (``("pin",)``)."""
    assert spec.knobs in (
        (),
        ("pin",),
        ("mxpin",),
        ("a1",),
        ("a1", "mxpin"),
        ("a1", "pin"),
    ), spec.knobs
    PIN = spec.has(
        "pin"
    )  # diagnostic twin: dumps + pinned atomic order (DUMP and PIN together)
    DUMP = PIN
    MXPIN = PIN or spec.has(
        "mxpin"
    )  # mixer-only twin: output K-slices in peer order; no BA pin, no dumps
    A1 = spec.has(
        "a1"
    )  # A1: one compiler-visible vmcnt(0) after the partials-stage Q loads
    b = IRBuilder(spec.symbol)
    b.kernel.attrs["max_workgroup_size"] = 256
    tab = b.param("pointer_table", PtrType(I64, "global"))
    tid = b.thread_id_x()
    block = b.block_id_x()
    arena = b.smem_alloc(I8, [LDS_BYTES], name_hint="arena")
    status = b.smem_alloc(I32, [12], name_hint="status")
    x = Ext(b)
    ci = b.const_i32
    c0, c1 = ci(0), ci(1)

    # ------------------------------------------------------------ helpers (FlyDSL names)
    def zx(c):  # bool -> i32 (FlyDSL .to(fx.Int32))
        return b.zext(c, I32)

    def asm(t, cons, ops=(), rt=None, se=True, hint="asm"):
        return b.inline_asm(
            t, cons, list(ops), rt, sideeffect=se, result_name_hint=hint
        )

    def rfl(v):
        return b.readfirstlane(v)

    def i64(v):
        return b.sext(v, I64) if v.type == I32 else v

    def raw_table_address(row, slot):
        r = row if not isinstance(row, int) else ci(row)
        idx = b.add(b.mul(r, ci(45)), ci(slot))
        p = b.global_ptr_to_i64(b.global_ptr_add(tab, b.mul(i64(idx), b.const_i64(8))))
        return asm(
            "s_load_dwordx2 $0, $1, 0x0\ns_waitcnt lgkmcnt(0)",
            "=&s,s,~{memory}",
            [p],
            I64,
            hint="tab",
        )

    def word(base64, w):  # i64 address of int32 word w (int or Value)
        off = (
            b.const_i64(4 * w) if isinstance(w, int) else b.mul(i64(w), b.const_i64(4))
        )
        return b.add(base64, off)

    def coherent_u32(a):
        return asm(
            "global_load_dword $0, $1, off sc0 sc1\ns_waitcnt vmcnt(0)",
            "=&v,v,~{memory}",
            [a],
            I32,
            hint="coh",
        )

    def system_store_u32(a, v):
        asm("global_store_dword $0, $1, off sc0 sc1", "v,v,~{memory}", [a, v])

    def system_store_u64(a, v):
        asm("global_store_dwordx2 $0, $1, off sc0 sc1", "v,v,~{memory}", [a, v])

    def plain_store_u32(a, v):  # FlyDSL fx.ptr_store to global (no cache bits)
        asm("global_store_dword $0, $1, off", "v,v,~{memory}", [a, v])

    def system_atomic_add(a):
        return asm(
            "flat_atomic_add $0, $1, $2 sc0 sc1\ns_waitcnt vmcnt(0) lgkmcnt(0)",
            "=&v,v,v,~{memory}",
            [a, c1],
            I32,
            hint="fat",
        )

    def system_atomic_add_nw(a):
        asm("global_atomic_add $0, $1, off sc1", "v,v,~{memory}", [a, c1])

    def lane0_atomic_add_nw(a):
        """``system_atomic_add_nw`` by lane 0 of a full-EXEC wave without a divergent region: EXEC is narrowed
        and restored inside the asm, so the layer loop does not end in an EXEC-masked flow block (where the register
        allocator rematerialized loop-invariant VGPR constants under partial EXEC; see KNOWN_ISSUES.md).
        """
        asm(
            "s_mov_b64 $0, exec\n s_and_b64 exec, exec, 1\n global_atomic_add $1, $2, off sc1\n s_mov_b64 exec, $0",
            "=&s,v,v,~{memory},~{scc}",
            [a, c1],
            I64,
            hint="l0add",
        )

    def wait_vm(n):
        asm(f"s_waitcnt vmcnt({n})", "~{memory}")

    def barrier():  # FlyDSL gpu.barrier: workgroup release fence, s_barrier, workgroup acquire fence
        b.fence(scope="workgroup", ordering="release")
        b.s_barrier_bare(fence="none")
        b.fence(scope="workgroup", ordering="acquire")

    def st_store(slot, v):
        s = slot if not isinstance(slot, int) else ci(slot)
        b.smem_store_vN(status, [s], v, 1)

    def st_load(
        slot,
    ):  # WG status word (read by every lane after a barrier): readfirstlane keeps the branches on it
        s = (
            slot if not isinstance(slot, int) else ci(slot)
        )  # scalar (no EXEC-masked region at the latch)
        return rfl(b.vec_extract(b.smem_load_vN(status, s, dtype=I32, n=1), 0))

    def lds_ld16(byte_off):  # <16 x i8> -> <4 x i32>
        return b.vec_bitcast(b.smem_load_vN(arena, byte_off, dtype=I8, n=16), V4I)

    def lds_st_f32(byte_off, v):
        b.smem_store_vN(
            arena, [byte_off], b.vec_bitcast(b.vec_pack([v], F32), VectorType(I8, 4)), 4
        )

    def lds_ld_f32(byte_off):
        return b.vec_extract(
            b.vec_bitcast(
                b.smem_load_vN(arena, byte_off, dtype=I8, n=4), VectorType(F32, 1)
            ),
            0,
        )

    arena_addr = b.trunc(b.smem_addr_of(arena), I32)

    def descriptor(addr64, size):
        lo = rfl(b.trunc(addr64, I32))
        hi = rfl(b.trunc(b.lshr(addr64, b.const_i64(32)), I32))
        sz = size if not isinstance(size, int) else ci(size)
        return b.vec_pack([lo, hi, sz, ci(DESC_W3)], I32)

    def dma(dst, off, desc, ins="buffer_load_dwordx4", bits=" nt"):
        asm(
            f"s_mov_b32 m0, $0\ns_nop 0\n{ins} $1, $2, 0 offen{bits} lds",
            "s,v,s,~{memory},~{m0}",
            [rfl(dst), off, desc],
        )

    def pin(v):
        return asm("", "=&v,0,~{memory}", [v], v.type, hint="pin")

    iy = [0]

    def if_yield(cond, vals, fn):
        """FlyDSL scf.if with results (rocKE scf_if has none): a 0/1-trip scf.for carrying ``vals``."""
        iy[0] += 1
        n = iy[0]
        lp_ = b.scf_for_iter(
            c0,
            zx(cond),
            c1,
            [(f"iy{n}_{i}", v) for i, v in enumerate(vals)],
            iv_name=f"iy{n}",
            elide_trailing_barrier=False,
        )
        with lp_ as (_, cur):
            b.scf_yield(*fn(list(cur)))
        return list(lp_.results)

    # fail_native: atomic CAS native[289] 0 -> 2, agent acq_rel (LLVM's gfx950 sequence for that cmpxchg)
    def fail_native():
        asm(
            "buffer_wbl2 sc1\ns_waitcnt vmcnt(0)\nglobal_atomic_cmpswap $0, $1, off\ns_waitcnt vmcnt(0)\nbuffer_inv sc1",
            "v,v,~{memory}",
            [word(native, 289), b.const_i64(2)],
        )

    # FlyDSL's compiled while loops read the failure word once BEFORE the loop (``failed = coherent_u32(native + 289)``,
    # source 106 / 150 / 257) and skip the loop when it is already set: PRE_T is that read + the loop head.
    PRE_T = (
        " global_load_dword $2, $4, off sc0 sc1\n s_waitcnt vmcnt(0)\n s_mov_b32 $0, 0\n v_mov_b32 $1, 0\n"
        " v_cmp_ne_u32 vcc, 0, $2\n s_cbranch_vccnz 2f\n1:\n"
    )

    def poll_ge(addr, target, limit, failed_addr=None, scalar=False):
        """FlyDSL while-loop poll (one lane's view, issuing lane = tid 0): returns ready (i32 0/1).
        scalar: s_dcache_inv + s_load_dword glc (uniform address), bounded by ``limit``; else coherent global loads of
        the word and of the failure word ``failed_addr`` each iteration, FlyDSL's pre-loop failure-word read (no poll
        when it is already set), bounded by ``limit``."""
        if scalar:
            t = (
                "s_mov_b32 $0, 0\n1:\n s_dcache_inv\n s_load_dword $1, $2, 0x0 glc\n s_waitcnt lgkmcnt(0)\n"
                " s_sleep 1\n s_add_u32 $0, $0, 1\n s_cmp_ge_i32 $1, $3\n s_cbranch_scc1 2f\n"
                " s_cmp_lt_u32 $0, $4\n s_cbranch_scc1 1b\n2:"
            )
            r = b.inline_asm_multi(
                t,
                "=&s,=&s,s,s,s,~{scc}",
                [addr, target, limit],
                result_types=[I32, I32],
                result_name_hint="spoll",
            )
            return zx(b.cmp_ge(r[1], target))
        t = (
            PRE_T + " global_load_dword $1, $3, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
            " global_load_dword $2, $4, off sc0 sc1\n s_waitcnt vmcnt(0)\n s_sleep 1\n s_add_u32 $0, $0, 1\n"
            " v_cmp_ge_i32 vcc, $1, $5\n s_cbranch_vccnz 2f\n v_cmp_ne_u32 vcc, 0, $2\n s_cbranch_vccnz 2f\n"
            " s_cmp_lt_u32 $0, $6\n s_cbranch_scc1 1b\n2:"
        )
        r = b.inline_asm_multi(
            t,
            "=&s,=&v,=&v,v,v,v,s,~{vcc},~{scc}",
            [addr, failed_addr, target, limit],
            result_types=[I32, I32, I32],
            result_name_hint="vpoll",
        )
        return zx(b.land(b.cmp_ge(r[1], target), b.cmp_eq(r[2], c0)))

    # ------------------------------------------------------------ entry (source 40-96, 237-269)
    native = raw_table_address(64, 42)
    xcc = asm(
        "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 16)",
        "=s,~{memory}",
        [],
        I32,
        hint="xcc",
    )
    safe_xcc = b.smax(c0, b.smin(xcc, ci(7)))
    command_epoch = coherent_u32(word(native, 288))
    with b.scf_if(b.cmp_eq(tid, c0)):
        system_store_u32(word(native, b.add(ci(CENSUS_WORD), block)), xcc)
        bad = b.lor(
            b.lor(b.cmp_lt(xcc, c0), b.cmp_ge(xcc, ci(8))),
            b.lor(
                b.cmp_lt(command_epoch, c1),
                b.cmp_gt(command_epoch, ci(MAX_COMMAND_EPOCH)),
            ),
        )
        with b.scf_if(bad):
            fail_native()
    safe_command = b.smax(c1, b.smin(command_epoch, ci(MAX_COMMAND_EPOCH)))
    command_base = b.mul(safe_command, ci(2048))

    def census():  # native_entry_census
        wait_vm(0)
        barrier()
        with b.scf_if(b.cmp_eq(tid, c0)):
            system_atomic_add(word(native, b.add(ci(293), safe_xcc)))
            census_prior = system_atomic_add(word(native, 291))
            with b.scf_if(b.cmp_eq(b.add(census_prior, c1), ci(256))):
                ok = c1
                for cx in range(8):
                    ok = b.select(
                        b.cmp_ne(coherent_u32(word(native, 293 + cx)), ci(32)),
                        ci(2),
                        ok,
                    )
                ok = b.select(b.cmp_ne(coherent_u32(word(native, 289)), c0), ci(2), ok)
                with b.scf_if(b.cmp_ne(ok, c1)):
                    fail_native()
                system_store_u32(word(native, 292), ok)
                wait_vm(0)
            t = (
                PRE_T + " global_load_dword $1, $3, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
                " global_load_dword $2, $4, off sc0 sc1\n s_waitcnt vmcnt(0)\n s_sleep 1\n s_add_u32 $0, $0, 1\n"
                " v_cmp_ne_u32 vcc, 0, $1\n s_cbranch_vccnz 2f\n v_cmp_ne_u32 vcc, 0, $2\n s_cbranch_vccnz 2f\n"
                " s_cmp_lt_u32 $0, 0x100000\n s_cbranch_scc1 1b\n2:"
            )
            g = b.inline_asm_multi(
                t,
                "=&s,=&v,=&v,v,v,~{vcc},~{scc}",
                [word(native, 292), word(native, 289)],
                result_types=[I32, I32, I32],
                result_name_hint="gate",
            )
            bad = b.lor(b.cmp_ne(g[1], c1), b.cmp_ne(g[2], c0))
            with b.scf_if(bad):
                fail_native()
            st_store(11, b.select(bad, ci(2), g[1]))
            st_store(
                10, c1
            )  # rocKE-only: the layer body's alive leaves its scf.if region through LDS slot 10
        barrier()
        return b.smax(c1, st_load(11))

    alive0 = census()

    # ------------------------------------------------------------ stage pieces (closures over the layer)
    def shuffle_xor_f(
        v, s
    ):  # FlyDSL shuffle_xor(s, 64) lowering: ds.swizzle (s < 16), permlane16/32.swap (16/32)
        vi = b.bitcast(v, I32)
        if s < 16:
            r = x.swizzle(vi, (s << 10) | 0x1F)
        else:
            p = x.permlane16_swap(vi, vi) if s == 16 else x.permlane32_swap(vi, vi)
            r = b.select(b.cmp_eq(p[0], vi), p[1], p[0])
        return b.bitcast(r, F32)

    def round_bf16(v):  # FlyDSL up_round_bf16
        r = b.sext(b.bitcast(b.trunc_f32_to_bf16(v), I16), I32)
        o = asm("", "=v,0", [r], I32, hint="opq")
        return b.cast_to_f32(b.bitcast(b.trunc(o, I16), BF16))

    def up_local_release(slot, layer):
        with b.scf_if(b.cmp_eq(tid, c0)):
            a = word(native, b.add(ci(CHUNK_WORD + 832), b.mul(safe_xcc, ci(16))))
            sa = b.lor(
                b.zext(rfl(b.trunc(a, I32)), I64),
                b.shl(
                    b.zext(rfl(b.trunc(b.lshr(a, b.const_i64(32)), I32)), I64),
                    b.const_i64(32),
                ),
            )
            ready = poll_ge(sa, b.add(layer, c1), ci(4194304), scalar=True)
            with b.scf_if(b.cmp_eq(ready, c0)):
                fail_native()
            st_store(slot, b.select(b.cmp_eq(ready, c0), ci(2), c1))
        barrier()
        return st_load(slot)

    def native_wait(epoch, slot, consume):  # source 103-121
        with b.scf_if(b.cmp_eq(tid, c0)):
            ready = poll_ge(
                word(native, b.add(ci(144), b.mul(safe_xcc, ci(16)))),
                epoch,
                ci(1048576),
                failed_addr=word(native, 289),
            )
            with b.scf_if(b.cmp_eq(ready, c0)):
                fail_native()
            with b.scf_if(b.land(b.cmp_eq(ready, c1), consume)):
                asm("s_waitcnt vmcnt(0)\nbuffer_inv sc1", "~{memory}")
            st_store(slot, b.select(b.cmp_eq(ready, c0), ci(2), c1))
        barrier()
        return st_load(slot)

    def native_postnorm(epoch, slot, layer, blk):  # source 139-169
        owner = b.lor(
            b.cmp_eq(block, c0),
            b.land(b.cmp_ge(block, ci(137)), b.cmp_le(block, ci(143))),
        )
        with b.scf_if_else(owner) as (t_ctx, f_ctx):
            with t_ctx:
                wait_vm(0)
                barrier()
                with b.scf_if(b.cmp_eq(tid, c0)):
                    system_atomic_add_nw(word(native, 272))
                    ready = poll_ge(
                        word(native, 272),
                        b.mul(b.add(layer, c1), ci(8)),
                        ci(1048576),
                        failed_addr=word(native, 289),
                    )
                    with b.scf_if(b.cmp_eq(ready, c0)):
                        fail_native()
                    asm("s_waitcnt vmcnt(0)\nbuffer_inv sc1", "~{memory}")
                    plain_store_u32(
                        word(
                            native, b.add(ci(CHUNK_WORD + 832), b.mul(safe_xcc, ci(16)))
                        ),
                        b.add(layer, c1),
                    )
                    wait_vm(0)
                    st_store(slot, b.select(b.cmp_eq(ready, c0), ci(2), c1))
                barrier()
            with f_ctx:
                native_wait(
                    epoch,
                    slot,
                    b.land(
                        b.cmp_gt(blk, c0),
                        b.lor(b.cmp_le(blk, ci(136)), b.cmp_ge(blk, ci(144))),
                    ),
                )
        return st_load(slot)

    def up_l2_touch(tbl, itid, blk, first, count, steps, k0=20):
        wd = descriptor(tbl(14), 89128960)
        dst = b.add(b.add(arena_addr, ci(156672)), b.mul(b.div(itid, ci(64)), ci(256)))
        xb = b.mod(blk, ci(8))
        lines = steps * 32
        for m in range(count):
            bb = (
                b.add(xb, ci((first + m) * 8))
                if isinstance(first, int)  # runtime first (output-prime touches)
                else b.add(xb, b.mul(b.add(first, ci(m)), ci(8)))
            )
            upw = zx(
                b.land(
                    b.cmp_gt(bb, c0),
                    b.lor(b.cmp_le(bb, ci(136)), b.cmp_ge(bb, ci(144))),
                )
            )
            unit = b.sub(b.sub(bb, c1), b.mul(zx(b.cmp_ge(bb, ci(144))), ci(7)))
            three = zx(b.cmp_lt(unit, ci(48)))
            gbase = b.add(
                b.mul(three, b.mul(unit, ci(3))),
                b.mul(
                    b.sub(c1, three), b.add(ci(144), b.mul(b.sub(unit, ci(48)), ci(2)))
                ),
            )
            groups = b.mul(b.add(ci(2), three), upw)
            for p in range((3 * 4 * lines + 255) // 256):
                line = b.add(itid, ci(p * 256))
                region = b.div(line, ci(lines))
                gg = b.div(region, ci(4))
                on = zx(b.cmp_lt(gg, groups))
                off = b.add(
                    b.mul(
                        b.add(
                            b.mul(
                                b.add(
                                    b.mul(b.add(gbase, gg), ci(4)), b.mod(region, ci(4))
                                ),
                                ci(40),
                            ),
                            ci(k0),
                        ),
                        ci(1024),
                    ),
                    b.mul(b.mod(line, ci(lines)), ci(128)),
                )
                off = b.add(b.mul(on, off), b.mul(b.sub(c1, on), ci(-128)))
                dma(dst, off, wd, "buffer_load_dword", "")

    def down_job_of(blk):
        j = blk
        for w_, d_ in DOWN_DELTA:
            j = b.add(j, b.mul(zx(b.cmp_eq(blk, ci(w_))), ci(d_)))
        return j

    def down_l2_touch(tbl, itid, blk):
        wd = descriptor(tbl(16), 44564480)
        dst = b.add(b.add(arena_addr, ci(156672)), b.mul(b.div(itid, ci(64)), ci(256)))
        xb = b.mod(blk, ci(8))
        first = zx(b.cmp_eq(xb, c0))
        for m in range(6):
            blk_ = b.add(xb, b.mul(b.add(first, ci(m)), ci(8)))
            job = down_job_of(blk_)
            tile, peer = b.div(job, ci(3)), b.mod(job, ci(3))
            start = b.add(b.mul(peer, ci(11)), b.smin(peer, c1))
            for p in range(4 * 128 // 256):
                line = b.add(itid, ci(p * 256))
                r = b.div(line, ci(128))
                atom = b.div(b.mod(line, ci(128)), ci(32))
                k = b.div(b.mod(line, ci(32)), ci(8))
                off = b.add(
                    b.mul(
                        b.add(
                            b.mul(b.add(b.mul(tile, ci(4)), atom), ci(136)),
                            b.add(b.mul(b.add(start, r), ci(4)), k),
                        ),
                        ci(1024),
                    ),
                    b.mul(b.mod(line, ci(8)), ci(128)),
                )
                dma(dst, off, wd, "buffer_load_dword", "")

    # ---------------------------------------------------------------- up_stage (source.py 1272-1553)
    def up_stage(P, layer, itid, blk, alive, slot):
        """FlyDSL up_stage (source 1272-1553): primes, XCD-local scalar-poll gate, scaled MFMAs, fused epilogue,
        K-slice sub-counter. Returns the gate status (scalar)."""
        lane, wave = b.mod(itid, ci(64)), rfl(b.div(itid, ci(64)))
        high = zx(b.cmp_ge(blk, ci(144)))
        unit = b.sub(b.sub(blk, c1), b.mul(high, ci(7)))
        three = zx(b.cmp_lt(unit, ci(48)))
        cond = b.land(
            b.land(b.cmp_eq(alive, c1), b.cmp_gt(blk, c0)),
            b.lor(b.cmp_le(blk, ci(136)), b.cmp_ge(blk, ci(144))),
        )
        with b.scf_if(cond):
            ad = descriptor(P["post_A"], 40960)
            asd = descriptor(P["post_AS"], 2752)
            bd = descriptor(P["up_w"], 89128960)
            bsd = descriptor(P["up_s"], 5570560)
            bd3 = descriptor(P["up_w"], rfl(b.mul(three, ci(89128960))))
            bsd3 = descriptor(P["up_s"], rfl(b.mul(three, ci(5570560))))
            ldsbase = rfl(arena_addr)
            abase = b.add(
                b.add(
                    b.mul(b.mod(lane, ci(8)), ci(5120)),
                    b.mul(b.div(lane, ci(16)), ci(16)),
                ),
                b.mul(b.div(b.mod(lane, ci(16)), ci(8)), ci(64)),
            )
            gbase = b.add(
                b.mul(three, b.mul(unit, ci(3))),
                b.mul(
                    b.sub(c1, three), b.add(ci(144), b.mul(b.sub(unit, ci(48)), ci(2)))
                ),
            )
            rbase = b.add(ldsbase, b.mul(rfl(wave), ci(39936)))

            def woff(i, kb):
                return b.add(
                    b.mul(
                        b.add(
                            b.mul(
                                b.add(
                                    b.mul(b.add(gbase, ci(i // 4)), ci(4)), ci(i % 4)
                                ),
                                ci(40),
                            ),
                            kb,
                        ),
                        ci(1024),
                    ),
                    b.mul(lane, ci(16)),
                )

            def soff(kb, g):
                return b.add(
                    b.mul(b.add(b.add(b.mul(kb, ci(544)), gbase), ci(g)), ci(256)),
                    b.mul(lane, ci(4)),
                )

            def kb_of(step):
                return b.add(ci(step * 4), wave)

            def prime_scales(step):
                kb = kb_of(step)
                return [
                    x.buf_ld4(bsd3 if g == 2 else bsd, soff(kb, g), 2) for g in range(3)
                ]

            def dma_step(step, slot_j):
                kb = kb_of(step)
                for i in range(12):
                    dma(
                        b.add(rbase, ci(slot_j * 12288 + i * 1024)),
                        woff(i, kb),
                        bd3 if i >= 8 else bd,
                    )

            bsc = [None] * 10
            for j in range(3):
                bsc[j] = prime_scales(j)
                dma_step(j, j)
            vb = [None, None]
            for v in range(2):
                bsc[3 + v] = prime_scales(3 + v)
                kb = kb_of(3 + v)
                vb[v] = [
                    x.buf_ld16(bd3 if i >= 8 else bd, woff(i, kb), 0) for i in range(12)
                ]
            res = up_local_release(slot, layer)
            with b.scf_if(b.cmp_eq(res, c1)):
                alo = [
                    x.buf_ld16(ad, b.add(abase, b.mul(kb_of(j), ci(128))), 17)
                    for j in range(10)
                ]
                asc = [
                    b.zext(
                        x.buf_ld1(asd, b.add(b.mul(kb_of(j), ci(64)), lane), 17), I32
                    )
                    for j in range(10)
                ]
                wait_vm(0)
                vb = [[pin(t) for t in row] for row in vb]
                alo = [pin(t) for t in alo]
                asc = [pin(t) for t in asc]
                bsc = [[pin(t) for t in bsc[j]] if j < 5 else bsc[j] for j in range(10)]
                acc = [b.zero_vec_f32(4) for _ in range(12)]

                def peer8(v):
                    return asm(
                        "s_nop 1\nv_mov_b32_dpp $0, $1 row_ror:8 row_mask:0xf bank_mask:0xf",
                        "=&v,v",
                        [v],
                        I32,
                        hint="dpp",
                    )

                def a_vec(j):
                    lo = [b.vec_extract(alo[j], e) for e in range(4)]
                    return b.vec_pack(lo + [peer8(t) for t in lo], I32)

                def lds_b(slot_j):
                    return [
                        lds_ld16(
                            b.add(
                                b.sub(rbase, ldsbase),
                                b.add(
                                    ci(slot_j * 12288 + i * 1024), b.mul(lane, ci(16))
                                ),
                            )
                        )
                        for i in range(12)
                    ]

                def mfmas(av, bv, sa, sbs):
                    for i in range(12):
                        acc[i] = asm(
                            f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 op_sel:[0,{i % 2},0] "
                            f"op_sel_hi:[0,{i % 4 // 2},0] cbsz:0 blgp:4",
                            "=a,v,v,v,v,0",
                            [av, bv[i], sa, sbs[i // 4], acc[i]],
                            V4F,
                            hint="umf",
                        )

                waits = {5: 30, 6: 30, 7: 30, 8: 15, 9: 0}
                for s in range(10):
                    if s in waits:
                        wait_vm(waits[s])
                    av = a_vec(s)
                    if s in (3, 4):
                        bv = vb[s - 3]
                    else:
                        bv = lds_b(s % 3 if s < 3 else (s - 5) % 3)
                        b.s_waitcnt(lgkmcnt=0)
                        if s in (
                            0,
                            1,
                            2,
                            5,
                            6,
                        ):  # refill into the slot just read: steps 5,6,7 (s<3), 8,9 (s=5,6)
                            tgt = s + 5 if s < 3 else s + 3
                            dma_step(tgt, s % 3 if s < 3 else (s - 5) % 3)
                            bsc[tgt] = prime_scales(tgt)
                    mfmas(av, bv, asc[s], bsc[s])
                wait_vm(0)
                b.s_waitcnt(lgkmcnt=0)
                barrier()
                for i in range(12):
                    for e in range(4):
                        lds_st_f32(
                            b.add(
                                b.mul(
                                    b.add(
                                        b.mul(
                                            b.add(b.mul(wave, ci(12)), ci(i)), ci(64)
                                        ),
                                        lane,
                                    ),
                                    ci(16),
                                ),
                                ci(4 * e),
                            ),
                            b.vec_extract(acc[i], e),
                        )
            res2 = st_load(slot)
            with b.scf_if(b.cmp_eq(res2, c1)):
                barrier()
                c32 = b.mod(itid, ci(32))
                r = b.div(itid, ci(32))
                atom_t = b.div(c32, ci(8))
                cp = b.mod(c32, ci(8))
                od = descriptor(P["act"], 139264)
                osd = descriptor(P["act_s"], 8896)

                def store_byte(desc, v32, off, en):
                    o = b.add(b.mul(en, off), b.mul(b.sub(c1, en), ci(0x7FFFFFF0)))
                    x.buf_st1(b.trunc(v32, I8), desc, o, 17)

                src_lane = b.add(b.mul(b.div(r, ci(4)), ci(16)), b.mul(cp, ci(2)))
                src_e = b.mod(r, ci(4))

                def sum_parts(g, h):
                    p = [
                        lds_ld_f32(
                            b.mul(
                                b.add(
                                    b.mul(
                                        b.add(
                                            b.mul(
                                                b.add(ci(w * 12 + g * 4), atom_t),
                                                ci(64),
                                            ),
                                            b.add(src_lane, ci(h)),
                                        ),
                                        ci(4),
                                    ),
                                    src_e,
                                ),
                                ci(4),
                            )
                        )
                        for w in range(4)
                    ]
                    return b.fadd(b.fadd(b.fadd(p[0], p[1]), p[2]), p[3])

                values, amax = [], []
                one = b.const_f32(1.0)
                for g in range(3):
                    gate = round_bf16(sum_parts(g, 0))
                    upv = round_bf16(sum_parts(g, 1))
                    sig = b.fdiv(one, b.fadd(one, x.exp(b.fneg(gate))))
                    actv = round_bf16(b.fmul(gate, sig))
                    values.append(round_bf16(b.fmul(actv, upv)))
                    amax.append(b.fabs(values[g]))
                for sh in (1, 2, 4, 8, 16):
                    amax = [x.maximum(m, shuffle_xor_f(m, sh)) for m in amax]
                for g in range(3):
                    on = c1 if g < 2 else three
                    grp = b.add(gbase, ci(g))
                    bits = b.bitcast(b.fmul(amax[g], b.const_f32(1.0 / 448.0)), I32)
                    ex = b.add(
                        b.land(b.lshr(bits, ci(23)), ci(255)),
                        zx(b.cmp_ne(b.land(bits, ci(8388607)), c0)),
                    )
                    ex = b.smax(c1, b.smin(ex, ci(254)))
                    scale = b.bitcast(b.shl(ex, ci(23)), F32)
                    pairv = b.vec_pack(
                        [
                            b.trunc_f32_to_bf16(values[g]),
                            b.trunc_f32_to_bf16(b.const_f32(0.0)),
                        ],
                        BF16,
                    )
                    pair = b.vec_extract(b.vec_bitcast(pairv, VectorType(I32, 1)), 0)
                    packed = asm(
                        "v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2",
                        "=v,v,v,0",
                        [pair, scale, c0],
                        I32,
                        se=False,
                        hint="cvt",
                    )
                    store_byte(
                        od,
                        packed,
                        b.add(b.add(b.mul(r, ci(17408)), b.mul(grp, ci(32))), c32),
                        on,
                    )
                    lead = zx(b.cmp_lt(c32, ci(2)))
                    sv = b.add(b.mul(b.sub(c1, c32), ex), b.mul(c32, ci(127)))
                    store_byte(
                        osd,
                        sv,
                        b.add(b.add(b.mul(grp, ci(16)), r), b.mul(c32, ci(8))),
                        b.mul(lead, on),
                    )
                wait_vm(0)
                barrier()
                with b.scf_if(b.cmp_eq(itid, c0)):
                    sl = b.add(zx(b.cmp_ge(unit, ci(72))), zx(b.cmp_ge(unit, ci(160))))
                    system_atomic_add_nw(
                        word(
                            native,
                            b.add(
                                ci(CHUNK_WORD + 256),
                                b.mul(
                                    b.add(b.mul(sl, ci(4)), b.mod(unit, ci(4))), ci(16)
                                ),
                            ),
                        )
                    )
        return st_load(slot)

    # ---------------------------------------------------------------- relay_up_slices (2716-2744)
    def relay_up_slices(layer, itid):
        with b.scf_if(b.cmp_lt(itid, ci(64))):
            lane = itid
            k12 = b.mod(lane, ci(12))
            sl = b.div(k12, ci(4))
            need = b.mul(
                b.sub(ci(22), b.mul(zx(b.cmp_eq(sl, c0)), ci(4))), b.add(layer, c1)
            )
            addr = word(native, b.add(ci(CHUNK_WORD + 256), b.mul(k12, ci(16))))
            ra = [
                word(
                    native,
                    b.add(
                        ci(CHUNK_WORD + 1024),
                        b.mul(b.add(b.mul(safe_xcc, ci(3)), ci(s_)), ci(16)),
                    ),
                )
                for s_ in range(3)
            ]
            grp = "".join(
                f" s_and_b32 $3, vcc_lo, {0xF << (4 * s_):#x}\n s_cmp_eq_u32 $3, {0xF << (4 * s_):#x}\n"
                f" s_cselect_b32 $3, {1 << s_}, 0\n s_or_b32 $2, $2, $3\n"
                for s_ in range(3)
            )
            sto = "".join(
                f" s_bitcmp1_b32 $3, {s_}\n s_cbranch_scc0 1{s_ + 3}f\n global_store_dword ${9 + s_}, $12, off\n"
                f"1{s_ + 3}:\n"
                for s_ in range(3)
            )
            t = (
                "s_mov_b32 $0, 0\n s_mov_b32 $1, 0\n1:\n global_load_dword $5, $7, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
                " v_cmp_ge_i32 vcc, $5, $8\n s_mov_b32 $2, 0\n"
                + grp
                + " s_andn2_b32 $3, $2, $1\n s_cmp_eq_u32 $3, 0\n s_cbranch_scc1 3f\n"
                " s_mov_b64 $4, exec\n s_mov_b64 exec, 1\n"
                + sto
                + " s_mov_b64 exec, $4\n3:\n"
                " s_or_b32 $1, $1, $2\n global_load_dword $6, $13, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
                " v_readfirstlane_b32 $3, $6\n s_sleep 1\n s_add_u32 $0, $0, 1\n"
                " s_cmp_eq_u32 $1, 7\n s_cbranch_scc1 2f\n s_cmp_lg_u32 $3, 0\n s_cbranch_scc1 2f\n"
                " s_cmp_lt_u32 $0, 0x100000\n s_cbranch_scc1 1b\n2:"
            )
            rr = b.inline_asm_multi(
                t,
                "=&s,=&s,=&s,=&s,=&s,=&v,=&v,v,v,v,v,v,v,v,~{vcc},~{scc}",
                [addr, need, ra[0], ra[1], ra[2], b.add(layer, c1), word(native, 289)],
                result_types=[I32, I32, I32, I32, I64, I32, I32],
                result_name_hint="relay",
            )
            with b.scf_if(b.land(b.cmp_eq(itid, c0), b.cmp_ne(rr[1], ci(7)))):
                fail_native()
        barrier()

    # ---------------------------------------------------------------- down_stage (1555-1688)
    def down_stage(P, layer, itid, blk, alive, slot):
        """FlyDSL down_stage (source 1555-1688) with M3's D3: 9 weight k-steps + scales primed, XCD-local scalar poll of
        the relayed word, the activation k-steps as one full-lane x4 load each + row_ror:8 DPP (lanes l % 16 < 8),
        scaled MFMAs with refill, 4-wave LDS reduction, wave 0 write-through stores, lane-0 chunk counter.
        """
        lane, wave = b.mod(itid, ci(64)), rfl(b.div(itid, ci(64)))
        job = down_job_of(blk)
        tile, peer = b.div(job, ci(3)), b.mod(job, ci(3))
        with b.scf_if(b.land(b.cmp_eq(alive, c1), b.cmp_lt(blk, ci(240)))):
            iters = b.select(b.cmp_eq(peer, c0), ci(12), ci(11))
            start = b.add(b.mul(peer, ci(11)), b.smin(peer, c1))
            ad = descriptor(P["act"], 139264)
            asd = descriptor(P["act_s"], 8896)
            bd = descriptor(P["dn_w"], 44564480)
            bsd = descriptor(P["dn_s"], 2785280)
            lane4 = b.land(lane, ci(-4))

            def kb_of(r):
                return b.add(b.mul(b.add(start, ci(r)), ci(4)), wave)

            def wload(r):
                kb = kb_of(r)
                ws, ss = [], []
                for atom in range(4):
                    ws.append(
                        x.buf_ld16(
                            bd,
                            b.add(
                                b.mul(
                                    b.add(
                                        b.mul(
                                            b.add(b.mul(tile, ci(4)), ci(atom)), ci(136)
                                        ),
                                        kb,
                                    ),
                                    ci(1024),
                                ),
                                b.mul(lane, ci(16)),
                            ),
                            2,
                        )
                    )
                    ss.append(
                        x.buf_ld4(
                            bsd,
                            b.add(
                                b.mul(
                                    b.add(
                                        b.add(b.mul(kb, ci(320)), b.mul(tile, ci(4))),
                                        ci(atom),
                                    ),
                                    ci(64),
                                ),
                                lane4,
                            ),
                            0,
                        )
                    )
                return ws, ss

            bv, bs = [None] * 36, [None] * 36
            for r in range(9):
                ws, ss = wload(r)
                for atom in range(4):
                    bv[r * 4 + atom], bs[r * 4 + atom] = ws[atom], ss[atom]
            with b.scf_if(b.cmp_eq(itid, c0)):
                a = word(
                    native,
                    b.add(
                        ci(CHUNK_WORD + 1024),
                        b.mul(b.add(b.mul(safe_xcc, ci(3)), peer), ci(16)),
                    ),
                )
                sa = b.lor(
                    b.zext(rfl(b.trunc(a, I32)), I64),
                    b.shl(
                        b.zext(rfl(b.trunc(b.lshr(a, b.const_i64(32)), I32)), I64),
                        b.const_i64(32),
                    ),
                )
                ready = poll_ge(sa, b.add(layer, c1), ci(4194304), scalar=True)
                with b.scf_if(b.cmp_eq(ready, c0)):
                    fail_native()
                st_store(slot, b.select(b.cmp_eq(ready, c0), ci(2), c1))
            barrier()
            res = st_load(slot)
            with b.scf_if(b.cmp_eq(res, c1)):
                alo, asc = [], []
                for r in range(
                    12
                ):  # D3: one full-lane load: lanes l % 16 < 8 read k bytes [0, 64) of row l % 8, lanes >= 8 [64, 128)
                    kb = kb_of(r)
                    off = b.add(
                        b.add(
                            b.add(
                                b.mul(b.mod(lane, ci(8)), ci(17408)), b.mul(kb, ci(128))
                            ),
                            b.mul(b.div(lane, ci(16)), ci(16)),
                        ),
                        b.mul(b.div(b.mod(lane, ci(16)), ci(8)), ci(64)),
                    )
                    alo.append(x.buf_ld16(ad, off, 17))
                    asc.append(x.buf_ld4(asd, b.add(b.mul(kb, ci(64)), lane4), 17))
                wait_vm(0)
                acc = [b.zero_vec_f32(4) for _ in range(4)]
                sh = b.mul(b.land(lane, ci(3)), ci(8))
                z = c0
                low8 = b.cmp_lt(
                    b.mod(lane, ci(16)), ci(8)
                )  # rows 8-15 (lanes l % 16 >= 8) stay 0 in both halves

                def a_d3(v):
                    """lanes l % 16 < 8: (own k[0,64), lane l+8's k[64,128)); lanes >= 8: (0, 0). hi: row_ror:8 DPP
                    with bank_mask 0x3 (banks 2-3 = lanes 8-15 of each row not written: keep the tied 0).
                    """
                    e4 = [b.vec_extract(v, e) for e in range(4)]
                    hi = b.inline_asm_multi(
                        "s_nop 1\n"
                        + "".join(
                            f" v_mov_b32_dpp ${i}, ${4 + i} row_ror:8 row_mask:0xf bank_mask:0x3\n"
                            for i in range(4)
                        ),
                        "=v,=v,=v,=v,v,v,v,v,0,1,2,3",
                        e4 + [z] * 4,
                        result_types=[I32] * 4,
                        result_name_hint="dpp",
                    )
                    lo = [b.select(low8, t, z) for t in e4]
                    return b.vec_pack(lo + list(hi), I32)

                def step(k):
                    wr = k % 9
                    lo, sa = pin(alo[k]), pin(asc[k])
                    sa = b.land(b.lshr(sa, sh), ci(255))
                    av = a_d3(lo)
                    for atom in range(4):
                        bb = pin(bv[wr * 4 + atom])
                        bf = b.vec_pack(
                            [b.vec_extract(bb, e) for e in range(4)] + [z] * 4, I32
                        )
                        sb = b.land(b.lshr(pin(bs[wr * 4 + atom]), sh), ci(255))
                        acc[atom] = x.mfma_scale(av, bf, acc[atom], sa, sb, 0, 4)

                def refill(k):
                    ws, ss = wload(k + 9)
                    for atom in range(4):
                        bv[(k % 9) * 4 + atom], bs[(k % 9) * 4 + atom] = (
                            ws[atom],
                            ss[atom],
                        )

                for k in range(11):
                    step(k)
                    if k + 9 < 11:
                        refill(k)
                    elif (
                        k + 9 == 11
                    ):  # FlyDSL: if down_k + 9 < down_iterations (peer-0 jobs only)

                        def rf(cur, k=k):
                            ws, ss = wload(k + 9)
                            return ws + ss

                        new = if_yield(
                            b.cmp_lt(ci(k + 9), iters),
                            [bv[(k % 9) * 4 + a] for a in range(4)]
                            + [bs[(k % 9) * 4 + a] for a in range(4)],
                            rf,
                        )
                        for a in range(4):
                            bv[(k % 9) * 4 + a], bs[(k % 9) * 4 + a] = (
                                new[a],
                                new[4 + a],
                            )

                def st11(cur):
                    acc[:] = cur
                    step(11)
                    return list(acc)

                acc[:] = if_yield(b.cmp_lt(ci(11), iters), list(acc), st11)
                wait_vm(0)

                def red_off(
                    w, atom, e, ln
                ):  # byte offset of wave w's partial (atom, e) of lane ln
                    wv = ci(w) if isinstance(w, int) else w
                    return b.mul(
                        b.add(
                            b.add(b.mul(wv, ci(1024)), ci(atom * 256)),
                            b.add(b.mul(ln, ci(4)), ci(e)),
                        ),
                        ci(4),
                    )

                for atom in range(4):
                    for e in range(4):
                        lds_st_f32(
                            red_off(wave, atom, e, lane), b.vec_extract(acc[atom], e)
                        )
                barrier()
                with b.scf_if(b.cmp_eq(wave, c0)):
                    for atom in range(4):
                        for e in range(4):
                            val = lds_ld_f32(red_off(0, atom, e, lane))
                            for w in range(1, 4):
                                val = b.fadd(val, lds_ld_f32(red_off(w, atom, e, lane)))
                            row = b.add(b.mul(b.div(lane, ci(16)), ci(4)), ci(e))
                            with b.scf_if(b.cmp_lt(row, ci(8))):
                                idx = b.add(
                                    b.add(
                                        b.add(
                                            b.mul(peer, ci(40960)), b.mul(row, ci(5120))
                                        ),
                                        b.add(b.mul(tile, ci(64)), ci(atom * 16)),
                                    ),
                                    b.mod(lane, ci(16)),
                                )
                                system_store_u32(
                                    b.add(P["part"], b.mul(i64(idx), b.const_i64(4))),
                                    b.bitcast(val, I32),
                                )
                    wait_vm(0)
                    # FlyDSL H5: chunk counter (wave 0 here; lane 0 by EXEC inside the asm)
                    lane0_atomic_add_nw(
                        word(
                            native,
                            b.add(ci(CHUNK_WORD), b.mul(b.div(tile, ci(10)), ci(16))),
                        )
                    )
                barrier()
        return st_load(slot)

    # ================================================================ fold, layer-0 RMS, QKV
    # source 321-406 (rms_stage), 503-623 (fold_chunk_stage), 625-649 (fold_ready_wait), 651-853 (qkv_stage),
    # 1697-1802 (attention_qgkv_stage).
    V8F, V8B = VectorType(F32, 8), VectorType(BF16, 8)
    PEER_WORD = 400
    DPP4 = (
        "s_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n s_nop 1\n"
        " v_add_f32_dpp $0, $0, $0 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n s_nop 1\n"
        " v_add_f32_dpp $0, $0, $0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n s_nop 1\n"
        " v_add_f32_dpp $0, $0, $0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
    )
    DPP2 = (
        "s_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n s_nop 1\n"
        " v_add_f32_dpp $0, $0, $0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
    )

    def nrm_offset(
        valid, byte_off
    ):  # invalid columns move past num_records (loads 0, stores dropped)
        return b.add(byte_off, b.mul(b.sub(c1, zx(valid)), ci(0x10000000)))

    def bf8(v4i):
        return b.vec_bitcast(v4i, V8B)

    def ext8(vb):
        return x.fpext_bf16x8(vb)

    def trunc8(vf):
        return b.vec_cast_f32_to(vf, BF16)

    def i4(vb):
        return b.vec_bitcast(vb, V4I)

    def tree8(
        acc,
    ):  # acc[0] + acc[1] + (acc[2] + acc[3]) + (acc[4] + acc[5] + (acc[6] + acc[7]))
        a = [b.vec_extract(acc, i) for i in range(8)]
        return b.fadd(
            b.fadd(b.fadd(a[0], a[1]), b.fadd(a[2], a[3])),
            b.fadd(b.fadd(a[4], a[5]), b.fadd(a[6], a[7])),
        )

    def shuffle_idx(
        v, c
    ):  # FlyDSL gpu.shuffle_idx(v, c, 64): mbcnt lane id (rocKE lane_id), width clamp, ds_bpermute
        me = b.lane_id()
        src = b.select(b.cmp_lt(ci(c), b.land(b.add(me, ci(64)), ci(-64))), ci(c), me)
        return x.bpermute(b.shl(src, ci(2)), v)

    def lf(
        idx,
    ):  # FlyDSL qkv_reduction + idx: float element pointer into the arena (LDS)
        return x.gep(x.ptr3(arena_addr), "float", idx, lds=True)

    def lbp(idx):  # FlyDSL qkv_demangle + idx: bfloat element pointer into the arena
        return x.gep(x.ptr3(arena_addr), "bfloat", idx, lds=True)

    def ldf(p):
        return x.load(p, "float", F32, 4, lds=True)

    def stf(p, v):
        x.store(p, v, "float", 4, lds=True)

    def fp8_quant(norm, branchy_ex=False):
        """amax over 8 values + shuffle_xor 1, 2 (32 columns), E8M0 ceil exponent clamped [1, 254], 4 x
        v_cvt_scalef32_pk_fp8_bf16 (asm sideeffect, FlyDSL). Returns (q0, q1, exponent).
        """
        raw = i4(norm)
        vals = ext8(norm)
        amax = b.const_f32(0.0)
        for e in range(8):
            amax = x.maximum(amax, b.fabs(b.vec_extract(vals, e)))
        for s_ in (1, 2):
            amax = x.maximum(amax, shuffle_xor_f(amax, s_))
        bits = b.bitcast(b.fmul(amax, b.const_f32(1.0 / 448.0)), I32)
        ex = b.land(b.lshr(bits, ci(23)), ci(255))
        nz = b.cmp_ne(b.land(bits, ci(8388607)), c0)
        ex = (
            if_yield(nz, [ex], lambda cur: [b.add(cur[0], c1)])[0]
            if branchy_ex
            else b.add(ex, zx(nz))
        )
        ex = b.smax(c1, b.smin(ex, ci(254)))
        scale = b.bitcast(b.shl(ex, ci(23)), F32)

        def cv(r, acc, hi):
            return asm(
                "v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2"
                + (" op_sel:[0,0,1]" if hi else ""),
                "=v,v,v,0",
                [r, scale, acc],
                I32,
                hint="cvt",
            )

        q0 = cv(b.vec_extract(raw, 1), cv(b.vec_extract(raw, 0), c0, False), True)
        q1 = cv(b.vec_extract(raw, 3), cv(b.vec_extract(raw, 2), c0, False), True)
        return q0, q1, ex

    def store_fp8(
        T, row, column, q0, q1, ex, tid
    ):  # FP8 dwordx2 + scale byte (sc0 sc1), caller guards validity
        asm(
            "global_store_dwordx2 $0, $1, off sc0 sc1",
            "v,v,~{memory}",
            [
                b.add(T["act"], i64(b.add(b.mul(row, ci(5120)), column))),
                b.vec_pack([q0, q1], I32),
            ],
        )
        with b.scf_if(b.cmp_eq(b.mod(tid, ci(4)), c0)):
            qb = b.div(column, ci(32))
            asm(
                "global_store_byte $0, $1, off sc0 sc1",
                "v,v,~{memory}",
                [
                    b.add(
                        T["act_s"],
                        i64(
                            b.add(
                                b.add(
                                    b.mul(b.div(qb, ci(4)), ci(64)),
                                    b.mul(b.mod(qb, ci(4)), ci(16)),
                                ),
                                row,
                            )
                        ),
                    ),
                    ex,
                ],
            )

    def padv():  # the padding byte 127
        return ci(127)

    def scale_padding(T, tid):  # block / row 0: 127 in the padding scale bytes
        for step in range(11):
            index = b.add(tid, ci(step * 256))
            with b.scf_if(b.cmp_lt(index, ci(2752))):
                with b.scf_if(
                    b.lor(
                        b.cmp_ge(index, ci(2560)), b.cmp_ge(b.mod(index, ci(16)), ci(8))
                    )
                ):
                    asm(
                        "global_store_byte $0, $1, off sc0 sc1",
                        "v,v,~{memory}",
                        [b.add(T["act_s"], i64(index)), padv()],
                    )

    def mode_save():
        return asm("s_getreg_b32 $0, 0xdc1", "=s,~{memory}", [], I32, hint="mode")

    def mode_restore(saved):
        asm(
            "s_waitcnt vmcnt(0) lgkmcnt(0)\ns_setreg_b32 0xdc1, $0",
            "s,~{memory}",
            [saved],
        )

    # ---------------------------------------------------------------- rms_stage (layer 0; source.py 321-406)
    def rms_stage(T, row, itid):
        tid = itid
        lane, wave = b.mod(tid, ci(64)), rfl(b.div(tid, ci(64)))
        asm("s_setreg_imm32_b32 0xdc1, 1;", "~{memory}")
        hd = descriptor(T["hidden"], 81920)
        rd = descriptor(T["residual"], 81920)
        gd = descriptor(T["gamma"], 10240)
        od = descriptor(T["residual_out"], 81920)
        nd = descriptor(T["normalized"], 81920)
        loaded = []
        for step in range(3):
            column = b.add(ci(step * 2048), b.mul(tid, ci(8)))
            valid = b.cmp_lt(column, ci(5120))
            o2 = nrm_offset(valid, b.mul(b.add(b.mul(row, ci(5120)), column), ci(2)))
            loaded.append(
                (
                    x.buf_ld16(hd, o2, 1),
                    x.buf_ld16(gd, nrm_offset(valid, b.mul(column, ci(2))), 0),
                    x.buf_ld16(rd, o2, 1),
                )
            )
        acc = b.zero_vec_f32(8)
        for step in range(3):
            summed = b.vector_add(
                ext8(bf8(loaded[step][0])), ext8(bf8(loaded[step][2]))
            )
            acc = x.fma8(summed, summed, acc)
        total = asm(DPP4, "=v,0", [tree8(acc)], F32, hint="dpp")
        total = b.fadd(total, shuffle_xor_f(total, 16))
        total = b.fadd(total, shuffle_xor_f(total, 32))
        with b.scf_if(b.cmp_eq(lane, c0)):
            lds_st_f32(b.mul(wave, ci(4)), total)
        barrier()
        total = if_yield(
            b.cmp_lt(lane, ci(4)),
            [b.const_f32(0.0)],
            lambda cur: [lds_ld_f32(b.mul(lane, ci(4)))],
        )[0]
        total = asm(DPP2, "=v,0", [total], F32, hint="dpp")
        total = b.bitcast(rfl(b.bitcast(total, I32)), F32)
        rms = x.rsq(x.fma(total, b.const_f32(1.0 / 5120.0), b.const_f32(1e-06)))
        zb = bf8(b.vec_pack([c0] * 4, I32))
        for step in range(3):
            column = b.add(ci(step * 2048), b.mul(tid, ci(8)))
            valid = b.cmp_lt(column, ci(5120))
            data = b.vector_select(valid, bf8(loaded[step][0]), zb)
            skip = b.vector_select(valid, bf8(loaded[step][2]), zb)
            gamma = b.vector_select(valid, bf8(loaded[step][1]), zb)
            rounded = trunc8(b.vector_add(ext8(data), ext8(skip)))
            norm = trunc8(
                b.vector_mul(
                    b.vector_mul(ext8(rounded), b.vector_splat(rms, 8)), ext8(gamma)
                )
            )
            so = nrm_offset(valid, b.mul(b.add(b.mul(row, ci(5120)), column), ci(2)))
            x.buf_st16(i4(rounded), od, so, 17)
            x.buf_st16(i4(norm), nd, so, 17)
            q0, q1, ex = fp8_quant(norm, branchy_ex=True)
            with b.scf_if(valid):
                store_fp8(T, row, column, q0, q1, ex, tid)
        with b.scf_if(b.cmp_eq(row, c0)):
            scale_padding(T, tid)

    # ---------------------------------------------------------------- fold_chunk_stage (source.py 503-623)
    def fold_chunk_stage(T, layer, itid, blk, epoch):
        """owner blk folds / normalizes columns [640 (blk // 8), +640) of row blk % 8. T: i64 addresses partial
        (layer L-1's Down planes), residual, gamma, normalized, residual_out, act, act_s.
        """
        row, chunk, tid = b.mod(blk, ci(8)), b.div(blk, ci(8)), itid
        lane, wave = b.mod(tid, ci(64)), rfl(b.div(tid, ci(64)))
        asm("s_setreg_imm32_b32 0xdc1, 1;", "~{memory}")
        with b.scf_if(
            b.cmp_eq(tid, c0)
        ):  # the chunk's 30 Down CTAs of the previous layer have published
            target = b.mul(layer, ci(30))
            bad0 = coherent_u32(word(native, 289))
            t = (
                "s_mov_b32 $0, 0\n v_mov_b32 $1, 0\n v_mov_b32 $2, $6\n1:\n v_cmp_gt_i32 vcc, $5, $1\n"
                " s_cbranch_vccz 2f\n v_cmp_eq_u32 vcc, 0, $2\n s_cbranch_vccz 2f\n s_cmp_lt_u32 $0, 0x100000\n"
                " s_cbranch_scc0 2f\n global_load_dword $1, $3, off sc0 sc1\n"  # P1-fold: one vmcnt(0) for both
                " global_load_dword $2, $4, off sc0 sc1\n s_waitcnt vmcnt(0)\n s_sleep 1\n s_add_u32 $0, $0, 1\n"
                " s_branch 1b\n2:"
            )
            rr = b.inline_asm_multi(
                t,
                "=&s,=&v,=&v,v,v,v,v,~{vcc},~{scc}",
                [
                    word(native, b.add(ci(CHUNK_WORD), b.mul(chunk, ci(16)))),
                    word(native, 289),
                    target,
                    bad0,
                ],
                result_types=[I32, I32, I32],
                result_name_hint="fcpoll",
            )
            bad = b.lor(b.cmp_lt(rr[1], target), b.cmp_ne(rr[2], c0))
            with b.scf_if(bad):
                fail_native()
            asm("s_waitcnt vmcnt(0)\nbuffer_inv sc1", "~{memory}")
            st_store(6, b.select(bad, ci(2), c1))
        barrier()
        chunk_ok = st_load(6)
        pd = descriptor(T["partial"], 491520)
        rd = descriptor(T["residual"], 81920)
        gd = descriptor(T["gamma"], 10240)
        od = descriptor(T["residual_out"], 81920)
        nd = descriptor(T["normalized"], 81920)
        column = b.add(b.mul(chunk, ci(640)), b.mul(tid, ci(8)))
        valid = b.cmp_lt(tid, ci(80))
        offset = b.add(b.mul(row, ci(5120)), column)
        parts = [
            x.buf_ld16(
                pd,
                nrm_offset(
                    valid,
                    b.add(b.mul(b.add(ci(pl * 40960), offset), ci(4)), ci(h * 16)),
                ),
                1,
            )
            for pl in range(3)
            for h in range(2)
        ]
        g = x.buf_ld16(gd, nrm_offset(valid, b.mul(column, ci(2))), 0)
        s = x.buf_ld16(rd, nrm_offset(valid, b.mul(offset, ci(2))), 1)
        planes = [
            b.vec_bitcast(
                b.vec_pack(
                    [b.vec_extract(parts[2 * p], i) for i in range(4)]
                    + [b.vec_extract(parts[2 * p + 1], i) for i in range(4)],
                    I32,
                ),
                V8F,
            )
            for p in range(3)
        ]
        gamma, skip = bf8(g), bf8(s)
        data = trunc8(b.vector_add(b.vector_add(planes[0], planes[1]), planes[2]))
        summed = b.vector_add(ext8(data), ext8(skip))
        acc = x.fma8(summed, summed, b.zero_vec_f32(8))
        total = asm(DPP4, "=v,0", [tree8(acc)], F32, hint="dpp")
        total = b.fadd(total, shuffle_xor_f(total, 16))
        total = b.fadd(total, shuffle_xor_f(total, 32))
        with b.scf_if(b.cmp_eq(lane, c0)):
            lds_st_f32(b.mul(wave, ci(4)), total)
        barrier()
        with b.scf_if(b.cmp_lt(tid, ci(64))):
            local = if_yield(
                b.cmp_lt(lane, ci(4)),
                [b.const_f32(0.0)],
                lambda cur: [lds_ld_f32(b.mul(lane, ci(4)))],
            )[0]
            local = asm(DPP2, "=v,0", [local], F32, hint="dpp")
            with b.scf_if(b.cmp_eq(tid, c0)):
                asm(
                    "global_store_dwordx2 $0, $1, off sc0 sc1\ns_waitcnt vmcnt(0)",
                    "v,v,~{memory}",
                    [
                        word(native, b.add(ci(PEER_WORD), b.mul(blk, ci(16)))),
                        b.vec_pack([epoch, b.bitcast(local, I32)], I32),
                    ],
                )
            # lane c < 8 polls chunk c's word of this row ({epoch, value} dwordx2 sc0 sc1); all eight must carry the
            # epoch. Per iteration the asm loop issues FlyDSL's compiled loop body (source 566-575; FlyDSL's compiled
            # ISA): load, tag compare, cndmask, min tree over xor 1/2/4 (ds_swizzle), readfirstlane, failure-word
            # load, s_sleep 1. The 64-bit word is split with v_pk_mov_b32 (asm operands are whole registers):
            # tmp = {epoch, got.hi} (tag ok <=> got == tmp), val = {got.hi, got.hi}.
            failed0 = rfl(coherent_u32(word(native, 289)))
            t = (
                "s_mov_b32 $0, 0\n s_mov_b32 $3, 0\n s_mov_b32 $4, $11\n v_mov_b64 $2, 0\n1:\n s_cmp_lg_u32 $3, 0\n"
                " s_cbranch_scc1 2f\n s_cmp_lg_u32 $4, 0\n s_cbranch_scc1 2f\n s_cmp_lt_u32 $0, 0x100000\n"
                " s_cbranch_scc0 2f\n global_load_dwordx2 $1, $9, off sc0 sc1\n"
                " global_load_dword $6, $12, off sc0 sc1\n s_waitcnt vmcnt(0)\n"  # P1-fold: tag + failure word
                " v_pk_mov_b32 $5, $10, $1 op_sel:[0,1]\n v_cmp_eq_u64 vcc, $1, $5\n s_nop 1\n"
                " v_cndmask_b32_e64 $7, 0, 1, vcc\n"
                + min_tree("$7", ("$8", None), (1, 2, 4))
                + " s_nop 0\n v_readfirstlane_b32 $3, $7\n v_pk_mov_b32 $2, $1, $1 op_sel:[1,1]\n"
                " s_sleep 1\n s_nop 0\n"
                " v_readfirstlane_b32 $4, $6\n s_add_u32 $0, $0, 1\n s_branch 1b\n2:"
            )
            V2I = VectorType(I32, 2)
            slot_a = word(
                native,
                b.add(
                    ci(PEER_WORD),
                    b.mul(b.add(b.mul(b.mod(lane, ci(8)), ci(8)), row), ci(16)),
                ),
            )
            rr = b.inline_asm_multi(
                t,
                "=&s,=&v,=&v,=&s,=&s,=&v,=&v,=&v,=&v,v,v,s,v,~{vcc},~{scc}",
                [slot_a, b.vec_pack([epoch, epoch], I32), failed0, word(native, 289)],
                result_types=[I32, V2I, V2I, I32, I32, V2I, I32, I32, I32],
                result_name_hint="rspoll",
            )
            value = b.vec_extract(rr[2], 0)
            parts8 = [b.bitcast(shuffle_idx(value, c), F32) for c in range(8)]
            total8 = b.const_f32(0.0)
            for c in range(8):
                total8 = b.fadd(total8, parts8[c])
            with b.scf_if(b.cmp_eq(tid, c0)):
                bad = b.lor(b.cmp_eq(rr[3], c0), b.cmp_ne(rr[4], c0))
                with b.scf_if(bad):
                    fail_native()
                st_store(5, b.select(bad, ci(2), c1))
                lds_st_f32(ci(32), total8)
        barrier()
        row_total = lds_ld_f32(ci(32))
        rms = x.rsq(x.fma(row_total, b.const_f32(1.0 / 5120.0), b.const_f32(1e-06)))
        rounded = trunc8(b.vector_add(ext8(data), ext8(skip)))
        norm = trunc8(
            b.vector_mul(
                b.vector_mul(ext8(rounded), b.vector_splat(rms, 8)), ext8(gamma)
            )
        )
        so = nrm_offset(valid, b.mul(offset, ci(2)))
        x.buf_st16(
            i4(rounded), od, so, 17
        )  # residual1 write-through (sc0 sc1; deviation, see README.md)
        x.buf_st16(i4(norm), nd, so, 3)
        q0, q1, ex = fp8_quant(norm)
        with b.scf_if(valid):
            store_fp8(T, row, column, q0, q1, ex, tid)
        with b.scf_if(b.cmp_eq(blk, c0)):
            scale_padding(T, tid)
        return b.smax(chunk_ok, st_load(5))

    # ---------------------------------------------------------------- fold_ready_wait (source.py 625-649)
    def fold_ready_wait(slot, layer, itid):
        with b.scf_if(
            b.cmp_lt(itid, ci(64))
        ):  # every fold counter has its 1 + 8 L publications
            # asm loop with FlyDSL's compiled loop body (FlyDSL's ISA): counter load, >= target, cndmask, min tree over
            # xor 1/2/4 (ds_swizzle), readfirstlane, failure-word load, s_sleep 1
            t = (
                "s_mov_b32 $0, 0\n1:\n global_load_dword $1, $8, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
                " v_cmp_ge_i32 vcc, $1, $9\n s_nop 1\n v_cndmask_b32_e64 $2, 0, 1, vcc\n"
                + min_tree("$2", ("$5", "$6"), (1, 2, 4))
                + " s_nop 0\n v_readfirstlane_b32 $3, $2\n global_load_dword $7, $10, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
                " s_sleep 1\n s_nop 0\n v_readfirstlane_b32 $4, $7\n s_add_u32 $0, $0, 1\n s_cmp_lg_u32 $4, 0\n"
                " s_cbranch_scc1 2f\n s_cmp_lg_u32 $3, 0\n s_cbranch_scc1 2f\n s_cmp_lt_u32 $0, 0x100000\n"
                " s_cbranch_scc1 1b\n2:"
            )
            f0 = rfl(coherent_u32(word(native, 289)))
            rr = b.inline_asm_multi(
                t,
                "=&s,=&v,=&v,=&s,=&s,=&v,=&v,=&v,v,v,v,~{vcc},~{scc}",
                [
                    word(
                        native,
                        b.add(ci(CHUNK_WORD + 512), b.mul(b.mod(itid, ci(8)), ci(16))),
                    ),
                    b.add(b.mul(layer, ci(8)), c1),
                    word(native, 289),
                ],
                result_types=[I32] * 8,
                result_name_hint="frpoll",
            )
            with b.scf_if(b.cmp_eq(itid, c0)):
                bad = b.lor(
                    b.cmp_ne(rr[3], c1), b.lor(b.cmp_ne(f0, c0), b.cmp_ne(rr[4], c0))
                )
                with b.scf_if(bad):
                    fail_native()
                asm("s_waitcnt vmcnt(0)\nbuffer_inv sc1", "~{memory}")
                st_store(slot, b.select(bad, ci(2), c1))
        barrier()
        return st_load(slot)

    # ---------------------------------------------------------------- qkv_stage / attention_qgkv_stage
    def qkv_common(T, layer, itid, blk, slot, attn):
        """FlyDSL qkv_stage (GDN, attn False; source.py 651-853) / attention_qgkv_stage (attn True; 1697-1802), the
        part inside ``if alive == 1`` (caller guards). T: i64 addresses A, AS, W, WS, raw (+ GDN: conv_w, hist,
        work, snap, BA). Returns the fold-wait status (all threads). M3's S6: on GDN layers the epilogue operands with
        no producer in this launch (BA weight k-block + scales, conv weights / history) are issued before the fold
        wait; the BA activation k-block after it."""
        tid, tile = itid, blk
        lane, wave = b.mod(tid, ci(64)), rfl(b.div(tid, ci(64)))
        lane4 = b.land(lane, ci(-4))
        nb, wsz, ssz = (896, 36700160, 2293760) if attn else (1032, 42270720, 2641920)
        ad = descriptor(T["A"], 40960)
        asd = descriptor(T["AS"], 2752)
        bd = descriptor(T["W"], wsz)
        bsd = descriptor(T["WS"], ssz)

        def woff(atom, kb, extra=None):
            o = b.add(
                b.mul(
                    b.add(b.mul(b.add(b.mul(tile, ci(4)), ci(atom)), ci(40)), kb),
                    ci(1024),
                ),
                b.mul(lane, ci(16)),
            )
            return o if extra is None else b.add(o, extra)

        def soff(atom, kb):
            return b.add(
                b.mul(
                    b.add(b.add(b.mul(kb, ci(nb)), b.mul(tile, ci(4))), ci(atom)),
                    ci(64),
                ),
                lane4,
            )

        bv, bs = [None] * 24, [None] * 24
        for r in range(6):  # P = 6 k-steps primed before the fold wait
            kb = b.add(ci(r * 4), wave)
            for atom in range(4):
                bv[r * 4 + atom] = x.buf_ld16(bd, woff(atom, kb), 2)
                bs[r * 4 + atom] = x.buf_ld4(bsd, soff(atom, kb), 0)
        bs2 = [None] * 16
        if (
            not attn
        ):  # k-steps 6-9 into LDS (DMA) and their scales into VGPRs, before the fold wait
            ldsw = rfl(b.add(b.add(arena_addr, ci(65536)), b.mul(wave, ci(16384))))
            for r2 in range(4):
                kb = b.add(ci((6 + r2) * 4), wave)
                for atom in range(4):
                    dma(b.add(ldsw, ci((r2 * 4 + atom) * 1024)), woff(atom, kb), bd)
                    bs2[r2 * 4 + atom] = x.buf_ld4(bsd, soff(atom, kb), 0)
        E = {}  # GDN epilogue operands (BA k-block, conv weights / history)

        def epi_idx():
            E["on"] = zx(b.cmp_ge(tile, ci(176)))
            job = b.smax(b.sub(tile, ci(176)), c0)
            E["tile_h"], peer_h = b.div(job, ci(10)), b.mod(job, ci(10))
            E["kb"] = b.add(b.mul(peer_h, ci(4)), wave)

        def epi_aoff():
            E["aoff"] = b.add(
                b.add(b.mul(b.mod(lane, ci(16)), ci(5120)), b.mul(E["kb"], ci(128))),
                b.mul(b.div(lane, ci(16)), ci(16)),
            )

        def epi_oob():
            E["oob"] = b.mul(b.sub(c1, E["on"]), ci(0x10000000))

        def epi_act():  # needs the fold output (input_A / input_AS)
            aoff, oob, kb = E["aoff"], E["oob"], E["kb"]
            E["lo"] = x.buf_ld16(ad, b.add(aoff, oob), 0)
            E["hi"] = x.buf_ld16(ad, b.add(b.add(aoff, ci(64)), oob), 0)
            E["sa"] = x.buf_ld4(asd, b.add(b.add(b.mul(kb, ci(64)), lane4), oob), 0)

        def epi_w():  # weights and conv history only (no producer in this launch)
            oob, kb, th = E["oob"], E["kb"], E["tile_h"]
            E["b"] = x.buf_ld16(
                bd,
                b.add(
                    b.add(
                        b.mul(b.add(b.mul(b.add(ci(1024), th), ci(40)), kb), ci(1024)),
                        b.mul(lane, ci(16)),
                    ),
                    oob,
                ),
                0,
            )
            E["sb"] = x.buf_ld4(
                bsd,
                b.add(
                    b.add(
                        b.mul(b.add(b.add(b.mul(kb, ci(1032)), ci(1024)), th), ci(64)),
                        lane4,
                    ),
                    oob,
                ),
                0,
            )
            conv_channel = b.add(b.mul(b.smin(tile, ci(159)), ci(64)), lane)
            cw0 = x.gep(x.ptr1(T["conv_w"]), "bfloat", b.mul(conv_channel, ci(4)))
            ch0 = x.gep(x.ptr1(T["hist"]), "bfloat", b.mul(conv_channel, ci(4)))
            E["cw"] = [
                b.cast_to_f32(
                    x.load(
                        cw0 if t_ == 0 else x.gep(cw0, "bfloat", ci(t_)),
                        "bfloat",
                        BF16,
                        2,
                    )
                )
                for t_ in range(4)
            ]
            E["ch"] = [
                b.cast_to_f32(
                    x.load(x.gep(ch0, "bfloat", ci(1 + t_)), "bfloat", BF16, 2)
                )
                for t_ in range(3)
            ]

        if (
            not attn
        ):  # S6: the producer-free epilogue operands leave before the fold wait
            epi_idx()
            epi_oob()
            epi_w()
        result = fold_ready_wait(slot, layer, itid)
        with b.scf_if(b.cmp_eq(result, c1)):
            if not attn:  # the BA activation k-block (fold output) after the wait
                epi_aoff()
                epi_act()
                ba_lo, ba_hi, ba_sa, ba_b, ba_sb = (
                    E["lo"],
                    E["hi"],
                    E["sa"],
                    E["b"],
                    E["sb"],
                )
                conv_w, conv_h = E["cw"], E["ch"]
            alo, ahi, asc = [], [], []
            for r in range(10):  # every activation k-block in one round trip
                kb = b.add(ci(r * 4), wave)
                off = b.add(
                    b.add(b.mul(b.mod(lane, ci(16)), ci(5120)), b.mul(kb, ci(128))),
                    b.mul(b.div(lane, ci(16)), ci(16)),
                )
                alo.append(x.buf_ld16(ad, off, 0))
                ahi.append(x.buf_ld16(ad, b.add(off, ci(64)), 0))
                asc.append(x.buf_ld4(asd, b.add(b.mul(kb, ci(64)), lane4), 0))
            acc = [b.zero_vec_f32(4) for _ in range(4)]
            sh = b.mul(b.land(lane, ci(3)), ci(8))
            z = c0
            for k in range(10):
                br = k % 6
                lo, hi, sa = pin(alo[k]), pin(ahi[k]), pin(asc[k])
                sa = b.land(b.lshr(sa, sh), ci(255))
                av = b.vec_pack(
                    [b.vec_extract(lo, e) for e in range(4)]
                    + [b.vec_extract(hi, e) for e in range(4)],
                    I32,
                )
                for atom in range(4):
                    if k < 6 or attn:
                        bb = pin(bv[br * 4 + atom])
                        sb = pin(bs[br * 4 + atom])
                    else:  # from LDS (the DMA landed before the activation wait)
                        bb = lds_ld16(
                            b.add(
                                b.add(
                                    ci(65536 + ((k - 6) * 4 + atom) * 1024),
                                    b.mul(wave, ci(16384)),
                                ),
                                b.mul(lane, ci(16)),
                            )
                        )
                        sb = pin(bs2[(k - 6) * 4 + atom])
                    sb = b.land(b.lshr(sb, sh), ci(255))
                    bf = b.vec_pack(
                        [b.vec_extract(bb, e) for e in range(4)] + [z] * 4, I32
                    )
                    acc[atom] = x.mfma_scale(av, bf, acc[atom], sa, sb, 0, 4)
                if attn and k + 6 < 10:  # attention refill (GDN: k + 6 < 6 never)
                    kb = b.add(ci((k + 6) * 4), wave)
                    for atom in range(4):
                        bv[br * 4 + atom] = x.buf_ld16(bd, woff(atom, kb), 2)
                        bs[br * 4 + atom] = x.buf_ld4(bsd, soff(atom, kb), 0)
            for atom in range(4):
                for e in range(4):
                    stf(
                        lf(
                            b.add(
                                b.add(ci(atom * 1024), b.mul(wave, ci(256))),
                                b.add(b.mul(lane, ci(4)), ci(e)),
                            )
                        ),
                        b.vec_extract(acc[atom], e),
                    )
            barrier()
            with b.scf_if(b.cmp_eq(wave, c0)):
                for atom in range(4):
                    folded = []
                    for e in range(4):
                        p = lf(b.add(ci(atom * 1024), b.add(b.mul(lane, ci(4)), ci(e))))
                        val = ldf(p)
                        for w in range(1, 4):
                            val = b.fadd(
                                val, ldf(x.gep(p, "float", ci(w * 256), lds=True))
                            )
                        folded.append(val)
                    rounded = b.vec_cast_f32_to(b.vec_pack(folded, F32), BF16)
                    for e in range(4):
                        x.store(
                            lbp(
                                b.add(
                                    b.add(
                                        ci(atom * 320),
                                        b.mul(
                                            b.add(
                                                b.mul(b.div(lane, ci(16)), ci(4)), ci(e)
                                            ),
                                            ci(20),
                                        ),
                                    ),
                                    b.mod(lane, ci(16)),
                                )
                            ),
                            b.vec_extract(rounded, e),
                            "bfloat",
                            2,
                            lds=True,
                        )
            barrier()
            rstride = 14336 if attn else 16512
            with b.scf_if(b.cmp_eq(wave, c0)):
                row, col = b.div(lane, ci(4)), b.mul(b.mod(lane, ci(4)), ci(4))
                with b.scf_if(b.cmp_lt(row, ci(8))):
                    for atom in range(4):
                        vals = x.load(
                            lbp(b.add(b.add(ci(atom * 320), b.mul(row, ci(20))), col)),
                            "<4 x bfloat>",
                            VectorType(BF16, 4),
                            2,
                            lds=True,
                        )
                        idx = b.add(
                            b.add(b.mul(row, ci(rstride)), b.mul(tile, ci(64))),
                            b.add(ci(atom * 16), col),
                        )
                        asm(
                            "global_store_dwordx2 $0, $1, off sc0 sc1",
                            "v,v,~{memory}",
                            [
                                b.add(T["raw"], i64(b.mul(idx, ci(2)))),
                                b.vec_bitcast(vals, VectorType(I32, 2)),
                            ],
                        )
            if not attn:
                barrier()
                with b.scf_if(b.cmp_lt(tile, ci(160))):
                    channel = b.add(b.mul(tile, ci(64)), lane)
                    for pair in range(2):
                        row = b.add(b.mul(wave, ci(2)), ci(pair))
                        value = b.const_f32(0.0)
                        for tap in range(4):
                            source = b.add(row, ci(tap - 3))
                            hv = b.select(
                                b.cmp_eq(source, ci(-3)), conv_h[0], conv_h[2]
                            )
                            hv = b.select(b.cmp_eq(source, ci(-2)), conv_h[1], hv)
                            xv = if_yield(
                                b.cmp_ge(source, c0),
                                [hv],
                                lambda cur, source=source: [
                                    b.cast_to_f32(
                                        x.load(
                                            lbp(
                                                b.add(
                                                    b.add(
                                                        b.mul(
                                                            b.div(lane, ci(16)), ci(320)
                                                        ),
                                                        b.mul(source, ci(20)),
                                                    ),
                                                    b.mod(lane, ci(16)),
                                                )
                                            ),
                                            "bfloat",
                                            BF16,
                                            2,
                                            lds=True,
                                        )
                                    )
                                ],
                            )[0]
                            value = x.fma(xv, conv_w[tap], value)
                        expo = b.fmul(b.fneg(value), b.const_f32(1.4426950408889634))
                        expn = asm(
                            "v_exp_f32 $0, $1",
                            "=v,v",
                            [expo],
                            F32,
                            se=False,
                            hint="vexp",
                        )
                        inv = asm(
                            "v_rcp_f32 $0, $1",
                            "=v,v",
                            [b.fadd(b.const_f32(1.0), expn)],
                            F32,
                            se=False,
                            hint="vrcp",
                        )
                        pv = b.vec_pack(
                            [
                                b.trunc_f32_to_bf16(b.fmul(value, inv)),
                                b.trunc_f32_to_bf16(b.const_f32(0.0)),
                            ],
                            BF16,
                        )
                        ws = b.vec_extract(b.vec_bitcast(pv, VectorType(I32, 1)), 0)
                        asm(
                            "global_store_short $0, $1, off sc0 sc1",
                            "v,v,~{memory}",
                            [
                                b.add(
                                    T["work"],
                                    i64(
                                        b.mul(
                                            b.add(b.mul(row, ci(16512)), channel), ci(2)
                                        )
                                    ),
                                ),
                                ws,
                            ],
                        )
                    saved = x.load(
                        lbp(
                            b.add(
                                b.add(
                                    b.mul(b.div(lane, ci(16)), ci(320)),
                                    b.mul(b.add(ci(4), wave), ci(20)),
                                ),
                                b.mod(lane, ci(16)),
                            )
                        ),
                        "bfloat",
                        BF16,
                        2,
                        lds=True,
                    )
                    x.store(
                        x.gep(
                            x.ptr1(T["snap"]),
                            "bfloat",
                            b.add(b.mul(channel, ci(4)), wave),
                        ),
                        saved,
                        "bfloat",
                        2,
                    )
                barrier()
                with b.scf_if(b.cmp_ge(tile, ci(176))):
                    job = b.sub(tile, ci(176))
                    ba_tile = b.div(job, ci(10))
                    sa = b.land(b.lshr(ba_sa, sh), ci(255))
                    sb = b.land(b.lshr(ba_sb, sh), ci(255))
                    av = b.vec_pack(
                        [b.vec_extract(ba_lo, e) for e in range(4)]
                        + [b.vec_extract(ba_hi, e) for e in range(4)],
                        I32,
                    )
                    bf = b.vec_pack(
                        [b.vec_extract(ba_b, e) for e in range(4)] + [z] * 4, I32
                    )
                    ba_acc = x.mfma_scale(av, bf, b.zero_vec_f32(4), sa, sb, 0, 4)
                    for e in range(4):
                        stf(
                            lf(
                                b.add(
                                    b.mul(wave, ci(256)),
                                    b.add(b.mul(lane, ci(4)), ci(e)),
                                )
                            ),
                            b.vec_extract(ba_acc, e),
                        )
                    barrier()
                    with b.scf_if(b.cmp_eq(wave, c0)):
                        folded = []
                        for e in range(4):
                            p = lf(b.add(b.mul(lane, ci(4)), ci(e)))
                            val = ldf(p)
                            for w in range(1, 4):
                                val = b.fadd(
                                    val, ldf(x.gep(p, "float", ci(w * 256), lds=True))
                                )
                            folded.append(val)
                        for e in range(4):
                            row = b.add(b.mul(b.div(lane, ci(16)), ci(4)), ci(e))
                            stf(
                                lf(b.add(b.mul(row, ci(20)), b.mod(lane, ci(16)))),
                                folded[e],
                            )
                    barrier()
                    if (
                        PIN
                    ):  # pin twin PB: the 10 K-split tiles of a BA tile add in tile order
                        with b.scf_if(b.cmp_eq(itid, c0)):
                            bp_t = b.add(
                                b.mul(b.sub(layer, b.div(layer, ci(4))), ci(10)),
                                b.mod(job, ci(10)),
                            )
                            t = (
                                "s_mov_b32 $0, 0\n1:\n global_load_dword $1, $2, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
                                " s_sleep 1\n s_add_u32 $0, $0, 1\n v_cmp_ge_i32 vcc, $1, $3\n s_cbranch_vccnz 2f\n"
                                " s_cmp_lt_u32 $0, 0x400000\n s_cbranch_scc1 1b\n2:"
                            )
                            rr = b.inline_asm_multi(
                                t,
                                "=&s,=&v,v,v,~{vcc},~{scc}",
                                [
                                    word(
                                        native,
                                        b.add(ci(BA_PIN), b.mul(ba_tile, ci(16))),
                                    ),
                                    bp_t,
                                ],
                                result_types=[I32, I32],
                                result_name_hint="bapoll",
                            )
                            with b.scf_if(b.cmp_lt(rr[1], bp_t)):
                                fail_native()
                        barrier()
                    with b.scf_if(b.cmp_eq(wave, c0)):
                        row, col = b.div(lane, ci(4)), b.mul(b.mod(lane, ci(4)), ci(4))
                        with b.scf_if(b.cmp_lt(row, ci(8))):
                            pb = x.gep(
                                x.ptr1(T["BA"]),
                                "float",
                                b.add(
                                    b.add(b.mul(row, ci(128)), b.mul(ba_tile, ci(16))),
                                    col,
                                ),
                            )
                            pl = lf(b.add(b.mul(row, ci(20)), col))
                            for e in range(4):
                                v = ldf(
                                    pl
                                    if e == 0
                                    else x.gep(pl, "float", ci(e), lds=True)
                                )
                                x.atomic_fadd_agent(
                                    pb if e == 0 else x.gep(pb, "float", ci(e)), v
                                )
                        if PIN:
                            wait_vm(0)
                            with b.scf_if(b.cmp_eq(itid, c0)):
                                system_atomic_add_nw(
                                    word(
                                        native,
                                        b.add(ci(BA_PIN), b.mul(ba_tile, ci(16))),
                                    )
                                )
                    barrier()
            wait_vm(0)  # outputs drained -> head counters
            barrier()
            with b.scf_if(b.cmp_eq(tid, c0)):
                if attn:
                    hc = b.select(
                        b.cmp_ge(tile, ci(192)),
                        b.add(
                            ci(24), b.div(b.mod(b.sub(tile, ci(192)), ci(16)), ci(4))
                        ),
                        b.div(tile, ci(8)),
                    )
                    system_atomic_add_nw(
                        word(native, b.add(ci(CHUNK_WORD + 3200), b.mul(hc, ci(16))))
                    )
                else:
                    vh = b.div(
                        b.sub(
                            b.sub(tile, ci(64)),
                            b.mul(zx(b.cmp_ge(tile, ci(160))), ci(96)),
                        ),
                        ci(2),
                    )
                    hc = b.select(
                        b.cmp_lt(tile, ci(64)),
                        b.add(
                            ci(2048), b.mul(b.div(b.mod(tile, ci(32)), ci(2)), ci(16))
                        ),
                        b.add(ci(2304), b.mul(vh, ci(16))),
                    )
                    system_atomic_add_nw(word(native, b.add(ci(CHUNK_WORD), hc)))
                    with b.scf_if(b.cmp_ge(tile, ci(176))):
                        system_atomic_add_nw(
                            word(
                                native,
                                b.add(
                                    ci(CHUNK_WORD + 3072),
                                    b.mul(b.div(b.sub(tile, ci(176)), ci(10)), ci(16)),
                                ),
                            )
                        )
        return result

    # ================================================================ output projection, mixer relay,
    # post-attention norm owners, finalize / reduce stand-ins (source 1150-1270, 2766-2799, 2940-2995)
    PIN_WORD = 8192  # native words 7,824-12,047 unused by the m8 source; zeroed by launch_begin

    def iatomic_add_agent(
        p, v
    ):  # fx.atomic_add(i32, Monotonic, syncscope agent) on a typed pointer
        return inst(
            b,
            'atomicrmw add ptr addrspace(1) {0}, i32 {1} syncscope("agent") monotonic, align 4',
            [p, v],
            I32,
            "iaa",
        )

    def relay_mixer_slices(layer, itid):
        """source 2766-2792: one CTA per XCD turns the 12 global mixer sub-counters (640 + 16 k, target 16 (L + 1)) into
        3 XCD-local words (1408 + 16 (3 xcc + s)). Same asm loop as relay_up_slices (uniform target).
        """
        with b.scf_if(b.cmp_lt(itid, ci(64))):
            lane = itid
            k12 = b.mod(lane, ci(12))
            need = b.mul(b.add(layer, c1), ci(16))
            addr = word(native, b.add(ci(CHUNK_WORD + 640), b.mul(k12, ci(16))))
            ra = [
                word(
                    native,
                    b.add(
                        ci(CHUNK_WORD + 1408),
                        b.mul(b.add(b.mul(safe_xcc, ci(3)), ci(s_)), ci(16)),
                    ),
                )
                for s_ in range(3)
            ]
            grp = "".join(
                f" s_and_b32 $3, vcc_lo, {0xF << (4 * s_):#x}\n s_cmp_eq_u32 $3, {0xF << (4 * s_):#x}\n"
                f" s_cselect_b32 $3, {1 << s_}, 0\n s_or_b32 $2, $2, $3\n"
                for s_ in range(3)
            )
            sto = "".join(
                f" s_bitcmp1_b32 $3, {s_}\n s_cbranch_scc0 1{s_ + 3}f\n global_store_dword ${9 + s_}, $12, off\n"
                f"1{s_ + 3}:\n"
                for s_ in range(3)
            )
            t = (
                "s_mov_b32 $0, 0\n s_mov_b32 $1, 0\n1:\n global_load_dword $5, $7, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
                " v_cmp_ge_i32 vcc, $5, $8\n s_mov_b32 $2, 0\n"
                + grp
                + " s_andn2_b32 $3, $2, $1\n s_cmp_eq_u32 $3, 0\n s_cbranch_scc1 3f\n"
                " s_mov_b64 $4, exec\n s_mov_b64 exec, 1\n"
                + sto
                + " s_mov_b64 exec, $4\n3:\n"
                " s_or_b32 $1, $1, $2\n global_load_dword $6, $13, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
                " v_readfirstlane_b32 $3, $6\n s_sleep 1\n s_add_u32 $0, $0, 1\n"
                " s_cmp_eq_u32 $1, 7\n s_cbranch_scc1 2f\n s_cmp_lg_u32 $3, 0\n s_cbranch_scc1 2f\n"
                " s_cmp_lt_u32 $0, 0x100000\n s_cbranch_scc1 1b\n2:"
            )
            rr = b.inline_asm_multi(
                t,
                "=&s,=&s,=&s,=&s,=&s,=&v,=&v,v,v,v,v,v,v,v,~{vcc},~{scc}",
                [addr, need, ra[0], ra[1], ra[2], b.add(layer, c1), word(native, 289)],
                result_types=[I32, I32, I32, I32, I64, I32, I32],
                result_name_hint="mrelay",
            )
            with b.scf_if(b.land(b.cmp_eq(itid, c0), b.cmp_ne(rr[1], ci(7)))):
                fail_native()
        barrier()

    def output_stage(T, tbl, layer, itid, blk, alive, slot, attn):
        """source 1150-1270: tile = b // 3 (64 columns), K-slice p = b % 3 (16 k-blocks); 16 weight + 16 scale loads
        (aux 2 / 0) primed before the gate, the Up L2 touches of blocks 192-239 (GDN layers) behind the prime, tid 0
        scalar poll of the XCD-local relayed word, mixer_A loads aux 17, 4 k x 4 atoms scaled MFMAs, the 4-wave LDS sum
        (qkv form), bf16, demangle, ``global_atomic_pk_add_bf16`` into ``mixer``, vmcnt(0), output counter. T: i64
        addresses A, AS, W, WS, mixer."""
        lane, wave = b.mod(itid, ci(64)), rfl(b.div(itid, ci(64)))
        tile, peer = b.div(blk, ci(3)), b.mod(blk, ci(3))
        with b.scf_if(b.land(b.cmp_eq(alive, c1), b.cmp_lt(blk, ci(240)))):
            ad = descriptor(T["A"], 49152)
            asd = descriptor(T["AS"], 3264)
            bd = descriptor(T["W"], 15728640)
            bsd = descriptor(T["WS"], 983040)
            lane4 = b.land(lane, ci(-4))

            def kb_of(r):
                return b.add(b.add(b.mul(peer, ci(16)), ci(r * 4)), wave)

            bv, bs = [None] * 16, [None] * 16
            for r in range(4):
                kb = kb_of(r)
                for atom in range(4):
                    base = b.add(b.mul(b.add(b.mul(tile, ci(4)), ci(atom)), ci(48)), kb)
                    bv[r * 4 + atom] = x.buf_ld16(
                        bd, b.add(b.mul(base, ci(1024)), b.mul(lane, ci(16))), 2
                    )
                    bs[r * 4 + atom] = x.buf_ld4(
                        bsd, b.add(b.mul(base, ci(64)), lane4), 0
                    )
            gdn = b.lnot(attn)
            with b.scf_if(
                b.land(b.land(b.cmp_ge(blk, ci(192)), b.cmp_lt(blk, ci(224))), gdn)
            ):
                g8 = b.div(b.sub(blk, ci(192)), ci(8))
                up_l2_touch(tbl, itid, blk, b.add(ci(3), g8), 1, 5)
                up_l2_touch(tbl, itid, blk, b.add(ci(7), b.mul(ci(2), g8)), 2, 1, 16)
            with b.scf_if(b.land(b.cmp_ge(blk, ci(224)), gdn)):
                up_l2_touch(
                    tbl,
                    itid,
                    blk,
                    b.add(ci(15), b.mul(ci(7), b.div(b.sub(blk, ci(224)), ci(8)))),
                    7,
                    1,
                    16,
                )
            with b.scf_if(b.cmp_eq(itid, c0)):
                a = word(
                    native,
                    b.add(
                        ci(CHUNK_WORD + 1408),
                        b.mul(b.add(b.mul(safe_xcc, ci(3)), peer), ci(16)),
                    ),
                )
                sa = b.lor(
                    b.zext(rfl(b.trunc(a, I32)), I64),
                    b.shl(
                        b.zext(rfl(b.trunc(b.lshr(a, b.const_i64(32)), I32)), I64),
                        b.const_i64(32),
                    ),
                )
                ready = poll_ge(sa, b.add(layer, c1), ci(4194304), scalar=True)
                with b.scf_if(b.cmp_eq(ready, c0)):
                    fail_native()
                st_store(slot, b.select(b.cmp_eq(ready, c0), ci(2), c1))
            barrier()
            res = st_load(slot)
            with b.scf_if(b.cmp_eq(res, c1)):
                alo, ahi, asc = [], [], []
                for r in range(4):
                    kb = kb_of(r)
                    off = b.add(
                        b.add(b.mul(b.mod(lane, ci(16)), ci(6144)), b.mul(kb, ci(128))),
                        b.mul(b.div(lane, ci(16)), ci(16)),
                    )
                    alo.append(x.buf_ld16(ad, off, 17))
                    ahi.append(x.buf_ld16(ad, b.add(off, ci(64)), 17))
                    asc.append(x.buf_ld4(asd, b.add(b.mul(kb, ci(64)), lane4), 17))
                acc = [b.zero_vec_f32(4) for _ in range(4)]
                sh = b.mul(b.land(lane, ci(3)), ci(8))
                z = c0
                for k in range(4):
                    lo, hi, sa_ = pin(alo[k]), pin(ahi[k]), pin(asc[k])
                    sa_ = b.land(b.lshr(sa_, sh), ci(255))
                    av = b.vec_pack(
                        [b.vec_extract(lo, e) for e in range(4)]
                        + [b.vec_extract(hi, e) for e in range(4)],
                        I32,
                    )
                    for atom in range(4):
                        bb = pin(bv[k * 4 + atom])
                        sb = b.land(b.lshr(pin(bs[k * 4 + atom]), sh), ci(255))
                        bf = b.vec_pack(
                            [b.vec_extract(bb, e) for e in range(4)] + [z] * 4, I32
                        )
                        acc[atom] = x.mfma_scale(av, bf, acc[atom], sa_, sb, 0, 4)
                for atom in range(4):
                    for e in range(4):
                        stf(
                            lf(
                                b.add(
                                    b.add(ci(atom * 1024), b.mul(wave, ci(256))),
                                    b.add(b.mul(lane, ci(4)), ci(e)),
                                )
                            ),
                            b.vec_extract(acc[atom], e),
                        )
                barrier()
                with b.scf_if(b.cmp_eq(wave, c0)):
                    for atom in range(4):
                        folded = []
                        for e in range(4):
                            p = lf(
                                b.add(ci(atom * 1024), b.add(b.mul(lane, ci(4)), ci(e)))
                            )
                            val = ldf(p)
                            for w in range(1, 4):
                                val = b.fadd(
                                    val, ldf(x.gep(p, "float", ci(w * 256), lds=True))
                                )
                            folded.append(val)
                        rounded = b.vec_cast_f32_to(b.vec_pack(folded, F32), BF16)
                        for e in range(4):
                            x.store(
                                lbp(
                                    b.add(
                                        b.add(
                                            ci(atom * 320),
                                            b.mul(
                                                b.add(
                                                    b.mul(b.div(lane, ci(16)), ci(4)),
                                                    ci(e),
                                                ),
                                                ci(20),
                                            ),
                                        ),
                                        b.mod(lane, ci(16)),
                                    )
                                ),
                                b.vec_extract(rounded, e),
                                "bfloat",
                                2,
                                lds=True,
                            )
                barrier()
                if MXPIN:  # pin twin: the K-slices of a tile add in peer order
                    with b.scf_if(b.cmp_eq(itid, c0)):
                        t = (
                            "s_mov_b32 $0, 0\n1:\n global_load_dword $1, $2, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
                            " s_sleep 1\n s_add_u32 $0, $0, 1\n v_cmp_ge_i32 vcc, $1, $3\n s_cbranch_vccnz 2f\n"
                            " s_cmp_lt_u32 $0, 0x400000\n s_cbranch_scc1 1b\n2:"
                        )
                        rr = b.inline_asm_multi(
                            t,
                            "=&s,=&v,v,v,~{vcc},~{scc}",
                            [
                                word(native, b.add(ci(PIN_WORD), b.mul(tile, ci(16)))),
                                b.add(b.mul(layer, ci(3)), peer),
                            ],
                            result_types=[I32, I32],
                            result_name_hint="pinpoll",
                        )
                        with b.scf_if(
                            b.cmp_lt(rr[1], b.add(b.mul(layer, ci(3)), peer))
                        ):
                            fail_native()
                    barrier()
                with b.scf_if(b.cmp_eq(wave, c0)):
                    row, col = b.div(lane, ci(4)), b.mul(b.mod(lane, ci(4)), ci(4))
                    with b.scf_if(b.cmp_lt(row, ci(8))):
                        dsl = raw_table_address(64, 6) if DUMP else None
                        for atom in range(4):
                            vals = x.load(
                                lbp(
                                    b.add(
                                        b.add(ci(atom * 320), b.mul(row, ci(20))), col
                                    )
                                ),
                                "<4 x bfloat>",
                                VectorType(BF16, 4),
                                2,
                                lds=True,
                            )
                            pk = b.vec_bitcast(vals, VectorType(I32, 2))
                            idx = b.add(
                                b.add(b.mul(row, ci(5120)), b.mul(tile, ci(64))),
                                b.add(ci(atom * 16), col),
                            )
                            if DUMP:
                                di = b.add(
                                    b.mul(
                                        b.add(
                                            b.mul(
                                                b.add(b.mul(layer, ci(3)), peer), ci(8)
                                            ),
                                            row,
                                        ),
                                        ci(5120),
                                    ),
                                    b.sub(idx, b.mul(row, ci(5120))),
                                )
                                asm(
                                    "global_store_dwordx2 $0, $1, off sc0 sc1",
                                    "v,v,~{memory}",
                                    [b.add(dsl, i64(b.mul(di, ci(2)))), pk],
                                )
                            for pair in range(2):
                                asm(
                                    "global_atomic_pk_add_bf16 $0, $1, off",
                                    "v,v,~{memory}",
                                    [
                                        b.add(
                                            T["mixer"],
                                            i64(b.mul(b.add(idx, ci(pair * 2)), ci(2))),
                                        ),
                                        b.vec_extract(pk, pair),
                                    ],
                                )
                    wait_vm(0)  # mixer atomics acknowledged -> output chunk counter
                    with b.scf_if(b.cmp_eq(itid, c0)):
                        system_atomic_add_nw(
                            word(
                                native,
                                b.add(
                                    ci(CHUNK_WORD + 128),
                                    b.mul(b.div(tile, ci(10)), ci(16)),
                                ),
                            )
                        )
                        if MXPIN:
                            system_atomic_add_nw(
                                word(native, b.add(ci(PIN_WORD), b.mul(tile, ci(16))))
                            )
                barrier()
        return st_load(slot)

    def owner_output_gate(layer, itid):
        """source 2963-2985: lanes 0-63 poll the 8 output counters (128 + 16 (lane % 8), target 30 (L + 1)); asm loop
        with FlyDSL's compiled body (FlyDSL's ISA: counter load, >=, cndmask, min tree xor 1/2/4, readfirstlane,
        failure-word load, s_sleep 1); tid 0: s_waitcnt vmcnt(0) + buffer_inv sc1, status slot 3.
        """
        with b.scf_if(b.cmp_lt(itid, ci(64))):
            t = (
                "s_mov_b32 $0, 0\n1:\n global_load_dword $1, $8, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
                " v_cmp_ge_i32 vcc, $1, $9\n s_nop 1\n v_cndmask_b32_e64 $2, 0, 1, vcc\n"
                + min_tree("$2", ("$5", "$6"), (1, 2, 4))
                + " s_nop 0\n v_readfirstlane_b32 $3, $2\n global_load_dword $7, $10, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
                " s_sleep 1\n s_nop 0\n v_readfirstlane_b32 $4, $7\n s_add_u32 $0, $0, 1\n s_cmp_lg_u32 $4, 0\n"
                " s_cbranch_scc1 2f\n s_cmp_lg_u32 $3, 0\n s_cbranch_scc1 2f\n s_cmp_lt_u32 $0, 0x100000\n"
                " s_cbranch_scc1 1b\n2:"
            )
            f0 = rfl(coherent_u32(word(native, 289)))
            rr = b.inline_asm_multi(
                t,
                "=&s,=&v,=&v,=&s,=&s,=&v,=&v,=&v,v,v,v,~{vcc},~{scc}",
                [
                    word(
                        native,
                        b.add(ci(CHUNK_WORD + 128), b.mul(b.mod(itid, ci(8)), ci(16))),
                    ),
                    b.mul(b.add(layer, c1), ci(30)),
                    word(native, 289),
                ],
                result_types=[I32] * 8,
                result_name_hint="ogpoll",
            )
            with b.scf_if(b.cmp_eq(itid, c0)):
                bad = b.lor(
                    b.cmp_ne(rr[3], c1), b.lor(b.cmp_ne(f0, c0), b.cmp_ne(rr[4], c0))
                )
                with b.scf_if(bad):
                    fail_native()
                asm("s_waitcnt vmcnt(0)\nbuffer_inv sc1", "~{memory}")
                st_store(3, b.select(bad, ci(2), c1))
        barrier()
        return st_load(3)

    def gdn_gate(layer, itid, offneed, hint):
        """FlyDSL's QKV -> core gate loop (core 857-892 / core tail 1053-1080; FlyDSL's compiled ISA: the per-lane
        word / target selects hoisted, loop = counter load, >=, s_sleep 1, cndmask, min tree xor 1..32, readfirstlane);
        tid 0: failure CAS, ``vmcnt(0)`` + ``buffer_inv sc1``, status slot 2. ``offneed(lane, d)`` -> (word, target).
        """
        with b.scf_if(b.cmp_lt(itid, ci(64))):
            d_ = b.add(b.sub(layer, b.div(layer, ci(4))), c1)
            off, need = offneed(itid, d_)
            f0 = rfl(coherent_u32(word(native, 289)))
            t = (
                "s_mov_b32 $0, 0\n1:\n global_load_dword $1, $6, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
                " v_cmp_ge_i32 vcc, $1, $7\n s_sleep 1\n s_nop 1\n v_cndmask_b32_e64 $2, 0, 1, vcc\n"
                + min_tree("$2", ("$3", "$4"), (1, 2, 4, 8, 16, 32))
                + " s_nop 0\n v_readfirstlane_b32 $5, $2\n s_add_u32 $0, $0, 1\n s_cmp_lg_u32 $5, 0\n s_cbranch_scc1 2f\n"
                " s_cmp_lt_u32 $0, 0x100000\n s_cbranch_scc1 1b\n2:"
            )
            rr = b.inline_asm_multi(
                t,
                "=&s,=&v,=&v,=&v,=&v,=&s,v,v,~{vcc},~{scc}",
                [word(native, b.add(ci(CHUNK_WORD), off)), need],
                result_types=[I32] * 6,
                result_name_hint=hint,
            )
            with b.scf_if(b.cmp_eq(itid, c0)):
                bad = b.lor(b.cmp_eq(rr[5], c0), b.cmp_ne(f0, c0))
                with b.scf_if(bad):
                    fail_native()
                asm("s_waitcnt vmcnt(0)\nbuffer_inv sc1", "~{memory}")
                st_store(2, b.select(bad, ci(2), c1))
        barrier()

    def core_offneed(blk):
        def f(lane, d_):
            h = b.div(blk, ci(4))
            off = b.select(
                b.cmp_eq(lane, c0), b.add(ci(2048), b.mul(b.div(h, ci(3)), ci(16))), c0
            )
            off = b.select(b.cmp_eq(lane, c1), b.add(ci(2304), b.mul(h, ci(16))), off)
            off = b.select(
                b.cmp_eq(lane, ci(2)),
                b.add(ci(3072), b.mul(b.div(h, ci(16)), ci(16))),
                off,
            )
            off = b.select(
                b.cmp_eq(lane, ci(3)),
                b.add(ci(3072), b.mul(b.div(b.add(ci(48), h), ci(16)), ci(16))),
                off,
            )
            need = b.select(b.cmp_lt(lane, ci(2)), b.mul(d_, ci(4)), c0)
            need = b.select(
                b.land(b.cmp_ge(lane, ci(2)), b.cmp_lt(lane, ci(4))),
                b.mul(d_, ci(10)),
                need,
            )
            return off, need

        return f

    def tail_offneed(
        lane, d_
    ):  # source 1060-1066: 48 value-head counters, 8 BA counters, lanes >= 56 need 0
        off = b.add(ci(2304), b.mul(b.smin(lane, ci(47)), ci(16)))
        need = b.mul(d_, ci(4))
        hi = b.cmp_ge(lane, ci(48))
        off = b.select(
            hi, b.add(ci(3072), b.mul(b.smin(b.sub(lane, ci(48)), ci(7)), ci(16))), off
        )
        need = b.select(hi, b.mul(d_, ci(10)), need)
        need = b.select(b.cmp_ge(lane, ci(56)), c0, need)
        return off, need

    def mixer_pub_gdn(
        itid, blk
    ):  # finalize's mixer sub-counter 640 + 16 (4 (b // 64) + b % 4) (source 2923-2924)
        with b.scf_if(b.cmp_eq(itid, c0)):
            system_atomic_add_nw(
                word(
                    native,
                    b.add(
                        ci(CHUNK_WORD + 640),
                        b.mul(
                            b.add(b.mul(b.div(blk, ci(64)), ci(4)), b.mod(blk, ci(4))),
                            ci(16),
                        ),
                    ),
                )
            )

    # ================================================================ GDN core / core tail / peer hand-off /
    # finalize (source 855-1148, 2638-2686, 2909-2924). Every op in source order; FlyDSL's typed global / LDS
    # accesses as ``getelementptr`` + ``load``/``store``; DPP / shuffles / inline asm as FlyDSL's IR has them.
    PEER_WORD, BA_PIN = 400, 9728
    LOG2E, LN2 = 1.4426950408889634, 0.6931471805599453
    LQ, LK, LV, LDEC, LBETA, LCORE = (
        0,
        1024,
        4096,
        2176,
        2184,
        2192,
    )  # arena element offsets (f32, f32, bf16, f32 ...)

    def fconst(v):
        return b.const_f32(v)

    def vasm(
        op, v
    ):  # llvm.inline_asm(T.f32, [v], '<op> $0, $1', '=v,v', has_side_effects=False)
        return asm(f"{op} $0, $1", "=v,v", [v], F32, se=False, hint=op[2:5])

    def sum16(
        v,
    ):  # core_sum16: row_shl 8 / 4 / 2 / 1 adds (update.dpp 0x100 + n, zero fill)
        for n in (8, 4, 2, 1):
            v = b.fadd(v, x.update_dpp_f32(v, 0x100 + n))
        return v

    def bperm(v, src):  # ds_bpermute of an f32 / i32 value from lane ``src``
        if v.type == F32:
            return b.bitcast(x.bpermute(b.shl(src, ci(2)), b.bitcast(v, I32)), F32)
        return x.bpermute(b.shl(src, ci(2)), v)

    def shfl_down(
        v, s, width=64
    ):  # gpu.shuffle_down(v, s, width): source lane me + s while inside the width segment
        me = b.lane_id()
        src = b.add(me, ci(s))
        return bperm(
            v,
            b.select(b.cmp_lt(src, b.land(b.add(me, ci(width)), ci(-width))), src, me),
        )

    def shfl_idx(v, idx):  # gpu.shuffle_idx(v, idx, 64)
        me = b.lane_id()
        i_ = ci(idx) if isinstance(idx, int) else idx
        return bperm(
            v, b.select(b.cmp_lt(i_, b.land(b.add(me, ci(64)), ci(-64))), i_, me)
        )

    _ty = {"float": ("float", F32, 4), "bfloat": ("bfloat", BF16, 2)}

    def gp(
        p, elem, var, k=0, lds=False
    ):  # FlyDSL ``base + var + k``: getelementptr var, then the constant one
        q = x.gep(p, elem, var, lds=lds)
        return q if k == 0 else x.gep(q, elem, ci(k), lds=lds)

    def ldp(p, elem, lds=False):  # fx.ptr_load of a typed pointer
        ty, rt, al = _ty[elem]
        return x.load(p, ty, rt, al, lds=lds)

    def stp(p, elem, v, lds=False):  # fx.ptr_store to a typed pointer
        x.store(p, v, elem, _ty[elem][2], lds=lds)

    def bf2f(v):
        return b.cast_to_f32(v)

    def core_stage(T, layer, itid, blk):
        """source 855-1049, blocks 0-191: value head h = b // 4, value columns 32 (b % 4) + 2 (tid // 16) + {0, 1}; lanes
        tid % 16 hold keys lane + 16 i. Gate, prologue loads (q / k / v / b / a, dt_bias, A_log, then the state), q / k
        L2 norms, decay / beta, then 8 rows: decay, prediction, delta, replay record, state update, output; final state;
        row squares of the 32 output columns -> PEER words."""
        gdn_gate(layer, itid, core_offneed(blk), "cgpoll")
        tid = itid
        head, split = b.div(blk, ci(4)), b.mod(blk, ci(4))
        key_head = b.div(head, ci(3))
        lane, sub = b.mod(tid, ci(16)), b.div(tid, ci(16))
        wave, wl = rfl(b.div(tid, ci(64))), b.mod(tid, ci(64))
        vbase = b.add(b.mul(split, ci(32)), b.mul(sub, ci(2)))
        ws, ba, cp = x.ptr1(T["work"]), x.ptr1(T["BA"]), x.ptr1(T["core"])
        rp, fs, ist = x.ptr1(T["squares"]), x.ptr1(T["final"]), x.ptr1(T["init"])
        L0 = x.ptr3(arena_addr)
        zero = fconst(0.0)
        sbase = b.add(b.add(b.mul(head, ci(16384)), b.mul(vbase, ci(128))), lane)
        pre = []
        for pair in range(2):  # every prologue load before any arithmetic
            row = b.add(wave, ci(pair * 4))
            packed = b.mul(row, ci(16512))
            kq = gp(ws, "bfloat", b.add(b.add(packed, b.mul(key_head, ci(128))), wl))
            vp = gp(
                ws,
                "bfloat",
                b.add(
                    b.add(b.add(packed, b.mul(head, ci(128))), b.mul(split, ci(32))),
                    b.mod(wl, ci(32)),
                ),
                4096,
            )
            bp = gp(ba, "float", b.add(b.mul(row, ci(128)), head))
            pre.append(
                (
                    ldp(kq, "bfloat"),
                    ldp(gp(kq, "bfloat", ci(64)), "bfloat"),
                    ldp(gp(kq, "bfloat", ci(2048)), "bfloat"),
                    ldp(gp(kq, "bfloat", ci(2112)), "bfloat"),
                    ldp(vp, "bfloat"),
                    ldp(bp, "float"),
                    ldp(gp(bp, "float", ci(48)), "float"),
                )
            )
        dt = ldp(gp(x.ptr1(T["dt_bias"]), "float", head), "float")
        alog = ldp(gp(x.ptr1(T["A_log"]), "float", head), "float")
        sp0 = gp(ist, "float", sbase)
        s0, s1 = [None] * 8, [None] * 8
        for i in range(8):  # recurrent state after the prologue loads
            s0[i] = ldp(sp0 if i == 0 else gp(sp0, "float", ci(i * 16)), "float")
            s1[i] = ldp(gp(sp0, "float", ci(128 + i * 16)), "float")
        eps = fconst(1e-06)
        r128 = None
        for pair in range(2):
            row = b.add(wave, ci(pair * 4))
            q0, q1, k0, k1 = (bf2f(pre[pair][j]) for j in range(4))
            qs = x.fma(q0, q0, b.fmul(q1, q1))
            ks = x.fma(k0, k0, b.fmul(k1, k1))
            for s_ in (32, 16):
                qs = b.fadd(qs, shfl_down(qs, s_))
                ks = b.fadd(ks, shfl_down(ks, s_))
            qs = sum16(qs)
            ks = sum16(ks)
            rq = vasm("v_rsq_f32", b.fadd(shfl_idx(qs, 0), eps))
            if (
                r128 is None
            ):  # core_native_rsqrt(128.0): one value (FlyDSL's IR has it once, CSE'd)
                r128 = vasm("v_rsq_f32", fconst(128.0))
            qi = b.fmul(rq, r128)
            ki = vasm("v_rsq_f32", b.fadd(shfl_idx(ks, 0), eps))
            ri = b.add(b.mul(row, ci(128)), wl)
            qp = gp(L0, "float", ri, LQ, lds=True)
            stp(qp, "float", b.fmul(q0, qi), lds=True)
            stp(gp(qp, "float", ci(64), lds=True), "float", b.fmul(q1, qi), lds=True)
            kp = gp(L0, "float", ri, LK, lds=True)
            stp(kp, "float", b.fmul(k0, ki), lds=True)
            stp(gp(kp, "float", ci(64), lds=True), "float", b.fmul(k1, ki), lds=True)
            with b.scf_if(b.cmp_lt(wl, ci(32))):
                stp(
                    gp(L0, "bfloat", b.add(b.mul(row, ci(32)), wl), LV, lds=True),
                    "bfloat",
                    pre[pair][4],
                    lds=True,
                )
            bv = bf2f(b.trunc_f32_to_bf16(pre[pair][5]))
            av = b.fadd(bf2f(b.trunc_f32_to_bf16(pre[pair][6])), dt)
            with b.scf_if(b.cmp_eq(wl, c0)):
                sp = if_yield(
                    b.fcmp("ole", av, fconst(20.0)),
                    [av],
                    lambda cur, av=av: [
                        b.fmul(
                            vasm(
                                "v_log_f32",
                                b.fadd(
                                    vasm("v_exp_f32", b.fmul(av, fconst(LOG2E))),
                                    fconst(1.0),
                                ),
                            ),
                            fconst(LN2),
                        )
                    ],
                )[0]
                stp(
                    gp(L0, "float", row, LDEC, lds=True),
                    "float",
                    x.exp(b.fmul(b.fneg(x.exp(alog)), sp)),
                    lds=True,
                )
                den = b.fadd(
                    vasm("v_exp_f32", b.fmul(b.fneg(bv), fconst(LOG2E))), fconst(1.0)
                )
                stp(
                    gp(L0, "float", row, LBETA, lds=True),
                    "float",
                    vasm("v_rcp_f32", den),
                    lds=True,
                )
        barrier()
        kb = gp(
            L0, "float", lane, LK, lds=True
        )  # core_key + lane (+ row * 128 + 16 i below)
        qb = gp(L0, "float", lane, lds=True)  # core_query + lane
        vb = gp(
            L0, "bfloat", b.mul(sub, ci(2)), LV, lds=True
        )  # core_value + 2 subgroup
        lcb = gp(
            L0, "float", b.mul(sub, ci(2)), LCORE, lds=True
        )  # core_local_core + 2 subgroup
        r0 = r1 = zero
        for row in range(8):
            keys = [None] * 8
            p0 = p1 = zero
            dd = ldp(gp(L0, "float", ci(LDEC + row), lds=True), "float", lds=True)
            for i in range(8):
                keys[i] = ldp(
                    (
                        gp(kb, "float", ci(row * 128 + i * 16), lds=True)
                        if row or i
                        else kb
                    ),
                    "float",
                    lds=True,
                )
                s0[i] = b.fmul(s0[i], dd)
                s1[i] = b.fmul(s1[i], dd)
                p0 = x.fma(s0[i], keys[i], p0)
                p1 = x.fma(s1[i], keys[i], p1)
            p0 = sum16(p0)
            p1 = sum16(p1)
            if row > 0:
                r0 = sum16(r0)
                r1 = sum16(r1)
                with b.scf_if(b.cmp_eq(lane, c0)):
                    cb = gp(
                        cp, "float", b.add(b.mul(head, ci(128)), vbase)
                    )  # FlyDSL: the gep in the store's block
                    stp(
                        gp(cb, "float", ci((row - 1) * 6144)) if row > 1 else cb,
                        "float",
                        r0,
                    )
                    stp(gp(cb, "float", ci((row - 1) * 6144 + 1)), "float", r1)
                    stp(
                        (
                            gp(lcb, "float", ci((row - 1) * 32), lds=True)
                            if row > 1
                            else lcb
                        ),
                        "float",
                        r0,
                        lds=True,
                    )
                    stp(
                        gp(lcb, "float", ci((row - 1) * 32 + 1), lds=True),
                        "float",
                        r1,
                        lds=True,
                    )

            def delta(cur, row=row, p0=p0, p1=p1):
                bt = ldp(gp(L0, "float", ci(LBETA + row), lds=True), "float", lds=True)
                v0 = gp(vb, "bfloat", ci(row * 32), lds=True) if row else vb
                d0_ = b.fmul(bt, b.fsub(bf2f(ldp(v0, "bfloat", lds=True)), p0))
                d1_ = b.fmul(
                    bt,
                    b.fsub(
                        bf2f(
                            ldp(
                                gp(vb, "bfloat", ci(row * 32 + 1), lds=True),
                                "bfloat",
                                lds=True,
                            )
                        ),
                        p1,
                    ),
                )
                return [d0_, d1_]

            d0, d1 = if_yield(b.cmp_eq(lane, c0), [zero, zero], delta)
            d0 = shfl_idx(d0, b.sub(wl, lane))
            d1 = shfl_idx(d1, b.sub(wl, lane))
            rec = b.mul(
                b.add(ci(row * 48), head), ci(260)
            )  # replay record of this row (squares slot)
            rb_ = gp(
                rp, "float", rec
            )  # core_rec (FlyDSL: in the row's block; the decay store uses it)
            with b.scf_if(b.cmp_eq(lane, c0)):
                dp = gp(rp, "float", b.add(rec, vbase))
                stp(dp, "float", d0)
                stp(gp(dp, "float", c1), "float", d1)
            with b.scf_if(b.land(b.cmp_eq(split, c0), b.cmp_eq(sub, c0))):
                kr = gp(rp, "float", b.add(rec, lane))
                for i in range(8):
                    stp(gp(kr, "float", ci(128 + i * 16)), "float", keys[i])
                with b.scf_if(b.cmp_eq(lane, c0)):
                    stp(gp(rb_, "float", ci(256)), "float", dd)
            r0 = r1 = zero
            for i in range(8):
                s0[i] = x.fma(d0, keys[i], s0[i])
                s1[i] = x.fma(d1, keys[i], s1[i])
                qv = ldp(
                    (
                        gp(qb, "float", ci(row * 128 + i * 16), lds=True)
                        if row or i
                        else qb
                    ),
                    "float",
                    lds=True,
                )
                r0 = x.fma(s0[i], qv, r0)
                r1 = x.fma(s1[i], qv, r1)
        r0 = sum16(r0)
        r1 = sum16(r1)
        with b.scf_if(b.cmp_eq(lane, c0)):
            cb = gp(cp, "float", b.add(b.mul(head, ci(128)), vbase))
            stp(gp(cb, "float", ci(7 * 6144)), "float", r0)
            stp(gp(cb, "float", ci(7 * 6144 + 1)), "float", r1)
            stp(gp(lcb, "float", ci(7 * 32), lds=True), "float", r0, lds=True)
            stp(gp(lcb, "float", ci(7 * 32 + 1), lds=True), "float", r1, lds=True)
        fp0 = gp(fs, "float", sbase)
        for i in range(8):
            stp(fp0 if i == 0 else gp(fp0, "float", ci(i * 16)), "float", s0[i])
            stp(gp(fp0, "float", ci(128 + i * 16)), "float", s1[i])
        barrier()
        srow, scol = b.div(tid, ci(32)), b.mod(tid, ci(32))
        cv = ldp(
            gp(L0, "float", b.add(b.mul(srow, ci(32)), scol), LCORE, lds=True),
            "float",
            lds=True,
        )
        sq = x.fma(cv, cv, shfl_down(b.fmul(cv, cv), 16, 32))
        for s_ in (8, 4, 2, 1):
            sq = b.fadd(sq, shfl_down(sq, s_, 32))
        with b.scf_if(b.cmp_eq(scol, c0)):
            asm(
                "global_store_dword $0, $1, off sc0 sc1\ns_waitcnt vmcnt(0)",
                "v,v,~{memory}",
                [
                    word(
                        native,
                        b.add(
                            ci(PEER_WORD),
                            b.add(
                                b.mul(b.add(b.mul(head, ci(4)), split), ci(16)), srow
                            ),
                        ),
                    ),
                    b.bitcast(sq, I32),
                ],
            )
            if (
                DUMP
            ):  # dump twin D3: the row square -> dump_squares[layer][head * 4 + split][row] (endpoint cell 9)
                dq = raw_table_address(64, 9)
                asm(
                    "global_store_dword $0, $1, off sc0 sc1",
                    "v,v,~{memory}",
                    [
                        b.add(
                            dq,
                            i64(
                                b.mul(
                                    b.add(
                                        b.mul(
                                            b.add(
                                                b.mul(layer, ci(192)),
                                                b.add(b.mul(head, ci(4)), split),
                                            ),
                                            ci(8),
                                        ),
                                        srow,
                                    ),
                                    ci(4),
                                )
                            ),
                        ),
                        b.bitcast(sq, I32),
                    ],
                )

    def core_tail_stage(T, layer, itid, blk):
        """source 1051-1096, blocks 192-255: gate on all 48 value-head and 8 BA counters, then copy z (raw columns
        10240-16383) and BA -> bf16 (raw and workspace columns 16384-16511) into the workspace (nothing reads it).
        """
        gdn_gate(layer, itid, tail_offneed, "ctpoll")
        rw, ws, ba = x.ptr1(T["raw"]), x.ptr1(T["work"]), x.ptr1(T["BA"])
        cb = b.sub(blk, ci(192))
        for p in range(4):
            idx = b.add(b.add(b.mul(cb, ci(256)), itid), ci(p * 16384))
            with b.scf_if(b.cmp_lt(idx, ci(8 * 6272))):
                trow, tcol = b.div(idx, ci(6272)), b.mod(idx, ci(6272))
                at = b.add(b.mul(trow, ci(16512)), tcol)
                with b.scf_if_else(b.cmp_lt(tcol, ci(6144))) as (t_, f_):
                    with t_:
                        stp(
                            gp(ws, "bfloat", at, 10240),
                            "bfloat",
                            ldp(gp(rw, "bfloat", at, 10240), "bfloat"),
                        )
                    with f_:
                        v = b.trunc_f32_to_bf16(
                            ldp(
                                gp(
                                    ba,
                                    "float",
                                    b.add(b.mul(trow, ci(128)), tcol),
                                    -6144,
                                ),
                                "float",
                            )
                        )
                        stp(gp(rw, "bfloat", at, 10240), "bfloat", v)
                        stp(gp(ws, "bfloat", at, 10240), "bfloat", v)

    def peer_handoff(epoch, itid, blk):
        """source 2638-2686, blocks 0-191: drain, publish this split's epoch (PEER + 8), lanes 0-63 poll the 4 splits'
        epochs (lane l -> split l % 4; asm loop = FlyDSL's compiled body: epoch load, >=, cndmask, min tree xor 1 / 2,
        readfirstlane, failure-word load, s_sleep 1), status slot 4; then lanes 8 p + r load split p's row-r square,
        lanes r < 8 sum in split order, rsq(total / 128 + 1e-6) -> LDS inverse[r]."""
        head, split = b.div(blk, ci(4)), b.mod(blk, ci(4))
        wait_vm(0)
        barrier()
        with b.scf_if(b.cmp_eq(itid, c0)):
            system_store_u32(
                word(
                    native,
                    b.add(
                        b.mul(b.add(b.mul(head, ci(4)), split), ci(16)),
                        ci(PEER_WORD + 8),
                    ),
                ),
                epoch,
            )
            wait_vm(0)
        with b.scf_if(b.cmp_lt(itid, ci(64))):
            f0 = rfl(coherent_u32(word(native, 289)))
            t = (
                "s_mov_b32 $0, 0\n1:\n global_load_dword $1, $8, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
                " v_cmp_ge_i32 vcc, $1, $9\n s_nop 1\n v_cndmask_b32_e64 $2, 0, 1, vcc\n"
                + min_tree("$2", ("$3", "$4"), (1, 2))
                + " s_nop 0\n v_readfirstlane_b32 $5, $2\n global_load_dword $6, $10, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
                " s_sleep 1\n s_nop 0\n v_readfirstlane_b32 $7, $6\n s_add_u32 $0, $0, 1\n s_cmp_lg_u32 $7, 0\n"
                " s_cbranch_scc1 2f\n s_cmp_lg_u32 $5, 0\n s_cbranch_scc1 2f\n s_cmp_lt_u32 $0, 0x100000\n"
                " s_cbranch_scc1 1b\n2:"
            )
            rr = b.inline_asm_multi(
                t,
                "=&s,=&v,=&v,=&v,=&v,=&s,=&v,=&s,v,v,v,~{vcc},~{scc}",
                [
                    word(
                        native,
                        b.add(
                            b.mul(
                                b.add(b.mul(head, ci(4)), b.mod(itid, ci(4))), ci(16)
                            ),
                            ci(PEER_WORD + 8),
                        ),
                    ),
                    epoch,
                    word(native, 289),
                ],
                result_types=[I32] * 8,
                result_name_hint="pepoll",
            )
            with b.scf_if(b.cmp_eq(itid, c0)):
                bad = b.lor(
                    b.cmp_eq(rr[5], c0), b.lor(b.cmp_ne(f0, c0), b.cmp_ne(rr[7], c0))
                )
                with b.scf_if(bad):
                    fail_native()
                st_store(4, b.select(bad, ci(2), c1))
        barrier()
        with b.scf_if(b.cmp_eq(st_load(4), c1)):
            g = if_yield(
                b.cmp_lt(itid, ci(32)),
                [c0],
                lambda cur: [
                    coherent_u32(
                        word(
                            native,
                            b.add(
                                b.add(
                                    b.mul(
                                        b.add(b.mul(head, ci(4)), b.div(itid, ci(8))),
                                        ci(16),
                                    ),
                                    b.mod(itid, ci(8)),
                                ),
                                ci(PEER_WORD),
                            ),
                        )
                    )
                ],
            )[0]
            wl = b.mod(itid, ci(64))
            parts = [g] + [shfl_idx(g, b.add(wl, ci(ps * 8))) for ps in range(1, 4)]
            with b.scf_if(b.cmp_lt(itid, ci(8))):
                total = fconst(0.0)
                for ps in range(4):
                    total = b.fadd(total, b.bitcast(parts[ps], F32))
                stp(
                    gp(x.ptr3(arena_addr), "float", itid, LDEC, lds=True),
                    "float",
                    vasm("v_rsq_f32", x.fma(total, fconst(1.0 / 128.0), fconst(1e-06))),
                    lds=True,
                )
        barrier()
        return st_load(4)

    def finalize_stage(T, itid, blk):
        """source 1098-1148, blocks 0-191: row tid // 32, column 128 h + 32 split + tid % 32: core * inverse * norm * z *
        sigmoid(z) -> bf16 (recurrent + LDS), then per 32 columns the MXFP8 quant (fold's rule) -> mixer_A / _AS
        (``sc0 sc1``); block 0 writes 127 into the padding scale bytes."""
        tid = itid
        head, split = b.div(blk, ci(4)), b.mod(blk, ci(4))
        row, ln = b.div(tid, ci(32)), b.mod(tid, ci(32))
        L0 = x.ptr3(arena_addr)
        inv = ldp(gp(L0, "float", row, LDEC, lds=True), "float", lds=True)
        colm = b.add(b.add(b.mul(head, ci(128)), b.mul(split, ci(32))), ln)
        z = bf2f(
            ldp(
                gp(
                    x.ptr1(T["raw"]),
                    "bfloat",
                    b.add(b.mul(row, ci(16512)), colm),
                    10240,
                ),
                "bfloat",
            )
        )
        sig = vasm(
            "v_rcp_f32",
            b.fadd(vasm("v_exp_f32", b.fmul(b.fneg(z), fconst(LOG2E))), fconst(1.0)),
        )
        co = b.add(b.mul(row, ci(6144)), colm)
        v = b.fmul(ldp(gp(x.ptr1(T["core"]), "float", co), "float"), inv)
        v = b.fmul(
            v,
            bf2f(
                ldp(
                    gp(x.ptr1(T["norm"]), "bfloat", b.add(b.mul(split, ci(32)), ln)),
                    "bfloat",
                )
            ),
        )
        v = b.fmul(v, z)
        v = b.fmul(v, sig)
        rb = b.trunc_f32_to_bf16(v)
        stp(gp(x.ptr1(T["recurrent"]), "bfloat", co), "bfloat", rb)
        x.store(lbp(tid), rb, "bfloat", 2, lds=True)
        barrier()
        asm("s_setreg_imm32_b32 0xdc1, 1", "~{memory}")
        with b.scf_if(b.cmp_lt(ln, ci(4))):
            nv = x.load(
                lbp(b.add(b.mul(row, ci(32)), b.mul(ln, ci(8)))),
                "<8 x bfloat>",
                VectorType(BF16, 8),
                2,
                lds=True,
            )
            q0, q1, ex = fp8_quant(nv, branchy_ex=True)
            asm(
                "global_store_dwordx2 $0, $1, off sc0 sc1",
                "v,v,~{memory}",
                [
                    b.add(
                        T["A"],
                        i64(
                            b.add(
                                b.add(
                                    b.add(b.mul(row, ci(6144)), b.mul(head, ci(128))),
                                    b.mul(split, ci(32)),
                                ),
                                b.mul(ln, ci(8)),
                            )
                        ),
                    ),
                    b.vec_pack([q0, q1], I32),
                ],
            )
            with b.scf_if(b.cmp_eq(ln, c0)):
                asm(
                    "global_store_byte $0, $1, off sc0 sc1",
                    "v,v,~{memory}",
                    [
                        b.add(
                            T["AS"],
                            i64(
                                b.add(
                                    b.add(b.mul(head, ci(64)), b.mul(split, ci(16))),
                                    row,
                                )
                            ),
                        ),
                        ex,
                    ],
                )
        with b.scf_if(b.cmp_eq(blk, c0)):
            for step in range(13):
                index = b.add(tid, ci(step * 256))
                with b.scf_if(b.cmp_lt(index, ci(3264))):
                    with b.scf_if(
                        b.lor(
                            b.cmp_ge(index, ci(3072)),
                            b.cmp_ge(b.mod(index, ci(16)), ci(8)),
                        )
                    ):
                        asm(
                            "global_store_byte $0, $1, off sc0 sc1",
                            "v,v,~{memory}",
                            [b.add(T["AS"], i64(index)), padv()],
                        )

    def gdn_layer(TG, layer, itid, blk, al_g, epoch_base):
        """source 2909-2924 (GDN layer, after QKV): core (0-191) | core tail + 240-247 Up L2 touch (192-255), peer
        hand-off (0-191, unconditional), finalize + its mixer sub-counter (0-191)."""
        with b.scf_if(b.land(b.cmp_eq(al_g, c1), b.cmp_lt(blk, ci(192)))):
            core_stage(TG, layer, itid, blk)
        with b.scf_if(b.land(b.cmp_eq(al_g, c1), b.cmp_ge(blk, ci(192)))):
            core_tail_stage(TG, layer, itid, blk)
            with b.scf_if(b.land(b.cmp_ge(blk, ci(240)), b.cmp_lt(blk, ci(248)))):
                up_l2_touch(TG["tbl"], itid, blk, 0, 3, 5)
        with b.scf_if(b.cmp_lt(blk, ci(192))):
            peer_handoff(b.add(epoch_base, ci(4)), itid, blk)
        al_p = b.select(b.cmp_lt(blk, ci(192)), b.smax(al_g, st_load(4)), al_g)
        with b.scf_if(b.land(b.cmp_eq(al_p, c1), b.cmp_lt(blk, ci(192)))):
            sm = mode_save()
            finalize_stage(TG, itid, blk)
            mode_restore(sm)
            barrier()
            mixer_pub_gdn(itid, blk)

    def zero_mixer_reserve(mixer, blk, itid):
        """source 2794-2799: owner j clears mixer[j * 256 + tid :: 2048]. Deviation from the reference: the bf16 zeros
        are written through (``global_store_short ... sc0 sc1``).
        """
        worker = b.select(b.cmp_eq(blk, c0), c0, b.sub(blk, ci(136)))
        iy[0] += 1  # unique loop name
        with b.scf_for(
            b.add(b.mul(worker, ci(256)), itid),
            ci(40960),
            ci(2048),
            iv_name=f"zmr{iy[0]}",
        ) as idx:
            asm(
                "global_store_short $0, $1, off sc0 sc1",
                "v,v,~{memory}",
                [b.add(mixer, i64(b.mul(idx, ci(2)))), c0],
            )

    def dump_mixer_row(mixer, layer, row, itid):
        """dump twin D2: the owner's mixer row -> dump_mixer[layer][row] (endpoint cell 7), plain 16-B accesses."""
        dm = x.ptr1(raw_table_address(64, 7))
        mp = x.ptr1(mixer)
        for step in range(3):
            column = b.add(b.mul(itid, ci(8)), ci(step * 2048))
            with b.scf_if(b.cmp_lt(column, ci(5120))):
                v = x.load(
                    x.gep(mp, "bfloat", b.add(b.mul(row, ci(5120)), column)),
                    "<8 x bfloat>",
                    VectorType(BF16, 8),
                    16,
                )
                x.store(
                    x.gep(
                        dm,
                        "bfloat",
                        b.add(b.mul(b.add(b.mul(layer, ci(8)), row), ci(5120)), column),
                    ),
                    v,
                    "<8 x bfloat>",
                    16,
                )

    # ================================================================ kernel prologue and layer-63 tail
    # (source 98-137 native_arrive / native_wait / native_grid, 237-293 census / embedding / entry grid, 1690-1695
    # fold_stage, 2503-2612 head_stream, 3002-3025 the tail).
    # Scalar-predicate rule: every wave-uniform decision is an SGPR value (wave = readfirstlane(tid // 64),
    # statuses via st_load = readfirstlane); no lane-dependent region ends a loop body (head_stream's tile loop ends
    # with a barrier after its lane-dependent store region).
    def native_arrive(epoch):
        """source 123-133: every wave and every CTA participates. tid 0: agent-monotonic add on the XCD's arrival word
        (atomicrmw, returned); the 32nd arriver of the XCD: release fence (agent) + system add on the global word 128;
        the 8th global arrival writes the 8 per-XCD release words 144 + 16 x (``sc0 sc1``) and drains.
        """
        wait_vm(0)
        barrier()
        with b.scf_if(b.cmp_eq(tid, c0)):
            local_prior = iatomic_add_agent(
                x.gep(x.ptr1(native), "i32", b.mul(safe_xcc, ci(16))), c1
            )
            with b.scf_if(b.cmp_eq(b.mod(b.add(local_prior, c1), ci(32)), c0)):
                b.fence(scope="agent", ordering="release")
                global_prior = system_atomic_add(word(native, 128))
                with b.scf_if(b.cmp_eq(b.mod(b.add(global_prior, c1), ci(8)), c0)):
                    for rx in range(8):
                        system_store_u32(word(native, 144 + rx * 16), epoch)
                    wait_vm(0)

    def native_grid(epoch, slot, consumer_blocks=256):
        """source 135-137: arrive, then tid 0 polls this XCD's release word >= epoch (+ failure word); consumers
        (block < consumer_blocks) ``vmcnt(0)`` + ``buffer_inv sc1``; status slot -> every thread (scalar).
        """
        native_arrive(epoch)
        return native_wait(epoch, slot, b.cmp_lt(block, ci(consumer_blocks)))

    def prologue(alive):
        """source 273-293, after the census (``alive`` = its scalar result): rows 0-7 (block = row) copy their token's
        embedding row into ``hidden0`` (zeros for a token outside [0, 248320)) and zero ``residual0``; all blocks zero the
        initial mixer (40,960 bf16, plain stores); entry grid ``command_base + 1`` (status slot 11). Returns the scalar
        alive. Plain write-back stores are made visible by the grid (per-XCD ``vmcnt(0)`` + agent release fence by the
        XCD's last arriver before the global release; consumers ``buffer_inv sc1``)."""
        ids = x.ptr1(raw_table_address(64, 44))
        emb = x.ptr1(raw_table_address(64, 43))
        hidden0 = x.ptr1(raw_table_address(0, 0))
        residual0 = x.ptr1(raw_table_address(0, 1))
        mixer0 = x.ptr1(raw_table_address(0, 32))
        V8BF = VectorType(BF16, 8)
        zero8 = bf8(b.vec_pack([c0] * 4, I32))
        with b.scf_if(b.land(b.cmp_eq(alive, c1), b.cmp_lt(block, ci(8)))):
            token = x.load(x.gep(ids, "i64", block), "i64", I64, 8)
            for step in range(3):
                column = b.add(b.mul(tid, ci(8)), ci(step * 2048))
                with b.scf_if(b.cmp_lt(column, ci(5120))):
                    inr = b.land(
                        b.cmp_ge(token, b.const_i64(0)),
                        b.cmp_lt(token, b.const_i64(248320)),
                    )

                    def load_row(cur, column=column):
                        e = b.add(b.mul(token, b.const_i64(5120)), b.sext(column, I64))
                        p = inst(
                            b,
                            "getelementptr bfloat, ptr addrspace(1) {0}, i64 {1}",
                            [emb, e],
                            PTR1,
                            "gep",
                        )
                        return [x.load(p, "<8 x bfloat>", V8BF, 2)]

                    value = if_yield(inr, [zero8], load_row)[0]
                    idx = b.add(b.mul(block, ci(5120)), column)
                    x.store(x.gep(hidden0, "bfloat", idx), value, "<8 x bfloat>", 2)
                    x.store(x.gep(residual0, "bfloat", idx), zero8, "<8 x bfloat>", 2)
        index = b.add(b.mul(block, ci(256)), tid)
        with b.scf_if(b.land(b.cmp_eq(alive, c1), b.cmp_lt(index, ci(40960)))):
            x.store(
                x.gep(mixer0, "bfloat", index),
                b.trunc_f32_to_bf16(b.const_f32(0.0)),
                "bfloat",
                2,
            )
        with b.scf_if(b.cmp_eq(alive, c1)):
            native_grid(b.add(command_base, c1), 11)
        return b.select(b.cmp_eq(alive, c1), b.smax(alive, st_load(11)), alive)

    def fold_stage(partial, out, itid, blk):
        """source 1690-1695 (blocks 0-159): element blk * 256 + tid of the 3 Down planes, (p0 + p1) + p2 in fp32 ->
        bf16 (RNE) -> final_hidden (plain stores)."""
        i = b.add(b.mul(blk, ci(256)), itid)
        q = x.gep(x.ptr1(partial), "float", i)
        p0 = x.load(q, "float", F32, 4)
        p1 = x.load(x.gep(q, "float", ci(40960)), "float", F32, 4)
        p2 = x.load(x.gep(q, "float", ci(81920)), "float", F32, 4)
        x.store(
            x.gep(x.ptr1(out), "bfloat", i),
            b.trunc_f32_to_bf16(b.fadd(b.fadd(p0, p1), p2)),
            "bfloat",
            2,
        )

    def head_stream(T, itid, blk):
        """source 2503-2612 (all blocks): LM head tiles blk + 256 it (64 vocabulary columns each, 15 or 16 tiles). The 10
        activation k-blocks (k-block 4 k + wave) + scale dwords are loaded once and pinned; the FP4 weights stream
        through a 5-slot per-wave LDS ring (LDS-DMA ``nt``, 4 atoms x 1 KB per slot) with the scale dwords in VGPRs
        carried across tiles; per k-step ``vmcnt(32)``, LDS reads, refill of the slot (k < 5: k-step k + 5 of this tile;
        else k - 5 of the next tile, OOB offset -16 when there is none), 4 scaled MFMAs (FP8 x FP4, cbsz 0, blgp 4); the
        4 waves' accumulators sum in LDS (arena + 81920) in wave order, bf16 RNE, demangle, wave 0 lanes < 32 store
        head_bf16 and its fp32 copy (logits). T: i64 addresses A, AS, W, WS, raw, logits.
        """
        tid = itid
        lane, wave = b.mod(tid, ci(64)), rfl(
            b.div(tid, ci(64))
        )  # scalar wave (wave-uniform decisions)
        lane4 = b.land(lane, ci(-4))
        ad = descriptor(T["A"], 40960)
        asd = descriptor(T["AS"], 2752)
        bd = descriptor(T["W"], 635699200)
        bsd = descriptor(T["WS"], 39731200)
        ring = b.add(rfl(arena_addr), b.mul(wave, ci(20480)))
        rf = x.gep(
            x.ptr3(arena_addr), "i8", ci(81920), lds=True
        )  # reduction (fp32) / demangle (bf16): arena + 81920
        ntiles = b.div(b.sub(ci(3880 + 255), blk), ci(256))
        lane16 = b.mul(lane, ci(16))
        alo, ahi, asc = [], [], []
        for k in range(10):
            kb = b.add(ci(k * 4), wave)
            off = b.add(
                b.add(b.mul(b.mod(lane, ci(16)), ci(5120)), b.mul(kb, ci(128))),
                b.mul(b.div(lane, ci(16)), ci(16)),
            )
            alo.append(x.buf_ld16(ad, off, 0))
            ahi.append(x.buf_ld16(ad, b.add(off, ci(64)), 0))
            asc.append(x.buf_ld4(asd, b.add(b.mul(kb, ci(64)), lane4), 0))

        def woff(t, atom, kb):
            return b.add(
                b.mul(
                    b.add(b.mul(b.add(b.mul(t, ci(4)), ci(atom)), ci(40)), kb), ci(1024)
                ),
                lane16,
            )

        def soff(t, atom, kb):
            return b.add(
                b.mul(
                    b.add(b.add(b.mul(kb, ci(15520)), b.mul(t, ci(4))), ci(atom)),
                    ci(64),
                ),
                lane4,
            )

        bs = [None] * 20
        for j in range(5):
            kb = b.add(ci(j * 4), wave)
            for atom in range(4):
                dma(b.add(ring, ci(j * 4096 + atom * 1024)), woff(blk, atom, kb), bd)
                bs[j * 4 + atom] = x.buf_ld4(bsd, soff(blk, atom, kb), 0)
        wait_vm(40)
        alo = [pin(v) for v in alo]
        ahi = [pin(v) for v in ahi]
        asc = [pin(v) for v in asc]
        sh = b.mul(b.land(lane, ci(3)), ci(8))
        lp_ = b.scf_for_iter(
            c0,
            ntiles,
            c1,
            [(f"hbs{i}", v) for i, v in enumerate(bs)],
            iv_name="hit",
            elide_trailing_barrier=False,
        )
        with lp_ as (it, cur):
            bs = list(cur)
            tile = b.add(blk, b.mul(it, ci(256)))
            nxt_on = zx(b.cmp_lt(b.add(it, c1), ntiles))
            acc = [b.zero_vec_f32(4) for _ in range(4)]
            for k in range(10):
                slot = k % 5
                wait_vm(32)
                sa = b.land(b.lshr(asc[k], sh), ci(255))
                av = b.vec_pack(
                    [b.vec_extract(alo[k], e) for e in range(4)]
                    + [b.vec_extract(ahi[k], e) for e in range(4)],
                    I32,
                )
                bvals = [
                    lds_ld16(
                        b.add(
                            b.mul(wave, ci(20480)),
                            b.add(ci(slot * 4096 + atom * 1024), lane16),
                        )
                    )
                    for atom in range(4)
                ]
                sbs = [pin(bs[slot * 4 + atom]) for atom in range(4)]
                b.s_waitcnt(
                    lgkmcnt=0
                )  # FlyDSL rocdl.s_waitcnt(lgkmcnt=0) = llvm.amdgcn.s.waitcnt(i32 49279)
                if k < 5:
                    rt, rkb = tile, b.add(ci((k + 5) * 4), wave)
                else:
                    rt, rkb = b.add(tile, ci(256)), b.add(ci((k - 5) * 4), wave)
                for atom in range(4):
                    wo, so = woff(rt, atom, rkb), soff(rt, atom, rkb)
                    if k >= 5:  # r_on * offset + (1 - r_on) * -16
                        neg = b.mul(b.sub(c1, nxt_on), ci(-16))
                        wo, so = b.add(b.mul(nxt_on, wo), neg), b.add(
                            b.mul(nxt_on, so), neg
                        )
                    dma(b.add(ring, ci(slot * 4096 + atom * 1024)), wo, bd)
                    bs[slot * 4 + atom] = x.buf_ld4(bsd, so, 0)
                for atom in range(4):
                    sb = b.land(b.lshr(sbs[atom], sh), ci(255))
                    bf = b.vec_pack(
                        [b.vec_extract(bvals[atom], e) for e in range(4)] + [c0] * 4,
                        I32,
                    )
                    acc[atom] = x.mfma_scale(av, bf, acc[atom], sa, sb, 0, 4)
            for atom in range(4):
                for e in range(4):
                    stf(
                        x.gep(
                            rf,
                            "float",
                            b.add(
                                b.add(ci(atom * 1024), b.mul(wave, ci(256))),
                                b.add(b.mul(lane, ci(4)), ci(e)),
                            ),
                            lds=True,
                        ),
                        b.vec_extract(acc[atom], e),
                    )
            barrier()
            with b.scf_if(b.cmp_eq(wave, c0)):
                for atom in range(4):
                    folded = []
                    for e in range(4):
                        p = x.gep(
                            rf,
                            "float",
                            b.add(ci(atom * 1024), b.add(b.mul(lane, ci(4)), ci(e))),
                            lds=True,
                        )
                        val = ldf(p)
                        for w in range(1, 4):
                            val = b.fadd(
                                val, ldf(x.gep(p, "float", ci(w * 256), lds=True))
                            )
                        folded.append(val)
                    rounded = b.vec_cast_f32_to(b.vec_pack(folded, F32), BF16)
                    for e in range(4):
                        x.store(
                            x.gep(
                                rf,
                                "bfloat",
                                b.add(
                                    b.add(
                                        ci(atom * 320),
                                        b.mul(
                                            b.add(
                                                b.mul(b.div(lane, ci(16)), ci(4)), ci(e)
                                            ),
                                            ci(20),
                                        ),
                                    ),
                                    b.mod(lane, ci(16)),
                                ),
                                lds=True,
                            ),
                            b.vec_extract(rounded, e),
                            "bfloat",
                            2,
                            lds=True,
                        )
            barrier()
            with b.scf_if(b.cmp_eq(wave, c0)):
                row, col = b.div(lane, ci(4)), b.mul(b.mod(lane, ci(4)), ci(4))
                with b.scf_if(b.cmp_lt(row, ci(8))):
                    for atom in range(4):
                        vals = x.load(
                            x.gep(
                                rf,
                                "bfloat",
                                b.add(b.add(ci(atom * 320), b.mul(row, ci(20))), col),
                                lds=True,
                            ),
                            "<4 x bfloat>",
                            VectorType(BF16, 4),
                            2,
                            lds=True,
                        )
                        idx = b.add(
                            b.add(b.mul(row, ci(248320)), b.mul(tile, ci(64))),
                            b.add(ci(atom * 16), col),
                        )
                        x.store(
                            x.gep(x.ptr1(T["raw"]), "bfloat", idx),
                            vals,
                            "<4 x bfloat>",
                            2,
                        )
                        f4 = inst(
                            b,
                            "fpext <4 x bfloat> {0} to <4 x float>",
                            [vals],
                            V4F,
                            "fpx",
                        )
                        x.store(
                            x.gep(x.ptr1(T["logits"]), "float", idx),
                            f4,
                            "<4 x float>",
                            4,
                        )
            barrier()
            b.scf_yield(*bs)

    def tail(layer, itid, blk, alive, epoch_base, norm_phase, partial=None):
        """source 3002-3025, layer 63 after down_stage (call it right after the Down alive update, before the
        layer-end control words): native_grid(epoch_base + norm_phase + 2, consumers block 0) -> fold_stage (blocks
        0-159: layer 63's Down planes -> final_hidden) -> native_grid(command_base + 754, slot 10) -> final rms_stage
        (blocks 0-7: final_hidden + final_skip with final_norm -> normalized / final_residual / A / AS; MODE saved /
        restored) -> native_grid(command_base + 755, slot 11) -> head_stream. The three grids run on every block and
        alive state (as FlyDSL). Runs inside ``scf_if(layer == 63)``; returns the merged scalar alive (= FlyDSL's
        max(alive, three grid statuses)) for every layer. ``partial``: layer 63's table slot 39 (default: loaded here).
        """
        s1 = b.add(norm_phase, ci(2))
        with b.scf_if(b.cmp_eq(layer, ci(63))):
            part = partial if partial is not None else raw_table_address(layer, 39)
            al = b.smax(alive, native_grid(b.add(epoch_base, s1), s1, 1))
            head_hidden = raw_table_address(64, 0)
            with b.scf_if(b.land(b.cmp_eq(al, c1), b.cmp_lt(blk, ci(160)))):
                fold_stage(part, head_hidden, itid, blk)
            al = b.smax(al, native_grid(b.add(command_base, ci(754)), 10))
            TF = {
                "hidden": head_hidden,
                "residual": raw_table_address(64, 1),
                "gamma": raw_table_address(64, 2),
                "normalized": raw_table_address(64, 18),
                "residual_out": raw_table_address(64, 19),
                "act": raw_table_address(64, 20),
                "act_s": raw_table_address(64, 21),
            }
            with b.scf_if(b.land(b.cmp_eq(al, c1), b.cmp_lt(blk, ci(8)))):
                sm = mode_save()
                rms_stage(TF, blk, itid)
                mode_restore(sm)
            al = b.smax(al, native_grid(b.add(command_base, ci(755)), 11))
            TH = {
                "A": TF["act"],
                "AS": TF["act_s"],
                "W": raw_table_address(64, 4),
                "WS": raw_table_address(64, 5),
                "raw": raw_table_address(64, 22),
                "logits": raw_table_address(64, 32),
            }
            with b.scf_if(b.cmp_eq(al, c1)):
                head_stream(TH, itid, blk)
        merged = b.smax(alive, b.smax(st_load(s1), b.smax(st_load(10), st_load(11))))
        return b.select(b.cmp_eq(layer, ci(63)), merged, alive)

    # ================================================================ the attention layer core as an exact
    # transliteration of FlyDSL (source 1804-1874 attention_qk_stage, 2095-2327 attention_partial_stage, 98-137
    # native_grid, 2329-2410 attention_reduce_stage, 2867-2884 the reduce's mixer sub-counter). Scalar-predicate rule:
    # the wave index and the LDS status words are readfirstlane'd; the counter publications are lane-0 asm under a
    # scalar wave-0 branch.
    NEG_MAX = -3.402823466e38
    SCORE_SCALE = 0.0625 * 1.442695041
    _ty["i32"] = ("i32", I32, 4)

    def publish0(
        itid, a
    ):  # FlyDSL ``if iteration_tid == 0: system_atomic_add_nw(a)`` in the scalar form
        with b.scf_if(b.cmp_eq(rfl(b.div(itid, ci(64))), c0)):
            lane0_atomic_add_nw(a)

    def counter_wait(addr, target, limit):
        """FlyDSL ``while (have < target) & (polls < limit): have = coherent_u32(p); s_sleep 1`` (tid 0), then
        ``if have < target: fail_native()`` and ``s_waitcnt vmcnt(0); buffer_inv sc1``.
        """
        t = (
            "s_mov_b32 $0, 0\n1:\n global_load_dword $1, $2, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
            " s_sleep 1\n s_add_u32 $0, $0, 1\n v_cmp_ge_i32 vcc, $1, $3\n s_cbranch_vccnz 2f\n"
            " s_cmp_lt_u32 $0, $4\n s_cbranch_scc1 1b\n2:"
        )
        rr = b.inline_asm_multi(
            t,
            "=&s,=&v,v,v,s,~{vcc},~{scc}",
            [addr, target, ci(limit)],
            result_types=[I32, I32],
            result_name_hint="apoll",
        )
        with b.scf_if(b.cmp_lt(rr[1], target)):
            fail_native()
        asm("s_waitcnt vmcnt(0)\nbuffer_inv sc1", "~{memory}")

    def bf16_word(v):  # fx.Vector.from_elements([v, BFloat16(0.0)]).bitcast(Int32)[0]
        pv = b.vec_pack([v, b.trunc_f32_to_bf16(fconst(0.0))], BF16)
        return b.vec_extract(b.vec_bitcast(pv, VectorType(I32, 1)), 0)

    def store_short_wt(addr64, v):  # global_store_short ... sc0 sc1 of a bf16
        asm(
            "global_store_short $0, $1, off sc0 sc1",
            "v,v,~{memory}",
            [addr64, bf16_word(v)],
        )

    def attention_qk_stage(T, layer, itid, blk):
        """source 1804-1874, blocks 0-223 (caller guards alive and blk < 224): row b // 28, head b % 28 (24 query heads,
        4 KV heads); thread d = one of the head's 256 dims. RoPE angle and gamma before the head's QGKV wait (tid 0 polls
        3200 + 16 head >= 8 (L // 4 + 1)); gate copy; RMS norm over 256 (shuffle_xor tree + 4 LDS partials, v_rsq);
        rotary on dims 0-63 (pairs d, d + 32); query / new K into the cache (write-through), new V copied; drain; the
        kv-head counter 3648 + 16 j."""
        row, head = b.div(blk, ci(28)), b.mod(blk, ci(28))
        d = itid
        isq = b.cmp_lt(head, ci(24))
        pos = rfl(
            ldp(gp(x.ptr1(T["position"]), "i32", row), "i32")
        )  # wave-uniform (row = block // 28): scalar
        gk = bf2f(ldp(gp(x.ptr1(T["kgamma"]), "bfloat", d), "bfloat"))
        gamma = if_yield(
            isq,
            [gk],
            lambda cur: [bf2f(ldp(gp(x.ptr1(T["qgamma"]), "bfloat", d), "bfloat"))],
        )[0]
        page = rfl(ldp(gp(x.ptr1(T["ctx_blocks"]), "i32", row), "i32"))
        freq = x.ocml(
            "pow",
            fconst(10000000.0),
            b.fmul(b.sitofp_f32(b.mod(d, ci(32))), fconst(-2 / 64)),
        )
        theta = b.fmul(b.sitofp_f32(pos), freq)
        cosine, sine = x.ocml("cos", theta), x.ocml("sin", theta)
        with b.scf_if(b.cmp_eq(itid, c0)):  # this head's 8 QGKV tiles
            counter_wait(
                word(native, b.add(ci(CHUNK_WORD + 3200), b.mul(head, ci(16)))),
                b.mul(b.add(b.div(layer, ci(4)), c1), ci(8)),
                4194304,
            )
        barrier()
        qg = x.ptr1(T["raw"])
        rbase = b.mul(row, ci(14336))
        kvh = b.sub(head, ci(24))
        src_kv = b.add(b.add(rbase, ci(12288)), b.mul(kvh, ci(256)))
        src_q = b.add(rbase, b.mul(head, ci(512)))
        hoff = b.add(b.mul(row, ci(6144)), b.mul(head, ci(256)))
        with b.scf_if(isq):  # gate copy (query heads)
            gv = ldp(gp(qg, "bfloat", b.add(src_q, d), 256), "bfloat")
            store_short_wt(b.add(T["gate"], i64(b.mul(b.add(hoff, d), ci(2)))), gv)
        source = b.select(isq, src_q, src_kv)
        value = bf2f(ldp(gp(qg, "bfloat", b.add(source, d)), "bfloat"))
        sq = b.fmul(value, value)
        for s_ in (1, 2, 4, 8, 16, 32):
            sq = b.fadd(sq, shuffle_xor_f(sq, s_))
        L0 = x.ptr3(arena_addr)
        with b.scf_if(b.cmp_eq(b.mod(d, ci(64)), c0)):
            stp(gp(L0, "float", b.div(d, ci(64)), lds=True), "float", sq, lds=True)
        barrier()
        red = [
            ldp(gp(L0, "float", ci(i_), lds=True) if i_ else L0, "float", lds=True)
            for i_ in range(4)
        ]
        total = b.fadd(b.fadd(red[0], red[1]), b.fadd(red[2], red[3]))
        inv = x.rsq(b.fadd(b.fmul(total, fconst(1 / 256)), fconst(1e-06)))
        norm = b.fmul(b.fmul(value, inv), gamma)
        partner = shuffle_xor_f(norm, 32)
        kbase = b.mul(b.add(b.mul(page, ci(4)), kvh), ci(16384))
        pslot = b.mul(b.mod(pos, ci(32)), ci(256))
        with b.scf_if(b.lnot(isq)):  # new V into the cache (copy)
            vv = ldp(
                gp(qg, "bfloat", b.add(b.add(rbase, b.mul(kvh, ci(256))), d), 13312),
                "bfloat",
            )
            store_short_wt(
                b.add(
                    T["context"],
                    i64(b.mul(b.add(b.add(b.add(kbase, ci(8192)), pslot), d), ci(2))),
                ),
                vv,
            )
        offset = b.select(isq, hoff, b.add(kbase, pslot))
        lo32 = b.cmp_lt(d, ci(32))
        plain = b.trunc_f32_to_bf16(norm)
        out0 = b.select(
            lo32,
            b.trunc_f32_to_bf16(b.fsub(b.fmul(norm, cosine), b.fmul(partner, sine))),
            plain,
        )
        out1 = b.select(
            lo32,
            b.trunc_f32_to_bf16(b.fadd(b.fmul(partner, cosine), b.fmul(norm, sine))),
            plain,
        )
        dst = b.select(isq, T["query"], T["context"])
        with b.scf_if(b.lor(lo32, b.cmp_ge(d, ci(64)))):
            a0 = b.add(dst, i64(b.mul(b.add(offset, d), ci(2))))
            store_short_wt(a0, out0)
            with b.scf_if(lo32):
                store_short_wt(b.add(a0, b.const_i64(64)), out1)
        wait_vm(
            0
        )  # outputs drained -> kv-head counter (query heads 6 j .. 6 j + 5 and kv head j)
        barrier()
        qp_j = b.select(isq, b.div(head, ci(6)), kvh)
        publish0(itid, word(native, b.add(ci(CHUNK_WORD + 3648), b.mul(qp_j, ci(16)))))

    def kv_tile_dma(kvd, lane, kvhead, pages, pages_hi, start, stop, dm_tile, lds_byte):
        """source 2131-2148: one 16-token K / V tile -> this wave's LDS buffer (16 x 1 KB ``buffer_load_dwordx4 ...
        lds``): K keys 2 j + lane // 32 with the 16-B chunk xor-swizzled by key % 16, V in the ds_read_tr16 layout. The
        tile's page is a readlane of the task's page lanes (pages = token blocks start // 32 + 0..63, pages_hi + 64..).
        """
        dm_pi = b.sub(b.div(dm_tile, ci(32)), b.div(start, ci(32)))
        dm_lo = x.readlane(pages, b.smin(dm_pi, ci(63)))
        dm_hi = x.readlane(pages_hi, b.smax(b.smin(b.sub(dm_pi, ci(64)), ci(63)), c0))
        dm_page = b.add(dm_lo, b.mul(b.sub(dm_hi, dm_lo), zx(b.cmp_ge(dm_pi, ci(64)))))
        dm_row = b.mul(b.add(b.mul(dm_page, ci(4)), kvhead), ci(16384))
        dm_lds = rfl(lds_byte)
        asm("s_waitcnt lgkmcnt(0)", "~{memory}")
        for j in range(16):
            if j < 8:
                key = b.add(ci(2 * j), b.div(lane, ci(32)))
                off = b.add(
                    b.add(
                        dm_row,
                        b.mul(
                            b.mod(b.smin(b.add(dm_tile, key), stop), ci(32)), ci(256)
                        ),
                    ),
                    b.mul(b.xor(b.mod(lane, ci(32)), b.mod(key, ci(16))), ci(8)),
                )
            else:
                db = b.add(ci((j - 8) * 1024), b.mul(lane, ci(16)))
                nt = b.div(db, ci(512))
                w2 = b.xor(b.mod(db, ci(128)), b.mul(b.land(nt, ci(3)), ci(16)))
                key2 = b.add(
                    b.mul(b.mod(b.div(db, ci(128)), ci(4)), ci(4)), b.div(w2, ci(32))
                )
                off = b.add(
                    b.add(
                        b.add(
                            b.add(dm_row, ci(8192)),
                            b.mul(
                                b.mod(b.smin(b.add(dm_tile, key2), stop), ci(32)),
                                ci(256),
                            ),
                        ),
                        b.mul(nt, ci(16)),
                    ),
                    b.mul(b.div(b.mod(w2, ci(32)), ci(16)), ci(8)),
                )
            dma(
                b.add(dm_lds, ci(j * 1024)),
                b.mul(off, ci(2)),
                kvd,
                "buffer_load_dwordx4",
                "",
            )

    def split_layout(past, short):
        """source 2098-2100 / 2332-2336: split_tokens and splits from the position (the 8-row build's row-quad
        layout: 32 ceil((p + 8) / 1024) tokens per split)."""
        p8 = b.add(past, ci(8))
        st = b.mul(b.div(b.add(past, ci(8 + 1023)), ci(1024)), ci(32))
        if short:  # reduce: the row-pair value first, replaced while p + 8 > 0 (always)
            st = b.select(
                b.cmp_gt(p8, c0),
                st,
                b.mul(b.div(b.add(past, ci(8 + 511)), ci(512)), ci(32)),
            )
        return st, b.div(b.sub(b.add(p8, st), c1), st)

    def attention_partial_stage(T, layer, itid, blk):
        """source 2095-2327: blocks b with b % 128 < 4 splits: task (kv head, split) = b % 128, row quad rq = b // 128;
        each wave takes the split's 16-token tiles first_tile + 64 i (double-buffered LDS-DMA, the first tile issued
        before the kv-head wait and again after it if it holds new tokens); per tile QK^T (8 x 2 MFMA 16x16x32, columns
        = 24 (row, head) of the row quad in two groups), online softmax in log2 units, PV (16 x 2 MFMA 16x16x16 with
        ds_read_tr16 V); the four waves merge through LDS; partial O (bf16), max and denominator per column.
        """
        tid = itid
        lane = b.mod(tid, ci(64))
        wave = rfl(b.div(tid, ci(64)))
        asm("s_setreg_imm32_b32 0xdc1, 1;", "~{memory}")
        kvd = descriptor(T["context"], 2147483632)
        past = rfl(
            x.load(x.ptr1(T["position"]), "i32", I32, 16)
        )  # drives the task / tile loop bounds: scalar
        split_tokens, splits = split_layout(past, False)
        rq, hs = b.div(blk, ci(128)), b.mod(blk, ci(128))
        bm, qp = x.ptr1(T["block_map"]), x.ptr1(T["query"])
        L0 = x.ptr3(arena_addr)
        iy[0] += 1
        with b.scf_for(
            hs, b.mul(splits, ci(4)), ci(256), iv_name=f"atask{iy[0]}"
        ) as task:
            kvhead, split = b.div(task, splits), b.mod(task, splits)
            start = b.mul(split, split_tokens)
            stop = b.smin(
                b.sub(b.add(start, split_tokens), c1),
                b.add(b.add(past, b.mul(rq, ci(4))), ci(3)),
            )
            head_base = b.mul(kvhead, ci(6))
            sblk = b.div(start, ci(32))
            span = b.sub(b.div(stop, ci(32)), sblk)
            spanc = b.smax(span, c0)
            pages = ldp(gp(bm, "i32", b.add(sblk, b.smin(lane, spanc))), "i32")
            pages_hi = ldp(
                gp(bm, "i32", b.add(sblk, b.smin(b.add(lane, ci(64)), spanc))), "i32"
            )
            with b.scf_if(b.land(b.cmp_eq(itid, c0), b.cmp_ge(span, ci(128)))):
                fail_native()
            column = b.mod(lane, ci(16))
            first_tile = b.add(b.mul(sblk, ci(32)), b.mul(wave, ci(16)))
            last16 = b.mul(b.div(stop, ci(16)), ci(16))
            wbuf = b.add(arena_addr, b.mul(wave, ci(32768)))
            kv_tile_dma(
                kvd,
                lane,
                kvhead,
                pages,
                pages_hi,
                start,
                stop,
                b.smin(first_tile, last16),
                wbuf,
            )
            with b.scf_if(
                b.cmp_eq(itid, c0)
            ):  # queries (6 heads x 8 rows) and new K / V (8 rows) of this kv head
                counter_wait(
                    word(native, b.add(ci(CHUNK_WORD + 3648), b.mul(kvhead, ci(16)))),
                    b.mul(b.add(b.div(layer, ci(4)), c1), ci(56)),
                    4194304,
                )
            barrier()
            with b.scf_if(
                b.land(
                    b.cmp_le(first_tile, stop),
                    b.cmp_gt(b.add(first_tile, ci(16)), past),
                )
            ):
                kv_tile_dma(
                    kvd,
                    lane,
                    kvhead,
                    pages,
                    pages_hi,
                    start,
                    stop,
                    b.smin(first_tile, last16),
                    wbuf,
                )
            crow, chead, cstop, qv = [], [], [], []
            for g in range(2):
                gcol = b.add(column, ci(g * 16))
                crow.append(b.add(b.mul(rq, ci(4)), b.div(gcol, ci(6))))
                chead.append(b.mod(gcol, ci(6)))
                cstop.append(b.smin(stop, b.add(past, crow[g])))
                qbase = b.add(
                    b.mul(
                        b.add(b.add(b.mul(crow[g], ci(24)), head_base), chead[g]),
                        ci(256),
                    ),
                    b.mul(b.div(lane, ci(16)), ci(8)),
                )
                z8 = b.zero_vec(BF16, 8)
                qv.append(
                    if_yield(
                        b.cmp_lt(gcol, ci(24)),
                        [z8] * 8,
                        lambda cur, qbase=qbase: [
                            x.load(
                                gp(
                                    qp,
                                    "bfloat",
                                    b.add(qbase, ci(kb * 32)) if kb else qbase,
                                ),
                                "<8 x bfloat>",
                                VectorType(BF16, 8),
                                2,
                            )
                            for kb in range(8)
                        ],
                    )
                )
            if (
                A1
            ):  # A1: the Q loads above complete here, visible to LLVM's waitcnt pass (no in-loop Q guards)
                b.s_waitcnt(vmcnt=0)
            lp = b.scf_for_iter(
                first_tile,
                b.add(stop, c1),
                ci(64),
                [
                    (f"at0_{iy[0]}", fconst(0.0)),
                    (f"at1_{iy[0]}", fconst(0.0)),
                    (f"am0_{iy[0]}", fconst(NEG_MAX)),
                    (f"am1_{iy[0]}", fconst(NEG_MAX)),
                    (f"aa0_{iy[0]}", b.zero_vec_f32(64)),
                    (f"aa1_{iy[0]}", b.zero_vec_f32(64)),
                ],
                iv_name=f"atile{iy[0]}",
                elide_trailing_barrier=False,
            )
            with lp as (tile, (tot0, tot1, mx0, mx1, acc0, acc1)):
                cur = b.mod(b.div(b.sub(tile, first_tile), ci(64)), ci(2))
                kv_tile_dma(
                    kvd,
                    lane,
                    kvhead,
                    pages,
                    pages_hi,
                    start,
                    stop,
                    b.smin(b.add(tile, ci(64)), last16),
                    b.add(wbuf, b.mul(b.sub(c1, cur), ci(16384))),
                )
                wait_vm(16)
                kt = b.add(
                    b.mul(wave, ci(16384)), b.mul(cur, ci(8192))
                )  # bf16 elements of this K tile
                scores = [b.zero_vec_f32(4), b.zero_vec_f32(4)]
                for kb in range(8):
                    ko = b.div(
                        b.add(
                            b.mul(column, ci(512)),
                            b.mul(
                                b.xor(b.add(ci(kb * 4), b.div(lane, ci(16))), column),
                                ci(16),
                            ),
                        ),
                        ci(2),
                    )
                    key = x.load(
                        gp(L0, "bfloat", b.add(kt, ko), lds=True),
                        "<8 x bfloat>",
                        VectorType(BF16, 8),
                        2,
                        lds=True,
                    )
                    for g in range(2):
                        scores[g] = x.mfma_bf16_32(key, qv[g][kb], scores[g])
                mins, tins = [mx0, mx1], [tot0, tot1]
                corr, probs, nmax, ntot = [], [], [], []
                pos4 = [
                    b.add(b.add(tile, b.mul(b.div(lane, ci(16)), ci(4))), ci(e))
                    for e in range(4)
                ]
                for g in range(2):
                    ok = [
                        b.land(b.cmp_ge(pos4[e], start), b.cmp_le(pos4[e], cstop[g]))
                        for e in range(4)
                    ]
                    scaled = [
                        b.fmul(b.vec_extract(scores[g], e), fconst(SCORE_SCALE))
                        for e in range(4)
                    ]
                    masked = [
                        b.select(ok[e], scaled[e], fconst(NEG_MAX)) for e in range(4)
                    ]
                    tmax = x.maximum(
                        x.maximum(masked[0], masked[1]), x.maximum(masked[2], masked[3])
                    )
                    tmax = x.maximum(tmax, shuffle_xor_f(tmax, 16))
                    tmax = x.maximum(tmax, shuffle_xor_f(tmax, 32))
                    newmax = x.maximum(mins[g], tmax)
                    corr.append(x.exp2(b.fsub(mins[g], newmax)))
                    # FlyDSL: exp2(masked - newmax) only under ``if valid`` (masked = scaled there)
                    p = [
                        b.select(ok[e], x.exp2(b.fsub(scaled[e], newmax)), fconst(0.0))
                        for e in range(4)
                    ]
                    ntot.append(
                        b.fadd(
                            b.fmul(tins[g], corr[g]),
                            b.fadd(b.fadd(p[0], p[1]), b.fadd(p[2], p[3])),
                        )
                    )
                    nmax.append(newmax)
                    probs.append(
                        b.vec_bitcast(
                            b.vec_pack(
                                [b.trunc_f32_to_bf16(p[e]) for e in range(4)], BF16
                            ),
                            VectorType(I16, 4),
                        )
                    )
                vt = b.add(
                    b.add(arena_addr, b.mul(kt, ci(2))), ci(8192)
                )  # V tile byte address
                accs, upd = [acc0, acc1], [[], []]
                for nt in range(16):
                    br_ = b.add(
                        b.mul(b.add(ci(nt * 4), b.div(lane, ci(16))), ci(128)),
                        b.xor(b.mul(column, ci(8)), ci((nt & 3) * 16)),
                    )
                    val = x.ds_read_tr16(x.ptr3(b.add(vt, br_)))
                    for g in range(2):
                        old = b.vec_pack(
                            [
                                b.fmul(b.vec_extract(accs[g], nt * 4 + e), corr[g])
                                for e in range(4)
                            ],
                            F32,
                        )
                        r_ = x.mfma_bf16_1k(val, probs[g], old)
                        upd[g] += [b.vec_extract(r_, e) for e in range(4)]
                asm("", "~{memory}")
                b.scf_yield(
                    ntot[0],
                    ntot[1],
                    nmax[0],
                    nmax[1],
                    b.vec_pack(upd[0], F32),
                    b.vec_pack(upd[1], F32),
                )
            tot_g, max_g, acc_g = lp.results[0:2], lp.results[2:4], lp.results[4:6]
            wait_vm(0)
            # merge the four waves' states; exchange [wave][g][column (stride 260)][dim] at arena + 0
            SCR = (
                147456 // 4
            )  # float element offset of the merge scratch [g][wave][max 16 | total 16]
            scr = x.gep(
                L0, "float", ci(SCR), lds=True
            )  # FlyDSL: one base pointer, then idx, then constant geps
            for g in range(2):
                folded = b.fadd(tot_g[g], shuffle_xor_f(tot_g[g], 16))
                folded = b.fadd(folded, shuffle_xor_f(folded, 32))
                with b.scf_if(b.cmp_lt(lane, ci(16))):
                    sp_ = x.gep(
                        scr,
                        "float",
                        b.add(b.mul(b.add(wave, ci(g * 4)), ci(32)), lane),
                        lds=True,
                    )
                    stp(sp_, "float", max_g[g], lds=True)
                    stp(
                        x.gep(sp_, "float", ci(16), lds=True), "float", folded, lds=True
                    )
            barrier()
            scol = x.gep(scr, "float", column, lds=True)
            wmax = [
                [
                    ldp(
                        (
                            x.gep(scol, "float", ci((g * 4 + w) * 32), lds=True)
                            if g or w
                            else scol
                        ),
                        "float",
                        lds=True,
                    )
                    for w in range(4)
                ]
                for g in range(2)
            ]
            common = [
                x.maximum(
                    x.maximum(wmax[g][0], wmax[g][1]), x.maximum(wmax[g][2], wmax[g][3])
                )
                for g in range(2)
            ]
            dead = [b.fcmp("olt", common[g], fconst(-3.0e38)) for g in range(2)]
            for g in range(2):
                w_own = b.select(
                    dead[g], fconst(0.0), x.exp2(b.fsub(max_g[g], common[g]))
                )
                eb = b.add(
                    b.add(
                        b.mul(b.add(b.mul(wave, ci(2)), ci(g)), ci(4160)),
                        b.mul(column, ci(260)),
                    ),
                    b.mul(b.div(lane, ci(16)), ci(4)),
                )
                ep = x.gep(L0, "float", eb, lds=True)
                for nt in range(16):
                    sc_ = b.vec_pack(
                        [
                            b.fmul(b.vec_extract(acc_g[g], nt * 4 + e), w_own)
                            for e in range(4)
                        ],
                        F32,
                    )
                    x.store(
                        x.gep(ep, "float", ci(nt * 16), lds=True) if nt else ep,
                        sc_,
                        "<4 x float>",
                        4,
                        lds=True,
                    )
            barrier()
            for g in range(2):
                cslot = b.add(
                    b.mul(
                        b.add(
                            b.mul(b.add(b.mul(crow[g], ci(4)), kvhead), splits), split
                        ),
                        ci(6),
                    ),
                    chead[g],
                )
                with b.scf_if(b.cmp_lt(b.add(column, ci(g * 16)), ci(24))):
                    with b.scf_if(b.cmp_eq(wave, c0)):
                        folded = fconst(0.0)
                        for w in range(4):
                            ww = b.select(
                                dead[g],
                                fconst(0.0),
                                x.exp2(b.fsub(wmax[g][w], common[g])),
                            )
                            sv = ldp(
                                x.gep(
                                    scol, "float", ci((g * 4 + w) * 32 + 16), lds=True
                                ),
                                "float",
                                lds=True,
                            )
                            folded = b.fadd(folded, b.fmul(sv, ww))
                        stp(
                            gp(x.ptr1(T["maximum"]), "float", cslot), "float", common[g]
                        )
                        stp(
                            gp(x.ptr1(T["denominator"]), "float", cslot),
                            "float",
                            folded,
                        )
                    for part in range(2):
                        dim0 = b.add(
                            b.add(
                                b.mul(wave, ci(64)), b.mul(b.div(lane, ci(16)), ci(16))
                            ),
                            ci(part * 8),
                        )
                        sums = [fconst(0.0)] * 8
                        xb_ = x.gep(
                            L0, "float", b.add(b.mul(column, ci(260)), dim0), lds=True
                        )
                        for w in range(4):
                            for q4 in range(2):
                                pc = x.load(
                                    (
                                        x.gep(
                                            xb_,
                                            "float",
                                            ci((w * 2 + g) * 4160 + q4 * 4),
                                            lds=True,
                                        )
                                        if (w * 2 + g) or q4
                                        else xb_
                                    ),
                                    "<4 x float>",
                                    V4F,
                                    4,
                                    lds=True,
                                )
                                for e in range(4):
                                    sums[q4 * 4 + e] = b.fadd(
                                        sums[q4 * 4 + e], b.vec_extract(pc, e)
                                    )
                        x.store(
                            gp(
                                x.ptr1(T["partial"]),
                                "bfloat",
                                b.add(b.mul(cslot, ci(256)), dim0),
                            ),
                            b.vec_pack([b.trunc_f32_to_bf16(s_) for s_ in sums], BF16),
                            "<8 x bfloat>",
                            2,
                        )
            barrier()

    def attention_reduce_stage(T, itid, blk):
        """source 2329-2410, blocks 0-191: pair b // 6 = (row, kv head), head kv 6 + b % 6, thread = dim. Split weights
        exp2(max_s - max) / sum_s(den_s w_s) (lanes = splits), the 32 partial loads (out-of-range splits read past
        num_records), the split-order fma chain, bf16, x sigmoid(gate) -> bf16 (gated), MXFP8 per 32 dims ->
        mixer_A / _AS (write-through); block 0 writes 127 into the scale padding."""
        tid = itid
        past = rfl(x.load(x.ptr1(T["position"]), "i32", I32, 16))
        _, splits = split_layout(past, True)
        pair, h = b.div(blk, ci(6)), b.mod(blk, ci(6))
        with b.scf_if(b.cmp_lt(pair, ci(32))):
            row, kvhead = b.div(pair, ci(4)), b.mod(pair, ci(4))
            first = b.mul(b.add(b.mul(row, ci(4)), kvhead), splits)
            head = b.add(b.mul(kvhead, ci(6)), h)
            lane = b.mod(tid, ci(64))
            inr = b.cmp_lt(lane, splits)
            mo = b.add(b.mul(b.add(first, lane), ci(6)), h)
            ownmax, ownsum = if_yield(
                inr,
                [fconst(NEG_MAX), fconst(0.0)],
                lambda cur: [
                    ldp(gp(x.ptr1(T["maximum"]), "float", mo), "float"),
                    ldp(gp(x.ptr1(T["denominator"]), "float", mo), "float"),
                ],
            )
            mv = ownmax
            for s_ in (32, 16, 8, 4, 2, 1):
                mv = x.maximum(mv, shuffle_xor_f(mv, s_))
            ow = b.select(inr, x.exp2(b.fsub(ownmax, mv)), fconst(0.0))
            tot = b.fmul(ownsum, ow)
            for s_ in (32, 16, 8, 4, 2, 1):
                tot = b.fadd(tot, shuffle_xor_f(tot, s_))
            divisor = b.select(b.fcmp("ogt", tot, fconst(0.0)), tot, fconst(1.0))
            nw = b.fdiv(ow, divisor)
            rd = descriptor(T["partial"], 0x7FFFFFF0)
            vals = []
            for s_ in range(32):
                okv = zx(b.cmp_gt(splits, ci(s_)))
                off = b.add(
                    b.mul(
                        okv,
                        b.mul(
                            b.add(
                                b.mul(
                                    b.add(b.mul(b.add(first, ci(s_)), ci(6)), h),
                                    ci(256),
                                ),
                                tid,
                            ),
                            ci(2),
                        ),
                    ),
                    b.mul(b.sub(c1, okv), ci(0x7FFFFFF0)),
                )
                vals.append(bf2f(b.bitcast(x.buf_ld2(rd, off, 0), BF16)))
            value = fconst(0.0)
            for s_ in range(32):
                value = x.fma(vals[s_], shfl_idx(nw, s_), value)
            rounded = b.trunc_f32_to_bf16(value)
            go = b.add(b.add(b.mul(row, ci(6144)), b.mul(head, ci(256))), tid)
            gv = bf2f(ldp(gp(x.ptr1(T["gate"]), "bfloat", go), "bfloat"))
            sig = b.fdiv(
                fconst(1.0),
                b.fadd(
                    x.ocml("exp2", b.fmul(b.fneg(gv), fconst(1.442695041))), fconst(1.0)
                ),
            )
            final = b.trunc_f32_to_bf16(b.fmul(bf2f(rounded), sig))
            stp(gp(x.ptr1(T["gated"]), "bfloat", go), "bfloat", final)
            amax = b.fabs(bf2f(final))
            for s_ in range(5):
                amax = x.maximum(amax, shuffle_xor_f(amax, 1 << s_))
            bits = b.bitcast(b.fmul(amax, fconst(1 / 448)), I32)
            ex = b.land(b.lshr(bits, ci(23)), ci(255))
            ex = b.select(b.cmp_ne(b.land(bits, ci(8388607)), c0), b.add(ex, c1), ex)
            ex = b.smax(b.smin(ex, ci(254)), c1)
            scale = b.bitcast(b.shl(ex, ci(23)), F32)
            own = b.land(b.sext(b.bitcast(final, I16), I32), ci(65535))
            packed = b.lor(own, b.shl(x.swizzle(own, (1 << 10) | 0x1F), ci(16)))
            with b.scf_if(b.cmp_eq(b.mod(tid, ci(2)), c0)):
                q = asm(
                    "v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2",
                    "=v,v,v,0",
                    [packed, scale, c0],
                    I32,
                    hint="cvq",
                )
                asm(
                    "global_store_short $0, $1, off sc0 sc1",
                    "v,v,~{memory}",
                    [b.add(T["A"], i64(go)), q],
                )
            with b.scf_if(b.cmp_eq(b.mod(tid, ci(32)), c0)):
                qb = b.div(b.add(b.mul(head, ci(256)), tid), ci(32))
                so = b.add(
                    b.add(
                        b.mul(b.div(qb, ci(4)), ci(64)), b.mul(b.mod(qb, ci(4)), ci(16))
                    ),
                    row,
                )
                asm(
                    "global_store_byte $0, $1, off sc0 sc1",
                    "v,v,~{memory}",
                    [b.add(T["AS"], i64(so)), ex],
                )
        with b.scf_if(b.cmp_eq(blk, c0)):
            for s_ in range(13):
                i_ = b.add(tid, ci(s_ * 256))
                with b.scf_if(b.cmp_lt(i_, ci(3264))):
                    with b.scf_if(
                        b.lor(
                            b.cmp_ge(i_, ci(3072)), b.cmp_ge(b.mod(i_, ci(16)), ci(8))
                        )
                    ):
                        asm(
                            "global_store_byte $0, $1, off sc0 sc1",
                            "v,v,~{memory}",
                            [b.add(T["AS"], i64(i_)), padv()],
                        )

    def attention_layer(TA, layer, itid, blk, alive, epoch_base):
        """source 2856-2884, after QGKV: qk (alive, blocks 0-223), partials (alive), native_grid(epoch_base + 4, status
        4: every block, inactive task blocks included), reduce + its mixer sub-counter (alive, blocks 0-191). ``alive``
        = this block's alive after QGKV (scalar); returns alive after the grid (scalar). TA: raw (tbl 22), context
        (tbl 25), tbl (the layer's table reader: slots 6-11 / 23-31 / 41 are read here in FlyDSL's order).
        """
        tbl_ = TA["tbl"]
        T = {"raw": TA["raw"], "context": TA["context"]}
        T.update(
            position=tbl_(8),
            qgamma=tbl_(6),
            kgamma=tbl_(7),
            query=tbl_(23),
            gate=tbl_(24),
            ctx_blocks=tbl_(9),
        )
        live = b.cmp_eq(alive, c1)
        with b.scf_if(b.land(live, b.cmp_lt(blk, ci(224)))):
            attention_qk_stage(T, layer, itid, blk)
        T.update(block_map=tbl_(10))
        tbl_(11)  # tasks, dispatch: loaded, never read by the 8-row build
        tbl_(41)
        T.update(partial=tbl_(26), maximum=tbl_(27), denominator=tbl_(28))
        with b.scf_if(live):
            sm = mode_save()
            with b.scf_if(
                b.cmp_gt(
                    b.add(rfl(x.load(x.ptr1(T["position"]), "i32", I32, 16)), ci(8)), c0
                )
            ):
                attention_partial_stage(T, layer, itid, blk)
            mode_restore(sm)
        al = b.smax(
            alive, native_grid(b.add(epoch_base, ci(4)), 4)
        )  # the kernel's one grid
        T.update(gated=tbl_(29), A=tbl_(30), AS=tbl_(31))
        with b.scf_if(b.land(b.cmp_eq(al, c1), b.cmp_lt(blk, ci(192)))):
            attention_reduce_stage(T, itid, blk)
            wait_vm(0)  # mixer_A slice published (slice head // 8, sub = row % 4)
            barrier()
            pair = b.div(blk, ci(6))
            mh = b.add(b.mul(b.mod(pair, ci(4)), ci(6)), b.mod(blk, ci(6)))
            publish0(
                itid,
                word(
                    native,
                    b.add(
                        ci(CHUNK_WORD + 640),
                        b.mul(
                            b.add(
                                b.mul(b.div(mh, ci(8)), ci(4)),
                                b.mod(b.div(pair, ci(4)), ci(4)),
                            ),
                            ci(16),
                        ),
                    ),
                ),
            )
        return al

    # ---------------------------------------------------------------- layer frame (source 296-319, 3026-3034)
    def layer_head(layer, alive):
        """layer entry: barrier, tid 0 status (alive, failure word) -> slot 11, barrier; iteration_tid / _block
        opacity; epoch_base; layer-63 current-slot record. Returns (alive, itid, blk, epoch_base, tbl, control).
        """
        barrier()
        with b.scf_if(b.cmp_eq(tid, c0)):
            bad = b.lor(
                b.cmp_ne(alive, c1), b.cmp_ne(coherent_u32(word(native, 289)), c0)
            )
            st_store(11, b.select(bad, ci(2), c1))
        barrier()
        alive = b.smax(alive, st_load(11))
        itid = b.land(asm("", "=v,0", [tid], I32, hint="itid"), ci(255))
        blk = b.land(asm("", "=s,0", [block], I32, hint="iblk"), ci(255))
        epoch_base = b.sub(
            b.add(b.add(command_base, ci(2)), b.mul(layer, ci(12))), b.div(layer, ci(4))
        )
        with b.scf_if(
            b.land(
                b.land(b.cmp_eq(block, c0), b.cmp_eq(tid, c0)),
                b.land(b.cmp_eq(alive, c1), b.cmp_eq(layer, ci(63))),
            )
        ):
            with b.scf_if(b.cmp_eq(coherent_u32(word(native, 289)), c0)):
                for cs in range(45):
                    system_store_u64(
                        b.add(native, b.const_i64((CURRENT_WORD // 2 + cs) * 8)),
                        raw_table_address(layer, cs),
                    )

        def tbl(slot_):
            return raw_table_address(layer, slot_)

        return alive, itid, blk, epoch_base, tbl, tbl(42)

    def layer_end(control, itid, blk, alive, layer):
        """layer exit: drain, alive -> slot 10 (rocKE: leaves the layer's scf.if region), barrier, tid 0 control words
        (Int32 indices 8 and 16 + blk, source 3031-3034; ``sc0 sc1``): failure word read, control[8] = 2 if (alive != 1
        or failed), control[16 + blk] = 2 / 1. Scalar-predicate rule: the layer loop must not end in a lane-dependent
        region, so wave 0 (scalar branch) runs ONE asm block that narrows EXEC to lane 0, does the read and the
        conditional stores, and restores EXEC (same memory operations and values as FlyDSL's tid-0 branch).
        """
        asm("s_waitcnt vmcnt(0) lgkmcnt(0)", "~{memory}")
        with b.scf_if(b.cmp_eq(itid, c0)):
            st_store(10, alive)
        barrier()
        with b.scf_if(b.cmp_eq(rfl(b.div(itid, ci(64))), c0)):
            t = (
                "s_mov_b64 $0, exec\n s_and_b64 exec, exec, 1\n"
                " global_load_dword $1, $3, off sc0 sc1\n s_waitcnt vmcnt(0)\n"
                " v_cmp_ne_u32 vcc, 0, $1\n s_cmp_lg_u32 $5, 1\n s_cselect_b64 $2, exec, 0\n s_or_b64 vcc, vcc, $2\n"
                " s_and_b64 vcc, vcc, exec\n v_mov_b32 $1, 2\n s_mov_b64 $2, exec\n s_mov_b64 exec, vcc\n"
                " global_store_dword $6, $1, off sc0 sc1\n s_mov_b64 exec, $2\n"
                " v_cndmask_b32_e64 $1, 1, 2, vcc\n global_store_dword $4, $1, off sc0 sc1\n s_mov_b64 exec, $0"
            )
            b.inline_asm_multi(
                t,
                "=&s,=&v,=&s,v,v,s,v,~{memory},~{vcc},~{scc}",
                [
                    word(native, 289),
                    word(control, b.add(ci(16), blk)),
                    rfl(alive),
                    word(control, 8),
                ],
                result_types=[I64, I32, I64],
                result_name_hint="ctl",
            )

    # ---------------------------------------------------------------- one layer (src_mlp.py 276-2878)
    def layer_body(layer, alive):
        """source 296-3034, one layer: frame, [layer-0 RMS | fold], QKV + GDN core / peer / finalize | QGKV + attention,
        relay_mixer, output, post-norm owners | Up + relay, Down, layer 63's tail, layer-end control words.
        """
        alive, itid, blk, epoch_base, tbl, control = layer_head(layer, alive)
        attn = b.cmp_eq(b.mod(layer, ci(4)), ci(3))
        norm_phase = b.select(attn, ci(7), ci(8))
        T0 = {
            nm: tbl(s_)
            for nm, s_ in (
                ("hidden", 0),
                ("residual", 1),
                ("gamma", 2),
                ("normalized", 18),
                ("residual_out", 19),
                ("act", 20),
                ("act_s", 21),
                ("W", 4),
                ("WS", 5),
                ("raw", 22),
            )
        }
        mixer = tbl(32)
        with b.scf_if(b.cmp_eq(alive, c1)):
            with b.scf_if(
                b.land(b.lnot(attn), b.cmp_lt(blk, ci(8)))
            ):  # Delta BA clear ahead of the fold
                early_ba = tbl(25)
                ec = b.add(b.mul(blk, ci(256)), itid)
                with b.scf_if(b.cmp_lt(ec, ci(1024))):
                    system_store_u32(b.add(early_ba, i64(b.mul(ec, ci(4)))), c0)
            with b.scf_if(b.land(b.cmp_lt(blk, ci(8)), b.cmp_eq(layer, c0))):
                sm = mode_save()
                rms_stage(T0, blk, itid)
                mode_restore(sm)
                barrier()
                with b.scf_if(b.cmp_eq(itid, c0)):
                    b.fence(scope="agent", ordering="release")
                    system_atomic_add_nw(
                        word(native, b.add(ci(CHUNK_WORD + 512), b.mul(blk, ci(16))))
                    )
            with b.scf_if(b.land(b.cmp_lt(blk, ci(64)), b.cmp_gt(layer, c0))):
                sm = mode_save()
                fold_chunk_stage(
                    dict(T0, partial=tbl(39 - 45)),
                    layer,
                    itid,
                    blk,
                    b.add(epoch_base, c1),
                )
                mode_restore(sm)
                barrier()
                with b.scf_if(b.cmp_eq(itid, c0)):
                    system_atomic_add_nw(
                        word(
                            native,
                            b.add(
                                ci(CHUNK_WORD + 512), b.mul(b.div(blk, ci(8)), ci(16))
                            ),
                        )
                    )
        # alive = max(alive, fold result): the fold's two statuses (LDS 6, 5) leave its region here
        fold_ran = b.land(
            b.cmp_eq(alive, c1), b.land(b.cmp_lt(blk, ci(64)), b.cmp_gt(layer, c0))
        )
        alive = b.select(fold_ran, b.smax(alive, b.smax(st_load(6), st_load(5))), alive)
        mixer_A = tbl(30)
        mixer_AS = tbl(31)
        QA = {
            "A": T0["act"],
            "AS": T0["act_s"],
            "W": T0["W"],
            "WS": T0["WS"],
            "raw": T0["raw"],
        }
        # (branch on "not attention", GDN body first: FlyDSL's object lays the GDN path out first)
        with b.scf_if_else(b.lnot(attn)) as (tg, ta):
            with (
                tg
            ):  # GDN layer: QKV + conv + BA, core / core tail (+ 240-247 Up L2 touch) / peer hand-off / finalize
                TG = dict(
                    QA,
                    BA=tbl(25),
                    conv_w=tbl(6),
                    hist=tbl(7),
                    work=tbl(23),
                    snap=tbl(24),
                )
                with b.scf_if(b.cmp_eq(alive, c1)):
                    qkv_common(TG, layer, itid, blk, 1, False)
                al_g = b.select(b.cmp_eq(alive, c1), b.smax(alive, st_load(1)), alive)
                TG.update(
                    init=tbl(8),
                    A_log=tbl(9),
                    dt_bias=tbl(10),
                    core=tbl(26),
                    final=tbl(27),
                    squares=tbl(28),
                )
                tbl(
                    41
                )  # state_snapshots (FlyDSL loads it; the non-diagnostic build never uses it)
                TG.update(
                    norm=tbl(11), recurrent=tbl(29), A=tbl(30), AS=tbl(31), tbl=tbl
                )
                gdn_layer(TG, layer, itid, blk, al_g, epoch_base)
            with (
                ta
            ):  # attention layer: QGKV, then qk / partials / grid / reduce (+ mixer sub-counter)
                ctx = tbl(25)  # context (the layer's KV pages)
                with b.scf_if(b.land(b.cmp_eq(alive, c1), b.cmp_lt(blk, ci(224)))):
                    qkv_common(QA, layer, itid, blk, 1, True)
                al_a = b.select(
                    b.land(b.cmp_eq(alive, c1), b.cmp_lt(blk, ci(224))),
                    b.smax(alive, st_load(1)),
                    alive,
                )
                attention_layer(
                    {"raw": T0["raw"], "context": ctx, "tbl": tbl},
                    layer,
                    itid,
                    blk,
                    rfl(al_a),
                    epoch_base,
                )
        qkv_ran = b.land(
            b.cmp_eq(alive, c1), b.lor(b.lnot(attn), b.cmp_lt(blk, ci(224)))
        )
        alive = b.select(qkv_ran, b.smax(alive, st_load(1)), alive)
        # GDN blocks 0-191: alive = max(alive, peer_handoff()); attention layers, every block: max(alive, grid status)
        # (both statuses leave their regions through LDS slot 4)
        alive = b.select(
            b.land(b.lnot(attn), b.cmp_lt(blk, ci(192))),
            b.smax(alive, st_load(4)),
            alive,
        )
        alive = b.select(attn, b.smax(alive, rfl(st_load(4))), alive)
        out_w = tbl(12)
        out_s = tbl(13)
        with b.scf_if(b.cmp_ge(blk, ci(248))):
            relay_mixer_slices(layer, itid)
            with b.scf_if(b.land(b.cmp_eq(alive, c1), b.lnot(attn))):
                up_l2_touch(tbl, itid, blk, 29, 3, 1, 16)
        oslot = b.select(attn, ci(5), ci(6))  # output_stage (source 2944)
        output_stage(
            {"A": mixer_A, "AS": mixer_AS, "W": out_w, "WS": out_s, "mixer": mixer},
            tbl,
            layer,
            itid,
            blk,
            alive,
            oslot,
            attn,
        )
        alive = b.select(
            b.land(b.cmp_eq(alive, c1), b.cmp_lt(blk, ci(240))),
            b.smax(alive, st_load(oslot)),
            alive,
        )
        post_norm, post_nz, resid2 = [tbl(s_) for s_ in (3, 33, 34)]
        P = {
            "post_A": tbl(35),
            "post_AS": tbl(36),
            "up_w": tbl(14),
            "up_s": tbl(15),
            "act": tbl(37),
            "act_s": tbl(38),
        }
        is_up = b.land(
            b.cmp_gt(blk, c0), b.lor(b.cmp_le(blk, ci(136)), b.cmp_ge(blk, ci(144)))
        )
        epoch_n = b.add(epoch_base, norm_phase)
        with b.scf_if_else(is_up) as (t_ctx, f_ctx):
            with t_ctx:
                up_stage(P, layer, itid, blk, alive, norm_phase)
                with b.scf_if(b.cmp_ge(blk, ci(248))):
                    relay_up_slices(layer, itid)
            with f_ctx:  # post-attention norm owners (source 2958-2995)
                al = b.smax(alive, owner_output_gate(layer, itid))
                norm_row = b.select(b.cmp_eq(blk, c0), c0, b.sub(blk, ci(136)))
                with b.scf_if(b.land(b.cmp_eq(al, c1), b.cmp_lt(norm_row, ci(8)))):
                    sm = mode_save()
                    rms_stage(
                        {
                            "hidden": mixer,
                            "residual": T0["residual_out"],
                            "gamma": post_norm,
                            "normalized": post_nz,
                            "residual_out": resid2,
                            "act": P["post_A"],
                            "act_s": P["post_AS"],
                        },
                        norm_row,
                        itid,
                    )
                    mode_restore(sm)
                    if DUMP:
                        dump_mixer_row(mixer, layer, norm_row, itid)
                al = b.smax(al, native_postnorm(epoch_n, norm_phase, layer, blk))
                with b.scf_if(b.cmp_eq(al, c1)):
                    down_l2_touch(tbl, itid, blk)
                with b.scf_if(b.cmp_eq(al, c1)):
                    zero_mixer_reserve(mixer, blk, itid)
        # alive after the Up / owner branch (all threads: the statuses were published behind barriers)
        alive = b.smax(alive, st_load(norm_phase))
        owner = b.lnot(is_up)
        alive = b.select(owner, b.smax(alive, st_load(3)), alive)
        with b.scf_if(b.cmp_eq(alive, c1)):  # fill_up_padding(activated_AS)
            index = b.add(b.mul(blk, ci(256)), itid)
            with b.scf_if(b.cmp_lt(index, ci(8896))):
                with b.scf_if(
                    b.lor(
                        b.cmp_ge(index, ci(8704)), b.cmp_ge(b.mod(index, ci(16)), ci(8))
                    )
                ):
                    asm(
                        "global_store_byte $0, $1, off",
                        "v,v,~{memory}",
                        [b.add(P["act_s"], i64(index)), padv()],
                    )
        P.update(dn_w=tbl(16), dn_s=tbl(17), part=tbl(39))
        dslot = b.add(norm_phase, c1)
        down_stage(P, layer, itid, blk, alive, dslot)
        alive = b.select(b.cmp_lt(blk, ci(240)), b.smax(alive, st_load(dslot)), alive)
        alive = tail(
            layer, itid, blk, alive, epoch_base, norm_phase, P["part"]
        )  # layer 63 only (inside)
        layer_end(control, itid, blk, alive, layer)

    # ------------------------------------------------------------ prologue + layer loop (source 273-3034)
    alive0 = prologue(alive0)
    lp = b.scf_for_iter(
        c0,
        ci(64),
        c1,
        [("alive", alive0)],
        iv_name="layer",
        elide_trailing_barrier=False,
    )
    with lp as (layer, (alive,)):
        with b.scf_if(b.cmp_eq(alive, c1)):
            layer_body(layer, alive)
        b.scf_yield(b.smax(alive, st_load(10)))
    b.ret()
    return b.kernel


def specs() -> list[KernelSpec]:
    """p2c_full (base port), p2c_full_pin / p2c_full_mxpin (diagnostic twins), p2c_full_a1 (A1, the timed kernel)
    and A1's two diagnostic twins."""
    return [
        KernelSpec("p2c_full", "p2c_full", ()),
        KernelSpec("p2c_full_pin", "p2c_full_pin", ("pin",)),
        KernelSpec("p2c_full_mxpin", "p2c_full_mxpin", ("mxpin",)),
        KernelSpec("p2c_full_a1", "p2c_full_a1", ("a1",)),
        KernelSpec("p2c_full_a1_mxpin", "p2c_full_a1_mxpin", ("a1", "mxpin")),
        KernelSpec("p2c_full_a1_pin", "p2c_full_a1_pin", ("a1", "pin")),
    ]
