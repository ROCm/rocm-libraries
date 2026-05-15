# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Lower CK DSL IR to AMDGPU LLVM IR text.

This is the fast-compile path: instead of emitting HIP C++ source and
forcing clang to re-parse `<hip/hip_runtime.h>` + STL, we go straight
to LLVM IR with AMDGPU intrinsics. The resulting `.ll` text is fed to
`clang -x ir` (subprocess) or `libamd_comgr` (in-process) to produce a
HSA code object that `hipModuleLoadData` can launch.

The lowering mirrors what MLIR's ROCDL dialect does on the AMDGPU
side: identical LLVM intrinsic targets, identical address-space
conventions.

What's hard, and how we handle it:

- `scf.for` with `iter_args` becomes 4 basic blocks (`entry → header →
  body → latch → exit`) plus phi nodes in the header for the induction
  variable and every loop-carried value. `scf.yield` is *recorded* by
  the body region; the latch block emits the IV increment and back-edge
  and feeds the recorded yielded values back into the header phis.
- `tile.smem_alloc` is a module-level addrspace(3) global; subsequent
  uses GEP into it. We collect smem allocs in a pre-pass.
- The three immarg operands to `mfma` (`cbsz`, `abid`, `blgp`) must be
  literal `i32 0` constants in the IR; we emit them as such, not as SSA
  values.
- LLVM SSA naming: we use the names already in our IR (`%v3`, `%tid8`,
  `%cz21`, …); they are valid LLVM identifiers. Block-local names get a
  block suffix where needed (e.g. `%iv.next`, `%cmp.13`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .ir import (
    KernelDef,
    Op,
    PtrType,
    Region,
    SmemType,
    Type,
    Value,
    VectorType,
)


# Datalayout / triple. Copied verbatim from clang's output for the same
# target on this box: clang -target amdgcn-amd-amdhsa -mcpu=gfx950
# -emit-llvm -S. If you change ROCm version, regenerate this string.
_DATALAYOUT = (
    "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
    "-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32"
    "-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048"
    "-n32:64-S32-A5-G1-ni:7:8:9"
)
_TRIPLE = "amdgcn-amd-amdhsa"

# Intrinsic declarations we may emit.
_INTRINSIC_DECLS: Dict[str, str] = {
    "workitem.x": "declare i32 @llvm.amdgcn.workitem.id.x()",
    "workitem.y": "declare i32 @llvm.amdgcn.workitem.id.y()",
    "workitem.z": "declare i32 @llvm.amdgcn.workitem.id.z()",
    "workgroup.x": "declare i32 @llvm.amdgcn.workgroup.id.x()",
    "workgroup.y": "declare i32 @llvm.amdgcn.workgroup.id.y()",
    "workgroup.z": "declare i32 @llvm.amdgcn.workgroup.id.z()",
    "s.barrier": "declare void @llvm.amdgcn.s.barrier()",
    "exp2.f32": "declare float @llvm.exp2.f32(float)",
    "sqrt.f32": "declare float @llvm.sqrt.f32(float)",
    "rsqrt.f32": "declare float @llvm.amdgcn.rsq.f32(float)",
    "tanh.f32": "declare float @llvm.tanh.f32(float)",
    "maxnum.f32": "declare float @llvm.maxnum.f32(float, float)",
    "minnum.f32": "declare float @llvm.minnum.f32(float, float)",
    "mfma.f32.16x16x16f16": (
        "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16("
        "<4 x half>, <4 x half>, <4 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.16x16x32.f16": (
        "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16("
        "<8 x half>, <8 x half>, <4 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.16x16x16bf16.1k": (
        # gfx950 lowers `16x16x16` bf16 MFMAs through the `_1k` variant
        # introduced for CDNA2; the operands take `<4 x i16>` (bitcast of
        # `<4 x bfloat>`). There is no plain `mfma.f32.16x16x16.bf16`
        # intrinsic on this LLVM target -- attempting to declare it
        # produces `undefined symbol` at link time, which is what blocked
        # bf16 head_size>=128 attention until this fix.
        "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k("
        "<4 x i16>, <4 x i16>, <4 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.16x16x32.bf16": (
        "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16("
        "<8 x bfloat>, <8 x bfloat>, <4 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.32x32x8f16": (
        "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16("
        "<4 x half>, <4 x half>, <16 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.32x32x16.f16": (
        "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16("
        "<8 x half>, <8 x half>, <16 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.4x4x4f16": (
        "declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x4f16("
        "<4 x half>, <4 x half>, <4 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "readfirstlane.i32": ("declare i32 @llvm.amdgcn.readfirstlane.i32(i32)"),
    "readfirstlane.i64": ("declare i64 @llvm.amdgcn.readfirstlane.i64(i64)"),
    "ds.bpermute": ("declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)"),
    "mbcnt.lo": ("declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)"),
    "mbcnt.hi": ("declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)"),
    "ds.read.tr16.b64": (
        "declare <4 x i16> @llvm.amdgcn.ds.read.tr16.b64(ptr addrspace(3))"
    ),
    "sched.barrier": ("declare void @llvm.amdgcn.sched.barrier(i32 immarg)"),
    "sched.group.barrier": (
        "declare void @llvm.amdgcn.sched.group.barrier("
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "s.setprio": ("declare void @llvm.amdgcn.s.setprio(i16 immarg)"),
    "s.waitcnt": ("declare void @llvm.amdgcn.s.waitcnt(i32 immarg)"),
    "make.buffer.rsrc.p1": (
        "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p1("
        "ptr addrspace(1) nocapture readnone, i16, i32, i32)"
    ),
    "raw.ptr.buffer.load.lds": (
        "declare void @llvm.amdgcn.raw.ptr.buffer.load.lds("
        "ptr addrspace(8) nocapture readonly, ptr addrspace(3) nocapture, "
        "i32, i32, i32, i32 immarg, i32 immarg)"
    ),
    "raw.ptr.buffer.load.v2i32": (
        "declare <2 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v2i32("
        "ptr addrspace(8) nocapture readonly, i32, i32, i32 immarg)"
    ),
    "raw.ptr.buffer.load.v4i32": (
        "declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32("
        "ptr addrspace(8) nocapture readonly, i32, i32, i32 immarg)"
    ),
    "raw.ptr.buffer.load.i32": (
        "declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32("
        "ptr addrspace(8) nocapture readonly, i32, i32, i32 immarg)"
    ),
    "raw.ptr.buffer.store.i32": (
        "declare void @llvm.amdgcn.raw.ptr.buffer.store.i32("
        "i32, ptr addrspace(8) nocapture writeonly, i32, i32, i32 immarg)"
    ),
    "raw.ptr.buffer.store.v2i32": (
        "declare void @llvm.amdgcn.raw.ptr.buffer.store.v2i32("
        "<2 x i32>, ptr addrspace(8) nocapture writeonly, i32, i32, i32 immarg)"
    ),
    "raw.ptr.buffer.store.v4i32": (
        "declare void @llvm.amdgcn.raw.ptr.buffer.store.v4i32("
        "<4 x i32>, ptr addrspace(8) nocapture writeonly, i32, i32, i32 immarg)"
    ),
    "raw.ptr.buffer.store.i16": (
        "declare void @llvm.amdgcn.raw.ptr.buffer.store.i16("
        "i16, ptr addrspace(8) nocapture writeonly, i32, i32, i32 immarg)"
    ),
    "amdgcn.cvt.f32.fp8": ("declare float @llvm.amdgcn.cvt.f32.fp8(i32, i32 immarg)"),
}


def _llvm_type(t: Type) -> str:
    """Map an IR Type to its LLVM IR textual form."""
    if isinstance(t, PtrType):
        if t.space == "global":
            return "ptr addrspace(1)"
        if t.space == "lds":
            return "ptr addrspace(3)"
        return "ptr"
    if isinstance(t, VectorType):
        return f"<{t.count} x {_llvm_type(t.elem)}>"
    if isinstance(t, SmemType):
        # The value-level type of a smem allocation token is "the pointer
        # to its base", since uses GEP into it.
        return "ptr addrspace(3)"
    if t.name == "i1":
        return "i1"
    if t.name == "i8":
        return "i8"
    if t.name == "i32":
        return "i32"
    if t.name == "i64":
        return "i64"
    if t.name == "f16":
        return "half"
    if t.name == "bf16":
        return "bfloat"
    if t.name == "fp8e4m3":
        return "i8"
    if t.name == "f32":
        return "float"
    raise NotImplementedError(f"no LLVM mapping for type {t!r}")


def _smem_storage_type(t: SmemType) -> str:
    """LLVM aggregate type for a smem allocation: nested arrays of halves/floats."""
    inner = _llvm_type(t.elem)
    out = inner
    for d in reversed(t.shape):
        out = f"[{d} x {out}]"
    return out


# ----------------------------- block model -------------------------------


@dataclass
class _Block:
    label: str
    lines: List[str] = field(default_factory=list)
    terminated: bool = False

    def emit(self, line: str) -> None:
        if self.terminated:
            raise RuntimeError(
                f"block {self.label!r} already terminated; cannot emit {line!r}"
            )
        self.lines.append(line)


# ----------------------------- lowerer -----------------------------------


class _Lowerer:
    def __init__(self, kernel: KernelDef) -> None:
        self.kernel = kernel
        self._needs_intrin: Dict[str, bool] = {}
        self._smem_globals: List[Tuple[str, SmemType]] = []
        self._smem_storage_name: Dict[str, str] = {}  # IR value name -> @global name
        self._blocks: List[_Block] = [_Block("entry")]
        self._block_counter = 0
        self._tmp_counter = 0
        # For scf.for nesting we record the body-region's recorded
        # yield-operand stack so the latch block can read it back.
        self._yield_stack: List[List[str]] = []

    # ----- helpers -----

    def _current(self) -> _Block:
        return self._blocks[-1]

    def _new_block(self, base: str) -> _Block:
        self._block_counter += 1
        blk = _Block(f"{base}.{self._block_counter}")
        self._blocks.append(blk)
        return blk

    def _fresh(self, hint: str) -> str:
        self._tmp_counter += 1
        return f"%{hint}.{self._tmp_counter}"

    def _operand(self, v: Value) -> str:
        """Return the textual LLVM operand for an IR Value.

        Constants are inlined as literals; other values use their SSA name.
        """
        op = v.op
        if op is None:
            return v.name
        if op.name == "arith.constant":
            ity = op.attrs.get("ity", "i32")
            val = op.attrs["value"]
            if ity == "f32":
                return _fp32_hex(val)
            if ity == "f16":
                return _fp16_hex(val)
            return str(int(val))
        return v.name

    def _operand_with_type(self, v: Value) -> str:
        return f"{_llvm_type(v.type)} {self._operand(v)}"

    def _need(self, key: str) -> None:
        self._needs_intrin[key] = True

    # ----- pre-pass: collect smem allocations -----

    def _collect_smem(self, region: Region) -> None:
        for op in region.ops:
            if op.name == "tile.smem_alloc":
                short = op.result.name.lstrip("%")
                gname = f"@{short}.{self.kernel.name}"
                self._smem_globals.append((gname, op.result.type))
                self._smem_storage_name[op.result.name] = gname
            for r in op.regions:
                self._collect_smem(r)

    # ----- per-op lowerings -----

    def lower_op(self, op: Op) -> None:
        method = getattr(self, f"_op_{op.name.replace('.', '_')}", None)
        if method is None:
            raise NotImplementedError(f"no LLVM lowering for op {op.name!r}")
        method(op)

    def lower_region(self, region: Region) -> None:
        for op in region.ops:
            self.lower_op(op)

    # arith

    def _op_arith_constant(self, op: Op) -> None:
        # Constants are emitted lazily at point of use. No-op here.
        return

    def _op_arith_constant_vec(self, op: Op) -> None:
        res = op.result
        if isinstance(res.type, VectorType):
            fill = float(op.attrs.get("fill", 0.0))
            if fill == 0.0:
                # zeroinitializer is the canonical form (works for any vector type)
                self._current().emit(
                    f"  {res.name} = select i1 true, {_llvm_type(res.type)} zeroinitializer, "
                    f"{_llvm_type(res.type)} zeroinitializer"
                )
                return
        raise NotImplementedError(f"arith.constant_vec: {op.attrs}")

    def _binop(self, op: Op, llvm_op: str) -> None:
        a, b = op.operands
        self._current().emit(
            f"  {op.result.name} = {llvm_op} {_llvm_type(op.result.type)} "
            f"{self._operand(a)}, {self._operand(b)}"
        )

    def _op_arith_add(self, op: Op) -> None:
        self._binop(op, "add nsw")

    def _op_arith_sub(self, op: Op) -> None:
        self._binop(op, "sub nsw")

    def _op_arith_mul(self, op: Op) -> None:
        self._binop(op, "mul nsw")

    def _op_arith_div(self, op: Op) -> None:
        self._binop(op, "sdiv")

    def _op_arith_mod(self, op: Op) -> None:
        self._binop(op, "srem")

    def _op_arith_fadd(self, op: Op) -> None:
        self._binop(op, "fadd")

    def _op_arith_fsub(self, op: Op) -> None:
        self._binop(op, "fsub")

    def _op_arith_fmul(self, op: Op) -> None:
        self._binop(op, "fmul")

    def _op_arith_fdiv(self, op: Op) -> None:
        self._binop(op, "fdiv")

    def _op_arith_fneg(self, op: Op) -> None:
        (v,) = op.operands
        self._current().emit(
            f"  {op.result.name} = fneg {_llvm_type(v.type)} {self._operand(v)}"
        )

    def _op_arith_cmp(self, op: Op) -> None:
        pred = op.attrs.get("pred", "lt")
        pmap = {
            "lt": "slt",
            "le": "sle",
            "gt": "sgt",
            "ge": "sge",
            "eq": "eq",
            "ne": "ne",
        }
        a, b = op.operands
        self._current().emit(
            f"  {op.result.name} = icmp {pmap[pred]} {_llvm_type(a.type)} "
            f"{self._operand(a)}, {self._operand(b)}"
        )

    def _op_arith_fcmp(self, op: Op) -> None:
        pred = op.attrs.get("pred", "olt")
        a, b = op.operands
        self._current().emit(
            f"  {op.result.name} = fcmp {pred} {_llvm_type(a.type)} "
            f"{self._operand(a)}, {self._operand(b)}"
        )

    def _op_arith_fmax(self, op: Op) -> None:
        a, b = op.operands
        self._need("maxnum.f32")
        self._current().emit(
            f"  {op.result.name} = call float @llvm.maxnum.f32(float {self._operand(a)}, float {self._operand(b)})"
        )

    def _op_arith_fmin(self, op: Op) -> None:
        a, b = op.operands
        self._need("minnum.f32")
        self._current().emit(
            f"  {op.result.name} = call float @llvm.minnum.f32(float {self._operand(a)}, float {self._operand(b)})"
        )

    def _op_arith_select(self, op: Op) -> None:
        cond, lhs, rhs = op.operands
        self._current().emit(
            f"  {op.result.name} = select i1 {self._operand(cond)}, "
            f"{_llvm_type(lhs.type)} {self._operand(lhs)}, "
            f"{_llvm_type(rhs.type)} {self._operand(rhs)}"
        )

    def _op_arith_and(self, op: Op) -> None:
        a, b = op.operands
        self._current().emit(
            f"  {op.result.name} = and {_llvm_type(a.type)} "
            f"{self._operand(a)}, {self._operand(b)}"
        )

    def _op_arith_or(self, op: Op) -> None:
        a, b = op.operands
        self._current().emit(
            f"  {op.result.name} = or {_llvm_type(a.type)} "
            f"{self._operand(a)}, {self._operand(b)}"
        )

    def _op_arith_not(self, op: Op) -> None:
        (a,) = op.operands
        ty = _llvm_type(a.type)
        if a.type.name == "i1":
            mask = "true"
        else:
            mask = "-1"
        self._current().emit(
            f"  {op.result.name} = xor {ty} {self._operand(a)}, {mask}"
        )

    def _op_arith_zext(self, op: Op) -> None:
        (v,) = op.operands
        self._current().emit(
            f"  {op.result.name} = zext {_llvm_type(v.type)} {self._operand(v)} "
            f"to {_llvm_type(op.result.type)}"
        )

    def _op_arith_sext(self, op: Op) -> None:
        (v,) = op.operands
        self._current().emit(
            f"  {op.result.name} = sext {_llvm_type(v.type)} {self._operand(v)} "
            f"to {_llvm_type(op.result.type)}"
        )

    def _op_arith_trunc_f32_to_f16(self, op: Op) -> None:
        (v,) = op.operands
        self._current().emit(
            f"  {op.result.name} = fptrunc float {self._operand(v)} to half"
        )

    def _op_arith_cast_to_f32(self, op: Op) -> None:
        (v,) = op.operands
        self._current().emit(
            f"  {op.result.name} = fpext {_llvm_type(v.type)} {self._operand(v)} to float"
        )

    def _op_arith_cast_f32_to(self, op: Op) -> None:
        (v,) = op.operands
        self._current().emit(
            f"  {op.result.name} = fptrunc float {self._operand(v)} to {_llvm_type(op.result.type)}"
        )

    def _op_arith_sitofp_f32(self, op: Op) -> None:
        (v,) = op.operands
        if v.type.name != "i32":
            raise NotImplementedError(
                f"arith.sitofp_f32 supports i32 input only, got {v.type.name}"
            )
        self._current().emit(
            f"  {op.result.name} = sitofp i32 {self._operand(v)} to float"
        )

    def _op_arith_cvt_fp8_to_f32(self, op: Op) -> None:
        """Lower fp8e4m3->f32 conversion to AMDGPU intrinsic.

        AMDGPU exposes a per-byte-lane FP8 conversion `llvm.amdgcn.cvt.f32.fp8`
        that takes an `i32` containing 4 packed fp8 elements plus a lane
        index (0..3) and returns the f32 value at that byte. We pack the
        single fp8e4m3 operand into the low byte of an `i32` and select
        lane 0. The full packed-vec dequant variant goes through
        `_op_arith_cvt_fp8x4_to_f32x4` so that we issue one intrinsic call
        per dword rather than one per element.
        """
        (v,) = op.operands
        self._need("amdgcn.cvt.f32.fp8")
        tmp = f"{op.result.name}x"
        self._current().emit(f"  {tmp} = zext i8 {self._operand(v)} to i32")
        self._current().emit(
            f"  {op.result.name} = call float @llvm.amdgcn.cvt.f32.fp8(i32 {tmp}, i32 0)"
        )

    def _op_math_exp2(self, op: Op) -> None:
        (v,) = op.operands
        if v.type.name != "f32":
            raise NotImplementedError("math.exp2 currently supports f32")
        self._need("exp2.f32")
        self._current().emit(
            f"  {op.result.name} = call float @llvm.exp2.f32(float {self._operand(v)})"
        )

    def _op_math_rcp(self, op: Op) -> None:
        (v,) = op.operands
        one = _fp32_hex(1.0) if v.type.name == "f32" else "1.000000e+00"
        self._current().emit(
            f"  {op.result.name} = fdiv {_llvm_type(v.type)} {one}, {self._operand(v)}"
        )

    def _op_math_sqrt(self, op: Op) -> None:
        (v,) = op.operands
        if v.type.name != "f32":
            raise NotImplementedError("math.sqrt currently supports f32")
        self._need("sqrt.f32")
        self._current().emit(
            f"  {op.result.name} = call float @llvm.sqrt.f32(float {self._operand(v)})"
        )

    def _op_math_rsqrt(self, op: Op) -> None:
        (v,) = op.operands
        if v.type.name != "f32":
            raise NotImplementedError("math.rsqrt currently supports f32")
        # llvm.amdgcn.rsq.f32 maps directly to v_rsq_f32 (~1 ulp). For higher precision
        # the user can compute rcp(sqrt(x)) explicitly. This matches Triton's tl.rsqrt.
        self._need("rsqrt.f32")
        self._current().emit(
            f"  {op.result.name} = call float @llvm.amdgcn.rsq.f32(float {self._operand(v)})"
        )

    def _op_math_tanh(self, op: Op) -> None:
        (v,) = op.operands
        if v.type.name != "f32":
            raise NotImplementedError("math.tanh currently supports f32")
        self._need("tanh.f32")
        self._current().emit(
            f"  {op.result.name} = call float @llvm.tanh.f32(float {self._operand(v)})"
        )

    # gpu

    def _op_gpu_thread_id(self, op: Op) -> None:
        axis = op.attrs.get("axis", "x")
        self._need(f"workitem.{axis}")
        self._current().emit(
            f"  {op.result.name} = call i32 @llvm.amdgcn.workitem.id.{axis}()"
        )

    def _op_gpu_block_id(self, op: Op) -> None:
        axis = op.attrs.get("axis", "x")
        self._need(f"workgroup.{axis}")
        self._current().emit(
            f"  {op.result.name} = call i32 @llvm.amdgcn.workgroup.id.{axis}()"
        )

    # memory

    def _op_tile_smem_alloc(self, op: Op) -> None:
        # Module-level global emitted at finalize time; nothing inline.
        return

    def _smem_global_name(self, smem_value: Value) -> Tuple[str, SmemType]:
        name = self._smem_storage_name[smem_value.name]
        return name, smem_value.type  # type: ignore[return-value]

    def _op_memref_global_load(self, op: Op) -> None:
        ptr, idx = op.operands
        gep = self._fresh("gep")
        align = int(op.attrs.get("align", 2))
        self._current().emit(
            f"  {gep} = getelementptr inbounds half, ptr addrspace(1) "
            f"{self._operand(ptr)}, i32 {self._operand(idx)}"
        )
        self._current().emit(
            f"  {op.result.name} = load half, ptr addrspace(1) {gep}, align {align}"
        )

    def _op_memref_global_load_typed(self, op: Op) -> None:
        ptr, idx = op.operands
        elem_ty = _llvm_type(op.result.type)
        gep = self._fresh("gep")
        align = int(op.attrs.get("align", 1))
        self._current().emit(
            f"  {gep} = getelementptr inbounds {elem_ty}, ptr addrspace(1) "
            f"{self._operand(ptr)}, i32 {self._operand(idx)}"
        )
        self._current().emit(
            f"  {op.result.name} = load {elem_ty}, ptr addrspace(1) {gep}, align {align}"
        )

    def _op_memref_global_store(self, op: Op) -> None:
        ptr, idx, val = op.operands
        gep = self._fresh("gep")
        align = int(op.attrs.get("align", 2))
        self._current().emit(
            f"  {gep} = getelementptr inbounds half, ptr addrspace(1) "
            f"{self._operand(ptr)}, i32 {self._operand(idx)}"
        )
        self._current().emit(
            f"  store half {self._operand(val)}, ptr addrspace(1) {gep}, align {align}"
        )

    def _op_memref_global_store_typed(self, op: Op) -> None:
        ptr, idx, val = op.operands
        elem_ty = _llvm_type(val.type)
        gep = self._fresh("gep")
        align = int(op.attrs.get("align", 1))
        self._current().emit(
            f"  {gep} = getelementptr inbounds {elem_ty}, ptr addrspace(1) "
            f"{self._operand(ptr)}, i32 {self._operand(idx)}"
        )
        self._current().emit(
            f"  store {elem_ty} {self._operand(val)}, ptr addrspace(1) {gep}, align {align}"
        )

    def _op_memref_global_load_vN(self, op: Op) -> None:
        """Vectorised <vec x 16-bit> load: a single naturally-aligned
        global_load_dwordx{1,2,4} on AMDGPU when the address is aligned."""
        ptr, idx = op.operands
        vec = int(op.attrs["vec"])
        elem_ty = _llvm_type(op.result.type.elem)  # type: ignore[attr-defined]
        gep = self._fresh("gep")
        # Cast the index-into-half offset into a pointer-cast: we GEP by
        # half, then bitcast to ptr to <vec x half>. This is exactly the
        # pattern Clang emits for `*(__fp16x4_t*)(ptr + idx)`.
        self._current().emit(
            f"  {gep} = getelementptr inbounds {elem_ty}, ptr addrspace(1) "
            f"{self._operand(ptr)}, i32 {self._operand(idx)}"
        )
        align = int(op.attrs.get("align", vec * 2))
        self._current().emit(
            f"  {op.result.name} = load <{vec} x {elem_ty}>, ptr addrspace(1) {gep}, "
            f"align {align}"
        )

    def _op_tile_smem_store(self, op: Op) -> None:
        smem = op.operands[0]
        indices = op.operands[1:-1]
        value = op.operands[-1]
        gname, stype = self._smem_global_name(smem)
        gep = self._fresh("gep")
        gidx = ["i32 0"] + [f"i32 {self._operand(i)}" for i in indices]
        agg_ty = _smem_storage_type(stype)
        self._current().emit(
            f"  {gep} = getelementptr inbounds {agg_ty}, ptr addrspace(3) {gname}, "
            f"{', '.join(gidx)}"
        )
        self._current().emit(
            f"  store {_llvm_type(value.type)} {self._operand(value)}, ptr addrspace(3) {gep}, align 2"
        )

    def _op_tile_smem_store_vN(self, op: Op) -> None:
        """Vectorised LDS store: stores <vec x half> at the given index.

        Lowers to `store <vec x half>, ptr addrspace(3) %p, align (vec*2)`
        which the AMDGPU backend turns into ds_write_b{32,64,128}.
        """
        smem = op.operands[0]
        indices = op.operands[1:-1]
        value = op.operands[-1]
        vec = int(op.attrs["vec"])
        gname, stype = self._smem_global_name(smem)
        agg_ty = _smem_storage_type(stype)
        gep = self._fresh("gep")
        gidx = ["i32 0"] + [f"i32 {self._operand(i)}" for i in indices]
        self._current().emit(
            f"  {gep} = getelementptr inbounds {agg_ty}, ptr addrspace(3) {gname}, "
            f"{', '.join(gidx)}"
        )
        align = int(op.attrs.get("align", vec * 2))
        elem_ty = _llvm_type(value.type.elem)  # type: ignore[attr-defined]
        self._current().emit(
            f"  store <{vec} x {elem_ty}> {self._operand(value)}, ptr addrspace(3) {gep}, "
            f"align {align}"
        )

    def _op_tile_smem_load_v4(self, op: Op) -> None:
        smem, row, col = op.operands
        gname, stype = self._smem_global_name(smem)
        agg_ty = _smem_storage_type(stype)
        base = self._fresh("smem.base")
        self._current().emit(
            f"  {base} = getelementptr inbounds {agg_ty}, ptr addrspace(3) {gname}, "
            f"i32 0, i32 {self._operand(row)}, i32 {self._operand(col)}"
        )
        # 4 contiguous fp16 loads + insertelement chain. We do separate
        # loads (not a single <4 x half> load) so we don't have to make
        # alignment claims we can't always honour. clang -O3 will fuse
        # them when alignment permits.
        elems = []
        for i in range(4):
            ep = self._fresh("smem.ep")
            self._current().emit(
                f"  {ep} = getelementptr inbounds half, ptr addrspace(3) {base}, i32 {i}"
            )
            ld = self._fresh("smem.ld")
            self._current().emit(f"  {ld} = load half, ptr addrspace(3) {ep}, align 2")
            elems.append(ld)
        prev = "undef"
        for i, e in enumerate(elems):
            tmp = op.result.name if i == 3 else self._fresh("vec")
            self._current().emit(
                f"  {tmp} = insertelement <4 x half> {prev}, half {e}, i32 {i}"
            )
            prev = tmp

    def _op_tile_smem_load_vN(self, op: Op) -> None:
        """Vector LDS load. Emits a single naturally-aligned vector
        load; the AMDGPU backend turns aligned `<vec x half>` loads
        into `ds_read_b{16,32,64,128}`.

        For `vec=1` we still return `<1 x half>` (a one-element vector) so
        callers consistently see the same type; LLVM folds it to scalar.
        """
        smem = op.operands[0]
        indices = list(op.operands[1:])
        vec = int(op.attrs["vec"])
        gname, stype = self._smem_global_name(smem)
        agg_ty = _smem_storage_type(stype)
        base = self._fresh("smem.base")
        idx_strs = ["i32 0"] + [f"i32 {self._operand(i)}" for i in indices]
        self._current().emit(
            f"  {base} = getelementptr inbounds {agg_ty}, ptr addrspace(3) {gname}, "
            f"{', '.join(idx_strs)}"
        )
        align = vec * 2
        elem_ty = _llvm_type(op.result.type.elem)  # type: ignore[attr-defined]
        if vec == 1:
            scalar = self._fresh("smem.s")
            self._current().emit(
                f"  {scalar} = load {elem_ty}, ptr addrspace(3) {base}, align {align}"
            )
            self._current().emit(
                f"  {op.result.name} = insertelement <1 x {elem_ty}> undef, {elem_ty} {scalar}, i32 0"
            )
        else:
            self._current().emit(
                f"  {op.result.name} = load <{vec} x {elem_ty}>, ptr addrspace(3) {base}, "
                f"align {align}"
            )

    def _op_tile_mfma_f32_16x16x16_f16(self, op: Op) -> None:
        a, b, c = op.operands
        self._need("mfma.f32.16x16x16f16")
        self._current().emit(
            f"  {op.result.name} = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16("
            f"<4 x half> {self._operand(a)}, "
            f"<4 x half> {self._operand(b)}, "
            f"<4 x float> {self._operand(c)}, "
            f"i32 0, i32 0, i32 0)"
        )

    def _op_tile_mfma_f32_16x16x32_f16(self, op: Op) -> None:
        a, b, c = op.operands
        self._need("mfma.f32.16x16x32.f16")
        self._current().emit(
            f"  {op.result.name} = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16("
            f"<8 x half> {self._operand(a)}, "
            f"<8 x half> {self._operand(b)}, "
            f"<4 x float> {self._operand(c)}, "
            f"i32 0, i32 0, i32 0)"
        )

    def _op_tile_mfma_f32_16x16x16_bf16(self, op: Op) -> None:
        a, b, c = op.operands
        self._need("mfma.f32.16x16x16bf16.1k")
        # bitcast <4 x bfloat> -> <4 x i16> for the `_1k` intrinsic.
        a_cast = self._fresh("mfma_a_i16")
        b_cast = self._fresh("mfma_b_i16")
        self._current().emit(
            f"  {a_cast} = bitcast <4 x bfloat> {self._operand(a)} to <4 x i16>"
        )
        self._current().emit(
            f"  {b_cast} = bitcast <4 x bfloat> {self._operand(b)} to <4 x i16>"
        )
        self._current().emit(
            f"  {op.result.name} = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k("
            f"<4 x i16> {a_cast}, "
            f"<4 x i16> {b_cast}, "
            f"<4 x float> {self._operand(c)}, "
            f"i32 0, i32 0, i32 0)"
        )

    def _op_tile_mfma_f32_16x16x32_bf16(self, op: Op) -> None:
        a, b, c = op.operands
        self._need("mfma.f32.16x16x32.bf16")
        self._current().emit(
            f"  {op.result.name} = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16("
            f"<8 x bfloat> {self._operand(a)}, "
            f"<8 x bfloat> {self._operand(b)}, "
            f"<4 x float> {self._operand(c)}, "
            f"i32 0, i32 0, i32 0)"
        )

    def _op_tile_mfma_f32_32x32x8_f16(self, op: Op) -> None:
        a, b, c = op.operands
        self._need("mfma.f32.32x32x8f16")
        self._current().emit(
            f"  {op.result.name} = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16("
            f"<4 x half> {self._operand(a)}, "
            f"<4 x half> {self._operand(b)}, "
            f"<16 x float> {self._operand(c)}, "
            f"i32 0, i32 0, i32 0)"
        )

    def _op_tile_mfma_f32_32x32x16_f16(self, op: Op) -> None:
        a, b, c = op.operands
        self._need("mfma.f32.32x32x16.f16")
        self._current().emit(
            f"  {op.result.name} = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16("
            f"<8 x half> {self._operand(a)}, "
            f"<8 x half> {self._operand(b)}, "
            f"<16 x float> {self._operand(c)}, "
            f"i32 0, i32 0, i32 0)"
        )

    def _op_tile_mfma_f32_4x4x4_f16(self, op: Op) -> None:
        a, b, c = op.operands
        self._need("mfma.f32.4x4x4f16")
        self._current().emit(
            f"  {op.result.name} = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x4f16("
            f"<4 x half> {self._operand(a)}, "
            f"<4 x half> {self._operand(b)}, "
            f"<4 x float> {self._operand(c)}, "
            f"i32 0, i32 0, i32 0)"
        )

    def _op_vector_bitcast(self, op: Op) -> None:
        (v,) = op.operands
        target = op.result.type
        self._current().emit(
            f"  {op.result.name} = bitcast {_llvm_type(v.type)} {self._operand(v)} "
            f"to {_llvm_type(target)}"
        )

    def _op_tile_readfirstlane(self, op: Op) -> None:
        (v,) = op.operands
        intrinsic_key, ty = {
            "i32": ("readfirstlane.i32", "i32"),
            "i64": ("readfirstlane.i64", "i64"),
        }[v.type.name]
        self._need(intrinsic_key)
        self._current().emit(
            f"  {op.result.name} = call {ty} @llvm.amdgcn.readfirstlane.{ty}({ty} {self._operand(v)})"
        )

    def _op_tile_ds_bpermute(self, op: Op) -> None:
        addr, data = op.operands
        self._need("ds.bpermute")
        self._current().emit(
            f"  {op.result.name} = call i32 @llvm.amdgcn.ds.bpermute("
            f"i32 {self._operand(addr)}, i32 {self._operand(data)})"
        )

    def _op_tile_ds_read_tr16_b64(self, op: Op) -> None:
        """`ds_read_b64_tr_b16` -- gfx950 transpose-read of a 16x16 fp16 tile.

        Returns `<4 x half>` per lane (MFMA B-operand layout for 16x16x16).
        Argument is the LDS address of the tile's [0,0] element.
        """
        smem = op.operands[0]
        indices = list(op.operands[1:])
        gname, stype = self._smem_global_name(smem)
        agg_ty = _smem_storage_type(stype)
        base = self._fresh("tr.base")
        idx_strs = ["i32 0"] + [f"i32 {self._operand(i)}" for i in indices]
        self._current().emit(
            f"  {base} = getelementptr inbounds {agg_ty}, ptr addrspace(3) {gname}, "
            f"{', '.join(idx_strs)}"
        )
        self._need("ds.read.tr16.b64")
        raw = self._fresh("tr.raw")
        self._current().emit(
            f"  {raw} = call <4 x i16> @llvm.amdgcn.ds.read.tr16.b64("
            f"ptr addrspace(3) {base})"
        )
        elem_ty = _llvm_type(op.result.type.elem)  # type: ignore[attr-defined]
        self._current().emit(
            f"  {op.result.name} = bitcast <4 x i16> {raw} to <4 x {elem_ty}>"
        )

    def _op_tile_lane_id(self, op: Op) -> None:
        """Build lane id from mbcnt: lane = mbcnt.hi(-1, mbcnt.lo(-1, 0))."""
        self._need("mbcnt.lo")
        self._need("mbcnt.hi")
        lo = self._fresh("mbcnt.lo")
        self._current().emit(f"  {lo} = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)")
        self._current().emit(
            f"  {op.result.name} = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 {lo})"
        )

    def _op_arith_bitcast(self, op: Op) -> None:
        (v,) = op.operands
        self._current().emit(
            f"  {op.result.name} = bitcast {_llvm_type(v.type)} {self._operand(v)} "
            f"to {_llvm_type(op.result.type)}"
        )

    def _op_arith_xor(self, op: Op) -> None:
        a, b = op.operands
        self._current().emit(
            f"  {op.result.name} = xor {_llvm_type(op.result.type)} "
            f"{self._operand(a)}, {self._operand(b)}"
        )

    def _op_arith_shl(self, op: Op) -> None:
        a, b = op.operands
        self._current().emit(
            f"  {op.result.name} = shl {_llvm_type(op.result.type)} "
            f"{self._operand(a)}, {self._operand(b)}"
        )

    def _op_tile_smem_addr_of(self, op: Op) -> None:
        (smem,) = op.operands
        gname = self._smem_storage_name[smem.name]
        # The global is ptr addrspace(3); cast to i64 for arithmetic.
        self._current().emit(
            f"  {op.result.name} = ptrtoint ptr addrspace(3) {gname} to i64"
        )

    def _op_tile_smem_ptr_add(self, op: Op) -> None:
        base, off = op.operands
        self._current().emit(
            f"  {op.result.name} = add i64 {self._operand(base)}, {self._operand(off)}"
        )

    def _op_tile_async_buffer_load_lds_addr(self, op: Op) -> None:
        rsrc, lds_addr, voff, soff = op.operands
        dwords = int(op.attrs["dwords"])
        size_bytes = dwords * 4
        self._need("raw.ptr.buffer.load.lds")
        # Convert the i64 LDS address back to ptr addrspace(3).
        ptr_name = self._fresh("lds_ptr")
        self._current().emit(
            f"  {ptr_name} = inttoptr i64 {self._operand(lds_addr)} to ptr addrspace(3)"
        )
        self._current().emit(
            f"  call void @llvm.amdgcn.raw.ptr.buffer.load.lds("
            f"ptr addrspace(8) {self._operand(rsrc)}, "
            f"ptr addrspace(3) {ptr_name}, "
            f"i32 {size_bytes}, "
            f"i32 {self._operand(voff)}, "
            f"i32 {self._operand(soff)}, "
            f"i32 0, i32 0)"
        )

    def _op_tile_sync(self, op: Op) -> None:
        # AMDGPU's ``s_barrier`` only synchronises waves -- it does NOT
        # wait for outstanding ``ds_write`` / ``ds_read`` instructions
        # to drain. Without an explicit ``s_waitcnt lgkmcnt(0)`` the
        # post-barrier readers can observe stale LDS contents (see the
        # transpose2d 24x24 grid failure that surfaced this). We also
        # drop ``vmcnt(0)`` so a ds_write whose source data came from a
        # global load completes its VMEM-to-VGPR-to-LDS chain before the
        # barrier; that matches what CK Tile's ``block_sync_lds`` does.
        # ``_encode_waitcnt_gfx9_10(vmcnt=0, lgkmcnt=0)`` evaluates to
        # ``0x70`` (= 112) -- ``vmcnt(0) lgkmcnt(0) expcnt(<max>)``.
        mask = _encode_waitcnt_gfx9_10(vmcnt=0, expcnt=-1, lgkmcnt=0)
        self._need("s.waitcnt")
        self._need("s.barrier")
        self._current().emit(f"  call void @llvm.amdgcn.s.waitcnt(i32 {mask})")
        self._current().emit("  call void @llvm.amdgcn.s.barrier()")

    def _op_tile_sync_lds_only(self, op: Op) -> None:
        # Workgroup barrier that drains LDS (lgkmcnt) but NOT VMEM (vmcnt).
        # Used by the async-DMA ping-pong pipeline: an outstanding
        # raw_ptr_buffer_load_lds (VMEM) for the *next* iter must keep
        # streaming while we wait on the *previous* iter's ds_reads.
        # Draining vmcnt here would defeat the whole point of the
        # overlap. Matches CK Tile's ``block_sync_lds``.
        mask = _encode_waitcnt_gfx9_10(vmcnt=-1, expcnt=-1, lgkmcnt=0)
        self._need("s.waitcnt")
        self._need("s.barrier")
        self._current().emit(f"  call void @llvm.amdgcn.s.waitcnt(i32 {mask})")
        self._current().emit("  call void @llvm.amdgcn.s.barrier()")

    def _op_tile_s_waitcnt(self, op: Op) -> None:
        # See ck_dsl/_ir.py:s_waitcnt for the encoding contract.
        self._need("s.waitcnt")
        vm = int(op.attrs.get("vmcnt", -1))
        lk = int(op.attrs.get("lgkmcnt", -1))
        ec = int(op.attrs.get("expcnt", -1))
        mask = _encode_waitcnt_gfx9_10(vm, ec, lk)
        self._current().emit(f"  call void @llvm.amdgcn.s.waitcnt(i32 {mask})")

    def _op_tile_sched_barrier(self, op: Op) -> None:
        self._need("sched.barrier")
        mask = int(op.attrs.get("mask", 0))
        self._current().emit(f"  call void @llvm.amdgcn.sched.barrier(i32 {mask})")

    def _op_tile_sched_group_barrier(self, op: Op) -> None:
        self._need("sched.group.barrier")
        mask = int(op.attrs["mask"])
        count = int(op.attrs["count"])
        group = int(op.attrs.get("group", 0))
        self._current().emit(
            f"  call void @llvm.amdgcn.sched.group.barrier("
            f"i32 {mask}, i32 {count}, i32 {group})"
        )

    def _op_tile_s_setprio(self, op: Op) -> None:
        self._need("s.setprio")
        level = int(op.attrs["level"])
        self._current().emit(f"  call void @llvm.amdgcn.s.setprio(i16 {level})")

    # ----- buffer-rsrc + async DRAM->LDS -----

    def _op_tile_buffer_rsrc(self, op: Op) -> None:
        """Build a buffer resource descriptor for a global pointer.

        Modern LLVM exposes `@llvm.amdgcn.make.buffer.rsrc.p1` which
        returns a `ptr addrspace(8)`. We keep our IR-level type as
        `<4 x i32>` for self-documenting printing, but the underlying
        LLVM value is the addrspace(8) ptr; the buffer_load lowerings
        below consume it as the addrspace(8) pointer directly via an
        inttoptr-free path.

        Flags = 0x00027000 (matches CK Tile's
        `__builtin_amdgcn_make_buffer_rsrc(p, 0, bytes, 0x00027000)`
        in `cktile_fixed_lean_kernel.hpp`). The flag word encodes the
        rsrc DWORD3 -- TYPE=2 (BUFFER_RESOURCE), DATA_FORMAT=4
        (32-bit dword), NUM_FORMAT=4 (UINT). Without these flags the
        AMDGPU compiler can lower buffer loads to "unbounded" loads
        (a single load_dword without bounds check) which then
        misreads padded boundary positions as the next row of A.
        """
        self._need("make.buffer.rsrc.p1")
        ptr, num_bytes = op.operands
        # CK Tile uses 0x00027000 — the buffer rsrc DWORD3 flag word
        # that gives "32-bit-uint, structured buffer, bounds-checked".
        self._current().emit(
            f"  {op.result.name} = call ptr addrspace(8) "
            f"@llvm.amdgcn.make.buffer.rsrc.p1("
            f"ptr addrspace(1) {self._operand(ptr)}, "
            f"i16 0, "
            f"i32 {self._operand(num_bytes)}, "
            f"i32 159744)"  # 0x00027000
        )

    def _op_tile_buffer_load_vN_f16(self, op: Op) -> None:
        """raw_ptr_buffer_load returning <dwords x i32>, bitcast to
        <2*dwords x half>. Bounds-checked: out-of-range voffset
        returns 0."""
        rsrc, voffset, soffset = op.operands
        dwords = int(op.attrs["dwords"])
        # Choose the right intrinsic variant.
        if dwords == 1:
            self._need("raw.ptr.buffer.load.i32")
            tmp = self._fresh("bli32")
            self._current().emit(
                f"  {tmp} = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32("
                f"ptr addrspace(8) {self._operand(rsrc)}, "
                f"i32 {self._operand(voffset)}, "
                f"i32 {self._operand(soffset)}, "
                f"i32 0)"
            )
            self._current().emit(
                f"  {op.result.name} = bitcast i32 {tmp} to <2 x half>"
            )
        else:
            intr = f"raw.ptr.buffer.load.v{dwords}i32"
            self._need(intr)
            tmp = self._fresh(f"blv{dwords}")
            self._current().emit(
                f"  {tmp} = call <{dwords} x i32> @llvm.amdgcn.raw.ptr.buffer.load.v{dwords}i32("
                f"ptr addrspace(8) {self._operand(rsrc)}, "
                f"i32 {self._operand(voffset)}, "
                f"i32 {self._operand(soffset)}, "
                f"i32 0)"
            )
            halves = dwords * 2
            self._current().emit(
                f"  {op.result.name} = bitcast <{dwords} x i32> {tmp} to <{halves} x half>"
            )

    def _op_tile_buffer_load_f16(self, op: Op) -> None:
        """Scalar half buffer load via `raw_ptr_buffer_load_u16` -> trunc i16
        -> bitcast to half. The amdgpu intrinsic for a u16 element load
        returns i32 (the low 16 bits hold the half); we trunc to i16 and
        bitcast to half. OOB returns 0 (clamped by the rsrc bounds)."""
        rsrc, voffset, soffset = op.operands
        self._need("raw.ptr.buffer.load.i32")
        tmp = self._fresh("blu16")
        self._current().emit(
            f"  {tmp} = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32("
            f"ptr addrspace(8) {self._operand(rsrc)}, "
            f"i32 {self._operand(voffset)}, "
            f"i32 {self._operand(soffset)}, "
            f"i32 0)"
        )
        trunc = self._fresh("trunc16")
        self._current().emit(f"  {trunc} = trunc i32 {tmp} to i16")
        self._current().emit(f"  {op.result.name} = bitcast i16 {trunc} to half")

    def _op_tile_buffer_store_vN_f16(self, op: Op) -> None:
        """raw_ptr_buffer_store of <2*dwords x half> via bitcast to
        <dwords x i32>. OOB voffsets are silently dropped (the rsrc
        bounds-check provides the runbook §6.2 tail safety)."""
        rsrc, voffset, soffset, val = op.operands
        dwords = int(op.attrs["dwords"])
        halves = dwords * 2
        if dwords == 1:
            self._need("raw.ptr.buffer.store.i32")
            bc = self._fresh("bsbc")
            self._current().emit(
                f"  {bc} = bitcast <2 x half> {self._operand(val)} to i32"
            )
            self._current().emit(
                f"  call void @llvm.amdgcn.raw.ptr.buffer.store.i32("
                f"i32 {bc}, "
                f"ptr addrspace(8) {self._operand(rsrc)}, "
                f"i32 {self._operand(voffset)}, "
                f"i32 {self._operand(soffset)}, "
                f"i32 0)"
            )
        else:
            intr = f"raw.ptr.buffer.store.v{dwords}i32"
            self._need(intr)
            bc = self._fresh("bsbc")
            self._current().emit(
                f"  {bc} = bitcast <{halves} x half> {self._operand(val)} to <{dwords} x i32>"
            )
            self._current().emit(
                f"  call void @llvm.amdgcn.raw.ptr.buffer.store.v{dwords}i32("
                f"<{dwords} x i32> {bc}, "
                f"ptr addrspace(8) {self._operand(rsrc)}, "
                f"i32 {self._operand(voffset)}, "
                f"i32 {self._operand(soffset)}, "
                f"i32 0)"
            )

    def _op_tile_buffer_store_f16(self, op: Op) -> None:
        """Single-half buffer store via i16 intrinsic. OOB drop."""
        rsrc, voffset, soffset, val = op.operands
        self._need("raw.ptr.buffer.store.i16")
        bc = self._fresh("bs1")
        self._current().emit(f"  {bc} = bitcast half {self._operand(val)} to i16")
        self._current().emit(
            f"  call void @llvm.amdgcn.raw.ptr.buffer.store.i16("
            f"i16 {bc}, "
            f"ptr addrspace(8) {self._operand(rsrc)}, "
            f"i32 {self._operand(voffset)}, "
            f"i32 {self._operand(soffset)}, "
            f"i32 0)"
        )

    def _op_tile_async_buffer_load_lds(self, op: Op) -> None:
        rsrc, lds_ptr, voffset, soffset = op.operands
        dwords = int(op.attrs["dwords"])
        bytes_per_lane = dwords * 4
        self._need("raw.ptr.buffer.load.lds")
        self._current().emit(
            f"  call void @llvm.amdgcn.raw.ptr.buffer.load.lds("
            f"ptr addrspace(8) {self._operand(rsrc)}, "
            f"ptr addrspace(3) {self._operand(lds_ptr)}, "
            f"i32 {bytes_per_lane}, "
            f"i32 {self._operand(voffset)}, "
            f"i32 {self._operand(soffset)}, "
            f"i32 0, "
            f"i32 0)"
        )

    # ----- f32 LDS ops (cshuffle epilogue) -----

    def _op_tile_smem_store_vN_f32(self, op: Op) -> None:
        smem = op.operands[0]
        indices = op.operands[1:-1]
        value = op.operands[-1]
        vec = int(op.attrs["vec"])
        gname, stype = self._smem_global_name(smem)
        agg_ty = _smem_storage_type(stype)
        gep = self._fresh("gep")
        gidx = ["i32 0"] + [f"i32 {self._operand(i)}" for i in indices]
        self._current().emit(
            f"  {gep} = getelementptr inbounds {agg_ty}, ptr addrspace(3) {gname}, "
            f"{', '.join(gidx)}"
        )
        align = vec * 4
        # vec=1 stores expect the scalar form regardless of how the value was typed.
        if vec == 1:
            # value may be a scalar `float` (from `vec_extract`) or `<1 x float>`.
            if isinstance(value.type, VectorType):
                # Extract scalar first.
                ext = self._fresh("v1ext")
                self._current().emit(
                    f"  {ext} = extractelement {_llvm_type(value.type)} {self._operand(value)}, i32 0"
                )
                self._current().emit(
                    f"  store float {ext}, ptr addrspace(3) {gep}, align {align}"
                )
            else:
                self._current().emit(
                    f"  store float {self._operand(value)}, ptr addrspace(3) {gep}, align {align}"
                )
        else:
            self._current().emit(
                f"  store <{vec} x float> {self._operand(value)}, ptr addrspace(3) {gep}, align {align}"
            )

    def _op_tile_smem_load_vN_f32(self, op: Op) -> None:
        smem = op.operands[0]
        indices = list(op.operands[1:])
        vec = int(op.attrs["vec"])
        gname, stype = self._smem_global_name(smem)
        agg_ty = _smem_storage_type(stype)
        base = self._fresh("smem.base")
        idx_strs = ["i32 0"] + [f"i32 {self._operand(i)}" for i in indices]
        self._current().emit(
            f"  {base} = getelementptr inbounds {agg_ty}, ptr addrspace(3) {gname}, "
            f"{', '.join(idx_strs)}"
        )
        align = vec * 4
        if vec == 1:
            scalar = self._fresh("smem.s")
            self._current().emit(
                f"  {scalar} = load float, ptr addrspace(3) {base}, align {align}"
            )
            self._current().emit(
                f"  {op.result.name} = insertelement <1 x float> undef, float {scalar}, i32 0"
            )
        else:
            self._current().emit(
                f"  {op.result.name} = load <{vec} x float>, ptr addrspace(3) {base}, align {align}"
            )

    # ----- packed f32->f16 + wide global store -----

    def _op_vector_trunc_f32_to_f16(self, op: Op) -> None:
        self._op_vector_trunc_f32_to(op)

    def _op_vector_trunc_f32_to(self, op: Op) -> None:
        (v,) = op.operands
        in_ty = _llvm_type(v.type)
        out_ty = _llvm_type(op.result.type)
        self._current().emit(
            f"  {op.result.name} = fptrunc {in_ty} {self._operand(v)} to {out_ty}"
        )

    def _op_memref_global_store_vN(self, op: Op) -> None:
        ptr, idx, val = op.operands
        vec = int(op.attrs["vec"])
        gep = self._fresh("gep")
        elem_ty = (
            _llvm_type(val.type.elem)
            if isinstance(val.type, VectorType)
            else _llvm_type(val.type)
        )
        self._current().emit(
            f"  {gep} = getelementptr inbounds {elem_ty}, ptr addrspace(1) "
            f"{self._operand(ptr)}, i32 {self._operand(idx)}"
        )
        align = vec * 2
        ty = _llvm_type(val.type)
        self._current().emit(
            f"  store {ty} {self._operand(val)}, ptr addrspace(1) {gep}, align {align}"
        )

    def _op_memref_global_atomic_add_f32(self, op: Op) -> None:
        ptr, idx, val = op.operands
        gep = self._fresh("gep")
        self._current().emit(
            f"  {gep} = getelementptr inbounds float, ptr addrspace(1) "
            f"{self._operand(ptr)}, i32 {self._operand(idx)}"
        )
        tmp = self._fresh("a")
        self._current().emit(
            f"  {tmp} = atomicrmw fadd ptr addrspace(1) {gep}, "
            f'float {self._operand(val)} syncscope("agent") monotonic, align 4'
        )

    def _op_vector_extract(self, op: Op) -> None:
        """Element extraction from a vector accumulator.

        Now parametric: derives the vector type from the operand so the
        same op handles `<4 x float>` (16x16 atoms) and `<16 x float>`
        (32x32 atoms) without a special case.
        """
        (v,) = op.operands
        i = op.attrs["index"]
        self._current().emit(
            f"  {op.result.name} = extractelement {_llvm_type(v.type)} "
            f"{self._operand(v)}, i32 {i}"
        )

    def _op_vector_splat(self, op: Op) -> None:
        (scalar,) = op.operands
        vec_ty = _llvm_type(op.result.type)
        elem_ty = _llvm_type(scalar.type)
        prev = "undef"
        count = op.result.type.count  # type: ignore[attr-defined]
        for i in range(count):
            name = op.result.name if i == count - 1 else self._fresh("splat")
            self._current().emit(
                f"  {name} = insertelement {vec_ty} {prev}, {elem_ty} {self._operand(scalar)}, i32 {i}"
            )
            prev = name

    def _op_vector_pack(self, op: Op) -> None:
        """Pack N scalars into `<N x elem>` via insertelement chain."""
        result_ty = op.result.type
        vec_ty = _llvm_type(result_ty)
        elem_ty = _llvm_type(result_ty.elem)  # type: ignore[attr-defined]
        prev = "undef"
        count = result_ty.count  # type: ignore[attr-defined]
        for i, comp in enumerate(op.operands):
            name = op.result.name if i == count - 1 else self._fresh("vpk")
            self._current().emit(
                f"  {name} = insertelement {vec_ty} {prev}, {elem_ty} {self._operand(comp)}, i32 {i}"
            )
            prev = name

    def _op_vector_concat(self, op: Op) -> None:
        """Concatenate two equal-typed vectors into a double-width vector."""
        a, b_op = op.operands
        a_ty = a.type
        b_ty = b_op.type
        a_n = a_ty.count  # type: ignore[attr-defined]
        b_n = b_ty.count  # type: ignore[attr-defined]
        elem_t = a_ty.elem  # type: ignore[attr-defined]
        elem_ll = _llvm_type(elem_t)
        out_ty_ll = _llvm_type(op.result.type)
        prev = "undef"
        for i in range(a_n):
            ext = self._fresh("vc.a")
            self._current().emit(
                f"  {ext} = extractelement {_llvm_type(a_ty)} {self._operand(a)}, i32 {i}"
            )
            nxt = self._fresh("vc.ins")
            self._current().emit(
                f"  {nxt} = insertelement {out_ty_ll} {prev}, {elem_ll} {ext}, i32 {i}"
            )
            prev = nxt
        for i in range(b_n):
            ext = self._fresh("vc.b")
            self._current().emit(
                f"  {ext} = extractelement {_llvm_type(b_ty)} {self._operand(b_op)}, i32 {i}"
            )
            is_last = i == b_n - 1
            nxt = op.result.name if is_last else self._fresh("vc.ins")
            self._current().emit(
                f"  {nxt} = insertelement {out_ty_ll} {prev}, {elem_ll} {ext}, i32 {a_n + i}"
            )
            prev = nxt

    def _op_vector_insert(self, op: Op) -> None:
        v, scalar = op.operands
        elem_ll = _llvm_type(scalar.type)
        idx = int(op.attrs["index"])
        self._current().emit(
            f"  {op.result.name} = insertelement {_llvm_type(v.type)} {self._operand(v)}, "
            f"{elem_ll} {self._operand(scalar)}, i32 {idx}"
        )

    def _vector_binop(self, op: Op, llvm_op: str) -> None:
        a, b = op.operands
        self._current().emit(
            f"  {op.result.name} = {llvm_op} {_llvm_type(a.type)} {self._operand(a)}, {self._operand(b)}"
        )

    def _op_vector_add(self, op: Op) -> None:
        elem = op.result.type.elem.name  # type: ignore[attr-defined]
        self._vector_binop(op, "fadd" if elem in ("f16", "bf16", "f32") else "add")

    def _op_vector_mul(self, op: Op) -> None:
        elem = op.result.type.elem.name  # type: ignore[attr-defined]
        self._vector_binop(op, "fmul" if elem in ("f16", "bf16", "f32") else "mul")

    def _op_vector_max(self, op: Op) -> None:
        a, b = op.operands
        # LLVM has vector maxnum intrinsics with overloaded names, but
        # per-element select keeps this backend-simple and optimizable.
        vec_ty = a.type
        count = vec_ty.count  # type: ignore[attr-defined]
        elem_ty = vec_ty.elem  # type: ignore[attr-defined]
        vals = []
        for i in range(count):
            ea = self._fresh("vmax.a")
            eb = self._fresh("vmax.b")
            cmp = self._fresh("vmax.cmp")
            sel = self._fresh("vmax.sel")
            self._current().emit(
                f"  {ea} = extractelement {_llvm_type(vec_ty)} {self._operand(a)}, i32 {i}"
            )
            self._current().emit(
                f"  {eb} = extractelement {_llvm_type(vec_ty)} {self._operand(b)}, i32 {i}"
            )
            self._current().emit(f"  {cmp} = fcmp ogt {_llvm_type(elem_ty)} {ea}, {eb}")
            self._current().emit(
                f"  {sel} = select i1 {cmp}, {_llvm_type(elem_ty)} {ea}, {_llvm_type(elem_ty)} {eb}"
            )
            vals.append(sel)
        prev = "undef"
        for i, v in enumerate(vals):
            name = op.result.name if i == count - 1 else self._fresh("vmax")
            self._current().emit(
                f"  {name} = insertelement {_llvm_type(vec_ty)} {prev}, {_llvm_type(elem_ty)} {v}, i32 {i}"
            )
            prev = name

    def _op_vector_select(self, op: Op) -> None:
        mask, lhs, rhs = op.operands
        self._current().emit(
            f"  {op.result.name} = select {_llvm_type(mask.type)} {self._operand(mask)}, "
            f"{_llvm_type(lhs.type)} {self._operand(lhs)}, {_llvm_type(rhs.type)} {self._operand(rhs)}"
        )

    def _op_vector_sum(self, op: Op) -> None:
        self._lower_vector_reduce(op, "fadd", "0.000000e+00")

    def _op_vector_reduce_max(self, op: Op) -> None:
        (v,) = op.operands
        vec_ty = v.type
        count = vec_ty.count  # type: ignore[attr-defined]
        elem_ty = vec_ty.elem  # type: ignore[attr-defined]
        acc = None
        for i in range(count):
            e = self._fresh("vred.e")
            self._current().emit(
                f"  {e} = extractelement {_llvm_type(vec_ty)} {self._operand(v)}, i32 {i}"
            )
            if acc is None:
                acc = e
            else:
                cmp = self._fresh("vred.cmp")
                nxt = op.result.name if i == count - 1 else self._fresh("vred.max")
                self._current().emit(
                    f"  {cmp} = fcmp ogt {_llvm_type(elem_ty)} {acc}, {e}"
                )
                self._current().emit(
                    f"  {nxt} = select i1 {cmp}, {_llvm_type(elem_ty)} {acc}, {_llvm_type(elem_ty)} {e}"
                )
                acc = nxt
        if count == 1:
            self._current().emit(
                f"  {op.result.name} = fadd {_llvm_type(elem_ty)} {acc}, 0.000000e+00"
            )

    def _lower_vector_reduce(self, op: Op, llvm_op: str, init: str) -> None:
        (v,) = op.operands
        vec_ty = v.type
        count = vec_ty.count  # type: ignore[attr-defined]
        elem_ty = vec_ty.elem  # type: ignore[attr-defined]
        acc = init
        for i in range(count):
            e = self._fresh("vred.e")
            self._current().emit(
                f"  {e} = extractelement {_llvm_type(vec_ty)} {self._operand(v)}, i32 {i}"
            )
            name = op.result.name if i == count - 1 else self._fresh("vred")
            self._current().emit(
                f"  {name} = {llvm_op} {_llvm_type(elem_ty)} {acc}, {e}"
            )
            acc = name

    # control flow

    def _op_scf_for(self, op: Op) -> None:
        num_iter = int(op.attrs.get("num_iter_args", 0))
        lower, upper, step = op.operands[:3]
        iter_inits = op.operands[3 : 3 + num_iter]
        iter_meta = op.attrs.get("iter_args", [])
        iv_name = op.attrs["iv"]
        iv_ty = _llvm_type(lower.type)

        # Capture the predecessor block (for the from-entry edge of phis).
        pred_block = self._current().label

        # Close current block with unconditional jump to header.
        header = self._new_block("for.header")
        self._blocks[-2].emit(f"  br label %{header.label}")
        self._blocks[-2].terminated = True

        # Emit phi nodes in header (filled in for the latch edge after the
        # body region is lowered).
        header.emit(
            f"  {iv_name} = phi {iv_ty} [ {self._operand(lower)}, %{pred_block} ], "
            f"[ %iv.next.{header.label}, %FOR_LATCH ]"
        )
        iter_phi_lines: List[int] = []
        for meta, init in zip(iter_meta, iter_inits):
            ty = meta["type"]
            ll_ty = _llvm_type_from_name(ty)
            header.emit(
                f"  {meta['name']} = phi {ll_ty} "
                f"[ {self._operand(init)}, %{pred_block} ], "
                f"[ {meta['name']}.next.{header.label}, %FOR_LATCH ]"
            )
            iter_phi_lines.append(len(header.lines) - 1)

        # Condition + branch.
        cmp = self._fresh("cmp")
        header.emit(f"  {cmp} = icmp slt {iv_ty} {iv_name}, {self._operand(upper)}")

        body = self._new_block("for.body")
        # We don't know the exit label yet; defer.
        header.emit(f"  br i1 {cmp}, label %{body.label}, label %FOR_EXIT")
        header.terminated = True

        # Lower body region into body block (and any sub-blocks it spawns).
        self._yield_stack.append([])
        self.lower_region(op.regions[0])

        # Body may have terminated itself if there was a scf.yield;
        # otherwise we expect the last op to be scf.yield emitted into the
        # current block. Branch to latch.
        last_body = self._current()
        latch = self._new_block("for.latch")
        last_body.emit(f"  br label %{latch.label}")
        last_body.terminated = True

        yielded = self._yield_stack.pop()
        if len(yielded) != num_iter:
            raise RuntimeError(
                f"scf.for expected {num_iter} yielded values, got {len(yielded)}"
            )

        iv_next = f"%iv.next.{header.label}"
        latch.emit(f"  {iv_next} = add nsw {iv_ty} {iv_name}, {self._operand(step)}")
        for meta, yld in zip(iter_meta, yielded):
            ll_ty = _llvm_type_from_name(meta["type"])
            latch.emit(
                f"  {meta['name']}.next.{header.label} = bitcast {ll_ty} {yld} to {ll_ty}"
            )
        latch.emit(f"  br label %{header.label}")
        latch.terminated = True

        exit_blk = self._new_block("for.exit")
        # Now back-patch FOR_LATCH and FOR_EXIT placeholders in header.
        for i, line in enumerate(header.lines):
            header.lines[i] = line.replace("%FOR_LATCH", f"%{latch.label}")
            header.lines[i] = header.lines[i].replace("%FOR_EXIT", f"%{exit_blk.label}")

        # Bind the for op's results: in LLVM IR, the header phi values
        # (which include the yielded values from the last latch iteration)
        # are the loop results. We add aliases via bitcast in the exit.
        for meta, result in zip(iter_meta, op.results):
            ll_ty = _llvm_type_from_name(meta["type"])
            exit_blk.emit(
                f"  {result.name} = bitcast {ll_ty} {meta['name']} to {ll_ty}"
            )

    def _op_scf_if(self, op: Op) -> None:
        (cond,) = op.operands
        then_region = op.regions[0]
        cur = self._current()
        then_blk = self._new_block("if.then")
        cur.emit(
            f"  br i1 {self._operand(cond)}, label %{then_blk.label}, label %IF_END"
        )
        cur.terminated = True

        self.lower_region(then_region)
        then_last = self._current()
        end_blk = self._new_block("if.end")
        if not then_last.terminated:
            then_last.emit(f"  br label %{end_blk.label}")
            then_last.terminated = True

        for i, line in enumerate(cur.lines):
            cur.lines[i] = line.replace("%IF_END", f"%{end_blk.label}")

    def _op_scf_yield(self, op: Op) -> None:
        if not self._yield_stack:
            raise RuntimeError("scf.yield without enclosing scf.for")
        self._yield_stack[-1].extend(self._operand(v) for v in op.operands)

    def _op_cf_return(self, op: Op) -> None:
        self._current().emit("  ret void")
        self._current().terminated = True

    # ----- finalize -----

    def finalize(self) -> str:
        # Terminate the entry (now potentially exit) block with ret.
        if not self._current().terminated:
            self._current().emit("  ret void")
            self._current().terminated = True

        out: List[str] = []
        out.append(f'target datalayout = "{_DATALAYOUT}"')
        out.append(f'target triple = "{_TRIPLE}"')
        out.append("")

        # smem globals.
        for gname, stype in self._smem_globals:
            agg = _smem_storage_type(stype)
            out.append(
                f"{gname} = internal unnamed_addr addrspace(3) global {agg} poison, align 4"
            )
        if self._smem_globals:
            out.append("")

        # Intrinsic declarations actually used.
        for key, decl in _INTRINSIC_DECLS.items():
            if self._needs_intrin.get(key):
                out.append(decl)
        if self._needs_intrin:
            out.append("")

        # Function header.
        params = [
            f"{_llvm_type(p.type)}{_param_attrs(p.attrs, p.type)} %{p.name}"
            for p in self.kernel.params
        ]
        out.append(
            f"define amdgpu_kernel void @{self.kernel.name}({', '.join(params)}) #0 {{"
        )

        for blk in self._blocks:
            out.append(f"{blk.label}:")
            out.extend(blk.lines)

        out.append("}")
        out.append("")
        max_wg = self.kernel.max_workgroup_size
        out.append(
            'attributes #0 = { "uniform-work-group-size"="true" '
            f'"amdgpu-flat-work-group-size"="64,{max_wg}" '
            "norecurse nounwind }"
        )
        out.append("")
        return "\n".join(out)


def _llvm_type_from_name(name: str) -> str:
    """Map our IR type-name string (from op.attrs) back to LLVM IR text."""
    if name == "i32":
        return "i32"
    if name == "i64":
        return "i64"
    if name == "i8":
        return "i8"
    if name == "f16":
        return "half"
    if name == "bf16":
        return "bfloat"
    if name == "f32":
        return "float"
    if name == "fp8e4m3":
        return "i8"
    if name.startswith("vec<"):
        # vec<f32x4> / vec<f16x4>
        inner = name[4:-1]
        elem, _, count = inner.partition("x")
        count = int(count)
        elem_map = {"f32": "float", "f16": "half", "i32": "i32"}
        return f"<{count} x {elem_map[elem]}>"
    raise NotImplementedError(f"no LLVM type for {name!r}")


def _param_attrs(attrs: Dict[str, object], t: Type) -> str:
    out: List[str] = []
    if not isinstance(t, PtrType):
        return ""
    if attrs.get("noalias"):
        out.append("noalias")
    if attrs.get("readonly"):
        out.append("readonly")
    if attrs.get("writeonly"):
        out.append("writeonly")
    if attrs.get("nocapture", True):
        out.append("nocapture")
    if attrs.get("nonnull"):
        out.append("nonnull")
    if "align" in attrs and attrs["align"] is not None:
        out.append(f"align {int(attrs['align'])}")
    if "dereferenceable" in attrs and attrs["dereferenceable"] is not None:
        out.append(f"dereferenceable({int(attrs['dereferenceable'])})")
    return (" " + " ".join(out)) if out else ""


def _encode_waitcnt_gfx9_10(vmcnt: int, expcnt: int, lgkmcnt: int) -> int:
    """Encode an AMDGPU ``s_waitcnt`` immediate for gfx9/gfx10-style ISAs.

    The gfx9/gfx10 encoding is not just a 4-bit VMCNT. LLVM's
    ``AMDGPUBaseInfo::encodeWaitcnt`` splits VMCNT across low bits
    ``[3:0]`` and high bits ``[15:14]`` for major versions 9 and 10.
    gfx950 uses that layout, and its VMCNT is 6 bits wide. If we mask
    ``vmcnt=16`` with ``0xf`` it becomes ``vmcnt(0)``, turning the 2D
    attention kernel's intended "leave next K in flight" partial wait
    into a full VMEM drain.

    ``-1`` means "no wait" and is encoded as the architectural maximum
    for each counter. Explicit values are clamped to the maximum instead
    of wrapping to zero; wrapping ``lgkmcnt=16`` to ``lgkmcnt(0)`` has
    the same full-drain bug for async global-to-LDS traffic.
    """

    vm_b = 0x3F if vmcnt < 0 else min(max(vmcnt, 0), 0x3F)
    ec_b = 0x7 if expcnt < 0 else min(max(expcnt, 0), 0x7)
    lk_b = 0xF if lgkmcnt < 0 else min(max(lgkmcnt, 0), 0xF)
    vm_lo = vm_b & 0xF
    vm_hi = (vm_b >> 4) & 0x3
    return vm_lo | (ec_b << 4) | (lk_b << 8) | (vm_hi << 14)


def _fp32_hex(x: float) -> str:
    import struct

    # LLVM textual IR spells `float` hex constants as a 64-bit hex encoding of
    # the exact double value of the rounded fp32 constant. Clang emits e.g.
    # `1.4426950408889634f` as `0x3FF7154760000000` (not the full double
    # precision source literal).
    rounded = struct.unpack("<f", struct.pack("<f", float(x)))[0]
    bits = struct.unpack("<Q", struct.pack("<d", rounded))[0]
    return f"0x{bits:016X}"


def _fp16_hex(x: float) -> str:
    # LLVM IR accepts fp16 constants as `half 0xH<4 hex digits>`. We use
    # a numerically-correct rounding via the struct module.
    import struct

    bits = struct.unpack("<H", struct.pack("<e", float(x)))[0]
    return f"0xH{bits:04X}"


def lower_kernel_to_llvm(kernel: KernelDef) -> str:
    """Return the AMDGPU LLVM IR text for the given kernel."""
    lowerer = _Lowerer(kernel)
    lowerer._collect_smem(kernel.body)
    lowerer.lower_region(kernel.body)
    return lowerer.finalize()
