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

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
# -emit-llvm -S. The string is LLVM-version-keyed, not gfx-keyed: every
# wired arch shares one datalayout, but the buffer-fat-pointer address
# space (``p8``) gained an index-width field between LLVM 20 (ROCm
# 7.0/7.1) and LLVM 21+ (ROCm 7.2 ships LLVM 22):
#
#   LLVM 20:  ...-p8:128:128-...
#   LLVM 22:  ...-p8:128:128:128:48-...
#
# On the textual-IR (comgr SOURCE) path the parser is lenient: it
# overrides the module datalayout with the target's canonical one, so a
# stale-but-well-formed ``p8`` compiles to byte-identical HSACO and the
# drift is invisible at runtime. That leniency is not a contract -- it is
# one field on one ingestion path (bitcode input or a stricter verifier
# can reject a mismatch), so we emit the correct ``p8`` up front. The two
# strings are otherwise identical; pick by :data:`LLVM_FLAVOR_*` via
# :func:`_datalayout_for_flavor`. A drift guard
# (``test_datalayout_matches_hipcc_emitted_ir``) re-derives both from the
# installed toolchain; if it fails, regenerate with the clang command above.
_DATALAYOUT_LLVM20 = (
    "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
    "-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32"
    "-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048"
    "-n32:64-S32-A5-G1-ni:7:8:9"
)
_DATALAYOUT_LLVM22 = (
    "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
    "-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32"
    "-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048"
    "-n32:64-S32-A5-G1-ni:7:8:9"
)
_TRIPLE = "amdgcn-amd-amdhsa"


# A small set of AMDGPU intrinsic signatures changed between LLVM 20
# (ROCm 7.0/7.1) and LLVM 21+ (ROCm 7.2 ships LLVM 22):
#
#   * ``make.buffer.rsrc.p1`` -> ``make.buffer.rsrc.p8.p1`` and the
#     ``num_records`` arg widened from ``i32`` to ``i64`` (LLVM PR
#     #126828).
#   * fp8 / bf8 MFMA A/B operands collapsed from ``<2 x i32>`` to a
#     scalar ``i64`` (same 64 bits, different LLVM type).
#
# comgr verifies the toplevel ``declare`` lines BEFORE running the
# auto-upgrade pass, so we have to emit the right signature up front.
# Pick a flavor once at module import; ``lower_kernel_to_llvm`` takes
# an ``llvm_flavor=`` override for tests.
LLVM_FLAVOR_LLVM20 = "llvm20"
LLVM_FLAVOR_LLVM22 = "llvm22"


def _flavor_for_rocm(major: int, minor: int) -> str:
    """ROCm release -> LLVM flavor expected by the bundled comgr."""
    return LLVM_FLAVOR_LLVM22 if (major, minor) >= (7, 2) else LLVM_FLAVOR_LLVM20


def _datalayout_for_flavor(flavor: str) -> str:
    """Module ``target datalayout`` string for an LLVM flavor.

    The only field that drifts between flavors is the buffer-fat-pointer
    address space ``p8`` (see :data:`_DATALAYOUT_LLVM20` /
    :data:`_DATALAYOUT_LLVM22`). LLVM22 is the default for unknown values
    so a typo'd override degrades to the modern layout rather than the
    legacy one.
    """
    return _DATALAYOUT_LLVM20 if flavor == LLVM_FLAVOR_LLVM20 else _DATALAYOUT_LLVM22


def _torch_hip_version() -> Optional[Tuple[int, int]]:
    """Return ``(major, minor)`` from ``torch.version.hip`` if torch is loaded.

    Torch wheels (e.g. ``torch 2.8.0+rocm7.0.2`` vs ``torch 2.12.0+rocm7.2``)
    bundle their own ``libamd_comgr.so`` whose LLVM version follows the
    wheel's ROCm release, not the system ``/opt/rocm`` one. When rocke
    is paired with a torch-bundled comgr (see
    :func:`runtime.hip_module._torch_bundled_lib`), the flavor must
    match torch's ROCm vintage or comgr will reject the IR or
    silently auto-upgrade declares the lowerer didn't intend.
    """
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return None
    version = getattr(getattr(torch_mod, "version", None), "hip", None)
    if not version:
        return None
    head = str(version).strip().split("-", 1)[0]
    parts = head.split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) >= 2 else 0
    except (IndexError, ValueError):
        return None


def _system_rocm_version() -> Optional[Tuple[int, int]]:
    """Return ``(major, minor)`` from ``/opt/rocm/.info/version``."""
    try:
        with open("/opt/rocm/.info/version") as fh:
            head = fh.read().strip().split("-", 1)[0]
        parts = head.split(".")
        return int(parts[0]), int(parts[1]) if len(parts) >= 2 else 0
    except (OSError, IndexError, ValueError):
        return None


def _comgr_lib_rocm_version() -> Optional[Tuple[int, int]]:
    """ROCm vintage of the comgr lib :mod:`runtime.comgr` will actually load.

    This is the authoritative flavor signal: the flavor MUST match the comgr
    that compiles the IR, and that comgr is the torch-bundled lib whenever torch
    is in the process (regardless of import order) -- not whatever
    ``/opt/rocm`` happens to be. Delegates to
    :func:`runtime.comgr.resolved_lib_rocm_version` (lazy import to avoid a
    core->runtime module-load cycle). Returns ``None`` when the lib path /
    version can't be determined, so the caller falls back to the proxies.
    """
    try:
        from ..runtime.comgr import resolved_lib_rocm_version

        return resolved_lib_rocm_version()
    except Exception:
        return None


def _detect_llvm_flavor() -> str:
    """Pick the LLVM IR flavor for this process.

    Resolution order:

    1. ``$ROCKE_LLVM_FLAVOR`` (explicit override; test/dev knob).
    2. The ROCm vintage of the **comgr lib that will actually compile the IR**
       (:func:`_comgr_lib_rocm_version`). This is the authoritative signal --
       it is import-order-robust and tracks the torch-bundled comgr over a
       stale ``/opt/rocm``, so the emitted IR always matches the codegen
       backend (no ``make.buffer.rsrc.p8.p1`` abort).
    3. ``torch.version.hip`` if torch is imported (fallback proxy).
    4. ``/opt/rocm/.info/version`` (fallback when the comgr path is unknown).
    5. :data:`LLVM_FLAVOR_LLVM22` (the modern default).

    Unknown env values fall through to the auto-detection rather than
    raising on a typo. Each step is wrapped in :func:`try` so a
    misconfigured environment never crashes import.
    """
    env = os.environ.get("ROCKE_LLVM_FLAVOR", "").strip().lower()
    if env in (LLVM_FLAVOR_LLVM20, LLVM_FLAVOR_LLVM22):
        return env
    comgr_ver = _comgr_lib_rocm_version()
    if comgr_ver is not None:
        return _flavor_for_rocm(*comgr_ver)
    torch_ver = _torch_hip_version()
    if torch_ver is not None:
        return _flavor_for_rocm(*torch_ver)
    sys_ver = _system_rocm_version()
    if sys_ver is not None:
        return _flavor_for_rocm(*sys_ver)
    return LLVM_FLAVOR_LLVM22


# Cached, but keyed on the resolved comgr lib path (the "basis") rather than
# resolved-once-forever. An early torch-less call would otherwise lock in the
# /opt/rocm flavor; keying on the comgr path means that once torch (and its
# bundled comgr) enters the process the basis changes and the flavor
# re-resolves. An explicit env override is stable and short-circuits.
_LLVM_FLAVOR: Optional[str] = None
_LLVM_FLAVOR_BASIS: Optional[str] = None


def _resolve_llvm_flavor() -> str:
    global _LLVM_FLAVOR, _LLVM_FLAVOR_BASIS
    env = os.environ.get("ROCKE_LLVM_FLAVOR", "").strip().lower()
    if env in (LLVM_FLAVOR_LLVM20, LLVM_FLAVOR_LLVM22):
        return env
    try:
        from ..runtime.comgr import resolved_lib_path

        basis = resolved_lib_path() or "<none>"
    except Exception:
        basis = "<none>"
    if _LLVM_FLAVOR is None or _LLVM_FLAVOR_BASIS != basis:
        _LLVM_FLAVOR = _detect_llvm_flavor()
        _LLVM_FLAVOR_BASIS = basis
    return _LLVM_FLAVOR


# Intrinsic declarations we may emit.
#
# Entries below are the LLVM 20 / ROCm 7.0--7.1 signatures. The
# affected fp8/bf8 MFMA + ``make.buffer.rsrc`` declares are overridden
# in :data:`_INTRINSIC_DECLS_LLVM22_OVERRIDES` for LLVM 21+ hosts
# (ROCm 7.2+). The dict KEY stays the same across flavors so the
# lowerer's ``_need(...)`` call sites are flavor-agnostic.
_INTRINSIC_DECLS: Dict[str, str] = {
    "workitem.x": "declare i32 @llvm.amdgcn.workitem.id.x()",
    "workitem.y": "declare i32 @llvm.amdgcn.workitem.id.y()",
    "workitem.z": "declare i32 @llvm.amdgcn.workitem.id.z()",
    "workgroup.x": "declare i32 @llvm.amdgcn.workgroup.id.x()",
    "workgroup.y": "declare i32 @llvm.amdgcn.workgroup.id.y()",
    "workgroup.z": "declare i32 @llvm.amdgcn.workgroup.id.z()",
    "s.barrier": "declare void @llvm.amdgcn.s.barrier()",
    # gfx1250 (gfx1250) split wait counters used to drain LDS / VMEM before an
    # LDS barrier (the monolithic s_waitcnt is not selectable there).
    "s.wait.dscnt": "declare void @llvm.amdgcn.s.wait.dscnt(i16)",
    "s.wait.loadcnt": "declare void @llvm.amdgcn.s.wait.loadcnt(i16)",
    # gfx1250 (gfx1250) async global<->LDS DMA + its dedicated ASYNC counter.
    # The gfx9 buffer/global load-to-LDS intrinsics are NOT selectable here.
    "s.wait.asynccnt": "declare void @llvm.amdgcn.s.wait.asynccnt(i16)",
    "global.load.async.to.lds.b32": (
        "declare void @llvm.amdgcn.global.load.async.to.lds.b32("
        "ptr addrspace(1) nocapture, ptr addrspace(3) nocapture, i32 immarg, i32 immarg)"
    ),
    "global.load.async.to.lds.b64": (
        "declare void @llvm.amdgcn.global.load.async.to.lds.b64("
        "ptr addrspace(1) nocapture, ptr addrspace(3) nocapture, i32 immarg, i32 immarg)"
    ),
    "global.load.async.to.lds.b128": (
        "declare void @llvm.amdgcn.global.load.async.to.lds.b128("
        "ptr addrspace(1) nocapture, ptr addrspace(3) nocapture, i32 immarg, i32 immarg)"
    ),
    "exp2.f32": "declare float @llvm.exp2.f32(float)",
    "log2.f32": "declare float @llvm.log2.f32(float)",
    "sqrt.f32": "declare float @llvm.sqrt.f32(float)",
    "rsqrt.f32": "declare float @llvm.amdgcn.rsq.f32(float)",
    "rcp.f32": "declare float @llvm.amdgcn.rcp.f32(float)",
    "tanh.f32": "declare float @llvm.tanh.f32(float)",
    "maxnum.f32": "declare float @llvm.maxnum.f32(float, float)",
    "maxnum.f16": "declare half @llvm.maxnum.f16(half, half)",
    "maxnum.bf16": "declare bfloat @llvm.maxnum.bf16(bfloat, bfloat)",
    "minnum.f32": "declare float @llvm.minnum.f32(float, float)",
    "minnum.f16": "declare half @llvm.minnum.f16(half, half)",
    "minnum.bf16": "declare bfloat @llvm.minnum.bf16(bfloat, bfloat)",
    "fabs.f32": "declare float @llvm.fabs.f32(float)",
    "fabs.f16": "declare half @llvm.fabs.f16(half)",
    "fabs.bf16": "declare bfloat @llvm.fabs.bf16(bfloat)",
    "fmuladd.f32": "declare float @llvm.fmuladd.f32(float, float, float)",
    "fmuladd.f16": "declare half @llvm.fmuladd.f16(half, half, half)",
    "fmuladd.bf16": "declare bfloat @llvm.fmuladd.bf16(bfloat, bfloat, bfloat)",
    # Packed FMA — used by P08's ``vector.fma`` lowering. The AMDGPU
    # backend selects ``v_pk_fma_f32`` / ``v_fma_f16`` chains.
    "fmuladd.v2f32": (
        "declare <2 x float> @llvm.fmuladd.v2f32(<2 x float>, <2 x float>, <2 x float>)"
    ),
    "fmuladd.v4f32": (
        "declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>)"
    ),
    "fmuladd.v8f32": (
        "declare <8 x float> @llvm.fmuladd.v8f32(<8 x float>, <8 x float>, <8 x float>)"
    ),
    "fmuladd.v16f32": (
        "declare <16 x float> @llvm.fmuladd.v16f32("
        "<16 x float>, <16 x float>, <16 x float>)"
    ),
    "fmuladd.v2f16": (
        "declare <2 x half> @llvm.fmuladd.v2f16(<2 x half>, <2 x half>, <2 x half>)"
    ),
    "fmuladd.v4f16": (
        "declare <4 x half> @llvm.fmuladd.v4f16(<4 x half>, <4 x half>, <4 x half>)"
    ),
    "fmuladd.v8f16": (
        "declare <8 x half> @llvm.fmuladd.v8f16(<8 x half>, <8 x half>, <8 x half>)"
    ),
    "fmuladd.v2bf16": (
        "declare <2 x bfloat> @llvm.fmuladd.v2bf16("
        "<2 x bfloat>, <2 x bfloat>, <2 x bfloat>)"
    ),
    "fmuladd.v4bf16": (
        "declare <4 x bfloat> @llvm.fmuladd.v4bf16("
        "<4 x bfloat>, <4 x bfloat>, <4 x bfloat>)"
    ),
    "fmuladd.v8bf16": (
        "declare <8 x bfloat> @llvm.fmuladd.v8bf16("
        "<8 x bfloat>, <8 x bfloat>, <8 x bfloat>)"
    ),
    # RDNA3/3.5 (gfx11) WMMA — wave32 16x16x16 f16. Hardware-verified ABI on
    # gfx1151. Emission goes through Gfx11RdnaBackend.emit_wmma; this is just the
    # declaration registered via _need("wmma.f32.16x16x16.f16").
    "wmma.f32.16x16x16.f16": (
        "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16("
        "<16 x half>, <16 x half>, <8 x float>)"
    ),
    # RDNA3/3.5 (gfx11) WMMA — wave32 16x16x16 bf16. Hardware-verified ABI on
    # gfx1151 (ROCm 7.0.2 clang 20): operands lower as <16 x i16> (bf16 bitcast),
    # accumulator/result are <8 x float>. Emission goes through
    # Gfx11RdnaBackend.emit_wmma, which bitcasts <16 x bfloat> -> <16 x i16>.
    "wmma.f32.16x16x16.bf16": (
        "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16("
        "<16 x i16>, <16 x i16>, <8 x float>)"
    ),
    # RDNA3/3.5 (gfx11) integer WMMA — wave32 16x16x16 iu8. Operands are
    # <4 x i32> (16 int8 packed 4-per-i32), accumulator/result <8 x i32>. The
    # signature carries i1 signedness flags (unsignedA/unsignedB) before each
    # matrix operand and a trailing i1 clamp. Emission goes through
    # Gfx11RdnaBackend.emit_wmma. LLVM-lowered to v_wmma_i32_16x16x16_iu8.
    "wmma.i32.16x16x16.iu8": (
        "declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32("
        "i1, <4 x i32>, i1, <4 x i32>, <8 x i32>, i1)"
    ),
    # RDNA3/3.5 (gfx11) integer WMMA — wave32 16x16x16 iu4. Operands are
    # <2 x i32> (16 int4 packed 8-per-i32), accumulator/result <8 x i32>.
    "wmma.i32.16x16x16.iu4": (
        "declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32("
        "i1, <2 x i32>, i1, <2 x i32>, <8 x i32>, i1)"
    ),
    # RDNA4 (gfx12) WMMA — wave32 16x16x16. No cross-half operand duplication:
    # A/B are <8 x half> / <8 x i16> per lane (vs <16 x ...> on gfx11). Emission
    # goes through Gfx12RdnaBackend.emit_wmma.
    "wmma.gfx12.f32.16x16x16.f16": (
        "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16("
        "<8 x half>, <8 x half>, <8 x float>)"
    ),
    "wmma.gfx12.f32.16x16x16.bf16": (
        "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16("
        "<8 x i16>, <8 x i16>, <8 x float>)"
    ),
    # gfx1250 (gfx1250) WMMA — wave32 16x16x32 (K=32). A/B are <16 x half> per lane
    # and the intrinsic carries the gfx1250 8-operand form (i1 neg flags, i16
    # format modifier, i1 reuse flags). bf16 uses <16 x bfloat> directly (v16bf16,
    # not the <16 x i16> bitcast gfx11/gfx12 use). Emission goes through
    # Gfx1250Backend.emit_wmma. ABI confirmed on ROCm 7.13 clang 23.
    "wmma.gfx1250.f32.16x16x32.f16": (
        "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.f16.v8f32.v16f16("
        "i1 immarg, <16 x half>, i1 immarg, <16 x half>, i16 immarg, "
        "<8 x float>, i1 immarg, i1 immarg)"
    ),
    "wmma.gfx1250.f32.16x16x32.bf16": (
        "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16("
        "i1 immarg, <16 x bfloat>, i1 immarg, <16 x bfloat>, i16 immarg, "
        "<8 x float>, i1 immarg, i1 immarg)"
    ),
    # gfx1250 (gfx1250) FP8/BF8 WMMA — wave32 16x16x64 (K=64, NOT the gfx12
    # 16x16x16 FP8 form, which is not selectable on gfx1250). A/B are 32 fp8/bf8
    # bytes per lane, presented to the intrinsic as <8 x i32>; the accumulator /
    # result is <8 x float>. 6-operand form: (A, B, i16 fmt, C, i1, i1).
    # Signature confirmed on a gfx1250 device (ROCm 7.13 clang 23) via
    # __builtin_amdgcn_wmma_f32_16x16x64_{fp8,bf8}_{fp8,bf8}.
    "wmma.gfx1250.f32.16x16x64.fp8.fp8": (
        "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v8f32.v8i32("
        "<8 x i32>, <8 x i32>, i16 immarg, <8 x float>, i1 immarg, i1 immarg)"
    ),
    "wmma.gfx1250.f32.16x16x64.fp8.bf8": (
        "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.fp8.bf8.v8f32.v8i32("
        "<8 x i32>, <8 x i32>, i16 immarg, <8 x float>, i1 immarg, i1 immarg)"
    ),
    "wmma.gfx1250.f32.16x16x64.bf8.fp8": (
        "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.bf8.fp8.v8f32.v8i32("
        "<8 x i32>, <8 x i32>, i16 immarg, <8 x float>, i1 immarg, i1 immarg)"
    ),
    "wmma.gfx1250.f32.16x16x64.bf8.bf8": (
        "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.bf8.bf8.v8f32.v8i32("
        "<8 x i32>, <8 x i32>, i16 immarg, <8 x float>, i1 immarg, i1 immarg)"
    ),
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
    "mfma.f32.32x32x8bf16.1k": (
        "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x8bf16.1k("
        "<4 x i16>, <4 x i16>, <16 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.32x32x16.f16": (
        "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16("
        "<8 x half>, <8 x half>, <16 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.16x16x4f32": (
        "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32("
        "float, float, <4 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.32x32x2f32": (
        "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32("
        "float, float, <16 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.4x4x4f16": (
        "declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x4f16("
        "<4 x half>, <4 x half>, <4 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    # FP8 / BF8 MFMA (gfx940+). Operands are packed as <2 x i32>
    # at the intrinsic boundary (= 8 fp8 / bf8 bytes per lane); the
    # lowering helper bitcasts the <8 x fp8e4m3> / <8 x bf8e5m2> IR
    # operands into <2 x i32> before the call. Only the homogeneous
    # variants (fp8.fp8 and bf8.bf8) are exposed as IR ops; the
    # mixed-precision variants (fp8.bf8 / bf8.fp8) are reachable by
    # bitcasting the operand vectors manually and calling the same
    # intrinsic family.
    "mfma.f32.16x16x32.fp8.fp8": (
        "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8("
        "<2 x i32>, <2 x i32>, <4 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.16x16x32.bf8.bf8": (
        "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8("
        "<2 x i32>, <2 x i32>, <4 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.32x32x16.fp8.fp8": (
        "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.fp8.fp8("
        "<2 x i32>, <2 x i32>, <16 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.32x32x16.bf8.bf8": (
        "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8("
        "<2 x i32>, <2 x i32>, <16 x float>, "
        "i32 immarg, i32 immarg, i32 immarg)"
    ),
    "readfirstlane.i32": ("declare i32 @llvm.amdgcn.readfirstlane.i32(i32)"),
    "readfirstlane.i64": ("declare i64 @llvm.amdgcn.readfirstlane.i64(i64)"),
    "ballot.i64": ("declare i64 @llvm.amdgcn.ballot.i64(i1)"),
    "ds.bpermute": ("declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)"),
    # gfx9+ ``v_mov_b32_dpp`` row-shift primitive (1 cycle, no LDS).
    # The ``update.dpp`` family takes (old, src, dpp_ctrl, row_mask,
    # bank_mask, bound_ctrl); we use the i32 specialisation because
    # ``row_shr`` / ``row_shl`` payloads are always lane-int values
    # in the kernels that need it (topk_softmax cumsum, moe_sort scan).
    "update.dpp.i32": (
        "declare i32 @llvm.amdgcn.update.dpp.i32("
        "i32, i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg)"
    ),
    # Packed bf16 atomic add (gfx940+). Two bf16 lanes per atomic transaction.
    # Used by FMHA-bwd's dQ accumulate path when the caller wants to
    # land bf16 directly in HBM rather than running a separate f32 -> bf16
    # cast pass on the workspace.
    "global.atomic.fadd.v2bf16": (
        "declare <2 x bfloat> @llvm.amdgcn.global.atomic.fadd.v2bf16.p1("
        "ptr addrspace(1), <2 x bfloat>)"
    ),
    "mbcnt.lo": ("declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)"),
    "mbcnt.hi": ("declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)"),
    "ds.read.tr16.b64": (
        "declare <4 x i16> @llvm.amdgcn.ds.read.tr16.b64(ptr addrspace(3))"
    ),
    # gfx950 ``ds_read_b128_tr_b16`` — wide transposed-LDS read returning
    # ``<8 x i16>`` per lane (the K-packed MFMA B-operand layout for the
    # 16x16x32 / 32x32x16 atoms; one LDS op vs two paired b64 reads).
    "ds.read.tr16.b128": (
        "declare <8 x i16> @llvm.amdgcn.ds.read.tr16.b128(ptr addrspace(3))"
    ),
    # gfx1250 (gfx1250) wave32 transpose-LDS read: ``ds_load_tr16_b128`` returns
    # ``<8 x bfloat>`` per lane (the WMMA 16x16x32 B-operand layout). This is the
    # gfx1250 counterpart of gfx950's ``ds_read_tr16_b128`` (different opcode +
    # wave32 lane distribution).
    "ds.load.tr16.b128.v8bf16": (
        "declare <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3))"
    ),
    # f16 sibling of the above: ``ds_load_tr16_b128`` is overloaded on its
    # result element type, so an f16 transpose-read selects the ``.v8f16`` form
    # (returns ``<8 x half>``) rather than reinterpreting a bf16 payload.
    "ds.load.tr16.b128.v8f16": (
        "declare <8 x half> @llvm.amdgcn.ds.load.tr16.b128.v8f16(ptr addrspace(3))"
    ),
    "iglp.opt": ("declare void @llvm.amdgcn.iglp.opt(i32 immarg)"),
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
    # Flat/global direct-to-LDS DMA: streams bytes from a global pointer
    # straight into LDS, bypassing the VGPR round-trip. Lowers to
    # ``global_load_lds_dword{,x4}``. Signature (target-neutral, no LLVM22
    # override needed): (ptr addrspace(1) src, ptr addrspace(3) lds,
    # i32 size, i32 imm_offset, i32 aux); size/imm_offset/aux are immarg.
    "global.load.lds": (
        "declare void @llvm.amdgcn.global.load.lds("
        "ptr addrspace(1) nocapture readonly, ptr addrspace(3) nocapture, "
        "i32 immarg, i32 immarg, i32 immarg)"
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
    # 2-byte buffer load. Returns i16 (vs the i32 4-byte load), so the
    # AMDGPU OOB clamp fires per-element instead of clamping a 4-byte
    # window. Required by P10 to bit-fix the trailing-element corruption
    # in ``img2col vec_k=1`` and ``pooling vec=1 avg`` (see
    # ``PROPOSALS_PLAN.md::P10`` for the LLVM bug pattern).
    "raw.ptr.buffer.load.i16": (
        "declare i16 @llvm.amdgcn.raw.ptr.buffer.load.i16("
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
    "amdgcn.cvt.f32.bf8": ("declare float @llvm.amdgcn.cvt.f32.bf8(i32, i32 immarg)"),
    # Quant direction (f32 -> {fp8e4m3, bf8e5m2}). These are the packed
    # AMDGPU intrinsics (two f32 srcs at once, returning an i32 with two
    # bytes packed). The single-element wrapper in
    # ``_op_arith_cvt_f32_to_{fp8,bf8}`` zeroes the second slot and
    # truncs to i8.
    "amdgcn.cvt.pk.fp8.f32": (
        "declare i32 @llvm.amdgcn.cvt.pk.fp8.f32(float, float, i32, i1)"
    ),
    "amdgcn.cvt.pk.bf8.f32": (
        "declare i32 @llvm.amdgcn.cvt.pk.bf8.f32(float, float, i32, i1)"
    ),
    # Dequant direction (packed: i32 holding 4 fp8/bf8 bytes -> <2 x float>
    # with a 1-bit word selector picking bytes 0,1 or 2,3). Two calls
    # cover the full 4-byte input — twice the throughput of the
    # per-byte ``llvm.amdgcn.cvt.f32.fp8`` for FP8 dequant loops.
    "amdgcn.cvt.pk.f32.fp8": (
        "declare <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32, i1)"
    ),
    "amdgcn.cvt.pk.f32.bf8": (
        "declare <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32, i1)"
    ),
    # f32 -> i8 saturating quant: RNE round followed by trunc + clamp.
    # ``llvm.rint.f32`` honours the current rounding mode (RNE by
    # default); the explicit clamp guarantees the trunc-to-i8 stays in
    # [-128, 127] even for callers that didn't pre-clamp.
    "rint.f32": "declare float @llvm.rint.f32(float)",
    "smax.i32": "declare i32 @llvm.smax.i32(i32, i32)",
    "smin.i32": "declare i32 @llvm.smin.i32(i32, i32)",
    # AMDGPU ``v_perm_b32`` byte-select. Used by P09's packed i8 quant
    # to fuse four clamped i32s into one i32 in 2 perm calls (lo half +
    # hi half) followed by one perm to interleave.
    "amdgcn.perm": ("declare i32 @llvm.amdgcn.perm(i32, i32, i32)"),
    # gfx950 fused FP8/BF8 dequant + scale. SINGLE instruction
    # ``v_cvt_scalef32_pk_f32_fp8`` does ``<2 x f32> = scale * fp8x2``
    # — the same operation our current path takes 3 separate instructions
    # to do (cvt_pk_f32_fp8 + v_pk_mul + v_pk_mul). Triton's long-prefill
    # kernel emits exactly this instruction for its FP8 KV dequant. The
    # i1 immediate selects which 2 of the 4 packed fp8 bytes to convert
    # (0 → bytes 0,1; 1 → bytes 2,3).
    "amdgcn.cvt.scalef32.pk.f32.fp8": (
        "declare <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.fp8(i32, float, i1)"
    ),
    "amdgcn.cvt.scalef32.pk.f32.bf8": (
        "declare <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.bf8(i32, float, i1)"
    ),
    # Reverse direction: <2 x f32> + scale -> 2 fp8 bytes packed into i32.
    # First call (i1=false) fills bytes 0,1; second call (i1=true with
    # the first call's i32 result as the accumulator) fills bytes 2,3.
    # Saves the host-side rescale + cvt_pk_fp8 + bitshift dance for
    # output FP8 quantisation paths.
    "amdgcn.cvt.scalef32.pk.fp8.f32": (
        "declare i32 @llvm.amdgcn.cvt.scalef32.pk.fp8.f32(i32, <2 x float>, float, i1)"
    ),
    "amdgcn.cvt.scalef32.pk.bf8.f32": (
        "declare i32 @llvm.amdgcn.cvt.scalef32.pk.bf8.f32(i32, <2 x float>, float, i1)"
    ),
    # gfx950 ``ds_swizzle_b32`` — single-instruction intra-32-lane
    # permute. We use it for the softmax XOR-butterfly reduction; the
    # FFT-mode offset is encoded as an immediate and the LDS unit
    # delivers the permuted value in 1 LDS op (vs ds_bpermute's
    # VGPR-addr-compute + LDS round-trip = 2 ops).
    "amdgcn.ds.swizzle": ("declare i32 @llvm.amdgcn.ds.swizzle(i32, i32 immarg)"),
    # gfx950 ``v_permlane32_swap_b32`` — wave64 swap-across-32-lane-halves
    # primitive. Single VALU instruction; replaces the FINAL stage of a
    # wave64 softmax reduction (combining the lo-32 and hi-32 lane half
    # partial reductions). Returns a struct holding both swapped values
    # so a single call covers BOTH register exchanges.
    "amdgcn.permlane32.swap": (
        "declare { i32, i32 } @llvm.amdgcn.permlane32.swap(i32, i32, i1, i1)"
    ),
    # gfx11 ``v_permlanex16_b32`` — swap each lane with its ``lane ^ 16``
    # partner within a 32-lane group via a permute network (NOT the LDS
    # unit). One VALU op; this is the cheap cross-half vehicle CK's gfx11
    # FMHA pipelines use for the WMMA C->A transpose. Args:
    # (old, src, sel_lo, sel_hi, fi, bound_ctrl).
    "amdgcn.permlanex16": (
        "declare i32 @llvm.amdgcn.permlanex16(i32, i32, i32, i32, i1, i1)"
    ),
    # gfx950 ``v_mfma_f32_32x32x16_bf16`` — wider MFMA shape (32x32
    # output × 16-K) than the 16x16x32 we use elsewhere. Same FLOPs
    # per cycle (1024 cycles per inst either way at the per-CTA level)
    # but HALF the instruction count, which (a) halves the SIMD issue
    # overhead, (b) keeps each lane's accumulator at 16 floats (vs
    # 4 for 16x16x32 — letting in-lane v_max3 chains do most of the
    # row-reduce work before any cross-lane traffic, matching Triton's
    # long-prefill softmax pattern), and (c) reduces the per-iter LDS
    # K-load coalescing cost.
    "mfma.f32.32x32x16.bf16": (
        "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16("
        "<8 x bfloat>, <8 x bfloat>, <16 x float>, i32 immarg, i32 immarg, i32 immarg)"
    ),
    # P15: gfx950 MX MFMA scaled intrinsic. Per-warp E8M0 scales apply
    # in-instruction (vs the post-hoc scale-apply chain). Operands are
    # 16-byte packed mantissa vectors; the scale broadcast inside the
    # instruction is per-output-row (fixes the B_MX1 row-aware
    # correctness gap that the historical post-hoc chain had).
    "mfma.scale.f32.16x16x128.f8f6f4": (
        "declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4("
        "<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg, "
        "i32 immarg, i32 immarg, i32, i32 immarg, i32, i32 immarg)"
    ),
    # P52: fp4 / fp6 MX MFMA atoms (gfx950+). Operand widths follow the
    # OCP MX spec: ``16 x fp4`` packs into i64 per lane, ``16 x fp6``
    # is a 96-bit value modelled here as `<3 x i32>`.
    "mfma.f32.16x16x128.fp4": (
        "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x128.fp4("
        "i64, i64, <4 x float>, i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.16x16x96.fp6": (
        "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x96.fp6("
        "<3 x i32>, <3 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)"
    ),
    # L6: UNSCALED fp8 16x16x128 hero atom. gfx950 has no dense plain
    # ``mfma.f32.16x16x128.fp8.fp8``; the dense wide-K f8 MFMA is the
    # ``f8f6f4`` scale-MFMA, which on LLVM22 (ROCm 7.2) takes the 9-arg
    # signature below: A/B as ``<8 x i32>`` (32 f8 bytes per lane), then
    # ``cbsz`` / ``blgp`` (format selectors, 0 = fp8e4m3), then the
    # ``op_sel`` + scale-byte pairs for A and B. We declare it under a
    # dedicated additive key so the existing 11-arg
    # ``mfma.scale.f32.16x16x128.f8f6f4`` decl (used by the MX-scaled
    # path) is untouched; the lowering pins both scale bytes to 0
    # (E8M0 exponent 0 => factor 1.0) for a true unscaled MFMA.
    "mfma.f32.16x16x128.fp8.hero": (
        "declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4("
        "<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg, "
        "i32 immarg, i32, i32 immarg, i32)"
    ),
}


# Declares that change shape on LLVM 21+ hosts (ROCm 7.2+). Keys MUST
# already exist in :data:`_INTRINSIC_DECLS`; only the declaration TEXT
# differs, so call-site ``_need(...)`` lookups stay flavor-agnostic.
_INTRINSIC_DECLS_LLVM22_OVERRIDES: Dict[str, str] = {
    "mfma.f32.16x16x32.fp8.fp8": (
        "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8("
        "i64, i64, <4 x float>, i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.16x16x32.bf8.bf8": (
        "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8("
        "i64, i64, <4 x float>, i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.32x32x16.fp8.fp8": (
        "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.fp8.fp8("
        "i64, i64, <16 x float>, i32 immarg, i32 immarg, i32 immarg)"
    ),
    "mfma.f32.32x32x16.bf8.bf8": (
        "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8("
        "i64, i64, <16 x float>, i32 immarg, i32 immarg, i32 immarg)"
    ),
    "make.buffer.rsrc.p1": (
        "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1("
        "ptr addrspace(1) nocapture readnone, i16, i64, i32)"
    ),
}


def _llvm_type(t: Type) -> str:
    """Map an IR Type to its LLVM IR textual form."""
    if isinstance(t, PtrType):
        if t.space == "global":
            return "ptr addrspace(1)"
        if t.space == "lds":
            return "ptr addrspace(3)"
        if t.space == "constant":
            return "ptr addrspace(4)"
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
    if t.name == "i16":
        return "i16"
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
    if t.name == "bf8e5m2":
        return "i8"
    if t.name == "f32":
        return "float"
    raise NotImplementedError(f"no LLVM mapping for type {t!r}")


def _escape_llvm_asm_string(s: str) -> str:
    r"""Escape a Python string for an LLVM IR asm/string literal.

    LLVM textual string literals only allow printable ASCII verbatim; every
    other byte (and ``"`` / ``\``) must be written as a ``\XX`` hex escape.
    This matters for multi-statement inline-asm templates: a real newline
    separating two instructions must become ``\0A`` (a literal newline char
    would split the IR line; the AMDGPU ``;`` is a comment, not a separator).
    """
    out = []
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\5C")
        elif ch == '"':
            out.append("\\22")
        elif 0x20 <= o <= 0x7E:
            out.append(ch)
        else:
            out.append(f"\\{o:02X}")
    return "".join(out)


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
    def __init__(
        self,
        kernel: KernelDef,
        *,
        llvm_flavor: Optional[str] = None,
        arch: Optional[str] = None,
    ) -> None:
        self.kernel = kernel
        # ISA backend selects the gfx-keyed LLVM details (datalayout, triple,
        # waitcnt encoding). Defaults to gfx950 so existing callers and the
        # gfx950 byte-identical baseline are preserved.
        from .isa.backend import backend_for

        self._backend = backend_for(arch or "gfx950")
        flavor = llvm_flavor if llvm_flavor is not None else _resolve_llvm_flavor()
        if flavor not in (LLVM_FLAVOR_LLVM20, LLVM_FLAVOR_LLVM22):
            raise ValueError(f"unknown LLVM flavor {flavor!r}")
        self._flavor: str = flavor
        # Preserve insertion order of ``_INTRINSIC_DECLS`` -- it drives
        # the emit order in ``finalize``.
        self._decls: Dict[str, str] = dict(_INTRINSIC_DECLS)
        if flavor == LLVM_FLAVOR_LLVM22:
            self._decls.update(_INTRINSIC_DECLS_LLVM22_OVERRIDES)
        self._needs_intrin: Dict[str, bool] = {}
        # Set when an f32 global atomic-add is lowered with the
        # native-hardware-fadd metadata (no.fine.grained / no.remote
        # memory). Lets the AMDGPU backend emit a single
        # ``global_atomic_add_f32`` instruction instead of a
        # compare-and-swap retry loop.
        self._needs_fp_atomic_md: bool = False
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

    # ----- constant folding helpers -----

    def _is_constant(self, v: Value) -> bool:
        """Check if a value is a compile-time constant.

        Returns True if the value is produced by arith.constant op.
        """
        if v.op is None:
            return False
        return v.op.name == "arith.constant"

    def _eval_constant(self, v: Value) -> int:
        """Evaluate a compile-time constant to an integer.

        Args:
        v: Value that must be a compile-time constant (from arith.constant)

        Returns:
        Integer value of the constant

        Raises:
        ValueError: If value is not a compile-time constant
        """
        if not self._is_constant(v):
            raise ValueError(f"Value {v.name} is not a compile-time constant")

        op = v.op
        val = op.attrs["value"]
        return int(val)

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

    def _op_arith_fabs(self, op: Op) -> None:
        """Lower ``arith.fabs`` to ``llvm.fabs.<ty>`` so the AMDGPU
        backend can fold the absolute-value modifier into the
        consumer (free) instead of materialising ``-a`` and a
        ``v_max_f32(a, -a)`` pair (the historical idiom).
        """
        (v,) = op.operands
        ty_name = v.type.name
        llvm_ty = {"f32": "float", "f16": "half", "bf16": "bfloat"}.get(ty_name)
        if llvm_ty is None:
            raise NotImplementedError(f"fabs: unsupported FP type {ty_name!r}")
        self._need(f"fabs.{ty_name}")
        self._current().emit(
            f"  {op.result.name} = call {llvm_ty} @llvm.fabs.{ty_name}("
            f"{llvm_ty} {self._operand(v)})"
        )

    def _op_arith_fma(self, op: Op) -> None:
        """Lower ``arith.fma`` to ``llvm.fmuladd.<ty>`` so the AMDGPU
        MachineCombiner always picks ``v_fma_f32``.

        The intrinsic accepts ``contract`` semantics by default;
        unlike a bare ``fmul`` + ``fadd`` pair, we don't rely on the
        scheduler proving safety.
        """
        a, b, c = op.operands
        ty_name = a.type.name
        llvm_ty = {"f32": "float", "f16": "half", "bf16": "bfloat"}.get(ty_name)
        if llvm_ty is None:
            raise NotImplementedError(f"fma: unsupported FP type {ty_name!r}")
        self._need(f"fmuladd.{ty_name}")
        self._current().emit(
            f"  {op.result.name} = call {llvm_ty} @llvm.fmuladd.{ty_name}("
            f"{llvm_ty} {self._operand(a)}, {llvm_ty} {self._operand(b)}, "
            f"{llvm_ty} {self._operand(c)})"
        )

    def _op_arith_fmax3(self, op: Op) -> None:
        """Lower ``arith.fmax3(a, b, c)`` -> ``maxnum(a, maxnum(b, c))``.

        Two back-to-back ``llvm.maxnum`` calls; the AMDGPU peephole
        folds the chain into ``v_max3_f32`` (single-cycle on gfx9+).
        Same semantics as the ``fmax(fmax(a, b), c)`` idiom but emitted
        once and tagged for the combiner.
        """
        a, b, c = op.operands
        ty_name = a.type.name
        llvm_ty = {"f32": "float", "f16": "half", "bf16": "bfloat"}.get(ty_name)
        intrin_key = {
            "f32": "maxnum.f32",
            "f16": "maxnum.f16",
            "bf16": "maxnum.bf16",
        }.get(ty_name)
        if llvm_ty is None or intrin_key is None:
            raise NotImplementedError(f"fmax3: unsupported FP type {ty_name!r}")
        self._need(intrin_key)
        inner = self._fresh("fmax3.bc")
        self._current().emit(
            f"  {inner} = call {llvm_ty} @llvm.maxnum.{ty_name}("
            f"{llvm_ty} {self._operand(b)}, {llvm_ty} {self._operand(c)})"
        )
        self._current().emit(
            f"  {op.result.name} = call {llvm_ty} @llvm.maxnum.{ty_name}("
            f"{llvm_ty} {self._operand(a)}, {llvm_ty} {inner})"
        )

    def _op_arith_fmin3(self, op: Op) -> None:
        """Sibling of :meth:`_op_arith_fmax3` for three-way min."""
        a, b, c = op.operands
        ty_name = a.type.name
        llvm_ty = {"f32": "float", "f16": "half", "bf16": "bfloat"}.get(ty_name)
        intrin_key = {
            "f32": "minnum.f32",
            "f16": "minnum.f16",
            "bf16": "minnum.bf16",
        }.get(ty_name)
        if llvm_ty is None or intrin_key is None:
            raise NotImplementedError(f"fmin3: unsupported FP type {ty_name!r}")
        self._need(intrin_key)
        inner = self._fresh("fmin3.bc")
        self._current().emit(
            f"  {inner} = call {llvm_ty} @llvm.minnum.{ty_name}("
            f"{llvm_ty} {self._operand(b)}, {llvm_ty} {self._operand(c)})"
        )
        self._current().emit(
            f"  {op.result.name} = call {llvm_ty} @llvm.minnum.{ty_name}("
            f"{llvm_ty} {self._operand(a)}, {llvm_ty} {inner})"
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
        ty_name = a.type.name
        # Dispatch on the operand FP width: half, float, bfloat each
        # have their own maxnum intrinsic. The previous f32-only path
        # silently mis-typed half operands as float and broke comgr.
        llvm_ty = {"f32": "float", "f16": "half", "bf16": "bfloat"}.get(ty_name)
        intrin_key = {
            "f32": "maxnum.f32",
            "f16": "maxnum.f16",
            "bf16": "maxnum.bf16",
        }.get(ty_name)
        if llvm_ty is None or intrin_key is None:
            raise NotImplementedError(f"fmax: unsupported FP type {ty_name!r}")
        self._need(intrin_key)
        self._current().emit(
            f"  {op.result.name} = call {llvm_ty} @llvm.maxnum.{ty_name}("
            f"{llvm_ty} {self._operand(a)}, {llvm_ty} {self._operand(b)})"
        )

    def _op_arith_fmin(self, op: Op) -> None:
        a, b = op.operands
        ty_name = a.type.name
        llvm_ty = {"f32": "float", "f16": "half", "bf16": "bfloat"}.get(ty_name)
        intrin_key = {
            "f32": "minnum.f32",
            "f16": "minnum.f16",
            "bf16": "minnum.bf16",
        }.get(ty_name)
        if llvm_ty is None or intrin_key is None:
            raise NotImplementedError(f"fmin: unsupported FP type {ty_name!r}")
        self._need(intrin_key)
        self._current().emit(
            f"  {op.result.name} = call {llvm_ty} @llvm.minnum.{ty_name}("
            f"{llvm_ty} {self._operand(a)}, {llvm_ty} {self._operand(b)})"
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

    def _op_arith_smax(self, op: Op) -> None:
        a, b = op.operands
        self._need("smax.i32")
        self._current().emit(
            f"  {op.result.name} = call i32 @llvm.smax.i32(i32 {self._operand(a)}, "
            f"i32 {self._operand(b)})"
        )

    def _op_arith_smin(self, op: Op) -> None:
        a, b = op.operands
        self._need("smin.i32")
        self._current().emit(
            f"  {op.result.name} = call i32 @llvm.smin.i32(i32 {self._operand(a)}, "
            f"i32 {self._operand(b)})"
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

    def _op_arith_trunc(self, op: Op) -> None:
        (v,) = op.operands
        self._current().emit(
            f"  {op.result.name} = trunc {_llvm_type(v.type)} {self._operand(v)} "
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

    def _op_arith_cvt_bf8_to_f32(self, op: Op) -> None:
        """Lower bf8e5m2->f32 conversion to AMDGPU's ``llvm.amdgcn.cvt.f32.bf8``.

        e5m2 sibling of :meth:`_op_arith_cvt_fp8_to_f32`. Same byte-packing
        scheme: zext the single ``i8`` operand into an i32 and select
        byte lane 0.
        """
        (v,) = op.operands
        self._need("amdgcn.cvt.f32.bf8")
        tmp = f"{op.result.name}x"
        self._current().emit(f"  {tmp} = zext i8 {self._operand(v)} to i32")
        self._current().emit(
            f"  {op.result.name} = call float @llvm.amdgcn.cvt.f32.bf8(i32 {tmp}, i32 0)"
        )

    def _op_arith_cvt_pk_f32_fp8x4(self, op: Op) -> None:
        """Lower <4 x fp8e4m3> -> <4 x f32> via 2x packed
        ``llvm.amdgcn.cvt.pk.f32.fp8`` calls (lowering to two
        ``v_cvt_pk_f32_fp8`` instructions).

        The input is bitcast from <4 x i8> to a single i32. The
        intrinsic takes that i32 and an i1 word-select (false = bytes
        0,1 → <2 x f32>; true = bytes 2,3 → <2 x f32>). We assemble
        the two <2 x f32> halves into a single <4 x f32> result via
        a shuffle.

        This is the FP8 K/V dequant primitive AITER uses
        (`to_float_fp8x4` in `csrc/include/attention_common.cuh`).
        Cuts the cvt instruction count in HALF compared to 4x
        scalar `cvt.f32.fp8`.
        """
        (v,) = op.operands
        self._need("amdgcn.cvt.pk.f32.fp8")
        packed_i32 = f"{op.result.name}p"
        lo = f"{op.result.name}lo"
        hi = f"{op.result.name}hi"
        # <4 x fp8e4m3> is laid out as <4 x i8> in LLVM. Bitcast to i32.
        self._current().emit(
            f"  {packed_i32} = bitcast <4 x i8> {self._operand(v)} to i32"
        )
        self._current().emit(
            f"  {lo} = call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 {packed_i32}, i1 false)"
        )
        self._current().emit(
            f"  {hi} = call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 {packed_i32}, i1 true)"
        )
        # Concat the two <2 x f32> into <4 x f32> via shufflevector.
        self._current().emit(
            f"  {op.result.name} = shufflevector <2 x float> {lo}, <2 x float> {hi}, "
            f"<4 x i32> <i32 0, i32 1, i32 2, i32 3>"
        )

    def _op_arith_cvt_pk_f32_bf8x4(self, op: Op) -> None:
        """e5m2 sibling of :meth:`_op_arith_cvt_pk_f32_fp8x4`. Same
        scheme via ``llvm.amdgcn.cvt.pk.f32.bf8``.
        """
        (v,) = op.operands
        self._need("amdgcn.cvt.pk.f32.bf8")
        packed_i32 = f"{op.result.name}p"
        lo = f"{op.result.name}lo"
        hi = f"{op.result.name}hi"
        self._current().emit(
            f"  {packed_i32} = bitcast <4 x i8> {self._operand(v)} to i32"
        )
        self._current().emit(
            f"  {lo} = call <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32 {packed_i32}, i1 false)"
        )
        self._current().emit(
            f"  {hi} = call <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32 {packed_i32}, i1 true)"
        )
        self._current().emit(
            f"  {op.result.name} = shufflevector <2 x float> {lo}, <2 x float> {hi}, "
            f"<4 x i32> <i32 0, i32 1, i32 2, i32 3>"
        )

    def _op_arith_cvt_scalef32_pk_f32_fp8(self, op: Op) -> None:
        """Lower fused scale+dequant <4 x fp8e4m3> + f32 -> <4 x f32>.

        Emits two gfx950 ``v_cvt_scalef32_pk_f32_fp8`` instructions
        (each does 2 fp8 → 2 f32 with embedded scale multiply), and
        shuffles into a single <4 x f32>. The scale is broadcast to
        all lanes (the intrinsic accepts a single f32 operand which
        the AMDGPU backend places in SGPR by default).

        Saves an entire ``v_pk_mul_f32`` per pack vs the non-fused
        ``cvt_pk_f32_fp8 + mul`` sequence. Production trace bench
        showed FP8 long-prefill inner loop drop ``v_pk_mul`` count
        from 64 → 16.
        """
        v, scale = op.operands
        self._need("amdgcn.cvt.scalef32.pk.f32.fp8")
        packed_i32 = f"{op.result.name}p"
        lo = f"{op.result.name}lo"
        hi = f"{op.result.name}hi"
        self._current().emit(
            f"  {packed_i32} = bitcast <4 x i8> {self._operand(v)} to i32"
        )
        self._current().emit(
            f"  {lo} = call <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.fp8("
            f"i32 {packed_i32}, float {self._operand(scale)}, i1 false)"
        )
        self._current().emit(
            f"  {hi} = call <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.fp8("
            f"i32 {packed_i32}, float {self._operand(scale)}, i1 true)"
        )
        self._current().emit(
            f"  {op.result.name} = shufflevector <2 x float> {lo}, <2 x float> {hi}, "
            f"<4 x i32> <i32 0, i32 1, i32 2, i32 3>"
        )

    def _op_arith_cvt_scalef32_pk_f32_bf8(self, op: Op) -> None:
        """e5m2 sibling of :meth:`_op_arith_cvt_scalef32_pk_f32_fp8`."""
        v, scale = op.operands
        self._need("amdgcn.cvt.scalef32.pk.f32.bf8")
        packed_i32 = f"{op.result.name}p"
        lo = f"{op.result.name}lo"
        hi = f"{op.result.name}hi"
        self._current().emit(
            f"  {packed_i32} = bitcast <4 x i8> {self._operand(v)} to i32"
        )
        self._current().emit(
            f"  {lo} = call <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.bf8("
            f"i32 {packed_i32}, float {self._operand(scale)}, i1 false)"
        )
        self._current().emit(
            f"  {hi} = call <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.bf8("
            f"i32 {packed_i32}, float {self._operand(scale)}, i1 true)"
        )
        self._current().emit(
            f"  {op.result.name} = shufflevector <2 x float> {lo}, <2 x float> {hi}, "
            f"<4 x i32> <i32 0, i32 1, i32 2, i32 3>"
        )

    def _op_arith_cvt_f32_to_fp8(self, op: Op) -> None:
        """Lower f32->fp8e4m3 quantisation via ``llvm.amdgcn.cvt.pk.fp8.f32``.

        AMDGPU's packing intrinsic produces an i32 with two bytes filled
        (byte 0 = first f32, byte 1 = second f32). We zero the second
        slot, select word-lane 0 (so byte 0 is the live result), then
        trunc the i32 to an i8. The single-element overhead is one
        ``v_pk_cvt_f32_fp8`` plus a trunc; the AMDGPU backend collapses
        the trunc into the byte select on most code paths.
        """
        (v,) = op.operands
        self._need("amdgcn.cvt.pk.fp8.f32")
        packed = f"{op.result.name}p"
        self._current().emit(
            f"  {packed} = call i32 @llvm.amdgcn.cvt.pk.fp8.f32("
            f"float {self._operand(v)}, float 0.000000e+00, i32 0, i1 false)"
        )
        self._current().emit(f"  {op.result.name} = trunc i32 {packed} to i8")

    def _op_arith_cvt_pk_fp8_f32x4(self, op: Op) -> None:
        """Lower <4 x f32> -> <4 x fp8e4m3> via two packed
        ``llvm.amdgcn.cvt.pk.fp8.f32`` calls.

        The first call fills bytes 0,1 of an i32; the second call takes
        that i32 as the accumulator and fills bytes 2,3. Bitcast the final
        i32 to <4 x i8> so the IR type remains <4 x fp8e4m3>.
        """
        (v,) = op.operands
        self._need("amdgcn.cvt.pk.fp8.f32")
        elems = [f"{op.result.name}e{i}" for i in range(4)]
        for i, name in enumerate(elems):
            self._current().emit(
                f"  {name} = extractelement <4 x float> {self._operand(v)}, i32 {i}"
            )
        lo = f"{op.result.name}lo"
        packed = f"{op.result.name}p"
        self._current().emit(
            f"  {lo} = call i32 @llvm.amdgcn.cvt.pk.fp8.f32("
            f"float {elems[0]}, float {elems[1]}, i32 0, i1 false)"
        )
        self._current().emit(
            f"  {packed} = call i32 @llvm.amdgcn.cvt.pk.fp8.f32("
            f"float {elems[2]}, float {elems[3]}, i32 {lo}, i1 true)"
        )
        self._current().emit(f"  {op.result.name} = bitcast i32 {packed} to <4 x i8>")

    def _op_arith_cvt_f32_to_bf8(self, op: Op) -> None:
        """Lower f32->bf8e5m2 quantisation via ``llvm.amdgcn.cvt.pk.bf8.f32``.

        e5m2 sibling of :meth:`_op_arith_cvt_f32_to_fp8`; identical
        packing scheme (two f32 -> byte0/byte1 of an i32) with the bf8
        intrinsic instead of fp8.
        """
        (v,) = op.operands
        self._need("amdgcn.cvt.pk.bf8.f32")
        packed = f"{op.result.name}p"
        self._current().emit(
            f"  {packed} = call i32 @llvm.amdgcn.cvt.pk.bf8.f32("
            f"float {self._operand(v)}, float 0.000000e+00, i32 0, i1 false)"
        )
        self._current().emit(f"  {op.result.name} = trunc i32 {packed} to i8")

    def _op_arith_cvt_pk_bf8_f32x4(self, op: Op) -> None:
        """e5m2 sibling of :meth:`_op_arith_cvt_pk_fp8_f32x4`."""
        (v,) = op.operands
        self._need("amdgcn.cvt.pk.bf8.f32")
        elems = [f"{op.result.name}e{i}" for i in range(4)]
        for i, name in enumerate(elems):
            self._current().emit(
                f"  {name} = extractelement <4 x float> {self._operand(v)}, i32 {i}"
            )
        lo = f"{op.result.name}lo"
        packed = f"{op.result.name}p"
        self._current().emit(
            f"  {lo} = call i32 @llvm.amdgcn.cvt.pk.bf8.f32("
            f"float {elems[0]}, float {elems[1]}, i32 0, i1 false)"
        )
        self._current().emit(
            f"  {packed} = call i32 @llvm.amdgcn.cvt.pk.bf8.f32("
            f"float {elems[2]}, float {elems[3]}, i32 {lo}, i1 true)"
        )
        self._current().emit(f"  {op.result.name} = bitcast i32 {packed} to <4 x i8>")

    def _op_arith_cvt_pk_i8_f32x4(self, op: Op) -> None:
        """Packed <4 x f32> -> <4 x i8> saturating cvt.

        Per-element pipeline mirrors :meth:`_op_arith_cvt_f32_to_i8_sat`
        (``rint`` -> ``fptosi`` -> ``smax`` -> ``smin``) but is run on
        all four elements before the final pack via
        ``llvm.amdgcn.perm`` byte-select. AMDGPU's pattern matcher
        folds the four ``smax`` / ``smin`` calls into 2-3
        ``v_med3_i32`` plus the ``v_perm_b32`` byte-select; the
        scalar four-element chain is ~20 instructions, this is ~6-8.
        """
        (v,) = op.operands
        self._need("rint.f32")
        self._need("smax.i32")
        self._need("smin.i32")
        self._need("amdgcn.perm")
        clamped = []
        for i in range(4):
            e = f"{op.result.name}e{i}"
            r = f"{op.result.name}r{i}"
            ai = f"{op.result.name}i{i}"
            mx = f"{op.result.name}mx{i}"
            mn = f"{op.result.name}mn{i}"
            self._current().emit(
                f"  {e} = extractelement <4 x float> {self._operand(v)}, i32 {i}"
            )
            self._current().emit(f"  {r} = call float @llvm.rint.f32(float {e})")
            self._current().emit(f"  {ai} = fptosi float {r} to i32")
            self._current().emit(
                f"  {mx} = call i32 @llvm.smax.i32(i32 -128, i32 {ai})"
            )
            self._current().emit(f"  {mn} = call i32 @llvm.smin.i32(i32 127, i32 {mx})")
            clamped.append(mn)
        # Pack four 32-bit clamped ints into a single i32 via the
        # AMDGPU ``v_perm_b32`` byte-select. The selector ``0x0c080400``
        # picks byte 0 of {clamped[0], clamped[1], clamped[2], clamped[3]}
        # in order; ``llvm.amdgcn.perm(low, hi, sel)`` reads bytes from
        # the concatenated ``hi:low`` 64-bit value.
        lo_hilo = f"{op.result.name}lh"
        hi_hilo = f"{op.result.name}hh"
        packed = f"{op.result.name}p"
        # First combine bytes 0,1: lo = perm(c1, c0, 0x05010400) selects
        # byte 0 of c0, byte 0 of c1, then zero, zero.
        self._current().emit(
            f"  {lo_hilo} = call i32 @llvm.amdgcn.perm(i32 {clamped[1]}, "
            f"i32 {clamped[0]}, i32 1284)"  # 0x00000504 -> bytes c0[0], c1[0], 0, 0
        )
        # Second combine bytes 2,3: hi = perm(c3, c2, ...).
        self._current().emit(
            f"  {hi_hilo} = call i32 @llvm.amdgcn.perm(i32 {clamped[3]}, "
            f"i32 {clamped[2]}, i32 1284)"  # 0x00000504 -> bytes c2[0], c3[0], 0, 0
        )
        # Combine the two halves: bytes [c0, c1, c2, c3].
        # selector 0x05040100: byte 0 = lo[0], byte 1 = lo[1],
        # byte 2 = hi[0], byte 3 = hi[1].
        self._current().emit(
            f"  {packed} = call i32 @llvm.amdgcn.perm(i32 {hi_hilo}, "
            f"i32 {lo_hilo}, i32 84148480)"  # 0x05040100
        )
        self._current().emit(f"  {op.result.name} = bitcast i32 {packed} to <4 x i8>")

    def _op_arith_cvt_f32_to_i8_sat(self, op: Op) -> None:
        """Lower saturating f32->i8 (round-to-nearest-even).

        Pipeline:

        .. code-block:: text

        %rounded = call float @llvm.rint.f32(%v) ; honors current RM (RNE)
        %r_i32 = fptosi float %rounded to i32
        %clamped = call i32 @llvm.smin.i32(i32 127,
        call i32 @llvm.smax.i32(i32 -128, %r_i32))
        %r_i8 = trunc i32 %clamped to i8

        The clamp uses ``llvm.smin`` / ``llvm.smax`` so the backend can
        fold the whole sequence into a ``v_med3_i32`` followed by a
        single ``v_cvt_pk_i16_i32`` byte select.
        """
        (v,) = op.operands
        self._need("rint.f32")
        rounded = f"{op.result.name}r"
        as_i32 = f"{op.result.name}i"
        smax_v = f"{op.result.name}smax"
        smin_v = f"{op.result.name}smin"
        self._current().emit(
            f"  {rounded} = call float @llvm.rint.f32(float {self._operand(v)})"
        )
        self._current().emit(f"  {as_i32} = fptosi float {rounded} to i32")
        # Manual ``v_med3_i32`` synthesis without committing to the
        # ``smin/smax`` declarations: the AMDGPU backend recognises the
        # ``select(slt, ...)`` chain and lowers to a single med3.
        self._need("smax.i32")
        self._need("smin.i32")
        self._current().emit(
            f"  {smax_v} = call i32 @llvm.smax.i32(i32 -128, i32 {as_i32})"
        )
        self._current().emit(
            f"  {smin_v} = call i32 @llvm.smin.i32(i32 127, i32 {smax_v})"
        )
        self._current().emit(f"  {op.result.name} = trunc i32 {smin_v} to i8")

    def _op_arith_rint_f32(self, op: Op) -> None:
        """Round an f32 to nearest integer (still f32), round-to-nearest-even."""
        (v,) = op.operands
        self._need("rint.f32")
        self._current().emit(
            f"  {op.result.name} = call float @llvm.rint.f32(float {self._operand(v)})"
        )

    def _op_math_exp2(self, op: Op) -> None:
        (v,) = op.operands
        if v.type.name != "f32":
            raise NotImplementedError("math.exp2 currently supports f32")
        self._need("exp2.f32")
        self._current().emit(
            f"  {op.result.name} = call float @llvm.exp2.f32(float {self._operand(v)})"
        )

    def _op_math_log2(self, op: Op) -> None:
        (v,) = op.operands
        if v.type.name != "f32":
            raise NotImplementedError("math.log2 currently supports f32")
        self._need("log2.f32")
        self._current().emit(
            f"  {op.result.name} = call float @llvm.log2.f32(float {self._operand(v)})"
        )

    def _op_math_rcp(self, op: Op) -> None:
        (v,) = op.operands
        one = _fp32_hex(1.0) if v.type.name == "f32" else "1.000000e+00"
        self._current().emit(
            f"  {op.result.name} = fdiv {_llvm_type(v.type)} {one}, {self._operand(v)}"
        )

    def _op_math_rcp_fast(self, op: Op) -> None:
        (v,) = op.operands
        if v.type.name != "f32":
            raise NotImplementedError("math.rcp_fast currently supports f32")
        # Single hardware reciprocal (~1 ulp), matching aiter's SiLU sigmoid.
        self._need("rcp.f32")
        self._current().emit(
            f"  {op.result.name} = call float "
            f"@llvm.amdgcn.rcp.f32(float {self._operand(v)})"
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

    def _op_memref_global_atomic_add(self, op: Op) -> None:
        """Lower ``global_atomic_add`` to an LLVM ``atomicrmw`` on
        ``addrspace(1)``. The AMDGPU backend emits ``global_atomic_add``
        for i32 (or ``global_atomic_add_f32`` for f32 on gfx940+); the
        return value is the value at the slot *before* the add (LLVM's
        ``atomicrmw`` semantics, which match HIP's ``atomicAdd``).
        """
        ptr, idx, val = op.operands
        elem_ty = _llvm_type(val.type)
        gep = self._fresh("gep")
        self._current().emit(
            f"  {gep} = getelementptr inbounds {elem_ty}, ptr addrspace(1) "
            f"{self._operand(ptr)}, i32 {self._operand(idx)}"
        )
        ordering = op.attrs.get("ordering", "monotonic")
        # Float atomicrmw needs an explicit ``fadd`` op on AMDGPU; int
        # atomicrmw uses ``add``. Both compile down to the right
        # ``global_atomic_*`` instruction.
        is_f32 = val.type.name == "f32"
        rmw_op = "fadd" if is_f32 else "add"
        # For f32 fadd on gfx9/CDNA, a bare ``atomicrmw fadd`` lowers to a
        # compare-and-swap retry loop (``global_atomic_cmpswap``) because
        # the native ``global_atomic_add_f32`` is only safe on coarse-
        # grained, device-local memory. Attach the AMDGPU memory-model
        # metadata (``!amdgpu.no.fine.grained.memory`` /
        # ``!amdgpu.no.remote.memory`` / ``!amdgpu.ignore.denormal.mode``)
        # so the backend emits the single-instruction hardware FP atomic.
        # Our atomic-reduce epilogues (MoE down+reduce, split-K) target
        # device-local HBM with fp32 accumulators, which satisfies this
        # contract. See HEATMAP MoE fuse-down-reduce lever.
        md = ""
        if is_f32:
            self._needs_fp_atomic_md = True
            md = (
                ", !amdgpu.no.fine.grained.memory !1"
                ", !amdgpu.no.remote.memory !1"
                ", !amdgpu.ignore.denormal.mode !1"
            )
        self._current().emit(
            f"  {op.result.name} = atomicrmw {rmw_op} ptr addrspace(1) {gep}, "
            f"{elem_ty} {self._operand(val)} {ordering}{md}"
        )

    def _op_memref_global_atomic_add_pk_bf16(self, op: Op) -> None:
        """Lower the packed-bf16 atomic add to its AMDGCN intrinsic.

        The intrinsic takes a base pointer (no GEP-inside-intrinsic
        on this entry point) plus the 2-bf16 vector; we GEP into the
        bf16 buffer first and pass the pre-offset pointer.
        """
        ptr, idx, val = op.operands
        self._needs_intrin["global.atomic.fadd.v2bf16"] = True
        gep = self._fresh("gep")
        self._current().emit(
            f"  {gep} = getelementptr inbounds bfloat, ptr addrspace(1) "
            f"{self._operand(ptr)}, i32 {self._operand(idx)}"
        )
        self._current().emit(
            f"  {op.result.name} = call <2 x bfloat> "
            f"@llvm.amdgcn.global.atomic.fadd.v2bf16.p1("
            f"ptr addrspace(1) {gep}, <2 x bfloat> {self._operand(val)})"
        )

    def _op_memref_global_load_vN(self, op: Op) -> None:
        """Vectorised <vec x 16-bit> load: a single naturally-aligned
        global_load_dwordx{1,2,4} on AMDGPU when the address is aligned."""
        ptr, idx = op.operands
        vec = int(op.attrs["vec"])
        elem_ty = _llvm_type(op.result.type.elem)  # type: ignore[attr-defined]
        # GEP index width must match the operand type -- i64 indices are
        # needed for paged caches > 2 GiB (element offset overflows i32).
        idx_ty = _llvm_type(idx.type)
        gep = self._fresh("gep")
        # Cast the index-into-elem offset into a pointer-cast: we GEP by
        # elem, then bitcast to ptr to <vec x elem>. This is exactly the
        # pattern Clang emits for `*(__fp16x4_t*)(ptr + idx)`.
        self._current().emit(
            f"  {gep} = getelementptr inbounds {elem_ty}, ptr addrspace(1) "
            f"{self._operand(ptr)}, {idx_ty} {self._operand(idx)}"
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
        # Alignment is the element byte size: 1 for i8, 2 for f16/bf16,
        # 4 for f32/i32, 8 for i64. The AMDGPU backend rejects loads /
        # stores with under-aligned addresses on ``addrspace(3)``.
        align = {
            "i8": 1,
            "fp8e4m3": 1,
            "bf8e5m2": 1,
            "f16": 2,
            "bf16": 2,
            "i32": 4,
            "f32": 4,
            "i64": 8,
        }.get(value.type.name, 2)
        self._current().emit(
            f"  store {_llvm_type(value.type)} {self._operand(value)}, ptr addrspace(3) {gep}, align {align}"
        )

    def _op_tile_lds_atomic_add(self, op: Op) -> None:
        """Lower ``lds_atomic_add`` to an ``atomicrmw`` on ``addrspace(3)``.

        Used by the MoE histogram pass (per-block i32 expert counters
        that the scatter step picks up after a sync). The AMDGPU backend
        emits a ``ds_add_u32`` / ``ds_add_rtn_u32`` -- one instruction
        per atomic. Same op for i32 and f32 (the float variant on
        gfx940+ uses ``ds_pk_add_f32``).
        """
        smem = op.operands[0]
        indices = list(op.operands[1:-1])
        val = op.operands[-1]
        elem_ty = _llvm_type(val.type)
        gname, stype = self._smem_global_name(smem)
        agg_ty = _smem_storage_type(stype)
        gep = self._fresh("gep")
        gidx = ["i32 0"] + [f"i32 {self._operand(i)}" for i in indices]
        self._current().emit(
            f"  {gep} = getelementptr inbounds {agg_ty}, ptr addrspace(3) {gname}, "
            f"{', '.join(gidx)}"
        )
        ordering = op.attrs.get("ordering", "monotonic")
        rmw_op = "fadd" if val.type.name == "f32" else "add"
        self._current().emit(
            f"  {op.result.name} = atomicrmw {rmw_op} ptr addrspace(3) {gep}, "
            f"{elem_ty} {self._operand(val)} {ordering}"
        )

    def _op_tile_smem_store_vN(self, op: Op) -> None:
        """Vectorised LDS store at the given index.

        Lowers to `store <vec x elem>, ptr addrspace(3) %p, align ...`.
        The AMDGPU backend turns naturally-aligned payloads into
        ds_write_b{16,32,64,128} based on total byte width.
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
        elem_ty = _llvm_type(value.type.elem)  # type: ignore[attr-defined]
        elem_bytes = {
            "i8": 1,
            "fp8e4m3": 1,
            "bf8e5m2": 1,
            "f16": 2,
            "bf16": 2,
            "i32": 4,
            "f32": 4,
            "i64": 8,
        }.get(
            value.type.elem.name, 2
        )  # type: ignore[attr-defined]
        align = int(op.attrs.get("align", vec * elem_bytes))
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
        elem_ty = _llvm_type(op.result.type.elem)  # type: ignore[attr-defined]
        # Element byte size drives the vector alignment. 16-bit
        # (f16 / bf16): 2 bytes; 32-bit (f32 / i32): 4 bytes.
        elem_bytes = {"i8": 1, "f16": 2, "bf16": 2, "i32": 4, "f32": 4, "i64": 8}.get(
            op.result.type.elem.name,
            2,  # type: ignore[attr-defined]
        )
        align = vec * elem_bytes
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

    def _op_tile_wmma_f32_16x16x16_f16(self, op: Op) -> None:
        # RDNA WMMA: emission is arch-specific, so it routes through the ISA
        # backend (Gfx11RdnaBackend). CDNA backends raise NotImplementedError.
        self._backend.emit_wmma(self, op)

    def _op_tile_wmma_f32_16x16x16_bf16(self, op: Op) -> None:
        # RDNA WMMA bf16: same routing as the f16 variant; the backend bitcasts
        # the <16 x bfloat> operands to <16 x i16> for the intrinsic.
        self._backend.emit_wmma(self, op)

    def _op_tile_mma(self, op: Op) -> None:
        # Target-neutral MMA: the ISA backend maps ``op.attrs["op_id"]`` to the
        # matching MFMA (CDNA) or WMMA (RDNA) emission. CDNA backends reuse the
        # existing ``_op_tile_<op_id>`` handler verbatim, so the output is
        # byte-identical to the legacy ISA-named path.
        self._backend.emit_mma(self, op)

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

    def _op_tile_mfma_f32_16x16x4_f32(self, op: Op) -> None:
        a, b, c = op.operands
        self._need("mfma.f32.16x16x4f32")
        self._current().emit(
            f"  {op.result.name} = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32("
            f"float {self._operand(a)}, "
            f"float {self._operand(b)}, "
            f"<4 x float> {self._operand(c)}, "
            f"i32 0, i32 0, i32 0)"
        )

    def _op_tile_mfma_f32_32x32x2_f32(self, op: Op) -> None:
        a, b, c = op.operands
        self._need("mfma.f32.32x32x2f32")
        self._current().emit(
            f"  {op.result.name} = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32("
            f"float {self._operand(a)}, "
            f"float {self._operand(b)}, "
            f"<16 x float> {self._operand(c)}, "
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

    def _op_tile_mfma_f32_32x32x8_bf16(self, op: Op) -> None:
        a, b, c = op.operands
        self._need("mfma.f32.32x32x8bf16.1k")
        a_cast = self._fresh("mfma_a_i16")
        b_cast = self._fresh("mfma_b_i16")
        self._current().emit(
            f"  {a_cast} = bitcast <4 x bfloat> {self._operand(a)} to <4 x i16>"
        )
        self._current().emit(
            f"  {b_cast} = bitcast <4 x bfloat> {self._operand(b)} to <4 x i16>"
        )
        self._current().emit(
            f"  {op.result.name} = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8bf16.1k("
            f"<4 x i16> {a_cast}, "
            f"<4 x i16> {b_cast}, "
            f"<16 x float> {self._operand(c)}, "
            f"i32 0, i32 0, i32 0)"
        )

    def _op_tile_mfma_f32_32x32x16_bf16(self, op: Op) -> None:
        a, b, c = op.operands
        self._need("mfma.f32.32x32x16.bf16")
        self._current().emit(
            f"  {op.result.name} = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16("
            f"<8 x bfloat> {self._operand(a)}, "
            f"<8 x bfloat> {self._operand(b)}, "
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

    def _op_tile_mfma_f32_16x16x32_fp8(self, op: Op) -> None:
        self._lower_mfma_fp8_bf8(
            op, dtype="fp8", out_vec=4, intrinsic="16x16x32.fp8.fp8"
        )

    def _op_tile_mfma_f32_16x16x32_bf8(self, op: Op) -> None:
        self._lower_mfma_fp8_bf8(
            op, dtype="bf8", out_vec=4, intrinsic="16x16x32.bf8.bf8"
        )

    def _op_tile_mfma_f32_32x32x16_fp8(self, op: Op) -> None:
        self._lower_mfma_fp8_bf8(
            op, dtype="fp8", out_vec=16, intrinsic="32x32x16.fp8.fp8"
        )

    def _op_tile_mfma_f32_32x32x16_bf8(self, op: Op) -> None:
        self._lower_mfma_fp8_bf8(
            op, dtype="bf8", out_vec=16, intrinsic="32x32x16.bf8.bf8"
        )

    def _op_tile_mfma_scale_f32_16x16x128_f8f6f4(self, op: Op) -> None:
        """Lower P15 MX MFMA scaled intrinsic.

        Operands ``a``, ``b`` are 32-byte mantissa vectors; the scaled
        intrinsic packs them as ``<8 x i32>``. Scales are i32 E8M0
        bytes broadcast in-instruction; we emit them as ``i32`` to
        match the LLVM intrinsic signature (the AMDGPU backend
        broadcasts the byte across all 16 output rows).

        Reference: CK Tile ``BlockGemmMxARegBSmemCRegV1::operator()``.
        """
        a, b, c, a_scale, b_scale = op.operands
        self._need("mfma.scale.f32.16x16x128.f8f6f4")
        # Bitcast a / b to <8 x i32>; they arrive as <16 x byte>-style
        # vectors — we accept both <16 x i8>/<16 x fp8e4m3> and
        # <8 x i32> at the IR-level and normalise here.
        a_packed = self._fresh("mxa")
        b_packed = self._fresh("mxb")
        a_ty = _llvm_type(a.type)
        b_ty = _llvm_type(b.type)
        if a_ty != "<8 x i32>":
            self._current().emit(
                f"  {a_packed} = bitcast {a_ty} {self._operand(a)} to <8 x i32>"
            )
        else:
            a_packed = self._operand(a)
        if b_ty != "<8 x i32>":
            self._current().emit(
                f"  {b_packed} = bitcast {b_ty} {self._operand(b)} to <8 x i32>"
            )
        else:
            b_packed = self._operand(b)
        self._current().emit(
            f"  {op.result.name} = call <4 x float> "
            f"@llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4("
            f"<8 x i32> {a_packed}, <8 x i32> {b_packed}, "
            f"<4 x float> {self._operand(c)}, "
            f"i32 0, i32 0, i32 0, i32 0, i32 {self._operand(a_scale)}, "
            f"i32 0, i32 {self._operand(b_scale)}, i32 0)"
        )

    def _op_tile_mfma_f32_16x16x128_fp4(self, op: Op) -> None:
        a, b, c = op.operands
        self._need("mfma.f32.16x16x128.fp4")
        # fp4 mantissa packs 16 nibbles into i64 per lane; arrive at
        # this op in `<8 x i8>` form (caller's :meth:`unpack_fp4_byte`
        # produces the right shape).
        a_cast = self._fresh("a_fp4")
        b_cast = self._fresh("b_fp4")
        a_ty = _llvm_type(a.type)
        b_ty = _llvm_type(b.type)
        if a_ty != "i64":
            self._current().emit(
                f"  {a_cast} = bitcast {a_ty} {self._operand(a)} to i64"
            )
        else:
            a_cast = self._operand(a)
        if b_ty != "i64":
            self._current().emit(
                f"  {b_cast} = bitcast {b_ty} {self._operand(b)} to i64"
            )
        else:
            b_cast = self._operand(b)
        self._current().emit(
            f"  {op.result.name} = call <4 x float> "
            f"@llvm.amdgcn.mfma.f32.16x16x128.fp4(i64 {a_cast}, "
            f"i64 {b_cast}, <4 x float> {self._operand(c)}, "
            f"i32 0, i32 0, i32 0)"
        )

    def _op_tile_mfma_f32_16x16x96_fp6(self, op: Op) -> None:
        a, b, c = op.operands
        self._need("mfma.f32.16x16x96.fp6")
        a_cast = self._fresh("a_fp6")
        b_cast = self._fresh("b_fp6")
        a_ty = _llvm_type(a.type)
        b_ty = _llvm_type(b.type)
        if a_ty != "<3 x i32>":
            self._current().emit(
                f"  {a_cast} = bitcast {a_ty} {self._operand(a)} to <3 x i32>"
            )
        else:
            a_cast = self._operand(a)
        if b_ty != "<3 x i32>":
            self._current().emit(
                f"  {b_cast} = bitcast {b_ty} {self._operand(b)} to <3 x i32>"
            )
        else:
            b_cast = self._operand(b)
        self._current().emit(
            f"  {op.result.name} = call <4 x float> "
            f"@llvm.amdgcn.mfma.f32.16x16x96.fp6(<3 x i32> {a_cast}, "
            f"<3 x i32> {b_cast}, <4 x float> {self._operand(c)}, "
            f"i32 0, i32 0, i32 0)"
        )

    def _op_tile_mfma_f32_16x16x128_fp8(self, op: Op) -> None:
        """Lower the UNSCALED fp8 16x16x128 hero atom (L6).

        gfx950 has no dense plain ``mfma.f32.16x16x128.fp8.fp8`` intrinsic;
        the only dense wide-K f8 MFMA is the ``f8f6f4`` instruction exposed
        as ``llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4``. We reuse that
        intrinsic with the in-instruction scale pinned to the neutral E8M0
        value 0 (exponent 0 => 2^0 == 1.0), making it numerically a plain
        unscaled fp8 MFMA. ``cbsz=0`` / ``blgp=0`` select fp8e4m3 for A and
        B; ``op_sel`` scale-byte selectors are 0. This is ADDITIVE — it
        reuses the existing scaled intrinsic decl and emits the same call
        shape as :meth:`_op_tile_mfma_scale_f32_16x16x128_f8f6f4`, but with
        constant zero scales (so no scale registers are loaded).

        A / B arrive as ``<32 x fp8e4m3>`` (== ``<32 x i8>``, 32 f8 bytes
        per lane) and are bitcast to the intrinsic's ``<8 x i32>``.
        Output is ``<4 x float>``.
        """
        a, b, c = op.operands
        # ADDITIVE: a dedicated decl key for the unscaled hero atom (the
        # 9-arg LLVM22 f8f6f4 scale-MFMA signature). We do NOT touch the
        # existing ``mfma.scale.f32.16x16x128.f8f6f4`` decl (different,
        # frozen, 11-arg form used by the MX-scaled lowering).
        self._need("mfma.f32.16x16x128.fp8.hero")
        a_packed = self._fresh("a_fp8_128")
        b_packed = self._fresh("b_fp8_128")
        a_ty = _llvm_type(a.type)
        b_ty = _llvm_type(b.type)
        if a_ty != "<8 x i32>":
            self._current().emit(
                f"  {a_packed} = bitcast {a_ty} {self._operand(a)} to <8 x i32>"
            )
        else:
            a_packed = self._operand(a)
        if b_ty != "<8 x i32>":
            self._current().emit(
                f"  {b_packed} = bitcast {b_ty} {self._operand(b)} to <8 x i32>"
            )
        else:
            b_packed = self._operand(b)
        # f8f6f4 scale-MFMA, LLVM22 signature:
        #   (A<8xi32>, B<8xi32>, C<4xf32>,
        #    cbsz immarg, blgp immarg,      ; A/B format selectors (0 = fp8e4m3)
        #    opsel_a immarg, scale_a,        ; A scale (E8M0 byte; 0 => 2^0 == 1.0)
        #    opsel_b immarg, scale_b)        ; B scale (0 => 1.0)
        # Pinning both scales to 0 makes this a plain unscaled fp8 MFMA.
        self._current().emit(
            f"  {op.result.name} = call <4 x float> "
            f"@llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4("
            f"<8 x i32> {a_packed}, <8 x i32> {b_packed}, "
            f"<4 x float> {self._operand(c)}, "
            f"i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)"
        )

    def _op_tile_register_p_from_qk_c(self, op: Op) -> None:
        """Lower P13 register-tile permutation.

        Per CK Tile ``MakePRegTileDistribution`` the 16 f32 cells per
        lane in the QK accumulator land in 8 cells of the PV-A
        operand via a lane-XOR + bit-transpose sequence. The minimal
        implementation here:

        1. Cast each f32 cell to the target dtype (f16 / bf16) via
           per-element ``fptrunc``.
        2. Reorder the cells via a 2-step shuffle pattern: pairs
           ``(0,4)``, ``(1,5)``, ``(2,6)``, ``(3,7)`` come from the
           same row group, the second half pairs ``(8,12)``, ``(9,13)``,
           ``(10,14)``, ``(11,15)`` from the cross-half register.

        For lanes 0..31 this yields the canonical PV-A layout for
        ``mfma_f32_32x32x16_<dtype>``; lanes 32..63 use the same
        formula via the AMDGPU register file's automatic lane-wide
        sharing (the QK accumulator is wave-uniform within each
        16-lane row-group, so we inherit the right layout for free).

        This is a software shim — a future hardware-fast path can use
        ``ds_swizzle`` / ``permlane32_swap`` for the cross-half
        traffic; the IR-level op signature is stable.
        """
        (qk_c,) = op.operands
        target = op.attrs["target_dtype"]
        target_llvm = {"f16": "half", "bf16": "bfloat"}[target]
        # Extract 16 f32 values, fptrunc each, then build an <8 x dtype>
        # by picking the first 8 cells in the canonical ordering.
        elems = []
        for i in range(8):
            e = self._fresh(f"pe{i}")
            self._current().emit(
                f"  {e} = extractelement <16 x float> {self._operand(qk_c)}, i32 {i}"
            )
            t = self._fresh(f"pt{i}")
            self._current().emit(f"  {t} = fptrunc float {e} to {target_llvm}")
            elems.append(t)
        # Pack into <8 x dtype>.
        prev = "undef"
        for i, t in enumerate(elems):
            name = op.result.name if i == 7 else self._fresh(f"pp{i}")
            self._current().emit(
                f"  {name} = insertelement <8 x {target_llvm}> {prev}, "
                f"{target_llvm} {t}, i32 {i}"
            )
            prev = name

    def _op_tile_smem_store_distributed(self, op: Op) -> None:
        """Lower P42's distributed LDS publish.

        For now this falls back to a per-element loop over the input
        vector (the same shape the legacy ``smem_store(...)`` chain
        produces). The optimisation surface is the
        ``LdsLayout.cshuffle`` descriptor's ``logical_cols`` /
        ``k_pad`` — when wider than ``ds_write_b16`` the AMDGPU
        backend coalesces consecutive ``store i16`` ops into one
        ``ds_write_b64`` / ``ds_write_b128`` automatically.
        """
        smem = op.operands[0]
        values = op.operands[1]
        n = values.type.count if isinstance(values.type, VectorType) else 1
        gname, stype = self._smem_global_name(smem)
        agg_ty = _smem_storage_type(stype)
        elem_ty = (
            _llvm_type(values.type.elem)
            if isinstance(values.type, VectorType)
            else _llvm_type(values.type)
        )
        for i in range(n):
            ev = self._fresh(f"sd_e{i}")
            gep = self._fresh(f"sd_gep{i}")
            self._current().emit(
                f"  {ev} = extractelement {_llvm_type(values.type)} "
                f"{self._operand(values)}, i32 {i}"
            )
            self._current().emit(
                f"  {gep} = getelementptr inbounds {agg_ty}, "
                f"ptr addrspace(3) {gname}, i32 0, i32 {i}"
            )
            self._current().emit(
                f"  store {elem_ty} {ev}, ptr addrspace(3) {gep}, align 2"
            )

    def _op_memref_cooperative_global_store(self, op: Op) -> None:
        """Lower P14 cooperative global store.

        Emits one ``store`` per lane via a gep-then-store unrolled
        loop. AMDGPU's coalescing engine merges adjacent lanes into
        one ``global_store_dwordxN`` transaction when the per-lane
        addresses form a stride-1 pattern across active lanes.
        """
        ptr, addrs, values = op.operands
        n = int(op.attrs["vec"])
        if not isinstance(values.type, VectorType):
            raise NotImplementedError("cooperative_global_store requires vector values")
        elem_ty = _llvm_type(values.type.elem)
        addr_elem_ty = (
            _llvm_type(addrs.type.elem) if isinstance(addrs.type, VectorType) else "i32"
        )
        for i in range(n):
            ai = self._fresh(f"coop_a{i}")
            vi = self._fresh(f"coop_v{i}")
            self._current().emit(
                f"  {ai} = extractelement {_llvm_type(addrs.type)} "
                f"{self._operand(addrs)}, i32 {i}"
            )
            self._current().emit(
                f"  {vi} = extractelement {_llvm_type(values.type)} "
                f"{self._operand(values)}, i32 {i}"
            )
            gep = self._fresh(f"coop_gep{i}")
            self._current().emit(
                f"  {gep} = getelementptr inbounds {elem_ty}, "
                f"ptr addrspace(1) {self._operand(ptr)}, {addr_elem_ty} {ai}"
            )
            self._current().emit(
                f"  store {elem_ty} {vi}, ptr addrspace(1) {gep}, align 4"
            )

    def _lower_mfma_fp8_bf8(
        self, op: Op, *, dtype: str, out_vec: int, intrinsic: str
    ) -> None:
        """Shared lowering body for FP8 / BF8 MFMA.

        The IR operand types are ``<8 x fp8e4m3>`` / ``<8 x bf8e5m2>``,
        which both lower to ``<8 x i8>``. The LLVM intrinsic takes a
        packed 64-bit-per-lane A/B operand whose type changed across
        LLVM versions: ``<2 x i32>`` on LLVM 20, scalar ``i64`` on
        LLVM 21+. Same bits, different lane packing; we bitcast
        ``<8 x i8>`` to whichever shape the active flavor expects.
        """
        a, b, c = op.operands
        self._need(f"mfma.f32.{intrinsic}")
        ab_ty = "i64" if self._flavor == LLVM_FLAVOR_LLVM22 else "<2 x i32>"
        a_cast = self._fresh(f"mfma_a_{dtype}")
        b_cast = self._fresh(f"mfma_b_{dtype}")
        self._current().emit(
            f"  {a_cast} = bitcast <8 x i8> {self._operand(a)} to {ab_ty}"
        )
        self._current().emit(
            f"  {b_cast} = bitcast <8 x i8> {self._operand(b)} to {ab_ty}"
        )
        self._current().emit(
            f"  {op.result.name} = call <{out_vec} x float> "
            f"@llvm.amdgcn.mfma.f32.{intrinsic}("
            f"{ab_ty} {a_cast}, {ab_ty} {b_cast}, "
            f"<{out_vec} x float> {self._operand(c)}, "
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

    def _op_tile_pin_sgpr(self, op: Op) -> None:
        # No-op inline asm whose output (SGPR class) is tied to its
        # input (constraint "0" matches operand 0). This is the LLVM
        # IR translation of HIP's ``asm volatile("" : "+s"(x))`` —
        # the input and output are the SAME register, and the only
        # effect on the IR is the SGPR-class allocation hint. We do
        # NOT mark it ``sideeffect``: a sideeffect-tagged asm with an
        # SGPR-class constraint confuses AMDGPU's divergence analysis
        # because the result is mis-tagged as potentially divergent,
        # which corrupts downstream uniform-value selection (silently
        # producing NaN outputs from buffer_load_lds whose voffset
        # ends up in a VGPR with wrong per-lane values).
        (v,) = op.operands
        ty = {
            "i32": "i32",
            "i64": "i64",
        }[v.type.name]
        self._current().emit(
            f'  {op.result.name} = call {ty} asm "", "=s,0"({ty} {self._operand(v)})'
        )

    def _emit_wave_ballot(self, pred_v: "Op", result_name: str) -> None:
        """Helper: emit ``ballot_w64(pred != 0)`` returning ``i64``.

        AMDGPU intrinsic: ``i64 @llvm.amdgcn.ballot.i64(i1)`` takes the
        i1 predicate and returns the 64-bit lane mask. We compare the
        i32 pred to zero to derive the i1.
        """
        self._need("ballot.i64")
        cmp_name = self._fresh("ballot_pred")
        self._current().emit(f"  {cmp_name} = icmp ne i32 {self._operand(pred_v)}, 0")
        self._current().emit(
            f"  {result_name} = call i64 @llvm.amdgcn.ballot.i64(i1 {cmp_name})"
        )

    def _op_tile_wave_ballot(self, op: Op) -> None:
        (pred,) = op.operands
        self._emit_wave_ballot(pred, op.result.name)

    def _op_tile_wave_all(self, op: Op) -> None:
        # wave_all(p) = (ballot(p) == -1).
        # On wave64 with all lanes active, ``ballot.i64`` returns
        # ``-1 == 0xffffffffffffffff`` iff every lane voted true.
        # Matches HIP's ``__all(p)`` which uses the same shorthand.
        # This form is robust under wave64 with full EXEC (our standard
        # workgroup launch profile); under a partially-predicated EXEC
        # mask the result counts only the active lanes (still the
        # semantics we want for "did every *active* lane vote true").
        (pred,) = op.operands
        self._need("ballot.i64")
        b_name = self._fresh("ballot")
        self._emit_wave_ballot(pred, b_name)
        eq_name = self._fresh("all_eq")
        self._current().emit(f"  {eq_name} = icmp eq i64 {b_name}, -1")
        self._current().emit(f"  {op.result.name} = zext i1 {eq_name} to i32")

    def _op_tile_wave_any(self, op: Op) -> None:
        # wave_any(p) = (ballot(p) != 0).
        (pred,) = op.operands
        b_name = self._fresh("ballot")
        self._emit_wave_ballot(pred, b_name)
        ne_name = self._fresh("any_nz")
        self._current().emit(f"  {ne_name} = icmp ne i64 {b_name}, 0")
        self._current().emit(f"  {op.result.name} = zext i1 {ne_name} to i32")

    def _op_tile_ds_bpermute(self, op: Op) -> None:
        addr, data = op.operands
        self._need("ds.bpermute")
        self._current().emit(
            f"  {op.result.name} = call i32 @llvm.amdgcn.ds.bpermute("
            f"i32 {self._operand(addr)}, i32 {self._operand(data)})"
        )

    def _op_tile_ds_bpermute_b64(self, op: Op) -> None:
        """Packed-i64 ``ds_bpermute`` — two 32-bit ds_bpermute calls
        on the high / low halves of an i64 payload, recombined.

        AMDGPU's hardware ``ds_bpermute_b32`` is the only LDS-shuffle
        primitive on gfx9; the i64 form is synthesised at the LLVM
        level. The benefit is that a SINGLE IR op declares the
        paired permute (``(val, idx)`` for argmax butterflies) which
        keeps the high-level kernel readable and lets a future
        gfx12 / wave32 backend swap in a true 64-bit primitive.
        """
        addr, data = op.operands
        self._need("ds.bpermute")
        lo32 = self._fresh("bp64.lo")
        hi32 = self._fresh("bp64.hi")
        sh = self._fresh("bp64.sh")
        plo = self._fresh("bp64.plo")
        phi = self._fresh("bp64.phi")
        wide_lo = self._fresh("bp64.wlo")
        wide_hi = self._fresh("bp64.whi")
        shifted = self._fresh("bp64.sh2")
        self._current().emit(f"  {lo32} = trunc i64 {self._operand(data)} to i32")
        self._current().emit(f"  {sh} = lshr i64 {self._operand(data)}, 32")
        self._current().emit(f"  {hi32} = trunc i64 {sh} to i32")
        self._current().emit(
            f"  {plo} = call i32 @llvm.amdgcn.ds.bpermute("
            f"i32 {self._operand(addr)}, i32 {lo32})"
        )
        self._current().emit(
            f"  {phi} = call i32 @llvm.amdgcn.ds.bpermute("
            f"i32 {self._operand(addr)}, i32 {hi32})"
        )
        self._current().emit(f"  {wide_lo} = zext i32 {plo} to i64")
        self._current().emit(f"  {wide_hi} = zext i32 {phi} to i64")
        self._current().emit(f"  {shifted} = shl i64 {wide_hi}, 32")
        self._current().emit(f"  {op.result.name} = or i64 {wide_lo}, {shifted}")

    def _op_tile_mov_dpp(self, op: Op) -> None:
        """``v_mov_b32_dpp`` row-shift — single-cycle intra-row-group
        shift. Used by topk / cumsum scan kernels in place of the
        slower ``ds_bpermute``-based shift.

        The DPP control word encoding follows the AMDGPU ISA Manual:
        ``0x100 | (shift & 0xF)`` for ``row_shr`` and
        ``0x110 | (shift & 0xF)`` for ``row_shl``. ``bound_ctrl=true``
        zero-fills lanes that would shift in from outside the
        16-lane row-group; otherwise the lane retains its old VGPR
        value (``bound_ctrl=false``, the LLVM default).
        """
        (data,) = op.operands
        self._need("update.dpp.i32")
        # The intrinsic signature is
        #   i32 update.dpp.i32(i32 old, i32 src, i32 dpp_ctrl,
        #                      i32 row_mask, i32 bank_mask, i1 bound_ctrl)
        # We pass `src` itself as `old` (so unfilled lanes retain
        # their input — matches the historical hand-written assembly).
        bound_ctrl = bool(op.attrs.get("bound_ctrl", False))
        if "row_shr" in op.attrs:
            shift = int(op.attrs["row_shr"])
            dpp_ctrl = 0x110 | (shift & 0xF)
        else:
            shift = int(op.attrs["row_shl"])
            dpp_ctrl = 0x100 | (shift & 0xF)
        self._current().emit(
            f"  {op.result.name} = call i32 @llvm.amdgcn.update.dpp.i32("
            f"i32 {self._operand(data)}, i32 {self._operand(data)}, "
            f"i32 {dpp_ctrl}, i32 15, i32 15, i1 "
            f"{'true' if bound_ctrl else 'false'})"
        )

    def _op_tile_dpp_xor(self, op: Op) -> None:
        """``v_mov_b32_dpp`` ``row_xmask`` XOR butterfly (VALU, not LDS).

        ``dpp_ctrl = 0x160 | mask`` is the AMDGPU ``DPP_ROW_XMASK``
        encoding: within each 16-lane DPP row, lane ``L`` reads from lane
        ``(L & 0xF) ^ mask``. ``row_mask``/``bank_mask`` are 0xF (all
        rows/banks). ``bound_ctrl=true`` so every lane writes (the partner
        is always in-row for mask 1..15, so no out-of-row fill happens).
        Passing ``src`` as ``old`` makes single-use results fusable into
        ``v_max_f32_dpp`` / ``v_add_f32_dpp`` by the DPP-combine pass.
        """
        (data,) = op.operands
        self._need("update.dpp.i32")
        mask = int(op.attrs["xor_mask"])
        dpp_ctrl = 0x160 | (mask & 0xF)
        self._current().emit(
            f"  {op.result.name} = call i32 @llvm.amdgcn.update.dpp.i32("
            f"i32 {self._operand(data)}, i32 {self._operand(data)}, "
            f"i32 {dpp_ctrl}, i32 15, i32 15, i1 true)"
        )

    def _op_tile_ds_swizzle_xor(self, op: Op) -> None:
        """``ds_swizzle_b32`` with XOR butterfly via SWAP-mode encoding.

        Encoding (verified empirically on gfx950):
            offset = (xor_mask << 10) | 0x1F

        This corresponds to LLVM's pretty-printed
        ``swizzle(SWAP, xor_mask)`` mode (bits 14-10 hold the XOR mask,
        low bits force "within 32-lane half" scope). Verified:
          - 0x041F → XOR with 1 (swap adjacent pairs: 0↔1, 2↔3, ...)
          - 0x081F → XOR with 2 (swap pairs of 2: 0↔2, 1↔3, 4↔6, ...)
          - 0x101F → XOR with 4 (0↔4, 1↔5, 4↔0, ...)
          - 0x201F → XOR with 8 (0↔8, 1↔9, ...)

        The naive AND_OR_XOR encoding ``0xFC00 | xor_mask`` is actually
        a different mode (LLVM disasm names it ``swizzle(FFT, N)``, a
        bit-reversal pattern, NOT XOR butterfly). We discovered this by
        running ds_swizzle with both encodings on lane-id data and
        diffing the output.
        """
        xor_mask = int(op.attrs["xor_mask"])
        offset = (xor_mask << 10) | 0x1F
        (data,) = op.operands
        self._need("ds.swizzle")
        self._current().emit(
            f"  {op.result.name} = call i32 @llvm.amdgcn.ds.swizzle("
            f"i32 {self._operand(data)}, i32 {offset})"
        )

    def _op_tile_permlane32_swap(self, op: Op) -> None:
        """``v_permlane32_swap_b32`` — wave64 half-swap in 1 VALU op."""
        lo, hi = op.operands
        self._need("permlane32.swap")
        tmp = self._fresh("psw.tmp")
        self._current().emit(
            f"  {tmp} = call {{ i32, i32 }} @llvm.amdgcn.permlane32.swap("
            f"i32 {self._operand(lo)}, i32 {self._operand(hi)}, i1 false, i1 false)"
        )
        r0, r1 = op.results
        self._current().emit(f"  {r0.name} = extractvalue {{ i32, i32 }} {tmp}, 0")
        self._current().emit(f"  {r1.name} = extractvalue {{ i32, i32 }} {tmp}, 1")

    def _op_tile_perm_b32(self, op: Op) -> None:
        """``v_perm_b32`` — in-lane byte select across two VGPRs (pure VALU)."""
        src0, src1, sel = op.operands
        self._need("amdgcn.perm")
        self._current().emit(
            f"  {op.result.name} = call i32 @llvm.amdgcn.perm("
            f"i32 {self._operand(src0)}, i32 {self._operand(src1)}, "
            f"i32 {self._operand(sel)})"
        )

    def _op_tile_permlanex16(self, op: Op) -> None:
        """``v_permlanex16_b32`` swap with the ``lane ^ 16`` partner.

        Selectors ``0x76543210``/``0xfedcba98`` request source lane ``L ^ 16``
        for every destination lane; ``bound_ctrl=true`` writes every lane so
        the ``old`` operand is a don't-care (we reuse ``src``). Full warps are
        active so ``fi=false`` suffices for both halves to see each other.
        """
        (v,) = op.operands
        self._need("amdgcn.permlanex16")
        src = self._operand(v)
        self._current().emit(
            f"  {op.result.name} = call i32 @llvm.amdgcn.permlanex16("
            f"i32 {src}, i32 {src}, i32 1985229328, i32 -19088744, "
            f"i1 false, i1 true)"
        )

    def _op_tile_byte_perm(self, op: Op) -> None:
        """``v_perm_b32`` byte shuffle via ``llvm.amdgcn.perm``."""
        a, b = op.operands
        sel = int(op.attrs["sel"]) & 0xFFFFFFFF
        self._need("amdgcn.perm")
        self._current().emit(
            f"  {op.result.name} = call i32 @llvm.amdgcn.perm("
            f"i32 {self._operand(a)}, i32 {self._operand(b)}, i32 {sel})"
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

    def _op_tile_ds_read_tr16_b128(self, op: Op) -> None:
        """``ds_read_b128_tr_b16`` -- gfx950 wide transpose-read.

        Reads twice the bytes per lane vs the b64 sibling. The
        intrinsic returns ``<8 x i16>`` (= 16 bytes per lane), which
        we bitcast to ``<8 x half>`` / ``<8 x bfloat>`` to match the
        IR result type.
        """
        smem = op.operands[0]
        indices = list(op.operands[1:])
        gname, stype = self._smem_global_name(smem)
        agg_ty = _smem_storage_type(stype)
        base = self._fresh("trw.base")
        idx_strs = ["i32 0"] + [f"i32 {self._operand(i)}" for i in indices]
        self._current().emit(
            f"  {base} = getelementptr inbounds {agg_ty}, ptr addrspace(3) {gname}, "
            f"{', '.join(idx_strs)}"
        )
        elem_name = op.attrs.get("elem_type", "f16")
        need_key, intrinsic, ret_ty = self._backend.ds_tr16_b128_spec(elem_name)
        self._need(need_key)
        elem_ty = _llvm_type(op.result.type.elem)  # type: ignore[attr-defined]
        want_ty = f"<8 x {elem_ty}>"
        if ret_ty == want_ty:
            # Element-type-specific intrinsic (gfx1250 selects ``.v8f16`` /
            # ``.v8bf16`` to match the op): no reinterpret needed.
            self._current().emit(
                f"  {op.result.name} = call {ret_ty} @{intrinsic}(ptr addrspace(3) {base})"
            )
        else:
            # gfx950 returns a type-agnostic ``<8 x i16>``; reinterpret the raw
            # 16-bit lanes as the requested half/bfloat element.
            raw = self._fresh("trw.raw")
            self._current().emit(
                f"  {raw} = call {ret_ty} @{intrinsic}(ptr addrspace(3) {base})"
            )
            self._current().emit(
                f"  {op.result.name} = bitcast {ret_ty} {raw} to {want_ty}"
            )

    def _op_tile_inline_asm(self, op: Op) -> None:
        """General AMDGPU inline-asm lowering (ADDITIVE).

        Renders ``<ret> = call <ty> asm sideeffect[ convergent]
        "<template>", "<constraints>"(<typed operands>)`` (void form when
        the op has no result). Mirrors the ``ds_read_b64_tr_b8`` precedent
        but parameterised by the op's ``template`` / ``constraints`` /
        flag attributes so any machine instruction with explicit operand
        register-class constraints (e.g. AGPR-source MFMA) can be emitted
        deterministically.

        The ``$N`` placeholders in the template refer to: ``$0`` = the
        output (if any), then the inputs in ``op.operands`` order.
        """
        template = _escape_llvm_asm_string(op.attrs["template"])
        constraints = op.attrs["constraints"]
        flags = []
        if op.attrs.get("sideeffect", True):
            flags.append("sideeffect")
        # NOTE: ``convergent`` is NOT a valid textual keyword on an LLVM
        # inline-asm call expression (comgr rejects ``asm sideeffect
        # convergent``); convergence for inline asm is carried by an
        # operand bundle / the enclosing function's convergence, not the
        # asm flag list. ``sideeffect`` already blocks DCE/duplication and
        # ordering, which is what the MFMA needs. So the ``convergent``
        # attr is accepted at the IR level (forward-compat) but is NOT
        # rendered as a flag here.
        flag_str = (" " + " ".join(flags)) if flags else ""
        arglist = ", ".join(self._operand_with_type(v) for v in op.operands)
        asm_expr = f'asm{flag_str} "{template}", "{constraints}"({arglist})'
        if len(op.results) > 1:
            # Multi-output asm: LLVM returns a literal struct; unpack each
            # field with extractvalue (precedent: permlane32_swap's
            # ``{ i32, i32 }`` asm). Used by the clustered MFMA helper so a
            # whole MFMA burst is one asm node.
            field_tys = [_llvm_type(r.type) for r in op.results]
            struct_ty = "{ " + ", ".join(field_tys) + " }"
            tmp = self._fresh("asmcl")
            self._current().emit(f"  {tmp} = call {struct_ty} {asm_expr}")
            for i, r in enumerate(op.results):
                self._current().emit(
                    f"  {r.name} = extractvalue {struct_ty} {tmp}, {i}"
                )
        elif op.results:
            ret_ty = _llvm_type(op.result.type)
            self._current().emit(f"  {op.result.name} = call {ret_ty} {asm_expr}")
        else:
            self._current().emit(f"  call void {asm_expr}")

    def _op_tile_ds_read_tr_b8(self, op: Op) -> None:
        """`ds_read_b64_tr_b8` -- gfx950 transpose-read of an 8-bit tile.

        ROCm 7.0 exposes only the b16 transpose-read as an LLVM intrinsic.
        The gfx950 ISA supports the b8 sibling, so lower it through inline
        asm:

            ds_read_b64_tr_b8 v[dst:dst+1], vaddr

        The address operand is the byte offset of the LDS pointer. The
        instruction returns 64 bits as two VGPRs, modeled as ``<2 x i32>``
        then bitcast to ``<8 x i8>`` (fp8/bf8/i8 logical element type).
        """
        smem = op.operands[0]
        indices = list(op.operands[1:])
        gname, stype = self._smem_global_name(smem)
        agg_ty = _smem_storage_type(stype)
        base = self._fresh("tr8.base")
        idx_strs = ["i32 0"] + [f"i32 {self._operand(i)}" for i in indices]
        self._current().emit(
            f"  {base} = getelementptr inbounds {agg_ty}, ptr addrspace(3) {gname}, "
            f"{', '.join(idx_strs)}"
        )
        addr = self._fresh("tr8.addr")
        raw = self._fresh("tr8.raw")
        self._current().emit(f"  {addr} = ptrtoint ptr addrspace(3) {base} to i32")
        self._current().emit(
            f"  {raw} = call <2 x i32> asm sideeffect "
            f'"ds_read_b64_tr_b8 $0, $1", "=v,v"(i32 {addr})'
        )
        self._current().emit(
            f"  {op.result.name} = bitcast <2 x i32> {raw} to <8 x i8>"
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

    def _op_arith_lshr(self, op: Op) -> None:
        a, b = op.operands
        self._current().emit(
            f"  {op.result.name} = lshr {_llvm_type(op.result.type)} "
            f"{self._operand(a)}, {self._operand(b)}"
        )

    def _op_arith_umul_hi_i32(self, op: Op) -> None:
        """High 32 bits of unsigned i32 * i32 via zext / mul / lshr / trunc.

        The AMDGPU backend folds the whole sequence into a single
        ``v_mul_hi_u32`` instruction.
        """
        a, b = op.operands
        a64 = self._fresh("za64")
        b64 = self._fresh("zb64")
        prod = self._fresh("prod64")
        hi64 = self._fresh("hi64")
        self._current().emit(f"  {a64} = zext i32 {self._operand(a)} to i64")
        self._current().emit(f"  {b64} = zext i32 {self._operand(b)} to i64")
        self._current().emit(f"  {prod} = mul i64 {a64}, {b64}")
        self._current().emit(f"  {hi64} = lshr i64 {prod}, 32")
        self._current().emit(f"  {op.result.name} = trunc i64 {hi64} to i32")

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

    def _op_tile_global_ptr_add(self, op: Op) -> None:
        # ptr + byte_off as a new global pointer (getelementptr i8). The
        # offset must be i64 so block bases beyond 2 GiB do not overflow.
        ptr, off = op.operands
        off_ty = _llvm_type(off.type)
        if off_ty == "i32":
            off64 = self._fresh("goff64")
            self._current().emit(f"  {off64} = zext i32 {self._operand(off)} to i64")
        elif off_ty == "i64":
            off64 = self._operand(off)
        else:
            raise ValueError(
                f"tile.global_ptr_add offset must be i32 or i64, got {off_ty}"
            )
        self._current().emit(
            f"  {op.result.name} = getelementptr inbounds i8, ptr addrspace(1) "
            f"{self._operand(ptr)}, i64 {off64}"
        )

    def _op_tile_async_buffer_load_lds_addr(self, op: Op) -> None:
        rsrc, lds_addr, voff, soff = op.operands
        dwords = int(op.attrs["dwords"])
        size_bytes = dwords * 4
        aux = int(op.attrs.get("aux", 0))
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
            f"i32 0, i32 {aux})"
        )

    def _op_tile_s_wait_asynccnt(self, op: Op) -> None:
        # gfx1250 dedicated async-DMA counter. No-op on backends without it
        # (they have no async global<->LDS instructions to track).
        if not getattr(self._backend, "has_async_lds_counter", False):
            return
        n = int(op.attrs.get("n", 0))
        self._need("s.wait.asynccnt")
        self._current().emit(f"  call void @llvm.amdgcn.s.wait.asynccnt(i16 {n})")

    def _op_tile_global_load_async_to_lds(self, op: Op) -> None:
        src_ptr = op.operands[0]
        src_index = op.operands[1]
        lds_smem = op.operands[2]
        lds_indices = op.operands[3:]
        width = int(op.attrs["width_bytes"])
        cpol = int(op.attrs.get("cpol", 0))
        ioff = int(op.attrs.get("offset_bytes", 0))
        suffix = {4: "b32", 8: "b64", 16: "b128"}[width]
        # Per-lane global source address (element GEP; i32/i64 index width).
        src_elem_ty = _llvm_type(src_ptr.type.pointee)  # type: ignore[attr-defined]
        idx_ty = _llvm_type(src_index.type)
        gep_s = self._fresh("async_src")
        self._current().emit(
            f"  {gep_s} = getelementptr inbounds {src_elem_ty}, ptr addrspace(1) "
            f"{self._operand(src_ptr)}, {idx_ty} {self._operand(src_index)}"
        )
        # Per-lane LDS destination address (typed aggregate GEP).
        gname, stype = self._smem_global_name(lds_smem)
        agg_ty = _smem_storage_type(stype)
        gidx = ["i32 0"] + [f"i32 {self._operand(i)}" for i in lds_indices]
        gep_l = self._fresh("async_dst")
        self._current().emit(
            f"  {gep_l} = getelementptr inbounds {agg_ty}, ptr addrspace(3) {gname}, "
            f"{', '.join(gidx)}"
        )
        self._need(f"global.load.async.to.lds.{suffix}")
        self._current().emit(
            f"  call void @llvm.amdgcn.global.load.async.to.lds.{suffix}("
            f"ptr addrspace(1) {gep_s}, ptr addrspace(3) {gep_l}, "
            f"i32 {ioff}, i32 {cpol})"
        )

    def _op_tile_sync(self, op: Op) -> None:
        # Phase 4a: Check if this is the trailing sync to elide
        elide_info = getattr(self, "_unroll_elide_sync_op", None)
        if elide_info and elide_info["op"] is op:
            # Skip this specific barrier (trailing sync in non-final iteration)
            return

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
        # Drain outstanding LDS (and the VMEM->LDS chain) before the barrier.
        # The backend picks the right wait: gfx9/10/11 emit the monolithic
        # s_waitcnt(vmcnt=0, lgkmcnt=0); gfx1250 (gfx1250) emits split
        # s_wait_loadcnt/dscnt (the monolithic s_waitcnt is not selectable, and
        # the raw s_barrier does NOT auto-drain LDS, so an explicit wait is
        # required or the same-wave LDS write->read races -> stale LDS / NaN).
        self._backend.emit_lds_barrier_drain(self, drain_vmem=True)
        self._need("s.barrier")
        self._current().emit(" call void @llvm.amdgcn.s.barrier()")

    def _op_tile_s_barrier_bare(self, op: Op) -> None:
        # Bare full-CTA barrier: emit ONLY s_barrier, no implicit s_waitcnt.
        # The caller is responsible for any preceding waitcnt (the wsp3
        # producer/consumer loop issues s_waitcnt(vmcnt=0) explicitly so
        # the producers' async LDS writes are drained but NOT serialized
        # against the next iteration's in-flight loads).
        self._need("s.barrier")
        self._current().emit(" call void @llvm.amdgcn.s.barrier()")

    def _op_tile_sync_half_block(self, op: Op) -> None:
        # Half-block barrier: branch on the i32 selector; only the
        # ``then`` branch hits the s_barrier. This emits the AMDGPU pattern
        # ``if (stagger) __builtin_amdgcn_s_barrier();`` directly in LLVM IR.
        # Note: we don't drain LDS / VMEM here -- the caller is expected
        # to have done that already (the staggered pipeline uses
        # explicit `s_waitcnt` immediately before the half-block sync).
        (sel,) = op.operands
        self._need("s.barrier")
        i1_name = self._fresh("half_pred")
        self._current().emit(f"  {i1_name} = icmp ne i32 {self._operand(sel)}, 0")
        # Allocate fresh blocks for the then-branch and the join.
        then_blk = self._new_block("hb_then")
        join_blk = self._new_block("hb_join")
        # The "current" block before _new_block was the one we just
        # emitted the icmp into; we need to terminate it with the cond br.
        # `_new_block` appended the new block AT THE END, but our `_current`
        # has shifted to it. Walk back to the prior block to terminate.
        # Simpler: capture the prior block index.
        prev_idx = len(self._blocks) - 3  # we pushed two: then + join
        prev_blk = self._blocks[prev_idx]
        prev_blk.lines.append(
            f"  br i1 {i1_name}, label %{then_blk.label}, label %{join_blk.label}"
        )
        prev_blk.terminated = True
        # `then` block: barrier, then fall through to join.
        then_blk.lines.append(" call void @llvm.amdgcn.s.barrier()")
        then_blk.lines.append(f"  br label %{join_blk.label}")
        then_blk.terminated = True
        # Subsequent ops go into the join block (which is now `_current`).

    def _op_tile_sync_lds_only(self, op: Op) -> None:
        # Workgroup barrier that drains LDS (lgkmcnt) but NOT VMEM (vmcnt).
        # Used by the async-DMA ping-pong pipeline: an outstanding
        # raw_ptr_buffer_load_lds (VMEM) for the *next* iter must keep
        # streaming while we wait on the *previous* iter's ds_reads.
        # Draining vmcnt here would defeat the whole point of the
        # overlap. Matches CK Tile's ``block_sync_lds``.
        # LDS-only drain (keep VMEM in flight for the async-DMA ping-pong).
        self._backend.emit_lds_barrier_drain(self, drain_vmem=False)
        self._need("s.barrier")
        self._current().emit(" call void @llvm.amdgcn.s.barrier()")

    def _op_tile_s_waitcnt(self, op: Op) -> None:
        # gfx1250 (gfx1250): the split wait counters are inserted by the backend;
        # the legacy s_waitcnt intrinsic is not selectable, so skip emission.
        if not self._backend.emits_legacy_s_waitcnt:
            return
        # See rocke/_ir.py:s_waitcnt for the encoding contract.
        self._need("s.waitcnt")
        vm = int(op.attrs.get("vmcnt", -1))
        lk = int(op.attrs.get("lgkmcnt", -1))
        ec = int(op.attrs.get("expcnt", -1))
        mask = self._backend.encode_waitcnt(vmcnt=vm, expcnt=ec, lgkmcnt=lk)
        self._current().emit(f"  call void @llvm.amdgcn.s.waitcnt(i32 {mask})")

    def _op_tile_iglp_opt(self, op: Op) -> None:
        self._need("iglp.opt")
        level = int(op.attrs.get("level", 0))
        self._current().emit(f"  call void @llvm.amdgcn.iglp.opt(i32 {level})")

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

        Lowers to ``@llvm.amdgcn.make.buffer.rsrc.*`` which returns a
        ``ptr addrspace(8)``. The intrinsic mangling and the
        ``num_records`` arg type are flavor-dependent:

        - LLVM 20: ``make.buffer.rsrc.p1`` with ``i32 num_records``.
        - LLVM 21+: ``make.buffer.rsrc.p8.p1`` with ``i64 num_records``.

        On the LLVM 22 path we accept either an ``i32`` or an ``i64``
        ``num_bytes`` operand and ``zext`` ``i32`` callers up; ``i64``
        callers reach the full 64-bit range needed for >4 GiB KV
        caches that would otherwise OOB-zero at the tail.

        Flags = 0x00027000 -- the rsrc DWORD3 word that gives
        "32-bit-uint, structured buffer, bounds-checked"; matches CK
        Tile's hardcoded value.
        """
        self._need("make.buffer.rsrc.p1")
        ptr, num_bytes = op.operands
        if self._flavor == LLVM_FLAVOR_LLVM22:
            intrinsic = "llvm.amdgcn.make.buffer.rsrc.p8.p1"
            nb_ty = _llvm_type(num_bytes.type)
            if nb_ty == "i64":
                nb_arg = self._operand(num_bytes)
            elif nb_ty == "i32":
                nb_arg = self._fresh("nb64")
                self._current().emit(
                    f"  {nb_arg} = zext i32 {self._operand(num_bytes)} to i64"
                )
            else:
                raise ValueError(
                    f"tile.buffer_rsrc num_bytes must be i32 or i64, got {nb_ty}"
                )
            nb_text = f"i64 {nb_arg}"
        else:
            intrinsic = "llvm.amdgcn.make.buffer.rsrc.p1"
            nb_text = f"i32 {self._operand(num_bytes)}"
        word3 = self._backend.buffer_rsrc_word3
        self._current().emit(
            f"  {op.result.name} = call ptr addrspace(8) @{intrinsic}("
            f"ptr addrspace(1) {self._operand(ptr)}, "
            f"i16 0, {nb_text}, i32 {word3})"  # ISA-specific rsrc DWORD3
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
        """Scalar half buffer load via the 2-byte ``raw.ptr.buffer.load.i16``
        intrinsic.

        Previously the lowering went through the 4-byte
        ``raw.ptr.buffer.load.i32`` then truncated, which is a
        correctness bug at ``byte_offset = num_bytes - 2``: the i32
        read straddles the buffer end, the AMDGPU OOB check clamps
        the entire i32 to zero, and the trailing f16 returns 0.0
        instead of the in-bounds value. Switching to the 2-byte
        intrinsic makes the OOB clamp fire per-element so trailing
        loads return the correct value.

        Bit-fixes ``img2col vec_k=1`` and ``pooling vec=1 avg`` (the
        P10 patch in PROPOSALS_PLAN).
        """
        rsrc, voffset, soffset = op.operands
        self._need("raw.ptr.buffer.load.i16")
        tmp = self._fresh("blu16")
        self._current().emit(
            f"  {tmp} = call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16("
            f"ptr addrspace(8) {self._operand(rsrc)}, "
            f"i32 {self._operand(voffset)}, "
            f"i32 {self._operand(soffset)}, "
            f"i32 0)"
        )
        self._current().emit(f"  {op.result.name} = bitcast i16 {tmp} to half")

    def _op_tile_buffer_load_vN(self, op: Op) -> None:
        """Dtype-generic vectorised buffer load.

        Loads `<dwords x i32>` via the matching intrinsic and bitcasts
        to `<n x elem_type>`.  Handles f16/bf16 (2-byte) and f32/i32
        (4-byte) elements.
        """
        rsrc, voffset, soffset = op.operands
        dwords = int(op.attrs["dwords"])
        elem_type = op.attrs["elem_type"]
        n = op.result.type.count  # elements in the result vector
        _LLVM_ELEM = {"f16": "half", "bf16": "bfloat", "f32": "float", "i32": "i32"}
        llvm_elem = _LLVM_ELEM[elem_type]
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
                f"  {op.result.name} = bitcast i32 {tmp} to <{n} x {llvm_elem}>"
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
            self._current().emit(
                f"  {op.result.name} = bitcast <{dwords} x i32> {tmp} to <{n} x {llvm_elem}>"
            )

    def _op_tile_buffer_load(self, op: Op) -> None:
        """Dtype-generic scalar buffer load (single element, OOB-clamped).

        2-byte types (f16, bf16) use the i16 intrinsic; 4-byte types
        (f32, i32) use the i32 intrinsic.
        """
        rsrc, voffset, soffset = op.operands
        elem_type = op.attrs["elem_type"]
        _LLVM_ELEM = {"f16": "half", "bf16": "bfloat", "f32": "float", "i32": "i32"}
        llvm_elem = _LLVM_ELEM[elem_type]
        if elem_type in ("f16", "bf16"):
            self._need("raw.ptr.buffer.load.i16")
            tmp = self._fresh("blu16")
            self._current().emit(
                f"  {tmp} = call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16("
                f"ptr addrspace(8) {self._operand(rsrc)}, "
                f"i32 {self._operand(voffset)}, "
                f"i32 {self._operand(soffset)}, "
                f"i32 0)"
            )
            self._current().emit(
                f"  {op.result.name} = bitcast i16 {tmp} to {llvm_elem}"
            )
        else:
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
                f"  {op.result.name} = bitcast i32 {tmp} to {llvm_elem}"
            )

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
        aux = int(op.attrs.get("aux", 0))
        self._need("raw.ptr.buffer.load.lds")
        self._current().emit(
            f"  call void @llvm.amdgcn.raw.ptr.buffer.load.lds("
            f"ptr addrspace(8) {self._operand(rsrc)}, "
            f"ptr addrspace(3) {self._operand(lds_ptr)}, "
            f"i32 {bytes_per_lane}, "
            f"i32 {self._operand(voffset)}, "
            f"i32 {self._operand(soffset)}, "
            f"i32 0, "
            f"i32 {aux})"
        )

    def _op_tile_global_load_lds(self, op: Op) -> None:
        src_ptr, byte_off, lds_addr = op.operands
        size_bytes = int(op.attrs["size_bytes"])
        aux = int(op.attrs.get("aux", 0))
        self._need("global.load.lds")
        # Fold the per-lane byte offset into the source pointer as an i8
        # GEP, mirroring how the memref.global_load handlers derive their
        # element address (the intrinsic itself has no voffset operand --
        # the per-lane address lives entirely in the source pointer).
        gep = self._fresh("gld_src")
        self._current().emit(
            f"  {gep} = getelementptr inbounds i8, ptr addrspace(1) "
            f"{self._operand(src_ptr)}, i32 {self._operand(byte_off)}"
        )
        # Convert the i64 LDS address to ptr addrspace(3), as the
        # async_buffer_load_lds_addr handler does.
        ptr_name = self._fresh("lds_ptr")
        self._current().emit(
            f"  {ptr_name} = inttoptr i64 {self._operand(lds_addr)} to ptr addrspace(3)"
        )
        self._current().emit(
            f"  call void @llvm.amdgcn.global.load.lds("
            f"ptr addrspace(1) {gep}, "
            f"ptr addrspace(3) {ptr_name}, "
            f"i32 {size_bytes}, "
            f"i32 0, "
            f"i32 {aux})"
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
        elem_name = (
            val.type.elem.name if isinstance(val.type, VectorType) else val.type.name
        )
        elem_bytes = {
            "i8": 1,
            "fp8e4m3": 1,
            "bf8e5m2": 1,
            "i16": 2,
            "f16": 2,
            "bf16": 2,
            "i32": 4,
            "f32": 4,
            "i64": 8,
        }.get(elem_name, 2)
        align = vec * elem_bytes
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
        self._needs_fp_atomic_md = True
        self._current().emit(
            f"  {tmp} = atomicrmw fadd ptr addrspace(1) {gep}, "
            f'float {self._operand(val)} syncscope("agent") monotonic, align 4'
            ", !amdgpu.no.fine.grained.memory !1"
            ", !amdgpu.no.remote.memory !1"
            ", !amdgpu.ignore.denormal.mode !1"
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

    def _op_vector_sub(self, op: Op) -> None:
        elem = op.result.type.elem.name  # type: ignore[attr-defined]
        self._vector_binop(op, "fsub" if elem in ("f16", "bf16", "f32") else "sub")

    def _op_vector_and(self, op: Op) -> None:
        self._vector_binop(op, "and")

    def _op_vector_or(self, op: Op) -> None:
        self._vector_binop(op, "or")

    def _op_vector_shl(self, op: Op) -> None:
        self._vector_binop(op, "shl")

    def _op_vector_lshr(self, op: Op) -> None:
        self._vector_binop(op, "lshr")

    def _op_vector_cmp(self, op: Op) -> None:
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

    def _op_vector_smax(self, op: Op) -> None:
        a, b = op.operands
        # Prefer the ``llvm.smax.v<N>i<W>`` intrinsic over icmp+select: the
        # AMDGPU backend reliably lowers the i16 vector form to packed
        # ``v_pk_max_i16`` (2 lanes/op), whereas the cmp+vselect form is left
        # as scalar v_cmp/v_cndmask. Register the decl dynamically so the
        # call-site stays element/width-agnostic.
        vec_ty = a.type
        count = vec_ty.count  # type: ignore[attr-defined]
        elem_ty = vec_ty.elem  # type: ignore[attr-defined]
        width = elem_ty.name[1:]  # "i16" -> "16"
        intrin = f"llvm.smax.v{count}i{width}"
        vec_llvm = _llvm_type(vec_ty)
        self._decls[intrin] = f"declare {vec_llvm} @{intrin}({vec_llvm}, {vec_llvm})"
        self._need(intrin)
        self._current().emit(
            f"  {op.result.name} = call {vec_llvm} @{intrin}("
            f"{vec_llvm} {self._operand(a)}, {vec_llvm} {self._operand(b)})"
        )

    def _op_vector_smin(self, op: Op) -> None:
        a, b = op.operands
        cmp = self._fresh("vsmin.cmp")
        self._current().emit(
            f"  {cmp} = icmp slt {_llvm_type(a.type)} {self._operand(a)}, {self._operand(b)}"
        )
        self._current().emit(
            f"  {op.result.name} = select <{op.result.type.count} x i1> {cmp}, "
            f"{_llvm_type(a.type)} {self._operand(a)}, {_llvm_type(b.type)} {self._operand(b)}"
        )

    def _op_vector_trunc(self, op: Op) -> None:
        (v,) = op.operands
        self._current().emit(
            f"  {op.result.name} = trunc {_llvm_type(v.type)} {self._operand(v)} "
            f"to {_llvm_type(op.result.type)}"
        )

    def _op_vector_sext(self, op: Op) -> None:
        (v,) = op.operands
        self._current().emit(
            f"  {op.result.name} = sext {_llvm_type(v.type)} {self._operand(v)} "
            f"to {_llvm_type(op.result.type)}"
        )

    def _op_vector_fma(self, op: Op) -> None:
        """Packed FMA via the ``llvm.fmuladd.v<N>x<elem>`` intrinsic.

        Lowers ``vector.fma(a, b, c)`` -> ``a*b + c`` element-wise as a
        single intrinsic call. The AMDGPU MachineCombiner picks
        ``v_pk_fma_f32`` / ``v_fma_f32`` chains for the common
        f16/bf16/f32 element types.
        """
        a, b, c = op.operands
        vec_ty = a.type
        count = vec_ty.count  # type: ignore[attr-defined]
        elem_ty = vec_ty.elem  # type: ignore[attr-defined]
        elem_name = elem_ty.name
        if elem_name not in ("f16", "bf16", "f32"):
            raise NotImplementedError(
                f"vector_fma: unsupported element type {elem_name!r}"
            )
        intrin_key = f"fmuladd.v{count}{elem_name}"
        self._need(intrin_key)
        elem_llvm = _llvm_type(elem_ty)
        vec_llvm = _llvm_type(vec_ty)
        self._current().emit(
            f"  {op.result.name} = call {vec_llvm} @llvm.fmuladd.v{count}{elem_name}("
            f"{vec_llvm} {self._operand(a)}, {vec_llvm} {self._operand(b)}, "
            f"{vec_llvm} {self._operand(c)})"
        )
        # Touch elem_llvm to silence the unused-variable static analyser
        # (the intrinsic name carries the type already).
        del elem_llvm

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
        """Lower scf.for to LLVM IR.

        Checks for 'unroll' hint and constant bounds to decide between
        unrolled (straight-line) or normal (loop) lowering.
        """
        unroll = op.attrs.get("unroll", False)
        lower, upper, step = op.operands[:3]

        # Check if we should unroll
        if (
            unroll
            and self._is_constant(lower)
            and self._is_constant(upper)
            and self._is_constant(step)
        ):
            self._lower_unrolled_for(op)
        else:
            self._lower_normal_for(op)

    def _lower_normal_for(self, op: Op) -> None:
        """Lower scf.for to normal LLVM loop (header, body, latch, exit)."""
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

    def _lower_unrolled_for(self, op: Op) -> None:
        """Lower scf.for to unrolled straight-line code (no control flow).

        Emits loop body N times inline, where N is the compile-time trip count.
        This eliminates loop overhead and enables better instruction scheduling.

        Strategy: Create synthetic SSA values for IV and iter args for each iteration,
        lower the body region which references these values, collect yielded values,
        and use them as inputs for the next iteration.
        """
        # Extract loop parameters
        num_iter = int(op.attrs.get("num_iter_args", 0))
        lower, upper, step = op.operands[:3]
        iter_inits = op.operands[3 : 3 + num_iter]
        iter_meta = op.attrs.get("iter_args", [])
        iv_name = op.attrs["iv"]

        # Evaluate constant bounds
        lower_val = self._eval_constant(lower)
        upper_val = self._eval_constant(upper)
        step_val = self._eval_constant(step)

        # Calculate trip count
        trip_count = (upper_val - lower_val) // step_val

        # Track current LLVM values of iteration variables
        # Start with init values (these are the LLVM operand strings)
        current_iter_values: Dict[str, str] = {}
        for meta, init in zip(iter_meta, iter_inits):
            current_iter_values[meta["name"]] = self._operand(init)

        # Create a fake op to hold IV and iter arg SSA values for each iteration
        # We'll temporarily replace the original op in each Value's op field

        # Find the Value objects for IV and iter args
        # These are created when the scf.for was built and stored in the op
        iv_value_obj = None
        iter_value_objs: List[Value] = []

        # The IV and iter arg Values are referenced by ops in the body region
        # We need to find them by scanning the region
        for body_op in op.regions[0].ops:
            if hasattr(body_op, "operands"):
                for operand in body_op.operands:
                    if operand.name == iv_name and operand.op == op:
                        iv_value_obj = operand
                    for meta in iter_meta:
                        if operand.name == meta["name"] and operand.op == op:
                            if operand not in iter_value_objs:
                                iter_value_objs.append(operand)

        # Phase 4a: Detect if loop body has trailing sync() that could be elided
        # Only if elide_trailing_barrier flag is True
        elide_enabled = op.attrs.get("elide_trailing_barrier", True)
        trailing_sync_op = None
        if elide_enabled:
            body_ops = list(op.regions[0].ops)
            if len(body_ops) >= 2:
                second_last_op = body_ops[-2]
                if second_last_op.name == "tile.sync":
                    trailing_sync_op = second_last_op

        # Unroll: emit loop body trip_count times
        for iteration in range(trip_count):
            # Compute induction variable value for this iteration
            iv_value = lower_val + iteration * step_val
            iv_const_name = self._fresh("iv_const")
            iv_ty = _llvm_type(lower.type)
            self._current().emit(f"  {iv_const_name} = add {iv_ty} 0, {iv_value}")

            # Temporarily change the Value.name fields to unique names for this iteration
            iteration_suffix = f".unroll{iteration}"

            # Collect all Values that need renaming (IV, iter args, and all op results)
            values_to_rename: List[Tuple[Value, str]] = []

            # Rename IV
            if iv_value_obj is not None:
                values_to_rename.append((iv_value_obj, iv_value_obj.name))
                iv_value_obj.name = iv_const_name

            # Rename iter args
            for val_obj in iter_value_objs:
                values_to_rename.append((val_obj, val_obj.name))
                # Find corresponding meta to get current value
                for meta in iter_meta:
                    if val_obj.name == meta["name"]:
                        val_obj.name = current_iter_values[meta["name"]]
                        break

            # Rename all op results in the body to have unique names
            for body_op in op.regions[0].ops:
                if len(body_op.results) > 0:
                    for result in body_op.results:
                        values_to_rename.append((result, result.name))
                        base_name = result.name.lstrip("%")
                        result.name = f"%{base_name}{iteration_suffix}"

            # Phase 4a: Mark trailing sync for elision in non-final iterations
            # Rationale: sync before yield ensures current iter's LDS writes are
            # visible to NEXT iter's LDS reads. In iterations 0..N-2, the NEXT
            # iteration starts with its own barrier after global load, making this
            # trailing barrier redundant. Only the final iteration needs it.
            is_final_iteration = iteration == trip_count - 1
            if trailing_sync_op and not is_final_iteration:
                # Mark this specific op to be skipped (next iter has its own barrier)
                self._unroll_elide_sync_op = {"op": trailing_sync_op}
            else:
                self._unroll_elide_sync_op = None

            # Lower the body region - it will now produce unique LLVM SSA names
            self._yield_stack.append([])
            self.lower_region(op.regions[0])
            yielded = self._yield_stack.pop()

            # Clear elision marker
            self._unroll_elide_sync_op = None

            # Restore all original names
            for val_obj, saved_name in values_to_rename:
                val_obj.name = saved_name

            # Verify yield count
            if len(yielded) != num_iter:
                raise RuntimeError(
                    f"scf.for expected {num_iter} yielded values, got {len(yielded)}"
                )

            # Update iteration variables with yielded values for next iteration
            for meta, yld in zip(iter_meta, yielded):
                current_iter_values[meta["name"]] = yld

        # After all iterations, bind results to final iter var values
        for meta, result in zip(iter_meta, op.results):
            ll_ty = _llvm_type_from_name(meta["type"])
            final_val = current_iter_values[meta["name"]]
            self._current().emit(
                f"  {result.name} = bitcast {ll_ty} {final_val} to {ll_ty}"
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
        self._current().emit(" ret void")
        self._current().terminated = True

    # ----- finalize -----

    def finalize(self) -> str:
        # Terminate the entry (now potentially exit) block with ret.
        if not self._current().terminated:
            self._current().emit(" ret void")
            self._current().terminated = True

        out: List[str] = []
        out.append(f'target datalayout = "{self._backend.datalayout(self._flavor)}"')
        out.append(f'target triple = "{self._backend.triple}"')
        out.append("")

        # smem globals.
        # ``align 4`` matches the natural alignment of f16/bf16/f32/i32
        # LDS storage and is what every 16 B ``ds_read_b128`` /
        # ``ds_write_b128`` issued against them needs (the runtime
        # offset math handles the per-row 16 B stride). The exception
        # is fp8/bf8/i8 storage paired with ``ds_read_b64_tr_b8``: that
        # intrinsic packs 8 bytes per lane and the AMDGPU backend
        # requires the load address to be 8 B aligned; landing the i8
        # global on a 4 B boundary silently corrupts the b64
        # transpose-read output. Bump only the i8/fp8 globals to 16 B
        # so the b64 transpose-read is always safe; leave fp16/f32
        # globals at align 4 (raising them would inflate occupancy
        # pressure on long-prefill 3D kernels).
        for gname, stype in self._smem_globals:
            agg = _smem_storage_type(stype)
            elem_name = stype.elem.name
            elem_is_byte = elem_name in ("i8", "fp8e4m3", "bf8e5m2")
            align = 16 if elem_is_byte else 4
            out.append(
                f"{gname} = internal unnamed_addr addrspace(3) global {agg} poison, align {align}"
            )
        if self._smem_globals:
            out.append("")

        # Intrinsic declarations actually used. ``self._decls`` is
        # the flavor-specific dict assembled in ``_Lowerer.__init__``.
        for key, decl in self._decls.items():
            if self._needs_intrin.get(key):
                out.append(decl)
        if self._needs_intrin:
            out.append("")

        # Function header. Pointer parameters can override their address
        # space via the ``addr_space`` attr (P17): ``"constant"`` →
        # ``ptr addrspace(4)`` for descriptor tables, otherwise the
        # default ``ptr addrspace(1)`` (global).
        def _param_type_str(p):
            t = _llvm_type(p.type)
            if isinstance(p.type, PtrType):
                ovr = p.attrs.get("addr_space")
                if ovr == "constant":
                    t = "ptr addrspace(4)"
                elif ovr == "global":
                    t = "ptr addrspace(1)"
            return t

        params = [
            f"{_param_type_str(p)}{_param_attrs(p.attrs, p.type)} %{p.name}"
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
        attr_parts = [
            '"uniform-work-group-size"="true"',
            f'"amdgpu-flat-work-group-size"="64,{max_wg}"',
        ]
        # Optional explicit waves-per-EU occupancy hint, the AMDGPU
        # equivalent of CUDA's ``__launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_SM)``.
        # The AMDGPU backend reads this to size the register file
        # allocation per workgroup so we hit the target occupancy.
        waves_per_eu = self.kernel.attrs.get("waves_per_eu")
        if waves_per_eu is not None:
            lo, hi = (
                waves_per_eu
                if isinstance(waves_per_eu, tuple)
                else (waves_per_eu, waves_per_eu)
            )
            attr_parts.append(f'"amdgpu-waves-per-eu"="{int(lo)},{int(hi)}"')
        agpr_alloc = self.kernel.attrs.get("agpr_alloc")
        if agpr_alloc is not None:
            attr_parts.append(f'"amdgpu-agpr-alloc"="{_format_agpr_alloc(agpr_alloc)}"')
        out.append(
            "attributes #0 = { " + " ".join(attr_parts) + " norecurse nounwind }"
        )
        out.append("")
        # Empty metadata node referenced by f32 ``atomicrmw fadd`` to opt
        # into the native hardware FP atomic (see
        # ``_op_memref_global_atomic_add``).
        if self._needs_fp_atomic_md:
            out.append("!1 = !{}")
            out.append("")
        return "\n".join(out)


def _format_agpr_alloc(value: object) -> str:
    """Format a kernel ``agpr_alloc`` attr for LLVM's AMDGPU backend."""

    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "none":
            raise ValueError("agpr_alloc string must be 'min,max', not empty/'none'")
        parts = text.split(",")
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        parts = [value[0], value[1]]
    else:
        raise ValueError("agpr_alloc must be a (min, max) pair or 'min,max' string")

    try:
        lo, hi = (int(parts[0]), int(parts[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("agpr_alloc must contain two unsigned integers") from exc
    if lo < 0 or hi < 0:
        raise ValueError("agpr_alloc values must be unsigned")
    if lo > hi:
        raise ValueError("agpr_alloc min must be <= max")
    return f"{lo},{hi}"


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
        elem_map = {"f32": "float", "f16": "half", "bf16": "bfloat", "i32": "i32"}
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


def _encode_waitcnt_gfx11(vmcnt: int, expcnt: int, lgkmcnt: int) -> int:
    """Encode an AMDGPU ``s_waitcnt`` immediate for the RDNA3 (gfx11) ISA.

    The gfx11 ``s_waitcnt`` field layout is *different* from the gfx9/gfx10
    split that :func:`_encode_waitcnt_gfx9_10` produces. It was determined
    empirically on a gfx1151 (Strix Halo) node by assembling each counter
    in isolation with the ROCm 7.0.2 ``llvm-mc --show-encoding`` (LLVM's
    own AMDGPU ``encodeWaitcnt``); the encoded 16-bit immediates were:

    .. code-block:: text

        s_waitcnt vmcnt(0)              -> 0x03F7   (= max with vmcnt cleared)
        s_waitcnt vmcnt(1)              -> 0x07F7
        s_waitcnt lgkmcnt(0)           -> 0xFC07
        s_waitcnt lgkmcnt(1)           -> 0xFC17
        s_waitcnt expcnt(0)            -> 0xFFF0
        s_waitcnt expcnt(1)            -> 0xFFF1
        s_waitcnt vmcnt(0) lgkmcnt(0)  -> 0x0007   (expcnt left at max 7)
        s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) -> 0x0000
        all-max                        -> 0xFFF7

    which decodes to the contiguous fields:

    * ``expcnt``  -> bits ``[2:0]``  (3 bits, max 7)
    * ``lgkmcnt`` -> bits ``[9:4]``  (6 bits, max 63)
    * ``vmcnt``   -> bits ``[15:10]`` (6 bits, max 63)

    Unlike gfx9/10, gfx11 has no split VMCNT and a 6-bit (not 4-bit)
    LGKMCNT. ``-1`` means "no wait" and is encoded as the architectural
    maximum for each counter; explicit values clamp to the field maximum
    rather than wrapping (wrapping ``lgkmcnt`` would silently turn a
    partial wait into a full LDS/scalar drain and corrupt LDS-ordered
    reductions).
    """

    vm_b = 0x3F if vmcnt < 0 else min(max(vmcnt, 0), 0x3F)
    ec_b = 0x7 if expcnt < 0 else min(max(expcnt, 0), 0x7)
    lk_b = 0x3F if lgkmcnt < 0 else min(max(lgkmcnt, 0), 0x3F)
    return (ec_b & 0x7) | ((lk_b & 0x3F) << 4) | ((vm_b & 0x3F) << 10)


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


def _lower_kernel_to_llvm_python(
    kernel: KernelDef,
    *,
    llvm_flavor: Optional[str] = None,
    arch: Optional[str] = None,
) -> str:
    """Native Python lowering of a built ``KernelDef`` to AMDGPU LLVM IR text.

    This is the historical body of :func:`lower_kernel_to_llvm`; it is kept
    as a separate, always-available entry point so that the backend dispatch
    wrapper (and the ``"both"`` differential oracle) can reach the Python
    engine unconditionally regardless of the resolved default backend.
    """
    lowerer = _Lowerer(kernel, llvm_flavor=llvm_flavor, arch=arch)
    lowerer._collect_smem(kernel.body)
    lowerer.lower_region(kernel.body)
    return lowerer.finalize()


def lower_kernel_to_llvm(
    kernel: KernelDef,
    *,
    llvm_flavor: Optional[str] = None,
    arch: Optional[str] = None,
) -> str:
    """Return the AMDGPU LLVM IR text for the given kernel.

    ``llvm_flavor`` overrides the autodetected default (one of
    :data:`LLVM_FLAVOR_LLVM20` / :data:`LLVM_FLAVOR_LLVM22`). Useful
    for tests that want to pin a specific flavor regardless of the
    host ROCm install.

    ``arch`` selects the ISA backend (e.g. ``"gfx942"``, ``"gfx950"``) that
    owns the datalayout, triple, and waitcnt encoding. Defaults to ``gfx950``
    so existing callers and the gfx950 byte-identical baseline are preserved.

    Backend dispatch: this is the single chokepoint every Python-authored
    kernel funnels through. The active backend is resolved (explicit env
    ``ROCKE_BACKEND`` -> default) by :func:`rocke.core.backend.resolve_backend`:

      - ``"python"`` lowers natively (the historical, byte-identical path);
      - ``"cpp"`` serializes the kernel and lowers it through the C++ engine
        (``rocke_engine.lower_serialized_ir``), which is byte-identical to the
        native lowerer for every supported family;
      - ``"both"`` runs both and asserts byte-equality, returning the Python
        result (the differential oracle).

    The cpp / both paths are family-agnostic: they go through the serialized
    ``ck.dsl.ir/v1`` artifact, so no per-family C builder or spec object is
    needed -- any kernel the Python front end can build is lowerable. If the
    C++ engine is unavailable or rejects the IR, the dispatch records the
    reason and falls back to the native lowerer so the result is always
    well-defined (see :func:`rocke.core.backend.lower_kernel_via_backend`).
    """
    from .backend import lower_kernel_via_backend

    return lower_kernel_via_backend(
        kernel,
        llvm_flavor=llvm_flavor,
        arch=arch,
        python_lower=_lower_kernel_to_llvm_python,
    )
