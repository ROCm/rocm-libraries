# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Thin Python compile-service entry point for the CK DSL provider.

M1 step I-3 ships :func:`noop_smoke`, which proves the C++ -> Python ->
``ck_dsl`` import path works inside the embedded interpreter.

M1 step I-4 adds :func:`compile_smoke`, which exercises the full DSL
compile pipeline (``build_elementwise`` -> ``compile_kernel`` -> HSACO
bytes) and returns the kernel artifact plus the minimum launch metadata
the C++ side needs to load and launch it via ``hipModuleLoadData`` +
``hipModuleLaunchKernel``. The smoke kernel is the simplest cleanly-
compiling instance in ``ck_dsl.instances``: an FP16 copy elementwise
with ``block_size=64`` and ``vec=2`` (no MFMA, no LDS, no scaled
converts -- but still gfx950 ISA since the DSL is gfx950-only). The
resulting kernel signature is ``(A: ptr, C: ptr, N: i32)`` so launching
over a one-element buffer copies a single FP16 value.

# TODO(I-7): add ``compile(op_kind: str, payload: dict) -> dict`` that
# instantiates the matching ``ck_dsl`` Spec dataclass via ``**payload``,
# calls ``ck_dsl.helpers.compile.compile_kernel``, and returns the same
# shape as :func:`compile_smoke` (hsaco + launch metadata).
"""

from __future__ import annotations

from typing import Any, Dict, List


def noop_smoke() -> dict:
    """Return a constant. Used by I-3 to prove import path + GIL handling."""
    import ck_dsl  # forces the cross-package import to succeed

    return {
        "service": "ck_dsl_provider.compile_service",
        "ck_dsl_module_path": ck_dsl.__file__,
        "smoke": "ok",
    }


# Pointer ABI on AMDGPU host-side: 8 bytes, 8-byte aligned. Matches the
# natural-alignment scheme used by `launcher.cpp` for the DSL kernels.
_PTR_SIZE = 8
_PTR_ALIGN = 8
_I32_SIZE = 4
_I32_ALIGN = 4


def _smoke_arg_schema() -> List[Dict[str, Any]]:
    """Schema for the elementwise-copy smoke kernel signature.

    Kernel ABI is ``(A: ptr<f16>, C: ptr<f16>, N: i32)`` as built by
    ``ck_dsl.instances.elementwise.build_elementwise`` for a unary op
    (see ``ElementwiseSpec`` docstring + ``build_elementwise`` body in
    ``ck_dsl/instances/elementwise.py``). The C++ ``LaunchAbi`` packs
    args back-to-back honouring each slot's ``align`` -- exactly the
    layout the AMDGPU calling convention expects when args are handed
    to ``hipModuleLaunchKernel`` via ``HIP_LAUNCH_PARAM_BUFFER_*``.
    """
    return [
        {"name": "A", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "C", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "N", "kind": "I32", "size": _I32_SIZE, "align": _I32_ALIGN},
    ]


def compile_smoke() -> dict:
    """Compile a trivial gfx950 HSACO and return the launch metadata.

    Used by I-4 to prove the C++ ``KernelArtifact`` / ``HipModule`` /
    ``LaunchAbi`` round-trip. The returned dict is the on-wire shape the
    C++ ``CompileServiceBridge::compileSmoke`` translates into a
    ``KernelArtifact``; keep the field names stable -- the C++ side
    looks them up by string.

    Returned fields:

    ``hsaco`` (bytes)
        The HSA code object produced by ``compile_kernel``. Loadable
        via ``hipModuleLoadData``.

    ``kernel_name`` (str)
        The mangled kernel symbol to pass to ``hipModuleGetFunction``.

    ``kind`` (str)
        Free-form tag for logs / debugging. Not consumed by the C++
        launch path.

    ``grid`` (tuple[int, int, int])
        ``(gx, gy, gz)`` for ``hipModuleLaunchKernel``. Sized for a
        one-element launch -- one block of ``block_size`` threads.

    ``block`` (tuple[int, int, int])
        ``(bx, by, bz)``. Always ``(block_size, 1, 1)`` for this kernel.

    ``lds_bytes`` (int)
        Dynamic shared-memory bytes (``sharedMemBytes`` arg). Zero for
        this kernel; the field exists so future kernels that need
        dynamic LDS can populate it without changing the wire shape
        (the launcher.cpp gap noted in PREP_FINDINGS P-1).

    ``arg_schema`` (list[dict])
        Per-arg metadata used by the C++ ``LaunchAbi::pack`` to lay out
        the argument buffer. Each entry is
        ``{"name": str, "kind": str, "size": int, "align": int}``.

    ``isa`` (str)
        The comgr ISA triple the artifact was built for. Returned for
        logging only; gfx950 is the only target the DSL supports today.
    """
    from ck_dsl.helpers.compile import compile_kernel
    from ck_dsl.instances.elementwise import ElementwiseSpec, build_elementwise

    # The simplest cleanly-compiling instance: FP16 copy, single warp.
    # Keeping block_size small (one wave) and vec small (one 32-bit
    # load) avoids stressing anything we don't need to exercise for a
    # smoke-level "did the launch succeed" check.
    spec = ElementwiseSpec(
        op="copy",
        dtype="f16",
        block_size=64,
        vec=2,
        name="ck_dsl_provider_smoke_copy",
    )
    kernel_def = build_elementwise(spec)
    artifact = compile_kernel(kernel_def)

    return {
        "hsaco": artifact.hsaco,
        "kernel_name": artifact.kernel_name,
        "kind": "elementwise_copy_smoke",
        # One-element launch: one block, one wave. The kernel's scalar
        # fallback path handles the N < block*vec tail; for N=1 all
        # threads except the lane-0 path are masked out by its in-bounds
        # check (see emit_scalar_path in instances/elementwise.py).
        "grid": (1, 1, 1),
        "block": (spec.block_size, 1, 1),
        "lds_bytes": 0,
        "arg_schema": _smoke_arg_schema(),
        "isa": artifact.isa,
    }
