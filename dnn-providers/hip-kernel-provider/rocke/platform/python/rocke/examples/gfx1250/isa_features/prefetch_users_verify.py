# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate the block-scaled GEMM data-prefetch hints, flag on against flag off."""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np

from rocke.examples.gfx1250.gemm.block_scaled_gemm_verify import (
    _launch,
    check_result,
    make_case_inputs,
    reference_result,
)
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
)

try:
    from .common import (
        Reporter,
        Runtime,
        ValidatedArtifact,
        compile_and_validate,
        make_parser,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        Reporter,
        Runtime,
        ValidatedArtifact,
        compile_and_validate,
        make_parser,
    )

# One spec per matrix path, each with more than one K step so the next-step
# ``global_prefetch`` is emitted.
_SPECS = tuple(
    BlockScaledGemmSpec(
        name="pf_verify",
        M=32,
        N=48,
        K=256,
        block_k=block_k,
        scale_dtype=scale_dtype,
        matrix_path=path,
    )
    for path, block_k, scale_dtype in (
        ("wmma", 128, "fp32"),
        ("wmma_scale", 32, "e8m0"),
        ("wmma_scale16", 16, "e8m0"),
    )
)
_CASE = "mixed"

_LLVM_ON = (
    "call void @llvm.amdgcn.s.setreg(i32 1537, i32 1)",
    "call void @llvm.amdgcn.s.prefetch.data.p1(ptr addrspace(1) ",
    "call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) ",
)
_ISA_ON = (r"\bs_setreg", r"\bs_prefetch_data\b", r"\bglobal_prefetch_b8\b")
_ISA_MMA = (r"\bv_wmma_",)


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def _compare_isa(off: ValidatedArtifact, on: ValidatedArtifact) -> tuple[bool, str]:
    """The flag adds prefetches and nothing the backend would not emit anyway.

    The llvm23 backend puts its own ``global_prefetch_b8`` and a
    ``HW_REG_WAVE_MODE`` bit 25 ``s_setreg`` in every gfx1250 prologue, so the
    flag's global prefetches are counted against the flag-off build. The
    scalar-prefetch enable (``WAVE_MODE`` bit 24) and ``s_prefetch_data`` come
    only from the flag.
    """
    counts = {
        name: (_count(pattern, off.isa_text), _count(pattern, on.isa_text))
        for name, pattern in (
            ("prefetch_enable", r"\bs_setreg\w*\s+hwreg\(HW_REG_WAVE_MODE, 24, 1\), 1\b"),
            ("s_prefetch_data", r"\bs_prefetch_data\b"),
            ("global_prefetch_b8", r"\bglobal_prefetch_b8\b"),
            ("v_wmma", r"\bv_wmma_\w+"),
        )
    }
    ok = (
        counts["prefetch_enable"] == (0, 1)
        and counts["s_prefetch_data"][0] == 0
        and counts["s_prefetch_data"][1] >= 1
        and counts["global_prefetch_b8"][1] > counts["global_prefetch_b8"][0]
        and counts["v_wmma"][0] == counts["v_wmma"][1] > 0
    )
    detail = ", ".join(f"{name} off={a} on={b}" for name, (a, b) in counts.items())
    return ok, detail


def _run_functional(
    spec: BlockScaledGemmSpec, off: ValidatedArtifact, on: ValidatedArtifact
) -> tuple[bool, str]:
    native = spec.resolved_matrix_path() != "wmma"
    inputs = make_case_inputs(spec, _CASE)
    expected = reference_result(*inputs, spec.block_k, native=native)
    runtime = Runtime()
    results = []
    for validated in (off, on):
        module = runtime.load_module(validated.artifact.hsaco)
        try:
            function = module.get_function(validated.artifact.kernel_name)
            results.append(_launch(runtime, function, spec, inputs))
        finally:
            module.unload()
    for got in results:
        check_result(got, expected, exact=native)
    same = results[0].tobytes() == results[1].tobytes()
    return same, f"case {_CASE}: flag on {'matches' if same else 'differs from'} flag off bit for bit"


def main(argv: list[str] | None = None) -> int:
    args = make_parser(__doc__).parse_args(argv)
    reporter = Reporter(args.arch)
    for base in _SPECS:
        path = base.resolved_matrix_path()
        builds = {}
        for flag in (False, True):
            spec = replace(base, prefetch=flag)
            name = f"prefetch-users.{path}.compile.{'on' if flag else 'off'}"
            try:
                builds[flag] = compile_and_validate(
                    build_block_scaled_gemm(spec, arch=args.arch),
                    arch=args.arch,
                    llvm_required=_LLVM_ON if flag else (),
                    isa_required=_ISA_ON + _ISA_MMA if flag else _ISA_MMA,
                )
            except Exception as exc:  # noqa: BLE001
                reporter.failed(name, f"{type(exc).__name__}: {exc}")
                continue
            reporter.passed(name, f"{spec.kernel_name()}: LLVM and ISA matched")

        name = f"prefetch-users.{path}.isa-diff"
        if len(builds) != 2:
            reporter.skipped(name, "compile validation failed")
        else:
            ok, detail = _compare_isa(builds[False], builds[True])
            (reporter.passed if ok else reporter.failed)(name, detail)

        name = f"prefetch-users.{path}.functional"
        if len(builds) != 2:
            reporter.skipped(name, "compile validation failed")
        elif args.compile_only:
            reporter.skipped(name, "--compile-only requested")
        else:
            try:
                ok, detail = _run_functional(base, builds[False], builds[True])
            except Exception as exc:  # noqa: BLE001
                reporter.failed(name, f"{type(exc).__name__}: {exc}")
            else:
                (reporter.passed if ok else reporter.failed)(name, detail)
    return reporter.finish()


if __name__ == "__main__":
    raise SystemExit(main())
