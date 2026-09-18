################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Phase G0 smoke test — proves the CPU-only codegen-emit harness works and is
deterministic, and records compact kernel-identity results.

The emit itself is what drives coverage of ``KernelWriterAssembly`` /
``KernelWriter``; this single gfx942 kernel already exercises thousands of lines
of the emitter. Identity snapshots record kernel selection and status. Tests
that protect a particular generated behavior assert a focused source pattern
instead of hashing all compiler-dependent instructions.
"""

import os
import shutil

import pytest

from codegen_harness import (
    canonicalize_asm,
    emit_kernels_from_logic,
)
from config_harness import (
    _select_benchmark_problem,
    assert_config_emits,
    benchmark_problem_fingerprint,
    emit_kernels_from_config,
    golden_digest,
    solutions_from_config,
)
from Tensile.Common.Architectures import gfxToIsa
from Tensile.Tests.rocisa_test_state import preserve_rocisa_kernel_state

pytestmark = pytest.mark.unit

# A small, self-contained logic file already shipped as a fixture for the
# LibraryIO suite (gfx942, HSS/BH GEMM). 1 solution -> 1 assembly kernel.
_LOGIC = os.path.join(
    os.path.dirname(__file__),
    "..",
    "LibraryIO",
    "data",
    "logic_gfx942_HSS_BH.yaml",
)

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "subtile3_gr_variants.yaml",
)

_MULTI_PROBLEM_CONFIG = "Tensile/Tests/common/gemm/use_beta_false.yaml"


def _pin_rocisa(arch, wavefront):
    from rocisa import rocIsa

    isa = gfxToIsa(arch)
    assert isa is not None, f"unrecognized test architecture: {arch}"
    ri = rocIsa.getInstance()
    ri.init(isa, shutil.which("amdclang++") or "/usr/bin/amdclang++")
    ri.setKernel(isa, wavefront)
    return ri


def test_emit_produces_assembly():
    """The harness emits non-trivial assembly for the fixture kernel."""
    results = emit_kernels_from_logic(_LOGIC)
    assert len(results) == 1
    base, src, err = results[0]
    assert err == 0
    assert src and len(src.splitlines()) > 1000
    # Sanity: it is real AMDGCN assembly for the expected target.
    assert ".amdgcn_target" in src
    assert "gfx942" in src
    assert base.startswith("Cijk_")


def test_emit_is_deterministic():
    """Two independent emits are byte-identical after canonicalization."""
    a = emit_kernels_from_logic(_LOGIC)
    b = emit_kernels_from_logic(_LOGIC)
    assert [t[0] for t in a] == [t[0] for t in b]
    assert [t[1] for t in a] == [t[1] for t in b]


@pytest.mark.parametrize(
    "emit,kwargs",
    [
        (emit_kernels_from_logic, {"logic_path": _LOGIC}),
        (emit_kernels_from_config, {"config_path": _CONFIG, "limit": 1, "arch": "gfx950"}),
    ],
    ids=["logic", "config"],
)
def test_emit_restores_callers_rocisa_kernel(emit, kwargs):
    """Logic/config harnesses must not leak their target ISA to callers."""
    with preserve_rocisa_kernel_state():
        ri = _pin_rocisa("gfx1250", 32)
        before = ri.getKernel()

        results = emit(**kwargs)

        after = ri.getKernel()
        assert results
        assert tuple(after.isa) == tuple(before.isa)
        assert after.wavefrontSize == before.wavefrontSize


def test_canonicalize_neutralizes_random_labels():
    """The canonicalizer maps random label suffixes to stable ids and is
    idempotent."""
    raw = (
        "s_cbranch_scc0 label_NoBranch_T8JHFHKM7BO5OHXW\n"
        "label_NoBranch_T8JHFHKM7BO5OHXW:\n"
        "s_branch label_Done_S4FDBQ587JJL6NOU\n"
        "label_Done_S4FDBQ587JJL6NOU:\n"
    )
    canon = canonicalize_asm(raw)
    assert "T8JHFHKM7BO5OHXW" not in canon
    assert canon.count("_LBL0") == 2  # def + reference preserved as a pair
    assert canon.count("_LBL1") == 2
    assert canonicalize_asm(canon) == canon  # idempotent


def test_config_source_patterns_observe_named_instruction(monkeypatch):
    results = [
        (
            "Cijk_kernel",
            '.amdgcn_target "amdgcn-amd-amdhsa--gfx942"\n'
            "v_cvt_f32_i32 v0, v0\n" + "s_nop 0\n" * 50,
            0,
        )
    ]
    monkeypatch.setattr("config_harness.emit_kernels_from_config", lambda *args, **kwargs: results)

    assert_config_emits(
        "unused.yaml",
        "gfx942",
        validate_source=True,
        required_source_patterns=(("integer-to-float conversion", r"^v_cvt_f32_i32\b"),),
    )


def test_config_source_patterns_reject_missing_instruction(monkeypatch):
    results = [
        (
            "Cijk_kernel",
            '.amdgcn_target "amdgcn-amd-amdhsa--gfx942"\n'
            "v_mov_b32 v0, v0\n" + "s_nop 0\n" * 50,
            0,
        )
    ]
    monkeypatch.setattr("config_harness.emit_kernels_from_config", lambda *args, **kwargs: results)

    with pytest.raises(AssertionError, match="integer-to-float conversion"):
        assert_config_emits(
            "unused.yaml",
            "gfx942",
            validate_source=True,
            required_source_patterns=(("integer-to-float conversion", r"^v_cvt_f32_i32\b"),),
        )


def test_config_emit_smoke_can_allow_known_errors(monkeypatch):
    results = [("kernel-ok", "", 0), ("kernel-error", "", -2)]
    monkeypatch.setattr("config_harness.emit_kernels_from_config", lambda *args, **kwargs: results)

    assert_config_emits("unused.yaml", "gfx942", all_ok=False)


def test_config_emit_smoke_rejects_unexpected_error(monkeypatch):
    results = [("kernel-ok", "", 0), ("kernel-error", "", -2)]
    monkeypatch.setattr("config_harness.emit_kernels_from_config", lambda *args, **kwargs: results)

    with pytest.raises(AssertionError, match="kernel-error"):
        assert_config_emits("unused.yaml", "gfx942")


def test_config_harness_selects_problem_entry():
    first = solutions_from_config(
        _MULTI_PROBLEM_CONFIG, arch="gfx942", limit_solutions=1, problem_index=0
    )
    second = solutions_from_config(
        _MULTI_PROBLEM_CONFIG, arch="gfx942", limit_solutions=1, problem_index=1
    )

    assert first and second
    assert first[0]["ProblemType"]["UseScaleCD"] is False
    assert second[0]["ProblemType"]["UseScaleCD"] is True


def test_problem_fingerprint_selection_survives_reordering():
    first = [{"OperationType": "GEMM", "DataType": "S"}, {"ForkParameters": []}]
    selected = [{"OperationType": "GEMM", "DataType": "H"}, {"ForkParameters": []}]
    fingerprint = benchmark_problem_fingerprint(selected)

    assert _select_benchmark_problem([first, selected], "config.yaml", 0, fingerprint) == selected
    assert _select_benchmark_problem([selected, first], "config.yaml", 0, fingerprint) == selected


def test_problem_fingerprint_selection_rejects_missing_group():
    entries = [[{"OperationType": "GEMM", "DataType": "S"}, {"ForkParameters": []}]]

    with pytest.raises(ValueError, match="does not exist"):
        _select_benchmark_problem(entries, "config.yaml", 0, "not-present")


def test_emit_golden_digest(snapshot):
    """Pin the order-invariant golden (kernel identity + emit success).

    The full assembly text is not hashed (it is order-coupled via the emitter's
    process-global MMA-scheduler state); coverage comes from running the emit.
    """
    results = emit_kernels_from_logic(_LOGIC)
    digests = [{"basename": b, "err": e} for (b, _s, e) in results]
    assert digests == snapshot
