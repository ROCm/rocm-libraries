# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""fp64 universal GEMM: numerics on hardware, refusals, and the LDS budget.

The numeric tests run on the GPU present when it is a target whose fp64 GEMM
numerics have passed on hardware. A target the validator admits fp64 GEMM on that
has not run them yet skips with a note to enable it. The refusal and LDS-budget
tests are CPU-only and check that both engines agree.
"""

import os
import re
from importlib.resources import files

import numpy as np
import pytest
from rocke.core.arch import ArchTarget, known_arches
from rocke.core.backend import universal_gemm_spec_to_dict
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    is_valid_spec,
)
from rocke.tests.arch_marks import arch_specific
from rocke.tests.engines import available_engines
from rocke.tests.gpu import current_device_arch_in
from rocke.tests.source_tree import require_dsl_docs
from rocke.tests.universal_gemm import launch_universal_gemm, lower_universal_gemm


def _spec(
    tile,
    pipeline,
    epilogue,
    pad=False,
    warp_tile=(16, 16, 4),
    acc="fp64",
    batched=False,
    **trait,
):
    tile_m, tile_n, tile_k, warp_m, warp_n = tile
    return UniversalGemmSpec(
        name="f64_gemm",
        tile=TileSpec(tile_m, tile_n, tile_k, warp_m, warp_n, 1, *warp_tile),
        trait=TraitSpec(
            pipeline=pipeline,
            epilogue=epilogue,
            pad_m=pad,
            pad_n=pad,
            pad_k=pad,
            **trait,
        ),
        data=DataSpec("fp64", "fp64", "fp64", acc),
        batched=batched,
    )


# The targets the validator admits a plain fp64 GEMM on, and those of them that
# have no DRAM->LDS DMA and so must refuse direct_to_lds.
_F64_GEMM_ARCHS = tuple(
    arch
    for arch in known_arches()
    if is_valid_spec(_spec((32, 32, 8, 1, 1), "mem", "default"), arch=arch)[0]
)
_F64_GEMM_ARCHS_WITHOUT_DMA = tuple(
    arch
    for arch in _F64_GEMM_ARCHS
    if ArchTarget.from_gfx(arch).async_lds_max_dwords == 0
)

# The targets these numerics have passed on in real hardware runs; the pull
# request that adds a target here records its run.
_F64_GEMM_HARDWARE_VERIFIED_ARCHS = ("gfx90a", "gfx942", "gfx950")


def _f64_gemm_device_arch() -> str:
    """The current GPU's arch when its fp64 GEMM numerics are hardware-verified.

    Skips on a GPU the validator does not admit fp64 GEMM on, and on an admitted
    target that has not run the numerics yet; fails under ``ROCKE_REQUIRE_<ARCH>=1``
    for an admitted arch the GPU is not (see ``current_device_arch_in``), and for
    the GPU's own arch when that arch is admitted but not yet verified, so a lane
    that requires it does not pass with no coverage.
    """
    arch = current_device_arch_in(_F64_GEMM_ARCHS, subject="fp64 GEMM numerics")
    if arch not in _F64_GEMM_HARDWARE_VERIFIED_ARCHS:
        unverified = f"fp64 GEMM has not run on {arch} hardware yet"
        if os.environ.get(f"ROCKE_REQUIRE_{arch.upper()}") == "1":
            pytest.fail(f"ROCKE_REQUIRE_{arch.upper()}=1 but {unverified}")
        pytest.skip(f"{unverified}; enable once it has")
    return arch


def _preshuffle_b(bt, tile_n, tile_k):
    n, k = bt.shape
    return np.ascontiguousarray(
        bt.reshape(n // tile_n, tile_n, k // tile_k, tile_k).transpose(2, 0, 1, 3)
    )


def _assert_both_engines_refuse(spec, arch, reason):
    admitted, verdict = is_valid_spec(spec, arch=arch)
    assert not admitted and reason in verdict
    if "cpp" in available_engines():
        import rocke_engine as native

        spec_dict = universal_gemm_spec_to_dict(spec)
        assert native.gemm_is_valid(spec_dict, arch=arch) == (admitted, verdict)


@arch_specific(
    _F64_GEMM_HARDWARE_VERIFIED_ARCHS,
    reason="fp64 GEMM numerics have passed on hardware for these targets only",
)
@pytest.mark.parametrize(
    "tile, pipeline, epilogue, pad, trait",
    [
        ((64, 64, 16, 2, 2), "compv4", "default", True, {}),
        ((64, 64, 16, 2, 2), "compv3", "cshuffle", False, {}),
        ((32, 32, 8, 1, 1), "mem", "default", False, {}),
        ((16, 16, 4, 1, 1), "mem", "cshuffle", True, {}),
        ((64, 64, 16, 2, 2), "compv3", "cshuffle", False, {"chiplet_swizzle": True}),
        ((64, 64, 32, 2, 2), "compv4", "default", False, {"lds_k_pad": 2}),
        ((64, 64, 16, 2, 2), "mem", "default", False, {"lds_k_pad": 4}),
        ((64, 64, 32, 2, 2), "compv4", "default", False, {"preshuffle_b": True}),
        ((32, 32, 8, 1, 1), "compv3", "cshuffle", False, {"preshuffle_b": True}),
        ((64, 64, 16, 2, 2), "compv4", "default", False, {"lds_swizzle": True}),
        ((32, 32, 8, 1, 1), "mem", "default", False, {"lds_swizzle": True}),
        ((64, 64, 16, 2, 2), "compv3", "cshuffle", False, {"direct_to_lds": True}),
        (
            (64, 64, 32, 2, 2),
            "compv4",
            "default",
            False,
            {"direct_to_lds": True, "dtl_prefetch": True},
        ),
        (
            (64, 64, 16, 2, 2),
            "compv4",
            "default",
            False,
            {"direct_to_lds": True, "lds_swizzle": True},
        ),
        (
            (32, 32, 8, 1, 1),
            "compv3",
            "cshuffle",
            False,
            {"direct_to_lds": True, "dtl_prefetch": True, "lds_swizzle": True},
        ),
    ],
)
def test_f64_gemm_numeric(tile, pipeline, epilogue, pad, trait):
    arch = _f64_gemm_device_arch()
    spec = _spec(tile, pipeline, epilogue, pad, **trait)
    if spec.trait.direct_to_lds and arch in _F64_GEMM_ARCHS_WITHOUT_DMA:
        pytest.skip(
            f"{arch} has no DRAM->LDS DMA; "
            "test_f64_gemm_direct_to_lds_refused_without_dma covers the refusal"
        )
    assert is_valid_spec(spec, arch=arch) == (True, "ok")
    tile_m, tile_n, tile_k = tile[:3]
    # Padded configs use extents that are not tile multiples.
    trim = 1 if pad else 0
    m, n, k = 2 * tile_m - trim, 3 * tile_n - trim, 4 * tile_k - trim
    rng = np.random.default_rng(tile_m * 100 + tile_k)

    def b_device(bt):
        if spec.trait.preshuffle_b:
            return _preshuffle_b(bt, tile_n, tile_k)
        return None

    for engine in available_engines():
        llvm_ir = lower_universal_gemm(spec, engine, arch=arch)
        a = rng.integers(-8, 9, (m, k)).astype(np.float64)
        bt = rng.integers(-8, 9, (n, k)).astype(np.float64)
        c = launch_universal_gemm(
            llvm_ir, spec, a, bt, arch=arch, b_device=b_device(bt)
        )
        np.testing.assert_array_equal(c, a @ bt.T)
        a = rng.standard_normal((m, k))
        bt = rng.standard_normal((n, k))
        c = launch_universal_gemm(
            llvm_ir, spec, a, bt, arch=arch, b_device=b_device(bt)
        )
        error = np.abs(c - a @ bt.T) / (np.abs(a) @ np.abs(bt).T)
        assert np.max(error) < k * np.finfo(np.float64).eps


@arch_specific(
    _F64_GEMM_HARDWARE_VERIFIED_ARCHS,
    reason="fp64 GEMM numerics have passed on hardware for these targets only",
)
@pytest.mark.parametrize(
    "tile, pipeline, epilogue",
    [
        ((64, 64, 16, 2, 2), "compv4", "default"),
        ((32, 32, 8, 1, 1), "mem", "default"),
    ],
)
def test_f64_batched_active_tile_skip_numeric(tile, pipeline, epilogue):
    arch = _f64_gemm_device_arch()
    spec = _spec(tile, pipeline, epilogue, batched=True, active_tile_skip=True)
    assert is_valid_spec(spec, arch=arch) == (True, "ok")
    tile_m, tile_n, tile_k = tile[:3]
    batches, m_tiles, n_tiles = 3, 2, 3
    m, n, k = m_tiles * tile_m, n_tiles * tile_n, 4 * tile_k
    rng = np.random.default_rng(tile_m)
    a = rng.integers(-8, 9, (batches, m, k)).astype(np.float64)
    bt = rng.integers(-8, 9, (batches, n, k)).astype(np.float64)
    expected = np.einsum("bmk,bnk->bmn", a, bt)
    # A tile is skipped when its first sorted token id is -1.
    skipped = {(0, 1), (2, 0)}
    ids = np.arange(batches * m, dtype=np.int32).reshape(batches, m)
    for z, mi in skipped:
        ids[z, mi * tile_m] = -1
    for engine in available_engines():
        llvm_ir = lower_universal_gemm(spec, engine, arch=arch)
        c = launch_universal_gemm(llvm_ir, spec, a, bt, arch=arch, sorted_ids=ids)
        for z in range(batches):
            for mi in range(m_tiles):
                rows = slice(mi * tile_m, (mi + 1) * tile_m)
                if (z, mi) in skipped:
                    assert np.isnan(c[z, rows]).all()
                else:
                    np.testing.assert_array_equal(c[z, rows], expected[z, rows])


# Refusals that follow the target: the warp-tile message and the LDS cap differ.
_F64_GEMM_REFUSALS = {
    "gfx942": {
        "warp_tile": "fp64 GEMM supports only warp_tile (16, 16, 4)",
        "lds_k_pad": (64, "LDS budget 81920 > 65536 cap"),
    },
    "gfx950": {
        "warp_tile": "unsupported fp64 warp_tile (4, 4, 4) on gfx950",
        "lds_k_pad": (160, "LDS budget 180224 > 163840 cap"),
    },
}


@pytest.mark.parametrize("arch", sorted(_F64_GEMM_REFUSALS))
@pytest.mark.parametrize(
    "case", ["warp_tile", "acc", "direct_to_lds_pad_k", "lds_k_pad"]
)
def test_f64_gemm_rejects_unsupported(arch, case):
    expected = _F64_GEMM_REFUSALS[arch]
    pad, pad_reason = expected["lds_k_pad"]
    spec, reason = {
        "warp_tile": (
            _spec((64, 64, 16, 2, 2), "mem", "default", warp_tile=(4, 4, 4)),
            expected["warp_tile"],
        ),
        "acc": (
            _spec((64, 64, 16, 2, 2), "mem", "default", acc="fp32"),
            "only supports fp64 accumulation for fp64",
        ),
        "direct_to_lds_pad_k": (
            _spec((64, 64, 16, 2, 2), "compv3", "default", True, direct_to_lds=True),
            "fp64 GEMM does not support direct_to_lds with pad_k",
        ),
        "lds_k_pad": (
            _spec((64, 64, 16, 2, 2), "compv4", "default", lds_k_pad=pad),
            pad_reason,
        ),
    }[case]
    _assert_both_engines_refuse(spec, arch, reason)


def test_f64_gemm_targets_are_the_documented_ones():
    """dtype_support.md and SUPPORT_MATRIX.md name the targets the validator
    admits fp64 GEMM on, and the one of them that refuses direct_to_lds."""
    assert _F64_GEMM_ARCHS == ("gfx90a", "gfx942", "gfx950")
    assert _F64_GEMM_ARCHS_WITHOUT_DMA == ("gfx90a",)


_F64 = re.compile(r"\bfp?64\b")
_UNIVERSAL_GEMM = re.compile(r"universal[ _]gemm|gemm_universal", re.IGNORECASE)
_GFX_TARGET = re.compile(r"\bgfx[0-9a-f]+\b")


def _doc_sentences(text):
    """Split markdown into sentences, keeping list items and table rows apart."""
    blocks = re.split(r"\n\s*\n|\n(?=\s*(?:[-*] |\|))", text)
    for block in blocks:
        yield from re.split(r"(?<=[.;])\s+|(?<=\.\*\*)\s+", " ".join(block.split()))


def _assert_names_every_admitted_target(doc):
    """A sentence that places fp64 universal GEMM on named targets names all of
    them, so no summary line describes it as narrower than the validator."""
    narrower = []
    for sentence in _doc_sentences(doc.read_text(encoding="utf-8")):
        if not (_F64.search(sentence) and _UNIVERSAL_GEMM.search(sentence)):
            continue
        named = set(_GFX_TARGET.findall(sentence))
        if named and not named.issuperset(_F64_GEMM_ARCHS):
            narrower.append(sentence)
    assert not narrower, f"{doc.name} names only some fp64 GEMM targets in: {narrower}"


def test_dtype_support_doc_names_every_f64_gemm_target():
    _assert_names_every_admitted_target(
        require_dsl_docs() / "reference" / "dtype_support.md"
    )


def test_support_matrix_names_every_f64_gemm_target():
    # Read through the package, so the CMake-installed test payload, which
    # installs the package directory whole, checks its copy too.
    _assert_names_every_admitted_target(files("rocke.instances") / "SUPPORT_MATRIX.md")


@pytest.mark.parametrize("arch", _F64_GEMM_ARCHS_WITHOUT_DMA)
def test_f64_gemm_direct_to_lds_refused_without_dma(arch):
    spec = _spec((64, 64, 16, 2, 2), "compv3", "cshuffle", direct_to_lds=True)
    _assert_both_engines_refuse(
        spec, arch, f"direct_to_lds: {arch} has no DRAM->LDS DMA instruction"
    )


@arch_specific(
    ("gfx942",),
    reason="the single-buffer fallback sizes follow the gfx942 64 KiB LDS cap; gfx950 has 160 KiB",
)
@pytest.mark.parametrize(
    "lds_k_pad, smem_bytes", [(16, 32768), (32, 49152), (48, 65536)]
)
def test_f64_gemm_lds_k_pad_budget(lds_k_pad, smem_bytes):
    # The padded row stride must count towards the double-buffer decision: with
    # lds_k_pad >= 16 a doubled A/B region no longer fits and compv4 falls back
    # to a single buffer.
    spec = _spec((64, 64, 16, 2, 2), "compv4", "default", lds_k_pad=lds_k_pad)
    assert is_valid_spec(spec, arch="gfx942") == (True, "ok")
    lowered = {
        engine: lower_universal_gemm(spec, engine, arch="gfx942")
        for engine in available_engines()
    }
    for llvm_ir in lowered.values():
        assert f"[{smem_bytes} x i8] poison" in llvm_ir
    assert len(set(lowered.values())) == 1
