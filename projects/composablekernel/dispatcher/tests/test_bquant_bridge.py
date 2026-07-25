#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the (non-grouped) bquant GEMM TileEngine -> Dispatcher bridge.

Locks the config name format (distinct `gemm_bquant` prefix, NOT the grouped bridge), the
byte-exact codegen<->utils kernel-name contract, the codegen-JSON projection, and the
fp8/bf8/fp8i4/bf8i4 + MX(bf16bf16/bf16bf8/bf16fp4) scope with preshuffleB / preshuffleQuant
families that Old-TE gemm_bquant_quantgrouped*.cpp register. No GPU / hipcc.
"""

import re
import sys
import tempfile
import unittest
from pathlib import Path

_DISP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))

from gemm_bquant_utils import (  # noqa: E402
    NAME_PREFIX,
    BQuantGemmProblem,
    _MX_VARIANTS,
    _require_mx_arch,
    _warp_tile_k_for,
    _generate_bquant_kernel,
    default_fp8_config,
    default_bf8_config,
    default_fp8i4_config,
    default_bf8i4_config,
    default_fp8_preshuffleb_config,
    default_bf8_preshuffleb_config,
    default_fp8i4_preshuffleb_config,
    default_bf8i4_preshuffleb_config,
    default_fp8_preshufflequant_config,
    default_fp8_preshuffleb_bquant_config,
    default_mx_bf16bf16_config,
    default_mx_bf16bf8_config,
    default_mx_bf16fp4_config,
    setup_multiple_bquant_dispatchers,
)
from codegen_common import make_bquant_kernel_name  # noqa: E402

# The ctypes lib source (checked for the B-matrix shuffle / pk_int4 permute steps,
# no GPU needed).
_CTYPES_SRC = (
    _DISP / "bindings" / "ctypes" / "gemm_bquant_ctypes_lib.cpp"
).read_text()

# The Python runner source (checked for the epilogue-dependent C de-permute).
_UTILS_SRC = (_DISP / "python" / "gemm_bquant_utils.py").read_text()


def _header_text(cfg):
    """Codegen the header for a config and return its text (no hipcc)."""
    tmp = Path(tempfile.mkdtemp(prefix="bq_test_"))
    hpp = _generate_bquant_kernel(cfg, tmp)
    assert hpp is not None, f"codegen failed for {cfg.name}"
    return hpp.read_text()


def _static_bool(text, field):
    m = re.search(rf"bool\s+{field}\s*=\s*(\w+)", text)
    assert m, f"{field} not found in generated header"
    return m.group(1) == "true"

_BASE = [default_fp8_config, default_bf8_config, default_fp8i4_config, default_bf8i4_config]
_MX = [default_mx_bf16bf16_config, default_mx_bf16bf8_config, default_mx_bf16fp4_config]
_ALL = _BASE + [
    default_fp8_preshuffleb_config,
    default_fp8_preshufflequant_config,
    default_fp8_preshuffleb_bquant_config,
] + _MX


class TestPrefix(unittest.TestCase):
    def test_name_prefix_is_gemm_bquant(self):
        self.assertEqual(NAME_PREFIX, "gemm_bquant")
        for ctor in _ALL:
            self.assertTrue(ctor().name.startswith("gemm_bquant_"), ctor().name)

    def test_not_grouped_prefix(self):
        # Must NOT collide with the grouped_gemm_bquant bridge namespace.
        for ctor in _ALL:
            self.assertFalse(ctor().name.startswith("grouped_"), ctor().name)


class TestNameContract(unittest.TestCase):
    def _assert_contract(self, cfg):
        expected = make_bquant_kernel_name(
            variant_key=cfg.variant_key,
            layout=cfg.layout,
            pipeline=cfg.pipeline,
            epilogue=cfg.epilogue,
            scheduler=cfg.scheduler,
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            warp_tile_m=cfg.warp_tile_m, warp_tile_n=cfg.warp_tile_n,
            warp_tile_k=cfg.warp_tile_k,
            quant_group_m=cfg.quant_group_m,
            quant_group_n=cfg.quant_group_n,
            quant_group_k=cfg.quant_group_k,
            preshuffle_b=cfg.preshuffle_b,
            preshuffle_bquant=cfg.preshuffle_bquant,
            name_prefix=NAME_PREFIX,
        )
        self.assertEqual(cfg.name, expected)

    def test_all_contracts(self):
        for ctor in _ALL:
            self._assert_contract(ctor())


class TestScope(unittest.TestCase):
    def test_base_variants(self):
        self.assertEqual([c().variant_key for c in _BASE],
                         ["fp8", "bf8", "fp8i4", "bf8i4"])

    def test_layout_is_rcr(self):
        for ctor in _ALL:
            self.assertEqual(ctor().layout, "rcr")

    def test_mx_pipeline_is_microscale(self):
        for ctor in _MX:
            self.assertEqual(ctor().pipeline, "microscale")

    def test_preshuffle_flags(self):
        self.assertFalse(default_fp8_config().preshuffle_b)
        self.assertTrue(default_fp8_preshuffleb_config().preshuffle_b)
        self.assertTrue(default_fp8_preshufflequant_config().preshuffle_bquant)
        pq = default_fp8_preshuffleb_bquant_config()
        self.assertTrue(pq.preshuffle_b and pq.preshuffle_bquant)


class TestCodegenProjection(unittest.TestCase):
    def test_to_codegen_config_roundtrip(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        self.assertEqual(d["variant_keys"], [cfg.variant_key])
        self.assertEqual(d["layouts"], [cfg.layout])
        self.assertEqual(d["quant_groups"][0]["quant_group_k"], cfg.quant_group_k)
        self.assertEqual(d["preshuffle_b"], cfg.preshuffle_b)


class TestProblem(unittest.TestCase):
    def test_problem_defaults(self):
        p = BQuantGemmProblem(M=256, N=256, K=256)
        self.assertEqual(p.k_batch, 1)
        self.assertEqual(p.quant_group_k, 128)


class TestArchSafety(unittest.TestCase):
    """Round-2 arch-safety hardening (get_arch+throw)."""

    def test_mx_requires_gfx950(self):
        # Every MX variant must reject a non-gfx950 arch with a clear error.
        for v in sorted(_MX_VARIANTS):
            with self.assertRaises(ValueError):
                _require_mx_arch(v, "gfx942")
        # gfx950 is accepted (no raise).
        for v in sorted(_MX_VARIANTS):
            _require_mx_arch(v, "gfx950")

    def test_non_mx_variant_any_arch_ok(self):
        # Non-MX variants are not restricted by the MX guard.
        for v in ("fp8", "bf8", "fp8i4", "bf8i4"):
            _require_mx_arch(v, "gfx942")
            _require_mx_arch(v, "gfx950")

    def test_setup_rejects_mx_on_non_gfx950(self):
        # The build entry point must fail early (before hipcc) for MX on gfx942.
        cfg = default_mx_bf16bf16_config(gfx_arch="gfx942")
        with self.assertRaises(ValueError):
            setup_multiple_bquant_dispatchers([cfg], gfx_arch="gfx942")


class TestArchAwareWarpTileK(unittest.TestCase):
    """Round-4: warp_tile_k must be arch-derived, mirroring get_k_warp_tile.

    The fp8/bf8 (and i4, which instantiate an 8-bit-float PrecType) default
    configs previously hardcoded warp_tile_k=128, which is a gfx950-only value.
    On gfx942 a warp_tile_k=128 fp8/bf8 kernel *compiles* but silently outputs
    ALL-ZEROS (there is no valid 16x16x128 fp8/bf8 warp-gemm on gfx942) -- the
    same trap already GPU-confirmed on the sibling tensor_quant/rowcolquant/
    aquant/abquant bridges.  So warp_tile_k MUST be 32 (decode) / 64 (preshuffle_b)
    on gfx942 and 128 on gfx950, and that value must flow into the byte-exact .name.
    """

    def test_helper_decode(self):
        # IsFlatMM=false (decode / preshufflequant): 128 gfx950, 32 gfx942.
        self.assertEqual(_warp_tile_k_for("gfx942"), 32)
        self.assertEqual(_warp_tile_k_for("gfx950"), 128)
        # Arch strings with feature suffixes must still resolve.
        self.assertEqual(_warp_tile_k_for("gfx942:sramecc+:xnack-"), 32)
        self.assertEqual(_warp_tile_k_for("gfx950:sramecc+:xnack-"), 128)

    def test_helper_preshuffleb_flatmm(self):
        # IsFlatMM=true (preshuffle_b): 128 gfx950, 64 gfx942.
        self.assertEqual(_warp_tile_k_for("gfx942", is_flatmm=True), 64)
        self.assertEqual(_warp_tile_k_for("gfx950", is_flatmm=True), 128)

    def test_decode_configs_arch_aware(self):
        # fp8/bf8 AND fp8i4/bf8i4 decode: 32 on gfx942, 128 on gfx950.
        for ctor in (default_fp8_config, default_bf8_config,
                     default_fp8i4_config, default_bf8i4_config):
            self.assertEqual(ctor(gfx_arch="gfx942").warp_tile_k, 32, ctor.__name__)
            self.assertEqual(ctor(gfx_arch="gfx950").warp_tile_k, 128, ctor.__name__)

    def test_preshufflequant_configs_arch_aware(self):
        # preshuffle_bquant (IsFlatMM=false): 32 gfx942, 128 gfx950.
        for ctor in (default_fp8_preshufflequant_config,):
            self.assertEqual(ctor(gfx_arch="gfx942").warp_tile_k, 32, ctor.__name__)
            self.assertEqual(ctor(gfx_arch="gfx950").warp_tile_k, 128, ctor.__name__)

    def test_preshuffleb_configs_arch_aware(self):
        # preshuffle_b (IsFlatMM=true): 64 gfx942, 128 gfx950 (fp8 + i4).
        for ctor in (default_fp8_preshuffleb_config,
                     default_fp8i4_preshuffleb_config,
                     default_bf8i4_preshuffleb_config,
                     default_fp8_preshuffleb_bquant_config):
            self.assertEqual(ctor(gfx_arch="gfx942").warp_tile_k, 64, ctor.__name__)
            self.assertEqual(ctor(gfx_arch="gfx950").warp_tile_k, 128, ctor.__name__)

    def test_warp_tile_k_flows_into_name(self):
        # The chosen warp_tile_k must appear byte-exact in the kernel .name.
        n942 = default_fp8_config(gfx_arch="gfx942").name
        n950 = default_fp8_config(gfx_arch="gfx950").name
        self.assertIn("16x16x32", n942)
        self.assertIn("16x16x128", n950)
        self.assertNotEqual(n942, n950)

    def test_mx_gfx950_values(self):
        # MX is gfx950-only; verified against Old-TE get_k_warp_tile<bf16,16>()
        # (=32 for bf16bf16/bf16fp4) and GemmConfigMixedPrecision (=64 for bf16bf8).
        self.assertEqual(default_mx_bf16bf16_config(gfx_arch="gfx950").warp_tile_k, 32)
        self.assertEqual(default_mx_bf16fp4_config(gfx_arch="gfx950").warp_tile_k, 32)
        self.assertEqual(default_mx_bf16bf8_config(gfx_arch="gfx950").warp_tile_k, 64)


class TestSplitKTrap(unittest.TestCase):
    """k_batch > 1 must be rejected, never silently passed through."""

    def test_kbatch_gt1_rejected(self):
        # Exercise the runner's guard without a GPU by stubbing the ctypes lib.
        import gemm_bquant_utils as gbu

        runner = gbu.BQuantGpuGemmRunner.__new__(gbu.BQuantGpuGemmRunner)
        runner._lib = None  # guard must fire before any lib access

        prob = BQuantGemmProblem(M=16, N=64, K=256, k_batch=2)
        with self.assertRaises(ValueError):
            runner.run(A=None, B=None, BQ=None, problem=prob)


class TestPreshuffleBMatrixShuffle(unittest.TestCase):
    """Round-3 BUG #1: PreshuffleB kernels must pre-shuffle the B WEIGHT matrix
    (Old-TE shuffle_b / shuffle_b_permuteN, run_gemm_quant_example.inc:770-789).
    Previously the ctypes lib only plain-copied B, so fp8/bf8 preshuffleb (+pq)
    returned garbage (max_rel ~67-69 on gfx950). The bq_permuteN path for the BQ
    scales (inc:799-815) must be applied too."""

    def test_ctypes_lib_has_b_matrix_shuffle_step(self):
        # The ctypes lib must call shuffle_b / shuffle_b_permuteN on B for
        # PreshuffleB kernels, gated by SelectedKernel::PreshuffleB.
        self.assertIn("SelectedKernel::PreshuffleB", _CTYPES_SRC)
        self.assertIn("shuffle_b<typename SelectedKernel::BShuffleConfig>", _CTYPES_SRC)
        self.assertIn(
            "shuffle_b_permuteN<typename SelectedKernel::BShuffleConfig>", _CTYPES_SRC
        )
        # permute_n variant is selected exactly when TiledMMAPermuteN && kN==1.
        self.assertIn("SelectedKernel::TiledMMAPermuteN", _CTYPES_SRC)
        self.assertIn("QuantGroupSize::kN == 1", _CTYPES_SRC)
        # The BQ scales must also be bq_permuteN'd for the permuteN case.
        self.assertIn("bq_permuteN<typename SelectedKernel::BShuffleConfig>", _CTYPES_SRC)

    def test_preshuffleb_headers_expose_bshuffle_config(self):
        preshuffleb_ctors = [
            default_fp8_preshuffleb_config,
            default_bf8_preshuffleb_config,
            default_fp8i4_preshuffleb_config,
            default_bf8i4_preshuffleb_config,
            default_fp8_preshuffleb_bquant_config,
        ]
        for ctor in preshuffleb_ctors:
            cfg = ctor()
            self.assertTrue(cfg.preshuffle_b, cfg.name)
            text = _header_text(cfg)
            self.assertTrue(_static_bool(text, "PreshuffleB"), cfg.name)
            self.assertIn("struct BShuffleConfig", text, cfg.name)
            # BShuffleConfig must expose the member names shuffle_b expects.
            for member in ("N_Tile", "N_Warp", "N_Warp_Tile", "K_Warp_Tile"):
                self.assertIn(member, text, f"{member} missing in {cfg.name}")
            # preshuffleb default (tile_n=128, warp_n=4, warp_tile_n=16):
            # N_Repeat = 128/16/4 = 2 -> TiledMMAPermuteN true.
            self.assertTrue(_static_bool(text, "TiledMMAPermuteN"), cfg.name)

    def test_non_preshuffleb_kernels_have_no_b_shuffle(self):
        # Non-preshuffleB kernels must NOT pre-shuffle B (PreshuffleB=false,
        # TiledMMAPermuteN=false).
        for ctor in (default_fp8_config, default_bf8_config, default_fp8i4_config,
                     default_bf8i4_config, default_fp8_preshufflequant_config,
                     default_mx_bf16bf16_config):
            cfg = ctor()
            self.assertFalse(cfg.preshuffle_b, cfg.name)
            text = _header_text(cfg)
            self.assertFalse(_static_bool(text, "PreshuffleB"), cfg.name)
            self.assertFalse(_static_bool(text, "TiledMMAPermuteN"), cfg.name)


class TestPkInt4Permute(unittest.TestCase):
    """Round-3 BUG #2: pk_int4 B (fp8i4 / bf8i4) must be permuted with
    permute_vectors_i4x4_b UNCONDITIONALLY before the device copy, exactly as
    Old-TE does (run_gemm_quant_example.inc:784-787). Without it fp8i4/bf8i4 were
    broken in all phases (NaN on random, all-zeros on constant)."""

    def test_ctypes_lib_permutes_pk_int4_b(self):
        self.assertIn("permute_vectors_i4x4_b", _CTYPES_SRC)
        # Applied for pk_int4 B specifically.
        self.assertIn("std::is_same_v<BDataType, ck_tile::pk_int4_t>", _CTYPES_SRC)
        # The permute helper header must be included.
        self.assertIn("ck_tile/host/permute_pk_int4.hpp", _CTYPES_SRC)

    def test_i4_variants_use_pk_int4_bdatatype(self):
        for ctor in (default_fp8i4_config, default_bf8i4_config,
                     default_fp8i4_preshuffleb_config, default_bf8i4_preshuffleb_config):
            cfg = ctor()
            text = _header_text(cfg)
            self.assertIn("using BDataType   = ck_tile::pk_int4_t", text, cfg.name)


class TestBCastPolicy(unittest.TestCase):
    """Round-3 BUG #3: mx_bf16bf16 (and every A==B kernel) must compile the same
    pipeline Old-TE uses. Old-TE (run_gemm_quant_example.inc:117-120) sets
    b_cast_policy = (A==B) ? BeforeLDSWrite : AfterLDSRead. The bridge previously
    left the GemmBQuantPipelineProblem BCastPolicy_ arg at its AfterLDSRead default
    for every kernel, so mx_bf16bf16 built a slower pipeline (~43% off on gfx950)."""

    def test_same_dtype_kernels_use_before_lds_write(self):
        # A == B: fp8/fp8, bf8/bf8, mx bf16/bf16.
        for ctor in (default_fp8_config, default_bf8_config, default_mx_bf16bf16_config):
            text = _header_text(ctor())
            self.assertIn("ck_tile::CastPolicy::BeforeLDSWrite", text, ctor().name)
            self.assertNotIn("ck_tile::CastPolicy::AfterLDSRead", text, ctor().name)

    def test_mixed_dtype_kernels_use_after_lds_read(self):
        # A != B: fp8i4 (fp8/pk_int4), bf8i4, mx_bf16bf8 (bf16/bf8),
        # mx_bf16fp4 (bf16/pk_fp4).
        for ctor in (default_fp8i4_config, default_bf8i4_config,
                     default_mx_bf16bf8_config, default_mx_bf16fp4_config):
            text = _header_text(ctor())
            self.assertIn("ck_tile::CastPolicy::AfterLDSRead", text, ctor().name)
            self.assertNotIn("ck_tile::CastPolicy::BeforeLDSWrite", text, ctor().name)


class TestPackedBCopyCount(unittest.TestCase):
    """Round-5 BUG #1: for packed B (pk_int4_t / pk_fp4_t; PackedSize=2) the
    host copy into b_k_n must copy the DESTINATION element count, not K*N.
    HostTensor<T>::get_element_space_size() divides by PackedSize, so the tensor
    holds only K*N/2 elements; copying K*N overran the buffer and corrupted the
    heap BEFORE permute_vectors_i4x4_b ran, crashing all i4 (fp8i4/bf8i4) and
    mx_bf16fp4 configs."""

    def test_packed_b_copy_uses_destination_size(self):
        # The overflowing copy (B_host + K * N into b_k_n) must be gone.
        self.assertNotIn("B_host + K * N", _CTYPES_SRC)
        # The copy must be bounded by the destination tensor's own size.
        self.assertIn("std::copy(B_host, B_host + b_k_n.size(), b_k_n.begin())",
                      _CTYPES_SRC)

    def test_packed_pk_int4_permute_still_runs(self):
        # The pk_int4 permute must still be present (it only runs once the copy
        # no longer corrupts the heap).
        self.assertIn("permute_vectors_i4x4_b", _CTYPES_SRC)
        self.assertIn("std::is_same_v<BDataType, ck_tile::pk_int4_t>", _CTYPES_SRC)


class TestEpilogueDependentCDepermute(unittest.TestCase):
    """Round-5 BUG #2: the permute_n C de-permute is EPILOGUE-DEPENDENT.
    PreshuffleB (WPQuantB) kernels need the FORWARD riffle C = C[:, _logical]
    (gfx942 tester: max_rel ~4.7e-4 on all 4 preshuffleb configs); CompV3 /
    preshufflequant kernels keep the INVERSE riffle _Cp[:, _logical] = C."""

    def test_utils_selects_direction_by_epilogue(self):
        # The runner must branch on whether the kernel name is a preshuffleb one,
        # using a delimiter-aware token match (not a bare substring, which would
        # false-positive on the "preshufflebq" preshufflequant token).
        self.assertIn(r"(?:^|_)preshuffleb(?:_|$)", _UTILS_SRC)
        # Forward riffle for PreshuffleB.
        self.assertIn("C = C[:, _logical]", _UTILS_SRC)
        # Inverse riffle retained for CompV3 / preshufflequant.
        self.assertIn("_Cp[:, _logical] = C", _UTILS_SRC)

    def test_preshuffleb_token_match_excludes_preshufflequant(self):
        # The token regex must fire for PreshuffleB names and NOT for the
        # preshufflequant ("preshufflebq") CompV3 name.
        import re as _re
        tok = re.compile(r'(?:^|_)preshuffleb(?:_|$)')
        # preshuffleb and preshuffleb+bquant -> forward riffle.
        self.assertTrue(tok.search(default_fp8_preshuffleb_config().name))
        self.assertTrue(tok.search(default_fp8_preshuffleb_bquant_config().name))
        # preshufflequant (preshufflebq only) -> inverse riffle.
        self.assertIsNone(tok.search(default_fp8_preshufflequant_config().name))

    def test_depermute_forward_then_inverse_are_true_inverses(self):
        # Sanity-check the two riffles: applying the inverse to a forward-riffled
        # array recovers the original, so the two epilogues use mirror ops.
        import numpy as np
        N, r = 8, 2
        half = N // r
        logical = [(c % r) * half + (c // r) for c in range(N)]
        col = np.arange(N).reshape(1, N)
        forward = col[:, logical]              # PreshuffleB path
        inverse = np.empty_like(col)
        inverse[:, logical] = forward          # CompV3 path applied to forward
        self.assertTrue(np.array_equal(inverse, col))

    def test_preshuffleb_kernel_names_are_detected(self):
        # The preshuffleb configs must produce names whose "preshuffleb" token
        # fires the forward-riffle branch; non-preshuffleb ones must not.
        tok = re.compile(r'(?:^|_)preshuffleb(?:_|$)')
        for ctor in (default_fp8_preshuffleb_config,
                     default_fp8_preshuffleb_bquant_config):
            self.assertTrue(tok.search(ctor().name), ctor().name)
        for ctor in (default_fp8_config, default_fp8_preshufflequant_config):
            self.assertIsNone(tok.search(ctor().name), ctor().name)


if __name__ == "__main__":
    unittest.main()
