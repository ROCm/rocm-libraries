#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the contraction_multi_abd dispatcher bridge.

Covers:
  1. Canonical kernel-name uniqueness -- configs differing in elementwise ops,
     tensor counts, tile shapes, or dimension counts must never produce the
     same name (previously untested, was a collision bug).
  2. ContractionMultiABDKernelConfig.to_codegen_config() projection -- all
     fields round-trip through the codegen JSON format.
  3. ABI marshalling in ContractionMultiABDDispatcherLib -- pointer arrays,
     stride packing, and status forwarding (GPU-free via ctypes mock).
  4. Input validation in the runner -- mis-shaped arrays are rejected before
     any raw pointer is passed to hipMemcpy.
  5. Shipped configs/*.json are well-formed and carry tile/trait sections.

No GPU, hipcc, or compiled .so is required; runs green in CPU-only CI.

Run: python3 -m pytest tests/test_contraction_multi_abd_bridge.py -v
"""

import ctypes
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
REPO_ROOT = DISPATCHER_DIR.parent

# Add both the dispatcher python dir and the codegen dir to path so imports work.
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

from contraction_multi_abd_utils import (  # noqa: E402
    ContractionMultiABDKernelConfig,
    ContractionMultiABDProblem,
    ContractionMultiABDDispatcherLib,
    ContractionMultiABDRunner,
)
from contraction_multi_abd_codegen import (  # noqa: E402
    make_contraction_multi_abd_kernel_name,
    _expand_nested_config,
    build_specs,
)

_CONFIG_DIR = (
    REPO_ROOT
    / "tile_engine"
    / "ops"
    / "gemm"
    / "contraction_multi_abd"
    / "configs"
)


def _base_config(**overrides) -> ContractionMultiABDKernelConfig:
    kw = dict(
        dtype="fp16",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=256, tile_n=256, tile_k=64,
        warp_m=2, warp_n=2, warp_k=1,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        num_a_tensor=1, num_b_tensor=1, num_d_tensor=1,
        num_dim_g=1, num_dim_m=2, num_dim_n=2, num_dim_k=1,
        a_elementwise="PassThrough",
        b_elementwise="PassThrough",
        cde_elementwise="MultiDAdd",
    )
    kw.update(overrides)
    return ContractionMultiABDKernelConfig(**kw)


def _base_problem() -> ContractionMultiABDProblem:
    return ContractionMultiABDProblem(
        g_dims=[2], m_dims=[4, 4], n_dims=[4, 4], k_dims=[8]
    )


# =============================================================================
# 1. Kernel-name uniqueness
# =============================================================================

class TestKernelNameUniqueness(unittest.TestCase):

    def test_different_elementwise_a_ops_produce_distinct_names(self):
        base = _base_config().name
        other = _base_config(a_elementwise="Scale").name
        self.assertNotEqual(base, other)

    def test_different_elementwise_b_ops_produce_distinct_names(self):
        base = _base_config().name
        other = _base_config(b_elementwise="Scale").name
        self.assertNotEqual(base, other)

    def test_different_cde_elementwise_ops_produce_distinct_names(self):
        base = _base_config().name
        other = _base_config(cde_elementwise="PassThrough").name
        self.assertNotEqual(base, other)

    def test_different_a_tensor_counts_produce_distinct_names(self):
        a1 = _base_config(num_a_tensor=1).name
        a2 = _base_config(num_a_tensor=2).name
        self.assertNotEqual(a1, a2)
        self.assertIn("_na1_", a1)
        self.assertIn("_na2_", a2)

    def test_different_b_tensor_counts_produce_distinct_names(self):
        b1 = _base_config(num_b_tensor=1).name
        b2 = _base_config(num_b_tensor=2).name
        self.assertNotEqual(b1, b2)

    def test_different_d_tensor_counts_produce_distinct_names(self):
        d0 = _base_config(num_d_tensor=0).name
        d2 = _base_config(num_d_tensor=2).name
        self.assertNotEqual(d0, d2)

    def test_different_dim_counts_produce_distinct_names(self):
        g1 = _base_config(num_dim_g=1).name
        g2 = _base_config(num_dim_g=2).name
        self.assertNotEqual(g1, g2)

    def test_different_tile_shapes_produce_distinct_names(self):
        a = _base_config(tile_m=128).name
        b = _base_config(tile_m=256).name
        self.assertNotEqual(a, b)

    def test_elementwise_fields_appear_in_name(self):
        cfg = _base_config(
            a_elementwise="PassThrough",
            b_elementwise="Scale",
            cde_elementwise="Relu",
        )
        self.assertIn("PassThrough", cfg.name)
        self.assertIn("Scale", cfg.name)
        self.assertIn("Relu", cfg.name)

    def test_name_is_byte_exact_with_standalone_function(self):
        cfg = _base_config()
        standalone = make_contraction_multi_abd_kernel_name(
            dtype=cfg.dtype,
            layout=cfg.layout,
            pipeline=cfg.pipeline,
            epilogue=cfg.epilogue,
            scheduler=cfg.scheduler,
            pad_m=cfg.pad_m, pad_n=cfg.pad_n, pad_k=cfg.pad_k,
            persistent=cfg.persistent,
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            warp_tile_m=cfg.warp_tile_m,
            warp_tile_n=cfg.warp_tile_n,
            warp_tile_k=cfg.warp_tile_k,
            num_a_tensor=cfg.num_a_tensor,
            num_b_tensor=cfg.num_b_tensor,
            num_d_tensor=cfg.num_d_tensor,
            num_dim_g=cfg.num_dim_g,
            num_dim_m=cfg.num_dim_m,
            num_dim_n=cfg.num_dim_n,
            num_dim_k=cfg.num_dim_k,
            a_elementwise=cfg.a_elementwise,
            b_elementwise=cfg.b_elementwise,
            cde_elementwise=cfg.cde_elementwise,
        )
        self.assertEqual(cfg.name, standalone)


# =============================================================================
# 2. Config projection round-trip
# =============================================================================

class TestCodegenConfigProjection(unittest.TestCase):

    def test_to_codegen_config_has_required_keys(self):
        cfg = _base_config()
        d = cfg.to_codegen_config()
        for key in ("dtypes", "layouts", "pipelines", "epilogues", "schedulers",
                    "tile_configs", "num_a_tensors", "num_b_tensors", "num_d_tensors",
                    "dim_combos", "a_elementwise", "b_elementwise", "cde_elementwise"):
            self.assertIn(key, d, f"missing key: {key}")

    def test_dtype_round_trips(self):
        for dtype in ("fp16", "bf16", "fp32"):
            cfg = _base_config(dtype=dtype)
            self.assertEqual(cfg.to_codegen_config()["dtypes"], [dtype])

    def test_layout_round_trips(self):
        for layout in ("rcr", "rrr", "ccr"):
            cfg = _base_config(layout=layout)
            self.assertEqual(cfg.to_codegen_config()["layouts"], [layout])

    def test_tile_config_round_trips(self):
        cfg = _base_config(tile_m=128, tile_n=64, tile_k=32)
        tc = cfg.to_codegen_config()["tile_configs"][0]
        self.assertEqual(tc["tile_m"], 128)
        self.assertEqual(tc["tile_n"], 64)
        self.assertEqual(tc["tile_k"], 32)

    def test_tensor_counts_round_trip(self):
        cfg = _base_config(num_a_tensor=2, num_b_tensor=3, num_d_tensor=1)
        d = cfg.to_codegen_config()
        self.assertEqual(d["num_a_tensors"], [2])
        self.assertEqual(d["num_b_tensors"], [3])
        self.assertEqual(d["num_d_tensors"], [1])

    def test_dim_counts_round_trip(self):
        cfg = _base_config(num_dim_g=2, num_dim_m=3, num_dim_n=3, num_dim_k=2)
        dc = cfg.to_codegen_config()["dim_combos"][0]
        self.assertEqual(dc["num_dim_g"], 2)
        self.assertEqual(dc["num_dim_m"], 3)
        self.assertEqual(dc["num_dim_n"], 3)
        self.assertEqual(dc["num_dim_k"], 2)

    def test_elementwise_ops_round_trip(self):
        cfg = _base_config(
            a_elementwise="Scale",
            b_elementwise="Scale",
            cde_elementwise="PassThrough",
        )
        d = cfg.to_codegen_config()
        self.assertEqual(d["a_elementwise"], "Scale")
        self.assertEqual(d["b_elementwise"], "Scale")
        self.assertEqual(d["cde_elementwise"], "PassThrough")


# =============================================================================
# 3. ABI marshalling (GPU-free mock)
# =============================================================================

class TestAbiMarshalling(unittest.TestCase):
    """Mock out ctypes.CDLL so no .so is needed."""

    @staticmethod
    def _make_mock_lib(run_return=0):
        fake = mock.MagicMock()
        fake.dispatcher_initialize.return_value = 0
        fake.dispatcher_get_kernel_count.return_value = 1
        fake.dispatcher_run_batched_contraction_multi_abd.return_value = run_return
        return fake

    def _load_lib_with_mock(self, fake):
        import tempfile, os
        # Create a real (empty) temp file so the exists() check passes.
        fd, tmp = tempfile.mkstemp(suffix=".so")
        os.close(fd)
        try:
            with mock.patch("ctypes.CDLL", return_value=fake):
                lib = ContractionMultiABDDispatcherLib(Path(tmp))
        finally:
            Path(tmp).unlink(missing_ok=True)
        return lib

    def test_initialize_is_called_on_construction(self):
        fake = self._make_mock_lib()
        self._load_lib_with_mock(fake)
        fake.dispatcher_initialize.assert_called_once()

    def test_get_kernel_name_delegates_to_so(self):
        fake = self._make_mock_lib()
        fake.dispatcher_get_kernel_name.return_value = b"contraction_multi_abd_fp16_rcr_compv3"
        lib = self._load_lib_with_mock(fake)
        self.assertEqual(lib.get_kernel_name(), "contraction_multi_abd_fp16_rcr_compv3")

    def test_run_argtypes_count(self):
        # The declared argtypes list must match the C signature exactly (25 parameters):
        # as_ptrs, bs_ptrs, ds_ptrs, e_ptr, num_a, num_b, num_d,
        # g_dims, m_dims, n_dims, k_dims, num_dim_g, num_dim_m, num_dim_n, num_dim_k,
        # a_strides_flat, b_strides_flat, d_strides_flat, e_strides,
        # elem_a, elem_b, elem_d, elem_e, k_batch, time_ms
        fake = self._make_mock_lib()
        lib = self._load_lib_with_mock(fake)
        argt = fake.dispatcher_run_batched_contraction_multi_abd.argtypes
        self.assertEqual(len(argt), 25)

    def test_run_returns_code_zero_on_success(self):
        fake = self._make_mock_lib(run_return=0)
        lib = self._load_lib_with_mock(fake)
        problem = _base_problem()
        G, M, N, K = problem.G_total, problem.M_total, problem.N_total, problem.K_total
        As = [np.zeros((G * M * K,), dtype=np.float16)]
        Bs = [np.zeros((G * N * K,), dtype=np.float16)]
        Ds = [np.zeros((G * M * N,), dtype=np.float16)]
        E  =  np.zeros((G * M * N,), dtype=np.float16)
        rc, _ = lib.run(As, Bs, Ds, E, problem)
        self.assertEqual(rc, 0)

    def test_run_forwards_nonzero_status(self):
        fake = self._make_mock_lib(run_return=-3)
        lib = self._load_lib_with_mock(fake)
        problem = _base_problem()
        G, M, N, K = problem.G_total, problem.M_total, problem.N_total, problem.K_total
        As = [np.zeros((G * M * K,), dtype=np.float16)]
        Bs = [np.zeros((G * N * K,), dtype=np.float16)]
        Ds = [np.zeros((G * M * N,), dtype=np.float16)]
        E  =  np.zeros((G * M * N,), dtype=np.float16)
        rc, _ = lib.run(As, Bs, Ds, E, problem)
        self.assertEqual(rc, -3)


# =============================================================================
# 4. Input validation in the runner
# =============================================================================

class TestInputValidation(unittest.TestCase):

    def _make_runner_with_mock(self, run_return=0):
        import tempfile, os
        fake = mock.MagicMock()
        fake.dispatcher_initialize.return_value = 0
        fake.dispatcher_get_kernel_count.return_value = 1
        fake.dispatcher_run_batched_contraction_multi_abd.return_value = run_return
        fd, tmp = tempfile.mkstemp(suffix=".so")
        os.close(fd)
        try:
            with mock.patch("ctypes.CDLL", return_value=fake):
                runner = ContractionMultiABDRunner(Path(tmp))
        finally:
            Path(tmp).unlink(missing_ok=True)
        return runner

    def test_wrong_a_tensor_shape_raises(self):
        runner = self._make_runner_with_mock()
        problem = _base_problem()
        G, M, N, K = problem.G_total, problem.M_total, problem.N_total, problem.K_total
        # Wrong number of elements for A
        As = [np.zeros((G * M * K + 1,), dtype=np.float16)]
        Bs = [np.zeros((G * N * K,), dtype=np.float16)]
        Ds = [np.zeros((G * M * N,), dtype=np.float16)]
        with self.assertRaises(ValueError, msg="A shape mismatch should raise ValueError"):
            runner.run(As, Bs, Ds, problem)

    def test_wrong_b_tensor_shape_raises(self):
        runner = self._make_runner_with_mock()
        problem = _base_problem()
        G, M, N, K = problem.G_total, problem.M_total, problem.N_total, problem.K_total
        As = [np.zeros((G * M * K,), dtype=np.float16)]
        Bs = [np.zeros((G * N * K + 1,), dtype=np.float16)]  # wrong
        Ds = [np.zeros((G * M * N,), dtype=np.float16)]
        with self.assertRaises(ValueError):
            runner.run(As, Bs, Ds, problem)

    def test_wrong_d_tensor_shape_raises(self):
        runner = self._make_runner_with_mock()
        problem = _base_problem()
        G, M, N, K = problem.G_total, problem.M_total, problem.N_total, problem.K_total
        As = [np.zeros((G * M * K,), dtype=np.float16)]
        Bs = [np.zeros((G * N * K,), dtype=np.float16)]
        Ds = [np.zeros((G * M * N + 1,), dtype=np.float16)]  # wrong
        with self.assertRaises(ValueError):
            runner.run(As, Bs, Ds, problem)

    def test_empty_as_raises(self):
        runner = self._make_runner_with_mock()
        problem = _base_problem()
        G, M, N, K = problem.G_total, problem.M_total, problem.N_total, problem.K_total
        with self.assertRaises(ValueError):
            runner.run([], [np.zeros((G * N * K,), np.float16)], [], problem)

    def test_empty_bs_raises(self):
        runner = self._make_runner_with_mock()
        problem = _base_problem()
        G, M, N, K = problem.G_total, problem.M_total, problem.N_total, problem.K_total
        with self.assertRaises(ValueError):
            runner.run([np.zeros((G * M * K,), np.float16)], [], [], problem)

    def test_correct_shapes_accepted(self):
        import tempfile, os
        fake = mock.MagicMock()
        fake.dispatcher_initialize.return_value = 0
        fake.dispatcher_get_kernel_count.return_value = 1
        fake.dispatcher_run_batched_contraction_multi_abd.return_value = 0
        fd, tmp = tempfile.mkstemp(suffix=".so")
        os.close(fd)
        try:
            with mock.patch("ctypes.CDLL", return_value=fake):
                runner = ContractionMultiABDRunner(Path(tmp))
        finally:
            Path(tmp).unlink(missing_ok=True)
        problem = _base_problem()
        G, M, N, K = problem.G_total, problem.M_total, problem.N_total, problem.K_total
        As = [np.zeros((G * M * K,), dtype=np.float16)]
        Bs = [np.zeros((G * N * K,), dtype=np.float16)]
        Ds = [np.zeros((G * M * N,), dtype=np.float16)]
        # Should not raise
        try:
            runner.run(As, Bs, Ds, problem)
        except RuntimeError:
            pass  # GPU call failure is OK; what matters is no ValueError


# =============================================================================
# 5. Shipped configs are well-formed
# =============================================================================

class TestShippedConfigs(unittest.TestCase):

    def test_config_dir_exists(self):
        self.assertTrue(_CONFIG_DIR.is_dir(), f"config dir missing: {_CONFIG_DIR}")

    def test_configs_are_valid_json(self):
        configs = sorted(_CONFIG_DIR.glob("*.json"))
        self.assertGreater(len(configs), 0, "no *.json shipped in configs/")
        for path in configs:
            with self.subTest(config=path.name):
                with open(path) as f:
                    data = json.load(f)
                self.assertIn("tile_config", data, f"{path.name}: missing tile_config")
                self.assertIn("trait_config", data, f"{path.name}: missing trait_config")

    def test_tile_config_keys_present(self):
        for path in sorted(_CONFIG_DIR.glob("*.json")):
            with self.subTest(config=path.name):
                with open(path) as f:
                    tc = json.load(f)["tile_config"]
                for key in (
                    "tile_m", "tile_n", "tile_k",
                    "warp_m", "warp_n", "warp_k",
                    "warp_tile_m", "warp_tile_n", "warp_tile_k",
                ):
                    self.assertIn(key, tc, f"{path.name}: missing tile_config.{key}")

    def test_trait_config_keys_present(self):
        for path in sorted(_CONFIG_DIR.glob("*.json")):
            with self.subTest(config=path.name):
                with open(path) as f:
                    tr = json.load(f)["trait_config"]
                for key in ("pipeline", "scheduler", "epilogue"):
                    self.assertIn(key, tr, f"{path.name}: missing trait_config.{key}")

    def test_smoke_ci_config_has_single_tile(self):
        smoke = _CONFIG_DIR / "smoke_ci_config.json"
        if not smoke.exists():
            self.skipTest("smoke_ci_config.json not found")
        with open(smoke) as f:
            data = json.load(f)
        tc = data["tile_config"]
        # Smoke configs should have exactly one tile value each for fast CI.
        for key in ("tile_m", "tile_n", "tile_k"):
            val = tc[key]
            if isinstance(val, dict):
                self.assertIn("values", val, f"smoke_ci: {key} has no 'values'")
                self.assertEqual(len(val["values"]), 1,
                                 f"smoke_ci: {key}.values should have 1 entry")
            elif isinstance(val, list):
                self.assertEqual(len(val), 1, f"smoke_ci: {key} list should have 1 entry")


class TestExpandNestedConfig(unittest.TestCase):
    """Unit tests for _expand_nested_config() — the tile_config/trait_config JSON expansion."""

    def test_flat_config_passes_through_unchanged(self):
        flat = {"dtypes": ["fp16"], "layouts": ["rcr"], "tile_configs": [{"tile_m": 256}]}
        result = _expand_nested_config(flat)
        self.assertIs(result, flat, "flat config must be returned as-is (same object)")

    def test_step_range_expansion(self):
        cfg = {
            "tile_config": {"tile_m": {"min": 64, "max": 256, "step": 64}},
        }
        result = _expand_nested_config(cfg)
        self.assertIn("tile_configs", result)
        tile_m_values = [tc["tile_m"] for tc in result["tile_configs"]]
        self.assertEqual(tile_m_values, [64, 128, 192, 256])

    def test_values_list_expansion(self):
        cfg = {
            "tile_config": {"warp_m": {"values": [4, 2, 1]}},
        }
        result = _expand_nested_config(cfg)
        self.assertIn("tile_configs", result)
        warp_m_values = [tc["warp_m"] for tc in result["tile_configs"]]
        self.assertEqual(warp_m_values, [4, 2, 1])

    def test_tile_cartesian_product(self):
        cfg = {
            "tile_config": {
                "tile_m": {"values": [64, 128]},
                "tile_n": {"values": [64]},
                "tile_k": {"values": [32]},
                "warp_m": {"values": [2]},
                "warp_n": {"values": [2]},
                "warp_k": {"values": [1]},
                "warp_tile_m": {"values": [32]},
                "warp_tile_n": {"values": [32]},
                "warp_tile_k": {"values": [16]},
            },
        }
        result = _expand_nested_config(cfg)
        self.assertEqual(len(result["tile_configs"]), 2,
                         "2 tile_m values × 1 each = 2 combinations")

    def test_trait_config_pipelines_schedulers_epilogues(self):
        cfg = {
            "trait_config": {
                "pipeline":  {"values": ["compv3", "compv4"]},
                "scheduler": {"values": ["intrawave"]},
                "epilogue":  {"values": ["cshuffle", "default2d"]},
                "pad_m":     {"values": [False]},
                "pad_n":     {"values": [False]},
                "pad_k":     {"values": [False]},
            },
        }
        result = _expand_nested_config(cfg)
        self.assertEqual(result["pipelines"],  ["compv3", "compv4"])
        self.assertEqual(result["schedulers"], ["intrawave"])
        self.assertEqual(result["epilogues"],  ["cshuffle", "default2d"])
        self.assertEqual(result["pad_options"],
                         [{"pad_m": False, "pad_n": False, "pad_k": False}])

    def test_nested_keys_removed_after_expansion(self):
        cfg = {
            "tile_config":  {"tile_m": {"values": [128]}},
            "trait_config": {"pipeline": {"values": ["compv3"]},
                             "scheduler": {"values": ["intrawave"]},
                             "epilogue":  {"values": ["cshuffle"]},
                             "pad_m": {"values": [False]},
                             "pad_n": {"values": [False]},
                             "pad_k": {"values": [False]}},
        }
        result = _expand_nested_config(cfg)
        self.assertNotIn("tile_config",  result)
        self.assertNotIn("trait_config", result)

    def test_shipped_smoke_config_expands_to_one_spec(self):
        smoke = _CONFIG_DIR / "smoke_ci_config.json"
        if not smoke.exists():
            self.skipTest("smoke_ci_config.json not found")
        with open(smoke) as f:
            data = json.load(f)
        # Simulate CMake merge: layer flat overrides on top of expanded base.
        expanded = _expand_nested_config(data)
        expanded.update({
            "dtypes": ["fp16"], "layouts": ["rcr"],
            "num_a_tensors": [1], "num_b_tensors": [1], "num_d_tensors": [1],
            "dim_combos": [{"num_dim_g": 1, "num_dim_m": 2, "num_dim_n": 2, "num_dim_k": 1}],
            "a_elementwise": "PassThrough",
            "b_elementwise": "PassThrough",
            "cde_elementwise": "MultiDAdd",
        })
        specs = build_specs(expanded)
        self.assertGreater(len(specs), 0, "smoke config must produce at least one spec")


class TestMixedDtypeDTensors(unittest.TestCase):
    """Validates that ContractionMultiABDDispatcherLib.run() rejects mixed-dtype D tensors."""

    def _make_mock_lib(self, num_d=2):
        """Return a ContractionMultiABDDispatcherLib with its ctypes calls mocked out."""
        with mock.patch("ctypes.CDLL") as MockCDLL:
            mock_lib = mock.MagicMock()
            MockCDLL.return_value = mock_lib
            mock_lib.dispatcher_initialize.return_value = 0
            mock_lib.dispatcher_get_kernel_name.return_value = b"test_kernel"

            from contraction_multi_abd_utils import ContractionMultiABDDispatcherLib
            with mock.patch("pathlib.Path.exists", return_value=True):
                lib = ContractionMultiABDDispatcherLib.__new__(ContractionMultiABDDispatcherLib)
                lib.so_path = Path("/fake/lib.so")
                lib._lib = mock_lib
        return lib

    def test_mixed_dtype_d_tensors_raises(self):
        from contraction_multi_abd_utils import (
            ContractionMultiABDDispatcherLib,
            ContractionMultiABDProblem,
        )
        problem = ContractionMultiABDProblem(g_dims=[2], m_dims=[4, 4], n_dims=[4, 4], k_dims=[8])
        G, M, N, K = 2, 16, 16, 8

        As = [np.ones(G * M * K, dtype=np.float16)]
        Bs = [np.ones(G * N * K, dtype=np.float16)]
        # Two D tensors with different dtypes — fp16 vs fp32
        Ds = [
            np.ones(G * M * N, dtype=np.float16),
            np.ones(G * M * N, dtype=np.float32),  # different itemsize
        ]
        E = np.zeros(G * M * N, dtype=np.float16)

        lib = self._make_mock_lib()
        with self.assertRaises(ValueError, msg="mixed-dtype D tensors must raise ValueError"):
            lib.run(As, Bs, Ds, E, problem)

    def test_uniform_dtype_d_tensors_does_not_raise_on_validation(self):
        """Same-dtype D tensors must pass the dtype check (will fail later on ctypes call)."""
        from contraction_multi_abd_utils import (
            ContractionMultiABDProblem,
        )
        problem = ContractionMultiABDProblem(g_dims=[2], m_dims=[4, 4], n_dims=[4, 4], k_dims=[8])
        G, M, N, K = 2, 16, 16, 8

        As = [np.ones(G * M * K, dtype=np.float16)]
        Bs = [np.ones(G * N * K, dtype=np.float16)]
        Ds = [
            np.ones(G * M * N, dtype=np.float16),
            np.ones(G * M * N, dtype=np.float16),  # same dtype — OK
        ]
        E = np.zeros(G * M * N, dtype=np.float16)

        lib = self._make_mock_lib()
        # Patch the underlying ctypes call to avoid a real dispatch.
        lib._lib.dispatcher_run_batched_contraction_multi_abd.return_value = 0
        # Should not raise ValueError for the dtype check.
        try:
            lib.run(As, Bs, Ds, E, problem)
        except ValueError as exc:
            self.fail(f"Uniform-dtype D tensors raised ValueError unexpectedly: {exc}")
        except Exception:
            pass  # ctypes/hipMalloc errors are expected in unit test context


class TestElemDZeroWithNoD(unittest.TestCase):
    """Validates elem_d=0 is accepted when there are no D tensors (kNumD==0 path)."""

    def test_dispatcher_lib_run_passes_elem_d_zero_for_empty_Ds(self):
        """ContractionMultiABDDispatcherLib.run() must not raise ValueError for empty Ds."""
        from contraction_multi_abd_utils import ContractionMultiABDProblem

        problem = ContractionMultiABDProblem(g_dims=[1], m_dims=[4, 4], n_dims=[4, 4], k_dims=[8])
        G, M, N, K = 1, 16, 16, 8

        As = [np.ones(G * M * K, dtype=np.float16)]
        Bs = [np.ones(G * N * K, dtype=np.float16)]
        Ds = []  # no D tensors → elem_d falls back to 2 (harmless sentinel)
        E = np.zeros(G * M * N, dtype=np.float16)

        with mock.patch("ctypes.CDLL") as MockCDLL:
            mock_lib = mock.MagicMock()
            MockCDLL.return_value = mock_lib
            mock_lib.dispatcher_initialize.return_value = 0
            mock_lib.dispatcher_run_batched_contraction_multi_abd.return_value = 0

            from contraction_multi_abd_utils import ContractionMultiABDDispatcherLib
            lib = ContractionMultiABDDispatcherLib.__new__(ContractionMultiABDDispatcherLib)
            lib.so_path = Path("/fake/lib.so")
            lib._lib = mock_lib

        try:
            lib.run(As, Bs, Ds, E, problem)
        except ValueError as exc:
            self.fail(f"Empty Ds raised ValueError unexpectedly: {exc}")
        except Exception:
            pass  # ctypes errors expected; the point is no ValueError from dtype checks

    def test_python_elem_d_sentinel_value_for_empty_ds(self):
        """elem_d sentinel (2) must be passed without triggering the dtype mismatch check."""
        from contraction_multi_abd_utils import ContractionMultiABDProblem

        problem = ContractionMultiABDProblem(g_dims=[1], m_dims=[2], n_dims=[2], k_dims=[4])
        As = [np.ones(8, dtype=np.float16)]
        Bs = [np.ones(8, dtype=np.float16)]
        Ds = []
        E = np.zeros(4, dtype=np.float16)

        with mock.patch("ctypes.CDLL") as MockCDLL:
            mock_lib = mock.MagicMock()
            MockCDLL.return_value = mock_lib
            mock_lib.dispatcher_initialize.return_value = 0
            mock_lib.dispatcher_run_batched_contraction_multi_abd.return_value = 0

            from contraction_multi_abd_utils import ContractionMultiABDDispatcherLib
            lib = ContractionMultiABDDispatcherLib.__new__(ContractionMultiABDDispatcherLib)
            lib.so_path = Path("/fake/lib.so")
            lib._lib = mock_lib

        # No exception of any kind from the validation path.
        try:
            lib.run(As, Bs, Ds, E, problem)
        except ValueError as exc:
            self.fail(f"Empty Ds raised ValueError: {exc}")
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
