#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = SCRIPTS_DIR / "generate_sdpa_bwd_golden.py"
FWD_SCRIPT_PATH = SCRIPTS_DIR / "generate_sdpa_fwd_golden.py"
VERIFIER_PATH = SCRIPTS_DIR / "verify_golden_bundles.py"
TORCH_MISSING = importlib.util.find_spec("torch") is None

B, H, S_Q, D_QK = 1, 2, 32, 16
S_KV, D_V = 32, 16
BF16_BYTES = 2
FP32_BYTES = 4


def run_generator(backward_base, forward_base, *extra_args):
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--base-filename",
        str(backward_base),
        "--forward-base-filename",
        str(forward_base),
        "--q-dims",
        str(B),
        str(H),
        str(S_Q),
        str(D_QK),
        "--v-dims",
        str(B),
        str(H),
        str(S_KV),
        str(D_V),
        "--attn-scale",
        "0.25",
        *extra_args,
    ]
    return subprocess.run(command, capture_output=True, text=True)


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestGenerateSdpaBwdGoldenBundles(unittest.TestCase):
    """One pipeline run must emit a paired forward bundle and a backward bundle."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.backward_base = root / "quick/SdpaBackward/bhsd/bf16/Small/Small"
        cls.forward_base = root / "quick/SdpaFwd/bhsd/bf16/Small/Small"
        cls.result = run_generator(cls.backward_base, cls.forward_base)
        if cls.result.returncode != 0:
            raise AssertionError(cls.result.stderr)
        cls.root = root
        cls.backward_graph = json.loads(Path(f"{cls.backward_base}.json").read_text())
        cls.backward_meta = json.loads(
            Path(f"{cls.backward_base}.meta.json").read_text()
        )
        cls.forward_meta = json.loads(Path(f"{cls.forward_base}.meta.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_both_bundles_are_written_in_one_run(self):
        for base in (self.forward_base, self.backward_base):
            self.assertTrue(Path(f"{base}.json").is_file(), base)
            self.assertTrue(Path(f"{base}.meta.json").is_file(), base)

    def test_backward_bundle_records_its_forward_source(self):
        # RFC 0011: a backward bundle must point at the forward pass whose O and
        # LSE it consumes, so a reviewer can retrace the single pipeline run.
        forward_source = self.backward_meta["forward_source"]
        self.assertEqual(
            forward_source["method"], self.forward_meta["reference_backend"]
        )
        self.assertIn(forward_source["version"], self.forward_meta["reference_source"])
        self.assertEqual(
            forward_source["generated_at"], self.forward_meta["generated_at"]
        )
        bundle = Path(f"{self.backward_base}.json").parent / forward_source["bundle"]
        self.assertTrue(bundle.resolve().is_file(), forward_source["bundle"])

    def test_reference_strategy_is_matched_precision(self):
        self.assertEqual(self.backward_meta["reference_strategy"], "matched_precision")

    def test_backward_graph_declares_every_gradient_output(self):
        node = self.backward_graph["nodes"][0]
        self.assertEqual(node["type"], "SdpaBackwardAttributes")
        outputs = node["outputs"]
        self.assertEqual(outputs["dq_tensor_uid"], 4)
        self.assertEqual(outputs["dk_tensor_uid"], 3)
        self.assertEqual(outputs["dv_tensor_uid"], 5)
        # This configuration has no --dbias, so the dBias UID must be explicitly absent.
        self.assertIsNone(outputs["dbias_tensor_uid"])

    def test_backward_stats_tensor_is_rank_four_fp32(self):
        # The CPU and GPU references both require [B, H_q, S_q, 1] FP32 stats.
        stats = next(t for t in self.backward_graph["tensors"] if t["name"] == "stats")
        self.assertEqual(stats["dims"], [B, H, S_Q, 1])
        self.assertEqual(stats["data_type"], "float")

    def test_tensor_payloads_have_the_declared_sizes(self):
        expected = {
            0: B * H * S_KV * D_QK * BF16_BYTES,  # K
            1: B * H * S_Q * D_V * BF16_BYTES,  # O
            2: B * H * S_Q * D_QK * BF16_BYTES,  # Q
            3: B * H * S_KV * D_QK * BF16_BYTES,  # dK
            4: B * H * S_Q * D_QK * BF16_BYTES,  # dQ
            5: B * H * S_KV * D_V * BF16_BYTES,  # dV
            6: B * H * S_KV * D_V * BF16_BYTES,  # V
            7: B * H * S_Q * D_V * BF16_BYTES,  # dO
            8: B * H * S_Q * 1 * FP32_BYTES,  # stats
        }
        for uid, size in expected.items():
            path = Path(f"{self.backward_base}.tensor{uid}.bin")
            self.assertEqual(path.stat().st_size, size, f"uid {uid}")

    def test_verifier_accepts_both_bundles(self):
        result = subprocess.run(
            [sys.executable, str(VERIFIER_PATH), "--require-data", str(self.root)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("0 error(s), 0 warning(s)", result.stderr)


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestGenerateSdpaBwdGoldenReproducibility(unittest.TestCase):
    def test_same_seed_reproduces_the_same_tensors(self):
        payloads = []
        for _ in range(2):
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                backward_base = root / "bwd/bwd"
                run_generator(backward_base, root / "fwd/fwd", "--seed", "7")
                payloads.append(
                    {
                        path.name: path.read_bytes()
                        for path in sorted(root.rglob("*.tensor*.bin"))
                    }
                )
        self.assertEqual(payloads[0], payloads[1])


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestGenerateSdpaBwdGoldenForwardAgreement(unittest.TestCase):
    """The paired forward bundle must match the forward generator exactly.

    Both generators share utilities/sdpa_common.py and the forward graph
    builder, so any drift between them is a bug rather than a tolerance
    question.
    """

    def test_forward_bundle_matches_the_forward_generator(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired = root / "paired/fwd"
            standalone = root / "standalone/fwd"
            run_generator(root / "paired_bwd/bwd", paired, "--seed", "11")
            subprocess.run(
                [
                    sys.executable,
                    str(FWD_SCRIPT_PATH),
                    "--base-filename",
                    str(standalone),
                    "--q-dims",
                    str(B),
                    str(H),
                    str(S_Q),
                    str(D_QK),
                    "--v-dims",
                    str(B),
                    str(H),
                    str(S_KV),
                    str(D_V),
                    "--attn-scale",
                    "0.25",
                    "--stats",
                    "--seed",
                    "11",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertEqual(
                json.loads(Path(f"{paired}.json").read_text()),
                json.loads(Path(f"{standalone}.json").read_text()),
            )
            for uid in range(5):
                self.assertEqual(
                    Path(f"{paired}.tensor{uid}.bin").read_bytes(),
                    Path(f"{standalone}.tensor{uid}.bin").read_bytes(),
                    f"uid {uid}",
                )


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestGenerateSdpaBwdGoldenRejectedConfigurations(unittest.TestCase):
    """Unsupported features must fail loudly rather than emit wrong goldens."""

    def generate(self, *extra_args):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            return run_generator(root / "bwd/bwd", root / "fwd/fwd", *extra_args)

    def test_heads_not_a_multiple_of_kv_heads_are_rejected(self):
        command = [
            sys.executable,
            str(SCRIPT_PATH),
            "--base-filename",
            "/dev/null/bwd",
            "--forward-base-filename",
            "/dev/null/fwd",
            "--q-dims",
            "1",
            "4",
            "32",
            "16",
            "--v-dims",
            "1",
            "3",
            "32",
            "16",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be a multiple of H_kv", result.stdout + result.stderr)

    def test_ragged_only_options_without_ragged_sides_are_rejected(self):
        for flag, value in (
            ("--ragged-blocks", "q=2,2"),
            ("--ragged-origins", "q=0,2"),
            ("--ragged-head-major", "q"),
            ("--ragged-head-pitch", "q=8"),
            ("--ragged-token-stride", "q=16"),
            ("--ragged-units", "tokens"),
        ):
            with self.subTest(flag=flag):
                result = self.generate(flag, value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"{flag} requires --ragged-sides", result.stderr)

    def test_fully_masked_query_row_is_rejected(self):
        # A fully masked row makes LSE -inf, so the softmax and every gradient
        # that depends on it would be NaN.
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--base-filename",
                "/dev/null/bwd",
                "--forward-base-filename",
                "/dev/null/fwd",
                "--q-dims",
                "1",
                "1",
                "64",
                "16",
                "--v-dims",
                "1",
                "1",
                "32",
                "16",
                "--right-bound",
                "0",
                "--diagonal-alignment",
                "bottom_right",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fully masked", result.stdout + result.stderr)


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestGenerateSdpaBwdGoldenMath(unittest.TestCase):
    """The reference gradients must agree with PyTorch autograd."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "generate_sdpa_bwd_golden", SCRIPT_PATH
        )
        cls.gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.gen)
        cls.torch = cls.gen.torch

    def _gradients(self, keep_mask=None, attn_bias=None):
        torch = self.torch
        rng = torch.Generator().manual_seed(3)
        shape = (1, 2, 8, 4)
        tensors = [
            torch.empty(shape, dtype=torch.float32).uniform_(-1, 1, generator=rng)
            for _ in range(4)
        ]
        Q, K, V, dO = tensors
        scale = 0.5
        heads = shape[1]
        O = self.gen.compute_forward(Q, K, V, scale, heads, heads, keep_mask, attn_bias)
        lse = self.gen.compute_lse(Q, K, scale, heads, heads, keep_mask, attn_bias)
        dQ, dK, dV, _ = self.gen.compute_backward(
            Q, K, V, O, lse, dO, scale, keep_mask, attn_bias
        )
        return Q, K, V, dO, scale, keep_mask, dQ, dK, dV, attn_bias

    def test_unmasked_gradients_match_autograd(self):
        deviation = self.gen.validate_against_autograd(*self._gradients())
        self.assertLess(deviation, 1e-5)

    def test_causal_gradients_match_autograd(self):
        keep_mask = self.gen.build_bound_mask(8, 8, None, 0, True)
        deviation = self.gen.validate_against_autograd(*self._gradients(keep_mask))
        self.assertLess(deviation, 1e-5)

    def test_additive_mask_gradients_match_autograd(self):
        torch = self.torch
        bias = torch.empty((1, 1, 8, 8), dtype=torch.float32).uniform_(
            -1, 1, generator=torch.Generator().manual_seed(11)
        )
        deviation = self.gen.validate_against_autograd(*self._gradients(attn_bias=bias))
        self.assertLess(deviation, 1e-5)

    def test_additive_mask_under_a_causal_band_matches_autograd(self):
        torch = self.torch
        keep_mask = self.gen.build_bound_mask(8, 8, None, 0, True)
        bias = torch.empty((1, 1, 8, 8), dtype=torch.float32).uniform_(
            -1, 1, generator=torch.Generator().manual_seed(12)
        )
        deviation = self.gen.validate_against_autograd(
            *self._gradients(keep_mask, bias)
        )
        self.assertLess(deviation, 1e-5)

    def test_a_finite_bias_cannot_resurrect_a_masked_position(self):
        """The band is applied after the bias, so -inf must survive any bias."""
        torch = self.torch
        keep_mask = self.gen.build_bound_mask(8, 8, None, 0, True)
        bias = torch.full((1, 1, 8, 8), 1e4, dtype=torch.float32)
        combined = self.gen.combined_attn_mask(keep_mask, bias)
        self.assertTrue(torch.isinf(combined[~keep_mask.expand_as(combined)]).all())
        self.assertTrue((combined[keep_mask.expand_as(combined)] == 1e4).all())


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestSdpaBwdAdditiveMaskBundles(unittest.TestCase):
    """--additive-mask must add the mask tensor to both halves of the pair."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.backward_base = root / "Bwd/Bwd"
        cls.forward_base = root / "Fwd/Fwd"
        cls.result = run_generator(
            cls.backward_base, cls.forward_base, "--additive-mask"
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _graph(self, base):
        return json.loads(Path(f"{base}.json").read_text())

    def test_generation_succeeds(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)

    def test_backward_graph_wires_the_mask_tensor(self):
        graph = self._graph(self.backward_base)
        uid = graph["nodes"][0]["inputs"]["attn_mask_tensor_uid"]
        self.assertIsNotNone(uid)
        tensor = next(t for t in graph["tensors"] if t["uid"] == uid)
        self.assertEqual(tensor["dims"], [B, 1, S_Q, S_KV][:1] + [1, S_Q, S_KV])
        self.assertEqual(tensor["data_type"], "bfloat16")

    def test_forward_half_carries_the_same_mask_values(self):
        bwd = self._graph(self.backward_base)
        fwd = self._graph(self.forward_base)
        bwd_uid = bwd["nodes"][0]["inputs"]["attn_mask_tensor_uid"]
        fwd_uid = fwd["nodes"][0]["inputs"]["attn_mask_tensor_uid"]
        self.assertIsNotNone(fwd_uid)
        # The forward generator owns a different UID map, so the UIDs differ;
        # the bytes behind them must not.
        self.assertEqual(
            Path(f"{self.backward_base}.tensor{bwd_uid}.bin").read_bytes(),
            Path(f"{self.forward_base}.tensor{fwd_uid}.bin").read_bytes(),
        )

    def test_mask_is_absent_without_the_flag(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_generator(root / "B/B", root / "F/F")
            self.assertEqual(result.returncode, 0, result.stderr)
            graph = json.loads((root / "B/B.json").read_text())
            self.assertIsNone(graph["nodes"][0]["inputs"]["attn_mask_tensor_uid"])


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestSdpaBwdSweepMode(unittest.TestCase):
    """Sweep mode must attach golden data to a checked-in template sweep."""

    CASE_ID = "golden_1_2_32_16_bf16_nomask"

    @classmethod
    def setUpClass(cls):
        cls._tmp = TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.sweep_dir = root / "Default"
        cls.sweep_dir.mkdir(parents=True)
        # A pre-existing graph-only case that regeneration must not disturb.
        cls.existing = {"id": "captured_case", "values": {"tensors": []}}
        (cls.sweep_dir / "sweep.json").write_text(
            json.dumps({"version": 1, "cases": [cls.existing]})
        )
        cls.result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--sweep-bundle-dir",
                str(cls.sweep_dir),
                "--case-id",
                cls.CASE_ID,
                "--forward-base-filename",
                str(root / "Fwd/Fwd"),
                "--q-dims",
                str(B),
                str(H),
                str(S_Q),
                str(D_QK),
                "--v-dims",
                str(B),
                str(H),
                str(S_KV),
                str(D_V),
                "--attn-scale",
                "0.25",
            ],
            capture_output=True,
            text=True,
        )
        cls.sweep = json.loads((cls.sweep_dir / "sweep.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _case(self):
        return next(c for c in self.sweep["cases"] if c["id"] == self.CASE_ID)

    def test_generation_succeeds(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)

    def test_tensors_land_in_the_case_golden_directory(self):
        golden = self.sweep_dir / "golden" / self.CASE_ID
        written = sorted(p.name for p in golden.glob("*.bin"))
        self.assertEqual(written, sorted(f"tensor{uid}.bin" for uid in range(9)))

    def test_case_points_at_its_tensors_dvc(self):
        self.assertEqual(
            self._case()["golden"]["path"], f"golden/{self.CASE_ID}/tensors.dvc"
        )

    def test_case_values_match_the_tensors_written(self):
        tensors = {t["uid"]: t for t in self._case()["values"]["tensors"]}
        self.assertEqual(tensors[8]["dims"], [B, H, S_Q, 1])
        self.assertEqual(tensors[8]["data_type"], "float")
        self.assertEqual(tensors[0]["dims"], [B, H, S_KV, D_QK])
        self.assertEqual(self._case()["values"]["io_data_type"], "bfloat16")

    def test_case_metadata_records_rfc_0011_provenance(self):
        metadata = self._case()["metadata"]
        self.assertEqual(metadata["reference_strategy"], "matched_precision")
        self.assertEqual(metadata["direction"], "backward")
        self.assertIn("forward_source", metadata)

    def test_existing_cases_are_left_alone(self):
        self.assertEqual(self.sweep["cases"][0], self.existing)
        self.assertEqual(len(self.sweep["cases"]), 2)

    def test_case_id_must_be_snake_case(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--sweep-bundle-dir",
                    str(root),
                    "--case-id",
                    "NotSnakeCase",
                    "--forward-base-filename",
                    str(root / "Fwd/Fwd"),
                    "--q-dims",
                    str(B),
                    str(H),
                    str(S_Q),
                    str(D_QK),
                    "--v-dims",
                    str(B),
                    str(H),
                    str(S_KV),
                    str(D_V),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("lowercase_snake_case", result.stderr)

    def test_sweep_bundle_dir_and_case_id_must_be_given_together(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--sweep-bundle-dir",
                    str(root),
                    "--forward-base-filename",
                    str(root / "Fwd/Fwd"),
                    "--q-dims",
                    str(B),
                    str(H),
                    str(S_Q),
                    str(D_QK),
                    "--v-dims",
                    str(B),
                    str(H),
                    str(S_KV),
                    str(D_V),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be given together", result.stderr)


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestSdpaBwdExtendedMath(unittest.TestCase):
    """GQA, padding, dBias and dropout: the explicit backward equals FP64 autograd
    of the forward definition on FP32 inputs (no IO rounding), so the bound is tight."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "generate_sdpa_bwd_golden", SCRIPT_PATH
        )
        cls.gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.gen)
        cls.torch = cls.gen.torch

    def _check(
        self,
        h_q=2,
        h_kv=2,
        s_q=8,
        s_kv=12,
        keep_mask=None,
        bias_dims=None,
        dropout_p=None,
        dropout_scale_inv=None,
    ):
        torch, gen = self.torch, self.gen
        rng = torch.Generator().manual_seed(5)
        draw = lambda *shape: torch.empty(shape).uniform_(-1, 1, generator=rng)
        Q, K, V = draw(2, h_q, s_q, 4), draw(2, h_kv, s_kv, 4), draw(2, h_kv, s_kv, 4)
        dO = draw(2, h_q, s_q, 4)
        bias = None if bias_dims is None else draw(*bias_dims)
        mask, scale_d, scale_inv = None, 1.0, 1.0
        if dropout_p is not None:
            mask = (
                torch.empty(2, h_q, s_q, s_kv).uniform_(0, 1, generator=rng)
                >= dropout_p
            ).float()
            scale_d = 1.0 / (1.0 - dropout_p)
            scale_inv = (
                1.0 - dropout_p if dropout_scale_inv is None else dropout_scale_inv
            )
        O, lse = gen.compute_forward_explicit(
            Q, K, V, 0.5, keep_mask, bias, mask, scale_d
        )
        dQ, dK, dV, dBias = gen.compute_backward(
            Q,
            K,
            V,
            O,
            lse,
            dO,
            0.5,
            keep_mask,
            bias,
            mask,
            scale_d,
            scale_inv,
            bias_dims,
        )
        aQ, aK, aV, aBias = gen.autograd_fp64_gradients(
            Q, K, V, dO, 0.5, keep_mask, bias, mask, scale_d
        )
        pairs = [("dQ", dQ, aQ), ("dK", dK, aK), ("dV", dV, aV)]
        if bias_dims is not None:
            pairs.append(("dBias", dBias, aBias))
        return gen.worst_relative_deviation(pairs), dQ, dK, dV, O, lse

    def test_gqa_sums_each_query_group(self):
        self.assertLess(self._check(h_q=4, h_kv=2)[0], 1e-5)
        self.assertLess(self._check(h_q=4, h_kv=1)[0], 1e-5)

    def test_broadcast_dbias_is_the_reduced_unscaled_ds(self):
        for dims in ([1, 1, 8, 12], [1, 2, 8, 12], [2, 1, 8, 12], [2, 2, 8, 12]):
            self.assertLess(self._check(bias_dims=dims)[0], 1e-5, dims)

    def test_dropout_matches_autograd_when_scale_inv_is_one_minus_p(self):
        self.assertLess(self._check(dropout_p=0.3)[0], 1e-5)

    def test_wrong_dropout_scale_inv_is_caught(self):
        # cuDNN's composition is only the exact gradient when scale_inv = 1/scale;
        # the generator's autograd gate must notice anything else.
        self.assertGreater(self._check(dropout_p=0.3, dropout_scale_inv=1.0)[0], 1e-2)

    def test_padding_with_dead_rows(self):
        keep = self.gen.build_padded_keep_mask(8, 12, [3, 8], [12, 5], None, 0, False)
        worst, dQ, dK, dV, O, lse = self._check(keep_mask=keep)
        self.assertLess(worst, 1e-5)
        # Batch 0: rows past seq_len_q are dead. Batch 1: bottom-right with
        # Sq = 8 > Skv = 5 leaves the first 3 rows with no key.
        for b, rows in ((0, range(3, 8)), (1, range(0, 3))):
            for r in rows:
                self.assertTrue(bool((O[b, :, r] == 0).all()))
                self.assertTrue(bool(self.torch.isneginf(lse[b, :, r]).all()))
                self.assertTrue(bool((dQ[b, :, r] == 0).all()))
        # Keys past batch 1's seq_len_kv get no gradient.
        self.assertTrue(bool((dK[1, :, 5:] == 0).all()))
        self.assertTrue(bool((dV[1, :, 5:] == 0).all()))

    def test_padded_keep_mask_uses_live_lengths_for_bottom_right(self):
        keep = self.gen.build_padded_keep_mask(4, 6, [2], [5], None, 0, False)[0, 0]
        # Live 2 x 5, offset 3: row 0 sees keys 0..3, row 1 keys 0..4; nothing else.
        self.assertEqual([j for j in range(6) if keep[0, j]], [0, 1, 2, 3])
        self.assertEqual([j for j in range(6) if keep[1, j]], [0, 1, 2, 3, 4])
        self.assertFalse(bool(keep[2:].any()))


# ctest reports exit code 77 as Skipped (SKIP_RETURN_CODE), so a run in which every
# test skipped, e.g. without PyTorch, does not look like a pass.
SKIP_EXIT = 77


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestSdpaBwdRaggedBundles(unittest.TestCase):
    """A ragged sweep case must store what the harness reads back: sequences at their
    offsets, NaN in output gaps, finite poison in input gaps, no aliasing."""

    RB, RH, RSQ, RSKV, RD = 3, 2, 8, 12, 16
    LENS_Q, LENS_KV = [3, 0, 8], [12, 0, 5]
    BLOCKS_Q, BLOCKS_KV = [5, 2, 8], [12, 3, 5]

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "generate_sdpa_bwd_golden", SCRIPT_PATH
        )
        cls.gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.gen)
        cls._tmp = TemporaryDirectory()
        cls.root = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _generate(self, name, kind, *extra):
        sweep_dir = self.root / name
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--write-template",
                str(sweep_dir),
                "--template-kind",
                kind,
                "--attn-scale",
                "0.25",
            ],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--sweep-bundle-dir",
                str(sweep_dir),
                "--case-id",
                "case_fp16_x",
                "--forward-base-filename",
                str(self.root / f"{name}_fwd/Small"),
                "--q-dims",
                str(self.RB),
                str(self.RH),
                str(self.RSQ),
                str(self.RD),
                "--v-dims",
                str(self.RB),
                str(self.RH),
                str(self.RSKV),
                str(self.RD),
                "--dtype",
                "fp16",
                "--attn-scale",
                "0.25",
                "--seq-lens-q",
                *map(str, self.LENS_Q),
                "--seq-lens-kv",
                *map(str, self.LENS_KV),
                *extra,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return sweep_dir

    def _tensor(self, sweep_dir, uid, dtype):
        raw = (sweep_dir / f"golden/case_fp16_x/tensor{uid}.bin").read_bytes()
        torch = self.gen.torch
        if dtype == "half":
            return torch.frombuffer(bytearray(raw), dtype=torch.float16)
        if dtype == "int32":
            return torch.frombuffer(bytearray(raw), dtype=torch.int32)
        return torch.frombuffer(bytearray(raw), dtype=torch.float32)

    def test_gaps_are_nan_in_outputs_and_finite_in_inputs(self):
        sweep_dir = self._generate(
            "gap",
            "ragged_qkv",
            "--ragged-sides",
            "qkv",
            "--ragged-blocks",
            "q=5,2,8",
            "--ragged-blocks",
            "dq=5,2,8",
            "--ragged-blocks",
            "k=12,3,5",
            "--ragged-blocks",
            "dk=12,3,5",
        )
        row = self.RH * self.RD
        gap_q = (sum(self.BLOCKS_Q) - sum(self.LENS_Q)) * row
        gap_kv = (sum(self.BLOCKS_KV) - sum(self.LENS_KV)) * row
        torch = self.gen.torch
        # dQ (4) and dK (3) are outputs; Q (2) and K (0) are inputs.
        self.assertEqual(
            int(torch.isnan(self._tensor(sweep_dir, 4, "half")).sum()), gap_q
        )
        self.assertEqual(
            int(torch.isnan(self._tensor(sweep_dir, 3, "half")).sum()), gap_kv
        )
        for uid in (0, 2):
            self.assertTrue(
                bool(torch.isfinite(self._tensor(sweep_dir, uid, "half")).all())
            )
        # O and dV were laid out compact: nothing to be NaN.
        self.assertEqual(int(torch.isnan(self._tensor(sweep_dir, 1, "half")).sum()), 0)

    def test_verifier_accepts_the_case(self):
        sweep_dir = self._generate(
            "verify",
            "ragged_qkv",
            "--ragged-sides",
            "qkv",
            "--ragged-blocks",
            "q=5,2,8",
        )
        result = subprocess.run(
            [sys.executable, str(VERIFIER_PATH), str(sweep_dir)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_token_offsets_carry_the_sequence_stride_as_multiplier(self):
        sweep_dir = self._generate(
            "tokens",
            "ragged_qkv",
            "--ragged-sides",
            "qkv",
            "--ragged-units",
            "tokens",
            "--ragged-blocks",
            "q=5,2,8",
        )
        offsets = self._tensor(sweep_dir, 16, "int32").tolist()
        self.assertEqual(offsets, [0, 5, 7, 15])
        case = json.loads((sweep_dir / "sweep.json").read_text())["cases"][0]
        multipliers = {
            t["uid"]: t["ragged_offset_multiplier"]
            for t in case["values"]["tensors"]
            if "ragged_offset_multiplier" in t
        }
        self.assertEqual(multipliers[2], self.RH * self.RD)  # Q
        self.assertEqual(multipliers[8], self.RH)  # Stats

    def test_q_only_layout_keeps_kv_dense_with_zero_padded_gradients(self):
        sweep_dir = self._generate(
            "qonly", "ragged_q", "--ragged-sides", "q", "--ragged-blocks", "q=5,2,8"
        )
        torch = self.gen.torch
        dk = self._tensor(sweep_dir, 3, "half").view(
            self.RB, self.RH, self.RSKV, self.RD
        )
        for b, length in enumerate(self.LENS_KV):
            self.assertEqual(float(dk[b, :, length:, :].abs().sum()), 0.0)
        self.assertTrue(bool(torch.isfinite(dk).all()))

    def test_blocks_shorter_than_the_sequence_are_rejected(self):
        with self.assertRaises(ValueError):
            self.gen.plan_ragged(
                [2, 1, 4, 8],
                [2, 1, 4, 8],
                [4, 4],
                [4, 4],
                "q",
                {"q": [3, 4]},
                "elements",
                "th",
            )


RAGGED_SIDES_OF_KIND = {"ragged_q": "q", "ragged_kv": "kv", "ragged_qkv": "qkv"}
UID = {"k": 0, "o": 1, "q": 2, "dk": 3, "dq": 4, "dv": 5, "v": 6, "do": 7, "stats": 8}
OFFSET_UID = {
    "q": 16,
    "k": 17,
    "v": 18,
    "o": 19,
    "do": 20,
    "dq": 21,
    "dk": 22,
    "dv": 23,
    "stats": 24,
}


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestSdpaBwdRaggedLayouts(unittest.TestCase):
    """Each ragged layout is written as a bundle whose files hold exactly the
    elements its sequences address (elementSpace, RFC 0014), and the verifier agrees."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "generate_sdpa_bwd_golden", SCRIPT_PATH
        )
        cls.gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.gen)
        cls._tmp = TemporaryDirectory()
        cls.root = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _run(self, name, kind, q_dims, v_dims, lens_q, lens_kv, *extra):
        sweep_dir = self.root / name
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--write-template",
                str(sweep_dir),
                "--template-kind",
                kind,
                "--attn-scale",
                "0.25",
            ],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--sweep-bundle-dir",
                str(sweep_dir),
                "--case-id",
                "case_fp16_x",
                "--forward-base-filename",
                str(self.root / f"{name}_fwd/Small"),
                "--q-dims",
                *map(str, q_dims),
                "--v-dims",
                *map(str, v_dims),
                "--dtype",
                "fp16",
                "--attn-scale",
                "0.25",
                "--seq-lens-q",
                *map(str, lens_q),
                "--seq-lens-kv",
                *map(str, lens_kv),
                "--ragged-sides",
                RAGGED_SIDES_OF_KIND[kind],
                *extra,
            ],
            capture_output=True,
            text=True,
        )
        return sweep_dir, result

    def _generate(self, name, kind, q_dims, v_dims, lens_q, lens_kv, *extra):
        sweep_dir, result = self._run(
            name, kind, q_dims, v_dims, lens_q, lens_kv, *extra
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return sweep_dir

    def _tensor(self, sweep_dir, uid):
        torch = self.gen.torch
        raw = (sweep_dir / f"golden/case_fp16_x/tensor{uid}.bin").read_bytes()
        if uid in OFFSET_UID.values():
            dtype = torch.int32
        elif uid == UID["stats"]:
            dtype = torch.float32
        else:
            dtype = torch.float16
        return torch.frombuffer(bytearray(raw), dtype=dtype)

    def _case_tensor(self, sweep_dir, uid):
        case = json.loads((sweep_dir / "sweep.json").read_text())["cases"][0]
        return {t["uid"]: t for t in case["values"]["tensors"]}[uid]

    def _verify(self, sweep_dir):
        return subprocess.run(
            [sys.executable, str(VERIFIER_PATH), str(sweep_dir)],
            capture_output=True,
            text=True,
        )

    def _check_row(self, name, kind, dims, lens, role, space, offsets, strides, *extra):
        """The bundle's `role` tensor holds `space` elements, its offset tensor is
        `offsets` and its H, S, D strides are `strides`; the verifier accepts it."""
        sweep_dir = self._generate(name, kind, dims, dims, lens, lens, *extra)
        self.assertEqual(self._tensor(sweep_dir, UID[role]).numel(), space)
        self.assertEqual(self._tensor(sweep_dir, OFFSET_UID[role]).tolist(), offsets)
        case = self._case_tensor(sweep_dir, UID[role])
        self.assertEqual(case["strides"][1:], strides)
        verified = self._verify(sweep_dir)
        self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
        return sweep_dir

    def test_l2_fused_token_stride_spans_the_padded_tokens(self):
        self._check_row(
            "l2",
            "ragged_kv",
            [2, 4, 3, 8],
            [2, 3],
            "v",
            416,
            [0, 192, 480],
            [8, 96, 1],
            "--ragged-token-stride",
            "v=96",
        )

    def test_l5s_block_longer_than_s_max_with_seq_len(self):
        self._check_row(
            "l5s",
            "ragged_q",
            [2, 4, 5, 8],
            [3, 5],
            "q",
            416,
            [0, 256, 512],
            [8, 32, 1],
            "--ragged-blocks",
            "q=8,8",
        )

    def test_l6_head_major_primary(self):
        self._check_row(
            "l6",
            "ragged_q",
            [2, 2, 5, 2],
            [2, 3],
            "q",
            20,
            [0, 4, 10],
            [10, 2, 1],
            "--ragged-head-major",
            "q",
        )

    def test_l7a_head_major_stats_with_several_heads(self):
        self._check_row(
            "l7a",
            "ragged_q",
            [2, 4, 3, 2],
            [2, 3],
            "stats",
            20,
            [0, 2, 5],
            [5, 1, 1],
            "--stats-layout",
            "ht",
        )

    def test_l7b_token_major_stats(self):
        self._check_row(
            "l7b",
            "ragged_q",
            [2, 4, 3, 2],
            [2, 3],
            "stats",
            20,
            [0, 8, 20],
            [1, 4, 1],
        )

    def test_l7c_head_pitch_beyond_the_tokens(self):
        self._check_row(
            "l7c",
            "ragged_q",
            [2, 4, 3, 2],
            [2, 3],
            "stats",
            35,
            [0, 2, 5],
            [10, 1, 1],
            "--stats-layout",
            "ht",
            "--ragged-head-pitch",
            "stats=10",
        )

    def test_l7d_gapped_head_major_stats_holds_finite_poison_in_the_gaps(self):
        sweep_dir = self._check_row(
            "l7d",
            "ragged_q",
            [2, 2, 3, 2],
            [2, 3],
            "stats",
            14,
            [0, 4, 7],
            [7, 1, 1],
            "--stats-layout",
            "ht",
            "--ragged-origins",
            "stats=0,4",
        )
        torch = self.gen.torch
        stats = self._tensor(sweep_dir, UID["stats"])
        gaps = [h * 7 + row for h in range(2) for row in (2, 3)]
        self.assertTrue(bool(torch.isfinite(stats[gaps]).all()))

    def test_l8_shifted_origin(self):
        self._check_row(
            "l8",
            "ragged_q",
            [2, 1, 4, 2],
            [2, 3],
            "q",
            14,
            [4, 8, 14],
            [2, 2, 1],
            "--ragged-origins",
            "q=4,8",
        )

    def test_l9_non_monotonic_origins(self):
        self._check_row(
            "l9",
            "ragged_q",
            [2, 1, 4, 2],
            [2, 3],
            "q",
            14,
            [10, 0, 14],
            [2, 2, 1],
            "--ragged-origins",
            "q=10,0",
        )

    def test_l9m_token_origins_carry_the_multiplier(self):
        sweep_dir = self._check_row(
            "l9m",
            "ragged_q",
            [2, 1, 4, 2],
            [2, 3],
            "q",
            14,
            [5, 0, 7],
            [2, 2, 1],
            "--ragged-units",
            "tokens",
            "--ragged-origins",
            "q=5,0",
        )
        self.assertEqual(
            self._case_tensor(sweep_dir, UID["q"])["ragged_offset_multiplier"], 2
        )

    def test_unwritten_elements_are_nan_in_outputs_and_finite_in_inputs(self):
        sweep_dir = self._generate(
            "unwritten",
            "ragged_q",
            [2, 1, 4, 2],
            [2, 1, 4, 2],
            [2, 3],
            [2, 3],
            "--ragged-origins",
            "q=10,0",
            "--ragged-origins",
            "dq=10,0",
        )
        torch = self.gen.torch
        # Batch 1 fills elements 0..5, batch 0 elements 10..13; 6..9 are unwritten.
        self.assertTrue(
            bool(torch.isfinite(self._tensor(sweep_dir, UID["q"])[6:10]).all())
        )
        dq = self._tensor(sweep_dir, UID["dq"])
        self.assertTrue(bool(torch.isnan(dq[6:10]).all()))
        live = list(range(6)) + list(range(10, 14))
        self.assertTrue(bool(torch.isfinite(dq[live]).all()))

    def test_head_major_stats_without_ragged_sides_is_rejected(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--base-filename",
                str(self.root / "dense_ht" / "Small"),
                "--forward-base-filename",
                str(self.root / "dense_ht_fwd" / "Small"),
                "--q-dims",
                "2",
                "2",
                "8",
                "8",
                "--v-dims",
                "2",
                "2",
                "8",
                "8",
                "--stats-layout",
                "ht",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--stats-layout ht needs Stats", result.stderr)

    def test_l7a_and_l7b_hold_the_same_lse_at_every_logical_coordinate(self):
        torch = self.gen.torch
        dims, lens = [2, 4, 3, 2], [2, 3]
        head_major = self._generate(
            "lse_ht", "ragged_q", dims, dims, lens, lens, "--stats-layout", "ht"
        )
        token_major = self._generate("lse_th", "ragged_q", dims, dims, lens, lens)
        ht = self._tensor(head_major, UID["stats"])
        th = self._tensor(token_major, UID["stats"])
        starts = [0, 2]
        for b, length in enumerate(lens):
            for h in range(4):
                for s in range(length):
                    row = starts[b] + s
                    self.assertEqual(float(ht[h * 5 + row]), float(th[row * 4 + h]))
        self.assertTrue(bool(torch.isfinite(ht).all()))

    def test_trailing_gap_is_not_written(self):
        # K's blocks are 12, 3 and 8 rows (23) but the last live row is row 19.
        sweep_dir = self._generate(
            "trailing",
            "ragged_qkv",
            [3, 2, 8, 16],
            [3, 2, 12, 16],
            [3, 0, 8],
            [12, 0, 5],
            "--ragged-blocks",
            "k=12,3,8",
        )
        row = 2 * 16
        self.assertEqual(self._tensor(sweep_dir, UID["k"]).numel(), 20 * row)
        self.assertEqual(
            self._tensor(sweep_dir, OFFSET_UID["k"]).tolist(),
            [0, 12 * row, 15 * row, 23 * row],
        )
        verified = self._verify(sweep_dir)
        self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)

    def test_l14_file_sized_to_the_last_offset_is_rejected_by_the_verifier(self):
        sweep_dir = self._check_row(
            "l14",
            "ragged_q",
            [2, 1, 4, 2],
            [2, 3],
            "q",
            10,
            [0, 4, 12],
            [2, 2, 1],
            "--ragged-blocks",
            "q=2,4",
        )
        path = sweep_dir / f"golden/case_fp16_x/tensor{UID['q']}.bin"
        path.write_bytes(path.read_bytes() + bytes(4))  # 12 elements = offset[B]
        verified = self._verify(sweep_dir)
        self.assertEqual(verified.returncode, 1)
        self.assertIn("file has 24 bytes but graph expects 20 bytes", verified.stderr)

    def test_kv_side_makes_only_k_and_v_roles_ragged(self):
        sweep_dir = self._generate(
            "kv",
            "ragged_kv",
            [3, 2, 8, 16],
            [3, 2, 12, 16],
            [3, 0, 8],
            [12, 0, 5],
            "--ragged-blocks",
            "k=12,3,8",
        )
        case = json.loads((sweep_dir / "sweep.json").read_text())["cases"][0]
        ragged = {
            t["uid"]
            for t in case["values"]["tensors"]
            if "ragged_offset_multiplier" in t
        }
        self.assertEqual(ragged, {UID[r] for r in ("k", "v", "dk", "dv")})
        verified = self._verify(sweep_dir)
        self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)

    def test_stats_that_differ_from_the_lse_fail_the_storage_check(self):
        original = self.gen.pack_ragged_role

        def corrupt_stats(dense, seq_lens, spec, pad):
            packed = original(dense, seq_lens, spec, pad)
            return packed + 1 if dense.shape[-1] == 1 else packed

        self.gen.pack_ragged_role = corrupt_stats
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "differs from the LSE"):
                    self.gen.generate_bundles(
                        base_filename=str(self.root / "lse_bad/bwd"),
                        forward_base_filename=str(self.root / "lse_bad/fwd"),
                        q_dims=[2, 2, 3, 2],
                        v_dims=[2, 2, 3, 2],
                        dtype="fp16",
                        attn_scale=0.25,
                        seq_lens_q=[2, 3],
                        seq_lens_kv=[2, 3],
                        ragged_sides="q",
                    )
        finally:
            self.gen.pack_ragged_role = original

    def _plan(self, sides="q", lens=(2, 3), dims=(2, 1, 4, 2), **options):
        return self.gen.plan_ragged(
            list(dims),
            list(dims),
            list(lens),
            list(lens),
            sides,
            options.pop("blocks", {}),
            options.pop("units", "elements"),
            options.pop("stats_layout", "th"),
            **options,
        )

    def test_head_major_stats_with_several_heads_is_accepted(self):
        plan = self.gen.plan_ragged(
            [2, 2, 4, 8], [2, 2, 4, 8], [4, 4], [4, 4], "qkv", {}, "elements", "ht"
        )
        self.assertEqual(plan["stats"]["strides"][1:], [8, 1, 1])
        self.assertEqual(plan["stats"]["space"], 16)

    def test_overlapping_sequences_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap rule"):
            self._plan(lens=(3, 3), origins={"q": [0, 4]})

    def test_origin_that_is_not_a_whole_token_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "whole-token origin rule"):
            self._plan(origins={"q": [1, 4]})

    def test_aliasing_strides_are_rejected(self):
        # BHSD strides of [2, 2, 4, 2] with origins 0 and 2 (rows 0..4): head stride
        # 8 is below the 10 elements the sequence axis spans (sorted-stride rule).
        with self.assertRaisesRegex(ValueError, "sorted-stride rule"):
            self.gen.ragged_footprint([2, 2, 4, 2], [0, 8, 2, 1], [0, 2], [2, 3])
        with self.assertRaisesRegex(ValueError, "negative-stride rule"):
            self.gen.ragged_footprint([2, 2, 4, 2], [0, 8, -2, 1], [0, 2], [2, 3])

    def test_head_pitch_below_the_last_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "head pitch"):
            self._plan(dims=(2, 2, 5, 2), head_major=["q"], head_pitch={"q": 4})

    def test_token_stride_below_one_token_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "token stride"):
            self._plan(dims=(2, 2, 5, 2), token_stride={"q": 3})

    def test_head_pitch_and_token_stride_need_the_matching_layout(self):
        with self.assertRaisesRegex(ValueError, "not head-major"):
            self._plan(head_pitch={"q": 9})
        with self.assertRaisesRegex(ValueError, "head-major"):
            self._plan(head_major=["q"], token_stride={"q": 4})

    def test_head_major_stats_needs_ragged_stats(self):
        with self.assertRaisesRegex(ValueError, "stats-layout ht needs Stats"):
            self._plan(sides="kv", stats_layout="ht")
        self.assertIn("stats", self._plan(sides="q", stats_layout="ht"))
        self.assertIn("stats", self._plan(sides="qkv", stats_layout="ht"))

    def test_options_name_only_ragged_roles(self):
        for option in ("origins", "head_major", "head_pitch", "token_stride"):
            value = ["k"] if option == "head_major" else {"k": [0, 2]}
            with self.assertRaisesRegex(ValueError, "not ragged"):
                self._plan(**{option: value})


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestSdpaBwdGraphOnlyAndTemplateCheck(unittest.TestCase):
    """A sweep case is registered only when the bundle's graph template expands to the
    graph the generator emits, and --graph-only accepts only what it can express."""

    CASE_ID = "case_bf16_x"
    SHAPE = ("--q-dims", "1", "2", "32", "16", "--v-dims", "1", "2", "32", "16")

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.sweep_dir = self.root / "Sweep"

    def tearDown(self):
        self._tmp.cleanup()

    def _template(self, kind="dense", scale="0.25"):
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--write-template",
                str(self.sweep_dir),
                "--template-kind",
                kind,
                "--attn-scale",
                scale,
            ],
            check=True,
            capture_output=True,
        )

    def _graph_only(self, *extra, scale="0.25"):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--sweep-bundle-dir",
                str(self.sweep_dir),
                "--case-id",
                self.CASE_ID,
                *self.SHAPE,
                "--attn-scale",
                scale,
                "--graph-only",
                *extra,
            ],
            capture_output=True,
            text=True,
        )

    def _registered(self):
        path = self.sweep_dir / "sweep.json"
        if not path.is_file():
            return []
        return [c["id"] for c in json.loads(path.read_text())["cases"]]

    def test_a_graph_only_case_is_registered_without_golden_data(self):
        self._template()
        result = self._graph_only()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._registered(), [self.CASE_ID])
        self.assertFalse((self.sweep_dir / "golden").exists())

    def test_a_template_of_another_topology_registers_nothing(self):
        self._template("dense_dbias")
        result = self._graph_only()
        self.assertEqual(result.returncode, 1)
        self.assertIn("does not express this case", result.stderr)
        self.assertEqual(self._registered(), [])

    def test_a_template_written_for_another_scale_registers_nothing(self):
        self._template(scale="0.5")
        result = self._graph_only()
        self.assertEqual(result.returncode, 1)
        self.assertIn("node parameters differ", result.stderr)
        self.assertEqual(self._registered(), [])

    def test_a_template_whose_tensor_differs_registers_nothing(self):
        # Every node section still agrees: only a tensor's own shape is wrong.
        self._template()
        path = self.sweep_dir / "graph.template.json"
        template = json.loads(path.read_text())
        template["tensors"][0]["dims"] = [9, 9, 9, 9]
        path.write_text(json.dumps(template))
        result = self._graph_only()
        self.assertEqual(result.returncode, 1)
        self.assertIn("tensors differ", result.stderr)
        self.assertNotIn("node", result.stderr.split("does not express")[-1])
        self.assertEqual(self._registered(), [])

    def test_a_graph_only_case_needs_a_template(self):
        self.sweep_dir.mkdir()
        result = self._graph_only()
        self.assertEqual(result.returncode, 1)
        self.assertIn("no graph.template.json", result.stderr)

    def test_graph_only_without_a_sweep_bundle_is_rejected(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--base-filename",
                str(self.root / "bwd/bwd"),
                *self.SHAPE,
                "--attn-scale",
                "0.25",
                "--graph-only",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "--graph-only requires --sweep-bundle-dir and --case-id", result.stderr
        )
        self.assertNotIn("Traceback", result.stderr)

    def test_graph_only_is_for_dense_graphs_without_padding_or_dropout(self):
        self._template()
        for extra in (
            ("--dropout-p", "0.1"),
            ("--seq-lens-q", "32", "--seq-lens-kv", "32"),
        ):
            with self.subTest(extra=extra):
                result = self._graph_only(*extra)
                self.assertEqual(result.returncode, 1)
                self.assertIn(
                    "--graph-only is for dense graphs without padding or dropout",
                    result.stderr,
                )
                self.assertEqual(self._registered(), [])

    def test_a_golden_case_is_checked_against_the_template_too(self):
        # The extended golden path (here dBias) must also refuse a template that does
        # not express the graph, and leave the sweep untouched.
        self._template("dense")
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--sweep-bundle-dir",
                str(self.sweep_dir),
                "--case-id",
                self.CASE_ID,
                "--forward-base-filename",
                str(self.root / "fwd/Small"),
                *self.SHAPE,
                "--attn-scale",
                "0.25",
                "--additive-mask",
                "--dbias",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("does not express this case", result.stderr)
        self.assertEqual(self._registered(), [])
        self.assertFalse((self.sweep_dir / "golden").exists())


if __name__ == "__main__":
    result = unittest.main(exit=False).result
    if not result.wasSuccessful():
        sys.exit(1)
    if result.testsRun > 0 and len(result.skipped) == result.testsRun:
        sys.exit(SKIP_EXIT)
