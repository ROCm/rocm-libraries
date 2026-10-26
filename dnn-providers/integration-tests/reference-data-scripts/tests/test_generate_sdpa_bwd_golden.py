#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
