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
        # dBias is not generated yet, so its UID must be explicitly absent.
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

    def test_gqa_is_rejected(self):
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
            "2",
            "32",
            "16",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GQA is not supported", result.stdout + result.stderr)

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

    def _gradients(self, keep_mask=None):
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
        O = self.gen.compute_forward(Q, K, V, scale, heads, heads, keep_mask)
        lse = self.gen.compute_lse(Q, K, scale, heads, heads, keep_mask)
        dQ, dK, dV = self.gen.compute_backward(Q, K, V, O, lse, dO, scale, keep_mask)
        return Q, K, V, dO, scale, keep_mask, dQ, dK, dV

    def test_unmasked_gradients_match_autograd(self):
        deviation = self.gen.validate_against_autograd(*self._gradients())
        self.assertLess(deviation, 1e-5)

    def test_causal_gradients_match_autograd(self):
        keep_mask = self.gen.build_bound_mask(8, 8, None, 0, True)
        deviation = self.gen.validate_against_autograd(*self._gradients(keep_mask))
        self.assertLess(deviation, 1e-5)


if __name__ == "__main__":
    unittest.main()
