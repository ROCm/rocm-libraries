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

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "generate_sdpa_fwd_golden.py"
VERIFIER_PATH = Path(__file__).resolve().parent.parent / "verify_golden_bundles.py"
TORCH_MISSING = importlib.util.find_spec("torch") is None


def load_generator_module():
    spec = importlib.util.spec_from_file_location(
        "generate_sdpa_fwd_golden", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestGenerateSdpaFwdGoldenScale(unittest.TestCase):
    def generate(self, *extra_args: str) -> tuple[dict, dict]:
        with TemporaryDirectory() as tmp:
            base = Path(tmp) / "bundle"
            command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--base-filename",
                str(base),
                "--q-dims",
                "1",
                "1",
                "4",
                "16",
                "--v-dims",
                "1",
                "1",
                "4",
                "16",
                *extra_args,
            ]
            subprocess.run(command, capture_output=True, text=True, check=True)
            graph = json.loads(base.with_suffix(".json").read_text())
            meta = json.loads(Path(f"{base}.meta.json").read_text())
            return graph, meta

    def test_no_attn_scale_writes_one(self):
        # An unset attn_scale_value means no scaling in hipDNN, as in cuDNN.
        graph, meta = self.generate()
        self.assertEqual(graph["nodes"][0]["attributes"]["attn_scale_value"], 1.0)
        self.assertEqual(meta["config"]["scale"], 1.0)

    def test_explicit_attn_scale_is_kept(self):
        graph, meta = self.generate("--attn-scale", "0.25")
        self.assertEqual(graph["nodes"][0]["attributes"]["attn_scale_value"], 0.25)
        self.assertEqual(meta["config"]["scale"], 0.25)


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestGenerateSdpaFwdGoldenRaggedHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = load_generator_module()

    def test_bshd_strides_keep_logical_dims(self):
        self.assertEqual(
            self.gen.compute_strides([2, 3, 5, 7], "bshd"), [105, 7, 21, 1]
        )

    def test_pack_ragged_concatenates_valid_rows_in_bshd_order(self):
        torch = self.gen.torch
        tensor = torch.arange(2 * 2 * 4 * 3, dtype=torch.float32).reshape(2, 2, 4, 3)

        packed = self.gen.pack_ragged(tensor, [3, 1])

        self.assertEqual(list(packed.shape), [4, 2, 3])
        expected_rows = [tensor[0, :, s, :] for s in range(3)] + [tensor[1, :, 0, :]]
        for row, expected in zip(packed, expected_rows):
            self.assertTrue(torch.equal(row, expected))

    def test_ragged_offsets_are_cumulative_tokens(self):
        torch = self.gen.torch

        offsets = self.gen.ragged_offsets([5, 16, 2])

        self.assertEqual(offsets.dtype, torch.int32)
        self.assertEqual(list(offsets.shape), [4, 1, 1, 1])
        self.assertEqual(offsets.flatten().tolist(), [0, 5, 21, 23])

    def test_physical_seqlens_pad_and_cap_at_s_max(self):
        pad = self.gen.RAGGED_PAD_ROWS
        self.assertEqual(
            self.gen.physical_seqlens([1, 16 - pad, 16 - pad + 1, 16], 16),
            [1 + pad, 16, 16, 16],
        )


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestGenerateSdpaFwdGoldenRaggedCli(unittest.TestCase):
    B, H_Q, H_KV, S, D_QK, D_V = 2, 2, 1, 16, 8, 4
    SEQ_LENS_Q = (5, 16)
    SEQ_LENS_KV = (3, 9)
    BUNDLE_DIR = Path("quick/SdpaFwd/bshd/bf16/test_ragged/Small")
    BF16_BYTES = 2

    @classmethod
    def setUpClass(cls):
        cls.gen = load_generator_module()
        cls.torch = cls.gen.torch

    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.base = self.root / self.BUNDLE_DIR / "Small"

    def run_generator(self, *extra_args: str) -> subprocess.CompletedProcess:
        command = [
            sys.executable,
            str(SCRIPT_PATH),
            "--base-filename",
            str(self.base),
            "--q-dims",
            *map(str, [self.B, self.H_Q, self.S, self.D_QK]),
            "--v-dims",
            *map(str, [self.B, self.H_KV, self.S, self.D_V]),
            *extra_args,
        ]
        return subprocess.run(command, capture_output=True, text=True, check=False)

    def generate_ragged(self, *extra_args: str) -> dict:
        completed = self.run_generator(
            "--ragged-offsets", "--layout", "bshd", *extra_args
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(self.base.with_suffix(".json").read_text())

    def seq_lens_args(self, seq_lens_q, seq_lens_kv) -> list[str]:
        return [
            "--seq-lens-q",
            *map(str, seq_lens_q),
            "--seq-lens-kv",
            *map(str, seq_lens_kv),
        ]

    def read_bin(self, uid: int) -> bytes:
        return Path(f"{self.base}.tensor{uid}.bin").read_bytes()

    def read_offsets(self, uid: int) -> list[int]:
        return self.torch.frombuffer(
            bytearray(self.read_bin(uid)), dtype=self.torch.int32
        ).tolist()

    def tensors_by_name(self, graph: dict) -> dict:
        return {tensor["name"]: tensor for tensor in graph["tensors"]}

    def test_uniform_ragged_bundle_layout(self):
        graph = self.generate_ragged()
        tensors = self.tensors_by_name(graph)

        expected_dims = {
            "Q": [self.B, self.H_Q, self.S, self.D_QK],
            "K": [self.B, self.H_KV, self.S, self.D_QK],
            "V": [self.B, self.H_KV, self.S, self.D_V],
            "O": [self.B, self.H_Q, self.S, self.D_V],
        }
        expected_offset_uid = {"Q": 10, "O": 10, "K": 11, "V": 11}
        uniform_offsets = [b * self.S for b in range(self.B + 1)]
        for name, dims in expected_dims.items():
            with self.subTest(tensor=name):
                tensor = tensors[name]
                self.assertEqual(tensor["dims"], dims)
                self.assertEqual(
                    tensor["strides"], self.gen.compute_strides(dims, "bshd")
                )
                self.assertEqual(
                    tensor["ragged_offset_multiplier"],
                    tensor["strides"][self.gen.SDPA_SEQ_AXIS],
                )
                self.assertEqual(
                    tensor["ragged_offset_tensor_uid"], expected_offset_uid[name]
                )

                offset_uid = expected_offset_uid[name]
                self.assertEqual(len(self.read_bin(offset_uid)), (self.B + 1) * 4)
                offsets = self.read_offsets(offset_uid)
                self.assertEqual(offsets, uniform_offsets)
                self.assertEqual(
                    len(self.read_bin(tensor["uid"])),
                    offsets[-1] * tensor["ragged_offset_multiplier"] * self.BF16_BYTES,
                )

        self.assertEqual(graph["nodes"][0]["attributes"]["mma_core_mode"], "unset")

    def test_ragged_with_seq_lens_packs_padding_with_nan_sentinel(self):
        graph = self.generate_ragged(
            *self.seq_lens_args(self.SEQ_LENS_Q, self.SEQ_LENS_KV)
        )
        tensors = self.tensors_by_name(graph)

        physical_q = self.gen.physical_seqlens(self.SEQ_LENS_Q, self.S)
        physical_kv = self.gen.physical_seqlens(self.SEQ_LENS_KV, self.S)
        self.assertEqual(
            self.read_offsets(10),
            self.gen.ragged_offsets(physical_q).flatten().tolist(),
        )
        self.assertEqual(
            self.read_offsets(11),
            self.gen.ragged_offsets(physical_kv).flatten().tolist(),
        )

        for name in ("SeqLenQ", "SeqLenKv"):
            self.assertEqual(tensors[name]["dims"], [self.B, 1, 1, 1])

        packed_o = self.torch.frombuffer(
            bytearray(self.read_bin(tensors["O"]["uid"])), dtype=self.torch.bfloat16
        ).reshape(sum(physical_q), self.H_Q, self.D_V)
        block_start = 0
        for valid, physical in zip(self.SEQ_LENS_Q, physical_q):
            block = packed_o[block_start : block_start + physical]
            self.assertTrue(self.torch.isfinite(block[:valid]).all())
            self.assertTrue(self.torch.isnan(block[valid:]).all())
            block_start += physical
        self.assertGreater(sum(physical_q), sum(self.SEQ_LENS_Q))

        verified = subprocess.run(
            [sys.executable, str(VERIFIER_PATH), "--require-data", str(self.root)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_ragged_with_seq_lens_and_stats_keeps_lse_dense(self):
        graph = self.generate_ragged(
            "--stats", *self.seq_lens_args(self.SEQ_LENS_Q, self.SEQ_LENS_KV)
        )
        lse_entry = self.tensors_by_name(graph)["LSE"]

        self.assertEqual(lse_entry["dims"], [self.B, self.H_Q, self.S, 1])
        self.assertNotIn("ragged_offset_tensor_uid", lse_entry)
        lse = self.torch.frombuffer(
            bytearray(self.read_bin(lse_entry["uid"])), dtype=self.torch.float32
        ).reshape(self.B, self.H_Q, self.S)
        for b, valid in enumerate(self.SEQ_LENS_Q):
            self.assertTrue(self.torch.isfinite(lse[b, :, :valid]).all())
            self.assertTrue(
                (lse[b, :, valid:] == float("-inf")).all(),
                f"batch {b} rows past seq_len_q must be -inf",
            )

    def assert_generator_rejects(self, expected_message: str, *args: str):
        completed = self.run_generator(*args)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(expected_message, completed.stderr)

    def test_ragged_requires_bshd_layout(self):
        self.assert_generator_rejects(
            "--ragged-offsets requires --layout bshd", "--ragged-offsets"
        )

    def test_unpaired_seq_lens_rejected(self):
        self.assert_generator_rejects(
            "must be provided together",
            "--ragged-offsets",
            "--layout",
            "bshd",
            "--seq-lens-q",
            *map(str, self.SEQ_LENS_Q),
        )

    def test_ragged_with_seq_lens_without_headroom_rejected(self):
        self.assert_generator_rejects(
            "needs headroom",
            "--ragged-offsets",
            "--layout",
            "bshd",
            *self.seq_lens_args([self.S] * self.B, [self.S] * self.B),
        )


if __name__ == "__main__":
    unittest.main()
