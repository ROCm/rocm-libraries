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

    def test_window_left_is_written_as_cudnn_left_bound(self):
        # --window-left is FlashAttention's window_size_left (W keeps W + 1 keys);
        # the graph's left_bound follows cuDNN (L keeps L keys), so it is W + 1.
        graph, meta = self.generate("--causal", "top_left", "--window-left", "2")
        attributes = graph["nodes"][0]["attributes"]
        self.assertEqual(attributes["left_bound"], 3)
        self.assertEqual(attributes["right_bound"], 0)
        self.assertEqual(meta["config"]["window_left"], 2)

    def test_left_bound_is_written_unchanged(self):
        # --left-bound is the graph's (cuDNN) left_bound: the same window as --window-left 2.
        graph, meta = self.generate("--causal", "top_left", "--left-bound", "3")
        self.assertEqual(graph["nodes"][0]["attributes"]["left_bound"], 3)
        self.assertEqual(meta["config"]["window_left"], 2)
        window_graph, _ = self.generate("--causal", "top_left", "--window-left", "2")
        self.assertEqual(graph["nodes"], window_graph["nodes"])

    def test_left_bound_one_is_diagonal_only(self):
        graph, meta = self.generate("--left-bound", "1", "--window-right", "0")
        self.assertEqual(graph["nodes"][0]["attributes"]["left_bound"], 1)
        self.assertEqual(meta["config"]["window_left"], 0)

    def test_invalid_left_bound_flags_are_rejected(self):
        for extra in (
            ("--left-bound", "0"),
            ("--left-bound", "-2"),
            ("--left-bound", "3", "--window-left", "2"),
        ):
            with self.subTest(extra=extra), self.assertRaises(
                subprocess.CalledProcessError
            ):
                self.generate(*extra)


def cudnn_keep_mask(torch, s_q, s_kv, diag_offset, left_bound, right_bound):
    """cuDNN's sliding-window rule, as cudnn-frontend's own reference states it
    (test/python/sdpa/fp16_ref.py): with rel = col - row, a key is masked when
    rel > diag_offset + right_bound or rel <= diag_offset - left_bound. It shares no
    code or convention with hipDNN's references, so it pins the contract from outside.
    """
    rows = torch.arange(s_q).view(s_q, 1)
    cols = torch.arange(s_kv).view(1, s_kv)
    rel = cols - rows
    masked = torch.zeros(s_q, s_kv, dtype=torch.bool)
    if right_bound is not None:
        masked |= rel > diag_offset + right_bound
    if left_bound is not None:
        masked |= rel <= diag_offset - left_bound
    return ~masked


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestGenerateSdpaFwdGoldenWindowMatchesCudnn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = load_generator_module()

    def assert_matches_cudnn(self, s_q, s_kv, causal, window_left, window_right):
        torch = self.gen.torch
        keep = self.gen.build_mask(s_q, s_kv, causal, window_left, window_right)
        diag_offset = s_kv - s_q if causal == "bottom_right" else 0
        if causal != "none":
            right_bound = 0
        else:
            right_bound = window_right if window_right >= 0 else None
        expected = cudnn_keep_mask(
            torch,
            s_q,
            s_kv,
            diag_offset,
            self.gen.left_bound_from_window_left(window_left),
            right_bound,
        )
        self.assertTrue(
            torch.equal(keep, expected),
            f"S_q={s_q} S_kv={s_kv} causal={causal} window=({window_left},"
            f"{window_right})\nbuild_mask:\n{keep.int()}\ncuDNN:\n{expected.int()}",
        )

    def test_left_bound_two_keeps_two_keys_including_the_diagonal(self):
        # cuDNN: left_bound = 2, TOP_LEFT, query 5 keeps keys {4, 5}.
        torch = self.gen.torch
        keep = cudnn_keep_mask(torch, 6, 6, 0, 2, 0)
        self.assertEqual(keep[5].nonzero().flatten().tolist(), [4, 5])
        self.assertEqual(self.gen.left_bound_from_window_left(1), 2)
        self.assertEqual(self.gen.window_left_from_left_bound(2), 1)
        self.assertEqual(self.gen.window_left_from_left_bound(-1), -1)

    def test_causal_windows_match_cudnn(self):
        for s_q, s_kv in ((6, 6), (4, 7), (7, 4)):
            for causal in ("top_left", "bottom_right"):
                for window_left in (0, 1, 2, 5):
                    self.assert_matches_cudnn(s_q, s_kv, causal, window_left, -1)

    def test_non_causal_windows_match_cudnn(self):
        for s_q, s_kv in ((6, 6), (4, 7)):
            for window_left in (0, 1, 2, 5):
                for window_right in (-1, 0, 2):
                    self.assert_matches_cudnn(
                        s_q, s_kv, "none", window_left, window_right
                    )


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

    def test_varied_seqlens_are_unequal_bounded_and_causal_safe(self):
        for batch, s_q, s_kv in [(2, 16, 16), (3, 256, 256), (5, 256, 256), (3, 8, 8)]:
            with self.subTest(batch=batch, s_q=s_q, s_kv=s_kv):
                q, kv = self.gen.varied_seqlens(batch, s_q, s_kv)
                self.assertEqual((len(q), len(kv)), (batch, batch))
                self.assertEqual((max(q), max(kv)), (s_q, s_kv))
                self.assertTrue(all(1 <= v <= s_q for v in q))
                self.assertTrue(all(1 <= v <= s_kv for v in kv))
                self.assertTrue(all(k >= v for k, v in zip(kv, q)))
                self.assertGreater(len(set(q)), 1)
                self.assertNotEqual(
                    self.gen.ragged_offsets(q).flatten().tolist(),
                    self.gen.ragged_offsets(kv).flatten().tolist(),
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

    def run_generator(
        self, *extra_args: str, batch: int | None = None
    ) -> subprocess.CompletedProcess:
        b = batch if batch is not None else self.B
        command = [
            sys.executable,
            str(SCRIPT_PATH),
            "--base-filename",
            str(self.base),
            "--q-dims",
            *map(str, [b, self.H_Q, self.S, self.D_QK]),
            "--v-dims",
            *map(str, [b, self.H_KV, self.S, self.D_V]),
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

    def test_ragged_bundle_layout(self):
        graph = self.generate_ragged()
        tensors = self.tensors_by_name(graph)

        self.assertEqual(
            set(tensors),
            {"Q", "K", "V", "O", "RaggedOffsetQ", "RaggedOffsetK"},
        )
        self.assertIsNone(graph["nodes"][0]["inputs"]["seq_len_q_tensor_uid"])

        expected_dims = {
            "Q": [self.B, self.H_Q, self.S, self.D_QK],
            "K": [self.B, self.H_KV, self.S, self.D_QK],
            "V": [self.B, self.H_KV, self.S, self.D_V],
            "O": [self.B, self.H_Q, self.S, self.D_V],
        }
        expected_offset_uid = {"Q": 10, "O": 10, "K": 11, "V": 11}
        lens_q, lens_kv = self.gen.varied_seqlens(self.B, self.S, self.S)
        expected_offsets = {
            10: self.gen.ragged_offsets(lens_q).flatten().tolist(),
            11: self.gen.ragged_offsets(lens_kv).flatten().tolist(),
        }
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
                self.assertEqual(offsets, expected_offsets[offset_uid])
                self.assertEqual(
                    len(self.read_bin(tensor["uid"])),
                    offsets[-1] * tensor["ragged_offset_multiplier"] * self.BF16_BYTES,
                )

        self.assertEqual(graph["nodes"][0]["attributes"]["mma_core_mode"], "unset")

    def test_ragged_bundle_matches_independent_per_batch_reference(self):
        graph = self.generate_ragged("--causal", "bottom_right")
        torch = self.torch
        lens_q, lens_kv = self.gen.varied_seqlens(self.B, self.S, self.S)
        scale = graph["nodes"][0]["attributes"]["attn_scale_value"]

        def load(uid, tokens, heads, dim):
            raw = bytearray(self.read_bin(uid))
            return (
                torch.frombuffer(raw, dtype=torch.bfloat16)
                .reshape(tokens, heads, dim)
                .float()
            )

        packed_q = load(self.gen.UID_Q, sum(lens_q), self.H_Q, self.D_QK)
        packed_k = load(self.gen.UID_K, sum(lens_kv), self.H_KV, self.D_QK)
        packed_v = load(self.gen.UID_V, sum(lens_kv), self.H_KV, self.D_V)
        packed_o = load(self.gen.UID_O, sum(lens_q), self.H_Q, self.D_V)

        q_start = kv_start = 0
        for b, (lq, lkv) in enumerate(zip(lens_q, lens_kv)):
            with self.subTest(batch=b):
                q_b = packed_q[q_start : q_start + lq]
                group = self.H_Q // self.H_KV
                k_b = packed_k[kv_start : kv_start + lkv].repeat_interleave(
                    group, dim=1
                )
                v_b = packed_v[kv_start : kv_start + lkv].repeat_interleave(
                    group, dim=1
                )
                scores = torch.einsum("qhd,khd->hqk", q_b, k_b) * scale
                keep = torch.arange(lkv)[None, :] <= torch.arange(lq)[:, None] + (
                    lkv - lq
                )
                scores = scores.masked_fill(~keep, float("-inf"))
                expected = torch.einsum(
                    "hqk,khd->qhd", torch.softmax(scores, dim=-1), v_b
                )
                torch.testing.assert_close(
                    packed_o[q_start : q_start + lq], expected, atol=2e-2, rtol=2e-2
                )
            q_start += lq
            kv_start += lkv

        verified = subprocess.run(
            [sys.executable, str(VERIFIER_PATH), "--require-data", str(self.root)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(verified.returncode, 0, verified.stderr)

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

    def assert_generator_rejects(
        self, expected_message: str, *args: str, batch: int | None = None
    ):
        completed = self.run_generator(*args, batch=batch)
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

    def test_ragged_without_seq_lens_rejects_degenerate_equal_lengths(self):
        self.assert_generator_rejects(
            "gives equal Q and KV lengths",
            "--ragged-offsets",
            "--layout",
            "bshd",
            batch=1,
        )

    def test_ragged_with_stats_rejected(self):
        for extra in ([], self.seq_lens_args(self.SEQ_LENS_Q, self.SEQ_LENS_KV)):
            with self.subTest(seq_lens=bool(extra)):
                self.assert_generator_rejects(
                    "does not support --stats",
                    "--ragged-offsets",
                    "--layout",
                    "bshd",
                    "--stats",
                    *extra,
                )


if __name__ == "__main__":
    unittest.main()
