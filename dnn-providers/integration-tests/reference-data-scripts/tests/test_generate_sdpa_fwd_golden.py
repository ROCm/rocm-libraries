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
        self.assertEqual(meta["config"]["left_bound"], 3)

    def test_left_bound_is_written_unchanged(self):
        # --left-bound is the graph's (cuDNN) left_bound: the same window as --window-left 2.
        graph, meta = self.generate("--causal", "top_left", "--left-bound", "3")
        self.assertEqual(graph["nodes"][0]["attributes"]["left_bound"], 3)
        self.assertEqual(meta["config"]["left_bound"], 3)
        window_graph, _ = self.generate("--causal", "top_left", "--window-left", "2")
        self.assertEqual(graph["nodes"], window_graph["nodes"])

    def test_left_bound_one_is_diagonal_only(self):
        graph, meta = self.generate("--left-bound", "1", "--window-right", "0")
        self.assertEqual(graph["nodes"][0]["attributes"]["left_bound"], 1)
        self.assertEqual(meta["config"]["left_bound"], 1)
        window_graph, _ = self.generate("--window-left", "0", "--window-right", "0")
        self.assertEqual(graph["nodes"], window_graph["nodes"])

    def test_invalid_left_bound_flags_are_rejected(self):
        for extra in (
            ("--left-bound", "0"),
            ("--left-bound", "-2"),
            ("--window-left", "-2"),
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
        # Loading the generator puts utilities/ on sys.path.
        cls.common = importlib.import_module("sdpa_common")

    def assert_matches_cudnn(self, s_q, s_kv, causal, left_bound, window_right):
        torch = self.gen.torch
        keep = self.gen.build_mask(s_q, s_kv, causal, left_bound, window_right)
        diag_offset = s_kv - s_q if causal == "bottom_right" else 0
        if window_right >= 0:
            right_bound = window_right
        else:
            right_bound = 0 if causal != "none" else None
        expected = cudnn_keep_mask(
            torch, s_q, s_kv, diag_offset, left_bound, right_bound
        )
        self.assertTrue(
            torch.equal(keep, expected),
            f"S_q={s_q} S_kv={s_kv} causal={causal} left_bound={left_bound} "
            f"window_right={window_right}\nbuild_mask:\n{keep.int()}\n"
            f"cuDNN:\n{expected.int()}",
        )

    def test_left_bound_two_keeps_two_keys_including_the_diagonal(self):
        # cuDNN: left_bound = 2, TOP_LEFT, query 5 keeps keys {4, 5}.
        torch = self.gen.torch
        keep = cudnn_keep_mask(torch, 6, 6, 0, 2, 0)
        self.assertEqual(keep[5].nonzero().flatten().tolist(), [4, 5])
        self.assertEqual(self.common.left_bound_from_window_left(1), 2)
        self.assertEqual(self.common.left_bound_from_window_left(-1), -1)
        self.assertEqual(self.common.window_left_from_left_bound(2), 1)
        self.assertEqual(self.common.window_left_from_left_bound(-1), -1)

    def test_causal_windows_match_cudnn(self):
        for s_q, s_kv in ((6, 6), (4, 7), (7, 4)):
            for causal in ("top_left", "bottom_right"):
                for left_bound in (1, 2, 3, 6):
                    self.assert_matches_cudnn(s_q, s_kv, causal, left_bound, -1)

    def test_non_causal_windows_match_cudnn(self):
        for s_q, s_kv in ((6, 6), (4, 7)):
            for left_bound in (1, 2, 3, 6):
                for window_right in (-1, 0, 2):
                    self.assert_matches_cudnn(
                        s_q, s_kv, "none", left_bound, window_right
                    )


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestGenerateSdpaFwdGoldenMask(unittest.TestCase):
    """The golden keep-mask must follow the bounds the graph serializes."""

    @classmethod
    def setUpClass(cls):
        cls.gen = load_generator_module()

    def keep(self, mask):
        return [[bool(x) for x in row] for row in mask.tolist()]

    def test_causal_alone_keeps_up_to_diagonal(self):
        mask = self.gen.build_mask(3, 3, "top_left", -1, -1)
        self.assertEqual(
            self.keep(mask),
            [[True, False, False], [True, True, False], [True, True, True]],
        )

    def test_explicit_right_bound_replaces_causal_right_bound(self):
        # The graph serializes causal + window_right=1 as right_bound=1, so one key past the
        # diagonal is visible; ANDing with causal would wrongly keep right_bound at 0.
        mask = self.gen.build_mask(3, 4, "top_left", -1, 1)
        self.assertEqual(
            self.keep(mask),
            [
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ],
        )

    def test_bottom_right_causal_shifts_diagonal(self):
        mask = self.gen.build_mask(2, 4, "bottom_right", -1, -1)
        self.assertEqual(
            self.keep(mask),
            [[True, True, True, False], [True, True, True, True]],
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


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestGenerateSdpaFwdGoldenGraphOnly(unittest.TestCase):
    """--graph-only writes the golden run's graph, no tensors, and input recipes that
    give every graph input a valid value (RFC 0011 bundle size budget)."""

    def run_generator(self, base: Path, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--base-filename", str(base), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def generate_both(self, *args: str) -> tuple[Path, dict, dict, dict]:
        """Golden and graph-only runs of the same arguments, into the same directory."""
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        bundle_dir = Path(tmp.name) / "quick/SdpaFwd/bhsd/x/case/Small"
        bundle_dir.mkdir(parents=True)
        base = bundle_dir / "Small"
        golden = self.run_generator(base, *args)
        self.assertEqual(golden.returncode, 0, golden.stderr)
        golden_graph = json.loads(base.with_suffix(".json").read_text())
        (bundle_dir / "Small.tensors.dvc").write_text("outs: []\n")
        graph_only = self.run_generator(base, *args, "--graph-only")
        self.assertEqual(graph_only.returncode, 0, graph_only.stderr)
        graph = json.loads(base.with_suffix(".json").read_text())
        meta = json.loads(Path(f"{base}.meta.json").read_text())
        return bundle_dir, golden_graph, graph, meta

    def assert_graph_only(self, bundle_dir, golden_graph, graph, meta):
        self.assertEqual(graph, golden_graph)
        self.assertEqual(
            sorted(p.name for p in bundle_dir.iterdir()),
            ["Small.json", "Small.meta.json"],
        )
        self.assertEqual(meta["format_version"], 1)
        self.assertEqual(meta["generator"], "generate_sdpa_fwd_golden.py")
        self.assertTrue(meta["reference_source"].strip())
        self.assertNotIn("input_range", meta)
        verified = subprocess.run(
            [sys.executable, str(VERIFIER_PATH), "--require-data", str(bundle_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(verified.returncode, 0, verified.stderr)
        # Every non-output tensor of the graph has a recipe.
        node = graph["nodes"][0]
        input_uids = {v for v in node["inputs"].values() if v is not None}
        self.assertEqual({int(uid) for uid in meta["inputs"]}, input_uids)

    def test_fp8_codes_span_the_format_with_positive_fixed_descales(self):
        args = (
            "--dtype",
            "fp8",
            "--q-dims",
            "1",
            "2",
            "8",
            "16",
            "--v-dims",
            "1",
            "2",
            "8",
            "16",
            "--causal",
            "bottom_right",
            "--stats",
            "--seed",
            "3",
        )
        bundle_dir, golden_graph, graph, meta = self.generate_both(*args)
        self.assert_graph_only(bundle_dir, golden_graph, graph, meta)
        for uid in ("0", "1", "2"):
            self.assertEqual(
                meta["inputs"][uid], {"kind": "free", "lo": -448.0, "hi": 448.0}
            )
        for uid in ("7", "8", "9"):
            self.assertEqual(
                meta["inputs"][uid], {"kind": "fixed", "value": 1.0 / 448.0}
            )
        self.assertEqual(meta["seed"], 3)

    def test_block_mask_bytes_keep_bit_seven(self):
        args = (
            "--block-mask",
            "--q-dims",
            "1",
            "2",
            "128",
            "16",
            "--v-dims",
            "1",
            "1",
            "1024",
            "16",
            "--stats",
            "--seed",
            "5",
        )
        bundle_dir, golden_graph, graph, meta = self.generate_both(*args)
        self.assert_graph_only(bundle_dir, golden_graph, graph, meta)
        self.assertEqual(
            meta["inputs"]["17"], {"kind": "free", "lo": 128.0, "hi": 255.0}
        )
        for uid in ("0", "1", "2"):
            self.assertEqual(
                meta["inputs"][uid], {"kind": "free", "lo": -1.0, "hi": 1.0}
            )

    def test_alibi_has_no_extra_inputs(self):
        args = (
            "--alibi",
            "--causal",
            "top_left",
            "--q-dims",
            "1",
            "2",
            "8",
            "16",
            "--v-dims",
            "1",
            "1",
            "8",
            "16",
            "--stats",
        )
        bundle_dir, golden_graph, graph, meta = self.generate_both(*args)
        self.assert_graph_only(bundle_dir, golden_graph, graph, meta)
        self.assertEqual(sorted(meta["inputs"]), ["0", "1", "2"])

    def test_rejects_inputs_a_recipe_cannot_express(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp) / "Small"
            dims = ("--q-dims", "2", "1", "8", "16", "--v-dims", "2", "1", "8", "16")
            for extra in (
                ("--seq-lens-q", "8", "4", "--seq-lens-kv", "8", "4"),
                ("--dtype", "fp8", "--fp8-scaling"),
                ("--block-mask",),  # fewer than 8 key tiles
            ):
                with self.subTest(extra=extra):
                    completed = self.run_generator(base, *dims, *extra, "--graph-only")
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn("--graph-only", completed.stderr)


# ctest reports exit code 77 as Skipped (SKIP_RETURN_CODE), so a run in which every
# test skipped, e.g. without PyTorch, does not look like a pass.
SKIP_EXIT = 77

if __name__ == "__main__":
    result = unittest.main(exit=False).result
    if not result.wasSuccessful():
        sys.exit(1)
    if result.testsRun > 0 and len(result.skipped) == result.testsRun:
        sys.exit(SKIP_EXIT)
