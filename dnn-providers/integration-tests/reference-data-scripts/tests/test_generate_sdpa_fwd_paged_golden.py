#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = SCRIPT_DIR / "generate_sdpa_fwd_paged_golden.py"
VERIFIER_PATH = SCRIPT_DIR / "verify_golden_bundles.py"
TORCH_MISSING = importlib.util.find_spec("torch") is None
NUMPY_MISSING = importlib.util.find_spec("numpy") is None

UID_Q, UID_K, UID_V, UID_O, UID_LSE = 0, 1, 2, 3, 4
UID_SEQ_LEN_Q, UID_SEQ_LEN_KV, UID_RAGGED_OFFSET_QO = 5, 6, 10
UID_PAGE_TABLE_K, UID_PAGE_TABLE_V = 12, 13
UID_SINK = 14

CANONICAL_SUPPORT = '{\n  "claims": {},\n  "version": 1\n}\n'


def load_generator_module():
    # The generator imports its helpers from generate_sdpa_fwd_golden.py next to it.
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "generate_sdpa_fwd_paged_golden", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PagedBundle:
    """A generated bundle read back with numpy, independent of the generator."""

    NP_DTYPES = {"half": "<f2", "float": "<f4", "int32": "<i4"}

    def __init__(self, base: Path):
        import numpy as np

        self.np = np
        self.base = base
        self.graph = json.loads(base.with_suffix(".json").read_text())
        self.tensors = {t["uid"]: t for t in self.graph["tensors"]}
        self.node = self.graph["nodes"][0]

    def bin_path(self, uid: int) -> Path:
        return Path(f"{self.base}.tensor{uid}.bin")

    def flat(self, uid: int):
        dtype = self.np.dtype(self.NP_DTYPES[self.tensors[uid]["data_type"]])
        return self.np.frombuffer(self.bin_path(uid).read_bytes(), dtype=dtype)

    def strided(self, uid: int):
        """Logical view through the declared dims/strides (non-ragged tensors)."""
        spec = self.tensors[uid]
        buf = self.flat(uid)
        return self.np.lib.stride_tricks.as_strided(
            buf,
            shape=spec["dims"],
            strides=[s * buf.itemsize for s in spec["strides"]],
            writeable=False,
        )

    def packed(self, uid: int):
        """[tokens, H, D] rows of a ragged (packed, token-major) tensor."""
        spec = self.tensors[uid]
        return self.flat(uid).reshape(-1, spec["dims"][1], spec["dims"][3])

    def is_ragged(self, uid: int) -> bool:
        return self.tensors[uid].get("ragged_offset_tensor_uid") is not None


def naive_paged_sdpa(bundle: PagedBundle):
    """Token-by-token paged attention in float64 straight from the bundle bytes.

    Returns {(b, h, i): (o_row, lse)} for every live row; a fully masked row has a
    zero o_row and lse -inf, or lse = sink with an attention sink.
    """
    np = bundle.np
    inputs = bundle.node["inputs"]
    attrs = bundle.node["attributes"]
    q_spec = bundle.tensors[UID_Q]
    B, H_q, S_q, _ = q_spec["dims"]
    k = bundle.strided(UID_K).astype(np.float64)
    v = bundle.strided(UID_V).astype(np.float64)
    _, H_kv, block_size, _ = bundle.tensors[UID_K]["dims"]
    pt_k = bundle.strided(inputs["page_table_k_tensor_uid"])
    pt_v = bundle.strided(inputs["page_table_v_tensor_uid"])
    seq_kv = bundle.flat(inputs["seq_len_kv_tensor_uid"])
    ragged = bundle.is_ragged(UID_Q)
    if ragged:
        q_rows = bundle.packed(UID_Q).astype(np.float64)
        offsets = bundle.flat(UID_RAGGED_OFFSET_QO)
        seq_q = bundle.flat(inputs["seq_len_q_tensor_uid"])
    else:
        q_dense = bundle.strided(UID_Q).astype(np.float64)
        seq_q = [S_q] * B
    scale = attrs["attn_scale_value"]
    left = attrs["left_bound"] if attrs["left_bound"] is not None else -1
    right = attrs["right_bound"] if attrs["right_bound"] is not None else -1
    bottom_right = attrs["diagonal_alignment"] == "BOTTOM_RIGHT"
    sink_uid = inputs.get("sink_token_tensor_uid")
    sinks = (
        bundle.strided(sink_uid).astype(np.float64).reshape(-1)
        if sink_uid is not None
        else None
    )

    rows = {}
    for b in range(B):
        sq, skv = int(seq_q[b]), int(seq_kv[b])
        off = skv - sq if bottom_right else 0
        for h in range(H_q):
            hk = h // (H_q // H_kv)
            for i in range(sq):
                q = q_rows[offsets[b] + i, h] if ragged else q_dense[b, h, i]
                keys = [
                    t
                    for t in range(skv)
                    if (right < 0 or t <= i + off + right)
                    and (left < 0 or t >= i + off - left)
                ]
                sink = sinks[h] if sinks is not None else -math.inf
                if not keys:
                    rows[(b, h, i)] = (np.zeros(v.shape[3]), sink)
                    continue
                scores = []
                values = []
                for t in keys:
                    kb = pt_k[b, 0, t // block_size, 0]
                    vb = pt_v[b, 0, t // block_size, 0]
                    scores.append(float(q @ k[kb, hk, t % block_size]) * scale)
                    values.append(v[vb, hk, t % block_size])
                scores = np.array(scores)
                # The sink joins the max and the denominator, unscaled, with no value row.
                peak = max(scores.max(), sink)
                weights = np.exp(scores - peak)
                total = weights.sum() + math.exp(sink - peak)
                o_row = (weights[:, None] * np.array(values)).sum(axis=0) / total
                rows[(b, h, i)] = (o_row, peak + math.log(total))
    return rows


@unittest.skipIf(TORCH_MISSING, "the generator requires PyTorch")
class TestGenerateSdpaFwdPagedHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = load_generator_module()
        cls.torch = cls.gen.torch

    def test_kv_container_strides_are_vllm_layout(self):
        # [nb, Hkv, bs, D] stored as [nb, bs, Hkv, D].
        self.assertEqual(
            self.gen.kv_container_strides([5, 2, 16, 8], 8), [256, 8, 16, 1]
        )
        # Interleaved: every D row is 2*D apart.
        self.assertEqual(
            self.gen.kv_container_strides([5, 2, 16, 8], 16), [512, 16, 32, 1]
        )

    def test_materialize_strided_fills_gaps(self):
        torch = self.torch
        logical = torch.arange(1, 7, dtype=torch.float32).reshape(1, 1, 3, 2)
        buf = self.gen.materialize_strided(logical, [12, 4, 4, 1], -5.0)
        self.assertEqual(
            buf.numel(), self.gen.element_space([1, 1, 3, 2], [12, 4, 4, 1])
        )
        self.assertEqual(buf.tolist(), [1, 2, -5, -5, 3, 4, -5, -5, 5, 6])

    def test_page_table_uses_distinct_blocks_and_marks_unused(self):
        torch = self.torch
        table = self.gen.build_page_table(
            [5, 0, 8], 4, 2, 6, torch.Generator().manual_seed(1)
        )
        self.assertEqual(list(table.shape), [3, 1, 2, 1])
        rows = table[:, 0, :, 0].tolist()
        self.assertEqual(rows[1], [-1, -1])
        used = rows[0] + rows[2]
        self.assertEqual(len(set(used)), 4)
        self.assertTrue(all(0 <= block < 6 for block in used))


@unittest.skipIf(TORCH_MISSING or NUMPY_MISSING, "the generator requires PyTorch")
class TestGenerateSdpaFwdPagedCli(unittest.TestCase):
    H_Q, H_KV, D, BS, MAX_KV = 4, 2, 8, 4, 12
    SPARE_FILL = 100.0

    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def base_for(self, variant: str) -> Path:
        return (
            self.root
            / "quick"
            / "SdpaFwd"
            / "bshd"
            / "fp16"
            / variant
            / "Small"
            / "Small"
        )

    def run_generator(self, variant: str, *args: str) -> subprocess.CompletedProcess:
        command = [
            sys.executable,
            str(SCRIPT_PATH),
            "--base-filename",
            str(self.base_for(variant)),
            "--dtype",
            "fp16",
            "--num-heads-q",
            str(self.H_Q),
            "--num-heads-kv",
            str(self.H_KV),
            "--head-dim",
            str(self.D),
            "--block-size",
            str(self.BS),
            "--max-seq-len-kv",
            str(self.MAX_KV),
            "--attn-scale",
            str(1.0 / math.sqrt(self.D)),
            *args,
        ]
        return subprocess.run(command, capture_output=True, text=True, check=False)

    def generate(self, variant: str, *args: str) -> PagedBundle:
        completed = self.run_generator(variant, *args)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return PagedBundle(self.base_for(variant))

    def dense(self, *extra: str) -> PagedBundle:
        return self.generate(
            "dense", "--seq-q", "2", "--seq-lens-kv", "7", "0", "12", "3", *extra
        )

    def ragged(self, *extra: str) -> PagedBundle:
        return self.generate(
            "ragged",
            "--ragged-q",
            "--seq-lens-q",
            "3",
            "0",
            "2",
            "--seq-lens-kv",
            "9",
            "5",
            "2",
            "--stats",
            "--interleaved-kv",
            "--window-left",
            "3",
            *extra,
        )

    def assert_matches_naive(self, bundle: PagedBundle):
        np = bundle.np
        rows = naive_paged_sdpa(bundle)
        self.assertTrue(rows)
        ragged = bundle.is_ragged(UID_O)
        if ragged:
            o_rows = bundle.packed(UID_O).astype(np.float64)
            offsets = bundle.flat(UID_RAGGED_OFFSET_QO)
        else:
            o_dense = bundle.strided(UID_O).astype(np.float64)
        has_lse = bundle.node["outputs"]["stats_tensor_uid"] is not None
        if has_lse:
            lse = (
                bundle.flat(UID_LSE).reshape(-1, self.H_Q)
                if ragged
                else bundle.strided(UID_LSE)
            )
        for (b, h, i), (expected, expected_lse) in rows.items():
            got = o_rows[offsets[b] + i, h] if ragged else o_dense[b, h, i]
            np.testing.assert_allclose(
                got, expected, rtol=1e-2, atol=2e-3, err_msg=f"O[b={b}, h={h}, row={i}]"
            )
            if has_lse:
                got_lse = lse[offsets[b] + i, h] if ragged else lse[b, h, i, 0]
                self.assertAlmostEqual(float(got_lse), expected_lse, delta=1e-3)

    def test_graph_fields_and_uids(self):
        bundle = self.ragged()
        tensors = bundle.tensors
        self.assertEqual(
            set(tensors),
            {
                UID_Q,
                UID_K,
                UID_V,
                UID_O,
                UID_LSE,
                UID_SEQ_LEN_Q,
                UID_SEQ_LEN_KV,
                UID_RAGGED_OFFSET_QO,
                UID_PAGE_TABLE_K,
                UID_PAGE_TABLE_V,
            },
        )
        inputs = bundle.node["inputs"]
        self.assertEqual(inputs["page_table_k_tensor_uid"], UID_PAGE_TABLE_K)
        self.assertEqual(inputs["page_table_v_tensor_uid"], UID_PAGE_TABLE_V)
        self.assertEqual(inputs["seq_len_q_tensor_uid"], UID_SEQ_LEN_Q)
        self.assertEqual(inputs["seq_len_kv_tensor_uid"], UID_SEQ_LEN_KV)
        self.assertEqual(bundle.node["outputs"]["stats_tensor_uid"], UID_LSE)

        attrs = bundle.node["attributes"]
        self.assertEqual(attrs["max_seq_len_kv"], self.MAX_KV)
        self.assertFalse(attrs["padding_mask"])
        self.assertEqual(attrs["diagonal_alignment"], "BOTTOM_RIGHT")
        self.assertEqual(attrs["right_bound"], 0)
        self.assertEqual(attrs["left_bound"], 3)
        self.assertEqual(bundle.graph["io_data_type"], "half")

        s_q = 3  # max(seq_lens_q)
        self.assertEqual(tensors[UID_Q]["dims"], [3, self.H_Q, s_q, self.D])
        for uid, multiplier in [(UID_Q, self.H_Q * self.D), (UID_O, self.H_Q * self.D)]:
            self.assertEqual(
                tensors[uid]["ragged_offset_tensor_uid"], UID_RAGGED_OFFSET_QO
            )
            self.assertEqual(tensors[uid]["ragged_offset_multiplier"], multiplier)
            self.assertEqual(tensors[uid]["strides"][2], multiplier)
        self.assertEqual(tensors[UID_LSE]["dims"], [3, self.H_Q, s_q, 1])
        self.assertEqual(
            tensors[UID_LSE]["ragged_offset_tensor_uid"], UID_RAGGED_OFFSET_QO
        )
        self.assertEqual(tensors[UID_LSE]["ragged_offset_multiplier"], self.H_Q)
        self.assertEqual(bundle.flat(UID_RAGGED_OFFSET_QO).tolist(), [0, 3, 3, 5])
        self.assertEqual(bundle.flat(UID_SEQ_LEN_Q).tolist(), [3, 0, 2])
        self.assertEqual(bundle.flat(UID_SEQ_LEN_KV).tolist(), [9, 5, 2])

        # Interleaved containers: D rows 2*D apart, filler in the other half.
        num_blocks = tensors[UID_K]["dims"][0]
        for uid in (UID_K, UID_V):
            self.assertEqual(
                tensors[uid]["dims"], [num_blocks, self.H_KV, self.BS, self.D]
            )
            pitch = 2 * self.D
            self.assertEqual(
                tensors[uid]["strides"],
                [self.BS * self.H_KV * pitch, pitch, self.H_KV * pitch, 1],
            )
            flat = bundle.flat(uid)
            self.assertEqual(
                flat.size, num_blocks * self.BS * self.H_KV * pitch - self.D
            )
            rows = flat[: (flat.size // pitch) * pitch].reshape(-1, pitch)
            self.assertTrue((rows[:, self.D :] == 0).all())

    def test_shared_page_table_binds_one_uid(self):
        bundle = self.ragged("--shared-page-table")
        inputs = bundle.node["inputs"]
        self.assertEqual(inputs["page_table_k_tensor_uid"], UID_PAGE_TABLE_K)
        self.assertEqual(inputs["page_table_v_tensor_uid"], UID_PAGE_TABLE_K)
        self.assertNotIn(UID_PAGE_TABLE_V, bundle.tensors)
        self.assertFalse(bundle.bin_path(UID_PAGE_TABLE_V).exists())
        self.assert_matches_naive(bundle)

    def test_page_tables_shape_and_unused_entries(self):
        bundle = self.dense()
        seq_kv = [7, 0, 12, 3]
        max_blocks = math.ceil(self.MAX_KV / self.BS)
        used = sum(math.ceil(s / self.BS) for s in seq_kv)
        num_blocks = bundle.tensors[UID_K]["dims"][0]
        self.assertEqual(num_blocks, used + 2)  # default --spare-blocks 2
        for uid in (UID_PAGE_TABLE_K, UID_PAGE_TABLE_V):
            spec = bundle.tensors[uid]
            self.assertEqual(spec["dims"], [4, 1, max_blocks, 1])
            self.assertEqual(spec["strides"], [max_blocks, max_blocks, 1, 1])
            self.assertEqual(spec["data_type"], "int32")
            table = bundle.strided(uid)[:, 0, :, 0]
            referenced = []
            for b, skv in enumerate(seq_kv):
                n = math.ceil(skv / self.BS)
                self.assertTrue((table[b, n:] == -1).all(), f"uid {uid} batch {b}")
                referenced += table[b, :n].tolist()
            self.assertEqual(len(set(referenced)), used)
            self.assertTrue(all(0 <= block < num_blocks for block in referenced))

            # Spare blocks hold the large finite fill.
            container = bundle.strided(UID_K if uid == UID_PAGE_TABLE_K else UID_V)
            for block in set(range(num_blocks)) - set(referenced):
                self.assertTrue((container[block] == self.SPARE_FILL).all())

    def test_dense_matches_naive_paged_loop(self):
        self.assert_matches_naive(self.dense())

    def test_ragged_window_interleaved_matches_naive_paged_loop(self):
        self.assert_matches_naive(self.ragged())

    def test_zero_seq_len_kv_rows_are_zero(self):
        bundle = self.generate(
            "zero_kv",
            "--ragged-q",
            "--seq-lens-q",
            "3",
            "2",
            "--seq-lens-kv",
            "9",
            "0",
        )
        np = bundle.np
        o_rows = bundle.packed(UID_O)
        self.assertTrue(np.isfinite(o_rows.astype(np.float32)).all())
        self.assertTrue((o_rows[3:5] == 0).all())
        self.assertTrue((o_rows[:3] != 0).any())
        self.assert_matches_naive(bundle)

        dense = self.dense()
        self.assertTrue((dense.strided(UID_O)[1] == 0).all())

    def test_dvc_lists_every_bin_with_md5(self):
        bundle = self.ragged()
        dvc_text = Path(f"{bundle.base}.tensors.dvc").read_text()
        lines = dvc_text.splitlines()
        self.assertEqual(lines[0], "outs:")
        entries = [lines[i : i + 5] for i in range(1, len(lines), 5)]
        bins = sorted(p.name for p in bundle.base.parent.glob("Small.tensor*.bin"))
        self.assertEqual([e[0][len("- path: ") :] for e in entries], bins)
        for entry in entries:
            data = (bundle.base.parent / entry[0][len("- path: ") :]).read_bytes()
            self.assertEqual(
                entry[1:],
                [
                    "  remote: golden-data",
                    f"  md5: {hashlib.md5(data).hexdigest()}",
                    f"  size: {len(data)}",
                    "  hash: md5",
                ],
            )
        self.assertEqual(
            {int(name.split("tensor")[1].split(".")[0]) for name in bins},
            set(bundle.tensors),
        )

    def test_support_sidecar_is_canonical_and_kept(self):
        bundle = self.dense()
        support = Path(f"{bundle.base}.support.json")
        self.assertEqual(support.read_bytes().decode(), CANONICAL_SUPPORT)

        claimed = '{\n  "claims": {"X": {"gfx942": ["linux"]}},\n  "version": 1\n}\n'
        support.write_text(claimed)
        self.dense()
        self.assertEqual(support.read_text(), claimed)

    def test_meta_records_paged_config(self):
        bundle = self.ragged("--shared-page-table")
        meta = json.loads(Path(f"{bundle.base}.meta.json").read_text())
        self.assertEqual(meta["format_version"], 1)
        self.assertEqual(meta["generator"], "generate_sdpa_fwd_paged_golden.py")
        self.assertEqual(len(meta["generator_sha256"]), 64)
        self.assertTrue(meta["reference_source"].startswith("PyTorch "))
        self.assertIn("gather-then-dense", meta["reference_backend"])
        self.assertEqual(meta["seed"], 42)
        config = meta["config"]
        self.assertEqual(config["block_size"], self.BS)
        self.assertEqual(config["max_seq_len_kv"], self.MAX_KV)
        self.assertEqual(config["num_blocks"], bundle.tensors[UID_K]["dims"][0])
        self.assertEqual(config["seq_lens_q"], [3, 0, 2])
        self.assertEqual(config["seq_lens_kv"], [9, 5, 2])
        self.assertTrue(config["shared_page_table"])
        self.assertEqual(
            config["page_table_k"],
            bundle.strided(UID_PAGE_TABLE_K)[:, 0, :, 0].tolist(),
        )

    def test_verifier_accepts_bundles(self):
        self.dense()
        self.ragged()
        verified = subprocess.run(
            [sys.executable, str(VERIFIER_PATH), "--require-data", str(self.root)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(verified.returncode, 0, verified.stderr)

    def assert_generator_rejects(self, expected_message: str, *args: str):
        completed = self.run_generator("rejected", *args)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(expected_message, completed.stderr)

    def test_stats_with_fully_masked_row_rejected(self):
        self.assert_generator_rejects(
            "LSE would be -inf", "--seq-q", "1", "--seq-lens-kv", "4", "0", "--stats"
        )

    def test_seq_len_kv_above_max_rejected(self):
        self.assert_generator_rejects(
            "must be in [0, max_seq_len_kv", "--seq-q", "1", "--seq-lens-kv", "13"
        )

    def test_ragged_requires_seq_lens_q(self):
        self.assert_generator_rejects(
            "--ragged-q requires --seq-lens-q", "--ragged-q", "--seq-lens-kv", "4"
        )

    def test_sink_tensor_graph_fields(self):
        for sink_dtype, json_dtype in (("io", "half"), ("float", "float")):
            bundle = self.dense("--sinks", "--sink-dtype", sink_dtype)
            self.assertEqual(bundle.node["inputs"]["sink_token_tensor_uid"], UID_SINK)
            spec = bundle.tensors[UID_SINK]
            self.assertEqual(spec["dims"], [1, self.H_Q, 1, 1])
            self.assertEqual(spec["strides"], [self.H_Q, 1, 1, 1])
            self.assertEqual(spec["data_type"], json_dtype)
            meta = json.loads(Path(f"{bundle.base}.meta.json").read_text())
            self.assertEqual(meta["config"]["sink_dtype"], sink_dtype)
            self.assertEqual(
                meta["config"]["sinks"],
                [float(x) for x in bundle.strided(UID_SINK).reshape(-1)],
            )
        self.assertIsNone(self.dense().node["inputs"]["sink_token_tensor_uid"])

    def test_sinks_match_naive_paged_loop(self):
        # Dense Q with an empty sequence and stats: keyless rows give LSE = sink.
        dense = self.dense("--sinks", "--stats")
        self.assert_matches_naive(dense)
        sinks = dense.strided(UID_SINK).reshape(-1)
        lse = dense.strided(UID_LSE)
        for h in range(self.H_Q):
            self.assertTrue((lse[1, h, :, 0] == sinks[h]).all())
        self.assertTrue((dense.strided(UID_O)[1] == 0).all())
        self.assert_matches_naive(self.ragged("--sinks", "--sink-dtype", "float"))

    def test_sinks_keep_the_other_tensors(self):
        # The sink is drawn last, so every other tensor matches the sink-free bundle.
        plain = self.dense()
        plain_bytes = {
            uid: plain.bin_path(uid).read_bytes() for uid in (UID_Q, UID_K, UID_V)
        }
        with_sinks = self.dense("--sinks")
        for uid, data in plain_bytes.items():
            self.assertEqual(with_sinks.bin_path(uid).read_bytes(), data)

    def test_sink_bundles_pass_the_verifier(self):
        self.dense("--sinks", "--stats")
        self.ragged("--sinks")
        verified = subprocess.run(
            [sys.executable, str(VERIFIER_PATH), "--require-data", str(self.root)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(verified.returncode, 0, verified.stderr)


# ctest reports exit code 77 as Skipped (SKIP_RETURN_CODE), so a run in which every
# test skipped, e.g. without PyTorch, does not look like a pass.
SKIP_EXIT = 77

if __name__ == "__main__":
    result = unittest.main(exit=False).result
    if not result.wasSuccessful():
        sys.exit(1)
    if result.testsRun > 0 and len(result.skipped) == result.testsRun:
        sys.exit(SKIP_EXIT)
