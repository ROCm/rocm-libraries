# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The shape corpus: what to compile, from the sources that decide it.

A kernel-side source answers "what is LEGAL?", never "what will anyone ASK for?", so
only a count against an external corpus shows an engine serving zero real workloads.
A mask spelling is never GUESSED, and provenance survives onto every shape.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_MINE = Path(__file__).resolve().parents[1] / "tools" / "mine_shapes.py"

_HEADER = (
    "shape_idx,model,variant,category,capability,dtype,mask,window_size,batch,"
    "seq_q,seq_kv,heads_q,heads_kv,head_dim,arch,ticket_group,priority\n"
)


def _row(**over) -> str:
    field = {
        "shape_idx": "0",
        "model": "Llama-3-8B",
        "variant": "v",
        "category": "prefill",
        "capability": "c",
        "dtype": "bf16",
        "mask": "causal",
        "window_size": "0",
        "batch": "1",
        "seq_q": "4096",
        "seq_kv": "4096",
        "heads_q": "32",
        "heads_kv": "8",
        "head_dim": "128",
        "arch": "gfx942",
        "ticket_group": "S1",
        "priority": "P0",
    }
    field.update({k: str(v) for k, v in over.items()})
    return ",".join(field[k] for k in _HEADER.strip().split(",")) + "\n"


def _mine(tmp_path: Path, csv_text: str, *extra) -> tuple[int, str, list]:
    csv_path = tmp_path / "published.csv"
    csv_path.write_text(csv_text)
    out = tmp_path / "shapes.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_MINE),
            "--published",
            str(csv_path),
            "--arch",
            "gfx942",
            "--out",
            str(out),
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    shapes = json.loads(out.read_text()) if out.exists() else []
    return result.returncode, result.stdout + result.stderr, shapes


class TestPublishedCsv:
    def test_mines_the_requested_arch_only(self, tmp_path):
        text = (
            _HEADER + _row() + _row(shape_idx=1, arch="gfx950", seq_q=512, seq_kv=512)
        )
        rc, _, shapes = _mine(tmp_path, text)
        assert rc == 0
        assert len(shapes) == 1

    def test_carries_priority_and_model_as_provenance(self, tmp_path):
        """`priority` comes from no other source, and the model is the axis a
        mixed-corpus result is split along."""
        rc, _, shapes = _mine(
            tmp_path, _HEADER + _row(priority="P0", model="Llama-3-8B")
        )
        assert rc == 0
        provenance = shapes[0]["_provenance"]
        assert provenance["priority"] == "P0"
        assert provenance["model"] == "Llama-3-8B"
        assert provenance["source"] == "published"

    def test_an_unknown_mask_spelling_is_refused_not_guessed(self, tmp_path):
        """Defaulting an unrecognised mask serves a windowed graph as plain causal: a
        wrong answer rather than a decline."""
        rc, output, _ = _mine(tmp_path, _HEADER + _row(mask="cheesecake"))
        assert rc != 0
        assert "unknown mask spelling" in output

    def test_an_unknown_dtype_spelling_is_refused_not_passed_through(self, tmp_path):
        """An unrecognised dtype builds the wrong binary and still validates, and the
        other two readers refuse it too."""
        rc, output, shapes = _mine(tmp_path, _HEADER + _row(dtype="fp8_e4m3"))
        assert rc != 0, f"bad dtype must be refused, not mined: {shapes}"
        assert "unknown dtype spelling" in output

    def test_every_dtype_spelling_normalises_the_same_as_the_other_readers(
        self, tmp_path
    ):
        for spelling, canonical in (
            ("bf16", "bf16"),
            ("bfloat16", "bf16"),
            ("torch.bfloat16", "bf16"),
            ("fp16", "fp16"),
            ("float16", "fp16"),
            ("half", "fp16"),
            ("torch.float16", "fp16"),
        ):
            rc, output, shapes = _mine(tmp_path, _HEADER + _row(dtype=spelling))
            assert rc == 0, f"{spelling!r} must mine cleanly: {output}"
            assert shapes[0]["dtype"] == canonical

    def test_an_absent_dtype_falls_back_to_bf16(self, tmp_path):
        """A row that does not say is a fallback, distinct from one saying something
        this table does not recognise."""
        rc, output, shapes = _mine(tmp_path, _HEADER + _row(dtype=""))
        assert rc == 0, output
        assert shapes[0]["dtype"] == "bf16"

    def test_windowed_rows_are_excluded_loudly_not_folded_onto_causal(self, tmp_path):
        """Folding `swin` onto `causal` collapses seven distinct shape keys."""
        text = (
            _HEADER
            + _row(mask="causal")
            + _row(shape_idx=1, mask="swin", seq_q=2048, seq_kv=2048, window_size=512)
        )
        rc, _, shapes = _mine(tmp_path, text)
        assert rc == 0
        assert len(shapes) == 1, "swin must not be mined by default"

        rc, _, with_windowed = _mine(tmp_path, text, "--include-windowed")
        assert rc == 0
        assert len(with_windowed) == 2
        masks = {s["mask_type"] for s in with_windowed}
        assert len(masks) == 2, "swin must keep its own mask_type, not become causal"

    def test_a_windowed_csv_row_carries_its_WIDTH_not_just_its_kind(self, tmp_path):
        """The published CSV states the width in its `window_size` column; a width
        arriving as 0 resolves to plain causal at the dispatcher."""
        rc, output, shapes = _mine(
            tmp_path, _HEADER + _row(mask="swin", window_size=512), "--include-windowed"
        )
        assert rc == 0, output
        assert len(shapes) == 1
        assert shapes[0]["mask_type"] == 3, "a windowed row is SLIDING_WINDOW"
        assert shapes[0]["sliding_window"] == 512, (
            "the window WIDTH must reach the request; a swin shape with width 0 is "
            "dispatched as plain causal"
        )
        assert shapes[0]["right_bound"] == 0, (
            "a request's right bound defaults to unbounded, which makes the same width "
            "a non-causal band"
        )

    def test_a_windowed_csv_row_without_a_width_is_refused(self, tmp_path):
        """Absent a width there is no windowed shape to mine, and defaulting one invents
        a shape nobody asked for."""
        rc, output, shapes = _mine(
            tmp_path, _HEADER + _row(mask="swin", window_size=0), "--include-windowed"
        )
        assert rc != 0, output
        assert not shapes, "a widthless windowed row must not reach the corpus"

    def test_identical_shapes_from_different_rows_merge_to_one_variant(self, tmp_path):
        """A corpus is a set of shapes: two rows asking for the same shape are one
        variant."""
        text = _HEADER + _row(shape_idx=0, model="A") + _row(shape_idx=1, model="B")
        rc, output, shapes = _mine(tmp_path, text)
        assert rc == 0
        assert len(shapes) == 1
        assert "1 duplicate shape(s) merged" in output

    def test_a_genuinely_different_shape_is_kept(self, tmp_path):
        text = (
            _HEADER
            + _row(seq_q=4096, seq_kv=4096)
            + _row(shape_idx=1, seq_q=8192, seq_kv=8192)
        )
        rc, _, shapes = _mine(tmp_path, text)
        assert rc == 0
        assert len(shapes) == 2

    def test_refuses_to_run_with_no_source(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(_MINE), "--out", str(tmp_path / "x.json")],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "at least one source" in (result.stdout + result.stderr)


class TestGraphCorpus:
    def _graph(self, path: Path, *, backward: bool = False) -> None:
        tensors = [
            {"name": "query", "dims": [1, 32, 4096, 128], "data_type": "bf16"},
            {"name": "key", "dims": [1, 8, 4096, 128], "data_type": "bf16"},
            {"name": "value", "dims": [1, 8, 4096, 128], "data_type": "bf16"},
        ]
        if backward:
            tensors.append({"name": "d_query", "dims": [1, 32, 4096, 128]})
        path.write_text(json.dumps({"tensors": tensors}))

    def test_backward_graphs_are_excluded_structurally(self, tmp_path):
        """A prefill kernel has no backward path, and one such graph routes to a
        third-party backward FMHA that takes the DEVICE down. Gradient tensors are the
        marker."""
        corpus = tmp_path / "graphs"
        corpus.mkdir()
        self._graph(corpus / "fwd_shape.json")
        # Deliberately named as though it were forward: filename must not decide.
        self._graph(corpus / "innocent_looking_name.json", backward=True)

        out = tmp_path / "shapes.json"
        result = subprocess.run(
            [sys.executable, str(_MINE), "--graphs", str(corpus), "--out", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        shapes = json.loads(out.read_text())
        assert len(shapes) == 1, "a gradient-carrying graph must not be mined"

    def test_suite_name_survives_as_provenance(self, tmp_path):
        corpus = tmp_path / "graphs" / "microbench_aiter"
        corpus.mkdir(parents=True)
        self._graph(corpus / "g.json")
        out = tmp_path / "shapes.json"
        subprocess.run(
            [
                sys.executable,
                str(_MINE),
                "--graphs",
                str(corpus.parent),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        shapes = json.loads(out.read_text())
        assert shapes[0]["_provenance"]["suite"] == "microbench_aiter", (
            "the suite is the axis a mixed-corpus result must be split along; a "
            "microbench/ path is a provenance label, not a synthetic-data warning"
        )


class TestCausalityComesFromTheGraphNotTheFilename:
    """`causal` decides which dispatcher branch resolves, so mining it wrong sizes a
    variant set that cannot serve the shapes it claims.

    hipDNN has no `causal` boolean: causality is (left_bound, right_bound,
    diagonal_alignment), and a set deprecated boolean only defaults an unbounded right
    bound to 0 (forcing BOTTOM_RIGHT for `causal_mask_bottom_right`), with the stored
    bounds and alignment applied on top. Every shipped causal bundle leaves both
    booleans false with `left_bound=-1, right_bound=0`. A filename heuristic is wrong
    for every causal graph here -- they carry `causal` in a PARENT DIRECTORY, never in
    the leaf name.
    """

    def _graph(self, path: Path, attrs: dict, seqlen_q: int = 4096) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "tensors": [
                        {
                            "name": "query",
                            "dims": [1, 32, seqlen_q, 128],
                            "data_type": "bf16",
                        },
                        {"name": "key", "dims": [1, 8, 4096, 128], "data_type": "bf16"},
                        {
                            "name": "value",
                            "dims": [1, 8, 4096, 128],
                            "data_type": "bf16",
                        },
                    ],
                    "nodes": [{"attributes": attrs}],
                }
            )
        )

    def _run(self, tmp_path: Path) -> tuple[subprocess.CompletedProcess, Path]:
        out = tmp_path / "shapes.json"
        result = subprocess.run(
            [
                sys.executable,
                str(_MINE),
                "--graphs",
                str(tmp_path / "graphs"),
                "--include-windowed",
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        return result, out

    def _mine(self, tmp_path: Path) -> list:
        result, out = self._run(tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(out.read_text())

    def _assert_refused(self, tmp_path: Path) -> None:
        result, out = self._run(tmp_path)
        assert result.returncode != 0
        assert not out.exists(), "a refused graph must not yield a mined corpus"

    def test_the_bound_form_is_read_as_causal_despite_a_neutral_filename(
        self, tmp_path
    ):
        """The shipped spelling: both booleans false, causality in the bounds."""
        self._graph(
            tmp_path / "graphs" / "hd128_causal_batch" / "Small" / "Small.json",
            {
                "causal_mask": False,
                "causal_mask_bottom_right": False,
                "left_bound": -1,
                "right_bound": 0,
                "diagonal_alignment": "BOTTOM_RIGHT",
            },
        )
        assert [s["mask_type"] for s in self._mine(tmp_path)] == [1]

    def test_no_bounds_at_all_is_not_causal_despite_a_causal_PATH(self, tmp_path):
        """A directory named `causal` containing an unmasked graph."""
        self._graph(
            tmp_path / "graphs" / "hd128_causal_batch" / "Small" / "Small.json",
            {
                "causal_mask": False,
                "causal_mask_bottom_right": False,
                "left_bound": None,
                "right_bound": None,
                "diagonal_alignment": "TOP_LEFT",
            },
        )
        assert [s["mask_type"] for s in self._mine(tmp_path)] == [0]

    def test_a_finite_left_bound_is_a_window_not_a_causal_variant(self, tmp_path):
        """Folding a window onto causal gets it SERVED as plain causal."""
        self._graph(
            tmp_path / "graphs" / "swa" / "g.json",
            {"causal_mask": False, "left_bound": 128, "right_bound": 0},
        )
        assert [s["mask_type"] for s in self._mine(tmp_path)] == [3]
        assert [s["right_bound"] for s in self._mine(tmp_path)] == [0]

    def test_the_deprecated_boolean_supplies_the_causal_right_bound(self, tmp_path):
        self._graph(
            tmp_path / "graphs" / "g.json",
            {"causal_mask": True, "left_bound": None, "right_bound": None},
        )
        assert [s["mask_type"] for s in self._mine(tmp_path)] == [1]

    def test_the_deprecated_boolean_keeps_a_stored_bottom_right_alignment(
        self, tmp_path
    ):
        """Chunked prefill (Sq != Skv) asks for a bottom-right diagonal; reading it as
        top-left causal mines a different request than the caller made, so the mask
        kind carries the anchor."""
        self._graph(
            tmp_path / "graphs" / "g.json",
            {"causal_mask": True, "diagonal_alignment": "BOTTOM_RIGHT"},
            seqlen_q=1024,
        )
        assert [s["mask_type"] for s in self._mine(tmp_path)] == [2]

    def test_the_deprecated_boolean_with_bottom_right_and_square_shape_is_causal(
        self, tmp_path
    ):
        self._graph(
            tmp_path / "graphs" / "g.json",
            {"causal_mask": True, "diagonal_alignment": "BOTTOM_RIGHT"},
        )
        assert [s["mask_type"] for s in self._mine(tmp_path)] == [1]

    def test_the_deprecated_boolean_keeps_a_stored_left_bound(self, tmp_path):
        self._graph(
            tmp_path / "graphs" / "g.json",
            {"causal_mask": True, "left_bound": 127, "right_bound": None},
        )
        shapes = self._mine(tmp_path)
        assert [
            (s["mask_type"], s["sliding_window"], s.get("right_bound")) for s in shapes
        ] == [(3, 128, 0)]

    def test_both_deprecated_booleans_set_is_refused(self, tmp_path):
        self._graph(
            tmp_path / "graphs" / "g.json",
            {"causal_mask": True, "causal_mask_bottom_right": True},
        )
        self._assert_refused(tmp_path)

    def test_causal_and_noncausal_shapes_do_not_collapse_onto_one_key(self, tmp_path):
        """Two graphs identical but for causality are TWO variants; merging them halves
        coverage."""
        self._graph(
            tmp_path / "graphs" / "a" / "g.json",
            {"causal_mask": False, "left_bound": -1, "right_bound": 0},
        )
        self._graph(
            tmp_path / "graphs" / "b" / "g.json",
            {"causal_mask": False, "left_bound": None, "right_bound": None},
        )
        assert sorted(s["mask_type"] for s in self._mine(tmp_path)) == [0, 1]

    def test_a_non_numeric_left_bound_is_refused_not_resolved_to_causal(self, tmp_path):
        """`left_bound` drives the branch below (`>= 0` -> window, else causal), so a
        non-numeric value falls through both comparisons and lands on causal."""
        self._graph(
            tmp_path / "graphs" / "g.json",
            {"causal_mask": False, "left_bound": "unbounded", "right_bound": 0},
        )
        self._assert_refused(tmp_path)


class TestRocKeBenchTree:
    """The third source: rocKE's own benchmark tree, the only one saying what the kernel
    team measures on an arch with no published results CSV. The files are JSONL, so
    `json.load` raises "Extra data"."""

    def _trace(self, path: Path, *records) -> None:
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def _record(self, **over) -> dict:
        base = {
            "ALL_DECODE": False,
            "kind": "2d",
            "num_seqs": 1,
            "num_query_heads": 64,
            "num_kv_heads": 8,
            "head_size": 64,
            "max_seqlen_q": 4096,
            "max_seqlen_k": 4096,
            "q_dtype": "torch.bfloat16",
            "window_size": [-1, -1],
            "has_sinks": False,
        }
        base.update(over)
        return base

    def _mine_bench(self, tmp_path: Path, *records):
        tree = tmp_path / "bench"
        tree.mkdir(exist_ok=True)
        self._trace(tree / "prefill_shapes.json", *records)
        out = tmp_path / "shapes.json"
        result = subprocess.run(
            [sys.executable, str(_MINE), "--rocke-bench", str(tree), "--out", str(out)],
            capture_output=True,
            text=True,
        )
        shapes = json.loads(out.read_text()) if out.exists() else []
        return result, shapes

    def test_an_explicit_noncausal_record_is_mined_unmasked(self, tmp_path):
        """`[-1, -1]` alone reads as causal; an explicit `causal: false` (the
        emitted benchmark lists carry one) is a full request, not its causal twin."""
        result, shapes = self._mine_bench(
            tmp_path, self._record(causal=False), self._record(causal=True)
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert sorted(s["mask_type"] for s in shapes) == [0, 1]

    def test_a_noncausal_record_with_a_window_is_refused(self, tmp_path):
        result, _ = self._mine_bench(
            tmp_path, self._record(causal=False, window_size=[127, 0])
        )
        assert result.returncode != 0
        assert "non-causal record cannot carry window" in result.stdout + result.stderr

    def test_a_varlen_record_is_skipped_and_counted(self, tmp_path):
        result, shapes = self._mine_bench(
            tmp_path, self._record(varlen=True, num_seqs=4), self._record()
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(shapes) == 1 and shapes[0]["batch"] == 1
        assert "1 rocKE varlen record(s) skipped" in result.stdout

    def test_a_jsonl_trace_is_read_at_all(self, tmp_path):
        """A json.load reader gets 'Extra data' and silently mines nothing."""
        result, shapes = self._mine_bench(
            tmp_path, self._record(), self._record(max_seqlen_q=8192, max_seqlen_k=8192)
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(shapes) == 2

    def test_an_unwindowed_prefill_trace_is_causal(self, tmp_path):
        """`[-1, -1]` is unbounded both ways, which for a prefill suite is full
        causal."""
        _, shapes = self._mine_bench(tmp_path, self._record(window_size=[-1, -1]))
        assert shapes[0]["mask_type"] == 1
        assert shapes[0]["sliding_window"] == 0

    def test_a_windowed_trace_carries_its_WIDTH_not_just_its_kind(self, tmp_path):
        """Dropping the width sends sliding_window=0 to the dispatcher, which computes a
        full causal triangle for a banded request instead of declining."""
        _, shapes = self._mine_bench(tmp_path, self._record(window_size=[127, 0]))
        assert shapes[0]["mask_type"] == 3, "a finite left bound is a window"
        assert shapes[0]["sliding_window"] == 128, (
            "the window WIDTH must reach the request; 127 is the left bound and the "
            "band includes the current token"
        )
        assert shapes[0]["right_bound"] == 0

    def test_sinks_are_carried_rather_than_filtered(self, tmp_path):
        """Shipping a sink variant is a scope decision made downstream; filtering the
        shape out here hides it from the reconciler."""
        _, shapes = self._mine_bench(tmp_path, self._record(has_sinks=True))
        assert shapes[0]["use_sinks"] is True
        assert shapes[0]["_provenance"]["has_sinks"] is True

    def test_a_trace_with_no_recorded_causality_is_skipped_not_defaulted(
        self, tmp_path
    ):
        """No record in rocKE's traces carries a causal/mask key, so defaulting it picks
        which branch the dispatcher resolves and which kernels get built."""
        result, shapes = self._mine_bench(
            tmp_path, self._record(window_size=None), self._record()
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(shapes) == 1, "the unknown-mask record must not be mined"
        assert "no recorded causality" in result.stdout

    def test_an_unknown_dtype_spelling_is_refused(self, tmp_path):
        """Three vocabularies meet here and none agree; a guessed dtype builds a
        different binary and still validates."""
        result, _ = self._mine_bench(
            tmp_path, self._record(q_dtype="torch.float8_e4m3")
        )
        assert result.returncode != 0
        assert "unknown dtype spelling" in result.stderr

    def test_decode_records_are_excluded(self, tmp_path):
        """ALL_DECODE marks a decode trace; a prefill kernel does not serve it."""
        _, shapes = self._mine_bench(
            tmp_path, self._record(ALL_DECODE=True), self._record()
        )
        assert len(shapes) == 1


class TestGradientSpellingsAreBothExcluded:
    """`sample_sdpa_backward` spells its gradients `dq`/`dk`/`dv`/`do`, not `d_query`,
    and a marker set matching one spelling mines a backward graph as forward."""

    @pytest.mark.parametrize("gradient", ["d_query", "dq", "dk", "dv", "do"])
    def test_either_gradient_spelling_excludes_the_graph(self, tmp_path, gradient):
        corpus = tmp_path / "graphs"
        corpus.mkdir()
        tensors = [
            {"name": "q", "dims": [1, 32, 4096, 128], "data_type": "bf16"},
            {"name": "k", "dims": [1, 8, 4096, 128], "data_type": "bf16"},
            {"name": "v", "dims": [1, 8, 4096, 128], "data_type": "bf16"},
            {"name": gradient, "dims": [1, 32, 4096, 128], "data_type": "bf16"},
        ]
        (corpus / "innocent.json").write_text(json.dumps({"tensors": tensors}))
        out = tmp_path / "shapes.json"
        result = subprocess.run(
            [sys.executable, str(_MINE), "--graphs", str(corpus), "--out", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, "no forward shapes should have been mined"
        assert "no shapes mined" in result.stderr

    def test_the_node_type_alone_excludes_a_backward_graph(self, tmp_path):
        """The graph DECLARES what it is, so the op type is the primary marker and
        tensor names the fallback."""
        corpus = tmp_path / "graphs"
        corpus.mkdir()
        (corpus / "g.json").write_text(
            json.dumps(
                {
                    "tensors": [
                        {"name": "q", "dims": [1, 32, 4096, 128], "data_type": "bf16"},
                        {"name": "k", "dims": [1, 8, 4096, 128], "data_type": "bf16"},
                        {"name": "v", "dims": [1, 8, 4096, 128], "data_type": "bf16"},
                    ],
                    "nodes": [{"type": "SdpaBackwardAttributes"}],
                }
            )
        )
        out = tmp_path / "shapes.json"
        result = subprocess.run(
            [sys.executable, str(_MINE), "--graphs", str(corpus), "--out", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, "a declared backward graph must not be mined"

    def test_a_forward_graph_with_similar_names_is_still_mined(self, tmp_path):
        """Control: the exclusion must not fire on a forward graph whose tensors merely
        start with d (`descale_q`), or the corpus empties silently."""
        corpus = tmp_path / "graphs"
        corpus.mkdir()
        (corpus / "g.json").write_text(
            json.dumps(
                {
                    "tensors": [
                        {"name": "q", "dims": [1, 32, 4096, 128], "data_type": "bf16"},
                        {"name": "k", "dims": [1, 8, 4096, 128], "data_type": "bf16"},
                        {"name": "v", "dims": [1, 8, 4096, 128], "data_type": "bf16"},
                        {"name": "descale_q", "dims": [1]},
                    ],
                    "nodes": [{"type": "SdpaAttributes"}],
                }
            )
        )
        out = tmp_path / "shapes.json"
        result = subprocess.run(
            [sys.executable, str(_MINE), "--graphs", str(corpus), "--out", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(json.loads(out.read_text())) == 1


class TestEveryRequestSemanticSurvivesMining:
    """A field dropped during mining sends the dispatcher a request the caller did not
    make; a field dropped from the corpus IDENTITY merges two requests into one variant,
    so only one is ever compiled."""

    @staticmethod
    def _graph(
        path: Path, *, attrs: dict, v_dims=(1, 8, 4096, 128), sink: bool = False
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = [
            {
                "uid": 1,
                "name": "query",
                "dims": [1, 32, 4096, 128],
                "data_type": "bf16",
            },
            {"uid": 2, "name": "key", "dims": [1, 8, 4096, 128], "data_type": "bf16"},
            {"uid": 3, "name": "value", "dims": list(v_dims), "data_type": "bf16"},
        ]
        node_attrs = {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3, **attrs}
        if sink:
            tensors.append(
                {"uid": 4, "name": "sink", "dims": [32], "data_type": "bf16"}
            )
            node_attrs["sink_token_tensor_uid"] = 4
        path.write_text(
            json.dumps(
                {
                    "name": path.stem,
                    "tensors": tensors,
                    "nodes": [{"type": "SdpaAttributes", "attributes": node_attrs}],
                }
            )
        )

    @staticmethod
    def _mine(tmp_path: Path) -> tuple[int, str, list]:
        out = tmp_path / "shapes.json"
        result = subprocess.run(
            [
                sys.executable,
                str(_MINE),
                "--graphs",
                str(tmp_path / "graphs"),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        shapes = json.loads(out.read_text()) if out.exists() else []
        return result.returncode, result.stdout + result.stderr, shapes

    _UNMASKED = {"causal_mask": False, "left_bound": None, "right_bound": None}
    _CAUSAL = {"causal_mask": False, "left_bound": -1, "right_bound": 0}
    _WINDOW = {"causal_mask": False, "left_bound": 127, "right_bound": 0}

    def test_an_asymmetric_v_head_dimension_is_carried_not_copied_from_q(
        self, tmp_path
    ):
        """Q and V head dimensions are INDEPENDENT: copying Q's onto V builds a variant
        with the wrong output width and leaves the asymmetric shape with none."""
        self._graph(
            tmp_path / "graphs" / "g.json",
            attrs=self._CAUSAL,
            v_dims=(1, 8, 4096, 64),
        )
        rc, output, shapes = self._mine(tmp_path)
        assert rc == 0, output
        assert shapes[0]["hdim_q"] == 128
        assert shapes[0]["hdim_v"] == 64, "V's head dimension was taken from Q"

    def test_a_v_tensor_disagreeing_on_batch_or_heads_is_refused(self, tmp_path):
        """V shares batch, heads and key length with K; a graph where it does not is not
        one attention request."""
        self._graph(
            tmp_path / "graphs" / "g.json",
            attrs=self._CAUSAL,
            v_dims=(1, 4, 4096, 128),
        )
        rc, output, _ = self._mine(tmp_path)
        assert rc != 0
        assert "incompatible independent Q/K/V dimensions" in output

    def test_a_missing_v_tensor_is_refused_rather_than_defaulted(self, tmp_path):
        path = tmp_path / "graphs" / "g.json"
        self._graph(path, attrs=self._CAUSAL)
        document = json.loads(path.read_text())
        document["tensors"] = [t for t in document["tensors"] if t["uid"] != 3]
        path.write_text(json.dumps(document))
        rc, output, _ = self._mine(tmp_path)
        assert rc != 0
        assert "independent Q, K and V" in output

    def test_unmasked_causal_and_windowed_requests_are_three_distinct_shapes(
        self, tmp_path
    ):
        """Identical in every dimension, different in what they mask: collapsing any
        pair sizes a variant set that cannot serve the other."""
        self._graph(tmp_path / "graphs" / "none.json", attrs=self._UNMASKED)
        self._graph(tmp_path / "graphs" / "causal.json", attrs=self._CAUSAL)
        self._graph(tmp_path / "graphs" / "window.json", attrs=self._WINDOW)
        rc, output, shapes = self._mine(tmp_path)
        assert rc == 0, output
        assert len(shapes) == 3, "mask semantics collapsed distinct requests"
        assert {(s["mask_type"], s["sliding_window"]) for s in shapes} == {
            (0, 0),
            (1, 0),
            (3, 128),
        }

    def test_two_windows_of_different_width_do_not_merge(self, tmp_path):
        """The mask KIND alone does not encode the window, so both reach the dispatcher
        as the same request."""
        self._graph(
            tmp_path / "graphs" / "w64.json",
            attrs={"causal_mask": False, "left_bound": 63, "right_bound": 0},
        )
        self._graph(
            tmp_path / "graphs" / "w128.json",
            attrs={"causal_mask": False, "left_bound": 127, "right_bound": 0},
        )
        rc, output, shapes = self._mine(tmp_path)
        assert rc == 0, output
        assert sorted(s["sliding_window"] for s in shapes) == [64, 128]

    def test_a_sink_request_does_not_merge_with_its_sinkless_twin(self, tmp_path):
        """Sinks are a recorded request attribute, not a tuning choice."""
        self._graph(tmp_path / "graphs" / "plain.json", attrs=self._CAUSAL)
        self._graph(tmp_path / "graphs" / "sinks.json", attrs=self._CAUSAL, sink=True)
        rc, output, shapes = self._mine(tmp_path)
        assert rc == 0, output
        assert len(shapes) == 2
        assert {s["use_sinks"] for s in shapes} == {False, True}

    def test_a_sink_uid_naming_no_tensor_is_refused(self, tmp_path):
        path = tmp_path / "graphs" / "g.json"
        self._graph(path, attrs=self._CAUSAL, sink=True)
        document = json.loads(path.read_text())
        document["tensors"] = [t for t in document["tensors"] if t["uid"] != 4]
        path.write_text(json.dumps(document))
        rc, output, _ = self._mine(tmp_path)
        assert rc != 0
        assert "sink_token_tensor_uid" in output

    def test_provenance_alone_never_makes_two_shapes_distinct(self, tmp_path):
        """Identity is computed from request fields, not the whole record: the same
        request from two suites is ONE variant and two votes for it."""
        self._graph(tmp_path / "graphs" / "suite_a" / "g.json", attrs=self._CAUSAL)
        self._graph(tmp_path / "graphs" / "suite_b" / "g.json", attrs=self._CAUSAL)
        rc, output, shapes = self._mine(tmp_path)
        assert rc == 0, output
        assert len(shapes) == 1, "provenance leaked into the shape identity"
        assert "1 duplicate shape(s) merged" in output
        suites = {p["suite"] for p in shapes[0]["_provenance_occurrences"]}
        assert suites == {"suite_a", "suite_b"}, "a merged duplicate lost its vote"


def _hipdnn_graph(
    path: Path,
    *,
    dtype: str = "bfloat16",
    inputs: dict | None = None,
    attrs: dict | None = None,
) -> None:
    """A graph in the shape dnn-benchmarking ships: operand UIDs under the node's
    `inputs`/`outputs`, every tensor with strides, and the full attribute set."""
    b, s = 1, 512

    def tensor(uid, name, heads):
        dims = [b, heads, s, 128]
        strides = [s * heads * 128, 128, heads * 128, 1]
        return {
            "uid": uid,
            "name": name,
            "dims": dims,
            "strides": strides,
            "data_type": dtype,
        }

    tensors = [
        tensor(1, "Q", 8),
        tensor(2, "K", 2),
        tensor(3, "V", 2),
        tensor(4, "O", 8),
    ]
    node_inputs = {
        "q_tensor_uid": 1,
        "k_tensor_uid": 2,
        "v_tensor_uid": 3,
        "sink_token_tensor_uid": None,
        "seq_len_q_tensor_uid": None,
        "seq_len_kv_tensor_uid": None,
    }
    for key, uid in (inputs or {}).items():
        node_inputs[key] = uid
        tensors.append({"uid": uid, "name": key, "dims": [1], "strides": [1]})
    node_attrs = {
        "causal_mask": False,
        "left_bound": -1,
        "right_bound": 0,
        "diagonal_alignment": "TOP_LEFT",
        "dropout_probability": None,
        "max_seq_len_kv": None,
        "generate_stats": None,
        "attn_scale_value": 0.088,
        **(attrs or {}),
    }
    for key, uid in (attrs or {}).items():
        if key.endswith("_tensor_uid") and uid is not None:
            tensors.append({"uid": uid, "name": key, "dims": [1], "strides": [1]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": path.stem,
                "tensors": tensors,
                "nodes": [
                    {
                        "type": "SdpaAttributes",
                        "inputs": node_inputs,
                        "outputs": {"o_tensor_uid": 4},
                        "attributes": node_attrs,
                    }
                ],
            }
        )
    )


def _mine_graphs(tmp_path: Path) -> tuple[int, str, list, list]:
    out = tmp_path / "shapes.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_MINE),
            "--graphs",
            str(tmp_path / "graphs"),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    shapes = json.loads(out.read_text()) if out.exists() else []
    excluded_path = tmp_path / "shapes.excluded.json"
    excluded = json.loads(excluded_path.read_text()) if excluded_path.exists() else []
    return result.returncode, result.stdout + result.stderr, shapes, excluded


class TestOutOfFamilyDtypesAreExcludedNotFatal:
    """dnn-benchmarking corpora routinely carry fp32 SDPA graphs. One of them must
    not abort the whole source; it is recorded as excluded with its reason."""

    def test_an_fp32_graph_is_excluded_and_the_rest_is_mined(self, tmp_path):
        _hipdnn_graph(tmp_path / "graphs" / "bf16.json")
        _hipdnn_graph(tmp_path / "graphs" / "fp32.json", dtype="float")
        rc, output, shapes, excluded = _mine_graphs(tmp_path)
        assert rc == 0, output
        assert len(shapes) == 1 and shapes[0]["dtype"] == "bf16"
        assert len(excluded) == 1, excluded
        assert excluded[0]["path"].endswith("fp32.json")
        assert "fp32" in excluded[0]["reason"]
        assert "excluded      :     1" in output

    def test_a_source_of_only_excluded_graphs_is_an_empty_corpus_not_a_failure(
        self, tmp_path
    ):
        _hipdnn_graph(tmp_path / "graphs" / "fp32.json", dtype="float32")
        rc, output, shapes, excluded = _mine_graphs(tmp_path)
        assert rc == 0, output
        assert shapes == [] and len(excluded) == 1

    def test_an_unrecognised_dtype_is_still_refused(self, tmp_path):
        """Exclusion is for recognised dtypes only; a spelling nobody mapped is
        still malformed input."""
        _hipdnn_graph(tmp_path / "graphs" / "g.json", dtype="cheesecake")
        rc, output, _, _ = _mine_graphs(tmp_path)
        assert rc != 0
        assert "unknown dtype spelling" in output


class TestOperandsBoundInNodeInputs:
    """hipDNN graphs bind operand UIDs under the SDPA node's `inputs`. A sink bound
    there must be mined as a sink request, not as the easier sinkless problem."""

    def test_a_sink_bound_in_node_inputs_is_mined_as_use_sinks(self, tmp_path):
        _hipdnn_graph(
            tmp_path / "graphs" / "sink.json", inputs={"sink_token_tensor_uid": 9}
        )
        rc, output, shapes, _ = _mine_graphs(tmp_path)
        assert rc == 0, output
        assert shapes[0]["use_sinks"] is True

    def test_a_null_input_slot_does_not_hide_an_attribute_bound_sink(self, tmp_path):
        """The helper's `inputs` carries `sink_token_tensor_uid: null`, as real graphs
        do for an unbound slot. The legacy attribute binding must still count."""
        _hipdnn_graph(
            tmp_path / "graphs" / "sink.json", attrs={"sink_token_tensor_uid": 9}
        )
        rc, output, shapes, _ = _mine_graphs(tmp_path)
        assert rc == 0, output
        assert shapes[0]["use_sinks"] is True


def _emitted(variant: str, seqlens: list[int], hq: int, hkv: int, **over) -> dict:
    """One record as the dense prefill benchmarks' `--emit-shapes` writes it: the
    rocKE trace schema plus an explicit `causal`, and `varlen`/`seqlens` for a packed
    ragged batch. Synthetic, so the tests do not follow the benchmarks' shape lists."""
    record = {
        "model": "benchmark_dense_prefill_live",
        "variant": variant,
        "label": variant,
        "num_seqs": len(seqlens),
        "max_seqlen_q": max(seqlens),
        "max_seqlen_k": max(seqlens),
        "num_query_heads": hq,
        "num_kv_heads": hkv,
        "head_size": 128,
        "q_dtype": "bf16",
        "causal": True,
        "window_size": [-1, 0],
        "has_sinks": False,
    }
    record.update(over)
    return record


class TestAnEmittedBenchmarkShapeListIsMinable:
    """The dense prefill benchmarks write their shape lists with `--emit-shapes` in
    the trace schema this miner reads. A whole emitted list, mined through the CLI,
    keeps the full, windowed and packed rows apart from the causal ones."""

    def _mine(self, tmp_path: Path, *records: dict):
        tree = tmp_path / "bench"
        tree.mkdir()
        (tree / "dense_prefill_live_shapes.json").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
        out = tmp_path / "shapes.json"
        mined = subprocess.run(
            [sys.executable, str(_MINE), "--rocke-bench", str(tree), "--out", str(out)],
            capture_output=True,
            text=True,
        )
        assert mined.returncode == 0, mined.stdout + mined.stderr
        return mined.stdout, json.loads(out.read_text())

    def test_full_and_windowed_rows_stay_distinct_and_repeats_merge(self, tmp_path):
        output, shapes = self._mine(
            tmp_path,
            _emitted("causal/mha", [1024], 8, 8),
            _emitted("causal/gqa", [1024], 8, 2),
            _emitted("full/mha", [1024], 8, 8, causal=False, window_size=[-1, -1]),
            _emitted("swa/w512", [1024], 8, 8, window_size=[511, 0]),
            # The persistent mode re-measures a causal shape on another grid.
            _emitted("persistent/mha", [1024], 8, 8),
        )
        assert len(shapes) == 4, output
        assert "1 duplicate shape(s) merged" in output
        by_mask = sorted((s["mask_type"], s["sliding_window"]) for s in shapes)
        assert by_mask == [(0, 0), (1, 0), (1, 0), (3, 512)], by_mask
        full = next(s for s in shapes if s["mask_type"] == 0)
        assert (full["nhead_q"], full["nhead_k"]) == (8, 8)

    def test_packed_varlen_rows_are_skipped_and_counted(self, tmp_path):
        output, shapes = self._mine(
            tmp_path,
            _emitted("dense/s512", [512], 8, 8),
            _emitted("dense/s2048", [2048], 8, 8),
            _emitted("varlen/4seq", [512, 1024, 300, 2048], 8, 8, varlen=True),
        )
        assert sorted(s["seqlen_q"] for s in shapes) == [512, 2048], output
        assert all(s["batch"] == 1 for s in shapes)
        assert "1 rocKE varlen record(s) skipped" in output


class TestShapeDirectory:
    """`--shape-dir` reads graph JSON and every tabular form from one directory."""

    def _mine(self, tmp_path: Path, *extra, arch="gfx942") -> tuple[int, str, list]:
        out = tmp_path / "shapes.json"
        result = subprocess.run(
            [
                sys.executable,
                str(_MINE),
                "--shape-dir",
                str(tmp_path / "pub"),
                "--arch",
                arch,
                "--out",
                str(out),
                *extra,
            ],
            capture_output=True,
            text=True,
        )
        shapes = json.loads(out.read_text()) if out.exists() else []
        return result.returncode, result.stdout + result.stderr, shapes

    @pytest.fixture
    def published(self, tmp_path) -> Path:
        directory = tmp_path / "pub"
        directory.mkdir()
        return directory

    def test_reads_csv_json_and_key_value_lines_from_one_flag(
        self, published, tmp_path
    ):
        (published / "a.csv").write_text(
            "batch,num_query_heads,num_kv_heads,seqlen_q,seqlen_k,head_size,dtype\n"
            "1,32,8,4096,4096,128,bf16\n"
        )
        (published / "b.json").write_text(
            json.dumps(
                [
                    {
                        "b": 2,
                        "hq": 16,
                        "hkv": 16,
                        "sq": 512,
                        "skv": 512,
                        "d": 64,
                        "data_type": "fp16",
                    }
                ]
            )
        )
        (published / "c.txt").write_text(
            "# a hand-maintained list\n"
            "batch=4 heads=8 seqlen_q=1 seqlen_kv=2048 head_dim=128 dtype=bf16\n"
            "\n"
        )
        rc, log, shapes = self._mine(tmp_path)
        assert rc == 0, log
        assert sorted(s["batch"] for s in shapes) == [1, 2, 4]
        assert all(s["_provenance"]["source"] == "shape_dir" for s in shapes)

    def test_a_row_missing_a_required_field_is_not_a_shape_row(
        self, published, tmp_path
    ):
        """Non-shape records (README tables, index files) are skipped."""
        (published / "index.json").write_text(
            json.dumps([{"model": "Llama-3-8B", "notes": "see ticket"}])
        )
        (published / "real.csv").write_text(
            "batch,heads_q,seqlen_q,seqlen_kv,head_dim\n1,32,4096,4096,128\n"
        )
        rc, log, shapes = self._mine(tmp_path)
        assert rc == 0, log
        assert len(shapes) == 1

    def test_a_narrower_v_head_and_a_window_are_carried_not_guessed(
        self, published, tmp_path
    ):
        """An MLA row keeps its 192/128 heads, and a windowed row its width."""
        (published / "mla.csv").write_text(
            "batch,heads_q,seqlen_q,seqlen_kv,head_dim,hdim_v,mask,window_size\n"
            "1,128,4096,4096,192,128,causal,\n"
            "2,32,4096,4096,128,,swin,1024\n"
        )
        rc, log, shapes = self._mine(tmp_path, "--include-windowed")
        assert rc == 0, log
        by_batch = {s["batch"]: s for s in shapes}
        assert (by_batch[1]["hdim_q"], by_batch[1]["hdim_v"]) == (192, 128)
        assert by_batch[2]["sliding_window"] == 1024
        assert by_batch[2]["right_bound"] == 0
        assert "right_bound" not in by_batch[1]

    def test_a_windowed_row_without_a_width_is_refused(self, published, tmp_path):
        (published / "swin.csv").write_text(
            "batch,heads_q,seqlen_q,seqlen_kv,head_dim,mask\n1,32,4096,4096,128,swin\n"
        )
        rc, log, _ = self._mine(tmp_path, "--include-windowed")
        assert rc != 0 and "positive window_size" in log

    def test_a_windowed_row_needs_include_windowed_like_the_published_csv(
        self, published, tmp_path
    ):
        """Without the flag a windowed row is excluded loudly -- recorded with its
        reason, never folded onto causal -- and the causal rows are unaffected."""
        (published / "mix.csv").write_text(
            "batch,heads_q,seqlen_q,seqlen_kv,head_dim,mask,window_size\n"
            "1,32,4096,4096,128,causal,\n"
            "2,32,4096,4096,128,swin,1024\n"
            "3,32,4096,4096,128,swin,\n"
        )
        rc, log, shapes = self._mine(tmp_path)
        assert rc == 0, log
        assert [s["batch"] for s in shapes] == [1], "windowed rows must be dropped"
        excluded = json.loads((tmp_path / "shapes.excluded.json").read_text())
        assert [record["row"] for record in excluded] == [1, 2]
        assert all("--include-windowed" in record["reason"] for record in excluded)

        rc, log, shapes = self._mine(tmp_path, "--include-windowed")
        assert (
            rc != 0 and "positive window_size" in log
        ), "with the flag a widthless windowed row is still refused"

    def test_a_graph_document_is_left_to_the_graph_reader(self, published, tmp_path):
        """A graph is mined once, as a graph, not again as a record."""
        (published / "g.json").write_text(
            json.dumps(
                {
                    "tensors": [
                        {
                            "uid": 1,
                            "name": "q",
                            "dims": [1, 32, 4096, 128],
                            "data_type": "bf16",
                        },
                        {
                            "uid": 2,
                            "name": "k",
                            "dims": [1, 8, 4096, 128],
                            "data_type": "bf16",
                        },
                        {
                            "uid": 3,
                            "name": "v",
                            "dims": [1, 8, 4096, 128],
                            "data_type": "bf16",
                        },
                    ],
                    "nodes": [
                        {
                            "type": "SdpaAttributes",
                            "attributes": {
                                "q_tensor_uid": 1,
                                "k_tensor_uid": 2,
                                "v_tensor_uid": 3,
                            },
                        }
                    ],
                }
            )
        )
        rc, log, shapes = self._mine(tmp_path)
        assert rc == 0, log
        assert [s["_provenance"]["source"] for s in shapes] == ["graphs"]

    def test_a_row_naming_another_arch_is_dropped_but_an_unnamed_one_is_kept(
        self, published, tmp_path
    ):
        """Rows naming no arch are kept: most publishers record no target."""
        (published / "a.csv").write_text(
            "batch,heads_q,seqlen_q,seqlen_kv,head_dim,arch\n"
            "1,32,4096,4096,128,gfx942\n"
            "2,32,512,512,128,gfx950\n"
            "4,32,128,128,128,\n"
        )
        rc, log, shapes = self._mine(tmp_path)
        assert rc == 0, log
        assert sorted(s["batch"] for s in shapes) == [1, 4]
        assert "named another arch" in log

    def test_an_unknown_mask_spelling_is_refused_rather_than_guessed(
        self, published, tmp_path
    ):
        """`bottom_right` differs from top-left causal, so it is not mapped."""
        (published / "a.csv").write_text(
            "batch,heads_q,seqlen_q,seqlen_kv,head_dim,mask\n"
            "1,32,4096,4096,128,bottom_right\n"
        )
        rc, log, _ = self._mine(tmp_path)
        assert rc != 0
        assert "unknown mask spelling" in log

    def test_a_boolean_causal_column_is_read_as_a_mask(self, published, tmp_path):
        (published / "a.csv").write_text(
            "batch,heads_q,seqlen_q,seqlen_kv,head_dim,is_causal\n"
            "1,32,4096,4096,128,true\n"
            "2,32,4096,4096,128,false\n"
        )
        rc, log, shapes = self._mine(tmp_path)
        assert rc == 0, log
        assert sorted(s["mask_type"] for s in shapes) == [0, 1]


class TestQueryCsv:
    """`--out-query-csv`, the handoff to `hipdnn_corpus_gen --model-shapes`."""

    def _mine(self, tmp_path: Path, rows: str) -> tuple[int, str, list[str]]:
        published = tmp_path / "pub"
        published.mkdir()
        (published / "a.csv").write_text(rows)
        query = tmp_path / "model-shapes.csv"
        result = subprocess.run(
            [
                sys.executable,
                str(_MINE),
                "--shape-dir",
                str(published),
                "--arch",
                "gfx942",
                "--include-windowed",
                "--out-query-csv",
                str(query),
            ],
            capture_output=True,
            text=True,
        )
        lines = query.read_text().splitlines() if query.exists() else []
        return result.returncode, result.stdout + result.stderr, lines

    def test_writes_the_q_columns_the_corpus_tool_reads(self, tmp_path):
        rc, log, lines = self._mine(
            tmp_path,
            "batch,heads_q,heads_kv,seqlen_q,seqlen_kv,head_dim,dtype,mask\n"
            "1,32,8,4096,4096,128,bf16,causal\n",
        )
        assert rc == 0, log
        assert lines[0] == (
            "name,op,q.batch,q.heads,q.heads_kv,q.seqlen_q,"
            "q.seqlen_k,q.head_dim,q.is_causal,q.alignment,q.generate_stats,"
            "q.dtype"
        )
        assert lines[1].split(",")[1:] == [
            "sdpa_fwd",
            "1",
            "32",
            "8",
            "4096",
            "4096",
            "128",
            "true",
            "top_left",
            "false",
            "bf16",
        ]

    def test_a_tabular_row_is_written_top_left(self, tmp_path):
        """Tabular sources cannot say bottom-right; the column is still written."""
        rc, log, lines = self._mine(
            tmp_path, "batch,heads_q,seqlen_q,seqlen_kv,head_dim\n1,32,4096,4096,128\n"
        )
        assert rc == 0, log
        assert lines[1].split(",")[-3] == "top_left"

    def test_generate_stats_is_always_false(self, tmp_path):
        """Sources record inference forwards only."""
        rc, log, lines = self._mine(
            tmp_path, "batch,heads_q,seqlen_q,seqlen_kv,head_dim\n1,32,4096,4096,128\n"
        )
        assert rc == 0, log
        assert lines[1].split(",")[-2] == "false"

    def test_a_windowed_shape_is_dropped_by_name_and_counted(self, tmp_path):
        """`sdpa_fwd` declares no window parameter."""
        rc, log, lines = self._mine(
            tmp_path,
            "batch,heads_q,seqlen_q,seqlen_kv,head_dim,mask,window_size\n"
            "1,32,4096,4096,128,swin,1024\n"
            "2,32,4096,4096,128,causal,\n",
        )
        assert rc == 0, log
        assert len(lines) == 2
        assert "dropped     1" in log and "windowed mask" in log

    def test_dropping_every_shape_fails_rather_than_writing_an_empty_pool(
        self, tmp_path
    ):
        rc, log, lines = self._mine(
            tmp_path,
            "batch,heads_q,seqlen_q,seqlen_kv,head_dim,mask,window_size\n"
            "1,32,4096,4096,128,swin,1024\n",
        )
        assert rc != 0
        assert len(lines) == 1
        assert "the model pool would be empty" in log

    def test_a_graph_keeps_its_bottom_right_anchor(self, tmp_path):
        """Where Sq != Sk the two anchors compute different outputs."""
        graphs = tmp_path / "graphs"
        graphs.mkdir()
        (graphs / "g.json").write_text(
            json.dumps(
                {
                    "name": "g",
                    "tensors": [
                        {
                            "uid": 1,
                            "name": "q",
                            "dims": [1, 32, 512, 128],
                            "data_type": "bf16",
                        },
                        {
                            "uid": 2,
                            "name": "k",
                            "dims": [1, 8, 4096, 128],
                            "data_type": "bf16",
                        },
                        {
                            "uid": 3,
                            "name": "v",
                            "dims": [1, 8, 4096, 128],
                            "data_type": "bf16",
                        },
                    ],
                    "nodes": [
                        {
                            "type": "SdpaAttributes",
                            "attributes": {
                                "q_tensor_uid": 1,
                                "k_tensor_uid": 2,
                                "v_tensor_uid": 3,
                                "causal_mask_bottom_right": True,
                            },
                        }
                    ],
                }
            )
        )
        query = tmp_path / "model-shapes.csv"
        result = subprocess.run(
            [
                sys.executable,
                str(_MINE),
                "--graphs",
                str(graphs),
                "--out-query-csv",
                str(query),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        header, row = query.read_text().splitlines()
        record = dict(zip(header.split(","), row.split(",")))
        assert record["q.is_causal"] == "true"
        assert record["q.alignment"] == "bottom_right"


def _sdpa_graph(path: Path, seqlen_q: int, seqlen_k: int, attrs: dict) -> None:
    """One forward SDPA graph, B1 Hq32 Hkv8 D128 bf16, with the given mask attributes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = [
        {"uid": uid, "name": name, "dims": [1, heads, seqlen, 128], "data_type": "bf16"}
        for uid, name, heads, seqlen in (
            (1, "q", 32, seqlen_q),
            (2, "k", 8, seqlen_k),
            (3, "v", 8, seqlen_k),
        )
    ]
    node = {
        "type": "SdpaAttributes",
        "attributes": {
            "q_tensor_uid": 1,
            "k_tensor_uid": 2,
            "v_tensor_uid": 3,
            **attrs,
        },
    }
    path.write_text(
        json.dumps({"name": path.stem, "tensors": tensors, "nodes": [node]})
    )


_CATALOG = """\
| model | D | q/kv heads | causal |
|---|---|---|---|
| llama | 128 | 32/8 | yes |

### `llama_run.py`

Validated in bf16 at Sq=Skv=4096.
"""


def _mine_sources(tmp_path: Path, *sources: str) -> tuple[list, list[dict]]:
    """Mine `sources` to both outputs; returns the `--out` shapes and the CSV rows."""
    out = tmp_path / "shapes.json"
    query = tmp_path / "model-shapes.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(_MINE),
            *sources,
            "--include-windowed",
            "--out",
            str(out),
            "--out-query-csv",
            str(query),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    header, *rows = query.read_text().splitlines()
    return json.loads(out.read_text()), [
        dict(zip(header.split(","), row.split(","))) for row in rows
    ]


class TestCatalogDecodeAttendsTheFullContext:
    """A decode row is the newest token of a KV-cached sequence: anchored top-left, its
    single query row would see key 0 alone."""

    def test_decode_rows_are_bottom_right_and_prefill_rows_top_left(self, tmp_path):
        catalog = tmp_path / "MODEL_CATALOG.md"
        catalog.write_text(_CATALOG)
        shapes, rows = _mine_sources(tmp_path, "--catalog", str(catalog))

        by_phase: dict[str, set] = {}
        for shape in shapes:
            phase = shape["_provenance"]["phase"]
            by_phase.setdefault(phase, set()).add(
                (shape["seqlen_q"], shape["seqlen_k"], shape["mask_type"])
            )
        assert by_phase == {"prefill": {(4096, 4096, 1)}, "decode": {(1, 4096, 2)}}
        assert {(r["name"], r["q.alignment"]) for r in rows} == {
            ("llama prefill", "top_left"),
            ("llama decode", "bottom_right"),
        }


class TestABottomRightWindowAcrossLengthsIsRefused:
    def test_the_window_kind_has_no_anchor_to_carry_it(self, tmp_path):
        """The request's window kind is top-left; mining a shifted band as one serves
        a different mask."""
        _sdpa_graph(
            tmp_path / "graphs" / "g.json",
            512,
            4096,
            {"left_bound": 127, "right_bound": 0, "diagonal_alignment": "BOTTOM_RIGHT"},
        )
        out = tmp_path / "shapes.json"
        result = subprocess.run(
            [
                sys.executable,
                str(_MINE),
                "--graphs",
                str(tmp_path / "graphs"),
                "--include-windowed",
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "unsupported translation" in result.stdout + result.stderr
        assert not out.exists()


class TestTheOutputIsWhatTheRequestClassAccepts:
    """`--out` is the corpus `registry_enumerator.py --shapes` hands to rocKE's frozen
    `AttentionRequest`; a key that class does not declare aborts the whole run."""

    @pytest.fixture
    def rocke(self, monkeypatch):
        """The checkout's rocKE request class, so what is asserted is the request each
        mined shape becomes."""
        monkeypatch.setattr(sys, "path", list(sys.path))
        monkeypatch.syspath_prepend(str(_MINE.parent))
        try:
            # registry_enumerator needs PyYAML, which a lane without rocKE's tooling lacks.
            import registry_enumerator
        except ImportError as exc:
            pytest.skip(f"the rocKE library cannot be imported here ({exc})")

        # Absolute, from this file: a copied source tree has no .git for the
        # repository-relative form to anchor on.
        provider = _MINE.parents[5] / "dnn-providers" / "hip-kernel-provider"
        try:
            registry_enumerator.bind_provider(str(provider))
            from dispatch.attention.common import AttentionRequest, _request_errors
        except (registry_enumerator.EnumerationError, ImportError) as exc:
            pytest.skip(f"the rocKE library cannot be imported here ({exc})")

        def resolve(shapes: list) -> list:
            """Each shape with the request it becomes; `_`-prefixed keys are carried
            provenance, not request fields."""
            return [
                (
                    shape,
                    AttentionRequest(
                        arch="gfx950",
                        **{k: v for k, v in shape.items() if not k.startswith("_")},
                    ),
                )
                for shape in shapes
            ]

        return resolve, _request_errors

    def test_every_mined_mask_resolves_to_the_request_it_describes(
        self, tmp_path, rocke
    ):
        resolve, request_errors = rocke
        graphs = tmp_path / "graphs"
        _sdpa_graph(graphs / "br.json", 512, 4096, {"causal_mask_bottom_right": True})
        _sdpa_graph(
            graphs / "tl.json",
            512,
            4096,
            {"left_bound": -1, "right_bound": 0, "diagonal_alignment": "TOP_LEFT"},
        )
        _sdpa_graph(
            graphs / "swa.json", 4096, 4096, {"left_bound": 127, "right_bound": 0}
        )
        catalog = tmp_path / "MODEL_CATALOG.md"
        catalog.write_text(_CATALOG)
        shapes, _ = _mine_sources(
            tmp_path, "--graphs", str(graphs), "--catalog", str(catalog)
        )

        features = {}
        windows = []
        for shape, request in resolve(shapes):
            assert request_errors(request) == [], shape
            origin = shape["_provenance"]
            label = origin.get("graph") or origin["phase"]
            features.setdefault(label, set()).add(request.features())
            if label == "swa":
                windows.append(request)
        # A causal sliding window is `causal` + `sliding_window`; a missing right bound
        # would read as unbounded and make it a non-causal `band`.
        assert features == {
            "br": {frozenset({"causal", "causal_bottom_right"})},
            "tl": {frozenset({"causal"})},
            "swa": {frozenset({"causal", "sliding_window"})},
            "prefill": {frozenset({"causal"})},
            "decode": {frozenset({"causal", "causal_bottom_right"})},
        }
        # left_bound 127, right_bound 0: the request states the width and a right bound
        # of 0, which rocKE normalizes to a band 128 keys wide ending on the diagonal.
        assert [
            (
                int(r.mask_type),
                int(r.sliding_window),
                int(r.right_bound),
                r.window_left,
                r.window_right,
            )
            for r in windows
        ] == [(3, 128, 0, None, None)]
