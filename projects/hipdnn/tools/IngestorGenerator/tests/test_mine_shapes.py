# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The shape corpus: what to compile, from the sources that decide it.

Every mining source the guidance originally named was kernel-side, so all of them
answered "what is LEGAL?" and none answered "what will anyone ASK for?". An
integration following it exactly shipped a legal, validated, fully-tested engine
that served zero real workloads -- three times, each caught only by counting against
an external corpus rather than from inside the integration.

Two properties carry most of the value and both are asserted here: that a mask
spelling is never GUESSED, and that provenance survives onto every shape. The second
looks like bookkeeping and is not -- the same measured result was a large win on one
synthetic suite and near-parity on real model traces, and only the provenance split
made that visible instead of merely suspected.
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
        """`priority` is a shipping signal available from no other source, and the
        model is the axis a mixed-corpus result has to be split along."""
        rc, _, shapes = _mine(
            tmp_path, _HEADER + _row(priority="P0", model="Llama-3-8B")
        )
        assert rc == 0
        provenance = shapes[0]["_provenance"]
        assert provenance["priority"] == "P0"
        assert provenance["model"] == "Llama-3-8B"
        assert provenance["source"] == "published"

    def test_an_unknown_mask_spelling_is_refused_not_guessed(self, tmp_path):
        """Defaulting an unrecognised mask is how a windowed graph gets served as
        plain causal -- a wrong answer rather than a decline, which already shipped
        once through a different layer."""
        rc, output, _ = _mine(tmp_path, _HEADER + _row(mask="cheesecake"))
        assert rc != 0
        assert "unknown mask spelling" in output

    def test_an_unknown_dtype_spelling_is_refused_not_passed_through(self, tmp_path):
        """The CSV reader once built `dtype` from `row.get("dtype") or "bf16"`
        directly, bypassing `_DTYPE_SPELLINGS` entirely -- the graph-corpus and
        rocKE-bench readers both refuse an unrecognised dtype, and the CSV path
        silently wrote it straight into the corpus instead. An unrecognised dtype
        here builds the wrong binary and still validates, exactly like the mask
        case above; a decline is the correct answer, not a passthrough."""
        rc, output, shapes = _mine(tmp_path, _HEADER + _row(dtype="fp8_e4m3"))
        assert rc != 0, f"bad dtype must be refused, not mined: {shapes}"
        assert "unknown dtype spelling" in output

    def test_every_dtype_spelling_normalises_the_same_as_the_other_readers(
        self, tmp_path
    ):
        """The converse of the refusal above: every spelling `_DTYPE_SPELLINGS`
        already recognises must still mine cleanly through the CSV path, and
        normalise to the same canonical value the graph/rocKE-bench readers
        produce -- the fix must not narrow what a valid CSV can express."""
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
        """A row that simply does not say is a fallback, not a refusal --
        distinct from a row that says something this table does not recognise."""
        rc, output, shapes = _mine(tmp_path, _HEADER + _row(dtype=""))
        assert rc == 0, output
        assert shapes[0]["dtype"] == "bf16"

    def test_windowed_rows_are_excluded_loudly_not_folded_onto_causal(self, tmp_path):
        """Folding `swin` onto `causal` collapsed seven distinct shape keys in an
        earlier join. Excluded by default, included on request, never merged."""
        text = (
            _HEADER
            + _row(mask="causal")
            + _row(shape_idx=1, mask="swin", seq_q=2048, seq_kv=2048)
        )
        rc, _, shapes = _mine(tmp_path, text)
        assert rc == 0
        assert len(shapes) == 1, "swin must not be mined by default"

        rc, _, with_windowed = _mine(tmp_path, text, "--include-windowed")
        assert rc == 0
        assert len(with_windowed) == 2
        masks = {s["mask_type"] for s in with_windowed}
        assert len(masks) == 2, "swin must keep its own mask_type, not become causal"

    def test_identical_shapes_from_different_rows_merge_to_one_variant(self, tmp_path):
        """A corpus is a set of shapes; two rows asking for the same shape is one
        variant to compile."""
        text = _HEADER + _row(shape_idx=0, model="A") + _row(shape_idx=1, model="B")
        rc, output, shapes = _mine(tmp_path, text)
        assert rc == 0
        assert len(shapes) == 1
        assert "1 duplicate shape(s) merged" in output

    def test_a_genuinely_different_shape_is_kept(self, tmp_path):
        """Guards the merge above from collapsing real coverage."""
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
        ]
        if backward:
            tensors.append({"name": "d_query", "dims": [1, 32, 4096, 128]})
        path.write_text(json.dumps({"tensors": tensors}))

    def test_backward_graphs_are_excluded_structurally(self, tmp_path):
        """A prefill kernel has no backward path, and one such graph routes to a
        third-party backward FMHA that takes the DEVICE down -- which killed a sweep
        two minutes into warmup. Gradient tensors are the marker; the filename is
        not authoritative, so the check reads the tensors.
        """
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
    """`causal` decides which dispatcher branch resolves, so mining it wrong
    silently sizes a variant set that cannot serve the shapes it claims.

    The first version read `"causal" in path.stem.lower()`. Against this repo's real
    bundle tree that is wrong for every causal graph there is: 25 carry `causal` in a
    PARENT DIRECTORY (`.../hd128_causal_batch/Small/Small.json`) and none carry it in
    the leaf name. The miner reported a corpus with zero causal graphs, and collapsed
    causal and non-causal shapes that differ in nothing else onto one key -- a
    coverage loss, not merely a mislabel.

    hipDNN has no `causal` boolean. The deprecated pair takes precedence when set;
    otherwise causality is (left_bound, right_bound, diagonal_alignment). Every
    shipped causal bundle leaves both booleans false and says `left_bound=-1,
    right_bound=0`, so a reader that trusts the booleans alone calls them all
    non-causal.
    """

    def _graph(self, path: Path, attrs: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "tensors": [
                        {
                            "name": "query",
                            "dims": [1, 32, 4096, 128],
                            "data_type": "bf16",
                        },
                        {"name": "key", "dims": [1, 8, 4096, 128], "data_type": "bf16"},
                    ],
                    "nodes": [{"attributes": attrs}],
                }
            )
        )

    def _mine(self, tmp_path: Path) -> list:
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
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(out.read_text())

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
        """The converse, and the one a filename heuristic gets backwards: a
        directory named `causal` containing an unmasked graph."""
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
        """Folding a window onto causal is how one gets SERVED as plain causal --
        a wrong answer rather than a decline."""
        self._graph(
            tmp_path / "graphs" / "swa" / "g.json",
            {"causal_mask": False, "left_bound": 128, "right_bound": 0},
        )
        assert [s["mask_type"] for s in self._mine(tmp_path)] == [2]

    def test_the_deprecated_boolean_still_takes_precedence_when_set(self, tmp_path):
        self._graph(
            tmp_path / "graphs" / "g.json",
            {"causal_mask": True, "left_bound": None, "right_bound": None},
        )
        assert [s["mask_type"] for s in self._mine(tmp_path)] == [1]

    def test_causal_and_noncausal_shapes_do_not_collapse_onto_one_key(self, tmp_path):
        """Two graphs identical but for causality are TWO variants to compile.
        Mining them both as non-causal merged them and silently halved coverage."""
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
        """`left_bound` drives the branch below (`>= 0` -> window, else causal),
        and a non-numeric value fell through neither comparison, landing on
        `return _MASK_TYPE["causal"]` by default -- the same wrong-answer-not-a-
        decline failure this reader exists to refuse for mask and dtype."""
        self._graph(
            tmp_path / "graphs" / "g.json",
            {"causal_mask": False, "left_bound": "unbounded", "right_bound": 0},
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
        assert "non-numeric left_bound" in (result.stdout + result.stderr)


class TestRocKeBenchTree:
    """The third source: rocKE's own benchmark tree.

    For an arch with no published results CSV -- gfx950 at the time this was added --
    it is the ONLY source that says what the kernel team measures, so a miner that
    cannot read it sizes a variant set from sample graphs alone.

    These files are JSONL (one record per line), not JSON documents: `json.load`
    raises "Extra data" on every one of them.
    """

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

    def test_a_jsonl_trace_is_read_at_all(self, tmp_path):
        """Guards the format itself: these are one-record-per-line, and a reader
        using json.load gets 'Extra data' and silently mines nothing."""
        result, shapes = self._mine_bench(
            tmp_path, self._record(), self._record(max_seqlen_q=8192, max_seqlen_k=8192)
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(shapes) == 2

    def test_an_unwindowed_prefill_trace_is_causal(self, tmp_path):
        """`[-1, -1]` is unbounded both ways, which for a prefill suite is full
        causal -- the paired live benchmark labels its W=0 arm 'full-causal'."""
        _, shapes = self._mine_bench(tmp_path, self._record(window_size=[-1, -1]))
        assert shapes[0]["mask_type"] == 1
        assert shapes[0]["sliding_window"] == 0

    def test_a_windowed_trace_carries_its_WIDTH_not_just_its_kind(self, tmp_path):
        """THE regression this class exists for. Dropping the width sends
        sliding_window=0 to the dispatcher, which resolves to plain causal -- the
        kernel then computes a full causal triangle for a banded request and returns
        a WRONG ANSWER instead of declining. The mask kind alone does not encode it.
        """
        _, shapes = self._mine_bench(tmp_path, self._record(window_size=[127, 0]))
        assert shapes[0]["mask_type"] == 2, "a finite left bound is a window"
        assert shapes[0]["sliding_window"] == 128, (
            "the window WIDTH must reach the request; 127 is the left bound and the "
            "band includes the current token"
        )

    def test_sinks_are_carried_rather_than_filtered(self, tmp_path):
        """Whether an integration SHIPS a sink variant is a scope decision made
        downstream. Filtering the shape out here hides it from the step-9
        reconciler, which is exactly where a declined-but-servable shape is
        supposed to surface."""
        _, shapes = self._mine_bench(tmp_path, self._record(has_sinks=True))
        assert shapes[0]["use_sinks"] is True
        assert shapes[0]["_provenance"]["has_sinks"] is True

    def test_a_trace_with_no_recorded_causality_is_skipped_not_defaulted(
        self, tmp_path
    ):
        """No record in rocKE's shipped traces carries a causal/mask key, so
        causality comes from window_size or from nowhere. Defaulting it picks which
        branch the dispatcher resolves and which kernels get built."""
        result, shapes = self._mine_bench(
            tmp_path, self._record(window_size=None), self._record()
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(shapes) == 1, "the unknown-mask record must not be mined"
        assert "no recorded causality" in result.stdout

    def test_an_unknown_dtype_spelling_is_refused(self, tmp_path):
        """Three vocabularies meet in this miner and none agree. A guessed dtype
        builds a different binary and still validates."""
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
    """`sample_sdpa_backward` spells its gradients `dq`/`dk`/`dv`/`do`, not
    `d_query`. The original marker set matched neither, so the graph passed the
    backward filter and was caught only incidentally by its `float` dtype -- a
    backward graph using a servable dtype would have been mined as forward, and one
    of that class takes the DEVICE down mid-sweep."""

    @pytest.mark.parametrize("gradient", ["d_query", "dq", "dk", "dv", "do"])
    def test_either_gradient_spelling_excludes_the_graph(self, tmp_path, gradient):
        corpus = tmp_path / "graphs"
        corpus.mkdir()
        tensors = [
            {"name": "q", "dims": [1, 32, 4096, 128], "data_type": "bf16"},
            {"name": "k", "dims": [1, 8, 4096, 128], "data_type": "bf16"},
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
        """Belt and braces: the graph DECLARES what it is, so the op type is the
        primary marker and tensor names are the fallback."""
        corpus = tmp_path / "graphs"
        corpus.mkdir()
        (corpus / "g.json").write_text(
            json.dumps(
                {
                    "tensors": [
                        {"name": "q", "dims": [1, 32, 4096, 128], "data_type": "bf16"},
                        {"name": "k", "dims": [1, 8, 4096, 128], "data_type": "bf16"},
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
        """Control: the exclusion must not fire on an ordinary forward graph whose
        tensors merely start with d (`descale_q`), or the corpus empties silently."""
        corpus = tmp_path / "graphs"
        corpus.mkdir()
        (corpus / "g.json").write_text(
            json.dumps(
                {
                    "tensors": [
                        {"name": "q", "dims": [1, 32, 4096, 128], "data_type": "bf16"},
                        {"name": "k", "dims": [1, 8, 4096, 128], "data_type": "bf16"},
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
