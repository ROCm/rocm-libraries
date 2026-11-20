# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A selector revision is re-stamped only on proof the change cannot move a prediction."""
from __future__ import annotations

import hashlib
import json
import math

import pytest

pytest.importorskip("pandas")

from uhd_gen.__main__ import main
from uhd_gen.provenance import validate_provenance
from uhd_gen.restamp import GENERIC_KIND, NATIVE_KIND, restamp

ENGINE = "hip-kernel-provider:asm_sdpa"
ARCH = "gfx942"
THRESHOLDS = {
    "max_median_change": 0.03,
    "outlier_change": 0.10,
    "max_outlier_fraction": 0.05,
    "min_graphs": 30,
}


def _generic(source_file="/old/kernel.cu", sha="aa", model_hash="m1"):
    return {
        "selector": GENERIC_KIND,
        "packs": {
            "pack/a": {
                "kernels": {
                    "k": {"source_file": source_file, "sha256": sha},
                },
            },
        },
        "ranker": {"model_hash": model_hash},
    }


def _generic_dump(description):
    text = json.dumps(
        description, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    )
    return {
        "engine": ENGINE,
        "selector_revision": f"{GENERIC_KIND}/{hashlib.sha256(text.encode()).hexdigest()}",
        "selector": description,
    }


def _native(inputs):
    return {"selector": NATIVE_KIND, "inputs": dict(inputs)}


def _native_dump(inputs):
    description = _native(inputs)
    lines = "".join(f"{name}:{inputs[name]}\n" for name in sorted(inputs))
    digest = hashlib.sha256(lines.encode()).hexdigest()[:16]
    return {
        "engine": ENGINE,
        "selector_revision": f"hip-kernel-provider/asm-sdpa-fwd/{digest}",
        "selector": description,
    }


def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _model(tmp_path, revision, **trained):
    path = tmp_path / "model.uhd.json"
    return _write(
        path,
        {
            "version": "1.0",
            "id": "727e5401-3b99-49ff-a2fc-68fd4eedbb54",
            "name": "l1",
            "adapter": "tree_data",
            "trained_against": {"selector_revision": revision, **trained},
            "objective": "max",
            "score": {"metric": "tflops", "calibrated": True, "transform": "log1p"},
            "features_signature": [],
            "tree_data": {"artifact": "model.bin"},
        },
    )


def _corpus(tmp_path, name, revision, elapsed, *, invalid=()):
    """An immediate corpus of ENGINE: graph `g{i}` takes `elapsed[i]` ms.

    `invalid` graphs carry a numerically wrong verdict: present, but without a label.
    """
    rows = []
    for index, milliseconds in enumerate(elapsed):
        graph = f"g{index}"
        row = {
            "engine_id": 7,
            "engine_name": ENGINE,
            "graph_id": graph,
            "graph_key": graph,
            "device_id": "board",
            "arch": ARCH,
            "binding": {
                "engine": ENGINE,
                "role": "predict_engine",
                "arch": ARCH,
                "metric": "tflops",
                "selector_revision": revision,
                "trained_against": {"selector_revision": revision},
            },
            "features": {"graph.flops": 2e12},
            "avgTimeMs": milliseconds,
            "robustMeanMs": milliseconds,
            "is_valid": True,
            "selection_mode": "immediate",
            "timing_statistic": "robustMeanMs",
        }
        if graph in invalid:
            row["numerically_valid"] = False
            for label in ("avgTimeMs", "robustMeanMs"):
                row.pop(label)
        rows.append(row)
    return _write(tmp_path / f"{name}.json", rows)


def _run(tmp_path, model, before, after, rows=None, write=True, **thresholds):
    before_path = _write(tmp_path / "before.selector.json", before)
    after_path = _write(tmp_path / "after.selector.json", after)
    return restamp(
        model,
        before_path,
        after_path,
        rows_before=rows[0] if rows else None,
        rows_after=rows[1] if rows else None,
        thresholds={**THRESHOLDS, **thresholds},
        write=write,
    )


def _native_pair(*, changed=("src/Plan.hpp",)):
    old = {"src/Plan.hpp": "01", "src/Other.hpp": "02", "kernels/a.co": "03"}
    new = {**old, **{name: "ff" for name in changed}}
    return _native_dump(old), _native_dump(new)


GRAPHS = 40


def _same(count=GRAPHS):
    return [2.0 + index * 0.01 for index in range(count)]


def test_generic_source_file_move_is_restamped_on_path_alone(tmp_path):
    before = _generic_dump(_generic())
    after = _generic_dump(_generic(source_file="/new/place/kernel.cu"))
    model = _model(tmp_path, before["selector_revision"], **{"x-note": 1})

    report = _run(tmp_path, model, before, after)

    assert report["passed"] and report["proof"] == "path-only"
    assert report["changes"]["soft"] == ["/packs/pack~1a/kernels/k/source_file"]
    trained = json.loads(model.read_text(encoding="utf-8"))["trained_against"]
    assert trained["selector_revision"] == after["selector_revision"]
    assert trained["x-restamps"] == [
        {
            "from": before["selector_revision"],
            "to": after["selector_revision"],
            "proof": "path-only",
            "changes": ["/packs/pack~1a/kernels/k/source_file"],
        }
    ]
    assert trained["x-note"] == 1
    # The loader still reads what was written, and the record does not enter comparisons.
    assert validate_provenance(trained) == {
        "selector_revision": after["selector_revision"]
    }


def test_restamp_keeps_the_files_formatting(tmp_path):
    before = _generic_dump(_generic())
    after = _generic_dump(_generic(source_file="/new/kernel.cu"))
    model = _model(tmp_path, before["selector_revision"])
    original = model.read_text(encoding="utf-8")

    _run(tmp_path, model, before, after)

    written = model.read_text(encoding="utf-8")
    assert written.endswith("}\n") and written.count("\n") > original.count("\n")
    assert written.startswith('{\n  "version"')


def test_dry_run_judges_without_writing(tmp_path):
    before = _generic_dump(_generic())
    after = _generic_dump(_generic(source_file="/new/kernel.cu"))
    model = _model(tmp_path, before["selector_revision"])
    original = model.read_bytes()

    report = _run(tmp_path, model, before, after, write=False)

    assert report["passed"] and not report["written"]
    assert model.read_bytes() == original


@pytest.mark.parametrize(
    "after",
    [
        _generic(sha="bb"),
        _generic(model_hash="m2"),
        _generic(source_file="/new/kernel.cu", sha="bb"),
    ],
    ids=["kernel-bytes", "ranker", "moved-and-rebuilt"],
)
def test_generic_change_beyond_a_path_move_is_refused(tmp_path, after):
    before = _generic_dump(_generic())
    model = _model(tmp_path, before["selector_revision"])
    original = model.read_bytes()

    report = _run(tmp_path, model, before, _generic_dump(after))

    assert not report["passed"] and report["changes"]["hard"]
    assert "retrain" in report["problems"][0]
    assert model.read_bytes() == original


def test_changed_selector_kind_is_refused(tmp_path):
    before = _generic_dump(_generic())
    after = _native_dump({"a.cpp": "01"})
    model = _model(tmp_path, before["selector_revision"])

    report = _run(tmp_path, model, before, after)

    assert not report["passed"]
    assert "selector kind changed" in report["changes"]["hard"][0]


def test_native_soft_change_without_rows_is_refused(tmp_path):
    before, after = _native_pair()
    model = _model(tmp_path, before["selector_revision"])
    original = model.read_bytes()

    report = _run(tmp_path, model, before, after)

    assert not report["passed"] and "--rows-before" in report["problems"][0]
    assert report["changes"]["soft"] == ["src/Plan.hpp"]
    assert model.read_bytes() == original


def _rows(tmp_path, before, after, now, then=None, **kwargs):
    return (
        _corpus(tmp_path, "rows_before", before["selector_revision"], then or _same()),
        _corpus(tmp_path, "rows_after", after["selector_revision"], now, **kwargs),
    )


def test_native_soft_change_with_matching_labels_is_restamped(tmp_path):
    before, after = _native_pair()
    model = _model(tmp_path, before["selector_revision"])
    # One graph drifts by 20%: an outlier, well inside the 5% allowance of 40 graphs.
    now = _same()
    now[0] *= 1.2
    rows = _rows(tmp_path, before, after, now)

    report = _run(tmp_path, model, before, after, rows)

    assert report["passed"] and report["proof"] == "measured-parity"
    assert report["parity"]["graphs"] == GRAPHS
    assert report["parity"]["median_change"] == 0.0
    assert report["parity"]["outlier_fraction"] == 1 / GRAPHS
    trained = json.loads(model.read_text(encoding="utf-8"))["trained_against"]
    assert trained["selector_revision"] == after["selector_revision"]
    entry = trained["x-restamps"][0]
    assert entry["proof"] == "measured-parity" and entry["changes"] == ["src/Plan.hpp"]
    assert entry["parity"] == report["parity"]


def test_native_parity_failure_is_refused(tmp_path):
    before, after = _native_pair()
    model = _model(tmp_path, before["selector_revision"])
    original = model.read_bytes()
    # Every graph is 8% slower: under the outlier line, over the median allowance.
    rows = _rows(tmp_path, before, after, [value * math.exp(0.08) for value in _same()])

    report = _run(tmp_path, model, before, after, rows)

    assert not report["passed"] and "median" in report["problems"][0]
    assert model.read_bytes() == original


def test_too_many_outliers_are_refused(tmp_path):
    before, after = _native_pair()
    model = _model(tmp_path, before["selector_revision"])
    now = _same()
    for index in range(4):  # 10% of 40 graphs, over the 5% allowance
        now[index] *= 1.5

    report = _run(tmp_path, model, before, after, _rows(tmp_path, before, after, now))

    assert not report["passed"] and "changed by more than" in report["problems"][0]


def test_parity_on_too_few_graphs_is_refused(tmp_path):
    before, after = _native_pair()
    model = _model(tmp_path, before["selector_revision"])
    rows = _rows(tmp_path, before, after, _same(10), _same(10))

    report = _run(tmp_path, model, before, after, rows)

    assert not report["passed"] and "at least 30" in report["problems"][0]


def test_newly_served_graph_is_refused_but_newly_declined_is_counted(tmp_path):
    before, after = _native_pair()
    model = _model(tmp_path, before["selector_revision"])

    declined = _rows(tmp_path, before, after, _same(), invalid={"g0"})
    report = _run(tmp_path, model, before, after, declined)
    assert report["passed"]
    assert report["parity"]["newly_declined"] == 1
    assert report["parity"]["graphs"] == GRAPHS - 1

    served = (
        _corpus(
            tmp_path,
            "served_before",
            before["selector_revision"],
            _same(),
            invalid={"g0"},
        ),
        _corpus(tmp_path, "served_after", after["selector_revision"], _same()),
    )
    model = _model(tmp_path, before["selector_revision"])
    report = _run(tmp_path, model, before, after, served)
    assert not report["passed"] and "not served at the old one" in report["problems"][0]


def test_a_graph_the_old_corpus_never_measured_is_reported_not_judged(tmp_path):
    before, after = _native_pair()
    model = _model(tmp_path, before["selector_revision"])
    rows = (
        _corpus(tmp_path, "rows_before", before["selector_revision"], _same()),
        _corpus(tmp_path, "rows_after", after["selector_revision"], _same(GRAPHS + 1)),
    )

    report = _run(tmp_path, model, before, after, rows)

    assert report["passed"], report["problems"]
    assert report["parity"]["unmatched"] == 1 and report["parity"]["newly_served"] == 0


def test_rows_collected_at_another_revision_prove_nothing(tmp_path):
    before, after = _native_pair()
    model = _model(tmp_path, before["selector_revision"])
    rows = (
        _corpus(tmp_path, "rows_before", before["selector_revision"], _same()),
        _corpus(tmp_path, "rows_after", before["selector_revision"], _same()),
    )

    report = _run(tmp_path, model, before, after, rows)

    assert (
        not report["passed"]
        and "not " + after["selector_revision"] in report["problems"][0]
    )


@pytest.mark.parametrize("name", ["kernels/a.co", "src/tuning.csv", "codegen.py"])
def test_native_hard_change_is_refused_even_with_perfect_rows(tmp_path, name):
    before, after = _native_pair(changed=(name,))
    model = _model(tmp_path, before["selector_revision"])
    original = model.read_bytes()

    report = _run(
        tmp_path, model, before, after, _rows(tmp_path, before, after, _same())
    )

    assert not report["passed"] and report["changes"]["hard"] == [name]
    assert model.read_bytes() == original


def test_dump_of_another_build_than_trained_is_refused(tmp_path):
    before = _generic_dump(_generic())
    after = _generic_dump(_generic(source_file="/new/kernel.cu"))
    model = _model(tmp_path, "generic-untuned-v1/" + "0" * 64)
    original = model.read_bytes()

    report = _run(tmp_path, model, before, after)

    assert not report["passed"] and "trained against" in report["problems"][0]
    assert model.read_bytes() == original


def test_model_already_at_the_after_revision_is_left_alone(tmp_path):
    before = _generic_dump(_generic())
    after = _generic_dump(_generic(source_file="/new/kernel.cu"))
    model = _model(tmp_path, after["selector_revision"])
    original = model.read_bytes()

    report = _run(tmp_path, model, before, after)

    assert report["passed"] and report["already_current"]
    assert model.read_bytes() == original


def test_inconsistent_dump_is_flagged_but_does_not_decide(tmp_path):
    before = _generic_dump(_generic())
    after = _generic_dump(_generic(source_file="/new/kernel.cu"))
    after["selector_revision"] = "generic-untuned-v1/" + "1" * 64
    model = _model(tmp_path, before["selector_revision"])

    report = _run(tmp_path, model, before, after)

    assert report["passed"]
    assert report["verified"] == {"before": True, "after": False}


def test_dumps_of_different_engines_are_refused(tmp_path):
    before = _generic_dump(_generic())
    after = _generic_dump(_generic(source_file="/new/kernel.cu"))
    after["engine"] = "other:engine"
    model = _model(tmp_path, before["selector_revision"])

    report = _run(tmp_path, model, before, after)

    assert not report["passed"] and "different engines" in report["problems"][0]


def test_command_exits_nonzero_on_refusal_and_zero_on_restamp(tmp_path):
    before = _generic_dump(_generic())
    soft = _generic_dump(_generic(source_file="/new/kernel.cu"))
    hard = _generic_dump(_generic(sha="bb"))
    model = _model(tmp_path, before["selector_revision"])
    arguments = ["restamp", str(model), "--selector-before"]

    refused = _write(tmp_path / "hard.json", hard)
    assert (
        main(
            [
                *arguments,
                str(_write(tmp_path / "b.json", before)),
                "--selector-after",
                str(refused),
                "--write",
            ]
        )
        == 1
    )
    report = tmp_path / "report.json"
    assert (
        main(
            [
                *arguments,
                str(_write(tmp_path / "b.json", before)),
                "--selector-after",
                str(_write(tmp_path / "a.json", soft)),
                "--report",
                str(report),
                "--write",
            ]
        )
        == 0
    )
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["schema"] == "uhd_gen.restamp_report/1" and written["written"]
    assert written["to"] == soft["selector_revision"]
