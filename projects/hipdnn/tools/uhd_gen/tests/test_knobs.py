# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The knob report has to separate three things a build budget confuses.

A knob that never varies, a knob that varies without mattering, and a knob that decides
the answer all look alike in a KMD: three declared fields. They cost very differently --
the first costs nothing, the second multiplies the AOT build for noise, and the third
earns every kernel it asks for.

These plant each case with a known answer and assert the report recovers it. A corpus
whose structure is invented here rather than measured is the point: on real data every
number is plausible and none is checkable.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uhd_gen.knobs import (  # noqa: E402
    analyse_knobs,
    format_author_report,
    knob_columns,
    rank_knobs,
)

_ENVELOPE = {
    "device": "devA",
    "pack": "pk",
    "dispatch": "dp",
    "minTimeMs": 0.0,
    "avgTimeMs": 0.0,
    "stddevMs": 0.001,
    "iters": 50,
    "is_valid": "true",
    "skip_reason": "",
    "collection_mode": "STANDARD",
    "problem_complete": "true",
    "shard_id": "0",
    "config_set_hash": "h",
    "applicability_id": "a",
}


def _corpus(problems: int = 60, seed: int = 5) -> pd.DataFrame:
    """A corpus with one knob of each kind, planted.

    - `kernel.block_m` DECIDES: which tile is fastest depends on the problem, so no
      single value can be pinned without roughly doubling the time on half the corpus.
    - `kernel.use_exp2_fast` is a uniform win: 1 is always at least as good, so pinning
      to it costs exactly nothing.
    - `kernel.waves_per_eu` is noise: no systematic effect.
    - `kernel.block_n` never varies.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(problems):
        seqlen = rng.choice([256, 512, 1024, 2048])
        best_bm = 256 if seqlen >= 1024 else 64
        for block_m in (64, 128, 256):
            for waves in (2, 4):
                for exp2 in (0, 1):
                    t = 0.05 if block_m == best_bm else 0.10
                    t *= 0.97 if exp2 else 1.0
                    t *= 1.0 + rng.gauss(0, 0.002)
                    rows.append(
                        {
                            **_ENVELOPE,
                            "benchmark": f"prob{i}",
                            "kernel": f"k{i}_{block_m}_{waves}_{exp2}",
                            "robustMeanMs": t,
                            "kernel.block_m": block_m,
                            "kernel.block_n": 64,
                            "kernel.waves_per_eu": waves,
                            "kernel.use_exp2_fast": exp2,
                        }
                    )
    return pd.DataFrame(rows)


def _knob(report: dict, name: str) -> dict:
    return next(k for k in report["knobs"] if k["name"] == name)


def test_knob_columns_are_the_kernel_axes_only():
    df = _corpus(4)
    df["q.seqlen_q"] = 512
    # `$q.*` describes the problem: it cannot be chosen away and is not an AOT axis.
    assert knob_columns(df) == [
        "kernel.block_m",
        "kernel.block_n",
        "kernel.waves_per_eu",
        "kernel.use_exp2_fast",
    ]


def test_a_field_that_never_varies_is_named_as_constant():
    # It costs no kernels, so it is not an AOT saving -- but it sits in the KMD claiming
    # to be a variant axis, and a model may be ranking on a column that cannot separate
    # anything. Silence about it would leave that undiscovered.
    report = analyse_knobs(_corpus())
    block_n = _knob(report, "kernel.block_n")
    assert block_n["constant"] is True
    assert block_n["distinct_values"] == 1
    assert block_n["per_value"] == []


def test_a_deciding_knob_is_expensive_to_pin():
    # The planted structure: the right tile depends on the problem, so every single
    # value is wrong on part of the corpus and pinning roughly doubles the time there.
    report = analyse_knobs(_corpus())
    block_m = _knob(report, "kernel.block_m")

    assert block_m["constant"] is False
    assert block_m["cost_of_pinning"] > 0.5, (
        "a knob that decides the winner cannot be pinned cheaply; "
        f"p95 regret was {block_m['cost_of_pinning']:.2%}"
    )


def test_a_noise_knob_is_nearly_free_to_pin():
    # Varies, costs kernels, changes nothing: the case the AOT budget wants to find.
    report = analyse_knobs(_corpus())
    waves = _knob(report, "kernel.waves_per_eu")

    assert waves["constant"] is False
    assert waves["cost_of_pinning"] < 0.02, (
        "a knob with no systematic effect must read as nearly free to pin; "
        f"p95 regret was {waves['cost_of_pinning']:.2%}"
    )


def test_a_uniformly_better_value_costs_exactly_nothing():
    # One value dominates everywhere, so only that value need ever be built.
    report = analyse_knobs(_corpus())
    exp2 = _knob(report, "kernel.use_exp2_fast")

    assert exp2["best_value"] == 1
    assert exp2["cost_of_pinning"] == pytest.approx(0.0, abs=1e-9)


def test_pinning_never_silently_drops_a_problem():
    """Coverage loss is not regret and must not be averaged into it.

    A value that cannot serve a problem leaves the engine with no kernel for it. That is
    categorically worse than a slower kernel, so it is counted separately and the best
    value is chosen on coverage first.
    """
    df = _corpus()
    # Make block_m=256 the only choice for one problem, then delete every other
    # candidate for it, so a pin to 64 or 128 cannot serve that problem at all.
    victim = df["benchmark"] == "prob0"
    df = df[~victim | (df["kernel.block_m"] == 256)]

    report = analyse_knobs(df)
    block_m = _knob(report, "kernel.block_m")
    by_value = {v["value"]: v for v in block_m["per_value"]}

    assert by_value[64]["uncovered"] == 1
    assert by_value[256]["uncovered"] == 0
    assert block_m["problems_lost_by_pinning"] == 0, (
        "the recommended value must be one that still serves every problem"
    )


def test_the_variant_curve_finds_the_cheap_covering_set():
    """The AOT question: how few kernels per geometry suffice.

    Twelve combinations are built here and only the tile actually matters, so a couple
    of variants should reach near-zero regret. The assertion is on the shape -- fewer
    variants than combinations, and flat by the end -- not on an exact count, which
    would pin the greedy tie-break rather than the finding.
    """
    report = analyse_knobs(_corpus())
    curve = report["variant_curve"]

    assert curve, "a corpus with varying knobs must produce a curve"
    assert curve[-1]["problems_uncovered"] == 0
    assert curve[-1]["mean_regret"] == pytest.approx(0.0, abs=1e-9)

    within_one_percent = next(
        (row["variants"] for row in curve if row["mean_regret"] < 0.01), None
    )
    assert within_one_percent is not None and within_one_percent <= 4, (
        "twelve built combinations should collapse to a handful; "
        f"needed {within_one_percent} variants to reach 1% mean regret"
    )


def test_an_empty_corpus_is_refused_rather_than_reported():
    df = _corpus(4)
    df["robustMeanMs"] = float("nan")
    with pytest.raises(ValueError, match="positive robustMeanMs"):
        analyse_knobs(df)


# ---- ranking: the sorted answer a kernel author acts on ----------------------


def _ranked(**kw):
    return rank_knobs(analyse_knobs(_corpus(**kw)))


def test_the_knob_that_decides_sorts_above_the_one_that_does_not():
    """The list is ordered by consequence, so the top row is where kernels are earned."""
    ranked = _ranked()
    names = [r["short_name"] for r in ranked]
    assert names.index("block_m") < names.index("waves_per_eu")


def test_a_constant_sorts_last_and_reads_as_a_defect_not_a_saving():
    """It costs no kernels, so ranking it by cost would put a defect above real findings."""
    ranked = _ranked()
    assert ranked[-1]["short_name"] == "block_n"
    assert ranked[-1]["verdict"] == "CONSTANT"
    # The reason a constant matters is that it breaks the model, not that it wastes a build.
    assert "6.3" in ranked[-1]["advice"]


def test_a_free_knob_is_named_droppable():
    ranked = {r["short_name"]: r for r in _ranked()}
    assert ranked["use_exp2_fast"]["verdict"] == "DROP"


def test_importance_is_reported_but_never_reorders_the_ranking():
    """Tree gain is a second opinion: a heavily-split knob can still be free to pin.

    If importance could reorder, a knob the model leans on would be recommended KEEP
    even where the measurements say pinning it is free -- which is the exact mistake
    the report exists to prevent.
    """
    report = analyse_knobs(_corpus())
    plain = [r["short_name"] for r in rank_knobs(report)]
    misleading = {
        "kernel.use_exp2_fast": {"gain": 999999.0, "split": 9999},
        "kernel.block_m": {"gain": 0.1, "split": 1},
    }
    weighted = rank_knobs(report, misleading)
    assert [r["short_name"] for r in weighted] == plain
    by_name = {r["short_name"]: r for r in weighted}
    assert by_name["use_exp2_fast"]["verdict"] == "DROP"
    assert by_name["use_exp2_fast"]["split"] == 9999


def test_missing_importance_leaves_the_columns_empty_rather_than_failing():
    ranked = rank_knobs(analyse_knobs(_corpus()), {})
    assert all(r["gain"] is None for r in ranked)


def test_author_report_names_the_edit_to_make():
    report = analyse_knobs(_corpus())
    text = format_author_report(report, rank_knobs(report), "eng")
    assert "# Knob value report -- eng" in text
    assert "## What to change" in text
    assert "block_n" in text.split("## What to change")[1]


# ---- matched fields and orphaned problems: two ways a zero cost lies ---------


def _corpus_with_geometry(problems: int = 40, seed: int = 7) -> pd.DataFrame:
    """A corpus carrying a field the matcher binds, beside a real knob.

    `kernel.head_size` is the shape the kernel was built for, so every candidate for a
    given problem shares it -- it varies across the corpus and never within a problem.
    That is exactly what the gfx942 sweep produced for seqlen_q, head_size and dtype.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(problems):
        head_size = 64 if i % 2 else 128
        for block_m in (64, 256):
            t = 0.05 if block_m == (256 if head_size == 128 else 64) else 0.10
            t *= 1.0 + rng.gauss(0, 0.002)
            rows.append(
                {
                    **_ENVELOPE,
                    "benchmark": f"prob{i}",
                    "kernel": f"k{i}_{block_m}",
                    "robustMeanMs": t,
                    "kernel.block_m": block_m,
                    "kernel.head_size": head_size,
                    # The twin in the problem namespace is what marks it graph-bound:
                    # the matcher fixed the kernel's head_size to the graph's.
                    "q.head_size": head_size,
                    # No `q.waves_per_eu` exists, and there is no such thing -- nothing
                    # binds it. The pack's generator simply built one value per geometry.
                    "kernel.waves_per_eu": 2 if head_size == 128 else 4,
                }
            )
    return pd.DataFrame(rows)


def test_a_matched_field_is_not_reported_as_droppable():
    """The defect the first real gfx942 run exposed.

    head_size varies across the corpus and never within a problem, so pinning it has no
    measurable cost -- the problems it orphans leave the comparison instead of scoring
    badly in it. Ranked on cost alone it reads 0.00% and the report recommends dropping
    it, which means shipping kernels for one head size.
    """
    ranked = {r["short_name"]: r for r in rank_knobs(analyse_knobs(_corpus_with_geometry()))}
    assert ranked["head_size"]["verdict"] == "MATCHED"
    assert ranked["head_size"]["tunable"] is False
    # And the real knob beside it is still judged on its merits.
    assert ranked["block_m"]["verdict"] == "KEEP"


def test_a_matched_field_sorts_below_every_real_choice():
    ranked = rank_knobs(analyse_knobs(_corpus_with_geometry()))
    names = [r["short_name"] for r in ranked]
    assert names.index("block_m") < names.index("head_size")


def test_a_matched_field_is_never_in_what_to_change():
    report = analyse_knobs(_corpus_with_geometry())
    text = format_author_report(report, rank_knobs(report), "eng")
    assert "head_size" not in text.split("## What to change")[1]


def _corpus_no_value_covers_everything(problems: int = 45, seed: int = 11) -> pd.DataFrame:
    """A knob that genuinely varies within every problem, yet cannot be pinned.

    Each problem was built with two of the three tile values, rotating, so every problem
    has a real choice while no single value exists everywhere. This is the shape a pack
    takes when the generator prunes variants per geometry -- and it is the one where a
    zero pin cost is most misleading, because the value that scores perfectly is the one
    that simply is not there for a third of the corpus.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(problems):
        available = [(64, 128), (128, 256), (256, 64)][i % 3]
        for block_m in available:
            t = 0.05 * (1.0 + rng.gauss(0, 0.002))
            rows.append(
                {
                    **_ENVELOPE,
                    "benchmark": f"prob{i}",
                    "kernel": f"k{i}_{block_m}",
                    "robustMeanMs": t,
                    "kernel.block_m": block_m,
                }
            )
    return pd.DataFrame(rows)


def test_pinning_that_orphans_problems_is_never_droppable():
    """Zero regret over the problems a value can serve says nothing about the ones it
    cannot. A knob is only free to pin if it orphans nothing."""
    ranked = {
        r["short_name"]: r
        for r in rank_knobs(analyse_knobs(_corpus_no_value_covers_everything()))
    }
    block_m = ranked["block_m"]

    assert block_m["tunable"] is True, "every problem offers a real choice of tile"
    assert block_m["problems_lost"] > 0, "no single tile exists across the whole corpus"
    # Timings are flat by construction, so cost alone would read 0.00% and DROP.
    assert block_m["verdict"] == "KEEP"
    assert "no kernel at all" in block_m["advice"]


def test_device_columns_are_not_knobs():
    """A merged multi-board corpus carries `device.*`, and none of it is tunable.

    The columns exist so one arch-keyed model can tell MI300X from MI325X. They are
    facts about the card, not choices an AOT build makes, so they must never reach the
    ranking -- a report suggesting a smaller `device.total_global_mem` would be
    recommending different hardware.
    """
    df = _corpus_with_geometry()
    df["device.cu_count"] = 304
    df["device.total_global_mem"] = 192 * 1024**3

    report = analyse_knobs(df)
    named = {k["name"] for k in report["knobs"]}
    assert not any(name.startswith("device.") for name in named), named
    assert "kernel.block_m" in named

    text = format_author_report(report, rank_knobs(report), "eng")
    assert "total_global_mem" not in text


def test_a_field_the_pack_pinned_is_told_apart_from_one_the_graph_binds():
    """Both look identical in the data -- constant within every problem -- and they need
    opposite answers.

    `head_size` is bound by the matcher: the kernel was built for that shape, and there
    is nothing to do. `waves_per_eu` is not bound by anything; the generator chose one
    value per geometry, so the model was never offered the choice and no sweep can say
    whether it matters. That is the author's decision to revisit, so it has to be named
    differently. The gfx942 pack has exactly this shape: waves_per_eu and persistent
    vary within 0 of its 664 geometries.
    """
    ranked = {r["short_name"]: r for r in rank_knobs(analyse_knobs(_corpus_with_geometry()))}

    assert ranked["head_size"]["verdict"] == "MATCHED"
    assert ranked["head_size"]["graph_bound"] is True
    assert "$q.head_size" in ranked["head_size"]["advice"]

    assert ranked["waves_per_eu"]["verdict"] == "PINNED"
    assert ranked["waves_per_eu"]["graph_bound"] is False
    assert "generator" in ranked["waves_per_eu"]["advice"]
    assert "re-sweep" in ranked["waves_per_eu"]["advice"]


def test_a_pinned_field_reaches_what_to_change_but_a_graph_bound_one_does_not():
    """The author can act on a pinned field and cannot act on a graph-bound one, so only
    the first belongs in the list of edits to make."""
    report = analyse_knobs(_corpus_with_geometry())
    text = format_author_report(report, rank_knobs(report), "eng")
    changes = text.split("## What to change")[1]

    assert "waves_per_eu" in changes
    assert "head_size" not in changes
