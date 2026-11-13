# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tools/variant_reduce.py on a tiny selection, KDP and sweep output built per test."""

import base64
import json
import re
import shutil
import struct
import sys
import uuid
from pathlib import Path

import pytest
import yaml

from codegen.config_loader import load_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import variant_reduce  # noqa: E402

PACK_ID = "3d3ae611-244d-558b-958c-bd385a92913a"
DISPATCH_ID = "451e1a7b-429c-5d37-85b3-afc5bbdbbdb0"
DEVICE = {
    "gcn_arch_name": "gfx950",
    "warp_size": 64,
    "multi_processor_count": 256,
    "total_global_mem": 309220868096,
    "memory_bus_width": 8192,
    "memory_clock_rate": 1900000,
    "shared_mem_per_block": 163840,
}

# (dtype, tile, unroll, default); tile is a `divides` knob, unroll is not.
KERNELS = [
    ("fp32", 128, 2, True),
    ("fp32", 64, 2, False),  # floor: divides the default's tile
    ("fp32", 128, 4, False),  # wins graph g_perf by 50%
    ("fp32", 128, 1, False),  # wins graph g_noise by 1%, under the noise floor
    ("fp32", 256, 4, False),  # never ranked
    ("fp16", 128, 2, True),
    ("fp16", 64, 2, False),
    ("fp16", 256, 4, False),  # the only kernel graph g_only ranks
]
GRAPHS = {
    "g_perf": [(("fp32", 128, 4), 1.0), (("fp32", 128, 2), 1.5)],
    "g_noise": [(("fp32", 128, 1), 1.0), (("fp32", 64, 2), 1.01)],
    "g_only": [(("fp16", 256, 4), 1.0)],
}


def kname(dtype, tile, unroll):
    return f"softmax_{dtype}_t{tile}_u{unroll}"


def graph_content(name, graph_id):
    """A minimal Graph flatbuffer: a vtable naming only field 9 (`id`), then ``name``."""
    vtable = struct.pack("<HH", 24, 20) + struct.pack("<10H", *([0] * 9 + [4]))
    table = struct.pack("<i", 24) + graph_id + name.encode()
    return base64.b64encode(struct.pack("<I", 28) + vtable + table).decode("ascii")


def identity(name):
    """`variant_reduce.graph_identity` of the fixture graph ``name``, whatever its id."""
    graph = {"content_base64": graph_content(name, bytes(16))}
    return variant_reduce.graph_identity(graph, "fixture")


def _attempt_dir(root, kind, number, retry=1):
    """``(attempt, shard dir)`` of one phase attempt of the sweep under ``root``."""
    attempt = root / "sweep" / "attempts" / f"{kind}__c__full__r{number}-{retry:04d}"
    return attempt, attempt / "winners" / "ingestor-winners" / "1.0.0" / "e" / "gfx950"


def build(
    tmp_path, edit=None, reduction=True, rounds=(GRAPHS,), checked=None, kernels=KERNELS
):
    """Write a selection, its installed KDP and a sweep of ``rounds``; return argv."""
    kernels = [
        {
            "name": kname(dtype, tile, unroll),
            "kernel_source": {},
            "metadata": {"dtype": dtype, "tile": tile, "unroll": unroll},
            **({"default": True} if default else {}),
        }
        for dtype, tile, unroll, default in kernels
    ]
    source = {"kind": "hip", "source": "k.hip", "entry": "k"}
    pack = {"name": "p", "arch": ["gfx950"], "kernel_defaults": source}
    selection = {
        "dialect": "packaged",
        "kernel_source_kind": "hip",
        "engine": {"name": "hipkernel:E"},
        "kmd_fields": [
            {"name": "dtype", "type": "string"},
            {"name": "tile", "type": "int"},
            {"name": "unroll", "type": "int"},
            {"name": "rows", "type": "int", "default_value": 64},
        ],
        "packs": [dict(pack, kernels=kernels)],
    }
    if reduction:
        performance = [{"name": "tile", "divides": "rows"}, {"name": "unroll"}]
        selection["reduction"] = {
            "fields": {"feature": ["dtype"], "performance": performance}
        }
    (tmp_path / "selection.yaml").write_text(yaml.safe_dump(selection))
    descriptors = [{"id": f"id-{k['name']}", **k} for k in kernels]
    kdp = {"id": PACK_ID, "dispatch": DISPATCH_ID, "kernelDescriptors": descriptors}
    (tmp_path / "installed.kdp.json").write_text(json.dumps(kdp))

    phases = [("timing", n, graphs) for n, graphs in enumerate(rounds, 1)]
    if checked:
        phases.append(("correctness", 1, {"g_perf": [(checked, 1.0)]}))
    written = []
    for kind, number, graphs in phases:
        # hipDNN mints a fresh Graph.id per finalization, so every round's id differs.
        graph_id = uuid.uuid4().bytes
        attempt, shard = _attempt_dir(tmp_path, kind, number)
        shard.mkdir(parents=True)
        phase = {"kind": kind, "corpus": "c", "arm": "full", "round": number}
        ledger = [{"correctness": {"passed": True}}]
        record = {"status": "success", "phase": phase, "ledger": ledger}
        written.append((phase, attempt))
        (attempt / "phase.json").write_text(json.dumps(record))
        ids = {"pack_id": PACK_ID, "dispatch_id": DISPATCH_ID}
        lines = ["1.0.0"]
        for graph, ranked in graphs.items():
            entries = [
                {"kernel_id": f"id-{kname(*k)}", **ids, "time_ms": t} for k, t in ranked
            ]
            line = {"v": 2, "graph": {"content_base64": graph_content(graph, graph_id)}}
            line["device"] = DEVICE
            line["entries"] = entries
            if edit:
                edit(line)
            lines.append(json.dumps(line))
        (shard / "winners.jsonl").write_text("\n".join(lines) + "\n")
    summary = {"expected_phases": [phase for phase, _ in written]}
    _summary(tmp_path).write_text(json.dumps(summary))
    files = "selection.yaml installed.kdp.json sweep reduced.yaml report.md".split()
    flags = ("--selection", "--kdp", "--sweep-output", "--out", "--report")
    return [a for flag, f in zip(flags, files) for a in (flag, str(tmp_path / f))]


def test_floor_coverage_and_performance_make_the_reduced_selection(tmp_path):
    assert variant_reduce.main(build(tmp_path, checked=("fp32", 128, 4))) == 0
    reduced = load_config(tmp_path / "reduced.yaml")
    selection = load_config(tmp_path / "selection.yaml")
    original = {k.name: k for k in selection.packs[0].kernels}
    floor = {kname(d, t, 2) for d in ("fp32", "fp16") for t in (128, 64)}
    coverage, perf = kname("fp16", 256, 4), kname("fp32", 128, 4)
    assert {k.name for k in reduced.packs[0].kernels} == floor | {coverage, perf}
    for kernel in reduced.packs[0].kernels:
        assert kernel.metadata == original[kernel.name].metadata
        assert kernel.default == original[kernel.name].default
    assert reduced.reduction == selection.reduction
    report = (tmp_path / "report.md").read_text().splitlines()
    rows = {line.split(" | ")[0][2:]: line for line in report if line[:3] == "| s"}
    assert rows[perf].endswith("; numerics checked |")
    assert rows[kname("fp16", 64, 2)].endswith("; numerics unchecked |")


def test_a_win_under_the_round_to_round_drift_is_not_kept(tmp_path):
    """A 6% win kept at one round is dropped once two rounds put eps over it."""
    r1 = {
        "g_perf": [(("fp32", 128, 4), 1.0), (("fp32", 128, 2), 1.08)],
        "g_noise": [(("fp32", 128, 1), 1.0), (("fp32", 64, 2), 1.06)],
        "g_only": GRAPHS["g_only"],
    }
    r2 = {
        "g_perf": [(("fp32", 128, 4), 1.0), (("fp32", 128, 2), 1.12)],
        "g_noise": [(("fp32", 64, 2), 1.0), (("fp32", 128, 1), 1.04)],
    }
    kept = {}
    for rounds in ((r1,), (r1, r2)):
        (tmp_path / str(len(rounds))).mkdir()
        args = build(tmp_path / str(len(rounds)), rounds=rounds)
        reduction = variant_reduce.analyse(*args[1:6:2])
        kept[len(rounds)] = kname("fp32", 128, 1) in reduction.keep(reduction.epsilon)
    assert reduction.epsilon >= 2 * reduction.noise.drift > 0
    assert kept == {1: True, 2: False}


def _noise_policy(tmp_path, rounds, **policy):
    args = build(tmp_path, rounds=rounds)
    selection = tmp_path / "selection.yaml"
    raw = yaml.safe_load(selection.read_text())
    raw["reduction"].update(policy)
    selection.write_text(yaml.safe_dump(raw))
    return variant_reduce.analyse(*args[1:6:2]).noise


def test_the_single_round_epsilon_is_the_selections(tmp_path):
    noise = _noise_policy(tmp_path, (GRAPHS,), single_round_epsilon=0.1)
    assert noise.epsilon_min == 0.1


def test_the_drift_multiplier_is_the_selections(tmp_path):
    r1 = {"g_perf": [(("fp32", 128, 4), 1.0), (("fp32", 128, 2), 1.08)]}
    r2 = {"g_perf": [(("fp32", 128, 4), 1.0), (("fp32", 128, 2), 1.12)]}
    noise = _noise_policy(tmp_path, (r1, r2), drift_multiplier=3)
    assert noise.drift > 0
    assert noise.epsilon_min == pytest.approx(3 * noise.drift)


def test_floor_and_coverage_over_budget_exit_1(tmp_path, capsys):
    assert variant_reduce.main(build(tmp_path) + ["--budget", "4"]) == 1
    assert "floor + coverage exceed budget" in capsys.readouterr().err
    assert not (tmp_path / "reduced.yaml").exists()


def test_applied_rules_replace_the_eps_keep_set(tmp_path, capsys):
    args = build(tmp_path)
    reduction = variant_reduce.analyse(*args[1:6:2])
    rule = next(r for r in reduction.rules if r.text == "drop unroll=1")
    apply = ["--apply-rules", rule.id]
    assert variant_reduce.main(args + apply + ["--budget", "6"]) == 1
    assert f"rules {rule.id} exceed budget: 7 instances > 6" in capsys.readouterr().err
    assert not (tmp_path / "reduced.yaml").exists()
    assert variant_reduce.main(args + apply) == 0
    kept = load_config(tmp_path / "reduced.yaml").packs[0].kernels
    every = {kname(d, t, u) for d, t, u, _ in KERNELS}
    assert {k.name for k in kept} == every - {kname("fp32", 128, 1)}
    # The report is what the user accepts, so it must describe the written set.
    report = (tmp_path / "report.md").read_text()
    assert f"- kept {len(every) - 1} of {len(every)} " in report
    assert f"| {kname('fp32', 128, 1)} | no | removed by {rule.id} |" in report
    assert f"chosen: rules {rule.id} (the eps search chose" in report
    assert "chosen eps:" not in report and "(chosen)" not in report

    # The report must describe the SAME set that was written, not the eps set.
    # "fp32_t128_u1" was dropped by the rule; it must appear as "no" in the
    # report's Variants table.
    report = (tmp_path / "report.md").read_text()
    dropped_name = kname("fp32", 128, 1)
    for line in report.splitlines():
        if dropped_name in line:
            assert (
                "| no |" in line
            ), f"report says {dropped_name!r} is kept, but the rule dropped it:\n{line}"
            break
    else:
        raise AssertionError(f"{dropped_name!r} not found in report Variants table")


def test_combined_rules_keep_a_kernel_for_every_ranked_graph(tmp_path):
    """Each rule alone loses nothing; together they would drop both kernels g_two ranks."""
    kernels = KERNELS + [("fp32", 256, 2, False)]
    g_two = [(("fp32", 256, 2), 1.0), (("fp32", 128, 4), 1.0)]
    args = build(tmp_path, rounds=({"g_two": g_two},), kernels=kernels)
    reduction = variant_reduce.analyse(*args[1:6:2])
    every_rule = ",".join(r.id for r in reduction.rules)
    assert variant_reduce.main(args + ["--apply-rules", every_rule]) == 0
    kept = {k.name for k in load_config(tmp_path / "reduced.yaml").packs[0].kernels}
    assert kept & {kname(*k) for k, _ in g_two}


# Two rules, each cheap alone, that together drop the best kernel of both graphs:
# "drop knob tile" removes t256_u2 and "drop knob unroll" removes t128_u4, leaving
# only the 2.0 ms default.
UNION_KERNELS = [
    ("fp32", 128, 2, True),
    ("fp32", 256, 2, False),
    ("fp32", 128, 4, False),
]
UNION_GRAPHS = {
    "graph_alpha": [
        (("fp32", 256, 2), 1.0),
        (("fp32", 128, 4), 1.01),
        (("fp32", 128, 2), 2.0),
    ],
    "graph_beta": [
        (("fp32", 128, 4), 1.0),
        (("fp32", 256, 2), 1.01),
        (("fp32", 128, 2), 2.0),
    ],
}


def _union_rules(tmp_path):
    args = build(tmp_path, rounds=(UNION_GRAPHS,), kernels=UNION_KERNELS)
    reduction = variant_reduce.analyse(*args[1:6:2])
    texts = ("drop knob tile", "drop knob unroll")
    return args, reduction, [r for r in reduction.rules if r.text in texts]


def test_applied_rules_whose_union_loses_more_than_epsilon_are_refused(
    tmp_path, capsys
):
    args, reduction, rules = _union_rules(tmp_path)
    assert len(rules) == 2
    assert all(r.worst_gain_lost <= reduction.rules_epsilon for r in rules)
    chosen = ",".join(r.id for r in rules)
    assert variant_reduce.main(args + ["--apply-rules", chosen]) == 1
    assert not (tmp_path / "reduced.yaml").exists()
    report = (tmp_path / "report.md").read_text()
    for name in ("graph_alpha", "graph_beta"):
        assert variant_reduce.graph_label(identity(name)) in report
    assert "applied" in capsys.readouterr().err


def test_applied_rules_within_epsilon_together_are_accepted(tmp_path):
    args, _, rules = _union_rules(tmp_path)
    (tile,) = [r for r in rules if r.text == "drop knob tile"]
    assert variant_reduce.main(args + ["--apply-rules", tile.id]) == 0
    assert (tmp_path / "reduced.yaml").exists()


def test_a_union_loses_at_least_what_its_worst_rule_loses(tmp_path):
    _, reduction, rules = _union_rules(tmp_path)
    union = set().union(*(r.removed for r in rules))
    losing, worst = variant_reduce._ranking_loss(union, reduction.rankings)
    assert worst >= max(r.worst_gain_lost for r in rules)
    assert len(losing) >= max(r.graphs_losing_best for r in rules)


def _summary(tmp_path):
    return tmp_path / "sweep" / "summary.json"


def _expect(tmp_path, *phases):
    summary = json.loads(_summary(tmp_path).read_text())
    summary["expected_phases"] += list(phases)
    _summary(tmp_path).write_text(json.dumps(summary))


def test_a_sweep_that_never_wrote_its_summary_is_incomplete(tmp_path, capsys):
    args = build(tmp_path, rounds=(GRAPHS, GRAPHS))
    _summary(tmp_path).unlink()
    assert variant_reduce.main(args) == 3
    assert not (tmp_path / "reduced.yaml").exists()


def test_a_summary_without_expected_phases_is_incomplete(tmp_path):
    """The warmup-failure summary carries no phase list."""
    args = build(tmp_path)
    _summary(tmp_path).write_text(json.dumps({"validated_complete": False}))
    assert variant_reduce.main(args) == 3


def test_an_expected_phase_with_no_record_is_incomplete(tmp_path, capsys):
    """A corpus killed before round 2's timing phase wrote anything."""
    args = build(tmp_path)
    _expect(tmp_path, {"kind": "timing", "corpus": "c", "arm": "full", "round": 2})
    assert variant_reduce.main(args) == 3
    assert "round" in capsys.readouterr().err
    assert not (tmp_path / "reduced.yaml").exists()


def test_an_expected_correctness_phase_with_no_record_is_incomplete(tmp_path):
    args = build(tmp_path)
    _expect(tmp_path, {"kind": "correctness", "corpus": "c", "arm": "full", "round": 1})
    assert variant_reduce.main(args) == 3


def test_a_timing_phase_with_no_winners_is_incomplete(tmp_path):
    args = build(tmp_path)
    for shard in (tmp_path / "sweep").rglob("winners.jsonl"):
        shard.write_text("1.0.0\n")
    assert variant_reduce.main(args) == 3


def test_another_arms_missing_phase_does_not_refuse_this_arm(tmp_path):
    """Round 2 is owed by `other` only; `full` has no round 2 to be missing."""
    args = build(tmp_path)
    _expect(tmp_path, {"kind": "timing", "corpus": "c", "arm": "other", "round": 2})
    assert variant_reduce.main(args + ["--arm", "full"]) == 0
    assert (tmp_path / "reduced.yaml").exists()


def test_another_arms_record_does_not_satisfy_this_arms_phase(tmp_path):
    """`full` owes round 2 and has no record for it; `other` has one, with winners,
    for exactly that (kind, corpus, round)."""
    args = build(tmp_path)
    attempts = tmp_path / "sweep" / "attempts"
    other = attempts / "timing__c__other__r2-0001"
    shutil.copytree(_attempt_dir(tmp_path, "timing", 1)[0], other)
    record = json.loads((other / "phase.json").read_text())
    record["phase"].update(arm="other", round=2)
    (other / "phase.json").write_text(json.dumps(record))
    _expect(tmp_path, {"kind": "timing", "corpus": "c", "arm": "full", "round": 2})
    assert variant_reduce.main(args + ["--arm", "full"]) == 3
    assert not (tmp_path / "reduced.yaml").exists()


@pytest.mark.parametrize(
    "expected",
    [
        [1],
        [{"kind": "timing", "arm": "full", "round": 1}],
        [{"kind": "timing", "arm": "full", "corpus": ["c"], "round": 1}],
        {"kind": "timing"},
        [],
        [{"kind": "timing", "corpus": "c", "arm": "other", "round": 1}],
    ],
    ids=[
        "non-mapping-item",
        "item-without-corpus",
        "unhashable-corpus",
        "not-a-list",
        "no-expected-phases",
        "phases-only-for-another-arm",
    ],
)
def test_an_unusable_expected_phase_list_is_incomplete(tmp_path, capsys, expected):
    args = build(tmp_path)
    _summary(tmp_path).write_text(json.dumps({"expected_phases": expected}))
    assert variant_reduce.main(args) == 3
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert not (tmp_path / "reduced.yaml").exists()


def test_a_zero_time_entry_is_a_ranking(tmp_path):
    edit = lambda d: d["entries"][0].update(time_ms=0)  # noqa: E731
    assert variant_reduce.main(build(tmp_path, edit)) == 0


@pytest.mark.parametrize(
    "ledger, gates, verdict",
    [
        ([({"passed": False}, True)], {"correctness": False}, "FAILED; investigate"),
        (
            [({"passed": False, "tolerance_match": None}, False)],
            {"reference": False},
            "no reference",
        ),
        (
            [({"passed": False, "tolerance_match": None}, False)],
            {"reference": False, "serving": False},
            "FAILED; investigate",
        ),
        (
            [
                ({"passed": False}, True),
                ({"passed": False, "tolerance_match": None}, False),
            ],
            {"correctness": False, "reference": False},
            "FAILED; investigate",
        ),
    ],
    ids=[
        "numerics-failed",
        "no-reference",
        "no-reference-and-another-gate-failed",
        "no-reference-and-numerics-failed",
    ],
)
def test_correctness_attempt_verdicts(tmp_path, capsys, ledger, gates, verdict):
    args = build(tmp_path, checked=("fp32", 128, 4))
    phase = _attempt_dir(tmp_path, "correctness", 1)[0] / "phase.json"
    record = json.loads(phase.read_text())
    entries = [
        {"correctness": correctness, "reference": {"usable": usable}}
        for correctness, usable in ledger
    ]
    record.update(status="failed", gates=gates, ledger=entries)
    phase.write_text(json.dumps(record))
    failed = verdict != "no reference"
    # A failed verdict is the RUNBOOK's "apply no reduction": nothing to adopt.
    assert variant_reduce.main(args) == (3 if failed else 0)
    assert (tmp_path / "reduced.yaml").exists() != failed
    warned = "do not apply this reduction" in capsys.readouterr().out
    report = (tmp_path / "report.md").read_text()
    assert f"- c: {verdict}, 1 winners" in report
    assert warned == ("WARNING: correctness attempts failed" in report) == failed
    assert ("not written" in report) == failed
    assert "; numerics checked |" not in report


def _fail_timing_round_two(tmp_path, args):
    """Mark round 2's timing phase of the sweep under ``tmp_path`` failed."""
    phase = _attempt_dir(tmp_path, "timing", 2)[0] / "phase.json"
    record = json.loads(phase.read_text())
    phase.write_text(json.dumps(dict(record, status="failed")))
    return args


def test_a_failed_timing_round_stops_the_reduction(tmp_path, capsys):
    """The noise floor is measured over fewer rounds than were swept, so the
    reduction is not one to adopt: the report is written, `--out` is not."""
    args = _fail_timing_round_two(tmp_path, build(tmp_path, rounds=(GRAPHS, GRAPHS)))
    assert variant_reduce.main(args) == 3
    captured = capsys.readouterr()
    assert "timing phases missing for (round, corpus) [(2, 'c')]" in captured.out
    assert "FAIL:" in captured.err
    assert not (tmp_path / "reduced.yaml").exists()
    report = (tmp_path / "report.md").read_text()
    assert "- missing (round, corpus): [(2, 'c')]" in report
    assert "output: not written" in report


def test_a_single_round_sweep_warns_that_its_noise_floor_is_assumed(tmp_path, capsys):
    assert variant_reduce.main(build(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "WARNING" in out and "0.05" in out


def test_two_rounds_do_not_warn_about_the_noise_floor():
    noise = variant_reduce.NoiseFloor(rounds=2, epsilon_min=0.1, drift=0.05)
    assert variant_reduce._single_round_warning(noise) is None
    assert variant_reduce._single_round_warning(variant_reduce.NoiseFloor(1, 0.05))


def test_a_sweep_with_no_timing_phase_exits_2(tmp_path, capsys):
    args = build(tmp_path, rounds=(), checked=("fp32", 128, 4))
    assert variant_reduce.main(args) == 2
    assert "no timing phase" in capsys.readouterr().err


def _retried_round_one(tmp_path, first, second):
    """Round 1's timing phase attempted twice; the later attempt has status `second`."""
    args = build(tmp_path)
    shutil.copytree(
        _attempt_dir(tmp_path, "timing", 1)[0],
        _attempt_dir(tmp_path, "timing", 1, retry=2)[0],
    )
    for retry, status, started in ((1, first, "01"), (2, second, "02")):
        path = _attempt_dir(tmp_path, "timing", 1, retry)[0] / "phase.json"
        record = json.loads(path.read_text())
        record.update(status=status, session={"started_at": f"2026-01-{started}"})
        path.write_text(json.dumps(record))
    return args


@pytest.mark.parametrize(
    "first, second, rc",
    [("failed", "success", 0), ("success", "failed", 2)],
    ids=["a-retry-rescues-the-phase", "a-failed-retry-supersedes-the-success"],
)
def test_the_latest_attempt_of_a_phase_is_the_one_read(tmp_path, first, second, rc):
    assert variant_reduce.main(_retried_round_one(tmp_path, first, second)) == rc


def test_a_graph_ranked_by_two_corpora_in_a_round_takes_the_later_corpus(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    ranked = {"g_perf": [(("fp32", 128, 1), 1.0), (("fp32", 128, 2), 1.5)]}
    build(other, rounds=(ranked,))
    args = build(tmp_path)
    attempts = tmp_path / "sweep" / "attempts"
    attempt = attempts / "timing__d__full__r1-0001"
    shutil.copytree(other / "sweep" / "attempts" / "timing__c__full__r1-0001", attempt)
    path = attempt / "phase.json"
    record = json.loads(path.read_text())
    record["phase"]["corpus"] = "d"
    path.write_text(json.dumps(record))
    reduction = variant_reduce.analyse(*args[1:6:2])
    assert reduction.duplicate_keys == 1
    (perf,) = [r for r in reduction.rankings if identity("g_perf") in r.key]
    assert (perf.corpus, perf.entries[0][0]) == ("d", kname("fp32", 128, 1))


def test_a_record_with_no_entries_is_skipped_not_fatal(tmp_path):
    def empty(line):
        if variant_reduce.graph_identity(line["graph"], "edit") == identity("g_noise"):
            line["entries"] = []

    args = build(tmp_path, empty)
    reduction = variant_reduce.analyse(*args[1:6:2])
    assert reduction.empty_records == 1
    assert len(reduction.rankings) == 2
    assert variant_reduce.main(args) == 0
    assert "records with no usable entry: 1" in (tmp_path / "report.md").read_text()


def test_equal_gains_take_the_smaller_kernel_name_first(tmp_path):
    kernels = [("fp32", 128, 2, True), ("fp32", 128, 1, False), ("fp32", 128, 4, False)]
    default = (("fp32", 128, 2), 1.2)
    ranked = {
        "g1": [(("fp32", 128, 4), 1.0), default],
        "g2": [(("fp32", 128, 1), 1.0), default],
    }
    args = build(tmp_path, rounds=(ranked,), kernels=kernels)
    reduction = variant_reduce.analyse(*args[1:6:2])
    names = [s.name for s in reduction.steps]
    assert names == [kname("fp32", 128, 1), kname("fp32", 128, 4)]
    assert reduction.steps[0].gain == reduction.steps[1].gain > 0


def test_a_gain_equal_to_epsilon_is_not_kept():
    ranking = variant_reduce.Ranking(1, "c", "g", [("a", 1.0)], ())
    steps = [variant_reduce.Step("a", 0.05, ranking)]
    assert variant_reduce.keep_at({"f": "floor"}, {}, steps, 0.05) == {"f"}
    assert variant_reduce.keep_at({"f": "floor"}, {}, steps, 0.049) == {"f", "a"}


def test_the_budget_is_met_by_an_equal_count(tmp_path):
    """Floor and coverage keep 5 kernels; g_perf's 50% win is the next one, so eps
    must reach 0.5 (a gain equal to eps is not kept) to fit a budget of 5."""
    assert variant_reduce.main(build(tmp_path) + ["--budget", "5"]) == 0
    kept = {k.name for k in load_config(tmp_path / "reduced.yaml").packs[0].kernels}
    assert len(kept) == 5 and kname("fp32", 128, 4) not in kept
    assert "chosen eps: 0.50" in (tmp_path / "report.md").read_text()


def test_the_noise_floor_is_the_median_spread_not_the_largest():
    """Per-kernel spreads over two rounds are 0, 0.4/2.2 and 4/6: eps_min is twice
    the middle one."""
    cell = ("fp32",)

    def ranking(number, *times):
        entries = list(zip("abc", times))
        return variant_reduce.Ranking(number, "c", "g", entries, cell)

    rounds = {1: {"g": ranking(1, 1.0, 2.0, 4.0)}, 2: {"g": ranking(2, 1.0, 2.4, 8.0)}}
    noise = variant_reduce.noise_floor(
        rounds,
        [],
        variant_reduce.DEFAULT_SINGLE_ROUND_EPSILON,
        variant_reduce.DEFAULT_DRIFT_MULTIPLIER,
    )
    assert noise.drift == pytest.approx(0.4 / 2.2)
    assert noise.epsilon_min == pytest.approx(0.8 / 2.2)
    assert (noise.rounds, noise.matched_keys, noise.unmatched_keys) == (2, 1, 0)


def test_a_rule_id_never_names_two_rules_across_budgets(tmp_path):
    """The proposal depends on `--budget`; `--apply-rules` ids must not."""
    args = build(tmp_path)
    seen = {}
    for budget in (5, 6, 7, 1000):
        reduction = variant_reduce.analyse(*args[1:6:2], budget=budget)
        for rule in reduction.rules:
            assert seen.setdefault(rule.id, rule.text) == rule.text


def test_a_subset_of_the_shipped_selection_writes_and_reloads(tmp_path):
    selection = (
        Path(__file__).resolve().parents[1] / "configs/gfx950_attention_dense.yaml"
    )
    shipped = load_config(selection).packs[0].kernels
    subset = {k.name: k for k in shipped[::3]}
    variant_reduce.write_selection(selection, list(subset), tmp_path / "reduced.yaml")
    reduced = load_config(tmp_path / "reduced.yaml").packs[0].kernels
    assert [k.name for k in reduced] == list(subset)
    for kernel in reduced:
        assert kernel.metadata == subset[kernel.name].metadata


def test_a_multi_launch_selection_writes_and_reloads_its_launches(tmp_path):
    """A role-only launch inherits each variant's spec. Written back as a full
    override, its spec differed per kernel, so the launches could not be
    factorised into kernel_defaults and the round trip failed."""
    shipped_path = (
        Path(__file__).resolve().parents[1] / "configs/gfx950_attention_dense.yaml"
    )
    raw = yaml.safe_load(shipped_path.read_text(encoding="utf-8"))
    raw["packs"][0]["kernel_defaults"]["launches"] = {
        "main": {},
        "merge": {"role": "merge"},
    }
    selection = tmp_path / "selection.yaml"
    selection.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    shipped = load_config(selection).packs[0].kernels
    subset = {k.name: k for k in shipped[::97]}
    assert len(subset) > 1

    variant_reduce.write_selection(selection, list(subset), tmp_path / "reduced.yaml")

    reduced = load_config(tmp_path / "reduced.yaml").packs[0].kernels
    assert [k.name for k in reduced] == list(subset)
    for kernel in reduced:
        assert (
            kernel.kernel_source.launches == subset[kernel.name].kernel_source.launches
        )


@pytest.mark.parametrize(
    "edit, reduction, extra, message",
    [
        (lambda d: d["entries"][0].update(pack_id="0"), True, [], "stale ranking"),
        (lambda d: d.update(v=1), True, [], "line format v=1"),
        (
            lambda d: d["graph"].update(content_base64="AAAA"),
            True,
            [],
            "not a Graph flatbuffer",
        ),
        (
            lambda d: d["entries"][0].update(score=1),
            True,
            [],
            "not in WinnerCacheFile.hpp",
        ),
        (None, False, [], "'reduction'"),
        (None, True, ["--apply-rules", "R99"], "R99"),
    ],
    ids=[
        "stale-pack-id",
        "line-v1",
        "bad-graph",
        "unknown-field",
        "no-reduction",
        "bad-rule",
    ],
)
def test_unusable_input_exits_2(tmp_path, capsys, edit, reduction, extra, message):
    assert variant_reduce.main(build(tmp_path, edit, reduction) + extra) == 2
    assert message in capsys.readouterr().err


def test_graph_identity_ignores_the_graph_id_and_nothing_else():
    """Rounds match on content: a re-minted id is the same graph, other bytes are not."""
    one, two = uuid.uuid4().bytes, uuid.uuid4().bytes
    same = variant_reduce.graph_identity(
        {"content_base64": graph_content("g", one)}, "x"
    )
    assert same == variant_reduce.graph_identity(
        {"content_base64": graph_content("g", two)}, "x"
    )
    assert same != variant_reduce.graph_identity(
        {"content_base64": graph_content("h", one)}, "x"
    )


def test_the_graph_id_slot_is_graph_fbs_field_order():
    """The zeroed slot is `id`'s position in `table Graph`, an inline 16-byte struct."""
    schema = Path(__file__).resolve().parents[3] / "flatbuffers_sdk/schemas/graph.fbs"
    text = schema.read_text(encoding="utf-8")
    body = re.search(r"^table Graph \{(.*?)^\}", text, re.S | re.M).group(1)
    fields = re.findall(r"^\s*(\w+)\s*:\s*(\w+|\[[^\]]+\])", body, re.M)
    assert [name for name, _ in fields].index("id") == variant_reduce._GRAPH_ID_FIELD
    assert dict(fields)["id"] == "Uuid"
    uuid_struct = re.search(r"^struct Uuid \{(.*?)^\}", text, re.S | re.M).group(1)
    assert "[ubyte:16]" in uuid_struct and variant_reduce._GRAPH_ID_BYTES == 16


# --- slotted engines: one reduction per slot, one selection out ------------------------

SLOTTED = Path(__file__).parent / "fixtures" / "slotted_three_slot.yaml"


def _slot_kernel(name, source, block_m, num_warps=None, default=False):
    metadata = {"block_m": block_m, "dtype": "BFLOAT16"}
    if num_warps is not None:
        metadata["num_warps"] = num_warps
    kernel = {
        "name": name,
        "kernel_source": {
            "kind": "embedded_source",
            "source_file": f"{source}.cpp",
            "entry_point": source,
        },
        "metadata": metadata,
    }
    if default:
        kernel["default"] = True
    return kernel


# dkdv: the default m64 plus three alternatives; dq has a pack per discriminator.
DKDV = [
    ("bwd.dkdv_m64", 64, 4, True),
    ("bwd.dkdv_m128", 128, 8, False),
    ("bwd.dkdv_m256", 256, 8, False),
    ("bwd.dkdv_m32", 32, 4, False),
]
SLOT_GRAPHS = {
    # dkdv: m128 beats the default by 50% on g1; m32 only wins g2 by 1%.
    "g1": {"dkdv": [("bwd.dkdv_m128", 1.0), ("bwd.dkdv_m64", 1.5)]},
    "g2": {"dkdv": [("bwd.dkdv_m32", 1.0), ("bwd.dkdv_m64", 1.01)]},
    "g3": {"prep": [("bwd.prep", 1.0)], "dq": [("bwd.dq_causal_m64", 1.0)]},
}


def _slotted_selection(tmp_path, dkdv_kernels=DKDV, dq_full_default=False, **reduction):
    raw = yaml.safe_load(SLOTTED.read_text())
    (dkdv,) = [p for p in raw["packs"] if p["name"] == "dkdv"]
    dkdv["kernels"] = [_slot_kernel(n, "Dkdv", m, w, d) for n, m, w, d in dkdv_kernels]
    (dq_full,) = [p for p in raw["packs"] if p["name"] == "dq_full"]
    dq_full["kernels"][0]["default"] = dq_full_default
    dq_full["kernels"].append(_slot_kernel("bwd.dq_full_m256", "DqFull", 256))
    raw["reduction"] = {
        "fields": {
            "feature": ["dtype"],
            "performance": [{"name": "block_m"}, {"name": "num_warps"}],
        },
        **reduction,
    }
    path = tmp_path / "selection.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def _slotted_sweep(
    tmp_path,
    generator,
    graphs=(SLOT_GRAPHS,),
    v2=False,
    checked=None,
    passed=True,
    extra_lines=(),
    **selection,
):
    """A slotted selection, its rendered KDPs and a `v:3` sweep of ``graphs`` rounds;
    returns argv. ``graphs`` maps a graph to ``{slot: [(kernel, ms), ...]}``.
    ``checked`` (same shape) adds a correctness phase that ``passed`` or failed;
    ``extra_lines`` are raw lines appended to each timing shard."""
    selection = _slotted_selection(tmp_path, **selection)
    config = load_config(selection)
    generator.render(config, tmp_path / "rendered")
    kdp_paths = sorted(
        (tmp_path / "rendered" / config.descriptor_dir).glob("*.kdp.json")
    )
    ids = {}
    for path in kdp_paths:
        kdp = json.loads(path.read_text())
        for d in kdp["kernelDescriptors"]:
            ids[d["name"]] = (kdp["id"], kdp["dispatch"], d["id"])
    program = [{"slot": "prep", "kernel_id": ids["bwd.prep"][2]}]
    written = []
    phases = [("timing", n, g) for n, g in enumerate(graphs, 1)]
    if checked:
        phases.append(("correctness", 1, checked))
    for kind, number, round_graphs in phases:
        graph_id = uuid.uuid4().bytes
        attempt, shard = _attempt_dir(tmp_path, kind, number)
        shard.mkdir(parents=True)
        phase = {"kind": kind, "corpus": "c", "arm": "full", "round": number}
        record = {"status": "success", "phase": phase, "ledger": []}
        if kind == "correctness":
            ok = {"correctness": {"passed": passed}, "reference": {"usable": True}}
            record.update(status="success" if passed else "failed", ledger=[ok])
            if not passed:
                record["gates"] = {"correctness": False}
        (attempt / "phase.json").write_text(json.dumps(record))
        written.append(phase)
        lines = ["1.0.0"]
        for graph, slots in round_graphs.items():
            base = {
                "graph": {"content_base64": graph_content(graph, graph_id)},
                "device": DEVICE,
            }
            for slot, ranked in slots.items():
                entries = [
                    {
                        "kernel_id": ids[k][2],
                        "pack_id": ids[k][0],
                        "dispatch_id": ids[k][1],
                        "time_ms": t,
                    }
                    for k, t in ranked
                ]
                if v2:
                    lines.append(json.dumps({"v": 2, **base, "entries": entries}))
                else:
                    lines.append(
                        json.dumps({"v": 3, **base, "slot": slot, "entries": entries})
                    )
            lines.append(
                json.dumps({"v": 3, **base, "program": program, "time_ms": 3.0})
            )
        if kind == "timing":
            lines += [json.dumps({"v": 3, **base, **extra}) for extra in extra_lines]
        (shard / "winners.jsonl").write_text("\n".join(lines) + "\n")
    _summary(tmp_path).write_text(json.dumps({"expected_phases": written}))
    argv = ["--selection", str(selection)]
    for path in kdp_paths:
        argv += ["--kdp", str(path)]
    return argv + [
        "--sweep-output", str(tmp_path / "sweep"),
        "--out", str(tmp_path / "reduced.yaml"),
        "--report", str(tmp_path / "report.md"),
    ]  # fmt: skip


def _kdps(argv):
    return [argv[i + 1] for i, a in enumerate(argv) if a == "--kdp"]


def _analyse_slots(argv, tmp_path, **kw):
    reductions = variant_reduce.analyse_slots(
        argv[1], _kdps(argv), tmp_path / "sweep", **kw
    )
    return {r.slot: r for r in reductions}


def test_a_slotted_sweep_reduces_each_slot_and_the_output_renders_and_verifies(
    tmp_path, generator, capsys
):
    argv = _slotted_sweep(tmp_path, generator, dq_full_default=True)
    assert variant_reduce.main(argv) == 0
    selection = load_config(tmp_path / "selection.yaml")
    reduced = load_config(tmp_path / "reduced.yaml")

    kept = {p.name: [k.name for k in p.kernels] for p in reduced.packs}
    assert kept == {
        "prep": ["bwd.prep"],
        # m128 wins g1 by 50%; m32's 1% win on g2 is under eps; m256 is never ranked.
        "dkdv": ["bwd.dkdv_m64", "bwd.dkdv_m128"],
        "dq_causal": ["bwd.dq_causal_m64"],
        "dq_full": ["bwd.dq_full_m128"],
    }
    # Every pack survives with its header, default markers, slots and reduction block.
    for before, after in zip(selection.packs, reduced.packs):
        assert (before.name, before.slot, before.arch, before.discriminator) == (
            after.name,
            after.slot,
            after.arch,
            after.discriminator,
        )
        originals = {k.name: k for k in before.kernels}
        for kernel in after.kernels:
            assert kernel.default == originals[kernel.name].default
            assert kernel.metadata == originals[kernel.name].metadata
    assert reduced.slots == selection.slots
    assert reduced.reduction == selection.reduction

    report = (tmp_path / "report.md").read_text()
    assert "## Slot `dkdv`" in report and "## Slot `dq`" in report
    assert "other slots at their reference picks" in report

    # generate.py consumes the reduced selection unchanged, and the gate accepts it.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import verify_variant_sets as gate_module

    out = tmp_path / "regenerated"
    generator.render(reduced, out)
    code = gate_module.main(
        ["--mode", "structural", "slotted", str(out / reduced.descriptor_dir)]
    )
    assert code == 0, capsys.readouterr().out


def test_each_slot_ranks_and_sets_its_noise_floor_on_its_own_records(
    tmp_path, generator
):
    """Two rounds: dkdv's kernel times move between them, prep's and dq's do not."""
    r2 = {
        "g1": {"dkdv": [("bwd.dkdv_m128", 1.0), ("bwd.dkdv_m64", 1.9)]},
        "g2": {"dkdv": [("bwd.dkdv_m32", 1.0), ("bwd.dkdv_m64", 1.3)]},
        "g3": SLOT_GRAPHS["g3"],
    }
    argv = _slotted_sweep(tmp_path, generator, graphs=(SLOT_GRAPHS, r2))
    by_slot = _analyse_slots(argv, tmp_path)
    assert list(by_slot) == ["prep", "dkdv", "dq"]
    assert by_slot["dkdv"].noise.drift > 0
    assert by_slot["prep"].noise.drift == by_slot["dq"].noise.drift == 0
    assert by_slot["dkdv"].noise.epsilon_min > by_slot["dq"].noise.epsilon_min
    # A slot's rankings hold only that slot's records, keyed by graph, device and slot.
    for slot, reduction in by_slot.items():
        assert {json.loads(r.key)[2] for r in reduction.rankings} == {slot}
        assert set(reduction.variants) <= {
            k.name
            for p in load_config(argv[1]).packs
            if p.slot == slot
            for k in p.kernels
        }
    # prep and dq never see dkdv's classification: it has no num_warps.
    assert by_slot["dq"].classification.performance == ("block_m",)
    assert by_slot["dkdv"].classification.performance == ("block_m", "num_warps")


def test_the_budget_bounds_each_pack_not_the_slot(tmp_path, generator, capsys):
    """dq keeps three kernels over two packs (the pack `dq_full` holds two), so a
    budget of 2 holds. A budget of 1 refuses dq, whose floor and coverage leave a pack
    two; dkdv raises its eps to fit and prep has one kernel."""
    argv = _slotted_sweep(tmp_path, generator)
    by_slot = _analyse_slots(argv, tmp_path)
    dq = by_slot["dq"]
    assert len(dq.keep(dq.epsilon)) == 3
    assert variant_reduce.instances(dq.variants, dq.keep(dq.epsilon)) == 2

    assert variant_reduce.main(argv + ["--budget", "2"]) == 0
    assert variant_reduce.main(argv + ["--budget", "1"]) == 1
    err = capsys.readouterr().err
    assert "slot dq: floor + coverage exceed budget: 2 instances > 1" in err
    assert "slot dkdv" not in err and "slot prep" not in err
    assert "WARNING" not in err
    assert "- budget 1 instances per pack per arch;" in (
        (tmp_path / "report.md").read_text()
    )


def test_a_failed_slotted_timing_round_stops_the_reduction(tmp_path, generator, capsys):
    argv = _slotted_sweep(tmp_path, generator, graphs=(SLOT_GRAPHS, SLOT_GRAPHS))
    _fail_timing_round_two(tmp_path, argv)
    assert variant_reduce.main(argv) == 3
    assert "FAIL:" in capsys.readouterr().err
    assert not (tmp_path / "reduced.yaml").exists()
    report = (tmp_path / "report.md").read_text().splitlines()
    assert "- missing (round, corpus): [(2, 'c')]" in report
    assert "output: not written" in report[2]


def test_a_slot_with_no_winners_in_a_timing_phase_is_an_incomplete_sweep(
    tmp_path, generator, capsys
):
    graphs = {"g3": {"prep": SLOT_GRAPHS["g3"]["prep"]}}
    argv = _slotted_sweep(tmp_path, generator, graphs=(graphs,))
    assert variant_reduce.main(argv) == 3
    assert "slot dkdv has no winners" in capsys.readouterr().err
    assert not (tmp_path / "reduced.yaml").exists()


# Two dkdv rules, each within eps alone, that together drop both kernels g_alpha
# and g_beta rank best: `block_m` removes m128, `num_warps` removes m64_nw8.
UNION_DKDV = [
    ("bwd.dkdv_m64", 64, 4, True),
    ("bwd.dkdv_m128", 128, 4, False),
    ("bwd.dkdv_m64_nw8", 64, 8, False),
]
UNION_SLOT_GRAPHS = {
    "g_alpha": {
        "dkdv": [
            ("bwd.dkdv_m128", 1.0),
            ("bwd.dkdv_m64_nw8", 1.01),
            ("bwd.dkdv_m64", 2.0),
        ]
    },
    "g_beta": {
        "dkdv": [
            ("bwd.dkdv_m64_nw8", 1.0),
            ("bwd.dkdv_m128", 1.01),
            ("bwd.dkdv_m64", 2.0),
        ]
    },
    "g3": SLOT_GRAPHS["g3"],
}


def _union_slot_rules(tmp_path, generator):
    argv = _slotted_sweep(
        tmp_path, generator, graphs=(UNION_SLOT_GRAPHS,), dkdv_kernels=UNION_DKDV
    )
    dkdv = _analyse_slots(argv, tmp_path)["dkdv"]
    rules = {r.text: r for r in dkdv.rules}
    return argv, dkdv, rules["drop knob block_m"], rules["drop knob num_warps"]


def test_applied_rules_losing_more_than_eps_in_a_slot_are_refused(
    tmp_path, generator, capsys
):
    argv, dkdv, block_m, num_warps = _union_slot_rules(tmp_path, generator)
    assert max(block_m.worst_gain_lost, num_warps.worst_gain_lost) <= dkdv.rules_epsilon
    chosen = f"{block_m.id},{num_warps.id}"
    assert variant_reduce.main(argv + ["--apply-rules", chosen]) == 1
    err = capsys.readouterr().err
    assert "slot dkdv: " in err
    assert (
        "the loss bounds each slot with the other slots at their reference picks" in err
    )
    assert not (tmp_path / "reduced.yaml").exists()
    report = (tmp_path / "report.md").read_text()
    for graph in ("g_alpha", "g_beta"):
        assert variant_reduce.graph_label(identity(graph)) in report
    assert "@gfx950/dkdv" in report

    # One of them alone stays within eps: it is written, and the other slots keep
    # every kernel (no rule of theirs was chosen).
    assert variant_reduce.main(argv + ["--apply-rules", block_m.id]) == 0
    kept = {
        p.name: [k.name for k in p.kernels]
        for p in load_config(tmp_path / "reduced.yaml").packs
    }
    assert kept["dkdv"] == ["bwd.dkdv_m64", "bwd.dkdv_m64_nw8"]
    assert kept["dq_full"] == ["bwd.dq_full_m128", "bwd.dq_full_m256"]


def test_two_slots_proposing_one_rule_text_get_distinct_ids(tmp_path, generator):
    """`drop block_m=256` is proposed by dkdv and by dq; applying dq's id leaves dkdv's
    kernels untouched."""
    argv = _slotted_sweep(tmp_path, generator, dq_full_default=True)
    by_slot = _analyse_slots(argv, tmp_path)
    text = "drop block_m=256"
    dkdv, dq = (
        {r.text: r for r in by_slot[slot].rules}[text] for slot in ("dkdv", "dq")
    )
    assert dkdv.id != dq.id
    assert variant_reduce.main(argv + ["--apply-rules", dq.id]) == 0
    kept = {
        p.name: [k.name for k in p.kernels]
        for p in load_config(tmp_path / "reduced.yaml").packs
    }
    assert kept["dq_full"] == ["bwd.dq_full_m128"]
    assert kept["dkdv"] == [name for name, *_ in DKDV]
    assert kept["dq_causal"] == ["bwd.dq_causal_m64"]


def test_a_slotted_selection_needs_one_kdp_per_pack(tmp_path, generator, capsys):
    argv = _slotted_sweep(tmp_path, generator)
    first = argv.index("--kdp")
    short = argv[:first] + argv[first + 2 :]
    assert variant_reduce.main(short) == 2
    assert "pack(s) need" in capsys.readouterr().err


def test_a_slotted_sweep_of_whole_graph_records_is_refused(tmp_path, generator, capsys):
    argv = _slotted_sweep(tmp_path, generator, v2=True)
    assert variant_reduce.main(argv) == 2
    assert "whole-graph v2 record in the sweep of a slotted engine" in (
        capsys.readouterr().err
    )


def test_an_unslotted_sweep_of_slot_records_is_refused(tmp_path, capsys):
    edit = lambda d: d.update(v=3, slot="s")  # noqa: E731
    assert variant_reduce.main(build(tmp_path, edit)) == 2
    assert "slot record ('s') in the sweep of an unslotted engine" in (
        capsys.readouterr().err
    )


def test_a_slot_the_selection_lacks_is_refused(tmp_path, generator, capsys):
    graphs = {
        "g1": {**SLOT_GRAPHS["g1"], "extra": [("bwd.dkdv_m64", 1.0)]},
        "g3": SLOT_GRAPHS["g3"],
    }
    argv = _slotted_sweep(tmp_path, generator, graphs=(graphs,))
    assert variant_reduce.main(argv) == 2
    assert "slot 'extra' is not a slot of the selection" in capsys.readouterr().err


def test_a_ranked_kernel_of_another_slot_is_refused(tmp_path, generator, capsys):
    graphs = {"g1": {"dkdv": [("bwd.prep", 1.0)]}, "g3": SLOT_GRAPHS["g3"]}
    argv = _slotted_sweep(tmp_path, generator, graphs=(graphs,))
    assert variant_reduce.main(argv) == 2
    assert "is not a kernel of slot dkdv" in capsys.readouterr().err


def _shard(tmp_path, *docs):
    base = {
        "graph": {"content_base64": graph_content("g", bytes(16))},
        "device": DEVICE,
    }
    entry = [{"kernel_id": "k", "pack_id": "p", "dispatch_id": "d", "time_ms": 1.0}]
    path = tmp_path / "winners.jsonl"
    lines = [json.dumps({**base, "entries": entry, **doc}) for doc in docs]
    path.write_text("1.0.0\n" + "\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize(
    "doc, message",
    [
        ({"v": 2, "slot": "s"}, r"\['slot'\] not in WinnerCacheFile.hpp"),
        ({"v": 3}, r"missing \['slot'\]"),
        ({"v": 3, "slot": ""}, "'slot' must be a non-empty string"),
        (
            {"v": 3, "slot": "s", "program": []},
            r"\['slot'\] not in WinnerCacheFile.hpp",
        ),
        ({"v": 3, "program": [], "time_ms": 1.0}, "'program' must be a non-empty list"),
        (
            {"v": 3, "program": [{"slot": "s"}], "time_ms": 1.0},
            r"missing \['kernel_id'\]",
        ),
        (
            {"v": 3, "program": [{"slot": "s", "kernel_id": "k"}], "time_ms": -1},
            "bad time_ms",
        ),
        ({"v": 4, "slot": "s"}, r"line format v=4, expected one of \[2, 3\]"),
        (
            {"v": 2, "program": [{"slot": "s", "kernel_id": "k"}], "time_ms": 1.0},
            r"\['program', 'time_ms'\] not in WinnerCacheFile.hpp",
        ),
        (
            {"v": 3, "program": [{"slot": "", "kernel_id": "k"}], "time_ms": 1.0},
            "slot and kernel_id must be non-empty strings",
        ),
        (
            {"v": 3, "program": [{"slot": "s", "kernel_id": ""}], "time_ms": 1.0},
            "slot and kernel_id must be non-empty strings",
        ),
        (
            {
                "v": 3,
                "slot": "s",
                "entries": [
                    {
                        "kernel_id": "k",
                        "pack_id": "p",
                        "dispatch_id": "d",
                        "time_ms": -1.0,
                    }
                ],
            },
            "entries.0.: bad time_ms",
        ),
    ],
    ids=[
        "v2-with-a-slot",
        "v3-without-a-slot",
        "empty-slot",
        "slot-and-program",
        "empty-program",
        "pick-without-kernel",
        "negative-program-time",
        "unknown-version",
        "v2-with-a-program",
        "pick-with-an-empty-slot",
        "pick-with-an-empty-kernel",
        "negative-entry-time",
    ],
)
def test_a_malformed_slotted_line_is_refused(tmp_path, doc, message):
    with pytest.raises(variant_reduce.ReduceError, match=message):
        variant_reduce.read_shard(_shard(tmp_path, doc))


def test_a_program_line_is_read_and_ranks_nothing(tmp_path):
    program = {
        "v": 3,
        "program": [{"slot": "s0", "kernel_id": "k"}],
        "time_ms": 2.0,
        "entries": [],
    }
    path = _shard(tmp_path, {"v": 3, "slot": "s0"}, program)
    (slot_line, program_line) = variant_reduce.read_lines(path)
    assert slot_line.slot == "s0" and slot_line.picks is None
    assert program_line.picks == (("s0", "k"),) and program_line.time_ms == 2.0
    (key,) = variant_reduce.read_shard(path)
    assert json.loads(key)[2] == "s0"


def test_a_v2_and_a_slot_record_of_one_graph_are_two_rankings(tmp_path):
    path = _shard(tmp_path, {"v": 2}, {"v": 3, "slot": "a"}, {"v": 3, "slot": "b"})
    keys = [json.loads(k) for k in variant_reduce.read_shard(path)]
    assert [k[2:] for k in keys] == [[], ["a"], ["b"]]


def test_an_unranked_pack_without_a_default_keeps_a_kernel(tmp_path, generator):
    """dq's pack `dq_full` is never ranked and declares no default. With `block_m`
    dividing, its m256 divides out of the floor beside m128 only within its own pack: a
    cell shared with `dq_causal` (m64) would floor both away and empty the pack."""
    fields = {
        "feature": ["dtype"],
        "performance": [
            {"name": "block_m", "divides": "block_m"},
            {"name": "num_warps"},
        ],
    }
    argv = _slotted_sweep(tmp_path, generator, fields=fields)
    assert variant_reduce.main(argv) == 0
    reduced = load_config(tmp_path / "reduced.yaml")
    kept = {p.name: [k.name for k in p.kernels] for p in reduced.packs}
    assert kept["dq_full"] == ["bwd.dq_full_m128"]
    assert kept["dq_causal"] == ["bwd.dq_causal_m64"]


CHECKED = {
    "g1": {"dkdv": [("bwd.dkdv_m128", 1.0)]},
    "g3": {"prep": [("bwd.prep", 1.0)], "dq": [("bwd.dq_causal_m64", 1.0)]},
}


def test_a_passing_slotted_correctness_phase_tags_the_winners_as_checked(
    tmp_path, generator
):
    argv = _slotted_sweep(tmp_path, generator, checked=CHECKED)
    assert variant_reduce.main(argv) == 0
    report = (tmp_path / "report.md").read_text()
    assert "- c: passed, 3 winners" in report
    rows = {line.split(" | ")[0][2:]: line for line in report.splitlines()}
    for name in ("bwd.dkdv_m128", "bwd.prep", "bwd.dq_causal_m64"):
        assert rows[name].endswith("; numerics checked |")
    assert rows["bwd.dkdv_m64"].endswith("; numerics unchecked |")


def test_a_failing_slotted_correctness_phase_exits_3(tmp_path, generator, capsys):
    argv = _slotted_sweep(tmp_path, generator, checked=CHECKED, passed=False)
    assert variant_reduce.main(argv) == 3
    assert "correctness attempts failed" in capsys.readouterr().err
    assert not (tmp_path / "reduced.yaml").exists()
    assert "- c: FAILED; investigate" in (tmp_path / "report.md").read_text()


def test_a_program_line_in_an_unslotted_sweep_is_refused(tmp_path, capsys):
    def program(line):
        del line["entries"]
        line.update(v=3, program=[{"slot": "s", "kernel_id": "k"}], time_ms=1.0)

    assert variant_reduce.main(build(tmp_path, program)) == 2
    assert "a program line in the sweep of an unslotted engine" in (
        capsys.readouterr().err
    )


def test_a_program_pick_naming_a_slot_the_selection_lacks_is_refused(
    tmp_path, generator, capsys
):
    picks = [{"slot": "prep", "kernel_id": "k"}, {"slot": "nope", "kernel_id": "k"}]
    pick = {"program": picks, "time_ms": 1.0}
    argv = _slotted_sweep(tmp_path, generator, extra_lines=[pick])
    assert variant_reduce.main(argv) == 2
    assert "a program line names slot 'nope'" in capsys.readouterr().err
