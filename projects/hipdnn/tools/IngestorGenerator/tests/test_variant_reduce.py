# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tools/variant_reduce.py on a tiny selection, KDP and sweep output built per test."""

import json
import sys
from pathlib import Path

import pytest
import yaml

from codegen.config_loader import load_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import variant_reduce  # noqa: E402

PACK_ID = "3d3ae611-244d-558b-958c-bd385a92913a"
DISPATCH_ID = "451e1a7b-429c-5d37-85b3-afc5bbdbbdb0"
DEVICE = {"gcn_arch_name": "gfx950", "warp_size": 64, "multi_processor_count": 256}

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
    for kind, number, graphs in phases:
        attempt = tmp_path / "sweep" / "attempts" / f"{kind}__c__full__r{number}-0001"
        shard = attempt / "winners" / "ingestor-winners" / "1.0.0" / "e" / "gfx950"
        shard.mkdir(parents=True)
        phase = {"kind": kind, "corpus": "c", "arm": "full", "round": number}
        ledger = [{"correctness": {"passed": True}}]
        record = {"status": "success", "phase": phase, "ledger": ledger}
        (attempt / "phase.json").write_text(json.dumps(record))
        ids = {"pack_id": PACK_ID, "dispatch_id": DISPATCH_ID}
        lines = ["1.0.0"]
        for graph, ranked in graphs.items():
            entries = [
                {"kernel_id": f"id-{kname(*k)}", **ids, "time_ms": t} for k, t in ranked
            ]
            line = {"v": 1, "graph": {"key": graph}, "device": DEVICE}
            line["entries"] = entries
            if edit:
                edit(line)
            lines.append(json.dumps(line))
        (shard / "winners.jsonl").write_text("\n".join(lines) + "\n")
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


def test_a_zero_time_entry_is_a_ranking(tmp_path):
    edit = lambda d: d["entries"][0].update(time_ms=0)  # noqa: E731
    assert variant_reduce.main(build(tmp_path, edit)) == 0


@pytest.mark.parametrize(
    "ledger, gates, verdict",
    [
        ({"passed": False}, {"correctness": False}, "FAILED; investigate"),
        (
            {"passed": False, "tolerance_match": None},
            {"reference": False},
            "no reference",
        ),
    ],
    ids=["numerics-failed", "no-reference"],
)
def test_correctness_attempt_verdicts(tmp_path, capsys, ledger, gates, verdict):
    args = build(tmp_path, checked=("fp32", 128, 4))
    phase = (
        tmp_path / "sweep" / "attempts" / "correctness__c__full__r1-0001" / "phase.json"
    )
    record = json.loads(phase.read_text())
    usable = "reference" not in gates
    entry = {"correctness": ledger, "reference": {"usable": usable}}
    record.update(status="failed", gates=gates, ledger=[entry])
    phase.write_text(json.dumps(record))
    assert variant_reduce.main(args) == 0
    warned = "do not apply this reduction" in capsys.readouterr().out
    report = (tmp_path / "report.md").read_text()
    assert f"- c: {verdict}, 1 winners" in report
    assert warned == ("WARNING" in report) == (verdict != "no reference")
    assert "; numerics checked |" not in report


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


@pytest.mark.parametrize(
    "edit, reduction, extra, message",
    [
        (lambda d: d["entries"][0].update(pack_id="0"), True, [], "stale ranking"),
        (lambda d: d.update(v=2), True, [], "line format v=2"),
        (
            lambda d: d["entries"][0].update(score=1),
            True,
            [],
            "not in WinnerCacheFile.hpp",
        ),
        (None, False, [], "'reduction'"),
        (None, True, ["--apply-rules", "R99"], "R99"),
    ],
    ids=["stale-pack-id", "line-v2", "unknown-field", "no-reduction", "bad-rule"],
)
def test_unusable_input_exits_2(tmp_path, capsys, edit, reduction, extra, message):
    assert variant_reduce.main(build(tmp_path, edit, reduction) + extra) == 2
    assert message in capsys.readouterr().err
