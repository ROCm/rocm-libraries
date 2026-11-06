# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bundle selection by per-tier pair-coverage target (select_by_target.py).

Most tests build a few rows in memory with hand-written pairs (bit numbers) and folders.
The folder tests take the folders from topology.py through select_by_target.assign_folders.
"""
import json
import random
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import select_by_target as sbt  # noqa: E402
import select_rows  # noqa: E402

HEADER = "case_key,level,origin"


def mask(bits):
    m = 0
    for b in bits:
        m |= 1 << b
    return m


def row(key, folder, bits, origin="low-change", level="quick", est="1.0"):
    return {
        "case_key": key,
        "level": level,
        "origin": origin,
        "detail": "",
        "est_seconds": est,
        "est_seconds_gfx950": est,
        "_folder": folder,
        "_pairs": mask(bits),
        "_line": f"{key},{level},{origin}",
    }


def selected_keys(sel):
    return {r["case_key"] for name in sel["order"] for r in sel["folders"][name]}


# Forcing


@pytest.mark.parametrize("origin", sbt.HAND_PICKED)
def test_hand_picked_row_forces_its_folder_without_new_pairs(origin):
    rows = [
        row("a1", "A", [0, 1, 2]),
        row("z1", "Z", [0], origin=origin),
    ]
    sel = sbt.select_tier(rows, 100)
    assert sel["why"] == {"Z": "forced", "A": "greedy"}
    assert sel["order"] == ["Z", "A"]


@pytest.mark.parametrize("origin", ["low-change", "pairwise", "size-sweep", "coverage"])
def test_other_origin_does_not_force(origin):
    rows = [
        row("a1", "A", [0, 1, 2]),
        row("z1", "Z", [0], origin=origin),
    ]
    sel = sbt.select_tier(rows, 100)
    assert sel["order"] == ["A"]


def test_hand_picked_row_that_is_not_bundle_expressible_is_reported_not_forced():
    rows = [row("a1", "A", [0, 1])]
    out = row("x1", "-", [], origin="targeted")
    out["_reasons"] = ["ragged_offsets"]
    selections = sbt.select_all(rows, {"quick": [out]})
    sel = selections["quick"]
    assert sel["order"] == ["A"]
    assert "x1" not in selected_keys(sel)
    reported = sel["hand_picked"]["not_bundle_expressible"]
    assert reported == [
        {
            "case_key": "x1",
            "origin": "targeted",
            "detail": "",
            "reasons": ["ragged_offsets"],
        }
    ]
    assert sel["hand_picked"]["per_origin"]["targeted"]["not_bundle_expressible"] == 1


# Target per tier


@pytest.mark.parametrize("level", ["quick", "standard", "comprehensive"])
def test_full_coverage_tiers_reach_every_reachable_pair(level):
    rows = [
        row("a1", "A", [0, 1, 2, 3], level=level),
        row("b1", "B", [3, 4], level=level),
        row("c1", "C", [5], level=level),
        row("d1", "D", [1, 2], level=level),
    ]
    sel = sbt.select_tier(rows, sbt.TARGET_PERCENT[level])
    f = sel["figures"]
    assert f["reachable_pairs"] == 6
    assert f["pairs_reached"] == 6
    assert f["shortfall"] == 0
    assert "D" not in sel["order"]


def test_full_tier_reaches_ninety_percent_with_no_extra_folder():
    rows = [
        row("f1", "F1", range(6), level="full"),
        row("f2", "F2", [6, 7], level="full"),
        row("f3", "F3", [8], level="full"),
        row("f4", "F4", [9], level="full"),
    ]
    sel = sbt.select_tier(rows, sbt.TARGET_PERCENT["full"])
    f = sel["figures"]
    assert f["reachable_pairs"] == 10
    assert f["pairs_reached"] * 100 >= 90 * f["reachable_pairs"]
    assert sel["order"] == ["F1", "F2", "F3"]
    without_last = 0
    for name in sel["order"][:-1]:
        for r in sel["folders"][name]:
            without_last |= r["_pairs"]
    assert sbt.popcount(without_last) * 100 < 90 * f["reachable_pairs"]


def test_forced_folders_alone_can_meet_the_target():
    rows = [
        row("f1", "F1", range(9), origin="real_world", level="full"),
        row("f2", "F2", [9], level="full"),
    ]
    sel = sbt.select_tier(rows, 90)
    assert sel["order"] == ["F1"]
    assert sel["figures"]["folders_greedy"] == 0


# Determinism and ties


def test_shuffled_inputs_give_byte_identical_csv(tmp_path):
    base = [
        row("k01", "A", [0, 1], level="quick"),
        row("k02", "A", [2], level="quick"),
        row("k03", "B", [1, 3], origin="targeted", level="quick"),
        row("k04", "C", [0, 4], level="standard"),
        row("k05", "D", [4, 5], level="standard"),
        row("k06", "D", [5], level="standard"),
        row("k07", "E", [6, 7], level="full"),
        row("k08", "F", [8], level="full"),
    ]
    outputs = []
    for seed in (1, 2, 3, 3):
        rows = [dict(r) for r in base]
        random.Random(seed).shuffle(rows)
        path = tmp_path / f"out-{len(outputs)}.csv"
        sbt.write_selected(path, HEADER, sbt.select_all(rows, {}))
        outputs.append(path.read_bytes())
    assert len(set(outputs)) == 1
    lines = outputs[0].decode().splitlines()
    assert lines[0] == HEADER + ",selection_folder,selection_why,selection_subset_kept"
    assert "k03,quick,targeted,B,forced,yes" in lines


def test_equal_gain_folders_go_to_the_smaller_name():
    rows = [
        row("x1", "Beta", [0]),
        row("y1", "Alpha", [1]),
        row("z1", "Gamma", [0, 1]),
    ]
    sel = sbt.select_tier(rows, 100)
    assert sel["order"] == ["Gamma"]
    rows = [row("x1", "Beta", [0]), row("y1", "Alpha", [0])]
    sel = sbt.select_tier(rows, 100)
    assert sel["order"] == ["Alpha"]


def test_equal_gain_tie_holds_after_a_gain_drops():
    rows = [
        row("f1", "First", range(6)),
        row("z1", "Zeta", [0, 1, 6]),
        row("a1", "Alpha", [7]),
    ]
    sel = sbt.select_tier(rows, 100)
    assert sel["order"] == ["First", "Alpha", "Zeta"]


def test_equal_gain_rows_in_subset_mode_go_to_the_smaller_case_key():
    rows = [row("bb", "A", [0, 1]), row("aa", "A", [0, 1]), row("cc", "A", [1])]
    sel = sbt.select_tier(rows, 100)
    assert sel["kept"] == {"aa"}
    assert sel["figures"]["cases_whole"] == 3
    assert sel["figures"]["cases_subset"] == 1


# Subset mode


def test_subset_keeps_hand_picked_rows_and_drops_rows_that_add_nothing():
    rows = [
        row("a1", "A", [0, 1, 2]),
        row("a2", "A", [1], origin="one-at-a-time"),
        row("a3", "A", [2]),
        row("b1", "B", [2, 3]),
        row("b2", "B", [0]),
    ]
    sel = sbt.select_tier(rows, 100)
    assert sel["order"] == ["A", "B"]
    assert sel["kept"] == {"a1", "a2", "b1"}
    f = sel["figures"]
    assert f["pairs_subset"] == f["pairs_whole"] == 4
    assert f["cases_whole"] == 5


def test_subset_counts_pairs_of_earlier_folders_as_covered():
    rows = [
        row("a1", "A", [0, 1, 2]),
        row("b1", "B", [0, 3]),
        row("b2", "B", [1]),
    ]
    sel = sbt.select_tier(rows, 100)
    assert sel["order"] == ["A", "B"]
    assert sel["kept"] == {"a1", "b1"}
    assert sel["figures"]["folders_keeping_one_case_subset"] == 2


# Lane times


def test_lane_over_budget_and_over_working_target():
    selections = {
        "quick": sbt.select_tier([row("q1", "A", [0], est="200")], 100),
        "standard": sbt.select_tier(
            [row("s1", "A", [0], level="standard", est="2000")], 100
        ),
    }
    lanes = sbt.lane_figures(selections)
    assert lanes["quick"]["status"] == "over_working_target"
    assert lanes["standard"]["lane_seconds_gfx942"] == 2200.0
    assert lanes["standard"]["status"] == "over_budget"


# Superset folders from topology.py

CLASS_VALUES = {
    "ragged_tensors": {"padded": []},
    "paged_sides": {"present": ["K", "V"]},
    "page_tables": {"separate": 2, "shared": 1},
    "packed_page_table": {"dense": False, "packed": True},
    "head_dim": {"d64": 64},
}

BASE_ROW = {
    "level": "standard",
    "case_key": "",
    "origin": "low-change",
    "batch_packing": "padded",
    "paged_kv": "absent",
    "page_table_storage": "-",
    "page_table_sharing": "-",
    "sequence_lengths": "uniform",
    "block_mask": "absent",
    "bias": "absent",
    "attention_sinks": "absent",
    "scale_source": "attribute_value",
    "dropout": "absent",
    "dropout_rng_state": "-",
    "dropout_pattern_output": "-",
    "stats_output": "not_requested",
    "max_sumexp_output": "not_requested",
    "amax_s": "not_requested",
    "amax_o": "not_requested",
    "descale_q": "absent",
    "descale_k": "absent",
    "descale_v": "absent",
    "s_quantization": "absent",
    "scale_o": "absent",
}


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "class-to-value.json"
    path.write_text(json.dumps(CLASS_VALUES))
    return select_rows.Model(path)


def model_row(key, **changes):
    r = dict(BASE_ROW, case_key=key)
    r.update(changes)
    return r


def test_nested_signatures_share_one_folder_and_exclusive_forms_do_not(model):
    rows = [
        model_row("base"),
        model_row("stats", stats_output="requested"),
        model_row("stats_bias", stats_output="requested", bias="present"),
        model_row("runtime", scale_source="runtime_parameter"),
        model_row("runtime_bias", scale_source="runtime_parameter", bias="present"),
    ]
    sbt.assign_folders(rows, model)
    folder = {r["case_key"]: r["_folder"] for r in rows}
    assert folder["base"] == folder["stats"] == folder["stats_bias"] == "Bias_Stats"
    assert folder["runtime"] == folder["runtime_bias"] == "Bias_ScRt"
    sbt.assign_folders(rows, model, exact=True)
    assert len({r["_folder"] for r in rows}) == 5


def test_folders_are_per_tier(model):
    rows = [
        model_row("q_base", level="quick"),
        model_row("s_base"),
        model_row("s_bias", bias="present"),
    ]
    sbt.assign_folders(rows, model)
    folder = {r["case_key"]: r["_folder"] for r in rows}
    assert folder["q_base"] == "Base"
    assert folder["s_base"] == folder["s_bias"] == "Bias"


def test_whole_folders_keep_their_folder_when_only_selected_rows_are_grouped(model):
    rows = [
        model_row("big", stats_output="requested", bias="present"),
        model_row("mid", stats_output="requested"),
        model_row("other", bias="present", attention_sinks="present"),
        model_row("small"),
    ]
    sbt.assign_folders(rows, model)
    by_folder = {}
    for r in rows:
        by_folder.setdefault(r["_folder"], []).append(r)
    assert len(by_folder) == 2
    for members in by_folder.values():
        sbt.check_regroup(members)
    part = [r for r in rows if r["case_key"] != "other"]
    with pytest.raises(SystemExit):
        sbt.check_regroup(part)


# Pairs from the model


def test_row_mask_holds_exactly_the_valid_pairs_of_the_row():
    from sdpa_case_flow_poc.model import load_model
    from sdpa_case_flow_poc.pairs import (
        axis_pos,
        candidate_pairs,
        feasible_pairs,
        make_pair,
    )
    from sdpa_case_flow_poc.solver import Solver

    index = sbt.PairIndex(select_rows.PLAN_DIR)
    model = load_model(select_rows.PLAN_DIR)
    valid = {
        ((a, str(x)), (b, str(y)))
        for (a, x), (b, y) in feasible_pairs(Solver(model), candidate_pairs(model))
    }
    assert index.valid_pairs == len(valid)
    pos = axis_pos(model)
    _, _, rows = select_rows.read_rows(select_rows.PLAN_DIR / "levels.csv")
    for r in rows[:3] + rows[-3:]:
        feats = [(a, r[a]) for a in model.axes if r[a] not in ("", "-")]
        want = {
            make_pair(pos, feats[i], feats[j])
            for i in range(len(feats))
            for j in range(i + 1, len(feats))
        } & valid
        assert sbt.popcount(index.row_mask(r)) == len(want)
        assert len(want) > 0


# Ship-all tiers


def test_ship_all_tier_selects_every_folder_and_other_tiers_keep_the_target():
    rows = [
        row("a1", "A", [0, 1, 2]),
        row("b1", "B", [0]),
        row("h1", "H", [1], origin="targeted"),
        row("s1", "SA", [0, 1], level="standard"),
        row("s2", "SB", [0], level="standard"),
    ]
    selections = sbt.select_all(rows, {}, ship_all_tiers=("quick",))
    quick = selections["quick"]
    assert quick["order"] == ["H", "A", "B"]
    assert quick["why"] == {"H": "forced", "A": "ship_all", "B": "ship_all"}
    assert quick["figures"]["folders_greedy"] == 0
    assert quick["figures"]["cases_whole"] == 3
    assert quick["figures"]["pairs_subset"] == quick["figures"]["pairs_whole"] == 3
    assert selections["standard"]["order"] == ["SA"]
    default = sbt.select_all(rows, {})
    assert default["quick"]["order"] == ["H", "A"]


# select_rows.py output checks


def write_select_dir(tmp_path, selected_lines, exclusions):
    levels = tmp_path / "levels.csv"
    levels.write_text(HEADER + "\nk1,quick,low-change\nk2,quick,targeted\n")
    out = tmp_path / "select"
    out.mkdir()
    (out / "summary.json").write_text(
        json.dumps({"input_sha256": select_rows.sha256(levels)})
    )
    (out / "selected.csv").write_text(
        HEADER + "\n" + "".join(line + "\n" for line in selected_lines)
    )
    (out / "exclusions.csv").write_text(
        "case_key,level,reasons\n" + "".join(f"{k},quick,{w}\n" for k, w in exclusions)
    )
    header, lines, rows = select_rows.read_rows(levels)
    return out, levels, lines, rows


def test_select_dir_is_read_when_it_matches_the_input(tmp_path):
    out, levels, lines, rows = write_select_dir(
        tmp_path, ["k1,quick,low-change"], [("k2", "ragged_offsets")]
    )
    selected, excluded, _ = sbt.read_select_dir(out, levels, lines, rows)
    assert selected == {"k1"}
    assert excluded == {"k2": ["ragged_offsets"]}


@pytest.mark.parametrize(
    "selected_lines, exclusions",
    [
        (["k1,quick,low-change"], [("k2", "x"), ("k9", "x")]),
        (["k1,quick,low-change", "k1,quick,low-change"], [("k2", "x")]),
        (["k1,quick,low-change"], []),
        (["k1,quick,low-change"], [("k1", "x"), ("k2", "x")]),
        (["k1,quick,other"], [("k2", "x")]),
    ],
)
def test_select_dir_that_does_not_match_the_input_stops(
    tmp_path, selected_lines, exclusions
):
    out, levels, lines, rows = write_select_dir(tmp_path, selected_lines, exclusions)
    with pytest.raises(SystemExit):
        sbt.read_select_dir(out, levels, lines, rows)
