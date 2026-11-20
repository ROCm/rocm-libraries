# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The renderer's floor-twin rule: a pin is not installed when another pin of the same object serves its head dims."""

from __future__ import annotations

import sys
from pathlib import Path

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL / "tools"))
import sdpa_bwd_tables_from_sweep as tool  # noqa: E402
from generators._sdpa_bwd_spec import Spec  # noqa: E402

SHA = "a" * 64


def pin(spec: str, sha: str = SHA) -> dict:
    parsed = Spec.parse(spec)
    return {"hash": parsed.hash(), "spec": parsed.key(), "sha256": sha}


def prune(*rows: dict) -> tuple[set[str], dict[str, str]]:
    kept, dropped = tool.prune_floor_twins({r["hash"]: r for r in rows})
    return set(kept), {d["hash"]: k["hash"] for d, k in dropped}


# bf16 tile 512: g1 serves 1..512, tail 385..512 (rung below is 384), floor 256 serves 257..512.
G1 = pin("dkdv,bf16,512,bias,bk,1,1,g1,0,1;block_q=32")
TAIL = pin("dkdv,bf16,512,bias,bk,1,1,tail,0,1;block_q=32")
F256 = pin("dkdv,bf16,512,bias,bk,1,1,256,0,1;block_q=32")


def test_the_narrower_twin_is_dropped_and_the_wider_one_kept() -> None:
    kept, dropped = prune(G1, TAIL)
    assert kept == {G1["hash"]}
    assert dropped == {TAIL["hash"]: G1["hash"]}


def test_a_superset_drops_every_row_it_contains() -> None:
    kept, dropped = prune(TAIL, F256, G1)
    assert kept == {G1["hash"]}
    assert dropped == {TAIL["hash"]: G1["hash"], F256["hash"]: G1["hash"]}


def test_a_dropped_twin_never_drops_a_row_it_contains_itself() -> None:
    """Nested ranges: only the widest survives, whatever the order of the rows."""
    kept, _ = prune(F256, TAIL, G1)
    assert kept == {G1["hash"]}


def test_equal_ranges_keep_the_smaller_hash() -> None:
    # fp32 tile 512: tail serves 257..512, so floor 256 serves the same dims.
    tail = pin("dkdv,fp32,512,bias,bk,1,1,tail,0,1;block_q=32")
    explicit = pin("dkdv,fp32,512,bias,bk,1,1,256,0,1;block_q=32")
    kept, dropped = prune(tail, explicit)
    keeper = min(tail["hash"], explicit["hash"])
    assert kept == {keeper}
    assert len(dropped) == 1 and dropped.popitem()[1] == keeper


def test_the_ladder_of_the_dtype_gives_the_tail() -> None:
    """fp32 has no 384 rung: its tail row serves 257..512, which the bf16 tail (385..512) does not."""
    assert tool.served_dims(Spec.parse(TAIL["spec"])) == (385, 512)
    fp32 = Spec.parse("dkdv,fp32,512,bias,bk,1,1,tail,0,1;block_q=32")
    assert tool.served_dims(fp32) == (257, 512)


def test_partial_overlap_is_kept(monkeypatch) -> None:
    """Neither range contains the other: both stay."""
    ranges = {G1["hash"]: (1, 300), TAIL["hash"]: (200, 512)}
    monkeypatch.setattr(tool, "served_dims", lambda spec: ranges[spec.hash()])
    kept, dropped = prune(G1, TAIL)
    assert kept == set(ranges)
    assert not dropped


def test_a_different_object_is_kept() -> None:
    other = pin("dkdv,bf16,512,bias,bk,1,1,tail,0,1;block_q=32", sha="b" * 64)
    kept, dropped = prune(G1, other)
    assert kept == {G1["hash"], other["hash"]}
    assert not dropped


def test_a_different_non_floor_field_is_kept() -> None:
    """Same object and nested ranges, but a knob or a head field differs: a knob twin is not pruned."""
    knob = pin("dkdv,bf16,512,bias,bk,1,1,tail,0,1;block_q=32;waves_per_eu=2")
    dropout = pin("dkdv,bf16,512,bias,bk,0,1,tail,0,1;block_q=32")
    kernel = pin("dq,bf16,512,bias,bk,1,1,tail,0,1")
    kept, dropped = prune(G1, knob, dropout, kernel)
    assert len(kept) == 4
    assert not dropped


def test_an_exact_row_is_not_a_twin_of_a_padded_one() -> None:
    """``padded`` is a Spec field: the exact row of the same tile stays."""
    exact = pin("dkdv,bf16,512,bias,bk,1,0,tail,0,1;block_q=32")
    kept, dropped = prune(G1, exact)
    assert len(kept) == 2
    assert not dropped
