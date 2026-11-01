# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The SDPA backward Spec grammar: the hashes the sweep records, the fp32 family's ladder and knob legality, and the merged grid knobs.

The pinned hashes are those of the sweep's manifests: a grammar change that moves one renames a shipped object.
Pure Python.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
from generators._instances import (  # noqa: E402
    PRIORITY_SPECIALIZED,
    SDPA_BWD_REDUCE_SYMBOL,
    instances_for,
)
from generators._sdpa_bwd_spec import (  # noqa: E402
    KNOBS,
    Spec,
    dbias_reduce_spec,
    rung_below,
    to_spec,
)

SWEEP_HASHES = {
    "dkdv,fp32,32,plain,bk,0,0,tail,0,1;block_q=32": "ccc139e916be",
    "dq,fp32,32,bias,bk,0,0,tail,0,1;block_m=32;block_n=16": "f2a3c5f87675",
    "prep,fp32,512,plain,bk,0,0,tail,0,1": "41156313bbdb",
    "dbias_reduce,fp32,32,plain,bk,0,0,tail,0,1": "9dcde734178d",
    "merged,bf16,224,bias,bk,0,0,tail,0,1;num_waves=2;mfma_rows=16;dkv_shards=1;store_db=0": "dcabd7878b0e",
}


@pytest.mark.parametrize(("key", "digest"), SWEEP_HASHES.items())
def test_a_sweep_key_parses_to_its_recorded_hash(key: str, digest: str) -> None:
    spec = Spec.parse(key)
    assert (spec.key(), spec.hash()) == (key, digest)


def test_knob_order_is_canonical_whatever_the_spelling() -> None:
    """Merged knobs come out as num_waves, mfma_rows, dkv_shards, role_order, store_db, gqa_pack."""
    assert KNOBS["merged"] == (
        "num_waves",
        "mfma_rows",
        "dkv_shards",
        "role_order",
        "store_db",
        "gqa_pack",
    )
    shuffled = Spec.parse(
        "merged,bf16,224,bias,bk,0,0,tail,0,1;store_db=0;dkv_shards=1;mfma_rows=16;num_waves=2"
    )
    assert shuffled.hash() == "dcabd7878b0e"
    assert list(shuffled.knobs) == ["num_waves", "mfma_rows", "dkv_shards", "store_db"]


def test_the_dbias_reduce_pin_of_a_dtype_is_its_placeholder_spec() -> None:
    assert dbias_reduce_spec("fp32").hash() == "9dcde734178d"
    assert (
        dbias_reduce_spec("bf16").key() == "dbias_reduce,bf16,32,plain,bk,0,0,tail,0,1"
    )


def test_fp32_has_no_384_rung_and_tile_512_floors_at_256() -> None:
    assert rung_below(512) == 384
    assert rung_below(512, "fp32") == 256
    assert rung_below(512, "bf16") == 384
    with pytest.raises(ValueError, match="tile"):
        Spec("dkdv", "fp32", 384)
    Spec("dkdv", "bf16", 384)


@pytest.mark.parametrize("kernel", ["merged", "dsink_reduce"])
def test_fp32_has_no_merged_or_dsink_family(kernel: str) -> None:
    with pytest.raises(ValueError, match="no fp32 family"):
        Spec(kernel, "fp32", 32 if kernel == "dsink_reduce" else 128)


@pytest.mark.parametrize(
    "knobs",
    [
        {"kernel": "dkdv", "block_q": 64},
        {"kernel": "dkdv", "num_waves": 4},
        {"kernel": "dkdv", "mfma_rows": 32},
        {"kernel": "dq", "block_m": 64},
        {"kernel": "dq", "bias_stage": "direct"},
        {"kernel": "prep", "sink": 1},
    ],
)
def test_fp32_rejects_knobs_outside_its_table(knobs: dict) -> None:
    kwargs = {"cls": "bias"} if knobs.get("bias_stage") else {}
    with pytest.raises(ValueError, match="fp32-legal"):
        Spec(dtype="fp32", tile=128, **kwargs, **knobs)


def test_fp32_accepts_its_geometry_knobs() -> None:
    Spec("dkdv", "fp32", 128, block_q=16, mfma_rows=16, num_stream_buffers=1)
    Spec("dq", "fp32", 128, block_m=16, block_n=16)


def test_fp32_never_drops_dbias() -> None:
    """store_db=0 builds a dQ object without the dB store: no fp32 spelling, so every fp32 bias dQ row stores dB."""
    with pytest.raises(ValueError, match="fp32-legal"):
        Spec("dq", "fp32", 128, "bias", store_db=0)


@pytest.mark.parametrize(
    "knobs",
    [{"kernel": "dkdv", "block_q": 16}, {"kernel": "dq", "block_m": 16}],
)
def test_sixteen_bit_rejects_the_fp32_only_geometries(knobs: dict) -> None:
    with pytest.raises(ValueError, match="fp32 knob value"):
        Spec(dtype="bf16", tile=128, **knobs)


def test_merged_grid_knobs_belong_to_merged_and_dkdv_only() -> None:
    Spec("merged", "bf16", 128, num_waves=4, mfma_rows=16, dkv_shards=2)
    Spec("dkdv", "bf16", 128, num_waves=4, mfma_rows=16, dkv_shards=2)
    with pytest.raises(ValueError, match="does not apply"):
        Spec("dq", "bf16", 128, dkv_shards=2)
    with pytest.raises(ValueError, match="does not apply"):
        Spec("prep", "bf16", 128, num_waves=4)


@pytest.mark.parametrize("name", ["num_waves", "mfma_rows", "dkv_shards"])
def test_merged_grid_knobs_are_positive_ints(name: str) -> None:
    with pytest.raises(ValueError, match="positive int"):
        Spec("merged", "bf16", 128, **{name: 0})
    with pytest.raises(ValueError, match="unknown knob"):
        Spec.parse("merged,bf16,128,plain,bk,0,0,tail,0,1;grid=2")


def _flaky() -> set[str]:
    path = FLYDSL / "tests" / "data" / "sdpa_bwd_known_flaky.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["hash"] for row in csv.DictReader(handle)}


@pytest.mark.parametrize("kind", ["prep", "dkdv", "dq"])
def test_policy_default_split_rows_outrank_the_rest(kind: str) -> None:
    """A split row that sets no perf knob and is not known-flaky is one priority above every other row of its op."""
    flaky = _flaky()
    rows = instances_for(f"sdpa_bwd_{kind}", "gfx950")
    ranks = {}
    for row in rows:
        spec = to_spec(row)
        policy_default = not spec.knobs and spec.hash() not in flaky
        assert row.priority == PRIORITY_SPECIALIZED + policy_default
        ranks[policy_default] = ranks.get(policy_default, 0) + 1
    assert set(ranks) == {True, False}


def test_a_dq_row_names_the_reduce_symbol_its_object_holds() -> None:
    """Only a dB-storing bias dQ row carries the reduce kernel; the symbol is the dtype's."""
    rows = instances_for("sdpa_bwd_dq", "gfx950")
    combined = [r for r in rows if r.knobs["reduce_symbol"]]
    assert combined
    for row in rows:
        spec = to_spec(row)
        stores_db = spec.cls == "bias" and spec.store_db != 0
        assert row.knobs["reduce_symbol"] == (
            SDPA_BWD_REDUCE_SYMBOL[spec.dtype] if stores_db else ""
        )


GFX11_KEYS = {
    "prep,f16,16,plain,bk,0,1,tail,0,1;arch=gfx11-generic": "112ee1714865",
    "dq,bf16,16,bias,bk,0,1,g1,0,1;arch=gfx11-generic;rne=1": None,
    "dbias_reduce,fp32,32,plain,bk,0,0,tail,0,1;arch=gfx11-generic": "f5f59e54fb48",
}


@pytest.mark.parametrize(("key", "digest"), GFX11_KEYS.items())
def test_a_gfx11_key_round_trips_with_the_arch_before_the_knobs(
    key: str, digest: str | None
) -> None:
    spec = Spec.parse(key)
    assert spec.key() == key
    assert spec.arch == "gfx11-generic"
    if digest:
        assert spec.hash() == digest


def test_every_pinned_gfx11_spec_keeps_its_hash() -> None:
    """The S1 catalogue's hashes are the evidence keys of the gfx11 objects: the grammar must reproduce all 432."""
    with (Path(__file__).parent / "data" / "sdpa_bwd_gfx11_reference.csv").open(
        newline="", encoding="utf-8"
    ) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 432
    for row in rows:
        spec = Spec.parse(row["spec"])
        assert (spec.key(), spec.hash()) == (row["spec"], row["hash"])


def test_the_arch_is_spelled_once_and_never_as_gfx950() -> None:
    with pytest.raises(ValueError, match="arch is spelled"):
        Spec.parse("prep,f16,16,plain,bk,0,1,tail,0,1;arch=gfx950")
    with pytest.raises(ValueError, match="arch is spelled"):
        Spec.parse(
            "prep,f16,16,plain,bk,0,1,tail,0,1;arch=gfx11-generic;arch=gfx12-generic"
        )


@pytest.mark.parametrize(
    "key",
    [
        # a gfx950 tile on a generic ladder
        "dkdv,bf16,384,plain,bk,0,1,g1,0,1;arch=gfx11-generic",
        # a padded generic dkdv carries floor g1
        "dkdv,bf16,64,plain,bk,0,1,tail,0,1;arch=gfx11-generic",
        # rne is a bf16 knob
        "dq,f16,64,plain,bk,0,1,g1,0,1;arch=gfx11-generic;rne=1",
        # the merged kernel has no generic build
        "merged,bf16,64,plain,bk,0,0,tail,0,1;arch=gfx11-generic",
        # contraction shards leave a column slice that is not a multiple of 16
        "dq,bf16,16,plain,bk,0,1,g1,0,1;arch=gfx11-generic;contraction_shards=2",
    ],
)
def test_a_gfx11_spec_outside_the_grammar_is_refused(key: str) -> None:
    with pytest.raises(ValueError):
        Spec.parse(key)
