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
from generators import _sdpa_bwd_nondet as nondet  # noqa: E402
from generators._instances import (  # noqa: E402
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


@pytest.mark.parametrize("kind", ["prep", "dkdv", "dq"])
def test_policy_default_split_rows_outrank_the_rest(kind: str) -> None:
    """A split row that sets no perf knob and is not in the nondeterministic list outranks every other row of its geometry that stores dB alike (a bias row without dB outranks its storing twin, default or not)."""
    listed = nondet.read(arch="gfx950").keys()
    groups: dict[tuple, dict[bool, list[int]]] = {}
    for row in instances_for(f"sdpa_bwd_{kind}", "gfx950"):
        spec = to_spec(row, "gfx950")
        policy_default = not spec.knobs and spec.hash() not in listed
        geometry = (
            row.dtype,
            row.knobs.get("store_db") == 0,
            *(
                row.knobs[k]
                for k in (
                    "head_dim",
                    "head_dim_v",
                    "causal",
                    "has_bias",
                    "has_dropout",
                    "padded",
                    "floor",
                    "dform",
                )
            ),
        )
        groups.setdefault(geometry, {}).setdefault(policy_default, []).append(
            row.priority
        )
    mixed = [g for g in groups.values() if len(g) == 2]
    assert mixed
    for ranks in mixed:
        assert min(ranks[True]) > max(ranks[False])


def test_a_dq_row_names_the_reduce_symbol_its_object_holds() -> None:
    """Only a dB-storing bias dQ row carries the reduce kernel; the symbol is the dtype's."""
    rows = instances_for("sdpa_bwd_dq", "gfx950")
    combined = [r for r in rows if r.knobs["reduce_symbol"]]
    assert combined
    for row in rows:
        spec = to_spec(row, "gfx950")
        stores_db = spec.cls == "bias" and spec.store_db != 0
        assert row.knobs["reduce_symbol"] == (
            SDPA_BWD_REDUCE_SYMBOL[spec.bias_dtype] if stores_db else ""
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
    """The S1 catalogue's hashes are the evidence keys of the gfx11 objects: the grammar must reproduce all 504."""
    with (Path(__file__).parent / "data" / "sdpa_bwd_gfx11_reference.csv").open(
        newline="", encoding="utf-8"
    ) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 504
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


# The hashes the AMD-internal sweep tooling records for these bias and split-V-head keys; that tooling is not part of this tree.
FEATURE_HASHES = {
    "dq,bf16,64,bias,bk,0,0,tail,0,1;bias_stage=direct;bias_f32=1": "5373cb474f4a",
    "dkdv,f16,128,plain,bk,0,0,tail,0,1;u2=dv": "05ab16cf6679",
    "dkdv,f16,128,causal,bk,1,0,tail,0,1;u2=dk": "83afb2c813b8",
    "dq,f16,64,causal,bk,0,0,tail,0,1;u2=dq": "6dc01fa40c8b",
    "dq,bf16,64,bias,bk,0,0,tail,0,1;arch=gfx11-generic;rne=1;bias_f32=1": "40a9b23235da",
    "dkdv,bf16,64,plain,bk,0,1,g1,0,1;arch=gfx11-generic;rne=1;u2=dk": "c8b99320d2f5",
}


@pytest.mark.parametrize(("key", "digest"), FEATURE_HASHES.items())
def test_a_feature_key_round_trips_to_its_hash(key: str, digest: str) -> None:
    spec = Spec.parse(key)
    assert (spec.key(), spec.hash()) == (key, digest)


def test_the_feature_knobs_come_last_in_every_knob_order() -> None:
    from generators._sdpa_bwd_spec import (  # noqa: PLC0415
        FEATURE_KNOBS,
        GENERIC_TABLE_KNOBS,
        KNOBS_CDNA9,
        KNOBS_GENERIC,
    )

    for kernel in ("dkdv", "dq"):
        for order in (KNOBS[kernel], KNOBS_GENERIC[kernel]):
            assert order[-2:] == FEATURE_KNOBS
        assert GENERIC_TABLE_KNOBS[kernel][-2:] == FEATURE_KNOBS
    for kernel in ("prep", "dbias_reduce", "merged"):
        assert not set(FEATURE_KNOBS) & set(KNOBS[kernel])
    # gfx942 and gfx90a: dK/dV spells grid_order before the feature knobs
    for kernel in ("dkdv", "dq"):
        assert KNOBS_CDNA9[kernel][-2:] == FEATURE_KNOBS
    assert KNOBS_CDNA9["dkdv"][-3] == "grid_order"
    for kernel in ("prep", "dbias_reduce"):
        assert not set(FEATURE_KNOBS) & set(KNOBS_CDNA9[kernel])


# --- gfx942 and gfx90a: one arch whose 16-bit rows are RDNA-family and whose fp32 rows are CDNA-family ---

CDNA9 = ("gfx942", "gfx90a")


def test_the_builder_family_follows_the_arch_and_the_dtype() -> None:
    from generators._sdpa_bwd_spec import family  # noqa: PLC0415

    assert [family("gfx950", d) for d in ("f16", "bf16", "fp32")] == ["cdna"] * 3
    assert [family("gfx11-generic", d) for d in ("f16", "bf16", "fp32")] == ["rdna"] * 3
    for arch in CDNA9:
        assert [family(arch, d) for d in ("f16", "bf16", "fp32")] == [
            "rdna",
            "rdna",
            "cdna",
        ]


def test_the_fp32_dbias_reduce_of_a_cdna9_arch_is_rdna_family() -> None:
    """The f32 dB reduce of a 16-bit row is the gfx11 kernel; every other fp32 row of gfx942 and gfx90a is CDNA-family."""
    from generators._sdpa_bwd_spec import family  # noqa: PLC0415

    for arch in CDNA9:
        assert family(arch, "fp32", "dbias_reduce") == "rdna"
        assert family(arch, "fp32", "dq") == family(arch, "fp32") == "cdna"
        spec = Spec.parse(f"dbias_reduce,fp32,32,plain,bk,0,0,tail,0,1;arch={arch}")
        assert spec.knobs == {}
    assert family("gfx950", "fp32", "dbias_reduce") == "cdna"


@pytest.mark.parametrize("arch", CDNA9)
def test_a_cdna9_row_takes_the_feature_knobs_of_the_16_bit_family_only(
    arch: str,
) -> None:
    from generators._sdpa_bwd_spec import U2_SYMBOLS  # noqa: PLC0415

    for key in (
        "dkdv,f16,64,bias,bk,0,1,g1,0,1;arch={};bias_f32=1",
        "dkdv,bf16,64,plain,bk,0,1,g1,0,1;arch={};rne=1;u2=dv",
        "dq,f16,128,causal,bk,1,1,g1,0,1;arch={};u2=dq",
        "dq,f16,128,bias,bk,0,1,g1,0,1;arch={};store_db=0;bias_f32=1",
    ):
        assert Spec.parse(key.format(arch)).key() == key.format(arch)
    for key in (
        "dkdv,fp32,64,plain,bk,0,1,g1,0,1;arch={};u2=dk",
        "dq,fp32,64,plain,bk,0,1,g1,0,1;arch={};u2=dq",
        "dkdv,fp32,64,bias,bk,0,1,g1,0,1;arch={};bias_f32=1",
        "dkdv,f16,64,bias,bk,0,1,g1,0,1;arch={};u2=dk",
    ):
        with pytest.raises(ValueError):
            Spec.parse(key.format(arch))
    for kernel in ("dkdv", "dq"):
        assert U2_SYMBOLS[arch, kernel] == U2_SYMBOLS["gfx11-generic", kernel]


def test_the_instance_module_spells_the_same_family_rule() -> None:
    """``_instances`` cannot import the Spec module (that would make it an input of every op), so it restates the rule."""
    from generators import _instances  # noqa: PLC0415
    from generators._sdpa_bwd_spec import ARCHES, family  # noqa: PLC0415

    for arch in ARCHES:
        for dtype in ("f16", "bf16", "fp32"):
            for kind in (None, "dq", "dbias_reduce"):
                assert _instances._sdpa_bwd_row_is_cdna(arch, dtype, kind) == (
                    family(arch, dtype, kind) == "cdna"
                ), (arch, dtype, kind)


@pytest.mark.parametrize("arch", CDNA9)
def test_the_cdna9_ladders(arch: str) -> None:
    from generators._sdpa_bwd_spec import FP32_LADDER, ladder  # noqa: PLC0415

    sixteen = ladder(arch, "f16")
    assert ladder(arch, "bf16") == sixteen == ladder(arch)
    assert sixteen == (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512)
    assert (
        ladder(arch, "fp32")
        == FP32_LADDER
        == (32, 64, 96, 128, 160, 192, 224, 256, 512)
    )
    # the head dims 257..512 ride the 512 rung, padded: there is no 320/384/448 rung
    assert rung_below(512, "f16", arch) == rung_below(512, "fp32", arch) == 256
    assert rung_below(16, "f16", arch) == 0
    for key, ok in (
        ("dkdv,bf16,512,plain,bk,0,1,g1,0,1;arch={}", True),
        ("dkdv,bf16,16,plain,bk,0,1,g1,0,1;arch={}", True),
        ("dkdv,bf16,384,plain,bk,0,1,g1,0,1;arch={}", False),
        ("dkdv,fp32,512,plain,bk,0,1,g1,0,1;arch={}", True),
        ("dkdv,fp32,16,plain,bk,0,1,g1,0,1;arch={}", False),
    ):
        if ok:
            assert Spec.parse(key.format(arch)).arch == arch
        else:
            with pytest.raises(ValueError, match="ladder"):
                Spec.parse(key.format(arch))


def test_the_cdna9_fp32_rows_take_the_gfx950_family_knobs_and_the_16_bit_rows_the_generic_ones() -> (
    None
):
    fp32 = Spec.parse("dkdv,fp32,512,causal,bk,1,1,g1,0,1;arch=gfx90a;block_q=16")
    assert fp32.knobs == {"block_q": 16}
    assert (
        Spec.parse(
            "dq,fp32,128,bias,bk,0,1,g1,0,1;arch=gfx942;block_m=32;block_n=16"
        ).block_n
        == 16
    )
    with pytest.raises(ValueError, match="does not apply"):
        Spec.parse("dkdv,fp32,128,plain,bk,0,1,g1,0,1;arch=gfx90a;split_head_dim=1")
    with pytest.raises(ValueError, match="does not apply"):
        Spec.parse("dkdv,f16,128,plain,bk,0,1,g1,0,1;arch=gfx90a;mfma_rows=16")
    with pytest.raises(ValueError, match="not built"):
        Spec.parse("merged,bf16,64,plain,bk,0,0,tail,0,1;arch=gfx90a")


@pytest.mark.parametrize(
    ("key", "other"),
    [
        ("dkdv,f16,64,plain,bk,0,1,g1,0,1;arch=gfx942", "grid_order=kv_fastest"),
        ("dkdv,f16,64,plain,bk,0,1,g1,0,1;arch=gfx90a", "grid_order=head_fastest"),
    ],
)
def test_grid_order_is_spelled_only_when_it_is_not_the_default_of_the_arch(
    key: str, other: str
) -> None:
    default = Spec.parse(key)
    assert default.grid_order is None and "grid_order" not in default.key()
    spelled = Spec.parse(f"{key};{other}")
    assert spelled.key() == f"{key};{other}"
    assert spelled.hash() != default.hash()
    default_order = {"gfx942": "head_fastest", "gfx90a": "kv_fastest"}[default.arch]
    with pytest.raises(ValueError, match="only when it is not the default"):
        Spec.parse(f"{key};grid_order={default_order}")


@pytest.mark.parametrize(
    "key",
    [
        "dq,f16,64,plain,bk,0,1,g1,0,1;arch=gfx90a;grid_order=head_fastest",  # dQ has one grid order
        "dkdv,fp32,64,plain,bk,0,1,g1,0,1;arch=gfx90a;grid_order=head_fastest",  # the native fp32 family launches head-fastest
        "dkdv,f16,64,plain,bk,0,1,g1,0,1;arch=gfx11-generic;grid_order=head_fastest",
    ],
)
def test_grid_order_belongs_to_the_cdna9_16_bit_dkdv_rows(key: str) -> None:
    with pytest.raises(ValueError, match="does not apply"):
        Spec.parse(key)


def test_a_cdna9_row_keeps_its_arch_through_to_spec() -> None:
    """A CDNA-family row of gfx942 or gfx90a is a Spec of that arch: its key spells the arch and hashes apart from gfx950's."""
    from generators import _instances  # noqa: PLC0415

    row = _instances._sdpa_bwd_cdna_row(
        "dkdv",
        ("fp32", 128, "plain", "bk", 0, 1, "g1", 0, 1, -1, -1, -1, 16, -1, -1, -1, -1),
        frozenset(),
    )
    for arch in CDNA9:
        spec = to_spec(row, arch)
        assert (spec.arch, spec.key()) == (
            arch,
            f"dkdv,fp32,128,plain,bk,0,1,g1,0,1;arch={arch};block_q=16",
        )
        assert Spec.parse(spec.key()) == spec
    assert (
        to_spec(row, "gfx950").key() == "dkdv,fp32,128,plain,bk,0,1,g1,0,1;block_q=16"
    )


@pytest.mark.parametrize("arch", CDNA9)
@pytest.mark.parametrize("dtype", ["f16", "bf16", "fp32"])
def test_vec_tail_is_a_spelled_knob_of_the_padded_cdna9_prep(
    arch: str, dtype: str
) -> None:
    plain = f"prep,{dtype},128,plain,bk,0,1,tail,0,1;arch={arch}"
    spelled = Spec.parse(f"{plain};vec_tail=1")
    assert spelled.key() == f"{plain};vec_tail=1"
    assert Spec.parse(spelled.key()) == spelled
    assert Spec.parse(plain).key() == plain and Spec.parse(plain).vec_tail is None
    assert spelled.hash() != Spec.parse(plain).hash()
    with pytest.raises(ValueError, match="vec_tail"):
        Spec.parse(f"{plain};vec_tail=0")


@pytest.mark.parametrize(
    "key",
    [
        "prep,f16,128,plain,bk,0,1,tail,0,1;vec_tail=1",  # gfx950
        "prep,fp32,128,plain,bk,0,1,tail,0,1;vec_tail=1",  # gfx950
        "prep,f16,128,plain,bk,0,1,tail,0,1;arch=gfx11-generic;vec_tail=1",
        "prep,fp32,128,plain,bk,0,1,tail,0,1;arch=gfx11-generic;vec_tail=1",
        "prep,f16,128,plain,bk,0,1,tail,0,1;arch=gfx12-generic;vec_tail=1",
        "prep,f16,128,plain,bk,0,0,tail,0,1;arch=gfx90a;vec_tail=1",  # exact tile: the knob has no effect, one spelling per object
        "prep,fp32,128,plain,bk,0,0,tail,0,1;arch=gfx942;vec_tail=1",
        "dkdv,f16,128,plain,bk,0,1,g1,0,1;arch=gfx90a;vec_tail=1",
        "dq,fp32,128,plain,bk,0,1,g1,0,1;arch=gfx942;vec_tail=1",
    ],
)
def test_vec_tail_is_refused_outside_the_padded_cdna9_prep(key: str) -> None:
    with pytest.raises(ValueError):
        Spec.parse(key)


@pytest.mark.parametrize(
    "key",
    [
        "dq,bf16,64,bias,bk,0,0,tail,0,1;bias_stage=direct;bias_f32=1",
        "dkdv,bf16,64,bias,bk,0,0,tail,0,1;bias_f32=1",
        "dq,bf16,64,bias,bk,0,0,tail,0,1;arch=gfx11-generic;rne=1;bias_f32=1",
        "dkdv,f16,64,plain,bk,0,0,tail,0,1;u2=dk",
        "dkdv,f16,64,causal,bk,0,0,tail,0,1;u2=dv",
        "dq,f16,64,plain,bk,0,0,tail,0,1;u2=dq",
        "dq,f16,64,causal,bk,0,0,tail,0,1;arch=gfx11-generic;u2=dq",
    ],
)
def test_a_feature_spec_inside_the_grammar_is_accepted(key: str) -> None:
    assert Spec.parse(key).key() == key


@pytest.mark.parametrize(
    "key",
    [
        # bias_f32: bias objects, 16-bit, dkdv or dq, one spelling; a gfx950 dQ needs the direct stage
        "dq,bf16,64,bias,bk,0,0,tail,0,1;bias_f32=1",
        "dkdv,f16,64,plain,bk,0,0,tail,0,1;bias_f32=1",
        "dkdv,f16,64,causal,bk,0,0,tail,0,1;bias_f32=1",
        "dkdv,fp32,64,bias,bk,0,0,tail,0,1;bias_f32=1",
        "dq,fp32,64,bias,bk,0,0,tail,0,1;bias_stage=direct;bias_f32=1",
        "merged,bf16,64,bias,bk,0,0,tail,0,1;bias_f32=1",
        "prep,bf16,64,plain,bk,0,0,tail,0,1;bias_f32=1",
        "dkdv,bf16,64,bias,bk,0,0,tail,0,1;bias_f32=2",
        # u2: each word on its kernel, plain or causal, 16-bit, never with a bias
        "dq,f16,64,plain,bk,0,0,tail,0,1;u2=dk",
        "dq,f16,64,plain,bk,0,0,tail,0,1;u2=dv",
        "dkdv,f16,64,plain,bk,0,0,tail,0,1;u2=dq",
        "dkdv,f16,64,bias,bk,0,0,tail,0,1;u2=dk",
        "dkdv,f16,64,bias,bk,0,0,tail,0,1;bias_f32=1;u2=dk",
        "dkdv,fp32,64,plain,bk,0,0,tail,0,1;u2=dk",
        "merged,bf16,64,plain,bk,0,0,tail,0,1;u2=dk",
        "dkdv,f16,64,plain,bk,0,0,tail,0,1;u2=both",
    ],
)
def test_a_feature_spec_outside_the_grammar_is_refused(key: str) -> None:
    with pytest.raises(ValueError):
        Spec.parse(key)


def test_a_float_bias_beside_16_bit_has_its_own_bias_dtype() -> None:
    plain = Spec.parse("dq,bf16,64,bias,bk,0,0,tail,0,1")
    f32 = Spec.parse("dq,bf16,64,bias,bk,0,0,tail,0,1;bias_stage=direct;bias_f32=1")
    fp32 = Spec.parse("dq,fp32,64,bias,bk,0,0,tail,0,1")
    assert (plain.bias_dtype, f32.bias_dtype, fp32.bias_dtype) == (
        "bf16",
        "fp32",
        "fp32",
    )


def test_the_ladders_of_the_tables_are_the_specs() -> None:
    """``_instances`` cannot import the Spec module (it would become an input of every op's digest), so it spells the rungs; they must be the Spec's."""
    from generators import _instances  # noqa: PLC0415
    from generators._sdpa_bwd_spec import GENERIC_LADDER, LADDER  # noqa: PLC0415

    assert _instances.SDPA_BWD_GFX11_LADDER == GENERIC_LADDER
    assert _instances.SDPA_BWD_TILE_RUNGS == LADDER


def test_a_feature_knob_in_a_table_row_is_the_knob_of_its_spec() -> None:
    """A table row may stop before its feature columns (they read as unset); one that spells them names the feature Spec."""
    from generators._instances import (  # noqa: PLC0415
        SDPA_BWD_FEATURE_KNOBS,
        SDPA_BWD_KNOBS,
        SDPA_BWD_OPTIONAL_KNOBS,
        sdpa_bwd_perf_knobs,
    )
    from generators._sdpa_bwd_spec import (
        FEATURE_KNOBS,
        OPTIONAL_TABLE_KNOBS,
        UNSET_INT,
        UNSET_WORD,
    )  # noqa: PLC0415

    assert tuple(SDPA_BWD_FEATURE_KNOBS) == FEATURE_KNOBS
    assert SDPA_BWD_FEATURE_KNOBS == {"bias_f32": UNSET_INT, "u2": UNSET_WORD}
    assert tuple(SDPA_BWD_OPTIONAL_KNOBS) == OPTIONAL_TABLE_KNOBS
    names = SDPA_BWD_KNOBS["dkdv"]
    short = sdpa_bwd_perf_knobs(names, (-1,) * (len(names) - 2))
    assert (short["bias_f32"], short["u2"]) == (UNSET_INT, UNSET_WORD)
    full = sdpa_bwd_perf_knobs(names, (*(-1,) * (len(names) - 2), -1, "dk"))
    assert full["u2"] == "dk"
    with pytest.raises(ValueError, match="knob columns"):
        sdpa_bwd_perf_knobs(names, (-1,) * (len(names) - 3))
    with pytest.raises(ValueError):
        sdpa_bwd_perf_knobs(names, (-1,) * (len(names) + 1))
