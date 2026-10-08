# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for tools/enumerate_conv_aot_instances.py (no compile, no GPU)."""

import dataclasses
import itertools
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="the rocKE builders need numpy")

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))

import enumerate_conv_aot_instances as tool  # noqa: E402
import launch_surface  # noqa: E402

_REPO = launch_surface.find_repo_root(_TOOLS)
_ROCKE = _REPO / "dnn-providers" / "hip-kernel-provider" / "rocke"
_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
_SPILLS = _CONFIGS / "gfx950_conv_fwd_aot.spills.json"


@pytest.fixture(scope="module", autouse=True)
def _rocke_on_path():
    """Bind rocKE for this module only: test_dispatch_parity asserts what a provider
    root ADDS to sys.path, so a binding left behind at collection would hide it."""
    before = list(sys.path)
    tool._bind_rocke(_ROCKE)
    yield
    sys.path[:] = before


def _identity(**knobs):
    from benchmarks.common.kernel_cache import KernelIdentity

    fields = dict(
        arch="gfx950",
        direction="fwd",
        algorithm="implicit_gemm",
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_d="fp16",
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
        pipeline="mem",
        epilogue="cshuffle",
        wave_size=64,
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=8,
        grouped=True,
        llvm_flavor="llvm22",
    )
    fields.update(knobs)
    return KernelIdentity(**fields).to_dict()


@pytest.fixture(scope="module")
def synthetic_grid():
    """The forward grid's knob cross product on one geometry."""
    grid = []
    for pipeline, epilogue, vab, vc, (unroll, dma) in itertools.product(
        ("mem", "compv3", "compv4"),
        ("default", "cshuffle"),
        (1, 2, 4, 8),
        (1, 2, 4, 8),
        ((False, False), (True, False), (False, True)),
    ):
        grid.append(
            _identity(
                pipeline=pipeline,
                epilogue=epilogue,
                vector_size_a=vab,
                vector_size_b=vab,
                vector_size_c=vc,
                unroll_k=unroll,
                async_dma=dma,
            )
        )
    return grid


def test_knob_rule_keeps_exactly_the_listed_values(synthetic_grid):
    kept = [i for i in synthetic_grid if not tool.knob_rule_rejection(i)]
    # pipeline 2 x epilogue 1 x A/B width 3 x C width 3 x K loop 2
    assert len(kept) == 2 * 1 * 3 * 3 * 2
    for knob, values in tool.KNOB_RULE.items():
        assert {i[knob] for i in kept} == set(values), knob


@pytest.mark.parametrize(
    ("change", "knob"),
    [
        ({"pipeline": "compv4"}, "pipeline"),
        ({"epilogue": "default"}, "epilogue"),
        ({"vector_size_a": 2, "vector_size_b": 2}, "vector_size_a"),
        ({"vector_size_c": 2}, "vector_size_c"),
        ({"async_dma": True}, "async_dma"),
    ],
)
def test_knob_rule_names_the_rejected_knob(change, knob):
    assert tool.knob_rule_rejection(_identity()) == ""
    assert tool.knob_rule_rejection(_identity(**change)).startswith(f"{knob}=")


def test_spill_list_round_trip(tmp_path, synthetic_grid):
    candidates = [i for i in synthetic_grid if not tool.knob_rule_rejection(i)]
    hashes = {"fp16": [tool.stable_hash(i) for i in candidates]}
    spilling = candidates[::5]
    entries = [
        tool.spill_entry(i, {"vgpr_spill": 3, "scratch_bytes": 16}) for i in spilling
    ]
    path = tmp_path / "spills.json"
    tool.write_spill_list(path, entries, hashes)

    assert tool.read_spill_list(path, hashes) == {tool.stable_hash(i) for i in spilling}
    document = json.loads(path.read_text())
    assert document["knob_rule"] == {k: list(v) for k, v in tool.KNOB_RULE.items()}
    assert document["spills"] == entries


def test_spill_list_refuses_another_candidate_set(tmp_path, synthetic_grid):
    hashes = {"fp16": [tool.stable_hash(i) for i in synthetic_grid[:4]]}
    path = tmp_path / "spills.json"
    tool.write_spill_list(path, [], hashes)
    with pytest.raises(SystemExit, match="--recompute-spills"):
        tool.read_spill_list(path, {"fp16": hashes["fp16"][:3]})


def test_r1_reads_scratch_or_spill():
    assert not tool.spills({"vgpr_spill": 0, "scratch_bytes": 0})
    assert tool.spills({"vgpr_spill": 0, "scratch_bytes": 4})
    assert tool.spills({"vgpr_spill": 2, "scratch_bytes": 0})


def test_committed_spill_list_is_hashes_of_knob_rule_candidates():
    document = json.loads(_SPILLS.read_text())
    assert document["knob_rule"] == {k: list(v) for k, v in tool.KNOB_RULE.items()}
    for entry in document["spills"]:
        assert len(entry["instance_id"]) == 40
        assert entry["dtype"] in tool.DTYPES
        assert entry["vgpr_spill"] > 0 or entry["scratch_bytes"] > 0
        assert entry["pipeline"] in tool.KNOB_RULE["pipeline"]
        assert entry["vector_size_ab"] in tool.KNOB_RULE["vector_size_a"]
        assert entry["vector_size_c"] in tool.KNOB_RULE["vector_size_c"]


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"unroll_k": True, "vector_size_c": 1},
        {"waves_per_eu": 2, "lds_k_pad": 8, "chiplet_swizzle": True},
        {"ws_replicas": 2, "knobs": "x"},
        {"llvm_flavor": ""},
    ],
    ids=["defaults", "kept-knobs", "layout-knobs", "replicas", "empty-flavor"],
)
def test_compact_identity_round_trips_to_the_same_hash(change):
    from benchmarks.common.kernel_cache import KernelIdentity

    full = _identity(**change)
    compact = tool.compact_identity(full)
    rebuilt = KernelIdentity.from_dict(compact)
    assert rebuilt.to_dict() == full
    assert rebuilt.stable_hash() == tool.stable_hash(full)
    # Only the fields without a plain default, and those that differ from it.
    fields = dataclasses.fields(KernelIdentity)
    assert set(compact) == {
        f.name
        for f in fields
        if f.default is dataclasses.MISSING or full[f.name] != f.default
    }


def test_compact_identity_keeps_a_default_value_of_another_type():
    full = dict(_identity(), unroll_k=0)
    assert tool.compact_identity(full)["unroll_k"] == 0


def test_written_instances_are_one_compact_entry_per_line(tmp_path, synthetic_grid):
    path = tmp_path / "instances.json"
    tool.write_instances(path, synthetic_grid[:5])
    lines = path.read_text().splitlines()
    assert lines[0] == "[" and lines[-1] == "]" and len(lines) == 7
    entries = json.loads(path.read_text())
    assert entries == [tool.compact_identity(i) for i in synthetic_grid[:5]]
    assert [tool.stable_hash(e) for e in entries] == [
        tool.stable_hash(i) for i in synthetic_grid[:5]
    ]


def test_make_conv_aot_config_accepts_the_written_instances(tmp_path):
    import make_conv_aot_config
    import yaml

    kept = [
        _identity(pipeline=p, vector_size_c=c, unroll_k=u)
        for p in ("mem", "compv3")
        for c in (8, 4)
        for u in (False, True)
    ]
    instances = tmp_path / "instances.json"
    tool.write_instances(instances, kept)
    out = tmp_path / "aot.yaml"
    argv = ["--rocke-root", str(_ROCKE), "--instances", str(instances)]
    assert make_conv_aot_config.main(argv + ["--output", str(out)]) == 0
    shapes = yaml.safe_load(out.read_text())["packs"][0]["variants"][0]["shapes"]
    assert sorted(s["instance_id"] for s in shapes) == sorted(
        tool.stable_hash(i) for i in kept
    )


# ---- Direct depthwise identities ----


@pytest.fixture(scope="module")
def direct_fp16():
    return tool.direct_identities("fp16", 1)


def test_direct_identities_are_every_forward_direct_identity_the_launch_runs(
    direct_fp16,
):
    import make_conv_aot_config

    packaged, grid_count = direct_fp16
    algorithms = sorted({i["algorithm"] for i in packaged})
    assert algorithms == sorted(make_conv_aot_config.DIRECT_ALGORITHMS)
    # The direct grid holds other forward algorithms, which have no packaged launch.
    assert grid_count > len(packaged)
    for identity in packaged:
        assert identity["dtype_a"] == "fp16"
        assert identity["direction"] == "direct_fwd"
        spec, why = make_conv_aot_config.instance_spec(identity)
        assert spec is not None and spec.is_direct, why
    assert len({tool.stable_hash(i) for i in packaged}) == len(packaged)


def test_direct_identities_round_trip_through_the_compact_form(direct_fp16):
    from benchmarks.common.kernel_cache import KernelIdentity

    for identity in direct_fp16[0]:
        compact = tool.compact_identity(identity)
        assert KernelIdentity.from_dict(compact).to_dict() == identity


def _fake_main(tmp_path, monkeypatch, implicit, direct):
    """Run main() over a synthetic grid and direct list; returns the written list."""
    monkeypatch.setattr(
        tool,
        "grid_identities",
        lambda dtype, jobs: [
            dict(i, dtype_a=dtype, dtype_b=dtype, dtype_d=dtype) for i in implicit
        ],
    )
    monkeypatch.setattr(tool, "read_spill_list", lambda path, candidates: set())
    monkeypatch.setattr(
        tool,
        "direct_identities",
        lambda dtype, jobs: (
            [dict(i, dtype_a=dtype, dtype_b=dtype, dtype_d=dtype) for i in direct],
            len(direct),
        ),
    )
    out = tmp_path / "instances.json"
    argv = ["--rocke-root", str(_ROCKE), "--output", str(out), "--jobs", "1"]
    assert tool.main(argv) == 0
    return json.loads(out.read_text())


def test_main_orders_by_dtype_then_family_then_hash(
    tmp_path, monkeypatch, synthetic_grid, direct_fp16
):
    implicit = synthetic_grid[:12]
    direct = direct_fp16[0][:5]
    written = _fake_main(tmp_path, monkeypatch, implicit, direct)

    kept = [i for i in implicit if not tool.knob_rule_rejection(i)]
    assert len(written) == 2 * (len(kept) + len(direct))
    keys = [
        (e["dtype_a"], e["algorithm"] != "implicit_gemm", tool.stable_hash(e))
        for e in written
    ]
    assert keys == sorted(keys)
    assert [k[0] for k in keys] == ["bf16"] * (len(keys) // 2) + ["fp16"] * (
        len(keys) // 2
    )
    assert sum(1 for k in keys if k[1]) == 2 * len(direct)


def test_main_writes_the_same_list_twice(
    tmp_path, monkeypatch, synthetic_grid, direct_fp16
):
    implicit, direct = synthetic_grid[:20], direct_fp16[0][:3]
    first = _fake_main(tmp_path, monkeypatch, implicit, direct)
    second = _fake_main(
        tmp_path, monkeypatch, list(reversed(implicit)), list(reversed(direct))
    )
    assert first == second
