# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for tools/enumerate_conv_aot_instances.py (no compile, no GPU)."""

import collections
import dataclasses
import itertools
import json
import os
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
    """Knob combinations on one geometry, each a valid identity of the forward grid."""
    grid = []
    for pipeline, vab, vc, (unroll, dma) in itertools.product(
        ("mem", "compv3", "compv4"),
        (1, 2, 4, 8),
        (1, 2, 4, 8),
        ((False, False), (True, False)),
    ):
        grid.append(
            _identity(
                pipeline=pipeline,
                vector_size_a=vab,
                vector_size_b=vab,
                vector_size_c=vc,
                unroll_k=unroll,
                async_dma=dma,
            )
        )
    return grid


@pytest.fixture(scope="module")
def selection():
    """The committed selection."""
    return tool.Selection.load(tool.DEFAULT_SELECTION)


@pytest.fixture(scope="module")
def grid_fp16(selection):
    """The fp16 implicit-GEMM identities the committed selection keeps."""
    return tool.grid_identities("fp16", min(16, os.cpu_count() or 1), selection)[0]


def _selection_of(implicit, direct=(), excluded=()):
    """A selection keeping exactly the configs of ``implicit`` and the ``direct``
    identities, less the ``excluded`` implicit identities."""
    fields = (
        "tile_m",
        "tile_n",
        "tile_k",
        "warp_m",
        "warp_n",
        "warp_tile_m",
        "warp_tile_n",
        "warp_tile_k",
    )
    configs = collections.defaultdict(set)
    for identity in implicit:
        geometry = tuple(identity[f] for f in fields)
        configs[geometry].add((identity["pipeline"], tool.k_loop(identity)))
    return {
        "implicit_gemm": {
            "geometry_fields": list(fields),
            "configs": [
                {"geometry": list(g), "pipeline_k_loop": sorted(map(list, loops))}
                for g, loops in sorted(configs.items())
            ],
            "excluded_instances": sorted(tool.selection_hash(i) for i in excluded),
        },
        "direct": {"instances": sorted(tool.selection_hash(i) for i in direct)},
    }


def _knob_class(identity):
    k_loop = (
        "async" if identity["async_dma"] else "unroll" if identity["unroll_k"] else ""
    )
    return (
        identity["pipeline"],
        identity["epilogue"],
        k_loop,
        identity["vector_size_a"],
        identity["vector_size_b"],
        identity["vector_size_c"],
        identity["async_chunk_a"],
        identity["async_chunk_b"],
    )


def test_the_implicit_catalog_is_rockes_forward_sweep_less_the_exclusions(
    grid_fp16, selection
):
    """Every valid identity kernel_sweep enumerates, once, except exactly those the
    selection leaves out; a kept config keeps every width pair and epilogue the
    sweep enumerates for it, less only its excluded instances."""
    from benchmarks.common import kernel_sweep
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx("gfx950")
    swept = [
        job.identity.to_dict()
        for job in kernel_sweep._direction_jobs(
            "fwd", "gfx950", "fp16", target, (2,), 64
        )
        if kernel_sweep._spec_is_valid(job, "gfx950", "fp16")
    ]
    kept = [i for i in swept if tool.excluded_reason(i, selection) is None]
    hashes = sorted(tool.stable_hash(i) for i in grid_fp16)
    assert hashes == sorted(tool.stable_hash(i) for i in kept)
    assert len(set(hashes)) == len(hashes)
    # Each rule drops something, so none of them is dead.
    reasons = collections.Counter(tool.excluded_reason(i, selection) for i in swept)
    assert set(reasons) == {
        None,
        "async DMA K loop",
        "config not selected",
        "instance excluded",
    }
    assert reasons["instance excluded"] == len(selection.excluded)
    # Every selected config is served, by every K loop and pipeline it names.
    assert {selection.config_of(i) for i in grid_fp16} == selection.configs
    classes = {_knob_class(i) for i in grid_fp16}
    assert {c[0] for c in classes} == {"mem", "compv3"}
    assert {c[1] for c in classes} == {"default", "cshuffle"}
    assert {c[2] for c in classes} == {"", "unroll"}

    def variants(identities):
        by_config = collections.defaultdict(set)
        for i in identities:
            by_config[selection.config_of(i)].add(tool.stable_hash(i))
        return by_config

    in_sweep, in_grid = variants(swept), variants(grid_fp16)
    excluded = {
        tool.stable_hash(i)
        for i in swept
        if tool.excluded_reason(i, selection) == "instance excluded"
    }
    for config, members in in_grid.items():
        assert members == in_sweep[config] - excluded, config


def test_the_selection_names_instances_alike_in_every_dtype():
    fp16 = _identity()
    bf16 = _identity(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16")
    assert tool.stable_hash(fp16) != tool.stable_hash(bf16)
    assert tool.selection_hash(fp16) == tool.selection_hash(bf16)
    selection = tool.Selection.from_dict(_selection_of([fp16], excluded=[fp16]))
    assert tool.excluded_reason(bf16, selection) == "instance excluded"


_SELECTED = [
    _identity,
    lambda: _identity(unroll_k=True),
]


@pytest.mark.parametrize(
    "change, reason",
    [
        ({}, None),
        ({"vector_size_a": 1, "vector_size_b": 1, "vector_size_c": 4}, None),
        ({"epilogue": "default", "vector_size_c": 1}, None),
        ({"unroll_k": True}, None),
        (
            {"async_dma": True, "async_chunk_a": 8, "async_chunk_b": 8},
            "async DMA K loop",
        ),
        ({"pipeline": "compv3"}, "config not selected"),
        ({"pipeline": "compv4"}, "config not selected"),
        ({"tile_m": 128}, "config not selected"),
        ({"vector_size_c": 4}, "instance excluded"),
    ],
    ids=[
        "8-8-8",
        "1-1-4",
        "default-8-8-1",
        "unroll",
        "async",
        "compv3",
        "compv4",
        "other-geometry",
        "excluded",
    ],
)
def test_excluded_reason_names_the_rule_an_identity_breaks(change, reason):
    selection = tool.Selection.from_dict(
        _selection_of(
            [make() for make in _SELECTED], excluded=[_identity(vector_size_c=4)]
        )
    )
    assert tool.excluded_reason(_identity(**change), selection) == reason


def test_every_knob_class_of_the_grid_converts(tmp_path, grid_fp16):
    """One identity of every pipeline, epilogue, K loop, chunk and width class."""
    import make_conv_aot_config
    import yaml

    sample = list({_knob_class(i): i for i in grid_fp16}.values())
    assert tool.validate(sample) == []
    instances = tmp_path / "instances.json"
    tool.write_instances(instances, sample)
    out = tmp_path / "aot.yaml"
    argv = ["--rocke-root", str(_ROCKE), "--instances", str(instances)]
    assert make_conv_aot_config.main(argv + ["--output", str(out)]) == 0
    shapes = [
        shape
        for pack in yaml.safe_load(out.read_text())["packs"]
        for variant in pack["variants"]
        for shape in variant["shapes"]
    ]
    assert sorted(s["instance_id"] for s in shapes) == sorted(
        tool.stable_hash(i) for i in sample
    )


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


# ---- Direct depthwise identities ----


@pytest.fixture(scope="module")
def direct_fp16():
    return tool.direct_identities("fp16", 1)


def test_direct_identities_are_every_forward_direct_identity_the_converter_accepts(
    direct_fp16,
):
    import make_conv_aot_config
    from benchmarks.common.direct_kernel_sweep import enumerate_direct_jobs
    from rocke.core.arch import ArchTarget

    packaged, refused = direct_fp16
    forward = enumerate_direct_jobs(
        arch="gfx950",
        target=ArchTarget.from_gfx("gfx950"),
        directions=("fwd",),
        dtype="fp16",
    )
    assert len(packaged) + sum(refused.values()) == len(forward)
    for identity in packaged:
        assert identity["dtype_a"] == "fp16"
        assert identity["direction"] == "direct_fwd"
        spec, why = make_conv_aot_config.instance_spec(identity)
        assert spec is not None and spec.is_direct, why
    assert len({tool.stable_hash(i) for i in packaged}) == len(packaged)
    # The packaged direct launch runs every forward direct kernel of the sweep, so
    # every identity of the direct grid is packaged and nothing is refused.
    assert sorted({i["algorithm"] for i in packaged}) == sorted(
        make_conv_aot_config.PACKAGED_DIRECT_ALGORITHMS
    )
    assert not refused, refused


def test_direct_identities_round_trip_through_the_compact_form(direct_fp16):
    from benchmarks.common.kernel_cache import KernelIdentity

    for identity in direct_fp16[0]:
        compact = tool.compact_identity(identity)
        assert KernelIdentity.from_dict(compact).to_dict() == identity


def _fake_main(tmp_path, monkeypatch, implicit, direct, refused=(), selection=None):
    """Run main() over a synthetic grid and direct list, with a selection keeping all
    of them unless one is given; returns the written lists, by dtype."""
    monkeypatch.setattr(
        tool,
        "sweep_identities",
        lambda dtype, jobs: [
            dict(i, dtype_a=dtype, dtype_b=dtype, dtype_d=dtype) for i in implicit
        ],
    )
    monkeypatch.setattr(
        tool,
        "direct_identities",
        lambda dtype, jobs: (
            [dict(i, dtype_a=dtype, dtype_b=dtype, dtype_d=dtype) for i in direct],
            collections.Counter(dict(refused)),
        ),
    )
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection or _selection_of(implicit, direct)))
    out = tmp_path / "lists"
    argv = ["--rocke-root", str(_ROCKE), "--output-dir", str(out), "--jobs", "1"]
    assert tool.main(argv + ["--selection", str(path)]) == 0
    return {
        dtype: json.loads((out / tool.INSTANCE_LIST.format(dtype=dtype)).read_text())
        for dtype in tool.DTYPES
    }


def test_main_writes_one_list_per_dtype_ordered_by_family_then_hash(
    tmp_path, monkeypatch, synthetic_grid, direct_fp16
):
    implicit = synthetic_grid[:12]
    direct = direct_fp16[0][:5]
    written = _fake_main(tmp_path, monkeypatch, implicit, direct)

    assert sorted(written) == sorted(tool.DTYPES)
    for dtype, entries in written.items():
        assert len(entries) == len(implicit) + len(direct)
        assert {e["dtype_a"] for e in entries} == {dtype}
        keys = [
            (e["algorithm"] != "implicit_gemm", tool.stable_hash(e)) for e in entries
        ]
        assert keys == sorted(keys)
        assert sum(1 for k in keys if k[0]) == len(direct)


def test_main_writes_the_same_lists_twice(
    tmp_path, monkeypatch, synthetic_grid, direct_fp16
):
    implicit, direct = synthetic_grid[:20], direct_fp16[0][:3]
    first = _fake_main(tmp_path, monkeypatch, implicit, direct)
    second = _fake_main(
        tmp_path, monkeypatch, list(reversed(implicit)), list(reversed(direct))
    )
    assert first == second


def test_main_reports_the_direct_identities_the_converter_refuses(
    tmp_path, monkeypatch, synthetic_grid, capsys
):
    refused = {("direct_grouped", "direct algorithm 'direct_grouped' has no launch"): 7}
    _fake_main(tmp_path, monkeypatch, synthetic_grid[:2], [], refused)
    assert "direct direct_grouped: 7 not packaged" in capsys.readouterr().out


def test_main_refuses_an_implicit_identity_the_converter_refuses(
    tmp_path, monkeypatch, synthetic_grid, capsys
):
    """Every implicit-GEMM identity the selection keeps is shipped: one the converter
    cannot package is an error, never a silent omission."""
    unsupported = dict(synthetic_grid[0], split_k=2)
    monkeypatch.setattr(tool, "sweep_identities", lambda dtype, jobs: [unsupported])
    monkeypatch.setattr(
        tool, "direct_identities", lambda dtype, jobs: ([], collections.Counter())
    )
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(_selection_of([unsupported])))
    out = tmp_path / "lists"
    argv = ["--rocke-root", str(_ROCKE), "--output-dir", str(out), "--jobs", "1"]
    assert tool.main(argv + ["--selection", str(path)]) == 1
    assert "split_k" in capsys.readouterr().err
    assert not out.exists()


def test_main_keeps_only_the_selected_configs_and_direct_instances(
    tmp_path, monkeypatch, synthetic_grid, direct_fp16
):
    implicit, direct = synthetic_grid[:12], direct_fp16[0][:4]
    chosen = [i for i in implicit if i["pipeline"] == "mem"]
    selection = _selection_of(chosen, direct[:2], excluded=chosen[:1])
    written = _fake_main(tmp_path, monkeypatch, implicit, direct, selection=selection)
    for dtype, entries in written.items():
        hashes = {tool.selection_hash(e) for e in entries}
        assert hashes == {tool.selection_hash(i) for i in chosen[1:] + direct[:2]}


@pytest.mark.parametrize("stale", ["config", "excluded", "direct"])
def test_main_refuses_a_selection_naming_what_the_sweep_lacks(
    tmp_path, monkeypatch, synthetic_grid, direct_fp16, capsys, stale
):
    """A selection that outlives a rocKE grid change fails rather than silently
    shrinking the catalog."""
    implicit, direct = synthetic_grid[:4], direct_fp16[0][:2]
    gone = _identity(tile_m=128)
    selection = _selection_of(
        implicit + ([gone] if stale == "config" else []),
        direct + ([dict(direct[0], knobs="gone")] if stale == "direct" else []),
        excluded=[gone] if stale == "excluded" else [],
    )
    monkeypatch.setattr(tool, "sweep_identities", lambda dtype, jobs: implicit)
    monkeypatch.setattr(
        tool, "direct_identities", lambda dtype, jobs: (direct, collections.Counter())
    )
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    out = tmp_path / "lists"
    argv = ["--rocke-root", str(_ROCKE), "--output-dir", str(out), "--jobs", "1"]
    assert tool.main(argv + ["--selection", str(path)]) == 1
    assert "the sweep lacks" in capsys.readouterr().err
    assert not out.exists()


def test_committed_instance_lists_are_what_the_tool_enumerates(tmp_path):
    """The shipped catalog's lists are the tool's output for the rocKE tree beside it:
    a rocKE grid change that moves the catalog has to regenerate the lists (and the
    bundle make_conv_aot_config.py builds from them)."""
    from benchmarks.common.kernel_cache import current_llvm_flavor

    committed = {
        dtype: _CONFIGS / tool.INSTANCE_LIST.format(dtype=dtype)
        for dtype in tool.DTYPES
    }
    with committed["fp16"].open() as handle:
        handle.readline()
        flavor = json.loads(handle.readline().rstrip(",\n"))["llvm_flavor"]
    if flavor != current_llvm_flavor():
        pytest.skip(
            f"the lists are for LLVM flavor {flavor}, this process lowers to "
            f"{current_llvm_flavor()}, so every identity differs"
        )
    out = tmp_path / "lists"
    jobs = str(min(16, os.cpu_count() or 1))
    argv = ["--rocke-root", str(_ROCKE), "--output-dir", str(out), "--jobs", jobs]
    assert tool.main(argv) == 0
    for dtype, path in committed.items():
        written = out / tool.INSTANCE_LIST.format(dtype=dtype)
        assert written.read_bytes() == path.read_bytes(), dtype
