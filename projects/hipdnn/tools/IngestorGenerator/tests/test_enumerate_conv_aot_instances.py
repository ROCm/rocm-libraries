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
    from kernels.conv.kernel_identity import KernelIdentity

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


def _label(identity):
    from kernels.conv.kernel_identity import KernelIdentity

    return KernelIdentity.from_dict(identity).label()


@pytest.fixture(scope="module")
def catalog():
    """The committed catalog's identities, in file order."""
    return tool.read_labels(tool.CATALOG_LABELS)


def test_the_catalog_lists_207_kernels_per_dtype_the_same_in_both(catalog):
    """79 implicit-GEMM and 128 direct kernels per dtype, and the two dtypes list the
    same kernels."""
    by_dtype = collections.defaultdict(list)
    for identity in catalog:
        by_dtype[identity["dtype_a"]].append(identity)
    assert sorted(by_dtype) == sorted(tool.DTYPES)
    for identities in by_dtype.values():
        families = collections.Counter(
            i["algorithm"] == "implicit_gemm" for i in identities
        )
        assert families == {True: 79, False: 128}

    def without_dtype(identity):
        return tuple(
            sorted((k, v) for k, v in identity.items() if not k.startswith("dtype_"))
        )

    assert sorted(map(without_dtype, by_dtype["fp16"])) == sorted(
        map(without_dtype, by_dtype["bf16"])
    )


def test_every_catalog_entry_is_a_forward_sweep_identity(catalog):
    """Each entry is a gfx950 forward identity of rocKE's implicit-GEMM AOT grid or of
    its direct grid."""
    jobs = min(16, os.cpu_count() or 1)
    for dtype in tool.DTYPES:
        swept = tool.sweep_hashes(dtype, jobs)
        listed = [i for i in catalog if i["dtype_a"] == dtype]
        assert all(tool.stable_hash(i) in swept for i in listed), dtype


def test_every_catalog_entry_converts(catalog):
    """The packaged launch runs every listed kernel: the converter refuses none."""
    import make_conv_aot_config

    assert tool.validate(catalog, min(16, os.cpu_count() or 1)) == []
    for identity in catalog:
        spec, _ = make_conv_aot_config.instance_spec(identity)
        assert spec.is_direct == (identity["algorithm"] != "implicit_gemm")


def test_read_labels_skips_comments_and_blank_lines(tmp_path, synthetic_grid):
    labels = tmp_path / "labels.txt"
    body = [_label(i) for i in synthetic_grid[:3]]
    labels.write_text("# a comment\n\n" + "\n".join(body) + "\n\n")
    assert tool.read_labels(labels) == synthetic_grid[:3]


def test_read_labels_refuses_a_repeated_kernel(tmp_path, synthetic_grid):
    labels = tmp_path / "labels.txt"
    labels.write_text(f"{_label(synthetic_grid[0])}\n{_label(synthetic_grid[0])}\n")
    with pytest.raises(ValueError, match="listed twice"):
        tool.read_labels(labels)


def test_read_labels_refuses_a_malformed_label(tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("fwd-implicit_gemm-fp16-not-a-label\n")
    with pytest.raises(ValueError):
        tool.read_labels(labels)


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
    from kernels.conv.kernel_identity import KernelIdentity

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


def _fake_main(tmp_path, monkeypatch, identities, swept=None):
    """Run main() over a label file of ``identities``, every one of them a sweep
    identity unless ``swept`` says otherwise. Returns main()'s status and the written
    lists, by dtype."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels = tmp_path / "labels.txt"
    labels.write_text("".join(f"{_label(i)}\n" for i in identities))
    hashes = {tool.stable_hash(i) for i in identities} if swept is None else swept
    monkeypatch.setattr(tool, "sweep_hashes", lambda dtype, jobs: hashes)
    out = tmp_path / "lists"
    argv = ["--rocke-root", str(_ROCKE), "--labels", str(labels)]
    status = tool.main(argv + ["--output-dir", str(out), "--jobs", "1"])
    if not out.exists():
        return status, None
    return status, {
        dtype: json.loads((out / tool.INSTANCE_LIST.format(dtype=dtype)).read_text())
        for dtype in tool.DTYPES
    }


def _both_dtypes(identities):
    return [
        dict(i, dtype_a=dtype, dtype_b=dtype, dtype_d=dtype)
        for dtype in tool.DTYPES
        for i in identities
    ]


def test_main_writes_one_list_per_dtype_ordered_by_family_then_hash(
    tmp_path, monkeypatch, synthetic_grid, catalog
):
    implicit = synthetic_grid[:12]
    # One dtype only: _both_dtypes reassigns dtype_a/b/d below, so two entries that
    # differ only by their original dtype would collide into the same label.
    direct = [
        i
        for i in catalog
        if i["algorithm"] != "implicit_gemm" and i["dtype_a"] == "bf16"
    ][:5]
    status, written = _fake_main(tmp_path, monkeypatch, _both_dtypes(implicit + direct))
    assert status == 0
    assert sorted(written) == sorted(tool.DTYPES)
    for dtype, entries in written.items():
        assert len(entries) == len(implicit) + len(direct)
        assert {e["dtype_a"] for e in entries} == {dtype}
        keys = [
            (e["algorithm"] != "implicit_gemm", tool.stable_hash(e)) for e in entries
        ]
        assert keys == sorted(keys)
        assert sum(1 for k in keys if k[0]) == len(direct)


def test_main_writes_the_same_lists_whatever_the_label_order(
    tmp_path, monkeypatch, synthetic_grid
):
    identities = _both_dtypes(synthetic_grid[:20])
    _, first = _fake_main(tmp_path / "a", monkeypatch, identities)
    _, second = _fake_main(tmp_path / "b", monkeypatch, list(reversed(identities)))
    assert first == second


def test_main_refuses_a_kernel_outside_the_sweeps(
    tmp_path, monkeypatch, synthetic_grid, capsys
):
    """The catalog names sweep kernels; anything else is an error, and nothing is
    written."""
    status, written = _fake_main(
        tmp_path, monkeypatch, _both_dtypes(synthetic_grid[:2]), swept=set()
    )
    assert status == 1 and written is None
    assert "not a gfx950 forward sweep identity" in capsys.readouterr().err


def test_main_refuses_a_kernel_the_converter_refuses(
    tmp_path, monkeypatch, synthetic_grid, capsys
):
    """A listed kernel the packaged launch cannot run is an error, never a silent
    omission."""
    unsupported = dict(synthetic_grid[0], split_k=2)
    status, written = _fake_main(tmp_path, monkeypatch, [unsupported])
    assert status == 1 and written is None
    assert "split_k" in capsys.readouterr().err


def test_main_refuses_a_dtype_the_catalog_does_not_package(
    tmp_path, monkeypatch, synthetic_grid, capsys
):
    fp32 = dict(synthetic_grid[0], dtype_a="fp32", dtype_b="fp32", dtype_d="fp32")
    status, written = _fake_main(tmp_path, monkeypatch, [fp32])
    assert status == 1 and written is None
    assert "fp32" in capsys.readouterr().err


def test_committed_instance_lists_are_what_the_tool_enumerates(tmp_path):
    """The shipped catalog's lists are the tool's output for the committed labels and
    the rocKE tree beside them: a change to either has to regenerate the lists (and
    the bundle make_conv_aot_config.py builds from them)."""
    from kernels.conv.kernel_identity import current_llvm_flavor

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
