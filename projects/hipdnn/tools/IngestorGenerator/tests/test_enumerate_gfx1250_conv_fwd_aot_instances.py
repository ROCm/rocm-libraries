# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for tools/enumerate_gfx1250_conv_fwd_aot_instances.py (no compile, no GPU)."""

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="the rocKE builders need numpy")

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))

import enumerate_gfx1250_conv_fwd_aot_instances as tool  # noqa: E402
import launch_surface  # noqa: E402

_REPO = launch_surface.find_repo_root(_TOOLS)
_ROCKE = _REPO / "dnn-providers" / "hip-kernel-provider" / "rocke"
_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
_JOBS = min(16, os.cpu_count() or 1)


@pytest.fixture(scope="module", autouse=True)
def _rocke_on_path():
    """Bind rocKE for this module only (see test_enumerate_conv_aot_instances)."""
    before = list(sys.path)
    tool._bind_rocke(_ROCKE)
    yield
    sys.path[:] = before


@pytest.fixture(scope="module")
def catalog():
    """rocKE's registered gfx1250 forward catalog, in registration order."""
    return tool.read_rocke_catalog()


@pytest.fixture
def catalog_flavor(catalog, monkeypatch):
    """Lower with the LLVM flavor the catalog's instances are named for (see
    check_flavor)."""
    (flavor,) = {i["llvm_flavor"] for i in catalog}
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
    return flavor


@pytest.fixture(scope="module")
def committed():
    """The committed per-dtype instance lists."""
    paths = {
        dtype: _CONFIGS / tool.INSTANCE_LIST.format(dtype=dtype)
        for dtype in tool.DTYPES
    }
    missing = sorted(str(p) for p in paths.values() if not p.exists())
    if missing:
        pytest.skip(
            "DVC-tracked instance list(s) not pulled, run `dvc pull` in "
            f"{_CONFIGS}: {missing}"
        )
    return {dtype: json.loads(path.read_text()) for dtype, path in paths.items()}


def test_the_lists_are_rockes_catalog_in_registration_order(catalog, committed):
    """The lists hold exactly rocKE's registered instances (no excludes are declared),
    each dtype's in the order rocKE registers them: that order is the catalog's
    priority order."""
    assert not tool.EXCLUDES.exists()
    for dtype, entries in committed.items():
        expected = [i for i in catalog if i["dtype_a"] == dtype]
        assert [tool.stable_hash(e) for e in entries] == [
            tool.stable_hash(i) for i in expected
        ]


def test_the_lists_are_written_compact(committed):
    """Each entry is the compact form of its identity (fields at their default left out),
    so it reads back as exactly that identity."""
    from kernels.conv.kernel_identity import KernelIdentity

    for entries in committed.values():
        for entry in entries:
            assert entry == tool.compact_identity(
                KernelIdentity.from_dict(entry).to_dict()
            )


def test_every_list_entry_is_a_forward_sweep_identity(committed, monkeypatch):
    """Each entry is a gfx1250 forward identity of rocKE's implicit-GEMM AOT grid or of
    its direct grid."""
    for dtype, entries in committed.items():
        (flavor,) = {entry["llvm_flavor"] for entry in entries}
        monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
        swept = tool.sweep_hashes(dtype, _JOBS)
        missing = [
            tool.identity_label(e) for e in entries if tool.stable_hash(e) not in swept
        ]
        assert not missing, missing[:5]


def test_every_list_entry_converts(committed):
    entries = [e for dtype in tool.DTYPES for e in committed[dtype]]
    assert tool.validate(entries) == []


def test_main_writes_the_lists_and_refuses_a_stale_exclude(
    tmp_path, catalog, catalog_flavor, capsys
):
    out = tmp_path / "configs"
    assert (
        tool.main(
            [
                "--rocke-root",
                str(_ROCKE),
                "--output-dir",
                str(out),
                "--jobs",
                str(_JOBS),
            ]
        )
        == 0
    )
    for dtype in tool.DTYPES:
        written = json.loads((out / tool.INSTANCE_LIST.format(dtype=dtype)).read_text())
        assert len(written) == sum(1 for i in catalog if i["dtype_a"] == dtype)
    excludes = tmp_path / "excludes.json"
    excludes.write_text(
        json.dumps([{"label": "fwd-no-such-instance", "reason": "test"}])
    )
    argv = [
        "--rocke-root",
        str(_ROCKE),
        "--output-dir",
        str(tmp_path / "x"),
        "--excludes",
        str(excludes),
    ]
    assert tool.main(argv) == 1
    assert "exclude not in rocKE's gfx1250 fwd catalog" in capsys.readouterr().err
    assert not (tmp_path / "x").exists()


def test_an_exclude_drops_its_instance(tmp_path, catalog, catalog_flavor):
    victim = catalog[0]
    excludes = tmp_path / "excludes.json"
    excludes.write_text(
        json.dumps([{"label": tool.identity_label(victim), "reason": "test"}])
    )
    out = tmp_path / "configs"
    argv = [
        "--rocke-root",
        str(_ROCKE),
        "--output-dir",
        str(out),
        "--excludes",
        str(excludes),
    ]
    assert tool.main(argv) == 0
    written = json.loads(
        (out / tool.INSTANCE_LIST.format(dtype=victim["dtype_a"])).read_text()
    )
    assert tool.stable_hash(victim) not in {tool.stable_hash(e) for e in written}


def test_main_names_the_flavor_it_needs(tmp_path, catalog_flavor, monkeypatch, capsys):
    """Under another LLVM flavor every entry would look like a stranger to the sweep;
    the tool says which flavor to rerun with instead, and writes nothing."""
    monkeypatch.setenv(
        "ROCKE_LLVM_FLAVOR", "llvm22" if catalog_flavor != "llvm22" else "llvm24"
    )
    out = tmp_path / "configs"
    assert tool.main(["--rocke-root", str(_ROCKE), "--output-dir", str(out)]) == 1
    err = capsys.readouterr().err
    assert f"ROCKE_LLVM_FLAVOR={catalog_flavor}" in err
    assert "not a gfx1250 forward sweep identity" not in err
    assert not out.exists()


def test_sweep_hashes_names_identities_for_this_processs_flavor(
    catalog, catalog_flavor, monkeypatch
):
    """A reused worker pool may hand back identities named for the flavor it was
    started under; sweep_hashes answers under this process's flavor regardless."""
    from dataclasses import replace
    from types import SimpleNamespace

    from benchmarks.common import direct_kernel_sweep, kernel_sweep
    from kernels.conv.kernel_identity import KernelIdentity

    stranger = "llvm22" if catalog_flavor != "llvm22" else "llvm24"
    dtype = "fp16"
    identities = [KernelIdentity.from_dict(i) for i in catalog if i["dtype_a"] == dtype]

    def jobs(direct):
        return [
            SimpleNamespace(identity=replace(i, llvm_flavor=stranger))
            for i in identities
            if i.is_direct == direct
        ]

    monkeypatch.setattr(kernel_sweep, "enumerate_jobs", lambda **_: jobs(False))
    monkeypatch.setattr(
        direct_kernel_sweep, "enumerate_direct_jobs", lambda **_: jobs(True)
    )
    assert any(i.is_direct for i in identities)
    assert any(not i.is_direct for i in identities)
    assert tool.sweep_hashes(dtype, 1) == {i.stable_hash() for i in identities}
