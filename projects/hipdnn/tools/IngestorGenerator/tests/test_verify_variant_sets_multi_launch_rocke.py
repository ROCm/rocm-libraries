# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""`verify_variant_sets --mode full` over a multi-launch bundle packed by the real
compiler.

The two packaging fixtures (`multi_launch_decode_merge`, `multi_launch_same_kernel_twice`)
are packed here with rocke and libamd_comgr, and the gate reads the result: every launch's
code object is fetched from the archive and checked against that launch's compiler
evidence. A machine without rocke, comgr or hipcc FAILS these tests rather than skipping
them, the way the packaging suite's `tests/rocke/conftest.py` does: nothing here is
meaningful without the real compiler.

The environment is the packaging suite's: HIPKERNELPROVIDER_ROCM_KPACK_DIR (the rocm_kpack
python directory), HKP_HIPCC, and PYTHONPATH holding the packer's python directory.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import verify_variant_sets as gate_module  # noqa: E402

_DESCRIPTOR_PACKAGING = gate_module._agreement_python_root().parent
_ROCKE_ROOT = _DESCRIPTOR_PACKAGING.parent / "rocke"
_KPACK_DIR = os.environ.get("HIPKERNELPROVIDER_ROCM_KPACK_DIR") or None
for _path in (
    _DESCRIPTOR_PACKAGING / "tests",
    _ROCKE_ROOT / "platform" / "python",
    _ROCKE_ROOT / "library",
    *([Path(_KPACK_DIR)] if _KPACK_DIR and Path(_KPACK_DIR).is_dir() else []),
):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from hkp_pack import provenance_sidecar  # noqa: E402
from pack_helpers import run_pipeline  # noqa: E402

ARCH = "gfx1151"
FIXTURES = _DESCRIPTOR_PACKAGING / "tests" / "fixtures"


@pytest.fixture(scope="session")
def hipcc():
    exe = os.environ.get("HKP_HIPCC") or shutil.which("hipcc")
    if not exe:
        pytest.fail("hipcc not found (set HKP_HIPCC or put hipcc on PATH)")
    return exe


@pytest.fixture(scope="session")
def rocke_available():
    """rocke, its kernels library and libamd_comgr, or a failure naming the gap."""
    try:
        import kernels  # noqa: F401
        import rocke  # noqa: F401
        from rocke.runtime import comgr

        comgr._resolve_lib()
    except Exception as exc:
        pytest.fail(
            f"rocke or libamd_comgr is not usable: {exc}. Provision both; they are "
            f"requirements of the multi-launch full-mode gate test."
        )


def _pack(fixture: str, tmp_path: Path, hipcc: str) -> Path:
    out = tmp_path / "out"
    run_pipeline(
        source_root=FIXTURES / fixture,
        arches=[ARCH],
        out_root=out,
        hipcc=hipcc,
        rocm_kpack_dir=_KPACK_DIR,
        inter_root=tmp_path / "inter",
    )
    return out / ARCH


def _gate_full(root: Path, tmp_path: Path, capsys):
    """The CLI in full mode over `root`. A manifest vocabulary on the one metadata
    field keeps the vocabulary check from being reported as not run."""
    kdp = json.loads((root / "multi_launch.kdp.json").read_text())
    family = kdp["kernelDescriptors"][0]["metadata"]["family"]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"specialization": {"vocabulary": {"family": [family]}}})
    )
    argv = ["--mode", "full", "--manifest", str(manifest)]
    if _KPACK_DIR:
        argv += ["--kpack-python-dir", _KPACK_DIR]
    code = gate_module.main([*argv, "packed", str(root)])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def _corrupt_kdp(root: Path, edit) -> None:
    """Edit the packed KDP's first UKD and re-seal it: the sidecar binds the
    descriptor's digest, so the provenance is reattached, the edit made, and both
    written back the way the packer ships them."""
    path = root / "multi_launch.kdp.json"
    doc = provenance_sidecar.attach(path, json.loads(path.read_text()))
    edit(doc["kernelDescriptors"][0]["sources"])
    name, data = provenance_sidecar.detach(path.name, doc)
    path.with_name(name).write_bytes(data)
    path.write_text(json.dumps(doc, separators=(",", ":")))


@pytest.mark.parametrize(
    "fixture", ["multi_launch_decode_merge", "multi_launch_same_kernel_twice"]
)
def test_the_gate_passes_a_packed_multi_launch_bundle(
    tmp_path, capsys, hipcc, rocke_available, fixture
):
    root = _pack(fixture, tmp_path, hipcc)
    code, output = _gate_full(root, tmp_path, capsys)
    assert code == 0, output
    assert "GATE PASSED" in output


def test_a_launch_whose_archive_member_is_missing_is_named(
    tmp_path, capsys, hipcc, rocke_available
):
    root = _pack("multi_launch_decode_merge", tmp_path, hipcc)

    def drop_member(sources):
        sources["merge"]["toc_key"] = "no_such_member"

    _corrupt_kdp(root, drop_member)
    code, output = _gate_full(root, tmp_path, capsys)
    assert code == 1
    assert "launch 'merge'" in output
    assert "launch 'decode'" not in output


def test_launches_whose_symbols_are_swapped_fail_naming_their_launches(
    tmp_path, capsys, hipcc, rocke_available
):
    """Each launch's entry is checked against its own compiler evidence, so a
    descriptor that points `decode` at the merge kernel cannot pass."""
    root = _pack("multi_launch_decode_merge", tmp_path, hipcc)

    def swap(sources):
        sources["decode"]["symbol"], sources["merge"]["symbol"] = (
            sources["merge"]["symbol"],
            sources["decode"]["symbol"],
        )

    _corrupt_kdp(root, swap)
    code, output = _gate_full(root, tmp_path, capsys)
    assert code == 1
    assert "launch 'decode'" in output, output
