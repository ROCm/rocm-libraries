# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The CI-matched venv setup script: pin parsing, nightly choice, venv reuse
and the activate hooks, without network or GPU."""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def _script_path() -> Path:
    """Find the script in source and relocated test layouts."""
    test_file = Path(__file__).resolve()
    candidates = (
        test_file.parents[3] / "tools" / "setup_dev_env.py",
        test_file.with_name("setup_dev_env.py"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.skip("top-level setup_dev_env.py is absent from this platform-only tree")


@pytest.fixture(scope="module")
def sde():
    spec = importlib.util.spec_from_file_location("setup_dev_env", _script_path())
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


PIN = "11621df52cb7e5790ab7a37515098cde8faed4ab"
LLVM = "6c98155189a4bcf6924a1a36196388e61794309b"
OTHER_LLVM = "f1eda33bf9000000000000000000000000000000"


def test_parse_therock_ref_reads_the_value_under_its_key(sde):
    text = (
        "outputs:\n"
        "  therock-ref:\n"
        '    description: "TheRock commit hash to use"\n'
        f'    value: "{PIN}" # 2026-11-17\n'
        "  kpack-ref:\n"
        '    value: "a022846cf553c2b135410a5168f97705f1b9c6ac"\n'
    )
    assert sde.parse_therock_ref(text) == PIN
    with pytest.raises(sde.SetupError):
        sde.parse_therock_ref("outputs:\n  kpack-ref:\n    value: x\n")


def test_venv_group_differs_per_python(sde, monkeypatch):
    monkeypatch.setattr(sde, "python_tag", lambda: "cp312")
    cp312 = sde.venv_group("gfx942", LLVM)
    monkeypatch.setattr(sde, "python_tag", lambda: "cp313")
    assert cp312 != sde.venv_group("gfx942", LLVM)
    assert cp312 == "gfx942-6c981551-cp312"


def _stamped_venv(sde, root: Path, python: str) -> Path:
    venv = root / "venv"
    exe = sde.venv_python(venv)
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    sde.write_stamp(venv, {"schema": sde.STAMP_SCHEMA, "python": python})
    return venv


def test_claim_venv_reuses_a_complete_venv(sde, tmp_path):
    venv = _stamped_venv(sde, tmp_path, sde.python_tag())
    assert sde.claim_venv(venv)["python"] == sde.python_tag()


def test_claim_venv_removes_only_an_unfinished_install(sde, tmp_path):
    venv = tmp_path / "venv"
    (venv / "lib").mkdir(parents=True)
    assert sde.claim_venv(venv) is None
    assert not venv.exists()


def test_claim_venv_never_deletes_a_venv_stamped_for_another_python(sde, tmp_path):
    # A shared base built by another interpreter belongs to other checkouts and
    # running jobs; a run that cannot use it must leave it alone.
    venv = _stamped_venv(sde, tmp_path, "cp399")
    with pytest.raises(sde.SetupError):
        sde.claim_venv(venv)
    assert (venv / sde.STAMP).is_file()


@pytest.mark.parametrize("signed", [False, True])
def test_write_hook_is_idempotent(sde, tmp_path, signed):
    script = tmp_path / "activate"
    body = "deactivate () { :; }\nexport PATH\n"
    if signed:
        body += "# SIG # Begin signature block\n# abc\n# SIG # End signature block\n"
    script.write_text(body)
    sde._write_hook(script, ["echo one"])
    first = script.read_text()
    sde._write_hook(script, ["echo one"])
    assert script.read_text() == first
    if signed:
        assert first.index(sde.HOOK_END) < first.index("# SIG # Begin")
        assert first.startswith("deactivate () { :; }\nexport PATH\n")


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="needs POSIX bash"
)
def test_bash_hook_restores_previous_values_on_deactivate(sde, tmp_path):
    activate = tmp_path / "activate"
    activate.write_text("deactivate () { :; }\n")
    env = {"ROCKE_COMGR_LIB": "/venv/libamd_comgr.so", "ROCKE_HIP_LIB": "/venv/hip.so"}
    sde._write_hook(activate, sde._bash_hook(env, tmp_path, PIN))
    probe = (
        "export ROCKE_COMGR_LIB=/before/comgr.so; unset ROCKE_HIP_LIB\n"
        f". {activate}\n"
        'echo "on:$ROCKE_COMGR_LIB:$ROCKE_HIP_LIB"\n'
        f". {activate}\n"  # a second activation must not save the venv's values
        "deactivate\n"
        'echo "off:$ROCKE_COMGR_LIB:${ROCKE_HIP_LIB-unset}"\n'
    )
    out = subprocess.run(
        ["bash", "-c", probe], capture_output=True, text=True, check=True
    ).stdout.split()
    assert out == [
        "on:/venv/libamd_comgr.so:/venv/hip.so",
        "off:/before/comgr.so:unset",
    ]


def test_zip64_extra_supplies_large_size_and_offset(sde):
    import struct

    extra = struct.pack("<HHQQQ", 1, 24, 2**33, 2**32 + 5, 2**34)
    assert sde._zip64_sizes(extra, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == (
        2**32 + 5,
        2**34,
    )


def test_pick_torch_takes_the_newest_build_with_a_device_wheel(sde):
    nightly = "10.3.0a20261118"
    torch = {f"2.15.0+rocm{nightly}", f"2.16.0a0+rocm{nightly}", "2.17.0+rocm9.9.9"}
    device = {f"2.15.0+rocm{nightly}", f"2.16.0a0+rocm{nightly}"}
    assert sde.pick_torch(nightly, torch, device) == f"2.16.0a0+rocm{nightly}"
    assert sde.pick_torch(nightly, torch, set()) is None
    assert sde.torch_key("2.16.0+rocmX") > sde.torch_key("2.16.0a0+rocmX")


def _fake_index(sde, monkeypatch, plat, py, arch, nightlies, torch_nightlies):
    def wheel(dist, ver, tags):
        return sde._WHEEL.match(f"{dist}-{ver}-{tags}-{plat}.whl")

    pages = {
        "rocm-sdk-core": [wheel("rocm_sdk_core", v, "py3-none") for v in nightlies],
        "torch": [
            wheel("torch", f"2.16.0a0+rocm{v}", f"{py}-{py}") for v in torch_nightlies
        ],
        f"amd-torch-device-{arch}": [
            wheel(f"amd_torch_device_{arch}", f"2.16.0a0+rocm{v}", f"{py}-{py}")
            for v in torch_nightlies
        ],
    }
    monkeypatch.setattr(sde, "index_wheels", lambda project: pages[project])


def test_resolve_nightly_picks_the_newest_build_of_ci_llvm(sde, monkeypatch):
    plat, py = sde.wheel_platform(), sde.python_tag()
    nightlies = ["10.3.0a20261117", "10.3.0a20261118", "10.3.0a20261119"]
    _fake_index(sde, monkeypatch, plat, py, "gfx942", nightlies, nightlies)
    cache = {
        "nightly_llvm": {
            f"{plat}/10.3.0a20261117": LLVM,
            f"{plat}/10.3.0a20261118": LLVM,
            f"{plat}/10.3.0a20261119": OTHER_LLVM,
        }
    }
    res = sde.resolve_nightly(PIN, LLVM, "gfx942", cache)
    assert (res.nightly, res.torch) == (
        "10.3.0a20261118",
        "2.16.0a0+rocm10.3.0a20261118",
    )


def test_resolve_nightly_refuses_a_different_compiler(sde, monkeypatch):
    plat, py = sde.wheel_platform(), sde.python_tag()
    _fake_index(sde, monkeypatch, plat, py, "gfx942", ["10.3.0a20261119"], [])
    cache = {"nightly_llvm": {f"{plat}/10.3.0a20261119": OTHER_LLVM}}
    with pytest.raises(sde.SetupError, match="Refusing"):
        sde.resolve_nightly(PIN, LLVM, "gfx942", cache)


def _serve(sde, monkeypatch, blob: bytes, honor_range: bool = True):
    from email.message import Message

    def fetch(url, headers=None, method="GET"):
        meta = Message()
        meta["Content-Length"] = str(len(blob))
        if method == "HEAD":
            return 200, meta, b""
        first, last = map(int, headers["Range"].split("=")[1].split("-"))
        if not honor_range:
            return 200, meta, blob
        return 206, meta, blob[first : last + 1]

    monkeypatch.setattr(sde, "_fetch", fetch)


def test_read_wheel_member_extracts_one_file_by_byte_ranges(sde, monkeypatch):
    manifest = {"submodules": [{"submodule_name": "llvm-project", "pin_sha": LLVM}]}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("_rocm_sdk_core/lib/big.so", b"\0" * 100_000)
        zf.writestr(sde.MANIFEST, json.dumps(manifest))
    _serve(sde, monkeypatch, buf.getvalue())
    cache: dict = {}
    assert sde.nightly_llvm("10.3.0a20261118", "linux_x86_64", cache) == LLVM
    assert sde.longest_member("https://example/x.whl") == len(sde.MANIFEST)


def test_a_server_ignoring_ranges_is_a_setup_error(sde, monkeypatch):
    _serve(sde, monkeypatch, b"PK" * 100, honor_range=False)
    with pytest.raises(sde.SetupError, match="byte-range"):
        sde.read_wheel_member("https://example/x.whl", sde.MANIFEST)
