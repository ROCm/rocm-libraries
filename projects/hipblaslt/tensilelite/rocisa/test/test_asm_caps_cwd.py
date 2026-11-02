# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import os
import shutil
import stat
import subprocess
import sys

import pytest

_ISA = (9, 0, 10)

# rocIsa caches capabilities per ISA for the life of the process, so each probe
# runs in a fresh interpreter that imports the same rocisa as this one.
_PROBE = """
import json, sys
sys.path[:] = json.loads(sys.argv[1])
import rocisa
isa = tuple(json.loads(sys.argv[2]))
ti = rocisa.rocIsa.getInstance()
ti.init(isa, sys.argv[3], False)
ti.setKernel(isa, 64)
print(json.dumps(dict(ti.getAsmCaps())))
"""


def _assembler():
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join(
        [
            os.path.join(rocm_path, "bin"),
            os.path.join(rocm_path, "lib", "llvm", "bin"),
        ]
    )
    return shutil.which("amdclang++", path=search_path) or "amdclang++"


def _probe_asm_caps(cwd):
    parent_path = json.dumps([os.path.abspath(p) for p in sys.path])
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, parent_path, json.dumps(_ISA), _assembler()],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _entries(path):
    return sorted(p.name for p in path.iterdir())


def test_asm_caps_probe_writes_nothing_to_cwd(tmp_path):
    caps = _probe_asm_caps(tmp_path)
    assert caps["SupportedISA"] == 1
    assert _entries(tmp_path) == []


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="needs a POSIX directory that this process cannot write to",
)
def test_asm_caps_probe_matches_in_read_only_cwd(tmp_path):
    writable = tmp_path / "writable"
    read_only = tmp_path / "read_only"
    writable.mkdir()
    read_only.mkdir()
    read_only.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        read_only_caps = _probe_asm_caps(read_only)
    finally:
        read_only.chmod(stat.S_IRWXU)
    writable_caps = _probe_asm_caps(writable)

    assert writable_caps["SupportedISA"] == 1
    assert read_only_caps == writable_caps
    assert _entries(read_only) == []
    assert _entries(writable) == []
