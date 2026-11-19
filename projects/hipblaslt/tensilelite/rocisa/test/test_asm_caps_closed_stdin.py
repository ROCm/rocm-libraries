# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import os
import shutil
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


def _probe_asm_caps(close_stdin):
    parent_path = json.dumps([os.path.abspath(p) for p in sys.path])
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, parent_path, json.dumps(_ISA), _assembler()],
        preexec_fn=(lambda: os.close(0)) if close_stdin else None,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(os.name != "posix", reason="closes fd 0 in the probe process")
def test_asm_caps_probe_matches_with_closed_stdin():
    # With fd 0 closed, the probe pipe's read end is allocated as fd 0. The
    # assembler must still get the probe source on stdin; otherwise it assembles
    # nothing, succeeds, and every capability is reported as supported.
    normal_caps = _probe_asm_caps(close_stdin=False)
    closed_stdin_caps = _probe_asm_caps(close_stdin=True)

    assert normal_caps["SupportedISA"] == 1
    assert 0 in normal_caps.values()
    assert closed_stdin_caps == normal_caps
