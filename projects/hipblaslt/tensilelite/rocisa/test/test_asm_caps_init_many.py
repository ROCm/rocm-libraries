# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import os
import shutil
import subprocess
import sys

import rocisa

_ISAS = [(9, 4, 2), (9, 5, 0), (12, 0, 1)]

# rocIsa caches capabilities per ISA for the life of the process, so each way of
# probing runs in a fresh interpreter that imports the same rocisa as this one.
_PROBE = """
import json, sys
sys.path[:] = json.loads(sys.argv[1])
import rocisa
isas = [tuple(isa) for isa in json.loads(sys.argv[2])]
ti = rocisa.rocIsa.getInstance()
if sys.argv[4] == "many":
    ti.initMany(isas + isas[:1], sys.argv[3], False)
for isa in isas:
    ti.init(isa, sys.argv[3], False)
caps = {}
for isa in isas:
    info = ti.getIsaInfo(isa)
    caps[str(isa)] = {
        "asmCaps": dict(info.asmCaps),
        "archCaps": dict(info.archCaps),
        "regCaps": dict(info.regCaps),
        "asmBugs": dict(info.asmBugs),
    }
print(json.dumps(caps, sort_keys=True))
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


def _probe(mode):
    parent_path = json.dumps([os.path.abspath(p) for p in sys.path])
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, parent_path, json.dumps(_ISAS), _assembler(), mode],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_concurrent_probing_matches_serial_probing():
    serial = _probe("serial")
    concurrent = _probe("many")

    # The probe must actually have reached an assembler.
    assert any(caps["asmCaps"]["SupportedISA"] == 1 for caps in serial.values())
    assert concurrent == serial


def test_init_many_accepts_any_iterable():
    # Empty inputs probe nothing, so no assembler is needed.
    ti = rocisa.rocIsa.getInstance()
    ti.initMany((), "unused", False)
    ti.initMany(iter([]), "unused", False)
