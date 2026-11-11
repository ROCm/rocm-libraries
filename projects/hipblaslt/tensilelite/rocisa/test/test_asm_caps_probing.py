# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import os
import signal
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="stands in for the assembler with shell scripts"
)

# rocIsa keeps capabilities for the life of the process, so each probe runs in a
# fresh interpreter that imports the same rocisa as this one.
_PROBE = """
import json, signal, sys
sys.path[:] = json.loads(sys.argv[1])
import rocisa
isas = [tuple(isa) for isa in json.loads(sys.argv[2])]
debug = sys.argv[4] == "1"
interval = float(sys.argv[5])
ti = rocisa.rocIsa.getInstance()
if interval:
    # Interrupts the system calls probing waits in, as a SIGALRM-based timeout would.
    signal.signal(signal.SIGALRM, lambda *_: None)
    signal.setitimer(signal.ITIMER_REAL, interval, interval)
for isa in isas:
    ti.init(isa, sys.argv[3], debug)
signal.setitimer(signal.ITIMER_REAL, 0)
print(json.dumps({str(isa): dict(ti.getIsaInfo(isa).asmCaps) for isa in isas}, sort_keys=True))
"""


def _probe(assembler, isas, debug=False, interval=0):
    parentPath = json.dumps([os.path.abspath(p) for p in sys.path])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            parentPath,
            json.dumps(isas),
            str(assembler),
            "1" if debug else "0",
            str(interval),
        ],
        # Under nohup the runner ignores SIGHUP and its children inherit that, so
        # restore it for the probe, whose assembler may have to die of it.
        preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_DFL),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    # debug prints each assembler run first.
    return json.loads(result.stdout.splitlines()[-1])


def _script(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


def test_probes_interrupted_by_a_signal_still_get_the_answer(tmp_path):
    # Python's signal handlers don't restart interrupted system calls, so a
    # timer going off while a probe waits for the assembler fails the wait with
    # EINTR. Taken as an answer, that wait reads as an accepted source.
    script = _script(
        tmp_path / "amdclang++", 'src=$(cat)\n[ -n "$src" ] || exit 0\nsleep 0.005\nexit 1\n'
    )

    uninterrupted = _probe(script, [(9, 4, 2)])
    assert all(c["SupportedISA"] == 1 for c in uninterrupted.values())
    assert _probe(script, [(9, 4, 2)], interval=0.001) == uninterrupted
