# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import os
import shutil
import signal
import subprocess
import sys

import pytest

_shell_assembler = pytest.mark.skipif(
    os.name != "posix", reason="stands in for the assembler with shell scripts"
)

_ISAS = [(9, 0, 10), (9, 4, 2), (9, 5, 0), (11, 0, 0), (12, 0, 1), (12, 5, 0)]

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
if interval:
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
        preexec_fn=(lambda: signal.signal(signal.SIGHUP, signal.SIG_DFL)) if os.name == "posix" else None,
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


def _runs(log):
    return len(log.read_text().splitlines()) if log.exists() else 0


@pytest.fixture
def assembler():
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join(
        [os.path.join(rocm_path, "bin"), os.path.join(rocm_path, "lib", "llvm", "bin")]
    )
    found = shutil.which("amdclang++", path=search_path)
    if not found:
        pytest.skip("no amdclang++ under ROCM_PATH")
    return found


def test_batched_probes_answer_like_one_at_a_time(assembler):
    # debug runs every probe on its own.
    assert _probe(assembler, _ISAS) == _probe(assembler, _ISAS, debug=True)


@pytest.mark.parametrize("isa", [(9, 4, 2), (12, 0, 1)], ids=["gfx942", "gfx1201"])
@_shell_assembler
def test_probing_an_isa_takes_a_few_assembler_runs(tmp_path, assembler, isa):
    log = tmp_path / "runs.log"
    script = _script(tmp_path / "bin" / "amdclang++", f'echo run >> "{log}"\nexec "{assembler}" "$@"\n')

    assert _probe(script, [isa]) == _probe(assembler, [isa], debug=True)
    # About a hundred probes, one at a time.
    assert _runs(log) < 10


@pytest.mark.parametrize("failure", ["unnamed-error", "killed-after-error"])
@_shell_assembler
def test_unreadable_batches_are_probed_one_source_at_a_time(tmp_path, assembler, failure):
    # Every probe is one line, so a run of several is a batch. Partial
    # diagnostics before a signal must not count as a complete rejection list.
    fail = {
        "unnamed-error": 'echo "error: out of luck" >&2; exit 1',
        "killed-after-error": 'echo "<stdin>:1:1: error: interrupted" >&2; kill -HUP $$',
    }[failure]
    script = _script(
        tmp_path / "amdclang++",
        'src=$(cat)\n'
        f'[ "$(printf "%s\\n" "$src" | wc -l)" -le 1 ] || {{ {fail}; }}\n'
        f'printf "%s\\n" "$src" | exec "{assembler}" "$@"\n',
    )

    assert _probe(script, _ISAS[1:2]) == _probe(assembler, _ISAS[1:2], debug=True)


@_shell_assembler
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


@pytest.mark.parametrize("isa", [(9, 4, 2), (11, 0, 0)], ids=["gfx942", "gfx1100"])
@_shell_assembler
def test_rejected_isa_stops_probing_after_the_empty_source(tmp_path, isa):
    # Rejecting every source is what an assembler that doesn't know the ISA does.
    # The empty source's command line rejects them all, so only probes with
    # other options still run: gfx10+ wave32 probes, which leave out
    # -mwavefrontsize64. debug runs every probe, for comparison.
    log = tmp_path / "runs.log"
    script = _script(tmp_path / "bin" / "amdclang++", f'echo run >> "{log}"\nexit 1\n')

    everyProbe = _probe(script, [isa], debug=True)
    allRuns = _runs(log)
    assert _probe(script, [isa]) == everyProbe
    runs = _runs(log) - allRuns
    if isa[0] < 10:
        assert runs == 1
    else:
        assert 1 < runs < 10 < allRuns
