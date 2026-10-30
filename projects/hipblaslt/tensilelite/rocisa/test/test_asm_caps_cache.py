# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import os
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="counts assembler runs with a shell-script assembler"
)

_ISAS = [(9, 4, 2), (9, 5, 0)]

# rocIsa keeps capabilities for the life of the process, so each probe runs in a
# fresh interpreter that imports the same rocisa as this one.
_PROBE = """
import json, sys
sys.path[:] = json.loads(sys.argv[1])
import rocisa
isas = [tuple(isa) for isa in json.loads(sys.argv[2])]
ti = rocisa.rocIsa.getInstance()
if sys.argv[4] == "many":
    ti.initMany(isas, sys.argv[3], False)
else:
    for isa in isas:
        ti.init(isa, sys.argv[3], False)
print(json.dumps({str(isa): dict(ti.getIsaInfo(isa).asmCaps) for isa in isas}, sort_keys=True))
"""


def _realAssembler():
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join(
        [os.path.join(rocm_path, "bin"), os.path.join(rocm_path, "lib", "llvm", "bin")]
    )
    return shutil.which("amdclang++", path=search_path)


@pytest.fixture
def assembler(tmp_path):
    """The real assembler behind a script that logs each run to runs.log."""
    real = _realAssembler()
    if not real:
        pytest.skip("no amdclang++ under ROCM_PATH")
    script = tmp_path / "bin" / "amdclang++"
    script.parent.mkdir()
    script.write_text(f'#!/bin/sh\necho run >> "{tmp_path / "runs.log"}"\nexec "{real}" "$@"\n')
    script.chmod(0o755)
    return script


def _runs(tmp_path):
    log = tmp_path / "runs.log"
    return len(log.read_text().splitlines()) if log.exists() else 0


def _probe(assembler, cacheDir, isas=_ISAS, mode="many", disable=False):
    env = {k: v for k, v in os.environ.items() if k != "TENSILE_DISABLE_ASM_CAPS_CACHE"}
    env["TENSILE_ASM_CAPS_CACHE_DIR"] = str(cacheDir)
    if disable:
        env["TENSILE_DISABLE_ASM_CAPS_CACHE"] = "1"
    parentPath = json.dumps([os.path.abspath(p) for p in sys.path])
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, parentPath, json.dumps(isas), str(assembler), mode],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("mode", ["many", "init"])
def test_second_run_reads_the_cache(tmp_path, assembler, mode):
    cache = tmp_path / "cache"
    probed = _probe(assembler, cache, mode=mode)
    runs = _runs(tmp_path)
    assert any(caps["SupportedISA"] == 1 for caps in probed.values())
    assert len(list(cache.glob("*.txt"))) == len(_ISAS)

    assert _probe(assembler, cache, mode=mode) == probed
    assert _runs(tmp_path) == runs


def test_changed_assembler_is_probed_again(tmp_path, assembler):
    cache = tmp_path / "cache"
    first = _probe(assembler, cache, isas=_ISAS[:1])
    runs = _runs(tmp_path)
    # A new size and modification time: a different assembler, as far as the cache can tell.
    with open(assembler, "a") as f:
        f.write("# updated\n")

    assert _probe(assembler, cache, isas=_ISAS[:1]) == first
    assert _runs(tmp_path) > runs


def test_disabled_cache_probes_and_writes_nothing(tmp_path, assembler):
    cache = tmp_path / "cache"
    first = _probe(assembler, cache, isas=_ISAS[:1], disable=True)
    runs = _runs(tmp_path)

    assert _probe(assembler, cache, isas=_ISAS[:1], disable=True) == first
    assert _runs(tmp_path) == 2 * runs
    assert not cache.exists()


def test_truncated_entry_is_probed_again_and_rewritten(tmp_path, assembler):
    cache = tmp_path / "cache"
    first = _probe(assembler, cache, isas=_ISAS[:1])
    (entry,) = cache.glob("*.txt")
    text = entry.read_text()
    entry.write_text(text[: len(text) // 2])
    runs = _runs(tmp_path)

    assert _probe(assembler, cache, isas=_ISAS[:1]) == first
    assert _runs(tmp_path) > runs
    runs = _runs(tmp_path)
    assert _probe(assembler, cache, isas=_ISAS[:1]) == first
    assert _runs(tmp_path) == runs


def test_results_from_a_killed_assembler_are_not_cached(tmp_path):
    # Killed runs read as rejected instructions, which is not what the assembler would say.
    script = tmp_path / "amdclang++"
    script.write_text("#!/bin/sh\nkill -9 $$\n")
    script.chmod(0o755)
    cache = tmp_path / "cache"

    caps = _probe(script, cache, isas=_ISAS[:1])
    assert all(c["SupportedISA"] == 0 for c in caps.values())
    assert not list(cache.glob("*.txt"))
