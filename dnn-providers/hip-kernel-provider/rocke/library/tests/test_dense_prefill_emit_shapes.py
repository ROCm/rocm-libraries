# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only checks for the dense prefill benchmarks' ``--emit-shapes``.

The emit path must run without torch: IngestorGenerator's ``mine_shapes.py`` reads
its output on hosts with no GPU stack. Each benchmark runs in a subprocess with
``sys.modules['torch'] = None``, so a module-scope ``import torch`` fails here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_LIBROOT = Path(__file__).resolve().parents[1]
_PYROOT = _LIBROOT.parent / "platform" / "python"

#: Keys every emitted record carries: the trace schema ``mine_shapes --rocke-bench``
#: reads, plus the explicit ``causal``.
_KEYS = {
    "model",
    "variant",
    "label",
    "num_seqs",
    "max_seqlen_q",
    "max_seqlen_k",
    "num_query_heads",
    "num_kv_heads",
    "head_size",
    "q_dtype",
    "causal",
    "window_size",
    "has_sinks",
}

#: Run a benchmark's main() with torch unimportable, then print, as JSON, the
#: number of configs its _configs() yields over the modes it emits.
_DRIVER = """
import json, runpy, sys
sys.modules["torch"] = None
bench, modes, argv = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3:]
module = runpy.run_path(bench, run_name="emit_shapes_test")
sys.argv = ["bench", *argv]
code = module["main"]()
if code:
    sys.exit(code)
print(json.dumps(sum(len(module["_configs"](m, 128, 8, 128)) for m in modes)))
"""


def _bench(arch: str) -> Path:
    return (
        _LIBROOT
        / "benchmarks"
        / arch
        / "attention"
        / "prefill"
        / "benchmark_dense_prefill_live.py"
    )


def _run(arch: str, modes: list[str], *argv: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_LIBROOT), str(_PYROOT), env.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-c", _DRIVER, str(_bench(arch)), json.dumps(modes), *argv],
        capture_output=True,
        text=True,
        env=env,
    )


def _emit(tmp_path: Path, arch: str, modes: list[str]) -> tuple[list[dict], int]:
    out = tmp_path / f"{arch}.jsonl"
    result = _run(arch, modes, "--emit-shapes", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in out.read_text().splitlines()]
    expected = json.loads(result.stdout.strip().splitlines()[-1])
    return records, expected


def test_gfx942_emits_every_measured_mode_without_torch(tmp_path):
    modes = ["causal", "mha", "gqa", "full", "swa", "persistent"]
    records, expected = _emit(tmp_path, "gfx942", modes)

    assert len(records) == expected > 0
    assert all(set(r) == _KEYS for r in records)
    assert {r["variant"].split("/")[0] for r in records} == set(modes)
    assert all(type(r["causal"]) is bool for r in records)


def test_gfx950_emits_mode_all_without_torch(tmp_path):
    records, expected = _emit(tmp_path, "gfx950", ["all"])

    assert len(records) == expected > 0
    for r in records:
        extra = set(r) - _KEYS
        assert _KEYS <= set(r) and extra <= {"varlen", "seqlens"}, r
        assert bool(extra) == bool(r.get("varlen")), r


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_an_unknown_argument_is_refused_with_emit_shapes(tmp_path, arch):
    """A typo'd filter must not quietly emit the default shape set."""
    out = tmp_path / "shapes.jsonl"

    result = _run(arch, [], "--emit-shapes", str(out), "--heads", "64")

    assert result.returncode == 2, result.stdout + result.stderr
    assert "unrecognized arguments: --heads 64" in result.stderr
    assert not out.exists()
