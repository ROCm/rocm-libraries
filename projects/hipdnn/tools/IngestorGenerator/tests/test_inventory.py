# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Baseline inventory: one command for extend.md's known-good-installation record.

The counts must come from the KDP key the loader reads (`kernelDescriptors`): a guessed
key returning zero would be a silent wrong inventory. The catalog digest must be the
RUNBOOK stage 8 recipe's, or a before/after comparison means nothing. Validator output
must fold one-message-per-kernel diagnostics so a different warning stays visible.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "inventory.py"
_ENGINE = "test:Inventory"


def _kernel(name: str, head_size: int, hq: int, kind: str = "kpack") -> dict:
    return {
        "name": name,
        "kernel_source": {"kind": kind},
        "metadata": {"head_size": head_size, "num_query_heads": hq},
    }


def _tree(root: Path) -> Path:
    """An install-like root holding one engine bundle with two KDPs, plus an unrelated
    bundle the inventory must not count."""
    bundle = root / "rocKE" / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "b.ued.json").write_text(json.dumps({"name": _ENGINE}))
    (bundle / "a.kdp.json").write_text(
        json.dumps(
            {
                "kernelDescriptors": [
                    _kernel("k0", 64, 8),
                    _kernel("k1", 64, 8),
                    _kernel("k2", 128, 16),
                ]
            }
        )
    )
    (bundle / "sub").mkdir()
    (bundle / "sub" / "b.kdp.json").write_text(
        json.dumps({"kernelDescriptors": [_kernel("k3", 128, 16, "hsaco")]})
    )
    other = root / "rocKE" / "other"
    other.mkdir()
    (other / "o.ued.json").write_text(json.dumps({"name": "test:Other"}))
    (other / "o.kdp.json").write_text(
        json.dumps({"kernelDescriptors": [_kernel("x", 1, 1)]})
    )
    return bundle


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_TOOL), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_counts_kernels_kinds_and_groups_of_the_named_engine(tmp_path):
    _tree(tmp_path)
    out = tmp_path / "report.json"

    result = _run(
        str(tmp_path),
        "--engine",
        _ENGINE,
        "--group-by",
        "head_size,num_query_heads",
        "--json",
        str(out),
    )

    assert result.returncode == 0, result.stderr
    inv = json.loads(out.read_text())["inventory"]
    assert inv["kernels"] == 4
    assert inv["source_kinds"] == {"kpack": 3, "hsaco": 1}
    assert inv["groups"] == [
        {"values": [64, 8], "kernels": 2},
        {"values": [128, 16], "kernels": 2},
    ]


def test_a_kdp_without_kernel_descriptors_fails_instead_of_counting_zero(tmp_path):
    bundle = _tree(tmp_path)
    (bundle / "a.kdp.json").write_text(json.dumps({"kernels": [_kernel("k0", 64, 8)]}))

    result = _run(str(tmp_path), "--engine", _ENGINE)

    assert result.returncode == 1
    assert "kernelDescriptors" in result.stderr


@pytest.mark.skipif(
    os.name == "nt"
    or shutil.which("bash") is None
    or shutil.which("sha256sum") is None,
    reason="the RUNBOOK recipe needs bash with GNU find/sort/sha256sum",
)
def test_catalog_digest_matches_the_runbook_recipe(tmp_path):
    bundle = _tree(tmp_path)
    recipe = (
        'set -euo pipefail; cd "$1"; '
        "lines=$(find . -name '*.json' -type f -print0 | LC_ALL=C sort -z "
        "| xargs -0r sha256sum); printf '%s\\n' \"$lines\" | sha256sum"
    )
    expected = subprocess.run(
        ["bash", "-c", recipe, "recipe", bundle.as_posix()],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()[0]

    result = _run(str(tmp_path), "--engine", _ENGINE)

    assert result.returncode == 0, result.stderr
    assert f"catalog digest: {expected}" in result.stdout


def _digest_line(result: subprocess.CompletedProcess) -> str:
    assert result.returncode == 0, result.stderr
    return next(
        line for line in result.stdout.splitlines() if line.startswith("catalog digest")
    )


def test_a_symlinked_json_is_left_out_of_the_digest_as_find_type_f_does(tmp_path):
    bundle = _tree(tmp_path)
    before = _digest_line(_run(str(tmp_path), "--engine", _ENGINE))
    try:
        (bundle / "linked.json").symlink_to(bundle / "a.kdp.json")
    except OSError as e:
        pytest.skip(f"cannot create a symlink here: {e}")

    after = _digest_line(_run(str(tmp_path), "--engine", _ENGINE))

    assert after == before


def test_a_validator_that_cannot_be_launched_is_an_inventory_failure(tmp_path):
    _tree(tmp_path)

    result = _run(
        str(tmp_path), "--engine", _ENGINE, "--validator", str(tmp_path / "missing")
    )

    assert result.returncode == 1
    assert result.stderr.startswith("FAIL: cannot run validator")
    assert "Traceback" not in result.stderr


def _fake_validator(tmp_path: Path, report: dict, exit_code: int) -> Path:
    script = tmp_path / "fake_validator.py"
    script.write_text(
        f"import json, sys\nprint(json.dumps({report!r}))\nsys.exit({exit_code})\n"
    )
    if os.name == "nt":
        wrapper = tmp_path / "fake_validator.cmd"
        wrapper.write_text(f'@"{sys.executable}" "{script}" %*\n')
    else:
        wrapper = tmp_path / "fake_validator"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        wrapper.chmod(0o755)
    return wrapper


def _diag(severity: str, message: str) -> dict:
    return {"severity": severity, "message": message}


def test_validator_diagnostics_are_folded_per_message(tmp_path):
    _tree(tmp_path)
    per_kernel = (
        "descriptor loader: extension key 'x-tag' in a 'kernelDescriptors' entry in "
    )
    report = {
        "success": True,
        "engines": [_ENGINE],
        "expected_engines_missing": [],
        "diagnostics": [
            _diag("WARN", per_kernel + f"/i/k{i}.kdp.json; ignoring it")
            for i in range(3)
        ]
        + [_diag("WARN", "a real warning"), _diag("INFO", "loaded")],
    }

    result = _run(
        str(tmp_path),
        "--engine",
        _ENGINE,
        "--validator",
        str(_fake_validator(tmp_path, report, 0)),
    )

    assert result.returncode == 0, result.stderr
    assert "validator: success" in result.stdout
    assert f"[WARN] x3 {per_kernel}<path>; ignoring it" in result.stdout
    assert "[WARN] x1 a real warning" in result.stdout


def test_a_failing_validator_fails_the_inventory(tmp_path):
    _tree(tmp_path)
    report = {
        "success": False,
        "engines": [],
        "expected_engines_missing": [_ENGINE],
        "diagnostics": [_diag("ERROR", "duplicate catalog tuple")],
    }

    result = _run(
        str(tmp_path),
        "--engine",
        _ENGINE,
        "--validator",
        str(_fake_validator(tmp_path, report, 1)),
    )

    assert result.returncode == 1
    assert "validator: FAILED" in result.stdout
    assert "expected engines missing" in result.stdout
