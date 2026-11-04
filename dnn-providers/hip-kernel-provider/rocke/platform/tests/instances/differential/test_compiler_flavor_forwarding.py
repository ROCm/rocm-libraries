# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Sibling native AUTO callers share owner evidence, not compiler discovery."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rocke.runtime import comgr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ir_artifact_diff  # noqa: E402
import fuzz_diff  # noqa: E402


@pytest.fixture(params=["artifact", "fuzz"])
def driver(request, monkeypatch, tmp_path):
    cli = tmp_path / "ir_lower_cli"
    cli.touch()
    if request.param == "artifact":
        module = ir_artifact_diff
        monkeypatch.setattr(
            module.run_diff, "find_families", lambda: [("fixture", tmp_path)]
        )

        def run_pass(cli, families, arch):
            rc, output, error = module.run_cli(cli, "gfx950", b"fixture IR")
            return 0 if rc == 0 and output == b"LLVM IR" else 1

        monkeypatch.setattr(module, "_run_pass", run_pass)
    else:
        module = fuzz_diff
        monkeypatch.setattr(
            module,
            "_ck_imports",
            lambda: (lambda kernel, **kwargs: "LLVM IR", lambda kernel: "fixture IR"),
        )

        def family(cli, name, count, seed, lower, serialize, verbose):
            verdict, note = module.diff_one(
                cli, lambda spec: object(), None, "gfx950", lower, serialize
            )
            failed = verdict != "IDENTICAL"
            return (
                {
                    "generated": 1,
                    "invalid": 0,
                    "identical": int(not failed),
                    "mismatch": 0,
                    "error": int(failed),
                },
                [],
                [("fixture", note)] if failed else [],
            )

        monkeypatch.setattr(module, "fuzz_family", family)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            request.param,
            "--cli",
            str(cli),
            "--only",
            "elementwise" if request.param == "fuzz" else "fixture",
        ],
    )
    return module


@pytest.mark.parametrize(
    "mode", ["auto", "offline", "unavailable", "invalid", "native-failure"]
)
def test_main_forwards_profile_and_restores_environment(driver, monkeypatch, mode):
    monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)
    info = comgr.CompilerInfo((24, 0, 0), "fixture", None, "selected", "selected")
    query = Mock(return_value=None if mode == "unavailable" else info)
    monkeypatch.setattr(comgr, "_loaded_compiler_info", query)
    original = (
        "llvm20" if mode == "offline" else "invalid" if mode == "invalid" else None
    )
    if original:
        monkeypatch.setenv("ROCKE_LLVM_FLAVOR", original)
    seen = []

    def run(command, **kwargs):
        seen.append(os.environ.get("ROCKE_LLVM_FLAVOR"))
        failed = mode == "native-failure"
        return SimpleNamespace(
            returncode=1 if failed else 0,
            stdout=b"" if failed else b"LLVM IR",
            stderr=b"native failure" if failed else b"",
        )

    monkeypatch.setattr(driver.subprocess, "run", run)
    if mode == "unavailable":
        with pytest.raises(comgr.ComgrError, match="Cannot determine LLVM flavor"):
            driver.main()
    elif mode == "invalid":
        with pytest.raises((ValueError, SystemExit), match="[Ff]lavor"):
            driver.main()
    else:
        assert driver.main() == (1 if mode == "native-failure" else 0)
        assert seen == [original or "llvm24"]
    assert os.environ.get("ROCKE_LLVM_FLAVOR") == original
    if mode in ("offline", "invalid"):
        query.assert_not_called()
    else:
        query.assert_called_once_with(required=True)
    if mode in ("unavailable", "invalid"):
        assert not seen


def test_artifact_main_preserves_discovered_family_names_and_sources(
    monkeypatch, tmp_path
):
    cli = tmp_path / "ir_lower_cli"
    cli.touch()
    source = tmp_path / "library emitters"
    source.mkdir()
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm20")
    monkeypatch.setattr(
        sys, "argv", ["artifact", "--cli", str(cli), "--only", "elementwise"]
    )
    monkeypatch.setattr(
        ir_artifact_diff.run_diff,
        "find_families",
        lambda: [("library_elementwise", source), ("other", tmp_path)],
    )
    family = Mock(
        return_value=[{"idx": 0, "verdict": "IDENTICAL", "arch": "gfx950", "note": ""}]
    )
    monkeypatch.setattr(ir_artifact_diff, "run_family", family)
    assert ir_artifact_diff.main() == 0
    family.assert_called_once_with(cli, "library_elementwise", source)


def test_artifact_python_emitter_uses_its_discovered_source(monkeypatch, tmp_path):
    source = tmp_path / "library emitters"
    source.mkdir()
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=b"IR", stderr=b""))
    monkeypatch.setattr(ir_artifact_diff.subprocess, "run", run)
    assert ir_artifact_diff.run_py("fixture", 0, "ir", source) == (0, b"IR", "")
    command = run.call_args.args[0]
    assert str(source / "fixture_emit.py") in command
    paths = run.call_args.kwargs["env"]["PYTHONPATH"].split(os.pathsep)
    assert str(source) in paths
    assert str(ir_artifact_diff.run_diff.LIB_ROOT) in paths


def test_main_restores_profile_after_a_subprocess_exception(driver, monkeypatch):
    monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)
    info = comgr.CompilerInfo((24, 0, 0), "fixture", None, "selected", "selected")
    monkeypatch.setattr(comgr, "_loaded_compiler_info", Mock(return_value=info))
    monkeypatch.setattr(
        driver.subprocess, "run", Mock(side_effect=OSError("subprocess unavailable"))
    )
    with pytest.raises(OSError, match="subprocess unavailable"):
        driver.main()
    assert "ROCKE_LLVM_FLAVOR" not in os.environ
