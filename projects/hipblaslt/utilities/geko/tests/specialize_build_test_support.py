# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""--build-library helpers: real CLI and TensileCreateLibrary (CPU), decoded msgpack
libraries and ELF symbols. Baseline .dat placeholders only identify the input build."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

import msgpack
import pytest
import yaml

from tests.specialize_test_support import (
    RECIPES, build_fixture, command, input_fixture, invoke, outputs, row, tree_hash,
)


def files(root):
    root = Path(root)
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*")) if path.is_file()}


def tree(root):
    root = Path(root)
    return {str(path.relative_to(root)): ("link", os.readlink(path)) if path.is_symlink()
            else ("directory",) if path.is_dir() else ("file", path.read_bytes())
            for path in sorted(root.rglob("*"))}


def subset_fixture(root, arch="gfx950"):
    baseline = build_fixture(root / "baseline")
    source = baseline / "source-logic"
    recipes = ("dp", "dp_other") if arch == "gfx950" else ("g942",)
    keep = {RECIPES[name]["logic_file"] for name in recipes}
    for path in source.rglob("*"):
        if path.is_file() and str(path.relative_to(source)) not in keep:
            path.unlink()
    table_path = baseline / "device-library/MatchTable.yaml"
    table = yaml.safe_load(table_path.read_text())
    table = {index: entry for index, entry in table.items() if Path(entry[0]).is_file()}
    table_path.write_text(yaml.safe_dump(table))
    records = [row(activation_type="none", call_count=2), row(activation_type="relu", call_count=3)]
    if arch == "gfx942":
        records = [row("g942", call_count=5)]
    logs, facts = input_fixture(root, records)
    metadata_path = logs / "run.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(library=str(baseline / "Tensile/library" / arch),
                    library_fp=tree_hash(baseline / "Tensile/library" / arch))
    metadata_path.write_text(json.dumps(metadata))
    return baseline, logs.parent, facts.parent, records


def build_case(root, monkeypatch, *, arch="gfx950", stock_only=False):
    baseline, logs, facts, records = subset_fixture(root, arch)
    before = files(baseline / "source-logic")
    destination = root / "built"
    work = root / "work"
    # Stale user logic must not leak into this run's merged input.
    stale = work / "logic" / arch / "stale-user.yaml"
    stale.parent.mkdir(parents=True)
    stale.write_text("not: a library logic file\n")
    extra = ["--keep-beta", "--jobs", "1", "--build-library", str(destination)]
    if stock_only:
        extra.append("--stock-only")
    start = time.perf_counter()
    code = invoke(command(root, logs, facts, work, arch, extra), monkeypatch)
    elapsed = time.perf_counter() - start
    result = outputs(work, code)
    result.update(baseline=baseline, before=before, destination=destination,
                  records=records, elapsed=elapsed)
    return result


def build(root, logs, facts, work, destination, monkeypatch, *, stock_only=False, jobs=1):
    extra = ["--keep-beta", "--jobs", str(jobs), "--build-library", str(destination)]
    if stock_only:
        extra.append("--stock-only")
    try:
        code = invoke(command(root, logs, facts, work, extra=extra), monkeypatch)
    except OSError as error:
        # Report the public success/clean-refusal contract, not a raw exception.
        return {"code": None, "exception": str(error), "work": work}
    return {**outputs(work, code), "exception": None}


def seed_work(root, logs, facts, work, monkeypatch):
    code = invoke(command(root, logs, facts, work, extra=["--keep-beta"]), monkeypatch)
    result = outputs(work, code)
    assert code == 0 and len(result["manifest"]["calls"]) == 2
    assert len({call["kernel_name"] for call in result["manifest"]["calls"]}) == 2


def assert_unchanged_refusal(result, work, destination, before_work, before_destination):
    assert result["exception"] is None, "unrecorded output content must be refused without an exception"
    assert result["code"] == 2, "unrecorded paths in OUT/logic or OUT/library must cause usage refusal"
    assert tree(work) == before_work, "refusal invalidated existing specialization outputs"
    assert tree(destination) == before_destination, "refusal changed owned or unowned build content"


def library_solutions(directory):
    solutions = {}
    for path in sorted(Path(directory).glob("TensileLibrary*.dat*")):
        data = path.read_bytes()
        if path.suffix == ".zlib":
            data = zlib.decompress(data)
        document = msgpack.unpackb(data, raw=False, strict_map_key=False)
        for solution in document.get("solutions", []):
            index = int(solution["index"])
            assert index not in solutions, "one built index must identify exactly one solution"
            solutions[index] = solution
    return solutions


def match_table(path):
    return yaml.safe_load(Path(path).read_text()) if Path(path).is_file() else {}


def symbols(directory):
    # llvm-nm is a real tool boundary; no GPU is touched by inspecting code objects.
    binary = shutil.which("llvm-nm") or str(Path(shutil.which("amdclang++")).resolve().parent / "llvm-nm")
    names = set()
    objects = sorted(path for path in Path(directory).rglob("*") if path.suffix in (".co", ".hsaco"))
    assert objects, "built library must contain actual code objects"
    for path in objects:
        result = subprocess.run([binary, "--defined-only", "--format=posix", str(path)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        names.update(line.split()[0] for line in result.stdout.splitlines() if line.split())
    return names


def run_tcl(logic, out, arch="gfx950", jobs=1):
    out.mkdir(parents=True)
    argv = [sys.executable, "-m", "tensilelite.TensileCreateLibrary", f"--architecture={arch}",
            "--cxx-compiler=/opt/rocm/bin/amdclang++", f"--jobs={jobs}", "--no-compress",
            "--disable-asm-comments", "--gfx1250v0", str(logic.resolve()), str(out.resolve()), "HIP"]
    with (out / "control-tcl.log").open("w") as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, cwd=out)
    assert result.returncode == 0, f"positive-control TCL failed; see {out / 'control-tcl.log'}"
    return out / "library" / arch


@pytest.fixture(scope="module")
def merged_case(tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        return build_case(tmp_path_factory.mktemp("merged"), monkeypatch)


@pytest.fixture(scope="module")
def gfx942_case(tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        return build_case(tmp_path_factory.mktemp("gfx942"), monkeypatch, arch="gfx942")


@pytest.fixture(scope="module")
def stock_case(tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        return build_case(tmp_path_factory.mktemp("stock"), monkeypatch, stock_only=True)
