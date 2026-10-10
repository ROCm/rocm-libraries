# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The launch record of a bench: launch overrides, the launch heuristic of
the bench build and the workspace, and their summaries."""

import json
import os

import pytest

from lib import launch_settings as ls

SCHEMA = (
    "Arguments:\n"
    "  - user_allocated_workspace: c_size_t\n"
    "Defaults:\n"
    "  iters: 10\n"
    "  user_allocated_workspace: 134217728\n"
)


def write_sources(root):
    """A rocm-libraries tree under `root` with every launch-rule file but
    the optional DefaultPersistentLaunch.hpp, and an unrelated source."""
    tl = root / "projects" / "hipblaslt" / "tensilelite"
    origami = root / "shared" / "origami"
    for comp, rel in ls.LAUNCH_RULE_FILES:
        if rel.endswith("DefaultPersistentLaunch.hpp"):
            continue
        path = (tl if comp == "tensilelite" else origami) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"// {comp} {rel}\n")
    (tl / "src" / "Unrelated.cpp").write_text("// not a launch rule\n")
    return tl, origami


def write_build(build, tl, origami, *, compile_commands=True):
    """A build of the tree: its compiled libraries, the clients' schema and
    either compile_commands.json or CMakeCache.txt naming the sources."""
    (build / "tensilelite").mkdir(parents=True)
    (build / "tensilelite" / "libtensilelite-host.so.1").write_bytes(b"\x7fELF")
    (build / "origami").mkdir()
    (build / "origami" / "liborigami.a").write_bytes(b"!<arch>")
    (build / "clients").mkdir()
    (build / "clients" / "hipblaslt_common.yaml").write_text(SCHEMA)
    (build / "CMakeCache.txt").write_text(
        f"CMAKE_PROJECT_NAME:STATIC=hipblaslt\n"
        f"hipblaslt_SOURCE_DIR:STATIC={tl.parent}\n"
    )
    if compile_commands:
        entries = [
            {"file": str(tl / "src" / "ContractionSolution.cpp")},
            {"file": str(tl / "src" / "Unrelated.cpp")},
            {"file": str(origami / "src" / "origami" / "streamk.cpp")},
        ]
        (build / "compile_commands.json").write_text(json.dumps(entries))
    old = 1_000_000_000
    for p in build.rglob("*"):
        os.utime(p, (old, old))
    return build


@pytest.fixture
def tree(tmp_path):
    tl, origami = write_sources(tmp_path / "src")
    build = write_build(tmp_path / "build", tl, origami)
    for p in (tmp_path / "src").rglob("*"):
        os.utime(p, (900_000_000, 900_000_000))
    return tl, origami, build


def test_the_heuristic_hashes_the_launch_rule_sources_the_build_compiled(tree):
    tl, origami, build = tree
    h = ls.heuristic(build)
    assert h["found_by"] == "compile_commands.json"
    assert list(h["files"]) == [f"{c}/{r}" for c, r in ls.LAUNCH_RULE_FILES]
    absent = "tensilelite/include/Tensile/DefaultPersistentLaunch.hpp"
    assert h["files"][absent] is None
    assert all(v.startswith("sha256:") for k, v in h["files"].items() if k != absent)
    assert h["newer_than_build"] == [] and h["digest"].startswith("sha256:")
    assert ls.heuristic(build) == h

    (tl / "src" / "Unrelated.cpp").write_text("// edited\n")
    os.utime(tl / "src" / "Unrelated.cpp", (900_000_000, 900_000_000))
    assert ls.heuristic(build)["digest"] == h["digest"]
    streamk = origami / "src" / "origami" / "streamk.cpp"
    streamk.write_text("// another grid rule\n")
    changed = ls.heuristic(build)
    assert changed["digest"] != h["digest"]
    assert changed["newer_than_build"] == ["origami/src/origami/streamk.cpp"]
    (tl / "include" / "Tensile" / "DefaultPersistentLaunch.hpp").write_text("//\n")
    assert ls.heuristic(build)["files"][absent] is not None


def test_the_cmake_cache_names_the_sources_without_compile_commands(tree):
    _, _, build = tree
    h = ls.heuristic(build)
    (build / "compile_commands.json").unlink()
    cached = ls.heuristic(build)
    assert cached["found_by"] == "CMakeCache.txt"
    assert cached["digest"] == h["digest"] and cached["files"] == h["files"]
    (build / "CMakeCache.txt").write_text("CMAKE_PROJECT_NAME:STATIC=hipblaslt\n")
    assert ls.heuristic(build) is None
    assert ls.heuristic(None) is None
    assert ls.heuristic(build / "missing") is None


def test_the_build_of_a_bench(tree, tmp_path):
    _, _, build = tree
    bench = build / "clients" / "hipblaslt-bench"
    bench.write_text("#!/bin/sh\n")
    assert ls.build_dir_of(bench) == build
    link = tmp_path / "bench-link"
    link.symlink_to(bench)
    assert ls.build_dir_of(link) == build
    wrapper = tmp_path / "ctx" / "hipblaslt-bench"
    wrapper.parent.mkdir()
    wrapper.write_text("#!/bin/sh\n")
    assert ls.build_dir_of(wrapper) is None
    cfg = {"runtime": {"rocm_libraries_root": str(tmp_path / "root")}}
    derived = tmp_path / "root" / "projects" / "hipblaslt" / "build" / "release"
    assert ls.build_dir_of(wrapper, cfg) == derived
    cfg["runtime"]["build_dir"] = str(build)
    assert ls.build_dir_of(wrapper, cfg) == build


def test_the_workspace_of_the_lines(tree):
    _, _, build = tree
    bench = build / "clients" / "hipblaslt-bench"
    default = ls.schema_workspace_bytes(bench, None)
    assert default == 134217728 == ls.schema_workspace_bytes(None, build)
    assert ls.schema_workspace_bytes(None, None) is None
    plain = "- {function: matmul, M: 8, N: 8, K: 8}\n"
    own = "- {function: matmul, M: 8, user_allocated_workspace: 4194304}\n"
    assert ls.workspace_bytes([plain, plain], default) == default
    assert ls.workspace_bytes([], default) == default
    assert ls.workspace_bytes([own], None) == 4194304
    assert ls.workspace_bytes([plain, own], default) == [4194304, default]
    assert ls.workspace_bytes([plain], None) is None


def test_the_bench_record(tree):
    _, _, build = tree
    bench = build / "clients" / "hipblaslt-bench"
    env = {"TENSILE_PRECISE_SM_TARGET": "1", "TENSILE_DB": "2"}
    environ = {"TENSILE_FIXED_WGM": "4", "TENSILE_PRECISE_SM_TARGET": "1", "X": "y"}
    rec = ls.bench_record(bench, None, env, environ, ["- {M: 8}\n"])
    assert rec["overrides"] == {"TENSILE_PRECISE_SM_TARGET": "1"}
    assert rec["stripped"] == ["TENSILE_FIXED_WGM"]
    assert rec["heuristic"] == ls.heuristic(build)
    assert rec["workspace_bytes"] == 134217728
    assert json.loads(json.dumps(rec)) == rec


def test_records_next_to_logs_and_their_summaries(tmp_path):
    a = {"overrides": {}, "stripped": [], "heuristic": {"digest": "sha256:aa"}}
    a["workspace_bytes"] = 134217728
    b = {"overrides": {"TENSILE_FIXED_WGM": "4"}, "heuristic": None}
    b["workspace_bytes"] = [4194304, 134217728]
    stage03 = tmp_path / "stage03"
    (stage03 / "logs").mkdir(parents=True)
    (stage03 / "platform.json").write_text(json.dumps({"gpus": [], "launch": a}))
    assert ls.read_record(stage03 / "logs") == a
    assert ls.read_record(tmp_path / "nowhere" / "logs") is None
    (tmp_path / "old" / "logs").mkdir(parents=True)
    (tmp_path / "old" / "platform.json").write_text(json.dumps({"gpus": []}))
    assert ls.read_record(tmp_path / "old" / "logs") is None

    one = ls.summarize([a, b, None])
    assert one == {
        "heuristics": ["sha256:aa"],
        "overrides": {"TENSILE_FIXED_WGM": ["4"]},
        "workspace_bytes": [4194304, 134217728],
        "unknown": 2,
    }
    s03b = tmp_path / "stage03b"
    (s03b / "logs").mkdir(parents=True)
    (s03b / "rebench_summary.json").write_text(json.dumps({"launch": one}))
    assert ls.read_record(s03b / "logs") == one
    c = {"overrides": {"TENSILE_FIXED_WGM": "8"}, "heuristic": {"digest": "sha256:bb"}}
    two = ls.summarize([one, c])
    assert two["heuristics"] == ["sha256:aa", "sha256:bb"] and two["unknown"] == 2
    assert two["overrides"] == {"TENSILE_FIXED_WGM": ["4", "8"]}
    assert ls.summarize([two]) == two
    assert ls.digests(c) == ["sha256:bb"] and ls.digests(None) == []
    # Sources edited after the build: the digest describes the edit, not the
    # build, and compares as unknown.
    stale = {"overrides": {}, "heuristic": {"digest": "sha256:cc"}}
    stale["heuristic"]["newer_than_build"] = ["origami/src/origami/streamk.cpp"]
    assert ls.digests(stale) == [] and ls.comparable_digest(stale["heuristic"]) is None
    three = ls.summarize([a, stale])
    assert three["heuristics"] == ["sha256:aa"] and three["unknown"] == 1
    assert "; 1 bench(es) with an unknown heuristic;" in ls.describe(three)
    assert ls.comparable_digest({"digest": "sha256:aa", "newer_than_build": []}) == (
        "sha256:aa"
    )
    assert ls.short("sha256:" + "f" * 64) == "f" * 16
    assert "launch heuristic aa, bb" in ls.describe(two)
    assert "TENSILE_FIXED_WGM=4|8" in ls.describe(two)
    assert ls.describe(ls.summarize([])).startswith("launch heuristic unknown")
