# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The C++ tensor GETT planner (rocke/tensor_gett_planner.h) and GEMM dispatcher
(rocke/gemm_dispatch.h) decide exactly as the Python ones (CPU only; needs a C++ compiler and
CMake to build the engine archive, or ``ROCKE_CORE_ARCHIVE``).

* On random contraction problems per target (tensor ranks 2-6 per operand, random mode orders and
  padded strides, f16 / bf16, scale / bilinear, CU counts of full devices and partitions), on
  operands laid out as (or one stride away from) the K-contiguous copies, on problems at the
  planner's thresholds (tile count, K, D run), on the same problems at other CU counts, plus the problems of the shipped AOT and tuned route tables, every plan is
  the same: route kind, GETT
  candidate and kernel name (the whole spec), grid, block, workspace, the pack steps (kernel
  names, grids, workspace) and both routes' predicted times (bit-equal doubles); problems the
  Python planner refuses are refused.
* On random GEMM requests of the layouts the planner issues, and on requests placed on the rule
  thresholds of dispatch/gemm (tile counts and K values one below, at and one above each), the
  GEMM dispatcher's pick (the registered candidate and its spec) and its predicted time are the
  same.
* On feature rows at, around and between every split threshold (and on LightGBM's
  missing-as-zero values), every shipped selector / skinny model and bag scores the same
  (bit-equal doubles, the skinny models with NumPy's pairwise summation), and so do copies of
  two of them with every missing type / default direction (the shipped models use one).
* On specs near the dispatcher's picks (every one-lever step of each, and random 0-4 tile /
  trait field changes), gemm_universal's
  is_valid_spec and (for valid specs) launch_shape_error agree.
* The C++ data tables (TraitSpec / SkinnyGemmSpec fields, the Python-only lever list, the
  gemm_skinny tables, the rule thresholds, selector margins and planner cost model) are the
  Python ones (tools/gen_gemm_dispatch_data.py --check).
* Every chosen GETT spec without a Python-engine-only lever converts to the C++ kernel builders'
  spec with the same kernel name.
* On random contraction problems (and ones on the planner's thresholds and K-contiguous operand
  layouts), every kernel of the planned route (the tensor_pack steps and the GETT), as the C++
  kernel builders build it from the C++ planner's route (what hipTensor's contraction JIT
  compiles), lowers to the same LLVM IR as the Python engine's kernel of the Python planner's
  route (SHA-256 of the .ll at ROCKE_LLVM_FLAVOR, llvm22 when unset).
* Problems with a batch mode or a mode of one operand only are refused by both planners.
* The engine archive the comparisons link is current: planner_ref.ensure_archive always runs the
  incremental build of rocke_core (an archive of older sources is never compared).

``ROCKE_GETT_PLANNER_PROBLEMS`` / ``ROCKE_GETT_PLANNER_GEMMS`` / ``ROCKE_GETT_PLANNER_SPECS`` /
``ROCKE_GETT_PLANNER_IR_PROBLEMS`` set the sample sizes.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import planner_ref as PR  # noqa: E402

# the shipped selectors (tests/conftest.py pins them off for the rule-layer tests)
GEMM_SELECTOR_TESTS = True
GEMM_SKINNY_SELECT_TESTS = True

N_PROBLEMS = int(os.environ.get("ROCKE_GETT_PLANNER_PROBLEMS", "2000"))
N_GEMMS = int(os.environ.get("ROCKE_GETT_PLANNER_GEMMS", "2000"))
N_SPECS = int(os.environ.get("ROCKE_GETT_PLANNER_SPECS", "3000"))
N_IR = int(os.environ.get("ROCKE_GETT_PLANNER_IR_PROBLEMS", "480"))
JOBS = min(32, os.cpu_count() or 1)


@pytest.fixture(autouse=True)
def _python_defaults(monkeypatch):
    for k in PR.PY_ONLY_ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def cli(tmp_path_factory):
    if not (PR.PLATFORM / "cpp" / "include").is_dir() or not PR.CLI_SRC.is_file():
        pytest.skip("no C++ engine sources here (the installed test tree)")
    if shutil.which("cmake") is None or not any(
        shutil.which(c) for c in ("c++", "clang++", "g++")
    ):
        if not os.environ.get("ROCKE_CORE_ARCHIVE"):
            pytest.skip(
                "no C++ toolchain to build the engine archive (set ROCKE_CORE_ARCHIVE)"
            )
    return PR.build_cli(tmp_path_factory.mktemp("planner_cli"))


def _compare(cli, kind, problems, tmp_path):
    cpp = PR.cpp_lines(cli, kind, problems, tmp_path, JOBS)
    py = PR.python_lines(kind, problems, JOBS)
    assert len(cpp) == len(py) == len(problems)
    bad = [(p, c, q) for p, c, q in zip(problems, cpp, py) if PR.cpp_comparable(c) != q]
    assert not bad, f"{len(bad)} of {len(problems)} differ:\n" + "\n".join(
        f"{p}\n  C++    {c}\n  Python {q}" for p, c, q in bad[:5]
    )
    return cpp, py


def test_generated_data_is_current():
    tool = HERE.parents[3] / "tools" / "gen_gemm_dispatch_data.py"
    if not tool.is_file():
        pytest.skip("tools/gen_gemm_dispatch_data.py is not here (installed test tree)")
    r = subprocess.run(
        [sys.executable, str(tool), "--check"], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr


def test_gemm_dispatch_equals_python(cli, tmp_path):
    rng = random.Random(9501)
    edges = PR._edge_values()
    problems = [PR.gemm(rng, "gfx950") for _ in range(N_GEMMS)]
    problems += [PR.gemm_edge(rng, "gfx950", edges) for _ in range(N_GEMMS // 2)]
    # the same requests at other CU counts (the C++ dispatcher keeps its decisions per request)
    problems += [dict(p, cus=c) for p in problems[:100] for c in (64, 304)]
    cpp, py = _compare(cli, "gemm", problems, tmp_path)
    cands = {q.split(" ")[0] for q in py if q != "none"}
    # the sample reaches the learned picks, the skinny body, the beat / ring bodies and both dtypes
    assert any("selected" in c for c in cands) and any("skinny" in c for c in cands)
    assert any("dtl_ring" in c for c in cands) and any("bf16" in c for c in cands)
    assert any(" t=" in q and not q.endswith("t=-") for q in py)


def test_model_scores_equal_python(cli, tmp_path):
    files = PR.synthetic_models(tmp_path)
    problems = PR.tree_problems(random.Random(9506), N_SPECS, files=files)
    cpp, py = _compare(cli, "trees", problems, tmp_path)
    assert "-" not in py, "a shipped model is missing"


def test_spec_validity_equals_python(cli, tmp_path):
    rng = random.Random(9505)
    pool = PR.spec_pool()
    types = PR._spec_types()
    problems = PR.spec_neighbours(pool, rng, types)
    problems += [PR.spec_problem(rng, pool, types) for _ in range(N_SPECS)]
    cpp, py = _compare(cli, "valid", problems, tmp_path)
    # both verdicts occur, for the spec and for the launch shape
    assert {q.split(" ")[0] for q in py} == {"valid=0", "valid=1"}
    assert {q.split(" ")[1] for q in py if q.startswith("valid=1")} == {
        "launch=0",
        "launch=1",
    }


@pytest.mark.parametrize("arch", ["gfx950", "gfx942", "gfx90a"])
def test_planner_equals_python(cli, tmp_path, arch):
    rng = random.Random({"gfx950": 9502, "gfx942": 9503, "gfx90a": 9504}[arch])
    n = N_PROBLEMS if arch == "gfx950" else max(1, N_PROBLEMS // 4)
    problems = [PR.contraction(rng, arch) for _ in range(n)]
    if arch == "gfx950":
        problems += [PR.contraction_kc_own(rng, arch) for _ in range(n // 8)]
        problems += [PR.contraction_edge(rng, arch) for _ in range(n // 4)]
        problems += [dict(p, cus=c) for p in problems[:60] for c in (64, 304)]
        problems = PR.table_problems() + problems
    cpp, py = _compare(cli, "contraction", problems, tmp_path)
    routes = [q for q in py if q != "none"]
    if arch != "gfx950":
        # the tensor GETT family serves gfx950 only
        assert not routes
        return
    kinds = {q.split("|")[0] for q in routes}
    cands = {q.split("|")[1] for q in routes}
    timed = [q for q in routes if not q.endswith("|-|-")]
    # the sample exercises every route kind and both route time predictions
    assert kinds == {"single", "pack_hybrid"}, kinds
    assert {"tensor_gett_mn", "tensor_gett_kc"} <= cands, cands
    assert len(timed) > n // 20, len(timed)
    assert len(routes) < len(py), "the sample has no refused problem"
    # a mode of one operand only (or a batch mode): refused by both planners, not dropped
    lonely = [
        i
        for i, p in enumerate(problems)
        if (set(p["a"]) - set(p["b"]) - set(p["d"]))
        or (set(p["b"]) - set(p["a"]) - set(p["d"]))
    ]
    assert lonely, "the sample has no problem with a mode of one operand only"
    assert all(py[i] == "none" for i in lonely), [
        problems[i] for i in lonely if py[i] != "none"
    ][:3]
    # every chosen spec without a Python-engine-only lever is the C++ builders' spec
    assert not [c for c in cpp if " builder=DIFF" in c]


def test_planned_kernels_lower_like_python(cli, tmp_path, monkeypatch):
    if not os.environ.get("ROCKE_LLVM_FLAVOR"):
        monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm22")
    rng = random.Random(9508)
    problems = [PR.contraction(rng, "gfx950") for _ in range(N_IR)]
    problems += [PR.contraction_kc_own(rng, "gfx950") for _ in range(N_IR // 8)]
    problems += [PR.contraction_edge(rng, "gfx950") for _ in range(N_IR // 4)]
    cpp = PR.cpp_lines(cli, "ir", problems, tmp_path, JOBS)
    keep = [i for i, c in enumerate(cpp) if c != "pyonly"]
    py = PR.python_lines("ir", [problems[i] for i in keep], JOBS)
    assert not [c for c in cpp if "ERR:" in c], [c for c in cpp if "ERR:" in c][:3]
    bad = [(problems[i], cpp[i], q) for i, q in zip(keep, py) if cpp[i] != q]
    assert not bad, f"{len(bad)} of {len(keep)} differ:\n" + "\n".join(
        f"{p}\n  C++    {c}\n  Python {q}" for p, c, q in bad[:5]
    )
    routes = [c for c in cpp if c not in ("none", "pyonly")]
    # the sample builds both route kinds and K-contiguous / MN GETT bodies
    assert {c.split("|")[0] for c in routes} == {"single", "pack_hybrid"}, routes[:5]
    assert len(routes) >= len(problems) // 5, len(routes)


def test_engine_archive_is_rebuilt(tmp_path, monkeypatch):
    """ensure_archive runs the incremental engine build on every call, also when an archive
    exists (a no-op when no source changed), and configures a directory of this tree's own when
    the shared default directory belongs to another rocKE tree."""
    monkeypatch.delenv("ROCKE_CORE_ARCHIVE", raising=False)
    calls = []
    monkeypatch.setattr(
        PR.subprocess, "run", lambda cmd, **kw: calls.append(list(map(str, cmd)))
    )
    root = tmp_path / "rocke_verify"
    root.mkdir()
    (root / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={PR.PLATFORM}\n"
    )
    (root / "librocke_core.a").write_bytes(b"!<arch>\n")
    assert PR.ensure_archive(root) == root / "librocke_core.a"
    assert calls == [
        [
            "cmake",
            "--build",
            str(root),
            "--target",
            "rocke_core",
            "-j",
            str(os.cpu_count() or 1),
        ]
    ], calls
    # no cache yet: configure, then build
    calls.clear()
    fresh = tmp_path / "fresh"
    PR.ensure_archive(fresh)
    assert [c[:2] for c in calls] == [["cmake", "-S"], ["cmake", "--build"]], calls
    # the default directory configured from another tree: a directory of this tree
    calls.clear()
    monkeypatch.setattr(PR.tempfile, "gettempdir", lambda: str(tmp_path / "t"))
    other = tmp_path / "t" / "rocke_verify"
    other.mkdir(parents=True)
    (other / "CMakeCache.txt").write_text(
        "CMAKE_HOME_DIRECTORY:INTERNAL=/elsewhere/platform\n"
    )
    got = PR.ensure_archive()
    assert got.parent != other and got.parent.name.startswith("rocke_verify_"), got
    assert [c[:2] for c in calls] == [["cmake", "-S"], ["cmake", "--build"]], calls
