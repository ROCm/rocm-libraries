# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the miopen-deep-review partition.py helper (areas, plan, packets, gate,
orient-gate)."""

import importlib.util
import os
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "partition.py"


@pytest.fixture
def part() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "miopen_deep_review_partition", SCRIPT
    )
    assert spec and spec.loader, f"could not load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(part, cmd, rd, *extra):
    part.main([cmd, "--round-dir", str(rd), *extra])


def make_round(tmp_path, numstat, files=None):
    """numstat: list of (added, deleted, path); files.txt defaults to the same paths."""
    (tmp_path / "numstat.txt").write_text(
        "".join(f"{a}\t{d}\t{p}\n" for a, d, p in numstat)
    )
    paths = files if files is not None else [p for _, _, p in numstat]
    (tmp_path / "files.txt").write_text("".join(f"{p}\n" for p in paths))
    return tmp_path


def read_tsv(path):
    return [tuple(ln.split("\t")) for ln in path.read_text().splitlines()]


@pytest.mark.parametrize(
    "path, slug",
    [
        ("projects/miopen/include/miopen/miopen.h", "public-api"),
        ("projects/miopen/src/private/wrapper.cpp", "wrapper-split"),
        ("projects/miopen/include_wrapper/miopen/hipdnn_forwarding.h", "wrapper-split"),
        ("projects/miopen/script/check_public_abi.py", "wrapper-split"),
        ("projects/miopen/src/solver/conv/conv_asm_1x1u.cpp", "solver"),
        ("projects/miopen/src/include/miopen/solver/implicitgemm_util.hpp", "solver"),
        ("projects/miopen/src/solver.cpp", "solver"),
        ("projects/miopen/src/conv/problem_description.cpp", "conv"),
        ("projects/miopen/src/include/miopen/conv/invokers/gcn_asm_1x1u.hpp", "conv"),
        ("projects/miopen/src/kernels/gfx942130.HIP.fdb.txt.bz2", "databases"),
        ("projects/miopen/src/kernels/gfx942130.db.txt.bz2", "databases"),
        ("projects/miopen/src/kernels/gfx908.kdb.bz2.dvc", "databases"),
        ("projects/miopen/src/kernels/gfx908.tn.model", "databases"),
        ("projects/miopen/src/kernels/lgbm_rank.bin", "databases"),
        ("projects/miopen/src/kernels/lgbm_model_meta.json", "databases"),
        ("projects/miopen/src/kernels/MIOpenBatchNormFwdTrainSpatial.cl", "kernels"),
        ("projects/miopen/src/kernels/winograd/a.s", "kernels"),
        ("projects/miopen/src/ck_impl/a.cpp", "ck"),
        ("projects/miopen/src/hipoc/hipoc_program.cpp", "runtime"),
        ("projects/miopen/fin/src/fin.cpp", "fin"),
        ("projects/miopen/src/batchnorm/problem_description.cpp", "op-batchnorm"),
        ("projects/miopen/src/include/miopen/batchnorm/solvers.hpp", "op-batchnorm"),
        ("projects/miopen/src/include/miopen/handle.hpp", "core"),
        ("projects/miopen/src/find_db.cpp", "core"),
        ("projects/miopen/test/gtest/conv_api.cpp", "tests"),
        ("projects/miopen/driver/conv_driver.hpp", "driver"),
        ("projects/miopen/CMakeLists.txt", "miopen-build"),
        ("projects/miopen/cmake/ClangTidy.cmake", "miopen-build"),
        ("projects/miopen/requirements.txt", "miopen-build"),
        ("dnn-providers/miopen-provider/MiopenApi.hpp", "miopen-provider"),
        ("CMakeLists.txt", "build-ci"),
        (".github/workflows/ci.yml", "build-ci"),
        ("shared/util/x.hpp", "shared-util"),
        ("projects/miopen/skills/miopen-deep-review/SKILL.md", "tooling"),
        ("tools/x.sh", "tooling"),
        ("projects/miopen/docs/conf.py", "miopen-docs"),
        ("projects/other/src/a.cpp", "other-src"),
        ("docs/guide.md", "docs"),
    ],
)
def test_area_table_rows_and_fallback(part, path, slug):
    assert part.area_of(path) == slug


def test_first_matching_row_wins_over_docs(part):
    """A markdown file under a table row's path takes that row's slug, not `docs`."""
    assert part.area_of("projects/miopen/src/solver/README.md") == "solver"


@pytest.mark.parametrize(
    "path, docs",
    [
        ("a/b.md", True),
        ("a/b.rst", True),
        ("LICENSE", True),
        ("projects/x/README", True),
        ("docs/notes.txt", True),
        ("a/CMakeLists.txt", False),
        ("a/requirements.txt", False),
        ("a/.clang-format", False),
        ("a/b.cpp", False),
    ],
)
def test_docs_detection(part, path, docs):
    assert part.is_docs(path) is docs


def test_areas_writes_tsv(part, tmp_path):
    rd = make_round(
        tmp_path, [(1, 1, "projects/miopen/src/solver/a.cpp"), (2, 0, "docs/x.md")]
    )
    run(part, "areas", rd)
    assert read_tsv(rd / "areas.tsv") == [
        ("projects/miopen/src/solver/a.cpp", "solver"),
        ("docs/x.md", "docs"),
    ]


def plan(part, rd, *extra):
    run(part, "areas", rd)
    run(part, "plan", rd, *extra)
    owners = read_tsv(rd / "owners.tsv")
    lanes = {}
    for owner, path in owners:
        lanes.setdefault(owner, []).append(path)
    return lanes


def test_plan_small_change_with_binary_and_delete_add_pair_is_one_lane(part, tmp_path):
    rd = make_round(
        tmp_path,
        [
            (10, 5, "projects/miopen/src/solver/old.cpp"),
            (10, 5, "projects/miopen/src/solver/new.cpp"),
            ("-", "-", "projects/miopen/test/golden.bin"),
        ],
    )
    lanes = plan(part, rd)
    assert list(lanes) == ["CORR1"]
    assert len(lanes["CORR1"]) == 3


def test_plan_groups_by_area_and_never_splits_a_file(part, tmp_path):
    rd = make_round(
        tmp_path,
        [
            (900, 0, "projects/miopen/src/solver/a.cpp"),
            (900, 0, "projects/miopen/src/solver/b.cpp"),
            (900, 0, "projects/miopen/src/conv/c.hpp"),
            (3000, 0, "projects/miopen/src/conv/huge.hpp"),
            (50, 0, "projects/miopen/driver/s.cpp"),
        ],
    )
    lanes = plan(part, rd)
    assert ["projects/miopen/src/conv/huge.hpp"] in lanes.values()
    flat = [p for ps in lanes.values() for p in ps]
    assert len(flat) == len(set(flat)) == 5
    solver = {o for o, ps in lanes.items() if "projects/miopen/src/solver/a.cpp" in ps}
    assert solver == {
        o for o, ps in lanes.items() if "projects/miopen/src/solver/b.cpp" in ps
    }
    for owner, paths in lanes.items():
        if paths != ["projects/miopen/src/conv/huge.hpp"]:
            lines = {"a.cpp": 900, "b.cpp": 900, "c.hpp": 900, "s.cpp": 50}
            assert sum(lines[p.rsplit("/", 1)[-1]] for p in paths) <= part.TARGET_LINES


def test_plan_sampled_files_get_their_own_lane(part, tmp_path):
    rd = make_round(
        tmp_path,
        [
            (5, 0, "projects/miopen/src/solver/a.cpp"),
            (9000, 0, "projects/miopen/src/solver/gen.cpp"),
        ],
    )
    sampled = tmp_path / "sampled.tsv"
    sampled.write_text("projects/miopen/src/solver/gen.cpp\tcodegen tool\tsample\n")
    lanes = plan(part, rd, "--sampled", str(sampled))
    assert lanes == {
        "CORR1": ["projects/miopen/src/solver/a.cpp"],
        "CORR2": ["projects/miopen/src/solver/gen.cpp"],
    }


def test_plan_docs_only_change_is_owned_by_docs(part, tmp_path):
    rd = make_round(
        tmp_path, [(3, 1, "docs/a.md"), (1, 0, "projects/miopen/src/solver/README.md")]
    )
    assert plan(part, rd) == {
        "DOCS": ["docs/a.md", "projects/miopen/src/solver/README.md"]
    }


def test_plan_fails_when_numstat_path_has_no_area(part, tmp_path):
    rd = make_round(tmp_path, [(1, 0, "tools/a.py")], files=[])
    (rd / "areas.tsv").write_text("")
    with pytest.raises(SystemExit) as exc:
        run(part, "plan", rd)
    assert "tools/a.py" in str(exc.value)


GATE_NUMSTAT = [(10, 2, "a.cpp"), ("-", "-", "b.bin"), (5, 5, "c.cpp")]


def gate(part, tmp_path, owners, capsys):
    rd = make_round(tmp_path, GATE_NUMSTAT)
    (rd / "owners.tsv").write_text("".join(f"{o}\t{p}\n" for o, p in owners))
    try:
        run(part, "gate", rd)
        code = 0
    except SystemExit as e:
        code = e.code
    return code, capsys.readouterr().out


def test_gate_passes_when_every_path_is_owned_once(part, tmp_path, capsys):
    code, out = gate(
        part,
        tmp_path,
        [("CORR1", "a.cpp"), ("CORR1", "b.bin"), ("CORR2", "c.cpp")],
        capsys,
    )
    assert code == 0
    assert "3 files, 22 lines" in out


@pytest.mark.parametrize(
    "owners, message",
    [
        ([("CORR1", "a.cpp"), ("CORR1", "b.bin")], "unowned: c.cpp"),
        (
            [
                ("CORR1", "a.cpp"),
                ("CORR2", "a.cpp"),
                ("CORR1", "b.bin"),
                ("CORR1", "c.cpp"),
            ],
            "owned twice: a.cpp",
        ),
        (
            [
                ("CORR1", "a.cpp"),
                ("CORR1", "b.bin"),
                ("CORR1", "c.cpp"),
                ("CORR1", "x.cpp"),
            ],
            "not in numstat: x.cpp",
        ),
    ],
)
def test_gate_failures(part, tmp_path, capsys, owners, message):
    code, out = gate(part, tmp_path, owners, capsys)
    assert code == 1
    assert message in out


def test_gate_line_sum_mismatch(part, tmp_path, capsys):
    """An unowned path leaves the owned lines short of the total."""
    code, out = gate(part, tmp_path, [("CORR1", "a.cpp"), ("CORR1", "b.bin")], capsys)
    assert code == 1
    assert "owned 12 of 22 lines" in out


@pytest.mark.parametrize("cmd", ["gate", "packets"])
@pytest.mark.parametrize(
    "owners, message",
    [
        ("TESTQ\ta.cpp\n", "owner is not CORR<k> or DOCS: 'TESTQ'"),
        ("CORR\ta.cpp\n", "owner is not CORR<k> or DOCS: 'CORR'"),
        ("CORR1 a.cpp\n", "no tab"),
    ],
)
def test_malformed_owners_row_fails_with_one_line(part, tmp_path, cmd, owners, message):
    rd = make_round(tmp_path, GATE_NUMSTAT)
    run(part, "areas", rd)
    (rd / "owners.tsv").write_text(owners)
    with pytest.raises(SystemExit) as exc:
        run(part, cmd, rd)
    assert message in str(exc.value)


def test_packets_lanes_tokens_are_stripped(part, tmp_path):
    rd = make_round(
        tmp_path,
        [(1, 0, "projects/miopen/src/solver/a.cpp"), (1, 0, "CMakeLists.txt")],
    )
    run(part, "areas", rd)
    (rd / "owners.tsv").write_text(
        "CORR1\tprojects/miopen/src/solver/a.cpp\nCORR1\tCMakeLists.txt\n"
    )
    run(part, "packets", rd, "--lanes", "TESTQ, DEPS")
    assert (rd / "packets" / "TESTQ.md").is_file()
    assert (rd / "packets" / "DEPS.md").is_file()


def test_gate_empty_owners(part, tmp_path):
    rd = make_round(tmp_path, GATE_NUMSTAT)
    (rd / "owners.tsv").write_text("")
    with pytest.raises(SystemExit) as exc:
        run(part, "gate", rd)
    assert "owners.tsv empty" in str(exc.value)


def test_planned_owners_pass_the_gate(part, tmp_path):
    rd = make_round(
        tmp_path,
        [
            (1500, 0, "projects/miopen/src/solver/a.cpp"),
            (1500, 0, "projects/miopen/src/conv/b.hpp"),
            ("-", "-", "tools/x.bin"),
        ],
    )
    plan(part, rd)
    run(part, "gate", rd)


def packets_round(part, tmp_path):
    rd = make_round(
        tmp_path,
        [
            (30, 10, "projects/miopen/src/solver/a.cpp"),
            ("-", "-", "projects/miopen/src/solver/gen.bin"),
            (4, 0, "CMakeLists.txt"),
        ],
    )
    (rd / "orient").mkdir()
    (rd / "orient" / "solver.md").write_text("x")
    return rd


def test_correctness_packet_lists_numstat_areas_orient_and_sampled(part, tmp_path):
    rd = packets_round(part, tmp_path)
    sampled = rd / "sampled.tsv"
    sampled.write_text("projects/miopen/src/solver/gen.bin\tgen.py\tsample\n")
    run(part, "areas", rd)
    (rd / "owners.tsv").write_text(
        "CORR1\tprojects/miopen/src/solver/a.cpp\nCORR1\tprojects/miopen/src/solver/gen.bin\nCORR2\tCMakeLists.txt\n"
    )
    run(part, "packets", rd, "--sampled", str(sampled))
    text = (rd / "packets" / "CORR1.md").read_text()
    assert "| `projects/miopen/src/solver/a.cpp` | 30 | 10 | solver |" in text
    assert "| `projects/miopen/src/solver/gen.bin` | - | - | solver |" in text
    assert "- orient/solver.md" in text
    assert "Sampled: gen.py; sample projects/miopen/src/solver/gen.bin" in text
    other = (rd / "packets" / "CORR2.md").read_text()
    assert "orient/build-ci.md (absent" in other
    assert "Sampled:" not in other


def test_sampled_lane_without_sample_file_fails(part, tmp_path):
    rd = packets_round(part, tmp_path)
    sampled = rd / "sampled.tsv"
    sampled.write_text("projects/miopen/src/solver/gen.bin\tgen.py\n")
    run(part, "areas", rd)
    (rd / "owners.tsv").write_text(
        "".join(
            f"CORR1\t{p}\n"
            for p in [
                "projects/miopen/src/solver/a.cpp",
                "projects/miopen/src/solver/gen.bin",
                "CMakeLists.txt",
            ]
        )
    )
    with pytest.raises(SystemExit) as exc:
        run(part, "packets", rd, "--sampled", str(sampled))
    assert "no sample file" in str(exc.value)


def test_concern_packets_scope_and_parent(part, tmp_path, capsys):
    rd = packets_round(part, tmp_path)
    run(part, "areas", rd)
    plan_owners = "".join(
        f"CORR1\t{p}\n"
        for p in [
            "projects/miopen/src/solver/a.cpp",
            "projects/miopen/src/solver/gen.bin",
            "CMakeLists.txt",
        ]
    )
    (rd / "owners.tsv").write_text(plan_owners)
    run(part, "packets", rd, "--lanes", "BUILD,TESTQ,DEPS", "--parent", "#123")
    build = (rd / "packets" / "BUILD.md").read_text()
    assert "`CMakeLists.txt`" in build and "a.cpp" not in build
    assert "Stack parent: #123" in build
    assert "a.cpp" in (rd / "packets" / "TESTQ.md").read_text()
    assert (rd / "packets" / "DEPS.md").is_file()
    assert (rd / "packets" / "CORR1.md").is_file()


def test_concern_lane_without_files_in_scope_is_not_written(part, tmp_path, capsys):
    rd = make_round(tmp_path, [(1, 0, "projects/miopen/src/solver/a.cpp")])
    run(part, "areas", rd)
    (rd / "owners.tsv").write_text("CORR1\tprojects/miopen/src/solver/a.cpp\n")
    run(part, "packets", rd, "--lanes", "BUILD")
    assert not (rd / "packets" / "BUILD.md").exists()
    assert "BUILD: no changed file in scope" in capsys.readouterr().out


def test_build_and_deps_scope_cover_registration_files_and_source(part):
    rows = [
        (p, 1, "")
        for p in (
            "projects/miopen/cmake/miopen-config.cmake.in",
            "projects/miopen/test/gtest/test_categories.yaml",
            "projects/miopen/Dockerfile",
            "projects/miopen/rbuild.ini",
            "projects/miopen/dev-requirements.txt",
            "projects/miopen/src/solver/a.cpp",
            "docs/guide.md",
        )
    ]
    build = {r[0] for r in part.concern_scope("BUILD", rows)}
    deps = {r[0] for r in part.concern_scope("DEPS", rows)}
    assert build == {rows[0][0], rows[1][0], rows[2][0], rows[3][0]}
    assert deps == {rows[0][0], rows[3][0], rows[4][0], rows[5][0]}


def test_plan_keeps_a_test_with_its_source_across_areas(part, tmp_path):
    rd = make_round(
        tmp_path,
        [
            (900, 0, "projects/miopen/test/test_widget.py"),
            (900, 0, "projects/miopen/src/solver/a.cpp"),
            (900, 0, "projects/miopen/src/solver/widget.cpp"),
            (900, 0, "projects/miopen/src/solver/b.cpp"),
        ],
    )
    lanes = plan(part, rd)
    pair = {
        "projects/miopen/test/test_widget.py",
        "projects/miopen/src/solver/widget.cpp",
    }
    assert any(pair <= set(ps) for ps in lanes.values())


def test_plan_pairs_a_test_with_its_header_and_implementation(part, tmp_path):
    paths = [
        "projects/miopen/test/TestWidget.cpp",
        "projects/miopen/src/solver/widget.cpp",
        "projects/miopen/src/solver/widget.hpp",
    ]
    key = "projects/miopen/src/solver/widget.cpp"
    assert part.pair_tests(paths) == {p: key for p in paths}


def test_plan_leaves_a_test_unpaired_when_same_stem_files_span_directories(part):
    paths = [
        "projects/miopen/test/TestWidget.cpp",
        "projects/miopen/src/solver/widget.cpp",
        "projects/miopen/src/conv/widget.hpp",
    ]
    assert part.pair_tests(paths) == {}
    assert part.pair_tests(paths[:2]) == {p: paths[1] for p in paths[:2]}


def test_plan_gives_an_oversized_pair_its_own_lane(part, tmp_path):
    rd = make_round(
        tmp_path,
        [
            (1500, 0, "projects/miopen/test/test_big.py"),
            (1500, 0, "projects/miopen/src/solver/big.cpp"),
            (50, 0, "projects/miopen/src/solver/small.cpp"),
        ],
    )
    lanes = plan(part, rd)
    assert [
        "projects/miopen/test/test_big.py",
        "projects/miopen/src/solver/big.cpp",
    ] in [sorted(ps, reverse=True) for ps in lanes.values()]
    assert len(lanes) == 2


def test_packets_drift_lane_follows_the_trigger(part, tmp_path):
    rd = make_round(tmp_path, [(3, 1, "projects/miopen/src/solver/a.cpp")])
    run(part, "areas", rd)
    run(part, "plan", rd)
    (rd / "drift.txt").write_text("Target tip: abc\nTrigger: no (range review)\n")
    run(part, "packets", rd, "--lanes", "DRIFT")
    assert not (rd / "packets" / "DRIFT.md").exists()
    (rd / "drift.txt").write_text("Target tip: abc\nTrigger: yes\n")
    run(part, "packets", rd, "--lanes", "DRIFT")
    assert "drift.txt" in (rd / "packets" / "DRIFT.md").read_text()


def test_orient_gate_accepts_added_files_and_flags_paths_absent_at_base(
    part, tmp_path, capsys
):
    clone = tmp_path / "clone"
    (clone / "src").mkdir(parents=True)
    (clone / "src" / "old.cpp").write_text("x\n")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t"}
    env.update(GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t", HOME=str(tmp_path))
    for args in (["init", "-q"], ["add", "."], ["commit", "-qm", "base"]):
        subprocess.run(["git", "-C", str(clone), *args], check=True, env=env)
    base = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (tmp_path / "r1").mkdir()
    rd = make_round(tmp_path / "r1", [(1, 0, "src/new.hpp")])
    (rd / "PIN.md").write_text(f"CLONE: {clone}\nBASE: {base}\n")
    (rd / "orient").mkdir()
    (rd / "orient" / "a.md").write_text(
        "Adds `src/new.hpp:3`; reads `./src/old.cpp`.\n"
    )
    (rd / "CONTEXT.md").write_text("Area `src/old.cpp:1` and dir `src/x`.\n")
    with pytest.raises(SystemExit) as exc:
        run(part, "orient-gate", rd)
    assert exc.value.code == 1
    assert capsys.readouterr().out == "not at base: src/x\n"
    (rd / "CONTEXT.md").write_text("Area `src/old.cpp:1` and dir `src`.\n")
    run(part, "orient-gate", rd)
    assert capsys.readouterr().out == ""
