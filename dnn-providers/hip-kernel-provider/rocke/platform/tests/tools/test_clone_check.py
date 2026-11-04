# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Behaviour of tools/clone_check.py on small synthetic git repositories."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[3] / "tools" / "clone_check.py"

if not TOOL.is_file():
    pytest.skip(
        "tools/clone_check.py is absent from this platform-only tree",
        allow_module_level=True,
    )

POSIX_ONLY = pytest.mark.skipif(
    os.name != "posix", reason="needs POSIX file names, modes or symlinks"
)

_GIT_ENV = dict(
    os.environ,
    GIT_AUTHOR_NAME="t",
    GIT_AUTHOR_EMAIL="t@example.com",
    GIT_COMMITTER_NAME="t",
    GIT_COMMITTER_EMAIL="t@example.com",
)

# One copied region: five non-trivial emitter statements, 66 tokens. {load}
# and {k} sit mid-block: a change there leaves fewer than 50 matching tokens on
# either side, even counting the shared end of the signature.
BLOCK = """\
    v1 = b.add(base, b.mul(row, stride))
    v2 = b.{load}(v1, b.const_f32({k}), ptr)
    v3 = b.select(ok, v2, zero)
    v4 = b.fmul(v3, v1)
    b.store(out, v3, v4)
"""


def func(name: str, body: str = BLOCK, load: str = "load", k: str = "32.0") -> str:
    head = f"def {name}(b, base, row, stride, ok, zero, ptr, out):\n"
    return head + body.format(load=load, k=k) + "\n\n"


UNRELATED = """\
    a1 = b.sub(base, row)
    a2 = b.cmp_lt(a1, stride)
    a3 = b.land(a2, ok)
    a4 = b.mod(a1, stride)
    b.barrier(a3, a4)
"""


class Repo:
    def __init__(self, path: Path):
        self.path = path
        self.git("init", "-q")
        self.git(
            "symbolic-ref", "HEAD", "refs/heads/develop"
        )  # `init -b` needs git 2.28

    def git(self, *args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=self.path,
            env=_GIT_ENV,
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()

    def write(self, rel: str, text: str) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")

    def commit(self) -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", "c")
        return self.git("rev-parse", "HEAD")

    def check(self, *args: str, root: Path | None = None) -> tuple[int, dict]:
        proc = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--root",
                str(root or self.path),
                "--format",
                "json",
                *args,
            ],
            capture_output=True,
            text=True,
        )
        return proc.returncode, (json.loads(proc.stdout) if proc.stdout else {})


def set_git_config(monkeypatch, tmp_path_factory, options: dict) -> None:
    """Apply user-level git settings through a global config file.

    A file under HOME works on every git; GIT_CONFIG_COUNT needs git 2.31 and
    older gits silently ignore it, which would leave the settings unapplied.
    """
    home = tmp_path_factory.mktemp("home")
    for key, value in options.items():
        subprocess.run(
            ["git", "config", "--file", str(home / ".gitconfig"), key, value],
            check=True,
        )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    r = Repo(tmp_path)
    r.write("library/kernels/keep.py", "X = 1\n")
    return r


def kinds(report: dict) -> list[str]:
    return [f["kind"] for f in report["findings"]]


def test_code_written_once_is_not_a_copy(repo):
    base = repo.commit()
    repo.write("library/kernels/a.py", func("f1"))
    head = repo.commit()
    rc, report = repo.check("--base", base, "--head", head)
    assert rc == 0 and report["findings"] == []


def test_new_code_written_twice_is_a_new_copy(repo):
    base = repo.commit()
    repo.write("library/kernels/a.py", func("f1") + func("f2"))
    head = repo.commit()
    rc, report = repo.check("--base", base, "--head", head)
    assert rc == 0
    assert kinds(report) == ["NEW COPY"]
    assert (
        report["findings"][0]["copies_before"],
        report["findings"][0]["copies_after"],
    ) == (0, 2)
    rc, _ = repo.check("--base", base, "--head", head, "--fail-on-new")
    assert rc == 1


def test_pasting_an_existing_block_is_an_exact_new_copy(repo):
    repo.write("library/kernels/a.py", func("f1") + func("f2"))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("f3"))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    (finding,) = report["findings"]
    assert (finding["kind"], finding["match"]) == ("NEW COPY", "exact")
    assert (finding["copies_before"], finding["copies_after"]) == (2, 3)
    new = [loc["path"] for loc in finding["locations"] if loc["class"] == "new"]
    assert new == ["library/kernels/b.py"]


def test_pasted_block_with_renamed_variables_is_a_renamed_copy(repo):
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    renamed = (
        BLOCK.replace("v1", "w1")
        .replace("v2", "w2")
        .replace("v3", "w3")
        .replace("v4", "w4")
    )
    repo.write("library/kernels/b.py", func("g", renamed))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert [(f["kind"], f["match"]) for f in report["findings"]] == [
        ("NEW COPY", "renamed")
    ]


@pytest.mark.parametrize("variant", [{"load": "store"}, {"k": "16.0"}])
def test_different_method_or_constant_is_not_a_copy(repo, variant):
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("g", **variant))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_rename_map_spans_the_whole_window(repo):
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    # Same statements, but the second one reads v2 instead of v1: a different data flow.
    repo.write(
        "library/kernels/b.py",
        func("g", BLOCK.replace("{load}(v1,", "{load}(v2,")),
    )
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


# Two statements, well over the 50-token minimum.
LONG = """\
    acc = b.mfma(b.load(a_ptr, b.add(base, b.mul(row, stride))), b.load(q_ptr, b.add(base, b.mul(stride, row))), acc)
    b.store(out, b.add(base, b.mul(row, stride)), b.select(ok, acc, zero))
"""


def test_copy_of_two_long_statements_is_a_copy(repo):
    repo.write("library/kernels/a.py", func("f1", LONG))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("g", LONG))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


ARGS = (
    "    {callee}(kernel, grid=(n_blocks, 1, 1), block=(256, 1, 1), shared=lds_bytes,"
    " args=[a_ptr, b_ptr, c_ptr, n_elems, stride_a, stride_b, stride_c, alpha, beta, gamma])\n"
)


def test_argument_list_copied_into_a_different_call_is_a_copy(repo):
    repo.write("library/kernels/a.py", func("f1", ARGS.format(callee="launch")))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("g", ARGS.format(callee="run_kernel")))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


# Five statements, but under 50 tokens even with the shared end of the signature.
TINY = """\
    x1 = b.f(base)
    x2 = b.g(x1)
    x3 = b.h(x2)
    x4 = b.k(x3)
    b.st(x4)
"""


def test_copy_below_the_token_minimum_is_not_a_copy(repo):
    repo.write("library/kernels/a.py", "def f1(b, base):\n" + TINY)
    base = repo.commit()
    repo.write("library/kernels/b.py", "def g(b, base, extra):\n" + TINY)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_moving_a_block_or_renaming_a_file_is_not_a_copy(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", "Y = 2\n\n\n" + func("f2"))
    base = repo.commit()
    repo.write("library/kernels/b.py", "Y = 2\n")
    repo.write("library/kernels/c.py", func("f2"))
    repo.git("mv", "library/kernels/a.py", "library/kernels/d.py")
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []
    assert report["existing_copies"] == []  # a moved copy was not edited


def existing(report: dict) -> list[tuple]:
    return [
        (e["kind"], sorted((loc["path"], loc["class"]) for loc in e["locations"]))
        for e in report["existing_copies"]
    ]


def test_same_fix_in_both_existing_copies_is_reported_as_existing_copies(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", func("f2"))
    base = repo.commit()
    fixed = BLOCK.replace("b.add(", "b.sub(")
    repo.write("library/kernels/a.py", func("f1", fixed))
    repo.write("library/kernels/b.py", func("f2", fixed))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []
    assert existing(report) == [
        (
            "EXISTING COPIES EDITED",
            [("library/kernels/a.py", "edited"), ("library/kernels/b.py", "edited")],
        )
    ]


def test_editing_one_of_three_existing_copies_lists_the_others(repo):
    for name in "abc":
        repo.write(f"library/kernels/{name}.py", func(f"f_{name}"))
    base = repo.commit()
    repo.write("library/kernels/a.py", func("f_a", BLOCK.replace("b.add(", "b.mul(")))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []
    assert existing(report) == [
        (
            "EXISTING COPIES EDITED",
            [
                ("library/kernels/a.py", "edited"),
                ("library/kernels/b.py", "unchanged"),
                ("library/kernels/c.py", "unchanged"),
            ],
        )
    ]


SHORT = """\
    v1 = b.add(base, row)
    v2 = b.select(ok, v1, zero)
    b.store(out, v2, v1)
"""


def short_func(name: str, body: str) -> str:
    return f"def {name}(b, base, row, ok, zero, out):\n{body}\n\n"


def test_short_existing_copies_edited_the_same_way_are_not_a_new_copy(repo):
    # The copies share a name, so their old text is identical. Each is 48 tokens,
    # below the window, so the base shows no copy at all. Adding the same two
    # lines to both makes them visible; they already existed.
    repo.write("library/kernels/a.py", short_func("f", SHORT))
    repo.write("library/kernels/b.py", short_func("f", SHORT))
    base = repo.commit()
    _, listing = repo.check("--backlog", "--head", base)
    assert listing["groups"] == []
    grown = SHORT.replace(
        "    b.store(",
        "    v3 = b.load(ptr, v2)\n    v4 = b.fmul(v3, v1)\n    b.store(",
    )
    repo.write("library/kernels/a.py", short_func("f", grown))
    repo.write("library/kernels/b.py", short_func("f", grown))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []
    assert existing(report) == [
        (
            "EXISTING COPIES EDITED",
            [("library/kernels/a.py", "edited"), ("library/kernels/b.py", "edited")],
        )
    ]


def test_replacing_unrelated_code_with_a_pasted_block_is_a_likely_copy(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", func("g", UNRELATED))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("g"))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    (finding,) = report["findings"]
    assert finding["kind"] == "LIKELY COPY"
    assert [
        loc["class"] for loc in finding["locations"] if loc["path"].endswith("b.py")
    ] == ["replaced"]


GROUPS_PATH = "tools/clone_groups.json"
GROUP = {
    "name": "per-arch",
    "files": ["library/kernels/gfx942/k.py", "library/kernels/gfx950/k.py"],
    "reason": "arch-specific schedules",
}
GROUPS = json.dumps({"groups": [GROUP]})


def test_editing_one_side_of_an_approved_fork_names_the_unchanged_sibling(repo):
    repo.write(GROUPS_PATH, GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", func("f"))
    base = repo.commit()
    repo.write(
        "library/kernels/gfx942/k.py", func("f", BLOCK.replace("b.add(", "b.mul("))
    )
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []
    assert existing(report) == [
        (
            "SIBLING NOTE",
            [
                ("library/kernels/gfx942/k.py", "edited"),
                ("library/kernels/gfx950/k.py", "unchanged"),
            ],
        )
    ]
    assert report["existing_copies"][0]["group"] == "per-arch"


def test_different_edits_on_both_sides_of_an_approved_fork_get_a_sibling_note(repo):
    repo.write(GROUPS_PATH, GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", func("f"))
    base = repo.commit()
    repo.write(
        "library/kernels/gfx942/k.py", func("f", BLOCK.replace("b.add(", "b.mul("))
    )
    repo.write(
        "library/kernels/gfx950/k.py", func("f", BLOCK.replace("b.add(", "b.sub("))
    )
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert existing(report) == [
        (
            "SIBLING NOTE",
            [
                ("library/kernels/gfx942/k.py", "edited"),
                ("library/kernels/gfx950/k.py", "edited"),
            ],
        )
    ]


def test_text_report_lists_ten_existing_copy_groups_then_counts_the_rest(repo):
    # Twelve separate copy pairs, one function per file, so no larger window
    # joins them into a single group.
    for i in range(12):
        repo.write(f"library/kernels/a{i}.py", func("f", k=f"{i}.0"))
        repo.write(f"library/kernels/b{i}.py", func("f", k=f"{i}.0"))
    base = repo.commit()
    for i in range(12):
        edited = func("f", BLOCK.replace("b.add(", "b.mul("), k=f"{i}.0")
        repo.write(f"library/kernels/a{i}.py", edited)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert len(report["existing_copies"]) == 12  # JSON keeps every entry
    text = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--root",
            str(repo.path),
            "--base",
            base,
            "--head",
            head,
        ],
        capture_output=True,
        text=True,
    ).stdout
    assert text.count("EXISTING COPIES EDITED  ") == 10
    assert "+2 more existing copy groups edited" in text


IF_BOTH = """\
    v1 = b.add(base, row)
    v2 = b.select(ok, v1, zero)
    if ok:
        v3 = b.load(ptr, v2)
        v4 = b.fmul(v3, b.const_f32(32.0))
    b.store(out, v2, v4)
"""
IF_ONE = IF_BOTH.replace(
    "        v4 = b.fmul(", "    v4 = b.fmul("
)  # same lines, different control flow


def test_forks_that_differ_only_in_indentation_get_a_sibling_note(repo):
    repo.write(GROUPS_PATH, GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", func("f"))
    base = repo.commit()
    repo.write("library/kernels/gfx942/k.py", func("f", IF_BOTH))
    repo.write("library/kernels/gfx950/k.py", func("f", IF_ONE))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert [e["kind"] for e in report["existing_copies"]] == ["SIBLING NOTE"]


def test_the_same_edit_on_approved_forks_saved_with_a_bom_and_with_crlf_is_not_a_note(
    repo,
):
    # One fork starts with a BOM, the other uses CRLF; the edit is identical, so
    # nothing differs between the copies and no sibling note is due.
    def save(rel: str, text: str, bom: bool, crlf: bool) -> None:
        data = text.replace("\n", "\r\n") if crlf else text
        target = repo.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((b"\xef\xbb\xbf" if bom else b"") + data.encode())

    repo.write(GROUPS_PATH, GROUPS)
    before = func("f")
    after = func("f", BLOCK.replace("b.add(", "b.mul("))
    save("library/kernels/gfx942/k.py", before, bom=True, crlf=False)
    save("library/kernels/gfx950/k.py", before, bom=False, crlf=True)
    base = repo.commit()
    save("library/kernels/gfx942/k.py", after, bom=True, crlf=False)
    save("library/kernels/gfx950/k.py", after, bom=False, crlf=True)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []
    assert existing(report) == []


def test_same_edit_on_both_sides_of_an_approved_fork_reports_nothing(repo):
    repo.write(GROUPS_PATH, GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", func("f"))
    base = repo.commit()
    fixed = func("f", BLOCK.replace("b.add(", "b.sub("))
    repo.write("library/kernels/gfx942/k.py", fixed)
    repo.write("library/kernels/gfx950/k.py", fixed)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []
    assert report["existing_copies"] == []  # the fork was kept in sync


def test_copy_between_approved_fork_files_is_a_sibling_copy(repo):
    repo.write(GROUPS_PATH, GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", "Z = 0\n")
    base = repo.commit()
    repo.write("library/kernels/gfx950/k.py", func("f"))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert [(f["kind"], f["group"]) for f in report["findings"]] == [
        ("SIBLING COPY", "per-arch")
    ]


def test_copy_into_a_file_outside_an_approved_group_is_a_new_copy(repo):
    repo.write(GROUPS_PATH, GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", func("f"))
    base = repo.commit()
    repo.write("library/kernels/gfx1250/k.py", func("f"))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


def test_second_copy_inside_one_approved_file_is_a_new_copy(repo):
    repo.write(GROUPS_PATH, GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", func("f"))
    base = repo.commit()
    repo.write("library/kernels/gfx942/k.py", func("f") + func("g"))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


SIX = [
    "    t1 = b.add(x, y)\n",
    "    t2 = b.mul(t1, y)\n",
    "    t3 = b.sub(t2, x)\n",
    "    t4 = b.select(t3, t1, t2)\n",
    "    t5 = b.load(t4, x)\n",
    "    b.store(t5, t4, t3)\n",
]


PIECE_P = [
    "p1 = b.add(base, b.mul(row, stride))",
    "p2 = b.select(ok, p1, b.const_f32(1.0))",
    "p3 = b.fmul(p2, b.load(ptr, p1))",
    "b.store(out, p3, p2)",
]
PIECE_Q = [
    "q1 = b.sub(stride, b.mod(row, base))",
    "q2 = b.cmp_lt(q1, b.const_i32(64))",
    "q3 = b.land(q2, ok)",
    "b.barrier(q3, b.load(ptr, q1))",
]


def two_pieces(name: str, first: list[str], second: list[str]) -> str:
    """Two pieces whose junction shares one line, so their windows overlap in lines."""
    lines = first[:-1] + [first[-1] + "; " + second[0]] + second[1:]
    head = f"def {name}(b, base, row, stride, ok, ptr, out):\n"
    return head + "".join(f"    {line}\n" for line in lines)


def test_reordered_pieces_from_one_source_stay_separate_findings(repo):
    repo.write("library/kernels/a.py", two_pieces("fa", PIECE_P, PIECE_Q))
    base = repo.commit()
    # Q then P: the copy advances while its source jumps backwards, so the two
    # pieces must not merge into one region that claims a:2-6 was copied as b:2-6.
    repo.write("library/kernels/b.py", two_pieces("fb", PIECE_Q, PIECE_P))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    spans = sorted(
        tuple((loc["path"][-4:], loc["start"], loc["end"]) for loc in f["locations"])
        for f in report["findings"]
    )
    assert spans == [
        (("a.py", 2, 5), ("b.py", 5, 8)),  # P
        (("a.py", 5, 8), ("b.py", 2, 5)),  # Q
    ]


def test_overlapping_fragments_from_two_sources_stay_separate(repo):
    repo.write("library/kernels/a.py", "def fa(b, x, y):\n" + "".join(SIX[:5]))
    # t1 is a parameter here, so it is a local variable like in fa and fc.
    repo.write("library/kernels/b.py", "def fb(b, x, y, t1):\n" + "".join(SIX[1:]))
    base = repo.commit()
    repo.write("library/kernels/c.py", "def fc(b, x, y):\n" + "".join(SIX))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    sources = sorted(
        tuple(loc["path"] for loc in f["locations"] if loc["class"] == "unchanged")
        for f in report["findings"]
    )
    assert sources == [("library/kernels/a.py",), ("library/kernels/b.py",)]


NESTED = """\
    v1 = b.add(base, row)
    v2 = {helper}(b, v1)
    v3 = b.select(ok, v2, zero)
    v4 = b.load(ptr, v3)
    b.store(out, v3, v4)

    def inner():
        {helper} = 0
        return {helper}
"""


def test_nested_function_locals_do_not_rename_outer_helpers(repo):
    repo.write("library/kernels/a.py", func("f1", NESTED.replace("{helper}", "emit_a")))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("f2", NESTED.replace("{helper}", "emit_b")))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


GLOBAL = """\
    global {mod}
    {mod} = load_module()
    v1 = {mod}.add(base, row)
    v2 = {mod}.select(ok, v1, zero)
    v3 = {mod}.load(ptr, v2)
    v4 = {mod}.fmul(v3, v1)
    v5 = {mod}.fadd(v4, v2)
    b.store(out, v4, v5)
"""


def test_declared_global_names_are_not_renamed(repo):
    # Assigning a `global` name binds the module-level name, not a local one.
    repo.write("library/kernels/a.py", func("f1", GLOBAL.replace("{mod}", "torch")))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("f2", GLOBAL.replace("{mod}", "np")))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_bare_field_declarations_are_skipped(repo):
    base = repo.commit()
    fields = "".join(f"    f{i}: int\n" for i in range(5))
    repo.write("library/kernels/a.py", f"class A:\n{fields}\n\nclass B:\n{fields}")
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


IMPORTS = """\
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
"""


def test_import_blocks_are_skipped(repo):
    base = repo.commit()
    repo.write("library/kernels/a.py", IMPORTS + "A = 1\n")
    repo.write("library/kernels/b.py", IMPORTS + "B = 2\n")
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


@pytest.mark.parametrize(
    "path",
    [
        "library/benchmarks/common/bench.py",
        "platform/python/rocke/examples/demo.py",
        "platform/python/rocke/benchmark/gemm/run.py",
        "library/builders/gfx950/attention/thing_verify.py",
        "library/builders/gfx950/attention/thing_sweep.py",
        "library/builders/gfx950/attention/bench_thing.py",
        "library/builders/gfx950/attention/benchmark_thing.py",
        "library/builders/gfx950/attention/reproduce_thing.py",
        "platform/python/rocke/portable_ir/examples/demo.py",
        "library/kernels/gfx942/experiments/d256/p1_prefetch.py",
        "library/tests/test_thing.py",
    ],
)
def test_benchmark_verify_and_example_scripts_are_out_of_scope(repo, path):
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    repo.write(path, func("f2"))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


@pytest.mark.parametrize("name", ["main", "make_inputs", "parse_args", "_parser"])
def test_cli_and_input_helper_functions_are_skipped(repo, name):
    repo.write("library/builders/gfx950/x/a.py", func(name))
    base = repo.commit()
    repo.write("library/builders/gfx950/x/b.py", func(name))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


AROUND_MAIN = """\
A1 = b0.add(x, y)
A2 = b0.mul(A1, y)


{main}
A3 = b0.sub(A2, x)
A4 = b0.select(A3, A1, A2)
"""


def test_skipped_function_does_not_complete_a_module_window(repo):
    # Four shared top-level statements plus an excluded main() must not form a
    # five-statement match.
    text = AROUND_MAIN.format(main=func("main"))
    repo.write("library/builders/gfx950/x/a.py", text)
    base = repo.commit()
    repo.write("library/builders/gfx950/x/b.py", text)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_dunder_main_block_is_skipped(repo):
    script = 'if __name__ == "__main__":\n' + BLOCK.format(load="load", k="32.0")
    repo.write("library/builders/gfx950/x/a.py", script)
    base = repo.commit()
    repo.write("library/builders/gfx950/x/b.py", script)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


@pytest.mark.parametrize("name", ["check", "bench"])
def test_check_and_bench_functions_are_still_checked(repo, name):
    repo.write("library/kernels/common/a.py", func(name))
    base = repo.commit()
    repo.write("library/kernels/common/b.py", func(name))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


CONFIG = """\
@dataclass
class Cfg:
    data: Spec = Spec(1)
    tile_m: int = field(default=64)
    tile_n: int = make_tile(64)
    warp_m: int = pick(2)
    warp_n: int = pick(4)
"""


def test_class_fields_with_defaults_are_skipped(repo):
    repo.write("library/kernels/a.py", CONFIG)
    base = repo.commit()
    repo.write("library/kernels/b.py", CONFIG)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_field_only_class_does_not_complete_a_module_window(repo):
    module = IMPORTS + "\n\n" + CONFIG
    repo.write("library/kernels/a.py", module)
    base = repo.commit()
    repo.write("library/kernels/b.py", module)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


BOOTSTRAP = """\
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.append(_HERE)
ROOT = Path(__file__).resolve().parents[2]
"""


def test_script_bootstrap_lines_are_setup(repo):
    repo.write("library/builders/gfx950/x/a.py", BOOTSTRAP)
    base = repo.commit()
    repo.write("library/builders/gfx950/x/b.py", BOOTSTRAP)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


KERNEL_SIGNATURE = """\
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    N = b.param("N", I32)
"""
SIGNATURE_CHAIN = """\
    sig = (
        SignatureBuilder("kernel")
        .ptr("q_ptr", io_dtype)
        .ptr("k_ptr", io_dtype)
        .ptr("v_ptr", io_dtype)
        .ptr("o_ptr", io_dtype)
        .scalar("scale", "f32")
        .scalar("num_seqs", "i32")
        .build()
    )
"""
# Braces are doubled: func() passes the body through str.format.
GUARDS = """\
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if spec.dtype not in ("f16", "bf16"):
        return False, f"unsupported dtype {{spec.dtype!r}}"
    if spec.head_size % 16 != 0:
        raise ValueError(f"head_size must be a multiple of 16, got {{spec.head_size}}")
    if spec.mask_mode not in ("none", "causal"):
        errors.append(f"unsupported mask_mode {{spec.mask_mode!r}}")
"""
PASS_THROUGH = """\
    return GemmSpec(
        tile_m=self.tile_m,
        tile_n=self.tile_n,
        tile_k=self.tile_k,
        warp_m=self.warp_m,
        warp_n=self.warp_n,
        wave_size=self.wave_size,
        pipeline=pipeline,
        epilogue=epilogue,
    )
"""


@pytest.mark.parametrize(
    "body", [KERNEL_SIGNATURE, SIGNATURE_CHAIN, GUARDS, PASS_THROUGH]
)
def test_declaration_validation_and_pass_through_boilerplate_is_not_a_copy(repo, body):
    repo.write("library/kernels/a.py", func("f1", body))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("f2", body))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_guard_with_logic_in_its_body_is_still_compared(repo):
    # One statement, so only what it does decides whether the `if` is a guard.
    body = GUARDS.replace(
        '        errors.append(f"unsupported mask_mode {{spec.mask_mode!r}}")\n',
        "        b.store(out, b.add(base, b.mul(row, stride)),"
        " b.select(ok, b.fmul(row, stride), b.const_f32(0.0)))\n",
    )
    repo.write("library/kernels/a.py", func("f1", body))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("f2", body))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


CONDITIONAL_PARTS = "".join(
    f"    if {flag}:\n        parts.append({flag})\n"
    for flag in "prologue smem_block launch_bounds extern_c debug_hooks epilogue".split()
)
DISPATCH = "".join(
    f'    if spec.kind == "{kind}":\n        return {kind}_kernel\n'
    for kind in "gemm conv norm moe attn gdn".split()
)


@pytest.mark.parametrize("body", [CONDITIONAL_PARTS, DISPATCH])
def test_conditional_appends_and_returns_of_values_are_compared(repo, body):
    # Only messages make a guard: appending or returning a value is logic.
    repo.write("library/kernels/a.py", func("f1", body))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("f2", body))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


FORWARDED = "".join(
    f"        {name}=self.{name},\n"
    for name in (
        "tile_m tile_n tile_k warp_m warp_n warp_tile_m warp_tile_n warp_tile_k"
        " wave_size pipeline epilogue"
    ).split()
)


def test_pass_through_keywords_inside_a_different_call_are_not_a_copy(repo):
    repo.write(
        "library/kernels/a.py",
        func(
            "f1",
            '    return ConvSpec(\n        problem=p.with_layout("nhwc"),\n'
            + FORWARDED
            + "    )\n",
        ),
    )
    base = repo.commit()
    repo.write(
        "library/kernels/b.py",
        func("f2", "    return GemmSpec(\n        layout=q,\n" + FORWARDED + "    )\n"),
    )
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_constant_setup_windows_are_skipped(repo):
    base = repo.commit()
    setup = "".join(f"    c{i} = b.const_i32({i})\n" for i in range(5))
    repo.write("library/kernels/a.py", func("f1", setup) + func("f2", setup))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


STORE_BLOCK = """\
    d_base_bytes = b.mul(d_base, c_half_bytes)
    safe_d_off = b.select(store_ok, d_base_bytes, oob_sentinel)
    acc_h = _trunc_f32(b, p.dtype, acc_to_flush)
    _buf_store_vN(b, p.dtype, d_rsrc, safe_d_off, c0, acc_h, 2)
    accs_flat[flat] = zero_acc
"""


def test_copied_straight_line_store_block_is_detected(repo):
    repo.write("library/kernels/a.py", func("f1", STORE_BLOCK))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("f2", STORE_BLOCK))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


def test_working_tree_mode_sees_uncommitted_and_untracked_files(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", "Y = 2\n")
    base = repo.commit()
    repo.write(
        "library/kernels/b.py", "Y = 2\n\n\n" + func("f2")
    )  # tracked, not committed
    repo.write("library/kernels/new.py", func("f3"))  # untracked
    _, report = repo.check("--base", base)
    assert report["head"] == "WORKTREE"
    (finding,) = report["findings"]
    new = sorted(loc["path"] for loc in finding["locations"] if loc["class"] == "new")
    assert new == ["library/kernels/b.py", "library/kernels/new.py"]


def test_linked_worktree_gives_the_same_result(repo, tmp_path):
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    linked = tmp_path.parent / (tmp_path.name + "-linked")
    repo.git("worktree", "add", "-q", str(linked), "HEAD")
    assert (linked / ".git").is_file()
    (linked / "library/kernels/new.py").write_text(func("f2"), newline="\n")
    _, report = repo.check("--base", base, root=linked)
    assert kinds(report) == ["NEW COPY"]


def test_unparsable_file_is_listed_and_the_run_continues(repo):
    base = repo.commit()
    repo.write("library/kernels/bad.py", "def broken(:\n")
    repo.write("library/kernels/a.py", func("f1") + func("f2"))
    head = repo.commit()
    rc, report = repo.check("--base", base, "--head", head)
    assert rc == 0
    assert [(s["side"], s["path"]) for s in report["skipped"]] == [
        ("head", "library/kernels/bad.py")
    ]
    assert kinds(report) == ["NEW COPY"]


@pytest.mark.parametrize(
    "groups",
    (
        json.dumps({"groups": [{k: v for k, v in GROUP.items() if k != "reason"}]}),
        json.dumps({"groups": [{**GROUP, "files": GROUP["files"][:1]}]}),
        json.dumps({"groups": [{**GROUP, "files": GROUP["files"][:1] * 2}]}),
        json.dumps({"groups": [GROUP, GROUP]}),
        json.dumps({"groups": ["library/kernels/gfx942/k.py"]}),
        json.dumps([GROUP]),
        '{"groups": [',
        json.dumps(
            {"groups": [{**GROUP, "files": ["library/a.py", "library/./a.py"]}]}
        ),
        json.dumps(
            {"groups": [{**GROUP, "files": ["/library/a.py", "/library/b.py"]}]}
        ),
        json.dumps(
            {
                "groups": [
                    {**GROUP, "files": ["library/tests/a.py", "library/tests/b.py"]}
                ]
            }
        ),
        json.dumps(
            {"groups": [{**GROUP, "files": ["library/../a.py", "library/b.py"]}]}
        ),
    ),
    ids=(
        "no-reason",
        "one-file",
        "same-file-twice",
        "name-twice",
        "entry-not-object",
        "no-groups-key",
        "bad-json",
        "same-file-spelled-twice",
        "absolute-paths",
        "out-of-scope-paths",
        "parent-directory",
    ),
)
def test_malformed_groups_file_is_a_tool_error(repo, groups):
    repo.write(GROUPS_PATH, groups)
    base = repo.commit()
    rc, _ = repo.check("--base", base)
    assert rc == 2


def backlog_groups(report: dict) -> list[tuple]:
    return [
        (g["kind"], g["copies"], sorted(loc["path"] for loc in g["locations"]))
        for g in report["groups"]
    ]


def test_backlog_lists_copies_that_already_exist_in_the_working_tree(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", func("f2"))
    repo.commit()
    rc, report = repo.check("--backlog")
    assert rc == 0
    assert report["head"] == "WORKTREE"
    assert backlog_groups(report) == [
        ("COPY GROUP", 2, ["library/kernels/a.py", "library/kernels/b.py"])
    ]
    assert report["groups"][0]["duplicated_lines"] > 0


def test_backlog_of_a_commit_ignores_the_working_tree(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", func("f2"))
    head = repo.commit()
    (repo.path / "library/kernels/b.py").unlink()
    _, at_commit = repo.check("--backlog", "--head", head)
    _, in_tree = repo.check("--backlog")
    assert len(at_commit["groups"]) == 1
    assert in_tree["groups"] == []


def test_backlog_puts_the_most_duplicated_lines_first(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", func("f2"))
    for name in "cde":
        repo.write(f"library/kernels/{name}.py", func("g", k="16.0"))
    repo.commit()
    _, report = repo.check("--backlog")
    assert [g["copies"] for g in report["groups"]] == [3, 2]
    sizes = [g["duplicated_lines"] for g in report["groups"]]
    assert sizes == sorted(sizes, reverse=True)


def test_backlog_labels_approved_forks_separately(repo):
    repo.write(GROUPS_PATH, GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", func("f"))
    repo.commit()
    _, report = repo.check("--backlog")
    assert [(g["kind"], g["group"]) for g in report["groups"]] == [
        ("APPROVED FORK", "per-arch")
    ]


@pytest.mark.parametrize("extra", [["--base", "HEAD"], ["--fail-on-new"]])
def test_backlog_rejects_options_that_compare_two_revisions(repo, extra):
    repo.commit()
    rc, _ = repo.check("--backlog", *extra)
    assert rc == 2


def test_backlog_text_lists_fifty_groups_then_counts_the_rest(repo):
    for i in range(52):
        repo.write(f"library/kernels/a{i}.py", func("f", k=f"{i}.0"))
        repo.write(f"library/kernels/b{i}.py", func("f", k=f"{i}.0"))
    repo.commit()
    text = subprocess.run(
        [sys.executable, str(TOOL), "--root", str(repo.path), "--backlog"],
        capture_output=True,
        text=True,
    ).stdout
    assert text.count("COPY GROUP ") == 50
    assert "+2 more copy groups" in text


# One statement of well over 50 tokens, written twice on ONE source line of ONE file.
STMT = (
    "v1 = b.add(base, b.mul(row, b.const_i32(7)), b.add(b.const_i32(1), b.const_i32(2)),"
    " b.add(b.const_i32(3), b.const_i32(4)), b.add(b.const_i32(5), b.const_i32(6)))"
)


def test_two_copies_on_one_line_of_one_file_stay_two_copies(repo):
    # Line ranges alone would join them into a single location (copies: 1).
    repo.write("library/kernels/a.py", f"def f(b, base, row):\n    {STMT}; {STMT}\n")
    repo.commit()
    _, report = repo.check("--backlog")
    assert report["groups"], "the backlog must list the repeated statement"
    assert {g["copies"] for g in report["groups"]} == {2}
    for g in report["groups"]:
        assert {loc["start"] for loc in g["locations"]} == {2}


def test_new_copy_on_the_same_line_as_an_old_one_is_reported(repo):
    repo.write("library/kernels/a.py", f"def f(b, base, row):\n    {STMT}\n")
    base = repo.commit()
    repo.write("library/kernels/a.py", f"def f(b, base, row):\n    {STMT}; {STMT}\n")
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"], "the pasted second statement is a new copy"
    for f in report["findings"]:
        assert len(f["locations"]) >= 2


def test_a_region_shared_by_fewer_files_is_a_separate_group(repo):
    # X is in a, b and c; the X+Y stretch only in a and b. The three-copy
    # group must cover X in all three files, not grow to X+Y in a and b only.
    y = BLOCK.replace("b.add(", "b.sub(")
    repo.write("library/kernels/a.py", func("fa", BLOCK + y))
    repo.write("library/kernels/b.py", func("fb", BLOCK + y))
    repo.write("library/kernels/c.py", func("fc", BLOCK))
    repo.commit()
    _, report = repo.check("--backlog")
    assert all(g["copies"] >= 2 for g in report["groups"])
    (triple,) = [g for g in report["groups"] if g["copies"] == 3]
    spans = {loc["end"] - loc["start"] for loc in triple["locations"]}
    assert len(spans) == 1
    assert any(g["copies"] == 2 for g in report["groups"])


def test_existing_copy_entries_with_different_copy_sets_stay_separate(repo):
    # R is in five files; the stretch R+Z only in four. Editing R in one of the
    # four touches both: a five-copy entry and a four-copy entry. The five-copy
    # entry must stay R-wide in every file, not grow to R+Z in four of them.
    z = BLOCK.replace("b.add(", "b.sub(")
    for name in "abcd":
        repo.write(f"library/kernels/{name}.py", func("f" + name, BLOCK + z))
    repo.write("library/kernels/e.py", func("fe", BLOCK))
    base = repo.commit()
    repo.write(
        "library/kernels/a.py",
        func("fa", BLOCK.replace("b.fmul", "b.fadd") + z),
    )
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    entries = report["existing_copies"]
    five = [e for e in entries if len(e["locations"]) == 5]
    four = [e for e in entries if len(e["locations"]) == 4]
    assert len(five) == 1 and four
    assert len({l["end"] - l["start"] for l in five[0]["locations"]}) == 1
    assert {l["path"][-4] for l in four[0]["locations"]} == set("abcd")


def test_same_inputs_give_identical_output(repo):
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("f2") + func("f3"))
    head = repo.commit()
    run = lambda: subprocess.run(  # noqa: E731
        [
            sys.executable,
            str(TOOL),
            "--root",
            str(repo.path),
            "--format",
            "json",
            "--base",
            base,
            "--head",
            head,
        ],
        capture_output=True,
    ).stdout
    first = run()
    assert first and first == run()


def test_overlapping_windows_in_one_run_are_not_a_copy(repo):
    base = repo.commit()
    repeated = "".join("    acc = b.fadd(acc, b.fmul(x, y))\n" for _ in range(6))
    repo.write("library/kernels/a.py", "def f(b, acc, x, y):\n" + repeated)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


DOC_BLOCK = '''\
    v1 = b.add(base, row)
    v2 = b.select(ok, v1, zero)

    def inner():
        """Line one.

        Line two.
        """
        return v2

    v3 = b.load(ptr, inner())
    b.store(out, v2, v3)
'''


def test_copy_at_a_deeper_nesting_level_is_still_a_copy(repo):
    # Re-indenting the copy also re-indents the nested docstring's text.
    repo.write("library/kernels/a.py", func("f1", DOC_BLOCK))
    base = repo.commit()
    deeper = "    if ok:\n" + "".join(
        "    " + line if line.strip() else line
        for line in DOC_BLOCK.splitlines(keepends=True)
    )
    repo.write("library/kernels/b.py", func("f2", deeper))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


STRING_BLOCK = """\
    v0 = {text}
    v1 = b.add(base, row)
    v2 = b.select(ok, v1, zero)
    v3 = b.load(ptr, v2)
    b.store(out, v0, v3)
"""


def test_string_values_that_differ_only_in_indentation_are_not_a_copy(repo):
    # Only docstrings ignore indentation; a string value keeps its exact text.
    repo.write(
        "library/kernels/a.py", func("f1", STRING_BLOCK.replace("{text}", '"a\\n  b"'))
    )
    base = repo.commit()
    repo.write(
        "library/kernels/b.py", func("f2", STRING_BLOCK.replace("{text}", '"a\\nb"'))
    )
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_two_separated_copies_in_one_block_are_a_copy(repo):
    base = repo.commit()
    body = BLOCK + "    marker = b.barrier()\n" + BLOCK
    repo.write("library/kernels/a.py", func("f", body))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


def run_tool(repo: Repo, *args: str, env: dict | None = None):
    return subprocess.run(
        [sys.executable, str(TOOL), "--root", str(repo.path), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_missing_upstream_branch_is_a_tool_error_that_names_the_fix(repo):
    repo.write("library/kernels/a.py", func("f"))
    repo.commit()  # the repo has no origin/develop, so there is no default base
    proc = run_tool(repo)
    assert proc.returncode == 2
    assert "--base" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_missing_git_binary_is_a_tool_error(repo, tmp_path):
    repo.write("library/kernels/a.py", func("f"))
    base = repo.commit()
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    proc = run_tool(repo, "--base", base, env={**os.environ, "PATH": str(empty)})
    assert proc.returncode == 2
    assert "git was not found" in proc.stderr
    assert "Traceback" not in proc.stderr


def stdout_of(repo: Repo, *args: str) -> str:
    proc = run_tool(repo, *args)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def commit_sibling_copy(repo: Repo) -> tuple[str, str]:
    """Base and head of a copy pasted between two approved fork files."""
    repo.write(GROUPS_PATH, GROUPS)
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", "Z = 0\n")
    base = repo.commit()
    repo.write("library/kernels/gfx950/k.py", func("f"))
    return base, repo.commit()


def commit_likely_copy(repo: Repo) -> tuple[str, str]:
    """Base and head of a pasted block that replaces unrelated code."""
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", func("g", UNRELATED))
    base = repo.commit()
    repo.write("library/kernels/b.py", func("g"))
    return base, repo.commit()


def test_text_report_names_a_new_copy_and_the_fix(repo):
    base = repo.commit()
    repo.write("library/kernels/a.py", func("f1") + func("f2"))
    head = repo.commit()
    out = stdout_of(repo, "--base", base, "--head", head)
    assert "NEW COPY" in out
    assert "Fix: extract one helper" in out


def test_text_report_names_a_likely_copy_and_the_fix(repo):
    base, head = commit_likely_copy(repo)
    out = stdout_of(repo, "--base", base, "--head", head)
    assert "LIKELY COPY" in out
    assert "Fix: call the existing code" in out


def test_text_report_names_a_sibling_copy_without_asking_for_action(repo):
    base, head = commit_sibling_copy(repo)
    out = stdout_of(repo, "--base", base, "--head", head)
    assert "SIBLING COPY" in out
    assert "no action needed" in out


def test_text_report_lists_skipped_files(repo):
    base = repo.commit()
    repo.write("library/kernels/bad.py", "def broken(:\n")
    head = repo.commit()
    out = stdout_of(repo, "--base", base, "--head", head)
    assert "SKIPPED (head)" in out
    assert "library/kernels/bad.py" in out


def test_fail_on_new_changes_only_the_exit_code_for_a_likely_copy(repo):
    base, head = commit_likely_copy(repo)
    plain = run_tool(repo, "--base", base, "--head", head)
    flagged = run_tool(repo, "--base", base, "--head", head, "--fail-on-new")
    assert (plain.returncode, flagged.returncode) == (0, 1)
    assert plain.stdout == flagged.stdout


def test_fail_on_new_passes_for_a_sibling_copy(repo):
    base, head = commit_sibling_copy(repo)
    rc, report = repo.check("--base", base, "--head", head, "--fail-on-new")
    assert kinds(report) == ["SIBLING COPY"]
    assert rc == 0


def test_fail_on_new_passes_when_only_an_existing_copy_was_edited(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", func("f2"))
    base = repo.commit()
    repo.write("library/kernels/a.py", func("f1", k="64.0"))
    head = repo.commit()
    rc, report = repo.check("--base", base, "--head", head, "--fail-on-new")
    assert report["findings"] == []
    assert existing(report)  # the edit really is reported, just not as a failure
    assert rc == 0


def test_files_under_platform_python_rocke_are_scanned(repo):
    base = repo.commit()
    repo.write("platform/python/rocke/helpers/a.py", func("f1") + func("f2"))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


@pytest.mark.parametrize("head_mode", (False, True), ids=("worktree", "commit"))
def test_a_rocke_root_below_the_git_toplevel_is_analysed(repo, head_mode):
    # In the real checkout rocKE sits in a subdirectory of the repository.
    nested = "dnn/rocke/library/kernels/a.py"
    repo.write(nested, func("f1"))
    base = repo.commit()
    repo.write(nested, func("f1") + func("f2"))
    args = ["--base", base] + (["--head", repo.commit()] if head_mode else [])
    _, report = repo.check(*args, root=repo.path / "dnn" / "rocke")
    assert kinds(report) == ["NEW COPY"]


GIT_SETTINGS = {
    "noprefix": {"diff.noprefix": "true"},
    "mnemonicPrefix": {"diff.mnemonicPrefix": "true"},
    "interHunkContext": {"diff.interHunkContext": "20"},
    "renames-off": {"diff.renames": "false"},
    "color": {"color.ui": "always"},
    "quotePath": {"core.quotePath": "true"},
    "autocrlf": {"core.autocrlf": "true"},
    "external-diff": {},  # not a config key: set through the environment below
    "diff-opts": {},  # likewise (GIT_DIFF_OPTS)
}
SETTING_SCENARIOS = {
    # f1 is untouched; the edit above it and the copy below it are 8 lines apart,
    # so a large diff.interHunkContext would fuse them into one hunk.
    "copy-below-an-edit": (
        {"library/kernels/a.py": "HEADER = 1\n\n" + func("f1")},
        {"library/kernels/a.py": "HEADER = 2\n\n" + func("f1") + func("f2")},
    ),
    "both-copies-edited-alike": (
        {"library/kernels/a.py": func("f1"), "library/kernels/b.py": func("f2")},
        {
            "library/kernels/a.py": func("f1", k="64.0"),
            "library/kernels/b.py": func("f2", k="64.0"),
        },
    ),
    # A renamed and edited copy: only rename detection can tell it from a new file.
    "renamed-and-edited-copy": (
        {"library/kernels/a.py": func("f1"), "library/kernels/b.py": func("f2")},
        {
            "library/kernels/a.py": None,
            "library/kernels/c.py": func("f1", k="64.0"),
        },
    ),
}


@pytest.mark.parametrize("scenario", sorted(SETTING_SCENARIOS))
@pytest.mark.parametrize("head_mode", (False, True), ids=("worktree", "commit"))
@pytest.mark.parametrize("setting", sorted(GIT_SETTINGS))
def test_user_git_settings_do_not_change_the_result(
    repo, monkeypatch, tmp_path_factory, scenario, head_mode, setting
):
    before, after = SETTING_SCENARIOS[scenario]
    for path, text in before.items():
        repo.write(path, text)
    base = repo.commit()
    for path, text in after.items():
        if text is None:
            (repo.path / path).unlink()
        else:
            repo.write(path, text)
    if not head_mode:
        repo.git("add", "-A")  # staged, so git diff sees the rename (and the add)
    args = ["--base", base] + (["--head", repo.commit()] if head_mode else [])
    expected = repo.check(*args)
    assert expected[1]["findings"] or expected[1]["existing_copies"]
    set_git_config(monkeypatch, tmp_path_factory, GIT_SETTINGS[setting])
    if setting == "external-diff":
        if os.name != "posix":
            pytest.skip("the external diff program is /bin/true")
        monkeypatch.setenv("GIT_EXTERNAL_DIFF", "/bin/true")
    if setting == "diff-opts":
        monkeypatch.setenv("GIT_DIFF_OPTS", "-u5")
    assert repo.check(*args) == expected


@pytest.mark.parametrize("flag", ("--base", "--head", "--root"))
def test_an_empty_revision_or_root_is_rejected_not_read_as_the_default(repo, flag):
    repo.commit()
    proc = run_tool(repo, flag, "")
    assert proc.returncode == 2
    assert "must not be empty" in proc.stderr


@pytest.mark.parametrize("extra", ((), ("--backlog",)), ids=("compare", "backlog"))
def test_a_root_without_rocke_sources_is_a_tool_error(tmp_path, extra):
    other = tmp_path / "other"
    other.mkdir()
    repo = Repo(other)
    repo.write("docs/x.py", "X = 1\n")
    base = repo.commit()
    proc = run_tool(repo, *(extra or ("--base", base)))
    assert proc.returncode == 2
    assert "rocKE root" in proc.stderr


def test_a_missing_root_is_a_tool_error(repo, tmp_path):
    proc = run_tool(repo, "--root", str(tmp_path / "missing"), "--base", "HEAD")
    assert proc.returncode == 2
    assert "not a directory" in proc.stderr


def test_a_groups_file_that_is_not_utf8_is_a_tool_error(repo):
    target = repo.path / GROUPS_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\xff\xfe{")
    base = repo.commit()
    assert repo.check("--base", base)[0] == 2  # read from the working tree
    assert repo.check("--base", base, "--head", base)[0] == 2  # read from a commit


def test_approved_forks_come_from_the_compared_head_not_the_working_tree(repo):
    base, head = commit_sibling_copy(repo)
    (repo.path / GROUPS_PATH).unlink()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["SIBLING COPY"]


FSTRING_BODY = """\
    v1 = b.log(f"row {row} of {base}")
    v2 = b.select(ok, v1, zero)
    b.store(out, v1, row)
"""


def test_an_f_string_counts_as_one_token_on_every_python(repo):
    # 16 + 8 + 12 + 10 = 46 code tokens, below the 50-token window. Tokenized
    # into pieces (Python 3.12+) the same code would have 55 and be reported.
    repo.write("library/kernels/a.py", short_func("f", FSTRING_BODY))
    base = repo.commit()
    repo.write("library/kernels/b.py", short_func("f", FSTRING_BODY))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert report["findings"] == []


def test_non_ascii_file_names_are_analysed(repo):
    base = repo.commit()
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/café.py", func("f2"))
    _, report = repo.check("--base", base)  # untracked files in the working tree
    assert kinds(report) == ["NEW COPY"]
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    paths = [loc["path"] for loc in report["findings"][0]["locations"]]
    assert "library/kernels/café.py" in paths


def test_the_default_base_is_the_merge_base_with_origin_develop(repo):
    first = repo.commit()
    repo.git("update-ref", "refs/remotes/origin/develop", first)
    repo.write("library/kernels/a.py", func("f1"))
    repo.commit()  # HEAD is one commit past origin/develop
    repo.write("library/kernels/a.py", func("f1") + func("f2"))
    _, report = repo.check()
    (finding,) = report["findings"]
    assert (finding["copies_before"], finding["copies_after"]) == (0, 2)


def test_a_head_commit_is_analysed_not_the_working_tree(repo):
    base = repo.commit()
    repo.write("library/kernels/a.py", func("f1") + func("f2"))
    head = repo.commit()
    repo.write("library/kernels/a.py", func("f1"))  # uncommitted: the copy is gone
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


def test_reported_spans_follow_the_lines_added_above_them(repo):
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    text = "A = 1\nB = 2\nC = 3\n" + func("f1") + func("f2")
    repo.write("library/kernels/a.py", text)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    spans = sorted((l["start"], l["end"]) for l in report["findings"][0]["locations"])
    lines = text.splitlines()
    first, second = (
        lines.index(f"def {n}(b, base, row, stride, ok, zero, ptr, out):") + 1
        for n in ("f1", "f2")
    )
    assert [s for s, _ in spans] == [first, second]
    assert spans[0][1] - spans[0][0] == spans[1][1] - spans[1][0]


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs POSIX file modes and a non-root user (root reads any file)",
)
def test_an_unreadable_source_file_is_a_tool_error_not_a_traceback(repo):
    base = repo.commit()
    target = repo.path / "library" / "kernels" / "a.py"
    target.write_text(func("f"))
    target.chmod(0)
    proc = run_tool(repo, "--base", base)
    target.chmod(0o644)
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


ODD_NAMES = (
    # a space: the path of a diff header cannot be split on it
    pytest.param("library/k b/a.py", id="space"),
    # names git quotes in its output; not valid file names on Windows
    pytest.param('library/qu"ote.py', id="quote", marks=POSIX_ONLY),
    pytest.param("library/we\\ird.py", id="backslash", marks=POSIX_ONLY),
    pytest.param("library/ta\tb.py", id="tab", marks=POSIX_ONLY),
)


@pytest.mark.parametrize("name", ODD_NAMES)
@pytest.mark.parametrize("head_mode", (False, True), ids=("worktree", "commit"))
def test_an_edited_file_with_an_unusual_name_is_analysed(repo, name, head_mode):
    repo.write(name, func("f1"))
    base = repo.commit()
    repo.write(name, func("f1") + func("f2"))
    args = ["--base", base] + (["--head", repo.commit()] if head_mode else [])
    rc, report = repo.check(*args)
    assert (rc, kinds(report)) == (0, ["NEW COPY"])


@pytest.mark.parametrize("name", ODD_NAMES)
def test_a_copy_pasted_into_a_file_with_an_unusual_name_is_found(repo, name):
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    repo.write(name, func("f2"))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    paths = [loc["path"] for loc in report["findings"][0]["locations"]]
    assert sorted(paths) == sorted([name, "library/kernels/a.py"])


@POSIX_ONLY
def test_a_file_name_that_is_not_utf8_is_ignored_unless_it_is_python(repo):
    kernels = os.fsencode(repo.path / "library" / "kernels")
    with open(kernels + b"/caf\xe9.txt", "wb") as handle:
        handle.write(b"x")
    base = repo.commit()
    repo.write("library/kernels/a.py", func("f1") + func("f2"))
    head = repo.commit()
    for args in (["--base", base], ["--base", base, "--head", head]):
        rc, report = repo.check(*args)
        assert (rc, kinds(report)) == (0, ["NEW COPY"])


@POSIX_ONLY
def test_a_python_file_name_with_a_newline_is_a_tool_error(repo):
    base = repo.commit()
    repo.write("library/kernels/a\nb.py", func("f1"))
    proc = run_tool(repo, "--base", base)
    assert proc.returncode == 2
    assert "newline" in proc.stderr


@pytest.mark.parametrize(
    "tail, is_copy",
    (("    return v2\n", True), ("    return\n", False)),
    ids=("50-tokens", "49-tokens"),
)
def test_a_copy_must_be_at_least_fifty_code_tokens(repo, tail, is_copy):
    # The header has 16 code tokens and SHORT 32; `return v2` adds 2, `return` 1.
    base = repo.commit()
    repo.write("library/kernels/a.py", short_func("f", SHORT + tail))
    repo.write("library/kernels/b.py", short_func("f", SHORT + tail))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == (["NEW COPY"] if is_copy else [])


def test_comments_do_not_hide_a_copy(repo):
    base = repo.commit()
    commented = func("f").replace("    v2 =", "    # explain this\n    v2 =")
    commented = commented.replace("zero)\n", "zero)  # trailing note\n")
    repo.write("library/kernels/a.py", func("f"))
    repo.write("library/kernels/b.py", commented)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


@pytest.mark.parametrize("head_mode", (False, True), ids=("worktree", "commit"))
def test_a_file_moved_into_scope_from_an_excluded_path_counts_as_added(repo, head_mode):
    repo.write("library/kernels/a.py", func("f"))
    repo.write("library/tests/helper.py", func("f"))  # tests are out of scope
    base = repo.commit()
    repo.git("mv", "library/tests/helper.py", "library/kernels/b.py")
    args = ["--base", base] + (["--head", repo.commit()] if head_mode else [])
    _, report = repo.check(*args)
    assert kinds(report) == ["NEW COPY"]


def test_a_file_touched_without_changes_is_not_a_tool_error(
    repo, monkeypatch, tmp_path_factory
):
    # A stat-dirty file that git does not refresh is listed by --name-status
    # but has no patch section, unless diff.autoRefreshIndex is pinned.
    set_git_config(monkeypatch, tmp_path_factory, {"diff.autoRefreshIndex": "false"})
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    (repo.path / "library/kernels/a.py").touch()
    os.utime(repo.path / "library/kernels/a.py", (1, 1))
    repo.write("library/kernels/c.py", func("f2"))
    rc, report = repo.check("--base", base)
    assert rc == 0
    assert kinds(report) == ["NEW COPY"]


@POSIX_ONLY
@pytest.mark.parametrize("head_mode", (False, True), ids=("worktree", "commit"))
def test_a_file_that_becomes_a_symlink_is_not_a_tool_error(repo, head_mode):
    repo.write("library/kernels/a.py", func("f"))
    base = repo.commit()
    (repo.path / "library/kernels/a.py").unlink()
    (repo.path / "library/kernels/a.py").symlink_to("keep.py")
    args = ["--base", base] + (["--head", repo.commit()] if head_mode else [])
    rc, _ = repo.check(*args)
    assert rc == 0


@pytest.mark.parametrize("rule", ("*.py -diff", "*.py binary"))
@pytest.mark.parametrize("head_mode", (False, True), ids=("worktree", "commit"))
def test_gitattributes_cannot_turn_python_sources_into_binary_files(
    repo, rule, head_mode
):
    repo.write(".gitattributes", rule + "\n")
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    repo.write("library/kernels/a.py", func("f1") + func("f2"))
    args = ["--base", base] + (["--head", repo.commit()] if head_mode else [])
    _, report = repo.check(*args)
    assert kinds(report) == ["NEW COPY"]


def test_a_head_commit_is_used_for_edited_files_too(repo):
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    repo.write("library/kernels/a.py", func("f1") + func("f2"))
    head = repo.commit()
    repo.write("library/kernels/a.py", func("f1"))  # the working tree is back at base
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


@POSIX_ONLY
def test_a_symlink_is_read_the_same_in_the_working_tree_and_in_a_commit(repo):
    repo.write("library/kernels/a.py", func("f"))
    base = repo.commit()
    (repo.path / "library/kernels/b.py").symlink_to("a.py")
    in_tree = repo.check("--base", base)
    head = repo.commit()
    committed = repo.check("--base", base, "--head", head)[1]
    for key in ("findings", "existing_copies", "skipped"):
        assert committed[key] == in_tree[1][key]
    assert in_tree[1]["findings"] == []  # a link is not a copy of its target


NON_ASCII_COPY = """\
    {p}1 = b.add(base, row)
    {p}2 = b.select(ok, b.const_str("é"), {p}1)
    {p}3 = b.select(ok, {p}2, zero)
    {p}4 = b.fmul({p}3, {p}1)
    b.store(out, {p}3, {p}4)
"""


def test_renamed_locals_match_after_a_non_ascii_character_on_the_line(repo):
    base = repo.commit()
    repo.write("library/kernels/a.py", func("f", NON_ASCII_COPY.replace("{p}", "v")))
    repo.write("library/kernels/b.py", func("f", NON_ASCII_COPY.replace("{p}", "u")))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert kinds(report) == ["NEW COPY"]


def test_form_feed_characters_in_source_lines_do_not_create_hunks(repo):
    # str.splitlines() would split `# x\x0c@@ -4 +4 @@` and read a hunk header
    # that claims line 4, inside the first copy, was edited.
    funcs = func("f1") + func("f2")
    repo.write("library/kernels/a.py", "# x\x0c@@ -4 +4 @@\n" + funcs)
    base = repo.commit()
    repo.write("library/kernels/a.py", "# y\x0c@@ -4 +4 @@\n" + funcs)
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert (report["findings"], report["existing_copies"]) == ([], [])


def test_ignored_files_are_not_scanned_and_untracked_files_are_new(repo):
    repo.write(".gitignore", "library/kernels/ignored.py\n")
    repo.write("library/kernels/a.py", func("f1"))
    base = repo.commit()
    repo.write("library/kernels/ignored.py", func("f2"))
    assert kinds(repo.check("--base", base)[1]) == []
    repo.write("library/kernels/b.py", func("f3"))  # untracked, not ignored
    (finding,) = repo.check("--base", base)[1]["findings"]
    classes = {loc["path"]: loc["class"] for loc in finding["locations"]}
    assert classes == {
        "library/kernels/a.py": "unchanged",
        "library/kernels/b.py": "new",
    }


@POSIX_ONLY
def test_a_python_file_name_that_is_not_utf8_is_a_tool_error(repo):
    base = repo.commit()
    kernels = os.fsencode(repo.path / "library" / "kernels")
    with open(kernels + b"/caf\xe9.py", "w") as handle:
        handle.write("X = 1\n")
    proc = run_tool(repo, "--base", base)
    assert proc.returncode == 2
    assert "UTF-8" in proc.stderr


def test_a_file_moved_out_of_scope_is_not_reported_under_its_new_path(repo):
    repo.write("library/kernels/a.py", func("f1"))
    repo.write("library/kernels/b.py", func("f2"))
    base = repo.commit()
    (repo.path / "library" / "tests").mkdir()
    repo.git("mv", "library/kernels/a.py", "library/tests/a.py")
    repo.write("library/tests/a.py", func("f1", k="64.0"))
    head = repo.commit()
    _, report = repo.check("--base", base, "--head", head)
    assert "library/tests" not in json.dumps(report)


def test_a_file_too_deeply_nested_to_parse_is_skipped_not_a_crash(repo):
    branches = "".join(f"    elif x == {i}:\n        return {i}\n" for i in range(3000))
    base = repo.commit()
    repo.write(
        "library/kernels/deep.py",
        "def f(x):\n    if x < 0:\n        return 0\n" + branches,
    )
    proc = run_tool(repo, "--base", base)
    assert proc.returncode == 0
    assert "Traceback" not in proc.stderr


def test_the_text_report_survives_a_console_that_cannot_show_its_symbols(repo):
    base = repo.commit()
    repo.write("library/kernels/a.py", func("f1") + func("f2"))
    head = repo.commit()
    env = {**os.environ, "PYTHONIOENCODING": "ascii"}
    proc = run_tool(repo, "--base", base, "--head", head, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "NEW COPY" in proc.stdout


MULTILINE_STRING = '''\
    v1 = b.add(base, b.mul(row, stride))
    v2 = b.log("""first line
second line""", v1)
    v3 = b.select(ok, v2, zero)
    v4 = b.fmul(v3, v1)
    b.store(out, v3, v4)
'''


def test_a_crlf_working_tree_file_matches_the_lf_blobs_of_its_copies(repo):
    # Windows checkouts with core.autocrlf=true hold CRLF files; the base side
    # always comes from LF blobs. A multi-line string keeps its line ending.
    block = func("f", MULTILINE_STRING)
    repo.write("library/kernels/a.py", block)
    repo.write("library/kernels/b.py", block)
    base = repo.commit()
    crlf = repo.path / "library" / "kernels" / "c.py"
    crlf.write_bytes(block.replace("\n", "\r\n").encode())
    (finding,) = repo.check("--base", base)[1]["findings"]
    assert (finding["copies_before"], finding["copies_after"]) == (2, 3)


@pytest.mark.parametrize("autocrlf", (None, "true", "input"))
def test_a_tracked_file_saved_with_crlf_still_reports_the_pasted_copy(
    repo, monkeypatch, tmp_path_factory, autocrlf
):
    # The blob is LF and the file on disk is CRLF. With core.autocrlf unset git
    # reports every line as changed, so the copy is labelled LIKELY, not NEW; it
    # must be reported either way.
    block = func("f", MULTILINE_STRING)
    repo.write("library/kernels/a.py", block)
    repo.write("library/kernels/b.py", block)
    base = repo.commit()
    crlf = (block + block).replace("\n", "\r\n").encode()
    (repo.path / "library" / "kernels" / "a.py").write_bytes(crlf)
    if autocrlf is not None:
        set_git_config(monkeypatch, tmp_path_factory, {"core.autocrlf": autocrlf})
    rc, report = repo.check("--base", base, "--fail-on-new")
    (finding,) = report["findings"]
    assert finding["kind"] in ("NEW COPY", "LIKELY COPY")
    assert (finding["copies_before"], finding["copies_after"]) == (2, 3)
    assert rc == 1


def test_a_file_changed_only_in_its_line_endings_does_not_break_the_run(repo):
    # git versions differ in whether they list such a file; a pasted copy in
    # another file must still be found without an error.
    block = func("f", MULTILINE_STRING)
    repo.write("library/kernels/a.py", block)
    base = repo.commit()
    (repo.path / "library" / "kernels" / "a.py").write_bytes(
        block.replace("\n", "\r\n").encode()
    )
    repo.write("library/kernels/b.py", block)
    rc, report = repo.check("--base", base)
    assert rc == 0
    assert kinds(report) == ["NEW COPY"]


def test_a_groups_file_with_a_utf8_bom_is_read(repo):
    target = repo.path / GROUPS_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\xef\xbb\xbf" + GROUPS.encode())
    repo.write("library/kernels/gfx942/k.py", func("f"))
    repo.write("library/kernels/gfx950/k.py", "Z = 0\n")
    base = repo.commit()
    repo.write("library/kernels/gfx950/k.py", func("f"))
    head = repo.commit()
    for args in (["--base", base], ["--base", base, "--head", head]):
        assert kinds(repo.check(*args)[1]) == ["SIBLING COPY"]


def test_a_committed_crlf_blob_matches_lf_blobs_in_head_mode(repo):
    block = func("f", MULTILINE_STRING)
    repo.write("library/kernels/a.py", block)
    repo.write("library/kernels/b.py", block)
    base = repo.commit()
    crlf = repo.path / "library" / "kernels" / "c.py"
    crlf.write_bytes(block.replace("\n", "\r\n").encode())  # git stores it as CRLF
    head = repo.commit()
    (finding,) = repo.check("--base", base, "--head", head)[1]["findings"]
    assert (finding["copies_before"], finding["copies_after"]) == (2, 3)
