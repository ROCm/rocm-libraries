# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the hipdnn-deep-review pin.py helper, using a local bare repo as origin."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "skills"
    / "hipdnn-deep-review"
    / "scripts"
    / "pin.py"
)


needs_posix_bash = pytest.mark.skipif(
    sys.platform == "win32",
    reason="a bare `bash` on Windows is WSL's, which cannot read C:\\ paths; env.sh is POSIX-shell only",
)


@pytest.fixture
def pin():
    spec = importlib.util.spec_from_file_location("hipdnn_deep_review_pin", SCRIPT)
    assert spec and spec.loader, f"could not load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(*cmd, cwd=None, stdin=None):
    r = subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@pytest.fixture
def origin(tmp_path):
    """Bare origin with develop and feature (edits under hipdnn, tools, docs)."""
    work, bare = tmp_path / "work", tmp_path / "origin.git"
    run("git", "init", "-q", "-b", "develop", str(work))
    run("git", "init", "-q", "--bare", str(bare))
    g = ["git", "-C", str(work), "-c", "user.name=t", "-c", "user.email=t@t"]

    def put(rel, text):
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    put("README.md", "root\n")
    put("projects/hipdnn/src/a.cpp", "a\n")
    put("projects/hipdnn/tools/x.sh", "x\n")
    put("dnn-providers/p/b.cpp", "b\n")
    put("docs/guide.md", "g\n")
    run(*g, "add", "-A")
    run(*g, "commit", "-qm", "base")
    base = run(*g, "rev-parse", "HEAD")
    run(*g, "checkout", "-qb", "feat/one")
    put("projects/hipdnn/src/a.cpp", "a\nb\n")
    put("projects/hipdnn/tools/x.sh", "x\ny\n")
    put("docs/guide.md", "g\nh\n")
    put("README.md", "root2\n")
    run(*g, "add", "-A")
    run(*g, "commit", "-qm", "change")
    head = run(*g, "rev-parse", "HEAD")
    run(*g, "push", "-q", str(bare), "develop", "feat/one", f"{head}:refs/pull/7/head")
    return {"url": str(bare), "base": base, "head": head, "tmp": tmp_path}


def init_args(origin, *target, n=1):
    t = origin["tmp"]
    return [
        "init",
        "--evidence",
        str(t / "ev"),
        "--scratch",
        str(t / "scratch"),
        "--round",
        str(n),
        "--owner-repo",
        "o/r",
        "--remote-url",
        origin["url"],
        *target,
    ]


@pytest.mark.parametrize(
    "text,key",
    [
        ("develop..users/me/feat", "develop-users-me-feat"),
        ("feat/one", "feat-one"),
        ("v1.2", "v1.2"),
        (".hidden", "hidden"),
        ("x.lock", "x"),
        ("a b?c", "a-b-c"),
    ],
)
def test_key_derivation(pin, capsys, text, key):
    pin.main(["key", text])
    assert capsys.readouterr().out.strip() == key


def test_key_rejects_unusable(pin):
    with pytest.raises(SystemExit) as e:
        pin.main(["key", "."])
    assert "unusable review key" in str(e.value)


def test_init_branch(pin, origin):
    pin.main(init_args(origin, "--branch", "feat/one"))
    t = origin["tmp"]
    r1 = t / "ev" / "r1"
    pin_md = (r1 / "PIN.md").read_text()
    assert f"HEAD: {origin['head']}\n" in pin_md
    assert f"BASE: {origin['base']}\n" in pin_md
    assert "KEY: feat-one\n" in pin_md and "TARGET: develop\n" in pin_md
    assert sorted((r1 / "files.txt").read_text().split()) == [
        "README.md",
        "docs/guide.md",
        "projects/hipdnn/src/a.cpp",
        "projects/hipdnn/tools/x.sh",
    ]
    assert (r1 / "sparse.txt").read_text().split() == [
        "docs",
        "projects/hipdnn",
        "tools",
    ]
    assert "1\t0\tprojects/hipdnn/src/a.cpp" in (r1 / "numstat.txt").read_text()
    assert (r1 / "commits.txt").read_text().strip() == f"{origin['head']} change"
    head_wt, base_wt = t / "scratch/r1/head", t / "scratch/r1/base"
    assert (head_wt / "projects/hipdnn/src/a.cpp").read_text() == "a\nb\n"
    assert (base_wt / "projects/hipdnn/src/a.cpp").read_text() == "a\n"
    assert not (head_wt / "dnn-providers").exists()


def test_init_range_with_sha_side(pin, origin):
    pin.main(init_args(origin, "--range", f"{origin['base']}..feat/one"))
    pin_md = (origin["tmp"] / "ev/r1/PIN.md").read_text()
    assert (
        f"BASE: {origin['base']}\n" in pin_md and f"HEAD: {origin['head']}\n" in pin_md
    )
    assert f"TARGET: {origin['base']}\n" in pin_md
    assert "KEY: " + pin.derive_key(f"{origin['base']}..feat/one") + "\n" in pin_md


def test_init_range_branches(pin, origin):
    pin.main(init_args(origin, "--range", "develop..feat/one"))
    pin_md = (origin["tmp"] / "ev/r1/PIN.md").read_text()
    assert "KEY: develop-feat-one\n" in pin_md
    assert f"BASE: {origin['base']}\n" in pin_md


@needs_posix_bash
def test_env_sh_fresh_and_persistent_shell(pin, origin):
    pin.main(init_args(origin, "--branch", "feat/one"))
    env = origin["tmp"] / "ev/r1/env.sh"
    out = run("bash", "-c", f'. "{env}"; echo "$KEY $HEAD_SHORT $BASE"; pwd')
    key, short, base, cwd = out.split()
    assert (key, short, base) == ("feat-one", origin["head"][:12], origin["base"])
    assert cwd == str(env.parent.resolve())
    # A persistent shell sources it twice from different directories; quotes in values survive.
    pin_md = env.parent / "PIN.md"
    pin_md.write_text(pin_md.read_text() + "KNOWLEDGE: /tmp/it's here\n")
    script = (
        f'cd /\n. "{env}"\necho "$KNOWLEDGE"\ncd /tmp\n. "{env}"\necho "$KEY"\npwd\n'
    )
    out = run("bash", stdin=script).splitlines()
    assert out == ["/tmp/it's here", "feat-one", str(env.parent.resolve())]


@needs_posix_bash
def test_env_sh_stops_when_unset(pin, origin, tmp_path):
    pin.main(init_args(origin, "--branch", "feat/one"))
    env = origin["tmp"] / "ev/r1/env.sh"
    pin_md = env.parent / "PIN.md"
    pin_md.write_text(
        "".join(
            l for l in pin_md.read_text().splitlines(True) if not l.startswith("KEY:")
        )
    )
    r = subprocess.run(
        ["bash", "-c", f'. "{env}"; echo reached'], capture_output=True, text=True
    )
    assert r.returncode != 0 and "reached" not in r.stdout


def test_cleanup_removes_worktrees_and_refs(pin, origin):
    pin.main(init_args(origin, "--branch", "feat/one"))
    t = origin["tmp"]
    clone = t / "scratch/clone"
    assert run("git", "-C", str(clone), "for-each-ref", "refs/review/feat-one/")
    run(
        "git",
        "-C",
        str(clone),
        "update-ref",
        "refs/review/other/target",
        origin["base"],
    )
    pin.main(["cleanup", "--evidence", str(t / "ev"), "--round", "1"])
    assert not (t / "scratch/r1/head").exists() and not (t / "scratch/r1/base").exists()
    assert run("git", "-C", str(clone), "for-each-ref", "refs/review/feat-one/") == ""
    assert run("git", "-C", str(clone), "for-each-ref", "refs/review/other/")
    assert clone.is_dir()
    wts = run("git", "-C", str(clone), "worktree", "list", "--porcelain")
    assert wts.count("worktree ") == 1


class FakeGh:
    """Stub for pin.gh serving a PR whose file list and head are configurable."""

    def __init__(self, head, files, head_oid=None, base="develop"):
        self.head, self.files = head, files
        self.head_oid = head_oid or head
        self.base = base

    def __call__(self, *args):
        if "baseRefName" in args and "--jq" in args:
            return self.base + "\n"
        if args[:2] == ("pr", "view"):
            return json.dumps(
                {"title": "t", "body": "desc", "headRefOid": self.head_oid}
            )
        if args[0] == "api" and args[-1].endswith("/files"):
            half = len(self.files) // 2
            return "".join(
                json.dumps([{"filename": f} for f in part])
                for part in (self.files[:half], self.files[half:])
            )
        if args[0] == "api" and "--jq" in args:
            return "ci\tsuccess\thttp://x\n"
        return "[]"


PR_FILES = [
    "README.md",
    "docs/guide.md",
    "projects/hipdnn/src/a.cpp",
    "projects/hipdnn/tools/x.sh",
]


def test_pr_saves_artifacts(pin, origin, monkeypatch):
    monkeypatch.setattr(pin, "gh", FakeGh(origin["head"], PR_FILES))
    pin.main(init_args(origin, "--pr", "7"))
    r1 = origin["tmp"] / "ev/r1"
    assert "KEY: pr7\n" in (r1 / "PIN.md").read_text()
    assert (r1 / "pr/body.md").read_text() == "desc"
    assert [
        f["filename"] for f in json.loads((r1 / "pr/files.json").read_text())
    ] == PR_FILES
    assert (r1 / "pr/checks.tsv").read_text() == "ci\tsuccess\thttp://x\n"


def test_pr_head_moved(pin, origin, monkeypatch):
    monkeypatch.setattr(pin, "gh", FakeGh(origin["head"], PR_FILES, head_oid="0" * 40))
    with pytest.raises(SystemExit) as e:
        pin.main(init_args(origin, "--pr", "7"))
    assert "head moved" in str(e.value)


def test_pr_file_count_mismatch(pin, origin, monkeypatch):
    monkeypatch.setattr(pin, "gh", FakeGh(origin["head"], PR_FILES[:3]))
    with pytest.raises(SystemExit) as e:
        pin.main(init_args(origin, "--pr", "7"))
    assert "file lists differ" in str(e.value)
