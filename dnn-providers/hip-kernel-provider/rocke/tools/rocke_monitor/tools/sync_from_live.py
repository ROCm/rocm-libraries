#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One sync from the live toolkit into the package's own repository: a file-level squash, one commit, never a merge
(docs/maintaining.md, "Syncing from a live toolkit"). Run it from the package repository's top:

  tools/sync_from_live.py SOURCE_CHECKOUT REF -- --deny FILE [PRIVACY_ARGS...]

SOURCE_CHECKOUT is a checkout of the live toolkit and REF the reviewed sync branch in it. The script replaces this
repository's tracked files with REF's tracked tree, writes the blank config.json (tools/make_package.py BLANK_CONFIG)
in place of REF's, stages the result, runs the privacy check over the files and over the staged change as the synced
tree's own code (rkmon_lib/privacy.py `command`, run at this repository's top; the arguments after `--` go to it:
--deny FILE, the deny list, which it requires, and --config, --state, --workspace, --git), runs the tests (the synced
tree's own `rkmon.py selftest`), and commits as rkmon <rkmon@localhost>. It refuses a work tree with changes and a
target that shares the source's repository; when a step fails it commits nothing and leaves the staged sync for you to
inspect (`git reset --hard` drops it).

Exit: 0 committed (or nothing to sync), 1 a refusal or a failed step, 2 a usage error (no deny list named among
them); a git step that fails exits with git's own code.
"""

import os
import subprocess
import sys
import tempfile

# The package this script is in: the privacy check's command line and its parser (rkmon_lib/privacy.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from rkmon_lib import privacy  # noqa: E402

# The usage, printed when the arguments are missing: the paragraphs above the exit codes, as comments.
USAGE = "".join(
    f"# {line}\n" for line in __doc__.split("\nExit:")[0].splitlines() if line
)
BLANK = 'import sys; sys.path[:0] = ["tools", "."]; import make_package; sys.stdout.write(make_package.BLANK_CONFIG)'
IDENTITY = {
    "GIT_AUTHOR_NAME": "rkmon",
    "GIT_AUTHOR_EMAIL": "rkmon@localhost",
    "GIT_COMMITTER_NAME": "rkmon",
    "GIT_COMMITTER_EMAIL": "rkmon@localhost",
}


class Stop(Exception):
    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


def out(text: str = "") -> None:
    print(text, flush=True)


def run(argv: list[str], **kw) -> subprocess.CompletedProcess:
    sys.stdout.flush()
    return subprocess.run(argv, **kw)


def must(argv: list[str], **kw) -> subprocess.CompletedProcess:
    """A step that must work: its exit code ends the sync when it fails."""
    p = run(argv, **kw)
    if p.returncode != 0:
        raise Stop(p.returncode)
    return p


def text_of(argv: list[str], **kw) -> tuple[int, str]:
    p = run(argv, stdout=subprocess.PIPE, text=True, **kw)
    return p.returncode, p.stdout.rstrip("\n")


def common(path: str) -> str:
    """The folder of the repository's common git data that `path` belongs to ("" when there is none)."""
    if not os.path.isdir(path):
        print(f"cd: {path}: No such file or directory", file=sys.stderr)
        return ""
    rc, rel = text_of(["git", "rev-parse", "--git-common-dir"], cwd=path)
    if rc != 0 or not rel:
        print("cd: null directory", file=sys.stderr)
        return ""
    d = os.path.join(path, rel)
    return os.path.realpath(d) if os.path.isdir(d) else ""


def stop(why: str) -> None:
    out(f"{why}: nothing committed")
    raise Stop(1)


def names_deny(args: list[str]) -> bool:
    """Whether the privacy check, reading its arguments as it does (privacy.parser), finds its deny list in them, which
    it requires: `--deny FILE` or `--deny=FILE`. They are read before the paths the sync adds, so an option left
    without its value cannot take a path for it. Arguments the parser refuses name none (the parser says why).
    """
    try:
        a = privacy.parser().parse_args(args)
    except SystemExit:
        return False
    return bool(a.deny)


def sync(argv: list[str], tmp: str, label: str = "sync") -> int:
    """The steps above for SOURCE_CHECKOUT REF -- --deny FILE [PRIVACY_ARGS...] (argv), committed as "rocKE monitor
    package: <label>" (tools/publish_package.py passes its own label). Arguments that name no deny list are a
    usage error, before anything changes."""
    src, ref = argv[0], argv[1]
    args = argv[2:]
    if args[:1] == ["--"]:
        args = args[1:]
    if not names_deny(args):
        out(
            "the privacy check requires the deny list: name it with --deny FILE after --; nothing changed"
        )
        return 2
    rc, top = text_of(["git", "rev-parse", "--show-toplevel"])
    if rc != 0:
        out("run it inside the package repository")
        return 2
    os.chdir(top)
    if common(top) == common(src):
        out(
            f"{top} shares the source's repository: the package repository is a repository of its own"
        )
        return 2
    if (
        run(
            ["git", "-C", src, "rev-parse", "-q", "--verify", f"{ref}^{{commit}}"],
            stdout=subprocess.DEVNULL,
        ).returncode
        != 0
    ):
        out(f"no commit {ref} in {src}")
        return 2
    if text_of(["git", "status", "--porcelain", "--untracked-files=normal"])[1]:
        out("the work tree has changes: commit or drop them")
        return 1
    # The source's tracked tree, by git archive: its files only, none of its history.
    if (
        run(
            ["git", "rev-parse", "-q", "--verify", "HEAD"], stdout=subprocess.DEVNULL
        ).returncode
        == 0
    ):
        must(["git", "rm", "-r", "-q", "."])
    archive = subprocess.Popen(
        ["git", "-C", src, "archive", "--format=tar", ref], stdout=subprocess.PIPE
    )
    tar = run(["tar", "-xf", "-", "-C", top], stdin=archive.stdout)
    archive.stdout.close()
    archive.wait()
    if tar.returncode != 0:
        raise Stop(tar.returncode)
    with open("config.json", "w", encoding="utf-8") as f:
        must([sys.executable, "-B", "-c", BLANK], stdout=f)
    must(["git", "add", "-A"])
    if run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        out(f"nothing to sync: this repository already carries {ref}'s tree")
        return 0
    stat = must(
        ["git", "diff", "--cached", "--stat"], stdout=subprocess.PIPE, text=True
    ).stdout.splitlines()
    if stat:
        out(stat[-1])
    # What the commit adds to the history: the diff's added lines, each without its + in front (an added Python
    # decorator line would read as a GitHub handle otherwise), and the file headers. A removed or unchanged line is
    # in the history already, so it is not read again.
    diff = run(["git", "diff", "--cached"], stdout=subprocess.PIPE).stdout
    staged = os.path.join(tmp, "staged.diff")
    with open(staged, "wb") as f:
        for line in diff.splitlines(keepends=True):
            if line[:1] == b"+":
                f.write(line[1:])
            elif line[:1] not in (b"-", b" "):
                f.write(line)
    # The check runs as the synced tree's own code: its command line, at the top of this repository.
    check_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out("privacy check, the files:")
    if run(privacy.command(*args, "."), cwd=top, env=check_env).returncode != 0:
        stop("the privacy check failed on the files")
    out("privacy check, the staged change (the commit's history):")
    if run(privacy.command(*args, staged), cwd=top, env=check_env).returncode != 0:
        stop("the privacy check failed on the staged change")
    # The tests run as the synced tree's own `rk selftest`, at the top of this repository.
    out("tests:")
    log = os.path.join(tmp, "tests.log")
    with open(log, "w", encoding="utf-8") as f:
        rc = run(
            [sys.executable, "-B", "rkmon.py", "selftest"],
            stdout=f,
            stderr=subprocess.STDOUT,
        ).returncode
    with open(log, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    if rc != 0:
        for line in lines[-40:]:
            out(line)
        stop("the tests failed")
    summary = [ln for ln in lines if ln.startswith(("Ran ", "OK", "FAILED"))]
    if not summary:
        raise Stop(1)
    for line in summary:
        out(line)
    status = text_of(["git", "status", "--porcelain", "--untracked-files=normal"])[1]
    if [
        ln
        for ln in status.splitlines()
        if not (len(ln) > 1 and ln[0] in "MADR" and ln[1] == " ")
    ]:
        out("the tests left files in the work tree:")
        run(["git", "status", "--porcelain", "--untracked-files=normal"])
        return 1
    msg = "rocKE monitor package"
    if (
        run(
            ["git", "rev-parse", "-q", "--verify", "HEAD"], stdout=subprocess.DEVNULL
        ).returncode
        == 0
    ):
        msg += f": {label}"
    must(
        [
            "git",
            "-c",
            "user.name=rkmon",
            "-c",
            "user.email=rkmon@localhost",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            msg,
        ],
        env=dict(os.environ, **IDENTITY),
    )
    must(["git", "log", "-1", "--format=committed %h %s (%an <%ae>)"])
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        sys.stdout.write(USAGE)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        try:
            return sync(argv, tmp)
        except Stop as s:
            return s.code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
