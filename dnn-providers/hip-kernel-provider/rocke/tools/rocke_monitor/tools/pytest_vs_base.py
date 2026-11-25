#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run a PR's test files on the tree under test, 10 files per pytest run, without stopping at the first failure. Each
test that fails is run again on develop (when the develop tree has its file): one that fails there too is reported
but is not the PR's doing. No -x: a stop at a test that fails on develop too would leave the PR's own suite unrun.

Every skipped test is listed with its reason (pytest_plugins/rkmon_skips.py): a pass with skips reads "pass (k
skipped: ...)", and a skip for a tool CI's lanes have and this run lacks (cmake, PyYAML) reads "skip NOT RUN: <tool>
missing", never a pass. The report ends with the exact command that re-runs each file as this run did, in its
environment, which the run writes to a file: with RKMON_RERUN_VIA (the job scripts set it to the package's
`rk pr-run`), `VIA --env ENV -C ROOT -- python -B -m pytest ... FILE` runs it as PR code in a slot of its own, ENV the
environment as `rk pr-run --env` reads it (tests_env.0) in the folder the job copies the results to (RKMON_RESULTS;
OUT without it); without it, a subshell sources the environment as bash's exports (OUT/tests_env.sh) and goes to
ROOT first.

Usage: pytest_vs_base.py OUT ROOT BASE_ROOT|- PYTHONPATH BASE_PYTHONPATH FILE...
FILEs are relative to ROOT (and BASE_ROOT). RKMON_PYEXTRA (optional) joins the end of both PYTHONPATHs (the checks'
PyYAML, review.tool_env_lines). RKMON_HEAD_ROOT and RKMON_HEAD_PYTHONPATH (optional, the drift check): the PR's head
alone, where a failure develop does not share runs again; one that fails there too is not counted as the merge's.
Prints the report, then one last line: "TESTS pass|fail|skip <detail>".

pytest is `python -B -m pytest ...` with the `python` on PATH (the checks' venv), under `timeout` (1500 s a batch, 300 s
a collection, 900 s a re-run) and LC_ALL=C. This file is standalone and runs on Python 3.8 or newer: the remote GPU
tests (rkmon_lib/gpu.py) stage it on the nodes with the plugins beside it, and run it with the nodes' `python`.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

PYTEST = [
    b"python",
    b"-B",
    b"-m",
    b"pytest",
    b"-q",
    b"-rfE",
    b"-p",
    b"no:cacheprovider",
    b"--no-header",
    b"-o",
    b"addopts=",
    b"--continue-on-collection-errors",
    b"-p",
    b"rkmon_env",
]
SUMMARY = re.compile(
    rb"[0-9]+ (passed|failed|skipped|errors?)|^FAILED|^ERROR|no tests ran"
)
ENV_NAME = re.compile(
    rb"(PATH|TMPDIR|HIP_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|ROCKE_[A-Za-z0-9_]*|RKMON_[A-Za-z0-9_]*)"
)
SUBTEST = re.compile(rb"(\)|]) [^ ]+\.py(::|$| )")


def say(line: bytes) -> None:
    sys.stdout.buffer.write(line + b"\n")
    sys.stdout.buffer.flush()


def read(path: bytes) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return b""


def write(path: bytes, data: bytes, mode: str = "wb") -> None:
    try:
        with open(path, mode) as fh:
            fh.write(data)
    except OSError as exc:
        sys.stderr.write("%s: %s\n" % (os.fsdecode(path), exc.strerror))


def lines_of(data: bytes) -> list:
    parts = data.split(b"\n")
    if parts and parts[-1] == b"":
        parts.pop()
    return parts


def text(lines) -> bytes:
    return b"".join(x + b"\n" for x in lines)


def grep(path: bytes, rx, invert: bool = False, binary_ok: bool = False) -> list:
    """grep's matching lines of a file; a file with a NUL byte is binary: its matches are not shown (grep says so on
    stderr), unless binary_ok (grep -a)."""
    data = read(path)
    found = [x for x in lines_of(data) if bool(rx.search(x)) != invert]
    if found and not binary_ok and b"\0" in data:
        sys.stderr.write("grep: %s: binary file matches\n" % os.fsdecode(path))
        return []
    return found


def ids(path: bytes) -> list:
    """The failing test ids in a pytest log's short summary: FAILED and ERROR lines, and SUBFAILED lines as their test.
    pytest prints a subtest's keyword label in parentheses and a positional one in brackets ("SUBFAILED(flag='b') <id>",
    "SUBFAILED[tdm_d2] <id>"), and counts the parent test as passed. An id can hold spaces ("test_x[<lambda>-k must have
    BSHD shape]"); a message after it starts with " - " (an id is never cut at its first space, and a SUBFAILED line
    with a bracketed label is a failure)."""
    out = []
    for s in lines_of(read(path)):
        if s.startswith((b"FAILED ", b"ERROR ")):
            s = s.split(b" ", 1)[1]
        elif s.startswith(b"SUBFAILED"):
            m = SUBTEST.search(s)
            if not m:
                continue
            s = s[m.start() + 2 :]
        else:
            continue
        i = s.find(b" - ")
        out.append(s[:i] if i >= 0 else s)
    return out


def comm(a: list, b: list, keep: str) -> list:
    """comm's merge of two sorted line lists (byte order): keep "1" (lines only in a) or "3" (lines in both)."""
    out, i, j = [], 0, 0
    while i < len(a) or j < len(b):
        if i >= len(a):
            j += 1
            continue
        if j >= len(b) or a[i] < b[j]:
            if keep == "1":
                out.append(a[i])
            i += 1
        elif a[i] > b[j]:
            j += 1
        else:
            if keep == "3":
                out.append(a[i])
            i += 1
            j += 1
    return out


def ansic_quote(v: bytes) -> bytes:
    """bash's $'...' quoting (the C locale's)."""
    named = {
        0x1B: b"E",
        0x07: b"a",
        0x0B: b"v",
        0x08: b"b",
        0x0C: b"f",
        0x0A: b"n",
        0x0D: b"r",
        0x09: b"t",
    }
    out = b"$'"
    for c in v:
        if c in named:
            out += b"\\" + named[c]
        elif c in (0x5C, 0x27):
            out += b"\\" + bytes([c])
        elif c < 0x20 or c >= 0x7F:
            out += b"\\%03o" % c
        else:
            out += bytes([c])
    return out + b"'"


def needs_ansic(v: bytes) -> bool:
    return any(c < 0x20 or c >= 0x7F for c in v)


def declare_x(name: bytes, v: bytes) -> bytes:
    """One line of bash's `export -p` (LC_ALL=C)."""
    if needs_ansic(v):
        q = ansic_quote(v)
    else:
        q = b'"' + re.sub(rb'(["$`\\])', rb"\\\1", v) + b'"'
    return b"declare -x " + name + b"=" + q


BSTAB = set(b" !\"$&'()*,;<>?[\\]^`{|}\t\n")


ENV_FILE = b"tests_env.0"  # the run's environment as `rk pr-run --env` reads it: NUL-separated NAME=VALUE
ENV_SH = b"tests_env.sh"  # the run's environment as bash's exports, for a re-run without rk pr-run


def rerun_line(env_dir: bytes, root: bytes, f: bytes, via: bytes = b"") -> bytes:
    """The command that re-runs one file as this run did, its environment in the folder `env_dir`. `via`
    (RKMON_RERUN_VIA, which the job scripts set: `rk pr-run`) runs it as PR code in a slot of its own, with ENV_FILE's
    environment, in ROOT; without it, a subshell sources ENV_SH and goes to ROOT first.
    """
    cmd = b" ".join(PYTEST) + b" --rootdir=" + printf_q(root) + b" " + printf_q(f)
    if via:
        return b"   %s --env %s -C %s -- %s" % (
            via,
            printf_q(env_dir + ENV_FILE),
            printf_q(root),
            cmd,
        )
    return b"   (. %s && cd %s && %s)" % (
        printf_q(env_dir + ENV_SH),
        printf_q(root),
        cmd,
    )


def printf_q(v: bytes) -> bytes:
    """bash's printf %q (LC_ALL=C)."""
    if not v:
        return b"''"
    if needs_ansic(v):
        return ansic_quote(v)
    out = b""
    for k, c in enumerate(v):
        if (
            c in BSTAB
            or (c == 0x23 and k == 0)
            or (c == 0x7E and (k == 0 or v[k - 1] in (0x3A, 0x3D)))
        ):
            out += b"\\"
        out += bytes([c])
    return out


def run_pytest(
    cwd: bytes, pythonpath: bytes, limit: str, args: list, log: bytes
) -> int:
    """(cd CWD && PYTHONPATH=... timeout LIMIT python -B -m pytest ... > LOG 2>&1); its exit code."""
    if not os.path.isdir(cwd):
        sys.stderr.write("cd: %s: No such file or directory\n" % os.fsdecode(cwd))
        return 1
    env = dict(os.environb)
    env[b"PYTHONPATH"] = pythonpath
    try:
        with open(log, "wb") as fh:
            rc = subprocess.run(
                [b"timeout", limit.encode()] + PYTEST + args,
                cwd=cwd,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
            ).returncode
    except OSError as exc:
        sys.stderr.write("%s: %s\n" % (os.fsdecode(log), exc.strerror))
        return 1
    return rc if rc >= 0 else 128 - rc


def present(base: bytes, todo: list) -> list:
    """The test files of the ids in TODO that BASE has (sed 's/::.*//' | sort -u, then [ -f ])."""
    files = (x.strip(b" \t") for x in sorted({x.split(b"::", 1)[0] for x in todo}))
    return [f for f in files if os.path.isfile(base + b"/" + f)]


def rerun_ids(base: bytes, todo: list, collected: bytes) -> list:
    """The ids of TODO that BASE has: test ids its collection lists, file ids whose file it has."""
    have = set(lines_of(read(collected)))
    out = []
    for i in (x.strip(b" \t") for x in todo):
        if b"::" in i:
            if i in have:
                out.append(i)
        elif os.path.isfile(base + b"/" + i):
            out.append(i)
    return out


def main(argv: list) -> int:
    args = [os.fsencode(a) for a in argv[1:]]
    out_, root, base, pp, bpp = (args + [b""] * 5)[:5]
    files = (
        args[5:] if len(args) >= 5 else args
    )  # `shift 5` shifts nothing when fewer are given
    O = out_ + b"/"
    if out_:  # a folder of the results' own in the job's out (rkmon_lib/review.py)
        os.makedirs(out_, exist_ok=True)
    # --continue-on-collection-errors: a file that fails to import must not stop the other files of its batch.
    # -p rkmon_env (pytest_plugins/): a test that fails only because this host's hipcc cannot compile HIP at all is
    # skipped, with the reason (a host with hipcc but no HIP headers fails every hipcc case).
    plug = os.path.join(logical_cwd(), os.path.dirname(os.fsencode(argv[0])) or b".")
    plug = os.path.normpath(plug) + b"/pytest_plugins"
    extra = os.environb.get(b"RKMON_PYEXTRA", b"")
    pp = plug + (b":" + pp if pp else b"") + (b":" + extra if extra else b"")
    bpp = plug + (b":" + bpp if bpp else b"") + (b":" + extra if extra else b"")
    os.environ["LC_ALL"] = "C"
    for name in (
        b"tests.log",
        b"tests_failed.txt",
        b"tests_all.log",
        b"tests_skipped.tsv",
    ):
        # tests_all.log: every batch's full log (tracebacks, SUBFAILED labels), not only the last one's (tests_batch.log
        # holds only the last batch). tests_skipped.tsv: every test the tree under test skips, with its reason
        # (rkmon_skips.py lines): the remote GPU tests run the cases skipped for want of a GPU (rkmon_lib/gpu.py).
        write(O + name, b"")
    # This run's environment, for the re-run commands: PATH, the temp folder, the GPU's visibility as this run has it
    # (the checks hide it: rkmon_lib/nogpu.py), the ROCKE_* exports, PyYAML's folder and the PYTHONPATH. Through
    # rk pr-run (RKMON_RERUN_VIA) a command gets that environment alone (ENV_FILE); a shell that sources ENV_SH unsets
    # any other ROCKE_* first.
    via = os.environb.get(b"RKMON_RERUN_VIA", b"")
    env_items = [
        (k, v) for k, v in sorted(os.environb.items()) if ENV_NAME.fullmatch(k)
    ]
    if via:
        env_items += [(b"LC_ALL", b"C"), (b"PYTHONPATH", pp)]
        write(O + ENV_FILE, b"".join(k + b"=" + v + b"\0" for k, v in env_items))
    else:
        env_lines = [b'for v in $(compgen -e | grep "^ROCKE_"); do unset "$v"; done']
        env_lines += [declare_x(k, v) for k, v in env_items]
        env_lines.append(b"export LC_ALL=C PYTHONPATH=" + printf_q(pp))
        write(O + ENV_SH, text(env_lines))
    rest = list(files)
    while rest:
        batch, rest = rest[:10], rest[10:]
        rc = run_pytest(
            root, pp, "1500", [b"--rootdir=" + root] + batch, O + b"tests_batch.log"
        )
        log = read(O + b"tests_batch.log")
        write(
            O + b"tests_all.log",
            b"=== batch (rc=%d): " % rc + b" ".join(batch) + b"\n" + log,
            "ab",
        )
        write(O + b"tests_failed.txt", text(ids(O + b"tests_batch.log")), "ab")
        write(
            O + b"tests_skipped.tsv",
            text(
                grep(
                    O + b"tests_batch.log",
                    re.compile(rb"^rkmon-skip:\t"),
                    binary_ok=True,
                )
            ),
            "ab",
        )
        # pytest exit codes: 0 passed, 1 tests failed, 5 nothing collected; anything else stopped the run itself.
        if rc not in (0, 1, 5):
            write(
                O + b"tests_failed.txt",
                b"RUN-ERROR(rc=%d):" % rc + batch[0] + b"\n",
                "ab",
            )
        for picked in (
            grep(O + b"tests_batch.log", SUMMARY)[-4:],
            grep(O + b"tests_batch.log", re.compile(rb"^rkmon(-import)?: ")),
        ):
            write(O + b"tests.log", text(picked), "ab")
            for x in picked:
                say(b"   " + x)
    failed = sorted(set(lines_of(read(O + b"tests_failed.txt"))))
    write(O + b"tests_failed.txt", text(failed))
    write(O + b"tests_develop_failed.txt", b"")
    write(O + b"tests_develop_unrun.txt", b"")
    if failed and base != b"-":
        f = present(base, failed)
        r = []
        if f:
            # Only ids develop has: one the PR adds makes pytest stop with "not found" and report nothing, and
            # develop's own failures would read "on the merge only".
            run_pytest(
                base,
                bpp,
                "300",
                [b"--collect-only", b"--rootdir=" + base] + f,
                O + b"tests_develop_ids.log",
            )
            r = rerun_ids(base, failed, O + b"tests_develop_ids.log")
        if r:
            run_pytest(
                base,
                bpp,
                "900",
                [b"-rA", b"--rootdir=" + base] + r,
                O + b"tests_develop.log",
            )
            dev_failed = sorted(set(ids(O + b"tests_develop.log")))
            write(O + b"tests_develop_failed.txt", text(dev_failed))
            # A failing id develop neither fails nor passes was skipped there (its own importorskip, a build develop's
            # leg lacks): not compared, never "on the merge only" (a test develop skips without the engine binding
            # may fail with it, on develop too).
            passed = sorted(
                {
                    x[len(b"PASSED ") :]
                    for x in grep(O + b"tests_develop.log", re.compile(rb"^PASSED "))
                }
            )
            write(O + b"tests_develop_passed.txt", text(passed))
            unrun = comm(
                comm(sorted({x for x in r if b"::" in x}), dev_failed, "1"), passed, "1"
            )
            write(O + b"tests_develop_unrun.txt", text(unrun))
    # The head alone (RKMON_HEAD_ROOT, its PYTHONPATH RKMON_HEAD_PYTHONPATH; the drift check sets them): a failing id
    # develop does not fail runs again on the PR's head under the same extraction, and one that fails there too is the
    # head's own (a test new to develop that the extraction cannot serve, or the PR's own failure), not develop's doing.
    write(O + b"tests_head_failed.txt", b"")
    hr = os.environb.get(b"RKMON_HEAD_ROOT", b"")
    if hr and os.path.isdir(hr):
        hpp_extra = os.environb.get(b"RKMON_HEAD_PYTHONPATH", b"")
        hpp = (
            plug
            + (b":" + hpp_extra if hpp_extra else b"")
            + (b":" + extra if extra else b"")
        )
        todo = [
            x
            for x in comm(failed, lines_of(read(O + b"tests_develop_failed.txt")), "1")
            if not x.startswith(b"RUN-ERROR")
        ]
        write(O + b"tests_head_todo.txt", text(todo))
        f = present(hr, todo)
        h = []
        if f:
            # Only ids the head has, as for develop above.
            run_pytest(
                hr,
                hpp,
                "300",
                [b"--collect-only", b"--rootdir=" + hr] + f,
                O + b"tests_head_ids.log",
            )
            h = rerun_ids(hr, todo, O + b"tests_head_ids.log")
        if h:
            run_pytest(
                hr, hpp, "900", [b"-rA", b"--rootdir=" + hr] + h, O + b"tests_head.log"
            )
            head_failed = comm(sorted(set(ids(O + b"tests_head.log"))), todo, "3")
            write(O + b"tests_head_failed.txt", text(head_failed))
            unrun = comm(
                lines_of(read(O + b"tests_develop_unrun.txt")), head_failed, "1"
            )
            write(O + b"tests_develop_unrun.tmp", text(unrun))
            try:
                os.replace(
                    O + b"tests_develop_unrun.tmp", O + b"tests_develop_unrun.txt"
                )
            except OSError as exc:
                sys.stderr.write("mv: %s\n" % exc.strerror)
    dev_failed = lines_of(read(O + b"tests_develop_failed.txt"))
    unrun = lines_of(read(O + b"tests_develop_unrun.txt"))
    head_failed = lines_of(read(O + b"tests_head_failed.txt"))
    pr_failed = comm(comm(comm(failed, dev_failed, "1"), unrun, "1"), head_failed, "1")
    old_failed = comm(failed, dev_failed, "3")
    write(O + b"tests_pr_failed.txt", text(pr_failed))
    write(O + b"tests_old_failed.txt", text(old_failed))
    tests_log = O + b"tests.log"
    sums = {}
    for m in (
        y
        for x in grep(tests_log, re.compile(rb"[0-9]+ (passed|failed|errors?|skipped)"))
        for y in re.finditer(rb"[0-9]+ (passed|failed|errors?|skipped)", x)
    ):
        word = m.group(0).replace(b"errors", b"error", 1).split(b" ")
        sums[word[1]] = sums.get(word[1], 0) + int(word[0])
    tot = b", ".join(
        b"%d %s" % (sums[k], k)
        for k in (b"passed", b"failed", b"error", b"skipped")
        if sums.get(k)
    )
    np_, no, nu, nh = len(pr_failed), len(old_failed), len(unrun), len(head_failed)
    # What pytest's own summary lines count as failed: a failure whose id no rule above reads must not pass.
    nf = sum(
        int(m.group(0).split(b" ")[0])
        for x in grep(tests_log, re.compile(rb"^(FAILED|ERROR) "), invert=True)
        for m in re.finditer(rb"[0-9]+ (failed|errors?)", x)
    )
    # Failures rkmon_env skipped as this host's (its "rkmon: N hipcc failures skipped" lines).
    ns = sum(
        int(re.match(rb"rkmon: ([0-9]+)", x).group(1))
        for x in grep(tests_log, re.compile(rb"^rkmon: [0-9]+"))
    )
    host = (
        b"; %d hipcc failures skipped: this host's hipcc cannot compile HIP" % ns
        if ns > 0
        else b""
    )
    # Tests rkmon_env skipped because they import a module this host lacks (its "rkmon-import: N tests skipped" line).
    ni = sum(
        int(re.match(rb"rkmon-import: ([0-9]+)", x).group(1))
        for x in grep(tests_log, re.compile(rb"^rkmon-import: [0-9]+"))
    )
    if ni > 0:
        host += (
            b"; %d skipped: they import a module this host lacks, without pytest.importorskip"
            % ni
        )
    # The skips: each test and its reason, and the tools CI has that kept tests from running here.
    skips = O + b"tests_skipped.tsv"
    plug_skips = plug + b"/rkmon_skips.py"
    try:
        res = subprocess.run(
            [b"python", b"-I", plug_skips, b"result", skips], stdout=subprocess.PIPE
        ).stdout
    except OSError as exc:
        sys.stderr.write("python: %s\n" % exc.strerror)
        res = b""
    got = [x.strip(b" \t") for x in lines_of(res)] + [b"", b"", b""]
    notrun, skipsum = got[1], got[2]
    sys.stdout.flush()
    try:
        subprocess.run(
            [b"python", b"-I", plug_skips, b"report", b"--where", skips, skips]
        )
    except OSError as exc:
        sys.stderr.write("python: %s\n" % exc.strerror)
    # The environment file is named where the job copies the results (RKMON_RESULTS), else here.
    env_dir = (os.environb.get(b"RKMON_RESULTS", b"") or out_) + b"/"
    say(
        b"re-run one file as this run did (its environment: "
        + env_dir
        + (ENV_FILE if via else ENV_SH)
        + b"; add -k or a test id after the file):"
    )
    for f in files:
        if os.path.exists(root + b"/" + f):
            say(rerun_line(env_dir, root, f, via))
    nr = b"; " + notrun if notrun else b""
    also = (b"; %d fail on develop too" % no if no > 0 else b"") + (
        b"; %d fail on the head alone too" % nh if nh > 0 else b""
    )
    for n, what, lines in (
        (no, b"fail on develop too, not this PR's doing", old_failed),
        (nh, b"fail on the head alone too, not develop's doing", head_failed),
        (nu, b"not compared, develop skips them", unrun),
    ):
        if n > 0:
            say(what + b" (%d):" % n)
            for x in lines[:6]:
                say(b"   " + x)
    if np_ > 0:
        where = b"not compared with develop" if base == b"-" else b"on the merge only"
        say(
            b"TESTS fail %d failing (" % np_
            + where
            + b"): "
            + two(pr_failed)
            + host
            + nr
        )
    elif nu > 0:
        say(
            b"TESTS fail %d failing (not compared: develop skips them): " % nu
            + two(unrun)
            + host
            + nr
        )
    elif not tot:
        say(b"TESTS fail no test result")
    elif no == 0 and nh == 0 and nf > 0:
        say(
            b"TESTS fail %d failed by pytest's count, but no failing test id was read"
            % nf
            + nr
        )
    elif notrun:
        say(b"TESTS skip " + notrun + b"; " + tot + also + host)
    else:
        say(
            b"TESTS pass "
            + (b"(" + skipsum + b") " if skipsum else b"")
            + tot
            + also
            + host
        )
    return 0


def two(lines: list) -> bytes:
    """`head -2 FILE | tr '\\n' ' ' | sed 's/ $//'`."""
    joined = b"".join(x + b" " for x in lines[:2])
    return joined[:-1] if joined.endswith(b" ") else joined


def logical_cwd() -> bytes:
    """The shell's working directory as `pwd` says it: $PWD when it names this directory, else the physical path."""
    pwd = os.environb.get(b"PWD", b"")
    try:
        if pwd.startswith(b"/") and os.path.samefile(pwd, b"."):
            return pwd
    except OSError:
        pass
    return os.getcwdb()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
