#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The representative golden gate as CI runs it (CTest rocke_golden_static: rocke_installed_golden_test.py checks every
flavor the golden holds, in one run), with its failing lines sorted: the golden's own load errors (a shard that does
not load beside the others: its alias map, schema or flavors differ; re-bless that shard) and case drift, counted by
family and flavor. A failing line develop's tree prints too is develop's, not the PR's, as pytest_vs_base.py does for
tests. The first "gate:" line alone would hide a load error ("<golden>.json has aliases {'llvm24': 'llvm23'}") behind
develop's own drift, and the drifting cases (lessons L-523).

Two steps: `gate` runs the gate, as a job that runs PR code; `report` reads what the jobs
left as data, runs nothing, and is the one step that writes develop's cached lines.

Usage: golden_vs_base.py gate OUT TESTS
  The gate in TESTS (a tree's platform/tests): `timeout ${GOLDEN_TIMEOUT:-2400} ${PYTHON:-python} -B
  rocke_installed_golden_test.py`, with LC_ALL=C. OUT gets its output (golden.log) and its failing lines
  (golden.log.lines: the indented lines under FAIL, sorted). Prints the first "gate:" line, then, on a pass or with no
  result, one last line "GOLDEN pass|fail <detail>". Exit 0 pass, 1 failing lines (the report's to sort), 2 no result.
Usage: golden_vs_base.py report OUT TESTS BASE [BASE_CACHE [DEVELOP_OUT]]
  The failing lines of the gate whose OUT this is (run in TESTS), compared with develop's. BASE: "-", no develop to
  compare with; "self", TESTS is develop's; else develop's platform/tests, whose failing lines come from BASE_CACHE when
  it holds them (an earlier complete run at the same develop SHA), else from DEVELOP_OUT (the OUT of develop's own gate
  job), and go to BASE_CACHE after a complete run. Prints the report, then one last line "GOLDEN pass|fail <detail>".
  OUT gets the develop lines it used (golden_develop.lines) and the sorted lines (golden_pr.txt, golden_old.txt,
  golden_pr_load.txt, golden_pr_drift.txt).
"""

import os
import re
import stat
import subprocess
import sys

GD = "golden/rocke_representative_ir_sha256"
# The harness imports rocke and the library kernels: without this PYTHONPATH the printed command fails.
REC = (
    "PYTHONPATH=python:../library python tests/instances/rocke_ir_parity_harness.py --write tests/"
    + GD
    + " --only"
).encode()
FLAVOR = re.compile(rb"^\[[A-Za-z0-9_]+\] ")
CASES = re.compile(rb"^(cases|expected_failures): (missing|new) current ")
ALIASES = re.compile(rb'"aliases":\{[^}]*\}')
BLANKS = re.compile(rb"[ \t\n]+")
# Below a tree's tests folder no link is followed: a PR's tree may hold a link to a file of the monitor's.
_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


def fields(line: bytes) -> list[bytes]:
    """awk's fields: the line split on runs of blanks."""
    return [f for f in BLANKS.split(line) if f]


def say(line: bytes) -> None:
    sys.stdout.buffer.write(line + b"\n")
    sys.stdout.buffer.flush()


def read(path: str) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return b""


def write(path: str, data: bytes) -> None:
    try:
        with open(path, "wb") as fh:
            fh.write(data)
    except OSError as exc:
        sys.stderr.write(f"{path}: {exc.strerror}\n")


def lines_of(data: bytes) -> list[bytes]:
    parts = data.split(b"\n")
    if parts and parts[-1] == b"":
        parts.pop()
    return parts


def text(lines: list[bytes]) -> bytes:
    return b"".join(x + b"\n" for x in lines)


def nonempty(path: str) -> bool:
    try:
        return os.stat(path).st_size > 0
    except OSError:
        return False


def first_gate_line(log: str, width: int) -> bytes:
    """The log's first line that holds "gate:", cut to `width` bytes; b"" when there is none."""
    return next((x for x in lines_of(read(log)) if b"gate:" in x), b"")[:width]


def comm(a: list[bytes], b: list[bytes], keep: str) -> list[bytes]:
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


def gate(tests: str, log: str) -> int:
    """Run the gate in TESTS: LOG gets its output, LOG.lines its failing lines (the indented ones under FAIL), sorted."""
    if os.path.isdir(tests):
        try:
            with open(log, "wb") as fh:
                rc = subprocess.run(
                    [
                        "timeout",
                        os.environ.get("GOLDEN_TIMEOUT") or "2400",
                        os.environ.get("PYTHON") or "python",
                        "-B",
                        "rocke_installed_golden_test.py",
                    ],
                    cwd=tests,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                ).returncode
            rc = rc if rc >= 0 else 128 - rc
        except OSError as exc:
            sys.stderr.write(f"{log}: {exc.strerror}\n")
            rc = 1
    else:
        sys.stderr.write(f"cd: {tests}: No such file or directory\n")
        rc = 1
    picked = sorted({x[2:] for x in lines_of(read(log)) if x.startswith(b"  ")})
    write(log + ".lines", text(picked))
    return rc


def family_flavor(line: bytes) -> bytes:
    """ "<family>@<flavor>" of a drift line: "[llvm24] attention/x: a -> b", "[llvm20] cases: new current gemm/y"."""
    s, fl = line, b"all"
    m = FLAVOR.match(s)
    if m:
        fl, s = s[1 : m.end() - 2], s[m.end() :]
    s = CASES.sub(b"", s, count=1)
    i = s.find(b"/")
    f = s[:i] if i >= 0 else s
    c = f.find(b":")
    return (f[:c] if c >= 0 else f) + b"@" + fl


def counts(path: str) -> bytes:
    """ "<family>@<flavor> ×<n>" for each, joined with ", " (sort | uniq -c, then awk's $2 and $1)."""
    keys = sorted(family_flavor(x) for x in lines_of(read(path)))
    groups = []
    for k in keys:
        if groups and groups[-1][0] == k:
            groups[-1][1] += 1
        else:
            groups.append([k, 1])
    parts = []
    for k, n in groups:
        f = fields(b"%7d " % n + k)
        parts.append((f[1] if len(f) > 1 else b"") + " ×".encode() + b"%d" % int(f[0]))
    return b", ".join(parts)


def fams(path: str) -> bytes:
    return b",".join(
        sorted({family_flavor(x).split(b"@", 1)[0] for x in lines_of(read(path))})
    )


def golden_folder(tests: str) -> int | None:
    """TESTS's golden folder, each folder below TESTS opened with O_NOFOLLOW: a descriptor, or None (none, or a link
    on the way)."""
    try:
        fd = os.open(tests, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError:
        return None
    for part in GD.split("/"):
        try:
            sub = os.open(part, _DIR, dir_fd=fd)
        except OSError:
            sub = None
        os.close(fd)
        if sub is None:
            return None
        fd = sub
    return fd


def head(folder: int, name: bytes) -> bytes | None:
    """The first 8 lines of the regular file `name` in the folder `folder`; None for a link or anything else."""
    try:
        fd = os.open(name, _FILE, dir_fd=folder)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as fh:
        if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
            return None
        return b"".join(fh.readline() for _ in range(8))


def amaps(tests: str) -> list[bytes]:
    """One line per distinct top-level alias map in TESTS's golden folder, fewest shards first: "<count> <map>
    <shard,shard>". Below TESTS no link is followed and only regular files are read (the report runs as the monitor;
    a link would show it a file of its own)."""
    folder = golden_folder(tests)
    if folder is None:
        return []
    n, s, order = {}, {}, []
    try:
        names = sorted(
            os.fsencode(x)
            for x in os.listdir(folder)
            if x.endswith(".json") and not x.startswith(".")
        )
        for name in names:
            first = head(folder, name)
            if first is None:
                continue
            m = ALIASES.search(first.replace(b" ", b"").replace(b"\n", b""))
            rec = (m.group(0) if m else b"") + b" " + name[:-5]
            f = fields(rec)
            if len(f) != 2:
                continue
            if f[0] not in n:
                order.append(f[0])
            n[f[0]] = n.get(f[0], 0) + 1
            s[f[0]] = s[f[0]] + b"," + f[1] if s.get(f[0]) else f[1]
    except OSError:
        pass
    finally:
        os.close(folder)
    out = [b"%d " % n[k] + k + b" " + s[k] for k in order]
    return sorted(out, key=lambda x: (int(x.split(b" ", 1)[0]), x))


def tail_cut(path: str, k: int, width: int) -> list[bytes]:
    return [x[:width] for x in lines_of(read(path))[-k:]]


def run_gate(out: str, tests: str) -> int:
    """`gate` (the module's usage): 0 pass, 1 failing lines, 2 no result."""
    os.makedirs(out, exist_ok=True)
    rc = gate(tests, f"{out}/golden.log")
    first = first_gate_line(f"{out}/golden.log", 160)
    say(first or b"no gate line (rc=%d)" % rc)
    if rc == 0 and b"gate: PASS" in first:
        say(b"GOLDEN pass " + first[first.index(b"PASS") :][:120])
        return 0
    if not nonempty(f"{out}/golden.log.lines"):
        for x in tail_cut(f"{out}/golden.log", 5, 200):
            say(b"   " + x)
        last = tail_cut(f"{out}/golden.log", 1, 140)
        say(b"GOLDEN fail no gate result (rc=%d): " % rc + (last[0] if last else b""))
        return 2
    return 1


def develop_lines(out: str, bt: str, bc: str, dev: str) -> None:
    """Develop's failing lines into OUT/golden_develop.lines: BASE_CACHE's when it holds them; else, when BASE names
    develop's tree, DEVELOP_OUT's, which go to BASE_CACHE when that run is complete (it printed a gate line: a timeout
    is never taken for develop's lines)."""
    write(f"{out}/golden_develop.lines", b"")
    if bc and nonempty(bc):
        cached = lines_of(read(bc))
        write(
            f"{out}/golden_develop.lines",
            text([x for x in cached if not x.startswith(b"#gate ")]),
        )
        say(
            b"develop (an earlier run at the same SHA): "
            + next((x for x in cached if x.startswith(b"#gate ")), b"")[6:130]
        )
    elif bt not in ("-", "self"):
        lines = read(f"{dev}/golden.log.lines") if dev else b""
        dfirst = first_gate_line(f"{dev}/golden.log", 120) if dev else b""
        write(f"{out}/golden_develop.lines", lines)
        say(b"develop: " + (dfirst or b"no gate line"))
        if bc and dfirst:
            write(bc, b"#gate " + dfirst + b"\n" + lines)


def report(out: str, t: str, bt: str, bc: str, dev: str) -> int:
    """`report` (the module's usage)."""
    if not nonempty(f"{out}/golden.log.lines"):
        say(b"GOLDEN fail no gate result: the gate's failing lines are not there")
        return 0
    develop_lines(out, bt, bc, dev)
    where = {"-": b"not compared with develop", "self": b""}.get(
        bt, b"on the merge only"
    )
    mine, devl = lines_of(read(f"{out}/golden.log.lines")), lines_of(
        read(f"{out}/golden_develop.lines")
    )
    pr, old = comm(mine, devl, "1"), comm(mine, devl, "3")
    write(f"{out}/golden_pr.txt", text(pr))
    write(f"{out}/golden_old.txt", text(old))
    load = [x for x in pr if x.startswith(b"golden ")]
    drift = [x for x in pr if not x.startswith(b"golden ")]
    write(f"{out}/golden_pr_load.txt", text(load))
    write(f"{out}/golden_pr_drift.txt", text(drift))
    nl, nd, no = len(load), len(drift), len(old)
    paren = b", " + where if where else b""
    if nl:
        say(b"the golden does not load (%d" % nl + paren + b"):")
        for x in load[:4]:
            say(b"   " + x[:220])
        m = amaps(t)
        if len(m) > 1:
            say(b"alias maps (shards per map, fewest first):")
            for x in m:
                say(b"   " + x[:200])
            say(
                b"re-bless the odd shards (from platform/, on a merge with develop): "
                + REC
                + b" "
                + m[0].split(b" ")[2]
            )
    if nd:
        say(b"drift (%d" % nd + paren + b"): " + counts(f"{out}/golden_pr_drift.txt"))
        for x in drift[:6]:
            say(b"   " + x[:200])
        say(
            b"re-record (from platform/, on a merge with develop): "
            + REC
            + b" "
            + fams(f"{out}/golden_pr_drift.txt")
        )
    if no:
        say(
            b"fail on develop too, not this PR's doing (%d): " % no
            + counts(f"{out}/golden_old.txt")
        )
    where_p = b" (" + where + b")" if where else b""
    if nl:
        say(b"GOLDEN fail the golden does not load" + where_p + b": " + load[0][:150])
    elif nd:
        say(
            b"GOLDEN fail %d drifting" % nd
            + where_p
            + b": "
            + counts(f"{out}/golden_pr_drift.txt")[:150]
        )
    else:
        say(
            b"GOLDEN pass every failing line fails on develop too (%d: " % no
            + counts(f"{out}/golden_old.txt")[:120]
            + b")"
        )
    return 0


def main(argv: list[str]) -> int:
    args = argv[1:]
    os.environ["LC_ALL"] = "C"
    if len(args) == 3 and args[0] == "gate":
        return run_gate(args[1], args[2])
    if 4 <= len(args) <= 6 and args[0] == "report":
        out, t, bt, bc, dev = (args[1:] + ["", ""])[:5]
        return report(out, t, bt, bc, dev)
    sys.stderr.write(
        "usage: golden_vs_base.py gate OUT TESTS | report OUT TESTS BASE [BASE_CACHE [DEVELOP_OUT]]\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
