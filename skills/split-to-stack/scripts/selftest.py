#!/usr/bin/env python3
"""Self-test for the split-to-stack scripts.

usage: selftest.py                         build a synthetic repo in a temp dir and check every script
       selftest.py --stress REPO BASE TIP  assign the real BASE..TIP ledger hunks round-robin to 3
                                           groups, build, and require tree(top) == tree(TIP). New
                                           objects go to a temporary object directory; REPO is not
                                           modified.
Exit 0 = all checks passed.
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from splitlib import (
    ASSIGN_COLUMNS,
    GROUP_COLUMNS,
    LEDGER_COLUMNS,
    TRANSITION_COLUMNS,  # noqa: E402
    read_tsv,
    write_tsv,
)

FAILS = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def run(cmd, env=None, check_rc=True):
    e = os.environ.copy()
    if env:
        e.update(env)
    p = subprocess.run(cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check_rc and p.returncode != 0:
        raise RuntimeError("%s failed: %s" % (cmd, p.stderr.decode()))
    return p


def script(name, *args, check_rc=True, env=None):
    e = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    e.update(env or {})
    return run(
        [sys.executable, os.path.join(HERE, name)] + list(args),
        check_rc=check_rc,
        env=e,
    )


def git(repo, *args, env=None):
    return run(["git", "-C", repo] + list(args), env=env).stdout.decode()


GENV = {
    "GIT_AUTHOR_NAME": "T",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "T",
    "GIT_COMMITTER_EMAIL": "t@example.com",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
}


def write(repo, path, data, mode=None):
    full = os.path.join(repo, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode())
    if mode:
        os.chmod(full, mode)


def lines(n, prefix="line"):
    return "".join("%s%d\n" % (prefix, i) for i in range(1, n + 1))


def blob_at(repo, rev, path):
    p = run(
        ["git", "-C", repo, "cat-file", "blob", "%s:%s" % (rev, path)], check_rc=False
    )
    return p.stdout if p.returncode == 0 else None


def mode_at(repo, rev, path):
    out = git(repo, "ls-tree", rev, "--", path).split()
    return out[0] if out else None


def synthetic():
    tmp = tempfile.TemporaryDirectory(prefix="split-selftest-")
    repo = os.path.join(tmp.name, "repo")
    os.makedirs(repo)
    git(repo, "init", "-q", env=GENV)
    base_a = lines(20)
    write(repo, "a.txt", base_a)
    write(repo, "b.txt", lines(3, "b"))
    write(repo, "old/name.txt", lines(10, "n"))
    write(repo, "bin.dat", b"\x00\x01\x02base")
    write(repo, "del.txt", "gone\n")
    write(repo, "run.sh", "echo hi\n", 0o644)
    write(repo, "noeol.txt", "x\ny")
    loop_base = (
        "int f() {\n  for (i = 0; i <= n; i++) use(old_name[i]);\n  return 0;\n}\n"
    )
    write(repo, "loop.cpp", loop_base)
    write(repo, "lnk", "target\n")
    for n in ("one", "two"):
        write(repo, "gen/%s.yaml" % n, lines(5, n))
    git(repo, "add", "-A", env=GENV)
    git(repo, "commit", "-q", "-m", "base", env=GENV)
    base = git(repo, "rev-parse", "HEAD").strip()

    # source: one commit with every kind of change
    a = base_a.split("\n")[:-1]
    new_a = (
        a[:3]
        + ["INS_AFTER_3"]
        + a[3:10]
        + ["INS_AFTER_10"]
        + a[10:16]
        + ["CHANGED_17"]
        + a[17:]
    )
    write(repo, "a.txt", "\n".join(new_a) + "\n")
    os.makedirs(os.path.join(repo, "new"), exist_ok=True)
    git(repo, "mv", "old/name.txt", "new/name.txt", env=GENV)
    write(repo, "new/name.txt", lines(10, "n").replace("n5\n", "n5-edited\n"))
    write(repo, "bin.dat", b"\x00\x01\x02tip")
    os.remove(os.path.join(repo, "del.txt"))
    write(repo, "run.sh", "echo hi\n", 0o755)
    write(repo, "noeol.txt", "x\ny\nz")
    write(
        repo,
        "loop.cpp",
        loop_base.replace("i <= n", "i < n").replace("old_name", "new_name"),
    )
    write(repo, "added.txt", "new file\n")
    os.remove(os.path.join(repo, "lnk"))
    os.symlink("target", os.path.join(repo, "lnk"))
    for n in ("one", "two"):
        write(repo, "gen/%s.yaml" % n, lines(5, n) + "more\n")
    write(repo, "gen/three.yaml", lines(3, "three"))
    git(repo, "add", "-A", env=GENV)
    git(repo, "commit", "-q", "-m", "source", env=GENV)
    tip = git(repo, "rev-parse", "HEAD").strip()
    return tmp, repo, base, tip, base_a, loop_base


def test_synthetic():
    tmp, repo, base, tip, base_a, loop_base = synthetic()
    runs = os.path.join(tmp.name, "run")
    os.makedirs(os.path.join(runs, "states"))
    out = script(
        "build_ledger.py",
        "--repo",
        repo,
        "--base",
        base,
        "--tip",
        tip,
        "--out",
        os.path.join(runs, "ledger.tsv"),
        "--generated-set",
        "gen=gen/*",
    ).stdout.decode()
    print(out.strip())
    meta, rows = read_tsv(os.path.join(runs, "ledger.tsv"), LEDGER_COLUMNS)
    by_path = {}
    for r in rows:
        by_path.setdefault(r["path"], []).append(r)

    def ids(path):
        return [r["id"] for r in by_path[path]]

    check(
        len(by_path["a.txt"]) == 3 and all(r["id"][0] == "H" for r in by_path["a.txt"]),
        "a.txt gives 3 hunks",
    )
    check(
        by_path["new/name.txt"][0]["status"] == "R"
        and by_path["new/name.txt"][0]["old_path"] == "old/name.txt",
        "rename detected as one F row",
    )
    check(by_path["bin.dat"][0]["binary"] == "yes", "binary file is a whole-file row")
    check(
        by_path["run.sh"][0]["mode"] == "100755"
        and by_path["run.sh"][0]["status"] == "M",
        "mode-only change gets its own row",
    )
    check(by_path["lnk"][0]["status"] == "T", "type change (file to symlink) is a row")
    check(
        all(
            r["id"][0] == "G"
            for p in ("gen/one.yaml", "gen/two.yaml", "gen/three.yaml")
            for r in by_path[p]
        ),
        "generated set files are G rows",
    )
    check(
        len(by_path["loop.cpp"]) == 1, "fix+rename on one line is one hunk (entangled)"
    )

    groups = [
        dict(
            NN="01",
            suffix="fix-bound",
            type="fix",
            scope="x",
            subject="fix the loop bound",
            author="",
            trailers="",
            depends_on="",
            summary="Fixes the loop bound.",
        ),
        dict(
            NN="02",
            suffix="rename",
            type="refactor",
            scope="x",
            subject="rename old_name",
            author="A B <ab@example.com>",
            trailers="Fixes #1",
            depends_on="01",
            summary="Renames things.",
        ),
        dict(
            NN="03",
            suffix="feature",
            type="feat",
            scope="",
            subject="add stuff",
            author="",
            trailers="",
            depends_on="02",
            summary="Adds stuff.",
        ),
        dict(
            NN="04",
            suffix="generated",
            type="chore",
            scope="",
            subject="regenerate yaml",
            author="",
            trailers="",
            depends_on="03",
            summary="Generated files.",
        ),
    ]
    write_tsv(os.path.join(runs, "groups.tsv"), GROUP_COLUMNS, groups)
    a_ids = ids("a.txt")
    asg = []

    def add(entry, step, nsteps, group, state="final"):
        asg.append(
            dict(
                entry=entry,
                step=str(step),
                nsteps=str(nsteps),
                group=group,
                state=state,
                purpose="p",
                note="",
            )
        )

    add(a_ids[0], 1, 1, "01")
    add(a_ids[1], 1, 1, "02")
    add(a_ids[2], 1, 1, "03")
    with open(os.path.join(runs, "states", "loop.1"), "w") as f:
        f.write("  for (i = 0; i < n; i++) use(old_name[i]);\n")
    add(ids("loop.cpp")[0], 1, 2, "01", "stored:states/loop.1")
    add(ids("loop.cpp")[0], 2, 2, "02")
    add(ids("del.txt")[0], 1, 1, "01")
    add(ids("new/name.txt")[0], 1, 1, "02")
    add(ids("bin.dat")[0], 1, 1, "02")
    add(ids("noeol.txt")[0], 1, 1, "03")
    add(ids("added.txt")[0], 1, 1, "03")
    add(ids("run.sh")[0], 1, 1, "03")
    add(ids("lnk")[0], 1, 1, "03")
    add("@gen", 1, 1, "04")
    write_tsv(os.path.join(runs, "plan_assign.tsv"), ASSIGN_COLUMNS, asg)
    with open(os.path.join(runs, "states", "b.shim"), "w") as f:
        f.write(lines(3, "b") + "SHIM\n")
    write_tsv(
        os.path.join(runs, "transitions.tsv"),
        TRANSITION_COLUMNS,
        [
            dict(
                id="X1",
                entry="-",
                added_in="01",
                removed_in="03",
                reason="keep 02 compiling",
                path="b.txt",
                stored="states/b.shim",
            )
        ],
    )

    p = script(
        "check_plan.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/t/",
        check_rc=False,
    )
    print(p.stdout.decode().strip())
    check(p.returncode == 0, "check_plan passes on a complete plan")

    keep = [x for x in asg if x["entry"] != ids("added.txt")[0]]
    write_tsv(os.path.join(runs, "plan_assign.tsv"), ASSIGN_COLUMNS, keep)
    p = script(
        "check_plan.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/t/",
        check_rc=False,
    )
    check(
        p.returncode == 1 and b"is not assigned" in p.stdout,
        "check_plan fails on an unassigned entry",
    )
    write_tsv(os.path.join(runs, "plan_assign.tsv"), ASSIGN_COLUMNS, asg)

    p = script(
        "build_stack.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/t/",
        "--refs",
        "create",
        "--date",
        "2020-01-01T00:00:00+0000",
        env=GENV,
        check_rc=False,
    )
    print(p.stdout.decode().strip(), p.stderr.decode().strip())
    check(
        p.returncode == 0 and b"OK: tree(top) == tree(tip)" in p.stdout,
        "build_stack reaches tree(top) == tree(tip)",
    )
    _, stack = read_tsv(os.path.join(runs, "stack.tsv"))
    s = {x["NN"]: x["sha"] for x in stack}
    p = script(
        "verify_stack.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/t/",
        "--refs",
        check_rc=False,
    )
    print(p.stdout.decode().strip())
    check(p.returncode == 0, "verify_stack passes")

    p = script(
        "render_plan.py",
        "--run",
        runs,
        "--prefix",
        "users/t/",
        "--repo",
        repo,
        "--table",
        "--transitions",
        "--states",
        check_rc=False,
    )
    out = p.stdout.decode()
    check(
        p.returncode == 0
        and "| # | type | branch suffix | description |" in out
        and "| 01 | fix | fix-bound | Fixes the loop bound. |" in out
        and "| X1 | b.txt | 01 | 03 |" in out
        and "+  for (i = 0; i < n; i++) use(old_name[i]);" in out,
        "render_plan prints the four-column table, the transition list and the stored-state diff",
    )

    a = base_a.split("\n")[:-1]
    g1 = "\n".join(a[:3] + ["INS_AFTER_3"] + a[3:]) + "\n"
    g2 = "\n".join(a[:3] + ["INS_AFTER_3"] + a[3:10] + ["INS_AFTER_10"] + a[10:]) + "\n"
    check(
        blob_at(repo, s["01"], "a.txt") == g1.encode(),
        "01: only the first insertion, at the right line",
    )
    check(
        blob_at(repo, s["02"], "a.txt") == g2.encode(),
        "02: second insertion lands after base line 10 although 01 shifted the file",
    )
    check(
        blob_at(repo, s["03"], "a.txt") == blob_at(repo, tip, "a.txt"),
        "03: a.txt equals the source",
    )
    check(
        blob_at(repo, s["01"], "loop.cpp")
        == loop_base.replace("i <= n", "i < n").encode(),
        "01: loop.cpp has the fix but the old name",
    )
    check(
        blob_at(repo, s["02"], "loop.cpp") == blob_at(repo, tip, "loop.cpp"),
        "02: loop.cpp is final",
    )
    check(
        blob_at(repo, s["01"], "del.txt") is None
        and blob_at(repo, base, "del.txt") is not None,
        "01: del.txt is deleted",
    )
    check(
        blob_at(repo, s["01"], "old/name.txt") is not None
        and blob_at(repo, s["01"], "new/name.txt") is None,
        "01: rename not applied yet",
    )
    check(
        blob_at(repo, s["02"], "old/name.txt") is None
        and blob_at(repo, s["02"], "new/name.txt") is not None,
        "02: rename applied",
    )
    check(
        blob_at(repo, s["01"], "bin.dat") == b"\x00\x01\x02base"
        and blob_at(repo, s["02"], "bin.dat") == b"\x00\x01\x02tip",
        "binary file changes at 02",
    )
    check(
        mode_at(repo, s["02"], "run.sh") == "100644"
        and mode_at(repo, s["03"], "run.sh") == "100755",
        "mode change lands at 03",
    )
    check(blob_at(repo, s["02"], "noeol.txt") == b"x\ny", "02: noeol.txt untouched")
    check(
        blob_at(repo, s["03"], "noeol.txt") == b"x\ny\nz",
        "03: no-newline-at-EOF text preserved",
    )
    check(
        mode_at(repo, s["03"], "lnk") == "120000", "type change to symlink lands at 03"
    )
    check(
        (blob_at(repo, s["01"], "b.txt") or b"").endswith(b"SHIM\n")
        and (blob_at(repo, s["02"], "b.txt") or b"").endswith(b"SHIM\n")
        and blob_at(repo, s["03"], "b.txt") == blob_at(repo, base, "b.txt"),
        "transition code present in 01-02, gone in 03",
    )
    check(
        blob_at(repo, s["03"], "gen/three.yaml") is None
        and blob_at(repo, s["04"], "gen/three.yaml") is not None,
        "generated set lands whole at 04",
    )
    msg = git(repo, "log", "-1", "--format=%an <%ae>%n%s%n%b", s["02"])
    check(
        msg.startswith("A B <ab@example.com>\nrefactor(x): rename old_name"),
        "commit author and subject come from groups.tsv",
    )
    check("Fixes #1" in msg, "trailers are added to the message")

    p = script(
        "build_stack.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/t/",
        "--refs",
        "update",
        env=GENV,
        check_rc=False,
    )
    _, stack2 = read_tsv(os.path.join(runs, "stack.tsv"))
    check(
        p.returncode == 0 and [x["sha"] for x in stack2] == [x["sha"] for x in stack],
        "rebuild is deterministic (same shas)",
    )

    asg2 = [dict(x) for x in asg]
    for x in asg2:
        if x["entry"] == a_ids[1]:
            x["group"] = "03"
    write_tsv(os.path.join(runs, "plan_assign.tsv"), ASSIGN_COLUMNS, asg2)
    p = script(
        "build_stack.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/t/",
        "--refs",
        "update",
        env=GENV,
        check_rc=False,
    )
    check(p.returncode == 0, "plan edit + rebuild works and moves the refs")
    _, stack3 = read_tsv(os.path.join(runs, "stack.tsv"))
    check(
        stack3[0]["sha"] == stack[0]["sha"] and stack3[1]["sha"] != stack[1]["sha"],
        "lower unchanged branch keeps its sha; the changed branch and above are new",
    )
    check(
        blob_at(repo, stack3[1]["sha"], "a.txt") == g1.encode(),
        "after the move, 02 has only the first insertion",
    )
    p = script(
        "verify_stack.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/t/",
        "--refs",
        check_rc=False,
    )
    check(p.returncode == 0, "verify_stack passes after the rebuild")

    git(repo, "branch", "-f", "users/t/feature", stack[0]["sha"])
    p = script(
        "build_stack.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/t/",
        "--refs",
        "update",
        check_rc=False,
        env=GENV,
    )
    check(
        p.returncode != 0,
        "update refuses when a branch no longer matches the recorded sha",
    )
    tmp.cleanup()


# ---------------------------------------------------------------- regression cases


def make_case(base_files, tip_files, modes=None, extra_repo_cmds=None):
    tmp = tempfile.TemporaryDirectory(prefix="split-case-")
    repo = os.path.join(tmp.name, "repo")
    os.makedirs(repo)
    git(repo, "init", "-q", env=GENV)
    for p, d in base_files.items():
        write(repo, p, d)
    git(repo, "add", "-A", env=GENV)
    git(repo, "commit", "-q", "--allow-empty", "-m", "base", env=GENV)
    base = git(repo, "rev-parse", "HEAD").strip()
    for p in base_files:
        if p not in tip_files:
            os.remove(os.path.join(repo, p))
    for p, d in tip_files.items():
        write(repo, p, d, (modes or {}).get(p))
    git(repo, "add", "-A", env=GENV)
    for cmd in extra_repo_cmds or []:
        git(repo, *cmd, env=GENV)
    git(repo, "commit", "-q", "-m", "tip", env=GENV)
    tip = git(repo, "rev-parse", "HEAD").strip()
    runs = os.path.join(tmp.name, "run")
    os.makedirs(os.path.join(runs, "states"))
    return tmp, repo, base, tip, runs


def mkgroups(n, suffixes=None):
    return [
        dict(
            NN="%02d" % (i + 1),
            suffix=(suffixes or ["g%d" % (i + 1) for i in range(n)])[i],
            type="feat",
            scope="",
            subject="s%d" % (i + 1),
            author="",
            trailers="",
            depends_on="",
            summary="group %d" % (i + 1),
        )
        for i in range(n)
    ]


def ledger_rows(repo, base, tip, runs, *extra):
    script(
        "build_ledger.py",
        "--repo",
        repo,
        "--base",
        base,
        "--tip",
        tip,
        "--out",
        os.path.join(runs, "ledger.tsv"),
        *extra
    )
    _, rows = read_tsv(os.path.join(runs, "ledger.tsv"), LEDGER_COLUMNS)
    by = {}
    for r in rows:
        by.setdefault(r["path"], []).append(r)
    return rows, by


def asgrow(entry, step, nsteps, group, state="final"):
    return dict(
        entry=entry,
        step=str(step),
        nsteps=str(nsteps),
        group=group,
        state=state,
        purpose="p",
        note="",
    )


def plan(runs, groups, asg, trans=None):
    write_tsv(os.path.join(runs, "groups.tsv"), GROUP_COLUMNS, groups)
    write_tsv(os.path.join(runs, "plan_assign.tsv"), ASSIGN_COLUMNS, asg)
    tp = os.path.join(runs, "transitions.tsv")
    if trans:
        write_tsv(tp, TRANSITION_COLUMNS, trans)
    elif os.path.exists(tp):
        os.remove(tp)


def chk(repo, runs, *extra):
    return script(
        "check_plan.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/t/",
        *extra,
        check_rc=False
    )


def bld(repo, runs, *extra, env=None):
    e = dict(GENV)
    e.update(env or {})
    return script(
        "build_stack.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/t/",
        *extra,
        check_rc=False,
        env=e
    )


def test_regressions():
    # file <-> directory replaced across groups: must be refused, nothing recorded
    tmp, repo, base, tip, runs = make_case({"p": "x\n"}, {"p/x": "y\n"})
    rows, by = ledger_rows(repo, base, tip, runs)
    plan(
        runs,
        mkgroups(2),
        [asgrow(by["p/x"][0]["id"], 1, 1, "01"), asgrow(by["p"][0]["id"], 1, 1, "02")],
    )
    p = bld(repo, runs)
    check(
        p.returncode != 0
        and b"file and a directory" in p.stderr
        and not os.path.exists(os.path.join(runs, "stack.tsv")),
        "a file/directory clash across groups is refused and nothing is recorded",
    )
    plan(
        runs,
        mkgroups(2),
        [asgrow(by["p"][0]["id"], 1, 1, "01"), asgrow(by["p/x"][0]["id"], 1, 1, "02")],
    )
    check(
        bld(repo, runs).returncode == 0,
        "the same change in a valid order (delete first) builds",
    )
    tmp.cleanup()

    # a deleted file with a stored intermediate state keeps that state until its final step
    tmp, repo, base, tip, runs = make_case(
        {"d.txt": "1\n", "k.txt": "k\n"}, {"k.txt": "k2\n"}
    )
    rows, by = ledger_rows(repo, base, tip, runs)
    with open(os.path.join(runs, "states", "d1"), "w") as f:
        f.write("mid\n")
    plan(
        runs,
        mkgroups(2),
        [
            asgrow(by["d.txt"][0]["id"], 1, 2, "01", "stored:states/d1"),
            asgrow(by["d.txt"][0]["id"], 2, 2, "02"),
            asgrow(by["k.txt"][0]["id"], 1, 1, "02"),
        ],
    )
    p = bld(repo, runs, "--refs", "create")
    _, st = read_tsv(os.path.join(runs, "stack.tsv"))
    check(
        p.returncode == 0
        and blob_at(repo, st[0]["sha"], "d.txt") == b"mid\n"
        and blob_at(repo, st[1]["sha"], "d.txt") is None,
        "a deleted file's intermediate state is applied",
    )
    tmp.cleanup()

    # paths with glob characters get only their own hunks
    body = lines(8)
    mod1 = body.replace("line2\n", "L2\n").replace("line6\n", "L6\n")
    mod2 = body.replace("line3\n", "L3\n")
    tmp, repo, base, tip, runs = make_case(
        {"[ab].txt": body, "a.txt": body}, {"[ab].txt": mod1, "a.txt": mod2}
    )
    rows, by = ledger_rows(repo, base, tip, runs)
    check(
        len(by["[ab].txt"]) == 2 and len(by["a.txt"]) == 1,
        "glob characters in a path do not pull in other files' hunks",
    )
    tmp.cleanup()
    tmp, repo, base, tip, runs = make_case({":c.txt": body}, {":c.txt": mod1})
    rows, by = ledger_rows(repo, base, tip, runs)
    check(
        len(by[":c.txt"]) == 2 and all(r["id"][0] == "H" for r in by[":c.txt"]),
        "a path starting with ':' still yields hunks",
    )
    tmp.cleanup()

    # a ledger made from a subdirectory of the repository is the same as one from the top level
    tmp, repo, base, tip, runs = make_case({"src/a.txt": body}, {"src/a.txt": mod1})
    rows, by = ledger_rows(os.path.join(repo, "src"), base, tip, runs)
    check(
        len(by["src/a.txt"]) == 2 and by["src/a.txt"][0]["id"][0] == "H",
        "--repo may be a subdirectory",
    )
    tmp.cleanup()

    # a rename with a mode change is one row
    tmp, repo, base, tip, runs = make_case(
        {"r.sh": "echo 1\necho 2\necho 3\necho 4\n"},
        {"r2.sh": "echo 1\necho 2\necho 3\necho 5\n"},
        modes={"r2.sh": 0o755},
    )
    rows, by = ledger_rows(repo, base, tip, runs)
    check(
        len(by["r2.sh"]) == 1
        and by["r2.sh"][0]["status"] == "R"
        and by["r2.sh"][0]["mode"] == "100755",
        "a rename with a mode change is a single whole-file row",
    )
    tmp.cleanup()

    # failed build records nothing; a plain rebuild does not break --refs update; refs move atomically
    tmp, repo, base, tip, runs = make_case(
        {"a.txt": "1\n2\n", "b.txt": "x\n"}, {"a.txt": "1\nTWO\n", "b.txt": "y\n"}
    )
    rows, by = ledger_rows(repo, base, tip, runs)
    ia, ib = by["a.txt"][0]["id"], by["b.txt"][0]["id"]
    with open(os.path.join(runs, "states", "bad"), "w") as f:
        f.write("WRONG\n")
    plan(
        runs,
        mkgroups(2),
        [asgrow(ia, 1, 1, "01", "stored:states/bad"), asgrow(ib, 1, 1, "02")],
    )
    p = bld(repo, runs)
    check(
        p.returncode == 1 and not os.path.exists(os.path.join(runs, "stack.tsv")),
        "a build whose top tree differs from the tip records nothing",
    )
    plan(runs, mkgroups(2), [asgrow(ia, 1, 1, "01"), asgrow(ib, 1, 1, "02")])
    check(bld(repo, runs, "--refs", "create").returncode == 0, "refs created")
    plan(runs, mkgroups(2), [asgrow(ia, 1, 1, "02"), asgrow(ib, 1, 1, "01")])
    check(bld(repo, runs).returncode == 0, "a rebuild without refs works")
    check(
        bld(repo, runs, "--refs", "update").returncode == 0,
        "--refs update still works after a plain rebuild",
    )
    git(
        repo, "branch", "-f", "users/t/g2", git(repo, "rev-parse", "users/t/g1").strip()
    )
    before = git(repo, "rev-parse", "users/t/g1").strip()
    plan(runs, mkgroups(2), [asgrow(ia, 1, 1, "01"), asgrow(ib, 1, 1, "02")])
    p = bld(repo, runs, "--refs", "update")
    check(
        p.returncode != 0 and git(repo, "rev-parse", "users/t/g1").strip() == before,
        "an update that cannot move every branch moves none",
    )
    plan(
        runs,
        mkgroups(2, ["feat", "feat/more"]),
        [asgrow(ia, 1, 1, "01"), asgrow(ib, 1, 1, "02")],
    )
    p = chk(repo, runs)
    check(
        p.returncode == 1 and b"cannot both exist" in p.stdout,
        "suffixes feat and feat/more are rejected",
    )
    g = mkgroups(2)
    g[0]["author"] = "Just A Name"
    plan(runs, g, [asgrow(ia, 1, 1, "01"), asgrow(ib, 1, 1, "02")])
    p = chk(repo, runs)
    check(
        p.returncode == 1 and b"author must look like" in p.stdout,
        "an author without an email is rejected by check_plan",
    )
    plan(runs, mkgroups(2), [asgrow(ia, 1, 1, "01"), asgrow(ib, 1, 1, "02")])
    other = tempfile.TemporaryDirectory(prefix="split-other-")
    git(other.name, "init", "-q", env=GENV)
    p = bld(repo, runs, env={"GIT_DIR": os.path.join(other.name, ".git")})
    check(
        p.returncode == 0, "a GIT_DIR in the environment does not redirect the scripts"
    )
    if p.returncode != 0:
        print(p.stderr.decode()[:400])
    other.cleanup()
    tmp.cleanup()

    # a transition may add a temporary file that exists at neither end; overlapping windows are rejected
    tmp, repo, base, tip, runs = make_case(
        {"a.txt": "1\n"}, {"a.txt": "2\n", "b.txt": "n\n"}
    )
    rows, by = ledger_rows(repo, base, tip, runs)
    with open(os.path.join(runs, "states", "shim"), "w") as f:
        f.write("shim\n")
    base_asg = [
        asgrow(by["a.txt"][0]["id"], 1, 1, "01"),
        asgrow(by["b.txt"][0]["id"], 1, 1, "03"),
    ]
    trans = [
        dict(
            id="X1",
            entry="-",
            added_in="01",
            removed_in="02",
            reason="temporary shim",
            path="compat/shim.h",
            stored="states/shim",
        )
    ]
    plan(
        runs,
        mkgroups(3),
        base_asg
        + [asgrow("H0", 1, 1, "02")][:0]
        + [asgrow(by["a.txt"][0]["id"], 1, 1, "01")][:0],
        trans,
    )
    p = chk(repo, runs)
    check(
        "group 02 has no assigned entries" in p.stdout.decode()
        and b"shim" not in p.stdout,
        "a transition on a new temporary path is accepted (the empty group is the only finding)",
    )
    plan(
        runs,
        mkgroups(2),
        [
            asgrow(by["a.txt"][0]["id"], 1, 1, "01"),
            asgrow(by["b.txt"][0]["id"], 1, 1, "02"),
        ],
        trans,
    )
    p = bld(repo, runs, "--refs", "create")
    _, st = read_tsv(os.path.join(runs, "stack.tsv"))
    check(
        p.returncode == 0
        and blob_at(repo, st[0]["sha"], "compat/shim.h") == b"shim\n"
        and blob_at(repo, st[1]["sha"], "compat/shim.h") is None,
        "a temporary new file lives in its window only",
    )
    trans2 = trans + [dict(trans[0], id="X2")]
    plan(
        runs,
        mkgroups(2),
        [
            asgrow(by["a.txt"][0]["id"], 1, 1, "01"),
            asgrow(by["b.txt"][0]["id"], 1, 1, "02"),
        ],
        trans2,
    )
    p = chk(repo, runs)
    check(
        p.returncode == 1 and b"overlap" in p.stdout,
        "overlapping transition windows are rejected",
    )
    tmp.cleanup()

    # a ledger with rows removed is caught by check_plan (mode-only and empty-file rows have no line counts)
    tmp, repo, base, tip, runs = make_case(
        {"s.sh": "echo\n", "k.txt": "a\n"},
        {"s.sh": "echo\n", "k.txt": "b\n", "empty.txt": ""},
        modes={"s.sh": 0o755},
    )
    rows, by = ledger_rows(repo, base, tip, runs)
    keep = [r for r in rows if r["path"] == "k.txt"]
    data = open(os.path.join(runs, "ledger.tsv"), "rb").read().split(b"\n")
    hdr = [ln for ln in data if ln.startswith(b"#") or ln.startswith(b"id\t")]
    body = [
        ln
        for ln in data
        if ln
        and not ln.startswith(b"#")
        and not ln.startswith(b"id\t")
        and b"k.txt" in ln
    ]
    open(os.path.join(runs, "ledger.tsv"), "wb").write(b"\n".join(hdr + body) + b"\n")
    plan(runs, mkgroups(1), [asgrow(r["id"], 1, 1, "01") for r in keep])
    p = chk(repo, runs)
    check(
        p.returncode == 1 and b"differs from a fresh run" in p.stdout,
        "a ledger with rows removed fails check_plan",
    )
    tmp.cleanup()

    # stored-state hygiene
    tmp, repo, base, tip, runs = make_case(
        {"a.txt": lines(4)}, {"a.txt": lines(4).replace("line2\n", "L2\n")}
    )
    rows, by = ledger_rows(repo, base, tip, runs)
    with open(os.path.join(runs, "states", "nonl"), "wb") as f:
        f.write(b"L2")
    plan(
        runs,
        mkgroups(2),
        [
            asgrow(by["a.txt"][0]["id"], 1, 2, "01", "stored:states/nonl"),
            asgrow(by["a.txt"][0]["id"], 2, 2, "02"),
        ],
    )
    p = chk(repo, runs)
    check(
        p.returncode == 1 and b"must end with a newline" in p.stdout,
        "a stored hunk state without a final newline is rejected",
    )
    plan(
        runs,
        mkgroups(2),
        [
            asgrow(by["a.txt"][0]["id"], 1, 2, "01", "stored:/etc/hostname"),
            asgrow(by["a.txt"][0]["id"], 2, 2, "02"),
        ],
    )
    p = chk(repo, runs)
    check(
        p.returncode == 1 and b"escapes the run directory" in p.stdout,
        "a stored path outside the run directory is rejected",
    )
    tmp.cleanup()

    # a submodule pointer cannot take a stored state
    tmp, repo, base, tip, runs = make_case(
        {"a.txt": "1\n"},
        {"a.txt": "1\n"},
        extra_repo_cmds=[
            ("update-index", "--add", "--cacheinfo", "160000,%s,sub" % ("1" * 40))
        ],
    )
    rows, by = ledger_rows(repo, base, tip, runs)
    with open(os.path.join(runs, "states", "s"), "w") as f:
        f.write("x\n")
    plan(
        runs,
        mkgroups(2),
        [
            asgrow(by["sub"][0]["id"], 1, 2, "01", "stored:states/s"),
            asgrow(by["sub"][0]["id"], 2, 2, "02"),
        ],
    )
    p = chk(repo, runs)
    check(
        p.returncode == 1 and b"submodule" in p.stdout,
        "a stored state on a submodule entry is rejected",
    )
    tmp.cleanup()

    # verify_stack notices a squashed stack and a wrong middle commit
    tmp, repo, base, tip, runs = make_case(
        {"a.txt": "1\n2\n", "b.txt": "x\n"}, {"a.txt": "1\nTWO\n", "b.txt": "y\n"}
    )
    rows, by = ledger_rows(repo, base, tip, runs)
    plan(
        runs,
        mkgroups(2),
        [
            asgrow(by["a.txt"][0]["id"], 1, 1, "01"),
            asgrow(by["b.txt"][0]["id"], 1, 1, "02"),
        ],
    )
    check(bld(repo, runs).returncode == 0, "build for the verify checks")
    _, st = read_tsv(os.path.join(runs, "stack.tsv"))
    tree = git(repo, "rev-parse", tip + "^{tree}").strip()
    squashed = git(
        repo, "commit-tree", tree, "-p", base, "-m", "squash", env=GENV
    ).strip()
    write_tsv(
        os.path.join(runs, "stack.tsv"),
        ["NN", "branch", "sha", "tree"],
        [dict(NN="01", branch="users/t/g1", sha=squashed, tree=tree)],
    )
    p = script(
        "verify_stack.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/t/",
        check_rc=False,
    )
    check(
        p.returncode == 1 and b"differ from groups.tsv" in p.stdout,
        "verify_stack rejects a squashed stack",
    )
    tmp.cleanup()


def forge(repo, parent, changes, msg="forged"):
    """Commit on `parent` with `changes` {path: bytes|None} using plumbing; returns (sha, tree)."""
    tmp = tempfile.mkdtemp(prefix="split-forge-")
    env = dict(GENV, GIT_INDEX_FILE=os.path.join(tmp, "index"))
    git(repo, "read-tree", parent, env=env)
    for p, data in changes.items():
        if data is None:
            git(repo, "update-index", "--force-remove", "--", p, env=env)
        else:
            blob = (
                run(
                    ["git", "-C", repo, "hash-object", "-w", "--stdin"],
                    env=GENV,
                    check_rc=True,
                )
                if False
                else None
            )
            proc = subprocess.run(
                ["git", "-C", repo, "hash-object", "-w", "--stdin"],
                input=data,
                stdout=subprocess.PIPE,
                env=dict(os.environ, **GENV),
            )
            git(
                repo,
                "update-index",
                "--add",
                "--cacheinfo",
                "100644,%s,%s" % (proc.stdout.decode().strip(), p),
                env=env,
            )
    tree = git(repo, "write-tree", env=env).strip()
    sha = git(repo, "commit-tree", tree, "-p", parent, "-m", msg, env=GENV).strip()
    return sha, tree


def set_stack(runs, items):
    write_tsv(
        os.path.join(runs, "stack.tsv"),
        ["NN", "branch", "sha", "tree"],
        [dict(NN=n, branch=b, sha=s, tree=t) for n, b, s, t in items],
    )


def verify(repo, runs, *extra):
    return script(
        "verify_stack.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/t/",
        *extra,
        check_rc=False
    )


def test_regressions_2():
    body = lines(9)
    # --- verify_stack negatives -------------------------------------------------------------
    a_tip = (
        body.replace("line2\n", "L2\n")
        .replace("line5\n", "L5\n")
        .replace("line8\n", "L8\n")
    )
    tmp, repo, base, tip, runs = make_case(
        {"a.txt": body, "b.txt": "x\n", "c.txt": "c\n"},
        {"a.txt": a_tip, "b.txt": "y\n", "c.txt": "c\n"},
    )
    rows, by = ledger_rows(repo, base, tip, runs)
    ah = [r["id"] for r in by["a.txt"]]
    ib = by["b.txt"][0]["id"]
    plan(
        runs,
        mkgroups(2),
        [
            asgrow(ah[0], 1, 1, "01"),
            asgrow(ah[1], 1, 1, "02"),
            asgrow(ah[2], 1, 1, "02"),
            asgrow(ib, 1, 1, "02"),
        ],
    )
    check(
        bld(repo, runs, "--refs", "create").returncode == 0,
        "plan for the verify negatives builds",
    )
    _, st = read_tsv(os.path.join(runs, "stack.tsv"))
    good = [(x["NN"], x["branch"], x["sha"], x["tree"]) for x in st]
    check(
        verify(repo, runs, "--refs").returncode == 0,
        "verify_stack passes the genuine stack (with --refs)",
    )
    tip_a = blob_at(repo, tip, "a.txt")
    tip_b = blob_at(repo, tip, "b.txt")
    # leak: 01 already holds every hunk of a.txt
    s1, t1 = forge(repo, base, {"a.txt": tip_a})
    s2, t2 = forge(repo, s1, {"b.txt": tip_b})
    set_stack(runs, [("01", good[0][1], s1, t1), ("02", good[1][1], s2, t2)])
    p = verify(repo, runs)
    check(
        p.returncode == 1 and b"does not match the plan's hunks" in p.stdout,
        "verify_stack catches a hunk that leaked into a lower branch",
    )
    # extra path changed
    s1, t1 = forge(
        repo,
        base,
        {"a.txt": blob_at(repo, st[0]["sha"], "a.txt"), "c.txt": b"changed\n"},
    )
    s2, t2 = forge(
        repo,
        s1,
        {"a.txt": tip_a, "b.txt": tip_b, "c.txt": blob_at(repo, base, "c.txt")},
    )
    set_stack(runs, [("01", good[0][1], s1, t1), ("02", good[1][1], s2, t2)])
    p = verify(repo, runs)
    check(
        p.returncode == 1 and b"does not assign to this group" in p.stdout,
        "verify_stack catches an unplanned path",
    )
    # non-linear parent
    s1 = good[0][2]
    s2, t2 = forge(repo, base, {"a.txt": tip_a, "b.txt": tip_b})
    set_stack(runs, [("01", good[0][1], s1, good[0][3]), ("02", good[1][1], s2, t2)])
    p = verify(repo, runs)
    check(
        p.returncode == 1 and b"parents" in p.stdout,
        "verify_stack catches a wrong parent",
    )
    # empty group commit
    s1, t1 = forge(repo, base, {})
    s2, t2 = forge(repo, s1, {"a.txt": tip_a, "b.txt": tip_b})
    set_stack(runs, [("01", good[0][1], s1, t1), ("02", good[1][1], s2, t2)])
    p = verify(repo, runs)
    check(
        p.returncode == 1 and b"changes nothing" in p.stdout,
        "verify_stack catches an empty group",
    )
    # refs mismatch and an inherited GIT_DIR
    set_stack(runs, good)
    git(repo, "branch", "-f", good[0][1], base)
    p = verify(repo, runs, "--refs")
    check(
        p.returncode == 1 and b"is " in p.stdout and b"expected" in p.stdout,
        "verify_stack --refs catches a moved branch",
    )
    git(repo, "branch", "-f", good[0][1], good[0][2])
    other = tempfile.TemporaryDirectory(prefix="split-other-")
    git(other.name, "init", "-q", env=GENV)
    e = {"GIT_DIR": os.path.join(other.name, ".git")}
    check(
        script(
            "verify_stack.py",
            "--repo",
            repo,
            "--run",
            runs,
            "--prefix",
            "users/t/",
            "--refs",
            check_rc=False,
            env=e,
        ).returncode
        == 0
        and chk(repo, runs, "--no-regen").returncode in (0, 1),
        "an inherited GIT_DIR does not redirect verify_stack",
    )
    other.cleanup()
    # a branch checked out in a worktree is refused
    wt = tempfile.mkdtemp(prefix="split-wt-")
    os.rmdir(wt)
    git(repo, "worktree", "add", "-q", wt, good[1][1])
    plan_ok = [
        asgrow(ah[0], 1, 1, "01"),
        asgrow(ah[1], 1, 1, "02"),
        asgrow(ah[2], 1, 1, "02"),
        asgrow(ib, 1, 1, "02"),
    ]
    plan(runs, mkgroups(2), plan_ok)
    p = bld(repo, runs, "--refs", "update")
    check(
        p.returncode != 0 and b"checked out in a worktree" in p.stderr,
        "build_stack refuses a branch checked out in a worktree",
    )
    git(repo, "worktree", "remove", "--force", wt)
    tmp.cleanup()

    # --- check_plan negatives ----------------------------------------------------------------
    tmp, repo, base, tip, runs = make_case(
        {"a.txt": body, "b.txt": "x\n", "g1.yaml": "1\n", "g2.yaml": "2\n"},
        {"a.txt": a_tip, "b.txt": "y\n", "g1.yaml": "11\n", "g2.yaml": "22\n"},
    )
    rows, by = ledger_rows(repo, base, tip, runs, "--generated-set", "gen=*.yaml")
    ah = [r["id"] for r in by["a.txt"]]
    ib = by["b.txt"][0]["id"]
    g1, g2 = by["g1.yaml"][0]["id"], by["g2.yaml"][0]["id"]

    def expect(groups, asg, needle, msg, trans=None, *extra):
        plan(runs, groups, asg, trans)
        p = chk(repo, runs, *extra)
        check(p.returncode == 1 and needle in p.stdout, msg)

    full = [
        asgrow(ah[0], 1, 1, "01"),
        asgrow(ah[1], 1, 1, "01"),
        asgrow(ah[2], 1, 1, "01"),
        asgrow(ib, 1, 1, "02"),
        asgrow("@gen", 1, 1, "03"),
    ]
    plan(runs, mkgroups(3), full)
    check(chk(repo, runs).returncode == 0, "baseline plan passes check_plan")
    expect(
        mkgroups(3),
        full[:4] + [asgrow(g1, 1, 1, "03"), asgrow(g2, 1, 1, "02")],
        b"split over groups",
        "check_plan rejects a generated set split over groups",
    )
    expect(
        mkgroups(3),
        [
            asgrow(ah[0], 1, 2, "01", "stored:states/x"),
            asgrow(ah[0], 2, 2, "02", "stored:states/x"),
        ]
        + full[1:],
        b"last step must be `final`",
        "check_plan rejects a last step that is not final",
    )
    expect(
        mkgroups(3),
        [asgrow(ah[0], 1, 2, "01"), asgrow(ah[0], 2, 2, "01")] + full[1:],
        b"strictly increase",
        "check_plan rejects two steps in one group",
    )
    g = mkgroups(3)
    g[0]["type"] = "feature"
    expect(g, full, b"not in", "check_plan rejects a type outside the closed list")
    g = mkgroups(3)
    g[0]["depends_on"] = "03"
    expect(
        g, full, b"not below it", "check_plan rejects a dependency on a higher group"
    )
    git(repo, "branch", "users/t/g2", base)
    expect(
        mkgroups(3),
        full,
        b"already exists locally",
        "check_plan rejects a branch name that already exists",
    )
    git(repo, "branch", "-D", "users/t/g2")
    git(repo, "branch", "users/t/s", base)
    expect(
        mkgroups(3, ["s/g1", "s/g2", "s/g3"]),
        full,
        b"cannot be created",
        "check_plan rejects a ref namespace clash",
    )
    git(repo, "branch", "-D", "users/t/s")
    p = script(
        "check_plan.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/t",
        check_rc=False,
    )
    check(
        p.returncode != 0 and b"must end with" in p.stderr,
        "check_plan requires the prefix to end with a slash",
    )
    plan(runs, mkgroups(3), full)
    p = chk(repo, runs, "--check-remote", "nosuchremote")
    check(
        p.returncode == 1 and b"cannot list heads" in p.stdout,
        "an unreachable remote is an error, not a pass",
    )
    d = [r for r in rows if r["path"] == "b.txt"]
    # mode-only entry with a stored state
    tmp2, repo2, base2, tip2, runs2 = make_case(
        {"s.sh": "echo\n"}, {"s.sh": "echo\n"}, modes={"s.sh": 0o755}
    )
    rows2, by2 = ledger_rows(repo2, base2, tip2, runs2)
    with open(os.path.join(runs2, "states", "m"), "w") as f:
        f.write("x\n")
    plan(
        runs2,
        mkgroups(2),
        [
            asgrow(by2["s.sh"][0]["id"], 1, 2, "01", "stored:states/m"),
            asgrow(by2["s.sh"][0]["id"], 2, 2, "02"),
        ],
    )
    p = chk(repo2, runs2)
    check(
        p.returncode == 1 and b"mode-only entry" in p.stdout,
        "check_plan rejects a stored state on a mode-only entry",
    )
    tmp2.cleanup()
    # deletion inside a transition window; transition under a file; non-numeric NN
    tmp3, repo3, base3, tip3, runs3 = make_case(
        {"d.txt": "1\n", "k.txt": "k\n", "p": "p\n"}, {"k.txt": "k2\n", "p": "p\n"}
    )
    rows3, by3 = ledger_rows(repo3, base3, tip3, runs3)
    with open(os.path.join(runs3, "states", "t"), "w") as f:
        f.write("t\n")
    tr = [
        dict(
            id="X1",
            entry="-",
            added_in="01",
            removed_in="03",
            reason="r",
            path="d.txt",
            stored="states/t",
        )
    ]
    plan(
        runs3,
        mkgroups(3),
        [
            asgrow(by3["d.txt"][0]["id"], 1, 1, "02"),
            asgrow(by3["k.txt"][0]["id"], 1, 1, "03"),
        ],
        tr,
    )
    p = chk(repo3, runs3)
    check(
        p.returncode == 1 and b"inside the window" in p.stdout,
        "check_plan rejects a deletion inside a transition window",
    )
    tr = [
        dict(
            id="X1",
            entry="-",
            added_in="01",
            removed_in="02",
            reason="r",
            path="p/x",
            stored="states/t",
        )
    ]
    plan(
        runs3,
        mkgroups(3),
        [
            asgrow(by3["d.txt"][0]["id"], 1, 1, "02"),
            asgrow(by3["k.txt"][0]["id"], 1, 1, "03"),
        ],
        tr,
    )
    p = chk(repo3, runs3)
    check(
        p.returncode == 1 and b"lies under" in p.stdout,
        "check_plan rejects a transition path under a file",
    )
    g = mkgroups(2)
    g[1]["NN"] = "ab"
    plan(
        runs3,
        g,
        [
            asgrow(by3["d.txt"][0]["id"], 1, 1, "01"),
            asgrow(by3["k.txt"][0]["id"], 1, 1, "02"),
        ],
    )
    p = chk(repo3, runs3)
    check(
        p.returncode == 2 and b"two digits" in p.stderr,
        "a non-numeric group number is a clear error",
    )
    os.remove(os.path.join(runs3, "groups.tsv"))
    p = chk(repo3, runs3)
    check(
        p.returncode == 2 and b"cannot read" in p.stderr,
        "a missing plan file is a clear error, not a traceback",
    )
    tmp3.cleanup()
    tmp.cleanup()

    # --- ledger pinning, `-` paths, commit: states, empty groups, trailers, table escaping -------
    tmp, repo, base, tip, runs = make_case(
        {"-": "a\nb\nc\n", "dir/-": "1\n2\n3\n"},
        {"-": "a\nB\nc\n", "dir/-": "1\nTWO\n3\n"},
    )
    rows, by = ledger_rows(repo, base, tip, runs)
    check(
        "-" in by and "dir/-" in by and by["-"][0]["path"] == "-",
        "paths named `-` survive the TSV format",
    )
    plan(runs, mkgroups(1), [asgrow(r["id"], 1, 1, "01") for r in rows])
    check(
        chk(repo, runs).returncode == 0 and bld(repo, runs).returncode == 0,
        "a plan with `-` paths checks and builds",
    )
    check(
        all(len(c) == 40 for r in rows for c in r["commits"].split(",") if c),
        "the commits column holds full ids",
    )
    git(repo, "config", "core.abbrev", "12")
    check(
        chk(repo, runs).returncode == 0,
        "core.abbrev does not change the regenerated ledger",
    )
    git(repo, "config", "diff.interHunkContext", "5")
    git(repo, "config", "diff.renameLimit", "1")
    git(repo, "config", "diff.renames", "copies")
    rows_b, _ = ledger_rows(repo, base, tip, runs)
    check(
        [r["hunk"] for r in rows_b] == [r["hunk"] for r in rows],
        "diff config (interHunkContext, renames) does not change the ledger",
    )
    open(os.path.join(repo, ".git", "info", "attributes"), "w").write("* -diff\n")
    p = script(
        "build_ledger.py",
        "--repo",
        repo,
        "--base",
        base,
        "--tip",
        tip,
        "--out",
        os.path.join(runs, "l2.tsv"),
        check_rc=False,
    )
    check(b"info/attributes" in p.stderr, "a non-empty info/attributes is reported")
    tmp.cleanup()

    tmp, repo, base, tip, runs = make_case(
        {"a.txt": "1\n"}, {"a.txt": "2\n", "b.txt": "n\n"}
    )
    # a source history with an intermediate commit: b.txt is "mid" first, then "n"
    git(repo, "checkout", "-q", "-b", "hist", base, env=GENV)
    write(repo, "b.txt", "mid\n")
    git(repo, "add", "-A", env=GENV)
    git(repo, "commit", "-q", "-m", "mid", env=GENV)
    midc = git(repo, "rev-parse", "HEAD").strip()
    write(repo, "b.txt", "n\n")
    write(repo, "a.txt", "2\n")
    git(repo, "add", "-A", env=GENV)
    git(repo, "commit", "-q", "-m", "end", env=GENV)
    tip = git(repo, "rev-parse", "HEAD").strip()
    rows, by = ledger_rows(repo, base, tip, runs)
    ia, ib = by["a.txt"][0]["id"], by["b.txt"][0]["id"]
    plan(
        runs,
        mkgroups(2),
        [
            asgrow(ia, 1, 1, "02"),
            asgrow(ib, 1, 2, "01", "commit:" + midc),
            asgrow(ib, 2, 2, "02"),
        ],
    )
    p = bld(repo, runs, "--refs", "create")
    _, st = read_tsv(os.path.join(runs, "stack.tsv"))
    check(
        p.returncode == 0
        and blob_at(repo, st[0]["sha"], "b.txt") == b"mid\n"
        and blob_at(repo, st[1]["sha"], "b.txt") == b"n\n",
        "a commit: state on a whole-file entry builds",
    )
    plan(
        runs,
        mkgroups(2),
        [
            asgrow(ia, 1, 1, "02"),
            asgrow(ib, 1, 2, "01", "commit:abc123"),
            asgrow(ib, 2, 2, "02"),
        ],
    )
    p = chk(repo, runs)
    check(
        p.returncode == 1 and b"full object id" in p.stdout,
        "check_plan rejects an abbreviated commit: state",
    )
    plan(
        runs,
        mkgroups(2),
        [
            asgrow(ia, 1, 2, "01", "commit:" + base),
            asgrow(ia, 2, 2, "02"),
            asgrow(ib, 1, 1, "01"),
        ],
    )
    p = chk(repo, runs)
    check(
        p.returncode == 1 and b"only valid for whole-file" in p.stdout,
        "commit: on a hunk entry is rejected",
    )
    plan(
        runs,
        mkgroups(2),
        [
            asgrow(ia, 1, 1, "02"),
            asgrow(ib, 1, 2, "01", "commit:" + base),
            asgrow(ib, 2, 2, "02"),
        ],
    )
    p = chk(repo, runs)
    check(
        p.returncode == 1 and b"does not exist at commit" in p.stdout,
        "commit: needs the path to exist at that commit",
    )
    plan(runs, mkgroups(2), [asgrow(ia, 1, 1, "01"), asgrow(ib, 1, 1, "02")])
    ia_, ib_ = ia, ib
    # trailer escape and table escaping
    g = mkgroups(2)
    g[0]["trailers"] = "Fixes: a\\|b|Co-authored-by: X <x@y.z>"
    g[0]["summary"] = "pipe | in summary"
    plan(runs, g, [asgrow(ia, 1, 1, "01"), asgrow(ib, 1, 1, "02")])
    p = bld(repo, runs, "--refs", "update")
    _, st = read_tsv(os.path.join(runs, "stack.tsv"))
    msg = git(repo, "log", "-1", "--format=%B", st[0]["sha"])
    check(
        "Fixes: a|b\nCo-authored-by: X <x@y.z>" in msg,
        "a literal pipe in a trailer is written \\| and kept",
    )
    out = script(
        "render_plan.py", "--run", runs, "--prefix", "users/t/", check_rc=False
    ).stdout.decode()
    check("pipe \\| in summary" in out, "a pipe in a summary is escaped in the table")
    # an empty group commit is refused
    plan(
        runs,
        mkgroups(3),
        [
            asgrow(ia, 1, 2, "01", "stored:states/a1"),
            asgrow(ia, 2, 2, "02"),
            asgrow(ib, 1, 1, "03"),
        ],
    )
    with open(os.path.join(runs, "states", "a1"), "w") as f:
        f.write("2\n")
    p = bld(repo, runs)
    check(
        p.returncode != 0 and b"empty commit" in p.stderr,
        "build_stack refuses a group that changes nothing",
    )
    # a generated-set glob with ';' and a bad set name are rejected
    p = script(
        "build_ledger.py",
        "--repo",
        repo,
        "--base",
        base,
        "--tip",
        tip,
        "--out",
        os.path.join(runs, "x.tsv"),
        "--generated-set",
        "gen=a;b",
        check_rc=False,
    )
    check(p.returncode != 0, "a generated-set glob containing ';' is rejected")
    p = script(
        "build_ledger.py",
        "--repo",
        repo,
        "--base",
        base,
        "--tip",
        tip,
        "--out",
        os.path.join(runs, "x.tsv"),
        "--generated-set",
        "=*.yaml",
        check_rc=False,
    )
    check(p.returncode != 0, "an empty generated-set name is rejected")
    tmp.cleanup()


def test_regressions_3():
    # ledger pinning: user config and the checkout's attributes must not change the ledger
    old = {
        "dir/a.txt": lines(10),
        "r1.txt": lines(10, "p"),
        "r2.txt": lines(10, "q"),
        "r3.txt": lines(10, "s"),
    }
    new = {
        "dir/a.txt": lines(10).replace("line3\n", "L3\n").replace("line6\n", "L6\n"),
        "n1.txt": lines(10, "p").replace("p5\n", "P5\n"),
        "n2.txt": lines(10, "q").replace("q5\n", "Q5\n"),
        "n3.txt": lines(10, "s").replace("s5\n", "S5\n"),
    }
    tmp, repo, base, tip, runs = make_case(old, new)
    git(repo, "config", "diff.renameLimit", "1")
    git(repo, "config", "diff.interHunkContext", "5")
    git(repo, "config", "diff.relative", "true")
    write(
        repo, ".gitattributes", "*.txt -diff\n"
    )  # untracked: not part of the tip's attributes
    rows, by = ledger_rows(repo, base, tip, runs)
    check(
        len(by["dir/a.txt"]) == 2 and all(r["id"][0] == "H" for r in by["dir/a.txt"]),
        "inter-hunk context and checkout attributes do not merge or hide hunks",
    )
    check(
        all(by[p][0]["status"] == "R" for p in ("n1.txt", "n2.txt", "n3.txt")),
        "a low diff.renameLimit does not turn renames into delete + add",
    )
    tmp.cleanup()

    # transitions: removed_in below added_in; commit: with a missing object; stored state keeps the tip's mode
    tmp, repo, base, tip, runs = make_case(
        {"a.txt": "1\n"}, {"a.txt": "2\n", "x.sh": "echo\n"}, modes={"x.sh": 0o755}
    )
    rows, by = ledger_rows(repo, base, tip, runs)
    ia, ix = by["a.txt"][0]["id"], by["x.sh"][0]["id"]
    with open(os.path.join(runs, "states", "t"), "w") as f:
        f.write("t\n")
    tr = [
        dict(
            id="X1",
            entry="-",
            added_in="02",
            removed_in="01",
            reason="r",
            path="tmp.h",
            stored="states/t",
        )
    ]
    plan(runs, mkgroups(2), [asgrow(ia, 1, 1, "01"), asgrow(ix, 1, 1, "02")], tr)
    p = chk(repo, runs)
    check(
        p.returncode == 1 and b"removed_in must be above added_in" in p.stdout,
        "a transition window must run upward",
    )
    plan(
        runs,
        mkgroups(2),
        [
            asgrow(ia, 1, 1, "01"),
            asgrow(ix, 1, 2, "01", "commit:" + "e" * 40),
            asgrow(ix, 2, 2, "02"),
        ],
    )
    p = chk(repo, runs)
    check(
        p.returncode == 1 and b"not found" in p.stdout,
        "commit: with a missing object is rejected",
    )
    with open(os.path.join(runs, "states", "xs"), "w") as f:
        f.write("echo different\n")
    plan(
        runs,
        mkgroups(2),
        [
            asgrow(ia, 1, 1, "01"),
            asgrow(ix, 1, 2, "01", "stored:states/xs"),
            asgrow(ix, 2, 2, "02"),
        ],
    )
    p = bld(repo, runs)
    _, st = read_tsv(os.path.join(runs, "stack.tsv"))
    check(
        p.returncode == 0
        and mode_at(repo, st[0]["sha"], "x.sh") == "100755"
        and blob_at(repo, st[0]["sha"], "x.sh") == b"echo different\n",
        "a stored state on an added executable keeps the tip's mode",
    )
    tmp.cleanup()

    # extract_state.py: the intermediate text of a hunk region at a source commit
    tmp, repo, base, tip, runs = make_case(
        {"a.txt": "p\nx = 1\nq\n"}, {"a.txt": "p\ny = 2\nq\n"}
    )
    git(repo, "checkout", "-q", "-b", "hist2", base, env=GENV)
    write(repo, "a.txt", "p\nx = 2\nq\n")
    git(repo, "add", "-A", env=GENV)
    git(repo, "commit", "-q", "-m", "mid", env=GENV)
    midc = git(repo, "rev-parse", "HEAD").strip()
    write(repo, "a.txt", "p\ny = 2\nq\n")
    git(repo, "add", "-A", env=GENV)
    git(repo, "commit", "-q", "-m", "end", env=GENV)
    tip = git(repo, "rev-parse", "HEAD").strip()
    rows, by = ledger_rows(repo, base, tip, runs)
    hid = by["a.txt"][0]["id"]
    p = script(
        "extract_state.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--entry",
        hid,
        "--commit",
        midc,
        "--out",
        "states/mid.txt",
        check_rc=False,
    )
    check(
        p.returncode == 0
        and open(os.path.join(runs, "states", "mid.txt"), "rb").read() == b"x = 2\n",
        "extract_state reproduces the intermediate text of a hunk region",
    )
    plan(
        runs,
        mkgroups(2),
        [asgrow(hid, 1, 2, "01", "stored:states/mid.txt"), asgrow(hid, 2, 2, "02")],
    )
    p = bld(repo, runs)
    _, st = read_tsv(os.path.join(runs, "stack.tsv"))
    check(
        p.returncode == 0
        and blob_at(repo, st[0]["sha"], "a.txt") == b"p\nx = 2\nq\n"
        and chk(repo, runs).returncode == 0,
        "a state extracted from history builds the intermediate branch",
    )
    p = script(
        "extract_state.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--entry",
        "F1",
        "--commit",
        midc,
        "--out",
        "states/z.txt",
        check_rc=False,
    )
    check(p.returncode != 0, "extract_state refuses a non-hunk entry")
    tmp.cleanup()


def put_state(runs, name, data):
    with open(os.path.join(runs, "states", name), "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode())
    return "subtractive:states/" + name


def test_subtractive():
    """`subtractive:` states must be the final text with lines removed (check_plan.py 3b)."""
    tmp, repo, base, tip, runs = make_case(
        {"t.cpp": "keep\n", "gone.txt": "g\n"},
        {"t.cpp": "keep\ncase1\ncase2\ncase3\n", "tests/new_test.cpp": "A\nB\nC\n"},
    )
    rows, by = ledger_rows(repo, base, tip, runs)
    h = by["t.cpp"][0]["id"]
    f = by["tests/new_test.cpp"][0]["id"]
    d = by["gone.txt"][0]["id"]
    check(
        len(by["t.cpp"]) == 1 and h[0] == "H" and f[0] == "F",
        "subtractive case: one hunk and one added file",
    )
    g3 = mkgroups(3)

    def asg(h1, h2, f1, f2, d_row=None):
        out = [
            asgrow(h, 1, 3, "01", h1),
            asgrow(h, 2, 3, "02", h2),
            asgrow(h, 3, 3, "03"),
            asgrow(f, 1, 3, "01", f1),
            asgrow(f, 2, 3, "02", f2),
            asgrow(f, 3, 3, "03"),
        ]
        out += d_row or [asgrow(d, 1, 1, "01")]
        return out

    good = asg(
        put_state(runs, "h1", "case1\n"),
        put_state(runs, "h2", "case1\ncase2\n"),
        put_state(runs, "f1", "A\n"),
        put_state(runs, "f2", "A\nC\n"),
    )
    plan(runs, g3, good)
    p = chk(repo, runs)
    check(
        p.returncode == 0 and b"4 subtractive states checked" in p.stdout,
        "check_plan passes subtractive states (lines removed, two-step chains, H and F entries)",
    )
    p = bld(repo, runs)
    _, st = read_tsv(os.path.join(runs, "stack.tsv"))
    check(
        p.returncode == 0
        and blob_at(repo, st[0]["sha"], "t.cpp") == b"keep\ncase1\n"
        and blob_at(repo, st[1]["sha"], "tests/new_test.cpp") == b"A\nC\n"
        and blob_at(repo, st[2]["sha"], "t.cpp") == blob_at(repo, tip, "t.cpp"),
        "build_stack builds subtractive states",
    )
    p = script(
        "render_plan.py",
        "--run",
        runs,
        "--prefix",
        "users/t/",
        "--repo",
        repo,
        "--states",
        check_rc=False,
    )
    check(
        p.returncode == 0 and b"state subtractive:states/h1" in p.stdout,
        "render_plan shows subtractive states",
    )

    def expect_fail(plan_rows, needle, msg):
        plan(runs, g3, plan_rows)
        p = chk(repo, runs)
        check(p.returncode == 1 and needle in p.stdout, msg)

    h2, f1, f2 = (
        "subtractive:states/h2",
        "subtractive:states/f1",
        "subtractive:states/f2",
    )
    expect_fail(
        asg(put_state(runs, "x", "case1\nEXTRA\n"), h2, f1, f2),
        b"'EXTRA' is not in the final text",
        "an added line fails the subtractive check",
    )
    expect_fail(
        asg(
            put_state(runs, "x", "case2\ncase1\n"),
            put_state(runs, "y", "case1\ncase2\ncase3\n"),
            f1,
            f2,
        ),
        b"is not the final text with lines removed",
        "reordered lines fail the subtractive check",
    )
    expect_fail(
        asg(put_state(runs, "x", "case1 \n"), h2, f1, f2),
        b"is not the final text with lines removed",
        "a whitespace-only change fails the subtractive check",
    )
    expect_fail(
        asg(put_state(runs, "x", "case1\n"), h2, put_state(runs, "z", "A"), f2),
        b"is not the final text with lines removed",
        "a missing final newline fails the subtractive check",
    )
    expect_fail(
        asg(
            put_state(runs, "x", "case1\ncase2\n"),
            put_state(runs, "y", "case1\n"),
            f1,
            f2,
        ),
        b"is not contained in the next step's state",
        "a later step that drops a kept line fails",
    )
    expect_fail(
        asg(
            put_state(runs, "x", "case1\n"),
            h2,
            f1,
            f2,
            [
                asgrow(d, 1, 2, "01", put_state(runs, "dd", "g\n")),
                asgrow(d, 2, 2, "02"),
            ],
        ),
        b"needs a text hunk",
        "a subtractive state on a deleted file is rejected",
    )
    with open(os.path.join(runs, "states", "s"), "w") as fh:
        fh.write("case1\nEXTRA\n")
    plan(runs, g3, asg("stored:states/s", h2, f1, f2))
    p = chk(repo, runs)
    check(
        p.returncode == 0 and b"3 subtractive states checked" in p.stdout,
        "a `stored:` state is not subjected to the subtractive check",
    )
    tmp.cleanup()


def test_generated_exceptions():
    """Deleted, renamed and type-changed generated files, and --exclude-generated paths, are F rows."""
    five = lines(5, "f")
    tmp, repo, base, tip, runs = make_case(
        {
            "gen/old.s": "o\n",
            "gen/foo.s": five,
            "gen/mod.s": "m\n",
            "src/use.txt": "uses foo\n",
        },
        {
            "gen/foo_v2.s": five,
            "gen/mod.s": "m2\n",
            "gen/new.s": "n\n",
            "gen/fixture.s": "fx\n",
            "src/use.txt": "uses foo_v2\n",
            ".gitattributes": "gen/* linguist-generated\n",
        },
    )
    rows, by = ledger_rows(
        repo,
        base,
        tip,
        runs,
        "--generated-set",
        "kernels=gen/*",
        "--exclude-generated",
        "gen/fixture.s",
    )
    meta, _ = read_tsv(os.path.join(runs, "ledger.tsv"), LEDGER_COLUMNS)
    old, ren, fx = by["gen/old.s"][0], by["gen/foo_v2.s"][0], by["gen/fixture.s"][0]
    check(
        old["id"][0] == "F" and old["status"] == "D" and old["generated"] == "kernels",
        "a deleted generated file is an F row that still names its set",
    )
    check(
        ren["id"][0] == "F" and ren["status"] == "R" and ren["old_path"] == "gen/foo.s",
        "a renamed generated file is an F row with old_path",
    )
    check(
        by["gen/mod.s"][0]["id"][0] == "G" and by["gen/new.s"][0]["id"][0] == "G",
        "added and modified generated files are still G rows",
    )
    check(
        fx["id"][0] == "F" and fx["generated"] == "",
        "--exclude-generated wins over a matching set",
    )
    check(
        meta.get("generated-exclude") == "gen/fixture.s",
        "the ledger header records the exclusions",
    )

    use = by["src/use.txt"][0]["id"]
    plan(
        runs,
        mkgroups(3),
        [
            asgrow(old["id"], 1, 1, "01"),
            asgrow(ren["id"], 1, 1, "02"),
            asgrow(use, 1, 1, "02"),
            asgrow("@kernels", 1, 1, "03"),
            asgrow(fx["id"], 1, 1, "03"),
            asgrow(by[".gitattributes"][0]["id"], 1, 1, "03"),
        ],
    )
    p = chk(repo, runs)
    check(
        p.returncode == 0 and b"reproduces byte for byte" in p.stdout,
        "check_plan regenerates a ledger with exclusions byte for byte",
    )
    p = bld(repo, runs)
    _, st = read_tsv(os.path.join(runs, "stack.tsv"))
    check(
        p.returncode == 0
        and blob_at(repo, st[0]["sha"], "gen/old.s") is None
        and blob_at(repo, st[0]["sha"], "gen/new.s") is None
        and blob_at(repo, st[1]["sha"], "gen/foo_v2.s") is not None
        and blob_at(repo, st[1]["sha"], "gen/foo.s") is None
        and blob_at(repo, st[1]["sha"], "gen/mod.s") == b"m\n",
        "the deletion and the rename land in their own groups, apart from the set",
    )
    p = script(
        "verify_stack.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/t/",
        check_rc=False,
    )
    check(
        p.returncode == 0,
        "verify_stack passes with generated deletions and renames placed separately",
    )

    _, by2 = ledger_rows(
        repo,
        base,
        tip,
        runs,
        "--attr-generated",
        "--exclude-generated",
        "gen/fixture.s",
    )
    check(
        by2["gen/fixture.s"][0]["id"][0] == "F"
        and by2["gen/fixture.s"][0]["generated"] == ""
        and by2["gen/new.s"][0]["generated"] == "attr-generated",
        "--exclude-generated wins over --attr-generated",
    )
    p = script(
        "build_ledger.py",
        "--repo",
        repo,
        "--base",
        base,
        "--tip",
        tip,
        "--out",
        os.path.join(runs, "x.tsv"),
        "--exclude-generated",
        "a;b",
        check_rc=False,
    )
    check(p.returncode != 0, "an --exclude-generated glob containing ';' is rejected")
    tmp.cleanup()


def test_stress(repo, base, tip):
    tmp = tempfile.TemporaryDirectory(prefix="split-stress-")
    objs = os.path.join(tmp.name, "objects")
    os.makedirs(objs)
    common = git(
        repo, "rev-parse", "--path-format=absolute", "--git-common-dir"
    ).strip()
    env = {
        "GIT_OBJECT_DIRECTORY": objs,
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": os.path.join(common, "objects"),
        "GIT_AUTHOR_NAME": "T",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "T",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    runs = os.path.join(tmp.name, "run")
    os.makedirs(runs)
    out = script(
        "build_ledger.py",
        "--repo",
        repo,
        "--base",
        base,
        "--tip",
        tip,
        "--out",
        os.path.join(runs, "ledger.tsv"),
        "--attr-generated",
        env=env,
    ).stdout.decode()
    print(out.strip())
    _, rows = read_tsv(os.path.join(runs, "ledger.tsv"), LEDGER_COLUMNS)
    groups = [
        dict(
            NN="0%d" % i,
            suffix="g%d" % i,
            type="feat",
            scope="",
            subject="s%d" % i,
            author="",
            trailers="",
            depends_on="",
            summary="g%d" % i,
        )
        for i in (1, 2, 3)
    ]
    write_tsv(os.path.join(runs, "groups.tsv"), GROUP_COLUMNS, groups)
    asg = []
    hcount = {}
    for r in rows:
        if r["id"][0] == "H":
            hcount[r["path"]] = hcount.get(r["path"], 0) + 1
            g = hcount[r["path"]] % 3 + 1
        else:
            g = (sum(map(ord, r["path"])) % 3) + 1
        asg.append(
            dict(
                entry=r["id"],
                step="1",
                nsteps="1",
                group="0%d" % g,
                state="final",
                purpose="p",
                note="",
            )
        )
    write_tsv(os.path.join(runs, "plan_assign.tsv"), ASSIGN_COLUMNS, asg)
    p = script(
        "check_plan.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/stress/",
        check_rc=False,
        env=env,
    )
    print(p.stdout.decode().strip().split("\n")[0])
    check(
        p.returncode == 0,
        "check_plan passes on the real range (totals reconcile with numstat)",
    )
    p = script(
        "build_stack.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/stress/",
        env=env,
        check_rc=False,
    )
    print(p.stdout.decode().strip().split("\n")[-1], p.stderr.decode()[:400])
    check(
        p.returncode == 0,
        "round-robin split of every hunk rebuilds tree(top) == tree(TIP)",
    )
    p = script(
        "verify_stack.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/stress/",
        env=env,
        check_rc=False,
    )
    check(p.returncode == 0, "verify_stack passes on the real range")
    for x in asg:
        x["group"] = "0%d" % (4 - int(x["group"][1]))
    write_tsv(os.path.join(runs, "plan_assign.tsv"), ASSIGN_COLUMNS, asg)
    p = script(
        "build_stack.py",
        "--repo",
        repo,
        "--run",
        runs,
        "--prefix",
        "users/stress/",
        env=env,
        check_rc=False,
    )
    check(p.returncode == 0, "reversed assignment also reaches the same top tree")
    tmp.cleanup()


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--stress":
        test_stress(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        test_synthetic()
        test_regressions()
        test_regressions_2()
        test_regressions_3()
        test_subtractive()
        test_generated_exceptions()
    if FAILS:
        print("\n%d check(s) failed" % len(FAILS))
        sys.exit(1)
    print("\nall checks passed")
