# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Mutation sweep of a PR's checks, in its review's checks. Each check the PR
adds or changes is disabled once and the PR's tests run against the mutant: a check no test notices is a check no test
covers. It does what the code quality pass's hand-made sweeps do, with no model.

**The checks**, on the PR's added and changed lines against merge-base(develop, head), outside test files
(`test_*.py`, `*_test.py`):
- an `if`/`elif` guard whose body refuses (a raise, an exit call: `sys.exit`, `parser.error`, `pytest.exit`,
  `fail`, `abort`, or a call to one of the file's own refusal helpers, a short function ending in one: a conftest
  that refuses through `_stop()`) or returns an error (None, False, a non-zero int, a string, the name its
  condition binds, or a tuple that starts with None or False: a validator's `return False, "reason"`): its condition
  becomes `False`. A condition of `or`-joined parts gets one mutant per part (at most
  PARTS): each part is a check of its own (the `evidence` half of `not isinstance(evidence, str) or not
  isinstance(verified_on, str)`);
- a `raise`, or a call to a refusal helper, outside such a guard (an `except` that refuses): it becomes `pass`;
- an `assert`: it becomes `pass`.
`if __name__ == "__main__":` runs a script and checks nothing. C/C++ checks (`throw`, `assert(`, and under an `if` an
error `return` or a refusal macro such as `ROCKE_SPEC_FAIL(…)`) are listed, not mutated: this version builds nothing.

**Which checks**: a first round takes every check of the PR. A follow-up (`--since`, our last reviewed head) takes only
the checks whose own lines (a guard's through its error exit) the push since then adds or changes; when there is none,
the section reads "no new checks since <head>" and nothing runs, or, when the push adds or changes C/C++ checks only,
"no new Python checks since <head>" with those listed (a merge-only round does not run the sweep at all). One
mutant per check, at most `--cap`: the PR's new checks first (a line the merge base's file lacks), then those it
changes or moves; within each, guards, raises and refusals before asserts (an assert in a test helper is often the
test itself), the files taken in turn.

**The tests**, per mutated module: the PR's changed and added test files that name the module, plus the test files
that import it (grep; at most IMPORTERS, the smallest first); when none does, every PR test file (at most PR_TESTS).
A set runs once unmutated, when its first mutant is due. Its failing tests are deselected for the mutants (they would
kill every one); a set whose clean run fails to run, or takes longer than the per-mutant limit, is NOT RUN. A set
whose clean run takes over SLOW_SET_S costs about that much per mutant: its mutants run after the other sets' ones.

**The runs**: in a scratch copy of the head's tree (the review's head worktree copied when it is at the head, else
`git archive`), never in the review's worktree; one mutant at a time under `nice -n 10`, `pytest -x -q` with the
per-mutant limit inside the total budget (the clean runs count), the file restored after each and checked at the end.
Each mutant reads killed (a test failed), survived (every test passed: no test catches the check), timeout, or NOT RUN
(the budget ran out, its set cannot run, the mutant does not compile). The scratch copy is removed at the end; OUT keeps
mutants.json and each run's log tail.

**The survivors' re-run** (lessons L-598): each survivor runs again against every test file that imports its module
(at most WIDE_MAX, the smallest after its own set; a file that fails unmutated, a develop-known failure or a module the
python lacks, is dropped, never the whole set), with the merged tree's engine build on the path (`--engine`): first
under `--python` (CI's environment: no torch), then under `--torch-python`. Killed by the first: killed, by the CI
importers. Killed only under torch: `vacuous: torch-only path` (CI has no torch: no CI test reaches the check), still a
survivor. Neither: `SURVIVED (CI importers, torch, engine)`, naming what ran. Its own `--rerun-budget` seconds.

**Earlier rounds** (`--history DIR --pr N`): the PR's sweep history, DIR/pr<N>.jsonl, which the sweep reads and never
writes: it runs PR code, and the history is the monitor's. The monitor's `append` step adds a
finished sweep's mutants and their states. It reads OUT/mutants.json as JSON by the result reader (worker/pr_results.py
read_json: a regular file, no link on the way, a size cap, no JSON refused), only when it is the sweep of the head the
step is given, and each mutant only with HISTORY_FIELDS of their types and a known state (a mutant of another shape is
left out). It writes the history never through a link. The report then lists the earlier rounds' survivors, timeouts and
NOT RUN items still present at this head (the check's text is still in its file there), each with the head of the sweep
that found it, unless this sweep mutated the same check again; a later round's state of a check replaces an earlier
one's. They are carried, not re-run: a test added since may catch one. `--carry-only` (a merge-only round) runs no sweep
and lists them alone.

Usage: python3 -I mutation_sweep.py --repo R --head SHA --out DIR [--since SHA] [--tree DIR] [--base REV]
       [--rocke-path K] [--python PY] [--cap 25] [--per-mutant 120] [--budget 600] [--history DIR --pr N]
       [--carry-only] [--torch-python PY] [--engine DIR] [--rerun-budget 300]
Prints the section, then one last line: "MUTANTS pass|fail|skip <detail>".
       python3 -I mutation_sweep.py --repo R --head SHA [--since SHA] [--base REV] [--rocke-path K] --fetch
Fetches the blobs of the clone the sweep reads, and runs nothing else (rkmon_lib/prtree.py, fetch_main).
       python3 -I mutation_sweep.py append --out DIR --history DIR --pr N --head SHA
Appends the sweep's mutants in OUT to the history and prints nothing; a line on stderr names what it refused. Exit 0
when it refused nothing, 1 when it left out some mutants, 2 when it appended nothing it was given.
"""
import argparse
import ast
import collections
import datetime as dt
import json
import os
import pathlib
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True  # the toolkit's folder is read-only at run time
sys.path.insert(0, str(ROOT))
from rkmon_lib import clashes, gitops, nogpu, overlaps, prtree  # noqa: E402

DEVELOP = "refs/remotes/origin/develop"
# Mutants at most, seconds per mutant, seconds in all (the clean runs too): the sweep holds the checks' heavy slot at
# about 40 reviews a day.
CAP, PER_MUTANT_S, BUDGET_S = 25, 120, 600
MIN_RUN_S = 30  # a clean run or a mutant is not started with less of the budget left
SLOW_SET_S = 30  # a test set whose clean run takes longer: its mutants run after the other sets' mutants
PARTS = 6  # mutants for one `or` condition, at most
IMPORTERS, PR_TESTS = (
    6,
    8,
)  # test files per mutated module: importers outside the PR, PR files when none names it
DESELECT_MAX = 40  # a set whose clean run fails more tests than this is NOT RUN
# The survivors' re-run (rerun_survivors): test files in a survivor's wider set, at most, and its own seconds.
WIDE_MAX, RERUN_BUDGET_S = 30, 300
PLUGINS = ROOT / "tools" / "pytest_plugins"
TEST_FILE = re.compile(r"(^|/)(test_[^/]*|[^/]*_test)\.py$")
EXITS = {"exit", "_exit", "error", "fail", "abort"}
# The source build's native binaries the checks' test section exports (tools/native_env.py): they belong to the merged
# tree's build, not to this copy of the head.
NATIVE_ENV = re.compile(r"^ROCKE_\w*(EMIT|TEST|CLI|EXE)$")


def git(repo: str, *args: str, timeout: int = 300, lazy: bool = False) -> str:
    """`lazy`: a blob the clone lacks is fetched, unless lazy fetches are off where the sweep runs (a PR runner's slot:
    it reads there what its `--fetch` fetched)."""
    env = dict(os.environ) if lazy else {**os.environ, "GIT_NO_LAZY_FETCH": "1"}
    p = subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if p.returncode:
        raise RuntimeError(f"git {' '.join(args[:2])}: {p.stderr.strip()[:200]}")
    return p.stdout


# --- the checks ------------------------------------------------------------------------------------------------------


def _in(ranges, a: int, b: int | None = None) -> bool:
    b = a if b is None else b
    return any(x <= b and a <= y for x, y in ranges)


def _offset(lines: list[str], line: int, col: int) -> int:
    """The character offset of an ast position (1-based line, UTF-8 byte column) in the joined text."""
    return sum(len(x) for x in lines[: line - 1]) + len(
        lines[line - 1].encode("utf-8")[:col].decode("utf-8", "replace")
    )


def _span(lines: list[str], node) -> tuple[int, int]:
    return _offset(lines, node.lineno, node.col_offset), _offset(
        lines, node.end_lineno, node.end_col_offset
    )


def _call_name(node) -> str:
    """The called name of a call statement (`f(…)`, `x.f(…)`), else ""."""
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        f = node.value.func
        return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
    return ""


def _body(node) -> list:
    """A function's statements without its docstring."""
    b = node.body
    doc = (
        b
        and isinstance(b[0], ast.Expr)
        and isinstance(b[0].value, ast.Constant)
        and isinstance(b[0].value.value, str)
    )
    return b[1:] if doc else b


def refusers(tree) -> set[str]:
    """The file's own refusal helpers: short functions (at most 3 statements) that end by raising or calling an exit
    (`def _stop(msg): pytest.exit(msg, returncode=4)`). A call to one refuses as a raise does.
    """
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and 0 < len(_body(node)) <= 3
        and (
            isinstance(_body(node)[-1], ast.Raise)
            or _call_name(_body(node)[-1]) in EXITS
        )
    }


def _refusal(st, stops: set[str]) -> bool:
    """A statement that refuses: a raise, an exit call (`parser.error`, `pytest.exit`, `sys.exit`) or a call to one
    of the file's refusal helpers."""
    return isinstance(st, ast.Raise) or _call_name(st) in EXITS | stops


def _error_value(v, bound: str | None) -> bool:
    """A returned value that reports an error: None, False, a non-zero int, a string, or the name the condition binds."""
    if v is None or (isinstance(v, ast.Name) and v.id == bound):
        return True
    if (
        isinstance(v, ast.UnaryOp)
        and isinstance(v.op, ast.USub)
        and isinstance(v.operand, ast.Constant)
    ):
        return True
    if isinstance(v, (ast.Constant, ast.JoinedStr)):
        c = getattr(v, "value", "")
        return (
            c is None
            or c is False
            or isinstance(c, str)
            or isinstance(v, ast.JoinedStr)
            or (isinstance(c, int) and not isinstance(c, bool) and c != 0)
        )
    return False


def _error_exit(body: list, bound: str | None, stops: set[str]):
    """The statement with which a guard's body leaves with an error, else None: a refusal (_refusal), or a return of
    an error value (_error_value), or of a tuple that starts with None or False (`return False, "pipeline must be
    set"`: a validator's (ok, reason) answer)."""
    for st in body:
        if _refusal(st, stops):
            return st
        if isinstance(st, ast.Return):
            v = st.value
            if _error_value(v, bound):
                return st
            if (
                isinstance(v, ast.Tuple)
                and v.elts
                and isinstance(v.elts[0], ast.Constant)
                and (v.elts[0].value is None or v.elts[0].value is False)
            ):
                return st
    return None


def python_checks(path: str, text: str, plus: list, old: str = "") -> list[dict]:
    """The checks on a Python file's added lines, each with its one mutation: (start, end) offsets in the text and
    the replacement, and `lines`, the check's own lines (a guard's through its error exit). `new`: the check's line is
    not in the file's merge-base text `old` (a check the PR adds, not one it moves or touches).
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    lines = [
        ln + "\n" for ln in text.split("\n")
    ]  # ast counts newlines only (str.splitlines also splits at \f)
    before = {ln.strip() for ln in old.split("\n")} - {""}
    stops = refusers(tree)
    out, covered = [], set()

    def add(
        kind: str, node, target, repl: str, what: str, last: int | None = None
    ) -> None:
        a, b = _span(lines, target)
        out.append(
            {
                "path": path,
                "line": node.lineno,
                "kind": kind,
                "lines": [node.lineno, last or node.end_lineno],
                "new": lines[node.lineno - 1].strip() not in before,
                "span": [a, b],
                "repl": repl,
                "what": what,
            }
        )

    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in stops
        ):
            covered.add(
                id(_body(node)[-1])
            )  # the helper's own raise is every caller's refusal, not a check
        if not isinstance(node, ast.If):
            continue
        if "__name__" in (ast.get_source_segment(text, node.test) or ""):
            # `if __name__ == "__main__": raise SystemExit(main())` runs a script; it checks nothing
            covered.update(
                id(st) for st in ast.walk(node) if isinstance(st, (ast.Raise, ast.Expr))
            )
            continue
        if not _in(plus, node.lineno, node.test.end_lineno):
            continue
        bound = (
            node.test.target.id
            if isinstance(node.test, ast.NamedExpr)
            and isinstance(node.test.target, ast.Name)
            else None
        )
        leave = _error_exit(node.body, bound, stops)
        if leave is None:
            continue
        covered.update(id(st) for st in node.body if _refusal(st, stops))
        t = node.test
        for part in (
            t.values[:PARTS]
            if isinstance(t, ast.BoolOp) and isinstance(t.op, ast.Or)
            else [t]
        ):
            add(
                "guard",
                node,
                part,
                "False",
                ast.get_source_segment(text, part) or "",
                leave.end_lineno,
            )
    for node in ast.walk(tree):
        if id(node) in covered or not _in(plus, getattr(node, "lineno", 0)):
            continue
        if isinstance(node, ast.Raise):
            add(
                "raise",
                node,
                node,
                "pass",
                ast.get_source_segment(text, node) or "raise",
            )
        elif isinstance(node, ast.Expr) and _call_name(node) in stops | {
            "error",
            "fail",
        }:
            # a refusal outside a guard (in an `except`, or a bare `parser.error(…)`); sys.exit ends a program
            add("refusal", node, node, "pass", ast.get_source_segment(text, node) or "")
        elif isinstance(node, ast.Assert):
            add(
                "assert",
                node,
                node,
                "pass",
                ast.get_source_segment(text, node) or "assert",
            )
    seen, uniq = set(), []
    for c in sorted(out, key=lambda c: (c["span"][0], c["kind"])):
        if tuple(c["span"]) not in seen:
            seen.add(tuple(c["span"]))
            uniq.append(c)
    return uniq


CPP_CHECK = re.compile(
    r"\bthrow\b|\bassert\s*\(|\breturn\s+(?:false|nullptr|NULL|-\s*\d+|"
    r"[\w:]*(?:ERR|Err|ERROR|Error|FAIL|Fail|INVALID|Invalid)\w*)|"
    r"\b[A-Z][A-Z0-9_]*(?:FAIL|ERROR|REFUSE|REJECT)[A-Z0-9_]*\s*\("
)


def cpp_checks(path: str, text: str, plus: list) -> list[dict]:
    """C/C++ checks on added lines (listed only): `throw`, `assert(`, and under an `if` (its line or the 2 above) a
    `return <error>` or a refusal macro (`ROCKE_SPEC_FAIL(reason, cap, "...")`: an upper-case name with FAIL, ERROR,
    REFUSE or REJECT in it)."""
    masked = overlaps.C_LIT.sub(overlaps._blank, text).split("\n")
    raw, out = text.split("\n"), []
    for a, b in plus:
        for ln in range(max(a, 1), min(b, len(masked)) + 1):
            m = CPP_CHECK.search(masked[ln - 1])
            if not m:
                continue
            if not m.group().startswith(("throw", "assert")) and not any(
                re.search(r"\bif\s*\(", masked[k - 1])
                for k in range(max(1, ln - 2), ln + 1)
            ):
                continue
            out.append(
                {
                    "path": path,
                    "line": ln,
                    "kind": "c",
                    "what": raw[ln - 1].strip()[:90],
                }
            )
    return out


def pick(checks: list[dict], cap: int) -> list[dict]:
    """At most `cap` checks: the PR's new ones first, then the ones it changes; within each, guards, raises and
    refusals before asserts (an assert in a test helper is often the test itself), the files in turn (round robin).
    """
    out = []
    for new, assert_ in ((True, False), (True, True), (False, False), (False, True)):
        by_file: dict[str, list] = {}
        for c in checks:
            if c["new"] == new and (c["kind"] == "assert") == assert_:
                by_file.setdefault(c["path"], []).append(c)
        queues = [by_file[f] for f in sorted(by_file)]
        while any(queues) and len(out) < cap:
            for q in queues:
                if q and len(out) < cap:
                    out.append(q.pop(0))
    return out


def mutate(text: str, c: dict) -> str | None:
    """The mutant's text: the check's span replaced; None when it does not compile or changes nothing."""
    a, b = c["span"]
    new = text[:a] + c["repl"] + text[b:]
    if new == text:
        return None
    try:
        compile(new, c["path"], "exec")
    except (SyntaxError, ValueError):
        return None
    return new


# --- the tests -------------------------------------------------------------------------------------------------------


def import_rx(stem: str) -> re.Pattern:
    s = re.escape(stem)
    return re.compile(
        rf"^[ \t]*(?:from[ \t]+[\w.]*\b{s}\b[\w.]*[ \t]+import\b|import[ \t]+[\w., \t]*\b{s}\b"
        rf"|from[ \t]+[\w.]+[ \t]+import[ \t]+\(?[^\n]*\b{s}\b)",
        re.M,
    )


def tests_for(
    scratch: pathlib.Path, modules: list[str], pr_tests: list[str]
) -> dict[str, list[str]]:
    """Each mutated module's test files (module docstring), relative to the rocKE tree."""
    texts: dict[str, str] = {}
    for p in sorted(scratch.rglob("test_*.py")):
        try:
            texts[str(p.relative_to(scratch))] = p.read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
    out = {}
    for mod in modules:
        stem = clashes.module_stem(mod)
        word, rx = re.compile(rf"\b{re.escape(stem)}\b"), import_rx(stem)
        named = [t for t in pr_tests if t in texts and word.search(texts[t])]
        others = sorted(
            (t for t, s in texts.items() if t not in pr_tests and rx.search(s)),
            key=lambda t: (len(texts[t]), t),
        )[:IMPORTERS]
        out[mod] = named + others or [t for t in pr_tests if t in texts][:PR_TESTS]
    return out


# --- running pytest --------------------------------------------------------------------------------------------------


def pytest_cmd(
    python: str, files: list[str], deselect: list[str], first_fail: bool
) -> list[str]:
    cmd = [
        "nice",
        "-n",
        "10",
        python,
        "-B",
        "-m",
        "pytest",
        "-q",
        "-rfE",
        "-p",
        "no:cacheprovider",
        "--no-header",
        "-o",
        "addopts=",
        "--continue-on-collection-errors",
    ]
    if first_fail:
        cmd.append("-x")
    if (PLUGINS / "rkmon_env.py").exists():
        cmd += ["-p", "rkmon_env"]
    for d in deselect:
        cmd += ["--deselect", d]
    return cmd + files


def run_pytest(
    cmd: list[str], cwd: pathlib.Path, env: dict, timeout: float
) -> tuple[str, int | None, str, float, int]:
    """(state, return code, output, seconds, strays killed): state ok or timeout. The run is its own process group,
    killed whole on a time-out; what it leaves running after it ends is killed too (kill_strays, in `cwd`, the scratch
    copy)."""
    t0 = time.monotonic()
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = p.communicate(timeout=timeout)
        state, rc = "ok", p.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        out, _ = p.communicate()
        state, rc = "timeout", None
    return state, rc, out or "", time.monotonic() - t0, kill_strays(cwd)


def become_subreaper() -> None:
    """Orphans of the test runs reparent to this process (Linux PR_SET_CHILD_SUBREAPER), so kill_strays finds them."""
    try:
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)
    except (OSError, AttributeError):
        pass


def _procs() -> dict[int, int]:
    """pid -> parent pid of every process (from /proc)."""
    out = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat", encoding="utf-8", errors="replace") as fh:
                    out[int(d)] = int(fh.read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
    return out


def kill_strays(scratch: pathlib.Path) -> int:
    """Kill what a test run left running, and reap it: this process's descendants (a test's worker in a session of its
    own survives its run's process group: a timed-out mutant can leave a lane worker running in the removed copy) and
    any process whose command line or working directory is in the scratch copy.
    Returns how many."""
    me, parents = os.getpid(), _procs()
    kids: dict[int, list[int]] = {}
    for pid, ppid in parents.items():
        kids.setdefault(ppid, []).append(pid)
    strays, todo = set(), [me]
    while todo:
        for c in kids.get(todo.pop(), []):
            strays.add(c)
            todo.append(c)
    where = str(scratch)
    for pid in parents:
        if pid == me or pid in strays:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().decode("utf-8", "replace")
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            continue
        if where in cmd or cwd.startswith(where):
            strays.add(pid)
    for pid in strays:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                break
        except ChildProcessError:
            break
    return len(strays)


def failed_ids(out: str) -> list[str]:
    """The failing test ids of a `-rfE` summary: FAILED and ERROR lines, an id up to its " - " message (an id can hold
    spaces: `test_x[a b]`), as tools/pytest_vs_base.py reads them."""
    ids = set()
    for line in out.splitlines():
        m = re.match(r"(?:FAILED|ERROR) (.+)$", line)
        if m:
            ids.add(m.group(1).split(" - ", 1)[0].strip())
    return sorted(ids)


def summary(out: str) -> str:
    lines = [
        ln
        for ln in out.splitlines()
        if re.search(r"\d+ (passed|failed|error|skipped|deselected)|no tests ran", ln)
    ]
    return (
        lines[-1].strip("= ") if lines else (out.strip().splitlines() or [""])[-1][:160]
    )


def pytest_env(scratch: pathlib.Path, tmp: pathlib.Path) -> dict:
    env = nogpu.env(
        {
            k: v
            for k, v in os.environ.items()
            if not NATIVE_ENV.match(k) and k not in ("PYTHONPATH", "GIT_DIR")
        }
    )
    paths = [str(PLUGINS)] if PLUGINS.is_dir() else []
    env.update(
        PYTHONPATH=os.pathsep.join(
            paths + [str(scratch / "platform" / "python"), str(scratch / "library")]
        ),
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(tmp),
    )
    return env


# --- the scratch copy ------------------------------------------------------------------------------------------------


def provider(K: str) -> str:
    """The provider's tree, the rocKE tree's parent: the scratch copy holds it, so paths that reach above rocke/
    resolve."""
    return K.rsplit("/", 1)[0] if "/" in K else K


def scratch_copy(
    repo: str, head: str, K: str, tree: str | None, into: pathlib.Path
) -> tuple[pathlib.Path, str]:
    """The head's provider tree under `into`; the rocKE tree's path and how it was made. From the review's head
    worktree when it holds the head, else `git archive` (prtree.extract)."""
    parent = provider(K)
    if tree:
        wt = pathlib.Path(tree)
        try:
            at = subprocess.run(
                ["git", "-C", str(wt), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            at = ""
        if at == head and (wt / parent).is_dir():
            shutil.copytree(
                wt / parent,
                into / parent,
                symlinks=True,
                ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
            )
            return into / K, f"a copy of {wt} (at the head)"
    prtree.extract(repo, head, [parent], into)
    return into / K, "git archive of the head"


# --- the sweep -------------------------------------------------------------------------------------------------------


def find_checks(
    repo: str, mb: str, head: str, K: str
) -> tuple[list[dict], list[str], list[dict]]:
    """The checks on the PR's added lines against its merge base (Python, and C/C++ to list), and the PR's test files
    (paths relative to the rocKE tree). A test file the PR deletes has no added line, so each one listed exists.
    """
    diff = git(
        repo,
        "-c",
        "core.quotepath=false",
        "diff",
        "-U0",
        "--no-color",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        mb,
        head,
        "--",
        K,
        timeout=600,
        lazy=True,
    )
    hs = {p: h for p, h in clashes.hunks(diff).items() if h["new"] and h["plus"]}
    specs = {p: f"{head}:{p}" for p in hs if overlaps.language(p) in ("py", "c")}
    olds = {
        p: f"{mb}:{h['old']}"
        for p, h in hs.items()
        if h["old"] and overlaps.language(p) == "py"
    }
    texts = clashes.gitops.cat_blobs(
        repo, sorted(set(specs.values()) | set(olds.values())), timeout=600
    )
    checks, tests, cpp = [], [], []
    for p, h in sorted(hs.items()):
        rel = p[len(K) + 1 :] if p.startswith(K + "/") else p
        text = texts.get(specs.get(p, ""))
        if text is None:
            continue
        lang = overlaps.language(p)
        if lang == "py" and TEST_FILE.search(p):
            tests.append(rel)
        elif lang == "py":
            checks += [
                dict(c, path=rel)
                for c in python_checks(
                    rel, text, h["plus"], texts.get(olds.get(p, "")) or ""
                )
            ]
        elif lang == "c":
            cpp += [dict(c, path=rel) for c in cpp_checks(rel, text, h["plus"])]
    return checks, tests, cpp


def needs(
    repo: str, head: str, K: str, base: str = DEVELOP, since: str | None = None
) -> list[str]:
    """The blobs of the clone the sweep reads (`--fetch`): both sides of the PR's changes under the rocKE tree, and of
    the push since `since`, and the head's provider tree (its scratch copy; the earlier rounds' checks at the head).
    """
    mb = git(repo, "merge-base", base, head).strip()
    ids = prtree.diff_blobs(repo, mb, head, [K]) + prtree.blobs(
        repo, head, [provider(K)]
    )
    if since and gitops.has_commit(repo, since):
        ids += prtree.diff_blobs(repo, since, head, [K])
    return ids


def interdiff(repo: str, since: str, head: str, K: str) -> dict[str, list] | None:
    """The new-side line ranges the push from `since` (our last reviewed head) to `head` adds or changes, per file
    (relative to the rocKE tree); None when `since` is not in the mirror. A check of the PR's own lines that none of
    them touches was there at our last review."""
    if subprocess.run(
        ["git", "-C", repo, "cat-file", "-e", f"{since}^{{commit}}"],
        capture_output=True,
    ).returncode:
        return None
    diff = git(
        repo,
        "-c",
        "core.quotepath=false",
        "diff",
        "-U0",
        "--no-color",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        since,
        head,
        "--",
        K,
        timeout=600,
        lazy=True,
    )
    return {
        (p[len(K) + 1 :] if p.startswith(K + "/") else p): h["plus"]
        for p, h in clashes.hunks(diff).items()
        if h["new"] and h["plus"]
    }


def clean_run(
    files: list[str],
    python: str,
    scratch: pathlib.Path,
    env: dict,
    limit: float,
    res: dict,
    log: pathlib.Path,
) -> dict:
    """A test set's run on the unmutated copy: its failing tests are deselected for the mutants (with them, any
    mutant reads killed), and the mutants' own command must then pass; `why` says why the set cannot be used.
    """
    state, rc, text, secs, strays = run_pytest(
        pytest_cmd(python, files, [], False), scratch, env, limit
    )
    res["strays"] += strays
    failed = failed_ids(text)
    broken = sorted(f for f in failed if "::" not in f)  # a file that does not collect
    files = [f for f in files if f not in broken]
    ids = [f for f in failed if "::" in f]
    why = None
    if state == "timeout":
        why = f"its clean run takes over {limit:.0f} s"
    elif rc not in (0, 1, 5) and not failed:
        why = f"its clean run stops (pytest exit {rc}): {summary(text)}"
    elif not files:
        why = "none of its test files collects"
    elif len(ids) > DESELECT_MAX:
        why = f"its clean run fails {len(ids)} tests"
    elif rc == 5 or re.search(r"\bno tests ran\b", text):
        why = "its clean run collects no test"
    elif failed:
        state, rc, again, more, strays = run_pytest(
            pytest_cmd(python, files, ids, True), scratch, env, limit
        )
        res["strays"] += strays
        secs += more
        if state != "ok" or rc != 0:
            why = f"its clean run still fails with {len(ids)} tests deselected: {summary(again)}"
    log.write_text(text[-20000:])
    return {
        "files": files,
        "deselect": ids,
        "clean": summary(text),
        "secs": round(secs, 1),
        "dropped": broken,
        "why": why,
    }


def wide_tests(scratch: pathlib.Path, module: str, have: list[str]) -> list[str]:
    """A survivor's wider test set: its set's files, then every test file that imports its module (the smallest
    first), WIDE_MAX files in all."""
    rx = import_rx(clashes.module_stem(module))
    found = []
    for p in sorted(scratch.rglob("test_*.py")):
        rel = str(p.relative_to(scratch))
        if rel in have:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if rx.search(text):
            found.append((len(text), rel))
    return list(have) + [r for _, r in sorted(found)][: max(0, WIDE_MAX - len(have))]


def wide_clean(
    files: list[str],
    python: str,
    scratch: pathlib.Path,
    env: dict,
    limit: float,
    res: dict,
    log: pathlib.Path,
) -> dict:
    """A wider set's run on the unmutated copy. A file that fails there (a develop-known failure, a module this
    python lacks) is dropped, never the whole set; `why` says why the set cannot be used at all.
    """
    state, rc, text, secs, strays = run_pytest(
        pytest_cmd(python, files, [], False), scratch, env, limit
    )
    res["strays"] += strays
    bad = sorted({f.split("::", 1)[0] for f in failed_ids(text)})
    keep = [f for f in files if f not in bad]
    why = None
    if state == "timeout":
        why = f"its clean run takes over {limit:.0f} s"
    elif rc not in (0, 1, 5) and not bad:
        why = f"its clean run stops (pytest exit {rc}): {summary(text)}"
    elif not keep:
        why = "every file fails unmutated"
    elif rc == 5 or re.search(r"\bno tests ran\b", text):
        why = "its clean run collects no test"
    log.write_text(text[-20000:])
    return {
        "files": keep,
        "dropped": bad,
        "why": why,
        "clean": summary(text),
        "secs": round(secs, 1),
    }


def rerun_survivors(
    res: dict,
    scratch: pathlib.Path,
    env: dict,
    python: str,
    torch_python: str | None,
    engine: str | None,
    per_mutant: float,
    budget: float,
    out: pathlib.Path,
) -> None:
    """Each survivor again, against every test file importing its module (wide_tests; a file failing unmutated is
    dropped, not the set), with the shared engine's build on the path when `engine` is a folder: first under `python`
    (CI's environment: no torch), then under `torch_python`. Killed by the first: `killed`, by the CI importers. Killed
    only under torch: it stays a survivor, `vacuous: torch-only path` (CI has no torch, so no CI test reaches the
    check). Neither: `SURVIVED (CI importers, torch, engine)`, naming what ran. Within `budget` seconds of its own.
    """
    surv = [m for m in res["mutants"] if m["state"] == "survived"]
    if not surv:
        return
    start = time.monotonic()

    def left() -> float:
        return budget - (time.monotonic() - start)

    eng = engine if engine and os.path.isdir(engine) else ""
    renv = dict(env)
    if eng:
        renv["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + eng
    runs = [("CI importers", python)] + (
        [("torch", torch_python)] if torch_python else []
    )
    wide: dict[str, list[str]] = {}
    cleans: dict[tuple, dict] = {}
    res["rerun"] = {"engine": eng, "torch": torch_python or "", "sets": {}}
    for m in surv:
        if m["path"] not in wide:
            wide[m["path"]] = wide_tests(
                scratch,
                m["path"],
                (res["sets"].get(m.get("set")) or {}).get("files") or [],
            )
        ran, killer, stop = [], None, ""
        for label, py in runs:
            if left() < MIN_RUN_S:
                stop = "the re-run budget ran out"
                break
            key = (m["path"], label)
            if key not in cleans:
                cleans[key] = wide_clean(
                    wide[m["path"]],
                    py,
                    scratch,
                    renv,
                    min(per_mutant, left()),
                    res,
                    out / f"rerun_{len(cleans) + 1:02d}.log",
                )
                res["rerun"]["sets"][f"{m['path']} ({label})"] = cleans[key]
            c = cleans[key]
            if c["why"] or left() < MIN_RUN_S:
                continue
            target = scratch / m["path"]
            orig = target.read_text(encoding="utf-8")
            text = mutate(orig, m)
            if text is None:
                break
            target.write_text(text, encoding="utf-8")
            try:
                state, rc, _log, _secs, strays = run_pytest(
                    pytest_cmd(py, c["files"], [], True),
                    scratch,
                    renv,
                    min(per_mutant, left()),
                )
                res["strays"] += strays
            finally:
                target.write_text(orig, encoding="utf-8")
            if state != "ok":
                continue
            ran.append(label)
            if rc in (1, 2, 3):
                killer = label
                break
        if killer == "CI importers":
            m["state"], m["rerun"] = "killed", "killed by the CI importers"
        elif killer:
            m["rerun"] = "vacuous: torch-only path"
        elif ran:
            m["rerun"] = f"SURVIVED ({', '.join(ran + (['engine'] if eng else []))})"
        else:
            m["rerun"] = f"not re-run: {stop or 'no wider set could run'}"


def defer_set(queue, mutants: list[dict], label: str, i: int):
    """The run queue with set `label`'s mutants (mutant i first) after every other set's: a slow set's clean run says
    each of its mutants costs about as much, so the cheaper ones run first within the budget.
    """
    return collections.deque(
        [j for j in queue if mutants[j]["set"] != label]
        + [i]
        + [j for j in queue if mutants[j]["set"] == label]
    )


def sweep(
    repo: str,
    head: str,
    out: pathlib.Path,
    K: str,
    base: str = DEVELOP,
    tree: str | None = None,
    python: str = sys.executable,
    cap: int = CAP,
    per_mutant: float = PER_MUTANT_S,
    budget: float = BUDGET_S,
    since: str | None = None,
    torch_python: str | None = None,
    engine: str | None = None,
    rerun_budget: float = RERUN_BUDGET_S,
) -> dict:
    """The whole sweep (module docstring); the result as mutants.json holds it. `since`: our last reviewed head, for a
    follow-up: only the checks the push since then adds or changes. The survivors are then re-run wider
    (rerun_survivors), within `rerun_budget` seconds of their own."""
    start = time.monotonic()

    def left() -> float:
        return budget - (time.monotonic() - start)

    res = {
        "head": head,
        "since": since,
        "checks": 0,
        "cpp": [],
        "mutants": [],
        "sets": {},
        "error": None,
        "strays": 0,
    }
    mb = git(repo, "merge-base", base, head).strip()
    res["mb"] = mb
    checks, pr_tests, res["cpp"] = find_checks(repo, mb, head, K)
    res["checks"], res["new_checks"] = len(checks), sum(1 for c in checks if c["new"])
    if since:
        touched = interdiff(repo, since, head, K)
        if touched is None:
            res["scope"] = (
                f"every check of the PR: our reviewed head {since[:10]} is not in the mirror"
            )
        else:
            checks = [c for c in checks if _in(touched.get(c["path"], []), *c["lines"])]
            res["scope"] = f"the checks the push since {since[:10]} adds or changes"
            res["cpp"] = [
                c for c in res["cpp"] if _in(touched.get(c["path"], []), c["line"])
            ]
    else:
        res["scope"] = "every check of the PR"
    res["in_scope"] = len(checks)
    res["mutants"] = [
        dict(c, state="NOT RUN", why="not reached") for c in pick(checks, cap)
    ]
    if not res["mutants"]:
        res["secs"] = round(time.monotonic() - start, 1)
        return res
    out.mkdir(parents=True, exist_ok=True)
    work = pathlib.Path(
        tempfile.mkdtemp(prefix="mutants_", dir=os.environ.get("TMPDIR") or None)
    )
    become_subreaper()
    try:
        scratch, res["copy"] = scratch_copy(repo, head, K, tree, work)
        tmp = work / "tmp"
        tmp.mkdir()
        env = pytest_env(scratch, tmp)
        sets = tests_for(scratch, sorted({m["path"] for m in res["mutants"]}), pr_tests)
        labels: dict[tuple, str] = {}
        for mod in sorted(sets):
            if sets[mod] and tuple(sets[mod]) not in labels:
                labels[tuple(sets[mod])] = (
                    chr(ord("A") + len(labels))
                    if len(labels) < 26
                    else f"S{len(labels)}"
                )
        originals: dict[str, str] = {}
        for m in res["mutants"]:
            m["set"] = labels.get(tuple(sets.get(m["path"]) or []))
        queue = collections.deque(range(len(res["mutants"])))
        while queue:
            i = queue.popleft()
            m = res["mutants"][i]
            label = m["set"]
            if label is None:
                m["why"] = "no test file names or imports its module"
                continue
            if left() < MIN_RUN_S:
                m["why"] = "the time budget ran out"
                continue
            if (
                label not in res["sets"]
            ):  # each set's clean run only when one of its mutants is due
                res["sets"][label] = clean_run(
                    list(sets[m["path"]]),
                    python,
                    scratch,
                    env,
                    min(per_mutant, left()),
                    res,
                    out / f"baseline_{label}.log",
                )
                if (
                    res["sets"][label]["secs"] > SLOW_SET_S
                    and not res["sets"][label]["why"]
                ):
                    queue = defer_set(queue, res["mutants"], label, i)
                    continue
            b = res["sets"][label]
            if b["why"]:
                m["why"] = f"set {label}: {b['why']}"
                continue
            if left() < MIN_RUN_S:
                m["why"] = "the time budget ran out"
                continue
            target = scratch / m["path"]
            orig = originals.setdefault(m["path"], target.read_text(encoding="utf-8"))
            text = mutate(orig, m)
            if text is None:
                m["why"] = "the mutant does not compile"
                continue
            target.write_text(text, encoding="utf-8")
            limit = min(per_mutant, left())
            try:
                state, rc, log, secs, strays = run_pytest(
                    pytest_cmd(python, b["files"], b["deselect"], True),
                    scratch,
                    env,
                    limit,
                )
                res["strays"] += strays
            finally:
                target.write_text(orig, encoding="utf-8")
            m["secs"], m["tail"] = round(secs, 1), summary(log)
            (out / f"mutant_{i + 1:02d}.log").write_text(log[-20000:])
            if state == "timeout":
                m["state"], m["why"] = "timeout", f"killed after {limit:.0f} s"
            elif rc == 0:
                m["state"], m["why"] = "survived", ""
            elif rc in (1, 2, 3):
                m["state"], m["why"] = "killed", ""
            else:
                m["why"] = f"pytest exit {rc}: {m['tail']}"
        if rerun_budget > 0:
            rerun_survivors(
                res,
                scratch,
                env,
                python,
                torch_python,
                engine,
                per_mutant,
                rerun_budget,
                out,
            )
        res["restored"] = all(
            (scratch / p).read_text(encoding="utf-8") == t for p, t in originals.items()
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        res["error"] = str(exc)[:300]
    finally:
        res["strays"] += kill_strays(work)
        shutil.rmtree(work, ignore_errors=True)
    res["secs"] = round(time.monotonic() - start, 1)
    return res


# --- earlier rounds --------------------------------------------------------------------------------------------------

CARRIED_SHOWN = 15  # earlier items listed, at most
# A history item's fields and their JSON types, as the report reads them, and the states a mutant ends in.
HISTORY_FIELDS = {
    "path": (str,),
    "line": (int, type(None)),
    "kind": (str, type(None)),
    "what": (str,),
    "repl": (str,),
    "state": (str,),
    "why": (str, type(None)),
}
STATES = ("killed", "survived", "timeout", "NOT RUN")


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def load_history(path: pathlib.Path) -> list[dict]:
    """The earlier sweeps of a PR, oldest first ({"head", "at", "items"}); [] without the file."""
    out = []
    try:
        for ln in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            if isinstance(rec, dict) and isinstance(rec.get("items"), list):
                out.append(rec)
    except OSError:
        pass
    return out


def history_record(res: dict) -> dict:
    """This sweep's line of the history: every mutant with its state."""
    return {
        "head": res["head"],
        "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "items": [{k: m.get(k) for k in HISTORY_FIELDS} for m in res["mutants"]],
    }


def history_item(m) -> bool:
    """Whether a finished sweep's mutant, read as data, is one the history keeps: HISTORY_FIELDS of their exact JSON
    types (true is no line number) and a state of STATES."""
    return (
        isinstance(m, dict)
        and all(type(m.get(k)) in types for k, types in HISTORY_FIELDS.items())
        and m["state"] in STATES
    )


def _append_line(path: pathlib.Path, line: str) -> None:
    """`line` appended to the file `path` (made when missing, as open() makes it), never through a link: a link at the
    file or its folder, or no regular file there, is an OSError."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dfd = os.open(
        path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
        fd = os.open(path.name, flags | os.O_CLOEXEC, 0o666, dir_fd=dfd)
    finally:
        os.close(dfd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"{path}: not a regular file")
        data = line.encode("utf-8")
        while data:
            data = data[os.write(fd, data) :]
    finally:
        os.close(fd)


def append_history(out: str, history: str, pr: int, head: str) -> int:
    """`append` (the module's usage): the finished sweep in `out` appended to PR `pr`'s history. Exit 0 when it refused
    nothing; 1 when it left out a mutant of another shape and appended the others; 2 when it appended nothing it was
    given: no mutants.json it reads as the sweep of `head`, no mutant of the history's shape, or a history it cannot
    write."""
    from rkmon_lib import paths

    pr_results = paths.worker_module("pr_results")  # the result reader

    def took_nothing(why: str) -> int:
        print(f"mutation history: nothing appended: {why}", file=sys.stderr)
        return 2

    try:
        res = pr_results.read_json(out, "mutants.json")
    except OSError as exc:  # the reader's Refused: no file its rules allow, or no JSON
        return took_nothing(str(exc)[:200])
    if (
        not isinstance(res, dict)
        or res.get("head") != head
        or not isinstance(res.get("mutants"), list)
    ):
        return took_nothing(f"mutants.json is no sweep of {head[:10]}")
    kept = [m for m in res["mutants"] if history_item(m)]
    left = len(res["mutants"]) - len(kept)
    if left:
        print(
            f"mutation history: {left} of {len(res['mutants'])} mutants left out (of another shape)",
            file=sys.stderr,
        )
    if kept:
        try:
            _append_line(
                pathlib.Path(history) / f"pr{pr}.jsonl",
                json.dumps(history_record({"head": head, "mutants": kept})) + "\n",
            )
        except OSError as exc:
            return took_nothing(str(exc)[:200])
    return 0 if not left else 1 if kept else 2


def carried(repo: str, head: str, K: str, history: list[dict], now: list[dict]) -> dict:
    """The earlier rounds' survivors, timeouts and NOT RUN items whose check is still at `head` and that this sweep did
    not mutate again (`now`): {"items": [(item, the sweep's head, its line at head)], "gone": n}.
    """
    key = lambda m: (
        m.get("path"),
        _norm(m.get("what") or ""),
        m.get("repl"),
    )  # noqa: E731
    latest: dict = {}
    for rec in history:
        for it in rec.get("items") or []:
            latest[key(it)] = (it, rec.get("head") or "?")
    redone = {key(m) for m in now}
    texts: dict[str, str | None] = {}
    items, gone = [], 0
    for k, (it, h) in latest.items():
        if it.get("state") == "killed" or k in redone:
            continue
        path = it.get("path") or ""
        if path not in texts:
            try:
                texts[path] = git(repo, "show", f"{head}:{K}/{path}")
            except (RuntimeError, OSError, subprocess.SubprocessError):
                texts[path] = None
        text = texts[path]
        if text is None or k[1] not in _norm(text):
            gone += 1
            continue
        first = (
            (it.get("what") or "").strip().splitlines()[0].strip()
            if (it.get("what") or "").strip()
            else ""
        )
        line = next(
            (i for i, ln in enumerate(text.splitlines(), 1) if first and first in ln),
            it.get("line"),
        )
        items.append((it, h, line))
    order = {"survived": 0, "timeout": 1, "NOT RUN": 2}
    items.sort(
        key=lambda x: (
            order.get(x[0].get("state"), 3),
            x[0].get("path") or "",
            x[2] or 0,
        )
    )
    return {"items": items, "gone": gone}


def carried_lines(c: dict | None) -> list[str]:
    if not c or not (c["items"] or c["gone"]):
        return []
    out = []
    if c["items"]:
        out.append(
            f"earlier rounds' items still at this head ({len(c['items'])}; from their sweeps, not re-run here):"
        )
        for it, h, line in c["items"][:CARRIED_SHOWN]:
            tag = {"survived": "SURVIVED", "timeout": "timeout"}.get(
                it.get("state"), "NOT RUN"
            )
            why = (
                f" ({it['why']})"
                if it.get("state") == "NOT RUN" and it.get("why")
                else ""
            )
            out.append(
                f"  {tag} (sweep at {h[:10]}): {it.get('path')}:{line} {_what(it)}{why}"
            )
        if len(c["items"]) > CARRIED_SHOWN:
            out.append(f"  +{len(c['items']) - CARRIED_SHOWN} more")
    if c["gone"]:
        out.append(
            f"({c['gone']} earlier survivors or NOT RUN items are not at this head)"
        )
    return out


def carried_tail(c: dict | None) -> str:
    if not c or not c["items"]:
        return ""
    surv = sum(1 for it, _h, _l in c["items"] if it.get("state") == "survived")
    rest = len(c["items"]) - surv
    parts = ([f"{surv} earlier survivor{'s' if surv != 1 else ''}"] if surv else []) + (
        [f"{rest} earlier timeout or NOT RUN"] if rest else []
    )
    return "; " + " and ".join(parts) + " still at this head"


# --- the report ------------------------------------------------------------------------------------------------------


def _what(m: dict) -> str:
    w = " ".join(m["what"].split())
    return f"`{w[:70]}{'…' if len(w) > 70 else ''}` -> `{m['repl']}`"


def nothing_since(res: dict) -> bool:
    """A follow-up whose push adds or changes no Python check: there is nothing new to mutate (its new C/C++ checks,
    `res["cpp"]`, are listed)."""
    return (
        bool(res.get("since"))
        and not res.get("in_scope")
        and not res.get("error")
        and str(res.get("scope", "")).startswith("the checks the push since")
    )


def _cpp_lines(res: dict) -> list[str]:
    out = [
        f"  C/C++ (listed): {c['path']}:{c['line']} `{c['what']}`"
        for c in res["cpp"][:8]
    ]
    return out + (
        [f"  C/C++ (listed): +{len(res['cpp']) - 8} more"]
        if len(res["cpp"]) > 8
        else []
    )


def render(res: dict, K: str, cap: int) -> list[str]:
    ms = res["mutants"]
    if res.get("merge_only"):
        return [
            f"no new checks since {res['since'][:10]} (merge-only since our review: the sweep does not run)",
            *carried_lines(res.get("carried")),
            " ".join(("MUTANTS",) + verdict(res)),
        ]
    if nothing_since(res) and res["cpp"]:
        return [
            f"no new Python checks since {res['since'][:10]}; the push since our last review adds or changes "
            f"{len(res['cpp'])} C/C++ checks (listed, not run: no build here; {res.get('secs', 0)} s):",
            *_cpp_lines(res),
            *carried_lines(res.get("carried")),
            " ".join(("MUTANTS",) + verdict(res)),
        ]
    if nothing_since(res):
        return [
            f"no new checks since {res['since'][:10]} (the push since our last review adds or changes none of the "
            f"PR's {res['checks']} Python checks and no C/C++ check; {res.get('secs', 0)} s)",
            *carried_lines(res.get("carried")),
            " ".join(("MUTANTS",) + verdict(res)),
        ]
    count = {s: sum(1 for m in ms if m["state"] == s) for s in STATES}
    scope = res.get("scope") or "every check of the PR"
    out = [
        f"checks on the PR's added and changed Python lines: {res['checks']} ({res.get('new_checks', 0)} new); "
        f"mutated: {scope} ({res.get('in_scope', res['checks'])}); C/C++ checks listed, not run (no build here): "
        f"{len(res['cpp'])}; mutants: {len(ms)}{f' (cap {cap})' if res.get('in_scope', 0) > len(ms) else ''}: "
        f"{count['survived']} survived, {count['killed']} killed, {count['timeout']} timeout, {count['NOT RUN']} "
        f"NOT RUN ({res.get('secs', 0)} s{', ' + res['copy'] if res.get('copy') else ''}"
        f"{f'; killed {s} processes the tests left running' if (s := res.get('strays')) else ''})"
    ]
    if res.get("error"):
        out.append(f"NOT RUN: {res['error']}")
    for label, b in sorted((res.get("sets") or {}).items()):
        state = (
            f"NOT RUN: {b['why']}"
            if b["why"]
            else f"clean run {b['clean']} in {b['secs']} s"
        )
        extra = (
            f"; {len(b['deselect'])} failing tests deselected"
            if b["deselect"] and not b["why"]
            else ""
        )
        extra += (
            f"; dropped (does not collect): {' '.join(b['dropped'])}"
            if b["dropped"]
            else ""
        )
        out.append(
            f"tests [{label}] ({state}{extra}): PYTHONPATH=platform/python:library python -B -m pytest -x -q "
            f"-o addopts= {' '.join(b['files'])}"
        )
    order = {"survived": 0, "timeout": 1, "NOT RUN": 2, "killed": 3}
    for m in sorted(ms, key=lambda m: (order[m["state"]], m["path"], m["line"])):
        where = f"{m['path']}:{m['line']}"
        if m["state"] == "killed":
            continue
        tag = {"survived": "SURVIVED", "timeout": "timeout", "NOT RUN": "NOT RUN"}[
            m["state"]
        ]
        if m["state"] == "survived" and m.get("rerun"):
            tag = m["rerun"]  # the wider re-run's verdict (rerun_survivors)
        why = f" ({m['why']})" if m.get("why") else ""
        out.append(
            f"  {tag}: {where} {_what(m)}{' [' + m['set'] + ']' if m.get('set') else ''}"
            f"{' (new)' if m['new'] else ''}{why}"
        )
    killed = [m for m in ms if m["state"] == "killed"]
    if killed:
        out.append(
            f"  killed ({len(killed)}): "
            + ", ".join(
                f"{m['path'].rsplit('/', 1)[-1]}:{m['line']} [{m.get('set')}] {m.get('secs')} s"
                + (" (by the CI importers' re-run)" if m.get("rerun") else "")
                for m in killed
            )
        )
    rr = res.get("rerun")
    if rr:
        out.append(
            f"survivors re-run against every test file importing their module (CI's python"
            f"{', torch: ' + rr['torch'] if rr['torch'] else ', no torch python here'}"
            f"{', engine: ' + rr['engine'] if rr['engine'] else ', no engine build'}; a file failing unmutated "
            "is dropped):"
        )
        for name, c in rr["sets"].items():
            out.append(
                f"  {name}: "
                + (
                    f"NOT RUN: {c['why']}"
                    if c["why"]
                    else f"{len(c['files'])} files, clean {c['clean']}"
                )
                + (
                    f"; dropped (fail unmutated): {' '.join(c['dropped'][:6])}"
                    if c["dropped"]
                    else ""
                )
            )
    out += _cpp_lines(res)
    if res.get("restored") is False:
        out.append(
            "!! a mutated file in the scratch copy did not read back as the head's: the copy is removed"
        )
    out += carried_lines(res.get("carried"))
    out.append(" ".join(("MUTANTS",) + verdict(res)))
    return out


def verdict(res: dict) -> tuple[str, str]:
    word, detail = _verdict(res)
    return word, detail + carried_tail(res.get("carried"))


def _verdict(res: dict) -> tuple[str, str]:
    ms = res["mutants"]
    if res.get("merge_only"):
        return "skip", f"no new checks since {res['since'][:10]} (merge-only)"
    ran = [m for m in ms if m["state"] != "NOT RUN"]
    surv = [m for m in ms if m["state"] == "survived"]
    if res.get("error") and not ran:
        return "skip", f"not run: {res['error'][:160]}"
    if nothing_since(res) and res["cpp"]:
        return (
            "skip",
            f"no new Python checks since {res['since'][:10]}; {len(res['cpp'])} new C/C++ checks listed",
        )
    if nothing_since(res):
        return "skip", f"no new checks since {res['since'][:10]}"
    if not ms:
        return "skip", "no check in the PR's added Python lines (nothing to mutate)"
    if not ran:
        return (
            "skip",
            f"no mutant ran ({len(ms)} NOT RUN: {ms[0].get('why', '')[:100]})",
        )
    tail = f"; {len(ms) - len(ran)} NOT RUN" if len(ran) < len(ms) else ""
    if surv:
        torch_only = sum(
            1 for m in surv if m.get("rerun") == "vacuous: torch-only path"
        )
        return "fail", (
            f"{len(surv)} of {len(ran)} mutants survive (checks no test catches): "
            + ", ".join(f"{m['path'].rsplit('/', 1)[-1]}:{m['line']}" for m in surv[:6])
            + (f" +{len(surv) - 6} more" if len(surv) > 6 else "")
            + (f"; {torch_only} only under torch (vacuous in CI)" if torch_only else "")
            + tail
        )
    return "pass", f"all {len(ran)} mutants killed{tail}"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["append"]:
        ap = argparse.ArgumentParser(
            prog="mutation_sweep.py append",
            description="A finished sweep's mutants appended to its PR's history.",
        )
        ap.add_argument(
            "--out", required=True, help="the sweep's folder (its mutants.json)"
        )
        ap.add_argument(
            "--history",
            required=True,
            help="the folder of the PRs' sweep histories (pr<N>.jsonl)",
        )
        ap.add_argument("--pr", type=int, required=True, help="the PR number")
        ap.add_argument("--head", required=True, help="the head the sweep ran at")
        a = ap.parse_args(argv[1:])
        if a.pr < 1:
            ap.error("--pr takes a PR number")
        return append_history(a.out, a.history, a.pr, a.head)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--tree",
        default=None,
        help="the review's head worktree (copied when it is at the head)",
    )
    ap.add_argument("--base", default=DEVELOP)
    ap.add_argument("--rocke-path", default="", type=prtree.folder_argument)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--per-mutant", type=float, default=PER_MUTANT_S)
    ap.add_argument("--budget", type=float, default=BUDGET_S)
    ap.add_argument(
        "--since",
        default=None,
        help="our last reviewed head: only the checks the push since then adds or "
        "changes (a follow-up)",
    )
    ap.add_argument(
        "--history",
        default=None,
        help="the folder of the PRs' sweep histories (pr<N>.jsonl): read, never written",
    )
    ap.add_argument("--pr", type=int, default=None, help="the PR number, for --history")
    ap.add_argument(
        "--torch-python",
        default=None,
        help="a python with torch: the survivors' re-run under it (vacuous: torch-only path)",
    )
    ap.add_argument(
        "--engine",
        default=None,
        help="the shared engine's build folder (the rocke_engine module): on the re-run's path",
    )
    ap.add_argument(
        "--rerun-budget",
        type=float,
        default=RERUN_BUDGET_S,
        help="seconds for the survivors' re-run (0: none)",
    )
    ap.add_argument(
        "--carry-only",
        action="store_true",
        help="no sweep (a merge-only round): the earlier rounds' items",
    )
    ap.add_argument("--fetch", action="store_true", help=prtree.FETCH_HELP)
    a = ap.parse_args(argv)
    if a.fetch:
        return prtree.fetch_main(
            a.repo, lambda: needs(a.repo, a.head, a.rocke_path, a.base, a.since)
        )
    if not a.out:
        ap.error("the following arguments are required: --out")
    out = pathlib.Path(a.out)
    hist = pathlib.Path(a.history) / f"pr{a.pr}.jsonl" if a.history and a.pr else None
    if a.carry_only:
        res = {
            "head": a.head,
            "since": a.since or a.head,
            "checks": 0,
            "cpp": [],
            "mutants": [],
            "error": None,
            "merge_only": True,
        }
    else:
        try:
            res = sweep(
                a.repo,
                a.head,
                out,
                a.rocke_path,
                a.base,
                a.tree,
                a.python,
                a.cap,
                a.per_mutant,
                a.budget,
                a.since,
                a.torch_python,
                a.engine,
                a.rerun_budget,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            res = {
                "head": a.head,
                "checks": 0,
                "cpp": [],
                "mutants": [],
                "error": str(exc)[:300],
            }
    if hist:
        try:
            res["carried"] = carried(
                a.repo, a.head, a.rocke_path, load_history(hist), res["mutants"]
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            res["carried"] = None
            res["carry_error"] = str(exc)[:200]
    out.mkdir(parents=True, exist_ok=True)
    (out / "mutants.json").write_text(json.dumps(res, indent=1))
    print("\n".join(render(res, a.rocke_path, a.cap)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
