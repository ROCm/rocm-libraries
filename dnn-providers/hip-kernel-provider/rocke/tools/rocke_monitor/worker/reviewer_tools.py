# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The review tools: a fixed menu of vetted tools on one PR's trees, shared by the first-pass reviewer
(review_agent.py), the offload passes (offload_agent.py), and `rk evidence-check` (rkmon_lib/offload_evidence.py), which re-runs
a recorded tool run by the same argv builder. The model never supplies a shell command: every tool builds its argv
from validated parameters, with shell=False; a path is relative to the tree it names, never absolute, never `..`.
The model's own code (a probe, a mutant) runs only in the sandbox (review_sandbox.py); a saved probe runs in it too when there
is one, else as the checks run it. Every tool that runs PR code (PR_CODE: the repository's tests, a probe, a mutant) runs
in a PR runner's slot of its own where there are PR runners (the container: pr_user.py), as the checks do.

`Tree` holds the PR's trees (wt<N>, the head; wt<N>m, the head merged with develop) and the job's paths from its RKMON_*
environment. Each tool takes a Tree and the model's arguments and returns text for the model; `execute` returns the
whole output too, for the evidence records.
"""

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pr_user  # noqa: E402
import review_sandbox as sandbox  # noqa: E402
import rkenv  # noqa: E402

SAFE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$")
NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,60}$")
OUT_CAP = (
    6000  # what one tool result shows the model: the head and tail of a longer output
)
READ_CAP = 8000
LINES_CAP = 400  # read_lines: the most lines in one call
CODE_CAP = 20000  # a probe's or mutant's source
QUOTE_CAP = 600  # a recorded quote: a few lines
QUOTE_PART_MIN = 12
TREES = ("head", "merged", "base")
# The tools that run PR code, in a PR runner's slot where there are PR runners (pr_user.wrap): there in the folder their
# builder names (True: a tree), or in the slot's home (False: the builder's folder is the review folder's, which no
# runner writes; a host runs the tool there).
PR_CODE = {"run_pytest": True, "run_probe": False, "run_mutant": True}


class Tree:
    """One PR's trees and the job's paths (the RKMON_* environment, or `get` in a test)."""

    def __init__(self, ws, n, pr_dir=None, get=None):
        get = get or rkenv.get
        self.ws, self.n = pathlib.Path(ws), str(n)
        self.root = pathlib.Path(get("RKMON_WORKING_DIR"))
        self.wt = (self.root / f"wt{n}").resolve()
        self.wtm = (self.root / f"wt{n}m").resolve()
        self.rocke = self.wt / get("RKMON_ROCKE_PATH")
        # The checks venv's Python, which has the repository's test dependencies; without the venv, this one (rule_scan
        # needs only the standard library, and run_pytest then says what is missing instead of failing the draft).
        venv = pathlib.Path(get("RKMON_VENV")) / "bin" / "python"
        self.pybin = venv if venv.exists() else pathlib.Path(sys.executable)
        self.toolkit = pathlib.Path(get("RKMON_TOOLKIT"))
        self.build = pathlib.Path(get("RKMON_BUILD"))
        self.tmp = pathlib.Path(get("RKMON_TMP"))
        self.probes = (
            pathlib.Path(get("RKMON_PROBES", required=False) or self.root / "probes")
            / f"pr{n}"
        )
        self.pr = pathlib.Path(pr_dir) if pr_dir else self.ws / "pr"
        self.tools_dir = pathlib.Path(__file__).resolve().parent
        self._base = None

    def env(self) -> dict:
        e = dict(os.environ)
        for k in ("GH_TOKEN", "GITHUB_TOKEN"):
            e.pop(k, None)
        e["ROCKE_LLVM_FLAVOR"] = e.get("ROCKE_LLVM_FLAVOR", "llvm22")
        # The system's temp folder may be a RAM-backed tmpfs; kernel-cache tests can fill it. Keep test temp files on
        # disk.
        self.tmp.mkdir(parents=True, exist_ok=True)
        e["TMPDIR"] = str(self.tmp)
        return e

    def rk(self) -> str:
        return str(self.toolkit / "rk")

    def sha(self, tree: str = "head") -> str:
        """The commit a tree holds: head (wt<N>), merged (wt<N>m) or base (the head's merge base with develop)."""
        if tree == "base":
            if self._base is None:
                self._base = _git(
                    self.wt, "merge-base", "HEAD", "refs/remotes/origin/develop"
                )
            return self._base
        return _git(self.wtm if tree == "merged" else self.wt, "rev-parse", "HEAD")

    def pytest_path(self) -> list[pathlib.Path]:
        """The checks' import path (monitor/tools/pytest_vs_base.py): the platform and library trees, the rkmon_env
        plugin and this head's native binding when it is built (`-I` drops PYTHONPATH: without this path, every test
        that imports the library tree would fail here and read as the PR's)."""
        pp = [
            self.toolkit / "tools" / "pytest_plugins",
            self.rocke / "platform" / "python",
            self.rocke / "library",
        ]
        if (self.build / f"wt{self.n}" / "pybind").is_dir():
            pp.append(self.build / f"wt{self.n}" / "pybind")
        return pp


def _git(tree: pathlib.Path, *args) -> str:
    p = subprocess.run(
        ["git", "-C", str(tree), *args], capture_output=True, text=True, timeout=60
    )
    return p.stdout.strip() if p.returncode == 0 else ""


def safe(tree: pathlib.Path, rel: str):
    """rel resolved inside tree; None for an absolute path, `..`, or an odd character."""
    if not rel or rel.startswith("/") or ".." in rel or not SAFE.match(rel):
        return None
    p = (tree / rel).resolve()
    return p if str(p).startswith(str(tree) + os.sep) else None


def safe_rel(rel: str) -> bool:
    """A repository path the model may name (it need not exist in the tree: a file at the merge base)."""
    return (
        bool(rel)
        and not rel.startswith("/")
        and ".." not in rel
        and bool(SAFE.match(rel))
    )


@dataclass
class Result:
    rc: int | None  # None: timed out
    out: str  # stdout, then stderr after a "[stderr]" line
    secs: float
    timeout: int

    def text(self) -> str:
        """What the model sees: the output's head and tail past OUT_CAP; "TIMEOUT after Ns." when it timed out."""
        if self.rc is None:
            return f"TIMEOUT after {self.timeout}s."
        return clip(self.out)


def clip(out: str, cap: int = OUT_CAP) -> str:
    if len(out) > cap:
        out = (
            out[: cap // 2]
            + f"\n...[truncated {len(out) - cap}]...\n"
            + out[-cap // 2 :]
        )
    return out.strip() or "(no output)"


def execute(argv, cwd, timeout: int, env: dict) -> Result:
    """Run a built tool (shell=False); a command line that runs PR code in a PR runner's slot (pr_user.wrap: the
    package's `rk pr-run --env /dev/stdin`) gets `env` on its stdin."""
    t0 = time.monotonic()
    data = pr_user.env_data(env) if pr_user.launched(argv) else None
    try:
        p = subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
            input=data,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return Result(None, "", round(time.monotonic() - t0, 2), timeout)
    except OSError as exc:
        return Result(
            127,
            f"cannot run {argv[0]}: {exc}",
            round(time.monotonic() - t0, 2),
            timeout,
        )
    stdout, stderr = (x.decode("utf-8", "replace") for x in (p.stdout, p.stderr))
    out = stdout + ("\n[stderr]\n" + stderr if stderr.strip() else "")
    return Result(p.returncode, out, round(time.monotonic() - t0, 2), timeout)


def run(argv, cwd, timeout: int, env: dict) -> str:
    return execute(argv, cwd, timeout, env).text()


# ---- the argv builders: (argv, cwd, timeout, extra environment) or an error string for the model --------------------


def grep_cmd(t: Tree, pattern: str = "", path: str = "."):
    if not pattern or len(pattern) > 200:
        return "pattern required (<=200 chars)"
    base = safe(t.wt, path) if path not in ("", ".") else t.wt
    if not base:
        return f"bad path: {path!r}"
    # Each unescaped `|` alternative is its own -e pattern (BRE reads `a|b` literally and matches nothing); BRE
    # stays, so a literal call such as "bar(x)" still matches itself.
    argv = ["grep", "-rnI"]
    for alt in [a for a in re.split(r"(?<!\\)\|", pattern) if a] or [pattern]:
        argv += ["-e", alt]
    return argv + ["--", str(base)], t.wt, 60, {}


def pytest_files(t: Tree, paths: str = ""):
    """The test files (or node ids, file::name) to run: [(absolute file or node id)], or an error string."""
    files = []
    for rel in [x for x in re.split(r"[\s,]+", paths or "") if x]:
        rel_file, sep, node = rel.partition("::")
        p = safe(t.wt, rel_file)
        if (
            not p
            or not p.is_file()
            or p.suffix != ".py"
            or (sep and not re.fullmatch(r"[\w\[\]:.,=-]+", node))
        ):
            return f"not a .py test file in worktree: {rel!r}"
        files.append(str(p) + (f"::{node}" if sep else ""))
    return files or "give one or more existing .py test paths from the worktree"


def pytest_cmd(t: Tree, paths: str = "", plugins=()):
    files = pytest_files(t, paths)
    if isinstance(files, str):
        return files
    argv = [
        str(t.pybin),
        "-B",
        "-m",
        "pytest",
        *files,
        "-q",
        "-p",
        "no:cacheprovider",
        "--no-header",
        "-rfE",
        "-o",
        "addopts=",
        "--continue-on-collection-errors",
        "-p",
        "rkmon_env",
        f"--rootdir={t.rocke}",
    ]
    for p in plugins:
        argv += ["-p", p]
    return (
        argv,
        t.rocke,
        1200,
        {"PYTHONPATH": os.pathsep.join(map(str, t.pytest_path()))},
    )


def rule_scan_cmd(t: Tree):
    return (
        [
            str(t.pybin),
            "-I",
            str(t.tools_dir / "rule_scan.py"),
            str(t.pr / "diff.patch"),
        ],
        t.wt,
        60,
        {},
    )


def merge_conflicts_cmd(t: Tree):
    if not t.wtm.exists():
        return "merged-with-develop worktree not prepared"
    return (
        ["git", "-C", str(t.wtm), "diff", "--name-only", "--diff-filter=U"],
        t.wtm,
        60,
        {},
    )


def git_diff_cmd(t: Tree, path: str = ""):
    if path and not safe_rel(path):
        return f"bad path: {path!r}"
    base = t.sha("base")
    if not base:
        return "no merge base with develop in the head worktree"
    return (
        ["git", "-C", str(t.wt), "diff", base, "HEAD", "--", *([path] if path else [])],
        t.wt,
        60,
        {},
    )


def git_log_cmd(t: Tree, path: str = ""):
    if path and not safe_rel(path):
        return f"bad path: {path!r}"
    base = t.sha("base")
    if not base:
        return "no merge base with develop in the head worktree"
    return (
        [
            "git",
            "-C",
            str(t.wt),
            "log",
            "--format=%h %ad %s",
            "--date=short",
            f"{base}..HEAD",
            "--",
            *([path] if path else []),
        ],
        t.wt,
        60,
        {},
    )


def ci_fails_cmd(t: Tree):
    """`rk ci-fails N`: the head's failing and running check runs (GitHub reads only)."""
    return [t.rk(), "ci-fails", t.n], t.ws, 120, {}


def job_log_cmd(t: Tree, job: str = ""):
    """`rk job-log ID`: a CI job's failing lines; the whole log is kept in the review folder (read_job_log)."""
    if not re.fullmatch(r"\d{1,15}", str(job or "")):
        return "job: a CI job's numeric id (rk ci-fails lists each failing run's url, which ends in it)"
    (t.ws / "tmp").mkdir(parents=True, exist_ok=True)
    return [t.rk(), "job-log", str(job)], t.ws, 180, {"TMPDIR": str(t.ws / "tmp")}


def probe_file(t: Tree, name: str):
    """A probe by name: the model's own (<review folder>/probes/<name>.sh) first, then the saved one
    (<probes>/pr<N>/<name>.sh); (path, saved?) or an error string."""
    if not NAME.match(name or ""):
        return f"bad probe name: {name!r}"
    own, saved = t.ws / "probes" / f"{name}.sh", t.probes / f"{name}.sh"
    if own.is_file():
        return own, False
    if saved.is_file():
        return saved, True
    names = sorted(
        p.stem
        for d in (t.ws / "probes", t.probes)
        if d.is_dir()
        for p in d.glob("*.sh")
    )
    return f"no probe {name!r}; the probes here: {', '.join(names) or 'none'}"


def probe_cmd(t: Tree, name: str = "", tree: str = "merged", timeout: int = 600):
    """`bash <probe> <tree root>`, as run_probes.py runs a saved probe: exit 1 while the finding reproduces, 0 once it
    is fixed, anything else an error. The model's own probe runs only in the sandbox; a saved one in it when there is
    one."""
    found = probe_file(t, name)
    if isinstance(found, str):
        return found
    path, saved = found
    if tree not in ("head", "merged"):
        return "tree: head or merged"
    root = t.wtm if tree == "merged" else t.wt
    if not root.exists():
        return f"the {tree} tree is not prepared"
    argv = ["nice", "-n", "10", "bash", str(path), str(root)]
    work = (
        t.ws / "runs"
    )  # the probe's working folder on a host (in a PR runner's slot, the slot's home: PR_CODE)
    work.mkdir(parents=True, exist_ok=True)
    if sandbox.kind():
        wrapped = sandbox.wrap(
            argv, t.ws, work, readonly=_readonly(t), timeout_s=timeout
        )
        return wrapped, work, timeout + 30, {"__sandbox__": "1"}
    if not saved:
        return (
            "refused: the model's own probes run only in a sandbox, and this machine has none ("
            + sandbox.describe()
            + ")"
        )
    return ["timeout", str(timeout), *argv], work, timeout + 30, {}


def mutant_cmd(
    t: Tree, name: str = "", tests: str = "", control: bool = False, timeout: int = 600
):
    """pytest on `tests` with the mutant plugin <review folder>/mutants/rkmut_<name>.py loaded (`-p`), or without it
    (the control), in the sandbox."""
    if not NAME.match(name or ""):
        return f"bad mutant name: {name!r}"
    plugin = t.ws / "mutants" / f"rkmut_{name}.py"
    if not plugin.is_file():
        return f"no mutant {name!r}: write it first (run_mutant's plugin)"
    if not sandbox.kind():
        return (
            "refused: mutants run only in a sandbox, and this machine has none ("
            + sandbox.describe()
            + ")"
        )
    built = pytest_cmd(t, tests, plugins=() if control else (f"rkmut_{name}",))
    if isinstance(built, str):
        return built
    argv, _cwd, _timeout, env = built
    env = {
        "PYTHONPATH": os.pathsep.join([str(plugin.parent), env["PYTHONPATH"]]),
        "__sandbox__": "1",
    }
    return (
        sandbox.wrap(argv, t.ws, t.rocke, readonly=_readonly(t), timeout_s=timeout),
        t.rocke,
        timeout + 30,
        env,
    )


def _readonly(t: Tree) -> list:
    """What the unshare sandbox binds read-only (bubblewrap makes everything read-only)."""
    return [
        t.root,
        os.environ.get("HOME") or "",
        os.environ.get("RKMON_HOME") or "",
        t.toolkit,
    ]


# The tools a recorded run may name, and their builders: `rk evidence-check` rebuilds the argv from the tool and its
# arguments, never from the record.
BUILDERS = {
    "grep_code": lambda t, a: grep_cmd(t, a.get("pattern", ""), a.get("path", ".")),
    "run_pytest": lambda t, a: pytest_cmd(t, a.get("paths", "")),
    "rule_scan": lambda t, a: rule_scan_cmd(t),
    "merge_conflicts": lambda t, a: merge_conflicts_cmd(t),
    "git_diff": lambda t, a: git_diff_cmd(t, a.get("path", "")),
    "git_log": lambda t, a: git_log_cmd(t, a.get("path", "")),
    "ci_fails": lambda t, a: ci_fails_cmd(t),
    "job_log": lambda t, a: job_log_cmd(t, a.get("job", "")),
    "run_probe": lambda t, a: probe_cmd(
        t, a.get("name", ""), a.get("tree") or "merged", int(a.get("timeout") or 600)
    ),
    "run_mutant": lambda t, a: mutant_cmd(
        t, a.get("name", ""), a.get("tests", ""), bool(a.get("control"))
    ),
}


def build(t: Tree, tool: str, args: dict):
    """(argv, cwd, timeout, environment) for a tool run, or an error string."""
    if tool not in BUILDERS:
        return f"unknown tool {tool!r}"
    built = BUILDERS[tool](t, args or {})
    if isinstance(built, str):
        return built
    argv, cwd, timeout, extra = built
    env = t.env()
    if extra.pop("__sandbox__", None):
        env = sandbox.env(env, t.ws)
    env.update(extra)
    if tool in PR_CODE:
        argv, _data = pr_user.wrap(argv, env, cwd if PR_CODE[tool] else None)
    return argv, cwd, timeout, env


def run_tool(t: Tree, tool: str, args: dict) -> tuple[str, Result | None, list]:
    """Run a tool: (what the model sees, the result or None for a refusal, the argv)."""
    built = build(t, tool, args)
    if isinstance(built, str):
        return built, None, []
    argv, cwd, timeout, env = built
    r = execute(argv, cwd, timeout, env)
    return r.text(), r, argv


# ---- the first-pass reviewer's tools (review_agent.py) ----------------------------------------------------------------


def read_file(t: Tree, path: str = "", **_) -> str:
    p = safe(t.wt, path)
    if not p or not p.is_file():
        return f"no such file in worktree: {path!r}"
    txt = p.read_text(encoding="utf-8", errors="replace")
    return txt[:READ_CAP] + (
        f"\n...[truncated {len(txt) - READ_CAP}]..." if len(txt) > READ_CAP else ""
    )


def grep_code(t: Tree, pattern: str = "", path: str = ".", **_) -> str:
    return run_tool(t, "grep_code", {"pattern": pattern, "path": path})[0]


def run_pytest(t: Tree, paths: str = "", **_) -> str:
    return run_tool(t, "run_pytest", {"paths": paths})[0]


def rule_scan(t: Tree, **_) -> str:
    return run_tool(t, "rule_scan", {})[0]


def merge_conflicts(t: Tree, **_) -> str:
    out = run_tool(t, "merge_conflicts", {})[0]
    return out or "(no conflicts)"


CLASSIC = {
    "read_file": read_file,
    "grep_code": grep_code,
    "run_pytest": run_pytest,
    "rule_scan": rule_scan,
    "merge_conflicts": merge_conflicts,
}


def schema(name: str, description: str, props: dict | None = None, required=()) -> dict:
    """A tool's function schema, as the chat request offers it: {name: type} properties, the required ones."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": v} for k, v in (props or {}).items()},
                **({"required": list(required)} if required else {}),
            },
        },
    }


CLASSIC_SCHEMAS = [
    schema(
        "read_file",
        "Read a file from the PR head worktree. Param: path (relative to the worktree root).",
        {"path": "string"},
        ["path"],
    ),
    schema(
        "grep_code",
        "Recursively grep the worktree for a literal/regex pattern. Params: pattern, optional path.",
        {"pattern": "string", "path": "string"},
        ["pattern"],
    ),
    schema(
        "run_pytest",
        "Run the repo's OWN pytest file(s) from the worktree on the review machine (hipcc available; tests that "
        "need a GPU skip when it has none). "
        "Param: paths (space/comma separated existing .py test paths). Use this to actually verify behavior.",
        {"paths": "string"},
        ["paths"],
    ),
    schema(
        "rule_scan",
        "Scan the PR diff for repo-rule violations (tracker IDs, absolute perf numbers, kernel-selecting "
        "env reads).",
    ),
    schema(
        "merge_conflicts",
        "List files that conflict when the PR is merged with current develop.",
    ),
]


# ---- line reads for quotes (the offload passes' read evidence) --------------------------------------------------------


def file_lines(t: Tree, path: str, tree: str = "head") -> list[str] | str:
    """A file's lines in one of the PR's trees: head (wt<N>), merged (wt<N>m), base (the merge base, from git)."""
    if tree not in TREES:
        return f"tree: one of {', '.join(TREES)}"
    if tree == "base":
        if not safe_rel(path):
            return f"bad path: {path!r}"
        base = t.sha("base")
        p = subprocess.run(
            ["git", "-C", str(t.wt), "show", f"{base}:{path}"],
            capture_output=True,
            timeout=60,
            env=dict(os.environ, GIT_NO_LAZY_FETCH="1"),
        )
        if not base or p.returncode:
            return f"no {path!r} at the merge base {base[:10]}"
        return p.stdout.decode("utf-8", "replace").splitlines()
    p = safe(t.wtm if tree == "merged" else t.wt, path)
    if not p or not p.is_file():
        return f"no such file in the {tree} tree: {path!r}"
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


def read_lines(t: Tree, path: str = "", start=1, end=0, tree: str = "head", **_) -> str:
    """Numbered lines of a file in one of the PR's trees, to quote exactly."""
    lines = file_lines(t, path, tree)
    return (
        lines
        if isinstance(lines, str)
        else numbered(lines, f"{path} ({tree})", start, end)
    )


def numbered(lines: list[str], label: str, start=1, end=0) -> str:
    """Lines `start` to `end` of a text (200 when no end is given, LINES_CAP at most), numbered as `  12| text`."""
    try:
        a = max(1, int(start or 1))
        b = int(end or 0) or a + 199
    except (TypeError, ValueError):
        return "start and end: line numbers"
    b = min(b, a + LINES_CAP - 1, len(lines))
    if a > len(lines):
        return f"{label} has {len(lines)} lines"
    return f"{label}, lines {a}-{b} of {len(lines)}:\n" + "\n".join(
        f"{i:>6}| {lines[i - 1]}" for i in range(a, b + 1)
    )


def quote_at(lines: list[str], line: int, quote: str) -> tuple[bool, int | None]:
    """Whether `quote` (one or more lines, compared without surrounding white space) is the file's text from `line`;
    and where it is when it is elsewhere (the first place), else None."""
    want = [q.strip() for q in str(quote).strip("\n").splitlines()]
    if not want or not any(want):
        return False, None
    # One line may be quoted in part (12 characters at least); several lines are quoted whole.
    part = len(want) == 1 and len(want[0]) >= QUOTE_PART_MIN

    def at(i: int) -> bool:
        return (
            i >= 1
            and i - 1 + len(want) <= len(lines)
            and all(
                lines[i - 1 + k].strip() == w or (part and w in lines[i - 1 + k])
                for k, w in enumerate(want)
            )
        )

    if at(int(line or 0)):
        return True, int(line)
    return False, next((i for i in range(1, len(lines) + 1) if at(i)), None)


# ---- what a run's output says, for comparing a re-run with the record -------------------------------------------------

PYTEST_COUNT = re.compile(
    r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)"
)


def normalize(tool: str, out: str, t: Tree | None = None) -> str:
    """A run's output reduced to what must match on a re-run: pytest's counts and failing node ids, grep's matches,
    a probe's verdict and last line, CI's failing and running names; the whole output, else, without the trees' own
    paths."""
    text = out or ""
    for root in ([t.wtm, t.wt, t.ws] if t else []):
        text = text.replace(str(root) + "/", "").replace(str(root), ".")
    if tool in ("run_pytest", "run_mutant"):
        counts = {}
        for n, kind in PYTEST_COUNT.findall(text):
            counts[kind.rstrip("s") if kind.startswith("error") else kind] = int(n)
        bad = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.M)))
        return json.dumps({"counts": counts, "failing": bad}, sort_keys=True)
    if tool in ("grep_code", "merge_conflicts", "rule_scan", "git_log"):
        return "\n".join(
            sorted(
                x.rstrip()
                for x in text.splitlines()
                if x.strip() and x != "(no output)"
            )
        )
    if tool == "run_probe":
        return text.strip().splitlines()[-1].strip() if text.strip() else ""
    if tool == "ci_fails":
        return "\n".join(
            sorted(
                "\t".join(x.split("\t")[:2])
                for x in text.splitlines()
                if x.startswith(("FAIL\t", "RUN\t"))
            )
        )
    return re.sub(r"\s+\n", "\n", text).strip()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def excerpt(out: str, lines: int = 6) -> str:
    """The first and last lines of an output, for the record."""
    rows = [x for x in (out or "").splitlines() if x.strip()]
    if len(rows) <= 2 * lines:
        return "\n".join(rows)[:1200]
    return ("\n".join(rows[:lines]) + "\n...\n" + "\n".join(rows[-lines:]))[:1200]


def probe_verdict(r: Result | None) -> str:
    if r is None:
        return ""
    if r.rc is None:
        return "ERROR (timeout)"
    return {0: "FIXED", 1: "REPRODUCES"}.get(r.rc, f"ERROR rc={r.rc}")
