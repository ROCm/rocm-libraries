# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The drafter's tools: a fixed menu of vetted, read-only tools on one PR's trees, shared by the tool-loop drafter
(drafter.py) and `rk evidence-check` (rkmon_lib/evidence.py), which re-runs a recorded tool run with the same argv
builder. The model never supplies a shell command: every tool builds its argv from validated parameters, with
shell=False; a path is relative to the tree it names, never absolute, never `..`. Local work stays light: git, grep,
file reads and the toolkit's diff scanner (rule_scan.py); no test or build runs here (the checks job runs those). One
tool is the script's own, never offered to the model: `probe`, one of our saved probes (tools/run_probes.py) on a tree;
it runs the PR's code, so it runs in a PR runner's slot where there are PR runners (the container: pr_user.py).

`Tree` holds the PR's trees (wt<N>, the head; wt<N>m, the head merged with develop; the merge base, read from git) and
the machine's paths: the review root (config paths.root), the project's path inside the repository, the checks' venv
(its Python runs the scanner), the saved probes (<root>/probes/pr<N>) and the temp folder. The trees are under the
review root unless a run names another folder (`trees`), as a draft outside the pipeline does.

Git reads by object id (`blobs`, `has_commit`) never fetch: they need only commits and trees, which a partial clone has.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pr_user  # noqa: E402

SAFE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./+-]*$")
OUT_CAP = (
    6000  # what one tool result shows the model: the head and tail of a longer output
)
LINES_CAP = 300  # read_lines: the most lines in one call
QUOTE_CAP = 600  # a recorded quote: a few lines
QUOTE_PART_MIN = 12
TREES = ("head", "merged", "base")
PROBE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.+-]*$")
PROBE_RUNNER = (
    pathlib.Path(__file__).resolve().parent.parent / "tools" / "run_probes.py"
)
# One line of run_probes.py: "<name>: REPRODUCES|FIXED|ERROR rc=N[ (crashed)]: <summary>".
PROBE_LINE = re.compile(
    r"^(\S+): (REPRODUCES|FIXED|ERROR rc=[^\s:]+(?: \(crashed\))?):? ?(.*)$", re.M
)
NO_FETCH = {"GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"}
PR_CODE = (
    "probe",
)  # the tools that run PR code: in a PR runner's slot (pr_user.wrap), in their tree


class Tree:
    """One PR's trees and the machine's paths."""

    def __init__(self, ws, n, root, project_path: str, trees=None):
        self.ws, self.n = pathlib.Path(ws), str(n)
        self.root = pathlib.Path(root)
        base = pathlib.Path(trees) if trees else self.root
        self.trees = base
        self.wt = (base / f"wt{n}").resolve()
        self.wtm = (base / f"wt{n}m").resolve()
        self.project_path = project_path.strip("/")
        self.project = self.wt / self.project_path
        venv = self.root / "venv" / "bin" / "python"
        self.pybin = venv if venv.exists() else pathlib.Path(sys.executable)
        self.rule_scan = self.root / "tools" / "rule_scan.py"
        self.probes = self.root / "probes" / f"pr{self.n}"
        self.tmp = self.root / "tmp"
        self.pr = self.ws / "pr"
        self._base = None

    def env(self) -> dict:
        e = {
            k: v for k, v in os.environ.items() if k not in ("GH_TOKEN", "GITHUB_TOKEN")
        }
        self.tmp.mkdir(parents=True, exist_ok=True)
        e["TMPDIR"] = str(self.tmp)
        return e

    def sha(self, tree: str = "head") -> str:
        """The commit a tree holds: head (wt<N>), merged (wt<N>m) or base (the head's merge base with develop)."""
        if tree == "base":
            if self._base is None:
                self._base = _git(
                    self.wt, "merge-base", "HEAD", "refs/remotes/origin/develop"
                )
            return self._base
        return _git(self.wtm if tree == "merged" else self.wt, "rev-parse", "HEAD")

    def resolve(self, rel: str) -> str:
        """A repository path the model named: as given when it is in the head tree, else under the project path (the PR
        map shows paths without it); as given when neither exists (a file only at the merge base).
        """
        rel = (
            (rel or "").strip().lstrip("./")
            if rel and not rel.startswith("/")
            else (rel or "")
        )
        if not safe_rel(rel):
            return rel
        if (self.wt / rel).exists():
            return rel
        long = f"{self.project_path}/{rel}"
        return long if (self.wt / long).exists() else rel

    def short(self, rel: str) -> str:
        """A repository path without the project path, as the post shows it."""
        return (
            rel[len(self.project_path) + 1 :]
            if rel.startswith(self.project_path + "/")
            else rel
        )

    def probe_names(self) -> list[str]:
        """Our saved probes for this PR (<probes>/<name>.sh, not the retired ones), in name order."""
        try:
            return sorted(
                p.stem
                for p in self.probes.glob("*.sh")
                if p.is_file() and PROBE_NAME.match(p.stem)
            )
        except OSError:
            return []


def _git(tree: pathlib.Path, *args) -> str:
    try:
        p = subprocess.run(
            ["git", "-C", str(tree), *args], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return p.stdout.strip() if p.returncode == 0 else ""


def git_out(repo, *args, fetch: bool = False, timeout: int = 60) -> tuple[int, str]:
    """(exit code, stdout) of a git command in `repo`; with fetch=False a partial clone fetches nothing (a missing
    object is an error). 127 when git could not run."""
    env = dict(os.environ) if fetch else dict(os.environ, **NO_FETCH)
    try:
        p = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return 127, ""
    return p.returncode, p.stdout


def has_commit(repo, sha: str) -> bool:
    """Whether the commit is in the clone (no fetch)."""
    return bool(sha) and git_out(repo, "cat-file", "-e", f"{sha}^{{commit}}")[0] == 0


def full_sha(repo, sha: str) -> str:
    """The full id of a commit the clone has ("" when it has none by that name)."""
    rc, out = (
        git_out(repo, "rev-parse", "--verify", "-q", f"{sha}^{{commit}}")
        if sha
        else (1, "")
    )
    return out.strip() if rc == 0 else ""


def blobs(repo, commit: str, paths: list[str]) -> dict[str, str] | None:
    """Each path's blob id at a commit, "-" where the commit has no such file; None when the commit is not in the
    clone. Trees only: nothing is fetched."""
    if not has_commit(repo, commit):
        return None
    found: dict[str, str] = {}
    for i in range(0, len(paths), 200):
        rc, out = git_out(repo, "ls-tree", "-z", commit, "--", *paths[i : i + 200])
        if rc:
            return None
        for ent in out.split("\0"):
            meta, _, path = ent.partition("\t")
            if path and len(meta.split()) == 3:
                found[path] = meta.split()[2]
    return {p: found.get(p, "-") for p in paths}


def merge_base(repo, a: str, b: str) -> str:
    rc, out = git_out(repo, "merge-base", a, b)
    return out.strip() if rc == 0 else ""


def tree_files(repo, commit: str = "HEAD", under: str = "") -> list[str]:
    """The files of a commit (under a folder), from its trees."""
    rc, out = git_out(
        repo,
        "ls-tree",
        "-r",
        "-z",
        "--name-only",
        commit,
        *(["--", under] if under else []),
        timeout=120,
    )
    return [p for p in out.split("\0") if p] if rc == 0 else []


def diff_text(repo, a: str, b: str, path: str, cap: int = 2500) -> str:
    """The diff of one file between two commits (blobs a partial clone lacks are fetched), at most `cap` characters;
    "" when there is none or git fails."""
    rc, out = git_out(
        repo, "diff", "--no-color", "-U2", a, b, "--", path, fetch=True, timeout=120
    )
    if rc or not out.strip():
        return ""
    return (
        out
        if len(out) <= cap
        else out[:cap] + f"\n[{len(out) - cap} more characters of this diff]"
    )


def safe(tree: pathlib.Path, rel: str):
    """rel resolved inside tree; None for an absolute path, `..`, or an odd character."""
    if not safe_rel(rel):
        return None
    p = (tree / rel).resolve()
    return p if str(p).startswith(str(tree) + os.sep) else None


def safe_rel(rel: str) -> bool:
    """A repository path the model may name (it need not exist: a file only at the merge base)."""
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
        if self.rc is None:
            return f"TIMEOUT after {self.timeout}s."
        return clip(self.out)


def clip(out: str, cap: int = OUT_CAP) -> str:
    if len(out) > cap:
        out = (
            out[: cap // 2]
            + f"\n...[{len(out) - cap} characters cut]...\n"
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
            input=None if data is None else data.decode("utf-8", "surrogateescape"),
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
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
    out = p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr.strip() else "")
    return Result(p.returncode, out, round(time.monotonic() - t0, 2), timeout)


# ---- the argv builders: (argv, cwd, timeout, extra environment) or an error string for the model --------------------


def grep_cmd(t: Tree, pattern: str = "", path: str = "."):
    if not pattern or len(pattern) > 200:
        return "pattern required (at most 200 characters)"
    rel = t.resolve(path) if path not in ("", ".") else "."
    base = safe(t.wt, rel) if rel != "." else t.wt
    if not base:
        return f"bad path: {path!r}"
    # Each unescaped `|` alternative is its own -e pattern (basic regular expressions read `a|b` literally).
    argv = ["grep", "-rnI", "--exclude-dir=.git"]
    for alt in [a for a in re.split(r"(?<!\\)\|", pattern) if a] or [pattern]:
        argv += ["-e", alt]
    return argv + ["--", str(base)], t.wt, 60, {}


def rule_scan_cmd(t: Tree):
    if not t.rule_scan.is_file():
        return f"no rule_scan.py at {t.rule_scan}"
    return (
        [str(t.pybin), "-I", str(t.rule_scan), str(t.pr / "diff.patch")],
        t.wt,
        60,
        {},
    )


def merge_conflicts_cmd(t: Tree):
    if not t.wtm.exists():
        return "the merged-with-develop worktree is not prepared"
    return (
        ["git", "-C", str(t.wtm), "diff", "--name-only", "--diff-filter=U"],
        t.wtm,
        60,
        {},
    )


def git_log_cmd(t: Tree, path: str = ""):
    rel = t.resolve(path) if path else ""
    if rel and not safe_rel(rel):
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
            *([rel] if rel else []),
        ],
        t.wt,
        60,
        {},
    )


def probe_cmd(t: Tree, name: str = "", tree: str = "merged", timeout_s=600):
    """One saved probe on a tree (merged or head), through tools/run_probes.py: its line says REPRODUCES, FIXED or
    ERROR. A probe checks its own controls: exit 2 (it cannot show its finding on this tree) reads as ERROR. The argv
    names `python3` from PATH, never this interpreter's path: `rk evidence-check` rebuilds it in another process and
    compares it with the recorded one."""
    if not PROBE_NAME.match(str(name or "")) or not (t.probes / f"{name}.sh").is_file():
        return f"no saved probe {name!r} in {t.probes}"
    if tree not in ("merged", "head"):
        return "tree: merged or head"
    root = t.wtm if tree == "merged" else t.wt
    if not root.exists():
        return f"the {tree} tree is not prepared"
    try:
        secs = max(1, int(timeout_s))
    except (TypeError, ValueError):
        return "timeout_s: seconds"
    return (
        ["python3", "-I", "-B", str(PROBE_RUNNER), str(t.probes), str(root), str(name)],
        t.wt,
        secs + 30,
        {"PROBE_TIMEOUT": str(secs)},
    )


# The tools a recorded run may name, and their builders: `rk evidence-check` rebuilds the argv from the tool and its
# arguments, never from the record.
BUILDERS = {
    "grep_code": lambda t, a: grep_cmd(t, a.get("pattern", ""), a.get("path", ".")),
    "rule_scan": lambda t, a: rule_scan_cmd(t),
    "merge_conflicts": lambda t, a: merge_conflicts_cmd(t),
    "git_log": lambda t, a: git_log_cmd(t, a.get("path", "")),
    "probe": lambda t, a: probe_cmd(
        t, a.get("name", ""), a.get("tree", "merged"), a.get("timeout_s", 600)
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
    env.update(extra)
    if tool in PR_CODE:
        argv, _data = pr_user.wrap(argv, env, cwd)
    return argv, cwd, timeout, env


def run_tool(t: Tree, tool: str, args: dict) -> tuple[str, Result | None, list]:
    """Run a tool: (what the model sees, the result or None for a refusal, the argv)."""
    built = build(t, tool, args)
    if isinstance(built, str):
        return built, None, []
    argv, cwd, timeout, env = built
    r = execute(argv, cwd, timeout, env)
    return r.text(), r, argv


def probe_verdict(out: str) -> tuple[str, str]:
    """A probe line's verdict (REPRODUCES, FIXED, or ERROR ...) and its summary; ("ERROR", why) when there is none."""
    m = PROBE_LINE.search(out or "")
    if m:
        return m.group(2), m.group(3).strip()
    rows = (out or "").strip().splitlines()
    return "ERROR", rows[-1][:200] if rows else "no output"


# ---- line reads for quotes -------------------------------------------------------------------------------------------


def file_lines(t: Tree, path: str, tree: str = "head") -> list[str] | str:
    """A file's lines in one of the PR's trees: head (wt<N>), merged (wt<N>m), base (the merge base, from git)."""
    if tree not in TREES:
        return f"tree: one of {', '.join(TREES)}"
    rel = t.resolve(path)
    if tree == "base":
        if not safe_rel(rel):
            return f"bad path: {path!r}"
        base = t.sha("base")
        if not base:
            return "no merge base with develop in the head worktree"
        try:
            p = subprocess.run(
                ["git", "-C", str(t.wt), "show", f"{base}:{rel}"],
                capture_output=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"git show failed: {exc}"
        if p.returncode:
            return f"no {rel!r} at the merge base {base[:10]}"
        return p.stdout.decode("utf-8", "replace").splitlines()
    p = safe(t.wtm if tree == "merged" else t.wt, rel)
    if not p or not p.is_file():
        return f"no such file in the {tree} tree: {path!r}"
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


def read_lines(t: Tree, path: str = "", start=1, end=0, tree: str = "head") -> str:
    """Numbered lines of a file in one of the PR's trees, to quote exactly."""
    lines = file_lines(t, path, tree)
    return (
        lines
        if isinstance(lines, str)
        else numbered(lines, f"{t.resolve(path)} ({tree})", start, end)
    )


def numbered(lines: list[str], label: str, start=1, end=0) -> str:
    """Lines `start` to `end` of a text (120 when no end is given, LINES_CAP at most), numbered as `  12| text`."""
    try:
        a = max(1, int(start or 1))
        b = int(end or 0) or a + 119
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
    and where it is when it is elsewhere (the first place), else None. One line may be quoted in part (QUOTE_PART_MIN
    characters at least); several lines are quoted whole."""
    want = [q.strip() for q in str(quote).strip("\n").splitlines()]
    if not want or not any(want):
        return False, None
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

    try:
        n = int(line or 0)
    except (TypeError, ValueError):
        n = 0
    if at(n):
        return True, n
    return False, next((i for i in range(1, len(lines) + 1) if at(i)), None)


# ---- what a run's output says, for comparing a re-run with the record ---------------------------------------------


def normalize(tool: str, out: str, t: Tree | None = None) -> str:
    """A run's output reduced to what must match on a re-run: grep's matches, the conflict list, the scanner's hits, the
    commit list, a probe's verdict; the whole output else, without the trees' own paths.
    """
    text = out or ""
    for root in ([t.wtm, t.wt, t.ws] if t else []):
        text = text.replace(str(root) + "/", "").replace(str(root), ".")
    if tool in ("grep_code", "merge_conflicts", "rule_scan", "git_log"):
        return "\n".join(
            sorted(
                x.rstrip()
                for x in text.splitlines()
                if x.strip() and x != "(no output)"
            )
        )
    if tool == "probe":
        return "\n".join(
            f"{m.group(1)}: {m.group(2)}" for m in PROBE_LINE.finditer(text)
        )
    return re.sub(r"\s+\n", "\n", text).strip()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def file_digest(path: pathlib.Path) -> str:
    """A file's content digest ("" when it cannot be read): a saved probe's, for its run's record."""
    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def excerpt(out: str, lines: int = 6) -> str:
    """The first and last lines of an output, for the record."""
    rows = [x for x in (out or "").splitlines() if x.strip()]
    if len(rows) <= 2 * lines:
        return "\n".join(rows)[:1200]
    return ("\n".join(rows[:lines]) + "\n...\n" + "\n".join(rows[-lines:]))[:1200]


def schema(name: str, description: str, props: dict | None = None, required=()) -> dict:
    """A tool's function schema as the request offers it: {name: type} properties, the required ones."""

    def prop(v):
        if isinstance(v, dict):
            return v
        if v == "array":
            return {"type": "array", "items": {"type": "string"}}
        return {"type": v}

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: prop(v) for k, v in (props or {}).items()},
                **({"required": list(required)} if required else {}),
            },
        },
    }
