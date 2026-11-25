# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Saved probes (tools/run_probes.py): the check before one is saved (`rk probe-check`) and the port of probes written
for another layout to the probe environment contract (`rk probe-port`).

A saved probe is `bash <probe>.sh <tree root>`: exit 1 while its finding reproduces, 0 once fixed, 2 on a tree where
the finding cannot show (no PR change, or a missing develop change), 3 when it cannot tell. A probe is trusted blindly
by the next round, so before it is saved it must read:
- 2 on DEVBASE (the round's develop: the PR's change is not there),
- 1 on the tree (the PR's head, or its merge with develop),
- 0 on a simulated fix (our fix line written another way, applied to a copy of the tree).
`check` runs it on the three trees through tools/run_probes.py (the same clean environment the checks give it, in a PR
runner's slot of its own: prcode.run) and prints `<probe>: DEVBASE 2 · tree 1 · simfix 0 · OK`. An uncaught exception
(run_probes' "crashed"), pytest's rc 5 (it ran no test) and every other exit code read 3. A tree is a folder given as
it is, or a commit of the tree's clone exported with `git archive` into a scratch folder (only the folders a worktree
checks out, paths.sparse_dirs), never a worktree of the clone: nothing of the clone's own state changes.

`flags` reads a probe and its helper files (the files of its folder its script names, or that start with its name):
- a hardcoded host path (a home folder, `~/`, `$HOME/`, `/opt/rocm...`, `/mnt/<drive>/`): a probe names no machine's
  folder; the contract's variables (PROBE_PYTHON, RKMON_ROCM_LLVM_BIN, RKMON_BUILD_DIR, RKMON_REPO, RKMON_REPO_DIR)
  and its tree root ($1) say where things are (`rk probe-port` rewrites the common forms);
- a grep of pytest's `^FAILED` lines with no `SUBFAILED`: pytest 9 prints a failing subtest as SUBFAILED and passes its
  parent;
- HIP_VISIBLE_DEVICES=-1, which the HIP runtime refuses with an abort (rc 134) as soon as torch is imported;
- a script that never reads its tree root ($1).

`port` rewrites one probe file from a live layout (a root folder in a home directory, LIVE_ROOT_NAMES or `--root`, and
ROCm under /opt/rocm) to the contract: shell scripts textually, Python helpers through their string literals and
comments (tokenize). What it cannot rewrite safely stays, and is listed with its reason (a hand edit).
"""

import io
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize

from . import prcode

PACKAGE = pathlib.Path(__file__).resolve().parent.parent
RUN_PROBES = PACKAGE / "tools" / "run_probes.py"
WANT = {"DEVBASE": 2, "tree": 1, "simfix": 0}
LINE = re.compile(
    r"^(?P<name>.+?): (?:REPRODUCES|FIXED|ERROR) rc=(?P<rc>-?\d+)(?P<crash> \(crashed\))?"
)
SEP = " · "

# --- the static flags ---------------------------------------------------------------------------------------------------

HOST_PATH = re.compile(
    r"(?:/home/[A-Za-z0-9_.-]+(?=/)|/Users/[A-Za-z0-9_.-]+(?=/)|/root(?=/)|(?<![\w$])~(?=/)|\$HOME\b|\$\{HOME\}"
    r"|\"\$HOME\"|/opt/rocm[A-Za-z0-9_.-]*|/mnt/[a-z](?=/))"
)
FAILED_GREP = re.compile(r"\^FAILED")
HIP_OFF = re.compile(r"HIP_VISIBLE_DEVICES=[\"']?-1")
READS_ROOT = re.compile(r"\$\{?1\b|\$\{?@|\$\*|\"\$@\"")


def _shown(path: str) -> str:
    """A path as a report shows it: the home folder's user name masked."""
    return re.sub(
        r"^/home/[^/]+", "/home/<user>", re.sub(r"^/Users/[^/]+", "/Users/<user>", path)
    )


def _code_lines(text: str, python: bool):
    """(line number, line) of the lines that are not comments."""
    for i, line in enumerate(text.splitlines(), 1):
        s = line.lstrip()
        if s.startswith("#") and not (i == 1 and s.startswith("#!")):
            continue
        yield i, line


def helpers(script: pathlib.Path) -> list[pathlib.Path]:
    """The files of a probe's folder that belong to it: those its script names, or whose name starts with its name."""
    try:
        text = script.read_text(encoding="utf-8", errors="replace")
        files = sorted(p for p in script.parent.iterdir() if p.is_file())
    except OSError:
        return []
    stem = script.stem
    return [
        p
        for p in files
        if p != script
        and p.suffix != ".sh"
        and not p.name.endswith(".pyc")
        and (p.name in text or p.name.startswith(stem + "_") or p.stem == stem)
    ]


def flags(script: pathlib.Path) -> list[str]:
    """The static flags of a probe (the module's docstring), each `<file>:<line>: <what>`."""
    out = []
    files = [script] + helpers(script)
    texts = {}
    for f in files:
        try:
            texts[f] = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    all_text = "\n".join(texts.values())
    for f, text in texts.items():
        py = f.suffix == ".py"
        for i, line in _code_lines(text, py):
            for m in HOST_PATH.finditer(line):
                rest = line[m.end() :]
                what = _shown(m.group(0) + re.split(r"[\s\"':;)]", rest, 1)[0])
                out.append(
                    f"{f.name}:{i}: hardcoded host path {what}: use the contract's variables "
                    "(PROBE_PYTHON, RKMON_ROCM_LLVM_BIN, RKMON_BUILD_DIR, RKMON_REPO_DIR) and $1 (rk probe-port)"
                )
                break
            if FAILED_GREP.search(line) and "SUBFAILED" not in all_text:
                out.append(
                    f"{f.name}:{i}: greps ^FAILED without SUBFAILED: pytest prints a failing subtest as SUBFAILED "
                    "and passes its parent"
                )
            if HIP_OFF.search(line):
                out.append(
                    f"{f.name}:{i}: sets HIP_VISIBLE_DEVICES=-1, which the HIP runtime refuses with an abort "
                    "(rc 134) once torch is imported: leave it unset"
                )
    if script in texts and not READS_ROOT.search(texts[script]):
        out.append(f"{script.name}: never reads its tree root ($1)")
    return out


# --- running a probe on the three trees --------------------------------------------------------------------------------


def reading(rc: int, crashed: bool) -> int:
    """The probe's verdict code: 0, 1 and 2 as they are; a crash, pytest's rc 5 and every other code read 3."""
    return rc if rc in (0, 1, 2) and not crashed else 3


def run_on(folder: pathlib.Path, tree: pathlib.Path, only: str, env: dict) -> dict:
    """{probe name: (verdict code, the run_probes line)} on one tree, in a PR runner's slot of its own (prcode.run),
    with what worker/pr_env.py lists of `env`.
    """
    argv = [sys.executable, "-I", "-B", str(RUN_PROBES), str(folder), str(tree)]
    if only:
        argv.append(only)
    p = prcode.run(
        argv, env=prcode.with_trees(dict(env, PROBE_DETAIL="1"), tree), timeout=None
    )
    out = {}
    for line in p.stdout.splitlines():
        m = LINE.match(line)
        if m:
            out[m.group("name")] = (
                reading(int(m.group("rc")), bool(m.group("crash"))),
                line,
            )
    return out


def git(tree: pathlib.Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def devbase_rev(tree: pathlib.Path, base: str) -> str:
    """The round's develop for a tree: the first parent of a merge (the merged tree), else the merge base with the base
    branch (origin's, else the local one)."""
    parents = git(tree, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if len(parents) == 3:
        return parents[1]
    for ref in (f"origin/{base}", base):
        mb = git(tree, "merge-base", "HEAD", ref)
        if mb:
            return mb
    return ""


def export(tree: pathlib.Path, rev: str, dest: pathlib.Path, dirs: list[str]) -> str:
    """`git archive REV [DIRS]` of the tree's clone into DEST; "" when done, else why not. The archive's pipes are
    closed, and git waited for, when it is read."""
    dest.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen(
        ["git", "-C", str(tree), "archive", "--format=tar", rev, "--", *dirs],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as a:
        t = subprocess.run(
            ["tar", "-x", "-C", str(dest)], stdin=a.stdout, capture_output=True
        )
        a.stdout.close()
        err = a.stderr.read().decode(errors="replace").strip()
    if a.returncode or t.returncode:
        return f"git archive {rev[:10]} failed: {err or t.stderr.decode(errors='replace').strip()}"
    return ""


def worktree_rev(tree: pathlib.Path) -> str:
    """The tree as it is: its uncommitted changes as a commit object (`git stash create`, no ref), else HEAD."""
    return git(tree, "stash", "create") or git(tree, "rev-parse", "HEAD")


def apply_patch(tree: pathlib.Path, patch: pathlib.Path) -> str:
    p = subprocess.run(
        ["git", "apply", "--unsafe-paths", "--directory=.", str(patch.resolve())],
        cwd=tree,
        capture_output=True,
        text=True,
    )
    return (
        ""
        if p.returncode == 0
        else f"the simulated fix does not apply: {p.stderr.strip()[:300]}"
    )


def check(
    folder: pathlib.Path,
    tree: pathlib.Path,
    *,
    only: str = "",
    devbase: str = "",
    devbase_tree: pathlib.Path | None = None,
    simfix: pathlib.Path | None = None,
    simfix_tree: pathlib.Path | None = None,
    base: str = "develop",
    dirs: list[str] | None = None,
    env: dict | None = None,
    scratch: pathlib.Path | None = None,
) -> tuple[list[str], int]:
    """The lines `rk probe-check` prints and its exit code: 0 every probe OK, 1 one is not, 2 nothing to check."""
    env = dict(os.environ if env is None else env)
    scripts = (
        [folder / f"{only}.sh"]
        if only
        else sorted(p for p in folder.glob("*.sh") if not p.name.startswith("."))
    )
    scripts = [s for s in scripts if s.is_file()]
    if not scripts:
        return [f"no saved probe in {folder}" + (f" named {only}" if only else "")], 2
    if not tree.is_dir():
        return [f"no tree: {tree}"], 2
    lines, notes = [], []
    work = pathlib.Path(tempfile.mkdtemp(prefix="probe_check_", dir=scratch))
    why = prcode.tree(work)  # the exported trees: the monitor's, which the runners read
    if why:
        notes.append(f"the scratch folder {work}: {why}")
    try:
        trees: dict[str, pathlib.Path | None] = {"tree": tree}
        if devbase_tree:
            trees["DEVBASE"] = devbase_tree
        else:
            rev = devbase or devbase_rev(tree, base)
            why = (
                export(tree, rev, work / "devbase", dirs or [])
                if rev
                else f"no DEVBASE: the tree is no merge and has no merge base with {base} (--devbase REV)"
            )
            trees["DEVBASE"] = None if why else work / "devbase"
            if why:
                notes.append(why)
            else:
                notes.append(f"DEVBASE {rev[:10]} (git archive into a scratch folder)")
        if simfix_tree:
            trees["simfix"] = simfix_tree
        elif simfix:
            rev = worktree_rev(tree)
            why = export(tree, rev, work / "simfix", dirs or []) or apply_patch(
                work / "simfix", simfix
            )
            trees["simfix"] = None if why else work / "simfix"
            if why:
                notes.append(why)
        else:
            trees["simfix"] = None
            notes.append(
                "no simulated fix: give --simfix PATCH (our fix line written another way) or --simfix-tree DIR"
            )
        got = {k: (run_on(folder, t, only, env) if t else {}) for k, t in trees.items()}
        bad = 0
        for s in scripts:
            name = s.stem
            parts, why = [], []
            for k, want in WANT.items():
                r = got[k].get(name)
                if r is None:
                    parts.append(f"{k} -")
                    why.append(f"{k} not run")
                    continue
                parts.append(f"{k} {r[0]}")
                if r[0] != want:
                    why.append(f"{k} must read {want}")
            fl = flags(s)
            if fl:
                why.append(f"{len(fl)} flag{'s' if len(fl) > 1 else ''}")
            verdict = "OK" if not why else "NOT OK (" + "; ".join(why) + ")"
            bad += bool(why)
            lines.append(f"{name}: {SEP.join(parts)}{SEP}{verdict}")
            for k in WANT:
                r = got[k].get(name)
                if r and r[0] != WANT[k]:
                    lines.append(f"   {k}: {r[1]}")
            lines += [f"   flag: {f}" for f in fl]
        lines += [f"({n})" for n in notes]
        return lines, 1 if bad else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def static(folder: pathlib.Path) -> tuple[list[str], int]:
    """The static flags of every probe in a folder, or in each probe folder of a probes root (its `retired*` folders
    left out: run_probes never runs them)."""
    lines, n, flagged = [], 0, 0
    for d in probe_folders(folder):
        for s in sorted(d.glob("*.sh")):
            n += 1
            fl = flags(s)
            if fl:
                flagged += 1
                lines.append(
                    f"{s.relative_to(folder) if d != folder else s.name}: {len(fl)} flag(s)"
                )
                lines += [f"   flag: {f}" for f in fl]
    lines.append(f"{n} probes, {flagged} flagged, {n - flagged} clean")
    return lines, 1 if flagged else 0


def probe_folders(folder: pathlib.Path) -> list[pathlib.Path]:
    """The folder itself when it holds probes, else its subfolders that do (a probes root), `retired*` left out."""
    if any(folder.glob("*.sh")):
        return [folder]
    return sorted(
        d
        for d in folder.iterdir()
        if d.is_dir() and not d.name.startswith("retired") and any(d.glob("*.sh"))
    )


# --- the port to the contract -----------------------------------------------------------------------------------------

LIVE_ROOT_NAMES = ("rocke_review",)  # the live toolkit's root folder, in a home folder
TMP_SH = "${TMPDIR:-/tmp}"
# Under the live root: (the path after it, its contract form in a shell script, in Python, or None: a hand edit and why)
UNDER_ROOT = (
    (
        r"venv/bin/python(?:3(?:\.\d+)?)?(?![\w.-])",
        "$PROBE_PYTHON",
        'os.environ.get("PROBE_PYTHON", "python3")',
        "",
    ),
    (
        r"venv/bin/(?=[\w.-])",
        "${PROBE_PYTHON%/*}/",
        None,
        "a tool of the checks venv: use $PROBE_PYTHON -m",
    ),
    (
        r"venv/bin(?![\w.-]|/[\w.-])",
        "${PROBE_PYTHON%/*}",
        None,
        "the checks venv's bin folder",
    ),
    (
        r"build/(?:\$\([^()]*(?:\([^()]*\)[^()]*)*\)|\$\{[^}]*\}|<[^>]*>|[^/\s\"'`;:)]+)",
        "$RKMON_BUILD_DIR",
        'os.environ["RKMON_BUILD_DIR"]',
        "",
    ),
    (r"repo(?![\w.-])", "$RKMON_REPO_DIR", 'os.environ["RKMON_REPO_DIR"]', ""),
    (r"(?:tmp|scratch)(?![\w.-])", TMP_SH, 'os.environ.get("TMPDIR", "/tmp")', ""),
    (
        r"venv-torch\b",
        None,
        None,
        "a torch venv: the contract has none (CI runs without torch)",
    ),
    (r"venv-gen\b", None, None, "a second venv (venv-gen): the contract has none"),
    (r"monitor\b", None, None, "the live toolkit's folder: the contract has none"),
    (r"wt\d+\w*", None, None, "a review's worktree: use the tree root ($1)"),
    (r"[\w.-]*", None, None, "a folder the contract has no variable for"),
)
ROCM = (
    (
        r"/opt/rocm(?:/core-[\w.-]+)?(?:/lib)?/llvm/bin",
        "$RKMON_ROCM_LLVM_BIN",
        'os.environ["RKMON_ROCM_LLVM_BIN"]',
        "",
    ),
    (
        r"/opt/rocm/core-[\w.-]+",
        "${RKMON_ROCM_LLVM_BIN%/lib/llvm/bin}",
        'os.environ["RKMON_ROCM_LLVM_BIN"].rsplit("/lib/llvm/bin", 1)[0]',
        "",
    ),
    (
        r"/opt/rocm(?![\w.-])",
        None,
        None,
        "names /opt/rocm itself (a host's ROCm, not the job's)",
    ),
)


def _roots(names) -> str:
    alt = "|".join(re.escape(n) for n in names)
    return (
        r"(?:/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/|\"\$HOME\"/|\$\{HOME\}/|\$HOME/|(?<![\w$])~/)"
        rf"(?:{alt})/"
    )


def _rules(names):
    root = _roots(names)
    return [
        (re.compile(root + tail), sh, py, why) for tail, sh, py, why in UNDER_ROOT
    ] + [(re.compile(rx), sh, py, why) for rx, sh, py, why in ROCM]


def _matches(line: str, rules):
    """The live paths of a line, left to right, without overlaps; at one place the first rule wins."""
    pos = 0
    while True:
        best = None
        for rule in rules:
            m = rule[0].search(line, pos)
            if m and (best is None or m.start() < best[0].start()):
                best = (m, rule)
        if best is None:
            return
        yield best
        pos = max(best[0].end(), best[0].start() + 1)


def _quoted_heredocs(text: str) -> set[int]:
    """The line numbers inside a quoted heredoc (<<'X', <<"X", <<\\X): no variable expands there."""
    out, end = set(), None
    for i, line in enumerate(text.splitlines(), 1):
        if end is not None:
            if line.strip() == end:
                end = None
            else:
                out.add(i)
            continue
        m = re.search(r"<<-?\s*(?:'(\w+)'|\"(\w+)\"|\\(\w+))", line)
        if m:
            end = m.group(1) or m.group(2) or m.group(3)
    return out


def _single_quoted(line: str, pos: int) -> bool:
    return line[:pos].count("'") % 2 == 1


def port_sh(text: str, names=LIVE_ROOT_NAMES) -> tuple[str, list[str], int]:
    """A shell script ported: (new text, the hand edits left, the paths rewritten)."""
    rules = _rules(names)
    frozen = _quoted_heredocs(text)
    out_lines, left, n = [], [], 0
    for i, line in enumerate(text.splitlines(keepends=True), 1):
        pos = 0
        new = ""
        for m, (_rx, sh, _py, why) in _matches(line, rules):
            new += line[pos : m.start()]
            comment = line.lstrip().startswith("#")
            if sh is None or (
                not comment and (i in frozen or _single_quoted(line, m.start()))
            ):
                reason = why or (
                    "inside a quoted heredoc" if i in frozen else "inside single quotes"
                )
                left.append(f"line {i}: {_shown(m.group(0))}: {reason}")
                new += m.group(0)
            else:
                new += sh
                n += 1
            pos = m.end()
        new += line[pos:]
        if i == 1 and new.startswith("#!") and "python" in new:
            new = "#!/usr/bin/env python3\n"
        out_lines.append(new)
    return "".join(out_lines), left, n


def _docstring(toks, k) -> bool:
    """Is token k a docstring (a string that is a whole statement of its own)?"""
    prev = next(
        (
            x
            for x in reversed(toks[:k])
            if x.type not in (tokenize.NL, tokenize.COMMENT)
        ),
        None,
    )
    nxt = toks[k + 1] if k + 1 < len(toks) else None
    return (
        prev is None
        or prev.type
        in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING)
    ) and (nxt is None or nxt.type in (tokenize.NEWLINE, tokenize.ENDMARKER))


STRING = re.compile(r"([rRuU]?)('|\")")


def port_py(text: str, names=LIVE_ROOT_NAMES) -> tuple[str, list[str], int]:
    """A Python helper ported through its tokens: a string literal that starts with a live path becomes the contract's
    expression plus the rest (`os.path.expanduser(...)` around it included); comments and docstrings name the
    contract's variable; anything else (a path inside a longer string, an f-string, bytes) is a hand edit.
    """
    rules = _rules(names)
    left, n = [], 0
    lines = text.splitlines(keepends=True)
    if lines and lines[0].startswith("#!") and HOST_PATH.search(lines[0]):
        lines[0] = "#!/usr/bin/env python3\n"
        n += 1
    text = "".join(lines)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError) as exc:
        return text, [f"line 1: the file: not read as Python ({exc})"], 0
    offsets = [0]
    for ln in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(ln))

    def off(rc):
        return offsets[rc[0] - 1] + rc[1]

    edits, use_os = [], False  # (start offset, end offset, replacement)
    expanduser = re.compile(r"os\.path\.expanduser\(\s*$")
    for k, t in enumerate(toks):
        if t.type == tokenize.COMMENT or (
            t.type == tokenize.STRING and _docstring(toks, k)
        ):
            s, pos, parts = t.string, 0, []
            for m, (_rx, sh, _py, _why) in _matches(s, rules):
                parts += [s[pos : m.start()], sh or m.group(0)]
                pos = m.end()
            s = "".join(parts) + s[pos:]
            if s != t.string:
                edits.append((off(t.start), off(t.end), s))
                n += 1
            continue
        m_prefix = STRING.match(t.string) if t.type == tokenize.STRING else None
        if (
            not m_prefix
            or t.string[m_prefix.end() - 1 : m_prefix.end() + 2]
            == m_prefix.group(2) * 3
        ):
            continue  # not a plain one-line string literal: the leftover scan below reports a path in it
        q = m_prefix.group(2)
        body = t.string[m_prefix.end() : -1]
        hit = next(_matches(body, rules), None)
        if not hit:
            continue
        m, (_rx, _sh, py, why) = hit
        if (
            py is None
            or m.start() != 0
            or m_prefix.group(1).lower() == "r"
            and "\\" in body
        ):
            continue  # the leftover scan reports it, with its reason
        rest = body[m.end() :]
        expr = py + (f" + {m_prefix.group(1)}{q}{rest}{q}" if rest else "")
        start, end = off(t.start), off(t.end)
        mm = expanduser.search(text[:start])
        if mm and k + 1 < len(toks) and toks[k + 1].string == ")":
            start, end = mm.start(), off(toks[k + 1].end)
        edits.append((start, end, expr))
        use_os = True
        n += 1
    for start, end, repl in sorted(edits, reverse=True):
        text = text[:start] + repl + text[end:]
    # What is left: every live path the rewrite did not reach, with its reason; and a home path built in Python.
    home = re.compile(
        r"(?:Path\.home\(\)|os\.environ\[[\"']HOME[\"']\]|os\.path\.expanduser\([\"']~[\"']\))\s*[/+,]\s*"
        rf"[\"']/?(?:{'|'.join(re.escape(x) for x in names)})\b"
    )
    for i, line in enumerate(text.splitlines(), 1):
        for m in home.finditer(line):
            left.append(f"line {i}: {m.group(0)}: a home path built in Python")
        for m, (_rx, _sh, _py, why) in _matches(line, rules):
            left.append(
                f"line {i}: {_shown(m.group(0))}: "
                + (why or "inside a longer string, an f-string or an expression")
            )
    if use_os and not re.search(r"(?m)^import os\b|^import .*\bos\b", text):
        lines = text.splitlines(keepends=True)
        at = 0
        while at < len(lines) and (lines[at].startswith("#") or not lines[at].strip()):
            at += 1
        futures = [
            i for i, ln in enumerate(lines) if ln.startswith("from __future__ import")
        ]
        if futures:
            at = futures[-1] + 1
        lines.insert(at, "import os  # rk probe-port: the probe environment contract\n")
        text = "".join(lines)
    return text, sorted(set(left), key=lambda x: int(x.split()[1].rstrip(":"))), n


def port_file(path: pathlib.Path, names=LIVE_ROOT_NAMES) -> tuple[str, list[str], int]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py" or (
        path.suffix == ""
        and text.startswith("#!")
        and "python" in text.split("\n", 1)[0]
    ):
        return port_py(text, names)
    return port_sh(text, names)


PORTED_SUFFIXES = (".sh", ".py")


def port(
    folder: pathlib.Path, write: bool = False, names=LIVE_ROOT_NAMES
) -> tuple[list[str], int]:
    """`rk probe-port`'s lines and exit code (0: nothing needs a hand edit; 1: something does; 2: nothing to read).
    Per probe folder, each file: converts (N paths), needs a hand edit (each path and why), or nothing to port; a
    probe (its script and its helpers) converts cleanly when all its files do. --write rewrites the files that convert
    cleanly, in place, keeping their mode; a file with a hand edit left is never written.
    """
    names = tuple(names) or LIVE_ROOT_NAMES
    folders = probe_folders(folder) if folder.is_dir() else []
    if not folders:
        return [f"no saved probe in {folder}"], 2
    lines = []
    counts = {"clean": 0, "hand": 0, "none": 0}
    pcounts = {"clean": 0, "hand": 0, "none": 0}
    reasons: dict[str, int] = {}
    for d in folders:
        state = {}
        for f in sorted(
            p for p in d.iterdir() if p.is_file() and p.suffix in PORTED_SUFFIXES
        ):
            try:
                new, left, n = port_file(f, names)
            except (OSError, UnicodeDecodeError) as exc:
                new, left, n = None, [f"not read ({exc})"], 0
            rel = f.relative_to(folder) if d != folder else pathlib.Path(f.name)
            if left:
                kind = "hand"
                lines.append(
                    f"{rel}: needs a hand edit ({len(left)}; {n} path(s) would convert)"
                )
                lines += [f"   {x}" for x in left]
                for x in left:
                    r = x.split(": ", 2)[-1]
                    reasons[r] = reasons.get(r, 0) + 1
            elif n:
                kind = "clean"
                lines.append(f"{rel}: converts ({n} path(s))")
                if write and new is not None:
                    mode = f.stat().st_mode
                    f.write_text(new, encoding="utf-8")
                    os.chmod(f, mode)
            else:
                kind = "none"
            counts[kind] += 1
            state[f] = kind
        for s in sorted(d.glob("*.sh")):
            kinds = {state.get(x, "none") for x in [s] + helpers(s)}
            pk = "hand" if "hand" in kinds else "clean" if "clean" in kinds else "none"
            pcounts[pk] += 1
    lines.append(
        f"probes: {sum(pcounts.values())}: {pcounts['clean']} convert cleanly, {pcounts['hand']} need a hand edit, "
        f"{pcounts['none']} need nothing"
    )
    lines.append(
        f"files: {sum(counts.values())}: {counts['clean']} convert cleanly, {counts['hand']} need a hand edit, "
        f"{counts['none']} need nothing"
        + (" (written)" if write else " (dry run: --write rewrites the clean ones)")
    )
    for r, k in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"   hand edits: {k} × {r}")
    return lines, 1 if counts["hand"] else 0
