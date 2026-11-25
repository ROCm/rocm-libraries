# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Docs against code, in every review's checks: what the verifier and the lane grepped changed docs for by hand in all
39 reviews of agent_script_overlap.md (fix 4). Leads, not verdicts: a doc may name an old name on purpose (a migration
note), and a new knob may be internal.

**Gone names**: the names the PR removes or renames, read from the removed lines of its code files under the rocKE tree
(tests and docs aside) against merge-base(develop, head), that its head's code does not have anywhere:
- a public Python function or class (`def name`, `class Name`: no `def`/`class` of that name is left);
- an environment variable or knob (`ROCKE_*`: the head's code has the token nowhere);
- a command-line flag (`add_argument("--flag"`: the head's code has the flag nowhere);
- an architecture (`gfxNNN`: the head's code has it nowhere).
Each doc line at the head that still names one (the rocKE tree's .md and .rst files: its docs, dsl_docs/reference,
SUPPORT_MATRIX*, README*, TESTING.md; changelogs and history notes aside) is a lead, with where the name was. Paths of
moved or deleted files are `rk moved-refs`' (the section above).

**New and not in the reference docs** (platform/dsl_docs/reference): on the PR's added lines,
- an environment variable the code reads (`getenv("ROCKE_X")`, `environ.get(...)`, `environ[...]`) that env_flags.md
  does not name;
- an IR op name of a dialect op_vocabulary.md lists (`"arith.x"`), on the IR and lowering files (core/ir*, core/lower*,
  cpp/core), that op_vocabulary.md does not name;
- an IR dtype (`Type("x")` in core/) that dtype_support.md does not name.

Git only, read-only. Usage: python3 -I doc_refs.py --repo R --head SHA [--base REV] [--rocke-path K] [--budget 120]
Prints the section, then one last line: "DOCREFS pass|fail|skip <detail>".
"""
import argparse
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True  # the toolkit's folder is read-only at run time
sys.path.insert(0, str(ROOT))
from rkmon_lib import gitops, prtree  # noqa: E402

BUDGET_S = 120
SHOWN = 30
REFERENCE = "platform/dsl_docs/reference"
DOC = re.compile(r"\.(md|rst)$")
HISTORY = re.compile(
    r"(?i)(^|/)(changelog|changes|history|news|release[-_]?notes)(\.[^/]*)?$"
)
TEST = re.compile(
    r"(^|/)(tests?|testing)/|(^|/)test_[^/]*$|_test\.[a-z]+$|(^|/)conftest\.py$"
)
CODE = re.compile(r"\.(py|c|cc|cpp|cxx|h|hpp|cmake|txt|yaml|yml|sh|json)$")
DEF = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z][A-Za-z0-9_]*)")
ENV_TOKEN = re.compile(r"\bROCKE_[A-Z0-9_]*[A-Z0-9]\b")
ENV_READ = re.compile(
    r"(?:getenv|environ\.get|environ\.setdefault|environ\[|env\.get)\s*\(?\s*[\"'](ROCKE_[A-Z0-9_]+)[\"']"
)
FLAG = re.compile(r"add_argument\(\s*[\"'](--[a-z][a-z0-9-]+)[\"']")
ARCH = re.compile(r"\bgfx[0-9]{3,4}[a-z]?\b")
OP = re.compile(r"[\"']([a-z][a-z0-9]*\.[a-z][a-z0-9_]*)[\"']")
DTYPE = re.compile(r"\bType\(\s*[\"']([a-z][a-z0-9]*)[\"']")
IR_FILES = re.compile(
    r"^platform/(python/rocke/core/(ir|lower)[^/]*(/.*)?|cpp/core/.*)$"
)


def diff_lines(repo: str, mb: str, head: str, K: str) -> dict[str, dict[str, list]]:
    """path (relative to the rocKE tree) -> {"-": [removed lines], "+": [(line number at head, added line)]}."""
    text = gitops.git(
        repo,
        "-c",
        "core.quotepath=false",
        "diff",
        "-U0",
        "--no-color",
        "--no-renames",
        mb,
        head,
        "--",
        K + "/",
        lazy=False,
        timeout=300,
    )
    out: dict[str, dict[str, list]] = {}
    cur, line = None, 0
    for ln in text.split("\n"):
        if ln.startswith("+++ "):
            p = ln[4:]
            cur = p[2 + len(K) + 1 :] if p.startswith("b/" + K + "/") else None
            continue
        if ln.startswith("--- "):
            p = ln[4:]
            cur = p[2 + len(K) + 1 :] if p.startswith("a/" + K + "/") else None
            continue
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", ln)
        if m:
            line = int(m.group(1))
            continue
        if cur is None:
            continue
        rec = out.setdefault(cur, {"-": [], "+": []})
        if ln.startswith("-"):
            rec["-"].append(ln[1:])
        elif ln.startswith("+"):
            rec["+"].append((line, ln[1:]))
            line += 1
    return out


def grep(
    repo: str,
    head: str,
    K: str,
    pattern: str,
    docs: bool,
    fixed_words: list[str] | None = None,
) -> str:
    """`git grep -n` at the head in the rocKE tree: in its docs, or in its code (docs aside)."""
    spec = (
        [f"{K}/*.md", f"{K}/*.rst"]
        if docs
        else [K + "/", f":(exclude){K}/*.md", f":(exclude){K}/*.rst"]
    )
    args = ["grep", "-n", "-I", "--no-color"]
    if fixed_words is not None:
        args += ["-w", "-F"] + [x for w in fixed_words for x in ("-e", w)]
    else:
        args += ["-E", "-e", pattern]
    p = subprocess.run(
        ["git", "-C", repo, *args, head, "--", *spec],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
    )
    return p.stdout if p.returncode in (0, 1) else ""


def gone_names(repo: str, head: str, K: str, diff: dict) -> dict[str, tuple[str, str]]:
    """name -> (kind, where it was) for the names the PR removes or renames and its head's code lacks."""
    cand: dict[str, tuple[str, str]] = {}
    for path, rec in diff.items():
        if not CODE.search(path) or TEST.search(path) or DOC.search(path):
            continue
        added = "\n".join(t for _n, t in rec["+"])
        for ln in rec["-"]:
            m = DEF.match(ln)
            if (
                m
                and not m.group(1).startswith("_")
                and not re.search(rf"\b(def|class)\s+{m.group(1)}\b", added)
            ):
                cand.setdefault(m.group(1), ("def", path))
            for rx, kind in ((ENV_TOKEN, "env"), (FLAG, "flag"), (ARCH, "arch")):
                for x in rx.findall(ln):
                    if not re.search(rf"(?<![\w-]){re.escape(x)}(?![\w-])", added):
                        cand.setdefault(x, (kind, path))
    if not cand:
        return {}
    code = grep(repo, head, K, "", docs=False, fixed_words=sorted(cand))
    present = set()
    for ln in code.splitlines():
        body = ln.split(":", 2)[2] if ln.count(":") >= 2 else ln
        for name, (kind, _p) in cand.items():
            if kind == "def":
                if re.search(rf"\b(def|class)\s+{re.escape(name)}\b", body):
                    present.add(name)
            elif re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", body):
                present.add(name)
    return {n: v for n, v in cand.items() if n not in present}


def stale_doc_lines(repo: str, head: str, K: str, gone: dict) -> list[dict]:
    if not gone:
        return []
    out = []
    for ln in grep(repo, head, K, "", docs=True, fixed_words=sorted(gone)).splitlines():
        parts = ln.split(":", 3)
        if len(parts) < 4:
            continue
        path, num, text = (
            parts[1][len(K) + 1 :] if parts[1].startswith(K + "/") else parts[1],
            parts[2],
            parts[3],
        )
        if HISTORY.search(path):
            continue
        for name in sorted(gone):
            kind = gone[name][0]
            hit = re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text)
            if hit and (
                kind != "def"
                or re.search(
                    rf"`[^`]*\b{re.escape(name)}\b[^`]*`|\b{re.escape(name)}\(", text
                )
            ):
                out.append(
                    {
                        "doc": path,
                        "line": int(num),
                        "name": name,
                        "kind": kind,
                        "was": gone[name][1],
                    }
                )
    return out


def reference_text(repo: str, head: str, K: str, name: str) -> str | None:
    try:
        return gitops.git(repo, "show", f"{head}:{K}/{REFERENCE}/{name}", lazy=False)
    except gitops.GitError:
        return None


def new_unlisted(repo: str, head: str, K: str, diff: dict, mb: str) -> list[dict]:
    """The env vars, ops and dtypes the PR's added lines bring that the reference docs do not name and that the merge
    base's code does not have (an old name on a moved or reformatted line is not new).
    """
    out = _unlisted(repo, head, K, diff)
    if not out:
        return out
    old = grep(
        repo, mb, K, "", docs=False, fixed_words=sorted({x["name"] for x in out})
    )
    return [
        x
        for x in out
        if not re.search(rf"[\"'`\s(]{re.escape(x['name'])}[\"'`\s),]", old)
    ]


def _unlisted(repo: str, head: str, K: str, diff: dict) -> list[dict]:
    refs = {
        n: reference_text(repo, head, K, n)
        for n in ("env_flags.md", "op_vocabulary.md", "dtype_support.md")
    }
    dialects = set(
        re.findall(
            r"`([a-z][a-z0-9]*)\.[a-z][a-z0-9_]*`", refs["op_vocabulary.md"] or ""
        )
    )
    seen, out = set(), []
    for path, rec in sorted(diff.items()):
        if TEST.search(path) or DOC.search(path):
            continue
        for num, text in rec["+"]:
            found = [("env", x, "env_flags.md") for x in ENV_READ.findall(text)]
            if IR_FILES.match(path):
                found += [
                    ("op", x, "op_vocabulary.md")
                    for x in OP.findall(text)
                    if x.split(".")[0] in dialects
                ]
            if path.startswith("platform/python/rocke/core/"):
                found += [("dtype", x, "dtype_support.md") for x in DTYPE.findall(text)]
            for kind, name, ref in found:
                doc = refs[ref]
                if (
                    (kind, name) in seen
                    or doc is None
                    or re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", doc)
                ):
                    continue
                seen.add((kind, name))
                out.append(
                    {
                        "kind": kind,
                        "name": name,
                        "at": f"{path}:{num}",
                        "ref": f"{REFERENCE}/{ref}",
                    }
                )
    return out


WORD = {
    "def": "function or class",
    "env": "environment variable",
    "flag": "command-line flag",
    "arch": "architecture",
    "op": "IR op",
    "dtype": "IR dtype",
}


def check(
    repo: str,
    head: str,
    K: str,
    base: str = prtree.DEVELOP,
    budget: float = BUDGET_S,
) -> dict:
    t0 = time.monotonic()
    res: dict = {"head": head, "error": None, "gone": {}, "stale": [], "new": []}
    try:
        res["mb"] = mb = prtree.merge_base(repo, base, head)
        diff = diff_lines(repo, mb, head, K)
        res["files"] = len(diff)
        res["gone"] = gone_names(repo, head, K, diff)
        if time.monotonic() - t0 > budget:
            raise prtree.NotRun(
                f"the time budget ({budget:.0f} s) ran out after the gone names"
            )
        res["stale"] = stale_doc_lines(repo, head, K, res["gone"])
        res["new"] = new_unlisted(repo, head, K, diff, mb)
    except (
        prtree.NotRun,
        gitops.GitError,
        RuntimeError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        res["error"] = str(exc)[:300]
    res["secs"] = round(time.monotonic() - t0, 1)
    return res


def verdict(res: dict) -> tuple[str, str]:
    if res.get("error"):
        return "skip", f"NOT RUN: {res['error'][:180]}"
    leads = len(res["stale"]) + len(res["new"])
    if not leads:
        return "pass", (
            f"no doc names a name the PR removes ({len(res['gone'])} gone), and the reference docs list "
            "every new env var, op and dtype"
        )
    parts = []
    if res["stale"]:
        names = sorted({x["name"] for x in res["stale"]})
        parts.append(
            f"{len(res['stale'])} doc lines name what the PR removes ({', '.join(names[:3])}"
            f"{' +' + str(len(names) - 3) if len(names) > 3 else ''})"
        )
    if res["new"]:
        parts.append(
            f"{len(res['new'])} new {', '.join(sorted({WORD[x['kind']] for x in res['new']}))} not in the "
            f"reference docs ({', '.join(x['name'] for x in res['new'][:3])})"
        )
    return "fail", f"{leads} lead{'s' if leads != 1 else ''}: " + "; ".join(parts)


def render(res: dict) -> list[str]:
    if res.get("error"):
        return [f"NOT RUN: {res['error']}", " ".join(("DOCREFS",) + verdict(res))]
    gone = res["gone"]
    out = [
        f"names the PR removes or renames and its head's code lacks: {len(gone)}"
        + (
            ": "
            + ", ".join(f"{n} ({WORD[k]})" for n, (k, _p) in sorted(gone.items())[:12])
            if gone
            else ""
        )
        + (f" +{len(gone) - 12} more" if len(gone) > 12 else "")
        + f" ({res.get('secs', 0)} s)"
    ]
    for x in res["stale"][:SHOWN]:
        out.append(
            f"LEAD {x['doc']}:{x['line']} names {x['name']}, a {WORD[x['kind']]} the PR removes or renames "
            f"(it was in {x['was']})"
        )
    if len(res["stale"]) > SHOWN:
        out.append(f"LEAD +{len(res['stale']) - SHOWN} more doc lines")
    for x in res["new"][:SHOWN]:
        out.append(
            f"LEAD new {WORD[x['kind']]} {x['name']} ({x['at']}) is not in {x['ref']}"
        )
    out.append(" ".join(("DOCREFS",) + verdict(res)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--base", default=prtree.DEVELOP)
    ap.add_argument("--rocke-path", default="", type=prtree.folder_argument)
    ap.add_argument("--budget", type=float, default=BUDGET_S)
    a = ap.parse_args(argv)
    print("\n".join(render(check(a.repo, a.head, a.rocke_path, a.base, a.budget))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
