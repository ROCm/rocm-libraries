#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Clang diagnostics the GCC build misses, over the C/C++ translation units a PR touches (the checks' "Superbuild tidy"
section, and `rk werror N`).

CI's superbuilds co-compile clang-tidy with the provider's `.clang-tidy` (WarningsAsErrors) and stop at the first error;
the checks' native build is GCC's, so a diagnostic only clang raises reaches the author a CI round late. Here:
- **the TUs**: every changed `.cpp`/`.cc`/`.cxx` the native build compiles (its `compile_commands.json`), and, for a
  changed header, the TUs of the build whose source names it in an `#include` (at most MAX_TUS in all). Each runs with
  its own command's flags (include paths, defines, standard; no output, no GCC warning flags); a flag clang does not
  know is dropped and said;
- **clang-tidy** (`<llvm bin>/clang-tidy`, else `clang-tidy-20` or `clang-tidy` on PATH) with the nearest
  `.clang-tidy` above the TU, the provider's, as CI reads it. A sparse worktree may hold it as a link to a folder it
  does not check out: its text is then read from the tree's commit (`git show`) into a scratch file;
- **clang -Werror**: `clang++ -fsyntax-only -ferror-limit=0 -Wall -Wextra -Werror` over the same TU (every error, past clang's
  default limit of 20);
- **feature macros off and on**: once as configured and, when the TU or a changed header tests macros the command
  does not define (`#if`/`#ifdef`/`defined(...)` of a `<PREFIX>_*` name), once more with each defined (`-D<M>=1`);
- **develop** (`--dev-tree`, the develop checkout the checks keep): each run again on develop's copy of the TU; an error
  develop has too (same file and message) is listed as such and not counted.
Every error is listed once (file:line:col, the message, which runs saw it). The lines are leads: the local clang may be
newer than CI's.

Usage: python3 -I tidy_check.py --tree ROOT --build DIR [--base REV] [--files F ...] [--rocke-path K]
                                [--dev-tree ROOT] [--llvm-bin DIR] [--macro-prefix ROCKE_] [--budget S]
       python3 -I tidy_check.py --repo R --tree ROOT [--dev-tree ROOT] ... --fetch
--base: the files the tree's HEAD adds or changes against BASE (the checks give the merge's develop), read from the
trees alone: no rename detection, which reads blobs (a moved file is one the head adds); --files names them instead
(paths from the tree's root). Prints the section, then one last line "TIDY pass|fail|skip <detail>".
--fetch: fetches into the clone R, whose worktrees the trees are, the blobs the check reads there (the file each
`.clang-tidy` link in the trees names), and runs nothing else (rkmon_lib/prtree.py, fetch_main): the check runs as PR
code, in a PR runner's slot, where git fetches nothing.
"""
import argparse
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rkmon_lib import prtree  # noqa: E402

CPP = (".cpp", ".cc", ".cxx")
CONFIG = ".clang-tidy"
HEADERS = (".h", ".hpp", ".hh", ".hxx", ".inl")
MAX_TUS = 12
MAX_MACROS = 8
SHOWN = 40
PER_RUN_S = 300
ERROR = re.compile(
    r"^(?P<file>[^\s:][^:]*):(?P<line>\d+):(?P<col>\d+): error: (?P<msg>.*)$"
)
UNKNOWN_ARG = re.compile(r"(?:unknown argument|unsupported option):? '([^']+)'")
NO_RUN = re.compile(
    r"Error: no checks enabled|USAGE: clang-tidy|Error while processing|Could not auto-detect"
)
MACRO_USE = r"(?:#\s*(?:if|ifdef|ifndef|elif)\b[^\n]*|defined\s*\(?\s*)"


def tool(names: list[str], llvm_bin: str) -> str:
    for n in names:
        if llvm_bin and os.access(os.path.join(llvm_bin, n), os.X_OK):
            return os.path.join(llvm_bin, n)
    for n in names:
        w = shutil.which(n)
        if w:
            return w
    return ""


def git(tree: pathlib.Path, *args: str) -> tuple[int, str]:
    p = subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True)
    return p.returncode, p.stdout


def changed(tree: pathlib.Path, base: str, files: list[str]) -> list[str] | None:
    """The files to check: `files`, else the ones the tree's HEAD adds or changes against `base`, read from the trees
    alone (rename detection reads blobs, which git fetches nowhere in a slot: a moved file is one the head adds); None
    when git cannot list them."""
    if files:
        return files
    if not base:
        return []
    rc, out = git(
        tree, "diff", "--name-only", "--no-renames", "--diff-filter=d", base, "HEAD"
    )
    return out.split() if rc == 0 else None


def link_target(tree: pathlib.Path, link: pathlib.Path) -> str:
    """The path from the tree's root of the file a link in the tree names."""
    return os.path.normpath(
        os.path.join(os.path.relpath(link.parent, tree), os.readlink(link))
    )


def needs(trees: list[pathlib.Path]) -> list[str]:
    """The blobs of the clone the check reads (`--fetch`): the file each `.clang-tidy` link in the trees names, which a
    sparse tree may not check out, and which the check then reads from the tree's commit (`Configs`). A link out of
    its tree names no file of the commit.
    """
    out = []
    for tree in trees:
        for folder, dirs, names in os.walk(tree):
            dirs[:] = [d for d in dirs if d != ".git"]
            link = pathlib.Path(folder) / CONFIG
            if CONFIG in names and link.is_symlink():
                target = link_target(tree, link)
                if not (target.startswith("..") or os.path.isabs(target)):
                    out += prtree.blobs(str(tree), "HEAD", [target])
    return out


def entries(build: pathlib.Path) -> list[dict]:
    try:
        return json.loads((build / "compile_commands.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def entry_file(e: dict) -> str:
    return os.path.realpath(os.path.join(e.get("directory", ""), e.get("file", "")))


def entry_args(e: dict) -> list[str]:
    return list(e.get("arguments") or shlex.split(e.get("command", "")))


def flags(args: list[str], src: str) -> list[str]:
    """A TU's own flags: no compiler, no output or dependency file, no -c, no source, no -W* (GCC's warnings)."""
    out, skip = [], False
    for a in args[1:]:
        if skip:
            skip = False
            continue
        if a in ("-o", "-MF", "-MT", "-MQ"):
            skip = True
            continue
        if (
            a in ("-c", "-MD", "-MMD")
            or a.startswith(("-o", "-W"))
            or os.path.realpath(a) == src
        ):
            continue
        out.append(a)
    return out


def pick_tus(
    tree: pathlib.Path, files: list[str], db: list[dict]
) -> tuple[list[dict], list[str]]:
    """The TUs to check and notes (a changed source the build compiles nowhere, TUs past MAX_TUS)."""
    by_file = {entry_file(e): e for e in db}
    tus, notes, seen = [], [], set()
    for f in files:
        if f.endswith(CPP):
            e = by_file.get(os.path.realpath(tree / f))
            if e is None:
                notes.append(f"{f}: no TU of the native build (not compiled here)")
            elif id(e) not in seen:
                tus.append(e)
                seen.add(id(e))
    heads = [f for f in files if f.endswith(HEADERS)]
    if heads:
        names = [re.escape(os.path.basename(h)) for h in heads]
        inc = re.compile(
            r'#\s*include\s*[<"](?:[^>"]*/)?(?:' + "|".join(names) + r')[>"]'
        )
        root = str(tree.resolve())
        for path, e in sorted(by_file.items()):
            if id(e) in seen or not path.startswith(root):
                continue
            try:
                if inc.search(
                    pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
                ):
                    tus.append(e)
                    seen.add(id(e))
            except OSError:
                continue
    if len(tus) > MAX_TUS:
        notes.append(f"{len(tus) - MAX_TUS} more TUs not checked (at most {MAX_TUS})")
        tus = tus[:MAX_TUS]
    return tus, notes


def feature_macros(texts: list[str], args: list[str], prefix: str) -> list[str]:
    rx = re.compile(MACRO_USE + rf"({re.escape(prefix)}[A-Z0-9_]+)")
    used = set()
    for t in texts:
        for line in t.splitlines():
            used.update(m.group(1) for m in rx.finditer(line))
            if re.match(r"\s*#\s*(?:if|elif)\b", line):
                used.update(re.findall(rf"\b({re.escape(prefix)}[A-Z0-9_]+)\b", line))
    defined = {a[2:].split("=")[0] for a in args if a.startswith("-D")}
    return sorted(used - defined)[:MAX_MACROS]


class Configs:
    """The .clang-tidy each TU reads: the nearest above it; a link to a folder the sparse tree lacks is read from the
    tree's commit into a scratch file."""

    def __init__(self, scratch: pathlib.Path):
        self.scratch, self.memo, self.notes = scratch, {}, []

    def for_file(self, tree: pathlib.Path, src: str) -> str:
        d = pathlib.Path(src).parent
        root = tree.resolve()
        while True:
            p = d / CONFIG
            if p in self.memo:
                return self.memo[p]
            if os.path.lexists(p):
                self.memo[p] = self._read(tree, p)
                return self.memo[p]
            if d == root or root not in d.parents:
                return ""
            d = d.parent

    def _read(self, tree: pathlib.Path, p: pathlib.Path) -> str:
        if p.is_file():
            return str(p)
        rel = link_target(tree, p)
        rc, text = git(tree, "show", f"HEAD:{rel}")
        if rc != 0:
            self.notes.append(
                f"{os.path.relpath(p, tree)} links to {rel}, which the tree and its commit lack"
            )
            return ""
        out = self.scratch / f"clang-tidy-{len(self.memo)}.yaml"
        out.write_text(text, encoding="utf-8")
        self.notes.append(
            f"{os.path.relpath(p, tree)}: read from the tree's commit ({rel})"
        )
        return str(out)


def run(argv: list[str], cwd: str, timeout: int) -> tuple[int, str]:
    try:
        p = subprocess.run(
            argv,
            cwd=cwd or None,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, ""
    except OSError as exc:
        return 127, str(exc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--base", default="")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--rocke-path", default="")
    ap.add_argument("--dev-tree", default="")
    ap.add_argument("--llvm-bin", default=os.environ.get("RKMON_LLVM_BIN", ""))
    ap.add_argument("--macro-prefix", default="ROCKE_")
    ap.add_argument("--budget", type=int, default=900)
    ap.add_argument(
        "--repo", default="", help="the clone the trees are worktrees of (--fetch)"
    )
    ap.add_argument("--fetch", action="store_true", help=prtree.FETCH_HELP)
    a = ap.parse_args(argv)
    if a.fetch and not a.repo:
        ap.error("--fetch needs --repo: the clone the blobs are fetched into")
    t0 = time.monotonic()
    tree, build = pathlib.Path(a.tree).resolve(), pathlib.Path(a.build)
    dev = (
        pathlib.Path(a.dev_tree).resolve()
        if a.dev_tree and os.path.isdir(a.dev_tree)
        else None
    )
    if a.fetch:
        return prtree.fetch_main(a.repo, lambda: needs([tree, *([dev] if dev else [])]))
    k = a.rocke_path.strip("/")
    files = changed(tree, a.base, a.files)
    if files is None:
        print(f"not run: git cannot list the files the tree changes against {a.base}")
        print("TIDY skip no list of the changed files")
        return 0
    files = [
        f
        for f in files
        if f.endswith(CPP + HEADERS) and (not k or f.startswith(k + "/"))
    ]
    if not files:
        print("no C/C++ file changed: nothing to check")
        print("TIDY pass no C/C++ file changed")
        return 0
    db = entries(build)
    if not db:
        print(
            f"not run: no compile_commands.json in {build} (the checks' native build writes it)"
        )
        print("TIDY skip no compile_commands.json")
        return 0
    tidy = tool(["clang-tidy", "clang-tidy-20"], a.llvm_bin)
    clangxx = tool(["clang++"], a.llvm_bin)
    if not tidy and not clangxx:
        print(
            "not run: no clang-tidy and no clang++ (ROCm's LLVM tools, paths.rocm_llvm_bin, or clang-tidy-20)"
        )
        print("TIDY skip no clang-tidy or clang++")
        return 0
    tus, notes = pick_tus(tree, files, db)
    head_texts = []
    for f in files:
        if f.endswith(HEADERS):
            try:
                head_texts.append(
                    (tree / f).read_text(encoding="utf-8", errors="replace")
                )
            except OSError:
                pass
    errors: dict[str, set[str]] = (
        {}
    )  # "file:line:col: error: msg" -> the runs that saw it
    dev_errors: set[tuple[str, str]] = set()  # (file, msg) develop's runs saw
    runs, macros_seen, dropped, cut, no_tidy = 0, set(), set(), False, set()
    with tempfile.TemporaryDirectory(prefix="tidy_") as scratch:
        configs = Configs(pathlib.Path(scratch))
        for e in tus:
            src = entry_file(e)
            rel = os.path.relpath(src, tree)
            base_flags = flags(entry_args(e), src)
            try:
                text = pathlib.Path(src).read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            macros = feature_macros([text, *head_texts], entry_args(e), a.macro_prefix)
            macros_seen.update(macros)
            variants = [("off", [])] + (
                [("on", [f"-D{m}=1" for m in macros])] if macros else []
            )
            sides = [("pr", tree)] + (
                [("dev", dev)] if dev and (dev / rel).is_file() else []
            )
            for label, defs in variants:
                for side, root in sides:
                    left = a.budget - (time.monotonic() - t0)
                    if left < 10:
                        cut = True
                        break
                    s = str(root / rel)
                    fl = [x.replace(str(tree), str(root)) for x in base_flags] + defs
                    jobs = []
                    if clangxx:
                        jobs.append(
                            (
                                "clang -Werror",
                                [
                                    clangxx,
                                    "-fsyntax-only",
                                    "-ferror-limit=0",
                                    "-Wall",
                                    "-Wextra",
                                    "-Werror",
                                ],
                                s,
                            )
                        )
                    cfg = configs.for_file(root, s) if tidy else ""
                    if tidy and cfg:
                        jobs.append(
                            (
                                "tidy",
                                [tidy, "--quiet", f"--config-file={cfg}", s, "--"],
                                "",
                            )
                        )
                    elif tidy:
                        no_tidy.add(rel)
                    for what, cmd, last in jobs:
                        full = (
                            cmd
                            + [x for x in fl if x not in dropped]
                            + ([last] if last else [])
                        )
                        rc, out = run(
                            full, e.get("directory", ""), int(min(PER_RUN_S, left))
                        )
                        bad = set(UNKNOWN_ARG.findall(out)) - dropped
                        if bad:
                            dropped |= bad
                            full = [x for x in full if x not in dropped]
                            rc, out = run(
                                full, e.get("directory", ""), int(min(PER_RUN_S, left))
                            )
                        runs += 1
                        if rc == 124:
                            notes.append(f"{rel}: {what} timed out ({side})")
                        if what == "tidy" and NO_RUN.search(out):
                            notes.append(
                                f"{rel}: clang-tidy did not run ({side}): {NO_RUN.search(out).group(0)}"
                            )
                        for line in out.splitlines():
                            m = ERROR.match(line.strip())
                            if not m:
                                continue
                            f = m.group("file")
                            if os.path.isabs(f):
                                f = os.path.relpath(f, root)
                            msg = m.group("msg").replace(
                                str(root) + "/", ""
                            )  # "lambda at <path>" reads alike
                            if side == "dev":
                                dev_errors.add((f, msg))
                                continue
                            key = (
                                f"{f}:{m.group('line')}:{m.group('col')}: error: {msg}"
                            )
                            errors.setdefault(key, set()).add(
                                what + (f", macros {label}" if macros else "")
                            )
                if cut:
                    break
            if cut:
                break
        notes += configs.notes
    on_dev = {
        k
        for k in errors
        if (k.split(":", 1)[0], k.split(": error: ", 1)[1]) in dev_errors
    }
    new = sorted(k for k in errors if k not in on_dev)
    feat = (
        f"feature macros off and on: {', '.join(sorted(macros_seen))}"
        if macros_seen
        else "no feature macro"
    )
    used = " + ".join(
        x for x, y in (("clang-tidy", tidy), ("clang -Werror", clangxx)) if y
    )
    n_tus = f"{len(tus)} TU{'s' if len(tus) != 1 else ''}"
    print(
        f"Superbuild tidy (ran): {len(new)} error{'s' if len(new) != 1 else ''} in {n_tus} ({used}; {feat}; "
        f"{runs} runs{'; develop compared' if dev else ''})"
    )
    for k in new[:SHOWN]:
        print(f"   {k}  [{'; '.join(sorted(errors[k]))}]")
    if len(new) > SHOWN:
        print(f"   … {len(new) - SHOWN} more")
    if on_dev:
        print(
            f"   {len(on_dev)} more error(s) develop has too, not counted; e.g. {sorted(on_dev)[0]}"
        )
    if dropped:
        print(f"   flags clang does not know, dropped: {' '.join(sorted(dropped))}")
    if no_tidy:
        print(
            f"   clang-tidy not run (no .clang-tidy above the TU): {' '.join(sorted(no_tidy))}"
        )
    for n in notes:
        print(f"   {n}")
    if cut:
        print(f"   cut at the {a.budget} s budget: not every TU ran")
    if not tus:
        print("TIDY skip no changed C/C++ file is a TU of the native build")
    elif new:
        print(f"TIDY fail {len(new)} error{'s' if len(new) != 1 else ''} in {n_tus}")
    elif cut:
        print(f"TIDY skip budget cut, 0 errors so far in {n_tus}")
    else:
        print(f"TIDY pass 0 errors in {n_tus} ({feat})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
