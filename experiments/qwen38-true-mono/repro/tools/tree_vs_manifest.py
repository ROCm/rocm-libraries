#!/usr/bin/env python3
"""Compare a local tree against a q38 stage manifest (sync_pack.py format: '<sha256>  <relpath>' of the
transmitted bytes, text files CRLF->LF). Uses sync_pack.py's own exclusion and text rules so the hashes match.

  tree_vs_manifest.py <tree> <manifest> [--only PREFIX ...] [--ignore GLOB ...] [--quiet]

Prints one line per difference (CHANGED / ONLY_TREE / ONLY_MANIFEST) and a summary; exit 0 iff no difference.
"""
import argparse
import fnmatch
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
try:
    import sync_pack as sp  # hackathon-prep/tools/sync_pack.py (same rules as the stage packer)
except ImportError:  # inside a stage: a vendored copy of the rules
    sp = None

EXC_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules"}
EXC_DIR_GLOBS = ["build", "build-*", "build_*", ".venv*", "venv*", "*.egg-info"]
EXC_FILE_GLOBS = ["*.pyc", "*.pyo", "*.o", "*.a", "*.so", "*.so.*", "*.obj", "*.dll", "*.lib", "*.exe", "*.pyd"]
TEXT_EXT = {".py", ".sh", ".txt", ".md", ".yaml", ".yml", ".json", ".cmake", ".cpp", ".cc", ".c", ".h", ".hpp",
            ".hh", ".cu", ".hip", ".in", ".toml", ".cfg", ".ini", ".s", ".ll", ".mlir", ".td", ".fbs", ".proto",
            ".rst", ".csv", ".cl"}
TEXT_NAMES = {"CMakeLists.txt", "Makefile", "Dockerfile", "LICENSE"}


def _long(path):
    # Windows MAX_PATH: rocKE has instance paths > 260 chars
    if os.name == "nt":
        path = os.path.abspath(path)
        if not path.startswith(r"\\?" + "\\"):
            path = r"\\?" + "\\" + path
    return path


def walk(src):
    src = _long(src)
    for root, dirs, files in os.walk(src):
        rel_root = os.path.relpath(root, src).replace("\\", "/")
        dirs[:] = sorted(d for d in dirs if d not in EXC_DIRS and not any(fnmatch.fnmatch(d, g) for g in EXC_DIR_GLOBS))
        for f in sorted(files):
            if any(fnmatch.fnmatch(f, g) for g in EXC_FILE_GLOBS):
                continue
            yield (f if rel_root == "." else rel_root + "/" + f), os.path.join(root, f)


def is_text(name):
    if sp is not None:
        return sp.is_text(name)
    return os.path.splitext(name)[1].lower() in TEXT_EXT or name in TEXT_NAMES


def tree_hashes(src):
    out = {}
    for rel, full in walk(src):
        with open(full, "rb") as fh:
            data = fh.read()
        if is_text(os.path.basename(full)) and b"\r\n" in data:
            data = data.replace(b"\r\n", b"\n")
        hs = {hashlib.sha256(data).hexdigest()}
        if b"\0" not in data:  # a non-text-extension file may sit in a stage as CRLF or LF (Windows worktree)
            lf = data.replace(b"\r\n", b"\n")
            hs.add(hashlib.sha256(lf).hexdigest())
            hs.add(hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest())
        out[rel] = hs
    return out


def read_manifest(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                h, rel = line.split("  ", 1)
                out[rel] = h
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("manifest")
    ap.add_argument("--only", action="append", default=[], help="restrict to paths with this prefix")
    ap.add_argument("--ignore", action="append", default=[], help="fnmatch glob of paths to skip")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    t, m = tree_hashes(a.tree), read_manifest(a.manifest)

    def keep(p):
        if a.only and not any(p.startswith(o) for o in a.only):
            return False
        return not any(fnmatch.fnmatch(p, g) for g in a.ignore)

    diffs = []
    for p in sorted(set(t) | set(m)):
        if not keep(p):
            continue
        if p in t and p in m:
            if m[p] not in t[p]:
                diffs.append(("CHANGED", p))
        elif p in t:
            diffs.append(("ONLY_TREE", p))
        else:
            diffs.append(("ONLY_MANIFEST", p))
    if not a.quiet:
        for k, p in diffs:
            print(f"{k} {p}")
    n = sum(1 for p in set(t) | set(m) if keep(p))
    print(f"SUMMARY files={n} changed={sum(k == 'CHANGED' for k, _ in diffs)} "
          f"only_tree={sum(k == 'ONLY_TREE' for k, _ in diffs)} only_manifest={sum(k == 'ONLY_MANIFEST' for k, _ in diffs)}")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
