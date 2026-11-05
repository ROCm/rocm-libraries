#!/usr/bin/env python3
"""EOL-insensitive file-level diff of two trees (A = pristine base, B = changed tree).

  tree_diff.py <A> <B> [--prefix P ...] [--exclude GLOB ...] [--json out.json]

A file is compared with CRLF->LF applied to both sides when it contains no NUL byte (the q38 stage packer converts
only some extensions, git archive emits LF). Caches and build outputs are always excluded (see EXC_*). Prints
ADDED / CHANGED / DELETED lines; DELETED = in A, not in B (may be a stage-packer exclusion: build*, *.so, ...).
"""
import argparse
import fnmatch
import json
import os
import sys

EXC_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "node_modules", "target"}
EXC_DIR_GLOBS = ["build", "build-*", "build_*", ".venv*", "venv*", "*.egg-info"]
EXC_FILE_GLOBS = ["*.pyc", "*.pyo", "*.o", "*.a", "*.so", "*.so.*", "*.obj", "*.dll", "*.lib", "*.exe", "*.pyd",
                  "*.log", ".coverage"]


def _long(path):
    # Windows MAX_PATH: rocKE has instance paths > 260 chars
    if os.name == "nt":
        path = os.path.abspath(path)
        if not path.startswith(r"\\?" + "\\"):
            path = r"\\?" + "\\" + path
    return path


def files(root):
    root = _long(root)
    out = {}
    for d, dirs, fs in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in EXC_DIRS and not any(fnmatch.fnmatch(x, g) for g in EXC_DIR_GLOBS))
        rel_d = os.path.relpath(d, root).replace("\\", "/")
        for f in fs:
            if any(fnmatch.fnmatch(f, g) for g in EXC_FILE_GLOBS):
                continue
            rel = f if rel_d == "." else rel_d + "/" + f
            out[rel] = os.path.join(d, f)
    return out


def norm(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if b"\0" not in data:
        data = data.replace(b"\r\n", b"\n")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--prefix", action="append", default=[])
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--json")
    a = ap.parse_args()
    fa, fb = files(a.a), files(a.b)

    def keep(p):
        if a.prefix and not any(p.startswith(x) for x in a.prefix):
            return False
        return not any(fnmatch.fnmatch(p, g) for g in a.exclude)

    res = {"ADDED": [], "CHANGED": [], "DELETED": []}
    for p in sorted(set(fa) | set(fb)):
        if not keep(p):
            continue
        if p in fa and p in fb:
            if norm(fa[p]) != norm(fb[p]):
                res["CHANGED"].append(p)
        elif p in fb:
            res["ADDED"].append(p)
        else:
            res["DELETED"].append(p)
    for k in ("CHANGED", "ADDED", "DELETED"):
        for p in res[k]:
            print(k, p)
    print("SUMMARY", " ".join(f"{k.lower()}={len(v)}" for k, v in res.items()))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
