"""WP-R3: classify every file that differs between the rust tree and the mono working tree.

Read-only on git (git cat-file / git merge-file -p on temp files). Prints, per path:
class, lines changed rust-vs-HEAD, mono-vs-HEAD, and whether a 3-way merge is clean.
"""
from __future__ import annotations
import filecmp, os, subprocess, sys, tempfile

REPO = "C:/Users/vanantha/Documents/dev/copy-rocm-lib"
MONO = REPO + "/dnn-providers/hip-kernel-provider/rocke"
RUST = "C:/Users/vanantha/Documents/dev/copy-rocm-lib-rust"
PFX = "dnn-providers/hip-kernel-provider/rocke/"
SKIP = {"target", "__pycache__", ".pytest_cache", ".ruff_cache", ".git"}

def walk(root):
    out = set()
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP]
        for f in files:
            if f.endswith(".pyc"):
                continue
            out.add(os.path.relpath(os.path.join(d, f), root).replace("\\", "/"))
    return out

def norm(b):
    return b.replace(b"\r\n", b"\n")

def head_blob(rel):
    r = subprocess.run(["git", "--no-optional-locks", "-C", REPO, "cat-file", "blob", "HEAD:" + PFX + rel],
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None

def nlines(a, b):
    import difflib
    return sum(1 for l in difflib.unified_diff(a.decode(errors="replace").splitlines(),
                                               b.decode(errors="replace").splitlines(), n=0)
               if l[:1] in "+-" and not l.startswith(("+++", "---")))

def merge3(ours, base, theirs):
    with tempfile.TemporaryDirectory() as t:
        p = []
        for n, b in (("o", ours), ("b", base), ("t", theirs)):
            q = os.path.join(t, n)
            open(q, "wb").write(b)
            p.append(q)
        r = subprocess.run(["git", "merge-file", "-p", *p], capture_output=True)
        return r.returncode, r.stdout

def main():
    rust, mono = walk(RUST), walk(MONO)
    rows = []
    for rel in sorted(rust | mono):
        if rel.startswith("platform/rust/"):
            if rel not in mono:
                rows.append(("crate-new", rel, "", "", ""))
            continue
        if rel not in mono:
            rows.append(("rust-only", rel, "", "", ""))
            continue
        if rel not in rust:
            continue  # mono-only: keep
        a, b = open(os.path.join(RUST, rel), "rb").read(), open(os.path.join(MONO, rel), "rb").read()
        if norm(a) == norm(b):
            if a != b:
                rows.append(("eol-only", rel, "", "", ""))
            continue
        base = head_blob(rel)
        if base is None:
            rows.append(("untracked-both", rel, nlines(norm(b), norm(a)), "", ""))
            continue
        base = norm(base)
        rr, mm = nlines(base, norm(a)), nlines(base, norm(b))
        if norm(b) == base:
            cls = "mono==HEAD"
        elif norm(a) == base:
            cls = "rust==HEAD"
        else:
            cls = "both"
        rc, _ = merge3(norm(b), base, norm(a))
        rows.append((cls, rel, rr, mm, "clean" if rc == 0 else f"CONFLICT({rc})"))
    for r in rows:
        print("\t".join(str(x) for x in r))

if __name__ == "__main__":
    main()
