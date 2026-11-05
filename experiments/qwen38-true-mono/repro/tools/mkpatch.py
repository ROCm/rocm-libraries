#!/usr/bin/env python3
"""Write ONE git-style unified patch that turns tree A into tree B for a list of files (LF, no binary hunks).

  mkpatch.py <A> <B> <out.patch> (--files list.txt | --json tree_diff.json) [--strip-prefix P] [--add-prefix P]

Both sides are copied to a scratch dir with CRLF->LF (files without NUL bytes only; a NUL file is refused), then
`git diff --no-index` per file (read-only: no repository is touched). Mode lines are dropped except `new file mode`
(100755 for *.sh / shebang files, else 100644). --add-prefix rewrites the paths inside the patch (e.g. the rocKE
stage is rooted at dnn-providers/hip-kernel-provider/rocke in the repo).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


def _long(path):
    if os.name == "nt":
        path = os.path.abspath(path)
        if not path.startswith(r"\\?" + "\\"):
            path = r"\\?" + "\\" + path
    return path


def read_norm(path):
    with open(_long(path), "rb") as fh:
        data = fh.read()
    if b"\0" in data:
        raise SystemExit(f"binary file refused: {path}")
    return data.replace(b"\r\n", b"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    ap.add_argument("--files")
    ap.add_argument("--json")
    ap.add_argument("--add-prefix", default="")
    a = ap.parse_args()
    if a.json:
        d = json.load(open(a.json, encoding="utf-8"))
        rels = sorted(d.get("CHANGED", []) + d.get("ADDED", []) + d.get("DELETED", []))
    else:
        rels = [l.strip() for l in open(a.files, encoding="utf-8") if l.strip() and not l.startswith("#")]
    tmp = tempfile.mkdtemp(prefix="q38rp-mkpatch-")
    out = []
    for rel in rels:
        pa, pb = os.path.join(a.a, rel), os.path.join(a.b, rel)
        ea, eb = os.path.exists(_long(pa)), os.path.exists(_long(pb))
        sides = []
        for tag, p, e in (("old", pa, ea), ("new", pb, eb)):
            if e:
                q = os.path.join(tmp, tag, rel)
                os.makedirs(os.path.dirname(q), exist_ok=True)
                with open(q, "wb") as fh:
                    fh.write(read_norm(p))
                sides.append(f"{tag}/{rel}")
            else:
                sides.append("/dev/null")
        r = subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", "diff", "--no-index",
                            "--no-color", "--no-ext-diff", "--full-index", sides[0], sides[1]],
                           cwd=tmp, capture_output=True)
        if r.returncode not in (0, 1):
            raise SystemExit(f"git diff failed for {rel}: {r.stderr.decode(errors='replace')}")
        txt = r.stdout.decode("utf-8", errors="strict").replace("\r\n", "\n")
        if not txt:
            continue
        lines = []
        for ln in txt.split("\n"):
            if ln.startswith(("old mode ", "new mode ")):
                continue
            if ln.startswith("new file mode "):
                data = read_norm(pb)
                ln = "new file mode " + ("100755" if rel.endswith(".sh") or data.startswith(b"#!") else "100644")
            if ln.startswith("deleted file mode "):
                ln = "deleted file mode 100644"
            if ln.startswith(("diff --git ", "--- ", "+++ ")):
                ln = re.sub(r"(^|\s)([ab])/(old|new)/", lambda m: f"{m.group(1)}{m.group(2)}/{a.add_prefix}", ln)
            lines.append(ln)
        out.append("\n".join(lines).rstrip("\n") + "\n")
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(out))
    shutil.rmtree(tmp, ignore_errors=True)
    n = sum(1 for x in "".join(out).split("\n") if x.startswith("diff --git "))
    print(f"patch {a.out}: {n} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
