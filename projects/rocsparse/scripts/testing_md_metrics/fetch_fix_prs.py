#!/usr/bin/env python3
# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

"""Fetch merged rocSPARSE fix PRs and classify whether each added tests.

A fix is a merged PR whose title is a rocSPARSE fix (fix(rocsparse), "[rocSPARSE] Fix",
"Fix rocSPARSE") and whose diff touches projects/rocsparse/library/ (the `library` field).
Rows without `library` are the wider title-only set.
A unit test: added lines in a C/C++ file under projects/rocsparse/clients/unittests/.
Any new test: also added lines in integration YAML or testing sources.
"""
import json, os, re, subprocess, sys

REPO = "ROCm/rocm-libraries"
TITLE = re.compile(r"^(fix\(rocsparse\)|\[rocsparse\]\s*fix|fix\s+rocsparse)", re.I)
CPP = (".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".hip", ".cuh")

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".files-cache")

def gh(*args):
    return subprocess.check_output(["gh", *args], text=True)

def list_prs(since):
    out = gh("pr", "list", "--repo", REPO, "--state", "merged",
             "--search", f"rocsparse in:title merged:>={since}",
             "--limit", "1000", "--json", "number,title,mergedAt,author,url")
    return json.loads(out)

def files(num):
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"{num}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    res = _files(num)
    json.dump(res, open(cache, "w"))
    return res

def _files(num):
    page, res = 1, []
    while True:
        chunk = json.loads(gh("api", f"repos/{REPO}/pulls/{num}/files?per_page=100&page={page}"))
        res += chunk
        if len(chunk) < 100:
            return res
        page += 1

TEST_DIRS = ("projects/rocsparse/clients/tests/", "projects/rocsparse/clients/testings/",
             "projects/rocsparse/clients/unittests/", "projects/rocsparse/clients/include/testing_")

def classify(fs):
    """Return (touches library, added a unit test, added any new test)."""
    lib = unit = any_test = False
    for f in fs:
        n = f["filename"]
        if n.startswith("projects/rocsparse/library/"):
            lib = True
        if f.get("additions", 0) == 0:
            continue
        is_cpp = n.endswith(CPP)
        if n.startswith("projects/rocsparse/clients/unittests/") and is_cpp:
            unit = True
        if n.startswith(TEST_DIRS) and (is_cpp or n.endswith((".yaml", ".yml"))):
            any_test = True
    return lib, unit, unit or any_test

def main(since="2025-07-01", out="fix_prs.json"):
    rows = []
    for pr in list_prs(since):
        if not TITLE.match(pr["title"]):
            continue
        lib, unit, anyt = classify(files(pr["number"]))
        rows.append(dict(number=pr["number"], title=pr["title"], merged=pr["mergedAt"][:10],
                         author=pr["author"]["login"], library=lib, unit=unit, any_test=anyt))
    rows.sort(key=lambda r: r["merged"])
    json.dump(rows, open(out, "w"), indent=1)
    print(len(rows), "fixes ->", out)

if __name__ == "__main__":
    main(*sys.argv[1:])
