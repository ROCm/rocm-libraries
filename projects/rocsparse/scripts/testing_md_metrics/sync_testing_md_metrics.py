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

"""Refresh the measured sections of rocSPARSE's testing.md.

Run from the repository root (needs `gh` signed in, network access, and matplotlib for the
plot). With S=projects/rocsparse/scripts/testing_md_metrics/sync_testing_md_metrics.py and
T=projects/rocsparse/clients/tests/testing.md:

    python3 $S coverage       # record new develop uploads in the CSV
    python3 $S fixes          # refresh fix_prs.json from GitHub
    python3 $S render         # print the generated markdown block
    python3 $S apply $T       # replace the block between the markers
    python3 $S all $T         # coverage + fixes + plot + apply

The block in testing.md sits between

    <!-- metrics:start -->  and  <!-- metrics:end -->

Everything between the markers is overwritten. Hand-written text goes outside them.

What each number is
-------------------
Coverage: the `rocSPARSE` and `rocSPARSE-unit` Codecov flags on the `develop` branch,
read from the upload the `codecov/rocsparse/develop` Jenkins job made for a commit
(`upload_type == uploaded`, never a carried-forward one). One point per day: the latest
commit of the day that has a qualifying upload for both flags; when a build uploaded two reports of a flag, the higher one.
An upload that covers fewer than MIN_FILES files is skipped, because Codecov
sometimes processes only part of a report (build 383 on 21 Nov 2026 read 100% on
1 file and 19 lines while its job had merged 549 files).

Fixes: merged pull requests titled as a rocSPARSE fix (see fetch_fix_prs.py) whose
diff touches projects/rocsparse/library/. A unit test is added lines in a C/C++ file
under clients/unittests/. "Any new test" also counts YAML or C/C++ under
clients/tests, clients/testings and clients/include/testing_*.
"""
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "coverage_history.csv")
FIXES_PATH = os.path.join(HERE, "fix_prs.json")
PLOT_PATH = os.path.normpath(
    os.path.join(HERE, "..", "..", "clients", "tests", "testing_coverage_history.svg"))
CODECOV = "https://api.codecov.io/api/v2/github/ROCm/repos/rocm-libraries"
MIN_FILES = 100
START = "2025-07-01"
LAYER = "2026-08-12"  # unit-test layer landed
DOC = "2026-09-23"  # testing.md merged
FIRST_WINDOW_END = "2026-11-06"

# Pull requests that fix a bug the unit tests found: PR -> (ticket, short summary).
# Maintained by hand: a bug counts when its ticket text says a unit test from the
# 26 Oct 2026 coverage suite reaches it.
UNIT_FOUND = {
    14461: ("AISPARSE-897", "spgeam uses the scalar type instead of the compute type"),
    14462: ("AISPARSE-898", "csritilu0 async_split faults on a null residual norm"),
    14429: ("AISPARSE-900", "csric0_set_tolerance dereferences a null info"),
    14480: ("AISPARSE-906", "csxsldu_preprocess leaves the column-direction base unset"),
    14430: ("AISPARSE-901", "CSC check_matrix applies the fill mode to the transpose"),
    14477: ("AISPARSE-904", "three headers document a status the function does not return"),
    14478: ("AISPARSE-905", "sptrsv ignores a device alpha when the types differ"),
    14481: ("AISPARSE-907", "spgemm ignores transposed operands"),
    14482: ("AISPARSE-908", "spgeam ignores transposed operands"),
    14490: ("AISPARSE-909", "spildlt0 ignores the descriptor boost"),
    14487: ("AISPARSE-910", "convert_array returns success for pairs it does not convert"),
    14488: ("AISPARSE-911", "ell2csr_nnz writes nothing on an empty matrix"),
    15497: ("AISPARSE-926", "empty spgeam analysis writes through a null C"),
    15814: ("AISPARSE-903", "csric0 conjugates the wrong operand in the inner sum"),
    14427: ("AISPARSE-896", "csric0 / csrildlt0 hash lookup hangs when the table is exactly full"),
    14428: ("AISPARSE-899", "empty csritsv reads a null tolerance"),
    14431: ("AISPARSE-902", "BSR-to-BSR sparse_to_sparse does not convert direction or block size"),
    15566: ("AISPARSE-924", "blocked ELL SpMM scales the wrong rows of C"),
}


def _get(url):
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "rocsparse-testing-md-sync"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


# ----------------------------------------------------------------- coverage

def read_csv():
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows):
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "develop_combined", "unit_only", "build"])
        w.writeheader()
        w.writerows(rows)


def _results(url):
    """All results of a paginated Codecov list, following `next`."""
    out = []
    while url:
        page = _get(url)
        out += page.get("results") or []
        url = page.get("next")
    return out


def sync_coverage():
    rows = read_csv()
    last = rows[-1]["date"]
    commits, page = [], 1
    while True:
        chunk = _get(f"{CODECOV}/commits?branch=develop&page_size=100&page={page}")["results"]
        commits += chunk
        if not chunk or chunk[-1]["timestamp"][:10] < last:
            break
        page += 1
    # The last recorded day is read again: a later commit that day replaces its point.
    new = sorted((c for c in commits if c["timestamp"][:10] >= last), key=lambda c: c["timestamp"])
    per_day = {}
    for c in new:
        flags = {}
        for u in _results(f"{CODECOV}/commits/{c['commitid']}/uploads/"):
            name = ",".join(u.get("flags") or [])
            if name not in ("rocSPARSE", "rocSPARSE-unit") or u.get("upload_type") != "uploaded":
                continue
            totals = u.get("totals") or {}
            if totals.get("coverage") is None or totals.get("files", 0) < MIN_FILES:
                print(f"skip {c['commitid'][:9]} {name}: {totals.get('files')} files")
                continue
            cov = float(totals["coverage"])
            if cov > flags.get(name, (-1, None))[0]:
                flags[name] = (cov, u.get("build_code"))
        if len(flags) == 2:
            per_day[c["timestamp"][:10]] = flags  # later commits overwrite earlier ones
    by_date = {r["date"]: r for r in rows}
    for day in sorted(per_day):
        f = per_day[day]
        by_date[day] = {"date": day, "develop_combined": f["rocSPARSE"][0],
                        "unit_only": f["rocSPARSE-unit"][0], "build": f["rocSPARSE"][1]}
        print("recorded", by_date[day])
    write_csv([by_date[d] for d in sorted(by_date)])
    if not per_day:
        print("no develop uploads on or after", last)


# -------------------------------------------------------------------- fixes

def sync_fixes():
    sys.path.insert(0, HERE)
    import fetch_fix_prs
    fetch_fix_prs.main(START, FIXES_PATH)


def _pct(a, b):
    return f"{a} of {b} ({round(100 * a / b)}%)" if b else "0"


def fix_table(today):
    with open(FIXES_PATH) as f:
        rows = json.load(f)
    lib = [r for r in rows if r["library"]]

    def window(a, b, only=None):
        sel = [r for r in lib if a <= r["merged"] <= b]
        if only is True:
            sel = [r for r in sel if r["number"] in UNIT_FOUND]
        elif only is False:
            sel = [r for r in sel if r["number"] not in UNIT_FOUND]
        return sel

    day_after = (dt.date.fromisoformat(FIRST_WINDOW_END) + dt.timedelta(days=1)).isoformat()
    lines = ["| Period | Library fixes | Added a unit test | Added any new test |",
             "|---|---:|---:|---:|"]

    def row(label, sel):
        n = len(sel)
        lines.append(f"| {label} | {n} | {_pct(sum(r['unit'] for r in sel), n)} | "
                     f"{_pct(sum(r['any_test'] for r in sel), n)} |")

    day_before_layer = (dt.date.fromisoformat(LAYER) - dt.timedelta(days=1)).isoformat()
    day_before_doc = (dt.date.fromisoformat(DOC) - dt.timedelta(days=1)).isoformat()
    row(f"Before the unit-test layer, {START} to {day_before_layer}", window(START, day_before_layer))
    row(f"Unit tests existed, {LAYER} to {day_before_doc}", window(LAYER, day_before_doc))
    row(f"Since this document, {DOC} to {FIRST_WINDOW_END}", window(DOC, FIRST_WINDOW_END))
    row(f"Since this document, {day_after} to {today}", window(day_after, today))
    row(f"**Since this document, {DOC} to {today}**", window(DOC, today))
    row("&nbsp;&nbsp;of which fix a bug the unit tests found", window(DOC, today, True))
    row("&nbsp;&nbsp;all other fixes", window(DOC, today, False))
    after = [r for r in window(DOC, today) if r["unit"]]
    counts = {}
    for r in after:
        counts[r["author"]] = counts.get(r["author"], 0) + 1
    top = max(counts.values()) if counts else 0
    note = (f"{len(after)} unit-tested fixes since this document were opened by "
            f"{len(counts)} different authors; the most active opened {top} of them.")
    return "\n".join(lines), note


def pr_state(n):
    out = json.loads(subprocess.check_output(
        ["gh", "pr", "view", str(n), "--repo", "ROCm/rocm-libraries",
         "--json", "state,mergedAt"], text=True))
    return "Merged" if out["mergedAt"] else out["state"].capitalize()


def bug_table():
    lines = ["| PR | Ticket | Status | Bug |", "|---|---|---|---|"]
    merged = 0
    items = sorted(UNIT_FOUND.items(), key=lambda kv: (pr_state(kv[0]) != "Merged", kv[0]))
    for pr, (ticket, text) in items:
        state = pr_state(pr)
        merged += state == "Merged"
        lines.append(f"| #{pr} | {ticket} | {state} | {text} |")
    return "\n".join(lines), len(UNIT_FOUND), merged


# ------------------------------------------------------------------- render

def render(today):
    cov = read_csv()[-1]
    table, note = fix_table(today)
    bugs, total, merged = bug_table()
    build = f", build {cov['build']}" if cov.get("build") else ""
    return f"""_Measured on {today}. Refresh it as described in "Refreshing this section" below._

**Latest develop coverage reading** ({cov['date']}{build}): `rocSPARSE` flag
{float(cov['develop_combined']):.2f}%, `rocSPARSE-unit` flag {float(cov['unit_only']):.2f}%,
against an 80% target.

![Line chart of rocSPARSE develop coverage by day: combined (blue) and unit-only (green) against the 80% target, with vertical markers for the unit-test layer, this document, #15395, and #15569.](testing_coverage_history.svg)

**Library fixes and whether they added tests:**

{table}

{note}

**Fixes for bugs the unit tests found:** {total} pull requests, {merged} merged.

{bugs}"""


def apply(path, today):
    with open(path) as f:
        text = f.read()
    block = f"<!-- metrics:start -->\n{render(today)}\n<!-- metrics:end -->"
    new, n = re.subn(r"<!-- metrics:start -->.*?<!-- metrics:end -->",
                     lambda _m: block, text, flags=re.S)
    if n != 1:
        sys.exit(f"{path}: expected one metrics marker pair, found {n}")
    with open(path, "w") as f:
        f.write(new)
    print("updated", path)


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "render"
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()  # GitHub dates are UTC
    if cmd == "coverage":
        sync_coverage()
    elif cmd == "fixes":
        sync_fixes()
    elif cmd == "render":
        print(render(today))
    elif cmd == "apply":
        apply(argv[2], today)
    elif cmd == "all":
        sync_coverage()
        sync_fixes()
        subprocess.check_call([sys.executable, os.path.join(HERE, "plot_coverage_history.py"),
                               CSV_PATH, PLOT_PATH])
        apply(argv[2], today)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
