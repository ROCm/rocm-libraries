#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build the gfx1250-strict nightly test report and post it to a webhook.

Reads the (already scrubbed) logs that gfx1250_hipblaslt_tensilelite_test.yml
writes: the hipBLASLt gtest shard logs and the TensileLite tox log. It
compares every test's result with the previous nightly's uploaded results
and POSTs {"subject", "html", "text", "ok", "run_url", "commit", "branch",
"tests": [{suite, test, status, new}]} as JSON to the webhook. The flow behind
the webhook (e.g. Power Automate) decides who gets it.

Usage:
    python gfx1250_nightly_report.py --results "$GITHUB_WORKSPACE" \
        --steps "SDK=success, Build=success, ..." --webhook-url "$URL"
    python gfx1250_nightly_report.py --results DIR --dry-run   # print, don't send
    # Compare with results on disk instead of fetching the previous nightly:
    python gfx1250_nightly_report.py --results DIR --previous PREV_DIR --previous-run-id ID --dry-run

Uses GITHUB_* environment variables for the run metadata, and GH_TOKEN (needs
actions: read) for the previous nightly's results. Standard library only.
"""

import argparse
import datetime
import glob
import html
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
import zipfile

ARTIFACT = "gfx1250-strict-test-results"
WORKFLOW = "gfx1250_hipblaslt_tensilelite_test.yml"
API = "https://api.github.com"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOOKBACK_DAYS = 7  # as the internal nightly's comparison
LAST_NIGHT_HOURS = (
    30  # the nightly starts at 03:00 UTC; allow for a late start or a long run
)
PASS, FAIL, SKIP = "pass", "fail", "skip"

# ---------------------------------------------------------------- parsing


def parse_gtest(results):
    """Per-test results from the gtest shard logs: (counts or None if not run, {name: status})."""
    logs = sorted(
        glob.glob(os.path.join(results, "test_results", "hipblaslt_test_shard*.log"))
    )
    status, total = {}, 0
    for path in logs:
        for line in open(path, errors="replace"):
            if m := re.match(r"^\[==========\] (\d+) tests? from .* ran", line):
                total += int(m.group(1))
            elif m := re.match(r"^\[       OK \] (\S+)", line):
                status[m.group(1)] = PASS
            elif m := re.match(r"^\[  SKIPPED \] ([^\s,]+)$", line.rstrip()):
                status[m.group(1)] = SKIP
            elif m := re.match(r"^\[  FAILED  \] ([^\s,]+)", line):
                if not m.group(1).isdigit():
                    status[m.group(1)] = FAIL
    if not total:
        return None, {}
    counts = {
        "Passed": sum(s == PASS for s in status.values()),
        "Failed": sum(s == FAIL for s in status.values()),
        "Skipped": sum(s == SKIP for s in status.values()),
    }
    counts["Total"] = total
    return counts, status


def parse_pytest(results):
    """Per-session counts and per-test results from the tox log: ({session: counts}, {session: {id: status}})."""
    path = os.path.join(results, "tensilelite_test_results", "common_test.log")
    sessions, status = {}, {"Rocisa": {}, "Common": {}}
    if not os.path.isfile(path):
        return sessions, status
    summaries = []
    word = {
        "PASSED": PASS,
        "FAILED": FAIL,
        "ERROR": FAIL,
        "XPASS": FAIL,
        "SKIPPED": SKIP,
        "XFAIL": SKIP,
    }
    for raw in open(path, errors="replace"):
        line = ANSI.sub("", raw.rstrip("\n"))
        if re.match(r"^=+ .*\b(passed|failed|error|errors|skipped)\b.* =+$", line):
            summaries.append(line)
        elif m := (
            re.search(r"\] (PASSED|FAILED|ERROR|XPASS|SKIPPED|XFAIL) (\S+)", line)
            or re.match(r"^(FAILED|ERROR) (\S+)", line)
        ):
            tid = m.group(2)
            session = "Rocisa" if tid.startswith("rocisa/") else "Common"
            if (
                status[session].get(tid) != FAIL
            ):  # a failure line wins over an earlier progress line
                status[session][tid] = word[m.group(1)]
    # tox runs the rocisa tests first, then the common tests (absent if tox stopped early).
    for name, line in (
        ("Rocisa", summaries[0] if summaries else None),
        ("Common", summaries[-1] if len(summaries) >= 2 else None),
    ):
        if line is None:
            continue
        c = {
            k: int(n)
            for n, k in re.findall(
                r"(\d+) (passed|failed|skipped|xfailed|xpassed|errors?)", line
            )
        }
        counts = {
            "Passed": c.get("passed", 0),
            "Failed": c.get("failed", 0)
            + c.get("error", 0)
            + c.get("errors", 0)
            + c.get("xpassed", 0),
            "Skipped": c.get("skipped", 0),
        }
        if c.get("xfailed"):
            counts["XFailed"] = c["xfailed"]
        counts["Total"] = sum(counts.values())
        sessions[name] = counts
    return sessions, status


def suites(results):
    """{suite label: (counts or None, {test: status})} for this job's three suites."""
    h_counts, h_status = parse_gtest(results)
    t_sessions, t_status = parse_pytest(results)
    return {
        "hipBLASLt": (h_counts, h_status),
        "TensileLite Common": (t_sessions.get("Common"), t_status["Common"]),
        "Rocisa": (t_sessions.get("Rocisa"), t_status["Rocisa"]),
    }


def compare(cur, prev):
    """Sort test changes between two runs of a suite.

    New: failing now, not failing before. Existing: failing in both.
    Fixed: failing before, passing now. Skipped/removed: failing or passing
    before, skipped now or not run at all (a skip is not a fix).
    """
    new = sorted(t for t, s in cur.items() if s == FAIL and prev.get(t) != FAIL)
    existing = sorted(t for t, s in cur.items() if s == FAIL and prev.get(t) == FAIL)
    fixed = sorted(t for t, s in prev.items() if s == FAIL and cur.get(t) == PASS)
    gone = sorted(
        t for t, s in prev.items() if s in (FAIL, PASS) and cur.get(t) in (SKIP, None)
    )
    return {"New": new, "Existing": existing, "Fixed": fixed, "Skipped/removed": gone}


def short(test):
    """A readable test name: drop the matmul and test_config boilerplate."""
    test = test.replace("_/matmul_test.matmul/", "")
    m = re.match(
        r"tensilelite/Tests/common/test_config\w*\.py::test_config\w*\[tensilelite/Tests/common/(.*)\]$",
        test,
    )
    return m.group(1) if m else test.replace("tensilelite/Tests/common/", "")


# ---------------------------------------------------------------- GitHub


def _api(url, token):
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
        },
    )
    return urllib.request.urlopen(req, timeout=60)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _download(art, token, dest):
    # The zip URL redirects to signed storage, which rejects the GitHub token: follow it without auth.
    req = urllib.request.Request(
        art["archive_download_url"], headers={"Authorization": f"Bearer {token}"}
    )
    try:
        urllib.request.build_opener(_NoRedirect).open(req, timeout=60)
        raise RuntimeError("artifact download did not redirect")
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise
        data = urllib.request.urlopen(e.headers["Location"], timeout=120).read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(dest)


def run_info(repo, run_id, token):
    r = json.load(_api(f"{API}/repos/{repo}/actions/runs/{run_id}", token))
    return {
        "date": r["created_at"][:10],
        "url": r["html_url"],
        "sha": r["head_sha"][:6],
    }


def fetch_previous(repo, run_id, token, now, query="event=schedule&branch=develop"):
    """Download the most recent earlier nightly that has results, within LOOKBACK_DAYS.

    Returns {"status": "ok" | "none", "date", "url", "sha", "dir", "hours", "missing": [(date, url)]}.
    "missing" lists the newer nightlies that left no results (crashed, cancelled or expired).
    """
    out = {"status": "none", "missing": []}
    if not (repo and token):
        return out
    runs = json.load(
        _api(
            f"{API}/repos/{repo}/actions/workflows/{WORKFLOW}/runs"
            f"?{query}&status=completed&per_page=20",
            token,
        )
    )
    for run in runs.get("workflow_runs", []):
        if str(run["id"]) == str(run_id):
            continue
        started = datetime.datetime.fromisoformat(
            run["created_at"].replace("Z", "+00:00")
        )
        hours = (now - started).total_seconds() / 3600
        if hours > LOOKBACK_DAYS * 24:
            break
        arts = json.load(
            _api(f"{API}/repos/{repo}/actions/runs/{run['id']}/artifacts", token)
        )
        art = next(
            (
                a
                for a in arts.get("artifacts", [])
                if a["name"] == ARTIFACT and not a["expired"]
            ),
            None,
        )
        if not art:
            out["missing"].append((run["created_at"][:10], run["html_url"]))
            continue
        dest = os.path.join(
            os.environ.get("RUNNER_TEMP", "/tmp"), f"previous-{run['id']}"
        )
        _download(art, token, dest)
        out.update(
            status="ok",
            date=run["created_at"][:10],
            url=run["html_url"],
            sha=run["head_sha"][:6],
            dir=dest,
            hours=hours,
        )
        return out
    return out


# ---------------------------------------------------------------- HTML
# Outlook desktop renders mail with Word's engine: tables, HTML attributes and inline styles only.

FONT = "font-family:'Segoe UI',Arial,sans-serif;font-size:14px;color:#1f2328"
TH = "background-color:#1f4e79;color:#ffffff;text-align:left;padding:4px 8px;border:1px solid #9bb7d4"
TD = "padding:4px 8px;border:1px solid #9bb7d4"
RED, GREEN, AMBER = "#b00020", "#1e7e34", "#9a6700"


def a(url, text):
    return f"<a href='{html.escape(url)}'>{html.escape(text)}</a>"


def cell(value, color=None):
    v = f"{value:,}" if isinstance(value, int) else html.escape(str(value))
    style = TD + (f";color:{color};font-weight:bold" if color else "")
    return f"<td style='{style}'>{v}</td>"


def table(head, rows):
    """rows: lists of cells, each a value or a (value, color) pair."""
    out = [
        "<table cellpadding='0' cellspacing='0' style='border-collapse:collapse'>",
        "<tr>"
        + "".join(f"<th style='{TH}'>{html.escape(h)}</th>" for h in head)
        + "</tr>",
    ]
    for r in rows:
        out.append(
            "<tr>"
            + "".join(cell(*c) if isinstance(c, tuple) else cell(c) for c in r)
            + "</tr>"
        )
    return "\n".join(out + ["</table>"])


def names_list(title, tests, color):
    if not tests:
        return ""
    items = "".join(
        f"<li><code title='{html.escape(t)}'>{html.escape(short(t))}</code></li>"
        for t in tests[:20]
    )
    more = f"<li>… and {len(tests) - 20:,} more</li>" if len(tests) > 20 else ""
    return f"<p style='margin:8px 0 2px'><b style='color:{color}'>{html.escape(title)}</b></p><ul style='margin:0'>{items}{more}</ul>"


def build(results, steps, previous):
    env = os.environ
    now = datetime.datetime.now(datetime.timezone.utc)
    repo, sha_full = env.get("GITHUB_REPOSITORY", ""), env.get("GITHUB_SHA", "")
    server, branch, sha = (
        env.get("GITHUB_SERVER_URL", "https://github.com"),
        env.get("GITHUB_REF_NAME", ""),
        sha_full[:6],
    )
    kind = "Nightly" if env.get("GITHUB_EVENT_NAME") == "schedule" else "Manual"
    run_url = f"{server}/{repo}/actions/runs/{env.get('GITHUB_RUN_ID', '')}"
    cur = suites(results)

    step_list = [tuple(x.strip().split("=", 1)) for x in steps.split(",") if "=" in x]
    # "Test job=failure" only repeats what the failing steps or tests already say.
    bad_steps = [
        (n, o)
        for n, o in step_list
        if o not in ("success", "skipped")
        and not (n == "Test job" and o == "failure" and len(step_list) > 1)
    ]
    failed = sum((c or {}).get("Failed", 0) for c, _ in cur.values())
    ran = [k for k, (c, _) in cur.items() if c]
    ok = not bad_steps and not failed and ran

    # Comparison with the previous nightly.
    prev = previous or {"status": "none", "missing": []}
    diff = {}
    if prev["status"] == "ok":
        old = suites(prev["dir"])
        for k in cur:
            if cur[k][0] and old[k][0]:
                diff[k] = compare(cur[k][1], old[k][1])
    new_total = sum(len(d["New"]) for d in diff.values())

    # Banner: a one-cell table with bgcolor survives Outlook and dark mode better than a styled <div>.
    if ok:
        head = "✅ PASSED"
    elif failed:
        head = f"❌ FAILED: {failed:,} failing test{'s' if failed != 1 else ''}" + (
            f" ({new_total:,} new)" if diff else ""
        )
    else:
        head = "❌ FAILED: " + (
            ", ".join(f"{n} {o}" for n, o in bad_steps) or "no tests ran"
        )
    parts = []
    for k, (c, _) in cur.items():
        if c:
            parts.append(
                f"{k}: {c['Passed']:,} passed"
                + (f", {c['Failed']:,} failed" if c["Failed"] else "")
            )
        else:
            parts.append(f"{k} not run")
    subject = f"{'✅' if ok else '❌'} gfx1250-strict Test Results - {kind} - {now:%Y-%m-%d} | Branch: {branch} | Commit: {sha}"
    banner = (
        f"<table width='100%' cellpadding='10' cellspacing='0'><tr><td bgcolor='{GREEN if ok else RED}' "
        f"style='color:#ffffff;font-size:15px'><b>{html.escape(head)}</b><br>{'<br>'.join(html.escape(x) for x in parts)}"
        "</td></tr></table>"
    )

    p = [
        "<h3 style='margin:0 0 4px'>Summary</h3>",
        banner,
        "<h2 style='margin:16px 0 4px'>Test Results Report</h2>",
        "<h3 style='margin:12px 0 4px'>Test Metadata</h3>",
    ]
    p.append(
        "<p style='margin:4px 0'>"
        + a(run_url, f"{kind} CI logs" if kind == "Nightly" else "CI logs")
        + "</p>"
    )
    steps_text = (
        f"✅ all {len(step_list)} steps passed"
        if step_list and not bad_steps
        else " · ".join(f"❌ {n} {o}" for n, o in bad_steps) or "-"
    )
    p.append(
        table(
            ("Field", "Value"),
            [
                *([("System", env["SYSTEM_LABEL"])] if env.get("SYSTEM_LABEL") else []),
                # In the report job, RUNNER_NAME is the GitHub-hosted runner; the test job passes its own.
                (
                    "Runner",
                    (
                        (env.get("TEST_RUNNER_NAME") or "- (the test job didn't start)")
                        if "TEST_RUNNER_NAME" in env
                        else env.get("RUNNER_NAME", "")
                    ),
                ),
                ("Architecture", "gfx1250-strict"),
                ("Branch", branch),
                ("Commit ID", sha),
                ("ROCm SDK", env.get("ROCM_SDK_VERSION", "")),
                ("Steps", (steps_text, None if not bad_steps else RED)),
            ],
        )
        .replace(
            f">{html.escape(sha)}</td>",
            f">{a(f'{server}/{repo}/commit/{sha_full}', sha)}</td>",
            1,
        )
        .replace(
            f">{html.escape(branch)}</td>",
            f">{a(f'{server}/{repo}/tree/{branch}', branch)}</td>",
            1,
        )
    )

    for title, keys in (
        ("hipBLASLt Tests", ("hipBLASLt",)),
        ("TensileLite Tests", ("TensileLite Common", "Rocisa")),
    ):
        p.append(f"<h3 style='margin:16px 0 4px'>{title}</h3>")
        for k in keys:
            if len(keys) > 1:
                p.append(
                    f"<h4 style='margin:10px 0 4px'>{k.replace('TensileLite ', '')} Tests</h4>"
                )
            c, status = cur[k]
            if not c:
                p.append("<p>Not run.</p>")
                continue
            p.append(
                table(
                    ("Status", "Count"),
                    [
                        (n, (v, RED if n == "Failed" and v else None))
                        for n, v in c.items()
                    ],
                )
            )
            p.append(
                names_list(
                    "Failing:", sorted(t for t, s in status.items() if s == FAIL), RED
                )
            )

    p.append("<h3 style='margin:16px 0 4px'>Failure Comparison</h3>")
    if prev["status"] == "ok":
        days = round(prev.get("hours", 24) / 24)
        ref = f"the nightly of {prev['date']}, commit {prev['sha']}"
        line = f"Compared with {html.escape(ref)} ({a(prev['url'], 'run')})."
        if prev.get("hours", 0) > LAST_NIGHT_HOURS or prev["missing"]:
            gap = (
                "; ".join(f"{d} ({a(u, 'run')})" for d, u in prev["missing"])
                or "none ran"
            )
            line = (
                f"<span style='color:{AMBER}'><b>⚠ The latest nightly with results was {days} night{'s' if days != 1 else ''} "
                f"back.</b></span> Compared with {html.escape(ref)} ({a(prev['url'], 'run')}). Nightlies since then "
                f"with no results: {gap}. New failures may come from any change since {html.escape(prev['date'])}."
            )
        p.append(f"<p>{line}</p>")
        rows, details = [], []
        for k in cur:
            if k not in diff:
                why = (
                    "not run in this run"
                    if not cur[k][0]
                    else "no results in that nightly"
                )
                rows.append((k, "-", "-", "-", f"- ({why})"))
                continue
            d = diff[k]
            rows.append(
                (
                    k,
                    (len(d["New"]), RED if d["New"] else None),
                    len(d["Existing"]),
                    (len(d["Fixed"]), GREEN if d["Fixed"] else None),
                    (
                        len(d["Skipped/removed"]),
                        AMBER if d["Skipped/removed"] else None,
                    ),
                )
            )
            details += [
                names_list(f"{k}, new failures:", d["New"], RED),
                names_list(f"{k}, fixed (now passing):", d["Fixed"], GREEN),
                names_list(
                    f"{k}, skipped or removed since:", d["Skipped/removed"], AMBER
                ),
            ]
        p.append(
            table(
                (
                    "Category",
                    "New Failures",
                    "Existing Failures",
                    "Fixed",
                    "Skipped/removed",
                ),
                rows,
            )
        )
        p.extend(x for x in details if x)
    elif prev.get("missing"):
        gap = "; ".join(f"{d} ({a(u, 'run')})" for d, u in prev["missing"])
        p.append(
            f"<p>No nightly to compare: none with results in the last {LOOKBACK_DAYS} days ({gap} left none).</p>"
        )
    else:
        p.append("<p>No nightly to compare.</p>")
    p.append(
        f"<p style='color:#57606a;font-size:12px'>Generated on {now:%Y-%m-%d %H:%M:%S} UTC</p>"
    )
    body = (
        f"<table width='640' cellpadding='0' cellspacing='0'><tr><td style=\"{FONT}\">"
        + "\n".join(p)
        + "</td></tr></table>"
    )

    # Plain-text version, for clients and flows that don't show HTML.
    text = [subject, head, *parts, f"Run: {run_url}", f"Steps: {steps_text}"]
    for k, d in diff.items():
        text += [
            f"{k}: {len(d['New'])} new, {len(d['Existing'])} existing, {len(d['Fixed'])} fixed, "
            f"{len(d['Skipped/removed'])} skipped or removed"
        ]
        text += [f"  new: {t}" for t in d["New"][:10]]
    # Per-test changes for whatever files tickets (e.g. the flow behind the webhook): test names only.
    tests = []
    for k, (c, status) in cur.items():
        d = diff.get(k)
        for t in sorted(x for x, st in status.items() if st == FAIL):
            tests.append(
                {
                    "suite": k,
                    "test": t,
                    "status": "failing",
                    "new": (t in d["New"]) if d else None,
                }
            )
        for kind_, key in (
            ("fixed", "Fixed"),
            ("skipped_or_removed", "Skipped/removed"),
        ):
            tests += [
                {"suite": k, "test": t, "status": kind_} for t in (d or {}).get(key, [])
            ]
    return {
        "subject": subject,
        "html": body,
        "text": "\n".join(text),
        "ok": bool(ok),
        "run_url": run_url,
        "commit": sha_full,
        "branch": branch,
        "tests": tests,
    }


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--results",
        required=True,
        help="Directory with test_results/ and tensilelite_test_results/",
    )
    ap.add_argument(
        "--steps", default="", help="Step outcomes, e.g. 'SDK=success, Build=failure'"
    )
    ap.add_argument("--webhook-url", default="", help="Where to POST the report")
    ap.add_argument(
        "--previous",
        default="",
        help="Results directory of a previous run (default: fetch the previous nightly)",
    )
    ap.add_argument(
        "--previous-run-id",
        default="",
        help="With --previous: the run those results came from",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print the report instead of sending it"
    )
    ap.add_argument(
        "--html-out", default="", help="Also write the HTML body to this file"
    )
    args = ap.parse_args()

    env, now = os.environ, datetime.datetime.now(datetime.timezone.utc)
    previous = {"status": "none", "missing": []}
    try:
        if args.previous:
            info = (
                run_info(
                    env.get("GITHUB_REPOSITORY"),
                    args.previous_run_id,
                    env.get("GH_TOKEN"),
                )
                if args.previous_run_id
                else {"date": "an earlier run", "url": "", "sha": "?"}
            )
            previous = dict(info, status="ok", dir=args.previous, hours=24, missing=[])
        else:
            previous = fetch_previous(
                env.get("GITHUB_REPOSITORY"),
                env.get("GITHUB_RUN_ID"),
                env.get("GH_TOKEN"),
                now,
            )
    except Exception as e:  # The report still goes out, without the comparison.
        print(
            f"::warning::Could not get the previous nightly's results ({type(e).__name__}: {e})"
        )
    report = build(args.results, args.steps, previous)
    if args.html_out:
        open(args.html_out, "w").write(report["html"])
    # CI logs are public: only --dry-run prints the report's content.
    if args.dry_run:
        print(report["text"])
        return 0
    if not args.webhook_url:
        print("::notice::No webhook URL; report not sent")
        return 0
    req = urllib.request.Request(
        args.webhook_url,
        data=json.dumps(report).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        print(f"Report sent (HTTP {r.status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
