#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build a test report from a CI run's logs and POST it to a webhook.

Parses gtest and pytest logs under --results, compares each test with the previous
nightly's uploaded results, and POSTs JSON to the webhook (e.g. a Power Automate
flow, which picks the recipients):

  subject, html, text, ok, run_url, commit, branch, steps [{name, conclusion}],
  outcome      passed | setup_failed | tests_failed | tests_incomplete | cancelled
  failed_step  first failed step, if any
  tests        [{suite, test, status, new}]; status failing | timeout | crashed |
               fixed | skipped_or_removed

Suites (--suite LABEL=KIND:PATH, repeatable, display order; PATH under --results):
  gtest:GLOB       gtest logs, e.g. sharded test_results/shard*.log
  pytest:FILE#N    Nth pytest session in FILE (1-based, negative from the end,
                   default 1), e.g. a tox log with two sessions

Usage:
    blas_nightly_report.py --results DIR --title NAME --artifact NAME --suite ... \
        --steps "Test job=success, Build=failure, ..."
    ... --dry-run                                 # print, don't send
    ... --previous PREV_DIR --previous-run-id ID  # compare with results on disk

Environment: GITHUB_* (run metadata), WEBHOOK_URL (default --webhook-url; keeps it
off the command line), GH_TOKEN (actions: read), SYSTEM_LABEL, TEST_RUNNER_NAME.
Standard library only.
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
import urllib.parse
import urllib.request
import zipfile

API = "https://api.github.com"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
PASS, FAIL, SKIP = "pass", "fail", "skip"
STRIP = []  # --strip: substrings removed from test names in the email
TEST_JOB = "Test job"  # the --steps entry for the job itself (job.status)


# ---------------------------------------------------------------- parsing


def parse_gtest(pattern, why):
    """(counts or None if not run, {test: status}) from the gtest logs matching
    pattern; fills why[test] with "timeout" or "crashed" for a test that stopped
    its shard."""
    logs = sorted(glob.glob(pattern))
    status, total = {}, 0
    for path in logs:
        running, alarm = None, False
        with open(path, errors="replace") as f:
            lines = f.readlines()
        for line in lines:
            if m := re.match(r"^\[ RUN      \] (\S+)", line):
                running = m.group(1)
            elif line.startswith(
                "Aborting tests due to an alarm timeout"
            ):  # test client
                alarm = True
            elif m := re.match(r"^\[==========\] (\d+) tests? from .* ran", line):
                total += int(m.group(1))
            elif m := re.match(r"^\[       OK \] (\S+)", line):
                status[m.group(1)] = PASS
            elif m := re.match(r"^\[  SKIPPED \] ([^\s,]+)$", line.rstrip()):
                status[m.group(1)] = SKIP
            elif (m := re.match(r"^\[  FAILED  \] ([^\s,]+)", line)) and not m.group(
                1
            ).isdigit():
                status[m.group(1)] = FAIL
            if running in status:
                running = None
        if running:  # the shard stopped in this test
            status[running], why[running] = FAIL, "timeout" if alarm else "crashed"
    if not total and not status:
        return None, {}
    counts = {
        "Passed": sum(s == PASS for s in status.values()),
        "Failed": sum(s == FAIL for s in status.values()),
        "Skipped": sum(s == SKIP for s in status.values()),
    }
    counts["Total"] = max(total, len(status))  # a stopped shard prints no total
    return counts, status


def parse_pytest(path, index, why):
    """(counts or None if absent, {id: status}) of the index-th pytest session in
    a log (1-based, negative from the end); fills why[id] = "timeout"."""
    if not os.path.isfile(path):
        return None, {}
    word = {
        "PASSED": PASS,
        "FAILED": FAIL,
        "ERROR": FAIL,
        "XPASS": FAIL,
        "SKIPPED": SKIP,
        "XFAIL": SKIP,
    }
    # Sessions are counted by their start line, so one that dies before its summary
    # keeps its place and the later ones keep their numbers. [summary or None, {id: status}]
    sessions = []
    with open(path, errors="replace") as f:
        lines = f.readlines()
    for raw in lines:
        line = ANSI.sub("", raw.rstrip("\n"))
        if re.match(r"^=+ test session starts =+$", line):
            sessions.append([None, {}])
        elif not sessions:
            continue
        elif re.match(r"^=+ .*\b(passed|failed|error|errors|skipped)\b.* =+$", line):
            sessions[-1][0] = line
        elif m := (
            re.search(r"\] (PASSED|FAILED|ERROR|XPASS|SKIPPED|XFAIL) (\S+)", line)
            or re.match(r"^(FAILED|ERROR) (\S+)", line)
        ):
            # A failure line wins over an earlier progress line.
            status = sessions[-1][1]
            if status.get(m.group(2)) != FAIL:
                status[m.group(2)] = word[m.group(1)]
            if re.search(r" - .*\bTimeout >", line):  # pytest-timeout
                why[m.group(2)] = "timeout"
    try:
        line, status = sessions[index - 1 if index > 0 else index]
    except IndexError:  # that session never started
        return None, {}
    if line is None:  # it started but never finished
        return None, {}
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
    return counts, status


def parse_suite_spec(spec):
    """'LABEL=gtest:GLOB' or 'LABEL=pytest:FILE[#N]' -> (label, kind, path, index)."""
    label, sep, rest = spec.partition("=")
    kind, sep2, path = rest.partition(":")
    if not (sep and sep2 and label and path and kind in ("gtest", "pytest")):
        sys.exit(
            f"bad --suite {spec!r}: want LABEL=gtest:GLOB or LABEL=pytest:FILE[#N]"
        )
    path, _, index = path.partition("#")
    return label.strip(), kind, path, int(index or 1)


def suites(results, specs, why=None):
    """{suite label: (counts or None, {test: status})}, in --suite order."""
    why = {} if why is None else why
    out = {}
    for label, kind, path, index in specs:
        full = os.path.join(results, path)
        out[label] = (
            parse_gtest(full, why)
            if kind == "gtest"
            else parse_pytest(full, index, why)
        )
    return out


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
    """A readable test name: drop the --strip substrings."""
    for part in STRIP:
        test = test.replace(part, "")
    return test


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
    # The zip URL redirects to signed storage that rejects the token: follow it without auth.
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


def _when(stamp):
    return datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def fetch_previous(repo, run_id, token, cfg):
    """Download the most recent earlier nightly of this run's workflow and branch whose
    cfg.artifact is still there, within cfg.lookback_days.

    Returns {"status": "ok" | "none", "date", "url", "sha", "dir", "hours", "missing": [(date, url)]}.
    "missing" lists the newer nightlies that left no results (crashed, cancelled or expired).
    """
    out = {"status": "none", "missing": []}
    if not (repo and token and run_id and cfg.artifact):
        return out
    this = json.load(_api(f"{API}/repos/{repo}/actions/runs/{run_id}", token))
    workflow = this["path"].split("/")[-1].split("@")[0]
    # Age from this run's start, not from now: the report can go out hours later.
    start = _when(this["created_at"])
    runs = json.load(
        _api(
            f"{API}/repos/{repo}/actions/workflows/{workflow}/runs"
            f"?event=schedule&branch={urllib.parse.quote(os.environ.get('GITHUB_REF_NAME', ''))}"
            "&status=completed&per_page=20",
            token,
        )
    )
    for run in runs.get("workflow_runs", []):
        started = _when(run["created_at"])
        if str(run["id"]) == str(run_id) or started >= start:
            continue  # this run, or a later one
        hours = (start - started).total_seconds() / 3600
        if hours > cfg.lookback_days * 24:
            break
        arts = json.load(
            _api(f"{API}/repos/{repo}/actions/runs/{run['id']}/artifacts", token)
        )
        art = next(
            (
                a
                for a in arts.get("artifacts", [])
                if a["name"] == cfg.artifact and not a["expired"]
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


def names_list(suite, what, tests, color, tags=None):
    """'<suite>:' / '(<what>)' then one test per line; tags: {test: [label, ...]}."""
    if not tests:
        return ""
    # One test per line: <br>, not <ul>, which some mail views run together.
    items = [
        f"• <code title='{html.escape(t)}'>{html.escape(short(t))}</code>"
        + "".join(f" <b>{html.escape(x)}</b>" for x in (tags or {}).get(t, []))
        for t in tests[:20]
    ]
    if len(tests) > 20:
        items.append(f"… and {len(tests) - 20:,} more")
    return (
        f"<p style='margin:8px 0 2px'><b style='color:{color}'>{html.escape(suite)}:</b>"
        f"<br>({html.escape(what)})</p>"
        f"<p style='margin:0 0 0 12px'>{'<br>'.join(items)}</p>"
    )


def build(results, steps, previous, cfg):
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
    why = {}  # failing test -> timeout or crashed
    cur = suites(results, cfg.suites, why)

    # A step that never started has an empty outcome: count it as skipped.
    step_list = [
        (n.strip(), o.strip() or "skipped")
        for n, _, o in (x.partition("=") for x in steps.split(","))
        if n.strip()
    ]
    # A skipped step is fine, a skipped test job isn't. A failed test job only repeats
    # what the failing steps or tests already say.
    bad_steps = [
        (n, o)
        for n, o in step_list
        if (o not in ("success", "skipped") or (n == TEST_JOB and o == "skipped"))
        and not (n == TEST_JOB and o == "failure" and len(step_list) > 1)
    ]
    failed = sum((c or {}).get("Failed", 0) for c, _ in cur.values())
    # Every --suite is expected in a nightly: one that didn't run fails the report.
    not_run = [k for k, (c, _) in cur.items() if not c]
    ok = not bad_steps and not failed and not not_run

    # Comparison with the previous nightly.
    prev = previous or {"status": "none", "missing": []}
    diff = {}
    if prev["status"] == "ok":
        old = suites(prev["dir"], cfg.suites)
        for k in cur:
            if cur[k][0] and old[k][0]:
                diff[k] = compare(cur[k][1], old[k][1])
    new_total = sum(len(d["New"]) for d in diff.values())
    # For flows that route by kind of failure.
    failed_step = next((n for n, o in bad_steps if n != TEST_JOB), None)
    if ok:
        outcome = "passed"
    elif (TEST_JOB, "cancelled") in step_list:
        outcome = "cancelled"
    elif why:
        outcome = "tests_incomplete"  # a test timed out or crashed
    elif failed:
        outcome = "tests_failed"
    else:
        outcome = "setup_failed"

    # Banner: a one-cell bgcolor table survives Outlook and dark mode; a styled <div> doesn't.
    if ok:
        head = "✅ PASSED"
    elif failed:
        kinds = [
            f"{n} {k}"
            for k in ("timeout", "crashed")
            if (n := sum(w == k for w in why.values()))
        ]
        head = f"❌ FAILED: {failed:,} failing test{'s' if failed != 1 else ''}" + (
            f" ({', '.join(([f'{new_total:,} new'] if diff else []) + kinds)})"
            if diff or kinds
            else ""
        )
    else:
        head = "❌ FAILED: " + ", ".join(
            [f"{n} {o}" for n, o in bad_steps] + [f"{k} not run" for k in not_run]
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
    subject = f"{'✅' if ok else '❌'} {cfg.title} Test Results - {kind} - {now:%Y-%m-%d} | Branch: {branch} | Commit: {sha}"
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
                ("Runner", env.get("TEST_RUNNER_NAME") or env.get("RUNNER_NAME", "-")),
                ("Branch", branch),
                ("Commit ID", sha),
                *cfg.meta_rows,
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

    for k in cur:
        p.append(f"<h3 style='margin:16px 0 4px'>{html.escape(k)} Tests</h3>")
        c, status = cur[k]
        if not c:
            p.append("<p>Not run.</p>")
            continue
        p.append(
            table(
                ("Status", "Count"),
                [(n, (v, RED if n == "Failed" and v else None)) for n, v in c.items()],
            )
        )

    # All test lists go here, at the end.
    p.append("<h3 style='margin:16px 0 4px'>Failures</h3>")
    lists = []
    for k, (_, status) in cur.items():
        new = set(diff[k]["New"]) if k in diff else set()
        failing = sorted(
            (t for t, s in status.items() if s == FAIL), key=lambda t: (t not in new, t)
        )
        tags = {
            t: [x for x in (why.get(t), "new" if t in new else None) if x]
            for t in failing
        }
        lists.append(names_list(k, "failing", failing, RED, tags))
    if prev["status"] == "ok":
        days = round(prev.get("hours", 24) / 24)
        ref = f"the nightly of {prev['date']}, commit {prev['sha']}"
        line = f"Compared with {html.escape(ref)} ({a(prev['url'], 'run')})."
        if prev.get("hours", 0) > cfg.last_night_hours or prev["missing"]:
            line = "<br>".join(
                [
                    f"<span style='color:{AMBER}'><b>⚠ The latest nightly with results was "
                    f"{days} night{'s' if days != 1 else ''} back.</b></span>",
                    f"Compared with {html.escape(ref)} ({a(prev['url'], 'run')}).",
                    f"New failures may come from any change since {html.escape(prev['date'])}.",
                ]
            )
        p.append(f"<p>{line}</p>")
        rows, details = [], []
        for k in cur:
            if k not in diff:
                gone = (
                    "not run in this run"
                    if not cur[k][0]
                    else "no results in that nightly"
                )
                rows.append((k, "-", "-", "-", f"- ({gone})"))
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
                names_list(k, "fixed, now passing", d["Fixed"], GREEN),
                names_list(k, "skipped or removed since", d["Skipped/removed"], AMBER),
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
        lists += details
    elif prev.get("missing"):
        p.append(
            f"<p>No nightly with results in the last {cfg.lookback_days} days to compare.</p>"
        )
    else:
        p.append("<p>No nightly to compare.</p>")
    p.extend(x for x in lists if x)
    p.append(
        f"<p style='color:#57606a;font-size:12px'>Generated on {now:%Y-%m-%d %H:%M:%S} UTC</p>"
    )
    body = (
        f"<table width='640' cellpadding='0' cellspacing='0'><tr><td style=\"{FONT}\">"
        + "\n".join(p)
        + "</td></tr></table>"
    )

    # Plain-text version, for clients and flows that don't show HTML.
    text = [subject, head, *parts]
    text += [f"Run: {run_url}", f"Steps: {steps_text}"]
    for k, d in diff.items():
        text += [
            f"{k}: {len(d['New'])} new, {len(d['Existing'])} existing, {len(d['Fixed'])} fixed, "
            f"{len(d['Skipped/removed'])} skipped or removed"
        ]
        text += [f"  new: {t}" for t in d["New"][:10]]
    # Per-test changes, for the flow (e.g. to file tickets).
    tests = []
    for k, (_, status) in cur.items():
        d = diff.get(k)
        for t in sorted(x for x, st in status.items() if st == FAIL):
            tests.append(
                {
                    "suite": k,
                    "test": t,
                    "status": why.get(t, "failing"),
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
        "outcome": outcome,
        "failed_step": failed_step,
        "steps": [{"name": n, "conclusion": o} for n, o in step_list],
    }


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--results", required=True, help="Directory with the run's test logs"
    )
    ap.add_argument("--title", required=True, help="Report name, e.g. gfx1250-strict")
    ap.add_argument(
        "--suite",
        action="append",
        required=True,
        help="LABEL=gtest:GLOB or LABEL=pytest:FILE[#N] (repeatable, in display order)",
    )
    ap.add_argument(
        "--meta",
        action="append",
        default=[],
        help="LABEL=VALUE: an extra metadata row (repeatable)",
    )
    ap.add_argument(
        "--strip",
        action="append",
        default=[],
        help="Substring to drop from test names in the email (repeatable)",
    )
    ap.add_argument(
        "--artifact", default="", help="Results artifact name, for the previous nightly"
    )
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Compare with the latest nightly that has results within this many days",
    )
    ap.add_argument(
        "--last-night-hours",
        type=int,
        default=30,
        help="Warn when the compared nightly started longer ago than this",
    )
    ap.add_argument(
        "--steps", default="", help="Step outcomes, e.g. 'SDK=success, Build=failure'"
    )
    ap.add_argument(
        "--webhook-url",
        default=os.environ.get("WEBHOOK_URL", ""),
        help="Where to POST the report (default: $WEBHOOK_URL)",
    )
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
    args.suites = [parse_suite_spec(x) for x in args.suite]
    args.meta_rows = []
    for spec in args.meta:
        label, sep, value = spec.partition("=")
        if not (sep and label):
            sys.exit(f"bad --meta {spec!r}: want LABEL=VALUE")
        args.meta_rows.append((label.strip(), value))
    STRIP[:] = args.strip

    env = os.environ
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
                args,
            )
    except Exception as e:  # The report still goes out, without the comparison.
        print(
            f"::warning::Could not get the previous nightly's results ({type(e).__name__}: {e})"
        )
    report = build(args.results, args.steps, previous, args)
    if args.html_out:
        with open(args.html_out, "w") as f:
            f.write(report["html"])
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
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            print(f"Report sent (HTTP {r.status})")
    except urllib.error.HTTPError as e:
        sys.exit(f"::error::Report not sent (HTTP {e.code})")
    except urllib.error.URLError as e:
        sys.exit(f"::error::Report not sent ({type(e.reason).__name__})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
