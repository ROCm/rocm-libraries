#!/usr/bin/env python3
"""Generate a hipBLASLt CI digest for a PR and upsert it as a sticky comment.

A compact, CDash-inspired summary of the hipBLASLt-relevant CI for a PR:
one row per build (platform x arch) with deep links to the configure/build
logs (computed from the deterministic TheRock S3 layout) and the hipBLASLt
test job, plus a green/red headline and a collapsed list of every hipBLASLt
check. See docs/ci-summary-cleanup/cdash-feasibility.md for the design.

v1 scope: hipBLASLt only; the native per-check list is collapsed by default;
no known-issue classification; test Run/Fail/Pass counts are deferred (the
Test column links the job log and shows its pass/fail). Configure/build/install
logs live in S3; test logs live in the GitHub job log.

Data comes from the GitHub API via the `gh` CLI (no extra Python deps; `gh`
picks up GH_TOKEN/GITHUB_TOKEN in Actions). S3 URLs are computed, not scraped.

Usage:
    python pr_ci_dashboard.py --repo ROCm/rocm-libraries --pr 8796 --dry-run
    python pr_ci_dashboard.py --repo "$REPO" --sha "$HEAD_SHA"   # resolves PR
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

COMMENT_MARKER = "<!-- hipblaslt-ci-digest -->"

# S3 layout, mirrored from TheRock build_tools/_therock_utils/workflow_outputs.py.
# rocm-libraries (repo != ROCm/TheRock) uses the external bucket + owner-repo prefix.
S3_HOST = "https://therock-ci-artifacts-external.s3.amazonaws.com"
S3_OWNER_REPO = "ROCm-rocm-libraries"

# CMake project name (S3 log filename casing) for the component this digest tracks.
HIPBLASLT_LOG = "hipBLASLt"

# Name of the workflow whose matrix builds feed the table.
THEROCK_WORKFLOW = "TheRock CI"

# A check run is hipBLASLt-relevant if its name matches this, or is always-included.
HIPBLASLT_RE = re.compile(r"hipblaslt|host_asan|\basan\b", re.IGNORECASE)
ALWAYS_INCLUDE = {"TheRock CI Summary", "Math CI Summary", "pre-commit"}

# Matrix job name, e.g.
#   "Linux (tensilelite,rocblas,hipblas,hipsparselt,hipblaslt | gfx94X-dcgpu) / Build (gfx94X-dcgpu)"
#   "... | gfx94X-dcgpu) / Test / hipblaslt / shard 1 of 6"   (post-rename)
#   "... | gfx94X-dcgpu) / Test (gfx94X-dcgpu) / Test hipblaslt / Test hipblaslt (shard 1 of 6)"
ROOT_RE = re.compile(
    r"^(?P<platform>Linux|Windows)\s*\((?P<proj>[^|]+?)\s*\|\s*"
    r"(?P<arch>[^)]+?)\)\s*/\s*(?P<rest>.*)$"
)


def gh(*args: str) -> str:
    return subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=True
    ).stdout


def gh_json(*args: str):
    return json.loads(gh(*args))


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def fmt_duration(start: Optional[datetime], end: Optional[datetime]) -> str:
    if not start or not end:
        return "—"
    secs = int((end - start).total_seconds())
    if secs < 0:
        return "—"
    h, rem = divmod(secs, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m}m" if h else f"{m}m"


def emoji(status: str) -> str:
    return {
        "success": "🟢",
        "failure": "🔴",
        "cancelled": "🟡",
        "timed_out": "🔴",
        "skipped": "⚪",
        "neutral": "⚪",
        "pending": "⏳",
        "in_progress": "⏳",
        "queued": "⏳",
    }.get(status, "⚪")


# Worst-wins ordering for aggregating shard statuses into one test verdict.
_RANK = {"failure": 4, "timed_out": 4, "cancelled": 3, "in_progress": 2,
         "queued": 2, "pending": 2, "success": 1, "skipped": 0, "neutral": 0}


@dataclass
class Row:
    platform: str
    arch: str
    run_id: str
    proj: str = ""
    build_status: str = "pending"
    build_url: str = ""
    test_status: Optional[str] = None
    test_url: str = ""
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    def add_test(self, status: str, completed: Optional[datetime], url: str) -> None:
        # Note: test jobs can start hours after the build (GPU-runner queue), so
        # the Time column reports build duration only (start..end below), not the
        # build-start..last-test-finish span which would be dominated by queue idle.
        if self.test_status is None or _RANK.get(status, 0) > _RANK.get(self.test_status, 0):
            self.test_status = status
        if url and not self.test_url:
            self.test_url = url

    @property
    def s3_dir(self) -> str:
        return f"{S3_HOST}/{S3_OWNER_REPO}/{self.run_id}-{self.platform.lower()}/logs/{self.arch}"

    def log(self, phase: str) -> str:
        return f"{self.s3_dir}/{HIPBLASLT_LOG}_{phase}.log"

    @property
    def index(self) -> str:
        return f"{self.s3_dir}/index.html"


def list_runs_for_sha(repo: str, sha: str) -> list[dict]:
    data = gh_json(
        "api", "--paginate",
        f"/repos/{repo}/actions/runs?head_sha={sha}&event=pull_request&per_page=100",
    )
    return data.get("workflow_runs", [])


def latest_run(runs: list[dict], workflow_name: str) -> Optional[dict]:
    matching = [r for r in runs if r.get("name") == workflow_name]
    return max(matching, key=lambda r: r.get("created_at", "")) if matching else None


def list_jobs(repo: str, run_id: str) -> list[dict]:
    data = gh_json(
        "api", "--paginate",
        f"/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
    )
    return data.get("jobs", [])


def check_runs(repo: str, sha: str) -> list[dict]:
    data = gh_json(
        "api", "--paginate",
        f"/repos/{repo}/commits/{sha}/check-runs?per_page=100",
    )
    return data.get("check_runs", [])


def resolve_pr(repo: str, sha: str) -> Optional[str]:
    try:
        pulls = gh_json("api", f"/repos/{repo}/commits/{sha}/pulls")
    except subprocess.CalledProcessError:
        return None
    open_pulls = [p for p in pulls if p.get("state") == "open"] or pulls
    return str(open_pulls[0]["number"]) if open_pulls else None


def collect_matrix(repo: str, run: Optional[dict]) -> list[Row]:
    """Build one Row per (platform, arch) from the latest TheRock CI run's jobs."""
    if not run:
        return []
    rows: dict[tuple, Row] = {}
    for job in list_jobs(repo, str(run["id"])):
        m = ROOT_RE.match(job.get("name", ""))
        if not m:
            continue
        gd = m.groupdict()
        rest = gd["rest"]
        # only matrices that actually build hipBLASLt
        if "hipblaslt" not in gd["proj"].lower():
            continue
        key = (gd["platform"], gd["arch"])
        row = rows.get(key) or Row(
            platform=gd["platform"], arch=gd["arch"],
            run_id=str(run["id"]), proj=gd["proj"].strip(),
        )
        rows[key] = row
        status = job.get("conclusion") or job.get("status") or "pending"
        if rest.startswith("Build"):
            row.build_status = status
            row.build_url = job.get("html_url", "")
            row.start = parse_dt(job.get("started_at"))
            row.end = parse_dt(job.get("completed_at"))
        elif "hipblaslt" in rest.lower():
            row.add_test(status, parse_dt(job.get("completed_at")), job.get("html_url", ""))
    return sorted(rows.values(), key=lambda r: (r.platform, r.arch))


def render(sha: str, rows: list[Row], checks: list[dict]) -> str:
    relevant = [
        c for c in checks
        if c.get("name") in ALWAYS_INCLUDE or HIPBLASLT_RE.search(c.get("name", ""))
    ]

    def concl(c: dict) -> str:
        return c.get("conclusion") or c.get("status") or "pending"

    passed = sum(1 for c in relevant if concl(c) == "success")
    failed = sum(1 for c in relevant if concl(c) in ("failure", "timed_out", "cancelled"))
    other = len(relevant) - passed - failed
    verdict = "🔴" if failed else ("🟢" if passed and not other else "⏳")
    headline = (
        f"**{passed} passed · {failed} failed · {other} pending/skipped** "
        f"({len(relevant)} hipBLASLt checks)"
    )

    out = [
        COMMENT_MARKER,
        f"## {verdict} hipBLASLt CI digest — `{sha[:7]}`",
        "",
        headline,
        "",
        "| Build (platform · arch) | Configure | Build | hipBLASLt tests | Build time |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        bad = r.build_status in ("failure", "timed_out") or r.test_status in ("failure", "timed_out")
        overall = "🔴" if bad else emoji(r.build_status)
        name = f"{overall} {r.platform} · {r.arch}"
        if r.build_url:
            name = f"[{name}]({r.build_url})"
        cfg = f"[log]({r.log('configure')})"
        bld = f"{emoji(r.build_status)} [log]({r.log('build')})"
        if r.test_status and r.test_url:
            tests = f"{emoji(r.test_status)} [job]({r.test_url})"
        elif r.test_status:
            tests = emoji(r.test_status)
        else:
            tests = "—"
        out.append(
            f"| {name} | {cfg} | {bld} | {tests} | {fmt_duration(r.start, r.end)} |"
        )

    index_link = rows[0].index if rows else S3_HOST
    out += [
        "",
        f"<sub>Configure / Build / Install → [S3 artifacts]({index_link}); "
        "Test → GitHub job log. Counts are job-level pass/fail "
        "(per-test counts are a follow-up).</sub>",
        "",
    ]

    # native per-check list, collapsed by default
    out.append(f"<details><summary>All hipBLASLt checks ({len(relevant)})</summary>")
    out.append("")
    out.append("| Check | Status |")
    out.append("| --- | --- |")
    for c in sorted(relevant, key=lambda c: c.get("name", "")):
        n = c.get("name", "")
        url = c.get("details_url") or c.get("html_url") or ""
        cell = f"[{n}]({url})" if url else n
        out.append(f"| {cell} | {emoji(concl(c))} {concl(c)} |")
    out.append("</details>")
    return "\n".join(out)


def upsert_comment(repo: str, pr: str, body: str, dry_run: bool) -> None:
    if dry_run:
        print(body)
        return
    comments = gh_json("api", "--paginate", f"/repos/{repo}/issues/{pr}/comments")
    existing = next((c for c in comments if COMMENT_MARKER in c.get("body", "")), None)
    if existing:
        gh("api", "--method", "PATCH",
           f"/repos/{repo}/issues/comments/{existing['id']}", "-f", f"body={body}")
    else:
        gh("api", "--method", "POST",
           f"/repos/{repo}/issues/{pr}/comments", "-f", f"body={body}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="owner/repo")
    ap.add_argument("--pr", help="PR number (else resolved from --sha)")
    ap.add_argument("--sha", help="head SHA (resolves the PR if --pr is absent)")
    ap.add_argument("--dry-run", action="store_true", help="print, do not post")
    args = ap.parse_args()

    if not args.pr and not args.sha:
        ap.error("one of --pr or --sha is required")

    if args.pr and not args.sha:
        args.sha = gh_json("api", f"/repos/{args.repo}/pulls/{args.pr}")["head"]["sha"]
    pr = args.pr or resolve_pr(args.repo, args.sha)
    if not pr:
        print(f"No open PR for {args.sha[:7]}; nothing to comment.", file=sys.stderr)
        return 0

    runs = list_runs_for_sha(args.repo, args.sha)
    rows = collect_matrix(args.repo, latest_run(runs, THEROCK_WORKFLOW))
    checks = check_runs(args.repo, args.sha)
    body = render(args.sha, rows, checks)
    upsert_comment(args.repo, pr, body, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
