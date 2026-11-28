# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fetch one PR's review context into Markdown files for a read-only worker.

Usage: python3 fetch_pr.py <PR number> <out dir>
Environment (set by the monitor's worker launch, lessons L-513): REVIEW_ROUND (First-round or Follow-up) and
REVIEW_LINE1 (the review's exact first line) go into meta.md as `- Round:` and `- Line 1:`. The repository, the
operator's login (config `operator.github_login`: their reviews and comments are left out, and meta.md names it as `- Operator:`) and
the clone come from the job's RKMON_* environment (rkenv.py).

Read-only: only `gh pr view`, `gh pr diff`, `gh pr checks` and GraphQL queries, each under the read login (the
job's GH_CONFIG_DIR, which rkmon_lib/gh.py job_env points at it).
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rkenv  # noqa: E402

REPO = rkenv.get("RKMON_REPO")
ME = rkenv.get("RKMON_OPERATOR_LOGIN")
OWNER, NAME = REPO.split("/", 1)
N, OUT = sys.argv[1], pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)


def fetch_locked(cmd: list, tries: int = 6, sleep=time.sleep) -> None:
    """A fetch into the monitor's clone. Every job fetches into it, and the live and shadow drafts stage the same PR at
    once: it holds the clone's lock (steps.clone_lock, as every staging does), and a fetch that still meets another's
    lock file or the shallow-file race waits (5 s, then 10 s, ...) and tries again, `tries` times at most.
    CalledProcessError (its message on stderr) when it fails otherwise or the last time.
    """
    from steps import SHALLOW_RACE, clone_lock

    repo = cmd[cmd.index("-C") + 1] if "-C" in cmd else "."
    with clone_lock(repo):
        for attempt in range(tries):
            f = subprocess.run(cmd, capture_output=True, text=True)
            if (
                f.returncode == 0
                or (".lock" not in f.stderr and SHALLOW_RACE not in f.stderr)
                or attempt == tries - 1
            ):
                break
            sleep(5 * (attempt + 1))
    if f.returncode != 0:
        sys.stderr.write(f.stderr)
        raise subprocess.CalledProcessError(f.returncode, cmd, f.stdout, f.stderr)


# The toolkit's rule (rkmon_lib/gh.py): retry every failure but a JSON 4xx other than 429, which is GitHub refusing the
# request. A transient GitHub or TLS failure (a handshake timeout) would otherwise fail a whole review's staging.
_PERMANENT = re.compile(r"\(HTTP 4(?!29)\d\d\)|\bHTTP 4(?!29)\d\d: ")


def gh(*args, check=True, attempts=3, sleep=time.sleep):
    for i in range(attempts):
        p = subprocess.run(
            [os.environ.get("RKMON_GH") or "gh", *args], capture_output=True, text=True
        )
        if p.returncode == 0 or not check:
            return p.stdout
        if _PERMANENT.search(p.stderr) or i == attempts - 1:
            break
        sleep(5 * (3**i))
    sys.exit(f"gh {' '.join(args[:3])} failed: {p.stderr.strip()}")


fields = (
    "number,title,body,author,labels,state,isDraft,mergeable,headRefName,"
    "headRefOid,baseRefName,createdAt,updatedAt,additions,deletions,"
    "commits,reviews,comments,files"
)
pr = json.loads(gh("pr", "view", N, "--repo", REPO, "--json", fields))
pr["baseRefOid"] = gh("api", f"repos/{REPO}/pulls/{N}", "--jq", ".base.sha").strip()
login = lambda a: (a or {}).get("login", "?")

meta = [
    f"# PR #{pr['number']}: {pr['title']}",
    f"- Author: @{login(pr['author'])}",
    f"- State: {pr['state']}{' (draft)' if pr['isDraft'] else ''}; mergeable: {pr['mergeable']}",
    f"- Head: {pr['headRefName']} @ {pr['headRefOid'][:10]}; base: {pr['baseRefName']} @ {pr['baseRefOid'][:10]}",
    f"- Created {pr['createdAt']}, updated {pr['updatedAt']}; +{pr['additions']}/-{pr['deletions']}",
    f"- Labels: {', '.join(l['name'] for l in pr['labels'])}",
    f"- Operator: @{ME}",
]
# The round this draft is and its exact line 1, from the monitor (rk's review.worker_script): a draft left to guess its
# round names the wrong one (lessons L-513, L-001). A run without them, by hand, stages neither.
meta += [
    f"- {key}: {os.environ[var]}"
    for key, var in (("Round", "REVIEW_ROUND"), ("Line 1", "REVIEW_LINE1"))
    if os.environ.get(var)
]
meta += ["", "## Description", pr["body"] or "(empty)"]
(OUT / "meta.md").write_text("\n".join(meta) + "\n")

lines = ["# Commits (oldest first)"]
for c in pr["commits"]:
    lines.append(f"- {c['oid'][:8]} {c['committedDate']} {c['messageHeadline']}")
(OUT / "commits.md").write_text("\n".join(lines) + "\n")

# "Ours": our own prior reviews are excluded so the worker assesses independently and
# does not parrot them. Other reviewers' comments are kept (the task requires checking
# whether they've been addressed). Ours = the account the monitor posts as (config `operator.github_login`).
OURS = {ME}
skipped_ours = 0

lines = [
    "# Reviews from OTHER reviewers (oldest first; commit = the head the reviewer saw).",
    "# Our own prior reviews are intentionally omitted — assess independently.",
]
for r in pr["reviews"]:
    if login(r["author"]) in OURS:
        skipped_ours += 1
        continue
    commit = (r.get("commit") or {}).get("oid", "")[:8]
    lines += [
        f"\n## @{login(r['author'])} {r['state']} at {commit or '?'} ({r['submittedAt']})",
        r["body"] or "(no body)",
    ]
(OUT / "reviews.md").write_text("\n".join(lines) + "\n")

lines = ["# Conversation comments from OTHERS (oldest first). Our own are omitted."]
for c in pr["comments"]:
    if login(c["author"]) in OURS:
        skipped_ours += 1
        continue
    lines += [f"\n## @{login(c['author'])} ({c['createdAt']})", c["body"]]
(OUT / "comments.md").write_text("\n".join(lines) + "\n")

query = """query($owner:String!,$repo:String!,$n:Int!,$after:String){repository(owner:$owner,name:$repo){
pullRequest(number:$n){reviewThreads(first:100,after:$after){pageInfo{hasNextPage endCursor}
nodes{isResolved isOutdated path line originalLine comments(first:50){nodes{author{login} body createdAt}}}}}}}"""
threads, after = [], None
while True:
    args = [
        "api",
        "graphql",
        "-f",
        f"query={query}",
        "-F",
        f"owner={OWNER}",
        "-F",
        f"repo={NAME}",
        "-F",
        f"n={N}",
    ]
    if after:
        args += ["-F", f"after={after}"]
    page = json.loads(gh(*args))["data"]["repository"]["pullRequest"]["reviewThreads"]
    threads += page["nodes"]
    if not page["pageInfo"]["hasNextPage"]:
        break
    after = page["pageInfo"]["endCursor"]
lines = ["# Inline review threads (our own comments omitted)"]
for t in threads:
    kept = [c for c in t["comments"]["nodes"] if login(c["author"]) not in OURS]
    if not kept:
        skipped_ours += 1
        continue
    status = ("RESOLVED" if t["isResolved"] else "OPEN") + (
        ", outdated" if t["isOutdated"] else ""
    )
    lines.append(f"\n## {t['path']}:{t['line'] or t['originalLine']} [{status}]")
    for c in kept:
        lines += [f"### @{login(c['author'])} ({c['createdAt']})", c["body"]]
(OUT / "threads.md").write_text("\n".join(lines) + "\n")

files = [
    f"{f['additions']:>6} +{f['deletions']:>6} -  {f['path']}" for f in pr["files"]
]
(OUT / "files.txt").write_text("\n".join(files) + "\n")
p = subprocess.run(
    [os.environ.get("RKMON_GH") or "gh", "pr", "diff", N, "--repo", REPO],
    capture_output=True,
    text=True,
)
if p.returncode == 0:
    diff = p.stdout
else:
    # GitHub refuses diffs over 20,000 lines; rebuild it from the monitor's clone (read-only), within the dates the
    # shallow clone keeps (config clone_since).
    R = rkenv.get("RKMON_REPO_DIR")
    since = rkenv.get("RKMON_CLONE_SINCE", required=False)
    base = pr["baseRefName"]
    fetch_locked(
        [
            "git",
            "-C",
            str(R),
            "fetch",
            "-q",
            "--filter=blob:none",
            *([f"--shallow-since={since}"] if since else []),
            "origin",
            f"+refs/pull/{N}/head:refs/remotes/origin/pr{N}",
            f"+refs/heads/{base}:refs/remotes/origin/base{N}",
        ]
    )
    mb = subprocess.run(
        ["git", "-C", str(R), "merge-base", f"origin/base{N}", f"origin/pr{N}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "-C", str(R), "diff", mb, f"origin/pr{N}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    print(
        f"(gh pr diff refused: {p.stderr.strip()[:100]}; rebuilt the diff from the local mirror)"
    )
(OUT / "diff.patch").write_text(diff)
# Every check run at the head with its real conclusion: `gh pr checks` buckets a cancelled job as "fail" (a bot relabel
# cancels a run, and its cancelled jobs are no failures).
runs = (
    gh(
        "api",
        "--paginate",
        f"repos/{REPO}/commits/{pr['headRefOid']}/check-runs?per_page=100",
        "--jq",
        ".check_runs[] | [(.conclusion // .status), .name, .html_url] | @tsv",
        check=False,
    )
    or ""
)
order = {
    "failure": 0,
    "timed_out": 0,
    "action_required": 1,
    "in_progress": 2,
    "queued": 2,
    "cancelled": 3,
    "success": 4,
    "neutral": 5,
    "skipped": 6,
}
# A failed or partial read (a line without tabs) must not crash the worker; say what was dropped.
lines = [r for r in runs.splitlines() if r.strip()]
rows = sorted(
    (r.split("\t") for r in lines if r.count("\t") >= 2),
    key=lambda r: (order.get(r[0], 3), r[1]),
)
bad = [r for r in lines if r.count("\t") < 2]
note = (
    f"(check-run read incomplete: {len(bad)} unreadable line(s), first: {bad[0][:120]})\n"
    if bad
    else ""
)
(OUT / "checks.txt").write_text(
    (
        "conclusion\tcheck\turl (cancelled = superseded or stopped, not a failure)\n"
        + "".join("\t".join(r) + "\n" for r in rows)
        if rows
        else "(no checks)\n"
    )
    + note
)
print(
    f"PR #{N}: {len(pr['reviews'])} reviews, {len(threads)} threads, {len(pr['comments'])} comments, "
    f"{len(pr['commits'])} commits, {len(pr['files'])} files -> {OUT} "
    f"(omitted {skipped_ours} of our own items)"
)
