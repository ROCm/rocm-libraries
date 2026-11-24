#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pin and intake for hipdnn-deep-review (RUNBOOK "Pin and intake" and "Iterate").

  pin.py key <text>
  pin.py init --evidence DIR --scratch DIR --round N --owner-repo O/R
              (--pr N | --branch NAME [--target BR] | --range BASE..HEAD)
              [--clone DIR] [--remote-url URL] [--skill DIR]
              [--reviewer LOGIN] [--since TIME]
  pin.py threads --evidence DIR --round N --owner-repo O/R --pr N
              [--reviewer LOGIN] [--since TIME]
  pin.py freshness --evidence DIR --owner-repo O/R [--pr N] [--reviewer LOGIN]
              [--prev-head SHA] [--prev-pinned TIME] [--prev-target BRANCH]
  pin.py cleanup --evidence DIR --round N

`init` also writes drift.txt: the target tip, whether it merges cleanly with the head, the
commits since BASE that touch the change, and whether the DRIFT lane triggers.

`init` with `--pr` and `threads` save the PR's review threads as r<N>/pr/threads.json and
write r<N>/threads.md and r<N>/reply-affected.txt under --evidence, record thread state in
the ledger's Threads rows and print `unlinked:` and `mismatch:` lines. `freshness` prints
whether the newest round is still current. TIME is `YYYY-MM-DD` (00:00 UTC) or
`YYYY-MM-DDTHH:MM:SSZ`.

Each subcommand prints terse results and exits non-zero with a one-line reason.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger  # noqa: E402
import reviewlib as rl  # noqa: E402

PIN_VARS = "SKILL|EVIDENCE|SCRATCH|CLONE|KEY|TARGET|TARGET_TIP|HEAD|HEAD_SHORT|BASE|REPO|REPO_BASE|KNOWLEDGE"

# Loads PIN.md values as quoted exports; eval-based, so it works in a persistent shell.
ENV_SH = (
    r"""eval "$(awk '/^("""
    + PIN_VARS
    + r"""):/{v=$0; sub(/^[A-Z_]+: */,"",v); k=$1; sub(/:$/,"",k); gsub(/\047/,"\047\\\047\047",v); print "export " k "=\047" v "\047"}' "$(dirname "${BASH_SOURCE[0]}")/PIN.md")"
GREP="$(command -v ggrep || echo /usr/bin/grep)"; export GREP
# fullclone <dir> <sha>: a full checkout of <sha> that never writes to $CLONE. $CLONE is blobless, so the copy keeps its objects
# by alternates and fetches missing blobs from the real origin; an existing <dir> is reused and moved to <sha>.
fullclone() { { [ -d "$1/.git" ] || git clone -q --shared --no-checkout "$CLONE" "$1"; } &&
  git -C "$1" remote set-url origin "$(git -C "$CLONE" remote get-url origin)" &&
  git -C "$1" config remote.origin.promisor true && git -C "$1" config remote.origin.partialclonefilter blob:none &&
  git -C "$1" checkout -q --detach "$2" || { echo "STOP: cannot check out $2 in $1" >&2; return 1; }; }
: "${EVIDENCE:?}" "${SCRATCH:?}" "${CLONE:?}" "${KEY:?}"; cd "$(dirname "${BASH_SOURCE[0]}")"
"""
)

MAX_GH_FILES = 3000


def die(msg):
    sys.exit(f"pin: {msg}")


def git(clone, *args, cwd=None, stdin=None, check=True):
    """Run git (-C clone); return stdout without the trailing newline."""
    code, out, err = rl.git(clone, *args, cwd=cwd, stdin=stdin)
    if check and code != 0:
        lines = err.strip().splitlines()
        die(f"git {args[0]} failed: {lines[-1] if lines else code}")
    return out.rstrip("\n")


def gh(*args):
    """Run gh and return stdout; the only place the GitHub CLI is invoked."""
    code, out, err = rl.run(["gh", *args])
    if code != 0:
        lines = err.strip().splitlines()
        die(f"gh {args[0]} failed: {lines[-1] if lines else code}")
    return out


def gh_paginated_list(endpoint):
    """Concatenate the JSON arrays printed by `gh api --paginate`."""
    text, dec, pos, out = gh("api", "--paginate", endpoint), json.JSONDecoder(), 0, []
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        arr, pos = dec.raw_decode(text, pos)
        out.extend(arr)
    return out


THREADS_QUERY = """
query($owner: String!, $repo: String!, $pr: Int!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $pr) {
      author { login }
      reviewThreads(first: 50, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          isResolved isOutdated path originalLine originalStartLine
          resolvedBy { login }
          comments(first: 100) {
            pageInfo { hasNextPage }
            nodes { author { login } createdAt updatedAt originalCommit { oid } url body }
          }
        }
      }
    }
  }
}
"""
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def parse_utc(text, what):
    """A UTC datetime from `YYYY-MM-DD` (00:00) or `YYYY-MM-DDTHH:MM:SS[Z]`."""
    for fmt in ("%Y-%m-%d", UTC_FORMAT, "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(text.strip(), fmt).replace(
                tzinfo=datetime.timezone.utc
            )
        except ValueError:
            pass
    die(f"{what}: not a UTC time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ): {text}")


def fetch_threads(owner_repo, pr):
    """{author, threads} of a PR from GraphQL `reviewThreads` (all pages). Fails loudly on a
    GraphQL error or a thread with more than 100 comments; there is no REST fallback."""
    owner, _, repo = owner_repo.partition("/")
    threads, author, cursor = [], None, None
    while True:
        args = [
            "api",
            "graphql",
            "-f",
            f"query={THREADS_QUERY}",
            "-f",
            f"owner={owner}",
            "-f",
            f"repo={repo}",
            "-F",
            f"pr={pr}",
        ]
        if cursor:
            args += ["-f", f"cursor={cursor}"]
        doc = json.loads(gh(*args))
        if doc.get("errors"):
            die(f"reviewThreads query failed: {doc['errors'][0].get('message')}")
        node = ((doc.get("data") or {}).get("repository") or {}).get("pullRequest")
        if not node:
            die(f"reviewThreads query returned no pull request {owner_repo}#{pr}")
        author = (node.get("author") or {}).get("login")
        page = node["reviewThreads"]
        for t in page["nodes"]:
            if t["comments"]["pageInfo"]["hasNextPage"]:
                die(f"thread {t['comments']['nodes'][0]['url']} has over 100 comments")
        threads += page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            return {"author": author, "threads": threads}
        cursor = page["pageInfo"]["endCursor"]


def role_of(login, reviewer, author):
    return "review" if login == reviewer else "author" if login == author else "other"


def thread_state(thread):
    return "resolved" if thread["isResolved"] else "unresolved"


def reopen_cutoff(evidence, n, since):
    """The time after which a reply is new to a reopened round n: --since, else round n's
    own Pinned:; fails when round n has neither."""
    if since:
        return parse_utc(since, "--since")
    rdir = rl.round_dir(evidence, n)
    pinned = rl.pin_value(rdir, "Pinned") if (rdir / "PIN.md").is_file() else None
    if not pinned:
        die(f"r{n} has no Pinned: pass --since <time>")
    return parse_utc(pinned, f"r{n} Pinned")


def reply_cutoff(evidence, n, since):
    """The time after which a reply is new to a fresh round n: --since, else the Pinned: of
    the newest round before round n; None when no earlier round exists."""
    if since:
        return parse_utc(since, "--since")
    before = [p for p in rl.round_dirs(evidence) if int(p.name[1:]) < n]
    if not before:
        return None
    pinned = (
        rl.pin_value(before[-1], "Pinned")
        if (before[-1] / "PIN.md").is_file()
        else None
    )
    if not pinned:
        die(f"{before[-1].name} has no Pinned: pass --since <time>")
    return parse_utc(pinned, f"{before[-1].name} Pinned")


def created_at(text):
    return parse_utc(text, "createdAt")


def reviewer_of(rdir, flag):
    """--reviewer, else PIN.md's Reviewer:; fails when neither exists."""
    if flag:
        return flag
    if (rdir / "PIN.md").is_file():
        found = rl.pin_value(rdir, "Reviewer")
        if found:
            return found
    die(f"{rdir.name} has no Reviewer: pass --reviewer <gh login>")


def intake(evidence, n, doc, cutoff):
    """Record `doc` (fetch_threads output with a `reviewer`) for round n: pr/threads.json,
    the Threads rows' State, Last reply and Round seen, threads.md, reply-affected.txt, and
    `unlinked:` and `mismatch:` lines on stdout. A finding's verdict, tier and status are
    never touched."""
    rdir = rl.round_dir(evidence, n)
    reviewer, author, threads = doc["reviewer"], doc["author"], doc["threads"]
    (rdir / "pr").mkdir(exist_ok=True)
    rl.write(rdir / "pr" / "threads.json", json.dumps(doc))
    by_url = {t["comments"]["nodes"][0]["url"]: t for t in threads}
    tables, statuses, rows = [], {}, []
    try:
        if ledger.ledger_path(evidence).is_file():
            tables = ledger.load_ledger(evidence)
            statuses = {
                r["ID"]: r["Status"] for r in rl.find_table(tables, "Findings").rows
            }
            rows = ledger.threads_table(tables).rows
    except rl.Fail as e:
        die(str(e))
    affected, linked = set(), set()
    for row in rows:
        thread = by_url.get(row["Thread URL"])
        if thread is None:
            print(f"missing: {row['Thread URL']} ({row['Finding']})")
            continue
        linked.add(row["Thread URL"])
        replies = thread["comments"]["nodes"][1:]
        row["State"] = thread_state(thread)
        row["Round seen"] = str(n)
        row["Last reply"] = (
            role_of(rl.login_of(replies[-1]), reviewer, author)
            + " "
            + replies[-1]["createdAt"][:10]
            if replies
            else ""
        )
        if cutoff and any(
            rl.new_by_other(c, reviewer, cutoff, created_at) for c in replies
        ):
            affected.add(row["Finding"])
        status = statuses.get(row["Finding"], "")
        if thread["isResolved"] and status in rl.NOT_CLOSED:
            print(f"mismatch: {row['Thread URL']} resolved, {row['Finding']} {status}")
        if not thread["isResolved"] and rl.is_closed(status):
            print(
                f"mismatch: {row['Thread URL']} unresolved, {row['Finding']} {status}"
            )
    if tables:
        ledger.save_ledger(evidence, tables)
    rl.write(rdir / "reply-affected.txt", "".join(f"{f}\n" for f in sorted(affected)))
    lines = [f"# Review threads (round {n})", "", f"{len(threads)} threads", ""]
    lines += [
        "| URL | Where | State | Opened by | Last reply | Linked |",
        "|---|---|---|---|---|---|",
    ]
    for url, t in by_url.items():
        comments = t["comments"]["nodes"]
        last = comments[-1] if len(comments) > 1 else None
        found = ", ".join(r["Finding"] for r in rows if r["Thread URL"] == url)
        lines.append(
            f"| {url} | {t['path']}:{rl.thread_line(t)} | {thread_state(t)}"
            f"{' (outdated)' if t['isOutdated'] else ''} | "
            f"{rl.login_of(comments[0])} | "
            + (f"{rl.login_of(last)} {last['createdAt'][:10]}" if last else "")
            + f" | {found} |"
        )
    rl.write(rdir / "threads.md", "\n".join(lines) + "\n")
    for url in by_url:
        if url not in linked:
            print(f"unlinked: {url}")
    print(
        f"threads: {len(threads)} saved, {len(linked)} linked, "
        f"{len(affected)} findings with a new reply"
    )


def cmd_threads(args):
    """Re-fetch a PR's review threads for round N (a round reopened on the same head) and
    record them as `init` does. A reply is new when it came after --since, else after round
    N's own Pinned:."""
    rdir = rl.round_dir(args.evidence, args.round)
    reviewer = reviewer_of(rdir, args.reviewer)
    cutoff = reopen_cutoff(args.evidence, args.round, args.since)
    doc = fetch_threads(args.owner_repo, args.pr)
    intake(args.evidence, args.round, {**doc, "reviewer": reviewer}, cutoff)


def cmd_freshness(args):
    """Print whether the newest round is still current: `stale: base branch changed`,
    `stale: head moved`, `new thread activity` or `fresh` (in that priority), then
    `next: start r<N+1>` or `next: reopen r<N>`; `N/A` for a review with no PR (`--pr`, else a round's PIN.md KEY `pr<N>`). The previous
    head, Pinned time and target come from the newest round's PIN.md, else the --prev-*
    flags (a missing target skips the base check and says so)."""
    rounds = rl.round_dirs(args.evidence)
    if not rounds:
        die(f"no round directory under {args.evidence}")
    last = rounds[-1]
    n = int(last.name[1:])
    has_pin = (last / "PIN.md").is_file()

    def pinned_value(key):
        return rl.pin_value(last, key) if has_pin else None

    pr = args.pr
    for rd in reversed(rounds):
        if pr:
            break
        key = rl.pin_value(rd, "KEY") if (rd / "PIN.md").is_file() else None
        m = re.fullmatch(r"pr(\d+)", key or "")
        pr = m.group(1) if m else None
    if not pr:
        print("N/A")
        return
    head = args.prev_head or pinned_value("HEAD")
    pinned = args.prev_pinned or pinned_value("Pinned")
    target = args.prev_target or pinned_value("TARGET")
    if not head or not pinned:
        die(
            f"{last.name} has no {'HEAD' if not head else 'Pinned'}: pass "
            f"{'--prev-head' if not head else '--prev-pinned'}"
        )
    since = parse_utc(pinned, "previous Pinned")
    info = json.loads(
        gh(
            "pr",
            "view",
            str(pr),
            "-R",
            args.owner_repo,
            "--json",
            "baseRefName,headRefOid",
        )
    )
    if not target:
        print("skipped: base branch check (no previous target)")
    if target and info["baseRefName"] != target:
        verdict = "stale: base branch changed"
    elif not (
        info["headRefOid"].startswith(head) or head.startswith(info["headRefOid"])
    ):
        verdict = "stale: head moved"
    else:
        reviewer = reviewer_of(last, args.reviewer)
        doc = fetch_threads(args.owner_repo, pr)
        active = any(
            rl.new_by_other(c, reviewer, since, created_at)
            for t in doc["threads"]
            for c in t["comments"]["nodes"]
        )
        verdict = "new thread activity" if active else "fresh"
    print(verdict)
    print(
        f"next: start r{n + 1}" if verdict.startswith("stale") else f"next: reopen r{n}"
    )


def derive_key(text):
    """Sanitize text into a review key and validate it as a ref component."""
    key = re.sub(r"[^A-Za-z0-9._-]", "-", text)
    key = re.sub(r"\.{2,}", "-", key)
    key = re.sub(r"^\.", "", key)
    key = re.sub(r"\.lock$", "", key)
    ok = subprocess.run(
        ["git", "check-ref-format", f"refs/review/{key}/target"], capture_output=True
    ).returncode
    if ok != 0:
        die(f"unusable review key: {key}")
    return key


def cmd_key(args):
    """Print the review key for a branch name or <base>..<head> text."""
    print(derive_key(args.text))


def sparse_set(files):
    """Sparse-checkout directories for changed paths (RUNBOOK sparse-set rules)."""
    out = set()
    for f in files:
        parts = f.split("/")
        if len(parts) == 1:
            continue
        if parts[0] in ("projects", "dnn-providers", "shared"):
            if len(parts) > 2:
                out.add(f"{parts[0]}/{parts[1]}")
            continue
        out.add("/".join(parts[:-1]))
    if any(re.match(r"projects/[^/]+/tools/", f) for f in files):
        out.add("tools")
    return sorted(out)


def pin_lines(values):
    """Format `NAME: value` PIN.md lines."""
    return "".join(f"{k}: {v}\n" for k, v in values)


def append_pin(rdir, text):
    rl.write(rdir / "PIN.md", (rdir / "PIN.md").read_text(encoding="utf-8") + text)


def drift_report(clone, tip, base, head, files, range_review):
    """drift.txt text: target tip, merge result of tip into head, conflicted paths, tip
    commits since base that touch a changed path or a file beside one, and the trigger.
    """
    if range_review:
        return (
            f"Target tip: {tip}\nMerge: not checked (range review)\nConflicts: \n"
            "Trigger: no (range review)\n"
        )
    code, out, err = rl.git(
        clone, "merge-tree", "--write-tree", "--name-only", tip, head
    )
    if code not in (0, 1):
        lines = err.strip().splitlines()
        die(f"git merge-tree failed: {lines[-1] if lines else code}")
    conflicts = []
    if code == 1:
        for line in out.splitlines()[1:]:
            if not line:
                break
            conflicts.append(line)
    changed = set(files)
    dirs = {f.rpartition("/")[0] for f in files}
    touching = []
    log = git(
        clone,
        "log",
        "--no-renames",
        "--name-only",
        "--format=%x01%H %s",
        f"{base}..{tip}",
    )
    for entry in log.split("\x01")[1:]:
        subject, _, names = entry.partition("\n")
        if any(n in changed or n.rpartition("/")[0] in dirs for n in names.split()):
            touching.append(subject)
    if conflicts or touching:
        trigger = "yes"
    elif tip == base:
        trigger = "no (target tip is BASE)"
    else:
        trigger = "no (no target commit touches the change)"
    return (
        f"Target tip: {tip}\n"
        f"Merge: {'conflicts' if code else 'clean'}\n"
        f"Conflicts: {', '.join(conflicts)}\n"
        + "".join(f"Touching commits: {t}\n" for t in touching)
        + f"Trigger: {trigger}\n"
    )


def is_sha(text):
    return re.fullmatch(r"[0-9a-f]{7,40}", text) is not None


def cmd_init(args):
    """Pin one review round: PIN.md, env.sh, refs, resolved SHAs, sparse worktrees, Gate checks."""
    targets = [bool(args.pr), bool(args.branch), bool(args.range)]
    if sum(targets) != 1:
        die("exactly one of --pr, --branch, --range is required")
    n = args.round
    evidence_dir, scratch_dir = Path(args.evidence), Path(args.scratch)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence, scratch = evidence_dir.resolve(), scratch_dir.resolve()
    skill = (
        Path(args.skill).resolve()
        if args.skill
        else Path(__file__).resolve().parent.parent
    )

    if args.pr:
        key = f"pr{args.pr}"
    else:
        key = derive_key(args.branch or args.range)
    clone = Path(args.clone or os.environ.get("CLONE") or scratch / "clone")
    repo, repo_base = scratch / f"r{n}" / "head", scratch / f"r{n}" / "base"
    rdir = evidence / f"r{n}"
    reviewer = args.reviewer or (
        gh("api", "user", "--jq", ".login").strip() if args.pr else None
    )
    cutoff = reply_cutoff(evidence, n, args.since) if args.pr else None
    rdir.mkdir(parents=True, exist_ok=True)
    rl.write(
        rdir / "PIN.md",
        pin_lines(
            [
                ("SKILL", skill),
                ("EVIDENCE", evidence),
                ("SCRATCH", scratch),
                ("CLONE", clone),
                ("KEY", key),
                ("REPO", repo),
                ("REPO_BASE", repo_base),
                (
                    "Pinned",
                    datetime.datetime.now(datetime.timezone.utc).strftime(UTC_FORMAT),
                ),
            ]
            + ([("Reviewer", reviewer)] if reviewer else []),
        ),
    )
    rl.write(rdir / "env.sh", ENV_SH)

    url = args.remote_url or f"https://github.com/{args.owner_repo}.git"
    code, top, _ = rl.git(clone, "rev-parse", "--show-toplevel")
    if code == 0 and same_path(top.strip()) == same_path(clone):
        pass
    elif clone.exists() and (not clone.is_dir() or any(clone.iterdir())):
        die(f"{clone} exists and is not a git clone")
    else:
        git(None, "clone", "--filter=blob:none", "--no-checkout", url, str(clone))
    git(clone, "config", "gc.auto", "0")

    ref_t, ref_h = f"refs/review/{key}/target", f"refs/review/{key}/r{n}"
    specs, shas = [], []
    if args.pr:
        target = gh(
            "pr",
            "view",
            args.pr,
            "-R",
            args.owner_repo,
            "--json",
            "baseRefName",
            "--jq",
            ".baseRefName",
        ).strip()
        specs = [f"+refs/heads/{target}:{ref_t}", f"+refs/pull/{args.pr}/head:{ref_h}"]
    elif args.branch:
        target = args.target or "develop"
        specs = [f"+refs/heads/{target}:{ref_t}", f"+refs/heads/{args.branch}:{ref_h}"]
    else:
        if args.range.count("..") != 1 or "..." in args.range:
            die(f"range must be <base>..<head>: {args.range}")
        target, head_side = args.range.split("..")
        if not target or not head_side:
            die(f"range must be <base>..<head>: {args.range}")
        for side, ref in ((target, ref_t), (head_side, ref_h)):
            if is_sha(side):
                shas.append((ref, side))
            else:
                specs.append(f"+refs/heads/{side}:{ref}")
    if specs:
        git(clone, "fetch", "origin", *specs)
    for ref, sha in shas:
        git(
            clone,
            "update-ref",
            ref,
            git(clone, "rev-parse", "--verify", f"{sha}^{{commit}}"),
        )

    head = git(clone, "rev-parse", f"{ref_h}^{{commit}}")
    if args.range:
        base = git(clone, "rev-parse", f"{ref_t}^{{commit}}")
    else:
        base = git(clone, "merge-base", ref_t, head)
    tip = git(clone, "rev-parse", f"{ref_t}^{{commit}}")
    append_pin(
        rdir,
        pin_lines(
            [
                ("TARGET", target),
                ("TARGET_TIP", tip),
                ("HEAD", head),
                ("HEAD_SHORT", head[:12]),
                ("BASE", base),
            ]
        ),
    )
    if base == head:
        print("empty diff: PR already merged by merge commit?", file=sys.stderr)

    files = git(clone, "diff", "--name-only", "--no-renames", base, head).splitlines()
    rl.write(rdir / "files.txt", "".join(f + "\n" for f in files))
    rl.write(rdir / "sparse.txt", "".join(d + "\n" for d in sparse_set(files)))
    rl.write(
        rdir / "numstat.txt",
        "".join(
            line + "\n"
            for line in git(
                clone, "diff", "--numstat", "--no-renames", base, head
            ).splitlines()
        ),
    )
    rl.write(
        rdir / "commits.txt",
        git(clone, "log", "--format=%H %s", f"{base}..{head}") + "\n",
    )
    rl.write(
        rdir / "drift.txt",
        drift_report(clone, tip, base, head, files, bool(args.range)),
    )
    sparse_text = (rdir / "sparse.txt").read_text(encoding="utf-8")
    for w, sha in ((repo, head), (repo_base, base)):
        git(clone, "worktree", "add", "--no-checkout", "--detach", str(w), sha)
        git(w, "sparse-checkout", "set", "--cone", "--stdin", stdin=sparse_text)
        git(w, "checkout", "HEAD")

    for w, sha in ((repo, head), (repo_base, base)):
        if git(w, "rev-parse", "HEAD") != sha:
            die(f"worktree {w} is not at {sha}")
    if args.pr:
        pr_gate(args, rdir, clone, base, head)
    print(
        f"KEY={key} TARGET={target} HEAD={head[:12]} BASE={base[:12]} files={len(files)}"
    )
    if args.pr:
        doc = fetch_threads(args.owner_repo, args.pr)
        intake(evidence, n, {**doc, "reviewer": reviewer}, cutoff)


def pr_gate(args, rdir, clone, base, head):
    """Save pr/ artifacts, check headRefOid and the file-set Gate, and record `Files:`."""
    pr, o = args.pr, args.owner_repo
    pr_dir = rdir / "pr"
    pr_dir.mkdir(exist_ok=True)
    fields = "title,body,baseRefName,headRefName,headRefOid,mergeStateStatus,mergeable,files,comments,reviews"
    meta = gh("pr", "view", pr, "-R", o, "--json", fields)
    rl.write(pr_dir / "pr.json", meta)
    info = json.loads(meta)
    rl.write(pr_dir / "body.md", info.get("body") or "")
    inline = gh_paginated_list(f"repos/{o}/pulls/{pr}/comments")
    rl.write(pr_dir / "inline.json", json.dumps(inline))
    # Plain text the LEAK sweep reads, one file per source; no jq needed on the reviewer's host.
    rl.write(pr_dir / "branch.txt", (info.get("headRefName") or "") + "\n")
    rl.write(
        pr_dir / "pr-text.txt", f"{info.get('title') or ''}\n{info.get('body') or ''}\n"
    )
    rl.write(
        pr_dir / "pr-comment.txt",
        "".join(
            f"{x.get('body') or ''}\n"
            for k in ("reviews", "comments")
            for x in info.get(k) or []
        ),
    )
    rl.write(
        pr_dir / "pr-inline.txt", "".join(f"{x.get('body') or ''}\n" for x in inline)
    )
    pr_files = gh_paginated_list(f"repos/{o}/pulls/{pr}/files")
    rl.write(pr_dir / "files.json", json.dumps(pr_files))
    rl.write(
        pr_dir / "checks.tsv",
        gh(
            "api",
            "--paginate",
            f"repos/{o}/commits/{head}/check-runs",
            "--jq",
            r".check_runs[] | [.name,.conclusion,.html_url] | @tsv",
        ),
    )

    if info.get("headRefOid") != head:
        die(f"PR head moved since pin: {info.get('headRefOid')} != {head}")
    if len(pr_files) >= MAX_GH_FILES:
        append_pin(rdir, f"Files: unverifiable (>{MAX_GH_FILES})\n")
        return
    local = git(clone, "diff", "--name-only", "-M", base, head).splitlines()
    both = set(
        git(clone, "diff", "--name-only", "--no-renames", base, head).splitlines()
    )
    theirs = {x["filename"] for x in pr_files}
    theirs |= {x["previous_filename"] for x in pr_files if x.get("previous_filename")}
    if both != theirs:
        die(
            f"file lists differ: git {len(local)} files, GitHub {len(pr_files)} files; "
            f"only git: {sorted(both - theirs)[:5]}, only GitHub: {sorted(theirs - both)[:5]}"
        )
    append_pin(rdir, f"Files: {len(local)} (GitHub {len(pr_files)})\n")


def read_pin(rdir):
    pin = {}
    for line in (rdir / "PIN.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"([A-Z_]+): ?(.*)", line)
        if m:
            pin[m.group(1)] = m.group(2)
    return pin


def remove_tree(path):
    """rmtree that also removes read-only git object files (Windows)."""

    def writable(func, p, _exc):
        os.chmod(p, 0o700)
        func(p)

    shutil.rmtree(path, onerror=writable)


def same_path(text):
    """Comparable form of a path: git prints `C:/x` where PIN.md holds `C:\\x`, and Windows
    paths ignore case."""
    return Path(os.path.normcase(os.path.realpath(text)))


def cmd_cleanup(args):
    """Remove review worktrees under $SCRATCH (except $CLONE) and delete refs/review/$KEY/*."""
    pin = read_pin(Path(args.evidence) / f"r{args.round}")
    clone, key = pin["CLONE"], pin["KEY"]
    clone_path, scratch_path = same_path(clone), same_path(pin["SCRATCH"])
    n = 0
    for rel in ("RUN/head", "RUN/base", "ROCK/src", "DRIFT/merge"):
        if (scratch_path / rel).is_dir():
            remove_tree(scratch_path / rel)
            n += 1
    if not Path(clone).is_dir():
        print(f"removed {n} worktrees, 0 refs (no clone)")
        return
    for line in git(clone, "worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            w = line[len("worktree ") :]
            wp = same_path(w)
            if wp != clone_path and wp.is_relative_to(scratch_path):
                git(clone, "worktree", "remove", "--force", w)
                n += 1
    refs = git(
        clone, "for-each-ref", "--format=%(refname)", f"refs/review/{key}/"
    ).splitlines()
    if refs:
        git(
            clone, "update-ref", "--stdin", stdin="".join(f"delete {r}\n" for r in refs)
        )
    print(f"removed {n} worktrees, {len(refs)} refs")


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("key", help=cmd_key.__doc__)
    s.add_argument("text")
    s.set_defaults(fn=cmd_key)
    s = sub.add_parser("init", help=cmd_init.__doc__)
    for a in ("--evidence", "--scratch", "--owner-repo"):
        s.add_argument(a, required=True)
    s.add_argument("--round", type=int, required=True)
    for a in (
        "--pr",
        "--branch",
        "--range",
        "--target",
        "--clone",
        "--remote-url",
        "--skill",
        "--reviewer",
        "--since",
    ):
        s.add_argument(a)
    s.set_defaults(fn=cmd_init)
    s = sub.add_parser("threads", help=cmd_threads.__doc__.split("\n\n")[0])
    for a in ("--evidence", "--owner-repo", "--pr"):
        s.add_argument(a, required=True)
    s.add_argument("--round", type=int, required=True)
    for a in ("--reviewer", "--since"):
        s.add_argument(a)
    s.set_defaults(fn=cmd_threads)
    s = sub.add_parser("freshness", help=cmd_freshness.__doc__.split("\n\n")[0])
    for a in ("--evidence", "--owner-repo"):
        s.add_argument(a, required=True)
    for a in ("--pr", "--reviewer", "--prev-head", "--prev-pinned", "--prev-target"):
        s.add_argument(a)
    s.set_defaults(fn=cmd_freshness)
    s = sub.add_parser("cleanup", help=cmd_cleanup.__doc__)
    s.add_argument("--evidence", required=True)
    s.add_argument("--round", type=int, required=True)
    s.set_defaults(fn=cmd_cleanup)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
