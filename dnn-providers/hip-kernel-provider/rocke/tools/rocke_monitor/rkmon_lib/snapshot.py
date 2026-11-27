# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Poll GitHub and git (read-only) into a snapshot: the open in-scope PRs and the develop branch."""

from __future__ import annotations

import concurrent.futures
import json
import pathlib
import re

from . import gh, gitops, idle
from . import config, paths, reviewer_name, whole_file
from .store import minutes_since, now

_SEARCH = """
query($q: String!, $cursor: String) {
  search(query: $q, type: ISSUE, first: 15, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes { ... on PullRequest {
      number title isDraft createdAt updatedAt additions deletions changedFiles
      author { login } baseRefName headRefName headRefOid mergeable mergeStateStatus reviewDecision
      autoMergeRequest { enabledAt mergeMethod enabledBy { login } }
      reviewRequests(first: 20) { nodes { requestedReviewer { ... on User { login } ... on Team { slug } } } }
      closingIssuesReferences(first: 10) { nodes { number repository { nameWithOwner } } }
      labels(first: 30) { nodes { name } }
      commits(last: 1) { nodes { commit { oid committedDate statusCheckRollup { state } } } }
      latestOpinionatedReviews(first: 25) { nodes { author { login } state submittedAt commit { oid } } }
    } }
  }
}"""

VERDICT = re.compile("[\U0001f534\U0001f7e0\U0001f7e1\U0001f7e2⚪⏸]")
HEAD_IN_META = re.compile(r"^- Head: .* @ ([0-9a-f]{7,40});", re.M)


def parse_verdict(body: str) -> str:
    """The verdict emoji of one of our reviews: the first status emoji on its '## PR #' heading line."""
    for line in body.splitlines():
        if line.startswith("## PR #") or line.startswith("# PR #"):
            m = VERDICT.search(line)
            return m.group(0) if m else "?"
    return "?"


# The scope cache, st["scope_cache"], keeps one answer per pull request, head and folder: for each pull request, the
# head it was checked at and, for each folder checked there, whether the pull request changes files in it:
# {"head": SHA, "folders": {FOLDER: bool}}. An entry from before the answers by folder holds one, "rocke", for the
# repository's folder, the one folder every shared label counted for. It stands, but for a True answer from before
# OLD_V, when develop came from the previous poll: a stale develop only adds files to the diff, so a False answer
# stands, and a True one is checked again.
OLD_V = 2


def answers(cfg: dict, cache: dict, n: int, head: str | None = None) -> dict[str, bool]:
    """The scope cache's answers for pull request N, by folder: whether it changes files there, at HEAD (None: at the
    head it was checked at); none at another head."""
    entry = cache.get(str(n)) or {}
    if not entry or (head is not None and entry.get("head") != head):
        return {}
    if "folders" in entry:
        return dict(entry["folders"])
    if entry.get("v") == OLD_V or not entry.get("rocke"):
        return {paths.repository_folder(cfg): bool(entry.get("rocke"))}
    return {}


def changes(
    cfg: dict,
    n: int,
    head: str,
    folders: list[str],
    cache: dict,
    ref: str | None = None,
) -> dict[str, bool]:
    """Whether pull request N, at HEAD, changes files in each of FOLDERS: its diff from its merge base with develop,
    asked of git for each folder the scope cache has no answer for, and kept there. A folder git cannot answer for
    counts as changed: the pull request stays in scope rather than silently dropped.

    Develop must hold the commit the PR is based on: with an older develop the merge base is older too, and the diff
    takes in develop's own commits since then. So develop is fetched after the PR ref, as the PR's base was on develop
    before its head was pushed (a PR rebased onto a newer develop, checked against an older one, would be put in scope
    by develop's own changes in the folder). `ref`: the PR's ref, when the caller has fetched it and develop after it
    (scope_filter)."""
    known = answers(cfg, cache, n, head)
    ask = [f for f in dict.fromkeys(folders) if f not in known]
    if ask:
        repo = cfg["paths"]["repo"]
        try:
            if ref is None:
                ref = gitops.fetch_pr(repo, n)
                gitops.fetch_develop(repo)
            mb = gitops.git(
                repo, "merge-base", "refs/remotes/origin/develop", ref
            ).strip()
            for f in ask:
                diff = gitops.git(repo, "diff", "--name-only", mb, ref, "--", f)
                known[f] = bool(diff.strip())
        except gitops.GitError:
            return {f: known.get(f, True) for f in folders}
        cache[str(n)] = {"head": head, "folders": known}
    return {f: known[f] for f in folders}


def in_scope(cfg: dict, pr: dict, cache: dict, ref: str | None = None) -> bool:
    """Whether pull request PR (its `n`, `head` and `labels`) is in scope, by the one rule (config.Scope.takes): its
    diff checked in its shared labels' folders (changes)."""
    sc = config.scope(cfg)
    folders = sc.folders(pr["labels"]) or []
    found = changes(cfg, pr["n"], pr["head"], folders, cache, ref) if folders else {}
    return sc.takes(pr["labels"], found.get)


def scope_filter(cfg: dict, prs: list[dict], cache: dict) -> list[dict]:
    """The pull requests in scope (in_scope). The refs of those with a folder to check are fetched first, then develop
    once (changes)."""
    sc, repo = config.scope(cfg), cfg["paths"]["repo"]
    need = [
        p
        for p in prs
        if set(sc.folders(p["labels"]) or ())
        - set(answers(cfg, cache, p["n"], p["head"]))
    ]
    refs = {}
    try:
        for p in need:
            refs[p["n"]] = gitops.fetch_pr(repo, p["n"])
        if need:
            gitops.fetch_develop(repo)
    except gitops.GitError:
        refs = {}  # each PR fetches its own ref and develop
    return [p for p in prs if in_scope(cfg, p, cache, refs.get(p["n"]))]


def label_query(
    cfg: dict,
    labels: list[str],
    kind: str = "pr",
    without: list[str] = (),
    repo: str = "",
) -> str:
    """The search for the open PRs (or issues) with any of `labels` (comma-separated values: any of them) and none of
    `without`, in `repo` (blank: the watched repository)."""
    return (
        f'repo:{repo or cfg["repo"]} is:{kind} is:open label:'
        + ",".join(f'"{l}"' for l in labels)
        + "".join(f' -label:"{l}"' for l in without)
    )


def list_prs(cfg: dict) -> list[dict]:
    """The open pull requests with a label in scope (config.Scope.searched); scope_filter keeps those in scope."""
    return _list_label(cfg, label_query(cfg, config.scope(cfg).searched()))


def corrected_rollup(runs: list[dict], statuses: list[dict]) -> str:
    """The head's CI state from its check runs and commit statuses, where a cancelled run whose check also has a passing
    run does not count (a duplicate a newer run replaced): FAILURE, PENDING or SUCCESS.
    """
    ok = {"success", "skipped", "neutral"}
    passed = {
        r["name"]
        for r in runs
        if r.get("status") == "completed" and r.get("conclusion") in ok
    }
    bad = [
        r
        for r in runs
        if r.get("status") == "completed"
        and r.get("conclusion") not in ok
        and not (r.get("conclusion") == "cancelled" and r["name"] in passed)
    ]
    if bad or any(x.get("state") in ("failure", "error") for x in statuses):
        return "FAILURE"
    if any(r.get("status") != "completed" for r in runs) or any(
        x.get("state") == "pending" for x in statuses
    ):
        return "PENDING"
    return "SUCCESS"


def correct_rollups(cfg: dict, prs: list[dict], cache: dict) -> list[dict]:
    """GitHub's rollup says FAILURE when any check run was cancelled, also when a newer run of the same check passed
    (a bot's run cancelled at a push while its twin passed is no CI failure). For a FAILURE rollup, read the head's check runs and statuses and decide with
    `corrected_rollup`; GitHub's own answer stays in ci_rollup_raw. Cached per head once final (not PENDING); a read
    error keeps GitHub's answer."""
    listed = set()
    for p in prs:
        key = str(p["n"])
        listed.add(key)
        if p.get("ci_rollup") != "FAILURE":
            continue
        c = cache.get(key) or {}
        if c.get("head") == p["head"] and c.get("state") in ("SUCCESS", "FAILURE"):
            state = c["state"]
        else:
            try:
                runs = gh.rest(
                    f"repos/{cfg['repo']}/commits/{p['head']}/check-runs?per_page=100",
                    paginate=True,
                    jq="[.check_runs[] | {name, status, conclusion}]",
                )
                statuses = gh.rest(
                    f"repos/{cfg['repo']}/commits/{p['head']}/status",
                    jq="[.statuses[] | {context, state}]",
                )
            except gh.GhError:
                continue
            state = corrected_rollup(runs, statuses)
            cache[key] = {"head": p["head"], "state": state}
        if state != "FAILURE":
            p["ci_rollup_raw"] = "FAILURE"
            p["ci_rollup"] = state
    for k in [k for k in cache if k not in listed]:
        del cache[k]
    return prs


# A read token with no organization access (a fine-grained token on the repository alone) cannot read review
# requests in the PR search: their teams are organization data, and GitHub refuses the whole query with this.
_NO_ORG_ACCESS = "Resource not accessible by personal access token"
_REVIEW_REQUESTS_LINE = "      reviewRequests(first: 20) { nodes { requestedReviewer { ... on User { login } ... on Team { slug } } } }\n"
REVIEW_REQUESTS_CACHE = "review_requests.json"


def _review_requests_file(cfg: dict) -> pathlib.Path | None:
    folder = (cfg.get("paths") or {}).get("cache")
    if not folder:
        try:
            from . import paths

            folder = paths.resolve(cfg).get("cache")
        except Exception:  # noqa: BLE001 - no folder known: no cache
            folder = ""
    return pathlib.Path(folder) / REVIEW_REQUESTS_CACHE if folder else None


def rest_review_requests(cfg: dict, n: int, updated_at: str) -> list[str]:
    """PR n's requested reviewers (user logins, team slugs) over REST, for a search that could not read them: kept in
    <cache>/review_requests.json per PR until the PR's updatedAt moves, written whole (whole_file) at the mode of the
    files the monitor shares (worker/perms.py). A failed read gives the last known list.
    """
    path = _review_requests_file(cfg)
    try:
        cache = json.loads(path.read_text()) if path else {}
    except (OSError, ValueError):
        cache = {}
    hit = cache.get(str(n)) if isinstance(cache, dict) else None
    if hit and hit.get("updated_at") == updated_at:
        return hit["reqs"]
    try:
        data = gh.rest(
            f"repos/{cfg['repo']}/pulls/{n}/requested_reviewers",
            jq="[.users[].login] + [.teams[].slug]",
        )
    except gh.GhError:
        return (hit or {}).get("reqs", [])
    if not path:
        return data
    cache = cache if isinstance(cache, dict) else {}
    cache[str(n)] = {"updated_at": updated_at, "reqs": data}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        whole_file.write(
            path, json.dumps(cache), paths.worker_module("perms").SHARED_FILE_MODE
        )
    except OSError:
        pass
    return data


def search_pages(q: str, first: int = 15, query: str = _SEARCH):
    """A search's result pages (`query`: a search with `first: <n>, after: $cursor`; the tick's PR search by default).
    GitHub answers HTTP 502 when a GraphQL request runs past its time limit (about 10 s): a page that does is asked
    again with half as many results, down to one, and the smaller size holds for the rest of the walk. A read login
    with no organization access (_NO_ORG_ACCESS) searches without the review requests, which _list_label then reads
    per PR over REST (rest_review_requests).
    """
    cursor, template = None, query
    while True:
        query = re.sub(
            r"first: \d+, after: \$cursor", f"first: {first}, after: $cursor", template
        )
        try:
            data = gh.graphql(
                query, attempts=1 if first > 1 else 3, q=q, cursor=cursor
            )["search"]
        except gh.GhError as e:
            if _NO_ORG_ACCESS in str(e) and _REVIEW_REQUESTS_LINE in template:
                template = template.replace(_REVIEW_REQUESTS_LINE, "")
                continue
            if first == 1 or not gh.transient(str(e)):
                raise
            first = max(1, first // 2)
            continue
        yield data
        if not data["pageInfo"]["hasNextPage"]:
            return
        cursor = data["pageInfo"]["endCursor"]


def closes(cfg: dict, node: dict) -> list[int]:
    """The develop issues PR NODE closes when it merges ("Fixes #N", "Fixes <org>/<repo>#N"): its closing references
    in the develop issues' repository (breakage.issue_repository). A reference that does not name its repository
    is in the watched repository."""
    # a call-time import: breakage imports this module
    from .breakage import issue_repository

    where, out = issue_repository(cfg).lower(), []
    for i in (node.get("closingIssuesReferences") or {}).get("nodes") or []:
        repo = ((i or {}).get("repository") or {}).get("nameWithOwner") or cfg["repo"]
        if i and repo.lower() == where:
            out.append(i["number"])
    return out


def _list_label(cfg: dict, q: str) -> list[dict]:
    prs = []
    for data in search_pages(q):
        for node in data["nodes"]:
            if not node:
                continue
            commit = (node["commits"]["nodes"] or [{}])[0].get("commit") or {}
            rollup = (commit.get("statusCheckRollup") or {}).get("state") or "NONE"
            labels = [l["name"] for l in node["labels"]["nodes"]]
            reviews = [
                {
                    "login": (r["author"] or {}).get("login", "?"),
                    "state": r["state"],
                    "at": r["submittedAt"],
                    "commit": (r["commit"] or {}).get("oid"),
                }
                for r in node["latestOpinionatedReviews"]["nodes"]
            ]
            prs.append(
                {
                    "n": node["number"],
                    "title": node["title"][:90],
                    "author": (node["author"] or {}).get("login", "?"),
                    "draft": node["isDraft"],
                    "labels": labels,
                    "base": node["baseRefName"],
                    "head_ref": node["headRefName"],
                    "head": node["headRefOid"],
                    "created_at": node["createdAt"],
                    "updated_at": node["updatedAt"],
                    "size": node["additions"] + node["deletions"],
                    "files": node["changedFiles"],
                    "mergeable": node["mergeable"],
                    "merge_state": node["mergeStateStatus"],
                    "review_decision": node["reviewDecision"],
                    "ci_rollup": rollup,
                    "reviews": reviews,
                    # When the head commit was made: the settle counts from it, not from the tick that first saw it.
                    "head_committed_at": commit.get("committedDate"),
                    # Users (login) and teams (slug) whose review is requested; GitHub drops a user once they review.
                    # A search without them (search_pages: no organization access) reads them per PR.
                    "review_requests": (
                        [
                            x
                            for x in (
                                (rr.get("requestedReviewer") or {}).get("login")
                                or (rr.get("requestedReviewer") or {}).get("slug")
                                for rr in (node.get("reviewRequests") or {}).get(
                                    "nodes"
                                )
                                or []
                            )
                            if x
                        ]
                        if "reviewRequests" in node
                        else rest_review_requests(
                            cfg, node["number"], node["updatedAt"]
                        )
                    ),
                    # The develop issues this PR closes when it merges: an open one among them makes it a develop
                    # fix (queue.refresh_develop_fixes).
                    "closes": closes(cfg, node),
                    # Who armed auto-merge, if anyone: the author, a maintainer, or us.
                    "auto_merge": (
                        {
                            "by": (
                                (node["autoMergeRequest"].get("enabledBy") or {}).get(
                                    "login"
                                )
                                or "?"
                            ),
                            "at": node["autoMergeRequest"].get("enabledAt"),
                            "method": node["autoMergeRequest"].get("mergeMethod"),
                        }
                        if node.get("autoMergeRequest")
                        else None
                    ),
                }
            )
    return prs


def account_reviews(cfg: dict, n: int) -> list[dict]:
    """Every review on PR n posted under this deployment's account (config.operator_login), oldest first, whoever
    posted it: this deployment, an older monitor or another deployment under the same account, or a person by hand. A
    review is a post of the account's that reviewer_name.is_review knows for one: an issue comment, or a formal PR
    review (older batches posted those). Each: {"kind": "comment" or "review", "id", "at" (a comment's created_at, a
    review's submitted_at), "url", "body" (its first 1500 characters), "commit" (the head a formal review names; None
    for a comment), "state" (a formal review's)}. our_latest_review keeps this deployment's own among them; `rk facts`
    lists the ones its ledger does not hold (facts.py), and `rk review-close` takes one that stands for ours
    (review.covering_review).
    """
    me = config.operator_login(cfg)
    rows = gh.rest(
        f"repos/{cfg['repo']}/issues/{n}/comments?per_page=100",
        paginate=True,
        jq=f'[.[] | select(.user.login=="{me}") | {{id, at: .created_at, url: .html_url, '
        f"body: (.body[0:1500])}}]",
    )
    revs = gh.rest(
        f"repos/{cfg['repo']}/pulls/{n}/reviews?per_page=100",
        paginate=True,
        jq=f'[.[] | select(.user.login=="{me}" and .submitted_at != null) | {{id, at: .submitted_at, '
        f'url: .html_url, state, commit: .commit_id, body: ((.body // "")[0:1500])}}]',
    )
    posts = [dict(r, kind="comment", commit=None) for r in rows]
    posts += [dict(r, kind="review") for r in revs]
    return sorted(
        (p for p in posts if reviewer_name.is_review(p.get("body"))),
        key=lambda p: p.get("at") or "",
    )


def as_ours(post: dict, count: int) -> dict:
    """The review POST, one of account_reviews, as our review's record (rec["ours"]), COUNT numbering our reviews of
    the PR. A formal review's head is the commit it names, exact; a comment's is None until resolve_reviewed_head
    fills it."""
    return {
        "kind": post["kind"],
        "id": post["id"],
        "at": post["at"],
        "url": post["url"],
        "verdict": parse_verdict(post["body"]),
        "line1": post["body"].splitlines()[0][:120],
        "count": count,
        "state": post.get("state"),
        "head": post.get("commit"),
        "head_src": "review-commit" if post.get("commit") else None,
    }


def our_latest_review(cfg: dict, n: int) -> dict | None:
    """Our latest review post on PR n, as our review's record (as_ours): the latest of the account's reviews
    (account_reviews) of this deployment's own: with a line 1 that gives one of its automated reviewer names, or
    gives none and was posted (a comment's created_at, a review's submitted_at) before reviews.unnamed_are_ours_until,
    or gives none in a mirror (reviewer_name.own_review). A review that gives another deployment's name under the
    same account is not ours.

    A formal review carries its commit, so its reviewed head is exact. Comments get theirs from
    resolve_reviewed_head().
    """
    posts = [
        p
        for p in account_reviews(cfg, n)
        if reviewer_name.own_review(cfg, p.get("body"), p.get("at"))
    ]
    if not posts:
        return None
    return as_ours(max(posts, key=lambda r: r["at"]), len(posts))


def head_at(cfg: dict, n: int, when: str) -> str | None:
    """Best estimate of PR n's head at time `when`, from its timeline (force pushes and commit dates)."""
    events = gh.rest(
        f"repos/{cfg['repo']}/issues/{n}/timeline?per_page=100",
        paginate=True,
        jq='[.[] | select(.event=="committed" or .event=="head_ref_force_pushed") | '
        "{event, sha: (.sha // .commit_id), t: (.committer.date // .created_at)}]",
    )
    head = None
    for e in events:
        if e["t"] and e["t"] <= when:
            head = e["sha"]
    return head


def worker_meta_head(cfg: dict, n: int) -> tuple[str | None, str | None]:
    """The head our last worker run fetched (workers/pr<n>/pr/meta.md) and that file's mtime (UTC)."""
    meta = pathlib.Path(cfg["paths"]["workers"]) / f"pr{n}" / "pr" / "meta.md"
    if not meta.exists():
        return None, None
    m = HEAD_IN_META.search(meta.read_text(errors="replace"))
    import datetime as dt

    mtime = dt.datetime.fromtimestamp(meta.stat().st_mtime, dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return (m.group(1) if m else None), mtime


def resolve_reviewed_head(cfg: dict, store, n: int, ours: dict) -> dict:
    """Fill ours['head'] from (0) a formal review's commit, (1) the ledger, (2) the worker's meta.md if it
    predates the post, (3) the timeline."""
    if ours.get("head_src") == "review-commit":
        return ours
    posts = [
        r
        for r in store.ledger()
        if r["kind"] == "post"
        and r["pr"] == n
        and (r["ref"] == ours["url"] or abs(minutes_since(ours["at"], r["ts"])) < 15)
    ]
    if posts:
        ours["head"], ours["head_src"] = posts[-1]["head"], "ledger"
        return ours
    sha, mtime = worker_meta_head(cfg, n)
    if (
        sha
        and mtime
        and mtime <= ours["at"]
        and minutes_since(mtime, ours["at"]) < 24 * 60
    ):
        full = gitops.git(
            cfg["paths"]["repo"],
            "rev-parse",
            "--verify",
            "-q",
            f"{sha}^{{commit}}",
            check=False,
        ).strip()
        ours["head"], ours["head_src"] = (full or sha), "worker-meta"
        return ours
    ours["head"], ours["head_src"] = head_at(cfg, n, ours["at"]), "timeline"
    return ours


def account_review_head(cfg: dict, n: int, review: dict) -> dict:
    """Fill the head of REVIEW (one of account_reviews, as_ours), a review of our account this deployment did not post,
    from GitHub alone: a formal review's commit (head_src "review-commit"), exact, else PR n's head when it was posted
    (head_at: head_src "timeline"), an estimate. Never from this deployment's ledger or worker files, as
    resolve_reviewed_head reads them for its own posts: another's review posted near a post of ours, or after our
    worker fetched the PR, is no review of that head."""
    if review.get("head_src") != "review-commit":
        review["head"], review["head_src"] = head_at(cfg, n, review["at"]), "timeline"
    return review


def required_contexts(cfg: dict) -> list[str]:
    """The status checks develop's ruleset requires (e.g. Math CI Summary, pre-commit, Multi-Arch CI Summary)."""
    return gh.rest(
        f"repos/{cfg['repo']}/rules/branches/develop",
        jq='[.[] | select(.type=="required_status_checks") | .parameters.required_status_checks[].context]',
    )


def required_status(cfg: dict, sha: str, names: list[str]) -> dict:
    """Each required check's state at `sha`: success / failure / pending / missing (never reported). The rollup GitHub
    gives covers only checks that reported, so a required check that never started looks green there.
    """
    runs = gh.rest(
        f"repos/{cfg['repo']}/commits/{sha}/check-runs?per_page=100",
        paginate=True,
        jq="[.check_runs[] | {name, status, conclusion}]",
    )
    statuses = gh.rest(
        f"repos/{cfg['repo']}/commits/{sha}/status",
        jq="[.statuses[] | {context, state}]",
    )
    out = {}
    for name in names:
        r = [x for x in runs if x["name"] == name]
        s = [x for x in statuses if x["context"] == name]
        if r:
            c = r[0]
            out[name] = (
                "pending"
                if c["status"] != "completed"
                else (
                    "success"
                    if c["conclusion"] in ("success", "skipped", "neutral")
                    else "failure"
                )
            )
        elif s:
            out[name] = {"success": "success", "pending": "pending"}.get(
                s[0]["state"], "failure"
            )
        else:
            out[name] = "missing"
    return out


_THREADS = """query($n: Int!) { repository(owner: "%s", name: "%s") { pullRequest(number: $n) {
  reviewThreads(first: 100) { nodes { isResolved comments(first: 1) { nodes { author { login } path } } } } } } }"""


def unresolved_threads(cfg: dict, n: int) -> list[dict]:
    """Unresolved review threads of PR n (the develop ruleset requires them resolved before a merge)."""
    owner, name = cfg["repo"].split("/")
    nodes = gh.graphql(_THREADS % (owner, name), n=n)["repository"]["pullRequest"][
        "reviewThreads"
    ]["nodes"]
    return [
        {
            "login": (
                (t["comments"]["nodes"][0].get("author") or {}).get("login") or "?"
            ),
            "path": t["comments"]["nodes"][0].get("path") or "",
        }
        for t in nodes
        if not t["isResolved"] and t["comments"]["nodes"]
    ]


def same_commit(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    k = min(len(a), len(b))
    return k >= 7 and a[:k] == b[:k]


def covers(ours: dict | None, head: str | None) -> bool:
    """Our review `ours` covers `head`: the head it reviewed, or a head recorded as equivalent to it since (a rebase-only
    or merge-only push). One rule for the plan, the review driver, G1 and the post ledger.
    """
    ours = ours or {}
    return any(
        same_commit(h, head)
        for h in [ours.get("head"), *(ours.get("equivalent_heads") or [])]
    )


def codeowner_org(cfg: dict) -> str:
    """The organization of the code-owner teams: config codeowner_org, else the repository's owner."""
    return cfg.get("codeowner_org") or cfg["repo"].split("/", 1)[0]


def codeowner_members(cfg: dict) -> dict:
    """Members of every CODEOWNERS team named in config (organization data: a login with no organization access gets
    a GhError, which refresh_codeowners answers)."""
    teams = sorted(
        {t for ts in (cfg.get("codeowner_teams") or {}).values() for t in ts}
    )
    org = codeowner_org(cfg)
    return {
        t: gh.rest(
            f"orgs/{org}/teams/{t}/members?per_page=100",
            paginate=True,
            jq="[.[].login]",
        )
        for t in teams
    }


def refresh_codeowners(cfg: dict, st: dict, at: str) -> list[str]:
    """The code-owner teams' members read again (codeowner_members) into the state's `codeowners`, {"at": AT, "teams":
    team -> logins}, and the lines the tick prints. GitHub refusing the read (gh.GhError: team membership is
    organization data, which a read login without Members on the organization cannot list) never fails the tick:
    - after a read that listed them, that list stands, tried again a day later ("tried": AT), and a line in
      parentheses says so;
    - with no list, the teams are none ("refused" true), so the queue knows a PR's code-owner approvals only from
      GitHub's review decision; the next try is a day later, and a FEATURE_SKIPPED line says what is missing and its
      fix once, as a feature that is on but not set up does (features.tick_reports), until a read lists them.
    """
    co = st.get("codeowners")
    try:
        teams = codeowner_members(cfg)
    except gh.GhError as exc:
        if co and not co.get("refused"):
            co["tried"] = at
            return [
                f"(code-owner teams not refreshed, the list from {co['at']} stands: {str(exc)[:160]})"
            ]
        st["codeowners"] = {"at": at, "teams": {}, "refused": True}
        return [] if co else [_codeowners_refused(cfg, exc)]
    st["codeowners"] = {"at": at, "teams": teams}
    return []


def _codeowners_refused(cfg: dict, exc: Exception) -> str:
    """The tick's line for the first refused read of the code-owner teams (refresh_codeowners): what is missing, what
    the queue does without it, and its fix (ghlogins.read_token_fix)."""
    from . import ghlogins  # a call-time import: ghlogins reads the logins' folders

    org = codeowner_org(cfg)
    return (
        f"FEATURE_SKIPPED code-owner approvals: the read login cannot read {org}'s team members ({str(exc)[:160]}): "
        f"it needs Members: read on {org}; until then the queue knows a PR's code-owner approvals only from GitHub's "
        f"review decision (fix: {ghlogins.read_token_fix(cfg['repo'])})"
    )


def develop_snapshot(cfg: dict, st: dict) -> dict:
    """Develop head, the first-parent commits since the last poll (and which touch rocKE), and develop CI."""
    repo = cfg["paths"]["repo"]
    sha = gitops.fetch_develop(repo)
    old = st["develop"].get("sha")
    snap = {"sha": sha, "old": old, "new_commits": [], "rocke_commits": []}
    if old and old != sha and gitops.has_commit(repo, old):
        snap["new_commits"] = gitops.first_parent_since(repo, old, sha)
        snap["rocke_commits"] = gitops.first_parent_since(
            repo, old, sha, paths.repository_folder(cfg)
        )
    # CI: the newest 8 first-parent SHAs; those whose CI has not settled are read again (ci_settled).
    recent = [c for c, _ in gitops.first_parent_recent(repo, sha, 8)] or [sha]
    ci = {}
    for s in recent:
        prev = st["develop"]["ci"].get(s)
        ci[s] = prev if ci_settled(prev) else develop_ci(cfg, s)
    snap["ci"] = ci
    return snap


# The develop CI verdicts a later run attempt can still change: a failure in a test (a flake can turn green on a re-run)
# or for infrastructure.
RERUN_VERDICTS = ("red", "red-other", "infra")


def ci_settled(prev: dict | None) -> bool:
    """A develop SHA's recorded CI is final: every watched run is complete and none failed. The tick reads every other
    SHA of its window again, and so does the change fingerprint."""
    return (
        bool(prev)
        and bool(prev.get("complete"))
        and ci_verdict(prev) not in RERUN_VERDICTS
    )


def watched_runs(cfg: dict, sha: str, branch: str = "develop") -> dict[str, dict]:
    """The latest run of each watched workflow (config `develop_watched_workflows`) for the push of `sha` to `branch`:
    name -> {id, name, status, conclusion, run_attempt}."""
    runs = gh.rest(
        f"repos/{cfg['repo']}/actions/runs?branch={branch}&event=push&head_sha={sha}&per_page=50",
        jq="[.workflow_runs[] | {id, name, status, conclusion, run_attempt}]",
    )
    watched = {}
    for r in runs:
        if r["name"] not in cfg["develop_watched_workflows"]:
            continue
        prev = watched.get(r["name"])
        if prev is None or r["id"] > prev["id"]:  # the latest attempt of each workflow
            watched[r["name"]] = r
    return watched


# A summary job only reports the other jobs' results: never a failure of its own.
SUMMARY_JOB = re.compile(r"\bSummary\b|\bNotify\b")
# The "Test" step states of a failed job that ran no failing test: the step never started, was skipped after a setup
# step failed, or passed before a later step failed.
NO_TEST_RUN = ("pending", "queued", "waiting", "requested", "skipped", "success")


def develop_ci(cfg: dict, sha: str) -> dict:
    watched = watched_runs(cfg, sha)
    pat = re.compile(cfg["develop_rocke_job_pattern"], re.I)
    out = {"checked_at": now(), "runs": {}, "complete": bool(watched)}
    for name, r in watched.items():
        rec = {"id": r["id"], "status": r["status"], "conclusion": r["conclusion"]}
        if r["status"] != "completed":
            out["complete"] = False
        # per_page=50: GitHub answers 502 for the first 100 jobs of some large TheRock runs.
        # The jobs of the attempt this conclusion belongs to: once the auto-retry starts the next attempt, the latest
        # jobs list shows the re-run rocKE job in progress, and the failure would read as outside rocKE.
        jobs = f"repos/{cfg['repo']}/actions/runs/{r['id']}/attempts/{r.get('run_attempt') or 1}/jobs"
        try:
            if r["conclusion"] == "failure":
                # Each failed job with its "Test" step's state. A job whose tests never ran failed for infrastructure
                # (a runner lost in "Run setup test environment", Test pending and no log), never a develop break.
                failed = _attempt_jobs(
                    jobs,
                    '[.jobs[] | select(.conclusion=="failure") | {name, test: '
                    '([.steps[]? | select(.name=="Test") | .conclusion // .status] | first)}]',
                )
                failed = [
                    j if isinstance(j, dict) else {"name": j, "test": None}
                    for j in failed
                ]
                infra = [j["name"] for j in failed if j.get("test") in NO_TEST_RUN]
                rec["failed_jobs"] = [j["name"] for j in failed][:30]
                rec["infra_failed"] = infra[:30]
                rec["rocke_failed"] = [
                    j["name"]
                    for j in failed
                    if pat.search(j["name"]) and j["name"] not in infra
                ]
            elif (
                r["conclusion"] == "success"
            ):  # which rocKE jobs actually ran and passed: path filters can skip them
                passed = _attempt_jobs(
                    jobs, '[.jobs[] | select(.conclusion=="success") | .name]'
                )
                rec["rocke_passed"] = [j for j in passed if pat.search(j)]
        except gh.GhError as exc:
            # GitHub can answer 502 and cut the stream on every page size for a run: the failure must not stop the
            # tick. The run reads as unread, so the verdict is "unknown" and the next tick asks again.
            if not gh.transient(str(exc)):
                raise
            rec["unread"] = str(exc)[:200]
            out["complete"] = False
        out["runs"][name] = rec
    return out


def _attempt_jobs(path: str, jq: str) -> list:
    """An attempt's jobs, every page. GitHub answers 502 for the first 100 jobs of some large TheRock runs, and at times
    for 50 too: a transient failure asks again with smaller pages."""
    err = None
    for per_page in (50, 25, 10):
        try:
            return gh.rest(f"{path}?per_page={per_page}", paginate=True, jq=jq)
        except gh.GhError as exc:
            if not gh.transient(str(exc)):
                raise
            err = exc
    raise err


def ci_verdict(ci: dict) -> str:
    """green / red / red-other / infra / pending / unknown for one develop SHA's watched runs. "infra": every failed job
    failed before its tests ran (or after they passed), apart from the summary jobs reporting it.
    """
    runs = ci.get("runs") or {}
    if not runs:
        return "unknown"
    if any(r.get("rocke_failed") for r in runs.values()):
        return "red"
    if any(r.get("unread") for r in runs.values()):
        return "unknown"
    failed = [r for r in runs.values() if r["conclusion"] == "failure"]
    if any(not _infra_only(r) for r in failed):
        return "red-other"
    if any(r["status"] != "completed" for r in runs.values()):
        return "pending"
    if failed:
        return "infra"
    if all(r["conclusion"] in ("success", "skipped", "neutral") for r in runs.values()):
        return "green"
    return "unknown"  # cancelled / superseded


def _infra_only(run: dict) -> bool:
    """A failed run whose failed jobs are all infrastructure (their tests never ran), or summary jobs reporting them."""
    infra = set(run.get("infra_failed") or [])
    real = [
        j
        for j in run.get("failed_jobs") or []
        if j not in infra and not SUMMARY_JOB.search(j)
    ]
    return bool(infra) and not real


def red_cleared_by(window: dict, sha: str) -> str | None:
    """The newer SHA of a develop CI window (newest first) that already shows `sha`'s red gone: green, with every rocKE
    job that failed at `sha` passing there (a red outside rocKE needs only a green). A newer red stops the search, as in
    guard.ci_cleared (a re-run attempt can turn red after newer SHAs passed its failed job: no incident opens and
    closes in one tick)."""
    shas = list(window)
    if sha not in shas:
        return None
    jobs = {
        j
        for r in (window[sha].get("runs") or {}).values()
        for j in r.get("rocke_failed") or []
    }
    for s in shas[: shas.index(sha)]:
        v = ci_verdict(window[s])
        if v == "green":
            passed = {
                j
                for r in (window[s].get("runs") or {}).values()
                for j in r.get("rocke_passed") or []
            }
            if jobs <= passed:
                return s
        elif v == "red" or (v == "red-other" and not jobs):
            return None
    return None


# --- the change fingerprint's parts (idle.py): what the poll reads, read cheaply --------------------------------------

_SLIM = """
query($q: String!, $cursor: String) {
  search(query: $q, type: ISSUE, first: 100, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes { ... on PullRequest {
      number isDraft updatedAt headRefOid mergeable autoMergeRequest { enabledAt }
      labels(first: 30) { nodes { name } } statusCheckRollup { state }
    } }
  }
}"""


def pr_signatures(cfg: dict, q: str, keep=lambda node: True) -> list[list]:
    """One row per open PR the search `q` finds (and `keep` keeps): number, head, last update, draft, mergeable, CI
    rollup, auto-merge and labels. The poll's search with the fields a change shows in.
    """
    out = []
    for data in search_pages(q, first=100, query=_SLIM):
        for node in data["nodes"]:
            if not node or not keep(node):
                continue
            out.append(
                [
                    node["number"],
                    node["headRefOid"],
                    node["updatedAt"],
                    node["isDraft"],
                    node["mergeable"],
                    (node.get("statusCheckRollup") or {}).get("state") or "NONE",
                    (node.get("autoMergeRequest") or {}).get("enabledAt"),
                    sorted(n["name"] for n in node["labels"]["nodes"]),
                ]
            )
    return sorted(out)


@idle.part("PRs by label", remote=True)
def _prs_by_label(cfg: dict, st: dict, store) -> list[list]:
    """The open PRs a scope label puts in scope."""
    return pr_signatures(cfg, label_query(cfg, config.scope(cfg).labels))


@idle.part("PRs by shared label", remote=True)
def _prs_by_shared_label(cfg: dict, st: dict, store) -> list[list]:
    """The open pull requests with a shared label and no scope label, but those the scope cache found to change no file
    in their shared labels' folders at their head: their updates are not the monitor's business.
    """
    sc = config.scope(cfg)
    if not sc.shared:
        return []
    cache = st.get("scope_cache") or {}

    def keep(node: dict) -> bool:
        known = answers(cfg, cache, node["number"], node["headRefOid"])
        labels = [x["name"] for x in node["labels"]["nodes"]]
        return sc.takes(labels, lambda folder: known.get(folder) is not False)

    return pr_signatures(
        cfg, label_query(cfg, list(sc.shared), without=sc.labels), keep
    )


_BRANCH = "query($o: String!, $n: String!%s) { repository(owner: $o, name: $n) { %s { name target { oid } } } }"


@idle.part("develop", remote=True)
def _develop(cfg: dict, st: dict, store) -> dict:
    """The base branch's head (config `base_branch`, else the repository's default branch), and the watched CI runs of
    each SHA of the tick's develop window whose CI has not settled (ci_settled)."""
    owner, name = cfg["repo"].split("/", 1)
    branch = cfg.get("base_branch")
    if branch:
        ref = gh.graphql(
            _BRANCH % (", $r: String!", "ref(qualifiedName: $r)"),
            o=owner,
            n=name,
            r=f"refs/heads/{branch}",
        )["repository"]["ref"]
    else:
        ref = gh.graphql(_BRANCH % ("", "defaultBranchRef"), o=owner, n=name)[
            "repository"
        ]["defaultBranchRef"]
    if not ref:
        raise gh.GhError(f"{cfg['repo']} has no branch {branch or '(default)'}")
    todo = [
        s
        for s, prev in ((st.get("develop") or {}).get("ci") or {}).items()
        if not ci_settled(prev)
    ]
    runs = {}
    if todo:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(8, len(todo))
        ) as pool:
            for s, w in zip(
                todo, pool.map(lambda s: watched_runs(cfg, s, ref["name"]), todo)
            ):
                # A run not complete yet reads the same queued or in progress: the tick tells only complete from not
                # (develop_ci, ci_verdict).
                runs[s[:12]] = sorted(
                    [
                        k,
                        r["id"],
                        r.get("run_attempt"),
                        r["status"] == "completed",
                        r["conclusion"],
                    ]
                    for k, r in w.items()
                )
    return {"head": ref["target"]["oid"], "ci": runs}
