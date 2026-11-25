# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Develop breakages live in GitHub issues, in the develop issues' repository (`issue_repository`: the develop guard's
issue repository, else the watched repository): one source of truth for breakages. Several people keep develop green and
force-revert where necessary, so the monitor reads the open ones every tick. It links our incidents to issues others
filed, never files a duplicate, and files or comments only through the guarded writes in post.py.

Two kinds, each with its label from the settings (`labels`, config `incident_issue`): RED, a develop break that makes
CI red, labelled with the develop guard's CI-red label (`labels[0]`) and its extra labels (`labels[1:]`,
`extra_labels`); STABILITY, a develop break CI does not see (e.g. a failing test no CI lane runs), labelled with the
stability label (`stability_label`) and assigned to the author of the breaking PR. The code names no label of its
own: reports print each kind by its label (`kind_label`, `entry_label`). st["post_merge"] keeps its name and has
the open issues of both kinds, each with its `kind` and `label`; an entry with no kind is a CI-red one.

Which develop issues concern the watched folder (`relevant`: an in-scope label, or a title that
incident_issue.title_pattern matches), the repository's own test names (incident_issue.test_name_pattern, beside the
test functions and test classes every repository's issues name) and those that run most of its tests
(incident_issue.umbrella_test_names), which `overlaps` reads, are the repository's: its repository defaults set them,
and the code names none of its own.

Every tick also sweeps the open PRs for fixes to the open rocKE issues (`fix_sweep`).

A file for `rk issue-file` starts with a front-matter block, then the body, whose line 1 is the report line
(post.notice_line). `labels:` carries exactly one of the two kinds' labels; a stability file needs `assign:`, the
breaking PR's author:

    ---
    title: [develop] <what is broken, where>
    labels: <the kind's label>[, <other labels>]
    assign: <login>[, <login>]       (required for a stability issue, optional for a CI-red one)
    incident: <incident id>          (optional)
    ---
    Automated develop-health report by <automated_reviewer_name>
    ...
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from urllib.parse import quote

from . import config, gh, gitops, idle, paths, registry, snapshot
from .store import now

# The two kinds (the module's docstring), named as the develop guard's settings name their labels.
RED, STABILITY = "ci_red", "stability"
# Each kind's word, where the settings name no label for it.
WORDS = {RED: "CI-red", STABILITY: "stability"}
REF = re.compile(r"(?<![\w/])#(\d{4,6})\b")
# The fix PRs an issue names: "**Fix candidate:** #101" (or a list, "**Fix candidates:** #1, #2"), "Fixed by #101",
# "fix PR #102". A candidate line that starts "none" names none, and a PR mentioned near the word "fix" is no fix
# ("land the fix with or before #103" does not make #103 a fix).
FIX_LINE = re.compile(r"\bfix candidates?\b[\s:*_-]{0,8}([^\n]*)", re.I)
FIX_BY = re.compile(
    r"\b(?:fixed\s+(?:by|in)|fix\s+PR|fix(?:es)?)\W{0,6}#(\d{4,6})\b", re.I
)
NO_FIX = re.compile(r"(?:none|no\b|n/?a\b|tbd\b|unknown|not yet)", re.I)
# The PR that broke develop: "fails on develop since #101", "**Introduced by:** #102".
INTRODUCED = re.compile(r"(?:introduced by|since)\W{0,6}#(\d{4,6})", re.I)
LOGIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}")
# What the fix sweep matches on: file paths with a directory, pytest node ids (file::Class::test), test files and
# test names, references (#N, owner/repo#N, .../issues/N, .../pull/N), and a title that reads as a fix.
PATH = re.compile(
    r"(?:[\w.-]+[/\\])+[\w.-]+\.(?:py|cpp|cc|h|hpp|txt|json|cmake|ya?ml|toml|md|sh|ll|mlir)\b"
)
NODE = re.compile(r"((?:[\w.-]+[/\\])*test_\w+\.py)((?:::[\w\[\]=.,'-]+)+)")
TEST_FILE = re.compile(r"\btest_\w+\.(?:py|cpp)$")
TEST_NAME = re.compile(r"\b(?:test_\w{3,}|Test[A-Z]\w{2,})")
MENTION = re.compile(r"(?:(?<![\w/])#|[\w.-]+/[\w.-]+#|/(?:issues|pull)/)(\d{4,6})\b")
FIXY = re.compile(
    r"\b(?:hot)?fix(?:e[sd]|ing)?\b|\b(?:repair|restore|revert|unbreak)", re.I
)
KNOWN_FAIL = re.compile(
    r"\bfail\w*\s+(?:identically|also|too)\s+on\s+develop"
    r"|\bfail\w*\s+on\s+develop\s+(?:too|as\s+well)"
    r"|\b(?:same|identical(?:ly)?)\s+(?:failures?\s+)?on\s+develop"
    r"|\b(?:known|pre-?existing|unrelated|remaining)\s+failures?",
    re.I,
)


def named_fixes(body: str) -> set[int]:
    """The fix PRs an issue body names (FIX_LINE, FIX_BY)."""
    found = {int(x) for x in FIX_BY.findall(body)}
    for rest in FIX_LINE.findall(body):
        if NO_FIX.match(rest.strip()):
            continue
        listed = re.split(r"[.;]\s|\(", rest, maxsplit=1)[
            0
        ]  # the list: before the first sentence or parenthesis
        found |= {int(x) for x in re.findall(r"#(\d{4,6})\b", listed)}
    return found


def labels(cfg: dict) -> dict[str, str]:
    """Kind -> its label, as the settings name it (config `incident_issue`: `labels[0]` and `stability_label`);
    blank where they name none."""
    ii = cfg.get("incident_issue") or {}
    return {
        RED: str((ii.get("labels") or [""])[0] or "").strip(),
        STABILITY: str(ii.get("stability_label") or "").strip(),
    }


def extra_labels(cfg: dict) -> list[str]:
    """The labels a CI-red issue carries beside its kind's (config `incident_issue.labels[1:]`, the develop guard's
    extra labels); a stability issue carries none of them."""
    given = ((cfg.get("incident_issue") or {}).get("labels") or [])[1:]
    return [str(x).strip() for x in given if str(x).strip()]


def label_key(kind: str) -> str:
    """The deploy settings file's key of KIND's label, as messages name it: its file key in the registry
    (incident_issue.labels comes from ci_red_label, then the extra labels)."""
    from . import registry

    name = (
        "incident_issue.stability_label"
        if kind == STABILITY
        else "incident_issue.labels"
    )
    return registry.key(name).file_keys[0]


def blank_labels(kinds: list[str]) -> str:
    """What a refusal says of KINDS whose labels the settings leave blank: their keys, and how to set them, through
    the host's command as the operator types it (doctor.apply_words)."""
    # a call-time import: doctor is a large module, read only for a refusal
    from .doctor import apply_words

    keys, one = [label_key(k) for k in kinds], len(kinds) == 1
    edit = f"set {'it' if one else 'them'} in your deploy settings file"
    return (
        f"{' and '.join(keys)} {'is' if one else 'are'} blank: {apply_words(edit=edit)}"
    )


def kind_label(cfg: dict, kind: str) -> str:
    """KIND as reports print it: its label, as the settings name it, else its word (WORDS)."""
    return labels(cfg).get(kind) or WORDS.get(kind, kind)


def entry_kind(cfg: dict, issue: dict) -> str:
    """The kind of ISSUE, an entry of st["post_merge"]: its `kind`, a CI-red one where it has none. An entry whose
    kind is neither (one stored with its label's name as its kind) has the kind its stored labels give (`kind_of`).
    """
    kind = issue.get("kind")
    if not kind:
        return RED
    return kind if kind in (RED, STABILITY) else kind_of(cfg, issue.get("labels") or [])


def entry_label(issue: dict) -> str:
    """ISSUE's kind as reports print it: the label `poll` read with it (`kind_label`). An entry stored with no label
    prints the name its kind was stored by, or the kind's word."""
    kind = issue.get("kind") or RED
    return issue.get("label") or WORDS.get(kind, kind)


def issue_repository(cfg: dict) -> str:
    """The develop issues' repository, <org>/<repo>: the develop guard's issue repository (`incident_issue.repo`),
    else the watched repository (`repo`). Develop issues are filed, read, labelled, assigned and commented on there,
    and their labels are checked and created there."""
    given = str((cfg.get("incident_issue") or {}).get("repo") or "").strip()
    return given or cfg.get("repo") or ""


def _elsewhere(cfg: dict) -> bool:
    """Are the develop issues in another repository than the watched one?"""
    return issue_repository(cfg).lower() != (cfg.get("repo") or "").lower()


def issue_ref(cfg: dict, n: int) -> str:
    """Develop issue n as a pull request's text names it: #N, or <org>/<repo>#N when the issues are in another
    repository."""
    return f"{issue_repository(cfg)}#{n}" if _elsewhere(cfg) else f"#{n}"


def pr_ref(cfg: dict, n: int) -> str:
    """Pull request n as a develop issue's text names it: #N, or <org>/<repo>#N when the issues are in another
    repository."""
    return f"{cfg['repo']}#{n}" if _elsewhere(cfg) else f"#{n}"


def kind_of(cfg: dict, names: list[str]) -> str:
    """An issue's kind from its labels; one with both labels is a CI-red one."""
    lab = labels(cfg)
    return (
        STABILITY
        if lab[STABILITY] and lab[STABILITY] in names and lab[RED] not in names
        else RED
    )


def kind_in_state(cfg: dict, st: dict, n: int) -> str:
    """The kind of open develop issue n as last read, or "develop" when it is not among them."""
    return next(
        (
            entry_kind(cfg, i)
            for i in (st.get("post_merge") or {}).get("open") or []
            if i["n"] == n
        ),
        "develop",
    )


def fetch_open(cfg: dict) -> list[dict]:
    """The open develop issues of both kinds, newest first: one query per label, merged by number, each with its
    kind."""
    out = {}
    for lab in [x for x in labels(cfg).values() if x]:
        for i in gh.rest(
            f"repos/{issue_repository(cfg)}/issues?labels={quote(lab)}&state=open&per_page=100",
            paginate=True,
            jq="[.[] | select(.pull_request == null) | {n: .number, title, url: .html_url, "
            "author: .user.login, labels: [.labels[].name], assignees: [.assignees[].login], "
            'created: .created_at, body: ((.body // "")[0:6000])}]',
        ):
            kind = kind_of(cfg, i["labels"])
            out[i["n"]] = dict(i, kind=kind, label=kind_label(cfg, kind))
    return sorted(out.values(), key=lambda i: -i["n"])


def relevant(cfg: dict, issue: dict) -> bool:
    """Does an issue concern the watched folder: a label in scope, a shared label too (config.Scope.searched: an issue
    has no diff to check against its folder), or a title incident_issue.title_pattern matches (without regard to case;
    a blank one matches none)?"""
    scoped = set(config.scope(cfg).searched())
    pattern = registry.value(cfg, "incident_issue.title_pattern")
    return bool(scoped & set(issue.get("labels") or [])) or bool(
        pattern and re.search(pattern, issue.get("title") or "", re.I)
    )


def scan(issue: dict) -> dict:
    """What the fix sweep matches an issue on: the files it names (a path with a directory, or a pytest node id), the
    test files among them (and one its title names), the failing tests it names (in its title or a node id), and the
    PRs it names (`refs`: the culprit, a fix candidate, PRs that reported the failure, related work).
    """
    title, text = issue["title"], issue["title"] + "\n" + (issue.get("body") or "")
    nodes = NODE.findall(text)
    tests = set(re.findall(r"\btest_\w+\.py\b", title))
    files = (
        {p.replace("\\", "/") for p in PATH.findall(text)}
        | {f.replace("\\", "/") for f, _ in nodes}
        | tests
    )
    names = set(TEST_NAME.findall(title)) | {
        t for f, rest in nodes for t in TEST_NAME.findall(f + rest)
    }
    return {
        "files": sorted(files)[:60],
        "tests": sorted(f for f in files if TEST_FILE.search(f))[:30],
        "names": sorted(names)[:30],
        "refs": sorted(mentions(text) - {issue.get("n")})[:100],
    }


def issue_lines(issues: list[dict], incidents: list[dict]) -> list[str]:
    """What `rk breakages` lists: the open develop issues, rocKE ones first and newest first, with their kind, author,
    assignees and our incident (`rk facts` prints the same lines)."""
    links = {i.get("issue"): i.get("id") for i in incidents if i.get("issue")}
    return [
        f"#{i['n']} {entry_label(i)} {'rocKE ' if i['rocke'] else ''}{i['created'][:10]} {i['author']}"
        + (f" -> {','.join(i['assignees'])}" if i["assignees"] else "")
        + (f" [incident {links[i['n']]}]" if i["n"] in links else "")
        + f": {i['title'][:110]}"
        for i in sorted(issues, key=lambda i: (not i["rocke"], -i["n"]))
    ]


def poll(cfg: dict, st: dict) -> list[str]:
    """Refresh st["post_merge"]: the open develop issues of both kinds (without bodies), the fix PRs their bodies name,
    and what the fix sweep matches the rocKE ones on. Report rocKE issues that opened or closed since, with their
    kind."""
    issues = fetch_open(cfg)
    pm = st.setdefault("post_merge", {"open": [], "at": None})
    before = {i["n"]: i for i in pm.get("open") or []}
    lines = []
    for i in issues:
        i["rocke"] = relevant(cfg, i)
        i.setdefault("kind", kind_of(cfg, i.get("labels") or []))
        i.setdefault("label", kind_label(cfg, i["kind"]))
        if i["rocke"]:
            i["scan"] = scan(i)
        if i["n"] not in before and pm.get("at") and i["rocke"]:
            lines.append(
                f"BREAKAGE_OPEN #{i['n']} {entry_label(i)} ({i['author']}): {i['title'][:120]}. If it is one of "
                f"our incidents, `rk incident-note ID issue {i['n']}`"
            )
    still = {i["n"] for i in issues}
    lines += [
        f"BREAKAGE_CLOSED #{n} {entry_label(i)}: {i['title'][:120]}"
        for n, i in before.items()
        if n not in still and i.get("rocke")
    ]
    pm["open"] = [{k: v for k, v in i.items() if k != "body"} for i in issues]
    named = {i["n"]: named_fixes(i.get("body") or "") - {i["n"]} for i in issues}
    pm["fixes"] = {str(n): sorted(prs) for n, prs in named.items() if prs}
    pm["at"] = now()
    return lines


# The test names every repository's issues name, which identify a breakage: test functions and test classes.
TEST_NAMES = r"\btest_\w{4,}|\bTest[A-Z]\w{3,}"


def _test_names(cfg: dict, text: str) -> set[str]:
    """The test names in TEXT, each whole: test functions and test classes, and the repository's own
    (incident_issue.test_name_pattern)."""
    own = registry.value(cfg, "incident_issue.test_name_pattern")
    pattern = TEST_NAMES + (f"|(?:{own})" if own else "")
    return {m.group(0) for m in re.finditer(pattern, text)}


def overlaps(cfg: dict, a: str, b: str) -> list[str]:
    """What two texts share that identifies a breakage: PR numbers, and test names (`_test_names`) but those that run
    most of the tests (incident_issue.umbrella_test_names: naming one identifies no breakage, else every issue that
    names one reads as a possible duplicate of every other)."""
    umbrella = set(registry.value(cfg, "incident_issue.umbrella_test_names") or ())
    return sorted(f"#{r}" for r in set(REF.findall(a)) & set(REF.findall(b))) + sorted(
        (_test_names(cfg, a) & _test_names(cfg, b)) - umbrella
    )


def duplicates(
    cfg: dict, title: str, body: str
) -> list[tuple[int, str, list[str], str]]:
    """Open develop issues of either kind that share a PR number or a test name with a draft: (n, title, shared,
    kind)."""
    out = []
    for i in fetch_open(cfg):
        shared = overlaps(
            cfg, i["title"] + "\n" + (i.get("body") or ""), title + "\n" + body
        )
        if shared:
            out.append((i["n"], i["title"], shared, i["kind"]))
    return out


def parse_file(path: str, cfg: dict | None = None) -> dict:
    """The front matter and body of an `rk issue-file` file. The labels must hold exactly one of the two kinds' labels,
    and a stability issue names the breaking PR's author in `assign:`."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not m:
        raise ValueError(
            "the file must start with a --- block holding title: and labels: (see breakage.py)"
        )
    meta = {
        k.strip().lower(): v.strip()
        for k, v in (ln.split(":", 1) for ln in m.group(1).splitlines() if ":" in ln)
    }
    labs = [lab.strip() for lab in meta.get("labels", "").split(",") if lab.strip()]
    names = labels(cfg or {})
    blank = [k for k, lab in names.items() if not lab]
    if blank:
        raise ValueError(blank_labels(blank))
    kinds = [k for k, lab in names.items() if lab in labs]
    if len(kinds) != 1:
        raise ValueError(
            f"labels: must carry exactly one of {names[RED]} (a develop break that makes CI red) "
            f"or {names[STABILITY]} (a break CI does not see, assigned to the breaking PR's author); got "
            f"{'both' if kinds else 'neither'}"
        )
    assign = [
        a.strip().lstrip("@") for a in meta.get("assign", "").split(",") if a.strip()
    ]
    bad = [a for a in assign if not LOGIN.fullmatch(a)]
    if bad:
        raise ValueError(f"assign: not GitHub logins: {', '.join(bad)}")
    if kinds[0] == STABILITY and not assign:
        raise ValueError(
            f"a {names[STABILITY]} issue is assigned to the author of the breaking PR: add `assign: LOGIN`"
        )
    extra = [x for x in extra_labels(cfg or {}) if x in labs]
    if kinds[0] == STABILITY and extra:
        raise ValueError(
            f"labels: {', '.join(extra)} {'goes' if len(extra) == 1 else 'go'} with {names[RED]} only: a "
            f"{names[STABILITY]} issue is a break "
            "CI does not see"
        )
    return {
        "title": meta.get("title", ""),
        "labels": labs,
        "kind": kinds[0],
        "assign": assign,
        "incident": meta.get("incident") or None,
        "body": text[m.end() :].lstrip("\n"),
    }


def mentions(text: str) -> set[int]:
    return {int(x) for x in MENTION.findall(text or "")}


def fix_names(body: str) -> list[str]:
    """The test names a PR body names as its own work. A sentence that lists known failures ("fail identically on
    develop", "known failures", "remaining failures") does not count: a PR that lists the failures it shares with
    develop is no fix to them."""
    keep = [
        s for s in re.split(r"(?<=[.!?])\s+|\n", body or "") if not KNOWN_FAIL.search(s)
    ]
    return sorted(set(TEST_NAME.findall("\n".join(keep))))[:200]


def same_file(named: str, changed: str) -> bool:
    """Is a changed file (repo path) a file an issue names? Their last components agree: all of a short name's, three of
    a longer one's, so a path from another root (the CTest copy `tests/library/tests/...`) still matches.
    """
    a = named.split("/")
    k = min(len(a), 3)
    return a[-k:] == changed.split("/")[-k:]


def signals(cfg: dict, issue: dict, rec: dict, sc: dict) -> list[str]:
    """Why PR rec looks like a fix for the issue; empty when nothing holds:
    a) its title or body mentions #N;
    b) it changes a test file the issue names, and its title reads as a fix;
    c) its title or body names a failing test the issue names, and it changes a file the issue names; naming only the
       test file's module needs a change outside the test files.
    A PR the issue names (the culprit, the fix, PRs that reported the failure, related work) counts only by a): the
    issue's author already weighed it. Without these limits, b) and c) flag feature PRs that merely share a test file
    with the issue, or list its failing tests as known: none a fix."""
    n, s, body = issue["n"], issue["scan"], sc.get("body") or {}
    root = paths.repository_folder(cfg) + "/"

    def short(files: list[str]) -> str:
        return ", ".join(f[len(root) :] if f.startswith(root) else f for f in files)

    out = [
        f"its {where} mentions #{n}"
        for where, refs in (
            ("title", mentions(rec["title"])),
            ("body", body.get("refs") or []),
        )
        if n in refs
    ]
    if rec["n"] in s.get("refs", []):
        return out
    hits = sc["hits"].get(str(n)) or []
    tests = [f for f in hits if any(same_file(t, f) for t in s["tests"])]
    if tests and FIXY.search(rec["title"]):
        out.append(f"its title reads as a fix and it changes {short(tests[:3])}")
    named = sorted(
        set(s["names"])
        & (set(TEST_NAME.findall(rec["title"])) | set(body.get("names") or []))
    )
    modules = {pathlib.PurePosixPath(t).stem for t in s["tests"]}
    code = [f for f in hits if not TEST_FILE.search(f)]
    if (code and named) or (hits and set(named) - modules):
        out.append(
            f"it names {', '.join(named[:3])} and changes {short((code or hits)[:2])}"
        )
    return out


def fix_sweep(cfg: dict, st: dict) -> list[str]:
    """Open PRs that look like fixes to the open rocKE develop issues (once verified, they are told so and go to the top
    of the review queue). A PR
    qualifies for an issue when it is open, in scope and not a draft, is not linked to it yet (a closing keyword, the
    issue naming it as the fix, `rk fix-link`) nor dismissed, and `signals` holds.

    Only data the tick has: changed files from the local PR refs, matched per head and issue set; the PR's body is read
    once per head, and only when its title or files already match an issue. Each candidate prints a FIX_CANDIDATE line
    every tick until the orchestrator verifies it and links it (`rk fix-link`) or dismisses it (`rk fix-dismiss`).
    """
    from .queue import DEVELOP, ensure_pr_ref

    issues = [
        i
        for i in (st.get("post_merge") or {}).get("open") or []
        if i.get("rocke") and i.get("scan")
    ]
    fp = hashlib.sha1(
        json.dumps([[i["n"], i["scan"]["files"]] for i in issues]).encode()
    ).hexdigest()[:12]
    done = {
        (int(k), n)
        for d in (st.get("fix_links") or {}, st.get("fix_dismissed") or {})
        for k, ns in d.items()
        for n in ns
    }
    fixes = st.get("develop_fixes") or {}
    lines = []
    for k, rec in sorted(st["prs"].items(), key=lambda kv: int(kv[0])):
        if rec.get("status") != "open":
            rec.pop("fix_scan", None)
            continue
        if rec.get("draft") or not issues:
            continue
        sc = rec.get("fix_scan") or {}
        if sc.get("head") != rec["head"] or sc.get("issues") != fp:
            try:
                files = gitops.changed_files(
                    cfg["paths"]["repo"], DEVELOP, ensure_pr_ref(cfg, rec)
                )
            except (gitops.GitError, OSError):
                continue
            hits = {
                str(i["n"]): [
                    f for f in files if any(same_file(p, f) for p in i["scan"]["files"])
                ][:10]
                for i in issues
            }
            sc = rec["fix_scan"] = {
                "head": rec["head"],
                "issues": fp,
                "hits": {n: v for n, v in hits.items() if v},
                "body": sc.get("body") if sc.get("head") == rec["head"] else None,
            }
        for i in issues:
            n, kind = i["n"], entry_label(
                i
            )  # the kind as its label, as the develop fixes name it
            if (
                (n, rec["n"]) in done
                or n in (rec.get("closes") or [])
                or f"{kind} #{n}" in fixes.get(k, [])
            ):
                continue
            if not (
                sc["hits"].get(str(n))
                or n in mentions(rec["title"])
                or set(i["scan"]["names"]) & set(TEST_NAME.findall(rec["title"]))
            ):
                continue
            if sc.get("body") is None:
                try:
                    body = gh.rest(
                        f"repos/{cfg['repo']}/pulls/{rec['n']}",
                        jq='{body: (.body // "")}',
                    )["body"]
                except gh.GhError:
                    break  # the next tick tries this PR again
                sc["body"] = {
                    "refs": sorted(mentions(body))[:100],
                    "names": fix_names(body),
                }
            sig = signals(cfg, i, rec, sc)
            if sig:
                lines.append(
                    f"FIX_CANDIDATE #{rec['n']} -> #{n} ({kind}): {'; '.join(sig)}. Verify it, then "
                    f"`rk fix-link {n} {rec['n']} [--evidence FILE]`; or `rk fix-dismiss {n} {rec['n']}`"
                )
    return lines


LISTED = "FIX_LISTED"


def listed_only(lines: list[str]) -> list[str]:
    """With fix-links off (or not set up): each FIX_CANDIDATE line as a FIX_LISTED line for the operator, which asks
    nothing of the orchestrator: no PR is linked, told or moved up."""
    out = []
    for line in lines:
        if line.startswith("FIX_CANDIDATE "):
            head, _, rest = line[len("FIX_CANDIDATE ") :].partition(": ")
            out.append(
                f"{LISTED} {head}: {rest.split('. Verify it,')[0]} (fix links are off: listed for you; "
                "nothing is linked)"
            )
        else:
            out.append(line)
    return out


def fix_dismiss(st: dict, store, issue: int, pr: int) -> str:
    """Stop the FIX_CANDIDATE line for (issue, pr): local only. A PR linked by mistake is unlinked too: it is no develop
    fix any more, and the comments already posted stay."""
    from . import guard, queue

    seen = st.setdefault("fix_dismissed", {}).setdefault(str(issue), [])
    if pr not in seen:
        seen.append(pr)
    links = st.get("fix_links") or {}
    unlinked = pr in (links.get(str(issue)) or [])
    if unlinked:
        links[str(issue)].remove(pr)
        if not links[str(issue)]:
            del links[str(issue)]
        queue.refresh_develop_fixes(st, guard.incidents(store))
    store.journal(
        "fix",
        f"#{pr} dismissed as a fix for #{issue}"
        + (" and unlinked" if unlinked else ""),
    )
    return f"#{pr} dismissed as a fix candidate for #{issue}" + (
        "; unlinked: it is no develop fix any more (the comments already posted stay)"
        if unlinked
        else ""
    )


# --- the change fingerprint's part (idle.py) --------------------------------------------------------------------------

_ISSUES = """
query($q: String!, $cursor: String) {
  search(query: $q, type: ISSUE, first: 100, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes { ... on Issue { number updatedAt title labels(first: 30) { nodes { name } } } }
  }
}"""


@idle.part("develop issues", remote=True)
def _issues(cfg: dict, st: dict, store) -> list[list]:
    """The open develop issues of both kinds that `poll` reports (relevant): number and last update."""
    names = [x for x in labels(cfg).values() if x]
    if not names:
        return []
    q = snapshot.label_query(cfg, names, kind="issue", repo=issue_repository(cfg))
    out = []
    for data in snapshot.search_pages(q, first=100, query=_ISSUES):
        for node in data["nodes"]:
            if node and relevant(
                cfg,
                {
                    "title": node["title"],
                    "labels": [n["name"] for n in node["labels"]["nodes"]],
                },
            ):
                out.append([node["number"], node["updatedAt"]])
    return sorted(out)
