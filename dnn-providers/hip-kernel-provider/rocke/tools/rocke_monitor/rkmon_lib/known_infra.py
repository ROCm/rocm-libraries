# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Known CI infrastructure failures, for a merge-only follow-up (review.advance).

A merge-only head whose only flag is CI failing at the head and not at our reviewed head is recorded as covered, like
a clean merge-only head, when every one of those failing jobs is known infrastructure:
- its log holds a known signature (`signatures`, below) whose job scope takes the job and that has not ended for this
  head; or
- its first error line is also the first error line of the same job (by name) failing at develop's latest SHA; or
- of the same job failing at the heads of at least OTHER_PRS other open PRs, its run completed today (UTC).

The signatures are data: config `ci_infra_signatures` (a repository's own, from its repository defaults), then the
state's `site.ci_infra_signatures` (the configuration's `ci_infra_site_signatures`: the failures this site's own
infrastructure causes, learned on this site), whose entry with the same pattern replaces the repository's. An entry is
a plain substring, or an object:
- `pattern`: the text the job's log holds (a plain substring; a regular expression when `regex` is true);
- `jobs`: a regular expression the job's name must match (searched, case-insensitive); blank: any job;
- `unless`: a regular expression that, when the log also holds it, leaves the job unknown (the PR's own failures
  beside the infrastructure's);
- `ends_at`: a develop commit that ends the signature: a head that contains it (the commit is an ancestor of the
  head, so the head's merge base with develop has it) is past the fix, and the signature no longer counts there. An
  ancestry that cannot be read counts as ended (the job goes to verification);
- `note`: what it is, for people.
An entry with a bad regular expression is left out.

A job's first error line is its log's first `##[error]` line, or else its first line that reads as a failure
(ci_runs.LOG_HITS), with the timestamp cut; the runner's closing "Process completed with exit code N" says nothing
about the cause and is skipped. Any read that fails (a check-run list, a log) leaves the job unknown, so the head goes
to verification as before. GETs only.
"""

from __future__ import annotations

import datetime as dt
import re

from . import ci_runs, gh

OTHER_PRS = 2  # other open PRs failing the same job the same way today
OTHER_PRS_READ = (
    25  # at most this many other open PRs' check runs are read, newest heads first
)
TIMESTAMP = re.compile(r"^\ufeff?\d{4}-\d\d-\d\dT[\d:.]+Z\s?")
ERROR = re.compile(r"##\[error\]")
GENERIC = re.compile(r"Process completed with exit code \d+")
JOB = re.compile(r"/job/(\d+)")


def _entry(raw) -> dict | None:
    """One signature, normalized: {pattern, regex, jobs, unless, ends_at, note, rx, jobs_rx, unless_rx}; None when
    it has no pattern or a regular expression does not compile."""
    if isinstance(raw, str):
        raw = {"pattern": raw}
    if not isinstance(raw, dict) or not str(raw.get("pattern") or "").strip():
        return None
    e = {
        "pattern": str(raw["pattern"]),
        "regex": bool(raw.get("regex")),
        "jobs": str(raw.get("jobs") or ""),
        "unless": str(raw.get("unless") or ""),
        "ends_at": str(raw.get("ends_at") or "").strip(),
        "note": str(raw.get("note") or ""),
    }
    try:
        e["rx"] = re.compile(e["pattern"] if e["regex"] else re.escape(e["pattern"]))
        e["jobs_rx"] = re.compile(e["jobs"], re.I) if e["jobs"] else None
        e["unless_rx"] = re.compile(e["unless"]) if e["unless"] else None
    except re.error:
        return None
    return e


def signatures(cfg: dict) -> list[dict]:
    """The known signatures: config `ci_infra_signatures`, then the state's `site.ci_infra_signatures` (the
    configuration's `ci_infra_site_signatures`); a site entry replaces the entry with the same pattern.
    """
    out: dict[str, dict] = {}
    for raw in [
        *(cfg.get("ci_infra_signatures") or []),
        *(cfg.get("ci_infra_site_signatures") or []),
    ]:
        e = _entry(raw)
        if e:
            out[e["pattern"]] = e
    return list(out.values())


def match(sigs: list[dict], name: str, text: str, ended=None) -> dict | None:
    """The first signature that marks job `name`'s failure (its log `text`) as known infrastructure: its pattern in the
    log, its job scope taking the name, no `unless` text in the log, and not ended (`ended(sha)`: whether the head
    contains that develop commit; without it no signature has ended)."""
    for e in sigs:
        if not e["rx"].search(text):
            continue
        if e["jobs_rx"] and not e["jobs_rx"].search(name):
            continue
        if e["unless_rx"] and e["unless_rx"].search(text):
            continue
        if e["ends_at"] and ended is not None and ended(e["ends_at"]):
            continue
        return e
    return None


class Ancestry:
    """Whether a head contains a develop commit (a signature's `ends_at`), read once per pair through GitHub's compare
    (GET): "ahead" or "identical" means the commit is the head's ancestor. A read that fails counts as contained, so
    an ended signature never hides a failure."""

    def __init__(self, cfg: dict, head: str):
        self.cfg, self.head, self.seen = cfg, head, {}

    def __call__(self, sha: str) -> bool:
        if sha not in self.seen:
            try:
                status = gh.rest(
                    f"repos/{self.cfg['repo']}/compare/{sha}...{self.head}",
                    jq=".status",
                )
                self.seen[sha] = status in ("ahead", "identical")
            except gh.GhError:
                self.seen[sha] = True
        return self.seen[sha]


def first_error(text: str) -> str:
    """The log's first error line, timestamp and `##[error]` cut, white space collapsed; "" when it has none."""
    lines = [TIMESTAMP.sub("", ln).rstrip("\r") for ln in text.split("\n")]
    for pick in (
        lambda ln: ERROR.search(ln),
        lambda ln: ci_runs.LOG_HITS.search(ln) and "error: 0" not in ln,
    ):
        for ln in lines:
            if pick(ln) and not GENERIC.search(ln):
                return re.sub(r"\s+", " ", ERROR.sub("", ln)).strip()[:200]
    return ""


def job_id(run: dict) -> int | None:
    m = JOB.search(run.get("url") or "")
    return int(m.group(1)) if m else None


class _Logs:
    """Job logs read once per classification."""

    def __init__(self, cfg: dict):
        self.cfg, self.text = cfg, {}

    def get(self, run: dict) -> str | None:
        job = job_id(run)
        if job is None:
            return None
        if job not in self.text:
            try:
                self.text[job] = gh.raw(
                    f"repos/{self.cfg['repo']}/actions/jobs/{job}/logs"
                )
            except gh.GhError:
                self.text[job] = None
        return self.text[job]


def _failing(cfg: dict, sha: str) -> dict[str, dict]:
    return {
        name: r
        for name, r in ci_runs.latest(ci_runs.check_runs(cfg, sha)).items()
        if ci_runs.failed(r)
    }


def classify(
    cfg: dict, st: dict, n: int, head: str, names: list[str], today: str | None = None
) -> dict | None:
    """{job name: why it is known infrastructure} when every one of `names` (jobs failing at PR n's `head`) is; None
    when one is not, or cannot be told."""
    today = today or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    logs = _Logs(cfg)
    try:
        at_head = _failing(cfg, head)
    except gh.GhError:
        return None
    sigs, ended = signatures(cfg), Ancestry(cfg, head)
    known: dict[str, str] = {}
    develop, others = None, None
    for name in names:
        run = at_head.get(name)
        text = logs.get(run) if run else None
        if text is None:
            return None
        sig = match(sigs, name, text, ended)
        if sig:
            known[name] = f"signature `{sig['pattern']}`"
            continue
        first = first_error(text)
        if not first:
            return None
        if develop is None:
            dev_sha = (st.get("develop") or {}).get("sha")
            try:
                develop = (dev_sha, _failing(cfg, dev_sha)) if dev_sha else ("", {})
            except gh.GhError:
                develop = ("", {})
        dev_sha, dev_failing = develop
        if (
            name in dev_failing
            and first_error(logs.get(dev_failing[name]) or "") == first
        ):
            known[name] = f"develop {dev_sha[:10]} fails the same: `{first}`"
            continue
        if others is None:
            others = _other_heads(st, n)
        same = []
        for k, sha in others:
            try:
                r = _failing(cfg, sha).get(name)
            except gh.GhError:
                continue
            if (
                r
                and (r.get("completed") or "")[:10] == today
                and first_error(logs.get(r) or "") == first
            ):
                same.append(k)
                if len(same) >= OTHER_PRS:
                    break
        if len(same) < OTHER_PRS:
            return None
        known[name] = f"the same on {', '.join(f'#{k}' for k in same)} today: `{first}`"
    return known


def _other_heads(st: dict, n: int) -> list[tuple[int, str]]:
    """Other open PRs' heads, the most recently pushed first, at most OTHER_PRS_READ."""
    recs = [
        r
        for k, r in st.get("prs", {}).items()
        if int(k) != n and r.get("status") == "open" and r.get("head")
    ]
    recs.sort(key=lambda r: r.get("head_seen_at") or "", reverse=True)
    return [(int(r["n"]), r["head"]) for r in recs[:OTHER_PRS_READ]]


def line(n: int, head: str, known: dict) -> str:
    """The tick line: the PR, each job and why it is known infrastructure."""
    return (
        f"CI_INFRA #{n} at {head[:10]}: merge-only, and its only flag is known CI infrastructure: "
        + "; ".join(f"{name} ({why})" for name, why in known.items())
    )
