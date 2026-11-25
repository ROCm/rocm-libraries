# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The only path from the toolkit to a GitHub comment. The Claude orchestrator calls it after verification.

Guards, all hard (a refusal names the rule):
- Line 1 of a review must be the line its review was staged with (`review_line`), `Automated review by
  <automated_reviewer_name>: Model - <model>, effort: <effort>`, which names the model that drafted it; with no staging
  of the review's head, the configured line (bindings.first_line: the drafter's `review_first_line`, or with no
  first-pass drafter the verifier's). A notice's first line must be its kind's, `<what it is> by
  <automated_reviewer_name>` (notice_line).
- Public repo: no URL outside the allow-list (internal hosts, SharePoint and Confluence links are refused), no absolute
  performance numbers, and no ticket tag inside a link.
- No duplicate: one post per PR head unless `again` is given, at most `max_posts_per_hour`, and every post is recorded
  in the ledger.

Posting is a plain issue comment (`gh pr comment`). Nothing here approves or dismisses a review. The one review the
toolkit submits is a change request on a held PR (`request_changes`), so a held PR
cannot merge until it is fixed and the user re-approves it. Before a review goes out, the automated reviewer name its
line 1 gives is recorded in the internal state (`record_name`), so that this deployment's reviews stay its own after
the name changes; an internal state it cannot read or write stops the post, refused with the fix.

The other writes, each guarded and ledgered the same way: notices (incident, queue, fix), arm/disarm, `automerge_off`
(auto-merge someone else enabled on a PR the queue knows is unsafe), develop issues of both kinds, CI-red and
stability (`issue_file`, `issue_comment`, `issue_relabel`), a verified fix linked to its issue (`fix_link`), and
the Teams alert for a CI-red develop break with no planned fix (`teams_breakage`). The Needs-you list's copy to the
same channel (needs_teams.py) goes through `teams_message` and `teams_post` here; each message opens with the line
that names the deployment that sends it (`automated_line`). In the container the writer alone reads the channel's
webhook, and sends each message (`teams_post`: `rk teams-send`, whose half is `teams_send`).

Every GitHub write here runs gh under the write login (`_gh_write`, gh.write); the rest of the toolkit, the
orchestrator session, its agents and every job read under the read login, whose writes GitHub refuses (gh.py).
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
from urllib.parse import quote

from . import config, daylog, derived, idle, outward, paths, registry, reviewer_name
from . import state_store
from . import gh as gh_cli
from .privacy import (
    PERF,
)  # one pattern for an absolute performance number: the posts' guard and the package's gate
from .state_store import (  # the types of unsent item
    ALERT,
    AUTO_MERGE,
    CHANGE_REQUEST,
    COMMENT,
    ISSUE,
    LABEL,
    REVIEW,
    TICKET,
)
from .store import minutes_since, now

# Every outward action is a ledger line, and one makes the next tick full (idle.py).
idle.file_part("ledger", lambda store: store.ledger_path)

ALLOWED_HOSTS = (
    "github.com",
    "rocm.docs.amd.com",
    "docs.amd.com",
    "llvm.org",
    "github.blog",
)
URL = re.compile(r"https?://([^/\s)>\]`\"']+)[^\s)>\]`\"']*")
# A ticket tag: KEY-NNN for the configured keys (config.tracker_keys), else any Jira-style key.
ANY_KEY = r"[A-Z][A-Z0-9]{1,9}"


def ticket_tag(cfg: dict | None = None) -> re.Pattern:
    keys = config.tracker_keys(cfg or {})
    return re.compile(
        r"\b(?:" + ("|".join(map(re.escape, keys)) if keys else ANY_KEY) + r")-\d+\b"
    )


def ticket_link(cfg: dict | None = None) -> re.Pattern:
    """A ticket tag inside a Markdown link, or in a URL: tags stay bare in a public repository."""
    tag = ticket_tag(cfg).pattern
    return re.compile(r"\[[^\]]*" + tag + r"[^\]]*\]\(|https?://\S*" + tag)


UNPOSTED = "unposted"  # the kind of our review record (rec["ours"]) that save_draft kept, unposted
# The kind in a kept review's file name, PR_<KIND>_<N>[_<k>].md in the outputs folder, as a post's copy is named by
# its round (FIRST_ROUND, FOLLOWUP), and the first word of the line that says a review was kept (kept_line). An older
# monitor named the file by KEPT_BEFORE, PR_DRAFT_<N>.md, and is_kept_file knows it still.
KEPT_KIND, KEPT_BEFORE = "UNPOSTED", "DRAFT"


def our_review(
    rec: dict, body: str, head: str, url: str | None = None, file: str = ""
) -> dict:
    """Our review of PR rec at `head`, as its record (rec["ours"]) from now on: posted as the comment at `url` (the URL
    gh printed), or, with url None, kept unposted (kind UNPOSTED, at the time it was kept) in the outputs folder's
    `file` (kept_file). The round logic reads both alike: the head is covered, a new head gets a follow-up, and
    `count` numbers our reviews of the PR.
    """
    from .snapshot import parse_verdict

    posted = url is not None
    return {
        "kind": "comment" if posted else UNPOSTED,
        "id": None,
        "at": now(),
        "url": url,
        "verdict": parse_verdict(body),
        "line1": body.splitlines()[0] if body.strip() else "",
        "count": ((rec.get("ours") or {}).get("count") or 0) + 1,
        "head": head,
        "head_src": "ledger" if posted else UNPOSTED,
        **({} if posted else {"file": file}),
    }


def kept_unposted(ours: dict | None) -> bool:
    """Whether our last review of a PR (rec["ours"]) is one save_draft kept unposted."""
    return (ours or {}).get("kind") == UNPOSTED


def kept_file(cfg: dict, ours: dict | None) -> pathlib.Path | None:
    """The file of a review kept unposted (rec["ours"] of kind UNPOSTED) in the outputs folder; None for any other."""
    if not kept_unposted(ours) or not ours.get("file"):
        return None
    return posted_dir(cfg) / ours["file"]


def covered_head(st: dict, n: int, head: str = "") -> str:
    """The head a review of PR n kept now covers: the head it was handed to verification at (its entry awaiting
    verification), else HEAD, the caller's; "" when neither is known."""
    entry = next((r for r in st.get("review_ready") or [] if r.get("pr") == n), None)
    return (entry or {}).get("head") or head


def save_draft(cfg: dict, st: dict, store, n: int, path: str, head: str = "") -> str:
    """Posting off (the review-posting feature not active): the verified review is kept unposted, a file in the outputs
    folder, with the GPU tests row when remote GPU tests ran for the head it covers (gpu.for_post), as its post would
    carry it. Nothing reaches GitHub, nothing is ledgered, and nothing posts it later. While review automation sends
    nothing (review_held: prototype, or standby) the file is an unsent item too, in STORE's state (keep). On a tracked
    PR it is logged in the day log as its post would be (daylog.from_review), of kind unposted. It is finished as a
    post finishes it (review.review_finished): the review PR n awaits verification for holds nothing in flight, and it
    is our last review of the PR (our_review, kind UNPOSTED, its record naming the file) at the head it covers
    (covered_head): the head it was handed to verification at, else `head`, the caller's (the PR's head for a keep by
    hand, as a post records it; the verified head for a checked post). With neither it is not recorded.
    """
    from . import gpu  # a call-time import, as post's own

    covered = covered_head(st, n, head)
    text = pathlib.Path(path).read_text(encoding="utf-8")
    body = gpu.for_post(st, n, covered, text) if covered else text
    out = posted_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    dest, i = out / f"PR_{KEPT_KIND}_{n}.md", 2
    while dest.exists():
        dest, i = out / f"PR_{KEPT_KIND}_{n}_{i}.md", i + 1
    if body == text:
        shutil.copyfile(path, dest)
    else:  # with the GPU tests row
        dest.write_text(body, encoding="utf-8")
    store.journal("post", f"#{n} kept unposted, posting is off: {dest}")
    held = review_held(cfg)
    if held:
        item = unsent_item(REVIEW, f"pull request {n}", file=str(dest))
        keep(dict(item, feature=REVIEWS_SEND_FOR, why=held), store)
    rec = st["prs"].get(str(n))
    # Its round as a post of it would be noted, read before the review is finished: that changes our last review and
    # what is in flight.
    said = round_word(post_note(st, n)) if rec is not None else ""
    if covered:
        # A call-time import: review loads the review pipeline's modules.
        from . import review

        ours = (
            our_review(rec, body, covered, file=dest.name) if rec is not None else None
        )
        review.review_finished(st, n, ours)
    if rec is not None:
        daylog.from_review(store, cfg, n, said, body, None)
    return str(dest)


POSTING = "review-posting"  # the capability that posts a verified review (features.CAPABILITIES)


def keeps(cfg: dict) -> bool:
    """Whether `rk post` and `rk spec-post` keep a verified review unposted (save_draft) rather than post it: review
    posting (POSTING) is off or not set up, outside a mirror (where mirror.divert writes it to the mirror's folder).
    """
    # Call-time imports: features reads the configuration's modules, and mirror imports this module.
    from . import features, mirror

    return not mirror.on(cfg) and not features.active(cfg, POSTING)


def last_review_at(rec: dict | None, head: str) -> str:
    """Our last review of PR rec (rec["ours"]), kept unposted or posted, when it is at `head` itself, in words: "kept
    unposted: <its file>" or "posted"; "" when it is at another head, or there is none. A keep there is a second review
    at that head, as a second post at a head we posted on is (preflight). A head recorded as equivalent to it since
    (snapshot.covers) is another head: a review there is a CI update (post_note)."""
    from .snapshot import same_commit

    ours = (rec or {}).get("ours") or {}
    if not same_commit(ours.get("head"), head):
        return ""
    if not kept_unposted(ours):
        return "posted"
    return f"kept unposted: {ours['file']}" if ours.get("file") else "kept unposted"


def keep_as_draft(
    cfg: dict,
    st: dict,
    store,
    n: int,
    path: str,
    again: bool = False,
    own=None,
    cost: bool = True,
) -> str:
    """Posting off (review automation in prototype or standby, or not set up): the verified review at `path` is kept
    unposted, never posted (save_draft: with no review awaiting verification, at the PR's head, as a post is recorded);
    the line that says so. One you type while review automation sends nothing is refused (keep_check), and so is one
    of a PR in review_skip (skip_check). A keep at the head our last review of PR n is at (last_review_at) is refused
    (PostRefused) unless `again` (a re-review), as a second post at a head we posted on is. The keep's line of
    state/review_cost.jsonl (review_cost.record_keep) counts it as a review, as a checked post's line does; `cost`
    False leaves the line to the caller (premises.finish, which has the post spec's tokens).
    """
    # Call-time imports: prep and review_cost load modules that import this one.
    from . import prep, review_cost

    keep_check(cfg, own)
    skip_check(st, n)
    rec = st["prs"].get(str(n)) or {}
    was = "" if again else last_review_at(rec, rec.get("head") or "")
    if was:
        raise PostRefused(
            f"our last review of #{n} is at head {rec['head'][:10]} already ({was}) (use --again if intended)"
        )
    head = covered_head(st, n, rec.get("head") or "")
    # Its round, read before the keep: the keep is one more of our reviews.
    rnd = prep.default_round(st, n)
    kept = save_draft(cfg, st, store, n, path, head=head)
    if cost:
        review_cost.record_keep(cfg, st, store, n, rnd, head, kept)
    return kept_line(cfg, kept)


REVIEWS_SEND_FOR = (
    "review_automation"  # the feature a review post sends for (features.SENDS)
)


def review_held(cfg: dict) -> str:
    """Why a verified review is not posted now while review automation is enabled (features.held): "prototype" or
    "standby"; "" while it posts, and while review automation is off (a review kept then is no unsent item).
    """
    from . import (
        features,
    )  # a call-time import: features reads the configuration's modules

    if not features.enabled(cfg, REVIEWS_SEND_FOR):
        return ""
    return features.held(cfg, REVIEWS_SEND_FOR)


def keep_check(cfg: dict, own=None) -> str:
    """The one outward gate (outward_check) for a verified review kept unposted (keeps), before anything is kept: while
    review automation sends nothing (review_held), one you type (outward.own) is refused, naming the mode or standby,
    and the monitor's own goes on, kept as an unsent item (save_draft). Why its review is held, or "".
    """
    if not review_held(cfg):
        return ""
    return outward_check(
        cfg, None, "a review post", (), own=own, feature=REVIEWS_SEND_FOR
    )


def skip_check(st: dict, n: int) -> None:
    """A review of PR n is neither posted nor kept while the PR is in review_skip (`rk review-skip N`), as its review
    does not start then (review.first_wait): PostRefused, naming the fix."""
    if n in (st.get("review_skip") or []):
        raise PostRefused(f"#{n} is in review_skip: run rk review-unskip {n} first")


def kept_line(cfg: dict, saved: str) -> str:
    """The line that says the verified review was kept as an unposted review, never posted: at `saved` (save_draft); it
    starts with KEPT_KIND (is_kept_line). It says why, in the words that say what posts the reviews after it: while
    review automation sends nothing (review_held), the mode or standby (features.held_words), and the review's unsent
    item; while review posting (POSTING) is off, what it runs while (features.why_off: review automation enabled, its
    mode live), as the outward gate refuses a post typed then; while it is on and not set up, what it lacks and the
    fix (features.missing, features.fix), as features.refusal says them. It never asks for a post by hand: the monitor
    records only what it posts.
    """
    from . import (
        features,
    )  # a call-time import: features reads the configuration's modules

    if review_held(cfg):
        return (
            f"{KEPT_KIND}: {features.held_words(cfg, REVIEWS_SEND_FOR)}, so the verified review stays unposted, "
            f"kept as an unsent item: {saved}."
        )
    posting = features.lowered(features.CAPABILITIES[POSTING].title)
    if not features.on(cfg, POSTING):
        why = f"is off ({features.why_off(cfg, POSTING)})"
    else:
        lacks = "; ".join(features.missing(cfg, POSTING))
        why = f"is on but needs {lacks} (fix: {features.fix(cfg, POSTING)})"
    return (
        f"{KEPT_KIND}: {posting} {why}, so the verified review stays unposted: {saved}."
    )


def is_kept_line(line: str) -> bool:
    """Whether LINE says a verified review was kept unposted (kept_line): what `rk post` prints for a post its gate
    holds since the post's checks (rkmon_lib/premises.py)."""
    return line.startswith(f"{KEPT_KIND}: ")


def posted_dir(cfg: dict) -> pathlib.Path:
    """Where each posted review's copy and each review kept unposted go, PR_<KIND>_<N>[_<k>].md: the outputs folder
    (`paths.posted_dir`)."""
    return pathlib.Path(paths.get(cfg, "posted_dir"))


def copies(cfg: dict, n: int) -> list[pathlib.Path]:
    """Our review files of PR n in the outputs folder (posted_dir), oldest first: each post's copy and each review kept
    unposted (KEPT_KIND)."""
    name = re.compile(rf"PR_[A-Z_]+_{n}(_\d+)?\.md")
    return sorted(
        (p for p in posted_dir(cfg).glob(f"PR_*_{n}*.md") if name.fullmatch(p.name)),
        key=lambda p: p.stat().st_mtime,
    )


def is_kept_file(p: pathlib.Path) -> bool:
    """Whether a file of copies() is a review kept unposted (KEPT_KIND, or an older monitor's KEPT_BEFORE), not a
    post's copy."""
    return p.name.startswith((f"PR_{KEPT_KIND}_", f"PR_{KEPT_BEFORE}_"))


class PostRefused(RuntimeError):
    pass


def _gh_write(args: list[str], timeout: float = 120, **_kw):
    """Every GitHub write of the toolkit: gh with `args` (gh's own arguments) under the write login (gh.write), after
    the caller's guards. gh.GhError before gh runs in a mirror, or when the write login's folder is the read one's.
    """
    return gh_cli.write(args, timeout)


def write_login() -> str:
    """What a dry run says of the post it stopped before: the login it would go out under."""
    d = gh_cli.config_dir("write")
    return (
        f"it would post under the write login (GH_CONFIG_DIR {d})"
        if d
        else "it would post under gh's own login (no read/write split)"
    )


def github_readable(cfg: dict, what: str, held: str) -> bool:
    """Whether the checks of WHAT, a GitHub write, may read GitHub here: not where this process reads GitHub with the
    write login's folder, as the writer's rk does (gh.login_dirs), while that folder holds no login (github_write is
    not set, as in prototype). Then a write that would go out (HELD "") is refused first, naming the command that sets
    the login; a held one (HELD: outward_check's word) makes its other checks and is kept as its unsent item.
    """
    from . import secrets_registry as sr

    folder = gh_cli.config_dir("write")
    if not folder or gh_cli.config_dir("read") != folder:
        return True  # its reads are the read login's
    login = sr.entry(cfg, "github_write")
    if sr.held(folder, sr.held_files(login)):
        return True
    if held:
        return False
    raise PostRefused(
        f"{what} is not sent: the GitHub write login is not set ({sr.how_to_set(login)})"
    )


# The capabilities each outward write is for (rkmon_lib/features.py): a review post's are the table's
# (rkmon_lib/outward.py), a notice's by its kind. Each is on by its switches, in prototype too; what it sends goes out
# only while its feature sends (outward_gate). rk notice's feature gate in rkmon.py reads NOTICE_FEATURE too.
POST_FEATURES = outward.row("post").features
NOTICE_FEATURE = {
    "incident": ("incident-issues",),
    "queue": ("queue-notices",),
    "overlap": ("overlap-management",),
    "drift": ("drift-notices",),
    "gpu": ("remote-gpu-tests", "review-posting"),
    "fix": ("fix-links",),
}
# The notices `rk notice --kind` posts, each by its kind in NOTICE_FEATURE (a fix notice is rk fix-link's).
NOTICE_CHOICES = ("incident", "queue", "overlap", "drift", "gpu")


class Held(PostRefused):
    """A write of the monitor's own that the one outward gate held (outward_gate): kept as an unsent item, or, a
    protective one in standby, left to wait. Its text is the one line that says so (UNSENT, WAITING); NEW, whether it
    kept an item not kept before, whose one journal line the gate wrote (keep), so that its caller writes no second
    line for it. A caller that catches PostRefused alone refuses it."""

    def __init__(self, line: str, new: bool = False):
        super().__init__(line)
        self.new = new


UNSENT, WAITING = "UNSENT", "WAITING"  # how the line of a held write starts (Held)


def held_line(line: str) -> bool:
    """Whether LINE is the line of a write the one outward gate held (Held): kept as an unsent item, or waiting."""
    return line.startswith((f"{UNSENT}: ", f"{WAITING}: "))


def outcome(done: str, said: str) -> str:
    """What a command prints for its write: the word that says the write went out (DONE, "filed:"), then what the
    write returned (SAID); or SAID alone when the one outward gate held the write (held_line).
    """
    return said if held_line(said) else f"{done} {said}"


def _sends_for(needs: tuple) -> str:
    """The feature that the capabilities NEEDS send for (features.Capability.feature): the first one named."""
    from . import features

    return next(
        (
            features.CAPABILITIES[n].feature
            for n in needs
            if features.CAPABILITIES[n].feature
        ),
        "",
    )


def outward_check(
    cfg: dict,
    st: dict,
    what: str,
    needs: tuple,
    protective: bool = False,
    own: bool | None = None,
    feature: str | None = None,
) -> str:
    """The one outward gate's first half (outward_gate), which a write asks before its own checks: PostRefused while a
    capability it acts for (NEEDS) is off, and for a write you type (outward.own) while its feature does not send,
    naming the mode or standby. Else why the write would be held: features.held's word for the monitor's
    own write, "" when it goes out. A protective write (`protective`: turning off an auto-merge the monitor armed) goes
    out in prototype too. FEATURE, when given, is the feature the write sends for (else the capabilities').
    """
    from . import features, state_store

    for name in needs:
        if not features.on(cfg, name):
            raise PostRefused(
                f"{features.lowered(features.CAPABILITIES[name].title)} is off ({features.why_off(cfg, name)}): {what} is "
                "not sent"
            )
    feature = _sends_for(needs) if feature is None else feature
    why = features.held(cfg, feature) if feature else ""
    if not why or (protective and why == state_store.PROTOTYPE):
        return ""
    if not (outward.own() if own is None else own):
        raise PostRefused(f"{what} is not sent: {features.held_words(cfg, feature)}")
    return why


def outward_gate(
    cfg: dict,
    st: dict,
    what: str,
    needs: tuple,
    item: dict | None = None,
    protective: bool = False,
    own: bool | None = None,
    feature: str | None = None,
    store=None,
) -> None:
    """The one outward gate, right before a write goes out: nothing when it goes out; PostRefused when it
    may not (outward_check); else Held, for the monitor's own write while its feature does not send: ITEM (its type,
    what it is for, its content and file: `unsent_item`) kept as an unsent item in STORE's state (keep), or, a
    protective write in standby, nothing kept: the first tick after standby ends makes it, if it is still due. With no
    ITEM there is nothing to keep, and the write is refused: a send the writer makes for another process, whose sender
    kept its own item at its own gate first (writer_item). Every outward write of the toolkit meets it: each of this
    module's right before it goes out, the alerts and the team dashboard's page where they are made, a found issue's
    ticket at its Approve (rkmon_lib/tickets.py), and the sends the writer makes for another process (rkmon.py
    write_gate). (`rk pause` pauses the tick only; it gates no command.)"""
    from . import features

    why = outward_check(cfg, st, what, needs, protective, own, feature)
    if not why:
        return
    feature = _sends_for(needs) if feature is None else feature
    words = features.held_words(cfg, feature)
    if protective:
        raise Held(f"{WAITING}: {what} waits, nothing was sent: {words}")
    if item is None:
        raise PostRefused(f"{what} is not sent: {words}")
    new = keep(dict(item, feature=feature, why=why), store)
    raise Held(
        f"{UNSENT}: {what} for {item['for']}, kept as an unsent item: {words}", new
    )


def keep(item: dict, store=None) -> bool:
    """Keep ITEM as an unsent item (state_store.keep_unsent) in STORE's state folder (the monitor's by default), and,
    when it is new, say so in its journal in one line: its type, what it was for and why it was kept (why_kept), never
    its content. Whether it was new."""
    from . import state_store
    from .store import Store

    store = store if store is not None else Store()
    new = state_store.keep_unsent(store.dir, item)
    if new:
        store.journal(
            "unsent",
            f"{item['type']} for {item['for']}: not sent "
            f"({why_kept(item['feature'], item['why'])})",
        )
    return new


def why_kept(feature, why) -> str:
    """Why an unsent item of FEATURE was kept, WHY the item's own word for it (state_store.PROTOTYPE or STANDBY), in
    words: "<feature> in prototype", the feature as features.SENDS names it, or "the monitor in standby", since standby
    is the whole monitor sending nothing, never a feature's mode. The item's journal line (keep) and
    the "Not sent" card's "Why kept" (rkmon_lib/webdash.py) say it."""
    from . import features, state_store

    if why == state_store.STANDBY:
        return "the monitor in standby"
    feature = str(feature or "")
    return f"{features.SENDS.get(feature, feature)} in {why or ''}"


def unsent_item(kind: str, for_: str, content: str = "", file: str = "") -> dict:
    """An unsent item's own fields: its type (state_store.TYPES), what it is for, its content and the file that holds
    it; the gate adds its feature and why."""
    return {
        "type": kind,
        "for": for_,
        "content": content,
        **({"file": file} if file else {}),
    }


def writer_item(cfg: dict, cmd: str, words=()) -> dict | None:
    """The unsent item that a send the writer makes for another process keeps when the gate holds it (rkmon.py
    write_gate): for a JIRA write of `rk atlassian` (the session's follow-up ticket; WORDS its words, as
    writer.atlassian_call reads them), a ticket for the JIRA project (tickets.unsent_for), its content the CLI's words
    as the writer would run them. None, so that the writer refuses it, for a write it would not make in live either
    (atlassian.outside), and for the other sends, whose sender keeps its own item at its own gate before it asks the
    writer (the alerts: teams_breakage, rkmon_lib/needs_teams.py; the team dashboard's page: rkmon_lib/publish.py).
    """
    from . import atlassian, tickets, writer

    if cmd != outward.ATLASSIAN:
        return None
    call = writer.atlassian_call(list(words))
    if atlassian.outside(cfg, call):
        return None
    return unsent_item(TICKET, tickets.unsent_for(cfg), shlex.join(call.argv))


def card_text(payload: dict) -> str:
    """A team chat message (teams_message) in plain words, as its unsent item keeps it: its summary, then each text,
    fact and link of its card, one a line, the automated-message line (is_automated_line) left out.
    """
    out = [str(payload.get("summary") or "")]
    for att in payload.get("attachments") or ():
        card = att.get("content") or {}
        for block in card.get("body") or ():
            if is_automated_line(block):
                continue
            if block.get("type") == "FactSet":
                out += [
                    f"{f.get('title')}: {f.get('value')}"
                    for f in block.get("facts") or ()
                ]
            elif block.get("text"):
                out.append(str(block["text"]))
        out += [
            f"{a.get('title')}: {a['url']}"
            for a in card.get("actions") or ()
            if a.get("url")
        ]
    return "\n".join(x for x in out if x)


def teams_channel(cfg: dict) -> str:
    """The team chat channel in words, as an unsent alert names what it was for: with its message label, when it has
    one."""
    label = message_label(cfg)
    return f"the team chat channel ({label})" if label else "the team chat channel"


def _gate(cfg: dict, st, store, what: str, needs: tuple, item: dict, **kw) -> str:
    """outward_gate for a write of this module: "" when it goes out, else the line that says it was held (Held)."""
    try:
        outward_gate(cfg, st, what, needs, item, store=store, **kw)
    except Held as held:
        return str(held)
    return ""


# --- writes whose outcome may be unknown -------------------------------------------------------------------------------
# A gh that times out or fails may still have made its comment or issue: GitHub took it, and the answer was lost. Before
# such a write is called a failure, GitHub is read back (a GET) for it, so a retry never makes it twice. One no read can
# settle stays in doubt in state/writes_in_doubt.json, and the next try of the same write reads GitHub first.
DOUBT_FILE = "writes_in_doubt.json"
DOUBT_SLACK_MIN = 2  # the read-back looks this much earlier than the try (clock skew)


def _doubts(store) -> dict:
    try:
        d = json.loads((store.dir / DOUBT_FILE).read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _doubt(store, key: str, val: dict | None) -> None:
    """KEY's write in doubt recorded as VAL, or dropped (VAL None), in DOUBT_FILE, written whole
    (state_store.write_file); with nothing in doubt there is no file."""
    d = _doubts(store)
    if val is None:
        if key not in d:
            return
        d.pop(key)
    else:
        d[key] = val
    if not d:  # nothing in doubt: no file
        (store.dir / DOUBT_FILE).unlink(missing_ok=True)
        return
    state_store.write_file(store.dir / DOUBT_FILE, d)


def _same(a: str, b: str) -> bool:
    return a.replace("\r\n", "\n").strip() == b.replace("\r\n", "\n").strip()


def find_comment(cfg: dict, n: int, body: str, since: str, repo: str = "") -> str:
    """Our own comment on issue or PR n of `repo` (blank: the watched repository) with this body, updated at or after
    `since`: its URL, or "" (a GET; gh.GhError when GitHub cannot be read). It reads the account alone, not whose
    comment it is (reviewer_name.own_review): the body must be the one this write sent, whose line 1 names this
    deployment, so another deployment's comment under the account never matches."""
    rows = gh_cli.rest(
        f"repos/{repo or cfg['repo']}/issues/{n}/comments?per_page=100&since={since}",
        paginate=True,
        jq=f'.[] | select(.user.login == "{config.operator_login(cfg)}") | {{html_url, body}}',
    )
    return next(
        (
            r.get("html_url") or ""
            for r in reversed(rows or [])
            if isinstance(r, dict) and _same(r.get("body") or "", body)
        ),
        "",
    )


def find_issue(cfg: dict, title: str, since: str) -> str:
    """Our own develop issue with this title, updated at or after `since`: its URL, or "" (a GET; gh.GhError)."""
    from . import breakage

    rows = gh_cli.rest(
        f"repos/{breakage.issue_repository(cfg)}/issues?creator={config.operator_login(cfg)}&state=all"
        f"&since={since}&per_page=100",
        paginate=True,
        jq=".[] | select(.pull_request == null) | {html_url, title}",
    )
    return next(
        (
            r.get("html_url") or ""
            for r in reversed(rows or [])
            if isinstance(r, dict) and (r.get("title") or "").strip() == title.strip()
        ),
        "",
    )


def _ago(minutes: float) -> str:
    import datetime as dt

    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def create(cfg: dict, store, args: list[str], key: str, found) -> str:
    """Run the gh command `args` (gh's own arguments: the configured gh runs them) that creates one comment or issue;
    its URL. `key` names the write (kind, PR, key); `found(since)` reads GitHub for it (find_comment, find_issue). A
    failure or timeout is reported only once the read-back finds nothing; a write the read-back found is returned as
    made. A mirror's refusal (gh.GhError, before gh runs) passes through and leaves nothing in doubt.
    """
    before = _doubts(store).get(key)
    if before:
        try:
            url = found(before["since"])
        except gh_cli.GhError as exc:
            raise PostRefused(
                f"an earlier try of this write is in doubt (it may have gone out), and GitHub could not "
                f"be read to settle it: {str(exc)[:200]}; nothing was written: try again"
            ) from None
        _doubt(store, key, None)
        if url:
            store.journal(
                "post", f"{key}: an earlier try that was in doubt did go out: {url}"
            )
            return url
    since = _ago(DOUBT_SLACK_MIN)
    _doubt(store, key, {"since": since, "at": now()})
    try:
        p = _gh_write(args, capture_output=True, text=True, timeout=120)
        err, out = (
            (p.stderr.strip()[:300] or f"exit {p.returncode}") if p.returncode else "",
            p.stdout,
        )
    except subprocess.TimeoutExpired:
        err, out = "timed out after 120 s", ""
    except gh_cli.GhError:  # a mirror: refused before gh ran
        _doubt(store, key, None)
        raise
    if not err:
        _doubt(store, key, None)
        return _gh_url(out)
    try:
        url = found(since)
    except gh_cli.GhError as exc:
        raise PostRefused(
            f"gh {' '.join(args[:2])} failed ({err}), and GitHub could not be read to check whether it went "
            f"out ({str(exc)[:160]}): in doubt; the next try reads GitHub first"
        ) from None
    _doubt(store, key, None)
    if url:
        store.journal(
            "post",
            f"{key}: gh reported a failure ({err[:120]}), but it is on GitHub: {url}",
        )
        return url
    raise PostRefused(f"gh {' '.join(args[:2])} failed: {err}")


# The line 1 a review's job was staged with (REVIEW_LINE1), as worker/fetch_pr.py writes it in the review folder's
# pr/meta.md, among the lines before the PR's description, which begins at META_DESCRIPTION
STAGED_LINE1 = re.compile(r"^- Line 1: (.+)$", re.M)
META_DESCRIPTION = "\n## Description\n"


def review_line(cfg: dict, n: int, head: str) -> str:
    """The line 1 PR N's review at HEAD must give: the line its review was staged with, the one the review folder's
    pr/meta.md (worker/fetch_pr.py) gives before the PR's description when its `- Head:` there is HEAD, so a review in
    flight keeps the line of the model that drafted it after a change of the passes; else the line the
    configuration gives now (bindings.first_line). The description is the PR author's text, so no line in it counts.
    """
    from . import bindings, snapshot

    workers = paths.resolve(cfg).get("workers")
    meta = pathlib.Path(workers) / f"pr{n}" / "pr" / "meta.md" if workers else None
    try:
        text = meta.read_text(encoding="utf-8", errors="replace") if meta else ""
    except OSError:
        text = ""
    text = text.split(META_DESCRIPTION, 1)[0]
    staged, at = STAGED_LINE1.search(text), snapshot.HEAD_IN_META.search(text)
    if staged and at and snapshot.same_commit(at.group(1), head):
        return staged.group(1).strip()
    return bindings.first_line(cfg)


def check(cfg: dict, body: str) -> list[str]:
    problems = []
    first = body.splitlines()[0].strip() if body.strip() else ""
    if first != cfg["review_first_line"]:
        problems.append(
            f"line 1 must be exactly {cfg['review_first_line']!r} (got {first[:80]!r})"
        )
    return problems + public(body, cfg)


ROUND_HEADER = re.compile(
    r"^## PR #(\d+)\b[^\n]*?—\s*(?:🔴|🟠|🟡|🟢|⚪)+\s*(First-round|Follow-up)\b([^:\n—]*)",
    re.M,
)


def round_check(body: str, n: int, reviewed: bool) -> list[str]:
    """The review header must name this PR and the right round (a stale first-round draft must never go out as a
    follow-up): a follow-up once we have reviewed the PR,
    else a first round. A "First-round re-review" may repeat a first round."""
    m = ROUND_HEADER.search(body)
    if not m:
        return [
            "no review header: expected '## PR #N — … — <🔴|🟠|🟡|🟢> First-round|Follow-up: …'"
        ]
    problems = []
    if int(m.group(1)) != n:
        problems.append(f"the header names #{m.group(1)}, not #{n}: wrong file?")
    rnd, rest = m.group(2), m.group(3)
    if reviewed and rnd == "First-round" and "re-review" not in rest:
        problems.append(
            f"the header says First-round, but #{n} already has our review: post the follow-up "
            "(wrong file?)"
        )
    elif not reviewed and rnd == "Follow-up":
        problems.append(
            f"the header says Follow-up, but we have not reviewed #{n} yet: post the first round"
        )
    return problems


def tag_check(header: str, sources: str, cfg: dict | None = None) -> list[str]:
    """Every ticket tag in the review header must be the PR's own, found in its title, body or commit messages (a draft
    can invent a tag, or copy one from an example)."""
    tag = ticket_tag(cfg)
    have = set(tag.findall(sources))
    return [
        f"the header's {t} is not in the PR's title, body or commits: drop it, or use the PR's own tag"
        for t in sorted(set(tag.findall(header)))
        if t not in have
    ]


# The skill's name for the operator (rkmon_lib/agentic.py LOGIN), filled in at install: never in an outward text.
LOGIN_PLACEHOLDER = "<user_login>"


def public(body: str, cfg: dict | None = None) -> list[str]:
    """Public-repo rules: no unfilled <user_login>, allow-listed hosts only, no absolute performance numbers, bare
    ticket tags."""
    problems = []
    if LOGIN_PLACEHOLDER in body:
        problems.append(
            f"the text still has the placeholder {LOGIN_PLACEHOLDER}: write the operator's login instead "
            "(integrations.github_cli.username; the installed skill names it)"
        )
    for m in URL.finditer(body):
        host = m.group(1).lower()
        if not any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS):
            problems.append(f"link to a non-public host: {m.group(0)[:80]}")
    for m in PERF.finditer(body):
        problems.append(f"absolute performance number: {m.group(0)!r} (use ratios)")
    if ticket_link(cfg).search(body):
        problems.append("ticket tags must be bare, not links")
    return problems


# What each kind of notice is: its first line's words before "by <automated_reviewer_name>" (notice_line). The kinds
# `notice` posts, the change request on a held PR (hold) and the develop issues' report (report).
NOTICES = {
    "incident": "Automated develop-health notice",
    "queue": "Automated merge-queue status",
    "fix": "Automated develop-health notice",
    # On a PR that drops its copy of work another open PR owns, once per group key.
    "overlap": "Automated overlap notice",
    # On a PR whose tests fail on today's develop while its head is unchanged (drift.py), once
    # per develop SHA.
    "drift": "Automated develop-drift notice",
    # On a PR whose GPU-only cases fail on a remote GPU node and pass on develop on the same node (gpu.py), once per
    # job.
    "gpu": "Automated GPU test notice",
    # A held PR's change request (request_changes).
    "hold": "Automated merge-queue hold",
    # A develop issue's body, and each comment of ours on one (issue_file, issue_comment, fix_link).
    "report": "Automated develop-health report",
}


def notice_line(cfg: dict, kind: str) -> str:
    """The first line of a notice of KIND (NOTICES): what it is, then "by" and this deployment's automated reviewer name
    (reviewer_name.name), so that a reader of two deployments' notices on one pull request or issue knows which posted
    which."""
    return f"{NOTICES[kind]} by {reviewer_name.name(cfg)}"


def notice(cfg: dict, st: dict, store, n: int, path: str, kind: str, key: str) -> str:
    """Post a develop-health (incident), merge-queue or fix (`fix_link`) notice on PR n (it may be merged, e.g. a
    culprit). `key` is the incident id, the queue-state key or the fixed issue; one notice per (kind, PR, key).
    """
    what = f"{'an' if kind[:1] in 'aeiou' else 'a'} {kind} notice"
    needs = NOTICE_FEATURE.get(kind, POST_FEATURES)
    outward_check(cfg, st, what, needs)
    body = pathlib.Path(path).read_text(encoding="utf-8")
    want = notice_line(cfg, kind)
    problems = [p for p in check(dict(cfg, review_first_line=want), body)]
    if problems:
        raise PostRefused("; ".join(problems))
    if store.ledger_has(f"notice-{kind}", n, key):
        raise PostRefused(f"{what} for {key} is already on #{n}")
    recent = [r for r in store.ledger() if minutes_since(r["ts"]) < 60]
    if len(recent) >= registry.value(cfg, "max_posts_per_hour") + 4:
        raise PostRefused(f"rate limit: {len(recent)} outward actions in the last hour")
    held = _gate(
        cfg, st, store, what, needs, unsent_item(COMMENT, f"pull request {n}", body)
    )
    if held:
        return held
    url = create(
        cfg,
        store,
        ["pr", "comment", str(n), "--repo", cfg["repo"], "--body-file", path],
        f"notice-{kind}:{n}:{key}",
        lambda since: find_comment(cfg, n, body, since),
    )
    store.ledger_add(f"notice-{kind}", n, key, url)
    store.journal("post", f"#{n} {kind} notice ({key}): {url}")
    daylog.add(store, cfg, "notice", n, f"{kind} notice ({key})", url=url)
    return url


def my_reviews(cfg: dict, n: int) -> list[dict]:
    """The user's approving, change-requesting and dismissed reviews on PR n, oldest first, read live. COMMENTED
    reviews never change a reviewer's state. It reads the account alone, not whose review it is
    (reviewer_name.own_review): GitHub keeps one review state per account, whichever deployment posted under it, and
    that state is what blocks a merge and what only the user's re-approval lifts."""
    from . import gh

    return gh.rest(
        f"repos/{cfg['repo']}/pulls/{n}/reviews?per_page=100",
        paginate=True,
        jq=f'.[] | select(.user.login == "{config.operator_login(cfg)}" and .state != "COMMENTED") | {{id, state, submitted_at}}',
    )


def request_changes(
    cfg: dict, st: dict, store, n: int, path: str, user: bool = False
) -> str:
    """Request changes on PR n from our account, so it cannot merge until it is fixed. A held PR gets one, so it cannot
    merge without its fix; the author clears it by re-requesting review once the fix is in, and the PR then needs the
    user's explicit re-approval.

    Only for a hold confirmed at the PR's current head (`rk hold`), or on the user's explicit instruction (`user`). The
    body says what fails, how to fix it, and to re-request a review once fixed. We never approve or dismiss: the
    user's explicit re-approval lifts it (merge.hold_reviews follows it). The PR is read on GitHub only where this
    process can read it (github_readable)."""
    from . import gh

    what, needs = "a change request", ("holds",)
    reads = github_readable(cfg, what, outward_check(cfg, st, what, needs))
    body = pathlib.Path(path).read_text(encoding="utf-8")
    problems = check(dict(cfg, review_first_line=notice_line(cfg, "hold")), body)
    rec = st["prs"].get(str(n))
    h = (st.get("holds") or {}).get(str(n))
    if not rec:
        problems.append(f"#{n} is not tracked")
    elif not user and not (
        h and h.get("status") == "held" and h["head"] == rec["head"]
    ):
        problems.append(
            f"#{n} has no confirmed hold at its head (`rk hold {n}` first, or --user on the user's "
            "instruction)"
        )
    elif rec.get("author") == config.operator_login(cfg):
        problems.append(
            f"#{n} is the user's own PR: GitHub does not let its author request changes"
        )
    elif store.ledger_has("request-changes", n, rec["head"]):
        problems.append(f"changes were already requested on #{n} at {rec['head'][:10]}")
    if problems:
        raise PostRefused("; ".join(problems))
    if reads:
        live = gh.rest(
            f"repos/{cfg['repo']}/pulls/{n}", jq="{state, merged, head: .head.sha}"
        )
        if live["state"] != "open":
            raise PostRefused(
                f"#{n} is {'merged' if live.get('merged') else live['state']}: a change request cannot "
                "block it"
            )
        if live["head"] != rec["head"]:
            raise PostRefused(
                f"#{n} moved to {live['head'][:10]}: a push ends the hold, so re-gate it first"
            )
        mine = my_reviews(cfg, n)
        if mine and mine[-1]["state"] == "CHANGES_REQUESTED":
            raise PostRefused(
                f"the user already requests changes on #{n} (since {mine[-1]['submitted_at']}); the tick "
                "records it as the hold's change request"
            )
    recent = [r for r in store.ledger() if minutes_since(r["ts"]) < 60]
    if len(recent) >= registry.value(cfg, "max_posts_per_hour") + 4:
        raise PostRefused(f"rate limit: {len(recent)} outward actions in the last hour")
    held = _gate(
        cfg,
        st,
        store,
        what,
        needs,
        unsent_item(CHANGE_REQUEST, f"pull request {n}", body),
    )
    if held:
        return held
    p = _gh_write(
        [
            "pr",
            "review",
            str(n),
            "--repo",
            cfg["repo"],
            "--request-changes",
            "--body-file",
            path,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if p.returncode:
        raise PostRefused(
            f"gh pr review --request-changes failed: {p.stderr.strip()[:300]}"
        )
    mine = my_reviews(cfg, n)
    rid = mine[-1]["id"] if mine and mine[-1]["state"] == "CHANGES_REQUESTED" else None
    url = f"{derived.github_address(cfg['repo'])}/pull/{n}" + (
        f"#pullrequestreview-{rid}" if rid else ""
    )
    reason = (h or {}).get("reason") or "the user's instruction"
    st.setdefault("hold_reviews", {})[str(n)] = {
        "head": rec["head"],
        "reason": reason[:200],
        "by": "us",
        "at": now(),
        "url": url,
    }
    store.ledger_add("request-changes", n, rec["head"], url, note=reason[:200])
    store.journal("post", f"#{n} changes requested ({'hold' if h else 'user'}): {url}")
    daylog.add(
        store,
        cfg,
        "hold",
        n,
        "changes requested: it cannot merge until fixed and re-approved",
        url=url,
    )
    return url


def arm(cfg: dict, st: dict, store, n: int) -> str:
    """Enable auto-merge (squash, pinned to the validated head) for PR n. Every condition is re-checked here, the PR's
    on GitHub only where this process can read it (github_readable)."""
    from . import gh

    what, needs = "auto-merge", ("auto-merge",)
    qs = st.get("queue_state") or {}
    rec = st["prs"].get(str(n))
    problems, reads = [], True
    try:
        reads = github_readable(cfg, what, outward_check(cfg, st, what, needs))
    except PostRefused as refused:
        # Off, one you type while it sends nothing, or no write login to read GitHub with: said with the rest.
        problems.append(str(refused))
    if n not in st.get("cleared", []):
        problems.append(
            f"#{n} is not cleared by the user (`rk clear {n}` on their instruction)"
        )
    if st.get("frozen"):
        problems.append(f"queue frozen: {st['frozen'].get('reason')}")
    if qs.get("armed") and qs["armed"]["pr"] != n:
        problems.append(f"#{qs['armed']['pr']} is already armed")
    b = (st.get("batches") or [None])[0]
    if not b or b.get("status") != "validated" or b["members"][0] != n:
        problems.append(f"#{n} is not the next member of a validated batch")
    if not rec:
        problems.append(f"#{n} is not tracked")
    if problems:
        raise PostRefused("; ".join(problems))
    if reads:
        live = gh.rest(
            f"repos/{cfg['repo']}/pulls/{n}",
            jq="{state, head: .head.sha, draft, mergeable, mergeable_state}",
        )
        if live["state"] != "open" or live["draft"] or live["head"] != rec["head"]:
            raise PostRefused(
                f"#{n} changed on GitHub (state {live['state']}, head {live['head'][:10]}): re-gate first"
            )
        # GitHub's own gates decide every merge: arming only turns on GitHub's auto-merge, which waits for the
        # required checks, approvals, code owners, conversations and branch protection. A PR GitHub calls
        # unmergeable (conflicts), or whose mergeability GitHub has not computed yet, is not armed.
        if live.get("mergeable") is False or live.get("mergeable_state") == "dirty":
            raise PostRefused(
                f"#{n} is not mergeable on GitHub ({live.get('mergeable_state') or 'conflicts'}): "
                "nothing armed"
            )
        if live.get("mergeable") is None:
            raise PostRefused(
                f"GitHub has not computed whether #{n} is mergeable yet: nothing armed; try the next tick"
            )
    held = _gate(
        cfg,
        st,
        store,
        what,
        needs,
        unsent_item(
            AUTO_MERGE,
            f"pull request {n}",
            f"auto-merge (squash) at head {rec['head']}",
        ),
    )
    if held:
        return held
    p = _gh_write(
        [
            "pr",
            "merge",
            str(n),
            "--repo",
            cfg["repo"],
            "--auto",
            "--squash",
            "--match-head-commit",
            rec["head"],
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if p.returncode:
        raise PostRefused(f"gh pr merge --auto failed: {p.stderr.strip()[:300]}")
    st.setdefault("queue_state", {})["armed"] = {
        "pr": n,
        "head": rec["head"],
        "develop": st["develop"]["sha"],
        "at": now(),
    }
    store.ledger_add("arm", n, rec["head"], f"develop {st['develop']['sha'][:12]}")
    store.journal("queue", f"armed #{n} at {rec['head'][:10]}")
    daylog.add(
        store,
        cfg,
        "arm",
        n,
        f"auto-merge enabled at {rec['head'][:10]} (develop {st['develop']['sha'][:10]})",
    )
    return (p.stdout.strip() or "auto-merge enabled")[:200]


def disarm(cfg: dict, st: dict, store, n: int, reason: str) -> str:
    """Turn auto-merge off on PR n and forget it as armed. For the PR the monitor armed (queue_state's armed PR) it is
    the protective write: it goes out with auto-merge off or in prototype; in standby it waits (Held:
    nothing kept), and the tick says DISARM_NEEDED again until standby ends. For any other PR, or with nothing armed,
    it is an auto-merge write, as arming is: it needs auto-merge on, and goes out only while the merge queue sends;
    else the monitor's own is kept as an unsent item (AUTO_MERGE), and one you type is refused, naming the mode or
    standby. Only a disarm that took clears the armed record and is ledgered: when gh fails (or times out), GitHub is
    read, and auto-merge already off (or the PR closed) counts as done; otherwise PostRefused, the PR stays armed, and
    the tick says DISARM_NEEDED again.
    """
    armed = ((st.get("queue_state") or {}).get("armed") or {}).get("pr") == n
    held = _gate(
        cfg,
        st,
        store,
        f"turning auto-merge off on #{n}" if armed else "turning auto-merge off",
        () if armed else ("auto-merge",),
        unsent_item(
            AUTO_MERGE, f"pull request {n}", f"auto-merge turned off: {reason}"
        ),
        **(dict(protective=True, feature="merge_queue") if armed else {}),
    )
    if held:
        return held
    try:
        p = _gh_write(
            ["pr", "merge", str(n), "--repo", cfg["repo"], "--disable-auto"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        err = (p.stderr.strip()[:200] or f"exit {p.returncode}") if p.returncode else ""
    except subprocess.TimeoutExpired:
        err = "timed out after 120 s"
    how = "auto-merge disabled"
    if err:
        try:
            live = gh_cli.rest(
                f"repos/{cfg['repo']}/pulls/{n}",
                jq="{state, auto: (.auto_merge != null)}",
            )
        except gh_cli.GhError as exc:
            raise PostRefused(
                f"gh pr merge --disable-auto failed ({err}), and the PR could not be read "
                f"({str(exc)[:120]}): #{n} stays armed; run `rk disarm {n}` again"
            ) from None
        if live.get("state") == "open" and live.get("auto"):
            raise PostRefused(
                f"gh pr merge --disable-auto failed: {err}; auto-merge is still on: #{n} stays armed; "
                f"run `rk disarm {n}` again"
            )
        why = (
            f"the PR is {live.get('state')}"
            if live.get("state") != "open"
            else f"gh: {err[:80]}"
        )
        how = f"auto-merge was already off ({why})"
    qs = st.setdefault("queue_state", {})
    if qs.get("armed", {}) and qs["armed"]["pr"] == n:
        qs["armed"] = None
    head = (st["prs"].get(str(n)) or {}).get("head", "?")
    store.ledger_add("disarm", n, head, reason)
    store.journal("queue", f"disarmed #{n}: {reason} ({how})")
    daylog.add(store, cfg, "disarm", n, f"auto-merge disabled: {reason}")
    return how


@contextlib.contextmanager
def _body_file(text: str):
    """`text` in a temporary file for gh to read, its path after `--body-file` or `-F body=@`: the `with` block gets the
    path, and the file goes when the block ends, however it ends (the write went out, gh failed or timed out, a guard
    refused).
    """
    import tempfile

    fd, name = tempfile.mkstemp(suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        yield name
    finally:
        pathlib.Path(name).unlink(missing_ok=True)


def queue_notice(
    cfg: dict,
    st: dict,
    store,
    n: int,
    text: str,
    key: str,
    explains_turn_off: bool = False,
) -> str:
    """Create, or edit in place, our single merge-queue status comment on PR n. One that `explains_turn_off` (why the
    auto-merge guard, or the user, turned auto-merge off) goes out with queue notices off, and skips the rate limit: a
    turn-off must never leave the author with auto-merge off and no reason why. Like every queue notice it goes out only
    while the merge queue sends; else the monitor's own is kept as an unsent item, and one you type is refused
    (outward_gate): no notice is the protective write."""
    what = "a queue notice"
    needs, gate = (
        ((), dict(feature="merge_queue"))
        if explains_turn_off
        else (NOTICE_FEATURE["queue"], {})
    )
    outward_check(cfg, st, what, needs, **gate)
    first = notice_line(cfg, "queue")
    problems = check(dict(cfg, review_first_line=first), text)
    if problems:
        raise PostRefused("; ".join(problems))
    recent = [r for r in store.ledger() if minutes_since(r["ts"]) < 60]
    if (
        not explains_turn_off
        and len(recent) >= registry.value(cfg, "max_posts_per_hour") + 4
    ):
        raise PostRefused(f"rate limit: {len(recent)} outward actions in the last hour")
    held = _gate(
        cfg,
        st,
        store,
        what,
        needs,
        unsent_item(COMMENT, f"pull request {n}", text),
        **gate,
    )
    if held:
        return held
    sticky = [
        r for r in store.ledger() if r["kind"] == "notice-queue-sticky" and r["pr"] == n
    ]
    if sticky:
        cid = sticky[-1]["note"]
        try:  # an edit in place: a retry edits it again, never a second comment
            with _body_file(text) as body:
                p = _gh_write(
                    [
                        "api",
                        "-X",
                        "PATCH",
                        f"repos/{cfg['repo']}/issues/comments/{cid}",
                        "-F",
                        f"body=@{body}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
        except subprocess.TimeoutExpired:
            raise PostRefused(
                f"editing comment {cid} timed out after 120 s: try again"
            ) from None
        if p.returncode:
            raise PostRefused(f"editing comment {cid} failed: {p.stderr.strip()[:300]}")
        store.ledger_add("notice-queue-edit", n, key, sticky[-1]["ref"], note=cid)
        url = sticky[-1]["ref"]
    else:
        with _body_file(text) as body:
            url = create(
                cfg,
                store,
                ["pr", "comment", str(n), "--repo", cfg["repo"], "--body-file", body],
                f"notice-queue:{n}:{key}",
                lambda since: find_comment(cfg, n, text, since),
            )
        cid = (re.findall(r"issuecomment-(\d+)", url) or [""])[-1]
        store.ledger_add("notice-queue-sticky", n, key, url, note=cid)
    st.setdefault("notices", {})[str(n)] = {"key": key, "url": url, "at": now()}
    store.journal("post", f"#{n} queue notice ({key}): {url}")
    daylog.add(store, cfg, "notice", n, f"queue status ({key})", url=url)
    return url


def _rate(cfg: dict, store) -> None:
    recent = [r for r in store.ledger() if minutes_since(r["ts"]) < 60]
    if len(recent) >= registry.value(cfg, "max_posts_per_hour") + 4:
        raise PostRefused(f"rate limit: {len(recent)} outward actions in the last hour")


def _gh_url(out: str) -> str:
    return (re.findall(r"https://github\.com/\S+", out) or [""])[-1]


def label_missing(cfg: dict, name: str) -> list[str]:
    """The refusal when the develop issues' repository (breakage.issue_repository) has no label `name` (a GET; 404
    means no), else nothing."""
    from . import breakage, gh

    repo = breakage.issue_repository(cfg)
    try:
        gh.rest(f"repos/{repo}/labels/{quote(name, safe='')}", jq=".name")
    except gh.GhError as exc:
        if "404" not in str(exc):
            raise
        return [
            f"{repo} has no {name} label: create the label first (`rk label-create {name} --description TEXT "
            "--color HEX`, on the user's instruction)"
        ]
    return []


def issue_file(cfg: dict, st: dict, store, path: str, new: bool = False) -> str:
    """Open a develop issue in the develop issues' repository (breakage.issue_repository): labelled
    with the CI-red label and its extra labels for a break that makes CI red, the stability label for one CI does not
    see, assigned to the breaking PR's author (`assign:`). The labels are the settings' (breakage.labels).
    Refused: a body whose line 1 is not the report line, public-repo violations (title too), a
    missing title, a label the repo does not have, an incident that already has an issue, a likely duplicate of an open
    develop issue of either kind (unless `new`, after checking it), the rate limit. The labels and the open develop
    issues are read where this process can read GitHub (github_readable); where it cannot, a held issue is kept as its
    unsent item without them, and one that would go out is refused before any check."""
    from . import breakage, guard

    what, needs = "a develop issue", ("stability-issues",)
    reads = github_readable(cfg, what, outward_check(cfg, st, what, needs))
    f = breakage.parse_file(path, cfg)
    kind, title = f["kind"], f["title"][:120]
    problems = check(
        dict(cfg, review_first_line=notice_line(cfg, "report")), f["body"]
    ) + public(f["title"], cfg)
    if not f["title"] or len(f["title"]) > 200:
        problems.append("title: missing, or longer than 200 characters")
    labs = list(f["labels"])
    # The develop guard's extra labels go with the CI-red label.
    added = (
        [x for x in dict.fromkeys(breakage.extra_labels(cfg)) if x not in labs]
        if kind == breakage.RED
        else []
    )
    labs += added
    if reads:
        for name in (breakage.labels(cfg)[kind], *added):
            problems += label_missing(cfg, name)
    incs = guard.incidents(store)
    inc = (
        next((i for i in incs if i.get("id") == f["incident"]), None)
        if f["incident"]
        else None
    )
    if f["incident"] and not inc:
        problems.append(f"no incident {f['incident']}")
    if inc and inc.get("issue"):
        problems.append(f"incident {f['incident']} already has issue #{inc['issue']}")
    # The ledger note is "<kind>: <title>"; older notes hold the title alone.
    if any(
        r["kind"] == "issue"
        and (r.get("note") == title or str(r.get("note") or "").endswith(f": {title}"))
        for r in store.ledger()
    ):
        problems.append("an issue with this title was already filed")
    if reads and not new and not problems:
        for n, other, shared, k in breakage.duplicates(cfg, f["title"], f["body"]):
            problems.append(
                f"possible duplicate of open {breakage.kind_label(cfg, k)} #{n} ({other[:80]}): "
                f"shares {', '.join(shared[:6])}. "
                f"Comment there (`rk issue-comment {n} FILE --key KEY`), or pass --new after checking"
            )
    if problems:
        raise PostRefused("; ".join(problems))
    _rate(cfg, store)
    assigned = f"; assigned to {', '.join(f['assign'])}" if f["assign"] else ""
    held = _gate(
        cfg,
        st,
        store,
        what,
        needs,
        unsent_item(
            ISSUE,
            f"a new issue in {breakage.issue_repository(cfg)}",
            f"{f['title']}\nlabels: {', '.join(labs) or 'none'}{assigned}\n\n{f['body']}",
        ),
    )
    if held:
        return held
    with _body_file(f["body"]) as body:
        args = [
            "issue",
            "create",
            "--repo",
            breakage.issue_repository(cfg),
            "--title",
            f["title"],
            "--body-file",
            body,
        ]
        for lab in labs:
            args += ["--label", lab]
        for login in f["assign"]:
            args += ["--assignee", login]
        url = create(  # the write login (gh.write)
            cfg,
            store,
            args,
            f"issue:{f['title'][:120]}",
            lambda since: find_issue(cfg, f["title"], since),
        )
    n = (
        int(url.rstrip("/").rsplit("/", 1)[-1])
        if url.rstrip("/").rsplit("/", 1)[-1].isdigit()
        else 0
    )
    who = f", assigned to {', '.join(f['assign'])}" if f["assign"] else ""
    store.ledger_add(
        "issue",
        n,
        f["incident"] or "-",
        url,
        note=f"{breakage.kind_label(cfg, kind)}: {title}",
    )
    store.journal(
        "post",
        f"{breakage.kind_label(cfg, kind)} issue #{n}: {f['title']}{who} ({url})",
    )
    daylog.add(
        store,
        cfg,
        "ticket",
        None,
        f"GitHub issue #{n} ({breakage.kind_label(cfg, kind)}): {f['title']}{who}",
        url=url,
    )
    if inc:
        inc["issue"] = n
        inc.setdefault("actions", []).append("issue")
        guard.save_incidents(store, incs)
    return url


def issue_relabel(
    cfg: dict, st: dict, store, n: int, to: str, assign: str | None = None
) -> str:
    """Move develop issue n to the other kind (CI-red only while it makes CI red; otherwise stability, assigned
    to the breaking PR's author): add the target label and remove the other
    one; `assign` adds an assignee. The extra labels go with the CI-red label: dropped with it, added back with it
    where the repo has them. `to` is breakage.RED or breakage.STABILITY.
    Ledgered once per (issue, target). Refused: the repo has no target
    label, the issue is closed or already carries the target label alone, the rate limit. The labels and the issue are
    read only where this process can read GitHub (github_readable); a held relabel not read keeps the whole edit (the
    other kind's labels off, the target's on) as its unsent item.
    """
    from . import breakage, gh

    lab = breakage.labels(cfg)
    kind = breakage.STABILITY if to == breakage.STABILITY else breakage.RED
    what = "relabelling a develop issue"
    needs = ("stability-issues" if kind == breakage.STABILITY else "incident-issues",)
    reads = github_readable(cfg, what, outward_check(cfg, st, what, needs))
    target, other = (
        lab[kind],
        lab[breakage.RED if kind == breakage.STABILITY else breakage.STABILITY],
    )
    problems = []
    if assign is not None and not breakage.LOGIN.fullmatch(assign.lstrip("@")):
        problems.append(f"--assign {assign!r} is not a GitHub login")
    if store.ledger_has("issue-relabel", n, kind):
        problems.append(f"#{n} was already relabelled {target}")
    if reads:
        problems += label_missing(cfg, target)
        issue = gh.rest(
            f"repos/{breakage.issue_repository(cfg)}/issues/{n}",
            jq="{state, labels: [.labels[].name], " "assignees: [.assignees[].login]}",
        )
        if issue["state"] != "open":
            problems.append(f"#{n} is {issue['state']}")
        elif target in issue["labels"] and other not in issue["labels"]:
            problems.append(f"#{n} already has the {target} label")
    else:  # not read: an issue of the other kind as filed, its labels the other kind's (the extra ones with CI-red)
        extra = breakage.extra_labels(cfg) if kind == breakage.STABILITY else []
        issue = {"labels": [x for x in (other, *extra) if x], "assignees": []}
    if problems:
        raise PostRefused("; ".join(problems))
    _rate(cfg, store)
    login = assign.lstrip("@") if assign else None
    add, drop = [target], [other] if other in issue["labels"] else []
    # The extra labels go with the CI-red label: an issue on its way to the stability label drops them.
    if kind == breakage.STABILITY:
        drop += [x for x in breakage.extra_labels(cfg) if x in issue["labels"]]
    else:
        add += [
            x
            for x in breakage.extra_labels(cfg)
            if x not in issue["labels"] and not (reads and label_missing(cfg, x))
        ]
    args = ["issue", "edit", str(n), "--repo", breakage.issue_repository(cfg)]
    for x in add:
        args += ["--add-label", x]
    for x in drop:
        args += ["--remove-label", x]
    if login and login not in issue["assignees"]:
        args += ["--add-assignee", login]
    held = _gate(
        cfg,
        st,
        store,
        what,
        needs,
        unsent_item(LABEL, f"issue {n}", " ".join(args[5:])),
    )
    if held:
        return held
    p = _gh_write(
        args, capture_output=True, text=True, timeout=120
    )  # the write login (gh.write)
    if p.returncode:
        raise PostRefused(f"gh issue edit failed: {p.stderr.strip()[:300]}")
    was = ", ".join(drop) or "unlabelled"
    who = f", assigned to {login}" if login else ""
    store.ledger_add(
        "issue-relabel", n, kind, _gh_url(p.stdout) or f"#{n}", note=f"from {was}{who}"
    )
    store.journal("post", f"issue #{n} relabelled {was} -> {target}{who}")
    daylog.add(
        store,
        cfg,
        "ticket",
        None,
        f"GitHub issue #{n} relabelled {target} (was {was}){who}",
    )
    for i in (st.get("post_merge") or {}).get(
        "open"
    ) or []:  # until the next poll reads it back
        if i["n"] == n:
            i.update(
                kind=kind,
                label=target,
                labels=[x for x in i.get("labels") or [] if x != other] + [target],
            )
    hint = (
        f"; assign the breaking PR's author (`rk issue-assign {n} LOGIN`)"
        if (kind == breakage.STABILITY and not login and not issue["assignees"])
        else ""
    )
    return f"#{n} relabelled {target} (was {was}){who}{hint}"


def issue_assign(cfg: dict, store, n: int, login: str) -> str:
    """Assign develop issue n (in the develop issues' repository, breakage.issue_repository) to `login`, on the
    user's instruction (e.g. a CI-red issue to the develop-green owner). Ledgered; refused when that login is
    already assigned (read only where this process can read GitHub: github_readable)."""
    from . import breakage, gh

    what, needs = "assigning a develop issue", ("incident-issues",)
    reads = github_readable(cfg, what, outward_check(cfg, None, what, needs))
    repo = breakage.issue_repository(cfg)
    have = (
        gh.rest(f"repos/{repo}/issues/{n}", jq="[.assignees[].login]") if reads else []
    )
    if login in (have or []):
        raise PostRefused(f"#{n} is already assigned to {login}")
    _rate(cfg, store)
    held = _gate(
        cfg,
        None,
        store,
        what,
        needs,
        unsent_item(ISSUE, f"issue {n}", f"assignee: {login}"),
    )
    if held:
        return held
    p = _gh_write(
        ["issue", "edit", str(n), "--repo", repo, "--add-assignee", login],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if p.returncode:
        raise PostRefused(f"gh issue edit failed: {p.stderr.strip()[:300]}")
    store.ledger_add("issue-assign", n, login, _gh_url(p.stdout) or f"#{n}")
    store.journal("post", f"issue #{n} assigned to {login}")
    daylog.add(store, cfg, "ticket", None, f"GitHub issue #{n} assigned to {login}")
    return f"#{n} assigned to {login}"


def label_create(cfg: dict, store, name: str, description: str, color: str) -> str:
    """Create a label in the develop issues' repository (breakage.issue_repository), where the develop issues'
    labels must exist, on the user's instruction. Ledgered; refused when the label already exists (read only where
    this process can read GitHub: github_readable).
    """
    import urllib.parse
    from . import breakage, gh

    what, needs = "a new label", ("incident-issues",)
    reads = github_readable(cfg, what, outward_check(cfg, None, what, needs))
    repo = breakage.issue_repository(cfg)
    if reads:
        try:
            gh.rest(
                f"repos/{repo}/labels/{urllib.parse.quote(name, safe='')}",
                jq=".name",
            )
        except gh.GhError as exc:
            if "404" not in str(exc):
                raise PostRefused(f"could not check label {name!r}: {exc}") from exc
        else:
            raise PostRefused(f"label {name!r} already exists in {repo}")
    problems = public(name, cfg) + public(description, cfg)
    if len(description) > 100:
        problems.append("description: GitHub allows at most 100 characters")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", color):
        problems.append("color: six hex digits, without #")
    if problems:
        raise PostRefused("; ".join(problems))
    _rate(cfg, store)
    held = _gate(
        cfg,
        None,
        store,
        what,
        needs,
        unsent_item(
            LABEL, f"the repository {repo}", f"{name}: {description} (color {color})"
        ),
    )
    if held:
        return held
    p = _gh_write(
        [
            "label",
            "create",
            name,
            "--repo",
            repo,
            "--description",
            description,
            "--color",
            color,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if p.returncode:
        raise PostRefused(f"gh label create failed: {p.stderr.strip()[:300]}")
    store.ledger_add("label", 0, name, f"label {name}", note=description[:120])
    store.journal("post", f"label {name!r} created in {repo}")
    daylog.add(
        store, cfg, "ticket", None, f"GitHub label {name} created: {description}"
    )
    return f"label {name} created in {repo}"


def issue_comment(cfg: dict, st: dict, store, n: int, path: str, key: str) -> str:
    """Comment on an existing develop issue of either kind (in the develop issues' repository,
    breakage.issue_repository), e.g. our bisect evidence on an issue someone else filed. Line 1 is the report line;
    one comment per (issue, key)."""
    from . import breakage

    what, needs = "an issue comment", ("incident-issues",)
    outward_check(cfg, st, what, needs)
    body = pathlib.Path(path).read_text(encoding="utf-8")
    problems = check(dict(cfg, review_first_line=notice_line(cfg, "report")), body)
    if store.ledger_has("issue-comment", n, key):
        problems.append(f"already commented on #{n} for {key}")
    if problems:
        raise PostRefused("; ".join(problems))
    _rate(cfg, store)
    held = _gate(cfg, st, store, what, needs, unsent_item(COMMENT, f"issue {n}", body))
    if held:
        return held
    repo = breakage.issue_repository(cfg)
    url = create(
        cfg,
        store,
        ["issue", "comment", str(n), "--repo", repo, "--body-file", path],
        f"issue-comment:{n}:{key}",
        lambda since: find_comment(cfg, n, body, since, repo),
    )
    kind = breakage.kind_label(cfg, breakage.kind_in_state(cfg, st, n))
    store.ledger_add("issue-comment", n, key, url)
    store.journal("post", f"comment on {kind} issue #{n} ({key}): {url}")
    daylog.add(
        store,
        cfg,
        "ticket",
        None,
        f"comment on GitHub issue #{n} ({kind}, {key})",
        url=url,
    )
    return url


def fix_notice_text(cfg: dict, rec: dict, n: int, title: str) -> str:
    """The PR comment for a verified fix of develop issue n (`fix_link`), with its TITLE when one is known, which names
    the issue in full when it is in another repository (breakage.issue_ref)."""
    from . import breakage

    ref = breakage.issue_ref(cfg, n)
    about = f" ({title})" if title else ""
    return (
        f"{notice_line(cfg, 'fix')}\n\n"
        f"Hi @{rec['author']}, this PR looks like the fix for {ref}{about}.\n\n"
        f'Please add "Fixes {ref}" to the description, so the issue closes when this PR merges. The rocKE monitor '
        "has moved this PR to the top of its review and merge queues.\n"
    )


def fix_link(
    cfg: dict, st: dict, store, issue: int, pr: int, evidence: str | None = None
) -> str:
    """Link PR `pr` as the fix for develop issue `issue`, once a human has verified it (an open PR that fixes a develop
    issue is told so, and goes to the top of the review queue):
    1. a comment on the issue, "Fix candidate: #P" with the optional evidence (`issue_comment`, key fix-P);
    2. one notice on the PR (kind fix): it looks like the fix, please add "Fixes #I" to the description, and the monitor
       has moved it to the top of its review and merge queues;
    3. st["fix_links"][I] gets P: a develop fix at once (review P0, merge tier 0), without waiting for an edit of the
       issue body (comments are not polled).
    The link is recorded before the writes, so a failed write leaves it standing; a re-run does only what is left.
    While the merge queue does not send, the link is recorded and each write is kept as an unsent item (outward_gate),
    whose line stands on a line of its own.
    The user's own PR gets no notice: it would ask its author, from the author's own account, to edit the description;
    the result line tells the user instead.
    Refused: all three done already, an issue that is not an open develop issue, a PR that is not open, tracked and
    ready for review, or already closes the issue, public-repo violations, the rate limit. The PR and the issue are
    read only where this process can read GitHub (github_readable); a held link not read there takes the issue's title
    and kind as the tick last read them (st post_merge).
    """
    from . import breakage, gh, guard, queue

    what = "a fix link"
    reads = github_readable(cfg, what, outward_check(cfg, st, what, ("fix-links",)))
    rec, links = st["prs"].get(str(pr)), st.setdefault("fix_links", {})
    own = bool(rec) and rec.get("author") == config.operator_login(cfg)
    done = (
        pr in links.get(str(issue), []),
        store.ledger_has("issue-comment", issue, f"fix-{pr}"),
        own or store.ledger_has("notice-fix", pr, f"issue-{issue}"),
    )
    if all(done):
        raise PostRefused(f"#{pr} is already linked as the fix for #{issue}")
    if not rec or rec.get("status") != "open":
        raise PostRefused(f"#{pr} is not a tracked open PR")
    if issue in (rec.get("closes") or []):
        raise PostRefused(f"#{pr} already closes #{issue}: it is a develop fix already")
    problems = []
    if reads:
        live = gh.rest(f"repos/{cfg['repo']}/pulls/{pr}", jq="{state, draft}")
        if live["state"] != "open" or live["draft"]:
            problems.append(
                f"#{pr} is {'a draft' if live['draft'] else live['state']}: link it once it is open and ready "
                "for review"
            )
        iss = gh.rest(
            f"repos/{breakage.issue_repository(cfg)}/issues/{issue}",
            jq="{title, state, labels: [.labels[].name], "
            "pr: (.pull_request != null)}",
        )
        lab = breakage.labels(cfg)
        if (
            iss["pr"]
            or iss["state"] != "open"
            or not set(lab.values()) & set(iss["labels"])
        ):
            problems.append(
                f"#{issue} is not an open develop issue (labelled {' or '.join(lab.values())})"
            )
        title, issue_kind = iss["title"], breakage.kind_of(cfg, iss["labels"])
    else:  # not read: the issue as the tick last read the open develop issues
        last = (st.get("post_merge") or {}).get("open") or []
        title = next((i.get("title") or "" for i in last if i["n"] == issue), "")
        issue_kind = breakage.kind_in_state(cfg, st, issue)
    ev = pathlib.Path(evidence).read_text(encoding="utf-8").strip() if evidence else ""
    comment = (
        f"{notice_line(cfg, 'report')}\n\n**Fix candidate:** {breakage.pr_ref(cfg, pr)}\n\n"
        + (f"{ev}\n\n" if ev else "")
        + "This monitor reviews and merges it first.\n"
    )
    text = fix_notice_text(cfg, rec, issue, title)
    if not done[1]:
        problems += check(
            dict(cfg, review_first_line=notice_line(cfg, "report")), comment
        )
    if not done[2]:
        problems += check(dict(cfg, review_first_line=notice_line(cfg, "fix")), text)
    if problems:
        raise PostRefused("; ".join(problems))
    _rate(cfg, store)
    kind = breakage.kind_label(cfg, issue_kind)
    if not done[0]:
        links.setdefault(str(issue), []).append(pr)
        queue.refresh_develop_fixes(st, guard.incidents(store))
        store.journal(
            "fix",
            f"#{pr} linked as the fix for {kind} #{issue}: review P0, merge tier 0",
        )
    out, held = [f"#{pr} is a develop fix ({kind} #{issue})"], []

    def said(word: str, line: str) -> None:
        # what a write returned, after WORD; the line of a held one stands alone, on a line of its own (held_line)
        (held if held_line(line) else out).append(outcome(word, line))

    if not done[1]:
        with _body_file(comment) as body:
            said(
                "issue comment",
                issue_comment(cfg, st, store, issue, body, f"fix-{pr}"),
            )
    if not done[2]:
        with _body_file(text) as body:
            said("PR notice", notice(cfg, st, store, pr, body, "fix", f"issue-{issue}"))
    if own:
        out.append(
            f'no PR notice: #{pr} is your own PR, so add "Fixes {breakage.issue_ref(cfg, issue)}" to its '
            "description yourself"
        )
    return "\n".join(["; ".join(out), *held])


def our_post_body(cfg: dict, ours: dict | None) -> str:
    """Our latest review comment's body, read from GitHub ("" when there is none, for a formal review, which the older
    batches posted, and for a comment someone deleted: it carries no open rows). Raises gh.GhError otherwise.
    """
    from . import gh

    m = re.search(r"issuecomment-(\d+)", (ours or {}).get("url") or "")
    if not m:
        return ""
    try:
        body = gh.rest(f"repos/{cfg['repo']}/issues/comments/{m.group(1)}", jq=".body")
    except gh.GhError as exc:
        if "HTTP 404" in str(exc):
            return ""
        raise
    return body if isinstance(body, str) else ""


def our_last_review(cfg: dict, n: int, ours: dict | None) -> tuple[str, str]:
    """Our last review of PR n (`ours`, its record rec["ours"]): its body and where it was read, ("", "") when none is
    read. A review kept unposted is read from its file (kept_file). A posted one is read from GitHub, the comment as it
    stands (our_post_body), else from the newest copy of our posts on the PR in the outputs folder; in a mirror
    our last post is the live monitor's, of which the folder holds no copy. Raises gh.GhError when GitHub cannot be
    read and no copy stands in for it.
    """
    from . import gh, mirror

    if kept_unposted(ours):
        path = kept_file(cfg, ours)
        try:
            return (path.read_text(encoding="utf-8"), path.name) if path else ("", "")
        except OSError:
            return "", ""
    try:
        body, failed = our_post_body(cfg, ours), None
    except gh.GhError as exc:
        body, failed = "", exc
    if body:
        return body, "GitHub"
    posted = [
        p for p in ([] if mirror.on(cfg) else copies(cfg, n)) if not is_kept_file(p)
    ]
    if posted:
        return posted[-1].read_text(encoding="utf-8"), posted[-1].name
    if failed is not None:
        raise failed
    return "", ""


def open_rows(body: str, prior: bool = False) -> list[tuple[str, str, str]]:
    """One of our posts' open 🔴/🟠 rows, as (key, where, text): its findings; with `prior`, also the prior items its
    status table still calls OPEN (key "Our 11-02 N3: what", where "", text "🟠 OPEN • why").
    """
    rows = [
        (k, where.strip(), text.strip())
        for k, sev, where, text in daylog.FINDING.findall(body)
        if sev in "🔴🟠"
    ]
    if prior:
        rows += [
            (
                item.strip(),
                "",
                " ".join(f"{sev} OPEN{rest}".replace("<br>", " ").split()),
            )
            for item, sev, rest in daylog.STILL_OPEN.findall(body)
        ]
    return rows


def open_findings(cfg: dict, ours: dict | None) -> list[str]:
    """Our latest review's open 🔴/🟠 rows ("N1 (where): what → how to fix"), read from GitHub for the auto-merge guard's
    comment, so the author learns how to fix it without opening the review."""
    from . import gh

    try:
        body = our_post_body(cfg, ours)
    except gh.GhError:
        return []
    return [f"{k} ({where}): {text[:240]}" for k, where, text in open_rows(body)][:4]


# The text of a team chat message's first line (automated_line): what it is, then "by" the name of the deployment that
# sends it, as a notice's first line gives it (notice_line).
AUTOMATED = "Automated message by {name}"


def automated_line(name: str) -> dict:
    """The first line of a team chat message (teams_message) from the deployment NAME, its automated reviewer name, else
    its container name (reviewer_name.name), so that the messages of several deployments in one channel say which sent
    each (AUTOMATED)."""
    return {
        "type": "TextBlock",
        "text": AUTOMATED.format(name=name),
        "isSubtle": True,
        "size": "Small",
    }


def is_automated_line(block) -> bool:
    """Whether BLOCK is a team chat message's first line (automated_line), naming any deployment: read by its form."""
    before, after = AUTOMATED.split("{name}")
    text = block.get("text") if isinstance(block, dict) else None
    if not (isinstance(text, str) and text.startswith(before) and text.endswith(after)):
        return False
    name = text[len(before) : len(text) - len(after)]
    return bool(name.strip()) and block == automated_line(name)


def message_label(cfg: dict) -> str:
    """The text each team chat message starts with (`teams.channel`), blank for none: it labels the message, and
    the webhook decides the channel."""
    return ((cfg.get("teams") or {}).get("channel") or "").strip()


def labelled(label: str, text: str) -> str:
    """TEXT after the message label LABEL and a colon; TEXT alone while the label is blank."""
    return f"{label}: {text}" if label else text


def teams_message(
    name: str, summary: str, body: list[dict], actions: list[dict] | None = None
) -> dict:
    """A Teams message from the deployment NAME (reviewer_name.name): one Adaptive Card (Teams Workflows webhooks
    accept it) that opens with the automated-message line naming it (automated_line), then `body`, and `actions` when
    given."""
    card = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": [automated_line(name)] + body,
    }
    if actions:
        card["actions"] = actions
    return {
        "type": "message",
        "summary": summary,
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": card,
            }
        ],
    }


def teams_breakage_card(
    issue: dict, culprit: str | None, label: str, name: str
) -> dict:
    """The Teams message for a develop break with no planned fix, from the deployment NAME (teams_message): clearly
    automated, links the CI-red issue, and asks for a quick response. Its summary starts with the message label LABEL.
    """
    facts = [
        {
            "title": "Issue",
            "value": f"[#{issue['number']} {issue['title']}]({issue['html_url']})",
        },
        {"title": "Fix planned", "value": "**none yet**"},
    ]
    if culprit:
        facts.insert(1, {"title": "Broken by", "value": culprit})
    body = [
        {
            "type": "TextBlock",
            "text": "develop is broken and nobody has a fix in flight",
            "weight": "Bolder",
            "size": "Medium",
            "wrap": True,
            "color": "Attention",
        },
        {"type": "FactSet", "facts": facts},
        {
            "type": "TextBlock",
            "wrap": True,
            "text": (
                "**Who can take this?** Please reply here or on the issue as soon as you can. Link your fix PR on the "
                f"issue (\"Fixes #{issue['number']}\"): the monitor moves it to the top of the review and merge queues. "
                "The develop-green team force-reverts where necessary."
            ),
        },
    ]
    return teams_message(
        name,
        labelled(label, f"develop is broken (#{issue['number']})"),
        body,
        [
            {
                "type": "Action.OpenUrl",
                "title": "Open the issue",
                "url": issue["html_url"],
            }
        ],
    )


def _teams_hook(cfg: dict) -> pathlib.Path:
    """The file that holds the channel's webhook URL (a secret: never in config.json, never printed), where the
    secrets registry places it (secrets_registry.path): in the container, the writer's volume, which only the writer
    reads."""
    from . import secrets_registry as sr

    return pathlib.Path(sr.path(cfg, sr.entry(cfg, "teams_webhook")))


def teams_hook_set(cfg: dict) -> bool:
    """Is the channel's webhook URL in its file (logins.found: in the container, the writer's answer)? Only
    `_teams_post` reads the URL: it is a credential."""
    from . import logins, secrets_registry as sr

    return logins.found(cfg, sr.entry(cfg, "teams_webhook"))["set"] is True


def _teams_url(cfg: dict) -> str:
    """The webhook URL: the first non-empty line of its file (a note on a later line is not part of it)."""
    try:
        return next(
            (
                ln.strip()
                for ln in _teams_hook(cfg).read_text().splitlines()
                if ln.strip()
            ),
            "",
        )
    except OSError:
        return ""


# Anything URL-like in an error: a scheme, a host with a path, or a query string (a webhook's signature is in it).
URLISH = re.compile(
    r"[A-Za-z][\w+.-]*://\S+|[\w-]+(?:\.[\w-]+)+(?::\d+)?/\S*|\S*[?&][\w.-]+=\S*"
)


def redact(text: str) -> str:
    """`text` with every URL-like part replaced by <url>: a Teams error must never carry the webhook (a credential)."""
    return URLISH.sub("<url>", text)


TEAMS_TIMEOUT_S = 30  # the most one message's POST takes


def _teams_post(cfg: dict, payload: dict) -> int:
    """POST a message to the channel's webhook, read here; the HTTP status (a Workflows webhook answers 202). Any
    failure is a PostRefused whose text names no URL. The mirror's refusal is weighed once in each process, before this:
    `teams_post` weighs it, and rk's gate for the command that sends (rkmon.py run_command; in the writer, write_gate).
    """
    import urllib.request

    try:
        req = urllib.request.Request(
            _teams_url(cfg),
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=TEAMS_TIMEOUT_S) as resp:
            return resp.status
    except Exception as exc:  # noqa: BLE001 - never echo the URL: it is a credential
        raise PostRefused(
            f"the Teams webhook failed: {type(exc).__name__}: {redact(str(exc))[:200]}"
        ) from None


# Every kind of message to the team chat runs on its own clock (features.alerts.message_cooldown_minutes, config
# teams.message_cooldown_min): st[SENT_KEY] = {kind: when its last message went}.
SENT_KEY = "teams_sent"
# The types of message, each the one its command sends, and what each sends for (outward_gate's capabilities): the
# message column of rkmon_lib/outward.py's table. `rk teams-send --type TYPE` takes these types.
KIND_BREAK = outward.row("teams-breakage").message
KIND_NEEDS_YOU = outward.row("needs-you-teams").message
TEAMS_FEATURES = {r.message: r.features for r in outward.TABLE if r.message}
# A message larger than this is refused (a card is a few kB).
TEAMS_MESSAGE_MAX_BYTES = 256 << 10


def teams_post(cfg: dict, payload: dict, kind: str) -> int:
    """Send PAYLOAD, a message of type KIND (TEAMS_FEATURES), to the team chat channel: the HTTP status. Where this
    process reads the webhook (logins.here), `_teams_post` sends it; in the container the writer, which alone reads
    it, does (`rk teams-send --type KIND`, the message on stdin: `teams_send`). PostRefused in a mirror, before any
    request, in mirror.guard's words for rk teams-send (the writer's rk asks again at its gate, in the same words),
    and for any failure, in words that name no URL."""
    from . import logins, mirror, secrets_registry as sr, writer

    why = mirror.guard(cfg, outward.TEAMS_SEND)
    if why:
        raise PostRefused(why)
    if logins.here(sr.entry(cfg, "teams_webhook")):
        return _teams_post(cfg, payload)
    p = writer.through(
        [writer.TEAMS_SEND, "--type", kind],
        input=json.dumps(payload),
        timeout_s=TEAMS_TIMEOUT_S + writer.START_S,
    )
    m = re.search(r"\bHTTP (\d{3})\b", p.stdout or "")
    if p.returncode == 0 and m:
        return int(m.group(1))
    raise PostRefused(redact(writer.said(p))[:300])


def is_message(payload) -> bool:
    """Whether PAYLOAD is a message as `teams_message` makes it: one Adaptive Card whose body opens with the
    automated-message line, naming any deployment (is_automated_line)."""
    try:
        (att,) = payload["attachments"]
        return (
            payload.get("type") == "message"
            and att.get("contentType") == "application/vnd.microsoft.card.adaptive"
            and is_automated_line(att["content"]["body"][0])
        )
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return False


def teams_send(cfg: dict, kind: str, text: str) -> int:
    """`rk teams-send --type KIND`: the message TEXT (JSON, as `teams_post` sends it) to the channel, from here (the
    writer's half of `teams_post`): a message of a listed type (TEAMS_FEATURES) that `teams_message` made, at most
    TEAMS_MESSAGE_MAX_BYTES; the HTTP status. PostRefused otherwise, before any request.
    """
    if kind not in TEAMS_FEATURES:
        raise PostRefused(
            f"{kind!r} is not a type of team chat message ({', '.join(TEAMS_FEATURES)}): nothing was sent"
        )
    if len(text.encode()) > TEAMS_MESSAGE_MAX_BYTES:
        raise PostRefused(
            f"the message is larger than {TEAMS_MESSAGE_MAX_BYTES} bytes: nothing was sent"
        )
    try:
        payload = json.loads(text)
    except ValueError:
        raise PostRefused("the message is not JSON: nothing was sent") from None
    if not is_message(payload):
        raise PostRefused(
            "not a team chat message as teams_message makes them (one Adaptive Card that opens with the "
            "automated-message line): nothing was sent"
        )
    return _teams_post(cfg, payload)


def message_cooldown(cfg: dict) -> float:
    """The fewest minutes between two messages of one kind; a value that is not a number >= 0 is refused."""
    v = registry.value(cfg, "teams.message_cooldown_min")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
        raise ValueError(
            f"features.alerts.message_cooldown_minutes must be a number of minutes, 0 or more, not {v!r}"
        )
    return v


def last_sent(st: dict, kind: str) -> str:
    """When the last message of this kind went ("" for never)."""
    return str((st.get(SENT_KEY) or {}).get(kind) or "")


def cooldown_until(cfg: dict, st: dict, kind: str) -> str:
    """When the next message of this kind may go ("" for now)."""
    last = last_sent(st, kind)
    gap = message_cooldown(cfg)
    if last and minutes_since(last) < gap:
        return idle.plus(last, gap)
    return ""


def mark_sent(st: dict, kind: str, at: str) -> None:
    st.setdefault(SENT_KEY, {})[kind] = at


def teams_test(cfg: dict, store) -> str:
    """One card, labelled as a test, to confirm the webhook reaches the channel (on the operator's request). It says it
    is a test that needs no action, and names the alerts the channel gets, the features of rk teams-test's row in
    outward.TABLE (the ones its feature gate reads), each by its title and what it does (features.CAPABILITIES).
    """
    from . import features

    if not teams_hook_set(cfg):
        raise PostRefused(f"no Teams webhook configured in {_teams_hook(cfg)}")
    alerts = [
        features.CAPABILITIES[f] for f in outward.row(outward.TEAMS_TEST).features
    ]
    code = _teams_post(
        cfg,
        teams_message(
            reviewer_name.name(cfg),
            labelled(message_label(cfg), "rocKE monitor test"),
            [
                {
                    "type": "TextBlock",
                    "text": "Test: the rocKE monitor can post to this channel",
                    "weight": "Bolder",
                    "size": "Medium",
                    "wrap": True,
                },
                {
                    "type": "TextBlock",
                    "wrap": True,
                    "text": (
                        "No action needed: this is a test card. The rocKE monitor sends these alerts to this "
                        "channel, each while it is on:"
                    ),
                },
                {
                    "type": "FactSet",
                    "facts": [{"title": c.title, "value": c.what} for c in alerts],
                },
            ],
        ),
    )
    store.ledger_add("teams", 0, f"test-{now()}", "test card", note=f"HTTP {code}")
    store.journal("post", f"Teams test card (HTTP {code})")
    return f"test card sent to the team chat (HTTP {code})"


def teams_breakage(cfg: dict, st: dict, store, n: int, force: bool = False) -> str:
    """Alert the team chat about CI-red develop issue n (an issue filed for a proven develop break with no fix planned
    is announced at once). Only a CI-red issue, which makes CI red: a stability one
    goes to the breaking PR's author.
    Refused when the issue is closed or lacks the CI-red label, when a fix is planned (a PR named on the issue,
    closing it or linked to it, for either kind; `force` overrides), when this issue was already announced, when a
    develop break message went less than features.alerts.message_cooldown_minutes ago (`force` overrides), or when no
    webhook is configured (its URL lives outside the repo, in `teams.webhook_file`). While alerts do not send
    (prototype, standby), the one outward gate keeps the message as an unsent item, refuses it when you typed it, and
    needs no webhook. A kept message counts as sent everywhere, as a kept Needs-you message does: its clock runs, and
    the ledger records the channel as alerted about the issue, so no later run sends it.
    """
    from . import breakage, gh, secrets_registry

    what, needs = "a develop break alert", ("chat-alerts",)
    held = outward_check(cfg, st, what, needs)
    problems = []
    if not held and not teams_hook_set(cfg):
        problems.append(
            "no Teams webhook configured: ask the operator to run "
            f"{secrets_registry.set_fix(secrets_registry.entry(cfg, 'teams_webhook'))}"
        )
    if store.ledger_has("teams", n, f"issue-{n}"):
        problems.append(f"the channel was already alerted about #{n}")
    issue = gh.rest(
        f"repos/{breakage.issue_repository(cfg)}/issues/{n}",
        jq="{number, title, html_url, state, body, " "labels: [.labels[].name]}",
    )
    if issue.get("state") != "open":
        problems.append(f"#{n} is {issue.get('state')}")
    lab = breakage.labels(cfg)
    if not lab[breakage.RED]:
        problems.append(
            "no issue counts as CI-red, so none goes to the channel: "
            + breakage.blank_labels([breakage.RED])
        )
    elif lab[breakage.RED] not in (issue.get("labels") or []):
        problems.append(
            f"#{n} lacks the {breakage.kind_label(cfg, breakage.RED)} label: a "
            f"{breakage.kind_label(cfg, breakage.STABILITY)} issue goes to the breaking PR's author, not the channel"
        )
    planned = sorted(breakage.named_fixes(issue.get("body") or "") - {n}) + sorted(
        int(k)
        for k, why in (st.get("develop_fixes") or {}).items()
        if {f"{x} #{n}" for x in (*lab.values(), breakage.RED, breakage.STABILITY) if x}
        & set(why)
    )
    if planned and not force:
        problems.append(
            f"a fix is planned ({', '.join('#' + str(p) for p in sorted(set(planned)))}): no alert needed"
        )
    wait = cooldown_until(cfg, st, KIND_BREAK)
    if wait and not force:
        problems.append(
            f"the last develop break message went at {last_sent(st, KIND_BREAK)[11:16]}Z: the next may go at "
            f"{wait[11:16]}Z (features.alerts.message_cooldown_minutes; --force sends it now)"
        )
    if problems:
        raise PostRefused("; ".join(problems))
    _rate(cfg, store)
    m = breakage.INTRODUCED.search(
        issue.get("body") or ""
    ) or breakage.INTRODUCED.search(issue.get("title") or "")
    culprit = None
    if m:
        rec = st["prs"].get(m.group(1)) or {}
        culprit = f"#{m.group(1)}" + (
            f" (@{rec['author']})" if rec.get("author") else ""
        )
    card = teams_breakage_card(
        issue, culprit, message_label(cfg), reviewer_name.name(cfg)
    )
    line = _gate(
        cfg,
        st,
        store,
        what,
        needs,
        unsent_item(ALERT, teams_channel(cfg), card_text(card)),
    )
    if line:
        mark_sent(st, KIND_BREAK, now())
        store.ledger_add("teams", n, f"issue-{n}", issue["html_url"], note=line[:300])
        return line
    code = teams_post(cfg, card, KIND_BREAK)
    mark_sent(st, KIND_BREAK, now())
    store.ledger_add("teams", n, f"issue-{n}", issue["html_url"], note=f"HTTP {code}")
    store.journal("post", f"Teams alert for {lab[breakage.RED]} #{n} (HTTP {code})")
    daylog.add(
        store,
        cfg,
        "notice",
        None,
        f"Teams alert: develop broken, no fix planned ({lab[breakage.RED]} #{n})",
        url=issue["html_url"],
    )
    return f"alerted the team chat about #{n} (HTTP {code})"


def automerge_off_text(
    cfg: dict, rec: dict, risk: dict, findings: list[str] | None = None
) -> str:
    """The PR comment for an auto-merge guard turn-off, in plain words: what we found, how to fix it, and that nothing
    else changed."""
    return (
        f"{notice_line(cfg, 'queue')}\n\n"
        f"Hi @{rec['author']}, **I turned auto-merge off on this PR for now.** Before a rocKE PR lands, the rocKE "
        "monitor checks it on top of today's develop branch, and this version would not land cleanly:\n\n"
        + "".join(f"- {r}\n" for r in risk["reasons"])
        + (
            "\nThe open items from our review, each with its fix:\n\n"
            + "".join(f"- {f}\n" for f in findings)
            if findings
            else ""
        )
        + "\n**What to do:** "
        + "; then ".join(risk["fixes"])
        + ". After that, turn auto-merge back on; the "
        "monitor re-checks every push.\n\n"
        "_Nothing else changed: the monitor does not approve, reject or merge PRs. It only turned auto-merge off to "
        "keep develop green. This comment is updated in place._\n"
    )


def automerge_off(cfg: dict, st: dict, store, n: int) -> str:
    """Turn off auto-merge on PR n when its gates fail (merge.automerge_risks records the reasons for its head), and say
    why and how to fix it, in plain words, in our sticky queue comment. Needs
    `automerge_guard` "act"; the user's own PRs also need `features.merge_queue.auto_merge_guard.on_your_pull_requests`
    at `same_as_others`. Once per head: a re-enable at the same head goes to the user. The PR is read on GitHub
    only where this process can read it (github_readable).
    """
    from . import features, gh
    from .merge import automerge_guard_acts

    rec = st["prs"].get(str(n))
    risk = (st.get("automerge_risks") or {}).get(str(n))
    if rec and store.ledger_has("automerge-off", n, rec["head"]):
        key = f"automerge-off-{rec['head'][:10]}"
        if not any(
            store.ledger_has(k, n, key)
            for k in ("notice-queue-sticky", "notice-queue-edit")
        ):
            # Turned off, but the explanation never posted (e.g. refused by the rate limit): finish it.
            if (
                not risk or risk["head"] != rec["head"]
            ):  # the next tick drops the risk once auto-merge is off
                off = [
                    r
                    for r in store.ledger()
                    if r["kind"] == "automerge-off" and r["pr"] == n
                ][-1]
                risk = {
                    "reasons": off["ref"].split("; "),
                    "fixes": ["fix the open items in our latest review on this PR"],
                }
            text = automerge_off_text(
                cfg,
                rec,
                risk,
                (
                    open_findings(cfg, rec.get("ours"))
                    if any(r.startswith("our latest review") for r in risk["reasons"])
                    else None
                ),
            )
            return outcome(
                "explanation posted:",
                queue_notice(cfg, st, store, n, text, key, explains_turn_off=True),
            )
    problems = []
    if not rec or not risk or risk["head"] != rec["head"]:
        problems.append(f"#{n} has no recorded auto-merge risk at its current head")
    elif features.guard_mode(cfg) != "act":
        problems.append(f"{features.guard_alert_words(cfg)}: alert the user instead")
    elif not automerge_guard_acts(cfg, rec):
        from .deploy_tree import name

        problems.append(
            f"#{n}: the user turned auto-merge on themselves, or it is their own PR and "
            f"{name('automerge_guard_own')} is not same_as_others: alert them instead"
        )
    elif store.ledger_has("automerge-off", n, rec["head"]):
        problems.append(
            f"auto-merge on #{n} was already turned off at this head; it was re-enabled, so tell the user"
        )
    if problems:
        raise PostRefused("; ".join(problems))
    what, needs = "turning auto-merge off", ("auto-merge-guard",)
    if github_readable(cfg, what, outward_check(cfg, st, what, needs)):
        # Who enabled auto-merge is read live, as the rest: the state's `auto_merge.by` is the tick's, and the user may
        # have turned it on again since (theirs is never turned off).
        live = gh.rest(
            f"repos/{cfg['repo']}/pulls/{n}",
            jq="{state, head: .head.sha, auto: (.auto_merge != null), by: .auto_merge.enabled_by.login}",
        )
        if live["state"] != "open" or not live["auto"] or live["head"] != rec["head"]:
            raise PostRefused(
                f"#{n} changed on GitHub (state {live['state']}, auto-merge {live['auto']}, head "
                f"{live['head'][:10]}): the next tick re-evaluates it"
            )
        if not automerge_guard_acts(
            cfg,
            dict(rec, auto_merge=dict(rec.get("auto_merge") or {}, by=live.get("by"))),
        ):
            raise PostRefused(
                f"#{n}: auto-merge is on now by {live.get('by') or 'an unknown login'}, the user's own: it is left "
                "on; alert them instead"
            )
        risk = dict(risk, by=live.get("by") or risk.get("by"))
    held = _gate(
        cfg,
        st,
        store,
        what,
        needs,
        unsent_item(
            AUTO_MERGE,
            f"pull request {n}",
            f"auto-merge turned off: {'; '.join(risk['reasons'])}",
        ),
    )
    if held:
        return held
    p = _gh_write(
        ["pr", "merge", str(n), "--repo", cfg["repo"], "--disable-auto"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if p.returncode:
        raise PostRefused(
            f"gh pr merge --disable-auto failed: {p.stderr.strip()[:300]}"
        )
    store.ledger_add("automerge-off", n, rec["head"], "; ".join(risk["reasons"])[:300])
    store.journal(
        "queue",
        f"auto-merge turned off on #{n} (enabled by {risk.get('by')}): {'; '.join(risk['reasons'])}",
    )
    daylog.add(
        store,
        cfg,
        "disarm",
        n,
        f"auto-merge (enabled by {risk.get('by')}) turned off: {'; '.join(risk['reasons'])}",
    )
    text = automerge_off_text(
        cfg,
        rec,
        risk,
        (
            open_findings(cfg, rec.get("ours"))
            if any(r.startswith("our latest review") for r in risk["reasons"])
            else None
        ),
    )
    said = queue_notice(
        cfg,
        st,
        store,
        n,
        text,
        f"automerge-off-{rec['head'][:10]}",
        explains_turn_off=True,
    )
    done = "auto-merge disabled"
    return f"{done}\n{said}" if held_line(said) else f"{done}; comment {said}"


def automerge_off_user(cfg: dict, st: dict, store, n: int, path: str) -> str:
    """Turn off auto-merge on PR n on the user's explicit instruction, and post their reason in our sticky queue comment
    (e.g. other merges go first, and auto-merge comes back on when they are in). Unlike the auto-merge guard, it needs
    no recorded gate failure. The file holds the reason in plain words; our first line is added. The message is checked
    before anything changes, so auto-merge is never left off without its reason; a re-run after a failed comment
    finishes it. Once per head, under its own ledger kind, so the auto-merge guard never mistakes it for one of its own
    turn-offs. It goes out only while the merge queue sends: it is not the protective write (outward_gate). The PR is
    read on GitHub only where this process can read it (github_readable)."""
    from . import gh

    what, needs = "turning auto-merge off", ("auto-merge-guard",)
    reads = github_readable(cfg, what, outward_check(cfg, st, what, needs))
    rec = st["prs"].get(str(n))
    if not rec:
        raise PostRefused(f"#{n} is not a tracked open PR")
    first = notice_line(cfg, "queue")
    reason = pathlib.Path(path).read_text(encoding="utf-8").strip()
    if reason.startswith(first):
        reason = reason[len(first) :].strip()
    if not reason:
        raise PostRefused(f"{path} gives no reason")
    text = f"{first}\n\n{reason}\n"
    key = f"automerge-off-user-{rec['head'][:10]}"
    problems = check(dict(cfg, review_first_line=first), text)
    if problems:
        raise PostRefused("; ".join(problems))
    if store.ledger_has("automerge-off-user", n, rec["head"]):
        if any(
            store.ledger_has(k, n, key)
            for k in ("notice-queue-sticky", "notice-queue-edit")
        ):
            raise PostRefused(
                f"auto-merge on #{n} was already turned off at this head on the user's instruction"
            )
        return outcome(
            "reason posted:",
            queue_notice(cfg, st, store, n, text, key, explains_turn_off=True),
        )
    if reads:
        live = gh.rest(
            f"repos/{cfg['repo']}/pulls/{n}",
            jq="{state, head: .head.sha, auto: (.auto_merge != null)}",
        )
        if live["state"] != "open" or not live["auto"] or live["head"] != rec["head"]:
            raise PostRefused(
                f"#{n} changed on GitHub (state {live['state']}, auto-merge {live['auto']}, head "
                f"{live['head'][:10]}): the next tick re-reads it"
            )
    held = _gate(
        cfg,
        st,
        store,
        what,
        needs,
        unsent_item(AUTO_MERGE, f"pull request {n}", f"auto-merge turned off: {text}"),
    )
    if held:
        return held
    p = _gh_write(
        ["pr", "merge", str(n), "--repo", cfg["repo"], "--disable-auto"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if p.returncode:
        raise PostRefused(
            f"gh pr merge --disable-auto failed: {p.stderr.strip()[:300]}"
        )
    store.ledger_add("automerge-off-user", n, rec["head"], reason.splitlines()[0][:300])
    store.journal("queue", f"auto-merge turned off on #{n} on the user's instruction")
    daylog.add(
        store, cfg, "disarm", n, "auto-merge turned off on the user's instruction"
    )
    said = queue_notice(cfg, st, store, n, text, key, explains_turn_off=True)
    done = "auto-merge disabled on the user's instruction"
    return f"{done}\n{said}" if held_line(said) else f"{done}; comment {said}"


def preflight(
    cfg: dict, st: dict, store, n: int, path: str, again: bool = False
) -> tuple[str, dict, str]:
    """Every guard of `post`, in its order, and no write: the body, PR n's record and the file to post (with the GPU
    tests row when remote GPU tests ran for this head), or PostRefused. `rk post --dry-run` stops after it. The first
    guard is the mode (outward_gate): review automation live; the next, that the PR is not in review_skip
    (skip_check).
    """
    outward_gate(cfg, st, "a review post", POST_FEATURES)
    skip_check(st, n)
    body = pathlib.Path(path).read_text(encoding="utf-8")
    head = (st["prs"].get(str(n)) or {}).get("head") or ""
    problems = check(dict(cfg, review_first_line=review_line(cfg, n, head)), body)
    if problems:
        raise PostRefused("; ".join(problems))
    rec = st["prs"].get(str(n))
    if not rec:
        raise PostRefused(f"#{n} is not a tracked open PR")
    problems = round_check(body, n, bool(rec.get("ours")))
    if problems:
        raise PostRefused("; ".join(problems))
    from . import gpu

    with_row = gpu.for_post(st, n, rec["head"], body)
    if (
        with_row != body
    ):  # remote GPU tests ran for this head: the review carries their row (pending, or the result)
        problems = public(with_row, cfg)
        if problems:
            raise PostRefused("; ".join(problems))
        body = with_row
        row_file = gpu.local_dir(cfg, f"row-{n}") / f"post-{rec['head'][:12]}.md"
        row_file.parent.mkdir(parents=True, exist_ok=True)
        row_file.write_text(body, encoding="utf-8")
        path = str(row_file)
    if store.ledger_has("post", n, rec["head"]) and not again:
        raise PostRefused(
            f"already posted on #{n} at head {rec['head'][:10]} (use --again if intended)"
        )
    from . import gh

    current = gh.rest(f"repos/{cfg['repo']}/pulls/{n}", jq=".head.sha")
    if current != rec["head"] and not again:
        raise PostRefused(
            f"#{n} head moved to {current[:10]} after the review of {rec['head'][:10]}: re-check first "
            "(or --again if the new commits do not change the review)"
        )
    hdr = next((ln for ln in body.splitlines() if ln.startswith("## PR #")), "")
    if ticket_tag(cfg).search(hdr):
        pr = gh.rest(f"repos/{cfg['repo']}/pulls/{n}", jq="{title, body}")
        msgs = gh.rest(
            f"repos/{cfg['repo']}/pulls/{n}/commits?per_page=100",
            paginate=True,
            jq="[.[].commit.message]",
        )
        problems = tag_check(
            hdr,
            "\n".join([pr.get("title") or "", pr.get("body") or "", *map(str, msgs)]),
            cfg,
        )
        if problems:
            raise PostRefused("; ".join(problems))
    recent = [
        r for r in store.ledger() if r["kind"] == "post" and minutes_since(r["ts"]) < 60
    ]
    if len(recent) >= registry.value(cfg, "max_posts_per_hour"):
        raise PostRefused(f"rate limit: {len(recent)} posts in the last hour")
    return body, rec, path


def post_note(st: dict, n: int) -> str:
    """The ledger note of a review post on PR n, made now: "first_round", "followup", or "ci-update" for a post at a head
    our review already covers (its own, or one recorded as rebase- or merge-only) with no review of ours on the PR in
    flight: a CI update (L-209), no review, so it starts no follow-up gap (plan.last_review_posts).
    """
    from .plan import CI_UPDATE
    from .snapshot import covers

    rec = st["prs"][str(n)]
    if not rec.get("ours"):
        return "first_round"
    pipes = (st.get("review_pipes") or []) + [
        st.get("review_pipeline") or {}
    ]  # every parallel review in flight
    inflight = any(p.get("pr") == n for p in pipes) or any(
        r["pr"] == n for r in st.get("review_ready") or []
    )
    return (
        CI_UPDATE if covers(rec["ours"], rec["head"]) and not inflight else "followup"
    )


def round_word(note: str) -> str:
    """A review's round as the day log says it, from its note (post_note): first round, follow-up or CI update."""
    return {"first_round": "first round", "followup": "follow-up"}.get(
        note, "CI update"
    )


def record_name(cfg: dict, store) -> None:
    """Add the automated reviewer name this deployment posts under (reviewer_name.name) to the names its reviews have
    given (reviewer_name.NAMES_USED), unless it is there: in the internal state of STORE's home, through state_store
    (STORE's folder is its home's state folder, config.state_dir). `post` calls it before the review goes out, so that
    after a change of the name the review stays this deployment's own. It is NAMES_USED's only writer, and
    it reads the names before it sets them, holding no lock of its own between the two: no other record_name runs in
    between, because every `rk post` holds the write lock (control.write_lock) from its first write to its ledger
    entry. PostRefused, with nothing written, when the internal state cannot be read, or cannot be written
    (state_store.unwritable): its message says why, then the fix, a line each (applied.cannot_run).
    """
    from . import applied, state_store

    home, name = store.dir.parent, reviewer_name.name(cfg)
    try:
        used = registry.get(state_store.read(home), reviewer_name.NAMES_USED)
        used = [] if used is registry.MISSING else list(used)
        if name and name not in used:
            state_store.set(home, reviewer_name.NAMES_USED, [*used, name])
    except state_store.StateError as exc:
        refused = exc
    except OSError as exc:
        refused = state_store.unwritable(home, exc)
    else:
        return
    lead = f"the automated reviewer name {name!r} cannot be recorded"
    raise PostRefused("\n".join(applied.cannot_run(refused, lead)))


def post(cfg: dict, st: dict, store, n: int, path: str, again: bool = False) -> str:
    from . import gpu

    body, rec, path = preflight(
        cfg, st, store, n, path, again
    )  # the file with the GPU tests row
    # Posting off: rk post and rk spec-post keep the review unposted before they get here; any other caller stops.
    outward_gate(cfg, st, "a review post", POST_FEATURES)
    note = post_note(st, n)
    record_name(cfg, store)
    url = create(
        cfg,
        store,
        ["pr", "comment", str(n), "--repo", cfg["repo"], "--body-file", path],
        f"post:{n}:{rec['head']}",
        lambda since: find_comment(cfg, n, body, since),
    )
    kind = "FOLLOWUP" if rec.get("ours") else "FIRST_ROUND"
    store.ledger_add("post", n, rec["head"], url, note=note)
    store.journal("post", f"#{n} {note} at {rec['head'][:10]}: {url}")
    gpu.mark_posted(
        st, n, rec["head"], url, body
    )  # `rk gpu-row N` edits this comment's GPU tests row later
    daylog.from_review(store, cfg, n, round_word(note), body, url)
    out = posted_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    dest, i = out / f"PR_{kind}_{n}.md", 2
    while dest.exists():
        dest, i = out / f"PR_{kind}_{n}_{i}.md", i + 1
    shutil.copyfile(path, dest)
    from .review import review_finished

    review_finished(st, n, our_review(rec, body, rec["head"], url))
    return f"{url} (copy: {dest})"
