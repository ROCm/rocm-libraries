# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The automated reviewer name: the name line 1 of each review this deployment posts gives, so that an
author with two deployments' reviews of one pull request knows which is which, and each deployment knows its own
reviews among the reviews of the account it posts under.

- `name(cfg)`: the name: features.review_automation.automated_reviewer_name, else container.name (the configuration's
  deployment.project, at its default while blank). Check 19 keeps the first to letters, digits, spaces, - and
  _, and container.name's form is narrower, so a line that gives the name stays one line and `given` reads it back.
- `review_line(cfg, model, effort)`: line 1 of a review, REVIEW_LINE: `Automated review by <name>: Model - <model>,
  effort: <effort>`. Which model and effort is bindings.first_line's rule.
- `given(line)`: the name the line 1 LINE gives, by REVIEW_LINE's form; "" for a line that gives none, such as an older
  monitor's `Automated review: Model - <model>, effort: <effort>`.
- `name_marked(line)`: the line 1 LINE with NAME_MARK in place of the name it gives, by REVIEW_LINE's form, so that two
  such lines compare apart from their names (the round trip's, tools/round_trip.py); LINE itself when it gives none.
- `first_line(body)`: the line 1 the post BODY gives: its first line that is not blank, read after its quote marks (>)
  and its bold or italic marks (* and _), so that a quoted or bold line 1 gives its name too.
- `automated(body)`: whether the post BODY is an automated one, of any monitor: its line 1 (`first_line`) starts
  "Automated", as each review's and each notice's does, or is an early review's header (`## PR #`), from before
  line 1. The account's other posts are the hand-written ones.
- `names(cfg)`: every name this deployment's reviews give: today's, and each one it has posted under
  (reviews.automated_reviewer_names_used), so its earlier reviews stay its own after the name changes.
- `own_review(cfg, body, at)`: whether BODY, a post of this deployment's account posted at AT (GitHub's created_at
  for a comment, submitted_at for a review), is one of its own reviews: a review (REVIEWISH) whose line 1
  (`first_line`) gives one of `names`, or gives none and was posted before reviews.unnamed_are_ours_until, an ISO 8601
  time in UTC in store.now()'s form (the reviews an older monitor posted, for the deployment that moved from it: the
  one-time move sets the time, and `rkmon standby off` sets it again). Outside a mirror, while that time is
  blank, or is no such time, no review that gives no name is its own, and neither is one whose time is unknown. In
  a mirror (mirror.on) every review that gives no name is its own, whenever it was posted: under the account, only
  the older live monitor that the mirror follows posts one. A review that gives another name, such as another
  deployment's under the same account, is never its own. The caller reads the account's posts alone: this reads the
  text and the time.
"""

from __future__ import annotations

import re

from . import registry, store

NAME = "features.review_automation.automated_reviewer_name"
CONTAINER_NAME = "deployment.project"  # container.name's configuration key
NAMES_USED = "reviews.automated_reviewer_names_used"
UNNAMED_ARE_OURS_UNTIL = "reviews.unnamed_are_ours_until"
REVIEW_LINE = "Automated review by {name}: Model - {model}, effort: {effort}"
NAME_MARK = "<automated_reviewer_name>"  # the name's place in a line 1 compared apart from it (`name_marked`)
# REVIEW_LINE's text before the name, and between the name and the model: no name holds the second (check 19).
_BEFORE, _AFTER = (
    REVIEW_LINE.split("{name}")[0],
    REVIEW_LINE.split("{name}")[1].split("{model}")[0],
)
_GIVEN = re.compile(re.escape(_BEFORE) + r"(.+?)" + re.escape(_AFTER))
# A post that is a review, of any monitor: line 1 begins "Automated review" (REVIEW_LINE, or an older monitor's line
# without a name), or an early review's header or legend, from before line 1.
REVIEWISH = re.compile(r"^(Automated review|## PR #|##? First-round)|Legend:", re.M)
# The marks a line's text is read without: quote marks (>) before it, and bold or italic marks (* and _) at either end.
_MARKS = re.compile(r"^(?:\s*>)*\s*[*_]*\s*|\s*[*_]*\s*$")
# How an automated post's line 1 starts: a review's or a notice's, or an early review's header (`automated`).
_AUTOMATED = ("Automated", "## PR #")


def name(cfg: dict) -> str:
    """This deployment's automated reviewer name: see the module's docstring."""
    own = str(registry.value(cfg, NAME) or "").strip()
    return own or str(registry.value(cfg, CONTAINER_NAME, registry.BLANK) or "").strip()


def review_line(cfg: dict, model: str, effort: str) -> str:
    """Line 1 of a review of this deployment's, giving MODEL and EFFORT: see the module's docstring."""
    return REVIEW_LINE.format(name=name(cfg), model=model, effort=effort)


def given(line: str) -> str:
    """The name the line 1 LINE gives; "" for one that gives none."""
    m = _GIVEN.match(str(line or "").strip())
    return m.group(1).strip() if m else ""


def name_marked(line: str) -> str:
    """The line 1 LINE with NAME_MARK in place of the name it gives: see the module's docstring."""
    line = str(line or "").strip()
    m = _GIVEN.match(line)
    return f"{_BEFORE}{NAME_MARK}{_AFTER}{line[m.end():]}" if m else line


def first_line(body: str) -> str:
    """The line 1 the post BODY gives: see the module's docstring."""
    for ln in str(body or "").splitlines():
        ln = _MARKS.sub("", ln)
        if ln:
            return ln
    return ""


def automated(body: str) -> bool:
    """Whether the post BODY is an automated one: see the module's docstring."""
    return first_line(body).startswith(_AUTOMATED)


def names(cfg: dict) -> list[str]:
    """Every automated reviewer name this deployment's reviews give, today's first: see the module's docstring."""
    out = [name(cfg)]
    for used in registry.value(cfg, NAMES_USED) or []:
        used = str(used or "").strip()
        if used and used not in out:
            out.append(used)
    return out


def own_review(cfg: dict, body: str, at: str) -> bool:
    """Whether BODY, a post of this deployment's account posted at AT, is one of its own reviews: see the module's
    docstring."""
    body = str(body or "")
    if not REVIEWISH.search(body):
        return False
    got = given(first_line(body))
    if got:
        return got in names(cfg)
    from . import mirror

    return mirror.on(cfg) or _before(at, registry.value(cfg, UNNAMED_ARE_OURS_UNTIL))


def _before(at, until) -> bool:
    """Whether the time AT is before the time UNTIL, each in store.now()'s form; False when either is blank or no such
    time."""
    try:
        return store.parse_ts(str(at or "")) < store.parse_ts(str(until or ""))
    except ValueError:
        return False
