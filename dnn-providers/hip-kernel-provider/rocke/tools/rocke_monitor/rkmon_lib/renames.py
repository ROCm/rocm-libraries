# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Configuration keys that have a new name, a new shape, or none: a configuration or a settings file that still uses an
older one is moved to the new one (RENAMED), has its feature switches made again from the older ones (SWITCHES), or
loses the key (DROPPED), with one plain line for each.

`migrate(raw)` returns the configuration with every older key at its new place (the input is not changed) and the lines
that say what moved; `moves(raw)` returns the same configuration with each move as (older key, new key or None, line),
which the deploy settings' move from format 1 to format 2 builds on (rkmon_lib/deploy_tree.py). `config.load` and
`config.normalize` run it silently, so the code reads only the new keys; `docker/rkmon adopt` runs it on the files it
reads and prints its lines; the next save writes the new keys only.

The feature switches (SWITCHES): the older deployment.phase (observe, review, guard, queue: how far the monitor acted),
deployment.profile (full, review-only, merge-queue: what it ran), merge_authority and the per-capability switches under
features.<name> become the features' own: features.review_automation and features.merge_queue each enabled and in mode
prototype or live, the merge queue's auto_merge, auto_hold and develop_guard, the dashboards, the alerts and ticket
creation. What the older ones did is kept: a deployment that posted reviews (review-posting, from phase review) runs
review automation live; one in phase queue runs the merge queue live. Each kind of alert has its own switch beside
the alerts' one (ALERT_KINDS): chat-alerts and needs-you-alerts become them. While neither older switch is on, both
kinds are on, as both types of ticket are, so the alerts' one switch, once turned on, sends both kinds as before;
for the same reason, a configuration with the alerts' one switch and neither kind's has both kinds set on. Alerts,
the team dashboard and ticket creation had no mode, and each was sent whenever it was on: each that is on with no mode,
whether an older switch or its own turns it on, is live (SENT_WHENEVER_ON); one that is off takes no mode, and starts
in prototype, as every feature does.

No package import: docker/rkmon runs it on the host, with the host's Python.
"""

from __future__ import annotations

import copy

from .removed_keys import NO_HOST_SERVICES

# Older key -> new key (dotted paths); each comment says why the new name.
RENAMED = {
    "me": "operator.github_login",  # the operator's GitHub login, under operator.*
    # the GitHub login to JIRA account map belongs with the other jira.* settings
    "github_jira_users": "jira.accounts",
    "worker_tag": "worker.tag",  # every other setting of the local worker is under worker.*
    # it is the free memory a native build (the checks slot) needs
    "min_free_gb": "checks.min_free_memory_gb",
    # days are spelled out, as in retention.job_days and conflicts.scan_days
    "scoreboard.lookback_d": "scoreboard.lookback_days",
    "mirror.on": "mirror.enabled",  # a switch is `enabled`, as in ramdisk.enabled and atlassian.enabled
    # every Claude role's effort is effort.<role>, over the base claude.effort
    "claude.quality_effort": "effort.quality",
    # what a deployment is and the image it runs: deployment.*
    "container.project": "deployment.project",
    "container.image": "deployment.image",
    # where `rkmon image pull` takes the image from, as the build settings' image.registry is where it goes
    "container.registry": "deployment.pull_registry",
    # the cooldown holds for every kind of message to the team chat, each on its own clock
    "teams.needs_you_every_min": "teams.message_cooldown_min",
}
# Why the JIRA MCP server's keys and its older switch (OLD_FEATURES' jira-mcp) are dropped.
_JIRA_MCP = "an unused feature, the JIRA MCP server"
# Older keys with no new place, each with why: removing one says so in its line.
DROPPED = {
    "rocm.mode": "every image carries its own ROCm",
    "rocm.host_dir": "every image carries its own ROCm, so no host's ROCm is mounted",
    "jira.mcp_enabled": _JIRA_MCP,
    "jira.mcp_server": _JIRA_MCP,
    "jira.mcp_url": _JIRA_MCP,
}

# The older feature switches (SWITCHES): where each one's meaning went. A key here is read, turned into the features'
# own switches, and removed.
PHASES = ("observe", "review", "guard", "queue")
SWITCH_KEYS = (
    "phase",
    "profile",
    "deployment.phase",
    "deployment.profile",
    "merge_authority",
)
OLD_FEATURES = {
    # older switch -> (the new key that holds its meaning, or None when the features' modes decide it now: why)
    "review-posting": ("features.review_automation.mode", ""),
    "auto-merge": ("features.merge_queue.auto_merge", ""),
    "holds": ("features.merge_queue.auto_hold", ""),
    "team-dashboard": ("features.dashboards.team.enabled", ""),
    "chat-alerts": (
        "features.alerts.develop_breaks",
        " (each kind of alert is sent while features.alerts.enabled is true)",
    ),
    "needs-you-alerts": (
        "features.alerts.needs_you",
        " (each kind of alert is sent while features.alerts.enabled is true)",
    ),
    "follow-up-tickets": ("features.ticket_creation.triggers.review_follow_up", ""),
    "found-issue-tickets": ("features.ticket_creation.triggers.out_of_scope_issue", ""),
    "remote-gpu-tests": ("gpu.enabled", ""),
    "weekly-report": ("weekly_report.enabled", ""),
    "services": (None, NO_HOST_SERVICES),
    "headless-claude": (
        None,
        "the Claude session runs while features.review_automation.enabled is true",
    ),
    "jira-mcp": (None, _JIRA_MCP),
    "overlap-management": (
        None,
        "overlap notices are posted while features.review_automation.mode is live",
    ),
    "queue-notices": (
        None,
        "queue notices are posted while features.merge_queue.mode is live and its queue_notices.enabled is true",
    ),
    "drift-notices": (
        None,
        "drift notices are posted while features.merge_queue.mode is live and its drift_notices.enabled is true",
    ),
    "auto-merge-guard": (
        None,
        "the auto-merge guard alerts in features.merge_queue.mode prototype and turns auto-merge off in live",
    ),
    "stability-issues": (
        None,
        "develop issues are filed while features.merge_queue.mode is live and its develop_guard.enabled is true",
    ),
    "incident-issues": (
        None,
        "incidents are filed while features.merge_queue.mode is live and its develop_guard.enabled is true",
    ),
    "fix-links": (
        None,
        "fixes are linked while features.merge_queue.mode is live and its develop_guard.enabled is true",
    ),
}
# The alerts' one switch and each kind's own (the module's docstring): a configuration with the one switch and neither
# kind's has both kinds set on.
ALERTS = "features.alerts.enabled"
ALERT_KINDS = ("features.alerts.develop_breaks", "features.alerts.needs_you")
# The features that had no mode (the module's docstring): one that is on with no mode is live.
SENT_WHENEVER_ON = (
    "features.alerts",
    "features.dashboards.team",
    "features.ticket_creation",
)
# The features' own switches (the deploy settings' features.*, config.json's `features`): a configuration with one of
# them is in the new shape already.
NEW_SWITCHES = (
    "review_automation",
    "merge_queue",
    "dashboards",
    "alerts",
    "ticket_creation",
)


def _get(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _pop(d: dict, dotted: str) -> None:
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.get(part)
        if not isinstance(cur, dict):
            return
    cur.pop(parts[-1], None)


def _set(d: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


class _Missing:
    pass


_MISSING = _Missing()


def _blank(v) -> bool:
    return v is _MISSING or v is None or v == "" or v == {} or v == []


def is_on(v) -> bool:
    """Whether an older switch's value turns it on: anything but false, "off", 0, blank or null."""
    return v not in (_MISSING, None, False, "off", "", 0)


def old_features(raw: dict) -> dict:
    """The older per-capability switches a configuration has: each features.<name> that OLD_FEATURES names."""
    feats = raw.get("features") if isinstance(raw.get("features"), dict) else {}
    return {k: v for k, v in feats.items() if k in OLD_FEATURES}


def _switches(raw: dict, out: dict, at: str) -> list:
    """The features' own switches made from the older ones (SWITCHES, the module's docstring), set in `out`: each
    older key's move as (older key, its new key or None, the line)."""
    feats = old_features(raw)
    named = [k for k in SWITCH_KEYS if _get(raw, k) is not _MISSING]
    if not feats and not named:
        return []
    phase = (
        str(_first(raw, "deployment.phase", "phase") or "observe").strip() or "observe"
    )
    profile = (
        str(_first(raw, "deployment.profile", "profile") or "full").strip() or "full"
    )
    authority = str(
        _get(raw, "merge_authority")
        if _get(raw, "merge_authority") is not _MISSING
        else "none"
    )
    acting = PHASES.index(phase) if phase in PHASES else 0
    # What the profile turned on unless features said otherwise, and what it turned off whatever they said.
    turned_on = {"review-posting"} if profile == "review-only" else set()

    def was(name: str) -> bool:
        if name in feats:
            return is_on(feats[name])
        return name in turned_on

    reviews = profile != "merge-queue"
    queue = profile != "review-only"
    new = {
        "features.review_automation.enabled": reviews,
        "features.review_automation.mode": (
            "live" if reviews and was("review-posting") and acting >= 1 else "prototype"
        ),
        "features.merge_queue.enabled": queue,
        "features.merge_queue.mode": (
            "live" if queue and phase == "queue" else "prototype"
        ),
        "features.merge_queue.auto_merge": (
            "enabled"
            if was("auto-merge") or authority == "arm-cleared"
            else "hints_only"
        ),
        "features.merge_queue.auto_hold": was("holds"),
        "features.merge_queue.develop_guard": queue,
        "features.dashboards.admin.enabled": queue,
        "features.dashboards.team.enabled": was("team-dashboard"),
        "features.alerts.enabled": was("chat-alerts") or was("needs-you-alerts"),
        "features.alerts.develop_breaks": was("chat-alerts")
        or not was("needs-you-alerts"),
        "features.alerts.needs_you": was("needs-you-alerts") or not was("chat-alerts"),
        "features.ticket_creation.enabled": was("follow-up-tickets")
        or was("found-issue-tickets"),
        "features.ticket_creation.triggers.review_follow_up": was("follow-up-tickets")
        or not was("found-issue-tickets"),
        "features.ticket_creation.triggers.out_of_scope_issue": was(
            "found-issue-tickets"
        )
        or not was("follow-up-tickets"),
    }
    for name in ("remote-gpu-tests", "weekly-report"):
        if name in feats:
            new[OLD_FEATURES[name][0]] = was(name)
    out_feats = out.get("features") if isinstance(out.get("features"), dict) else {}
    for name in feats:
        out_feats.pop(name, None)
    out["features"] = out_feats
    for k in SWITCH_KEYS:
        _pop(out, k)
    if isinstance(out.get("deployment"), dict) and not out["deployment"]:
        del out["deployment"]
    for k, v in new.items():
        _set(out, k, v)
    moves = []
    for k in named:
        v = _get(raw, k)
        if k.endswith("phase"):
            line = (
                f"{k} {v}{at} is now two keys: features.review_automation.mode "
                f"{new['features.review_automation.mode']}, features.merge_queue.mode "
                f"{new['features.merge_queue.mode']}"
            )
            if phase == "guard" and (was("stability-issues") or was("incident-issues")):
                line += " (develop issues and incidents are filed once the merge queue is live)"
        elif k.endswith("profile"):
            line = (
                f"{k} {v}{at} is now the features' switches: features.review_automation.enabled "
                f"{str(reviews).lower()}, features.merge_queue.enabled {str(queue).lower()}, "
                f"features.dashboards.admin.enabled {str(queue).lower()}"
            )
        else:
            line = (
                f"{k} {v}{at} is now features.merge_queue.auto_merge "
                f"{new['features.merge_queue.auto_merge']}"
            )
        moves.append((k, None, line))
    for name, v in feats.items():
        target, why = OLD_FEATURES[name]
        if target is None:
            moves.append(
                (f"features.{name}", None, f"features.{name}{at} dropped: {why}")
            )
        else:
            shown = new.get(target, v)
            moves.append(
                (
                    f"features.{name}",
                    target,
                    f"features.{name}{at} is now {target} {_word(shown)}{why}",
                )
            )
    return moves


def _first(raw: dict, *keys):
    for k in keys:
        v = _get(raw, k)
        if v is not _MISSING and v not in (None, ""):
            return v
    return None


def _word(v) -> str:
    return str(v).lower() if isinstance(v, bool) else str(v)


def moves(raw: dict, where: str = "") -> tuple[dict, list]:
    """(the configuration with the new keys, each move as (older key, new key or None, its line)). An older key whose
    new key already holds a value is dropped (the new one wins), and the line says so. `where` names the file in the
    lines."""
    if not isinstance(raw, dict):
        return raw, []
    out = None
    found = []
    at = f" in {where}" if where else ""
    for old, new in RENAMED.items():
        v = _get(raw, old)
        if v is _MISSING:
            continue
        out = out if out is not None else copy.deepcopy(raw)
        have = _get(out, new)
        _pop(out, old)
        if _blank(have):
            _set(out, new, v)
            found.append((old, new, f"{old}{at} is now {new}: moved"))
        else:
            found.append(
                (
                    old,
                    None,
                    f"{old}{at} is now {new}, which is set already: {old} dropped",
                )
            )
    for old, why in DROPPED.items():
        if _get(raw, old) is _MISSING:
            continue
        out = out if out is not None else copy.deepcopy(raw)
        _pop(out, old)
        found.append((old, None, f"{old}{at} dropped: {why}"))
    if old_features(raw) or any(_get(raw, k) is not _MISSING for k in SWITCH_KEYS):
        out = out if out is not None else copy.deepcopy(raw)
        found += _switches(raw, out, at)
    have = out if out is not None else raw
    if _get(have, ALERTS) is not _MISSING and all(
        _get(have, k) is _MISSING for k in ALERT_KINDS
    ):
        out = out if out is not None else copy.deepcopy(raw)
        for k in ALERT_KINDS:
            _set(out, k, True)
        found.append(
            (
                ALERTS,
                None,
                f"{ALERTS}{at} is the alerts' one switch, and each kind of alert has its own now: "
                f"{ALERT_KINDS[0]} true, {ALERT_KINDS[1]} true",
            )
        )
    for feature in SENT_WHENEVER_ON:
        have = out if out is not None else raw
        switch, mode = f"{feature}.enabled", f"{feature}.mode"
        if is_on(_get(have, switch)) and _get(have, mode) is _MISSING:
            out = out if out is not None else copy.deepcopy(raw)
            _set(out, mode, "live")
            found.append(
                (
                    switch,
                    None,
                    f"{switch}{at} is true and {mode} is not set, which an older configuration sent whenever it "
                    f"was on: {mode} is now live",
                )
            )
    for old in {k.split(".", 1)[0] for k in list(RENAMED) + list(DROPPED) if "." in k}:
        if out is not None and out.get(old) == {}:
            del out[old]
    return (out if out is not None else raw), found


def migrate(raw: dict, where: str = "") -> tuple[dict, list[str]]:
    """(the configuration with the new keys, one line per key moved): `moves`, its lines only."""
    cfg, found = moves(raw, where)
    return cfg, [line for _old, _new, line in found]
