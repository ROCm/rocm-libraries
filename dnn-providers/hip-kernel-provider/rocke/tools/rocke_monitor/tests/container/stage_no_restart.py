# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The image test's no-restart stage (tests/container/image.py): each deploy key of class no_restart (its x-applies in
the deploy schema), changed on a running deployment as a person changes a setting, is
read at its next use, with no restart.

The stage walks the deploy path as the deploy-path stage does (tests/container/stage_deploy_path.py DeployPath), on a
deployment of its own, with the tick every TICK_MINUTES minutes, each key of CHECKS at its first value (filled), and
the stand-in GitHub serving the open PRs of `open_prs`. Once the path is done, it approves the large PR's in-depth
review as the operator does (approve_large); once a full tick is on record with what each key's reader shows
(recorded), it edits every key of CHECKS to its second value in the settings file, runs `rkmon confirm --yes`, which
shows each change as read at its next use with no restart and names apply, then the command its next line gives,
`rkmon apply`, which neither drains nor restarts, and each next line after it to the done line. After it, the ticks
(their files, the monitor's state and the reads the stand-in GitHub logs) and the commands the session runs show each
key's new value where they showed its old one before: one result for each key, naming its reader and when it reads
the key. The container, when it started, and the monitor's long-lived processes (LONG_LIVED) are the same at the end
as before the change.

CHECKS is the one list of the keys the stage proves: the stage fails at its start unless the keys the deploy schema
classes no_restart are exactly its keys, and tests/test_change_plan.py checks the same, so that no key is classed
no_restart without its check here.

Every result is named after the stage, the walk's own among them (DeployPath.result). The stage's objects are the
walk's, named after its scope, <names>-no-restart, and removed at the end as the walk removes them.
"""

import datetime as dt
import json
import os
import re
import sys
import time
from typing import Callable, NamedTuple

import hostrun
from stage_deploy_path import (
    REPOSITORY,
    DeployPath,
    edited,
    next_command,
    next_line,
    outcome,
)

STAGE = "no-restart"
# The tick's interval, in minutes: the schema's least, so that a tick reads each key within a minute or two
TICK = "monitor.tick_interval_minutes"
TICK_MINUTES = 1
# How long the stage waits, in minutes: for a full tick on record with what each key's reader shows (recorded); then,
# after apply, for the ticks to show each key's new value
READY_MINUTES = 10
AFTER_MINUTES = 8
POLL_S = 10
# The monitor's long-lived processes, by the last word of their command (rkmon.py serve, rkmon.py orchestrate): each
# keeps what it read as it started, so a restart shows as another process
LONG_LIVED = ("serve", "orchestrate")
# apply's lines when it restarts the container (docker/rkmon restart) or drains it first (rkmon_lib/drain.py)
RESTART_WORDS = ("restart:", "drain:")
# The open PRs the stand-in GitHub serves (open_prs), by number: LARGE, of LARGE_LINES changed lines, settled long
# ago, over the autopilot's cap (in_depth_review_over_lines) at its first value and within it at its second, so that
# its review, once the stage approves it as the operator does (approve_large), is left to be started by hand before
# the change and starts by itself after it; PUSHED, pushed as the stage starts, waiting for the push cool-down;
# REVIEWED, settled long ago, with our review of its first push, submitted as the stage starts, waiting for the
# follow-up cool-down
LARGE, PUSHED, REVIEWED = 1, 2, 3
LARGE_LINES = 30
# The first word of the advance's line on a review it leaves to be started by hand (rkmon_lib/review.py manual_lines),
# and of its line on a review it starts (start_review)
LEFT, STARTED = "REVIEW_MANUAL", "REVIEW_START"
# A price in US dollars per million tokens, for each of a model's prices: invented
PRICES = {
    "input": 5.0,
    "output": 25.0,
    "cache_read": 0.5,
    "cache_write_5_minutes": 6.25,
    "cache_write_1_hour": 10.0,
}
RA = "features.review_automation"
GUARD = "features.merge_queue.develop_guard"
FULL_TICK_MAX = "monitor.full_tick_max_interval_minutes"
SPEND = f"{RA}.spend.enabled"
PRICE_TABLE = f"{RA}.spend.usd_per_million_tokens"
CI_RED = f"{GUARD}.ci_red_label"
EXTRA = f"{GUARD}.ci_red_extra_labels"
STABILITY = f"{GUARD}.stability_label"
# A develop issue's breaking PR's author, a stability issue's assignee
AUTHOR = "dev-a"
# rk review-split's first line (rkmon_lib/split.py summary): the most changed lines in a part
SPLIT_SIZE = re.compile(
    r"SPLIT #\d+ \([^()]*\): \d+ parts? of at most (\d+) changed lines"
)
# rk issue-file's refusal of a stability issue that carries the extra labels (rkmon_lib/breakage.py parse_file): the
# extra labels, the CI-red label they go with, and the stability label
EXTRA_REFUSED = re.compile(
    r"labels: (.+?) go(?:es)? with (\S+) only: a (\S+) issue is a break CI does not see"
)
# The tick's read of the open develop issues (rkmon_lib/breakage.py fetch_open), as the stand-in GitHub logs it: the
# repository it reads them in
DEVELOP_ISSUES = re.compile(
    r"/repos/([\w.-]+/[\w.-]+)/issues\?labels=[^&]*&state=open&"
)
# What the stage reads of the monitor's state, in its container, as the monitor's user: one JSON object, a Seen. Its
# argument, JSON: "state", the state folder; "files", the state files by what they hold ("cap": the GitHub request
# cap's record, "scaler": the scaler's state, "spend": the spend card, "ticks": the folder of the files each tick and
# each review driver round write); "idle", the state's key of the last full tick; "large", the large PR's number, and
# "decisions", the first words of the advance's lines on it (LEFT, STARTED), of which it reads the last in those files,
# in their order; "prs", the numbers of the PRs whose records it reads.
SEEN_SCRIPT = r"""
import json, pathlib, re, sys
a = json.loads(sys.argv[1])
state, files = pathlib.Path(a["state"]), a["files"]
def load(name):
    try:
        data = json.loads((state / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
st = load("state.json")
decided = []
for f in sorted((state / files["ticks"]).glob("*.md")):
    text = f.read_text(encoding="utf-8", errors="replace")
    decided += re.findall(r"\b(%s) #%d\b" % ("|".join(map(re.escape, a["decisions"])), a["large"]), text)
prs = st.get("prs") or {}
mine = ("head_seen_at", "head_committed_at", "ours")
print(json.dumps({
    "idle": st.get(a["idle"]) or {},
    "prs": {str(n): {k: (prs.get(str(n)) or {}).get(k) for k in mine} for n in a["prs"]},
    "spend": {"at": (st.get("spend") or {}).get("at"), "card": (state / files["spend"]).is_file()},
    "cap": load(files["cap"]).get("cap_per_hour"),
    "pipes": (load(files["scaler"]).get("pipes") or {}).get("start"),
    "decided": decided[-1] if decided else None,
}))
"""
# rk issue-file on the file it reads on stdin, as the session runs it: the file written with the container's umask
# (docker/entrypoint.sh), so that the writer, of the monitor's group, reads it too (rkmon_lib/writer_run.py), and
# removed after it
ISSUE_FILE = 'umask 027; f="/tmp/no-restart-issue-$$.md"; cat > "$f" && rk issue-file "$f"; rc=$?; rm -f "$f"; exit $rc'


def _time(text) -> float:
    """TEXT, a time as the monitor writes one (store.now: 2001-02-03T04:05:06Z), in seconds."""
    return dt.datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()


def _last_full(seen: dict) -> dict:
    """The last full tick's record in SEEN (rkmon_lib/idle.py record_full), {} for none."""
    idle = seen.get("idle") or {}
    return idle if idle.get("full_at") else {}


def _reason(seen: dict):
    """Whether the last full tick ran because the longest time between full ticks is 0 (rkmon_lib/idle.py decide)."""
    full = _last_full(seen)
    return f"{FULL_TICK_MAX} is 0" in (full.get("why") or []) if full else None


def _timer(label: str) -> Callable:
    """Whether the last full tick recorded the timer LABEL (rkmon_lib/idle.py timer)."""

    def show(seen: dict):
        full = _last_full(seen)
        return label in (full.get("due") or {}) if full else None

    return show


def _minutes(label: str, since: Callable) -> Callable:
    """The minutes from SINCE(seen) to the time the last full tick recorded for the timer LABEL."""

    def show(seen: dict):
        due, start = (_last_full(seen).get("due") or {}).get(label), since(seen)
        return round((_time(due) - _time(start)) / 60) if due and start else None

    return show


def _pushed(seen: dict):
    """When the pushed PR's head was pushed, as the push cool-down counts it (rkmon_lib/store.py head_age_min): the
    earlier of its commit's time and the time a tick first saw it."""
    rec = (seen.get("prs") or {}).get(str(PUSHED)) or {}
    times = [t for t in (rec.get("head_seen_at"), rec.get("head_committed_at")) if t]
    return min(times, key=_time) if times else None


def _reviewed(seen: dict):
    """When our review of the reviewed PR was submitted, as the follow-up cool-down counts it (rkmon_lib/plan.py
    last_review_time)."""
    rec = (seen.get("prs") or {}).get(str(REVIEWED)) or {}
    return (rec.get("ours") or {}).get("at")


def _spend(seen: dict):
    """Whether the spend card was written: its time in the state, and its file."""
    s = seen.get("spend")
    return bool(s.get("at") and s.get("card")) if isinstance(s, dict) else None


def _value(name: str) -> Callable:
    def show(seen: dict):
        return seen.get(name)

    return show


def _labels(part: str) -> Callable:
    """What rk issue-file's refusal of a stability issue that carries the extra labels names (refused_labels): PART
    "red", the CI-red label; "stability", the stability label; "extra", the extra labels.
    """

    def show(seen: dict):
        return (seen.get("labels") or {}).get(part)

    return show


class Check(NamedTuple):
    """A no_restart key's check: VALUES, its value in the settings file before the change and after it; SHOWN, what
    its reader shows of each, as SHOW reads that from what the stage saw (a Seen); READER, what reads the key, and
    when; TICK, whether a tick reads it (else a command the stage runs as the session runs it).
    """

    values: tuple
    shown: tuple
    reader: str
    tick: bool
    show: Callable


# Each key the deploy schema classes no_restart, with its check (the module's docstring)
CHECKS = {
    FULL_TICK_MAX: Check(
        (60, 0),
        (False, True),
        "the tick's choice of a full or an idle tick (rkmon_lib/idle.py decide), at each tick",
        True,
        _reason,
    ),
    "integrations.github_cli.max_requests_per_hour": Check(
        (4000, 3000),
        (4000, 3000),
        "the GitHub request cap's record (rkmon_lib/gh_cap.py), which each tick writes as it starts, and each gh call "
        "reads",
        True,
        _value("cap"),
    ),
    f"{RA}.parallel_reviews": Check(
        (3, 2),
        (3, 2),
        "the parallel reviews' starting number (`rkmon_lib/scaler.py`), at each full tick and each review driver round",
        True,
        _value("pipes"),
    ),
    f"{RA}.large_prs.in_depth_review_over_lines": Check(
        (20, 40),
        (LEFT, STARTED),
        "the advance's choice between starting a review and leaving it to be started by hand (rkmon_lib/review.py "
        "pick_next, oversized), at each full tick and each review driver round",
        True,
        _value("decided"),
    ),
    f"{RA}.large_prs.split_part_lines": Check(
        (8000, 7000),
        (8000, 7000),
        "rk review-split (rkmon_lib/split.py part_lines), each time the session runs it",
        False,
        _value("split"),
    ),
    f"{RA}.schedule.push_cooldown_minutes": Check(
        (600, 500),
        (600, 500),
        "the push cool-down's timer (rkmon_lib/store.py settle_left), at each full tick, each review driver round and "
        "each review start",
        True,
        _minutes(f"settle #{PUSHED}", _pushed),
    ),
    f"{RA}.schedule.follow_up_cooldown_minutes": Check(
        (600, 500),
        (600, 500),
        "the follow-up cool-down's timer (rkmon_lib/plan.py gap_until), at each full tick, each review driver round "
        "and each review start",
        True,
        _minutes(f"gap #{REVIEWED}", _reviewed),
    ),
    SPEND: Check(
        (False, True),
        (False, True),
        "the spend card (rkmon_lib/spend.py tick), at each full tick",
        True,
        _spend,
    ),
    "features.weekly_report.enabled": Check(
        (False, True),
        (False, True),
        "the weekly report's timer (rkmon.py _weekly_due), at each full tick",
        True,
        _timer("weekly report"),
    ),
    f"{GUARD}.issue_repository": Check(
        ("", "acme/issues"),
        (REPOSITORY, "acme/issues"),
        "the read of the open develop issues (rkmon_lib/breakage.py fetch_open), at each full tick",
        True,
        _value("develop_issues"),
    ),
    CI_RED: Check(
        ("ci-red", "ci-red-2"),
        ("ci-red", "ci-red-2"),
        "rk issue-file's check of an issue's labels (rkmon_lib/breakage.py parse_file), each time the session runs it",
        False,
        _labels("red"),
    ),
    EXTRA: Check(
        (["ci-extra"], ["ci-extra-2"]),
        (["ci-extra"], ["ci-extra-2"]),
        "rk issue-file's check of an issue's labels (rkmon_lib/breakage.py parse_file), each time the session runs it",
        False,
        _labels("extra"),
    ),
    STABILITY: Check(
        ("stability", "stability-2"),
        ("stability", "stability-2"),
        "rk issue-file's check of an issue's labels (rkmon_lib/breakage.py parse_file), each time the session runs it",
        False,
        _labels("stability"),
    ),
}


def no_restart_keys(schema: dict) -> set:
    """The keys the deploy schema SCHEMA classes no_restart: each key whose own x-applies is no_restart, as the
    takes-effect table reads the classes (tools/generated.py)."""
    from rkmon_lib import change_plan, settings_file  # noqa: E402

    return {
        path
        for path, node, _required in settings_file.schema_keys(schema)
        if node.get("x-applies") == change_plan.NO_RESTART
    }


def filled(values: dict) -> None:
    """VALUES, the settings file's, as the stage fills them in after the walk's own values: the tick every TICK_MINUTES
    minutes, each key of CHECKS at its first value, and the prices of each built-in model, set while spend is off and
    read once it is on."""
    from rkmon_lib import deploy_tree  # noqa: E402

    deploy_tree.put(values, TICK, TICK_MINUTES)
    for key, check in CHECKS.items():
        deploy_tree.put(values, key, check.values[0])
    prices = {model: dict(PRICES) for model in deploy_tree.BUILTIN.values()}
    deploy_tree.put(values, PRICE_TABLE, prices)


def open_prs(at: str, line1: str) -> list:
    """The open PRs the stand-in GitHub serves (tests/fakes/fake_net.py CONFIG "prs"; LARGE, PUSHED and REVIEWED say
    what each is), AT being the stage's start and LINE1 the line 1 our review gives."""
    review = f"{line1}\n\n## PR #{REVIEWED}\n\nNo findings.\n"
    return [
        {"number": LARGE, "lines": LARGE_LINES},
        {"number": PUSHED, "at": at},
        {
            "number": REVIEWED,
            "pushes": 2,
            "reviews": [{"push": 1, "at": at, "body": review}],
        },
    ]


def issue_text(at: int) -> str:
    """An rk issue-file file (rkmon_lib/breakage.py parse_file) of a stability issue, with the settings at their values
    AT (0 before the change, 1 after it): labelled with the stability label and the extra labels of both values, and
    assigned to AUTHOR, the breaking PR's author. rk issue-file refuses it before it reads GitHub, as a stability issue
    carries no extra label, naming the extra labels it carries, the CI-red label they go with and the stability label
    (refused_labels)."""
    extras = [label for value in CHECKS[EXTRA].values for label in value]
    return (
        "---\n"
        "title: develop: a break the image test files\n"
        f"labels: {', '.join([CHECKS[STABILITY].values[at], *extras])}\n"
        f"assign: {AUTHOR}\n"
        "---\n"
        "A develop issue that its labels refuse.\n"
    )


def refused_labels(out: str) -> dict:
    """The labels rk issue-file's OUT, its refusal of a stability issue that carries the extra labels, names
    (EXTRA_REFUSED, OUT read unwrapped): {"extra": the extra labels, "red": the CI-red label, "stability": the stability
    label}; {} for none."""
    m = EXTRA_REFUSED.search(" ".join(out.split()))
    if not m:
        return {}
    return {"extra": m.group(1).split(", "), "red": m.group(2), "stability": m.group(3)}


def split_size(out: str):
    """The most changed lines in a part, as rk review-split's OUT gives it (SPLIT_SIZE, unwrapped); None for none."""
    m = SPLIT_SIZE.search(" ".join(out.split()))
    return int(m.group(1)) if m else None


def applies(out: str, key: str) -> str:
    """How confirm's OUT says the change of KEY applies: the line under the change's own line (rkmon_lib/confirm.py
    _setting), stripped; "" unless exactly one change line names KEY."""
    lines = [x.strip() for x in out.splitlines()]
    at = [
        i for i, x in enumerate(lines) if re.match(rf"[~+] {re.escape(key)}(\s|$)", x)
    ]
    return lines[at[0] + 1] if len(at) == 1 and at[0] + 1 < len(lines) else ""


def seen_args(state: str) -> dict:
    """SEEN_SCRIPT's argument, for the monitor's state folder STATE."""
    from rkmon_lib import gh_cap, idle, orchestrator, scaler, spend  # noqa: E402

    return {
        "state": state,
        "files": {
            "cap": gh_cap.RECORD,
            "scaler": scaler.STATE,
            "spend": spend.SUMMARY,
            "ticks": orchestrator.TICKS,
        },
        "idle": idle.KEY,
        "large": LARGE,
        "decisions": [LEFT, STARTED],
        "prs": [PUSHED, REVIEWED],
    }


def issues_read(log: list):
    """The repository of the last read of the open develop issues in LOG, the stand-ins' log (DEVELOP_ISSUES),
    answered; None for none."""
    found = [
        m.group(1)
        for r in log
        if r.get("status") == 200 and (m := DEVELOP_ISSUES.match(r.get("path") or ""))
    ]
    return found[-1] if found else None


def recorded(seen: dict) -> bool:
    """Whether SEEN shows a full tick on record with its fingerprint (rkmon_lib/idle.py), and what each key a tick
    reads shows."""
    full = _last_full(seen)
    ticked = [c for c in CHECKS.values() if c.tick]
    return isinstance(full.get("fp"), dict) and all(
        c.show(seen) is not None for c in ticked
    )


def utc_now() -> str:
    """Now, as the monitor writes a time (store.now)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class NoRestart(DeployPath):
    """One run of the stage: the walk (DeployPath) under the stage's name, with the stand-ins' open PRs and the keys
    filled in, then the change and what reads it (the module's docstring)."""

    def __init__(
        self, engine: str, image: str, pkg: str, names: str, run_opts: list, res
    ):
        super().__init__(engine, image, pkg, names, run_opts, res, STAGE)
        # the pushed PR's push, and our review of the reviewed one
        self.started = utc_now()

    # --- the walk, as this stage has it --------------------------------------------------------------------------
    def fakes_config(self) -> dict:
        """The walk's stand-ins, with the open PRs (open_prs), our review giving the deployment's name in its line 1
        (rkmon_lib/reviewer_name.py: the container's name, as the settings give no other).
        """
        from rkmon_lib import reviewer_name  # noqa: E402

        config = super().fakes_config()
        line1 = reviewer_name.REVIEW_LINE.format(
            name=self.scope, model="opus", effort="high"
        )
        config["prs"] = open_prs(self.started, line1)
        return config

    def fill_in(self) -> None:
        """The walk's values, then the stage's (filled), through the package's one writer of settings files."""
        from rkmon_lib import file_types, settings_file  # noqa: E402

        super().fill_in()
        doc = settings_file.read(str(self.file), file_types.DEPLOY)
        filled(doc.values)
        self.file.write_text(settings_file.dump(doc), encoding="utf-8")

    # --- the stage's start -----------------------------------------------------------------------------------------
    def classes(self) -> bool:
        """Whether the package's deploy schema classes no_restart exactly the keys of CHECKS, with its result."""
        sys.path.insert(0, self.pkg)
        from rkmon_lib import file_types  # noqa: E402

        classed = no_restart_keys(file_types.DEPLOY.load())
        ok = classed == set(CHECKS)
        if not ok:
            hostrun.indented(
                f"classed no_restart with no check here: {sorted(classed - set(CHECKS))}; checked here, and not "
                f"classed no_restart: {sorted(set(CHECKS) - classed)}"
            )
        return self.result(
            ok,
            "the deploy schema classes no_restart exactly the keys this stage checks",
        )

    # --- the deployment's container ------------------------------------------------------------------------------
    def running(self) -> dict:
        """What runs: the container (DeployPath.monitor), by its id, when it started and how often docker restarted it;
        and each of the monitor's long-lived processes (LONG_LIVED), by its pid and when it started, as ps in the
        container (DeployPath.inside) shows them.
        """
        rc, out = self.docker(
            "inspect",
            "--format",
            "{{.Id}} {{.State.StartedAt}} {{.RestartCount}}",
            self.monitor(),
        )
        found = {"container": out if rc == 0 else f"docker inspect exited {rc}"}
        rc, out = self.inside("ps", "-ww", "-eo", "pid,lstart,args")
        rows = [x.split() for x in out.splitlines()[1:]] if rc == 0 else []
        for name in LONG_LIVED:
            found[name] = sorted(
                " ".join(w[:6])
                for w in rows
                if len(w) > 7 and w[-1] == name and w[-2].endswith("rkmon.py")
            )
        return found

    # --- what the monitor shows --------------------------------------------------------------------------------------
    def seen(self) -> dict:
        """What the monitor shows now: its state (SEEN_SCRIPT), {} when that cannot be read, and the repository of its
        last read of the open develop issues, from the stand-ins' log (issues_read): a Seen, which commands() completes.
        """
        from rkmon_lib import container_layout, paths  # noqa: E402

        args = seen_args(f"{container_layout.MONITOR_HOME}/{paths.STATE}")
        rc, out = self.inside(
            "python3", "-I", "-B", "-c", SEEN_SCRIPT, json.dumps(args)
        )
        try:
            seen = json.loads(out.splitlines()[-1]) if rc == 0 else None
        except (ValueError, IndexError):
            seen = None
        if not isinstance(seen, dict):
            hostrun.indented(
                f"the monitor's state was not read: exit {rc}: {out[-500:]}"
            )
            return {}
        return dict(seen, develop_issues=issues_read(self.requests()))

    def commands(self, at: int) -> dict:
        """What the commands the session runs show of the keys they read, each run now, with the settings at their
        values AT (0 before the change, 1 after it): "split", the most changed lines in a part, from rk review-split of
        the large PR; "labels", the labels rk issue-file's refusal of a stability issue names (issue_text,
        refused_labels).
        """
        from rkmon_lib import outward  # noqa: E402

        rc, out = self.inside("rk", "review-split", str(LARGE), timeout=600)
        hostrun.indented(f"rk review-split {LARGE}: exit {rc}\n{out[-1500:]}")
        rc, said = self.inside(
            "sh",
            "-c",
            ISSUE_FILE,
            stdin=issue_text(at).encode(),
            env={outward.OWN_VAR: "1"},
        )
        hostrun.indented(f"rk issue-file, a stability issue: exit {rc}\n{said[-1500:]}")
        return {"split": split_size(out), "labels": refused_labels(said)}

    def approve_large(self) -> bool:
        """The large PR's in-depth review approved, as the operator approves it (rk review-approve), once a tick has
        read the PR: at the cap's first value it waits for that approval, and the advance decides nothing on it before.
        Whether it was approved within READY_MINUTES."""
        end = time.monotonic() + READY_MINUTES * 60
        while True:
            rc, out = self.inside("rk", "review-approve", str(LARGE))
            if rc == 0 or time.monotonic() >= end:
                break
            time.sleep(POLL_S)
        hostrun.indented(f"rk review-approve {LARGE}: exit {rc}\n{out[-500:]}")
        return self.result(
            rc == 0,
            "the large PR's in-depth review is approved, as the operator approves it",
        )

    def wait_for(self, done: Callable, minutes: float, also: dict) -> tuple:
        """seen(), with ALSO, every POLL_S seconds until DONE is true of it, for MINUTES at most: (the last one, whether
        DONE is true of it)."""
        end = time.monotonic() + minutes * 60
        while True:
            seen = {**self.seen(), **also}
            if done(seen) or time.monotonic() >= end:
                return seen, done(seen)
            time.sleep(POLL_S)

    # --- the change, and what reads it -------------------------------------------------------------------------------
    def change(self) -> bool:
        """Every key of CHECKS edited to its second value in the settings file, as a person edits a line; `rkmon
        confirm --yes`, then the command its next line gives, `rkmon apply`, and each next line after it to the done
        line, each with its result (the module's docstring): whether apply passed."""
        from rkmon_lib import change_plan, confirm  # noqa: E402

        text = self.file.read_text(encoding="utf-8")
        for key, check in CHECKS.items():
            text = edited(text, key, check.values[1])
        self.file.write_text(text, encoding="utf-8")
        n, no_restart = len(CHECKS), confirm.APPLIES[change_plan.NO_RESTART]
        rc, out = self.rkmon("confirm", "--yes")
        found = outcome(
            out, rf"CONFIRMED: {n} changes wait for apply; nothing runs differently yet"
        )
        unlike = [key for key in CHECKS if applies(out, key) != no_restart]
        ok = (
            rc == 0
            and bool(found)
            and not unlike
            and next_command(next_line(out))[1] == ["apply"]
        )
        if not ok:
            hostrun.indented(
                f"confirm: exit {rc}, outcome {found!r}, next line {next_line(out)!r}; not said to apply with no "
                f"restart: {unlike}"
            )
        self.result(
            ok,
            "confirm shows each key's change as read at its next use with no restart, and names apply",
        )
        if not ok:
            return False
        rc, out = self.follow(out)
        found = outcome(
            out,
            rf"APPLIED: the deployment runs your confirmed settings \({n} changes\)",
        )
        restarted = [
            x.strip() for x in out.splitlines() if x.strip().startswith(RESTART_WORDS)
        ]
        said = [
            f"apply: exit {rc}, outcome {found!r}, next line {next_line(out)!r}, a drain or a restart: {restarted}"
        ]
        ok = rc == 0 and bool(found) and not restarted and self.to_done(out, said)
        hostrun.indented("\n".join(said))
        return self.result(
            ok,
            "apply neither drains nor restarts, and its next lines end at the done line",
        )

    def change_and_read(self) -> None:
        """Once the path is done: a full tick on record with what each key's reader shows, and the commands' showing
        (before); the change; then what each reader shows after it, the ticks' waited for to show each key's new value;
        one result for each key, and one that the container and the long-lived processes are those of before.
        """
        if not self.approve_large():
            return
        before, ok = self.wait_for(recorded, READY_MINUTES, {})
        if not ok:
            hostrun.indented(f"what the stage saw: {json.dumps(before)[-3000:]}")
        self.result(
            ok,
            "a full tick is on record with what each key's reader shows, before the change",
        )
        before.update(self.commands(0))
        ran = self.running()
        if not self.change():
            return
        also = self.commands(1)
        after, _ok = self.wait_for(
            lambda s: all(c.show(s) == c.shown[1] for c in CHECKS.values()),
            AFTER_MINUTES,
            also,
        )
        for key, check in CHECKS.items():
            was, now = check.show(before), check.show(after)
            ok = was == check.shown[0] and now == check.shown[1]
            hostrun.indented(
                f"{key}: {json.dumps(was)} before, {json.dumps(now)} after; {json.dumps(check.shown[0])} and "
                f"{json.dumps(check.shown[1])} wanted"
            )
            self.result(
                ok,
                f"{key}, changed with no restart, is read at its next use: {check.reader}",
            )
        now = self.running()
        same = now == ran and all(ran.get(name) for name in LONG_LIVED)
        if not same:
            hostrun.indented(
                f"before the change: {json.dumps(ran)}\nat the end: {json.dumps(now)}"
            )
        self.result(
            same,
            "the container and the monitor's long-lived processes are the ones that ran before the change",
        )


def no_restart_stage(
    engine: str, image: str, pkg: str, names: str, run_opts: list, res
) -> None:
    """The stage (the module's docstring): its results through RES."""
    if os.name == "nt":
        print(f"SKIP  {STAGE}: rkmon runs on a Linux or macOS host, not on Windows")
        return
    t = NoRestart(engine, image, pkg, names, run_opts, res)
    try:
        if t.classes():
            t.prepare()
            t.path()
            if t.done:
                t.change_and_read()
            else:
                t.result(
                    False,
                    "the deploy path ends at the done line, after which the stage changes the keys",
                )
            t.after()
    except (RuntimeError, OSError, ValueError) as exc:
        hostrun.indented(str(exc))
        res.result("FAIL", f"{STAGE}: the stage's own setup: {exc.__class__.__name__}")
    finally:
        t.clean()
