# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The drain (apply step 2): a flag in the monitor's state under which nothing new starts while the work
in flight finishes; `rk drain` and `rk undrain`.

The flag is the state's `draining` record (store.drain_record): the command that set it, its limit in minutes, and
the process that set it. While it holds, no review, gate, develop sanity run or drift check starts (store.halted); it
holds nothing once the process that set it is gone, or DRAIN_MARGIN_MIN minutes after its limit (store.drain_stale).
A pause you set yourself (rk pause) is no drain: a drain neither sets it nor lifts it.

- WHY: the reason `rk drain` gives its flag, which `rk undrain` lifts. KIND: the kind `rk drain` registers its process
  under (procs.register), which rk stop leaves alone, so that the drain sees the stop and ends as it says.
- DRAINED and DRAIN_TIMEOUT: the outcome words of a drain, `rk drain`'s and `rk stop --drain`'s: nothing is in
  flight, or the limit passed with work still in flight. UNDRAINED: a flag lifted, `rk undrain`'s, and `rk drain`'s
  and `rk stop --drain`'s when another command lifted its flag. NOT_DRAINED: `rk stop --drain`'s when the
  configuration cannot be loaded: no review in flight can finish then, so the stop goes at once (rkmon.py cmd_stop).
- `in_flight(st, store) -> list[str]`: the work still running: reviews, rechecks, gates, develop sanity, a bisect,
  drafts, the model roles' jobs, GPU staging, builds and drift checks, each in a few words.
- `Busy`: the exception `hold` raises while another drain is in force: its `record`, that drain's; its words name the
  drain and what ends it.
- `hold(store, why, until_min)`: set the flag for WHY, the command, naming this process and UNTIL_MIN; Busy while
  another drain is in force: one whose process runs, or one left in place by another command (a flag WHY's command left
  in place, once drained, is taken over).
- `lift(store, why) -> dict | None`: lift WHY's flag; the record lifted, None when WHY's flag did not hold.
- `wait(cfg, store, why, timeout_min, poll_s, keep=False, say=print, sleep=time.sleep, clock=time.monotonic) ->
  Waited`: set WHY's flag (`hold`: Busy) and wait for the work in flight, advancing the parallel reviews meanwhile so
  that they finish without a tick, for at most TIMEOUT_MIN minutes. Drained, the flag stays for the caller to lift.
  Past the limit, the flag is lifted, unless KEEP: then it stays for the caller too (`rk stop --drain`, whose stop
  lifts it as it records itself, so that nothing starts in between); and the wait ends, with nothing to lift, when
  another command lifts or replaces the flag. A stop (rk stop, rk kill) ends it at its next look, the flag lifted: a
  stop holds more than a drain, as nothing runs under it and nothing starts until rk start, which ends a drain too;
  the reviews never advance under it. Anything that ends the wait early (an error, Ctrl-C, or SIGTERM under
  procs.signals_exit) lifts the flag before it goes on.
- `outcome_line(waited) -> str`: the outcome line of a wait that no other command lifted, `rk drain`'s and `rk stop
  --drain`'s, each adding what follows from it: DRAINED, nothing is left to drain, with the stop's words, when a stop
  ended the wait; DRAINED, nothing is in flight, and after how many minutes when it took one or more; DRAIN TIMEOUT,
  the limit and the work still in flight.
- `limits(cfg, timeout_min=None) -> (float, float)`: how long a drain waits and how often it looks, `rk drain`'s and
  `rk stop --drain`'s: its limit in minutes, TIMEOUT_MIN when given, else the configuration's update.drain_timeout_min;
  and the seconds between two looks at the work in flight, its update.poll_s; each the registry's default where the
  configuration has none.
- `main(cfg, store, timeout_min=None, say=print, sleep=time.sleep, clock=time.monotonic) -> int`: `rk drain
  [--timeout-minutes N]`: `wait`, with the limit and the looks `limits` gives (N minutes when given). DRAINED, exit 0,
  once nothing is in flight: the flag stays, no longer naming this process, so that it holds until `rk undrain` lifts
  it, or DRAIN_MARGIN_MIN minutes after the drain ended; DRAINED too, with no flag, when a stop holds (nothing is left
  to drain). Exit 1, the flag lifted: DRAIN TIMEOUT when the work is still in flight after N minutes, UNDRAINED when
  another command lifted the flag meanwhile. REFUSED, exit 4, while another drain is in force (`hold`).
- `undrain(store, out=print) -> int`: `rk undrain`: lift the flag `rk drain` set or left in place, so that new work
  starts again; a pause you set stays, and says so. Exit 0, and NOTHING TO DO when no flag of rk drain's is there.
"""

from __future__ import annotations

import os
import time
from typing import NamedTuple

from . import config, jobs, procs, registry, review
from . import store as store_mod

WHY = "rk drain"  # the reason `rk drain` gives its flag, which `rk undrain` lifts
KIND = "drain"  # the kind `rk drain` registers its process under, which rk stop leaves alone
# The outcome words of a drain: rk drain's and rk stop --drain's (DRAINED, DRAIN_TIMEOUT), and a flag lifted.
DRAINED, DRAIN_TIMEOUT, UNDRAINED = "DRAINED", "DRAIN TIMEOUT", "UNDRAINED"
# rk stop --drain's when the configuration cannot be loaded: no review in flight can finish, and the stop goes at once.
NOT_DRAINED = "NOT DRAINED"
TIMEOUT_KEY = "update.drain_timeout_min"
POLL_KEY = "update.poll_s"


def in_flight(st: dict, store) -> list[str]:
    """The work that is still running: see the module's docstring."""

    def running(job) -> bool:
        # A job record this code cannot read counts as running.
        try:
            return bool(job) and jobs.status(store, job) == "running"
        except Exception:  # noqa: BLE001
            return True

    out = []
    pipes = (
        list((st.get("review_pipes") or {}).values())
        if isinstance(st.get("review_pipes"), dict)
        else list(st.get("review_pipes") or [])
    )
    if st.get("review_pipeline"):
        pipes.append(st["review_pipeline"])
    out += [f"review #{p.get('pr')}" for p in pipes if p]
    out += [f"checks re-run of #{k}" for k in (st.get("rechecks") or {})]
    qs = st.get("queue_state") or {}
    if qs.get("gate") and running(qs["gate"].get("job")):
        out.append(f"gate on #{qs['gate'].get('pr')}")
    if qs.get("batch_gate") and running(qs["batch_gate"].get("job")):
        out.append("batch gate")
    g = st.get("guard") or {}
    if g.get("running") and running(g["running"].get("job")):
        out.append(f"develop sanity on {str(g['running'].get('sha'))[:10]}")
    if g.get("bisect") and running(g["bisect"].get("job")):
        out.append("bisect")
    # The draft pool's and the shadow drafts' jobs, the model roles' jobs and the remote GPU jobs still staging here:
    # each runs the code it started with, so a restart waits for them too.
    for key, what in (("draft_pool", "pool draft"), ("shadow_pool", "shadow draft")):
        for k, e in (st.get(key) or {}).items():
            if isinstance(e, dict) and running(e.get("job")):
                out.append(f"{what} #{k}")
    for role, r in (st.get("roles") or {}).items():
        if isinstance(r, dict) and running(r.get("job")):
            out.append(f"role job {role}")
    for j in (st.get("gpu") or {}).get("jobs") or []:
        if isinstance(j, dict) and running(j.get("stage_job")):
            out.append(f"GPU staging {j.get('id')}")
    js = st.get("jobs") or {}
    for key, what in (("heavy", "build job"), ("drift", "drift job")):
        held = js.get(key)
        # one job, or a list of holders
        for job in held if isinstance(held, list) else [held]:
            if isinstance(job, dict) and running(job):
                out.append(f"{what} {job.get('id')}")
    return sorted(set(out), key=out.index)


def _ours(d, why: str) -> bool:
    """Whether the drain record D is WHY's flag, set by this process: no other process runs with its pid, and a record
    that a process gone left with the same pid, `hold` replaced (store.drain_stale)."""
    return bool(d) and d.get("why") == why and d.get("pid") == os.getpid()


class Busy(Exception):
    """Another drain is in force: RECORD, its flag; the words say what ends it."""

    def __init__(self, record: dict):
        super().__init__(
            f"a drain is in force already ({record.get('why')}, since {record.get('since')}): "
            "it ends with that command, or rk start"
        )
        self.record = record


def hold(store, why: str, until_min: float) -> None:
    """Set WHY's flag for this process: see the module's docstring."""
    with store.locked(wait=True):
        st = store.load()
        d = store_mod.draining(st)
        if d and not store_mod.drain_stale(d) and (d.get("why") != why or d.get("pid")):
            raise Busy(d)
        st["draining"] = store_mod.drain_record(why, until_min)
        store.save(st)


def lift(store, why: str) -> dict | None:
    """Lift WHY's flag: see the module's docstring."""
    with store.locked(wait=True):
        st = store.load()
        d = st.get("draining")
        if not d or d.get("why") != why:
            return None
        st.pop("draining")
        store.save(st)
        return d


def _lift_ours(store, why: str) -> None:
    """Lift WHY's flag when this process still holds it (a flag another command set since stays)."""
    with store.locked(wait=True):
        st = store.load()
        if _ours(st.get("draining"), why):
            st.pop("draining")
            store.save(st)


class Waited(NamedTuple):
    """How a drain's wait ended: DRAINED, nothing in flight; else LEFT, the work still in flight when the limit
    passed; LIFTED, another command lifted or replaced the flag meanwhile; MINUTES, how long it waited; LIMIT, its
    limit in minutes; STOPPED, when a stop ended the wait (DRAINED too), the stop in store.halted's words.
    """

    drained: bool
    left: list
    lifted: bool
    minutes: float
    limit: float
    stopped: str = ""


def _listed(items) -> str:
    return ", ".join(items)


def outcome_line(waited: Waited) -> str:
    """The outcome line of a wait that no other command lifted: see the module's docstring."""
    if waited.stopped:
        return f"{DRAINED}: nothing is left to drain: {waited.stopped}"
    if waited.drained:
        minutes = int(waited.minutes)
        return f"{DRAINED}: nothing is in flight" + (
            f" after {minutes} minutes" if minutes else ""
        )
    return f"{DRAIN_TIMEOUT} after {waited.limit:.0f} minutes: {_listed(waited.left)} still in flight"


def wait(
    cfg: dict,
    store,
    why: str,
    timeout_min: float,
    poll_s: float,
    keep: bool = False,
    say=print,
    sleep=time.sleep,
    clock=time.monotonic,
) -> Waited:
    """Set WHY's flag and wait for the work in flight: see the module's docstring."""
    hold(store, why, timeout_min)
    start = clock()
    end = start + timeout_min * 60
    said = None
    try:
        while True:
            with store.locked(wait=True):
                st = store.load()
                if store_mod.stopped(st):
                    if _ours(st.get("draining"), why):
                        st.pop("draining")
                        store.save(st)
                    minutes = (clock() - start) / 60
                    halt = store_mod.halted(st)
                    return Waited(True, [], False, minutes, (end - start) / 60, halt)
                if not _ours(st.get("draining"), why):
                    minutes = (clock() - start) / 60
                    return Waited(False, [], True, minutes, (end - start) / 60)
                # A review that cannot advance here still counts as in flight.
                try:
                    review.advance(cfg, st, store, st.get("last_plan", []))
                except Exception as exc:  # noqa: BLE001
                    say(f"  (the reviews did not advance here: {str(exc)[:200]})")
                store.save(st)
            left = in_flight(st, store)
            now = clock()
            limit = (end - start) / 60
            if said is None:
                say(
                    f"drain: nothing new starts; the work in flight finishes first (up to {limit:.0f} minutes): "
                    f"{_listed(left)}"
                    if left
                    else "drain: nothing new starts; nothing is in flight"
                )
            elif left and left != said:
                say(f"drain: still in flight: {_listed(left)}")
            said = left
            if not left:
                return Waited(True, [], False, (now - start) / 60, limit)
            if now >= end:
                if not keep:
                    _lift_ours(store, why)
                return Waited(
                    False, left, False, (now - start) / 60, (end - start) / 60
                )
            sleep(poll_s)
    except BaseException:
        _lift_ours(store, why)
        raise


def _setting(cfg: dict, name: str):
    """The configuration's NAME, else its default (the registry's)."""
    value = config.lookup(cfg or {}, name)
    return value if value not in (None, "") else registry.key(name).default


def limits(cfg: dict, timeout_min=None) -> tuple:
    """A drain's limit in minutes and the seconds between its looks: see the module's docstring."""
    return (
        float(_setting(cfg, TIMEOUT_KEY) if timeout_min is None else timeout_min),
        float(_setting(cfg, POLL_KEY)),
    )


def _handover(store, minutes: float) -> None:
    """The drained flag kept for the caller: no longer this process's, it holds until `rk undrain` lifts it, or
    DRAIN_MARGIN_MIN minutes after the drain ended (store.drain_stale)."""
    with store.locked(wait=True):
        st = store.load()
        d = st.get("draining")
        if _ours(d, WHY):
            d.pop("pid", None)
            d.pop("pstart", None)
            d["until_min"] = round(minutes, 1)
            store.save(st)


def main(
    cfg: dict,
    store,
    timeout_min=None,
    say=print,
    sleep=time.sleep,
    clock=time.monotonic,
) -> int:
    """`rk drain [--timeout-minutes N]`: see the module's docstring."""
    limit, poll_s = limits(cfg, timeout_min)
    try:
        with procs.signals_exit():
            waited = wait(
                cfg,
                store,
                WHY,
                limit,
                poll_s,
                say=say,
                sleep=sleep,
                clock=clock,
            )
    except Busy as exc:
        say(f"REFUSED: {exc}")
        return 4
    if waited.lifted:
        say(f"{UNDRAINED}: the flag was lifted meanwhile (rk undrain, or rk start)")
        return 1
    if waited.drained and not waited.stopped:
        _handover(store, waited.minutes)
    say(outcome_line(waited) + ("" if waited.drained else "; the drain is lifted"))
    return 0 if waited.drained else 1


def undrain(store, out=print) -> int:
    """`rk undrain`: see the module's docstring."""
    d = lift(store, WHY)
    if d is None:
        out("NOTHING TO DO: no drain of rk drain's is there")
        return 0
    paused = (store.load().get("paused") or {}).get("at")
    out(
        f"{UNDRAINED}: the drain since {d.get('since')} is lifted; "
        + (
            f"the pause you set at {paused} stays (rk resume ends it)"
            if paused
            else "new work starts again"
        )
    )
    return 0
