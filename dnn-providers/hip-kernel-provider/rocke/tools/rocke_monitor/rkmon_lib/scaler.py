# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The scaler: how many parallel reviews and parallel builds run at once, from this machine's load, and whether another
draft may start now. Every change and every draft decision is one line of state/scaler.jsonl with its inputs and its
reason, so the policy can be tuned from real data.

**The current settings** (`adjust`, at every review advance: the review driver's minute and the tick's). Each runs
within a fixed range, `ranges`: pipes 1..config.pipes_max(), heavy jobs 1..config.checks_max(). `review.pipes`
and `checks.limit` are the starting settings (and a new value there starts again from it). The scaler moves the current
setting one step at a time, from samples at least SAMPLE_S seconds apart:
- down, when SAMPLES_DOWN samples in a row are over: the CPU (the load average over the cores; inside a CPU-limited
  container, the larger of that and its own CPU time over its limit) or the memory (used over total; inside a
  memory-limited container, the fuller of the machine and its limit; tmpfs pages count as used, so the RAM disk's files
  do) above
  `max_host_usage_pct` (`ceiling_pct`), or, for the pipes, the drafter binding's endpoint spent (an HTTP 429, or
  BUDGET_SHARE_MAX of its budget used, in the last BUDGET_MINUTES minutes);
- up, when SAMPLES_UP samples in a row have headroom and COOLDOWN_MIN minutes passed since the last change: every
  piece of the setting busy (no point in more otherwise), the CPU and the memory at least HEADROOM_PCT points under
  the limit with one more piece's RAM disk footprint counted, the RAM disk with room for that footprint, the endpoint
  not spent, and the pipes under what the reviewer endpoint allows (config.pipes: a local Ollama allows 1 unless
  confirmed; nothing caps them with no drafter);
- otherwise it holds, and the streaks start again.
Lowering never stops running work: the pipes and the heavy jobs in flight finish, and nothing new starts until the
running count is under the new setting (review._fill, jobs.heavy_busy).

**The RAM disk** (rkmon_lib/ramdisk.py), while it is on: each sample also measures what each review in flight (its
worktrees, its review tools' build, its review directories) and each heavy job (its build) holds there, with the slots
its PR holds when they are there (ramdisk.slot_folders: a heavy job's while one runs, else the review's), and a piece's
peak, when it ends, raises the running high estimate of its kind (ramdisk.SEED_GB before the first; a lower peak pulls
the estimate down by DECAY of the difference). `room` lets a pipe or a heavy job start only when the RAM disk's free
space, less what the running pieces may still grow to (their estimates over their use; a review awaiting verification,
whose draft and checks are done, at most READY_RESERVE_GB: its trees no longer grow; the whole never more than the
RAM disk), holds its estimate, and the memory stays under the limit with that counted as used. Otherwise it waits:
nothing spills to the disk.

**Draft starts** (`may_start`, the hook the draft stages call before each draft: draft_stage.pool_advance,
draft_stage.shadow_start, review.start_review). Inputs (`inputs`):
- budget: the drafting role's binding's bucket over the last BUDGET_MINUTES minutes (llm_client.window_stats): the
  share of the budget its tokens and requests used, the seconds callers waited, the HTTP 429s, the callers active and
  the requests in flight now;
- host: this machine's load against max_host_usage_pct (`host_load`);
- queue: the reviews waiting in the plan (follow-ups and first rounds), the oldest one's wait, and the develop fixes
  and priorities among them;
- running: the drafts running now, live and shadow together, and the ceiling (bindings.parallel_max).
The policy (`decide`): another draft may start while the last BUDGET_MINUTES minutes used under BUDGET_SHARE_MAX of
the budget, no HTTP 429 came in them, and the drafts running are under the ceiling.

Its own state (the current settings, the streaks, the last sample, the footprints) is state/scaler_state.json, apart
from state.json, so a sample never makes the next tick full.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

from . import bindings, config, idle, ramdisk, registry, state_store
from .store import minutes_since, now

BUDGET_MINUTES = 5
BUDGET_SHARE_MAX = 0.8
LOG = "scaler.jsonl"
STATE = "scaler_state.json"
NAMES = {"pipes": "parallel reviews", "heavy": "parallel builds"}
KIND = {
    "pipes": "pipe",
    "heavy": "heavy",
}  # the footprint kind of each setting's pieces
SAMPLE_S = 50  # samples closer than this are not counted
SAMPLES_DOWN = 2
SAMPLES_UP = 3
COOLDOWN_MIN = 5
HEADROOM_PCT = 10
DECAY = 0.1
MIN_PEAK_GB = 0.05  # a piece that held less than this on the RAM disk (it ran elsewhere) teaches nothing
# What a review awaiting verification may still add to the RAM disk: its draft and checks are done, so its trees and
# builds no longer grow; the verifier's own probes and notes may add a little.
READY_RESERVE_GB = 0.5
MEMINFO = "/proc/meminfo"
CGROUP = "/sys/fs/cgroup"


def ranges() -> dict:
    """Each setting's range: the pipes 1..config.pipes_max(), the heavy jobs 1..config.checks_max() (the deploy
    schema's maximums, read at their first use)."""
    return {"pipes": (1, config.pipes_max()), "heavy": (1, config.checks_max())}


def ceiling_pct(cfg: dict) -> int:
    """`max_host_usage_pct`: the share of the CPU and of the memory the scaler keeps the machine under (blank or 0: its
    default)."""
    return registry.number(cfg, "max_host_usage_pct", int, registry.FALSY)


# --- the host ---------------------------------------------------------------------------------------------------------


def _cgroup_memory(root: str) -> tuple[float, float] | None:
    """(total, used) bytes inside a memory-limited cgroup (v2): its memory.max, and memory.current less the inactive
    file cache it can drop (tmpfs pages are not file cache: they count as used). None without a limit.
    """
    base = pathlib.Path(root)
    try:
        limit = (base / "memory.max").read_text().strip()
        if not limit.isdigit():
            return None
        current = int((base / "memory.current").read_text().strip())
        stat = dict(
            ln.split()
            for ln in (base / "memory.stat").read_text().splitlines()
            if len(ln.split()) == 2
        )
        return float(limit), float(max(0, current - int(stat.get("inactive_file", 0))))
    except (OSError, ValueError):
        return None


def _cgroup_cpu(root: str, prev: dict | None, clock=None) -> float | None:
    """Inside a CPU-limited cgroup (v2, cpu.max with a quota): its own CPU time since the previous sample over its
    limit, in percent (None on the first sample, or without a limit). `prev` keeps the last reading.
    """
    base = pathlib.Path(root)
    try:
        quota, period = (base / "cpu.max").read_text().split()[:2]
        if not quota.isdigit():
            return None
        cpus = int(quota) / int(period)
        usec = int(
            next(
                ln.split()[1]
                for ln in (base / "cpu.stat").read_text().splitlines()
                if ln.startswith("usage_usec ")
            )
        )
    except (OSError, ValueError, StopIteration, IndexError):
        return None
    if prev is None:
        return None
    t, last = (clock or time.time)(), prev.get("cpu")
    prev["cpu"] = [t, usec]
    if not last or t <= last[0]:
        return None
    return round(100.0 * (usec - last[1]) / 1e6 / ((t - last[0]) * cpus), 1)


def cpu_count() -> int:
    """The CPUs this process may run on (its affinity: a container's or taskset's cpuset), else the machine's."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def host_load(
    cfg: dict,
    meminfo: str = MEMINFO,
    cgroup: str | None = None,
    loadavg=os.getloadavg,
    prev: dict | None = None,
) -> dict:
    """The machine's CPU and memory use in percent, the larger of the two, the memory in GB, and the configured
    ceiling. CPU: the load average over the cores it may use (cpu_count); inside a CPU-limited container, the larger of
    that and its own CPU time over its limit since the previous sample (`prev`, the scaler's state, keeps the
    reading); each at most 100% (a load past the cores, or a burst past the quota, is full). Memory: used over
    total from `meminfo` (MemAvailable leaves out tmpfs pages, so the RAM disk's files count as used); inside a
    memory-limited container (the default /proc/meminfo, a cgroup with a memory.max), its own use over its limit when
    that is the fuller of the two (its use counts the tmpfs pages it holds)."""
    out = {
        "limit_pct": ceiling_pct(cfg),
        "cpu_pct": None,
        "mem_pct": None,
        "pct": None,
        "mem_total_gb": None,
        "mem_used_gb": None,
    }
    try:
        out["cpu_pct"] = min(100.0, round(100.0 * loadavg()[0] / cpu_count(), 1))
    except OSError:
        pass
    use_cg = bool(cgroup) or meminfo == MEMINFO
    cpu = _cgroup_cpu(cgroup or CGROUP, prev) if use_cg else None
    if cpu is not None:  # the busier of the machine and the container's own limit
        out["cpu_pct"] = max(min(cpu, 100.0), out["cpu_pct"] or 0.0)
    total = used = None
    try:
        info = {}
        for line in pathlib.Path(meminfo).read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition(":")
            if k in ("MemTotal", "MemAvailable"):
                info[k] = int(v.split()[0]) * 1024
        total, used = float(info["MemTotal"]), float(
            info["MemTotal"] - info["MemAvailable"]
        )
    except (OSError, ValueError, KeyError, IndexError):
        pass
    cg = _cgroup_memory(cgroup or CGROUP) if use_cg else None
    if cg and (
        total is None or cg[1] / cg[0] > used / total
    ):  # the fuller of the machine and the container's limit
        total, used = cg
    if total:
        out["mem_pct"] = round(100.0 * used / total, 1)
        out["mem_total_gb"], out["mem_used_gb"] = round(total / 2**30, 2), round(
            used / 2**30, 2
        )
    known = [x for x in (out["cpu_pct"], out["mem_pct"]) if x is not None]
    out["pct"] = max(known) if known else None
    return out


def queue_pressure(st: dict, plan: list[dict]) -> dict:
    """The reviews waiting in the plan, the oldest one's wait in minutes, and the develop fixes and priorities among
    them."""
    waiting = [i for i in plan or [] if i.get("kind") in ("FOLLOW_UP", "FIRST_ROUND")]
    ages = []
    for i in waiting:
        seen = (st["prs"].get(str(i["pr"])) or {}).get("head_seen_at")
        if seen:
            try:
                ages.append(minutes_since(seen))
            except ValueError:
                pass
    prio = {str(k) for k in st.get("develop_fixes") or {}} | {
        str(k) for k in st.get("user_priority") or {}
    }
    return {
        "backlog": len(waiting),
        "oldest_min": round(max(ages)) if ages else 0,
        "priorities": sum(1 for i in waiting if str(i["pr"]) in prio),
    }


def budget(cfg: dict, role: str = "drafter") -> dict:
    """The role's binding's bucket over the last BUDGET_MINUTES minutes; {} when the role has no binding (no drafter:
    no endpoint budget binds the pipes)."""
    from . import first_pass

    s = bindings.client_settings(cfg, role) or {}
    return first_pass.llm_client().window_stats(s, BUDGET_MINUTES) if s else {}


def _spent(b: dict) -> str:
    """Why the endpoint's budget is spent ("" when it is not)."""
    if b.get("rate_limited"):
        return f"{b['rate_limited']} HTTP 429 in the last {BUDGET_MINUTES} min"
    share = float(b.get("share") or 0)
    if share >= BUDGET_SHARE_MAX:
        return (
            f"the endpoint's budget: {share:.0%} used in the last {BUDGET_MINUTES} min"
        )
    return ""


# --- draft starts -----------------------------------------------------------------------------------------------------


def inputs(
    cfg: dict, st: dict, plan: list[dict], running: int, role: str = "drafter"
) -> dict:
    return {
        "budget": budget(cfg, role),
        "host": host_load(cfg),
        "queue": queue_pressure(st, plan),
        "running": running,
        "ceiling": bindings.parallel_max(cfg),
    }


def decide(inp: dict) -> tuple[bool, str]:
    """The policy: under the ceiling, no 429 and under BUDGET_SHARE_MAX of the budget in the last minutes."""
    b = inp.get("budget") or {}
    if inp["running"] >= inp["ceiling"]:
        return (
            False,
            f"{inp['running']} drafts running: the ceiling is {inp['ceiling']}",
        )
    if b.get("rate_limited"):
        return False, f"{b['rate_limited']} HTTP 429 in the last {BUDGET_MINUTES} min"
    share = float(b.get("share") or 0)
    if share >= BUDGET_SHARE_MAX:
        return False, (
            f"the last {BUDGET_MINUTES} min used {share:.0%} of the budget (under "
            f"{BUDGET_SHARE_MAX:.0%} starts another)"
        )
    return True, (
        f"{share:.0%} of the budget used in the last {BUDGET_MINUTES} min, no 429, {inp['running']} of "
        f"{inp['ceiling']} drafts running"
    )


def _log(store, rec: dict) -> None:
    try:
        with open(store.dir / LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def may_start(
    cfg: dict,
    st: dict,
    store,
    plan: list[dict],
    pr: int,
    running: int,
    role: str = "drafter",
) -> tuple[bool, str]:
    """Whether a draft of PR `pr` by the role's binding may start now; the answer is recorded in state/scaler.jsonl."""
    inp = inputs(cfg, st, plan, running, role)
    ok, reason = decide(inp)
    _log(
        store,
        {
            "at": now(),
            "kind": "draft",
            "pr": pr,
            "role": role,
            "allow": ok,
            "reason": reason,
            "inputs": inp,
        },
    )
    return ok, reason


def recent(store, n: int = 10, kind: str | None = None) -> list[dict]:
    """The last n records (of one kind: "draft", "setting" or "footprint"; a record from before the kinds is a
    draft's)."""
    try:
        lines = (store.dir / LOG).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for x in reversed(lines):
        try:
            rec = json.loads(x)
        except ValueError:
            continue
        if kind is None or rec.get("kind", "draft") == kind:
            out.append(rec)
            if len(out) >= n:
                break
    return out[::-1]


def last(store, tail_bytes: int = 65536) -> dict | None:
    """The last draft decision, read from the end of the log."""
    try:
        with open(store.dir / LOG, "rb") as f:
            f.seek(max(0, f.seek(0, os.SEEK_END) - tail_bytes))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for x in reversed(lines):
        try:
            rec = json.loads(x)
        except ValueError:
            continue
        if rec.get("kind", "draft") == "draft":
            return rec
    return None


@idle.timer("held draft")
def _held_due(cfg: dict, st: dict, store):
    """When a draft the scaler held for its budget (the tokens, requests or HTTP 429s of the last BUDGET_MINUTES
    minutes) may start: BUDGET_MINUTES after the decision. A hold at the ceiling ends when a draft job ends.
    """
    rec = last(store)
    inp = (rec or {}).get("inputs") or {}
    if (
        rec
        and not rec.get("allow")
        and rec.get("at")
        and inp.get("running", 0) < inp.get("ceiling", 1)
    ):
        yield "held draft", idle.plus(rec["at"], BUDGET_MINUTES)


# --- the scaler's own state -------------------------------------------------------------------------------------------


def load_state(store) -> dict:
    try:
        s = json.loads((store.dir / STATE).read_text(encoding="utf-8"))
        return s if isinstance(s, dict) else {}
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
    ):  # no state yet, or a store without a folder
        return {}


def save_state(store, s: dict) -> None:
    """The scaler's state S in the state folder, written whole as its other JSON files are (state_store.write_file);
    a write that fails leaves the last one."""
    p = store.dir / STATE
    try:
        state_store.write_file(p, s)
    except OSError:
        pass


def start_value(cfg: dict, what: str) -> int:
    """The starting setting: review.pipes or checks.limit (held to the range)."""
    return config.pipes_asked(cfg) if what == "pipes" else config.checks_limit(cfg)


def pipe_cap(cfg: dict) -> tuple[int, str]:
    """What the reviewer endpoint allows the pipes, and why (config.pipes: a local Ollama allows 1 unless confirmed; no
    cap with no drafter or a drafter binding)."""
    p = config.pipes(dict(cfg, review={"pipes": config.pipes_max()}))
    return p["effective"], p["why"]


def _setting(cfg: dict, s: dict, what: str) -> dict:
    """The setting's record in the scaler's state, (re)seeded from the configuration's starting value: on first use,
    and whenever that value changes."""
    lo, hi = ranges()[what]
    start = start_value(cfg, what)
    rec = s.setdefault(what, {})
    if rec.get("start") != start or not isinstance(rec.get("current"), int):
        rec.update(
            start=start,
            current=start,
            over=0,
            under=0,
            changed_at=None,
            reason="the starting setting",
        )
    rec["current"] = max(lo, min(hi, rec["current"]))
    return rec


def current(cfg: dict, store, what: str) -> int:
    """The current setting of `what` ("pipes" or "heavy"): the scaler's, else the configuration's starting value."""
    s = load_state(store) if store is not None else {}
    rec = s.get(what) or {}
    lo, hi = ranges()[what]
    if rec.get("start") == start_value(cfg, what) and isinstance(
        rec.get("current"), int
    ):
        return max(lo, min(hi, rec["current"]))
    return start_value(cfg, what)


def pipes_in_effect(cfg: dict, store=None) -> int:
    """The pipes that may run now: the current setting, never past what the reviewer endpoint allows."""
    return min(current(cfg, store, "pipes"), pipe_cap(cfg)[0])


# --- footprints on the RAM disk ---------------------------------------------------------------------------------------


def estimates(s: dict) -> dict:
    """The running high estimates of a pipe's and a heavy job's RAM disk use, in GB."""
    return dict(ramdisk.SEED_GB, **(s.get("est") or {}))


def active(st: dict, store) -> dict[str, set[int]]:
    """The pieces that hold the RAM disk now: "pipe", every PR with a review in flight, a draft awaiting verification
    (its trees and review directory stay until its review is finished: review.review_finished) or a pool draft (it
    stages the trees); "heavy", every PR whose checks run."""
    from . import jobs, review

    pipes = {p["pr"] for p in review.pipes(st)} | {
        r["pr"] for r in st.get("review_ready") or []
    }
    pipes |= {int(k) for k in st.get("draft_pool") or {}}
    heavy = {
        p["pr"]
        for p in review.pipes(st)
        if jobs.status(store, p.get("checks")) == "running"
    }
    heavy |= {
        int(k)
        for k, j in (st.get("rechecks") or {}).items()
        if jobs.status(store, j) == "running"
    }
    return {"pipe": pipes, "heavy": heavy}


def awaiting(st: dict) -> set[int]:
    """The PRs whose review awaits verification (review_ready) and holds no pipe: their trees no longer grow."""
    from . import review

    return {r["pr"] for r in st.get("review_ready") or []} - {
        p["pr"] for p in review.pipes(st)
    }


def reserved_gb(
    s: dict,
    act: dict[str, set[int]],
    ready: set[int] = frozenset(),
    cap: float | None = None,
) -> float:
    """What the running pieces may still add to the RAM disk: each one's estimate over its last measured use; a review
    awaiting verification (`ready`), at most READY_RESERVE_GB. Never more than `cap` (the RAM disk's size).
    """
    est, uses = estimates(s), s.get("uses") or {}
    total = 0.0
    for kind, ns in act.items():
        for n in ns:
            grow = max(0.0, est[kind] - float(uses.get(f"{kind}:{n}") or 0.0))
            total += (
                min(grow, READY_RESERVE_GB) if kind == "pipe" and n in ready else grow
            )
    return round(min(total, cap) if cap else total, 2)


def reservation(st: dict, store, s: dict, u: dict) -> float:
    """reserved_gb for the work running now, against the RAM disk `u` (ramdisk.usage)."""
    return reserved_gb(s, active(st, store), awaiting(st), u.get("total_gb"))


def learn(
    cfg: dict, s: dict, act: dict[str, set[int]], store, measure=ramdisk.gb_of
) -> list[dict]:
    """Measure each active piece's RAM disk use, keep its peak, and fold the peaks of the pieces that ended into the
    estimates. Returns the footprint records it logged."""
    uses, peaks, est = (
        s.setdefault("uses", {}),
        s.setdefault("peaks", {}),
        s.setdefault("est", {}),
    )
    live = set()
    held = ramdisk.slot_folders(cfg)
    heavy = act.get("heavy") or set()
    for kind, ns in act.items():
        for n in ns:
            key = f"{kind}:{n}"
            live.add(key)
            folders = (
                ramdisk.pipe_folders(cfg, n)
                if kind == "pipe"
                else ramdisk.heavy_folders(cfg, n)
            )
            if kind == ("heavy" if n in heavy else "pipe"):
                folders = folders + held.get(n, [])  # the PR's slots: once
            uses[key] = measure(folders)
            peaks[key] = max(float(peaks.get(key) or 0.0), uses[key])
    out = []
    for key in sorted(set(peaks) - live):
        kind, peak = key.split(":")[0], float(peaks.pop(key))
        uses.pop(key, None)
        if peak < MIN_PEAK_GB or kind not in ramdisk.SEED_GB:
            continue
        old = estimates(s)[kind]
        new = round(peak if peak > old else old - (old - peak) * DECAY, 3)
        est[kind] = new
        rec = {
            "at": now(),
            "kind": "footprint",
            "what": kind,
            "pr": int(key.split(":")[1]),
            "peak_gb": peak,
            "from_gb": old,
            "to_gb": new,
        }
        _log(store, rec)
        out.append(rec)
    for key in set(uses) - live:
        uses.pop(key, None)
    return out


def room(
    cfg: dict, st: dict, store, kind: str, n: int, usage=None, host=None
) -> tuple[bool, str]:
    """Whether a pipe ("pipe") or a heavy job ("heavy") of PR n may start as far as the RAM disk goes: (True, "") while
    the RAM disk is off; else the RAM disk mounted, its free space less what the running pieces may still grow to
    holding the piece's estimate (nothing for a pipe whose trees are there already), and the memory under the limit with
    that counted as used."""
    root = ramdisk.path(cfg)
    if not root:
        return True, ""
    u = usage or ramdisk.usage(cfg)
    if not u["mounted"]:
        from . import deploy_tree, doctor  # only this message reads them

        return False, (
            f"the RAM disk is not mounted at {root} ({deploy_tree.name('ramdisk.enabled')}): nothing that goes on "
            f"it starts ({doctor.rk_command('doctor')})"
        )
    s = load_state(store)
    need = estimates(s)[kind]
    if kind == "pipe" and (pathlib.Path(root) / f"wt{n}").is_dir():
        need = 0.0
    held = reservation(st, store, s, u)
    if u["free_gb"] - held < need:
        return False, (
            f"the RAM disk has {u['free_gb']:.1f} GiB free, {held:.1f} GiB of it held for the work running; a "
            f"{'review' if kind == 'pipe' else 'build'} needs about {need:.1f} GiB there"
        )
    h = host or host_load(cfg)
    if h["mem_total_gb"]:
        pct = 100.0 * (h["mem_used_gb"] + held + need) / h["mem_total_gb"]
        if pct > h["limit_pct"]:
            return False, (
                f"memory: {h['mem_used_gb']:.1f} GiB used and {held + need:.1f} GiB more on the RAM disk "
                f"would be {pct:.0f}% of {h['mem_total_gb']:.0f} GiB, over {h['limit_pct']}%"
            )
    return True, ""


# --- the current settings ---------------------------------------------------------------------------------------------


def sample(cfg: dict, st: dict, store, s: dict) -> dict:
    """The inputs of one adjustment: the host, the RAM disk, the drafter's budget, the pieces running and the
    estimates."""
    from . import jobs, review

    u = ramdisk.usage(cfg)
    return {
        "host": host_load(cfg, prev=s.setdefault("prev", {})),
        "ramdisk": u,
        "budget": budget(cfg),
        "running": {
            "pipes": len(review.pipes(st)),
            "heavy": len(jobs.heavy_running(st, store)),
        },
        "pipe_cap": pipe_cap(cfg),
        "est": estimates(s),
        "reserved_gb": reservation(st, store, s, u) if u["mounted"] else 0.0,
    }


def load_text(inp: dict) -> str:
    """ "CPU 42%, mem 51%, RAM disk 9.2/24 GiB": the sample in words."""
    h, u = inp.get("host") or {}, inp.get("ramdisk") or {}
    parts = [
        f"CPU {h['cpu_pct']:.0f}%" if h.get("cpu_pct") is not None else "CPU ?",
        f"mem {h['mem_pct']:.0f}%" if h.get("mem_pct") is not None else "mem ?",
    ]
    if u.get("enabled"):
        parts.append(
            f"RAM disk {u['used_gb']:.1f}/{u['total_gb']:.0f} GiB"
            if u.get("mounted")
            else "RAM disk not mounted"
        )
    return ", ".join(parts)


def _over(what: str, inp: dict) -> list[str]:
    h = inp["host"]
    out = [
        f"{name} {h[key]:.0f}% over {h['limit_pct']}%"
        for name, key in (("CPU", "cpu_pct"), ("memory", "mem_pct"))
        if h.get(key) is not None and h[key] > h["limit_pct"]
    ]
    if what == "pipes" and _spent(inp.get("budget") or {}):
        out.append(_spent(inp["budget"]))
    return out


def _blocks_up(what: str, inp: dict, cur: int) -> list[str]:
    """Why the setting may not go up now ([] when it may)."""
    lo, hi = ranges()[what]
    h, u, kind = inp["host"], inp.get("ramdisk") or {}, KIND[what]
    if cur >= hi:
        return [f"at the top of its range ({hi})"]
    if what == "pipes" and cur >= inp["pipe_cap"][0]:
        return [
            f"the reviewer endpoint allows {inp['pipe_cap'][0]}: {inp['pipe_cap'][1]}"
        ]
    running = inp["running"][what]
    if running < cur:
        return [f"{running} of {cur} busy: no call for more"]
    out = []
    need = float(inp["est"][kind]) if u.get("enabled") else 0.0
    room_pct = h["limit_pct"] - HEADROOM_PCT
    if h.get("cpu_pct") is not None and h["cpu_pct"] > room_pct:
        out.append(
            f"CPU {h['cpu_pct']:.0f}% is within {HEADROOM_PCT} points of {h['limit_pct']}%"
        )
    if h.get("mem_total_gb"):
        mem = (
            100.0
            * (h["mem_used_gb"] + float(inp.get("reserved_gb") or 0) + need)
            / h["mem_total_gb"]
        )
        if mem > room_pct:
            out.append(
                f"memory would be {mem:.0f}% with one more {'review' if kind == 'pipe' else 'build'}"
                + (" on the RAM disk" if need else "")
                + f", within {HEADROOM_PCT} points of {h['limit_pct']}%"
            )
    if u.get("enabled"):
        if not u.get("mounted"):
            out.append("the RAM disk is not mounted")
        elif u["free_gb"] - float(inp.get("reserved_gb") or 0) < need:
            out.append(
                f"the RAM disk has no room for one more ({u['free_gb']:.1f} GiB free, about {need:.1f} GiB "
                "needed)"
            )
    if what == "pipes" and _spent(inp.get("budget") or {}):
        out.append(_spent(inp["budget"]))
    return out


def step(what: str, rec: dict, inp: dict, at: str) -> tuple[int, str]:
    """One sample of one setting: (its new current value, why). Updates the streaks in `rec`; pure otherwise."""
    lo, hi = ranges()[what]
    cur = rec["current"]
    over = _over(what, inp)
    if over:
        rec["over"], rec["under"] = int(rec.get("over") or 0) + 1, 0
        why = "; ".join(over)
        if cur <= lo:
            return cur, f"{why}; at the bottom of its range ({lo})"
        if rec["over"] >= SAMPLES_DOWN:
            rec["over"] = 0
            return cur - 1, f"{why} ({SAMPLES_DOWN} samples in a row): one fewer"
        return cur, f"{why} (sample {rec['over']} of {SAMPLES_DOWN} before one fewer)"
    rec["over"] = 0
    blocks = _blocks_up(what, inp, cur)
    if blocks:
        rec["under"] = 0
        return cur, "stays: " + "; ".join(blocks)
    rec["under"] = int(rec.get("under") or 0) + 1
    since = (
        minutes_since(rec["changed_at"], at) if rec.get("changed_at") else COOLDOWN_MIN
    )
    if rec["under"] >= SAMPLES_UP and since >= COOLDOWN_MIN:
        rec["under"] = 0
        return cur + 1, f"headroom ({SAMPLES_UP} samples in a row): one more"
    return cur, (
        f"headroom (sample {rec['under']} of {SAMPLES_UP} before one more"
        + (
            f"; {COOLDOWN_MIN - since:.0f} min of cool-down left"
            if since < COOLDOWN_MIN
            else ""
        )
        + ")"
    )


def adjust(
    cfg: dict,
    st: dict,
    store,
    inp: dict | None = None,
    at: str | None = None,
    clock=time.time,
) -> list[str]:
    """One adjustment of both settings (at most one sample every SAMPLE_S seconds; the RAM disk's footprints measured
    with it): a SCALER line for each change, which is also a line of state/scaler.jsonl.
    """
    s = load_state(store)
    t = clock()
    # A clock stepped back (an NTP correction, a WSL resync) puts the last sample in the future: sample now, rather
    # than stop sampling until the clock passes it.
    if 0 <= t - float(s.get("sampled_at") or 0) < SAMPLE_S:
        return []
    at = at or now()
    if ramdisk.path(cfg) and inp is None:
        u = ramdisk.usage(cfg)
        if u["mounted"]:
            learn(cfg, s, active(st, store), store)
    inp = inp or sample(cfg, st, store, s)
    s["sampled_at"] = t
    s["last"] = {
        "at": at,
        "host": inp["host"],
        "ramdisk": inp.get("ramdisk"),
        "reserved_gb": inp.get("reserved_gb"),
        "load": load_text(inp),
    }
    out = []
    for what in ranges():
        rec = _setting(cfg, s, what)
        cur = rec["current"]
        new, why = step(what, rec, inp, at)
        rec["reason"] = why
        if new != cur:
            rec.update(current=new, changed_at=at)
            lo, hi = ranges()[what]
            _log(
                store,
                {
                    "at": at,
                    "kind": "setting",
                    "what": what,
                    "from": cur,
                    "to": new,
                    "range": [lo, hi],
                    "reason": why,
                    "load": load_text(inp),
                    "inputs": inp,
                },
            )
            store.journal(
                "scaler",
                f"{NAMES[what]} {cur} -> {new} of {lo}-{hi}: {why} ({load_text(inp)})",
            )
            out.append(
                f"SCALER {NAMES[what]} {cur} -> {new} (of {lo}-{hi}): {why}; {load_text(inp)}"
            )
    save_state(store, s)
    return out


# --- rk status --------------------------------------------------------------------------------------------------------


def status_lines(cfg: dict, store, n: int = 5) -> list[str]:
    """Each current setting against its range with the scaler's reason and the last sample, the reviewer endpoint's
    cap, the footprint estimates, and the last `n` changes. Local files only."""
    s = load_state(store)
    last_ = s.get("last") or {}
    out = []
    for what, (lo, hi) in ranges().items():
        rec = s.get(what) or {}
        cur = current(cfg, store, what)
        line = f"{NAMES[what]} {cur} of {lo}–{hi}"
        line += (
            f" — {last_['load']}"
            if last_.get("load")
            else " — no sample yet (the review driver samples)"
        )
        if rec.get("reason") and rec.get("start") == start_value(cfg, what):
            line += f"; {rec['reason']}"
        if what == "pipes":
            cap, why = pipe_cap(cfg)
            if cap < cur:
                line += f"; {cap} run at most: {why}"
        out.append(line)
    if last_.get("at"):
        out.append(
            f"last sample {last_['at'][:16]}Z; starting values: parallel reviews {start_value(cfg, 'pipes')}, "
            f"parallel builds {start_value(cfg, 'heavy')}; max_host_usage_pct "
            f"{ceiling_pct(cfg)}"
        )
    if ramdisk.path(cfg):
        e = estimates(s)
        out.append(
            f"footprints on the RAM disk (running high estimates): a review {e['pipe']:.2f} GiB, a build "
            f"{e['heavy']:.2f} GiB"
        )
    changes = recent(store, n, kind="setting")
    out.append("last changes:" + ("" if changes else " none"))
    for c in changes:
        out.append(
            f"  {str(c.get('at'))[:16]}Z {NAMES.get(c.get('what'), c.get('what'))} {c.get('from')} -> "
            f"{c.get('to')}: {c.get('reason')} ({c.get('load', '')})"
        )
    return out
