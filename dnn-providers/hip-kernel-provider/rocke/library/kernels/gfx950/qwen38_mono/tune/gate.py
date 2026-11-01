# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measurement discipline of the monokernel tuner (DESIGN 9.1).

- :func:`shuffled_rounds`: every round times every arm once, in a seeded
  shuffled order; the round is the unit of replication.
- Two controls per batch: ``ctl_ava`` (a second instance of the baseline
  arm: same code object, its own buffers / graph) and ``ctl_rc`` (the
  baseline compiled again into a separate code object). A separately
  compiled copy of the same code moves a layer by 0.1-0.3 %, the size of real
  wins, so a win must clear both.
- :func:`win_rule`: per-round ratios vs that round's baseline; WIN = median
  below the minimum of BOTH controls' rounds by ``margin`` (0.002) AND at
  least 3/4 of the rounds below 1; LOSS mirrored; else NOISE.
- :func:`e2e_rule`: served A/B: WIN = median ratio below the A-vs-A minimum
  AND at least 5 of 6 pairs below 1.
- :class:`ClockSampler`: SCLK / power sampled during the timed section
  (``amd-smi``, else sysfs); :func:`throttled_rounds` names rounds whose
  median SCLK fell below ``(1 - tol)`` of the session median, which
  :func:`win_rule` drops.

Ratios only leave this module; absolute samples stay in the caller's private
record.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

__all__ = [
    "BASE",
    "CTL_AVA",
    "CTL_RC",
    "ClockSampler",
    "Verdict",
    "e2e_rule",
    "per_round_ratios",
    "shuffled_rounds",
    "throttled_rounds",
    "win_rule",
]

BASE = "base"
CTL_AVA = "ctl_ava"
CTL_RC = "ctl_rc"


@dataclass(frozen=True)
class Verdict:
    """A lever decision: ``WIN`` / ``LOSS`` / ``NOISE``, with the evidence."""

    verdict: str
    median: float
    frac_below: float
    floor: float
    ceil: float
    rounds: int
    dropped: tuple = ()

    def to_json(self) -> dict:
        return {
            "verdict": self.verdict,
            "median": self.median,
            "frac_below": self.frac_below,
            "control_floor": self.floor,
            "control_ceil": self.ceil,
            "rounds": self.rounds,
            "dropped_rounds": list(self.dropped),
        }


def shuffled_rounds(
    arms: Sequence[str],
    rounds: int,
    run: Callable[[str, int], float],
    *,
    seed: int = 0,
    on_round: Callable[[int, str], None] | None = None,
) -> list[dict[str, float]]:
    """``rounds`` rounds; in each, ``run(arm, round)`` for every arm in a
    seeded shuffled order (a new order every round). Returns one ``{arm:
    time}`` per round."""
    rng = random.Random(f"rounds:{seed}")
    out = []
    for r in range(int(rounds)):
        order = list(arms)
        rng.shuffle(order)
        row = {}
        for a in order:
            if on_round is not None:
                on_round(r, a)
            row[a] = float(run(a, r))
        out.append(row)
    return out


def per_round_ratios(rounds: Sequence[Mapping[str, float]], base: str = BASE) -> dict[str, list[float]]:
    """``{arm: [time / that round's base time, ...]}`` (base excluded)."""
    out: dict[str, list[float]] = {}
    for row in rounds:
        b = row.get(base)
        if not b or not math.isfinite(b):
            continue
        for a, t in row.items():
            if a == base or not math.isfinite(t):
                continue
            out.setdefault(a, []).append(t / b)
    return out


def win_rule(
    arm: Sequence[float],
    controls: Sequence[Sequence[float]],
    *,
    margin: float = 0.002,
    frac: float = 0.75,
    drop: Sequence[int] = (),
) -> Verdict:
    """DESIGN 9.1: ``arm`` and each control are per-round ratios vs the
    baseline (rounds aligned); ``drop``: round indices to leave out
    (throttled)."""
    keep = [i for i in range(len(arm)) if i not in set(drop)]
    a = [arm[i] for i in keep]
    cs = [[c[i] for i in keep if i < len(c)] for c in controls]
    cs = [c for c in cs if c]
    if not a or not cs:
        return Verdict("NOISE", math.nan, math.nan, math.nan, math.nan, len(a), tuple(drop))
    med = statistics.median(a)
    floor = min(min(c) for c in cs)
    ceil = max(max(c) for c in cs)
    below = sum(x < 1.0 for x in a) / len(a)
    above = sum(x > 1.0 for x in a) / len(a)
    if med < floor - margin and below >= frac:
        v = "WIN"
    elif med > ceil + margin and above >= frac:
        v = "LOSS"
    else:
        v = "NOISE"
    return Verdict(v, med, below, floor, ceil, len(a), tuple(drop))


def e2e_rule(pairs: Sequence[float], a_vs_a: Sequence[float], *, need: int = 5) -> Verdict:
    """Served A/B (DESIGN 9.1): ``pairs`` are per-pair ratios (arm / pinned
    hybrid), ``a_vs_a`` the hybrid-vs-hybrid ratios of the same cell."""
    if not pairs or not a_vs_a:
        return Verdict("NOISE", math.nan, math.nan, math.nan, math.nan, len(pairs))
    med = statistics.median(pairs)
    n_below = sum(x < 1.0 for x in pairs)
    n_above = sum(x > 1.0 for x in pairs)
    lo, hi = min(a_vs_a), max(a_vs_a)
    need_frac = need / max(6, len(pairs))
    if med < lo and n_below >= need_frac * len(pairs):
        v = "WIN"
    elif med > hi and n_above >= need_frac * len(pairs):
        v = "LOSS"
    else:
        v = "NOISE"
    return Verdict(v, med, n_below / len(pairs), lo, hi, len(pairs))


# ----------------------------------------------------------------- clocks
def _amd_smi_sample(gpu: int) -> tuple[float, float] | None:
    exe = shutil.which("amd-smi") or ("/opt/rocm/bin/amd-smi" if os.path.exists("/opt/rocm/bin/amd-smi") else None)
    if exe is None:
        return None
    try:
        r = subprocess.run(
            [exe, "metric", "-g", str(gpu), "--clock", "--power", "--json"],
            capture_output=True, text=True, timeout=10,
        )  # fmt: skip
        data = json.loads(r.stdout)
    except (subprocess.SubprocessError, ValueError, OSError):
        return None
    return parse_amd_smi_metric(data)


def _num(x) -> float:
    """``2100``, ``"2100 MHz"`` or ``{"value": 2100, "unit": "MHz"}`` -> 2100.0."""
    if isinstance(x, Mapping):
        return _num(x.get("value"))
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.split()[0])
        except (ValueError, IndexError):
            return math.nan
    return math.nan


def parse_amd_smi_metric(data) -> tuple[float, float]:
    """``(sclk_mhz, power_w)`` of the first GPU in ``amd-smi metric --clock
    --power --json`` output (``[{"gpu": 0, "clock": {"gfx_0": {"clk": ...}},
    "power": {"socket_power": ...}}]``; older releases nest the list under
    ``"gpu_data"``)."""
    if isinstance(data, Mapping) and "gpu_data" in data:
        data = data["gpu_data"]
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, Mapping):
        return math.nan, math.nan
    clock = data.get("clock") or {}
    sclk = math.nan
    for name in ("gfx_0", "sclk", "gfx"):
        c = clock.get(name)
        if c is not None:
            sclk = _num(c.get("clk") if isinstance(c, Mapping) and "clk" in c else c)
            break
    power = data.get("power") or {}
    pw = math.nan
    for name in ("socket_power", "average_socket_power", "current_socket_power"):
        if name in power:
            pw = _num(power[name])
            if math.isfinite(pw):
                break
    return sclk, pw


def _sysfs_sample(bdf: str = "") -> tuple[float, float] | None:
    """SCLK (the ``*`` level of ``pp_dpm_sclk``) and power of the card at PCI
    ``bdf`` (``0000:65:00.0``; empty: the first card that has them)."""
    for card in sorted(glob.glob("/sys/class/drm/card*/device")):
        if bdf and os.path.basename(os.path.realpath(card)).lower() != bdf.lower():
            continue
        f = os.path.join(card, "pp_dpm_sclk")
        if not os.path.exists(f):
            continue
        try:
            sclk = math.nan
            for line in open(f).read().splitlines():
                if "*" in line:
                    sclk = float(line.split(":")[1].strip().split("Mhz")[0].split("MHz")[0])
            power = math.nan
            for h in glob.glob(os.path.join(card, "hwmon/hwmon*/power1_average")):
                power = float(open(h).read().strip()) / 1e6
            return sclk, power
        except (OSError, ValueError, IndexError):
            continue
    return None


@dataclass
class ClockSampler:
    """Samples ``(t, sclk_mhz, power_w)`` every ``period`` s in a thread
    while running (``with ClockSampler() as cs: ...``). ``gpu``: the job's
    GPU index as ``amd-smi`` numbers it (inside a q38 job ``amd-smi`` sees
    only the job's GPUs). ``marks`` record ``(t, label)`` so samples can be
    assigned to rounds."""

    gpu: int = 0
    bdf: str = ""
    period: float = 0.5
    samples: list = field(default_factory=list)
    marks: list = field(default_factory=list)
    source: str = ""

    def __post_init__(self):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _one(self):
        # the job's own card by PCI address first (sysfs); amd-smi numbering
        # inside a job is not guaranteed to be the HIP device's
        if self.bdf:
            s = _sysfs_sample(self.bdf)
            if s is not None and not all(math.isnan(x) for x in s):
                self.source = "sysfs:" + self.bdf
                return s
        s = _amd_smi_sample(self.gpu)
        if s is not None and not all(math.isnan(x) for x in s):
            self.source = "amd-smi"
            return s
        if not self.bdf:
            s = _sysfs_sample()
            if s is not None:
                self.source = "sysfs:first-card"
        return s

    def _loop(self):
        while not self._stop.is_set():
            s = self._one()
            if s is not None:
                self.samples.append((time.time(), s[0], s[1]))
            self._stop.wait(self.period)

    def mark(self, label: str) -> None:
        """Start a labelled segment and take one sample at its start, so even
        sub-second rounds (shorter than ``period``) get a reading."""
        t = time.time()
        self.marks.append((t, label))
        s = self._one()
        if s is not None:
            self.samples.append((t, s[0], s[1]))

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, name="clock-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
        return False

    def per_label(self) -> dict[str, dict]:
        """``{label: {"sclk": median, "power": median, "n": count}}`` over
        the samples between a mark and the next."""
        out: dict[str, dict] = {}
        marks = sorted(self.marks) + [(math.inf, "")]
        for (t0, lab), (t1, _l) in zip(marks, marks[1:]):
            sc = [s for t, s, _p in self.samples if t0 <= t < t1 and math.isfinite(s)]
            pw = [p for t, _s, p in self.samples if t0 <= t < t1 and math.isfinite(p)]
            d = out.setdefault(lab, {"sclk": [], "power": []})
            d["sclk"] += sc
            d["power"] += pw
        return {
            k: {
                "sclk": statistics.median(v["sclk"]) if v["sclk"] else None,
                "power": statistics.median(v["power"]) if v["power"] else None,
                "n": len(v["sclk"]),
            }
            for k, v in out.items()
        }


def throttled_rounds(per_round_sclk: Mapping[int, float | None], *, tol: float = 0.05) -> list[int]:
    """Round indices whose median SCLK is below ``(1 - tol)`` of the
    session's median (rounds without samples are kept)."""
    vals = [v for v in per_round_sclk.values() if v]
    if not vals:
        return []
    ref = statistics.median(vals)
    return sorted(r for r, v in per_round_sclk.items() if v and v < (1.0 - tol) * ref)
