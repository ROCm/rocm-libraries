# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What the two skinny GEMM kernel selectors share: the config identity, the
sweep results loader, the disjoint assignment and the argparse types.

Importing this module does not initialize HIP."""

from __future__ import annotations

import argparse
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple, TypeVar

import yaml

import skinny_gemm_support as support
from skinny_gemm_support import Problem
from sweep_skinny_gemm_results import msgpack, read_results, without_launch_lists

LABEL_KEYS = ("name", "m", "n", "k", "spk", "useBeta", "betaMode")
# The verified values whose timing is not trusted: a mismatch and no check.
UNTRUSTED_VERIFIED = ("failed", "no")
REASON_NO_TIMINGS = "no verified configurations in the sweep top"
REASON_NO_TIMINGS_USE_BETA = "no verified USE_BETA1 configurations in the sweep top"
# Everything reading a malformed or foreign results file may raise.
INPUT_ERRORS: tuple[type[Exception], ...] = (
    OSError,
    UnicodeError,
    ValueError,
    KeyError,
    TypeError,
    RuntimeError,
    yaml.YAMLError,
    *((msgpack.exceptions.UnpackException,) if msgpack is not None else ()),
)


class ConfigIdentity(NamedTuple):
    bm: int
    tn: int
    lanes_k: int
    waves: int
    pf: int
    up_iters: int
    group: int
    xcds: int
    grid3d: bool
    blocks: int | None
    mfma: bool
    lds: bool
    use_beta: bool
    use_split_k: bool


@dataclass(frozen=True)
class SweepCase:
    index: int
    problem: dict
    best_us: float | None
    top: list
    reason: str | None = None

    @property
    def label(self) -> dict:
        return {key: self.problem[key] for key in LABEL_KEYS}

    def text(self) -> str:
        p = self.problem
        spk = "k" if p["spk"] is None else p["spk"]
        return f"{p['name']} m={p['m']} n={p['n']} k={p['k']} spk={spk}"


def identity_of(cfg: dict) -> ConfigIdentity:
    grid3d = bool(cfg["grid3d"])
    return ConfigIdentity(
        cfg["bm"],
        cfg["tn"],
        cfg["lanesK"],
        cfg["waves"],
        cfg["prefetch"],
        int(cfg["upIters"]),
        cfg["group"],
        cfg["numXcds"],
        grid3d,
        None if grid3d else cfg["blocks"],
        bool(cfg.get("mfma", False)),
        bool(cfg.get("lds", False)),
        bool(cfg.get("useBeta", False)),
        # The kernel without split-K records no spk.
        cfg.get("spk") is not None,
    )


def tie_break_key(identity: ConfigIdentity) -> tuple[int, ...]:
    return tuple(-1 if value is None else int(value) for value in identity)


def rounded(value: float) -> float:
    return round(float(value), 4)


def identity_record(identity: ConfigIdentity) -> dict:
    record = {name: getattr(identity, name) for name in identity._fields}
    record["grid3d"] = bool(identity.grid3d)
    return record


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be at least 0, got {value}")
    return value


def non_negative_finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(
            f"must be a finite non-negative number, got {text}"
        )
    return value


def positive_finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(
            f"must be a finite positive number, got {text}"
        )
    return value


def read_document(path: str) -> dict:
    """Loads a results document and checks its shape."""
    document = read_results(path)
    if not isinstance(document, dict) or not isinstance(
        document.get("results"), list
    ):
        raise ValueError(f"{path}: not a sweep results document (no results list)")
    if not isinstance(document.get("meta"), dict):
        raise ValueError(f"{path}: results meta is not a mapping")
    return document


def check_complete(meta: dict, require_one_shot: bool) -> None:
    if require_one_shot and meta.get("oneShot") is not True:
        raise ValueError(
            "results must come from sweep_skinny_gemm.py --one-shot "
            "(meta.oneShot is not true)"
        )
    if meta.get("interrupted") or meta.get("aborted"):
        raise ValueError("results are from an interrupted or aborted sweep")
    cases_done = meta.get("casesDone")
    cases_total = meta.get("casesTotal")
    if cases_done is None or cases_total is None or cases_done != cases_total:
        raise ValueError(
            f"results are incomplete ({cases_done} of {cases_total} cases finished)"
        )


@contextmanager
def entry_errors(position: int):
    """Reports a missing or mistyped field of results[position] as a ValueError."""
    try:
        yield
    except (AttributeError, KeyError, TypeError) as err:
        raise ValueError(f"results[{position}]: malformed entry: {err!r}") from err


def valid_time(item: dict, position: int, name: str) -> float:
    time_us = item["timeUs"]
    valid = (
        isinstance(time_us, (int, float))
        and not isinstance(time_us, bool)
        and math.isfinite(time_us)
        and time_us > 0
    )
    if not valid:
        raise ValueError(
            f"results[{position}] ({name}): timeUs {time_us!r} is not a finite "
            "positive number"
        )
    return time_us


def trusted_top(entry: dict, position: int, use_beta_only: bool) -> list:
    """The ranked entries of a result that may be selected, fastest first."""
    name = entry["problem"]["name"]
    top = []
    for item in entry.get("top") or []:
        if item.get("verified") in UNTRUSTED_VERIFIED:
            continue
        if not isinstance(item["config"], dict):
            raise TypeError("top entry config is not a mapping")
        if use_beta_only and not item["config"].get("useBeta"):
            continue
        valid_time(item, position, name)
        top.append(without_launch_lists(item))
    return sorted(top, key=lambda item: (item["timeUs"], item["rank"]))


def make_case(
    index: int, position: int, entry: dict, use_beta_only: bool
) -> SweepCase:
    with entry_errors(position):
        top = trusted_top(entry, position, use_beta_only)
        if top:
            return SweepCase(index, entry["problem"], top[0]["timeUs"], top)
        reason = REASON_NO_TIMINGS_USE_BETA if use_beta_only else REASON_NO_TIMINGS
        return SweepCase(index, entry["problem"], None, top, reason)


def wanted_entry(entry: dict, only: str | None, use_beta_only: bool) -> bool:
    problem = entry["problem"]
    if use_beta_only and not problem["useBeta"]:
        return False
    return not only or only in problem["name"]


def load_cases(
    document: dict,
    only: str | None,
    limit: int | None,
    *,
    require_one_shot: bool,
    use_beta_only: bool,
) -> list[SweepCase]:
    """The cases of a validated document. A case without a trusted timing is kept
    with best_us None and the reason, so that it is reported as uncovered."""
    check_complete(document["meta"], require_one_shot)
    wanted = []
    for position, entry in enumerate(document["results"]):
        with entry_errors(position):
            if wanted_entry(entry, only, use_beta_only):
                wanted.append((position, entry))
    if limit is not None:
        wanted = wanted[:limit]
    return [
        make_case(index, position, entry, use_beta_only)
        for index, (position, entry) in enumerate(wanted)
    ]


def fastest_winner(options: dict):
    """The (identity, entry) with the least entry.time_us; ties go to the
    identity with the least tie_break_key."""
    return min(
        options.items(),
        key=lambda pair: (pair[1].time_us, tie_break_key(pair[0])),
    )


X = TypeVar("X")


def assign_disjoint(
    qualified: dict[int, dict[ConfigIdentity, X]], library: set[ConfigIdentity]
) -> dict[int, tuple[ConfigIdentity, X]]:
    """Each case goes to its fastest qualifier in the library: its own fastest
    one when that is in the library, else the fastest of the library."""
    covers: dict[int, tuple[ConfigIdentity, X]] = {}
    for index, per_id in qualified.items():
        options = {i: e for i, e in per_id.items() if i in library}
        if options:
            covers[index] = fastest_winner(options)
    return covers


def target_problem(problem_record: dict, cfg: dict, spk: int | None) -> Problem:
    """Re-targets cfg to the swept problem problem_record, with the split spk."""
    tp = problem_record
    align = support.alignUp(tp["k"], 4)
    # Explicit, so that an old VALU config never gets the MFMA defaults.
    mfma = bool(cfg.get("mfma", False))
    return support.makeProblem(
        tp["m"],
        tp["n"],
        tp["k"],
        spk=spk,
        bm=cfg["bm"],
        tn=cfg["tn"],
        lanes_k=cfg["lanesK"],
        waves=cfg["waves"],
        pf=cfg["prefetch"],
        group=cfg["group"],
        xcds=cfg["numXcds"],
        up_scales=0 if mfma else int(cfg["upIters"] > 0),
        blocks=None if cfg["grid3d"] else cfg["blocks"],
        padding=(tp["ldA"] - align, tp["ldB"] - align, 0),
        ldD=tp["ldD"] // 2,
        ldC=tp["ldC"] // 2,
        alpha=tp["alpha"],
        beta=tp["beta"],
        use_beta=tp["useBeta"],
        mfma=mfma,
        lds=bool(cfg.get("lds", False)),
    )
