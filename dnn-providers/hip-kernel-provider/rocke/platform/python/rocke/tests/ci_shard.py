# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Which CI shard a run is, and which keys that shard owns.

This is the shard rule of the ``rocke.tests.ctest_shard`` pytest plugin,
without pytest, so a runner that is not pytest (a golden check that runs as one
CTest entry) splits its work the same way. The shard comes from these sources,
in this order:

1. A non-blank ``ROCKE_TEST_SHARD`` (``i/n`` or ``i,j,.../n``, 0-based).
2. The 0-based gtest pair ``GTEST_SHARD_INDEX`` and ``GTEST_TOTAL_SHARDS``,
   only when the run asks for it (the plugin's and the runners'
   ``--rocke-shard-from-gtest-env``).
3. No shard: the run does everything.

TheRock's 1-based ``SHARD_INDEX`` and ``TOTAL_SHARDS`` are not read. A key (a
test's shard key, a golden family) belongs to the shard ``sha256(key) mod n``,
so each key falls in exactly one shard of ``n``, whatever else is selected.

A key may also carry a declared cost (:func:`place_keys`): the keys with a cost
are spread over the shards heaviest first, each onto the shard holding the
least declared cost so far, so a few heavy keys no longer land together by
chance. Keys without a cost keep the hash rule, so a run that declares no cost
splits exactly as before. Every shard of one run places the same keys with the
same costs, so each key still runs in exactly one shard.

A runner prints :func:`header_line` and :meth:`Shard.split_line`, the lines
the plugin prints, so one log parser reads both.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
from collections.abc import Iterable, Mapping

__all__ = [
    "ACTIVATION_FLAG",
    "CHILD_ENV_EXCLUDED",
    "GTEST_INDEX_ENV",
    "GTEST_TOTAL_ENV",
    "SHARD_ENV",
    "Shard",
    "check_cost",
    "current_shard",
    "header_line",
    "parse_shard",
    "place_keys",
    "resolve_shard",
    "shard_of",
    "unsharded_environment",
]

SHARD_ENV = "ROCKE_TEST_SHARD"
GTEST_INDEX_ENV = "GTEST_SHARD_INDEX"
GTEST_TOTAL_ENV = "GTEST_TOTAL_SHARDS"
# The option that makes a run take its shard from the gtest pair. A CTest entry
# whose command carries it runs only its CI shard's share.
ACTIVATION_FLAG = "--rocke-shard-from-gtest-env"
# A child pytest that must run unsharded, with only the plugins that it names,
# must not inherit these variables. SHARD_INDEX and TOTAL_SHARDS are
# TheRock's 1-based pair.
CHILD_ENV_EXCLUDED = (
    SHARD_ENV,
    GTEST_INDEX_ENV,
    GTEST_TOTAL_ENV,
    "SHARD_INDEX",
    "TOTAL_SHARDS",
    "PYTEST_ADDOPTS",
    "PYTEST_PLUGINS",
)


def parse_shard(text: str) -> tuple[frozenset[int], int]:
    """``"i/n"`` or ``"i,j,.../n"`` -> ``({i, j, ...}, n)``."""
    picks, sep, total = text.strip().partition("/")
    if not sep:
        raise ValueError(f"shard {text!r} is not i/n")
    n = int(total)
    idx = frozenset(int(p) for p in picks.split(",") if p.strip())
    if n < 1 or not idx or any(not 0 <= i < n for i in idx):
        raise ValueError(f"shard {text!r} out of range")
    return idx, n


def shard_of(key: str, n: int) -> int:
    """The shard of ``n`` that ``key`` belongs to."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % n


def check_cost(key: str, cost: object) -> None:
    """Raise ``ValueError`` unless ``cost``, the declared cost of ``key``, is
    a positive finite number."""
    if not (
        isinstance(cost, (int, float))
        and not isinstance(cost, bool)
        and math.isfinite(cost)
        and cost > 0
    ):
        raise ValueError(
            f"shard cost {cost!r} of {key!r} is not a positive finite number"
        )


def place_keys(
    keys: Iterable[str], total: int, costs: Mapping[str, float] | None = None
) -> dict[str, int]:
    """The shard of ``total`` that each of ``keys`` runs in.

    A key without a cost in ``costs`` goes to :func:`shard_of`'s shard. The
    keys with a cost (any positive unit the keys of one run share, e.g.
    seconds) go heaviest first, ties by key, each to the shard holding the
    least declared cost so far; among equally loaded shards, the first at or
    after the key's own hash shard. A cost for a key not in ``keys`` is
    ignored. Raises ``ValueError`` for a cost that is not a positive finite
    number."""
    unique = list(dict.fromkeys(keys))
    declared = {} if costs is None else costs
    weighted = {key: declared[key] for key in unique if key in declared}
    for key, cost in weighted.items():
        check_cost(key, cost)
    placement = {key: shard_of(key, total) for key in unique if key not in weighted}
    load = [0.0] * total
    for key in sorted(weighted, key=lambda key: (-weighted[key], key)):
        home = shard_of(key, total)
        shard = min(
            range(total), key=lambda index: (load[index], (index - home) % total)
        )
        placement[key] = shard
        load[shard] += weighted[key]
    return placement


def _count(name: str, raw: str) -> int:
    if not (raw.isascii() and raw.isdigit()):
        raise ValueError(f"{name}={raw!r} is not a non-negative integer")
    return int(raw)


def _gtest_shard(env: Mapping[str, str]) -> str | None:
    """``"<index>/<total>"`` from the gtest pair; ``None`` when both are
    blank."""
    index_text = env.get(GTEST_INDEX_ENV, "").strip()
    total_text = env.get(GTEST_TOTAL_ENV, "").strip()
    if not index_text and not total_text:
        return None
    if not index_text:
        raise ValueError(f"{GTEST_TOTAL_ENV}={total_text} needs {GTEST_INDEX_ENV}")
    if not total_text:
        raise ValueError(f"{GTEST_INDEX_ENV}={index_text} needs {GTEST_TOTAL_ENV}")
    total = _count(GTEST_TOTAL_ENV, total_text)
    if total < 1:
        raise ValueError(f"{GTEST_TOTAL_ENV}={total_text!r} is not at least 1")
    index = _count(GTEST_INDEX_ENV, index_text)
    if index >= total:
        raise ValueError(
            f"{GTEST_INDEX_ENV}={index_text!r} is not in 0..{total - 1} "
            f"({GTEST_TOTAL_ENV}={total_text})"
        )
    return f"{index}/{total}"


def _gtest_source(env: Mapping[str, str]) -> str:
    index = env.get(GTEST_INDEX_ENV, "").strip()
    total = env.get(GTEST_TOTAL_ENV, "").strip()
    return f"{GTEST_INDEX_ENV}={index}, {GTEST_TOTAL_ENV}={total}"


def unsharded_environment(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """A copy of ``env`` (default ``os.environ``) without the variables in
    ``CHILD_ENV_EXCLUDED``, for a child pytest that runs every test."""
    env = os.environ if env is None else env
    return {k: v for k, v in env.items() if k not in CHILD_ENV_EXCLUDED}


def resolve_shard(
    env: Mapping[str, str] | None = None, from_gtest: bool = False
) -> tuple[str, str] | None:
    """``(spec, source)`` of this run, or ``None`` when there is no shard.

    ``spec`` is ``i/n`` or ``i,j,.../n``. ``source`` names the variables
    that give the shard. A non-blank ``ROCKE_TEST_SHARD`` gives the shard,
    and then the gtest pair is not read. When ``from_gtest`` is true, the
    gtest pair gives the shard if ``ROCKE_TEST_SHARD`` is blank. The 1-based
    ``SHARD_INDEX`` and ``TOTAL_SHARDS`` are not read. Raises ``ValueError``
    for a malformed or out-of-range value of the source that gives the
    shard, and for a half-set gtest pair when that pair is the source."""
    env = os.environ if env is None else env
    text = env.get(SHARD_ENV, "").strip()
    if text:
        parse_shard(text)
        return text, f"{SHARD_ENV}={text}"
    if not from_gtest:
        return None
    spec = _gtest_shard(env)
    return None if spec is None else (spec, _gtest_source(env))


@dataclasses.dataclass(frozen=True)
class Shard:
    """One run's shard: ``spec`` (``i/n`` or ``i,j,.../n``) and the variables
    that gave it (``source``, for :func:`header_line`)."""

    spec: str
    source: str = ""
    picks: frozenset[int] = dataclasses.field(init=False)
    total: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        picks, total = parse_shard(self.spec)
        object.__setattr__(self, "picks", picks)
        object.__setattr__(self, "total", total)

    def owns(self, key: str) -> bool:
        """True when ``key``, a key without a declared cost, belongs to this
        shard. For keys that may carry a cost use :meth:`keeps`."""
        return place_keys([key], self.total)[key] in self.picks

    def keeps(
        self, keys: Iterable[str], costs: Mapping[str, float] | None = None
    ) -> set[str]:
        """The keys of ``keys`` that this shard runs, by :func:`place_keys`."""
        placement = place_keys(keys, self.total, costs)
        return {key for key, shard in placement.items() if shard in self.picks}

    def split_line(
        self, keys: Iterable[str], costs: Mapping[str, float] | None = None
    ) -> str:
        """How ``keys`` split over every shard of this run (by
        :func:`place_keys`), and how many this one keeps; the same split on
        every shard of one run."""
        keys = list(keys)
        placement = place_keys(keys, self.total, costs)
        counts = [0] * self.total
        for key in keys:  # a key repeated in ``keys`` counts each time
            counts[placement[key]] += 1
        kept = sum(counts[pick] for pick in self.picks)
        split = "/".join(str(count) for count in counts)
        return (
            f"rocke-ctest-shard {self.spec}: split {split} of {sum(counts)}; "
            f"kept {kept}"
        )


def current_shard(
    env: Mapping[str, str] | None = None, from_gtest: bool = False
) -> Shard | None:
    """This run's :class:`Shard` by :func:`resolve_shard`'s rule, or ``None``."""
    resolved = resolve_shard(env, from_gtest)
    return None if resolved is None else Shard(*resolved)


def header_line(shard: Shard | None) -> str:
    """The line naming a run's shard and its source, or that it has none."""
    if shard is None:
        return "rocke-ctest-shard: off"
    return f"rocke-ctest-shard: {shard.spec} ({shard.source})"
