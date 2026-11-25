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

A runner that is not pytest takes the flag with :func:`add_shard_argument`,
gets its shard with :func:`shard_from_arguments`, and prints
:func:`header_line` and :meth:`Shard.split_line`, the lines the plugin prints,
so one log parser reads both.

Given ``LIST_KEYS_FLAG`` and a path, a run lists instead of running: it writes
the cases it would split, each with its shard key, as a
:class:`ShardKeyListing`, so a check can place them by this module's rule
without running them (``tools/check_entry_times.py``).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

__all__ = [
    "ACTIVATION_FLAG",
    "CHILD_ENV_EXCLUDED",
    "GTEST_INDEX_ENV",
    "GTEST_TOTAL_ENV",
    "LIST_KEYS_FLAG",
    "SHARD_ENV",
    "WORKER_ENV",
    "Shard",
    "ShardKeyListing",
    "add_shard_argument",
    "adopt_worker",
    "check_cost",
    "current_shard",
    "current_worker",
    "header_line",
    "list_keys_path",
    "parse_shard",
    "place_keys",
    "resolve_shard",
    "shard_from_arguments",
    "shard_of",
    "unsharded_environment",
]

SHARD_ENV = "ROCKE_TEST_SHARD"
GTEST_INDEX_ENV = "GTEST_SHARD_INDEX"
GTEST_TOTAL_ENV = "GTEST_TOTAL_SHARDS"
# ``"<index>/<count>"`` (0-based) in a worker process of
# ``rocke.tests.ctest_workers``: the share of the run's tests it runs.
# ``"<count>/<count>"`` in the process that runs the run's run-alone tests
# after its ``count`` workers have exited.
WORKER_ENV = "ROCKE_TEST_WORKER"
# The option that makes a run take its shard from the gtest pair. A CTest entry
# whose command carries it runs only its CI shard's share.
ACTIVATION_FLAG = "--rocke-shard-from-gtest-env"
# The option, followed by a path, that makes a run write its ShardKeyListing
# there instead of running.
LIST_KEYS_FLAG = "--rocke-list-shard-keys"
# A child pytest that must run unsharded, with only the plugins that it names,
# must not inherit these variables. SHARD_INDEX and TOTAL_SHARDS are
# TheRock's 1-based pair.
CHILD_ENV_EXCLUDED = (
    SHARD_ENV,
    WORKER_ENV,
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


@dataclasses.dataclass(frozen=True)
class ShardKeyListing:
    """The cases a run splits over the CI shards, without running them.

    ``key_of_case`` maps each case (a test's node id, a golden case id) to its
    shard key, in the run's order. ``costs`` holds the declared cost of each
    key that has one, which :func:`place_keys` places by. A run in worker
    processes also lists each key's cost over its cases that do not run alone
    (``worker_costs``) and the cases that run alone after the workers
    (``run_alone_cases``).
    """

    key_of_case: Mapping[str, str]
    costs: Mapping[str, float] = dataclasses.field(default_factory=dict)
    worker_costs: Mapping[str, float] = dataclasses.field(default_factory=dict)
    run_alone_cases: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        keys = set(self.key_of_case.values())
        for field in ("costs", "worker_costs"):
            costs = getattr(self, field)
            unknown = sorted(set(costs) - keys)
            if unknown:
                raise ValueError(
                    f"ShardKeyListing failed - {field} keys={unknown}, expected "
                    "only keys of key_of_case"
                )
            for key, cost in costs.items():
                check_cost(key, cost)
        unknown = sorted(self.run_alone_cases - set(self.key_of_case))
        if unknown:
            raise ValueError(
                f"ShardKeyListing failed - run_alone_cases={unknown}, expected only "
                "cases of key_of_case"
            )

    def write(self, path: Path) -> None:
        """Write this listing to ``path`` as JSON."""
        listing = {
            "key_of_case": dict(self.key_of_case),
            "costs": dict(self.costs),
            "worker_costs": dict(self.worker_costs),
            "run_alone_cases": sorted(self.run_alone_cases),
        }
        Path(path).write_text(json.dumps(listing), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> ShardKeyListing:
        """The listing :meth:`write` wrote to ``path``."""
        listing = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            key_of_case=listing["key_of_case"],
            costs=listing["costs"],
            worker_costs=listing["worker_costs"],
            run_alone_cases=frozenset(listing["run_alone_cases"]),
        )


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


def add_shard_argument(
    parser: argparse.ArgumentParser, help: str, list_help: str
) -> None:
    """Give a runner's ``parser`` the :data:`ACTIVATION_FLAG` option and the
    :data:`LIST_KEYS_FLAG` option, which every split runner takes
    (``tools/check_entry_times.py`` lists each split entry with it).

    :func:`shard_from_arguments` and :func:`list_keys_path` read the parsed
    values. ``help`` says what the runner's share is; ``list_help`` what the
    runner lists, each case with its shard key, instead of running.
    """
    parser.add_argument(
        ACTIVATION_FLAG, dest="shard_from_gtest_env", action="store_true", help=help
    )
    parser.add_argument(
        LIST_KEYS_FLAG,
        dest="list_shard_keys",
        type=Path,
        default=None,
        help=f"with {ACTIVATION_FLAG}: {list_help}",
    )


def shard_from_arguments(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> Shard | None:
    """The run's :class:`Shard` when ``arguments`` carry :data:`ACTIVATION_FLAG`
    (from :func:`add_shard_argument`) and a shard is set, else ``None``.

    A malformed or out-of-range shard, or :data:`LIST_KEYS_FLAG` without
    :data:`ACTIVATION_FLAG`, is a usage error, reported through ``parser``
    (exit status 2).
    """
    if arguments.list_shard_keys is not None and not arguments.shard_from_gtest_env:
        parser.error(f"{LIST_KEYS_FLAG} needs {ACTIVATION_FLAG}")
    if not arguments.shard_from_gtest_env:
        return None
    try:
        return current_shard(from_gtest=True)
    except ValueError as error:
        parser.error(f"rocke-ctest-shard: {error}")


def list_keys_path(arguments: argparse.Namespace) -> Path | None:
    """Where the run writes its :class:`ShardKeyListing` instead of running,
    from the :data:`LIST_KEYS_FLAG` of :func:`add_shard_argument`; ``None``
    when the run runs."""
    return arguments.list_shard_keys


# This process's worker identity once adopt_worker() has taken it from the
# environment: (adopted, (index, count) or None).
_adopted_worker: tuple[bool, tuple[int, int] | None] = (False, None)


def adopt_worker() -> tuple[int, int] | None:
    """This process's :func:`current_worker`, taken out of the environment: the
    variable is removed, so a pytest process a test starts is not taken for a
    worker, and :func:`current_worker` answers from what was taken. Raises
    ``ValueError`` as :func:`current_worker` does."""
    global _adopted_worker
    if not _adopted_worker[0]:
        worker = current_worker()
        os.environ.pop(WORKER_ENV, None)
        _adopted_worker = (True, worker)
    return _adopted_worker[1]


def current_worker(env: Mapping[str, str] | None = None) -> tuple[int, int] | None:
    """``(index, count)`` from :data:`WORKER_ENV` of ``env`` (default: this
    process's, as :func:`adopt_worker` took it, else ``os.environ``) when the
    process is a worker of ``rocke.tests.ctest_workers`` (``index < count``) or
    the process that runs the run-alone tests after the ``count`` workers
    (``index == count``), else ``None``. Raises ``ValueError`` for a value
    other than ``<index>/<count>`` with ``0 <= index <= count`` and
    ``count >= 1``."""
    if env is None and _adopted_worker[0]:
        return _adopted_worker[1]
    env = os.environ if env is None else env
    text = env.get(WORKER_ENV, "").strip()
    if not text:
        return None
    index, _, count = text.partition("/")
    digits = all(part.isascii() and part.isdigit() for part in (index, count))
    if not (digits and 1 <= int(count) and int(index) <= int(count)):
        raise ValueError(
            f"current_worker failed - {WORKER_ENV}={text!r}, expected "
            "<index>/<count> with 0 <= index <= count and count >= 1"
        )
    return int(index), int(count)


def header_line(shard: Shard | None) -> str:
    """The line naming a run's shard and its source, or that it has none."""
    if shard is None:
        return "rocke-ctest-shard: off"
    return f"rocke-ctest-shard: {shard.spec} ({shard.source})"
