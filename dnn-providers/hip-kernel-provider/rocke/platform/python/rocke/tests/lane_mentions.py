# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Where Python sources name a runner lane or write a lane declaration by hand.

Which runner lane a GPU case waits for is a fact of :mod:`rocke.tests.lanes`
alone: a test states the arches it needs, and declares a lane only through
``skip_until`` or ``skip_until_reason``. ``platform/tests/tools/test_pending_lanes.py``
holds the tree to that with :func:`lane_mentions`. Parsing every source of the
tree is the cost of that check, so the files are parsed in worker processes,
one per CPU this process may use.
"""

from __future__ import annotations

import ast
import enum
import functools
import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import lanes
from .host_cpus import available_cpus

__all__ = [
    "LaneMention",
    "LaneMentionKind",
    "lane_mentions",
    "lane_mentions_in_source",
]

# The names rocke.tests.lanes binds to runner lanes (lanes that stand for arches).
_RUNNER_LANE_NAMES = frozenset(
    name
    for name, value in vars(lanes).items()
    if isinstance(value, lanes.Lane) and value.arches
)

# Files each worker parses per task: large enough that sending tasks costs
# little next to parsing them.
_FILES_PER_TASK = 32


class LaneMentionKind(enum.Enum):
    """How a source mentions a lane."""

    RUNNER_LANE_NAME = "runner lane name"
    HAND_WRITTEN_DECLARATION = "hand-written declaration"


@dataclass(frozen=True)
class LaneMention:
    """One place a source mentions a lane: ``text`` is the runner lane's name
    or the declaring string."""

    path: str
    line: int
    kind: LaneMentionKind
    text: str


def _declares_a_lane(text: str) -> bool:
    """Whether ``text`` reads as a lane declaration, a known lane or not."""
    try:
        return bool(lanes.lanes_waited_for(text))
    except ValueError:
        return True


def lane_mentions_in_source(source: str, path: str) -> list[LaneMention]:
    """Each runner lane name and hand-written lane declaration of Python
    ``source``, labelled ``path``, in :func:`ast.walk` order. Raises
    ``SyntaxError`` naming ``path`` when ``source`` does not parse."""
    mentions = []
    for node in ast.walk(ast.parse(source, filename=path)):
        if isinstance(node, ast.Name) and node.id in _RUNNER_LANE_NAMES:
            mentions.append(
                LaneMention(
                    path, node.lineno, LaneMentionKind.RUNNER_LANE_NAME, node.id
                )
            )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _declares_a_lane(node.value)
        ):
            mentions.append(
                LaneMention(
                    path,
                    node.lineno,
                    LaneMentionKind.HAND_WRITTEN_DECLARATION,
                    node.value,
                )
            )
    return mentions


def _mentions_in_file(root: Path, path: str) -> list[LaneMention]:
    return lane_mentions_in_source((root / path).read_text(encoding="utf-8"), path)


def lane_mentions(root: Path, paths: Sequence[str]) -> list[LaneMention]:
    """The lane mentions of each Python file ``paths`` names, relative to
    ``root``, in the order of ``paths``. A file that does not parse raises
    its ``SyntaxError``."""
    scan = functools.partial(_mentions_in_file, root)
    workers = min(available_cpus(), len(paths))
    if workers <= 1:
        found = [scan(path) for path in paths]
    else:
        # spawn: the same start method on every platform, and no fork of a
        # process that may hold threads.
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            found = list(pool.map(scan, paths, chunksize=_FILES_PER_TASK))
    return [mention for mentions in found for mention in mentions]
