# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Where Python sources name the runner lane, write a lane reason by hand, or
need hardware no pull-request runner has.

Which lane a skipped case waits for is a fact of :mod:`rocke.tests.lanes`
alone: a test states the arches it needs, and says what CI lacks only through
the lane helpers. ``platform/tests/tools/test_pending_lanes.py`` holds the
tree to that with :func:`lane_mentions`, and gives each case that needs runner
hardware a record. Parsing every source of the tree is the cost of those
checks, so the files are parsed once, in worker processes, one per CPU this
process may use.
"""

from __future__ import annotations

import ast
import enum
import functools
import inspect
import multiprocessing
import re
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import gpu, lanes
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

# What a skip reason says when it claims CI lacks something: that the case
# waits for something, or a CI runner or CI itself.
_LANE_WORDS = re.compile(r"(?i:\bwait(?:s|ing)?\s+(?:for|on)\b|\brunners?\b)|\bCI\b")
# The pytest and unittest calls that skip, with the index of their first
# reason argument (``skipif(condition, reason)``); a keyword ``reason`` or
# ``msg`` is a reason too.
_SKIP_CALLS = {
    "skip": 0,
    "skipTest": 0,
    "skipif": 1,
    "skipIf": 1,
    "skipUnless": 1,
    "importorskip": 2,
}
# The helpers a test passes the arches its case needs, which declare the runner
# hardware when no pull-request runner has any of them, with the position of
# their ``arches`` parameter.
_ARCH_NEED_HELPERS = {
    helper.__name__: list(inspect.signature(helper).parameters).index("arches")
    for helper in (
        lanes.skip_needing_arch,
        lanes.arch_skip_reason,
        gpu.current_device_arch_in,
        gpu.supported_device_arch,
        gpu.run_on_device,
    )
}

# Files each worker parses per task: large enough that sending tasks costs
# little next to parsing them.
_FILES_PER_TASK = 32


class LaneMentionKind(enum.Enum):
    """How a source mentions a lane."""

    RUNNER_LANE_NAME = "runner lane name"
    HAND_WRITTEN_DECLARATION = "hand-written declaration"
    HAND_WRITTEN_SKIP_REASON = "hand-written skip reason"
    RUNNER_HARDWARE_NEED = "runner hardware need"


@dataclass(frozen=True)
class LaneMention:
    """One place a source mentions a lane. ``text`` is the runner lane's name,
    the declaring string, the skip's written reason, or, for a runner hardware
    need, the arch helper's name; ``arches`` is that need's arches, and empty
    for every other kind."""

    path: str
    line: int
    kind: LaneMentionKind
    text: str
    arches: tuple[str, ...] = ()


def _declares_a_lane(text: str) -> bool:
    """Whether ``text`` reads as a lane declaration, a known lane or not."""
    try:
        return bool(lanes.lanes_waited_for(text))
    except ValueError:
        return True


def _called_name(call: ast.Call) -> str | None:
    return getattr(call.func, "attr", getattr(call.func, "id", None))


def _written_text(node: ast.AST) -> str:
    """The text written in ``node`` itself: its string constants, f-string
    parts included, but none a call returns (a lane helper's reason)."""
    if isinstance(node, ast.Call):
        return ""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else ""
    return " ".join(_written_text(child) for child in ast.iter_child_nodes(node))


def _hand_written_skip_reason(call: ast.Call) -> str | None:
    """The written reason of a skip ``call`` that says what CI lacks
    (:data:`_LANE_WORDS`) instead of naming a lane through
    ``rocke.tests.lanes``, or reads as a lane declaration; None for any other
    call. A reason held in a variable or a constant is not read."""
    name = _called_name(call)
    if name not in _SKIP_CALLS:
        return None
    reasons = call.args[_SKIP_CALLS[name] :] + [
        keyword.value for keyword in call.keywords if keyword.arg in ("reason", "msg")
    ]
    text = " ".join(_written_text(reason) for reason in reasons)
    return text if _LANE_WORDS.search(text) or _declares_a_lane(text) else None


def _literal_strings(node: ast.AST | None) -> tuple[str, ...] | None:
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)) and all(
        isinstance(element, ast.Constant) and isinstance(element.value, str)
        for element in node.elts
    ):
        return tuple(element.value for element in node.elts)
    return None


def _module_arches(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    """Each module-level name of ``tree`` bound to a literal of strings."""
    return {
        target.id: arches
        for statement in tree.body
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
        and (arches := _literal_strings(statement.value)) is not None
        for target in (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if isinstance(target, ast.Name)
    }


def _runner_hardware_need(
    call: ast.Call, module_arches: dict[str, tuple[str, ...]]
) -> tuple[str, ...] | None:
    """The arches of an arch helper ``call``, a literal or a module-level name
    bound to one, when no pull-request runner has any of them; None for any
    other call. Arches computed at run time are not seen; the limbo audit sees
    their cases."""
    name = _called_name(call)
    if name not in _ARCH_NEED_HELPERS:
        return None
    index = _ARCH_NEED_HELPERS[name]
    argument = (
        call.args[index]
        if len(call.args) > index
        else next((kw.value for kw in call.keywords if kw.arg == "arches"), None)
    )
    arches = _literal_strings(argument)
    if arches is None and isinstance(argument, ast.Name):
        arches = module_arches.get(argument.id)
    return arches if arches and lanes.runner_lanes_for(arches) else None


def lane_mentions_in_source(source: str, path: str) -> list[LaneMention]:
    """Each lane mention of Python ``source``, labelled ``path``, in
    :func:`ast.walk` order. Raises ``SyntaxError`` naming ``path`` when
    ``source`` does not parse."""
    tree = ast.parse(source, filename=path)
    module_arches = _module_arches(tree)
    mentions = []
    for node in ast.walk(tree):
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
        elif isinstance(node, ast.Call):
            reason = _hand_written_skip_reason(node)
            if reason is not None:
                mentions.append(
                    LaneMention(
                        path,
                        node.lineno,
                        LaneMentionKind.HAND_WRITTEN_SKIP_REASON,
                        reason,
                    )
                )
            arches = _runner_hardware_need(node, module_arches)
            if arches is not None:
                mentions.append(
                    LaneMention(
                        path,
                        node.lineno,
                        LaneMentionKind.RUNNER_HARDWARE_NEED,
                        _called_name(node),
                        arches,
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
