# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Decline taxonomy: the reason codes recorded for every fallback.

The C++ runtime (``csrc/decline.h``) uses the same stage names and the same
``STAGE[:detail]`` string form, so reasons coming from ``_C`` and from the
Python routing layers aggregate under one vocabulary. Backend error messages
are kept next to the reason for the report but are never parsed.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass

__all__ = ["Decline", "DeclineError", "Stage", "parse_op_error", "parse_reason"]


class Stage(str, enum.Enum):
    NO_BUILD = "NO_BUILD"  # build_operation_graph (or graph validation) failed
    NO_ENGINE = "NO_ENGINE"  # the ranked engine list is empty
    CHECK_SUPPORT = "CHECK_SUPPORT"
    BUILD_PLANS = "BUILD_PLANS"
    UNSUPPORTED_FEATURE = "UNSUPPORTED_FEATURE"  # plugin-side pre-filter
    DISABLED = "DISABLED"  # excluded by configuration
    CAPTURE_INVARIANT = "CAPTURE_INVARIANT"  # forced to the capture-time variant
    EXECUTE = "EXECUTE"  # a built plan failed to launch


# Stages that require a detail (``UNSUPPORTED_FEATURE:softcap``). The others
# may carry one, but do not need to.
_DETAIL_REQUIRED = {Stage.UNSUPPORTED_FEATURE}
_DETAIL_RE = re.compile(r"^[a-z0-9_.]+$")


@dataclass(frozen=True)
class Decline:
    stage: Stage
    detail: str = ""
    message: str = ""

    def __post_init__(self):
        if self.stage in _DETAIL_REQUIRED and not self.detail:
            raise ValueError(f"{self.stage.value} needs a feature name")
        if self.detail and not _DETAIL_RE.match(self.detail):
            raise ValueError(f"decline detail {self.detail!r} must match {_DETAIL_RE.pattern}")

    @property
    def reason(self) -> str:
        return f"{self.stage.value}:{self.detail}" if self.detail else self.stage.value

    @classmethod
    def feature(cls, name: str, message: str = "") -> "Decline":
        return cls(Stage.UNSUPPORTED_FEATURE, name, message)


def parse_reason(reason: str) -> Decline:
    """Inverse of :attr:`Decline.reason`. Raises ``ValueError`` for strings
    outside the taxonomy."""
    stage_name, _, detail = reason.partition(":")
    try:
        stage = Stage(stage_name)
    except ValueError:
        raise ValueError(f"{reason!r} is not a decline reason") from None
    return Decline(stage, detail)


class DeclineError(RuntimeError):
    """Raised by strict mode, and by Python wrappers when an op declines."""

    def __init__(self, op: str, decline: Decline):
        self.op = op
        self.decline = decline
        msg = f"hipdnn_vllm: {op} declined: {decline.reason}"
        if decline.message:
            msg += f" ({decline.message})"
        super().__init__(msg)


# Must match csrc/graphs/norm.cpp: "hipdnn_vllm decline [<op>]: <reason> :: <message>".
_OP_ERROR_RE = re.compile(
    r"hipdnn_vllm decline \[(?P<op>[^\]]+)\]: (?P<reason>\S+) :: (?P<msg>.*)", re.S
)


def parse_op_error(exc: BaseException) -> tuple[str, Decline] | None:
    """Decode the RuntimeError a ``torch.ops.hipdnn_vllm`` op raises on decline.
    Returns ``(op, Decline)``, or ``None`` if ``exc`` is not a decline."""
    m = _OP_ERROR_RE.search(str(exc))
    if m is None:
        return None
    d = parse_reason(m.group("reason"))
    return m.group("op"), Decline(d.stage, d.detail, m.group("msg").strip())
