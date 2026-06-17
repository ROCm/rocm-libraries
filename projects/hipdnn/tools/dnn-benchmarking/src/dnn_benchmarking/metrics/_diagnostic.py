# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Single-fire warning helper for metric collection.

Metric probes can fail silently in many places (psutil missing, amdsmi
init error, /proc read denied). To avoid spamming N warnings per suite
when the same dependency is missing, ``warn_once`` deduplicates by
``(source, reason)`` for the lifetime of the process.
"""

import sys
from typing import Set, Tuple

_seen: Set[Tuple[str, str]] = set()


def warn_once(source: str, reason: str) -> None:
    """Emit a warning to stderr the first time a (source, reason) is seen.

    Args:
        source: Short identifier of the metric source (e.g. "amdsmi",
            "psutil", "machine_info").
        reason: One-line explanation of the failure.
    """
    key = (source, reason)
    if key in _seen:
        return
    _seen.add(key)
    print(f"[metrics:{source}] {reason}", file=sys.stderr)


def reset() -> None:
    """Clear the seen set. Intended for tests."""
    _seen.clear()
