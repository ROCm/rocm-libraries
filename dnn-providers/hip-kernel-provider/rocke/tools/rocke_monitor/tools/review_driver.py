#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The continuous review driver: it works through the review backlog without waiting for ticks.

`rk review-driver` advances the parallel reviews (the drafts that run at once) every 60 s, keeping up to
features.review_automation.parallel_reviews of them running at once (one worktree pair and one workspace per PR). It
exits, which wakes the orchestrator, as soon as:
  - a draft is ready for verification (REVIEW_READY; every one of that round is printed), or
  - nothing can be started now (backlog empty, the remaining heads still settling, the remaining follow-ups
    waiting for the gap after our last review of their PR: `rk review-advance` never starts one early, or the
    remaining in-depth reviews waiting for the operator's approval); the ticks resume, and a tick starts each once
    its time comes, or
  - only reviews too large for the autopilot are left (REVIEW_MANUAL): the orchestrator starts one by hand, or
  - the monitor is stopped (rk stop, rk kill), or
  - the last tick is over 25 minutes old (TICK_STALE): the tick loop may be down, so no review starts until a tick
    runs (`rk tick`); jobs already running keep running, or
  - two hours pass (a stuck job; the orchestrator checks `rk review-status`).
The drafts in flight keep running when it exits: they are detached jobs. One driver at a time: it holds
state/review_driver.lock, and registers its process so rk stop and rk kill find it.
The toolkit this script belongs to (tools/..) runs it, in this process's place; its rk finds the monitor's home
(`rk home`).

Usage: python3 -B tools/review_driver.py [ARGS]   (ARGS go to `rk review-driver`)
"""

import os
import sys


def main() -> int:
    rk = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/rk"
    try:
        os.execv(rk, [rk, "review-driver", *sys.argv[1:]])
    except OSError as exc:  # as the shell's `exec` that cannot run its command
        sys.stderr.write(f"{sys.argv[0]}: line 20: {rk}: {exc.strerror}\n")
        return 126 if isinstance(exc, PermissionError) else 127


if __name__ == "__main__":
    sys.exit(main())
