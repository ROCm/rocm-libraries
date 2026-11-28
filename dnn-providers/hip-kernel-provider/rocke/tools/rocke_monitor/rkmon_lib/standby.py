# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk standby-off`: the end of standby, the whole monitor's (features.standby), so that the monitor sends
again, as your settings allow. `rkmon standby off` runs it in the running container, in the deployment's state 14
alone, once the monitor you moved from has stopped; the Claude session never does (rkmon_lib/outward.py). It
reads nothing of any other deployment.

- `off(cfg, home=None, out=None) -> int`: while the monitor is in standby, the one-time move from an older install
  ended when one is open (one_time_move.end), its lines printed as it returns them (what it renamed, removed
  and kept, and the time it set), then `standby.on` cleared in the internal state (state_store.set), then, last, the
  mark of an adoption removed (adopt.unmark), then RELEASED (states.RELEASED), exit 0. NOTHING TO DO
  (states.NOT_IN_STANDBY) while the monitor is not in standby, nothing changed. While an adoption has not finished
  (adopt.unfinished: rkmon adopt stopped part-way), REFUSED, exit 4, nothing changed, its fix adopt again: rk
  standby-off checks it itself, whatever runs it. Standby stays on when the move cannot end
  (one_time_move.MoveError), `standby.on` cannot be cleared (state_store.StateError, or an OSError of its lock or its
  write: state_store.unwritable) or the mark cannot be removed: REFUSED, exit 4, when nothing changed; ERROR, exit 1,
  once the end has done some of its work, which its lines, printed first, say, or once `standby.on` is cleared. Each
  with the fix the error names, as the host's command runs it (config.ConfigError.fix). CFG: the configuration the
  monitor runs, by which the end finds the older copies of the secrets; HOME: the monitor's home (config.home() by
  default); OUT: the stream the lines go to (stdout by default).
"""

from __future__ import annotations

import sys

from . import adopt, config, features, one_time_move, outcome, state_store, states


def off(cfg: dict, home=None, out=None) -> int:
    """`rk standby-off`: see the module's docstring."""
    out = sys.stdout if out is None else out
    home = config.home() if home is None else home
    if not features.standby(home):
        return outcome.finish("NOTHING TO DO", states.NOT_IN_STANDBY, out=out)
    unfinished = adopt.unfinished(home)
    if unfinished is not None:
        return _stays_on(unfinished, False, out)
    try:
        lines = one_time_move.end(home, cfg)
    except one_time_move.MoveError as exc:
        _say(exc.done, out)
        return _stays_on(exc, bool(exc.done), out)
    _say(lines, out)
    try:
        state_store.set(home, "standby.on", False)
    except state_store.StateError as exc:
        return _stays_on(exc, bool(lines), out)
    except OSError as exc:
        return _stays_on(state_store.unwritable(home, exc), bool(lines), out)
    try:
        adopt.unmark(home)
    except config.ConfigError as exc:
        return _stays_on(exc, True, out)
    return outcome.finish("RELEASED", states.RELEASED, out=out)


def _say(lines, out) -> None:
    """LINES, what the end did, a line each, as it returns them."""
    for line in lines:
        out.write(f"{line}\n")


def _stays_on(exc: config.ConfigError, part_way: bool, out) -> int:
    """Standby stays on for EXC: REFUSED when nothing changed, ERROR when the end did some of its work (PART_WAY),
    each with EXC's fix as the host's command runs it."""
    # a call-time import: doctor is a large module, read only on a failure
    from .doctor import wrapper_name

    fix = exc.fix(wrapper_name())
    if part_way:
        return outcome.finish("ERROR", f"{exc}; standby stays on", fix=fix, out=out)
    text = f"{exc}; standby stays on; nothing changed"
    return outcome.finish("REFUSED", text, fix=fix, out=out)
