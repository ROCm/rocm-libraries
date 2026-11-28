# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Which delivered deploy settings the monitor runs: the applied record, and `rk applied`.

The applied record names the delivered layer that passed the configuration's load (runtime_config.load) and was
recorded (runtime_config.record_applied). Two commands write it: rk start, at each start of the monitor
(control.start, which the container's start runs), and `rk applied --load`, for a delivery that needs no restart, so
that the record holds the new layer as soon as the command returns, well within the two minutes apply waits for it
(apply step 7).

- `record(home=None, move=False) -> Recorded`: the configuration of HOME, the monitor's home (config.home() by
  default), loaded now, its layer recorded as applied, and the lines that say so. With MOVE, as rk start runs it, the
  work volume's one-time move is made between the two, when one is due (one_time_move.at_start), its lines after the
  load's. ConfigError when the configuration cannot run (runtime_config.LayerError when its layer cannot, or is
  missing where this monitor has run one); nothing is moved or recorded then.
- `Recorded(resolved, lines, holds)`: RESOLVED, the configuration loaded; LINES, what is said of its layer: loaded,
  and recorded as applied now or applied at an earlier time, or loaded with a record that cannot be written or read
  back, why and the fix (runtime_config.RecordError); none with no layer; HOLDS, whether the record holds that layer
  now.
- `cannot_run(exc, lead) -> list[str]`: why the configuration cannot run, EXC from `record` or any other load of it,
  in plain words after LEAD, then the fix the error names (config.ConfigError.fix), as the host's command runs it
  (doctor.wrapper_name).
- `refuse(exc, out=print) -> int`: `cannot_run`'s lines led by REFUSED, through OUT; exit 4: what
  `rk applied --load` and every rk command that cannot load the configuration say.
- `at_start(home=None) -> list[str]`: `record` as rk start runs it, the one-time move with it: its lines, or
  `cannot_run`'s led by "not loaded". `rk applied --load` never moves.
- `answer(home=None) -> dict`: what the host's deploy side asks the running monitor (state_facts.Monitor):
  {"applied": the applied record or None, "paused": whether a pause you set holds (rk pause), "standby": whether the
  monitor is in standby (features.standby)}. runtime_config.RecordError when the record cannot be read.
- `words(rec) -> str`: the applied record REC in a line.
- `main(load=False, as_json=False, home=None, out=print) -> int`: `rk applied [--load] [--json]`: the applied record in
  words, or `answer` as JSON; exit 0 with a record, 1 without. --load first loads the configuration and records its
  layer (`record`), and says so in place of the record's words: exit 0 when the record holds that layer, 1 when it
  does not (no layer, or a record that cannot be written or read back). REFUSED, exit 4: `cannot_run`'s lines when
  the configuration cannot run, or with --json `{"refused": why, "fix": the fix in its JSON form}`
  (checks.fix_to_json; null for none), which rkmon apply keeps with the change it puts back; without --load, when the
  record cannot be read, why and its fix.
"""

import json
import pathlib
from typing import NamedTuple

from . import checks, config, features, one_time_move, runtime_config
from .store import Store

NO_LAYER = (
    "loaded: your confirmed settings are not here: the configuration is config.json's"
)


class Recorded(NamedTuple):
    """What `record` did: see the module's docstring."""

    resolved: runtime_config.Resolved
    lines: list
    holds: bool


def _home(home) -> pathlib.Path:
    return pathlib.Path(home) if home is not None else config.home()


def _fix(fix) -> str:
    """The line of FIX, a check row's fix."""
    return f"{checks.FIX_LABEL}{fix}"


def record(home=None, move: bool = False) -> Recorded:
    """The configuration loaded now, its layer recorded as applied: see the module's docstring."""
    home = _home(home)
    resolved = runtime_config.load(home)
    layer = resolved.layer
    if layer is None:
        return Recorded(resolved, [], False)
    moved = one_time_move.at_start(home, resolved) if move else []
    what = (
        f"loaded: the settings you confirmed at {layer.get('confirmed_at')} (package "
        f"{layer.get('package_version')})"
    )
    try:
        if runtime_config.record_applied(home, resolved):
            return Recorded(resolved, [f"{what}, recorded as applied", *moved], True)
        rec = runtime_config.applied(home) or {}
    except runtime_config.RecordError as exc:
        return Recorded(resolved, [f"{what}; {exc}", _fix(exc.fix()), *moved], False)
    return Recorded(
        resolved, [f"{what}, applied at {rec.get('applied_at')}", *moved], True
    )


def cannot_run(exc: config.ConfigError, lead: str) -> list:
    """Why the configuration cannot run, in lines: see the module's docstring."""
    # a call-time import: doctor is a large module, read only when the configuration cannot run
    from .doctor import wrapper_name

    fix = exc.fix(wrapper_name())
    return [f"{lead}: {exc}"] + ([_fix(fix)] if fix is not None else [])


def refuse(exc: config.ConfigError, out=print) -> int:
    """REFUSED, exit 4: see the module's docstring."""
    for line in cannot_run(exc, "REFUSED"):
        out(line)
    return 4


def at_start(home=None) -> list:
    """`record` as rk start runs it: see the module's docstring."""
    try:
        return record(home, move=True).lines
    except config.ConfigError as exc:
        return cannot_run(exc, "not loaded")


def answer(home=None) -> dict:
    """What the host's deploy side asks the running monitor: see the module's docstring."""
    home = _home(home)
    folder = config.state_dir(home)
    try:
        st = Store(folder).load() if folder.is_dir() else {}
    except ValueError:  # a state this code cannot read holds no pause it knows of
        st = {}
    return {
        "applied": runtime_config.applied(home),
        "paused": bool(st.get("paused")),
        "standby": features.standby(home),
    }


def words(rec) -> str:
    """The applied record REC in a line."""
    if not rec:
        return "applied: none: the monitor has run none of your confirmed settings here"
    return (
        f"applied: your confirmed settings {rec.get('fingerprint')} (package {rec.get('package_version')}), at "
        f"{rec.get('applied_at')}"
    )


def main(load: bool = False, as_json: bool = False, home=None, out=print) -> int:
    """`rk applied [--load] [--json]`: see the module's docstring."""
    holds = None
    if load:
        try:
            done = record(home)
        except config.ConfigError as exc:
            if not as_json:
                return refuse(exc, out)
            from .doctor import wrapper_name  # a call-time import, as cannot_run's

            fix = checks.fix_to_json(exc.fix(wrapper_name()))
            out(json.dumps({"refused": str(exc), "fix": fix}, sort_keys=True))
            return 4
        holds = done.holds
        if not as_json:
            for line in done.lines or [NO_LAYER]:
                out(line)
            return 0 if holds else 1
    try:
        said = answer(home)
    except runtime_config.RecordError as exc:
        out(f"REFUSED: {exc}")
        out(_fix(exc.fix()))
        return 4
    out(json.dumps(said, sort_keys=True) if as_json else words(said["applied"]))
    if holds is not None:
        return 0 if holds else 1
    return 0 if said["applied"] else 1
