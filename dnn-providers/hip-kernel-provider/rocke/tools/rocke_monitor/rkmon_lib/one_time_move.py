# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The work volume's one-time move: the config.json an older package left in the monitor's home, split
once by the registry's owners when your confirmed settings are delivered, the monitor in standby until
`rkmon standby off` ends the move.

The move is made where the monitor loads the delivered layer at its start: rk start loads the
configuration, makes the move when one is due (`at_start`, through state_store.migrate, under the internal state's
lock), then records the layer as applied (rkmon_lib/applied.py). No other load makes it: with a layer delivered,
runtime_config.load never reads config.json, so no command's load moves anything.

- `pending(home) -> bool`: whether a move is due or open: your settings are delivered here (runtime_config.delivered:
  in the container, the layer's file in its settings folder), HOME is the work volume's monitor home
  (container_layout.MONITOR_HOME), and its record says a move is open, or there is none and a move is due (`due`);
  never once a move has ended. While one is pending, the monitor is in standby (features.standby), its internal state
  missing too. A home elsewhere, as a test's is, never has a move pending: no test moves, or is put in standby, unless
  it makes its own home the work volume's.
- `due(home) -> bool`: your settings are delivered here, HOME is the work volume's monitor home, no record of a move
  is there, HOME holds a config.json, and the work volume was an older install's when this package first read it
  (provision.older_install: decided once, at that first read, and kept in the internal state). So an older install
  moves once: a second start moves nothing, nor does a config.json that changes while the move is open, and a
  config.json written later on a work volume this package set up, or after the end, by a tool or a person, is never
  moved.
- `due_again(home, again) -> list`: the move due again, for `rkmon adopt`, which calls it each time it has copied an
  earlier monitor's config.json into HOME and the internal state is in place:
  under the internal state's lock, the answer that the work volume is an older install's (provision.OLDER_INSTALL)
  set, then the record of an earlier move removed, so that the next start makes the move. Its line; [] when HOME
  holds no config.json. MoveError, with its fix, when the internal state or the record cannot be written: the
  internal state's own, or the state folder made writable (`writable`), then AGAIN, the caller's words and command
  that run it again.
- `writable(folder, again) -> Steps`: the fix of FOLDER, which the monitor cannot write: made writable for it, then
  AGAIN, the words and command that run again what failed (the end's, due_again's and rkmon adopt's).
- `older_settings(home) -> Older(values, notes, problem, sha256)`: config.json as the configuration reads it
  (runtime_config.carried: its older keys at their names today) and NOTES, what renames.migrate says of it; else VALUES
  None and PROBLEM, why it cannot be read. SHA256: the sha256 of the bytes read, the ones the split splits; "" when
  none could be read. This part's one reading of config.json's settings, which goes with the rest of the code
  that reads config.json.
- `split(values, running, state) -> Split(carried, lines, full)`: config.json's VALUES split by the registry's owners,
  pure. RUNNING is the configuration the monitor runs, loaded from the layer (runtime_config.Resolved); STATE, what the
  internal state holds, under the configuration's names. Each key at the registry's grain (registry.leaves):
  - a state key is carried to the internal state (CARRIED: name -> value) where the state holds none and the value is
    not the key's default; where the state holds another value, the state's is kept, and a line says so;
  - a deploy key whose value differs from the one your deploy settings give is said: yours runs; one your deploy
    settings set none of (a model binding's keys, while no binding names it) is dropped, named;
  - a package key that config.json sets to a value of its own (repository_defaults.own) other than the one that runs
    (config.defaults.json's, or the repository defaults') is said, and not carried;
  - a key no feature reads (rkmon_lib/removed_keys.py: a removed key, or one whose value the package sets), a derived
    key, which each load computes, and a name no key of this package has are dropped, each named, by why.
  A state, deploy or package key config.json sets to nothing (null, "", an empty list or map) is never carried or
  said: nothing of it is lost. LINES: what the state, the deploy and the package keys' lines say, then what was
  dropped, by why. A value is shown as rk config show shows it (config.safe_value), as JSON, at most SHOWN_MOST
  characters; the line of a private key (registry.Key.klass) shows neither value, for private values stay out of
  the logs, and rk start's output goes to its log. FULL: the same lines, each value shown, which the record keeps.
- `recorded(home, running, older, split, refused, now) -> list`: the move state_store.migrate has just made, written
  in its record, and its report: that config.json was moved, or why nothing of it could be carried, and that the
  monitor is in standby until `rkmon standby off`; then what renames.migrate said; the state keys carried; each one
  of REFUSED (name -> why), which the internal state's schema refused, not carried (a private key's line without the
  why, which quotes the value); the split's lines; and UNTIL, set to NOW, the move's time. The lines it returns show
  no private key's value; the record keeps them with each value shown.
- `at_start(home, running, environ=None) -> list`: what rk start does of the move, in lines: the move made, when one
  is due and RUNNING was loaded from a delivered layer, and its report; else, once a move has ended, `retry`. A move
  that cannot be made is said, with why and its fix: it stays pending, the monitor in standby, and the next start
  makes it.

The record of the move: RECORD_FILE in the state folder of the monitor's home (config.state_dir), of the kind
RECORD_KIND at RECORD_SCHEMA_VERSION, written whole under the internal state's lock (state_store.write_file):
`moved_at`; the `fingerprint` and the `package_version` of the layer it was made with; `config_json_sha256`, the
sha256 of the bytes it split; `report`, its lines, each value shown, for the terminal; and, once the move has ended,
`ended_at`, `end`, the end's lines, and `kept`, each older copy it kept, [{entry, place}]. rk start prints the report
with no private key's value, and the container's start log (logs/start.log) keeps what it prints.
- `record(home) -> dict | None`: the record; None when there is none, it cannot be read or it is no record of the
  move: a move made again writes it anew.
- `is_open(rec) -> bool`: whether REC records a move that has not ended.
- `ended(home) -> bool`: whether HOME's record says its move has ended: from then on no older copy of a secret counts
  as its entry set (logins.older_copy), as the end has removed each one it could.

Until the end, the move only adds: config.json, the older copies of the secrets and the older package's code stay as
they are. The end, which `rkmon standby off` runs (`end(home, cfg, environ=None) -> list`), ends the open move:
config.json becomes MIGRATED, read-only (the name provision.OLDER_SETUP reads), and the older copies of the secrets
go, by the secrets registry's older names (rkmon_lib/secrets_registry.py, each entry's `older`, for CFG, the
configuration the monitor runs):
- an older copy the move copies (`copied`: the team chat's webhook, the web host's key, the claude login's older
  settings, a model binding's key, the write login from before the read/write split) goes once its entry is set in its
  own place (logins.found: the writer answers for its own); else it is kept, and its line says why, then the fix that
  sets the entry (secrets_registry.set_fix): a copy the root step refused (a shape it cannot carry, two older copies
  that differ) or did not make (a key no feature uses);
- an older copy of a secret the move never copies, the admin access token's, which a deployment makes anew, goes
  once its entry is set;
- an older copy of a login the move never copies goes: the older atlassian login (Atlassian replaces its refresh
  token at each refresh: `rkmon login atlassian` makes the writer's own) and the files the read login's token was kept
  in beside the login (the login is its one copy); its line says whether the login is set in its own place, then,
  while it is not, the fix that sets it (secrets_registry.set_fix: for the atlassian login, `rkmon login atlassian`);
- the files an older package kept a secret in that no entry claims (secrets_registry.OLDER_SECRET_FILES) go: the local
  worker's extra headers, the JIRA MCP server's headers, and the local worker's key while no binding names the first
  review pass's key, whose older copy it is otherwise.
The older package's code goes too (OLDER_CODE: its folder in the work volume, container_layout.WORK_DIR, with the
releases its update installed and the link to the one that ran), when HOME is the work volume's monitor home: until the
end, the way back to the older package runs it. No entry's own place is ever removed. Then the end sets UNTIL again,
to its own time, under the internal state's lock (state_store.set), which it holds. It returns its lines, which say
what it renamed, removed and kept, and UNTIL; [] when no move is open or due. Each MoveError carries its fix
(config.ConfigError.fix) and `done`, the lines of what the end did before it stopped, as it returns them (() when it
changed nothing):
- a move that is due and has no record (`due`: the start could not make it), nothing changed: the monitor stays in
  standby while the move is pending, so the end refuses rather than let standby off end a standby it cannot end. The
  fix: restart the monitor, whose start makes the move again, then `rkmon standby off`;
- the places cannot be found, as the configuration names no secrets folder (paths.PathsError), nothing changed and
  the move open: the fix, `rkmon doctor` and what it says put right, then `rkmon standby off`;
- the internal state's lock cannot be taken in its state folder, nothing changed and the move open: the state folder
  made writable, then `rkmon standby off`;
- config.json cannot be renamed, nothing changed and the move open: its folder made writable, then
  `rkmon standby off`;
- the internal state or the record cannot be written once the end's work is done, which `done` says: the state folder
  made writable, or the internal state's own fix, then `rkmon standby off` (the end, run again, ends it).
- `retry(home, cfg, environ=None) -> list`: each older copy the end kept, removed once the end's rule lets it go (its
  entry set since), in lines; the record keeps the ones still there. rk start runs it (`at_start`).
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import stat
from typing import NamedTuple

from . import config, registry, reviewer_name, runtime_config
from .checks import FIX_LABEL, Steps

RECORD_FILE = "one_time_move.json"
RECORD_KIND = "one_time_move"
RECORD_SCHEMA_VERSION = 1
MIGRATED = runtime_config.CONFIG_JSON + ".migrated"  # config.json, once the move ends
REASON = "moved from an older install"  # standby's reason while the move is open
# The internal state's time before which the reviews an older monitor posted, whose line 1 has no automated reviewer
# name, are this deployment's own: the move's time, then the end's. reviewer_name reads it.
UNTIL = reviewer_name.UNNAMED_ARE_OURS_UNTIL
SHOWN_MOST = 80  # the most characters of a value the report shows
# The older package's code, in the work volume beside the monitor's home: the releases its update installed and the
# link to the one that ran. The way back to the older package runs it, so the end alone removes it.
OLDER_CODE = "code"


class MoveError(config.ConfigError):
    """The end of the move cannot be made, or its record cannot be written: see the module's docstring."""

    done: tuple = ()  # the lines of what the end did before it stopped


class Older(NamedTuple):
    """config.json's settings as the move reads them: see the module's docstring."""

    values: dict | None
    notes: list
    problem: str
    sha256: str = ""


class Split(NamedTuple):
    """config.json split by owner: see the module's docstring."""

    carried: dict
    lines: list
    full: list


def config_file(home) -> pathlib.Path:
    """config.json in the monitor's home HOME."""
    return pathlib.Path(home) / runtime_config.CONFIG_JSON


def record_file(home) -> pathlib.Path:
    """The record of the move, in the state folder of the monitor's home HOME."""
    return config.state_dir(home) / RECORD_FILE


def _standby_off():
    """The command that ends standby and, with it, the move, as you type it (steps.command_line)."""
    from . import steps

    return steps.command_line("standby off")


def _locked(home):
    """The internal state's lock in HOME's state folder, held while the block runs (state_store.locked_in)."""
    from . import state_store

    return state_store.locked_in(config.state_dir(home))


# --- whether a move is due --------------------------------------------------------------------------------------------


def _here(home) -> bool:
    """Whether your settings are delivered here and HOME is the work volume's monitor home."""
    from . import container_layout

    if not runtime_config.delivered():
        return False
    return os.path.realpath(home) == os.path.realpath(container_layout.MONITOR_HOME)


def _older(home) -> bool:
    """Whether HOME holds a config.json and the work volume was an older install's (provision.older_install)."""
    from . import provision  # a call-time import: provision reads this module

    return os.path.lexists(config_file(home)) and provision.older_install(home)


def pending(home) -> bool:
    """Whether a move is due or open: see the module's docstring."""
    if not _here(home):
        return False
    rec = record(home)
    return is_open(rec) if rec is not None else _older(home)


def due(home) -> bool:
    """Whether a move is due: see the module's docstring."""
    return _here(home) and record(home) is None and _older(home)


def due_again(home, again: tuple) -> list:
    """The move due again, for rkmon adopt: see the module's docstring."""
    from . import provision, state_store
    from .doctor import wrapper_name

    path = config_file(home)
    if not os.path.lexists(path):
        return []
    lead = "the one-time move cannot be made due again"
    try:
        with _locked(home):
            state_store.set(home, provision.OLDER_INSTALL, True, held=True)
            record_file(home).unlink(missing_ok=True)
    except OSError as exc:
        folder = config.state_dir(home)
        why = exc.strerror or exc
        raise MoveError(f"{lead} in {folder}: {why}", writable(folder, again)) from None
    except state_store.StateError as exc:
        raise MoveError(f"{lead}: {exc}", exc.fix(wrapper_name())) from None
    off = _standby_off()
    return [
        f"the one-time move is due again: the next start splits {path} by owner, once, and the monitor stands by "
        f"until you run {off}"
    ]


def record(home) -> dict | None:
    """The record of the move: see the module's docstring."""
    try:
        rec = json.loads(record_file(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) and rec.get("kind") == RECORD_KIND else None


def is_open(rec) -> bool:
    """Whether REC records a move that has not ended."""
    return bool(rec) and not rec.get("ended_at")


def ended(home) -> bool:
    """Whether HOME's record says its move has ended: see the module's docstring."""
    rec = record(home)
    return bool(rec) and not is_open(rec)


# --- the split --------------------------------------------------------------------------------------------------------


def older_settings(home) -> Older:
    """config.json's settings as the move reads them: see the module's docstring."""
    path, notes = config_file(home), []
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return Older(None, notes, f"{path} is not there")
    except OSError as exc:
        return Older(None, notes, f"{path} cannot be read: {exc.strerror or exc}")
    sha256 = hashlib.sha256(data).hexdigest()
    try:
        values = runtime_config.carried(path, notes, data)
    except config.ConfigError as exc:
        return Older(None, notes, str(exc), sha256)
    return Older(values, notes, "", sha256)


def _shown(value) -> str:
    """VALUE as the report shows it: see the module's docstring; "none" for no value."""
    if value is registry.MISSING:
        return "none"
    text = json.dumps(config.safe_value(value), sort_keys=True, default=str)
    return text if len(text) <= SHOWN_MOST else text[: SHOWN_MOST - 3] + "..."


def _blank(value) -> bool:
    """Whether VALUE sets nothing: no value, null, "" or an empty list or map."""
    return value is registry.MISSING or value in registry.BLANK or value in ([], {})


# A key whose values stay out of the logs (registry.Key.klass): the report's lines show neither of them.
PRIVATE = "private"


def _not_carried(key, value, holder: str, held, verb: str, where: str = "") -> tuple:
    """The line of the key KEY (a registry.Key) whose VALUE in config.json is not carried, and of HELD, the value that
    holds instead, HOLDER's (at WHERE, its name there, when it has one of its own), which VERB says of: (the report's,
    the record's). The report's shows neither value for a private key (PRIVATE)."""
    at = f"{holder} {where}" if where else holder
    full = f"{key.name}: config.json's {_shown(value)} is not carried; {at} {_shown(held)} {verb}"
    if key.klass != PRIVATE:
        return full, full
    return (
        f"{key.name}: config.json's value is not carried; {holder} {where or 'value'} {verb}",
        full,
    )


def _refused(key, value, why: str) -> tuple:
    """The line of the state key KEY whose VALUE in config.json the internal state's schema refuses, WHY: (the
    report's, the record's). The report's leaves out a private key's why, which quotes the value.
    """
    lead = f"{key.name}: config.json's {_shown(value)} is not carried; the internal state's schema refuses it"
    if key.klass != PRIVATE:
        return f"{lead} ({why})", f"{lead} ({why})"
    shown = f"{key.name}: config.json's value is not carried; the internal state's schema refuses it"
    return shown, f"{lead} ({why})"


# Why a key of config.json is dropped, by what it is (the keys of removed_keys say their own why: removed_keys.why)
UNSET = "your deploy settings set none of them"
COMPUTED = "each load computes them"
UNKNOWN = "no key of this package has the name"


def _owned(key, v, running, state: dict, package: dict) -> tuple:
    """What becomes of the key KEY (a registry.Key) at the value V of config.json, by its owner: (whether it is carried,
    the lines that say what holds instead, `_not_carried`'s, why it is dropped); None for no lines, "" for no why. See
    the module's docstring.
    """
    from . import deploy_tree, repository_defaults

    name, runs = key.name, registry.get(running.values, key.name)
    if key.owner == registry.DERIVED:
        return False, None, COMPUTED
    if _blank(v):
        return False, None, ""
    if key.owner == registry.STATE:
        held = registry.get(state, name)
        if held is registry.MISSING:
            return v != key.default, None, ""
        if held == v:
            return False, None, ""
        return False, _not_carried(key, v, "the internal state's", held, "is kept"), ""
    if key.owner == registry.DEPLOY:
        if runs is registry.MISSING:
            return False, None, UNSET
        if v == runs:
            return False, None, ""
        where = deploy_tree.name(name)
        where = where if where != name else ""
        return (
            False,
            _not_carried(key, v, "your deploy settings'", runs, "runs", where),
            "",
        )
    own = repository_defaults.own(v, registry.get(package, name))
    if not own or v == runs:
        return False, None, ""
    theirs = running.source.get(name) == runtime_config.REPOSITORY
    whose = "the repository defaults'" if theirs else "the package's"
    return False, _not_carried(key, v, whose, runs, "runs"), ""


def split(values: dict, running, state: dict) -> Split:
    """config.json's VALUES split by owner: see the module's docstring."""
    from . import removed_keys

    package, state = registry.package_defaults(), state or {}
    carried, said = {}, {registry.STATE: [], registry.DEPLOY: [], registry.PACKAGE: []}
    removed, dropped = {}, {}
    for name in registry.leaves(values or {}, {k.name for k in registry.keys()}):
        v, kind = registry.get(values, name), registry.kind(name)
        if removed_keys.why(name):
            names = removed.setdefault(removed_keys.why(name), [])
            names.append(removed_keys.listed(name))
        elif kind == registry.KEY:
            key = registry.key(name)
            carry, pair, why = _owned(key, v, running, state, package)
            if carry:
                carried[name] = v
            if pair:
                said[key.owner].append(pair)
            if why:
                dropped.setdefault(why, []).append(name)
        elif not (kind == registry.SECTION and _blank(v)):
            dropped.setdefault(UNKNOWN, []).append(name)
    pairs = said[registry.STATE] + said[registry.DEPLOY] + said[registry.PACKAGE]
    for why, names in [*removed.items(), *dropped.items()]:
        line = f"dropped ({why}): {', '.join(dict.fromkeys(names))}"
        pairs.append((line, line))
    return Split(carried, [shown for shown, _ in pairs], [full for _, full in pairs])


# --- the record -------------------------------------------------------------------------------------------------------


def _restart(lead: str = "restart the monitor:") -> tuple:
    """LEAD, then the commands that restart the monitor, whose start runs again what failed, as you type them."""
    from . import steps

    return (lead, steps.command_line("stop"), ", then", steps.command_line("start"))


def _off() -> tuple:
    """The step that runs `rkmon standby off` again."""
    return ("run", _standby_off())


def writable(folder, again: tuple) -> Steps:
    """The fix of FOLDER, which the monitor cannot write: see the module's docstring (AGAIN: `_restart`, `_off`, or
    the caller's)."""
    return Steps((f"make {folder} writable for the monitor, then", *again))


def _write(home, rec: dict, again: tuple) -> None:
    """REC written whole as the record (state_store.write_file); MoveError when it cannot be, its fix `writable`'s,
    with AGAIN."""
    from . import state_store

    path = record_file(home)
    try:
        state_store.write_file(path, rec)
    except OSError as exc:
        why = exc.strerror or exc
        what = f"the record of the one-time move cannot be written in {path.parent}"
        raise MoveError(f"{what}: {why}", writable(path.parent, again)) from None


def _until_line(now, then: str = "") -> str:
    """The line that says UNTIL is NOW, then THEN."""
    return (
        f"{UNTIL} is {now}: the reviews the older monitor posted before then, whose line 1 names no automated "
        f"reviewer, are this deployment's own{then}"
    )


def recorded(home, running, older: Older, split_: Split, refused: dict, now) -> list:
    """The move just made, written in its record, and its report: see the module's docstring."""
    from . import states  # a call-time import: standby's words have their home there

    path = config_file(home)
    off = _standby_off()
    standby = f"the monitor is {states.in_standby()}"
    if older.values is None:
        what = f"nothing of config.json is carried, as {older.problem}"
    else:
        what = f"{path}, an older install's configuration, split by owner, once"
    carried = [n for n in split_.carried if n not in refused]
    head = [f"moved: {what}; {standby}", *older.notes]
    if carried:
        head.append(f"carried into the internal state: {', '.join(carried)}")
    pairs = [
        _refused(registry.key(n), split_.carried[n], why) for n, why in refused.items()
    ]
    tail = [_until_line(now, f"; {off} sets it again, to its own time")]
    shown = head + [p[0] for p in pairs] + split_.lines + tail
    report = head + [p[1] for p in pairs] + split_.full + tail
    layer = running.layer or {}
    rec = dict(kind=RECORD_KIND, schema_version=RECORD_SCHEMA_VERSION, moved_at=now)
    rec.update({k: layer.get(k) for k in runtime_config.APPLIED_MARK})
    rec.update(config_json_sha256=older.sha256, report=report)
    rec.update(ended_at="", end=[], kept=[])
    _write(home, rec, _restart())
    return shown


# --- the end ----------------------------------------------------------------------------------------------------------


class _Place(NamedTuple):
    """An older copy the end looks at: ENTRY, its entry's name ("" for a file no entry claims, and for the older
    package's code); PLACE; GONE, the line that says it goes, else KEPT, the line that says why it stays, and FIX, the
    fix that sets its entry (None for none).
    """

    entry: str
    place: str
    gone: str = ""
    kept: str = ""
    fix: object = None


def _once_set(cfg: dict, entry, place: str, environ) -> _Place:
    """The older copy at PLACE of ENTRY, which goes once ENTRY is set in its own place (logins.found)."""
    from . import logins
    from . import secrets_registry as sr

    got = logins.found(cfg, entry, environ)
    what = f"an older copy of {entry.name}"
    if got["set"] is True:
        gone = f"removed {place}: {what}, which is set in its own place"
        return _Place(entry.name, place, gone=gone)
    if got["set"] is False:
        why = "it is not set in its own place"
    else:
        why = f"whether it is set in its own place is not known ({got['where']})"
    kept = f"kept {place}: {what}, as {why}; once it is set, the next start removes this copy"
    return _Place(entry.name, place, kept=kept, fix=sr.set_fix(entry))


def _never_copied(cfg: dict, entry, place: str, environ) -> _Place:
    """The older copy at PLACE of the login ENTRY, which the move never copies: it goes, its line saying whether the
    login is set in its own place (logins.found), with the fix that sets it while it is not.
    """
    from . import logins
    from . import secrets_registry as sr

    got = logins.found(cfg, entry, environ)
    gone = f"removed {place}: an older copy of the login {entry.name}, which the move never copies"
    if got["set"] is True:
        return _Place(
            entry.name, place, gone=f"{gone}; the login is set in its own place"
        )
    if got["set"] is False:
        why = "the login is not set in its own place"
    else:
        why = f"whether the login is set in its own place is not known ({got['where']})"
    return _Place(entry.name, place, gone=f"{gone}; {why}", fix=sr.set_fix(entry))


def _older_code(home) -> str:
    """The older package's code (OLDER_CODE in the work volume), when HOME is the work volume's monitor home and the
    folder is there; else ""."""
    from . import container_layout

    if not _here(home):
        return ""
    place = os.path.join(container_layout.WORK_DIR, OLDER_CODE)
    return place if os.path.lexists(place) else ""


def _places(home, cfg: dict, environ) -> list:
    """Each older copy there is for the monitor's home HOME, and whether the end removes it: see the module's
    docstring. paths.PathsError when no secrets folder is known."""
    from . import paths
    from . import secrets_registry as sr

    entries = sr.entries(cfg)
    own = set()
    for e in entries:
        own |= {sr.home(cfg, e, environ), sr.path(cfg, e, environ)}
    claimed, out = set(own), {}
    for e in entries:
        for older, place in sr.older_places(cfg, e, environ):
            claimed.add(place)
            if place in own or place in out or not os.path.lexists(place):
                continue
            if older.copied or e.shape != sr.LOGIN:
                out[place] = _once_set(cfg, e, place, environ)
            else:
                out[place] = _never_copied(cfg, e, place, environ)
    folder = paths.get(cfg, "secrets_dir")
    for name in sr.OLDER_SECRET_FILES:
        place = os.path.abspath(os.path.expanduser(f"{folder}/{name}"))
        if place not in claimed and os.path.lexists(place):
            gone = f"removed {place}: an older package kept a secret there, which nothing reads now"
            out[place] = _Place("", place, gone=gone)
    code = _older_code(home)
    if code:
        gone = f"removed {code}: the older package's code, which nothing runs once the move has ended"
        out[code] = _Place("", code, gone=gone)
    return list(out.values())


def _remove(place: str) -> str:
    """PLACE removed: a file or a link (the link alone), or a folder and all it holds; "" once it is gone, else why."""
    p = pathlib.Path(place)
    try:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        return exc.strerror or str(exc)
    return ""


def _said(line: str, fix) -> list:
    """LINE, then the line of FIX when there is one."""
    return [line] + ([f"{FIX_LABEL}{fix}"] if fix is not None else [])


def _act(found: list) -> tuple:
    """Each place of FOUND removed or kept: (its lines, each place kept, as the record keeps it)."""
    lines, kept = [], []
    for p in found:
        if p.gone:
            why = _remove(p.place)
            if not why:
                lines += _said(p.gone, p.fix)
                continue
            said = f"kept {p.place}: it cannot be removed: {why}; the next start removes it"
            lines.append(said)
        else:
            lines += _said(p.kept, p.fix)
        kept.append({"entry": p.entry, "place": p.place})
    return lines, kept


def _rename(home) -> list:
    """config.json renamed MIGRATED, read-only, in lines; MoveError when it cannot be renamed."""
    src = config_file(home)
    dst = src.with_name(MIGRATED)
    if not os.path.lexists(src):
        return [f"{src} is not there: nothing is renamed"]
    try:
        os.replace(src, dst)
    except OSError as exc:
        why = exc.strerror or exc
        what = f"the one-time move cannot end: {src} cannot be renamed {dst.name}"
        raise MoveError(f"{what}: {why}", writable(src.parent, _off())) from None
    try:
        if not dst.is_symlink():
            os.chmod(dst, stat.S_IMODE(os.lstat(dst).st_mode) & ~0o222)
    except OSError as exc:
        why = exc.strerror or exc
        return [f"renamed {src} {dst.name}; it cannot be made read-only: {why}"]
    return [f"renamed {src} {dst.name}, read-only"]


def end(home, cfg: dict, environ=None) -> list:
    """The end of the open move: see the module's docstring."""
    from . import paths, steps, store

    lead = "the one-time move cannot end"
    if not is_open(record(home)):
        if due(home):
            restart = _restart("restart the monitor, whose start makes the move:")
            fix = Steps((*restart, "; then run", _standby_off()))
            raise MoveError(
                f"{lead}: it is not made yet; the monitor's start makes it", fix
            )
        return []
    # the places are found before the lock is taken: logins.found may ask the writer
    try:
        found = _places(home, cfg, environ)
    except paths.PathsError:
        what = "the configuration names no secrets folder, so the older copies of the secrets cannot be found"
        doctor = steps.command_line("doctor")
        fix = Steps(
            ("run", doctor, "and put right what it says, then run", _standby_off())
        )
        raise MoveError(f"{lead}: {what}", fix) from None
    try:
        with _locked(home):
            rec = record(home)
            if not is_open(rec):
                return []
            lines = _rename(home)
            done, kept = _act(found)
            lines += done
            try:
                now = store.now()
                lines += _until(home, now)
                rec.update(ended_at=now, end=lines, kept=kept)
                _write(home, rec, _off())
            except MoveError as exc:
                exc.done = tuple(lines)
                raise
    # the lock's: each step under it says its own failure as a MoveError
    except OSError as exc:
        folder = config.state_dir(home)
        why = exc.strerror or exc
        what = f"{lead}: the internal state's lock cannot be taken in {folder}"
        raise MoveError(f"{what}: {why}", writable(folder, _off())) from None
    return lines


def _until(home, now) -> list:
    """UNTIL set to NOW, the end's time, in the internal state, whose lock the end holds (state_store.set), in lines;
    MoveError, with its fix, when it cannot be: the end's work is done, and the end, run again, ends it.
    """
    from . import state_store
    from .doctor import wrapper_name

    lead = f"the one-time move cannot end: {UNTIL} cannot be set"
    try:
        state_store.set(home, UNTIL, now, held=True)
    except OSError as exc:
        folder = config.state_dir(home)
        why = exc.strerror or exc
        raise MoveError(
            f"{lead} in {folder}: {why}", writable(folder, _off())
        ) from None
    except state_store.StateError as exc:
        raise MoveError(f"{lead}: {exc}", exc.fix(wrapper_name())) from None
    return [_until_line(now)]


def _there(kept) -> bool:
    """Whether KEPT, a place the record keeps, is still there."""
    return isinstance(kept, dict) and os.path.lexists(str(kept.get("place")))


def retry(home, cfg: dict, environ=None) -> list:
    """Each older copy the end kept, removed once the end's rule lets it go: see the module's docstring."""
    rec = record(home)
    if not rec or is_open(rec) or not rec.get("kept"):
        return []
    kept = {str(k.get("place")) for k in rec["kept"] if isinstance(k, dict)}
    found = [p for p in _places(home, cfg, environ) if p.place in kept and p.gone]
    lines = []
    with _locked(home):
        rec = record(home)
        if not rec or is_open(rec):
            return []
        for p in found:
            if not _remove(p.place):
                lines += _said(p.gone, p.fix)
        was = rec.get("kept") or []
        left = [k for k in was if _there(k)]
        if left != was:
            rec["kept"] = left
            _write(home, rec, _restart())
    return lines


# --- at the start -----------------------------------------------------------------------------------------------------


def _failed(lead: str, exc: Exception, then: str) -> list:
    """LEAD, why (EXC: a config.ConfigError, or an OSError, by its file) and THEN in a line, then the fix a ConfigError
    names (config.ConfigError.fix), as the host's command runs it."""
    from .doctor import wrapper_name

    fix = None
    if isinstance(exc, config.ConfigError):
        why, fix = str(exc), exc.fix(wrapper_name())
    elif isinstance(exc, OSError) and exc.filename and exc.strerror:
        why = f"{exc.filename}: {exc.strerror}"
    else:
        why = str(exc)
    return [f"{lead}: {why}{then}"] + ([f"{FIX_LABEL}{fix}"] if fix is not None else [])


def at_start(home, running, environ=None) -> list:
    """What rk start does of the move: see the module's docstring."""
    from . import paths, state_store

    if running.layer is not None and due(home):
        try:
            return state_store.migrate(home, running)
        except (OSError, config.ConfigError) as exc:
            then = "; the monitor is in standby until the move is made, which the next start tries again"
            return _failed("not moved", exc, then)
    try:
        return retry(home, running.values, environ)
    except (OSError, config.ConfigError, paths.PathsError) as exc:
        return _failed("the older copies the one-time move kept stay", exc, "")
