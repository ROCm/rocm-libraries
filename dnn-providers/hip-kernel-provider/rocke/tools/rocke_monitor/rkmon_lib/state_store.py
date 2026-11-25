# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The monitor's internal state: what the monitor found or recorded as it ran, in STATE_FILE
in the monitor's state folder (config.state_dir), of the kind file_types.INTERNAL_STATE
(schema/internal_state.schema.json). This module is the file's only writer.

The file holds each state key of the registry (rkmon_lib/registry.py, STATE) at its key in the file (the key's
FILE_KEYS); the configuration reads it under the key's name. What the file holds of a model binding
(model_bindings.<binding>.<field>) is the configuration's for the binding today's container runs as its local worker
(deploy_tree.WORKER) only; the file keeps every other binding's as it is.

- `path(home)`: the file, in the state folder of the monitor's home HOME.
- `file_key(key) -> str`: where the state key KEY (a registry.Key) is in the file, by the file's names: its file key, a
  binding's field at the local worker's binding.
- `read(home) -> dict`: the state keys the file holds, nested under the configuration's names; {} when there is no
  file. StateError when the file cannot be read, is no JSON object, is of another kind or layout, or holds a value
  its schema refuses, each naming its fix (config.ConfigError.fix): the file's mode, when it keeps the monitor out;
  the file mended where its JSON breaks, or moved aside (settings_file.aside_fix); each key whose value the schema
  refuses deleted, by name.
- `set(home, name, value, fix=None, held=False)`: NAME at VALUE in the file: a state key, or an entry of a state key
  whose value is a mapping (jira.accounts.<login>), by the configuration's name. The file is read, changed and
  written whole (`write_file`) under its lock (LOCK_FILE, beside it), so two writers never lose each other's change;
  HELD: the caller holds the lock already (`locked_in`), as one_time_move.end does. StateError, and nothing written,
  when NAME is not state (the message says who sets it), when VALUE is one the schema refuses (its fix FIX, the
  caller's), or when the file cannot be read (as `read` refuses it, with its fix). An OSError of its lock or of its
  write goes through to the caller, with nothing written (`unwritable` says it, with its fix).
- `staged(home, values)`: VALUES ({name: value}, each name as `set` takes it) set in the file of HOME, a copy that no
  monitor reads yet: rkmon adopt's stage, which docker/rkmon makes on the host, where worker/perms.py is not
  (paths.worker_module). Read, changed and written whole as `set` writes it, but with no lock and at STAGED_MODE, its
  maker's alone. StateError, and nothing written, as `set` raises it.
- `unwritable(home, exc) -> StateError`: the file that cannot be written for EXC, an OSError of `set`'s lock or of its
  write, as a StateError that says why and names its fix. Where a mode keeps the monitor out, the fix gives the mode,
  as `read`'s gives the file's (the monitor and the writer share them): the lock file's, when the lock is what EXC
  names; else that of the folder the write makes its new file in. Another error names no fix: the reason says what to
  put right (a full disk, a file system mounted read-only).
- `migrate(home, running=None) -> list`: the work volume's one-time move (rkmon_lib/one_time_move.py),
  when one is due (one_time_move.due), under the file's lock: config.json split by owner (one_time_move.split, with
  the state the file holds), each state key it carries put in the file where the schema takes it, the time before
  which an older monitor's unnamed reviews are this deployment's own (one_time_move.UNTIL) set to now, standby
  entered (standby.on, its reason one_time_move.REASON, since now) unless it is on already, the file written whole,
  then the move recorded (one_time_move.recorded). Its report, in lines; [] when no move is due. RUNNING is the
  configuration the monitor runs, loaded from the delivered layer (runtime_config.load(home) when None). StateError,
  nothing written, when the file cannot be read, as `read` refuses it; one_time_move.MoveError when the record cannot
  be written, the file written: the move stays due, and made again it finds the state keys carried already.
- `keep_unsent(folder, item) -> bool`: ITEM, an unsent item (what a feature in prototype, or the
  whole monitor in standby, would have sent), appended as one JSON line to UNSENT_FILE in the state folder FOLDER
  (config.state_dir: beside the file), under the file's lock, at the shared files' mode: its type (TYPES), what it
  was for, when (`at`: now, when ITEM has none), the feature, why it was kept (prototype or standby), the content as
  it would have gone out, and the file that holds a content kept as a file (a review, the team dashboard's page).
  True when it was kept now; False, and nothing written, when UNSENT_FILE holds the same item already (SAME: the same
  type, for, feature, why, content and file), so that a write the monitor asks for at every tick is one item. The
  team dashboard's page (PAGE) is one item, replaced at each keep: the file is written whole without the page items
  it held (every other line as it was), then ITEM; True only when none of those was the same item.
  ValueError when ITEM lacks a field of REQUIRED, its type is not one of TYPES, or WHY is neither PROTOTYPE nor
  STANDBY. retention.LOGS rotates the file, so an item is kept for the package's retention, then dropped; no item is
  ever sent.
- `locked_in(folder)`: the file's lock in the state folder FOLDER, held while the block runs: the lock `set` and
  `keep_unsent` take, which retention takes too while it moves UNSENT_FILE, so that a keep never writes the moved lines
  back.
- `unsent_items(folder) -> list`: the unsent items kept in the state folder FOLDER, oldest first (the rotated file's,
  then the file's; a line that does not parse is skipped), with the team dashboard's page (type PAGE) once: its
  newest item.
- `write_file(file, data)`: DATA as JSON at the path FILE, written whole and durable (whole_file.write: a crash
  leaves the old file or the new one), at the mode of the files the monitor and the writer share (worker/perms.py,
  SHARED_FILE_MODE). A missing folder is made at the shared folders' mode (SHARED_DIR_MODE).
- `StateError`: a config.ConfigError.
"""

from __future__ import annotations

import contextlib
import copy
import errno
import json
import os
import pathlib

from . import config, paths, registry, whole_file
from .checks import DELETE, Command, Edit, Steps

STATE_FILE = "internal_state.json"
LOCK_FILE = "internal_state.lock"
STAGED_MODE = 0o600  # a staged copy's (`staged`): its maker's alone
UNSENT_FILE = "unsent.jsonl"  # beside STATE_FILE: the unsent items, one JSON line each
PROTOTYPE, STANDBY = "prototype", "standby"  # why an item was kept
# The types of unsent item: what the write would have made. The team dashboard's page (PAGE) is one item.
REVIEW, COMMENT, LABEL, CHANGE_REQUEST, AUTO_MERGE, ISSUE, ALERT, PAGE, TICKET = (
    "review",
    "comment",
    "label",
    "change request",
    "auto-merge",
    "issue",
    "alert",
    "page push",
    "ticket",
)
TYPES = (REVIEW, COMMENT, LABEL, CHANGE_REQUEST, AUTO_MERGE, ISSUE, ALERT, PAGE, TICKET)
REQUIRED = ("type", "for", "feature", "why")
SAME = ("type", "for", "feature", "why", "content", "file")  # what makes two items one


class StateError(config.ConfigError):
    """The internal state cannot be read, or a change to it is refused: see the module's docstring."""


def _perms():
    return paths.worker_module("perms")


def _file_type():
    from . import file_types

    return file_types.INTERNAL_STATE


def path(home) -> pathlib.Path:
    """The internal state file of the monitor's home HOME."""
    return config.state_dir(home) / STATE_FILE


# --- names ------------------------------------------------------------------------------------------------------------


def file_key(key: registry.Key) -> str:
    """Where the state key KEY is in the file: see the module's docstring."""
    from . import deploy_tree

    return key.file_keys[0].replace("*", deploy_tree.WORKER, 1)


def _not_state(name: str) -> str:
    """Why NAME is neither a state key nor an entry of one, naming who sets it: a package key's value is
    config.defaults.json's, or the repository defaults' for the folders they were written for when they set it.
    """
    from . import deploy_tree, repository_defaults

    try:
        owner = registry.key(name).owner
    except KeyError:
        owner = None
    if owner == registry.DEPLOY:
        return f"{name} is not internal state: your deploy settings file sets it ({deploy_tree.name(name)})"
    if owner == registry.PACKAGE:
        where = "config.defaults.json"
        if name in repository_defaults.keys():
            where += (
                ", or the repository defaults that apply to your repository and folder"
            )
        return f"{name} is not internal state: the package sets it ({where})"
    if owner == registry.DERIVED:
        return (
            f"{name} is not internal state: it is computed from other keys at each load"
        )
    if registry.kind(name) == registry.SECTION:
        return f"{name} is a section, not a key: set its keys one by one"
    return f"{name} is no key of the configuration"


def _place(name: str) -> str:
    """Where NAME, a state key or an entry of a state key's mapping, is in the file; StateError for any other name."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        try:
            key = registry.key(".".join(parts[:i]))
        except KeyError:
            continue
        if key.owner == registry.STATE and (
            i == len(parts) or isinstance(key.default, dict)
        ):
            return ".".join([file_key(key), *parts[i:]])
        break
    raise StateError(_not_state(name))


# --- the file ---------------------------------------------------------------------------------------------------------


def _kind(body: dict) -> str:
    """Why BODY, the file's whole content, is of another kind or layout than internal state; "" when it is not."""
    state = _file_type()
    if body.get("kind") != state.kind or body.get("schema_version") != state.version:
        return (
            f"it is of the kind {body.get('kind')!r} at layout {body.get('schema_version')!r}, not "
            f"{state.kind!r} at layout {state.version}"
        )
    return ""


def _misfits(body: dict) -> list:
    """The check rows of the values in BODY, internal state's whole content, that their keys do not take
    (settings_file.misfits)."""
    from . import settings_file

    state = _file_type()
    values = {k: v for k, v in body.items() if k not in settings_file.HEADER_KEYS}
    return settings_file.misfits(settings_file.Document(state, state.version, values))


def _problems(body: dict) -> list:
    """What keeps BODY, the file's whole content, from being internal state, one line each."""
    kind = _kind(body)
    return [kind] if kind else [f"{r.name}: {r.found}" for r in _misfits(body)]


def _aside(p: pathlib.Path, mend: bool = False):
    """The fix of the file P that is no internal state the monitor reads: mend it (MEND), or move it aside."""
    from . import settings_file

    return settings_file.aside_fix(p, mend=mend)


def _unreadable_fix(p: pathlib.Path, exc: OSError):
    """The fix of the file P that cannot be read for EXC: its mode, when that keeps the monitor out (the monitor
    and the writer share the file); else move it aside."""
    from . import settings_file

    if exc.errno not in (errno.EACCES, errno.EPERM):
        return _aside(p)
    mode = format(_perms().SHARED_FILE_MODE, "o")
    chmod = Command(("chmod", mode, settings_file.shown(p)))
    return Steps((chmod, "as its owner: the monitor and the writer share it"))


def unwritable(home, exc: OSError) -> StateError:
    """The file that cannot be written for EXC, an OSError of `set`, as a StateError: see the module's docstring."""
    p = path(home)
    return StateError(
        f"the internal state {p} cannot be written: {exc.strerror or exc}",
        _unwritable_fix(p, exc),
    )


def _unwritable_fix(p: pathlib.Path, exc: OSError):
    """The fix of the file P that cannot be written for EXC (`unwritable`): the mode of the lock file EXC names, or of
    the folder it makes a file in, when that keeps the monitor out; else None."""
    from . import settings_file

    if exc.errno not in (errno.EACCES, errno.EPERM):
        return None
    named = pathlib.Path(exc.filename) if exc.filename else None
    if named is not None and named.is_file():
        place, mode = named, _perms().SHARED_FILE_MODE
    else:
        place, mode = (named.parent if named else p.parent), _perms().SHARED_DIR_MODE
    chmod = Command(("chmod", format(mode, "o"), settings_file.shown(place)))
    return Steps((chmod, "as its owner: the monitor and the writer share it"))


def _schema_fix(p: pathlib.Path, body: dict):
    """The fix of the file P whose content BODY breaks its schema: move it aside when it is of another kind or
    layout; else delete each key whose value the schema refuses, by name."""
    from . import settings_file

    if _kind(body):
        return _aside(p)
    where, parts = settings_file.shown(p), []
    names = list(dict.fromkeys(r.name for r in _misfits(body)))
    for i, name in enumerate(names):
        if i:
            parts.append("and" if i == len(names) - 1 else ",")
        parts.append(Edit(name, where, DELETE))
    return parts[0] if len(parts) == 1 else Steps(tuple(parts))


def _body(home) -> dict:
    """The file's whole content, checked; a new file's header when there is no file."""
    p = path(home)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        state = _file_type()
        return {"kind": state.kind, "schema_version": state.version}
    except OSError as exc:
        raise StateError(
            f"the internal state {p} cannot be read: {exc.strerror or exc}",
            _unreadable_fix(p, exc),
        ) from None
    try:
        body = json.loads(text)
    except ValueError as exc:
        raise StateError(
            f"the internal state {p} is no JSON: {exc}", _aside(p, mend=True)
        ) from None
    if not isinstance(body, dict):
        raise StateError(f"the internal state {p} is no JSON object", _aside(p))
    problems = _problems(body)
    if problems:
        raise StateError(
            f"the internal state {p} breaks its schema: " + "; ".join(problems),
            _schema_fix(p, body),
        )
    return body


def _values(body: dict) -> dict:
    """The state keys BODY, the file's whole content, holds, nested under the configuration's names."""
    out: dict = {}
    for key in registry.keys():
        if key.owner == registry.STATE:
            v = registry.get(body, file_key(key))
            if v is not registry.MISSING:
                registry.put(out, key.name, copy.deepcopy(v))
    return out


def read(home) -> dict:
    """The state keys the file holds, under the configuration's names: see the module's docstring."""
    return _values(_body(home))


@contextlib.contextmanager
def _locked(home):
    """The file's lock, held (locked_in its state folder)."""
    with locked_in(path(home).parent):
        yield


@contextlib.contextmanager
def locked_in(folder):
    """The file's lock in the state folder FOLDER, held: LOCK_FILE beside the file, at the shared files' mode
    (perms.shared_lock). See the module's docstring."""
    p = pathlib.Path(folder) / LOCK_FILE
    _folder(p.parent)
    with _perms().shared_lock(p):
        yield


def _changed(home, values: dict, fix=None) -> dict:
    """The file's whole content with VALUES (name -> value) set, checked: StateError, with FIX, as `set` raises it."""
    places = {name: _place(name) for name in values}
    body = _body(home)
    for name, value in values.items():
        registry.put(body, places[name], copy.deepcopy(value))
    problems = _problems(body)
    if problems:
        names = f"{', '.join(values)} {'is' if len(values) == 1 else 'are'}"
        raise StateError(f"{names} not set: " + "; ".join(problems), fix)
    return body


def set(home, name: str, value, fix=None, held: bool = False) -> None:
    """NAME at VALUE in the file: see the module's docstring."""
    _place(name)  # a name that is not state is refused before the lock is taken
    with contextlib.nullcontext() if held else _locked(home):
        write_file(path(home), _changed(home, {name: value}, fix))


def staged(home, values: dict) -> None:
    """VALUES set in a copy of the file that no monitor reads yet: see the module's docstring."""
    body = _changed(home, values)
    p = path(home)
    p.parent.mkdir(parents=True, exist_ok=True)
    whole_file.write(p, _text(body), STAGED_MODE)


def migrate(home, running=None) -> list:
    """The one-time move of the work volume, when one is due: see the module's docstring."""
    # call-time imports: the move reads the configuration, whose load reads this module
    from . import one_time_move as move
    from . import runtime_config, store

    running = runtime_config.load(home) if running is None else running
    with _locked(home):
        if not move.due(home):
            return []
        body = _body(home)
        older = move.older_settings(home)
        split = move.split(older.values, running, _values(body))
        refused = {}
        for name, value in split.carried.items():
            trial = copy.deepcopy(body)
            registry.put(trial, _place(name), copy.deepcopy(value))
            problems = _problems(trial)
            if problems:
                refused[name] = "; ".join(problems)
            else:
                body = trial
        now = store.now()
        set_now = {move.UNTIL: now}
        if registry.get(body, _place("standby.on")) is not True:
            set_now.update(
                {
                    "standby.on": True,
                    "standby.reason": move.REASON,
                    "standby.since": now,
                }
            )
        for name, value in set_now.items():
            registry.put(body, _place(name), value)
        write_file(path(home), body)
        return move.recorded(home, running, older, split, refused, now)


# --- the unsent items -------------------------------------------------------------------------------------------------


def unsent_path(folder) -> pathlib.Path:
    """The unsent items' file in the state folder FOLDER (config.state_dir), beside the internal state."""
    return pathlib.Path(folder) / UNSENT_FILE


def _lines(p: pathlib.Path) -> list[dict]:
    """The items of the file P, in order; a line that does not parse (a write cut short) is skipped."""
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    out = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def _same(item: dict) -> tuple:
    return tuple(item.get(k) or "" for k in SAME)


def keep_unsent(folder, item: dict) -> bool:
    """ITEM appended to the unsent items: see the module's docstring."""
    gone = [k for k in REQUIRED if not item.get(k)]
    if gone:
        raise ValueError(f"an unsent item needs {', '.join(gone)}")
    if item["why"] not in (PROTOTYPE, STANDBY):
        raise ValueError(
            f"an unsent item is kept for {PROTOTYPE} or {STANDBY}, not {item['why']!r}"
        )
    if item["type"] not in TYPES:
        raise ValueError(
            f"an unsent item is of a type of {', '.join(TYPES)}, not {item['type']!r}"
        )
    if not item.get("at"):
        from .store import (
            now,
        )  # a call-time import: store reads config, which reads this module

        item = dict(item, at=now())
    p = unsent_path(folder)
    line = (json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    with locked_in(folder):
        if item["type"] == PAGE:
            return _replace_page(p, item, line)
        if any(_same(old) == _same(item) for old in _lines(p)):
            return False
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(p, flags, _perms().SHARED_FILE_MODE)
        try:
            if os.fstat(fd).st_uid == os.getuid():
                os.fchmod(
                    fd, _perms().SHARED_FILE_MODE
                )  # the umask may have taken a bit away
            size = os.lseek(fd, 0, os.SEEK_END)
            if size and os.pread(fd, 1, size - 1) != b"\n":
                line = b"\n" + line  # a line cut short stays apart, and is skipped
            os.write(fd, line)
        finally:
            os.close(fd)
    return True


def _replace_page(p: pathlib.Path, item: dict, line: bytes) -> bool:
    """The page item ITEM (its LINE) in place of the page items the file P holds, the file written whole (the caller
    holds its lock); whether none of them was the same item (keep_unsent)."""
    try:
        old = p.read_bytes().splitlines(keepends=True)
    except FileNotFoundError:
        old = []
    rest, new = [], True
    for raw in old:
        try:
            was = json.loads(raw)
        except ValueError:
            was = None
        if isinstance(was, dict) and was.get("type") == PAGE:
            new = new and _same(was) != _same(item)
            continue
        rest.append(raw if raw.endswith(b"\n") else raw + b"\n")
    whole_file.write(p, b"".join(rest) + line, _perms().SHARED_FILE_MODE, durable=True)
    return new


def unsent_items(folder) -> list[dict]:
    """The unsent items kept: see the module's docstring."""
    from .retention import with_older

    items = [item for p in with_older(unsent_path(folder)) for item in _lines(p)]
    pages = [i for i, item in enumerate(items) if item.get("type") == PAGE]
    return [
        item
        for i, item in enumerate(items)
        if item.get("type") != PAGE or i == pages[-1]
    ]


# --- writing ----------------------------------------------------------------------------------------------------------


def _folder(folder: pathlib.Path) -> None:
    """FOLDER, made at the shared folders' mode when it is missing."""
    if folder.is_dir():
        return
    folder.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):  # another user made it first
        os.chmod(folder, _perms().SHARED_DIR_MODE)


def _text(data) -> str:
    """DATA as the JSON text of a file this module writes."""
    return json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def write_file(file, data) -> None:
    """DATA as JSON at the path FILE, whole: see the module's docstring."""
    p = pathlib.Path(file)
    _folder(p.parent)
    whole_file.write(p, _text(data), _perms().SHARED_FILE_MODE, durable=True)
