# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The host's records: what only the host keeps about this package folder's deployment, in
FILE, written by confirm, apply, rollback and doctor, by deliver (rkmon_lib/deliver.py) with each delivery and
put-back, the first start's included, and by the first command that sees a tick since a start, through private_files
(a new file at 600, renamed into place). The state table reads them (rkmon_lib/state_facts.py).

- `Confirmed(settings_file, fingerprint, at, file_sha256, package_version, code_sha256, container_name, image,
  changes)`: the last confirm: the settings file, as messages show it (`~/…` under the home folder,
  settings_file.shown); the settings' fingerprint (deploy_plan.fingerprint); when they were confirmed (`timestamp`);
  the file's own values' SHA-256 (deploy_plan.file_sha256), which tells a change of the file from a change of this
  package; this package's version then, and the SHA-256 of its code (image.code_sha256); the confirmed
  container.name; the image they name, an `Image`; and how many changes the confirm showed.
- `Image(registry, name, tag)`: an image as the settings name it (image.registry, "" for none; image.name;
  image.tag). `reference`: NAME:TAG, as the image is tagged on this host.
- `UnderWay(command, since, installed, baseline, failing, code_back, kept, first_apply)`: an apply, rollback or adopt
  under way (COMMANDS); since when; whether a release went into the code volume during the change
  (`Lease.record_install`), which the command that takes the place of one that stopped part-way carries on (begin),
  so that its put-back puts that code back too; whether the doctor ran before the change (BASELINE, apply's step 1),
  with the names of the checks it found failing then (FAILING), which that command keeps too (`baseline`); whether a
  rollback puts the code back with the settings (CODE_BACK), which the rollback that takes its place keeps
  (`code_back`); whether the delivery of an apply's change kept the settings in place before it (KEPT:
  deliver.keep_previous; `Lease.record_kept`), which the command that takes its place carries on too; and whether the
  change is a first apply over an older package (`first_apply`), which
  begin records as it is given, and the command that takes its place carries on too. False, False, (), False, False
  and False in a record without them.
- `PutBack(fingerprint, reason, at, failure, fix)`: a confirmed change that was put back: its fingerprint; why (one of
  REASONS: by a rollback, or after a restart, a load or a doctor check that failed; or FIRST_APPLY, a change not put
  back: the first apply over an older package failed and stopped the new container); when; what failed, in a few
  words (for a rollback, what failed after it; "" for none); and the fix for it (a checks.Fix, or None), kept in its
  JSON form (checks.fix_to_json).
- `Doctor(at, checks, failing)`: the last doctor result: when, how many checks ran, and the names of those that
  failed.
- `Written(env, override, layer, kept_layer)`: the fingerprint (deliver.fingerprint) of each file deliver last wrote
  for this folder's deployment, which each delivery and put-back keeps (deliver.wrote): docker/.env's, the
  override's, the delivered layer's, and that of the layer the settings volume keeps, which a put-back puts in its
  place. "" for a file with no fingerprint kept (an older package wrote it): it counts as unchanged (state 9).
  NOT_KEPT: each of them "".
- `FirstTick(at)`: when deploy step 8, the first tick, first passed after a start: the time of the tick since a start
  that the first command to see one saw. From then on a next line names a deploy step's command without its step
  (states.step_8_passed).
- `Installed(at)`: the last apply put a release into the code volume, and when it ended: a rollback puts the code back
  with the settings (`code_back`). None once an apply that put none in and kept the settings before it ends (its
  rollback puts those settings back alone), and once the code is back (a rollback, or an apply's put-back). An apply
  that put none in and kept none (the confirmed change delivered again: state 9) leaves it as it was: the
  settings kept are still those from before the apply it records.
- `Records(confirmed, under_way, put_back, doctor, written, first_tick, installed)`: each None while none is recorded.
- `read(package_dir, carried=False) -> Records`: the records, every one None while there is no FILE; checked first
  as every reader of a private file checks it (private_files.checked: Refused, with the fix), or with CARRIED as an
  update's carry leaves FILE, at 600: its permissions are passed over (an update's dry run reads them so), and one
  you cannot read fails to open (an OSError). `Unreadable` (a ValueError) when FILE is no host records this package
  reads: not JSON, another kind, a newer layout (SCHEMA_VERSION), a record that lacks a field (but UnderWay's
  installed, baseline, failing, code_back, kept and first_apply, which may be missing) or has one of another type, or
  a command or a reason none of COMMANDS or REASONS. A field the record has besides its own is passed over.
  Unreadable's message ends with its fix, `fix`: FILE moved aside (private_files.aside), then your settings confirmed
  again (the confirm step's command).
- `confirm(package_dir, confirmed)`: CONFIRMED as the last confirm, its settings file as messages show it; a confirm
  of the fingerprint already confirmed keeps the time it was first confirmed (the delivered layer carries that time:
  deploy_plan).
- `put_back(package_dir, record)` and `clear_put_back(package_dir)`: RECORD as the change put back; none, once an
  apply made the confirmed settings run.
- `doctor_result(package_dir, doctor)`: the last doctor result.
- `change_written(package_dir, change)`: the fingerprints kept, made CHANGE(those kept before, NOT_KEPT while none
  are): deliver.wrote's, as each file is written.
- `first_tick(package_dir, first)`: FIRST, a FirstTick, unless one is kept already: the first one stays, and FILE is
  not written again. `clear_first_tick(package_dir)`: none, once the deployment's volumes are removed: its next start
  is a first start again; records that keep none are not written, nor made in a folder that has none.
- `begin(package_dir, command, failing=None, code_back=False, first_apply=False) -> Lease`: COMMAND, one of COMMANDS,
  recorded as under way, with the doctor's baseline, FAILING (the names of the checks it found failing before the
  change; None when it did not run), CODE_BACK as it is given (a rollback's: `code_back`), and FIRST_APPLY, the mark of
  a first apply over an older package, while the lease holds UNDER_WAY_LOCK, a lock file beside FILE (flock: the
  kernel lets it go when the process ends, however it ends); `Busy` when another process holds it after BUSY_WAIT
  seconds of trying (`runs` holds it a moment, to look). `Lease.finish(code=None)`: the command ended (done, or nothing
  changed), and the record goes, with CODE in the same write: True, the apply put a release in (Installed, now); False,
  it put none in, or the code is back (none); None, Installed as it was. `Lease.applied()`: an apply ended well, finish
  with CODE by what its change did (see Installed): True when a release went in (`installed`), else False when its
  delivery kept the settings before it (`kept`), else None. `Lease.release()`, as leaving `with begin(...) as lease:`
  does, lets the lock go and leaves the record: an apply that stopped part-way (state 3). A lease taken while the record
  names a command that stopped part-way records COMMAND in its place, and carries on its `installed`, `kept` and
  `first_apply`, and its baseline when it has one (`baseline`); `PartWay`, nothing written, when COMMAND does not take
  that command's place (`takes_place`). `Lease.failing`: the baseline the command judges the doctor after the change
  against (None: none). `Lease.installed`: whether a release went into the code volume during the change, by this
  command or by the one that stopped part-way before it; `Lease.record_install()`: one did, and the record says so.
  `Lease.kept`: whether the delivery of the change kept the settings in place before it, by this command or by the one
  that stopped part-way before it; `Lease.record_kept()`: it does, and the record says so before the delivery writes.
  `Lease.part_way`: whether the lease took the place of such a record, the change part-way when this command began.
  `Lease.stop(changed)`: the command ends before it is done: as finish() when nothing changed (CHANGED false) and the
  change was not part-way when it began; else as release(), the change part-way still (state 3). Whether the record
  stays.
- `Busy(command, where)`: an apply, rollback or adopt that runs, its `command`, in the package folder WHERE names
  (this one by default).
- `PartWay(command)`: the records keep COMMAND, its `command`, as stopped part-way, whose place the command that
  began does not take (begin).
- `takes_place(command, part_way) -> bool`: whether COMMAND takes the place of PART_WAY, the command that stopped
  part-way: the same command, or an apply and a rollback, one for the other (CHANGES, the changes of the settings and
  the code); an adopt finishes an adopt's alone, and only an adopt finishes one.
- `runs(package_dir) -> bool`: whether the command the records say is under way still runs: a process holds its lock.
- `baseline(part_way, failing) -> tuple | None`: the doctor's baseline a command judges the doctor after its change
  against (apply's steps 1 and 7): that of the change that stopped part-way, PART_WAY (its UnderWay, or None), when its
  record has one, so that a failure that change caused never counts as failing already; else FAILING, the names of
  the checks the command's own doctor found failing (None: it did not run).
- `code_back(records) -> bool`: whether a rollback, or the put-back of an apply, puts the code back with the settings,
  by the RECORDS: after a rollback that stopped part-way, as it decided (UnderWay.code_back), so that the
  command again ends where that one would have; while an apply is under way (one that stopped part-way, or the one
  whose put-back asks), whether a release went in during it (UnderWay.installed), or, when its delivery kept no
  settings (UnderWay.kept), whether the last apply put one in (Installed), since the settings kept are those from
  before that apply; else (none under way, or an adopt) whether the last apply put one in (Installed).
- `first_apply(records) -> bool`: whether the change under way, or the failed change the RECORDS keep, is a first
  apply over an older package, by the records alone: the record of the change
  under way has the mark (UnderWay.first_apply), or the change put back is a first apply that failed (FIRST_APPLY).
  The one home of the question: apply goes its first apply's way while it is true, and rollback has nothing to do.
- `timestamp(when=None)` and `parsed(text)`: a time as the records keep it, TIME_FORMAT in UTC (now, by default), and
  back as an aware datetime (None for text in no such form).
- FOLDER, FILE: the records' folder in the package folder, the package's own folder of private files
  (private_files.RECORDS_FOLDER), and their file in it (private_files.PACKAGE_FILES lists FILE).
  RECORDS_LOCK, UNDER_WAY_LOCK: the two lock files beside FILE, empty, 600: one for each change of FILE (a read, the
  change and the write, so that two commands never lose each other's), one for the lease.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import dataclasses
import datetime
import fcntl
import json
import os
import pathlib
import time
from typing import Callable, Optional

from . import private_files, settings_file, steps
from .checks import FIX_LABEL, Command, Fix, Steps, fix_from_json, fix_to_json

FOLDER = private_files.RECORDS_FOLDER
FILE = f"{FOLDER}/records.json"
RECORDS_LOCK = f"{FOLDER}/records.lock"
UNDER_WAY_LOCK = f"{FOLDER}/under_way.lock"
KIND = "host_records"
SCHEMA_VERSION = 1
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
COMMANDS = ("apply", "rollback", "adopt")
# the changes of the settings and the code: one takes the other's place (takes_place)
CHANGES = ("apply", "rollback")
# why a confirmed change is in state 10 (PutBack.reason): put back by a rollback, or after a failed restart, load or
# doctor check; or FIRST_APPLY, the first apply over an older package, which failed and stopped the new container
FIRST_APPLY = "first apply"
REASONS = ("rollback", "restart", "load", "doctor", FIRST_APPLY)
BUSY_WAIT = 1.0


@dataclasses.dataclass(frozen=True)
class Image:
    """An image as the settings name it: see the module's docstring."""

    registry: str
    name: str
    tag: str

    @property
    def reference(self) -> str:
        """NAME:TAG, as the image is tagged on this host."""
        return f"{self.name}:{self.tag}"


@dataclasses.dataclass(frozen=True)
class Confirmed:
    """The last confirm: see the module's docstring."""

    settings_file: str
    fingerprint: str
    at: str
    file_sha256: str
    package_version: str
    code_sha256: str
    container_name: str
    image: Image
    changes: int


@dataclasses.dataclass(frozen=True)
class UnderWay:
    """An apply, rollback or adopt under way: see the module's docstring."""

    command: str
    since: str
    installed: bool = False
    baseline: bool = False
    failing: tuple = ()
    code_back: bool = False
    kept: bool = False
    first_apply: bool = False


@dataclasses.dataclass(frozen=True)
class PutBack:
    """A confirmed change that was put back: see the module's docstring."""

    fingerprint: str
    reason: str
    at: str
    failure: str
    fix: Optional[Fix]


@dataclasses.dataclass(frozen=True)
class Doctor:
    """The last doctor result: see the module's docstring."""

    at: str
    checks: int
    failing: tuple


@dataclasses.dataclass(frozen=True)
class Written:
    """The fingerprint of each file deliver last wrote: see the module's docstring."""

    env: str
    override: str
    layer: str
    kept_layer: str


NOT_KEPT = Written("", "", "", "")


@dataclasses.dataclass(frozen=True)
class FirstTick:
    """When deploy step 8 first passed after a start: see the module's docstring."""

    at: str


@dataclasses.dataclass(frozen=True)
class Installed:
    """The last apply put a release in: see the module's docstring."""

    at: str


@dataclasses.dataclass(frozen=True)
class Records:
    """The host's records: see the module's docstring."""

    confirmed: Optional[Confirmed] = None
    under_way: Optional[UnderWay] = None
    put_back: Optional[PutBack] = None
    doctor: Optional[Doctor] = None
    written: Optional[Written] = None
    first_tick: Optional[FirstTick] = None
    installed: Optional[Installed] = None


SECTIONS = {
    "confirmed": Confirmed,
    "under_way": UnderWay,
    "put_back": PutBack,
    "doctor": Doctor,
    "written": Written,
    "first_tick": FirstTick,
    "installed": Installed,
}
# each field's type in JSON, by its annotation; PutBack.fix is checks.fix_to_json's form, and a block a mapping
_JSON_TYPES = {"str": str, "int": int, "bool": bool, "tuple": list}
_BLOCKS = {"Image": Image}
# the fields that hold one of a few words
_WORDS = {(UnderWay, "command"): COMMANDS, (PutBack, "reason"): REASONS}


class Unreadable(ValueError):
    """The records file at PATH is no host records this package reads, WHY: see the module's docstring."""

    def __init__(self, path, why: str):
        shown = settings_file.shown(path)
        move = Command(("mv", shown, settings_file.shown(private_files.aside(path))))
        self.fix = Steps(
            (move, ", then", steps.command_line(steps.DEPLOY_STEPS[2].command))
        )
        super().__init__(
            f"{shown} is no host records this package reads: {why} ({FIX_LABEL}{self.fix})"
        )


def _with_article(command: str) -> str:
    """COMMAND after its article: "an apply", "a rollback"."""
    return f"an {command}" if command[:1] in "aeiou" else f"a {command}"


class Busy(RuntimeError):
    """Another process holds the lease: an apply, rollback or adopt runs, in the package folder WHERE names."""

    def __init__(self, command: str, where: str = "this package folder"):
        self.command = command
        super().__init__(f"{_with_article(command)} runs in {where} already")


class PartWay(RuntimeError):
    """The records keep COMMAND as stopped part-way, whose place the command that began does not take: see the
    module's docstring."""

    def __init__(self, command: str):
        self.command = command
        super().__init__(
            f"{_with_article(command)} stopped part-way in this package folder"
        )


def timestamp(when: datetime.datetime | None = None) -> str:
    """WHEN (now by default) as the records keep a time."""
    when = when or datetime.datetime.now(datetime.timezone.utc)
    return when.astimezone(datetime.timezone.utc).strftime(TIME_FORMAT)


def parsed(text: str) -> datetime.datetime | None:
    """A time the records keep, as an aware datetime; None for text in no such form."""
    try:
        when = datetime.datetime.strptime(str(text), TIME_FORMAT)
    except ValueError:
        return None
    return when.replace(tzinfo=datetime.timezone.utc)


def _to_json(records: Records) -> dict:
    out = {"kind": KIND, "schema_version": SCHEMA_VERSION}
    for name in SECTIONS:
        record = getattr(records, name)
        if record is None:
            out[name] = None
            continue
        fields = {}
        for f in dataclasses.fields(record):
            value = getattr(record, f.name)
            if f.name == "fix":
                value = fix_to_json(value)
            elif isinstance(value, tuple):
                value = list(value)
            elif dataclasses.is_dataclass(value):
                value = dataclasses.asdict(value)
            fields[f.name] = value
        out[name] = fields
    return out


def _section(cls, data, name: str, path):
    """The record CLS from DATA, the section NAME of the records file at PATH; None for none."""
    if data is None:
        return None
    if not isinstance(data, dict):
        raise Unreadable(path, f"its {name} is no mapping")
    values = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            if f.default is dataclasses.MISSING:
                raise Unreadable(path, f"its {name} has no {f.name}")
            values[f.name] = f.default
            continue
        value = data[f.name]
        if f.name == "fix":
            try:
                values[f.name] = fix_from_json(value)
            except (KeyError, TypeError, ValueError, AttributeError):
                raise Unreadable(path, f"its {name}.fix is no fix") from None
            continue
        if f.type in _BLOCKS:
            if not isinstance(value, dict):
                raise Unreadable(path, f"its {name}.{f.name} is no mapping")
            values[f.name] = _section(_BLOCKS[f.type], value, f"{name}.{f.name}", path)
            continue
        kind = _JSON_TYPES[f.type]
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise Unreadable(path, f"its {name}.{f.name} is no {f.type}")
        words = _WORDS.get((cls, f.name))
        if words and value not in words:
            raise Unreadable(path, f"its {name}.{f.name} is none of {', '.join(words)}")
        values[f.name] = tuple(value) if kind is list else value
    return cls(**values)


def _from_json(data, path) -> Records:
    if not isinstance(data, dict) or data.get("kind") != KIND:
        raise Unreadable(path, f'its kind is not "{KIND}"')
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise Unreadable(
            path,
            f"its layout is {version!r}, and this package reads layout {SCHEMA_VERSION}",
        )
    return Records(
        **{
            name: _section(cls, data.get(name), name, path)
            for name, cls in SECTIONS.items()
        }
    )


def _path(package_dir) -> pathlib.Path:
    return pathlib.Path(package_dir) / FILE


def read(package_dir, carried=False) -> Records:
    """The records: see the module's docstring."""
    path = _path(package_dir)
    if not os.path.lexists(path):
        return Records()
    p = private_files.checked(path, package_dir, carried=carried)
    fd = os.open(p, os.O_RDONLY | private_files.NOFOLLOW)
    with os.fdopen(fd, "rb") as f:
        data = f.read()
    try:
        loaded = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise Unreadable(path, "it is not JSON") from None
    return _from_json(loaded, path)


def _lock_file(package_dir, rel: str, create: bool):
    """The lock file REL's descriptor, made (empty, 600) when CREATE; None when it is not there."""
    path = pathlib.Path(package_dir) / rel
    if create:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_RDWR | os.O_CREAT
    else:
        flags = os.O_RDONLY
    try:
        return os.open(path, flags | private_files.NOFOLLOW, 0o600)
    except FileNotFoundError:
        return None


def _update(package_dir, change: Callable[[Records], Records]) -> Records:
    """CHANGE made to the records, under RECORDS_LOCK: FILE read (set to 600 first, when it is looser), changed and
    written; the records it made."""
    fd = _lock_file(package_dir, RECORDS_LOCK, create=True)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        private_files.tighten(_path(package_dir))
        records = change(read(package_dir))
        text = json.dumps(_to_json(records), indent=2, ensure_ascii=False) + "\n"
        private_files.write(_path(package_dir), text, package_dir, replace=True)
        return records
    finally:
        os.close(fd)


def confirm(package_dir, confirmed: Confirmed) -> Records:
    """CONFIRMED as the last confirm: see the module's docstring."""
    shown = settings_file.shown(confirmed.settings_file)

    def change(r: Records) -> Records:
        earlier = r.confirmed
        same = earlier is not None and earlier.fingerprint == confirmed.fingerprint
        at = earlier.at if same else confirmed.at
        new = dataclasses.replace(confirmed, settings_file=shown, at=at)
        return dataclasses.replace(r, confirmed=new)

    return _update(package_dir, change)


def put_back(package_dir, record: PutBack) -> Records:
    """RECORD as the change that was put back."""
    if record.reason not in REASONS:
        raise ValueError(
            f"a put back's reason is one of {', '.join(REASONS)}, not {record.reason!r}"
        )
    return _update(package_dir, lambda r: dataclasses.replace(r, put_back=record))


def clear_put_back(package_dir) -> Records:
    """No change put back: an apply made the confirmed settings run."""
    return _update(package_dir, lambda r: dataclasses.replace(r, put_back=None))


def doctor_result(package_dir, doctor: Doctor) -> Records:
    """DOCTOR as the last doctor result."""
    return _update(package_dir, lambda r: dataclasses.replace(r, doctor=doctor))


def change_written(package_dir, change: Callable[[Written], Written]) -> Records:
    """The fingerprints kept, made CHANGE(those kept before): see the module's docstring."""
    return _update(
        package_dir,
        lambda r: dataclasses.replace(r, written=change(r.written or NOT_KEPT)),
    )


def first_tick(package_dir, first: FirstTick) -> Records:
    """FIRST, unless a first tick is kept already: see the module's docstring."""
    kept = read(package_dir)
    if kept.first_tick is not None:
        return kept

    def change(r: Records) -> Records:
        if r.first_tick is not None:
            return r
        return dataclasses.replace(r, first_tick=first)

    return _update(package_dir, change)


def clear_first_tick(package_dir) -> Records:
    """No first tick kept: the deployment's volumes are removed. Records that keep none are not written."""
    kept = read(package_dir)
    if kept.first_tick is None:
        return kept
    return _update(package_dir, lambda r: dataclasses.replace(r, first_tick=None))


class Lease:
    """An apply, rollback or adopt under way, while this process holds its lock: see the module's docstring."""

    def __init__(self, package_dir, command: str, fd: int):
        self.package_dir, self.command, self._fd = package_dir, command, fd
        self.installed = self.kept = self.part_way = self.first_apply = False
        self.failing = None

    def finish(self, code: bool | None = None) -> None:
        """The command ended: the record goes, with CODE (see the module's docstring), then the lock."""
        if self._fd is None:
            raise RuntimeError("the lease was let go before the command finished")

        def change(r: Records) -> Records:
            r = dataclasses.replace(r, under_way=None)
            if code is None:
                return r
            return dataclasses.replace(
                r, installed=Installed(timestamp()) if code else None
            )

        _update(self.package_dir, change)
        self.release()

    def applied(self) -> None:
        """An apply ended well: finish, with CODE by what its change did (see the module's docstring)."""
        self.finish(code=True if self.installed else (False if self.kept else None))

    def record_install(self) -> None:
        """A release went into the code volume during the change: the record says so (UnderWay.installed)."""
        if self._fd is None:
            raise RuntimeError("the lease was let go before the install was recorded")

        def change(r: Records) -> Records:
            under_way = r.under_way or UnderWay(self.command, timestamp())
            return dataclasses.replace(
                r, under_way=dataclasses.replace(under_way, installed=True)
            )

        _update(self.package_dir, change)
        self.installed = True

    def record_kept(self) -> None:
        """The delivery of the change keeps the settings in place before it: the record says so (UnderWay.kept)."""
        if self._fd is None:
            raise RuntimeError("the lease was let go before the keep was recorded")

        def change(r: Records) -> Records:
            under_way = r.under_way or UnderWay(self.command, timestamp())
            return dataclasses.replace(
                r, under_way=dataclasses.replace(under_way, kept=True)
            )

        _update(self.package_dir, change)
        self.kept = True

    def stop(self, changed: bool) -> bool:
        """The command ends before it is done: see the module's docstring. Whether the record stays."""
        if changed or self.part_way:
            self.release()
            return True
        self.finish()
        return False

    def release(self) -> None:
        """The lock goes; the record stays, unless finish() cleared it."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> Lease:
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _taken(fd: int) -> bool:
    """Whether this process took the lock FD, trying for BUSY_WAIT seconds."""
    deadline = time.monotonic() + BUSY_WAIT
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)


def begin(
    package_dir,
    command: str,
    failing=None,
    code_back: bool = False,
    first_apply: bool = False,
) -> Lease:
    """COMMAND recorded as under way, under a lease: see the module's docstring."""
    if command not in COMMANDS:
        raise ValueError(f"under way is one of {', '.join(COMMANDS)}, not {command!r}")
    fd = _lock_file(package_dir, UNDER_WAY_LOCK, create=True)
    if not _taken(fd):
        os.close(fd)
        under_way = read(package_dir).under_way
        raise Busy(under_way.command if under_way else command)
    lease = Lease(package_dir, command, fd)

    def change(r: Records) -> Records:
        # a record left here is that of a command that stopped part-way: COMMAND takes its place when it may, and its
        # install, its keep, its first apply's mark and its baseline carry on
        if r.under_way is not None and not takes_place(command, r.under_way.command):
            raise PartWay(r.under_way.command)
        lease.part_way = r.under_way is not None
        lease.installed = lease.part_way and r.under_way.installed
        lease.kept = lease.part_way and r.under_way.kept
        lease.first_apply = bool(first_apply) or (
            lease.part_way and r.under_way.first_apply
        )
        lease.failing = baseline(r.under_way, failing)
        record = UnderWay(
            command,
            timestamp(),
            lease.installed,
            lease.failing is not None,
            lease.failing or (),
            bool(code_back),
            lease.kept,
            lease.first_apply,
        )
        return dataclasses.replace(r, under_way=record)

    try:
        _update(package_dir, change)
    except BaseException:
        lease.release()
        raise
    return lease


def takes_place(command: str, part_way: str) -> bool:
    """Whether COMMAND takes the place of PART_WAY, the command that stopped part-way: see the module's docstring."""
    return command == part_way or {command, part_way} <= set(CHANGES)


def baseline(part_way: Optional[UnderWay], failing) -> Optional[tuple]:
    """The doctor's baseline a command judges the doctor after its change against: see the module's docstring."""
    if part_way is not None and part_way.baseline:
        return part_way.failing
    return None if failing is None else tuple(failing)


def code_back(records: Records) -> bool:
    """Whether a rollback puts the code back with the settings, by the RECORDS: see the module's docstring."""
    part_way = records.under_way
    if part_way is None or part_way.command not in CHANGES:
        return records.installed is not None
    if part_way.command == "rollback":
        return part_way.code_back
    return part_way.installed or (not part_way.kept and records.installed is not None)


def first_apply(records: Records) -> bool:
    """Whether the change under way, or the failed change kept, is a first apply over an older package: see the
    module's docstring."""
    if records.under_way is not None:
        return records.under_way.first_apply
    return records.put_back is not None and records.put_back.reason == FIRST_APPLY


def runs(package_dir) -> bool:
    """Whether a process holds the lease's lock."""
    fd = _lock_file(package_dir, UNDER_WAY_LOCK, create=False)
    if fd is None:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
