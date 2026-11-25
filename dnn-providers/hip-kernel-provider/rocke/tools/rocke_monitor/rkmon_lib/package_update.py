# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rkmon package update`: the package folder FOLDER replaced by a newer package, every local
file of its docker/ folder brought along, and the folder it replaced kept beside it as FOLDER.prev, the way back.

- `update(folder, source=None, *, dry_run=False, named=False, hold=None) -> Done`: FOLDER, a package folder
  (paths.is_package; resolved, so that a link to it goes on pointing at it), replaced by the package in SOURCE, a
  folder or a tarball the package build made, or, with no SOURCE, by the one fetched (`fetch`) from the repository
  and branch that the updates section of your settings file names (`updates_of`), at its GitHub address
  (derived.github_address). NAMED: the command named FOLDER (`--folder`), so each update a fix or a next line names
  carries `--folder FOLDER` and updates FOLDER too, and each command after it is named by FOLDER's own rkmon
  (`rkmon_of`), which runs it there. In order:
  1. ASIDE beside FOLDER, which a way back that stopped part-way left, is refused, its fix the rename that puts it
     where it belongs; FOLDER is a package (when it is gone and FOLDER.prev has one, as when an update's renames
     stopped after FOLDER became FOLDER.prev, the refusal's fix puts that in its place); what an earlier update left
     beside it goes: STAGE, FETCHED, UNPACKED, and OLD, which is FOLDER.prev again when that is missing (the renames
     stopped after their first);
  2. with SOURCE the folder FOLDER.prev, a package: the way back. The two folders change places by three renames
     through ASIDE, each with its own local files, so the earlier package comes back exactly as it was;
  3. FOLDER passes the place rule of the package's own private files (private_files.carry's, checked on FOLDER: carry
     checks it again on STAGE, beside FOLDER, whose fix would name STAGE);
  4. git is on this host: the update reads with it which files a package includes, and fetches with it. Else REFUSED as
     an earlier step, once the updates section is read when there is no SOURCE;
  5. the new package: SOURCE's, a tarball unpacked into UNPACKED first, or the one fetched into FETCHED. REFUSED when
     nothing is at SOURCE, or it has no package, neither itself nor in its one folder (package_files.package_in);
     for a tarball, when it is none or no whole one (cut short or damaged), and, naming the member, for one with an
     absolute path or `..`, a device, or a member or a link that would land outside UNPACKED (package_files.unpack);
  6. NOTHING TO DO when its code is FOLDER's (`differs`);
  7. STAGE: the files of the new package that a package includes, copied as the package build copies them
     (package_files.copy: what git tracks there or leaves unignored, a link's file, and the blank config.json), so
     that what only a package in use or a clone has stays out: its own private files, a file moved aside, a root
     certificate, a monitor's state, a venv. REFUSED, naming each, as content no package includes (exit 4, not as an
     earlier step): a link to a file outside the new package, which the copy would bring in (before the copy); a
     private file that no package includes in STAGE, by its name or its header (deployment.stray_rows), in a row. Then
     the image row, a WARN, when the files the image keeps its own copies of (code_volume.IMAGE_BOUND) differ between
     FOLDER and STAGE (the first code_volume.MOST of them, then how many more): the new package needs a new image, its
     fix releases.new_image_fix. A dry run stops next,
     PASSED, with the rows of step 8 as they would be (carry's checks run, nothing is set, and only STAGE, which goes,
     gets the other local files, so that the next line reads the code the update would put in FOLDER), the image row
     and its next line (`next_line`);
  8. FOLDER's own private files carried into STAGE (private_files.carry: each at 600, a looser one set to 600 where
     it is too, as its row says); then each other local file of FOLDER's docker/ where STAGE has nothing (`others`),
     as it is. Each other file of FOLDER's docker/ that the update leaves in FOLDER.prev alone (`left`) gets a row
     too, which says where it is;
  9. the renames: an earlier FOLDER.prev to OLD, FOLDER to FOLDER.prev, STAGE to FOLDER; then OLD goes.
  FETCHED, UNPACKED and STAGE go whatever the outcome. A failure or an interrupt renames back what was renamed, so
  nothing changes but a looser private file set to 600; a rename back that fails is an ERROR whose fix is that rename.
  A dry run leaves nothing past step 1. Returns Done: UPDATED, with the PASS row of each file carried, naming its
  place in FOLDER, and of each file left, naming its place in FOLDER.prev, then the image row, and the state's next
  line (`next_line`), which the way back's UPDATED has too; NOTHING TO DO; PASSED for a dry run, the way back's too,
  with a dry run's rows and next line. Raises Stopped for a refusal or a failure, and lets private_files.Refused,
  settings_file.Refused and records.Unreadable through as they come; nothing changed.
  HOLD, a function of one folder (rkmon's holds that folder's lock exclusive until the command ends:
  rkmon_lib/package_lock.py; None: no lock is held), is called with each folder beside FOLDER that the update moves
  or makes, before it moves it, so that, with FOLDER's own, which the caller holds, the update holds the lock of
  every folder it moves or makes from before its first move until its last line. It is called with
  FOLDER.prev once step 1 is done (an update moves it to OLD, which goes; the way back puts it in FOLDER's place),
  and with STAGE as step 7 makes it, STAGE given a records' folder (700) first when FOLDER has one, so that STAGE
  has a lock, which moves with it into FOLDER's place: from the renames on, FOLDER's lock is the update's until it
  ends, its clean-up and next line included. A dry run calls it too, so that it refuses as the update would. What
  HOLD raises, as for a hold refused, goes through as it comes; nothing changed.
- `next_line(folder, named=False) -> str`: the next line of the state that FOLDER is in once an update has put another
  package in it (state 4 when your file is in another layout, else 5; 2 when none is confirmed), as the state
  table gives it (states.next_line), from the host's records of FOLDER and the settings file they name, for the
  package now in FOLDER (state_facts.gather, with FOLDER as its PACKAGE, since the code that runs the update is often
  the package it replaced, and a DOCKER that asks nothing). As nothing is asked of the container, nothing has passed
  since a start (states.NOT_STARTED), so a command that is a deploy step's comes with the step's number and name
  (`next: step 3, confirm: rkmon confirm`). `next: rkmon next` when those facts do not decide the state alone
  (states.RECORDS_ONLY), or cannot be read: the package says no layout of its deploy file (state_facts.NoLayout), or
  the records break a private file's rules, are none this package reads or do not open. A dry run's next line runs the
  update, then names the command that the update's own next line would, without the step's words: the one for the
  package it would put in FOLDER, with FOLDER's records as its carry leaves them (state_facts.gather's CARRIED), or,
  for the way back, for FOLDER.prev with its own (`next: rkmon package update --from ~/new, then rkmon migrate
  --settings ~/rkmon-deploy.yaml`), so that it says which case this is. Records you cannot read now the update
  reads once its carry has set them to 600: the dry run's line says so, then names `rkmon next`. NAMED, as after an
  update that named FOLDER (`--folder`): FOLDER's own rkmon names the command (`rkmon_of`, program.run_by), since it
  runs there, and so does a dry run's line, after the update it runs from here.
- `rkmon_of(folder) -> str`: FOLDER's own rkmon, as a command names it: its path, as a message shows it.
- `interrupted(folder, before) -> (text, done)`: what an update that an interrupt (Ctrl-C) stopped changed, the text
  of its outcome, FOLDER's os.stat before the update being BEFORE (None: no folder was there, and the update changed
  nothing). Until its renames are done, the update renames back what it renamed: nothing changed; FOLDER gone, as
  when a second interrupt stops that, the update run again names the rename that puts it back. Once they are done,
  FOLDER is another folder than BEFORE's: only the clean-up after them stopped, which the next update finishes, and
  DONE is true.
- `updates_of(folder, named=None) -> (repository, branch)`: the updates section of the settings file that the host's
  records of FOLDER keep (the last confirm's), read as it is now, after the checks every reader of a private file
  makes. Stopped, REFUSED as an earlier step (exit 3), when no settings file is confirmed, the file is gone or it has
  no updates section, its fix both ways: `--from` (with `--folder NAMED` when the command named FOLDER so), or the
  section (`rkmon template section updates` prints it); REFUSED, with the section's own FAIL rows, when the section
  is not filled in.
- `fetch(url, branch, into) -> path`: the package's folder of the newest commit of BRANCH in the repository at URL,
  cloned by git into the new folder INTO, shallow and sparse: that commit alone, and of its files only the package
  folder's (git's cone mode adds the few files that lie directly in the folders above it, never another folder's).
  That folder is the one with a package in it, as the commit's list of files shows (package_files.package_folders),
  read before any file is: the update names no folder of the repository. git asks for no password: a repository
  that needs a login takes git logged in to it. FetchFailed, with git's last line, when git fails (the network, no
  such branch, no login), or with why when the commit has no package, or more than one, which it names.
- `differs(folder, new)`: whether the package NEW has other code than the package folder FOLDER: a file of NEW's
  that FOLDER lacks or has otherwise (other bytes, or whether it runs: code_volume.runs; a link by its file, as the
  copy takes it), or one of FOLDER's that NEW lacks, but for a local file of its docker/ (`local`). The files of each
  are the ones the copy of a package copies (package_files.copied): what a package includes, but config.json, which
  it writes blank.
- `local(rel)`: whether REL, a path in a package folder, may be a local file: it lies in docker/, and is none of the
  package's own kinds of file there (a script, the image's recipe, the wrapper), so one the new package dropped stays
  dropped. `others(folder, new)`: the local files of FOLDER where NEW (an update's STAGE) has nothing, but for what no
  package includes (package_files.excluded: a backup a write keeps, a compiled file, a cache) and NOT_PACKAGE (carry
  takes the package's own private files, and the host's records' folder has nothing else to keep), sorted.
  `left(folder, new)`: the files of FOLDER's docker/ that an update to NEW (its STAGE) neither carries nor takes as
  they are, but for what `others` passes over too, sorted: they stay in FOLDER.prev alone, until the next update
  removes it. A local file where NEW has another file, or a folder; a script, the image's recipe or the wrapper that
  NEW lacks, which the new package dropped or which is yours. The update cannot tell those two apart, so it gives each
  a PASS row that says where it is.
- `Done(word, text, rows=(), next="")`: an update that ended well: its outcome word, what happened, the rows of the
  files carried and left, and the image row, and its next line (`next_line`: an update's, a way back's or a dry
  run's); "" for the caller's own.
- `Stopped`: outcome.Stopped, for an update refused (REFUSED) or failed (ERROR), with the FAIL rows that say why;
  nothing changed, unless its text says what did.
- `FetchFailed`: a fetch that gave no package; its message says why.
- PREVIOUS, STAGE, FETCHED, UNPACKED, ASIDE and OLD: the endings of the folders beside FOLDER that take its name.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import contextlib
import dataclasses
import filecmp
import os
import pathlib
import shutil
import subprocess

from . import (
    code_volume,
    deployment,
    derived,
    file_types,
    migrate,
    outcome,
    package_files,
    paths,
    private_files,
    program,
    records,
    releases,
    settings_file,
    state_facts,
    states,
    steps,
)
from .checks import Check, Command, Status, Steps
from .from_layout1 import UPDATES
from .outcome import Stopped

PREVIOUS = ".prev"  # FOLDER.prev: the folder an update replaced, the way back
STAGE = ".update-new"  # the new package folder, until it takes FOLDER's place
FETCHED = ".update-fetch"  # git's clone of the repository
UNPACKED = ".update-unpack"  # the tarball SOURCE, unpacked
ASIDE = ".update-aside"  # FOLDER, while the way back's renames run
OLD = ".update-old"  # the earlier FOLDER.prev, while an update's renames run
# What only a package in use has, never a package: the host's records' folder and the package's own private files.
NOT_PACKAGE = (records.FOLDER, *private_files.PACKAGE_FILES)
# The folder of a package folder whose local files an update carries.
LOCAL_FOLDER = "docker/"
# The package's own kinds of file in docker/, never a local file: its scripts, the image's recipe and the wrapper.
OWN_ENDINGS = (".sh", ".py")
OWN_NAMES = ("Dockerfile", "rkmon")
IMAGE_ROW = "image"  # the name of the row that says the new package needs a new image
# whose the container name is, as next_line's facts read it: none of identity's words, since it asks docker nothing
NOT_ASKED = "NOT ASKED"
_shown = settings_file.shown  # a path as a message shows it


@dataclasses.dataclass(frozen=True)
class Done:
    """An update that ended well: see the module's docstring."""

    word: str
    text: str
    rows: tuple = ()
    next: str = ""


class FetchFailed(Exception):
    """A fetch that gave no package: see the module's docstring."""


class _Unasked:
    """state_facts' DOCKER for next_line, which asks docker nothing: whose the container name is reads as NOT_ASKED,
    so that gather reads the records and the settings file alone."""

    @staticmethod
    def owner(project) -> str:
        return NOT_ASKED


def _named(command: Command, named) -> Command:
    """COMMAND, an update, with `--folder NAMED` after it when the command named its folder NAMED (as shown), so that
    it updates that folder too; else as it is."""
    return command if named is None else Command((*command.argv, "--folder", named))


def _from(named=None) -> Command:
    """The update from a package folder, as a fix names it, of the folder NAMED (`_named`)."""
    return _named(steps.command_line(settings_file.PACKAGE_UPDATE), named)


def _source_fix(named=None) -> Steps:
    """The fix for a SOURCE that gives no package: the update from one, of the folder NAMED (`_named`)."""
    return Steps(
        (
            _from(named),
            ", DIR being a package folder or a tarball the package build made",
        )
    )


def _again(source, named=None) -> Command:
    """The update run again, as a fix names it: from SOURCE, or fetched when there is none; of the folder NAMED
    (`_named`)."""
    if source is None:
        again = steps.command_line(steps.command_name(settings_file.PACKAGE_UPDATE))
    else:
        again = steps.command_line(settings_file.PACKAGE_UPDATE, DIR=_shown(source))
    return _named(again, named)


def _rel(at: str, name: str) -> str:
    """The path of NAME in the folder AT of a package folder ("." for its top), its parts joined by /."""
    return name if at == "." else f"{at}/{name}"


def _skipped(rel: str) -> bool:
    """Whether REL, a path in a package folder, is what `others` passes over: what no package includes
    (package_files.excluded), or NOT_PACKAGE."""
    return package_files.excluded(rel) or rel in NOT_PACKAGE


def _tree(top) -> dict:
    """{REL: its path} for each file and link in the folder TOP, REL its path there, but for what `others` passes
    over (`_skipped`); a link to a folder is one entry, not a folder walked."""
    top = pathlib.Path(top)
    out = {}
    for here, folders, files in os.walk(top):
        base = pathlib.Path(here)
        at = base.relative_to(top).as_posix()
        walked = []
        for name in folders:
            rel = _rel(at, name)
            if _skipped(rel):
                continue
            if (base / name).is_symlink():
                out[rel] = base / name
            else:
                walked.append(name)
        folders[:] = walked
        for name in files:
            rel, p = _rel(at, name), base / name
            if not _skipped(rel) and (p.is_symlink() or p.is_file()):
                out[rel] = p
    return out


def _files(top) -> list:
    """The files of the folder TOP that a package includes (package_files.tracked), git pointed at no other repository;
    OSError, with git's last line, when git fails."""
    try:
        return package_files.tracked(top, env=package_files.git_env())
    except subprocess.CalledProcessError as exc:
        lines = (exc.stderr or "").strip().splitlines()
        why = lines[-1] if lines else f"git exited with {exc.returncode}"
        raise OSError(f"git could not list the files of {_shown(top)}: {why}") from None


def _code(top) -> dict:
    """{REL: its path} for each file of the folder TOP that the copy of a package copies (package_files.copied): what
    a package includes (`_files`), but config.json, which it writes blank."""
    return package_files.copied(top, _files(top))


def _same(a: pathlib.Path, b: pathlib.Path) -> bool:
    """Whether the files A and B are the same as a package has them: the same bytes, which their package runs both
    or neither (code_volume.runs); a link by its file, as the package build copies it.
    """
    runs = {code_volume.runs(os.stat(p).st_mode) for p in (a, b)}
    return len(runs) == 1 and filecmp.cmp(a, b, shallow=False)


def local(rel: str) -> bool:
    """Whether REL may be a local file: see the module's docstring."""
    name = rel.rsplit("/", 1)[-1]
    return rel.startswith(LOCAL_FOLDER) and not (
        name.endswith(OWN_ENDINGS) or name in OWN_NAMES
    )


def differs(folder, new) -> bool:
    """Whether the package NEW has other code than FOLDER: see the module's docstring."""
    old, fresh = _code(folder), _code(new)
    for rel, path in fresh.items():
        if rel not in old or not _same(old[rel], path):
            return True
    return any(not local(rel) for rel in old.keys() - fresh.keys())


def others(folder, new) -> list:
    """The local files of FOLDER where NEW has nothing: see the module's docstring."""
    new = pathlib.Path(new)
    return sorted(
        rel for rel in _tree(folder) if local(rel) and not os.path.lexists(new / rel)
    )


def _as_it_is(path: pathlib.Path, there: pathlib.Path) -> bool:
    """Whether THERE, a path in a new package, is a file the same as FOLDER's file PATH (`_same`); False when PATH
    is a link that points nowhere."""
    try:
        return there.is_file() and _same(path, there)
    except OSError:
        return False


def left(folder, new) -> list:
    """The files of FOLDER's docker/ that an update to NEW leaves in FOLDER.prev alone: see the module's docstring."""
    new = pathlib.Path(new)
    out = []
    for rel, path in _tree(folder).items():
        there = new / rel
        if local(rel):
            if os.path.lexists(there) and not _as_it_is(path, there):
                out.append(rel)
        elif rel.startswith(LOCAL_FOLDER) and not os.path.lexists(there):
            out.append(rel)
    return sorted(out)


def _git(env: dict, *args: str) -> str:
    """What git prints for ARGS, run with ENV and no input; FetchFailed, with git's last line, when it fails."""
    p = subprocess.run(
        ["git", *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if p.returncode:
        lines = p.stderr.decode("utf-8", "replace").strip().splitlines()
        raise FetchFailed(lines[-1] if lines else f"git exited with {p.returncode}")
    return p.stdout.decode("utf-8", "surrogateescape")


def fetch(url: str, branch: str, into) -> pathlib.Path:
    """The package's folder of the newest commit of BRANCH at URL, cloned into INTO: see the module's docstring."""
    into = pathlib.Path(into)
    clone = ("clone", "--quiet", "--depth=1", "--filter=blob:none", "--sparse")
    try:
        env = package_files.git_env()
        _git(env, *clone, "--no-tags", f"--branch={branch}", "--", url, str(into))
        files = _git(env, "-C", str(into), "ls-tree", "-r", "-z", "--name-only", "HEAD")
        found = package_files.package_folders(files.split("\0"))
        if not found:
            raise FetchFailed("it has no package")
        if len(found) > 1:
            shown = [f or "its top folder" for f in found]
            raise FetchFailed(
                f"it has a package in each of {', '.join(shown[:-1])} and {shown[-1]}"
            )
        folder = found[0]
        # cone mode checks out a folder; at the top, the package is the whole tree
        only = ("set", folder) if folder else ("disable",)
        _git(env, "-C", str(into), "sparse-checkout", *only)
    except OSError as exc:
        raise FetchFailed(f"git: {exc.strerror or exc}") from None
    new = into / folder
    if not paths.is_package(new):
        raise FetchFailed("it has no package")
    return new


def _no_source(why: str, where: str = "", named=None) -> Stopped:
    """The refusal of an update with no --from and no updates section to fetch from (WHY), naming both ways, the
    first of the folder NAMED (`_named`)."""
    section = steps.command_line(file_types.DEPLOY.section_command, KEY=UPDATES)
    within = where or "your settings file"
    fix = Steps(
        (
            _from(named),
            f", or an {UPDATES} section in {within} (",
            section,
            "prints it)",
        )
    )
    text = f"there is no --from, and {why}; nothing changed"
    return Stopped("REFUSED", text, fix=fix, earlier_step=True)


def updates_of(folder, named=None) -> tuple:
    """(repository, branch), from the updates section of your settings file: see the module's docstring."""
    confirmed = records.read(folder).confirmed
    if confirmed is None:
        raise _no_source(
            f"no settings file is confirmed in {_shown(folder)}", named=named
        )
    where = confirmed.settings_file
    path = os.path.expanduser(where)
    if not os.path.lexists(path):
        raise _no_source(
            f"{where}, the settings file confirmed here, is gone", named=named
        )
    private_files.checked(path, folder, migrate.NAME, shown=where)
    doc = settings_file.read(path, file_types.DEPLOY)
    section = doc.values.get(UPDATES)
    if section is None:
        raise _no_source(f"{where} has no {UPDATES} section", where, named)
    rows = [
        r
        for r in settings_file.validate(doc)
        if r.status == Status.FAIL and r.name.split(".")[0] == UPDATES
    ]
    if rows:
        text = f"the {UPDATES} section of {where} is not filled in; {outcome.FIX_EACH_FAIL}"
        raise Stopped("REFUSED", text, rows=rows)
    return section["repository"], section["branch"]


def _clear(path) -> None:
    """What is at PATH removed: a folder with all that is in it, or a file or a link (never what a link points to)."""
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.unlink(path)


def _why(exc: OSError) -> str:
    """What failed, as a message says it: the file and the system's words; for a copy, the first file it failed on."""
    copied = exc.args[0] if isinstance(exc, shutil.Error) and exc.args else None
    if isinstance(copied, list) and copied:
        src, _dst, why = copied[0]
        return f"{_shown(src)}: {why}"
    if exc.filename is not None and exc.strerror:
        return f"{_shown(exc.filename)}: {exc.strerror}"
    return str(exc)


def _renames(pairs) -> None:
    """Each (FROM, TO) of PAIRS renamed, in order, each TO free before its rename. A failure or an interrupt renames
    back those done (FROM gone and TO there), the last first, then goes on; Stopped (ERROR) when one cannot be renamed
    back, whose fix is that rename."""
    begun = []
    try:
        for src, dst in pairs:
            begun.append((src, dst))
            os.rename(src, dst)
    except BaseException:
        for src, dst in reversed(begun):
            if os.path.lexists(src) or not os.path.lexists(dst):
                continue  # not renamed: it failed, or the interrupt came first
            try:
                os.rename(dst, src)
            except OSError as exc:
                text = f"{_shown(dst)} could not be renamed back to {_shown(src)}: {exc.strerror}"
                fix = Command(("mv", _shown(dst), _shown(src)))
                raise Stopped("ERROR", text, fix=fix) from None
        raise


def _aside_left(folder, prev, aside) -> Stopped:
    """The refusal while ASIDE, which a way back that stopped part-way left, is there: its fix renames it to the one of
    FOLDER and PREV that is missing, or else out of the way."""
    gone = next((p for p in (folder, prev) if not os.path.lexists(p)), None)
    target = gone or private_files.aside(aside)
    move = Command(("mv", _shown(aside), _shown(target)))
    then = (
        "puts it back"
        if gone
        else "moves it out of the way: look at what is in it before you remove it"
    )
    text = f"{_shown(aside)} is left from a way back that stopped part-way; nothing changed"
    return Stopped("REFUSED", text, fix=Steps((move, then)))


def _not_package(folder, prev) -> Stopped:
    """The refusal while FOLDER has no package: when FOLDER is gone and PREV has a package, its fix puts that in
    FOLDER's place; else it names the folder to update."""
    if not os.path.lexists(folder) and paths.is_package(prev):
        text = f"{_shown(folder)} is gone, and {_shown(prev)} has a package; nothing changed"
        move = Command(("mv", _shown(prev), _shown(folder)))
        return Stopped("REFUSED", text, fix=Steps((move, "puts it in its place")))
    text = (
        f"{_shown(folder)} has no package: no rkmon.py and rkmon_lib/; nothing changed"
    )
    again = steps.command_line(f"{settings_file.PACKAGE_UPDATE} --folder FOLDER")
    fix = Steps((again, ", FOLDER being the package folder to update"))
    return Stopped("REFUSED", text, fix=fix)


def _leftovers(folder, prev, made, old) -> None:
    """What an earlier update left beside FOLDER, gone: the folders an update MADE (STAGE, FETCHED and UNPACKED), and
    OLD, which is PREV again when that is missing."""
    try:
        if os.path.lexists(old) and not os.path.lexists(prev):
            os.rename(old, prev)
        for p in (*made, old):
            _clear(p)
    except OSError as exc:
        text = (
            f"what an earlier update left beside {_shown(folder)} could not be cleared away: {_why(exc)}; nothing "
            "changed"
        )
        raise Stopped("ERROR", text) from None


def _back(folder, prev, aside, dry_run: bool, named=None) -> Done:
    """The way back: FOLDER and PREV change places, through ASIDE. With NAMED, the folder as the command named it, the
    update a fix or a dry run's line names carries `--folder NAMED`, and FOLDER's own rkmon names each command after
    it (`rkmon_of`)."""
    by = None if named is None else rkmon_of(folder)
    both = f"{_shown(folder)} and {_shown(prev)}"
    if not paths.is_package(prev):
        text = (
            f"{_shown(prev)} has no package, so there is no way back; nothing changed"
        )
        fix = Steps((_from(named), ", DIR being an earlier package"))
        raise Stopped("REFUSED", text, fix=fix)
    if dry_run:
        return Done(
            "PASSED",
            f"{both} can change places",
            next=_dry_next(prev, prev, prev, named=named, by=by),
        )
    try:
        _renames(((folder, aside), (prev, folder), (aside, prev)))
    except OSError as exc:
        text = f"{both} could not change places: {_why(exc)}; nothing changed"
        raise Stopped("ERROR", text) from None
    text = f"{both} changed places, each with its own local files"
    return Done("UPDATED", text, next=next_line(folder, named is not None))


def _need_git(source, words: str, named=None) -> None:
    """Stopped, REFUSED as an earlier step, when git is not on this host: the update reads with it which files of the
    new package (WORDS) a package includes, and with no SOURCE fetches it too. Its fix runs it again, of the folder
    NAMED."""
    if shutil.which("git") is not None:
        return
    does = (
        f"fetches {words} and reads which of its files a package includes"
        if source is None
        else f"reads which files of {words} a package includes"
    )
    text = f"git, which {does}, is not on this host; nothing changed"
    fix = Steps(("install git, then run", _again(source, named)))
    raise Stopped("REFUSED", text, fix=fix, earlier_step=True)


def _unpack(source: pathlib.Path, into: pathlib.Path, named=None) -> None:
    """The tarball SOURCE unpacked into the new folder INTO (package_files.unpack); Stopped, REFUSED, when it is not a
    whole tarball, or has a member that is not unpacked, its fix of the folder NAMED.
    """
    try:
        with open(source, "rb") as f:
            package_files.unpack(f, into)
    except package_files.Unsafe as exc:
        text = f"{_shown(source)} has {exc.member}, which {exc.why}; nothing changed"
        raise Stopped("REFUSED", text, fix=_source_fix(named)) from None
    except package_files.UNWHOLE:
        text = f"{_shown(source)} is not a tarball, or not a whole one; nothing changed"
        raise Stopped("REFUSED", text, fix=_source_fix(named)) from None


def _new(folder, source, fetched, unpacked, named=None) -> tuple:
    """(the new package, the words that say where it comes from): SOURCE's, a folder or a tarball unpacked into
    UNPACKED, or the one fetched into FETCHED; git on this host first (`_need_git`). Its fixes of the folder NAMED.
    """
    if source is None:
        repository, branch = updates_of(folder, named)
        words = f"the newest commit of {branch} in {repository}"
        _need_git(source, words, named)
        try:
            url = derived.github_address(repository)
            return fetch(url, branch, fetched), words
        except FetchFailed as exc:
            text = f"the fetch of {words} failed: {exc}; nothing changed"
            fix = Steps(("update from a package folder instead:", _from(named)))
            raise Stopped("ERROR", text, fix=fix) from None
    _need_git(source, _shown(source), named)
    if not os.path.exists(source):
        text = f"there is no folder or file at {_shown(source)}; nothing changed"
        raise Stopped("REFUSED", text, fix=_source_fix(named))
    top = source
    if not source.is_dir():
        _unpack(source, unpacked, named)
        top = unpacked
    new = package_files.package_in(top)
    if new is None:
        text = (
            f"{_shown(source)} has no package: neither it nor its one folder has rkmon.py and rkmon_lib/; "
            "nothing changed"
        )
        raise Stopped("REFUSED", text, fix=_source_fix(named))
    return new, _shown(new if top is source else source)


def _staged(
    new: pathlib.Path, stage: pathlib.Path, words: str, source, named=None
) -> None:
    """STAGE made: the files of the package NEW that a package includes, as the package build copies them. Stopped,
    REFUSED, naming each, when NEW holds a link to a file outside it (before the copy), or STAGE a private file that
    no package includes (in a row); its fixes run it again, of the folder NAMED."""
    code, top = _code(new), os.path.realpath(new)
    away = [
        rel
        for rel, f in code.items()
        if f.is_symlink() and not package_files.within(f, top)
    ]
    if away:
        text = (
            f"the package from {words} has links to files outside it, which the copy would bring in: "
            f"{', '.join(away)}; nothing changed"
        )
        then = f"take each out of {words}, or put its file in its place, then run"
        fix = Steps((then, _again(source, named)))
        raise Stopped("REFUSED", text, fix=fix)
    package_files.copy(new, stage, files=list(code.values()))
    rows = deployment.stray_rows(stage, _again(source, named), source=words)
    if rows:
        text = f"the package from {words} has private files, which no package includes; {outcome.FIX_EACH_FAIL}"
        raise Stopped("REFUSED", text, rows=rows)


def _image_rows(folder, stage, words: str) -> list:
    """The image row, when the package from WORDS, copied to STAGE, needs a new image: the files the image keeps its
    own copies of (code_volume.IMAGE_BOUND) differ between it and FOLDER; else none."""
    bound = code_volume.image_changes(folder, stage)
    if not bound:
        return []
    found = (
        f"the package from {words} needs a new image: it changes files the image keeps its own copies of: "
        f"{code_volume.listed(bound)}"
    )
    return [Check(IMAGE_ROW, Status.WARN, found, releases.new_image_fix())]


def _what_next(folder, package, carried=False, by=None) -> tuple:
    """(the next line of FOLDER's state, the command it names) once FOLDER has the package in the folder PACKAGE,
    with FOLDER's records, as the update's carry leaves them with CARRIED, named by the rkmon BY (program.run_by;
    None: the one that runs): see `next_line` in the module's docstring."""
    with program.run_by(by):
        try:
            facts = state_facts.gather(
                folder, _Unasked(), package=package, carried=carried
            )
        except (
            state_facts.NoLayout,
            records.Unreadable,
            private_files.Refused,
            OSError,
        ):
            facts = None
        state = None if facts is None else states.current(facts)
        if state is None or state.n not in states.RECORDS_ONLY:
            command = str(steps.command_line("next"))
            return f"{steps.NEXT_LABEL}{command}", command
        line = states.next_line(state, facts, states.NOT_STARTED)
        return line, str(state.next_command(facts))


def rkmon_of(folder) -> str:
    """FOLDER's own rkmon, as a command names it: see the module's docstring."""
    return _shown(pathlib.Path(folder) / "docker" / "rkmon")


def next_line(folder, named: bool = False) -> str:
    """The next line of FOLDER's state after an update: see the module's docstring."""
    by = rkmon_of(folder) if named else None
    return _what_next(folder, folder, by=by)[0]


def _dry_next(source, folder, package, carried=False, named=None, by=None) -> str:
    """A dry run's next line: the update from SOURCE, of the folder NAMED (`_named`), then the command that the
    update's own next line would name, by the rkmon BY, FOLDER holding the package in PACKAGE, with its records as
    the update's carry leaves them with CARRIED: see `next_line` in the module's docstring.
    """
    again = str(_again(source, named))
    held = folder / records.FILE
    if carried and os.path.isfile(held) and not os.access(held, os.R_OK):
        again += f", which sets {_shown(held)} to 600 first"
    then = _what_next(folder, package, carried, by)[1]
    return f"{steps.NEXT_LABEL}{again}, then {then}"


def interrupted(folder, before) -> tuple:
    """What an update that an interrupt stopped changed: see the module's docstring."""
    if before is None:
        return "nothing changed", False
    try:
        now = os.stat(folder)
    except OSError:
        return f"the update stopped part-way: {_shown(folder)} is not there", False
    if os.path.samestat(before, now):
        return "nothing changed", False
    return (
        "the update is done; only its clean-up stopped, which the next update finishes",
        True,
    )


def _carried(folder: pathlib.Path, stage: pathlib.Path, dry_run=False) -> list:
    """FOLDER's local files carried into STAGE: its own private files, then each other local file of its docker/
    where STAGE has nothing; the PASS row of each, naming its place in FOLDER, which STAGE becomes. With DRY_RUN,
    carry's checks alone for the private files, nothing set, and each row says what would be; the other local files
    are copied all the same, into STAGE alone, which goes, so that the dry run's next line reads the code the update
    would put in FOLDER (image.code_sha256: a local file the package's .dockerignore lets in is code too).
    """
    rows = [
        Check(
            r.name,
            r.status,
            r.found.replace(str(stage / r.name), str(folder / r.name), 1),
        )
        for r in private_files.carry(folder, stage, dry_run=dry_run)
    ]
    for rel in others(folder, stage):
        (stage / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / rel, stage / rel, follow_symlinks=False)
        found = (
            f"{'would be carried' if dry_run else 'carried'} to {folder / rel} as it is"
        )
        rows.append(Check(rel, Status.PASS, found))
    return rows


def _left_rows(
    folder: pathlib.Path, stage: pathlib.Path, prev: pathlib.Path, dry_run=False
) -> list:
    """The PASS row of each file of FOLDER's docker/ that the update to STAGE leaves in PREV alone (`left`),
    naming its place there and why. With DRY_RUN, each row says what would be."""
    rows = []
    for rel in left(folder, stage):
        why = (
            "the new package has its own, which takes its place"
            if local(rel)
            else "the new package has none, and an update never carries a script, the image's recipe or the wrapper"
        )
        found = f"{'would be left' if dry_run else 'left'} in {prev / rel}, which the next update removes: {why}"
        rows.append(Check(rel, Status.PASS, found))
    return rows


def update(
    folder, source=None, *, dry_run: bool = False, named: bool = False, hold=None
) -> Done:
    """FOLDER replaced by a newer package: see the module's docstring."""
    folder = private_files.real(folder)
    shown = _shown(folder) if named else None
    by = rkmon_of(folder) if named else None
    source = None if source is None else private_files.real(source)
    prev, stage, fetched, unpacked, aside, old = (
        folder.parent / (folder.name + ending)
        for ending in (PREVIOUS, STAGE, FETCHED, UNPACKED, ASIDE, OLD)
    )
    if os.path.lexists(aside):
        raise _aside_left(folder, prev, aside)
    if not paths.is_package(folder):
        raise _not_package(folder, prev)
    _leftovers(folder, prev, (stage, fetched, unpacked), old)
    # an update moves FOLDER.prev to OLD, which goes; the way back puts it in FOLDER's place
    if hold is not None:
        hold(prev)
    if source == prev:
        return _back(folder, prev, aside, dry_run, shown)
    # The place rule alone, its row naming FOLDER (carry checks it again on STAGE, beside FOLDER).
    private_files.carry(folder, folder, names=())
    try:
        new, words = _new(folder, source, fetched, unpacked, shown)
        if not differs(folder, new):
            text = f"{_shown(folder)} has the package from {words} already"
            return Done("NOTHING TO DO", text)
        _staged(new, stage, words, source, shown)
        if hold is not None:  # STAGE's lock, which moves with it into FOLDER's place
            if os.path.isdir(folder / records.FOLDER):
                (stage / records.FOLDER).mkdir(mode=0o700, parents=True, exist_ok=True)
            hold(stage)
        image = _image_rows(folder, stage, words)
        left_rows = _left_rows(folder, stage, prev, dry_run)
        rows = _carried(folder, stage, dry_run) + left_rows + image
        if dry_run:
            text = f"the package from {words} can replace {_shown(folder)}"
            line = _dry_next(source, folder, stage, carried=True, named=shown, by=by)
            return Done("PASSED", text, tuple(rows), line)
        earlier = [(prev, old)] if os.path.lexists(prev) else []
        _renames([*earlier, (folder, prev), (stage, folder)])
    except OSError as exc:
        text = f"the update of {_shown(folder)} stopped: {_why(exc)}; nothing changed"
        raise Stopped("ERROR", text) from None
    finally:
        for p in (fetched, unpacked, stage):
            with contextlib.suppress(OSError):
                _clear(p)
    with contextlib.suppress(OSError):
        _clear(old)
    text = f"{_shown(folder)} has the package from {words}; the one it replaced is {_shown(prev)}"
    return Done("UPDATED", text, tuple(rows), next_line(folder, named))
