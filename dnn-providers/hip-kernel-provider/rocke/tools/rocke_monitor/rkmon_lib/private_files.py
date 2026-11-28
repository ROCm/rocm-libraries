# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Private files: the files that may hold a private value (a settings file, a file a secret is read from, the
package's own .env, override and records), checked and written one way, by every reader and every writer.

A private file is a regular file, not a link, owned by you, that only you can read: permissions 600 (nothing for the
group and others, and readable by you, so 400 passes too; a write makes it exactly 600). Your own private files lie
outside every folder that carries its files off: the package folder (into the image's build context and the
package's commits), a git work tree (into its commits) and a cloud-synced folder (OneDrive, Dropbox, iCloud
Drive, Google Drive, Box: the sync service copies it off this machine). The package's own private files
(`PACKAGE_FILES`, by their place under the package folder: docker/.env and docker/compose.override.yaml, their .prev
copies, and the host's records, records.json in the package's own folder of private files, `RECORDS_FOLDER`,
docker/.rkmon) have fixed places in the package folder instead: their one place rule is that the package folder lies
outside cloud-synced folders, and the ignore lists keep them out of git and out of the image. Any other path in the
package folder is refused. Links are resolved before a place is judged, so a link cannot hide where a file lies.

Every row's fix is a command, or steps of commands (rkmon_lib/checks.py). Where the fix is to name another file, it is
the command you ran, naming that file: RETRY, which the caller passes, is a function from a path to that command (or to
the edit that names the file); `again` makes one from the command's words. A RETRY that names only another folder, never
another file (a secret's file, whose name is fixed), is marked so (`folder_only`): in a link's row it names nothing, and
the link alone is removed. FILE stands for a path only you know. A suggested path is a free one: ~/NAME in your home
folder, NAME.old beside the file, or the name with -2, -3, ... before its suffix when that is taken.

A row shows the file's path as SHOWN, the caller's way of showing it (`~/rkmon-deploy.yaml`, as the command's own
lines show the file), when the caller gives one: the file's folder as SHOWN's, and a path beside the file (NAME.old,
a free name) beside SHOWN; any other path, and every path without SHOWN, absolute.

- `problems(path, package_dir, name=..., retry=None, shown=None, carried=False)`: the checks of the file at PATH as
  Check rows named NAME: a FAIL row for each rule it breaks, each with its fix, or one PASS row when it breaks none. A
  missing file, a link and anything that is no regular file get that one row alone; an owner, permissions and a
  place are each checked. A file in one of the three folders is moved to your home folder, then named again (RETRY).
  Without RETRY, the fixes find the file (ls) or move it, and the words after them say to name it. A package's own
  file in a cloud-synced package folder: the package folder moves to your home folder. With CARRIED, the checks of
  the file as `carry` leaves it, at 600: its permissions are passed over (an update's dry run reads its records so).
- `checked(path, package_dir, name=..., retry=None, shown=None, carried=False)`: the one check a reader makes before
  it reads a private file: PATH, absolute, when `problems` (with CARRIED) finds no FAIL; else Refused, with those
  rows.
- `tighten(path)`: the permissions rule's fix, for a writer that keeps its own private files: a regular file of yours
  whose permissions break the rule is set to 600, and True says it was; any other file is left for `problems` to say.
- `write(path, text, package_dir, replace=False, retry=..., dry_run=False, shown=None)`: TEXT (UTF-8, line ends as
  they are) as the file PATH: a new file beside it, created at 600 (O_CREAT|O_EXCL), written and synced, then put at
  PATH, and its folder synced; a missing folder is made (the last one at 700). With `replace` the new file is renamed
  onto PATH. Without, PATH becomes its second name (a hard link), which never replaces a file, so a file another
  writer makes at PATH meanwhile stays; where the file system makes no hard link, PATH itself is made O_CREAT|O_EXCL
  and written. PATH comes back absolute. Refused, with nothing written, when PATH breaks its place rule (the fix:
  RETRY with a path in your home folder; for a package's own file, the package folder moved), is a link or no
  regular file, or exists while `replace` is False (RETRY with a free name, or the file moved aside). A TypeError
  without RETRY, but for a package's own file: its place is fixed, so its fixes name no other file. With `dry_run` (a
  command's --dry-run, which runs every guard and changes nothing), it stops once PATH passes those checks: nothing
  is written, no folder is made, and PATH comes back absolute all the same.
- `carry(src_dir, dst_dir, names=PACKAGE_FILES, dry_run=False)`: the package's own private files NAMES that the
  package folder SRC_DIR has, to their places in the package folder DST_DIR (a package update), each written as
  `write` writes it, never over a file: a PASS row for each file carried. Each is a regular file, not a link, and
  yours, and DST_DIR lies outside cloud-synced folders. Other permissions are no refusal: the file is set to 600
  where it is, its copy is at 600, and its row says what they were. All or nothing: refused, with nothing written,
  when a file breaks a rule or DST_DIR has it already (the fix: move that one aside), and a failure part-way removes
  the copies made and the folders made for them. A ValueError for a name not in PACKAGE_FILES. With DRY_RUN (the
  update's --dry-run), the checks alone: nothing is set or written, and each row says what the carry would do.
- `aside(path)`: where the file at PATH moves out of the way, beside it: NAME.old, or the first free name like it
  (with -2, -3, ... before .old); absolute. A refusal's fix offers it, and so may a writer that keeps the file that
  was there.
- `again(argv, given="", flag="--settings")`: a RETRY: the command ARGV (its program first, as you ran it) with the
  path in the place of GIVEN, the file you named as you typed it: its last word that is GIVEN or FLAG=GIVEN; when no
  word is, FLAG and the path after them.
- `folder_only(retry)`: RETRY, marked as one that names only another folder (secrets_registry.retry): a link's row
  then says to remove the link alone and run the command again.
- `real(path)`: PATH resolved, as a place is judged: `~` expanded, absolute, every link followed.
- `place_problem(path, package_dir)`: why a file or a folder at PATH may not hold private files, as "it is inside
  <folder>: <what carries it off>" for the first of the three folders that holds it; "" when none does.
- `git_top(path)`: the top folder of the git work tree PATH lies in, or "".
- `own_names(name)`: NAME, the name of one of the package's own private files (no folder in it), and its kin beside
  it, as globs: NAME, its copies (NAME.*), and the new file a write of it that stopped part-way left
  (whole_file.NEW_FILE), which holds what NAME would. The ignore lists hold them wherever they are
  (secrets_registry.ignore_patterns).
- `own_places()`: where the package's own private files and their kin lie in the package folder, as globs relative
  to it: each file of PACKAGE_FILES with its kin (`own_names`), then RECORDS_FOLDER, whose every file is the
  package's own.
- `in_own_places(rel)`: whether REL, a path relative to the package folder (with /), lies in one of `own_places`,
  RECORDS_FOLDER's files included: the scan for stray private files leaves it alone in a package folder in use
  (deployment.stray_private_files).
- `Refused`: a refused read, write or carry (a ValueError); its `rows` are the FAIL rows that say why, each with its
  fix. Its message is the one way a refusal is shown: those rows as `checks.render(rows, "problems")` prints them,
  then RULES, once: the line that names where README.md says what a private file is and where it may lie.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import contextlib
import fnmatch
import os
import pathlib
import pwd
import re
import stat

from . import whole_file
from .checks import Check, Command, Status, Steps, render

# The folders a sync service copies off this machine, by the name the service gives its folder: the whole name, or how
# it starts (an organization's OneDrive folder adds a dash and the organization's name; "Dropbox (Personal)",
# "GoogleDrive-<account>"). macOS keeps them under CloudStorage, and iCloud Drive under Mobile Documents and
# com~apple~CloudDocs; Google Drive's own folder is My Drive.
SYNCED = re.compile(
    r"(onedrive|dropbox|icloud ?drive|google ?drive|box sync|box-box).*|box|mobile documents|com~apple~clouddocs|"
    r"my drive|cloudstorage",
    re.I,
)
ROW = "private file"  # a row's name when the caller gives none
FILE = "FILE"  # a fix's stand-in for a path only you know
# The line a refusal ends with, once, after its rows: where README.md says what a private file is and where it may
# lie.
RULES = 'What a private file is and where it may lie: README.md, "Where things live".'
# The package's own folder of private files, by its place under the package folder: the host's records. Every file in
# it is the package's own.
RECORDS_FOLDER = "docker/.rkmon"
# The package's own private files, by their place under the package folder: the two files rkmon writes for docker
# compose (.env and the override), the copies it keeps of them, and the host's records.
PACKAGE_FILES = (
    "docker/.env",
    "docker/compose.override.yaml",
    "docker/.env.prev",
    "docker/compose.override.yaml.prev",
    f"{RECORDS_FOLDER}/records.json",
)


class Refused(ValueError):
    """A file refused, nothing read or written: `rows` are the FAIL rows that say why, each with its fix; the message
    is the rows as checks.render prints them, then RULES."""

    def __init__(self, rows: list):
        self.rows = list(rows)
        super().__init__(f'{render(self.rows, "problems")}\n{RULES}')


def _path(path) -> pathlib.Path:
    return pathlib.Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def real(path) -> pathlib.Path:
    """PATH resolved: see the module's docstring."""
    return pathlib.Path(os.path.realpath(os.path.expanduser(os.fspath(path))))


def _as_is(path):
    return path


def _show(p: pathlib.Path, shown):
    """How the rows about the file at P show a path (the module's docstring): P as SHOWN, P's folder as SHOWN's, a
    path beside P beside SHOWN, and any other path as it is; with SHOWN None, every path as it is.
    """
    if shown is None:
        return _as_is
    shown = str(shown)
    folder = os.path.dirname(shown)
    if folder == "~":
        folder = "~/"  # the home folder as the shell expands it: a command keeps a leading ~/ as it is, not ~ alone

    def show(path):
        q = pathlib.Path(path)
        if q == p:
            return shown
        if q == p.parent:
            return folder or "."
        if q.parent == p.parent:
            return os.path.join(folder, q.name)
        return path

    return show


def git_top(path) -> str:
    """The top folder of the git work tree PATH lies in: the nearest folder with a .git entry (a folder, or a
    worktree's file), PATH's own when it is a folder, links resolved; "" when none. PATH need not exist.
    """
    p = real(path)
    for d in [p, *p.parents] if p.is_dir() else list(p.parents):
        if (d / ".git").exists():
            return str(d)
    return ""


def own_names(name: str) -> tuple:
    """NAME, one of the package's own private files' names, and its kin beside it, as globs (the module's docstring)."""
    return (name, f"{name}.*", whole_file.NEW_FILE.format(name=name, unique="*"))


def own_places() -> list:
    """Where the package's own private files and their kin lie in the package folder, as globs relative to it (the
    module's docstring)."""
    out = []
    for rel in PACKAGE_FILES:
        folder, _, name = rel.rpartition("/")
        if folder != RECORDS_FOLDER:
            out += [f"{folder}/{glob}" for glob in own_names(name)]
    return out + [RECORDS_FOLDER]


def in_own_places(rel: str) -> bool:
    """Whether REL, a path relative to the package folder, lies in one of own_places (the module's docstring)."""
    rel = rel.strip("/")
    return rel.startswith(f"{RECORDS_FOLDER}/") or any(
        fnmatch.fnmatchcase(rel, glob) for glob in own_places()
    )


def _cloud(p: pathlib.Path):
    """(where the resolved path P lies, what carries it off from there) when a cloud-synced folder holds it, or
    None."""
    for i, part in enumerate(p.parts):
        if SYNCED.fullmatch(part):
            folder = pathlib.Path(*p.parts[: i + 1])
            return (
                f"inside the cloud-synced folder {folder}",
                "its sync service copies it off this machine",
            )
    return None


def _place(path, package_dir):
    """(where PATH lies, what carries it off from there) for the first of the three folders that holds it, or None."""
    p = real(path)
    try:
        p.relative_to(real(package_dir))
        return (
            "inside the package folder",
            "the image's build context and a commit would carry it",
        )
    except ValueError:
        pass
    top = git_top(p)
    if top:
        return f"inside the git work tree {top}", "a commit there would publish it"
    return _cloud(p)


def _own(p: pathlib.Path, package_dir) -> str:
    """P's place under the package folder when P is one of the package's own private files, else "". P's folder is
    resolved, P itself is not: a link there breaks the file rules."""
    try:
        rel = (real(p.parent) / p.name).relative_to(real(package_dir)).as_posix()
    except ValueError:
        return ""
    return rel if rel in PACKAGE_FILES else ""


def _package_rows(p: pathlib.Path, package_dir, name: str, show=_as_is) -> list:
    """The FAIL row of the one place rule of the package's own files, P being one of them, or the package folder:
    the package folder lies outside cloud-synced folders. The fix moves the package folder to your home folder.
    """
    pkg = real(package_dir)
    cloud = _cloud(pkg)
    if not cloud:
        return []
    move = Command(("mv", pkg, _home(pkg.name)))
    fix = Steps((move, ", then use the package folder from there"))
    return [Check(name, Status.FAIL, f"{show(p)} is {cloud[0]}: {cloud[1]}", fix)]


def place_problem(path, package_dir) -> str:
    """Why a file or a folder at PATH may not hold private files ("it is inside <folder>: <what carries it off>"), or
    "" when none of the package folder, a git work tree and a cloud-synced folder holds it.
    """
    place = _place(path, package_dir)
    return f"it is {place[0]}: {place[1]}" if place else ""


def _user(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def _free(p: pathlib.Path) -> pathlib.Path:
    """P when nothing is there, else the first of P's name with -2, -3, ... before its suffix that names nothing."""
    q, n = p, 2
    while os.path.lexists(q):
        q, n = p.with_name(f"{p.stem}-{n}{p.suffix}"), n + 1
    return q


def aside(path) -> pathlib.Path:
    """Where the file at PATH moves out of the way, beside it: NAME.old when nothing has that name, else the first name
    with -2, -3, ... before .old that names nothing (a link to nothing takes a name too); absolute, `~` expanded.
    """
    p = _path(path)
    return _free(p.with_name(p.name + ".old"))


def _home(name: str) -> str:
    """A place in your home folder for a file named NAME, as ~/NAME (the first free name like it)."""
    return "~/" + _free(pathlib.Path(os.path.expanduser("~")) / name).name


def again(argv, given: str = "", flag: str = "--settings"):
    """A retry: the command ARGV (its program first, as you ran it) with the path in the place of GIVEN, the file you
    named as you typed it (its last word that is GIVEN or FLAG=GIVEN); when no word is, FLAG and the path after them.
    """
    argv = tuple(str(a) for a in argv)

    def retry(path) -> Command:
        words, path = list(argv), str(path)
        for i in range(len(words) - 1, 0, -1):
            if given and words[i] in (given, f"{flag}={given}"):
                words[i] = words[i][: len(words[i]) - len(given)] + path
                break
        else:
            words += [flag, path]
        return Command(words)

    return retry


def folder_only(retry):
    """RETRY, marked as one that names only another folder, never another file (the module's docstring): a link's
    row then names nothing (_link_row). RETRY itself, so it serves as a decorator."""
    retry.folder_only = True
    return retry


def _find(p: pathlib.Path, retry, what: str, show=_as_is):
    """The fix that names the file again: the command you ran, FILE standing for WHAT; without RETRY, a look at what
    is at P (in its folder, when nothing is), then the words that say to name it."""
    if retry:
        return Steps((retry(FILE), f", FILE being {what}"))
    there = p if os.path.lexists(p) else p.parent
    return Steps(
        (Command(("ls", "-la", show(there))), f"shows what is there: name {what}")
    )


def _link_row(p: pathlib.Path, name: str, retry, show=_as_is) -> Check:
    """P is a link: the file it points to is named instead (RETRY); without RETRY, a copy of that file takes the
    link's place (the file itself stays where it is). With a RETRY that names only another folder (folder_only), the
    link alone is removed, and the command run again."""
    target = pathlib.Path(os.path.realpath(p))
    if getattr(retry, "folder_only", False):
        fix = Steps(
            (Command(("rm", show(p))), "(the link alone), then run the command again")
        )
    elif not target.exists():  # a link to nothing
        fix = (
            _find(p, retry, "the file's full path")
            if retry
            else Command(("rm", show(p)))
        )
    elif retry:
        fix = Steps(("name the file it points to:", retry(str(target))))
    else:
        fix = Steps(
            (Command(("rm", show(p))), ", then", Command(("cp", target, show(p))))
        )
    return Check(name, Status.FAIL, f"{show(p)} is a link, not a file", fix)


def _odd_row(p: pathlib.Path, name: str, retry, show=_as_is) -> Check:
    return Check(
        name,
        Status.FAIL,
        f"{show(p)} is not a regular file",
        _find(p, retry, "a regular file's full path", show),
    )


def _taken_row(p: pathlib.Path, name: str, retry, show=_as_is) -> Check:
    """A file is at P, where one is to be written: another file is named (a free name), or that one moves aside."""
    move = Command(("mv", show(p), show(aside(p))))
    fix = (
        Steps((retry(str(show(_free(p)))), ", or move that one aside first:", move))
        if retry
        else move
    )
    return Check(name, Status.FAIL, f"{show(p)} exists", fix)


def _regular_rows(
    p: pathlib.Path, name: str, retry=None, st=None, show=_as_is
) -> tuple:
    """(the FAIL rows of the rules on what is at P: a file, a regular one, not a link, yours; P's stat when it is a
    regular file, else None). ST is P's stat when the caller has it (an open file's), else P's lstat is taken. SHOW
    shows a path in a row (`_show`).
    """
    if st is None:
        try:
            st = os.lstat(p)
        except FileNotFoundError:
            fix = _find(p, retry, "the file's full path", show)
            return [Check(name, Status.FAIL, f"no file at {show(p)}", fix)], None
        except OSError as exc:
            fix = Command(("ls", "-l", show(p)))
            return [Check(name, Status.FAIL, f"{show(p)}: {exc.strerror}", fix)], None
    if stat.S_ISLNK(st.st_mode):
        return [_link_row(p, name, retry, show)], None
    if not stat.S_ISREG(st.st_mode):
        return [_odd_row(p, name, retry, show)], None
    if st.st_uid != os.getuid():
        found = f"{show(p)} belongs to {_user(st.st_uid)}, not to you"
        fix = Command(("sudo", "chown", _user(os.getuid()), show(p)))
        return [Check(name, Status.FAIL, found, fix)], st
    return [], st


def _mode_rows(p: pathlib.Path, name: str, st, show=_as_is) -> list:
    """The FAIL row of the permissions rule, ST being P's stat: only you can read the file, and you can."""
    mode = stat.S_IMODE(st.st_mode)
    wrong = []
    if mode & 0o077:
        wrong.append(
            "others on this machine can read it"
            if mode & 0o044
            else "others on this machine have access to it"
        )
    if not mode & 0o400:
        wrong.append("you cannot read it")
    if not wrong:
        return []
    found = f"{show(p)} has permissions {mode:03o}: {' and '.join(wrong)}"
    return [Check(name, Status.FAIL, found, Command(("chmod", "600", show(p))))]


def problems(
    path, package_dir, name: str = ROW, retry=None, shown=None, carried=False
) -> list:
    """The checks of the private file at PATH, as Check rows named NAME: a FAIL row, with its fix, for each rule it
    breaks (a missing file, a link or no regular file: that row alone; else its owner, its permissions and its place,
    PACKAGE_DIR being the package folder: a package's own file at its place has the package folder's place rule), or
    one PASS row when it breaks none. RETRY names another file, and the rows show PATH as SHOWN (the module's
    docstring). With CARRIED, the checks of the file as `carry` leaves it, at 600: its permissions are passed over.
    """
    p = _path(path)
    show = _show(p, shown)
    out, st = _regular_rows(p, name, retry, show=show)
    if st is None:
        return out
    if not carried:
        out += _mode_rows(p, name, st, show)
    own = _own(p, package_dir)
    place = None if own else _place(p, package_dir)
    if own:
        out += _package_rows(p, package_dir, name, show)
    elif place:
        home = _home(p.name)
        move = Command(("mv", show(p), home))
        fix = (
            Steps((move, ", then", retry(home)))
            if retry
            else Steps((move, ", then name it by its new path"))
        )
        found = f"{show(p)} is {place[0]}: {place[1]}"
        out.append(Check(name, Status.FAIL, found, fix))
    passed = (
        f"{show(p)} is yours only, at its place in the package folder, outside cloud sync"
        if own
        else f"{show(p)} is yours only, outside the package, git and cloud sync"
    )
    return out or [Check(name, Status.PASS, passed)]


def checked(
    path, package_dir, name: str = ROW, retry=None, shown=None, carried=False
) -> pathlib.Path:
    """The one check a reader makes before it reads the private file PATH: PATH, absolute, when it breaks no rule of
    problems() (with CARRIED, the file as `carry` leaves it); else Refused, its rows the FAIL rows, each with its fix,
    showing PATH as SHOWN.
    """
    rows = [
        r
        for r in problems(path, package_dir, name, retry, shown, carried)
        if r.status == Status.FAIL
    ]
    if rows:
        raise Refused(rows)
    return _path(path)


def tighten(path) -> bool:
    """The permissions rule's fix: the private file PATH set to 600 when its permissions break the rule; whether they
    did. Only a regular file of yours, not a link, is set, through the open file when you can read it (a link put
    there meanwhile is not followed); any other is left as it is, for problems() to say.
    """
    p = _path(path)
    rows, st = _regular_rows(p, ROW)
    if rows or not _mode_rows(p, ROW, st):
        return False
    try:
        fd = os.open(p, os.O_RDONLY | NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    except PermissionError:  # you cannot read it: yours, so set by its path
        os.chmod(p, 0o600)
        return True
    except OSError:
        if _regular_rows(p, ROW)[0]:  # changed since it was checked
            return False
        raise
    try:
        if _regular_rows(p, ROW, st=os.fstat(fd))[0]:  # changed since it was checked
            return False
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)
    return True


class _Taken(Exception):
    """A file is at the path that a write that never replaces a file was to take."""


NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)  # a link at the path fails the open


def _create(p: pathlib.Path, data: bytes) -> None:
    """DATA as the new file P, made in its place (O_CREAT|O_EXCL): _Taken when a file is there; P goes again when the
    write fails."""
    try:
        fd = os.open(p, whole_file.FLAGS, 0o600)
    except FileExistsError:
        raise _Taken() from None
    try:
        whole_file.fill(fd, data, 0o600, durable=True)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(p)
        raise


def _put(p: pathlib.Path, data: bytes, replace: bool) -> None:
    """DATA as the file P at 600, then its folder synced. With REPLACE it is written whole (whole_file.write).
    Without, a new file beside P (whole_file.beside), written and synced, becomes P's second name (a hard link), which
    never replaces a file: _Taken when a file is there, whoever made it and whenever; where the file system makes no
    hard link, P is made new (O_CREAT|O_EXCL) and written. The file beside P goes either way, and on a failure P is
    as it was.
    """
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if replace:
        whole_file.write(p, data, 0o600, durable=True)
        return
    tmp = whole_file.beside(p)
    fd = os.open(tmp, whole_file.FLAGS, 0o600)
    try:
        whole_file.fill(fd, data, 0o600, durable=True)
        try:
            os.link(tmp, p)
        except FileExistsError:
            raise _Taken() from None
        except OSError:  # no hard link here
            _create(p, data)
    finally:
        with contextlib.suppress(OSError):  # P, its other name, keeps the file
            os.unlink(tmp)
    whole_file.sync_folder(p.parent)


def write(
    path,
    text: str,
    package_dir,
    replace: bool = False,
    *,
    retry=None,
    dry_run: bool = False,
    shown=None,
) -> pathlib.Path:
    """TEXT as the private file PATH, PACKAGE_DIR being the package folder: a new file at 600 beside it, written,
    synced, then put at PATH (the module's docstring); its absolute path. Refused (nothing written) when PATH breaks
    its place rule (the three folders; for a package's own file, a cloud-synced folder that holds the package folder),
    is a link or no regular file, or exists while REPLACE is False, a file another writer makes there meanwhile
    included (it stays). Each fix names another file through RETRY (the module's docstring), which is required but for
    a package's own file, whose place is fixed: its fixes name no other file. The rows show PATH as SHOWN (the
    module's docstring). With DRY_RUN it returns once those checks pass, before anything is written or made.
    """
    p = _path(path)
    show = _show(p, shown)
    own = _own(p, package_dir)
    if own:
        retry, rows = None, _package_rows(p, package_dir, ROW, show)
    elif retry is None:
        raise TypeError(
            "write() takes retry: a refusal's fix is the command you ran, naming another file"
        )
    else:
        rows, place = [], _place(p, package_dir)
        if place:
            found = f"{show(p)} is {place[0]}: {place[1]}"
            rows.append(Check(ROW, Status.FAIL, found, retry(_home(p.name))))
    if os.path.islink(p):
        rows.append(_link_row(p, ROW, retry, show))
    elif os.path.lexists(p) and not os.path.isfile(p):
        rows.append(_odd_row(p, ROW, retry, show))
    elif os.path.lexists(p) and not replace:
        rows.append(_taken_row(p, ROW, retry, show))
    if rows:
        raise Refused(rows)
    if dry_run:  # every guard has run: nothing is written
        return p
    try:
        _put(p, text.encode("utf-8"), replace)
    except _Taken:  # made since it was checked
        raise Refused([_taken_row(p, ROW, retry, show)]) from None
    return p


def _take(p: pathlib.Path, name: str) -> tuple:
    """(the content of the package's own file P, its permissions as they were), P set to 600: opened without following
    a link or waiting on a pipe, and checked again once open, so a file changed since its check is refused.
    """
    rows, st = _regular_rows(p, name)
    if rows:  # changed since it was checked
        raise Refused(rows)
    mode = stat.S_IMODE(st.st_mode)
    try:
        if not mode & 0o400:  # you cannot read it: yours, so it is set to 600 first
            os.chmod(p, 0o600)
        fd = os.open(p, os.O_RDONLY | NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        rows = _regular_rows(p, name)[0]
        if rows:  # changed since it was checked
            raise Refused(rows) from None
        raise
    with os.fdopen(fd, "rb") as f:
        rows = _regular_rows(p, name, st=os.fstat(f.fileno()))[0]
        if rows:  # changed since it was checked
            raise Refused(rows)
        os.fchmod(f.fileno(), 0o600)
        return f.read(), mode


def carry(src_dir, dst_dir, names=PACKAGE_FILES, dry_run=False) -> list:
    """The package's own private files NAMES (PACKAGE_FILES, or some of them) that the package folder SRC_DIR has, to
    their places in the package folder DST_DIR, each written as write() writes it, never over a file: a PASS row for
    each file carried, in the order of NAMES. Each is a regular file, not a link, and yours, and DST_DIR lies outside
    cloud-synced folders; other permissions are no refusal: the file is set to 600 where it is, and its row says what
    they were. All or nothing: refused (nothing written) when one breaks a rule or DST_DIR has it already, and a
    failure part-way removes the copies made and the folders made for them. A ValueError for a name not in
    PACKAGE_FILES. With DRY_RUN, the checks alone: nothing is set or written, and each row says what would be.
    """
    for n in names:
        if n not in PACKAGE_FILES:
            raise ValueError(
                f"{n!r} is not one of the package's own private files: {', '.join(PACKAGE_FILES)}"
            )
    src, dst = _path(src_dir), _path(dst_dir)
    todo = [n for n in dict.fromkeys(names) if os.path.lexists(src / n)]
    rows = _package_rows(dst, dst, "package folder")
    modes = {}  # each file's permissions as they are
    for n in todo:
        failed, st = _regular_rows(src / n, n)
        rows += failed
        if st is not None:
            modes[n] = stat.S_IMODE(st.st_mode)
        if os.path.lexists(dst / n):
            rows.append(_taken_row(dst / n, n, None))
    if rows:
        raise Refused(rows)
    if dry_run:
        return [_carried_row(n, dst / n, modes[n], dry_run) for n in todo]
    made = []  # (what this carry made, whether it is a folder), in the order made
    out = []
    try:
        for n in todo:
            data, mode = _take(src / n, n)
            p = dst / n
            made += [(d, True) for d in reversed(p.parents) if not os.path.lexists(d)]
            try:
                _put(p, data, replace=False)
            except _Taken:  # made since it was checked
                raise Refused([_taken_row(p, n, None)]) from None
            made.append((p, False))
            out.append(_carried_row(n, p, mode))
    except BaseException:
        for q, folder in reversed(made):  # the copies, then the folders made for them
            with contextlib.suppress(OSError):  # a folder goes only when empty
                (os.rmdir if folder else os.unlink)(q)
        raise
    return out


def _carried_row(name: str, p: pathlib.Path, mode: int, dry_run=False) -> Check:
    """The PASS row of the package's own file NAME carried to P, its permissions MODE before it was; with DRY_RUN,
    what would be."""
    if dry_run:
        found = f"would be carried to {p} at 600"
        if mode != 0o600:
            found += (
                f"; its permissions are {mode:03o}, and would be 600 in both folders"
            )
    else:
        found = f"carried to {p} at 600"
        if mode != 0o600:
            found += (
                f"; its permissions were {mode:03o}, and are 600 in both folders now"
            )
    return Check(name, Status.PASS, found)
