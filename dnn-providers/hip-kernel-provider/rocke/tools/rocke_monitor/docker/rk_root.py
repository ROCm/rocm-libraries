#!/usr/bin/python3 -I
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The container's root steps: what only root may do in the container.

The entry point runs it as root, and only so: in its root phase, at each start of `run` (`start`, before it drops to
the monitor's user) and before an rk command in a one-off container the host hands your confirmed settings
(`hand-layer`), and as `entrypoint.sh ACTION` run as root (RKMON_AS=root: `docker exec --user 0`, or a one-off
container of the image) for the other actions: `code-install` (`rkmon apply`) and `code-back` (`rkmon rollback`). No
sudo rule names it: no other user of the container runs it.

It reads no environment, and its argument is the action (code-install takes one option, below): every path is the
image's (rkmon_lib/container_layout.py and rkmon_lib/code_volume.py, read from the image's code, and the write broker's
check of a code tree, docker/rk_write.py `problems`), and so is the code it loads (the reader of the older Claude
settings, rkmon_lib/claude_env_shape.py, with rkmon_lib/custom_headers.py beside it). The secrets registry's facts that
the one-time move's copies act on are a table here (KEEPERS, COPIES, CLAUDE, KEY_SUFFIX, OLDER_KEY_NAMES, SECRET_NAME,
FIXED_NAMES), which tests/test_root_copies.py holds equal to rkmon_lib/secrets_registry.py, their one home: root acts on
no list that the monitor's user can write. It prints a line for each thing it did or found, for the entry point's log or
the host's command.

- `start`, in this order:
  - the write volume's own folder: owned by the write user, mode 700 (a volume made before the write user existed).
  - the GitHub write login's move, idempotent: a deployment from before the read/write split keeps its one gh login
    in the secrets volume (/secrets/gh). When the write volume holds no login yet and that folder holds one, its files
    are copied into /secrets-write/gh (mode 700, files 600, owned by the write user), and the old folder is removed. A
    folder that is a symbolic link, or that holds anything but regular files, is left as it is and named (nothing
    moves). It never prints a file's content.
  - the one-time move's copies (`copies`, below).
  - whatever those did, the code volume's (`Volume.start`): its folder made root's, mode 755; what a change of the
    code that died left removed; the volume checked as the write broker checks the code it runs (root's and read-only
    all the way, its links CURRENT and KEPT root's, KEPT naming a release or the image's code); and the releases
    nothing uses removed (`Volume.prune`). FAIL, exit 5, with its fix, when the check fails, or cannot be made
    whatever stops it: CURRENT is removed first, so that no release runs (the launcher, the health check's rk with
    it, and the broker run the image's own code).
  The entry point starts the monitor only when this action ends with exit 0.
- `code-install`: the package's code, a tar archive on stdin, installed as a release over this image (`Volume.install`):
  INSTALLED, KEPT naming the code CURRENT named before, a release whatever image it was installed over (the image's
  own code when CURRENT named none, or one that is not whole), so that a rollback to that image runs it again; with
  the option code_volume.LEAVE_KEPT (a start's install: docker/rkmon start_code), KEPT stays as it is, so that a
  rollback still goes back to the code from before the last apply; NOTHING TO DO when that code runs already;
  REFUSED, exit 4, nothing changed, for an archive that does not read or has what a package does not (a link, a
  special file, a name outside its folder, a name twice, a release's record) or a package that is not whole; exit 6
  when it needs a new image, whose fix is the host's to say (rkmon_lib/releases.py new_image_fix). A release's id names
  its package's version, and its code and the image it is installed over by a digest of the two: the same code over
  another image is a release of its own. The archive is unpacked, checked and made a release in a folder of its own
  first, so that what is refused changes nothing. A code volume that fails its check is emptied only then (the image's
  own code runs until the release is in place), but for the one-time move's marks while they pass the check
  themselves, and the lines say so; a step after that which fails says what changed.
- `code-back`: the code KEPT names in CURRENT's place again (`Volume.back`), the last install undone whatever image
  runs now: the launcher runs a release only over the image it was installed over. RESTORED; NOTHING TO DO when no
  code is kept or CURRENT names it already; REFUSED, exit 4, nothing changed, when the release kept is not whole;
  FAIL, exit 5, when the code volume fails its check, CURRENT removed first as at the start.
- `hand-layer`: the layer your confirmed settings make, on stdin, which the host's wrapper hands a one-off container of
  the logins' walk (docker/rkmon; the entry point takes it from container_layout's HANDED_LAYER_VARIABLE), written as
  LAYER_FILE in HANDED_LAYER_DIR, in the container's own file system, where rk reads it in place of the settings
  volume's layer (rkmon_lib/runtime_config.py handed_layer): the folder made root's, group rkmon, mode 750, and the
  file root's, group rkmon, mode 640, put in place whole (a rename), so that the monitor and the writer read it and no
  other user changes it. Nothing printed when done; ERROR, exit 1, when the layer is empty or longer than HANDED_MAX
  bytes, or when it cannot be written. Nothing is written in the folder's place when something else is there (a link,
  a file, another user's folder): rk's read of the layer refuses that, with its fix (rkmon_lib/runtime_config.py
  handed_layer).
Each change of the code volume holds its LOCK; REFUSED, exit 4, while another change holds it. ERROR, exit 1, when a
file cannot be read or written; exit 2 for another action, or a user other than root; exit 3 when the image has no
write user.

The one-time move's copies (`copies`), at each start: each secret an older package kept under another
name, or in the monitor's volume, whose older copy holds a value while its new copy is missing, copied to its new copy,
owned by its keeper's user, mode 600, with a line that names both files, never a value; the older copy stays until the
move ends (`rkmon standby off`). They are the team chat's webhook and the web host's key, into the writer's volume,
byte for byte (COPIES); the claude login's key (CLAUDE): the key alone, when the settings carry the rest of the
older Claude settings (rkmon_lib/claude_env_shape.py), else nothing, and the line names what the settings cannot carry
and the fix, `rkmon login claude`; and each model binding's key that the delivered deploy layer names (a binding's
api_key_secret_name), from the older names `older_key_names` gives it, while review automation is on and one of its
passes runs on such a binding, else the line says why it is not copied: no feature uses it. A new copy is written whole
under a name of its own, then linked in place, never through a link. An older copy that is a link, not a regular file or
larger than COPY_LIMIT is not copied, nor are two older copies of one key that differ: the line gives the fix, the
command that sets the secret. ERROR, exit 1, with the same fix, when a copy cannot be read or written; the code
volume's start runs all the same. Each copy is made once: once a start has dealt with it, whether it made the copy,
found the new copy there or no older copy, found nothing in it or refused it, a mark in the code volume, which only
root writes (MARKS: code_volume.py MOVED, a file named after the new copy's keeper and name), keeps every later start
from making it, so that no file the monitor's user writes later at an older name becomes a new copy, the writer's
least of all; while its new copy is missing, a line names the older copy and the fix. A copy no feature uses, and one
that cannot be read or written, has no mark: a later start deals with it. Marks that another user may write make no
copy at all: ERROR, exit 1. The GPU host's key keeps its name and folder, the atlassian login is never copied, and
GitHub's write login moves as `start` says.

A release is removed only when nothing uses it (`Volume.prune`, at the start and after an install): neither CURRENT
nor KEPT names it, no live process works in it or runs it (its PIN), and the worker tools' values file does not name
its rk, which each PR run starts (code_volume.in_use). Root may not read another user's processes (the container keeps
no SYS_PTRACE), so each user that may run a release scans its own, in a process of its ids (`_scans`): root, the
monitor's user, which reads the values file too, and the write user. A release in use stays, and a line says so; when
a scan cannot be made, no release is removed, and a line says why. The next start or install removes what nothing uses
then.
"""

import contextlib
import datetime as dt
import errno
import fcntl
import grp
import hashlib
import json
import os
import pwd
import re
import runpy
import shutil
import stat
import subprocess
import sys
import tarfile
import types

CODE = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)  # the image's code: read-only, owned by root
LAYOUT = types.SimpleNamespace(
    **runpy.run_path(os.path.join(CODE, "rkmon_lib", "container_layout.py"))
)
VOLUME = types.SimpleNamespace(
    **runpy.run_path(os.path.join(CODE, "rkmon_lib", "code_volume.py"))
)
BROKER = types.SimpleNamespace(
    **runpy.run_path(os.path.join(CODE, "docker", "rk_write.py"))
)
WHOLE_FILE = types.SimpleNamespace(
    **runpy.run_path(os.path.join(CODE, "rkmon_lib", "whole_file.py"))
)
WRITE_USER = LAYOUT.WRITER.name
WRITE_VOLUME = LAYOUT.WRITE_SECRETS_DIR
NEW = LAYOUT.WRITE_LOGIN_DIR
OLD = os.path.join(
    LAYOUT.SECRETS_DIR, os.path.basename(NEW)
)  # the single login of a deployment from before the split
LOGIN = "hosts.yml"  # gh's login, in its folder: rkmon_lib/gh.py HOSTS_FILE (tests/test_container.py)
ACTIONS = ("start", "code-install", "code-back", "hand-layer")
OPTIONS = {"code-install": VOLUME.LEAVE_KEPT}  # an action's one option
DONE, ERROR, USAGE, NO_USER, REFUSED, FAILS = 0, 1, 2, 3, 4, 5
# code that needs a new image, whose fix the host's command says (rkmon apply)
NEW_IMAGE = VOLUME.NEW_IMAGE
# What a change of the code leaves in the volume's folder while it runs, or when it died, by the start of its name: its
# STAGE folder, and a link made before its rename (Volume.link).
LEFT = (VOLUME.STAGE, f".{VOLUME.CURRENT}-", f".{VOLUME.KEPT}-")
# What fixes a code volume that fails its check, after the host's command that does it (rkmon apply): the start's FAIL
# line says it (Volume.failed), and so does the doctor's row of the code volume (rkmon_lib/doctor.py).
VOLUME_FIX = (
    "installs this package's code again, as root, and empties the code volume first"
)
# How long a scan of a user's processes may take (`_scan`): one that takes longer fails, and no release is removed.
SCAN_WAIT = 60
# The one-time move's copies (`copies`) act on the secrets registry's facts (rkmon_lib/secrets_registry.py, their one
# home), kept here as a table, since root acts on no list that the monitor's user can write and loads only the image's
# own code: tests/test_root_copies.py holds the table equal to the registry. Each keeper's secrets folder and user:
KEEPERS = {
    "monitor": (LAYOUT.SECRETS_DIR, LAYOUT.MONITOR.name),
    "writer": (WRITE_VOLUME, WRITE_USER),
}
WHOSE = {"monitor": "the monitor user's", "writer": "the write user's only"}
# Each secret copied byte for byte: (the keeper and name of its older copy, its keeper and file name, the command that
# sets it, after rkmon's name). GitHub's write login moves on its own (`move_write_login`).
COPIES = (
    ("monitor", "teams_webhook", "writer", "teams_webhook", "secret teams_webhook"),
    (
        "monitor",
        "webhost_ed25519",
        "writer",
        "web_host_ssh_key",
        "secret web_host_ssh_key",
    ),
)
# The claude login, whose older copy holds the Claude session's settings with the key among them: the key alone is
# copied (`_claude_value`).
CLAUDE = ("monitor", "claude_env", "monitor", "claude_key", "login claude")
# A model binding's key: the end of its name, the older names that the rule for that end does not give, a secret's name
# (the registry's BINDING_KEY_SUFFIX, OLDER_KEY_NAMES and NAME), and the names the fixed entries keep (their names,
# places and older names), which no binding's key is copied from.
KEY_SUFFIX = "_api_key"
OLDER_KEY_NAMES = {"worker_api_key": "first_review_pass_api_key"}
SECRET_NAME = re.compile(r"[a-z0-9][a-z0-9_]*")
FIXED_NAMES = frozenset(
    {
        "admin_access_token",
        "admin_token",
        "atlassian",
        "claude",
        "claude_env",
        "claude_key",
        "gh",
        "gh-read",
        "gh-read-token",
        "gh_read_token",
        "github_read",
        "github_write",
        "gpu_ssh_key",
        "teams_webhook",
        "web_host_ssh_key",
        "webhost_ed25519",
    }
)
# The delivered deploy layer, which names the bindings' keys and the passes: the settings volume's, which nothing in the
# container writes. The most root reads of an older copy, and of the layer.
LAYER = os.path.join(LAYOUT.SETTINGS_DIR, LAYOUT.LAYER_FILE)
COPY_LIMIT, LAYER_LIMIT = 64 * 1024, 4 * 1024 * 1024
# What open says of a name that is no file it opens: a socket, or a device with no driver (`_read`).
NOT_A_FILE = (errno.ENXIO, errno.ENODEV)
# The one-time move's marks (`copies`): the folder, in the code volume, which only root writes.
MARKS = os.path.join(LAYOUT.CODE_DIR, VOLUME.MOVED)
# The layer handed to a one-off container of the logins (`hand-layer`): where it goes, and the most it reads, what one
# variable carries (container_layout's HANDED_LAYER_MAX).
HANDED = os.path.join(LAYOUT.HANDED_LAYER_DIR, LAYOUT.LAYER_FILE)
HANDED_MAX = LAYOUT.HANDED_LAYER_MAX


def _owner() -> tuple[int, int]:
    """The write user's uid and group id (the monitor's group: docker/Dockerfile)."""
    p = pwd.getpwnam(WRITE_USER)
    return p.pw_uid, p.pw_gid


def _scans() -> list[tuple[int, int, str | None]]:
    """Who scans its own processes for the releases in use (the module's docstring): (uid, gid, the values file it
    reads too, or None) each. KeyError when the image has no such user."""
    out = [(0, 0, None)]
    values = os.path.join(LAYOUT.WORK_DIR, VOLUME.WORKER_VALUES)
    for name, reads in ((LAYOUT.MONITOR.name, values), (WRITE_USER, None)):
        p = pwd.getpwnam(name)
        out.append((p.pw_uid, p.pw_gid, reads))
    return out


def _scan(folder: str, uid: int, gid: int, values: str | None) -> set[str]:
    """The releases in use in the code volume FOLDER that the processes of the user UID, group GID, show, and the one
    VALUES names (code_volume.in_use, run as that user): their names. Raises when the scan fails or takes longer than
    SCAN_WAIT seconds."""
    ids = (
        {}
        if (uid, gid) == (os.getuid(), os.getgid())
        else {"user": uid, "group": gid, "extra_groups": []}
    )
    script = os.path.join(CODE, "rkmon_lib", "code_volume.py")
    p = subprocess.run(
        [sys.executable, "-I", script, "in-use", folder, *([values] if values else [])],
        capture_output=True,
        text=True,
        timeout=SCAN_WAIT,
        cwd="/",
        env={},
        **ids,
    )
    if p.returncode:
        raise OSError(
            f"the scan as uid {uid} ended with exit {p.returncode}: {p.stderr.strip()[-300:]}"
        )
    return set(json.loads(p.stdout))


def _files(old: str) -> list[str]:
    """The names of the regular files in folder `old` (gh keeps its login in a few files, no folder); ValueError when
    `old` is not a folder or holds anything else (a link, a folder)."""
    s = os.lstat(old)
    if not stat.S_ISDIR(s.st_mode):
        raise ValueError(f"{old} is not a folder (a link or a file): left as it is")
    out = []
    for name in sorted(os.listdir(old)):
        if not stat.S_ISREG(os.lstat(os.path.join(old, name)).st_mode):
            raise ValueError(
                f"{os.path.join(old, name)} is not a file (a link or a folder): {old} is left as it is"
            )
        out.append(name)
    return out


def move_write_login(
    old: str = OLD, new: str = NEW, owner: tuple[int, int] | None = None, chown=None
) -> list[str]:
    """Move the write login of a deployment from before the read/write split from `old` to `new` (owned by `owner`,
    the write user by default).
    Idempotent: nothing to do when `new` holds a login or `old` holds none. The lines to print.
    """
    if os.path.isfile(os.path.join(new, LOGIN)):
        return []
    if not os.path.lexists(os.path.join(old, LOGIN)):
        return []
    uid, gid = owner or _owner()
    chown = chown or os.chown
    try:
        entries = _files(old)
    except (OSError, ValueError) as exc:
        return [f"the write login was not moved: {exc}"]
    os.makedirs(new, mode=0o700, exist_ok=True)
    if os.path.islink(new) or not os.path.isdir(new):
        return [f"the write login was not moved: {new} is not a folder"]
    chown(new, uid, gid, follow_symlinks=False)
    os.chmod(new, 0o700)
    for name in entries:
        src, dst = os.path.join(old, name), os.path.join(new, name)
        fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as f_in, open(
            os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600),
            "wb",
        ) as f_out:
            shutil.copyfileobj(f_in, f_out)
        chown(dst, uid, gid, follow_symlinks=False)
        os.chmod(dst, 0o600)
    shutil.rmtree(old)
    return [
        f"moved the GitHub write login from {old} to {new} (the write user's only; {len(entries)} file(s))"
    ]


def write_volume(
    path: str = WRITE_VOLUME, owner: tuple[int, int] | None = None, chown=None
) -> list[str]:
    """The write volume's folder: the write user's, mode 700. The lines to print."""
    try:
        s = os.lstat(path)
    except FileNotFoundError:
        return []
    if not stat.S_ISDIR(s.st_mode):
        return [f"{path} is not a folder: the write login has no volume"]
    uid, gid = owner or _owner()
    out = []
    if (s.st_uid, s.st_gid) != (uid, gid):
        (chown or os.chown)(path, uid, gid, follow_symlinks=False)
        out.append(f"{path} is now the write user's")
    if stat.S_IMODE(s.st_mode) != 0o700:
        os.chmod(path, 0o700)
        out.append(f"{path} is now mode 700")
    return out


class NotCopied(Exception):
    """A copy the one-time move does not make: LINE, what it says; ERROR, True when a file could not be read."""

    def __init__(self, line: str, error: bool = False):
        super().__init__(line)
        self.line, self.error = line, error


def _at(tree, *keys):
    """The value at KEYS in the nested mappings TREE; None when there is none."""
    for key in keys:
        tree = tree.get(key) if isinstance(tree, dict) else None
    return tree


def _read(path: str, limit: int) -> bytes | None:
    """The file at PATH, read without following a link; None when there is none. ValueError for a link, anything but a
    regular file (a socket, or a device with no driver, does not open: NOT_A_FILE), or a file larger than LIMIT
    bytes."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in NOT_A_FILE:
            raise ValueError(f"{path} is not a file") from None
        if exc.errno != errno.ELOOP:
            raise
        raise ValueError(f"{path} is a link, not a file") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"{path} is not a file")
        data = b""
        while len(data) <= limit:
            more = os.read(fd, limit + 1 - len(data))
            if not more:
                break
            data += more
    finally:
        os.close(fd)
    if len(data) > limit:
        raise ValueError(f"{path} is larger than {limit // 1024} KiB")
    return data


def _put(
    folder: str, name: str, data: bytes, owner: tuple[int, int], fchown=os.fchown
) -> bool:
    """DATA as the file NAME in FOLDER, owned by OWNER (uid, gid), mode 600: written whole under a name of its own,
    then linked as NAME, never through a link. False, and nothing changed, when NAME is taken by then.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    folder_fd = os.open(folder, flags)
    try:
        temp = f".{name}.{os.urandom(8).hex()}"
        fd = os.open(
            temp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=folder_fd,
        )
        try:
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view) :]
                os.fchmod(fd, 0o600)
                fchown(fd, *owner)
                os.fsync(fd)
            finally:
                os.close(fd)
            try:
                os.link(
                    temp,
                    name,
                    src_dir_fd=folder_fd,
                    dst_dir_fd=folder_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                return False
        finally:
            os.unlink(temp, dir_fd=folder_fd)
        os.fsync(folder_fd)
        return True
    finally:
        os.close(folder_fd)


def _key_name(name: str) -> str:
    """The name the move copies a model binding's key named NAME to (secrets_registry.binding_key_name)."""
    if name in OLDER_KEY_NAMES:
        return OLDER_KEY_NAMES[name]
    if SECRET_NAME.fullmatch(name) and not name.endswith(KEY_SUFFIX):
        return name.removesuffix("_key") + KEY_SUFFIX
    return name


def older_key_names(name: str, taken) -> list[str]:
    """The older names of a model binding's key named NAME, in order, but the names in TAKEN: each name `_key_name`
    gives NAME from (secrets_registry's older names of a binding's key)."""
    olders = [o for o, new in OLDER_KEY_NAMES.items() if new == name]
    stem = name.removesuffix(KEY_SUFFIX)
    if stem != name:
        olders += [f"{stem}_key", stem]
    return [o for o in dict.fromkeys(olders) if o not in taken and _key_name(o) == name]


def pass_models(settings) -> dict:
    """{its model key: the model it names} for each pass of review automation in the deploy SETTINGS (a layer's): each
    key NAME.model beside NAME.effort, as rkmon_lib/deploy_tree.py finds a pass's model.
    """
    out = {}
    todo = [
        ("features.review_automation", _at(settings, "features", "review_automation"))
    ]
    while todo:
        path, node = todo.pop()
        if not isinstance(node, dict):
            continue
        if "model" in node and "effort" in node:
            out[f"{path}.model"] = node["model"]
        todo += [(f"{path}.{k}", v) for k, v in node.items() if isinstance(v, dict)]
    return out


def layer_keys(path: str = LAYER) -> tuple[dict, bool, str]:
    """What the delivered deploy layer at PATH says of the model bindings' keys: ({each name a binding's
    api_key_secret_name gives: whether a pass of review automation runs on such a binding}, whether review automation
    is on, ""); ({}, False, why it says nothing) when none is delivered, or it cannot be read.
    """
    try:
        data = _read(path, LAYER_LIMIT)
        layer = None if data is None else json.loads(data.decode("utf-8"))
    except (OSError, ValueError) as exc:
        return {}, False, f"your confirmed settings cannot be read ({exc})"
    if layer is None:
        return {}, False, "your confirmed settings are not in the container yet"
    settings = _at(layer, "settings")
    if (
        not isinstance(settings, dict)
        or layer.get("kind") != LAYOUT.LAYER_KIND
        or layer.get("schema_version") != LAYOUT.LAYER_SCHEMA_VERSION
    ):
        return {}, False, "your confirmed settings are of another kind or layout"
    on = _at(settings, "features", "review_automation", "enabled") is True
    models = set(pass_models(settings).values()) if on else set()
    bindings = settings.get("model_bindings")
    named = {}
    for binding, fields in (bindings if isinstance(bindings, dict) else {}).items():
        key = _at(fields, "api_key_secret_name")
        if (
            isinstance(key, str)
            and SECRET_NAME.fullmatch(key)
            and key.endswith(KEY_SUFFIX)
        ):
            named[key] = named.get(key, False) or binding in models
    return named, on, ""


def _older_value(paths: list, command: str) -> tuple[str, bytes] | None:
    """(the older copy among PATHS that holds a value, its value), copied byte for byte; None when none holds one.
    NotCopied when one is refused (`_read`) or cannot be read, or two hold different values; COMMAND sets the secret.
    """
    held = {}
    for path in paths:
        try:
            data = _read(path, COPY_LIMIT)
        except ValueError as exc:
            raise NotCopied(
                f"the one-time move did not copy {path}: {exc}; the fix: rkmon {command}, on the host"
            ) from None
        except OSError as exc:
            raise NotCopied(
                f"ERROR: the one-time move could not read {path}: {exc.strerror or exc}; the fix: rkmon {command}, "
                "on the host",
                True,
            ) from None
        if data is not None and data.strip():
            held[path] = data
    if len({data.strip() for data in held.values()}) > 1:
        raise NotCopied(
            f"the one-time move did not copy {' or '.join(held)}: they hold different values; the fix: rkmon "
            f"{command}, on the host"
        )
    return next(iter(held.items()), None)


def _claude_value(paths: list, command: str) -> tuple[str, bytes]:
    """(what is copied, in words; its value): the key alone in the older Claude settings at PATHS' one path, as the
    claude login keeps it (one line), read by the one reader of their shape (rkmon_lib/claude_env_shape.py, loaded by
    its path from the image's own code, with no judgement of what looks like a secret: its lines name variables and
    counts only). NotCopied when it is refused or cannot be read, and when its `move` says why no key is copied: the
    settings cannot carry the rest of what is in the file, or there is no key in it. Check 16 shows the same words.
    """
    (path,) = paths
    fix = f"the fix: rkmon {command}, on the host"
    shape = runpy.run_path(os.path.join(CODE, "rkmon_lib", "claude_env_shape.py"))
    try:
        found, key = shape["read_key"](path)
    except ValueError as exc:
        raise NotCopied(
            f"the one-time move did not copy the key in {path}: {exc}; {fix}"
        ) from None
    except OSError as exc:
        raise NotCopied(
            f"ERROR: the one-time move could not read {path}: {exc.strerror or exc}; {fix}",
            True,
        ) from None
    why = shape["move"](found)
    if why == shape["NO_KEY"]:
        raise NotCopied(f"the one-time move copied nothing from {path}: {why}; {fix}")
    if why:
        raise NotCopied(
            f"the one-time move did not copy the key in {path}: {why}; {fix}"
        )
    return f"the key alone in {path}", f"{key}\n".encode("utf-8")


def _copies_to_make(folders: dict, layer: str) -> list:
    """The copies the move makes when their new copy is missing, in order: (the older copies' paths, the keeper, the
    new copy's file name, the command that sets it, how its value is read (`_older_value`, `_claude_value`: what is
    copied, in words, and its value), why no feature uses it or "").
    """

    def at(keeper: str, name: str) -> str:
        return os.path.join(folders[keeper], name)

    out = [
        ([at(was, older)], keeper, name, command, _older_value, "")
        for was, older, keeper, name, command in COPIES
    ]
    was, older, keeper, name, command = CLAUDE
    out.append(([at(was, older)], keeper, name, command, _claude_value, ""))
    named, on, why = layer_keys(layer)
    taken = FIXED_NAMES | set(named)
    for name, used in sorted(named.items()):
        if name in FIXED_NAMES:
            continue
        if used:
            unused = ""
        elif on:
            unused = f"no pass of review automation runs on a model binding that names {name}, so no feature uses it"
        else:
            unused = "review automation is off, so no feature uses it"
        olders = [at("monitor", o) for o in older_key_names(name, taken)]
        out.append((olders, "monitor", name, f"secret {name}", _older_value, unused))
    for older, name in OLDER_KEY_NAMES.items():
        if name not in named:
            unused = why or f"no model binding names {name}, so no feature uses it"
            out.append(
                (
                    [at("monitor", older)],
                    "monitor",
                    name,
                    f"secret {name}",
                    _older_value,
                    unused,
                )
            )
    return out


def _marks(path: str) -> int:
    """The folder of the one-time move's marks, PATH, opened: made when it is missing, mode 755 whatever the umask, so
    that the doctor's check of the code volume reads it. OSError when it is not a folder, or not this user's alone to
    write (root's, at a start): then no mark can be trusted."""
    with contextlib.suppress(FileExistsError):
        os.mkdir(path, 0o755)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        if exc.errno not in (errno.ELOOP, errno.ENOTDIR):
            raise
        raise OSError(exc.errno, "it is not a folder") from None
    try:
        s = os.fstat(fd)
        if s.st_uid != os.geteuid() or s.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise OSError(
                errno.EPERM,
                f"another user may write it (uid {s.st_uid}, mode {stat.S_IMODE(s.st_mode):04o})",
            )
        if stat.S_IMODE(s.st_mode) != 0o755:
            os.fchmod(fd, 0o755)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _marked(marks: int, name: str) -> bool:
    """Whether the folder MARKS (`_marks`) holds the mark NAME."""
    try:
        os.stat(name, dir_fd=marks, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _mark(marks: int, name: str) -> None:
    """The mark NAME made in the folder MARKS (`_marks`), to last: an empty file, mode 644. One that exists stays."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(name, flags, 0o644, dir_fd=marks)
    except FileExistsError:
        return
    try:
        os.fchmod(fd, 0o644)
    finally:
        os.close(fd)
    os.fsync(marks)


def _copy(
    present: list, keeper: str, name: str, command: str, value, owners, fchown, folder
) -> tuple[str, bool]:
    """One copy a start makes: the new copy NAME in KEEPER's FOLDER, from the older copies PRESENT, read by VALUE
    (`_copies_to_make`). (Its line, "" for none; whether it cannot be read or written, an ERROR, which a later start
    deals with.)"""
    new = os.path.join(folder, name)
    try:
        found = value(present, command)
    except NotCopied as no:
        return no.line, no.error
    if found is None:
        return "", False
    what, data = found
    if owners:
        ids = owners[keeper]
    else:
        user = pwd.getpwnam(KEEPERS[keeper][1])
        ids = user.pw_uid, user.pw_gid
    try:
        if not _put(folder, name, data, ids, fchown):
            return "", False
    except OSError as exc:
        return (
            f"ERROR: the one-time move could not write {new}: {exc.strerror or exc}; the fix: rkmon {command}, on the "
            "host",
            True,
        )
    return (
        f"the one-time move copied {what} to {new} ({WHOSE[keeper]}, mode 600); the older copy stays until the move "
        "ends",
        False,
    )


def copies(
    folders: dict | None = None,
    owners: dict | None = None,
    layer: str = LAYER,
    fchown=os.fchown,
    marks: str = MARKS,
) -> tuple[list[str], bool]:
    """The one-time move's copies at a start, each made once: see the module's docstring. FOLDERS and OWNERS, by
    keeper: its secrets folder and its user's (uid, gid), KEEPERS' by default; LAYER, the delivered deploy layer;
    MARKS, the folder of the move's marks. (The lines to print, whether a copy could not be made.)
    """
    folders = folders or {k: folder for k, (folder, _user) in KEEPERS.items()}
    try:
        fd = _marks(marks)
    except OSError as exc:
        return [
            f"ERROR: the one-time move made no copy: its marks cannot be kept in {marks}: {exc.strerror or exc}"
        ], True
    out, failed = [], False
    try:
        for olders, keeper, name, command, value, unused in _copies_to_make(
            folders, layer
        ):
            new = os.path.join(folders[keeper], name)
            mark = f"{keeper}.{name}"
            present = [p for p in olders if os.path.lexists(p)]
            if present and not os.path.lexists(new):
                older = " or ".join(present)
                if _marked(fd, mark):
                    out.append(
                        f"the one-time move did not copy {older} to {new}: the move makes each copy once, and an "
                        f"earlier start dealt with this one; the fix: rkmon {command}, on the host"
                    )
                    continue
                if unused:
                    out.append(
                        f"the one-time move did not copy {older} to {new}: {unused}"
                    )
                    continue
                line, error = _copy(
                    present,
                    keeper,
                    name,
                    command,
                    value,
                    owners,
                    fchown,
                    folders[keeper],
                )
                out += [line] if line else []
                if error:
                    failed = True
                    continue
            try:
                _mark(fd, mark)
            except OSError as exc:
                out.append(
                    f"ERROR: the one-time move could not mark its copy to {new} as dealt with, in {marks}: "
                    f"{exc.strerror or exc}"
                )
                failed = True
    finally:
        os.close(fd)
    return out, failed


class Stop(Exception):
    """An action ended short of its change: CODE, its exit code, and LINES, what it says."""

    def __init__(self, code: int, *lines: str):
        super().__init__(lines[0])
        self.code, self.lines = code, list(lines)


def hand_layer(
    data: bytes, path: str = HANDED, owner: tuple[int, int] | None = None
) -> list[str]:
    """`hand-layer`: DATA, the layer handed to this container, written as the file PATH, its folder too, owned by
    OWNER (root and the monitor's group by default): see the module's docstring. The lines to print (none); Stop when
    it is not written, but for something else in the folder's place, which rk's read refuses.
    """
    folder = os.path.dirname(path)
    not_written = "your confirmed settings handed to this container are not written"
    if not data or len(data) > HANDED_MAX:
        why = "empty" if not data else f"longer than {HANDED_MAX} bytes"
        raise Stop(ERROR, f"ERROR: {not_written}: they are {why}")
    uid, gid = owner or (0, grp.getgrnam(LAYOUT.MONITOR.group).gr_gid)
    with contextlib.suppress(FileExistsError):
        os.mkdir(folder, 0o750)
    s = os.lstat(folder)
    if not stat.S_ISDIR(s.st_mode) or s.st_uid != uid:
        return []  # nothing written in another's place: rk's read refuses it
    os.chown(folder, uid, gid, follow_symlinks=False)
    os.chmod(folder, 0o750)
    WHOLE_FILE.write(path, data, 0o640, durable=True, owner=(uid, gid))
    return []


@contextlib.contextmanager
def _said_first(out: list[str]):
    """OUT, the lines of what was done so far, said first when the block ends short: by a Stop, or by an OSError,
    which ends it as an ERROR (exit 1)."""
    try:
        yield
    except Stop as stop:
        stop.lines[:0] = out
        raise
    except OSError as exc:
        raise Stop(ERROR, *out, f"ERROR: the code volume: {exc}") from None


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _taken(stage: str, parts: list[str]) -> str:
    """The archive's name for what the member PARTS meets where it goes in the folder STAGE: the first folder on its way
    that is a file there, else its own name."""
    for n in range(1, len(parts)):
        if not os.path.isdir(os.path.join(stage, *parts[:n])):
            return "/".join(parts[:n])
    return "/".join(parts)


class Volume:
    """The code volume in FOLDER, over the image's own code in IMAGE, whose digest the file ID_FILE holds. CHOWN and
    LSTAT are os.chown and os.lstat, and SCANS who scans its processes for the releases in use (`_scans`, by
    default); the tests give a fake root's, and their own user."""

    def __init__(
        self,
        folder: str = LAYOUT.CODE_DIR,
        image: str = CODE,
        id_file: str = LAYOUT.IMAGE_ID_FILE,
        chown=os.chown,
        lstat=os.lstat,
        scans=None,
    ):
        self.folder, self.image, self.id_file = folder, os.path.realpath(image), id_file
        self.chown, self.lstat, self.scans = chown, lstat, scans

    def path(self, *names: str) -> str:
        return os.path.join(self.folder, *names)

    def image_id(self) -> str:
        try:
            with open(self.id_file, encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return ""

    def running(self) -> str | None:
        """The release that runs, resolved, or None when the image's own code runs (code_volume.running)."""
        rel = VOLUME.running(self.folder, self.image_id())
        return None if rel is None else str(rel)

    def name(self, tree: str | None) -> str:
        """The code in the folder TREE (None: the image's own) in words; a release installed over another image says
        so."""
        if tree is None or os.path.realpath(tree) == self.image:
            return "the image's own code"
        over = VOLUME.release_info(tree).get("base_image")
        other = (
            ""
            if over and over == self.image_id()
            else " (installed over another image)"
        )
        return f"release {os.path.basename(tree)}{other}"

    def current(self) -> str | None:
        """The release CURRENT names, resolved, whatever image it was installed over; None for the image's own code:
        when CURRENT names no release, or one that is not whole, which runs on no image.
        """
        current = self.path(VOLUME.CURRENT)
        if not os.path.lexists(current):
            return None
        tree = os.path.realpath(current)
        if tree == self.image or VOLUME.missing(tree):
            return None
        return tree

    def own(self) -> list[str]:
        """The volume's folder, made when it is missing: root's, mode 755. The lines to print."""
        if not os.path.lexists(self.folder):
            os.makedirs(self.folder, mode=0o755)
        s = self.lstat(self.folder)
        if not stat.S_ISDIR(s.st_mode):
            raise Stop(
                FAILS,
                f"FAIL: the code volume's folder {self.folder} is not a folder: the image's own code runs, and the "
                "monitor does not start; the fix: mount the code volume there (docker/compose.yaml)",
            )
        out = []
        if (s.st_uid, s.st_gid) != (0, 0):
            self.chown(self.folder, 0, 0, follow_symlinks=False)
            out.append(f"{self.folder} is now root's")
        if stat.S_IMODE(s.st_mode) != 0o755:
            os.chmod(self.folder, 0o755)
            out.append(f"{self.folder} is now mode 755")
        return out

    @contextlib.contextmanager
    def locked(self, wait: bool = False):
        """LOCK held while the block runs, so that one change of the code volume runs at a time; while another holds
        it, Stop (REFUSED), or with WAIT the wait until it is free. Then what a change that died left goes. A LOCK that
        is not root's one file (a folder, a link, another user's file or a second name of a file) was made while
        another user could write the folder (own() has made it root's since): it goes, and a new one takes its place.
        """
        lock = self.path(VOLUME.LOCK)
        try:
            s = self.lstat(lock)
        except FileNotFoundError:
            s = None
        if s and not (stat.S_ISREG(s.st_mode) and s.st_uid == 0 and s.st_nlink == 1):
            VOLUME.remove(lock)
        fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            except BlockingIOError:
                raise Stop(
                    REFUSED,
                    "REFUSED: another change of the code runs now (rkmon apply or rkmon rollback): run this one "
                    "again when it has ended; nothing changed",
                ) from None
            if self.lstat(lock).st_uid != 0:
                self.chown(lock, 0, 0, follow_symlinks=False)
            os.fchmod(fd, 0o600)
            for name in os.listdir(self.folder):
                if name.startswith(LEFT):
                    VOLUME.remove(self.path(name))
            yield
        finally:
            os.close(fd)

    def problems(self, leftovers: bool = True) -> list[str]:
        """What keeps the volume from being root's and read-only all the way, as the write broker checks it. Without
        LEFTOVERS, what a change of the code leaves while it runs, or left when it died (LEFT), is left out: each change
        and the start remove it before they check, and a user other than root cannot read a change's STAGE folder (the
        doctor's row)."""
        skip = []
        if not leftovers:
            try:
                skip = [
                    self.path(n) for n in os.listdir(self.folder) if n.startswith(LEFT)
                ]
            except OSError:
                pass
        return BROKER.problems(self.folder, self.lstat, kept=(self.image,), skip=skip)

    def failed(self, problems=(), unchecked: Exception | None = None) -> Stop:
        """The Stop (FAILS) for a volume with PROBLEMS, or that could not be checked (UNCHECKED, what stopped the
        check). First CURRENT goes, so that no release runs while the volume fails: the launcher, and the broker, run
        the image's own code without it."""
        what = (
            f"could not be checked: {unchecked}"
            if unchecked is not None
            else f"is not root's and read-only all the way: {VOLUME.listed(problems)}"
        )
        current = self.path(VOLUME.CURRENT)
        if not os.path.lexists(current):
            runs = "no release runs: the image's own code does"
        elif VOLUME.remove(current):
            runs = f"so that no release runs, root removed {current}: the image's own code runs"
        else:
            runs = f"{current} could not be removed: its release may still run"
        return Stop(
            FAILS,
            f"FAIL: the code volume ({self.folder}) {what}; {runs}",
            f"  the fix: rkmon apply, on the host, {VOLUME_FIX}; until then the monitor does not start",
        )

    def link(self, name: str, target: str | None) -> None:
        """The link NAME (CURRENT or KEPT) pointed at TARGET in one rename, or removed for None."""
        path = self.path(name)
        if target is None:
            if os.path.lexists(path):
                os.unlink(path)
            return
        new = self.path(f".{name}-{os.getpid()}")
        if os.path.lexists(new):
            os.unlink(new)
        os.symlink(target, new)
        os.replace(
            new, path
        )  # a link, not a file: rkmon_lib/whole_file.py writes files

    def prune(self) -> list[str]:
        """Each release that neither CURRENT nor KEPT names removed when nothing uses it: see the module's docstring.
        The lines to print."""
        keep = {
            os.path.realpath(self.path(n))
            for n in (VOLUME.CURRENT, VOLUME.KEPT)
            if os.path.lexists(self.path(n))
        }
        releases = self.path(VOLUME.RELEASES)
        names = [
            name
            for name in (
                sorted(os.listdir(releases)) if os.path.isdir(releases) else []
            )
            if os.path.realpath(os.path.join(releases, name)) not in keep
        ]
        if not names:
            return []
        later = "the next start or install removes what nothing uses then"
        try:
            used = set()
            for uid, gid, values in _scans() if self.scans is None else self.scans:
                used |= _scan(self.folder, uid, gid, values)
        # No release goes while the releases in use are not known, whatever stops a scan.
        except Exception as exc:  # noqa: BLE001
            return [
                f"no release was removed: the releases in use could not be found ({exc}); {later}"
            ]
        out = []
        for name in names:
            if name in used:
                out.append(
                    f"release {name} stays: it is in use (a process runs it, or the worker tools name it); {later}"
                )
            else:
                VOLUME.remove(os.path.join(releases, name))
        return out

    def start(self) -> list[str]:
        """The code volume at a start of `run`: see the module's docstring. Whatever stops the check fails it."""
        out = []
        with _said_first(out):
            try:
                out += self.own()
                with self.locked(wait=True):
                    bad = self.problems()
                    if bad:
                        raise self.failed(bad)
                    out += self.prune()
            except Stop:
                raise
            # A check that could not be made fails, whatever stopped it.
            except Exception as exc:  # noqa: BLE001
                raise self.failed(unchecked=exc) from None
        return out

    def unpack(self, stream, stage: str) -> str:
        """The package in the tar archive STREAM, unpacked in the folder STAGE: its top folder. Only its folders and
        regular files, Python's bytecode left out, each file read-only as it is written (RUN_MODE when its mode in the
        archive is one that runs, code_volume.runs, else FILE_MODE); Stop (REFUSED) for any other member, a name
        outside the archive's folder, a name twice, or an archive that does not read.
        """
        try:
            with tarfile.open(fileobj=stream, mode="r|*") as tar:
                for m in tar:
                    parts = [p for p in m.name.split("/") if p not in ("", ".")]
                    if m.name.startswith("/") or ".." in parts:
                        raise Stop(
                            REFUSED,
                            f"REFUSED: the archive names {m.name}, outside its folder; nothing changed",
                        )
                    if not parts or "__pycache__" in parts or m.name.endswith(".pyc"):
                        continue
                    path = os.path.join(stage, *parts)
                    if m.isdir():
                        os.makedirs(path, mode=0o700, exist_ok=True)
                    elif m.isreg():
                        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
                        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                        with tar.extractfile(m) as src, os.fdopen(
                            os.open(path, flags | os.O_CLOEXEC, 0o600), "wb"
                        ) as out:
                            shutil.copyfileobj(src, out)
                            os.fchmod(
                                out.fileno(),
                                (
                                    VOLUME.RUN_MODE
                                    if VOLUME.runs(m.mode)
                                    else VOLUME.FILE_MODE
                                ),
                            )
                            out.flush()
                            os.fsync(out.fileno())
                    else:
                        raise Stop(
                            REFUSED,
                            f"REFUSED: the archive has {m.name}, a link or a special file, where a package may "
                            "carry only files and folders; nothing changed",
                        )
        except (FileExistsError, NotADirectoryError, IsADirectoryError):
            raise Stop(
                REFUSED,
                f"REFUSED: the archive names {_taken(stage, parts)} twice; nothing changed",
            ) from None
        except (tarfile.TarError, EOFError) as exc:
            raise Stop(
                REFUSED,
                f"REFUSED: what came on stdin is not a package's tar archive ({exc}); nothing changed",
            ) from None
        names = os.listdir(stage)
        top = os.path.join(stage, names[0]) if len(names) == 1 else stage
        return top if os.path.isdir(top) else stage

    def install(self, stream, leave_kept: bool = False) -> list[str]:
        """The package's code in the tar archive STREAM installed as a release over this image, KEPT left as it is
        with LEAVE_KEPT (a start's install): see the module's docstring for `code-install`. The lines to print; Stop
        when it ends short."""
        out = []
        with _said_first(out):
            out += self.own()
            with self.locked():
                stage = self.path(f"{VOLUME.STAGE}{os.getpid()}")
                os.mkdir(stage, 0o700)
                try:
                    self._install(stage, stream, out, leave_kept)
                finally:
                    if os.path.lexists(stage):
                        VOLUME.remove(stage)
        return out

    def empty(self, stage: str, problems: list[str], out: list[str]) -> None:
        """The volume, which failed its check (PROBLEMS), emptied but for LOCK, the folder STAGE, which holds the
        release that takes its place, ready, and the one-time move's marks (MOVED: `copies`) while they pass the check
        themselves; the line that says so added to OUT. Stop (ERROR) when a name stays.
        """
        keep = (VOLUME.LOCK, os.path.basename(stage))
        marks = self.path(VOLUME.MOVED)
        if (
            os.path.isdir(marks)
            and not os.path.islink(marks)
            and not BROKER.problems(marks, self.lstat)
        ):
            keep += (VOLUME.MOVED,)
        for name in os.listdir(self.folder):
            if name not in keep:
                VOLUME.remove(self.path(name))
        why = f"the code volume was not root's and read-only all the way ({VOLUME.listed(problems)})"
        left = sorted(self.path(n) for n in os.listdir(self.folder) if n not in keep)
        if left:
            raise Stop(
                ERROR,
                f"ERROR: {why}, and root could not empty it: {VOLUME.listed(left)} could not be removed, the rest of "
                "it was; the release was not put in place",
            )
        out.append(
            f"{why}: it was emptied first, so the image's own code runs until the release is in place"
        )

    def _install(
        self, stage: str, stream, out: list[str], leave_kept: bool = False
    ) -> None:
        """The package in STREAM unpacked and checked in the folder STAGE, and made a release there, ready, before
        the volume changes: an archive or a package refused changes nothing. A volume that fails its check is emptied
        only then, and a step after that which fails says so. KEPT names the code CURRENT named before, but with
        LEAVE_KEPT. Its lines added to OUT."""
        top = self.unpack(stream, stage)
        if os.path.lexists(os.path.join(top, VOLUME.RELEASE_FILE)):
            raise Stop(
                REFUSED,
                f"REFUSED: the package has {VOLUME.RELEASE_FILE}, the record root writes in each release it "
                "installs and no package may carry; nothing changed",
            )
        lost = VOLUME.missing(top)
        if lost:
            raise Stop(
                REFUSED,
                f"REFUSED: not a whole package: {VOLUME.listed(lost)} missing or empty; nothing changed",
            )
        changed = VOLUME.image_changes(self.image, top)
        if changed:
            raise Stop(
                NEW_IMAGE,
                f"REFUSED: this code needs a new image: {VOLUME.listed(changed)} differ from this image's; nothing "
                "changed",
            )
        image_id = self.image_id()
        if not image_id:
            raise Stop(
                ERROR,
                f"ERROR: the image has no code digest ({self.id_file}): build the image again (QUICK_BUILD_GUIDE.md); "
                "nothing changed",
            )
        # The volume's check, what this change leaves while it runs aside (its STAGE): a volume that fails it holds
        # nothing that runs on, and nothing to keep.
        bad = self.problems(leftovers=False)
        digest, was = VOLUME.digest(top), None if bad else self.running()
        if digest == (
            VOLUME.release_info(was).get("digest") if was else VOLUME.digest(self.image)
        ):
            if bad:
                self.empty(stage, bad, out)
            out.append(f"NOTHING TO DO: this code runs already ({self.name(was)})")
            return
        # The code it takes the place of: CURRENT's release whatever image it was installed over, which KEPT names then
        # (but with LEAVE_KEPT), so that a rollback to that image runs it again.
        replaced = None if bad else self.current()
        version = VOLUME.version(top)
        over = hashlib.sha256(f"{digest} {image_id}".encode()).hexdigest()
        rid = (
            f"{re.sub(r'[^0-9A-Za-z._+-]', '_', version) or 'unversioned'}-{over[:12]}"
        )
        record = {
            "id": rid,
            "version": version,
            "digest": digest,
            "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "base_image": image_id,
        }
        with open(
            os.open(
                os.path.join(top, VOLUME.RELEASE_FILE),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(record, indent=1) + "\n")
            os.fchmod(f.fileno(), VOLUME.FILE_MODE)
            f.flush()
            os.fsync(f.fileno())
        # Read-only to every user: each folder but the top, which a rename into RELEASES must still write (the files
        # are already, from the unpack).
        for folder, _dirs, _files in os.walk(top, topdown=False):
            if folder != top:
                os.chmod(folder, VOLUME.DIR_MODE)
        if bad:
            self.empty(stage, bad, out)
        try:
            releases = self.path(VOLUME.RELEASES)
            os.makedirs(releases, mode=0o755, exist_ok=True)
            # Mode 755 whatever the umask (the entry point's is 027): the launcher and the broker, which run as other
            # users, must reach the release, else the image's own code runs for them.
            os.chmod(releases, 0o755)
            dest = os.path.join(releases, rid)
            if os.path.lexists(dest):
                VOLUME.remove(dest)
            os.rename(top, dest)
            os.chmod(dest, VOLUME.DIR_MODE)
            _fsync_dir(releases)
            if not leave_kept:
                self.link(
                    VOLUME.KEPT,
                    (
                        os.path.join(VOLUME.RELEASES, os.path.basename(replaced))
                        if replaced
                        else self.image
                    ),
                )
            self.link(VOLUME.CURRENT, os.path.join(VOLUME.RELEASES, rid))
        except OSError as exc:
            if not bad:
                raise
            raise Stop(
                ERROR,
                f"ERROR: the release was not put in place ({exc}): the code volume was emptied, and the image's own "
                "code runs",
            ) from None
        _fsync_dir(self.folder)
        pruned = self.prune()
        done = (
            f"INSTALLED: release {rid} ({version or 'no version'}) over this image, in place of "
            f"{self.name(replaced)}"
        )
        kept = self.path(VOLUME.KEPT)
        if not leave_kept:
            out.append(f"{done}, which is kept: rkmon rollback goes back to it")
        elif os.path.lexists(kept):
            out.append(
                f"{done}; the code kept stays {self.name(os.path.realpath(kept))}: rkmon rollback goes back to it"
            )
        else:
            out.append(f"{done}; no code is kept")
        out += pruned

    def back(self) -> list[str]:
        """The code KEPT names in CURRENT's place again: see the module's docstring for `code-back`. The lines to print;
        Stop when it ends short."""
        out = []
        with _said_first(out):
            out += self.own()
            with self.locked():
                bad = self.problems()
                if bad:
                    raise self.failed(bad)
                kept, current = self.path(VOLUME.KEPT), self.path(VOLUME.CURRENT)
                if not os.path.lexists(kept):
                    return out + ["NOTHING TO DO: no code is kept to go back to"]
                to = os.path.realpath(kept)
                was = (
                    os.path.realpath(current)
                    if os.path.lexists(current)
                    else self.image
                )
                if to == was:
                    return out + [
                        f"NOTHING TO DO: the code kept is in place already ({self.name(to)})"
                    ]
                if to != self.image and VOLUME.missing(to):
                    raise Stop(
                        REFUSED,
                        f"REFUSED: the code kept ({self.name(to)}) is not whole: it runs on no image; rkmon apply "
                        "installs a package's code; nothing changed",
                    )
                self.link(
                    VOLUME.CURRENT,
                    (
                        None
                        if to == self.image
                        else os.path.join(VOLUME.RELEASES, os.path.basename(to))
                    ),
                )
                self.link(VOLUME.KEPT, None)
                _fsync_dir(self.folder)
        runs = "runs again" if to in (self.image, self.running()) else "is back"
        return out + [f"RESTORED: {self.name(to)} {runs}, in place of {self.name(was)}"]


def main(argv: list[str], stdin=None, volume: Volume | None = None) -> int:
    action, options = (argv[0], argv[1:]) if argv else ("", [])
    if action not in ACTIONS or options not in ([], [OPTIONS.get(action)]):
        each = (f"{a} [{OPTIONS[a]}]" if a in OPTIONS else a for a in ACTIONS)
        print(f"usage: rk_root.py {' | '.join(each)}", file=sys.stderr)
        return USAGE
    if os.geteuid() != 0:
        print(
            "rk_root.py: as root only (the entry point runs it as root)",
            file=sys.stderr,
        )
        return USAGE
    lines, code = [], DONE
    if action == "hand-layer":
        try:
            lines += hand_layer(
                (sys.stdin.buffer if stdin is None else stdin).read(HANDED_MAX + 1)
            )
        except Stop as stop:
            lines, code = lines + stop.lines, stop.code
        except OSError as exc:
            lines, code = (
                lines
                + [
                    f"ERROR: your confirmed settings handed to this container are not written: {exc}"
                ],
                ERROR,
            )
    if action == "start":
        try:
            grp.getgrgid(_owner()[1])
            lines += write_volume() + move_write_login()
        except KeyError:
            lines, code = (
                lines + [f"ERROR: no user {WRITE_USER} in this image"],
                NO_USER,
            )
        # The start checks the code volume whatever the move does.
        except Exception as exc:  # noqa: BLE001
            lines, code = lines + [f"ERROR: the write login's move: {exc}"], ERROR
    if action == "start" and code != NO_USER:
        try:
            said, failed = copies()
            lines, code = lines + said, ERROR if failed else code
        # The start checks the code volume whatever the copies do.
        except Exception as exc:  # noqa: BLE001
            lines, code = lines + [f"ERROR: the one-time move's copies: {exc}"], ERROR
    if action in ("start", "code-install", "code-back"):
        v = volume or Volume()
        try:
            if action == "start":
                lines += v.start()
            elif action == "code-install":
                lines += v.install(
                    sys.stdin.buffer if stdin is None else stdin,
                    leave_kept=bool(options),
                )
            else:
                lines += v.back()
        except Stop as stop:
            lines, code = lines + stop.lines, stop.code
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
