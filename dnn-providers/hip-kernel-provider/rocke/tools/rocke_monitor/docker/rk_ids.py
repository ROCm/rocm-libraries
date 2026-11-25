#!/usr/bin/python3 -I
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The deployment's user ids, mapped onto the image's users at each start (docker/entrypoint.sh runs it as root, before
any command drops to its user). The image is built with ids of its own (rkmon_lib/container_layout.py: rkmon 1000:1000,
rkwrite 1999, the PR runners rkpr1 to rkprN, each in a group of its own and in their group rkpr), never the build
machine's. The deployment names the monitor's in docker/.env, which is made from the confirmed deploy settings
(host.users.monitor_uid and monitor_gid, the operator's `id -u` and `id -g` by default, so a file of theirs mounted
read-only, or kept in a home on NFS, is readable inside):

  RKMON_UID, RKMON_GID   the monitor's user, rkmon, and its group (rkwrite's group too)

A variable that is unset keeps the image's id. The GitHub writer's id and the PR runners' are the package's
(container_layout), the image's in every deployment: no variable changes them. The volumes' files keep up: when the ids
differ from the ones the volumes were last given (/work/.rkmon_ids.json records them; none yet, the image's), each file
of an old id is given the new one, so a deployment keeps its state, its secrets and their modes when its ids change.
Once, the files of the single PR-code user the volumes had before the runners (on the first runner's id,
container_layout's PR_RUNNER) become the monitor's, and its group's the runners' group: no runner owns a file the
monitor kept. With the ids in place it changes nothing; a container made again from the image renumbers the monitor's
user and group and walks no volume. The volumes' own folders get their owners at every start (root_owners): docker
gives an empty named volume's folder the image's owner again in each new container. The record is the monitor's, in
its folder, so root takes nothing of it on trust: it reads a regular file there alone (recorded), moves no file of an
id another user or group of the container holds, whatever the record names (taken), and writes the record whole and
never through a link (write_record).

A one-off container of the logins' walk (docker/entrypoint.sh: an rk command handed your confirmed settings) runs the
image you confirmed beside the running container, which may run another package's image, whose record and owners
this image's rules need not share: there (--users) the users and their homes alone get the ids, and a volume's own
folder gets its owner only while it has the image's own (root_owners of the image's ids: a volume docker just made,
or an empty one it gave the image's owner again), so that the walk's rk reaches the secrets volumes. No file of the
volumes is walked, and the record is neither read nor written. A one-off container whose command only reads (a
question, adopt's plan: container_layout LEAVE_VOLUMES_VARIABLE) runs it the same way.

Refused (exit 2, nothing changed): an id that is not a number from 1 to 2147483647; an id that would mix the users
(clashes, the one home of these rules): the monitor's id on a system account's (system_id: below 1000, or nobody's
65534), on a PR runner's or on the writer's, and its group on the runners' group's id or on a runner's own group's id;
or an id a system account of the image holds (nobody, nogroup). A refusal names the deploy setting to change (SETTING).
A regular account of the base image on a wanted id (Ubuntu's `ubuntu`) is removed, as the image's build does. The
deploy settings' checks word the same clashes for the monitor's ids they give, with the package's for the writer's
(settings_ids): rkmon_lib/deployment.py container_checks (rkmon confirm and rkmon check) and rkmon_lib/coherence.py
check 8.

  rk_ids.py            map the ids from the environment (as root)
  rk_ids.py --users    the users alone, in a one-off container of the logins' walk or one that only reads (above)
  rk_ids.py --plan     print what it would change, change nothing
"""

from __future__ import annotations

import contextlib
import json
import os
import runpy
import stat
import subprocess
import sys
import types
from typing import NamedTuple

CODE = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)  # the image's code: read-only, owned by root
LAYOUT = types.SimpleNamespace(
    **runpy.run_path(os.path.join(CODE, "rkmon_lib", "container_layout.py"))
)


def perms() -> types.SimpleNamespace:
    """worker/perms.py, the modes' home, read when a start writes its record: the deploy settings' checks load this
    file from a package folder that holds no worker/ (rkmon_lib/deployment.py rk_ids).
    """
    return types.SimpleNamespace(
        **runpy.run_path(os.path.join(CODE, "worker", "perms.py"))
    )


def whole_file() -> types.SimpleNamespace:
    """rkmon_lib/whole_file.py, the whole-file writes' home, read when a start writes its record (as `perms`)."""
    return types.SimpleNamespace(
        **runpy.run_path(os.path.join(CODE, "rkmon_lib", "whole_file.py"))
    )


MONITOR, WRITER = LAYOUT.MONITOR.name, LAYOUT.WRITER.name
RUNNER_IDS = frozenset(u.uid for u in LAYOUT.PR_RUNNERS)
RUNNER_GIDS = frozenset(u.gid for u in LAYOUT.PR_RUNNERS)  # each runner's own group's
OURS = tuple(u.name for u in LAYOUT.USERS)
VOLUMES = LAYOUT.VOLUMES  # the work volume, the secrets, the write login
RECORD = f"{LAYOUT.WORK_DIR}/.rkmon_ids.json"
PASSWD, GROUP = "/etc/passwd", "/etc/group"
ENV = {
    "uid": "RKMON_UID",
    "gid": "RKMON_GID",
}  # the ids a deployment sets: the monitor's user and group
IDS = (
    "uid",
    "gid",
    "write_uid",
)  # the ids a start gives (wanted): the deployment's, then the writer's, the package's
USER_OF = {
    "uid": MONITOR
}  # the user this renumbers: the writer and the PR runners keep the image's ids
HOMED = (
    MONITOR,
    WRITER,
)  # the users whose homes follow a renumbering: the monitor, and the writer in its group
RUNNERS_KEY = "pr_runners"  # in the record: the volumes have the runners' layout (the one-time move is done)
SYSTEM = ("nobody", "nfsnobody", "nogroup")
MAX_ID = 2**31 - 1
FIRST_ID = 1000  # the first id of a regular account (Debian's and the RHEL family's): below it, the system's
NOBODY = 65534  # nobody and nogroup
# The deploy settings' keys of the ids a deployment sets: the monitor's user and group.
SETTING = {"uid": "host.users.monitor_uid", "gid": "host.users.monitor_gid"}


class Refused(Exception):
    pass


def system_id(value: int) -> bool:
    """An id that is a system account's on any base the image builds on: below FIRST_ID, or nobody's. None of the
    users takes one (the monitor's group may be such a group: an existing group is taken as it is).
    """
    return value < FIRST_ID or value == NOBODY


def read_db(passwd: str, group: str) -> tuple[dict, dict]:
    """({user: (uid, gid)}, {group: gid}) from /etc/passwd's and /etc/group's text."""
    users, groups = {}, {}
    for line in passwd.splitlines():
        f = line.split(":")
        if len(f) >= 4 and f[2].isdigit() and f[3].isdigit():
            users[f[0]] = (int(f[2]), int(f[3]))
    for line in group.splitlines():
        f = line.split(":")
        if len(f) >= 3 and f[2].isdigit():
            groups[f[0]] = int(f[2])
    return users, groups


def homes(passwd: str) -> tuple:
    """The homes /etc/passwd's text gives HOMED's users: the image's, which its useradd made."""
    names = set(HOMED)
    out = []
    for line in passwd.splitlines():
        f = line.split(":")
        if len(f) >= 6 and f[0] in names and f[5].startswith("/") and f[5] != "/":
            out.append(f[5])
    return tuple(out)


def current(users: dict, groups: dict) -> dict:
    """The ids the image's users have now (IDS)."""
    return {
        "uid": users[MONITOR][0],
        "gid": users[MONITOR][1],
        "write_uid": users[WRITER][0],
    }


def fix(key: str, what: str) -> str:
    """What a refusal of the monitor's id `key` (ENV's keys) says to do: set it to `what` in the deploy settings."""
    return f"set {SETTING[key]} to {what} in your deploy settings"


class Clash(NamedTuple):
    """An id that would mix the users: `key`, the wanted id's (ENV's keys: the monitor's or its group's); `why`, whose id
    it is already, in words that follow "is"; `what`, the id to take instead. The entry point refuses it (wanted); the
    deploy settings' checks word it with their own keys."""

    key: str
    why: str
    what: str


def span(ids) -> str:
    """A range of the package's ids, in words."""
    return f"{min(ids)} to {max(ids)}, the package's"


def clashes(want: dict) -> list:
    """Each of the monitor's ids in `want` (IDS: the monitor's `uid` and `gid`, None for one not known, which is checked
    against nothing, and the writer's `write_uid`) that would mix the users, with the first reason found for it, in
    IDS's order: the one home of these rules. The writer's and the PR runners' ids are the package's, which never clash
    (tests/test_entrypoint.py): each clash is the monitor's user's or its group's, the ids a deployment sets.
    """
    uid, gid, write = (want.get(k) for k in IDS)
    found = {}

    def clash(key: str, why: str, what: str) -> None:
        found.setdefault(key, Clash(key, why, what))

    if uid is not None:
        if system_id(uid):
            clash(
                "uid",
                f"a system account's id (below {FIRST_ID}, or nobody's {NOBODY})",
                f"an id of {FIRST_ID} or more",
            )
        if uid in RUNNER_IDS:
            clash("uid", f"a PR runner's id ({span(RUNNER_IDS)})", "another id")
        if uid == write:
            clash("uid", "the GitHub writer's id", "another id")
    if gid is not None:
        if gid == LAYOUT.PR_GID:
            clash("gid", "the PR runners' group's id", "another group's id")
        if gid in RUNNER_GIDS:
            clash(
                "gid",
                f"a PR runner's own group's id ({span(RUNNER_GIDS)})",
                "another group's id",
            )
    return [found[k] for k in IDS if k in found]


def settings_ids(uid, gid) -> dict:
    """The ids a start gives (IDS), as clashes takes them: the monitor's `uid` and `gid` (None: not known), and the
    GitHub writer's, the package's, which no setting changes."""
    return {"uid": uid, "gid": gid, "write_uid": LAYOUT.WRITER.uid}


def wanted(env: dict, have: dict) -> dict:
    """The ids a start gives (settings_ids): the deployment's (ENV), each unset one as the image has it, and the
    writer's, the package's; Refused when they cannot be: a value that is no id, or the first of their clashes.
    """
    out = settings_ids(have["uid"], have["gid"])
    for k, var in ENV.items():
        v = str(env.get(var) or "").strip()
        if not v:
            continue
        if not v.isdigit() or not 1 <= int(v) <= MAX_ID:
            raise Refused(f"{var}={v} is not a user id (a number from 1 to {MAX_ID})")
        out[k] = int(v)
    found = clashes(out)
    if found:
        c = found[0]
        raise Refused(f"{ENV[c.key]} ({out[c.key]}) is {c.why}: " + fix(c.key, c.what))
    return out


def plan(users: dict, groups: dict, want: dict) -> list:
    """The commands that give the monitor and its group the wanted ids (groupmod, usermod, and userdel of a base image's
    account on a wanted id); [] when they have them. The writer's group is the monitor's: it follows.
    """
    have = current(users, groups)
    cmds = []

    def holder(table: dict, value: int, skip) -> str:
        for n, v in table.items():
            if n not in skip and (v[0] if isinstance(v, tuple) else v) == value:
                return n
        return ""

    def check_system(name: str, value: int, key: str) -> None:
        if system_id(value) or name in SYSTEM:
            raise Refused(
                f"{ENV[key]} ({value}) is the image's system account {name}: "
                + fix(key, "another id")
            )

    # The monitor's group: renumbered, or a group of the image that holds the id already, taken as it is.
    if want["gid"] != have["gid"]:
        mine = holder(groups, have["gid"], ()) or MONITOR
        other = holder(groups, want["gid"], {mine})
        if other:
            cmds += [["usermod", "-g", str(want["gid"]), u] for u in (MONITOR, WRITER)]
        else:
            cmds.append(["groupmod", "-g", str(want["gid"]), mine])
    # The monitor's user. No other of ours is on its id: clashes refused that.
    moving = [k for k in USER_OF if want[k] != have[k]]
    for k in moving:
        other = holder(users, want[k], OURS)
        if other:
            check_system(other, want[k], k)
            cmds.append(["userdel", "-r", other])
    cmds += [["usermod", "-u", str(want[k]), USER_OF[k]] for k in moving]
    return cmds


def last_ids(have: dict, last: dict) -> dict:
    """The ids the volumes' files were last given (IDS): the record's, LAST; none recorded, the image's, HAVE."""
    return last if all(isinstance(last.get(k), int) for k in IDS) else have


def taken(before: dict, want: dict) -> dict:
    """The ids of BEFORE (last_ids) that differ from WANT's, the ids a start gives, and that another user or group of
    the container holds, {key: id}: as a user's, root's or a system account's (system_id), a PR runner's, the monitor's
    or the GitHub writer's; as a group's, root's, the runners' group's or a runner's own (the monitor's group may be a
    system group: plan takes the image's group of its id as it is). The record is the monitor's to write: no file of
    such an id moves (file_maps), whatever the record names.
    """
    out = {}
    for k in IDS:
        v = before[k]
        if v == want[k]:
            continue
        if k == "gid":
            held = v == 0 or v == LAYOUT.PR_GID or v in RUNNER_GIDS
        else:
            held = (
                system_id(v) or v in RUNNER_IDS or v in (want["uid"], want["write_uid"])
            )
        if held:
            out[k] = v
    return out


def file_maps(have: dict, want: dict, last: dict) -> tuple[dict, dict]:
    """({old uid: new uid}, {old gid: new gid}) for the volumes' files: from the ids they were last given (last_ids),
    when those differ from the wanted ones, the writer's included (a record may hold another id of its), but for one
    another user or group of the container holds (taken); and, until the record says the volumes have the runners'
    layout, the single PR-code user's files to the monitor and its group to the runners'. That user's id is the
    package's: the first runner's (container_layout's PR_RUNNER), its group numbered as it.
    """
    before = last_ids(have, last)
    skip = taken(before, want)
    uids = {
        before[k]: want[k]
        for k in ("uid", "write_uid")
        if before[k] != want[k] and k not in skip
    }
    gids = (
        {before["gid"]: want["gid"]}
        if before["gid"] != want["gid"] and "gid" not in skip
        else {}
    )
    if not last.get(RUNNERS_KEY):
        old = LAYOUT.PR_RUNNER.uid
        if old not in (want["uid"], want["write_uid"]):
            uids.setdefault(old, want["uid"])
        if old not in (want["gid"], LAYOUT.PR_GID):
            gids.setdefault(old, LAYOUT.PR_GID)
    return uids, gids


def remap_files(roots, uids: dict, gids: dict) -> int:
    """Each file under ROOTS owned by an old id, given the new one (owner and group; a link itself). How many."""
    if not uids and not gids:
        return 0
    n = 0

    def one(p: str) -> None:
        nonlocal n
        try:
            st = os.lstat(p)
        except OSError:
            return
        u, g = uids.get(st.st_uid, -1), gids.get(st.st_gid, -1)
        if u != -1 or g != -1:
            try:
                os.lchown(p, u, g)
                n += 1
            except OSError:
                pass

    for root in roots:
        if not os.path.isdir(root):
            continue
        one(root)
        for dirpath, dirnames, files in os.walk(root):
            for name in dirnames + files:
                one(os.path.join(dirpath, name))
    return n


def root_owners(want: dict) -> dict:
    """The owner each volume's own folder must have: the monitor's for /work and /secrets, the write user's for
    /secrets-write, each in the monitor's group. Docker gives an empty named volume's folder the image's owner again at
    each new container (its copy of the image's folder), whatever the volume's files say: these are set at every start.
    """
    work, secrets, write = VOLUMES
    return {
        work: (want["uid"], want["gid"]),
        secrets: (want["uid"], want["gid"]),
        write: (want["write_uid"], want["gid"]),
    }


def fresh_roots(have: dict, want: dict) -> dict:
    """Each volume's own folder that still has the image's own owner (root_owners of HAVE, the image's ids), which
    docker gave it, and that WANT, the ids a start gives, owns otherwise: with the owner it must have (root_owners).
    """
    image, owners = root_owners(have), root_owners(want)
    out = {}
    for path, owner in image.items():
        try:
            st = os.lstat(path)
        except OSError:
            continue
        if (st.st_uid, st.st_gid) == owner and owner != owners[path]:
            out[path] = owners[path]
    return out


def fix_roots(owners: dict) -> int:
    """Each folder of `owners` that has another owner, given its own (its mode kept). How many."""
    n = 0
    for path, (u, g) in owners.items():
        try:
            st = os.lstat(path)
        except OSError:
            continue
        if (st.st_uid, st.st_gid) != (u, g):
            try:
                os.lchown(path, u, g)
                n += 1
            except OSError:
                pass
    return n


def recorded(path: str = RECORD) -> dict:
    """The record at PATH, {} when there is none. Its folder is the monitor's: a regular file alone is read, opened
    never through a link and without waiting on a pipe; any other kind of file at PATH (a link, a pipe, a folder) is
    no record, said in one line.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return {}
    except OSError:  # a link: O_NOFOLLOW opens none
        fd = None
    try:
        if fd is None or not stat.S_ISREG(os.fstat(fd).st_mode):
            print(
                f"rk_ids: {path} is not a regular file: read as no record",
                file=sys.stderr,
            )
            return {}
        with os.fdopen(fd, "rb") as f:
            fd = None  # f closes it
            v = json.loads(f.read())
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}
    finally:
        if fd is not None:
            os.close(fd)


def write_record(record: dict, uid: int, gid: int) -> None:
    """RECORD written whole (rkmon_lib/whole_file.py: a new file beside it, made new and never through a link, given
    its owner by its descriptor, then renamed onto it), the user UID's and the group GID's, at the mode of a file the
    monitor shares. The folder is the monitor's: root follows no link there. Nothing on a failure: the next start
    writes it again.
    """
    with contextlib.suppress(OSError):
        whole_file().write(
            RECORD, json.dumps(record), perms().SHARED_FILE_MODE, owner=(uid, gid)
        )


def main(argv: list) -> int:
    with open(PASSWD, encoding="utf-8") as f:
        passwd = f.read()
    with open(GROUP, encoding="utf-8") as f:
        group = f.read()
    users, groups = read_db(passwd, group)
    have = current(users, groups)
    try:
        want = wanted(os.environ, have)
        cmds = plan(users, groups, want)
    except Refused as exc:
        print(f"rk_ids: {exc}: nothing changed", file=sys.stderr)
        return 2
    users_only = (
        "--users" in argv
    )  # a one-off container of the logins' walk, or one that only reads: see the module's docstring
    if users_only:
        uids, gids, roots = {}, {}, fresh_roots(have, want)
    else:
        last = recorded(RECORD)
        uids, gids = file_maps(have, want, last)
        roots = root_owners(want)
        skipped = taken(last_ids(have, last), want)
        if skipped:
            named = ", ".join(f"{k} {v}" for k, v in skipped.items())
            print(
                f"rk_ids: {RECORD} names ids of another user or group ({named}): no file of them changes owner",
                file=sys.stderr,
            )
    if "--plan" in argv:
        for c in cmds:
            print(" ".join(c))
        print(f"files: owners {uids or 'kept'}, groups {gids or 'kept'}")
        return 0
    for c in cmds:
        p = subprocess.run(
            c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        if p.returncode:
            print(
                f"rk_ids: {' '.join(c)} failed: {p.stdout.strip()[:300]}",
                file=sys.stderr,
            )
            return 1
    moved = remap_files(VOLUMES, uids, gids) + fix_roots(roots)
    if (
        cmds
    ):  # the users' homes are the image's, with its ids, in every new container (usermod moves the owner only)
        remap_files(
            homes(passwd),
            {have[k]: want[k] for k in USER_OF if have[k] != want[k]},
            {have["gid"]: want["gid"]} if have["gid"] != want["gid"] else {},
        )
    if users_only:
        return 0
    record = dict(want, **{RUNNERS_KEY: LAYOUT.SLOTS})
    if moved or (cmds and last != record):
        print(
            f"rk_ids: the users' ids are the deployment's: {MONITOR} {want['uid']}:{want['gid']}, {WRITER} "
            f"{want['write_uid']}"
            + (f"; {moved} files of the volumes given the new ids" if moved else "")
        )
    if os.path.isdir(os.path.dirname(RECORD)) and last != record:
        write_record(record, want["uid"], want["gid"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
