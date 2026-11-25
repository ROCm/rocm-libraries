#!/usr/bin/python3 -I
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The working volume's modes, the RAM disk's folder with the PR runners' slots, and the system's temporary folders,
given at each start, as root:
docker/entrypoint.sh runs it right after docker/rk_ids.py, for every command but the health check, code-install and
code-back. The logins' walk's rk command, in a one-off container beside the running one, gets that container's own
temporary folders alone (--system-tmp, below), and none of the volumes' modes, which the running container keeps.
It takes no argument but --plan and --system-tmp, and reads no environment: every path, user and mode is the image's
(rkmon_lib/container_layout.py and worker/perms.py, read from the image's code).

The working folder's folders, by kind (by their names in the default layout: rkmon_lib/paths.py UNDER_WORKING and
worktree, rkmon_lib/deploy.py venv_commands):
- the working folder itself: the monitor's, group rkmon, 2771 (a PR runner passes through it to the trees it reads,
  and lists nothing), its default ACL SHARED; the files in it are shared ones;
- TREES, the trees PR code reads (the clone, the worker tools, the probes, the virtual environments, the worktrees),
  and CACHES in the cache folder (the develop-side results, the mutation history and the CTest listings of
  tools/lane_of.py, which the monitor writes and a job reads): the monitor's, group rkpr, 2750, no ACL. MADE and
  CACHES are made empty when missing, so the clone and the worker tools go into a folder of that kind;
- SCRATCH (the review directories, the jobs' scratch folders): shared, 2771, its default ACL SCRATCH: the runners'
  group enters each folder made there, and lists, reads and writes none. A runner reads only the trees made there
  (worker/perms.py tree_folder: the job scripts' prtree, and rk pr-tree for the worktrees the session's agents add);
  they keep their modes;
- CACHE: shared, 2771, so a runner reaches CACHES in it;
- every other folder (the state, the logs, Claude's folder, the outputs): shared: group rkmon, 2770, default ACL
  SHARED.

The RAM disk's folder (container_layout RAMDISK: the deployment's tmpfs, else a folder of the container's own, made
when missing) is laid out as the working folder: the monitor's, group rkmon, 2771, its default ACL SHARED (a PR
runner passes through it to its slot and to the trees it reads, and writes nothing at its top); SCRATCH is made in it
(paths.tmp is there while the RAM disk is on), and its folders and files get their kinds' modes (the PRs' worktrees
there are trees), the slots apart. The slots (SLOTS_DIR): root's, 0755, no ACL. Each runner's slot folder: the
runner's, group rkmon, 2751, no ACL (made when missing; what it holds is never walked or emptied here: the launcher,
docker/rk_prcode.py, empties it as its user, and a kept slot's build stays). The records' folder (SLOT_LOCKS):
shared, 2770. The inputs (SLOT_INPUTS): root's, group rkpr, 0750, holding a folder per runner: the monitor's, group
rkpr, 2750, no ACL.

The system's temporary folders (container_layout SYSTEM_TMP: /tmp and /var/tmp): root's, group rkmon, sticky, 1770, no
ACL: the monitor, the writer and root write them, and no PR runner writes or reads them (a job's TMPDIR is its slot's
`tmp`). What is in them is left as it is. The image takes others' writing off them (docker/Dockerfile), so a container
without this step has no temporary folder its users may write: --system-tmp gives them these modes alone, and says
nothing (what the walk's command prints is the walk's to read). /dev/shm and /dev/mqueue stay every user's: the
launcher, docker/rk_prcode.py, removes a runner's own entries in them.

A top folder whose group, mode or ACLs are not its kind's is walked, by descriptors and never through a link, and each
folder and regular file in it that the monitor or the writer owns gets its kind's group and modes: the group's bits
are the owner's (shared) or the owner's reading and entering (trees), and none where the group had none (a private
file stays private); others get none; folders get the setgid bit and their kind's default ACL; SCRATCH's entries get
an access ACL by which the runners' group enters its folders and reads none of its files, the others none, and a tree
in it (worker/perms.py is_tree) is left as it is, with all it holds. An entry another user owns is left as it is. So a
volume from before these modes gets them once, and a later start walks nothing.

A file system that keeps no ACL: a line says so, and the modes alone are set (the files the monitor shares are then
0640: the writer reads them, and writes none). Exit 0, also when a folder is a link or no folder (a line names it and
it is left as it is; a RAM disk's folder that is one gets no slots); 1 when the image has no such users.

  rk_layout.py               give the working folder, the RAM disk's folder and the system's temporary folders their
                             modes, and make the slots (as root)
  rk_layout.py --system-tmp  the system's temporary folders alone, saying nothing (the logins' walk's one-off)
  rk_layout.py --plan        print what it would change, change nothing; with --system-tmp, for those folders alone
"""

from __future__ import annotations

import errno
import fnmatch
import grp
import os
import pwd
import runpy
import stat
import sys
import types

CODE = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)  # the image's code: read-only, owned by root
LAYOUT = types.SimpleNamespace(
    **runpy.run_path(os.path.join(CODE, "rkmon_lib", "container_layout.py"))
)
PERMS = types.SimpleNamespace(
    **runpy.run_path(os.path.join(CODE, "worker", "perms.py"))
)
PATHS = types.SimpleNamespace(
    **runpy.run_path(os.path.join(CODE, "rkmon_lib", "paths.py"))
)
TREES = ("repo", "tools", "probes", "venv", "venv-*", "wt*")
MADE = ("repo", "tools", "probes")
SCRATCH = "tmp"
CACHE = "cache"
CACHES = PATHS.CACHES
SHARED, TREE, SCRATCHED = "shared", "tree", "scratch"
PASS_MODE = 0o2771  # the working folder, SCRATCH and CACHE: a runner passes through, and lists nothing
SLOTS_MODE = 0o755
INPUTS_MODE = 0o750
SYSTEM_TMP_MODE = 0o1770  # /tmp and /var/tmp: root's, the monitor's group's, sticky; nothing for a PR runner
_OPEN_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_OPEN_ANY = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY | os.O_CLOEXEC


def ids() -> types.SimpleNamespace:
    """The ids the image's users have now (after rk_ids.py): root, monitor, group (the monitor's), writer, pr (the
    runners' group), runners ({name: uid})."""
    mon = pwd.getpwnam(LAYOUT.MONITOR.name)
    return types.SimpleNamespace(
        root=0,
        monitor=mon.pw_uid,
        group=mon.pw_gid,
        writer=pwd.getpwnam(LAYOUT.WRITER.name).pw_uid,
        pr=grp.getgrnam(LAYOUT.PR_GROUP).gr_gid,
        runners={u.name: pwd.getpwnam(u.name).pw_uid for u in LAYOUT.PR_RUNNERS},
    )


def _acl(get) -> tuple | None:
    try:
        return get()
    except OSError as exc:
        if exc.errno in (errno.ENODATA, errno.EOPNOTSUPP, errno.ENOTSUP):
            return None
        raise


class Layout:
    """The modes of one working folder (`work`) for the users `who` (ids()); `say` gets each line; `plan`: change
    nothing, say what would change; `tmp_dirs`: the system's temporary folders, closed to the PR runners; `ramdisk`: the
    RAM disk's folder, which holds the slots ("": no slots).
    """

    def __init__(
        self,
        work: str,
        who,
        say=print,
        plan: bool = False,
        tmp_dirs=(),
        ramdisk: str = "",
    ):
        self.work, self.who, self.say, self.plan = (
            work.rstrip("/") or "/",
            who,
            say,
            plan,
        )
        self.tmp_dirs = tuple(tmp_dirs)
        self.ramdisk = ramdisk.rstrip("/")
        self.acls = True  # the file system keeps ACLs, until one is refused
        self.owners = {who.monitor, who.writer}

    # -- what an entry of a kind gets -------------------------------------------------------------------------------
    def gid(self, kind: str) -> int:
        return self.who.pr if kind == TREE else self.who.group

    @staticmethod
    def mode(have: int, kind: str, folder: bool) -> int:
        """The mode an entry of `kind` gets, from its mode `have`: the owner's bits kept; the group's from them
        (reading and entering only in a tree), none when the group had none; none for others; setgid on a folder.
        """
        u = (have >> 6) & 7
        g = 0 if not have & 0o070 else (u & 5 if kind == TREE else u)
        return (u << 6) | (g << 3) | (stat.S_ISGID if folder else 0)

    def default_acl(self, kind: str, mode: int):
        """The default ACL of a folder of `kind` with `mode`; None for a tree and a private folder."""
        if kind == TREE or not mode & 0o070:
            return None
        return PERMS.scratch(self.who.pr) if kind == SCRATCHED else PERMS.SHARED

    def access_acl(self, kind: str, mode: int):
        """The access ACL of an entry of `kind` with `mode`; None: none. SCRATCH's, as its default ACL gives one: the
        runners' group's x, which the mask (the mode's group bits) lets through on a folder and on a program, so a
        runner enters the folder and reads no file."""
        if kind != SCRATCHED or not mode & 0o070:
            return None
        u, g = (mode >> 6) & 7, (mode >> 3) & 7
        text = PERMS._text
        return (
            ("user_obj", None, text(u)),
            ("group_obj", None, text(g)),
            ("group", self.who.pr, "x"),
            ("mask", None, text(g)),
            ("other", None, text(mode & 7)),
        )

    # -- one open entry ---------------------------------------------------------------------------------------------
    def _set_acl(self, fd: int, name: str, entries) -> None:
        """Set (`entries`) or remove (None) one of the entry's ACLs; a file system that keeps none is said once."""
        if not self.acls:
            return
        try:
            if entries is None:
                os.removexattr(fd, name)
            else:
                os.setxattr(fd, name, PERMS.acl_bytes(entries))
        except OSError as exc:
            if exc.errno == errno.ENODATA:
                return
            if exc.errno not in (errno.EOPNOTSUPP, errno.ENOTSUP):
                raise
            self.acls = False
            self.say(
                f"rk_layout: {self.work}'s file system keeps no ACL: the permission bits alone are set, so the "
                "files the monitor shares are 0640 (the writer reads them, and writes none)"
            )

    def differs(self, fd: int, uid: int, gid: int, mode: int, default, access) -> bool:
        st = os.fstat(fd)
        if (uid != -1 and st.st_uid != uid) or st.st_gid != gid:
            return True
        if stat.S_IMODE(st.st_mode) != mode:
            return True
        if not self.acls:
            return False
        have = _acl(lambda: PERMS.default_acl(fd)) if stat.S_ISDIR(st.st_mode) else None
        if (have is None) != (default is None) or (
            default is not None and not PERMS.same_acl(have, default)
        ):
            return True
        have = _acl(lambda: PERMS.acl_entries(os.getxattr(fd, PERMS.ACCESS)))
        return (have is None) != (access is None) or (
            access is not None and not PERMS.same_acl(have, access)
        )

    def give(
        self, fd: int, uid: int, gid: int, mode: int, default=None, access=None
    ) -> bool:
        """Give the open entry `fd` its owner (-1 keeps it), group, mode and ACLs; whether it had other ones."""
        if not self.differs(fd, uid, gid, mode, default, access):
            return False
        if self.plan:
            return True
        st = os.fstat(fd)
        if access is None:
            self._set_acl(fd, PERMS.ACCESS, None)
        if (uid != -1 and st.st_uid != uid) or st.st_gid != gid:
            os.fchown(fd, uid, gid)
        os.fchmod(fd, mode)
        if stat.S_ISDIR(st.st_mode):
            self._set_acl(fd, PERMS.DEFAULT, default)
        if access is not None:
            self._set_acl(fd, PERMS.ACCESS, access)
        return True

    def entry(self, fd: int, kind: str) -> bool:
        """An entry of a walked folder: given its kind's modes when the monitor or the writer owns it."""
        st = os.fstat(fd)
        folder = stat.S_ISDIR(st.st_mode)
        if st.st_uid not in self.owners or not (folder or stat.S_ISREG(st.st_mode)):
            return False
        mode = self.mode(st.st_mode, kind, folder)
        return self.give(
            fd,
            -1,
            self.gid(kind),
            mode,
            self.default_acl(kind, mode) if folder else None,
            self.access_acl(kind, mode),
        )

    def tree_in(self, dfd: int, name: str) -> bool:
        """Whether the folder `name` in the open folder `dfd` is a tree PR code reads (worker/perms.py is_tree), read
        without following a link."""
        try:
            st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
        except OSError:
            return False
        return PERMS.is_tree(st, self.who.pr)

    def walk(self, fd: int, kind: str, skip=()) -> int:
        """Each folder and regular file under the open folder `fd` (not itself), by descriptors, never through a link,
        the folders `skip` names at its top left out, and in SCRATCH the trees PR code reads with all they hold: given
        its kind's modes. How many changed.
        """
        n = 0
        top = True
        for _, dirs, files, dfd in os.fwalk(".", dir_fd=fd, follow_symlinks=False):
            if top:
                dirs[:] = [d for d in dirs if d not in skip]
                top = False
            else:
                n += self.entry(dfd, kind)
            if kind == SCRATCHED:  # a job's or an agent's tree keeps its modes
                dirs[:] = [d for d in dirs if not self.tree_in(dfd, d)]
            for name in files:
                try:
                    f = os.open(name, _OPEN_ANY, dir_fd=dfd)
                except OSError:  # a link (ELOOP), a socket, gone
                    continue
                try:
                    n += self.entry(f, kind)
                finally:
                    os.close(f)
        return n

    # -- the folders ------------------------------------------------------------------------------------------------
    def open_dir(self, name: str, dir_fd: int, path: str, make_as=None):
        """An open descriptor of folder `name` (in `dir_fd`; `path` names it), never through a link. When missing and
        `make_as` is (uid, gid, mode): made so. None when missing, or a link or no folder (a line says so).
        """
        try:
            return os.open(name, _OPEN_DIR, dir_fd=dir_fd)
        except FileNotFoundError:
            if make_as is None:
                return None
        except OSError as exc:
            self.say(
                f"rk_layout: {path} is a link or no folder ({exc.strerror}): left as it is"
            )
            return None
        if self.plan:
            self.say(f"rk_layout: {path} would be made")
            return None
        try:
            os.mkdir(name, 0o700, dir_fd=dir_fd)
        except FileExistsError:
            pass
        fd = self.open_dir(name, dir_fd, path)
        if fd is not None:
            uid, gid, mode = make_as
            st = os.fstat(fd)
            if st.st_uid == os.geteuid() and not os.listdir(fd):  # the one just made
                self._set_acl(fd, PERMS.ACCESS, None)
                self._set_acl(fd, PERMS.DEFAULT, None)
                os.fchown(fd, uid, gid)
                os.fchmod(fd, mode)
        return fd

    def top(
        self, fd: int, path: str, kind: str, mode: int | None = None, skip=()
    ) -> None:
        """A top folder of `kind` (mode `mode`, else its kind's from its own): walked, and given its modes, when it
        does not have them."""
        st = os.fstat(fd)
        want = mode if mode is not None else self.mode(st.st_mode, kind, True)
        default = self.default_acl(kind, want)
        access = None if mode is not None else self.access_acl(kind, want)
        if not self.differs(fd, -1, self.gid(kind), want, default, access):
            return
        if self.plan:
            self.say(
                f"rk_layout: {path} would be given the permissions of a {kind} folder"
            )
            return
        n = self.walk(fd, kind, skip)
        self.give(fd, -1, self.gid(kind), want, default, access)
        self.say(
            f"rk_layout: {path}: a {kind} folder ({n} entries in it given their permissions)"
        )

    @staticmethod
    def kind_of(name: str) -> str:
        if name == SCRATCH:
            return SCRATCHED
        if any(fnmatch.fnmatchcase(name, p) for p in TREES):
            return TREE
        return SHARED

    def entries(self, fd: int, path: str, skip=(), cache: bool = False) -> None:
        """Each entry of the open folder `fd` (`path` names it) but those `skip` names: a file a shared one, a folder a
        top folder of its name's kind; with `cache`, CACHE is the cache folder."""
        for name in sorted(os.listdir(fd)):
            if name in skip:
                continue
            try:
                sub = os.open(name, _OPEN_ANY, dir_fd=fd)
            except OSError:  # a link, a socket, gone
                continue
            try:
                if not stat.S_ISDIR(os.fstat(sub).st_mode):
                    self.entry(sub, SHARED)  # a file of the folder: a shared one
                elif cache and name == CACHE:
                    self.cache(sub, f"{path}/{name}")
                else:
                    kind = self.kind_of(name)
                    self.top(
                        sub,
                        f"{path}/{name}",
                        kind,
                        PASS_MODE if kind == SCRATCHED else None,
                    )
            finally:
                os.close(sub)

    def work_folder(self) -> None:
        """The working folder, and the folders and files in it."""
        try:
            fd = os.open(self.work, _OPEN_DIR)
        except OSError as exc:
            self.say(f"rk_layout: {self.work}: {exc.strerror}: nothing to do")
            return
        try:
            if os.fstat(fd).st_uid != self.who.monitor:
                self.say(
                    f"rk_layout: {self.work} is not the monitor's (rk_ids.py gives it its owner): left as it is"
                )
                return
            self.give(fd, -1, self.who.group, PASS_MODE, PERMS.SHARED)
            mon, tree_mode = self.who.monitor, PERMS.TREE_DIR_MODE
            for name in MADE + (SCRATCH, CACHE):
                kind = self.kind_of(name)
                made = self.open_dir(
                    name,
                    fd,
                    f"{self.work}/{name}",
                    (mon, self.gid(kind), tree_mode if kind == TREE else PASS_MODE),
                )
                if made is not None:
                    os.close(made)
            self.entries(fd, self.work, cache=True)
        finally:
            os.close(fd)

    def cache(self, fd: int, path: str) -> None:
        """The cache folder (shared, 2771) and CACHES in it (trees)."""
        self.top(fd, path, SHARED, PASS_MODE, skip=CACHES)
        for name in CACHES:
            sub = self.open_dir(
                name,
                fd,
                f"{path}/{name}",
                (self.who.monitor, self.who.pr, PERMS.TREE_DIR_MODE),
            )
            if sub is not None:
                try:
                    self.top(sub, f"{path}/{name}", TREE)
                finally:
                    os.close(sub)

    def ramdisk_folder(self) -> None:
        """The RAM disk's folder (made when missing), SCRATCH in it, and its folders and files but the slots."""
        if not self.ramdisk:
            return
        parent, name = os.path.split(self.ramdisk)
        try:
            pfd = os.open(parent or "/", _OPEN_DIR)
        except OSError as exc:
            self.say(f"rk_layout: {parent}: {exc.strerror}: no RAM disk folder made")
            return
        try:
            fd = self.open_dir(
                name, pfd, self.ramdisk, (self.who.monitor, self.who.group, PASS_MODE)
            )
        finally:
            os.close(pfd)
        if fd is None:
            return
        try:
            if self.give(fd, self.who.monitor, self.who.group, PASS_MODE, PERMS.SHARED):
                self.say(
                    f"rk_layout: {self.ramdisk}: the monitor's, group {LAYOUT.MONITOR.group}, {PASS_MODE:o}: "
                    "a PR runner passes through it"
                )
            made = self.open_dir(
                SCRATCH,
                fd,
                f"{self.ramdisk}/{SCRATCH}",
                (self.who.monitor, self.who.group, PASS_MODE),
            )
            if made is not None:
                os.close(made)
            self.entries(fd, self.ramdisk, skip=(os.path.basename(LAYOUT.SLOTS_DIR),))
        finally:
            os.close(fd)

    def slots(self) -> None:
        """The slots' folder in the RAM disk's, each runner's slot folder, the records' folder and the inputs."""
        if not self.ramdisk:
            return
        try:
            rfd = os.open(self.ramdisk, _OPEN_DIR)
        except OSError:
            return
        path = f"{self.ramdisk}/{os.path.basename(LAYOUT.SLOTS_DIR)}"
        try:
            fd = self.open_dir(
                os.path.basename(LAYOUT.SLOTS_DIR),
                rfd,
                path,
                (self.who.root, self.who.root, SLOTS_MODE),
            )
        finally:
            os.close(rfd)
        if fd is None:
            return
        try:
            if self.give(fd, self.who.root, self.who.root, SLOTS_MODE):
                self.say(f"rk_layout: {path}: root's, {SLOTS_MODE:o}")
            changed = []
            for runner, uid in sorted(self.who.runners.items()):
                s = self.open_dir(
                    runner,
                    fd,
                    f"{path}/{runner}",
                    (uid, self.who.group, LAYOUT.SLOT_MODE),
                )
                if s is not None:
                    try:
                        if self.give(s, uid, self.who.group, LAYOUT.SLOT_MODE):
                            changed.append(runner)
                    finally:
                        os.close(s)
            if changed:
                self.say(
                    f"rk_layout: the slot folders of {', '.join(changed)}: each its runner's, group "
                    f"{LAYOUT.MONITOR.group}, {LAYOUT.SLOT_MODE:o}"
                )
            name = os.path.basename(LAYOUT.SLOT_LOCKS)
            locks = self.open_dir(
                name,
                fd,
                f"{path}/{name}",
                (self.who.monitor, self.who.group, PERMS.SHARED_DIR_MODE),
            )
            if locks is not None:
                try:
                    self.top(locks, f"{path}/{name}", SHARED, PERMS.SHARED_DIR_MODE)
                finally:
                    os.close(locks)
            name = os.path.basename(LAYOUT.SLOT_INPUTS)
            inputs = self.open_dir(
                name, fd, f"{path}/{name}", (self.who.root, self.who.pr, INPUTS_MODE)
            )
            if inputs is not None:
                try:
                    self.give(inputs, self.who.root, self.who.pr, INPUTS_MODE)
                    for runner in sorted(self.who.runners):
                        r = self.open_dir(
                            runner,
                            inputs,
                            f"{path}/{name}/{runner}",
                            (self.who.monitor, self.who.pr, PERMS.TREE_DIR_MODE),
                        )
                        if r is not None:
                            try:
                                self.give(
                                    r,
                                    self.who.monitor,
                                    self.who.pr,
                                    PERMS.TREE_DIR_MODE,
                                )
                            finally:
                                os.close(r)
                finally:
                    os.close(inputs)
        finally:
            os.close(fd)

    def system_tmp(self) -> None:
        """Each of `tmp_dirs`: root's, the monitor's group's, SYSTEM_TMP_MODE, no ACL; never through a link, and what is
        in it left as it is."""
        for path in self.tmp_dirs:
            try:
                fd = os.open(path, _OPEN_DIR)
            except OSError as exc:
                self.say(
                    f"rk_layout: {path} is a link, no folder or missing ({exc.strerror}): left as it is"
                )
                continue
            try:
                if self.give(fd, self.who.root, self.who.group, SYSTEM_TMP_MODE):
                    self.say(
                        f"rk_layout: {path}: root's, group {LAYOUT.MONITOR.group}, {SYSTEM_TMP_MODE:o}: "
                        "no PR runner writes or reads it"
                    )
            finally:
                os.close(fd)

    def apply(self) -> None:
        self.work_folder()
        self.ramdisk_folder()
        self.slots()
        self.system_tmp()


def main(argv: list) -> int:
    try:
        who = ids()
    except KeyError as exc:
        print(
            f"rk_layout: the image has no user or group {exc}: nothing changed",
            file=sys.stderr,
        )
        return 1
    plan = "--plan" in argv
    # the logins' walk's one-off container: its own temporary folders alone, said under --plan only (the docstring)
    if "--system-tmp" in argv:
        Layout(
            LAYOUT.WORK_DIR,
            who,
            say=print if plan else (lambda _line: None),
            plan=plan,
            tmp_dirs=LAYOUT.SYSTEM_TMP,
        ).system_tmp()
        return 0
    Layout(
        LAYOUT.WORK_DIR,
        who,
        plan=plan,
        tmp_dirs=LAYOUT.SYSTEM_TMP,
        ramdisk=LAYOUT.RAMDISK,
    ).apply()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
