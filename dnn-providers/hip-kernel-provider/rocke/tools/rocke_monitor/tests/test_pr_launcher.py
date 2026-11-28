# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The PR-code launcher, docker/rk_prcode.py, outside a container: a temporary folder stands in for a
slot, the test's own user for its runner, and the test's own children for the runner's processes. No test lets it
signal every process or remove the machine's System V objects: kill(-1) and IPC_RMID are for the image's runner ids
only, and each test that could reach them forbids them, or lists only an object of its own.

- PR code gets only what the lists name (worker/pr_env.py): a planted X_PASSWORD or X_KEY, a login folder, a token, a
  key or a variable of the monitor's own does not reach it; HOME, TMPDIR, RKMON_OUT and RKMON_BUILD are its slot's.
- Its git fetches nothing, and takes as safe only the job's trees: the one CMD runs in and the ones RKMON_TREES names,
  each a folder of the monitor's; a repository anywhere else, a runner's included, is refused.
- It works as a PR runner only, by name and id, and only a PR runner's id signals every process it may signal.
- A stop sends SIGTERM, then SIGKILL to what is left after the grace, or SIGKILL at once; it waits until none is left,
  and says so when some outlive it.
- What CMD left are the launcher's descendants, by their parents in /proc: a zombie is not one that runs, and the
  root's own entry, read as one of theirs (a pid used again), is walked once. A signal to each of them passes over one
  that has ended or is another user's.
- A slot is used only as root made it: the runner's, group rkmon, mode 2751, no link. `begin` empties it and makes its
  four folders fresh (2750, group rkmon); emptying never follows a link, opens a folder its owner closed, and moves a
  folder past DEPTH_MAX up before it goes on.
- `keep` makes the slot's build, and nothing else, readable by the other runners and writable by none of them, with
  no set-user-ID or set-group-ID file, never through a link, and its other folders fresh: what PR code left beside the
  build, or opened to the other runners, is gone; a kept slot, or one not begun, runs nothing; `end` empties it.
  `begin`, `keep` and `end` remove the ACLs PR code set on the slot's folder (its access ACL only when it has one: a
  runner removing one, even one that is not there, takes the folder's set-group-ID bit, and the slot is then refused),
  and, once the runner's processes are stopped, the runner's own entries in /dev/shm and /dev/mqueue, the folders
  every user writes, never through a link, and the System V objects it owns or made: another user's stay; a line says
  what went.
- `run`: CMD in the slot (its home, or -C DIR) with stdin /dev/null and no new privileges; its exit code, or 128 + the
  signal that ended it;
  127 when it cannot start; a SIGTERM to the launcher reaches CMD; what CMD left running is stopped before the launcher
  returns, and no other process of the runner's (a `kill`, `begin`, `end` or `keep` of the slot runs as the runner
  too); the launcher adopts and reaps what CMD leaves. A launcher the kernel does not make a subreaper runs nothing
  and exits 127 with one line, as one whose monitor's group is missing does.
"""

import contextlib
import ctypes
import errno
import grp
import io
import os
import pathlib
import pwd
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docker"))
sys.dont_write_bytecode = True

import rk_prcode  # noqa: E402

L = rk_prcode.LAYOUT
REAP = rk_prcode.reap
PR_GET_CHILD_SUBREAPER = 37


def forbidden(*_args, **_kwargs):
    raise AssertionError("a signal to every process")


def status(pid: int) -> dict | None:
    """/proc/<pid>/status as a dict; None when the process is gone."""
    try:
        with open(f"/proc/{pid}/status") as f:
            return dict(
                (k, v.strip()) for k, v in (ln.split(":", 1) for ln in f if ":" in ln)
            )
    except OSError:
        return None


def running(pid: int) -> bool:
    st = status(pid)
    return bool(st) and not st["State"].startswith("Z")


def fake_proc(top: pathlib.Path, procs) -> str:
    """A /proc in `top` with these (pid, parent, state) processes, each the test's user's."""
    uid = os.geteuid()
    for pid, parent, state in procs:
        (top / str(pid)).mkdir(parents=True)
        (top / str(pid) / "status").write_text(
            f"Name:\tx\nState:\t{state}\nPPid:\t{parent}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n"
        )
    (top / "self").mkdir(exist_ok=True)
    return str(top)


class Clock:
    """time.monotonic and time.sleep, by hand."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


class TestEnv(unittest.TestCase):
    def test_pr_code_gets_only_what_the_lists_name(self):
        given = {
            "PATH": "/work/venv/bin:/usr/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "RKMON_JOB": "/work/monitor/state/jobs/checks-pr7-1",
            "ROCKE_LLVM_FLAVOR": "llvm22",
            "HOME": "/work/monitor",
            "TMPDIR": "/work/tmp",
            "RKMON_OUT": "/work/monitor/out",
            "RKMON_BUILD": "/work/build",
            "X_PASSWORD": "p",
            "X_KEY": "k",
            "GH_CONFIG_DIR": "/secrets/gh-read",
            "GH_TOKEN": "t",
            "ANTHROPIC_API_KEY": "k",
            "CLAUDE_CONFIG_DIR": "/work/claude/config",
            "ATLASSIAN_CONFIG_DIR": "/secrets/atlassian",
            "RKMON_GH_WRITE_DIR": "/secrets-write/gh",
            "RKMON_PR_RUNNER": "/opt/x/docker/rk_prcode.py",
            "SUDO_USER": "rkmon",
            "LD_PRELOAD": "/work/tmp/x.so",
            "GIT_CONFIG_GLOBAL": "/work/monitor/gitconfig",
            "MY_SECRET": "s",
            "ROCKE_API_KEY": "k",
            "PYTHONDONTWRITEBYTECODE": "0",
        }
        env = rk_prcode.job_env(given, "/ramdisk/slots/rkpr3", "rkpr3")
        self.assertEqual(
            sorted(env),
            sorted(
                [
                    "PATH",
                    "LANG",
                    "LC_ALL",
                    "RKMON_JOB",
                    "ROCKE_LLVM_FLAVOR",
                    "HOME",
                    "TMPDIR",
                    "RKMON_OUT",
                    "RKMON_BUILD",
                    "USER",
                    "LOGNAME",
                    "GIT_NO_LAZY_FETCH",
                    "PYTHONDONTWRITEBYTECODE",
                ]
            ),
        )
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")  # whatever the job's says
        self.assertEqual(
            (
                env["HOME"],
                env["TMPDIR"],
                env["RKMON_OUT"],
                env["RKMON_BUILD"],
                env["USER"],
            ),
            (
                "/ramdisk/slots/rkpr3/home",
                "/ramdisk/slots/rkpr3/tmp",
                "/ramdisk/slots/rkpr3/out",
                "/ramdisk/slots/rkpr3/build",
                "rkpr3",
            ),
        )
        self.assertEqual(env["PATH"], given["PATH"])
        self.assertEqual(
            env["GIT_NO_LAZY_FETCH"], "1"
        )  # the monitor fetched what it reads
        self.assertEqual(
            rk_prcode.job_env({}, "/s/rkpr1", "rkpr1")["PATH"], rk_prcode.PATH
        )

    def trees(self) -> pathlib.Path:
        """A temporary folder with a git tree `wt7m` (a `.git` folder), a link to it, a folder that is no tree and a
        file."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = pathlib.Path(os.path.realpath(tmp.name))
        (d / "wt7m" / ".git").mkdir(parents=True)
        (d / "wt7m" / "rocke" / "platform").mkdir(parents=True)
        (d / "plain").mkdir()
        (d / "link").symlink_to(d / "wt7m")
        (d / "file").write_text("x\n")
        return d

    def test_git_takes_the_jobs_trees_as_safe_and_no_other_folder(self):
        d = self.trees()
        me = os.getuid()
        self.assertEqual(
            rk_prcode.git_top(str(d / "wt7m" / "rocke" / "platform")), str(d / "wt7m")
        )
        self.assertEqual(rk_prcode.git_top(str(d / "link" / "rocke")), str(d / "wt7m"))
        self.assertEqual(rk_prcode.git_top(str(d / "plain")), "")
        given = [
            "",
            "wt7m",  # not absolute
            str(d / "file"),
            str(d / "gone"),
            str(d / "link"),
            str(d / "wt7m"),
            str(d / "plain"),
        ]
        self.assertEqual(
            rk_prcode.trees(given, me), [str(d / "wt7m"), str(d / "plain")]
        )
        # a folder of another user's (a runner's), or no monitor's user to tell: none
        self.assertEqual(rk_prcode.trees(given, me + 1), [])
        self.assertEqual(rk_prcode.trees(given, None), [])
        env = rk_prcode.job_env({}, "/s/rkpr1", "rkpr1", ["/w/wt7m", "/w/repo"])
        self.assertEqual(
            [(k, env[k]) for k in sorted(env) if k.startswith("GIT_CONFIG_")],
            [
                ("GIT_CONFIG_COUNT", "2"),
                ("GIT_CONFIG_KEY_0", "safe.directory"),
                ("GIT_CONFIG_KEY_1", "safe.directory"),
                ("GIT_CONFIG_VALUE_0", "/w/wt7m"),
                ("GIT_CONFIG_VALUE_1", "/w/repo"),
            ],
        )
        self.assertNotIn("GIT_CONFIG_COUNT", rk_prcode.job_env({}, "/s/rkpr1", "rkpr1"))

    @unittest.skipUnless(shutil.which("git"), "no git")
    def test_the_jobs_git_reads_its_trees_and_refuses_any_other_repository(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = pathlib.Path(os.path.realpath(tmp.name))
        (d / "slot" / "home").mkdir(parents=True)
        for name in ("tree", "other"):
            subprocess.run(["git", "init", "-q", str(d / name)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(d / name),
                    "-c",
                    "user.name=x",
                    "-c",
                    "user.email=x@example.com",
                    "commit",
                    "-q",
                    "--allow-empty",
                    "-m",
                    "x",
                ],
                check=True,
            )
        safe = rk_prcode.trees(
            [rk_prcode.git_top(str(d / "tree")), str(d / "tree")], os.getuid()
        )
        env = rk_prcode.job_env(
            {"PATH": os.environ["PATH"]}, str(d / "slot"), "rkpr1", safe
        )
        # git as another user would see them: neither repository is the user's own
        env["GIT_TEST_ASSUME_DIFFERENT_OWNER"] = "1"

        def head(name):
            return subprocess.run(
                ["git", "-C", str(d / name), "rev-parse", "HEAD"],
                env=env,
                capture_output=True,
                text=True,
                cwd=d,
            )

        self.assertEqual(head("tree").returncode, 0, head("tree").stderr)
        p = head("other")
        self.assertEqual(p.returncode, 128)
        self.assertIn("dubious ownership", p.stderr)

    def test_the_environment_is_read_as_data(self):
        self.assertEqual(
            rk_prcode.PR_ENV.parse(b"A=1\0B=x=y\0junk\0C=\xff\0D=\0"),
            {"A": "1", "B": "x=y", "D": ""},
        )


class TestIdentity(unittest.TestCase):
    def test_it_works_as_a_pr_runner_only(self):
        r = L.PR_RUNNERS[2]
        users = {r.uid: r.name, 1000: L.MONITOR.name, 4242: L.PR_RUNNERS[0].name}

        def getpwuid(uid):
            if uid not in users:
                raise KeyError(uid)
            return types.SimpleNamespace(pw_name=users[uid])

        self.assertEqual(rk_prcode.runner(r.uid, getpwuid), r)
        self.assertIsNone(rk_prcode.runner(1000, getpwuid))
        self.assertIsNone(
            rk_prcode.runner(4242, getpwuid)
        )  # a runner's name with another id
        self.assertIsNone(rk_prcode.runner(7, getpwuid))  # no such user
        # This test's own user is no PR runner: nothing is stopped, nothing is touched.
        self.assertIsNone(rk_prcode.runner(os.geteuid()))
        old = os.umask(0o022)
        try:
            with mock.patch.object(rk_prcode, "stop", forbidden), mock.patch.object(
                rk_prcode, "open_slot", forbidden
            ), contextlib.redirect_stderr(io.StringIO()) as err:
                for argv in (["begin"], ["run", "--", "true"], ["kill"]):
                    self.assertEqual(rk_prcode.main(argv), 2)
            self.assertEqual(os.umask(0o022), 0o022)
        finally:
            os.umask(old)
        self.assertIn("runs as a PR runner only", err.getvalue())

    def test_only_a_pr_runners_id_signals_every_process(self):
        self.assertEqual(rk_prcode.RUNNER_IDS, frozenset(u.uid for u in L.PR_RUNNERS))
        self.assertNotIn(os.geteuid(), rk_prcode.RUNNER_IDS)
        with mock.patch.object(rk_prcode.os, "kill", forbidden):
            with self.assertRaisesRegex(RuntimeError, "PR runners' only"):
                rk_prcode.kill_all(signal.SIGKILL)
        sent = []
        with mock.patch.object(
            rk_prcode, "RUNNER_IDS", frozenset({os.geteuid()})
        ), mock.patch.object(
            rk_prcode.os, "kill", lambda pid, sig: sent.append((pid, sig))
        ):
            rk_prcode.kill_all(signal.SIGTERM)
        self.assertEqual(sent, [(-1, signal.SIGTERM)])

        def none_left(pid, sig):
            raise ProcessLookupError

        with mock.patch.object(
            rk_prcode, "RUNNER_IDS", frozenset({os.geteuid()})
        ), mock.patch.object(rk_prcode.os, "kill", none_left):
            rk_prcode.kill_all(signal.SIGKILL)  # nothing was left to signal: no error


class TestStop(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(rk_prcode, "reap", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.clock = Clock()
        self.sent = []

    def stop(self, now: bool, alive: set, ends: dict):
        """stop() over the processes `alive`, each ended by the signals `ends` names for it."""

        def send(sig):
            self.sent.append(sig)
            alive.difference_update([p for p in alive if sig in ends[p]])

        return rk_prcode.stop(
            now,
            lambda: sorted(alive),
            send=send,
            sleep=self.clock.sleep,
            clock=self.clock,
        )

    def test_sigterm_then_sigkill_to_what_is_left(self):
        # 5 ends on SIGTERM; 6 only on SIGKILL
        got = self.stop(
            False,
            {5, 6},
            {5: {signal.SIGTERM, signal.SIGKILL}, 6: {signal.SIGKILL}},
        )
        self.assertEqual(got, (2, 1))
        self.assertEqual(self.sent, [signal.SIGTERM, signal.SIGKILL])
        self.assertGreaterEqual(self.clock.t, rk_prcode.GRACE_S)

    def test_at_once_sigkill(self):
        got = self.stop(True, {5}, {5: {signal.SIGTERM, signal.SIGKILL}})
        self.assertEqual(got, (1, 1))
        self.assertEqual(self.sent, [signal.SIGKILL])
        self.assertLess(self.clock.t, rk_prcode.GRACE_S)

    def test_nothing_to_stop(self):
        self.assertEqual(self.stop(False, set(), {}), (0, 0))
        self.assertEqual(self.sent, [])

    def test_a_process_that_outlives_sigkill_is_named(self):
        with self.assertRaisesRegex(rk_prcode.Left, "1 process.*: 9"):
            self.stop(True, {9}, {9: set()})
        self.assertGreaterEqual(self.clock.t, rk_prcode.GONE_S)


class TestTree(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_descendants_by_their_parents(self):
        """100's children, theirs and so on, that still run. A process of the same user that is none of them is not
        one, as a `kill` of the same slot is not; the root's own entry, read as one of theirs, is walked once.
        """
        proc = fake_proc(
            pathlib.Path(self.tmp.name),
            [
                (
                    100,
                    104,
                    "S (sleeping)",
                ),  # its parent's pid used again by its descendant
                (101, 100, "S (sleeping)"),
                (102, 100, "Z (zombie)"),
                (103, 101, "R (running)"),
                (104, 103, "S (sleeping)"),
                (200, 1, "S (sleeping)"),
                (201, 200, "S (sleeping)"),
            ],
        )
        self.assertEqual(rk_prcode.descendants(100, proc), [101, 103, 104])
        self.assertEqual(rk_prcode.descendants(200, proc), [201])
        self.assertEqual(rk_prcode.descendants(999, proc), [])

    def test_a_signal_to_each_passes_over_one_that_ended_or_is_another_users(self):
        sent = []

        def kill(pid, sig):
            sent.append((pid, sig))
            if pid == 5:
                raise ProcessLookupError(pid)
            if pid == 6:
                raise PermissionError(pid)

        with mock.patch.object(rk_prcode.os, "kill", kill):
            rk_prcode.kill_each([5, 6, 7], signal.SIGKILL)
        self.assertEqual(sent, [(p, signal.SIGKILL) for p in (5, 6, 7)])


# The first lines of /proc/sysvipc's files, as Linux writes them.
SHM_HEAD = (
    "       key      shmid perms                  size  cpid  lpid nattch   uid   gid  cuid  cgid"
    "      atime      dtime      ctime                   rss                  swap"
)
SEM_HEAD = "       key      semid perms      nsems   uid   gid  cuid  cgid      otime      ctime"
MSG_HEAD = (
    "       key      msqid perms      cbytes       qnum lspid lrpid   uid   gid  cuid  cgid"
    "      stime      rtime      ctime"
)


class TestIpc(unittest.TestCase):
    """clear_ipc over a /proc/sysvipc of the test's own: only the objects it lists, never the machine's."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def sysvipc(self, shm=(), sem=(), msg=()) -> str:
        """A /proc whose sysvipc files list these objects: each a dict of its columns, the others 0."""
        top = pathlib.Path(self.tmp.name) / "proc"
        (top / "sysvipc").mkdir(parents=True)
        for kind, head, rows in (
            ("shm", SHM_HEAD, shm),
            ("sem", SEM_HEAD, sem),
            ("msg", MSG_HEAD, msg),
        ):
            lines = [head] + [
                " ".join(str(r.get(n, 0)) for n in head.split()) for r in rows
            ]
            (top / "sysvipc" / kind).write_text("\n".join(lines) + "\n")
        return str(top)

    def test_the_runners_own_objects_go_and_no_one_elses(self):
        me = os.geteuid()
        proc = self.sysvipc(
            shm=[
                {"shmid": 10, "uid": me, "cuid": me},
                {
                    "shmid": 11,
                    "uid": me + 1,
                    "cuid": me,
                },  # made by the runner, given to another user
                {"shmid": 12, "uid": me + 1, "cuid": me + 1},
                {"shmid": 13, "uid": me, "cuid": me},  # one that cannot be removed
            ],
            sem=[{"semid": 20, "uid": me, "cuid": me}],
            msg=[{"msqid": 30, "uid": me + 1, "cuid": me + 1}],
        )
        with open(f"{proc}/sysvipc/msg", "a") as f:
            f.write("a line that names no object\n")
        removed = []

        def remove(kind, ident):
            if ident == 13:
                raise OSError(errno.EPERM, "not permitted")
            removed.append((kind, ident))

        self.assertEqual(rk_prcode.clear_ipc(me, proc, remove), (3, 1))
        self.assertEqual(removed, [("shm", 10), ("shm", 11), ("sem", 20)])
        none = str(pathlib.Path(self.tmp.name) / "none")
        self.assertEqual(rk_prcode.clear_ipc(me, none, remove), (0, 0))

    def test_a_real_object_is_removed_and_only_as_a_runner(self):
        libc = ctypes.CDLL(None, use_errno=True)
        ident = libc.shmget(
            0, 4096, 0o1600
        )  # IPC_PRIVATE, IPC_CREAT and 0600: the test's own
        if ident < 0:
            self.skipTest(
                f"no System V shared memory here: {os.strerror(ctypes.get_errno())}"
            )
        self.addCleanup(libc.shmctl, ident, rk_prcode.IPC_RMID, None)
        me = os.geteuid()
        with self.assertRaisesRegex(RuntimeError, "the PR runners' only"):
            rk_prcode.ipc_rmid("shm", ident)
        proc = self.sysvipc(shm=[{"shmid": ident, "uid": me, "cuid": me}])
        with mock.patch.object(rk_prcode, "RUNNER_IDS", frozenset({me})):
            self.assertEqual(rk_prcode.clear_ipc(me, proc), (1, 0))
            rk_prcode.ipc_rmid("shm", ident)  # removed already: passed over
        info = ctypes.create_string_buffer(512)
        self.assertEqual(libc.shmctl(ident, 2, info), -1)  # IPC_STAT: it is gone
        self.assertEqual(ctypes.get_errno(), errno.EINVAL)


class SlotCase(unittest.TestCase):
    """A slot folder as root makes it, this test's user standing in for its runner, under umask 027; and a file and a
    folder of the monitor's beside it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        old = os.umask(rk_prcode.UMASK)
        self.addCleanup(os.umask, old)
        self.d = pathlib.Path(self.tmp.name)
        self.name = pwd.getpwuid(os.geteuid()).pw_name
        self.slot = self.d / self.name
        self.slot.mkdir()
        os.chmod(self.slot, 0o2751)
        s = os.stat(self.slot)
        self.uid, self.gid = s.st_uid, s.st_gid
        self.assertEqual(stat.S_IMODE(s.st_mode), 0o2751)
        self.mine = self.d / "monitor"
        self.mine.mkdir()
        self.mine_file = self.mine / "state.json"
        self.mine_file.write_text("the monitor's\n")
        os.chmod(self.mine_file, 0o600)
        self.mine_mode = stat.S_IMODE(os.stat(self.mine).st_mode)
        self.addCleanup(self.reopen)

    def reopen(self):
        """Every folder left in the temporary folder open to its owner, so that its cleanup can remove it."""
        for dirpath, dirnames, _files in os.walk(self.d):
            for n in dirnames:
                p = os.path.join(dirpath, n)
                if not os.path.islink(p):
                    os.chmod(p, 0o700)

    def open(self) -> int:
        return rk_prcode.open_slot(str(self.slot), self.uid, self.gid)

    def begin(self) -> None:
        fd = self.open()
        try:
            rk_prcode.empty(fd)
            rk_prcode.make_folders(fd, self.gid)
        finally:
            os.close(fd)

    def mode(self, *parts) -> int:
        return stat.S_IMODE(os.stat(self.slot.joinpath(*parts)).st_mode)


class TestSlot(SlotCase):
    def test_a_slot_is_used_only_as_root_made_it(self):
        os.close(self.open())
        for mode in (0o751, 0o2755, 0o2771):
            os.chmod(self.slot, mode)
            with self.assertRaisesRegex(rk_prcode.Refused, "next start makes it again"):
                self.open()
        os.chmod(self.slot, 0o2751)
        with self.assertRaises(rk_prcode.Refused):
            rk_prcode.open_slot(str(self.slot), self.uid + 1, self.gid)
        with self.assertRaises(rk_prcode.Refused):
            rk_prcode.open_slot(str(self.slot), self.uid, self.gid + 1)
        (self.d / "link").symlink_to(self.slot)
        with self.assertRaises(rk_prcode.Refused):
            rk_prcode.open_slot(str(self.d / "link"), self.uid, self.gid)

    def test_begin_empties_the_slot_and_makes_its_four_folders(self):
        s = self.slot
        (s / "out" / "sub").mkdir(parents=True)
        (s / "out" / "sub" / "r.txt").write_text("r")
        closed = s / "build" / "closed"
        closed.mkdir(parents=True)
        (closed / "f").write_text("f")
        closed.chmod(0)
        (s / "tmp").mkdir()
        (s / "tmp" / "link").symlink_to(self.mine_file)
        (s / "tmp" / "dirlink").symlink_to(self.mine)
        os.mkfifo(s / "tmp" / "fifo")
        deep = s / "home"
        for i in range(12):
            deep = deep / f"d{i}"
        deep.mkdir(parents=True)
        (deep / "bottom.txt").write_text("b")
        (s / "stray.txt").write_text("x")
        aside, real = [], rk_prcode._aside

        def counted(top):
            aside.append(real(top))
            return aside[-1]

        with mock.patch.object(rk_prcode, "DEPTH_MAX", 4), mock.patch.object(
            rk_prcode, "_aside", counted
        ):
            fd = self.open()
            try:
                rk_prcode.empty(fd)
                self.assertEqual(os.listdir(fd), [])
                rk_prcode.make_folders(fd, self.gid)
            finally:
                os.close(fd)
        self.assertTrue(aside)  # the deep folder was moved up, not walked down
        self.assertEqual(sorted(os.listdir(s)), sorted(L.SLOT_FOLDERS))
        for name in L.SLOT_FOLDERS:
            st = os.stat(s / name)
            self.assertEqual(
                (st.st_uid, st.st_gid, stat.S_IMODE(st.st_mode)),
                (self.uid, self.gid, 0o2750),
            )
        self.assertEqual(self.mine_file.read_text(), "the monitor's\n")
        self.assertEqual(os.listdir(self.mine), ["state.json"])

    def test_the_four_folders_are_checked_as_made(self):
        fd = self.open()
        try:
            with self.assertRaisesRegex(rk_prcode.Refused, "not .* mode 2750"):
                rk_prcode.make_folders(fd, self.gid + 1)
        finally:
            os.close(fd)

    def test_keep_opens_the_build_and_nothing_else_never_through_a_link(self):
        self.begin()
        core = self.slot / "build" / "core"
        core.mkdir()
        (core / "lib.so").write_text("l")
        (core / "run.sh").write_text("r")
        os.chmod(core / "run.sh", 0o750)
        (core / "setuid").write_text("s")
        os.chmod(core / "setuid", 0o6750)
        (core / "open.txt").write_text("o")
        os.chmod(core / "open.txt", 0o666)
        (core / "open").mkdir()
        os.chmod(core / "open", 0o2777)
        (core / "link").symlink_to(self.mine_file)
        (core / "dirlink").symlink_to(self.mine)
        os.mkfifo(core / "fifo")
        (self.slot / "out" / "r.txt").write_text("r")
        fd = self.open()
        try:
            rk_prcode.begun(fd, self.uid)
            self.assertFalse(rk_prcode.kept(fd))
            done, left = rk_prcode.readable(fd)
            self.assertTrue(rk_prcode.kept(fd))
            self.assertEqual(rk_prcode.readable(fd), (done, left))  # again: the same
        finally:
            os.close(fd)
        self.assertEqual((done, left), (7, 3))
        self.assertEqual(self.mode("build"), 0o2755)
        self.assertEqual(self.mode("build", "core"), 0o2755)
        self.assertEqual(self.mode("build", "core", "lib.so"), 0o644)
        self.assertEqual(self.mode("build", "core", "run.sh"), 0o755)
        self.assertEqual(
            self.mode("build", "core", "setuid"), 0o755
        )  # runs as whoever runs it
        self.assertEqual(
            self.mode("build", "core", "open.txt"), 0o644
        )  # no other runner writes it
        self.assertEqual(self.mode("build", "core", "open"), 0o2755)
        for name in ("out", "tmp", "home"):
            self.assertEqual(self.mode(name), 0o2750)
        self.assertEqual(self.mode("out", "r.txt"), 0o640)
        self.assertEqual(stat.S_IMODE(os.stat(self.mine_file).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(self.mine).st_mode), self.mine_mode)

    def test_clear_shm_removes_the_runners_own_entries_and_no_one_elses(self):
        """In the folder every user writes, the runner's own entries go with what they hold: a link without its target,
        a folder its owner closed, one past DEPTH_MAX too. Another user's stay; one that cannot be removed is counted.
        """
        shm = self.d / "shm"
        shm.mkdir()
        os.chmod(shm, 0o1777)
        (shm / "sem.mp-x").write_text("s")
        (shm / "dir" / "closed").mkdir(parents=True)
        (shm / "dir" / "closed" / "f").write_text("f")
        os.chmod(shm / "dir" / "closed", 0)
        deep = shm / "deep"
        for _ in range(6):
            deep = deep / "d"
        deep.mkdir(parents=True)
        (shm / "mine").symlink_to(self.mine)
        names = ["deep", "dir", "mine", "sem.mp-x"]
        self.assertEqual(
            rk_prcode.clear_shm(self.uid + 1, str(shm)), (0, 0)
        )  # another user's: none of them
        self.assertEqual(sorted(os.listdir(shm)), names)
        with mock.patch.object(rk_prcode, "DEPTH_MAX", 3):
            self.assertEqual(rk_prcode.clear_shm(self.uid, str(shm)), (4, 0))
        self.assertEqual(os.listdir(shm), [])
        self.assertEqual(os.listdir(self.mine), ["state.json"])  # the link's target
        self.assertEqual(rk_prcode.clear_shm(self.uid, str(self.d / "none")), (0, 0))
        if os.geteuid() == 0:
            return  # root removes from a folder closed to it
        (shm / "kept").write_text("k")
        os.chmod(shm, 0o555)
        try:
            self.assertEqual(rk_prcode.clear_shm(self.uid, str(shm)), (0, 1))
        finally:
            os.chmod(shm, 0o1777)
        self.assertEqual(os.listdir(shm), ["kept"])

    def test_a_slot_whose_build_is_a_link_is_not_begun(self):
        self.begin()
        os.rmdir(self.slot / "build")
        (self.slot / "build").symlink_to(self.mine)
        fd = self.open()
        try:
            with self.assertRaisesRegex(rk_prcode.Refused, "build is not a folder"):
                rk_prcode.begun(fd, self.uid)
            self.assertEqual(
                rk_prcode.readable(fd), (0, 1)
            )  # the link is left as it is
        finally:
            os.close(fd)
        self.assertEqual(stat.S_IMODE(os.stat(self.mine).st_mode), self.mine_mode)


@contextlib.contextmanager
def fd2(path: pathlib.Path):
    """File descriptor 2 (what a forked child writes its error to) into `path` for the block."""
    saved = os.dup(2)
    with open(path, "w") as f:
        os.dup2(f.fileno(), 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


class TestRun(unittest.TestCase):
    """run_cmd with real processes: the test is the launcher, and not the subreaper of anything."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        patcher = mock.patch.object(rk_prcode, "reap", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cmd(self, cmd, cwd=None, left=lambda: [], send=forbidden) -> int:
        return rk_prcode.run_cmd(
            cmd,
            str(cwd or self.d),
            {"PATH": os.environ["PATH"], "FOO": "1"},
            left,
            send=send,
            become_subreaper=lambda: None,
        )

    def test_cmd_runs_in_its_folder_with_stdin_dev_null_and_its_code_comes_back(self):
        code = self.run_cmd(
            [
                "sh",
                "-c",
                'pwd > w; readlink /proc/$$/fd/0 > i; echo "$FOO $HOME" > e; exit 7',
            ]
        )
        self.assertEqual(code, 7)
        self.assertEqual((self.d / "w").read_text(), f"{self.d}\n")
        self.assertEqual((self.d / "i").read_text(), "/dev/null\n")
        self.assertEqual((self.d / "e").read_text(), "1 \n")  # its environment only

    def test_cmd_gains_no_privilege_by_exec(self):
        with open("/proc/self/status") as f:
            if "NoNewPrivs:\t1" in f.read():
                self.skipTest("this test runs with no new privileges already")
        self.assertEqual(
            self.run_cmd(["sh", "-c", "grep NoNewPrivs /proc/self/status > p"]), 0
        )
        self.assertEqual((self.d / "p").read_text().split(), ["NoNewPrivs:", "1"])

    def test_a_signal_that_ends_cmd_and_one_it_cannot_start(self):
        self.assertEqual(self.run_cmd(["sh", "-c", "kill -KILL $$"]), 128 + 9)
        with fd2(self.d / "err"):
            self.assertEqual(self.run_cmd(["/nonexistent/cmd"]), 127)
            self.assertEqual(self.run_cmd(["true"], cwd=self.d / "nonexistent"), 127)
        err = (self.d / "err").read_text()
        self.assertIn("rk_prcode.py: /nonexistent/cmd:", err)
        self.assertIn(f"rk_prcode.py: {self.d / 'nonexistent'}:", err)

    def test_a_launcher_that_cannot_be_the_subreaper_runs_nothing(self):
        def refused():
            raise OSError(errno.EINVAL, os.strerror(errno.EINVAL), "the subreaper")

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = rk_prcode.run_cmd(
                ["sh", "-c", "touch ran"],
                str(self.d),
                {"PATH": os.environ["PATH"]},
                lambda: [],
                send=forbidden,
                become_subreaper=refused,
            )
        self.assertEqual(code, 127)
        self.assertEqual(
            err.getvalue(),
            f"rk_prcode.py: the subreaper: {os.strerror(errno.EINVAL)}: CMD not run\n",
        )
        self.assertFalse((self.d / "ran").exists())

    def test_what_prctl_returns_is_checked(self):
        libc = mock.Mock()
        libc.prctl.return_value = -1
        with mock.patch.object(
            rk_prcode.ctypes, "CDLL", return_value=libc
        ), mock.patch.object(rk_prcode.ctypes, "get_errno", return_value=errno.EPERM):
            for act, option in (
                (rk_prcode.subreaper, rk_prcode.PR_SET_CHILD_SUBREAPER),
                (rk_prcode.no_new_privs, rk_prcode.PR_SET_NO_NEW_PRIVS),
            ):
                with self.assertRaises(OSError) as cm:
                    act()
                self.assertEqual(cm.exception.errno, errno.EPERM)
                libc.prctl.assert_called_with(option, 1, 0, 0, 0)
            libc.prctl.return_value = 0
            rk_prcode.subreaper()  # the kernel took it

    def test_a_sigterm_to_the_launcher_reaches_cmd(self):
        before = signal.getsignal(signal.SIGTERM)
        code = self.run_cmd(["sh", "-c", "kill -TERM $PPID; exec sleep 30"])
        self.assertEqual(code, 128 + signal.SIGTERM)
        self.assertIs(signal.getsignal(signal.SIGTERM), before)

    def test_what_cmd_left_running_is_stopped_before_it_returns(self):
        pidfile = self.d / "pid"
        sent = []

        def left():
            pid = int(pidfile.read_text())
            return [pid] if running(pid) else []

        def send(sig):
            sent.append(sig)
            os.kill(int(pidfile.read_text()), sig)

        code = self.run_cmd(
            ["sh", "-c", f"sleep 60 & echo $! > {pidfile}; exit 3"],
            left=left,
            send=send,
        )
        self.assertEqual(code, 3)
        self.assertEqual(sent, [signal.SIGKILL])
        self.assertFalse(running(int(pidfile.read_text())))

    def test_what_cmd_left_is_stopped_and_no_other_process_of_the_runners(self):
        """By default the launcher stops its own descendants: what CMD left goes, in a session of its own too, and a
        process of the same user that CMD did not start runs on, as a `kill`, `begin`, `end` or `keep` of the slot
        does."""
        pidfile = self.d / "pids"
        other = subprocess.Popen(["sleep", "60"])
        self.addCleanup(other.wait)
        self.addCleanup(other.kill)
        pid = os.fork()
        if (
            pid == 0
        ):  # the launcher: a process of its own, so that it may be a subreaper
            code = 1
            try:
                with mock.patch.object(rk_prcode, "reap", REAP):
                    rc = rk_prcode.run_cmd(
                        [
                            "sh",
                            "-c",
                            f"sleep 60 & echo $! > {pidfile}; setsid sleep 60 & echo $! >> {pidfile}",
                        ],
                        str(self.d),
                        {"PATH": os.environ["PATH"]},
                    )
                left = [int(p) for p in pidfile.read_text().split()]
                code = (
                    0
                    if (rc, len(left), [status(p) for p in left])
                    == (0, 2, [None, None])
                    else 1
                )
            finally:
                os._exit(code)
        _pid, st = os.waitpid(pid, 0)
        self.assertEqual(os.waitstatus_to_exitcode(st), 0)
        self.assertIsNone(other.poll())

    def test_the_launcher_adopts_and_reaps_what_cmd_leaves(self):
        pidfile = self.d / "pid"
        pid = os.fork()
        if (
            pid == 0
        ):  # the launcher: a process of its own, so that it may be a subreaper
            code = 1
            try:
                parents = []

                def left():
                    p = int(pidfile.read_text())
                    st = status(p)
                    if st and not parents:
                        parents.append(int(st["PPid"]))
                    return [p] if running(p) else []

                def send(sig):
                    os.kill(int(pidfile.read_text()), sig)

                with mock.patch.object(rk_prcode, "reap", REAP):
                    rc = rk_prcode.run_cmd(
                        ["sh", "-c", f"sleep 60 & echo $! > {pidfile}"],
                        str(self.d),
                        {"PATH": os.environ["PATH"]},
                        left,
                        send=send,
                    )
                libc = ctypes.CDLL(None, use_errno=True)
                flag = ctypes.c_int(0)
                libc.prctl(PR_GET_CHILD_SUBREAPER, ctypes.byref(flag), 0, 0, 0)
                gone = status(int(pidfile.read_text())) is None  # reaped, not a zombie
                code = (
                    0
                    if (rc, parents, flag.value, gone)
                    == (
                        0,
                        [os.getpid()],
                        1,
                        True,
                    )
                    else 1
                )
            finally:
                os._exit(code)
        _pid, st = os.waitpid(pid, 0)
        self.assertEqual(os.waitstatus_to_exitcode(st), 0)


class TestMain(SlotCase):
    """main() as this test's user, which a stand-in layout names a runner, in the temporary slot; nothing it runs
    reaches the real kill(-1) or the machine's System V objects (clear_ipc stood in for), and CMD itself is not started
    (run_cmd stood in for)."""

    def setUp(self):
        super().setUp()
        group = grp.getgrgid(self.gid).gr_name
        me = L.User(self.name, self.uid, group, self.gid)
        self.shm = self.d / "shm"  # never the machine's own
        self.mq = self.d / "mqueue"
        for d in (self.shm, self.mq):
            d.mkdir()
            os.chmod(d, 0o1777)
        layout = types.SimpleNamespace(
            **{
                **vars(L),
                "SLOTS_DIR": str(self.d),
                "SHM_DIR": str(self.shm),
                "MQUEUE_DIR": str(self.mq),
                "PR_RUNNERS": (me,),
                "PR_RUNNER": me,
                "MONITOR": L.User("rkmon", self.uid, group, self.gid),
            }
        )
        self.alive = []
        self.ran, self.ran_kw, self.ipc = [], [], []
        patches = [
            mock.patch.object(rk_prcode, "reap", lambda: None),
            mock.patch.object(rk_prcode, "LAYOUT", layout),
            mock.patch.object(rk_prcode.os, "kill", forbidden),
            mock.patch.object(
                rk_prcode, "others", lambda uid, me, proc="/proc": list(self.alive)
            ),
            mock.patch.object(
                rk_prcode,
                "run_cmd",
                lambda *a, **k: self.ran.append(a) or self.ran_kw.append(k) or 5,
            ),
            mock.patch.object(
                rk_prcode,
                "clear_ipc",
                lambda uid, proc="/proc": self.ipc.append(list(self.alive)) or (0, 0),
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def main(self, *argv, env=b"") -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rk_prcode.main(list(argv), stdin=io.BytesIO(env))
        return code, out.getvalue(), err.getvalue()

    def test_a_slots_life(self):
        code, _out, err = self.main("run", "--", "true")
        self.assertEqual(code, 3)
        self.assertIn("the slot was not begun", err)
        os.umask(0o022)  # main() sets its own
        self.assertEqual(self.main("begin")[0], 0)
        self.assertEqual(os.umask(rk_prcode.UMASK), rk_prcode.UMASK)
        self.assertEqual(sorted(os.listdir(self.slot)), sorted(L.SLOT_FOLDERS))
        env = b"PATH=/usr/bin\0X_PASSWORD=p\0X_KEY=k\0RKMON_JOB=/j\0"
        self.assertEqual(
            self.main("run", "-C", "out", "--", "make", "-j2", env=env)[0], 5
        )
        self.assertEqual(self.main("run", "--", "true")[0], 5)
        (cmd, cwd, job), (cmd2, cwd2, _job2) = self.ran
        self.assertEqual(
            self.ran_kw, [{}, {}]
        )  # what CMD left: the launcher's own descendants
        self.assertEqual((cmd, cwd), (["make", "-j2"], f"{self.slot}/out"))
        self.assertEqual((cmd2, cwd2), (["true"], f"{self.slot}/home"))
        self.assertNotIn("X_PASSWORD", job)
        self.assertNotIn("X_KEY", job)
        self.assertEqual((job["RKMON_JOB"], job["HOME"]), ("/j", f"{self.slot}/home"))
        code, out, _err = self.main("keep")
        self.assertEqual(code, 0)
        self.assertIn("kept: its build is readable by the other runners", out)
        self.assertEqual(self.mode("build"), 0o2755)
        code, _out, err = self.main("run", "--", "true")
        self.assertEqual(code, 3)
        self.assertIn("no job runs in a kept slot", err)
        self.assertEqual(len(self.ran), 2)
        self.assertEqual(self.main("keep")[0], 0)  # again: still kept
        self.assertEqual(self.main("end")[0], 0)
        self.assertEqual(os.listdir(self.slot), [])
        self.assertEqual(self.main("run", "--", "true")[0], 3)

    def test_keep_leaves_the_build_and_makes_the_other_folders_fresh(self):
        """What PR code put in a kept slot beside its build, or opened to the other runners there, is gone: its out, tmp
        and home made again (empty, 2750, group rkmon), every other entry of the slot's folder removed, never through a
        link; the build stays, readable."""
        self.assertEqual(self.main("begin")[0], 0)
        s = self.slot
        (s / "build" / "lib.so").write_text("l")
        (s / "out" / "r.txt").write_text("r")
        os.chmod(s / "out" / "r.txt", 0o644)
        os.chmod(s / "out", 0o2755)
        (s / "tmp" / "shared").mkdir()
        os.chmod(s / "tmp" / "shared", 0o777)
        os.chmod(s / "tmp", 0o2777)
        (s / "home" / ".cache").mkdir()
        (s / "home" / "mine").symlink_to(self.mine_file)
        os.chmod(s / "home", 0o2755)
        (s / "stray").mkdir()
        (s / "stray" / "f").write_text("f")
        os.chmod(s / "stray", 0o755)
        (s / "stray.txt").write_text("x")
        (s / "link").symlink_to(self.mine)
        code, out, _err = self.main("keep")
        self.assertEqual(code, 0)
        self.assertIn("kept: its build is readable by the other runners", out)
        self.assertEqual(sorted(os.listdir(s)), sorted(L.SLOT_FOLDERS))
        for name in L.SLOT_FOLDERS:
            st = os.stat(s / name)
            self.assertEqual(
                (st.st_uid, st.st_gid, stat.S_IMODE(st.st_mode)),
                (self.uid, self.gid, 0o2755 if name == L.SLOT_BUILD else 0o2750),
                name,
            )
            if name != L.SLOT_BUILD:
                self.assertEqual(os.listdir(s / name), [], name)
        self.assertEqual((s / "build" / "lib.so").read_text(), "l")
        self.assertEqual(self.mine_file.read_text(), "the monitor's\n")
        self.assertEqual(os.listdir(self.mine), ["state.json"])
        self.assertEqual(stat.S_IMODE(os.stat(self.mine).st_mode), self.mine_mode)

    def test_begin_keep_and_end_remove_the_runners_entries_in_shm(self):
        for act in ("begin", "keep", "end"):
            (self.shm / f"sem.{act}").write_text("s")
            code, out, _err = self.main(act)
            self.assertEqual(code, 0, act)
            self.assertEqual(os.listdir(self.shm), [], act)
            self.assertIn(f"{self.shm}: removed 1 of the runner's entries", out, act)
        code, out, _err = self.main("begin")
        self.assertNotIn(str(self.shm), out)  # nothing to remove: no line

    def test_run_names_the_jobs_trees_to_git(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = pathlib.Path(os.path.realpath(tmp.name))
        for t in ("wt7m", "repo"):
            (d / t / ".git").mkdir(parents=True)
        (d / "wt7m" / "rocke").mkdir()
        self.assertEqual(self.main("begin")[0], 0)
        named = f"RKMON_TREES={d / 'repo'}:relative:{d / 'gone'}\0".encode()
        with mock.patch.object(rk_prcode, "monitor_uid", lambda: self.uid):
            self.main("run", "-C", str(d / "wt7m" / "rocke"), "--", "true", env=named)
            self.main("run", "--", "true")
        (_c, _w, job), (_c2, _w2, job2) = self.ran
        self.assertEqual(
            [job.get(f"GIT_CONFIG_VALUE_{i}") for i in range(3)],
            [str(d / "wt7m"), str(d / "repo"), None],
        )
        self.assertEqual(job["GIT_CONFIG_COUNT"], "2")
        self.assertNotIn(
            "GIT_CONFIG_COUNT", job2
        )  # in its home, with no tree named: none

    def test_begin_keep_and_end_remove_the_acls_pr_code_set_on_the_slots_folder(self):
        perms = rk_prcode.PERMS
        other = (
            ("user", self.uid + 1, "rx"),
            ("mask", None, "rx"),
        )  # another runner reads
        access = (
            ("user_obj", None, "rwx"),
            ("group_obj", None, "rx"),
            ("other", None, "x"),
        )

        def plant():
            perms.set_default_acl(self.slot, perms.SHARED + other)
            os.setxattr(self.slot, perms.ACCESS, perms.acl_bytes(access + other))

        def acl(path, name):
            try:
                return perms.acl_entries(os.getxattr(path, name))
            except OSError as exc:
                if exc.errno == errno.ENODATA:
                    return None
                raise

        try:
            plant()
        except OSError:
            self.skipTest("this file system keeps no ACL")
        self.assertEqual(self.mode(), 0o2751)  # what root made, as far as the mode says
        self.assertEqual(self.main("begin")[0], 0)
        self.assertEqual(
            (acl(self.slot, perms.DEFAULT), acl(self.slot, perms.ACCESS)), (None, None)
        )
        for name in L.SLOT_FOLDERS:  # made with no ACL of PR code's
            self.assertEqual(
                (
                    acl(self.slot / name, perms.DEFAULT),
                    acl(self.slot / name, perms.ACCESS),
                ),
                (None, None),
                name,
            )
        plant()
        os.setxattr(self.slot / "home", perms.ACCESS, perms.acl_bytes(access + other))
        self.assertEqual(self.main("keep")[0], 0)
        self.assertEqual(
            (acl(self.slot, perms.DEFAULT), acl(self.slot, perms.ACCESS)), (None, None)
        )
        for (
            name
        ) in (
            L.SLOT_FOLDERS
        ):  # the build's own are the build's: readable() closes it to writes
            if name != L.SLOT_BUILD:
                self.assertEqual(
                    (
                        acl(self.slot / name, perms.DEFAULT),
                        acl(self.slot / name, perms.ACCESS),
                    ),
                    (None, None),
                    name,
                )
        plant()
        self.assertEqual(self.main("end")[0], 0)
        self.assertEqual(
            (acl(self.slot, perms.DEFAULT), acl(self.slot, perms.ACCESS)), (None, None)
        )

    def test_an_access_acl_is_removed_only_when_the_slots_folder_has_one(self):
        """A runner is in no group of its slot's folder, and for such a user the kernel takes the folder's set-group-ID
        bit with any access ACL removed, even one that is not there (the stand-in removexattr does so here): the
        folders made in it next would not be the monitor's group's. The launcher removes the folder's access ACL only
        when it has one, and refuses the slot when its folder lost that bit."""
        perms, real, removed = rk_prcode.PERMS, os.removexattr, []

        def removexattr(path, name, *args, **kwargs):
            removed.append(name)
            if name == perms.ACCESS:
                os.chmod(path, stat.S_IMODE(os.stat(path).st_mode) & ~stat.S_ISGID)
            return real(path, name, *args, **kwargs)

        with mock.patch.object(rk_prcode.os, "removexattr", removexattr):
            for act in ("begin", "keep", "end"):
                with self.subTest(act=act):
                    code, _out, err = self.main(act)
                    self.assertEqual(code, 0, err)
        self.assertNotIn(perms.ACCESS, removed)
        self.assertEqual(self.mode(), 0o2751)
        access = (
            ("user_obj", None, "rwx"),
            ("user", self.uid + 1, "rx"),  # another runner reads
            ("group_obj", None, "rx"),
            ("mask", None, "rx"),
            ("other", None, "x"),
        )
        try:
            os.setxattr(self.slot, perms.ACCESS, perms.acl_bytes(access))
        except OSError:
            self.skipTest("this file system keeps no ACL")
        self.assertEqual(self.mode(), 0o2751)
        with mock.patch.object(rk_prcode.os, "removexattr", removexattr):
            code, _out, err = self.main("begin")
        self.assertEqual(code, 3)
        self.assertIn("removing it took the folder's set-group-ID bit", err)

    def test_a_slot_pr_code_changed_is_refused_until_the_next_start(self):
        self.main("begin")
        os.chmod(self.slot, 0o2771)
        for act in ("begin", "keep", "end"):
            code, _out, err = self.main(act)
            self.assertEqual(code, 3)
            self.assertIn("refused", err)
        self.assertEqual(self.main("run", "--", "true")[0], 3)
        self.assertEqual(self.ran, [])
        self.assertEqual(sorted(os.listdir(self.slot)), sorted(L.SLOT_FOLDERS))

    def test_every_action_first_stops_the_runners_processes(self):
        sent, cleared = [], []

        def send(sig):
            sent.append(sig)
            self.alive.clear()

        def clear(uid, path=""):
            cleared.append((path, list(self.alive)))
            return 0, 0

        with mock.patch.object(rk_prcode, "kill_all", send), mock.patch.object(
            rk_prcode, "GRACE_S", 0
        ), mock.patch.object(rk_prcode, "clear_shm", clear):
            self.alive[:] = [41, 42]
            code, out, _err = self.main("kill")
            self.assertEqual((code, sent), (0, [signal.SIGTERM]))
            self.assertIn("stopped 2 process(es)", out)
            self.alive[:] = [43]
            code, out, _err = self.main("kill", "--now")
            self.assertEqual(sent[-1], signal.SIGKILL)
            self.assertIn("stopped 1 process(es); 1 needed SIGKILL", out)
            for act in ("begin", "keep", "end"):
                self.alive[:] = [44]
                code, out, _err = self.main(act)
                self.assertEqual((code, sent[-1]), (0, signal.SIGKILL))
                self.assertIn("stopped 1 process(es)", out)
        # What every user shares is cleared once the stop is over: no process of the runner's is left to make more
        # there.
        self.assertEqual(cleared, [(str(self.shm), []), (str(self.mq), [])] * 3)
        self.assertEqual(self.ipc, [[], [], []])

    def test_a_line_says_what_went_from_what_every_user_shares(self):
        (self.mq / "rk-q").write_text("q")  # a message queue of the runner's
        with mock.patch.object(
            rk_prcode, "clear_ipc", lambda uid, proc="/proc": (2, 1)
        ):
            code, out, _err = self.main("begin")
        self.assertEqual(code, 0)
        self.assertIn(f"{self.mq}: removed 1 of the runner's entries", out)
        self.assertIn(
            "System V IPC: removed 2 of the runner's objects; 1 could not be removed",
            out,
        )
        self.assertEqual(os.listdir(self.mq), [])

    def test_a_missing_monitor_group_is_one_line(self):
        self.assertEqual(self.main("begin")[0], 0)
        gone = types.SimpleNamespace(
            **{
                **vars(rk_prcode.LAYOUT),
                "MONITOR": L.User("rkmon", self.uid, "rk-no-such-group", self.gid),
            }
        )
        with mock.patch.object(rk_prcode, "LAYOUT", gone):
            for argv in (["begin"], ["keep"], ["end"], ["run", "--", "true"]):
                code, _out, err = self.main(*argv)
                self.assertEqual(code, 127, argv)
                self.assertEqual(
                    err,
                    f"PR_SLOT {self.name}: no group rk-no-such-group (the monitor's): the container's start "
                    "makes it\n",
                    argv,
                )
        self.assertEqual(self.ran, [])

    def test_usage_errors(self):
        self.main("begin")
        for argv in (
            [],
            ["run", "true"],
            ["run", "--"],
            ["kill", "--soon"],
            ["begin", "x"],
        ):
            code, _out, err = self.main(*argv)
            self.assertEqual(code, 2, argv)
            self.assertIn("usage:", err)
        with mock.patch.object(rk_prcode.PR_ENV, "ENV_MAX", 8):
            code, _out, err = self.main("run", "--", "true", env=b"A=123456789")
        self.assertEqual(code, 2)
        self.assertIn("an environment over 8 bytes", err)
        self.assertEqual(self.ran, [])


if __name__ == "__main__":
    unittest.main()
