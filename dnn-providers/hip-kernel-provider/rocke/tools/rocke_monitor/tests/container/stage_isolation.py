# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The isolation proof, in the image, as root for the setup only (tests/container/image.py runs it with RKMON_AS=root, in
a throwaway container with no network). Fake logins and keys only; every check runs as the user it names.

  python3 -B stages.py isolation

- The start's root step (docker/rk_root.py start, as the entry point's root phase runs it): an older deployment's write
  login in /secrets/gh moves to /secrets-write/gh, the write user's, once; nothing of it is printed; the session user
  cannot run it, nor anything else as root.
- The write login (the fix pass's proof): as the session user, neither its folder, its file, gh pointed at it, an alias,
  a Python subprocess nor any sudo but the broker's reaches it; the broker refuses everything but rk's GitHub writes
  (approve, merge-now, dismiss, --premises in any spelling, spec-post, the login's file, a link to it, a configuration
  that writes where the write user keeps it), and rk's writes reach it.
- The PR runners, as the job scripts run PR code (rkmon_lib/prcode.py SHELL: prslot, prc, prr, prkeep)
  and as `rk pr-run` does: rkpr1 to rkpr16, each with a group of its own and in the runners' group, one per slot, the
  slots made at the start. PR code runs as the runner of the slot its job took, with the listed environment only (no
  login, key or variable of the monitor's); it reads no secret (/secrets, the read login, the atlassian login,
  /secrets-write, Claude's folder, the broker's copies, the session's processes' environments, the first process's)
  and nothing of the monitor's (its configuration, its state, the checks' report, the logs, its home), runs no sudo,
  and writes its slot's four folders and its own entries in /dev/shm and /dev/mqueue and its System V objects (gone
  when the slot is freed), and nothing else: not the work volume, the monitor's home and state, the logs, the outputs,
  the caches, Claude's folder, the clone and the PR's tree (`.git/config` through git too, `.git/hooks/`, the
  attributes, the objects), the probes, the image's code, the virtual environments, a program on the search path,
  the system's Python, the shells' start files, /tmp, /var/tmp, the RAM disk's top, the scratch folder, the slots'
  folder, their records, the inputs the monitor hands it, or another slot, which it cannot list either; no try makes
  or changes a file. The monitor reads its results through the result reader, which refuses a link planted in `out`
  to a file or a folder of the monitor's and copies nothing; emptying the slot removes the links, not what they name.
  Every process it left is stopped when its command ends, before its results are read, one in a session of its own,
  its environment cleared, deaf to SIGTERM and under another name included; the launcher's `kill` stops one the
  session user cannot signal. Named work keeps its folders until it ends, or until its owner is gone and the review's
  end comes; other work gets fresh folders. Its git takes as safe only the tree it runs in and the trees the job
  names, and fetches nothing: a blob the clone lacks is there once the monitor fetched it. The monitor's git refuses
  a repository a runner made, so its core.fsmonitor never runs. It makes no namespace. No folder of the container's is
  every user's but the kernel's folders of shared memory and message queues (NOTE lines: what the container can and
  cannot give a runner of its own).
- The scratch folder (paths.tmp): a runner enters a review directory the session made there, and lists, reads and
  writes none of it, `rk pr-run` there included; it reads the trees made there with `rk pr-tree` (a worktree added as
  the session's agents add theirs, and a folder of files PR code reads), runs git in the worktree, and writes none of
  them. The start's layout step, walking the scratch folder again, leaves those trees as they are.
- A kept slot (the checks' build, kept for its review): its other folders made fresh (what PR code opened there is
  gone), its build readable by the other runners and writable by none, with no set-user-ID file; another slot's job
  reads that build and nothing else of the slot; the launcher runs no job in it; ctest runs on it through
  tools/ctest_read_only.py, as PR code in a slot of its own, and writes nothing there. The review's end frees it
  (review.end_slots), and a job that finds every other slot taken takes it, the shared build record removed both ways.
- Every other run of PR code goes through the same launcher: a PR's test file and a saved probe that try to read both
  secrets volumes, run by the first-pass tools (run_pytest, run_probe), the drafter's saved probe and `rk pr-run`,
  run as a PR runner and read nothing.
- The code is root's: root installs two releases of the image's own code, each with a mark in
  its writer's runner, as `rkmon apply` has root install one (entrypoint.sh code-install, as root), so that CURRENT
  names the second and KEPT the first. The monitor, the writer and a PR runner (as the job scripts run PR code) each
  try to make, write, rename, open up, point elsewhere or remove the code volume's folder, its releases' folder, the
  releases, their files, the lock, CURRENT and KEPT, and change nothing. The launcher runs CURRENT's release; so does
  the broker, whatever its caller names (RKMON_CODE, an argument); the broker refuses, and runs none of it, a release
  with a file another user may write or owns or with a link that leaves it, until root puts it back. `rk code
  install`, as the monitor, refuses and names the host's commands.
- After the run, by the modes and ACLs of what the monitor and the writer made in the working folder, the RAM disk's,
  the system's temporary folders, their homes and the secrets volumes (each entry's access and default ACL, read from
  its extended attributes as getfacl reads them: the image has no getfacl): no PR runner may write any of it, and
  each file they share is 0660, group rkmon, each folder 2770 (2771 where a runner passes through), group rkmon,
  with a shared folder's default ACL. A private file or folder (its group has nothing) is the one other kind.
- Each sudo call (the broker's, the launcher's) writes its audit record: a check whose output holds sudo's word that
  it could not (the container lacks the AUDIT_WRITE capability) fails.
"""

import collections
import errno
import glob
import grp
import io
import json
import os
import pathlib
import pwd
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile

PKG = "/opt/rocke_monitor"
sys.path.insert(0, PKG)
from rkmon_lib import container_layout as L  # noqa: E402
from rkmon_lib import code_volume, gh, paths, prcode, writer_run  # noqa: E402
from rkmon_lib import secrets_registry as sr  # noqa: E402

pr_env = paths.worker_module("pr_env")  # the environment PR code gets

RUNUSER = shutil.which("runuser", path="/usr/sbin:/sbin:/usr/bin:/bin") or "runuser"
BROKER = f"{PKG}/docker/rk_write.py"
RUNNER = f"{PKG}/docker/rk_prcode.py"
ROOT_STEP = f"{PKG}/docker/rk_root.py"
ROOT_START = f"python3 -I {ROOT_STEP} start"  # the start's root step, as the entry point's root phase runs it
# The writer's runner of each release the stage installs says first which release it is (MARK): the broker ran its code.
WRITER_RUNNER = "rkmon_lib/writer_run.py"
MARK = "isolation: the writer runs release {}"
WRITE_TOKEN, READ_TOKEN, ATL_TOKEN, KEY = (
    "FAKE-WRITE-TOKEN",
    "FAKE-READ-TOKEN",
    "FAKE-ATLASSIAN-TOKEN",
    "FAKE-MODEL-KEY",
)
SECRETS = (WRITE_TOKEN, READ_TOKEN, ATL_TOKEN, KEY)
# What sudo says on each call when the container lacks the AUDIT_WRITE capability, which it writes its audit record
# of the call with: it runs the command all the same. A check whose output holds it fails.
SUDO_UNAUDITED = "sudo: unable to send audit message"


def secret_place(name: str, older: bool = False) -> str:
    """Where the container keeps the secrets registry's entry NAME, a login's folder or a secret's file: its own place
    (secrets_registry.home), or, OLDER, where an older deployment kept it (its first older place).
    """
    # the container's folders: the monitor's secrets folder, and the writer's own volume (doctor.deployment_image)
    cfg, environ = {"paths": {"secrets_dir": L.SECRETS_DIR}}, {"RKMON_CONTAINER": "1"}
    entry = sr.entry(cfg, name)
    return (
        sr.older_places(cfg, entry, environ)[0][1]
        if older
        else sr.home(cfg, entry, environ)
    )


# The fake logins and key, where the container keeps them: gh's read login, the session's; the write login, in the write
# user's own volume, and where an older deployment kept it (the start's root step moves it from there); the atlassian
# login where a container from before the writer kept it, and a file in it; the Claude session's key.
READ_LOGIN = secret_place("github_read")
READ_HOSTS = f"{READ_LOGIN}/{gh.HOSTS_FILE}"
WRITE_LOGIN = L.WRITE_LOGIN_DIR
WRITE_HOSTS = f"{WRITE_LOGIN}/{gh.HOSTS_FILE}"
OLDER_WRITE_LOGIN = secret_place("github_write", older=True)
OLDER_ATLASSIAN = secret_place("atlassian", older=True)
ATLASSIAN_FILE = f"{OLDER_ATLASSIAN}/config.json"
CLAUDE_KEY = secret_place(sr.CLAUDE_LOGIN)
# The image's environment for the session user (docker/Dockerfile ENV), as the container's processes have it.
SESSION_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": f"/home/{L.MONITOR.name}",
    "LANG": "C.UTF-8",
    "RKMON_CONTAINER": "1",
    "RKMON_HOME": L.MONITOR_HOME,
    "RKMON_WRITE_BROKER": BROKER,
    "RKMON_PR_RUNNER": RUNNER,
    "GH_CONFIG_DIR": READ_LOGIN,
    "PYTHONDONTWRITEBYTECODE": "1",
}
JOB = f"{L.MONITOR_HOME}/state/jobs/checks-pr7-1"
CONFIG = f"{L.MONITOR_HOME}/config.json"  # the monitor's configuration
LOGS = f"{L.WORK_DIR}/{L.LOGS_FOLDER}"  # the logs, the broker's (writer_run.LOG) among them
CLAUDE = f"{L.WORK_DIR}/claude"  # Claude's folder: its config folder (docker/Dockerfile CLAUDE_CONFIG_DIR) is in it
CLAUDE_LOGIN = f"{CLAUDE}/config/.credentials.json"  # Claude's login, a fake one
ORIGIN = f"{L.WORK_DIR}/origin"  # the clone's remote: a folder of the monitor's that no runner reads
REPO = f"{L.WORK_DIR}/repo"  # the monitor's blobless clone of it, a tree PR code reads
WT = f"{L.WORK_DIR}/wt7"  # PR 7's tree: a test file that reads for secrets, and a CMake project with one test
PR7 = f"{L.WORK_DIR}/workers/pr7"  # PR 7's folder of the workers' (the review tools' Tree)
CHECKS = f"{PR7}/checks"  # the checks job's folder: the shared build record goes there
PROBES = f"{L.WORK_DIR}/probes/pr7"  # PR 7's saved probes (RKMON_PROBES/pr<N>)
RECORD = f"{CHECKS}/shared_build"
REPORT = f"{CHECKS}/report.txt"  # the checks job's report: an output of the monitor's
STATE = f"{L.MONITOR_HOME}/state/proof.json"  # a file of the monitor's state
RD = f"{L.WORK_DIR}/tmp/v7a"  # a review directory in the scratch folder (paths.tmp: the RAM disk is off)
TRY = "rk-isolation-try"  # the name of each file a runner tries to make where it may write nothing
FUNCS = f"{L.WORK_DIR}/jobfuncs.sh"
SESSION_PID = f"{L.WORK_DIR}/session_pid"  # a process of the session's, whose environment no runner reads
BODY = f"{L.WORK_DIR}/body.md"  # a post's body
LINK = f"{L.WORK_DIR}/link.md"  # a link to the write login's file
LATER = f"{L.WORK_DIR}/later.md"  # a name that names nothing yet
# The job scripts' functions, and the proof's own: `pc [-C DIR] CMD...` is prc with a first line that names the user CMD
# runs as (RAN), so a check that wants CMD to fail cannot pass when nothing ran; `other` names a slot not the job's.
PROOF_FUNCS = r"""
pc() { local c=(); if [ "${1:-}" = -C ]; then c=(-C "$2"); shift 2; fi
  prc "${c[@]}" sh -c 'echo "RAN as $(id -un)" >&2; exec "$@"' sh "$@"; }
other() { if [ "$SLOT" = @LAST@ ]; then echo @BEFORE_LAST@; else echo @LAST@; fi; }
"""
RAN = f"RAN as {L.PR_GROUP}"
# A job script's start, as a check job's: its job folder, a key its environment should never pass on, and a slot.
SLOT7 = (
    f"export RKMON_JOB={JOB} ANTHROPIC_API_KEY={KEY}; . {FUNCS}; "
    "prslot 7 || { echo NO-SLOT; exit 99; }; "
)
# What the launcher sets itself, and what a shell adds, beside the listed variables (worker/pr_env.py).
LAUNCHER_NAMES = {"HOME", "TMPDIR", "RKMON_OUT", "RKMON_BUILD", "USER", "LOGNAME"}
LAUNCHER_NAMES |= {
    "GIT_NO_LAZY_FETCH",
    "GIT_CONFIG_COUNT",
    "PWD",
    "OLDPWD",
    "SHLVL",
    "_",
}
NEVER = (
    "GH_CONFIG_DIR",
    "ANTHROPIC_API_KEY",
    "RKMON_WRITE_BROKER",
    "RKMON_PR_RUNNER",
    "RKMON_HOME",
    "SUDO_",
)
ENV_PROBE = (
    "import os; e = os.environ; print('NAMES', *sorted(e)); "
    "print('VALUES', *(e.get(k) for k in "
    "('GIT_NO_LAZY_FETCH', 'USER', 'HOME', 'TMPDIR', 'RKMON_OUT', 'RKMON_BUILD'))); "
    f"print('FIXED', *(k + '=' + str(e.get(k)) for k in {sorted(pr_env.FIXED)!r}))"
)
CMAKE = """cmake_minimum_required(VERSION 3.20)
project(isolation NONE)
enable_testing()
add_test(NAME the_build_is_there COMMAND ${CMAKE_COMMAND} -E true)
"""
END_SLOTS = (
    f"import sys; sys.path.insert(0, {PKG!r}); from rkmon_lib import review; "
    "print(*(review.end_slots({}) or ['nothing to say']), sep='\\n')"
)


def start_step(name: str) -> str:
    """A shell line that runs one step of the start, the image's entry point's function `name` (docker/entrypoint.sh),
    with the entry point's rk, as the start runs it: the step from its one home, not a copy of it.
    """
    return f"RK={PKG}/rk; eval \"$(sed -n '/^{name}() {{/,/^}}/p' {PKG}/docker/entrypoint.sh)\"; {name}"


class Proof:
    def __init__(self):
        self.n = self.failed = 0

    def run(
        self,
        user: str,
        cmd: str,
        env: dict | None = None,
        timeout: float = 600,
        stdin: str | None = None,
    ):
        """CMD in bash as USER, with the image's environment and ENV, and STDIN on its standard input: (its exit code,
        its output and its errors)."""
        argv = ["bash", "-c", cmd]
        if user != "root":
            argv = [RUNUSER, "-u", user, "--", *argv]
        try:
            p = subprocess.run(
                argv,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=dict(SESSION_ENV, **(env or {})),
                cwd=L.WORK_DIR,
            )
        except subprocess.TimeoutExpired:
            return 124, "(timed out)"
        return p.returncode, (p.stdout + p.stderr)

    def leaked(self, out: str) -> list:
        """The secrets OUT holds: here, the fake ones."""
        return [s for s in SECRETS if s in out]

    def expect(self, good: bool, what: str, out: str = "") -> bool:
        """One check's verdict: PASS when `good`, no secret is in `out` (`leaked`), and sudo wrote its audit record of
        each call `out` comes from (SUDO_UNAUDITED is not in it)."""
        leaked = self.leaked(out)
        why = [f"read {leaked[0]}"] if leaked else []
        if SUDO_UNAUDITED in out:
            why.append("sudo wrote no audit record")
        good = good and not why
        self.n += 1
        self.failed += not good
        verdict = "PASS" if good else "FAIL" + (f" ({'; '.join(why)})" if why else "")
        print(f"[{self.n:02d}] {verdict}: {what}", flush=True)
        return good

    def check(self, want: str, what: str, user: str, cmd: str, need: str = "", **kw):
        """want: ok (exit 0), fail (exit not 0), reached (past the broker: rk's own guards answer), or a text the
        output must hold; `need`, a text it must hold besides. Any fake secret in the output fails the check, and so
        does sudo's word that it wrote no audit record (SUDO_UNAUDITED).
        """
        rc, out = self.run(user, cmd, **kw)
        if want == "ok":
            good = rc == 0
        elif want == "fail":
            good = rc != 0
        elif want == "reached":
            good = "the write broker" not in out
        else:
            good = want in out
        good = self.expect(good and need in out, what, out)
        print(f"     as {user}: {cmd[:240]}\n     exit {rc}")
        for line in out.strip().splitlines()[: 4 if good else 40]:
            print(f"     | {line[:200]}")
        return out

    def note(self, what: str, user: str, cmd: str) -> None:
        """What the container lets a user do, said and not judged."""
        rc, out = self.run(user, cmd)
        print(f"NOTE: {what}\n     as {user}: {cmd[:240]}\n     exit {rc}")
        for line in out.strip().splitlines()[:8]:
            print(f"     | {line[:200]}")


def what_is_at(path: str):
    """What is at `path`, to compare before and after (as root): its inode, size, mode and times; None when nothing."""
    try:
        s = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    return (s.st_ino, s.st_size, s.st_mode, s.st_mtime_ns, s.st_ctime_ns)


def setup(p: Proof) -> None:
    """As root: fake logins in the layouts the checks need, a configuration, the clone, PR 7's tree, a file of the
    monitor's state, the checks job's report, the job scripts' functions; then the start's layout step, as each start
    runs it, gives the trees their modes, and the session user saves PR 7's probe in the probes tree it made.
    """
    gid = int(
        subprocess.run(["id", "-g", "rkmon"], capture_output=True, text=True).stdout
    )
    for d in (
        OLDER_WRITE_LOGIN,
        READ_LOGIN,
        OLDER_ATLASSIAN,
        LOGS,
        L.MONITOR_HOME,
        os.path.dirname(CLAUDE_LOGIN),
        f"{WT}/rk/tests",
        f"{WT}/cm",
    ):
        os.makedirs(d, exist_ok=True)
    pathlib.Path(f"{OLDER_WRITE_LOGIN}/{gh.HOSTS_FILE}").write_text(
        f"github.com:\n    oauth_token: {WRITE_TOKEN}\n    user: fake-writer\n    git_protocol: https\n"
    )
    pathlib.Path(READ_HOSTS).write_text(
        f"github.com:\n    oauth_token: {READ_TOKEN}\n    user: fake-reader\n    git_protocol: https\n"
    )
    pathlib.Path(ATLASSIAN_FILE).write_text(json.dumps({"token": ATL_TOKEN}))
    pathlib.Path(CLAUDE_KEY).write_text(f"{KEY}\n")
    pathlib.Path(CLAUDE_LOGIN).write_text(json.dumps({"key": KEY}))
    pathlib.Path(f"{WT}/rk/tests/test_secret.py").write_text(LEAK_TEST)
    pathlib.Path(f"{WT}/cm/CMakeLists.txt").write_text(CMAKE)
    last, before_last = L.PR_RUNNERS[-1].name, L.PR_RUNNERS[-2].name
    pathlib.Path(FUNCS).write_text(
        prcode.SHELL
        + PROOF_FUNCS.replace("@LAST@", last).replace("@BEFORE_LAST@", before_last)
    )
    mine = [L.SECRETS_DIR, LOGS, L.MONITOR_HOME, CLAUDE, WT]
    subprocess.run(["chown", "-R", f"rkmon:{gid}", *mine, FUNCS], check=True)
    subprocess.run(["chmod", "-R", "go-rwx", L.SECRETS_DIR], check=True)
    os.chmod(CLAUDE_LOGIN, 0o600)
    # /secrets-write as a new volume has it: the write user's, mode 700, empty (the login is still in /secrets/gh)
    subprocess.run(["chown", f"rkwrite:{gid}", L.WRITE_SECRETS_DIR], check=True)
    os.chmod(L.WRITE_SECRETS_DIR, 0o700)
    cfg = json.load(open(f"{PKG}/config.defaults.json"))
    cfg.pop("_doc", None)
    cfg.update(
        features={"review_automation": {"mode": "live"}},
        repo="ROCm/r",
        rocke_path="a/b/rocke",
        operator={"github_login": "fake-writer"},
        scope={"labels": ["x"], "path_labels": []},
    )
    # config.defaults.json carries the package's own keys only, no paths: the working folder is the container's
    cfg.setdefault("paths", {})["working_dir"] = L.WORK_DIR
    pathlib.Path(CONFIG).write_text(json.dumps(cfg, indent=1))
    subprocess.run(["chown", f"rkmon:{gid}", CONFIG], check=True)
    git = "git -c user.name=fake -c user.email=fake@example.com"
    p.check(
        "ok",
        "the session user makes the clone (blobless, from a local remote), PR 7's tree and the files the checks use",
        "rkmon",
        f"set -e; printf 'body\\n' > {BODY}; ln -s {WRITE_HOSTS} {LINK}; "
        f"mkdir -p {CHECKS} {os.path.dirname(STATE)}; printf 'the checks report\\n' > {REPORT}; "
        f"printf '{{}}\\n' > {STATE}; "
        f"git init -q -b develop {ORIGIN}; printf 'the clone\\n' > {ORIGIN}/README.md; "
        f"{git} -C {ORIGIN} add README.md; {git} -C {ORIGIN} commit -qm one; "
        f"git -C {ORIGIN} config uploadpack.allowFilter true; "
        f"git -C {ORIGIN} config uploadpack.allowAnySHA1InWant true; "
        f"git clone -q --filter=blob:none --no-checkout file://{ORIGIN} {REPO}; "
        f"git init -q {WT}; {git} -C {WT} add -A; {git} -C {WT} commit -qm pr7",
    )
    p.check(
        f"{paths.worker_module('perms').TREE_FILE_MODE:04o}",
        "the start's step keeps the clone's own files a tree's, the runners' group's to read (docker/entrypoint.sh "
        "shared_repository)",
        "rkmon",
        f"set -e; {start_step('shared_repository')}; git -C {REPO} config core.sharedRepository",
    )
    p.check(
        "ok",
        "the start's layout step (docker/rk_layout.py, as root) gives them their permissions",
        "root",
        f"python3 -I {PKG}/docker/rk_layout.py",
    )
    p.check(
        f"leak.sh {L.PR_GROUP} 640",
        "the session user saves PR 7's probe, as the drafter does: in the probes tree the start made, which the "
        "runners' group reads",
        "rkmon",
        f"umask 027; mkdir -p {PROBES} && printf %s {shlex.quote(LEAK_PROBE)} > {PROBES}/leak.sh && "
        f"stat -c '%n %G %a' {PROBES}/leak.sh | sed 's|.*/||'",
    )
    p.run(
        "rkmon",
        f"sleep 900 >/dev/null 2>&1 & echo $! > {SESSION_PID}; disown",
        env={"FAKE_SESSION_KEY": KEY},
    )


def root_step(p: Proof) -> None:
    print(
        "\n--- the start's root step: an older layout's write login moves to the write user's volume"
    )
    p.check(
        "fail",
        "the session user cannot run it: no sudo rule names it",
        "rkmon",
        f"sudo -n {ROOT_STEP} start",
    )
    p.check(
        "moved the GitHub write login",
        "the move runs as root, as the entry point's root phase runs it, saying what it did",
        "root",
        ROOT_START,
    )
    p.check(
        "ok",
        "the login is the write user's now, mode 600 in a 700 folder",
        "root",
        f"test \"$(stat -c '%U %a' {WRITE_HOSTS})\" = 'rkwrite 600' && "
        f"test \"$(stat -c '%U %a' {WRITE_LOGIN})\" = 'rkwrite 700' && ! test -e {OLDER_WRITE_LOGIN}",
    )
    p.check(
        "ok",
        "again: nothing to do, nothing printed",
        "root",
        f'test -z "$({ROOT_START})"',
    )
    p.check("fail", "sudo as root for anything else", "rkmon", "sudo -n true")


def write_login(p: Proof) -> None:
    b = BROKER
    print("\n--- the write login, as the session user")
    p.check("fail", "its folder is not listable", "rkmon", f"ls -la {WRITE_LOGIN}")
    p.check("fail", "its file is not readable", "rkmon", f"cat {WRITE_HOSTS}")
    p.check(
        "fail",
        "python cannot open it either",
        "rkmon",
        f"python3 -I -c 'print(open(\"{WRITE_HOSTS}\").read())'",
    )
    p.check(
        "fail",
        "gh pointed at the write folder",
        "rkmon",
        f"GH_CONFIG_DIR={WRITE_LOGIN} gh auth token",
    )
    p.check(
        "fail",
        "gh, a write under the write folder",
        "rkmon",
        f"GH_CONFIG_DIR={WRITE_LOGIN} gh api -X POST repos/ROCm/r/issues/1/comments -f body=x",
    )
    p.check(
        "fail",
        "gh with no folder named (gh's own default has no login)",
        "rkmon",
        "env -u GH_CONFIG_DIR gh auth token",
    )
    p.check(
        "ok",
        "an alias made under the read login",
        "rkmon",
        f"GH_CONFIG_DIR={READ_LOGIN} gh alias set wr 'api -X POST repos/ROCm/r/issues/1/comments -f body=x'",
    )
    p.check(
        "fail",
        "that alias run against the write folder",
        "rkmon",
        f"GH_CONFIG_DIR={WRITE_LOGIN} gh wr",
    )
    p.check(
        "fail",
        "an alias written into the write folder",
        "rkmon",
        f"GH_CONFIG_DIR={WRITE_LOGIN} gh alias set tok 'auth token'",
    )
    p.check(
        "fail",
        "a python subprocess: gh auth token under the write folder",
        "rkmon",
        f'python3 -I -c \'import os, subprocess, sys; e = dict(os.environ, GH_CONFIG_DIR="{WRITE_LOGIN}"); '
        'sys.exit(subprocess.run(["gh", "auth", "token"], env=e).returncode)\'',
    )
    p.check(
        "ok",
        "the read login is the session's (its token's length only)",
        "rkmon",
        f"GH_CONFIG_DIR={READ_LOGIN} gh auth token | wc -c",
    )
    print("\n--- sudo: the two rules only")
    p.check(
        "rk_prcode.py",
        "sudo -l lists the broker and the PR runners' launcher",
        "rkmon",
        "sudo -n -l",
        need="rk_write.py",
    )
    p.check(
        "fail",
        "sudo gh as the write user",
        "rkmon",
        "sudo -n -u rkwrite /usr/local/bin/gh auth token",
    )
    p.check(
        "fail",
        "sudo cat as the write user",
        "rkmon",
        f"sudo -n -u rkwrite /bin/cat {WRITE_HOSTS}",
    )
    p.check(
        "fail",
        "sudo python as the write user",
        "rkmon",
        "sudo -n -u rkwrite /usr/bin/python3 -c 'print(1)'",
    )
    p.check(
        "fail",
        "the broker with the caller's environment kept (-E)",
        "rkmon",
        f"sudo -n -E -u rkwrite {b} post 7 {BODY} --dry-run",
    )
    p.check(
        "fail",
        "the broker with a variable set",
        "rkmon",
        f"sudo -n -u rkwrite GH_CONFIG_DIR=/tmp {b} post 7 {BODY} --dry-run",
    )
    print("\n--- the broker's own refusals")
    for what, args in (
        ("a command that is not a GitHub write (status)", "status"),
        ("approve", "approve 7"),
        ("merge-now", "merge-now 7"),
        ("dismiss", "dismiss 7"),
        (
            "post --premises (its AFTER lines would run as the write user)",
            f"post 7 {BODY} --premises {BODY}",
        ),
        ("post --prem (an abbreviation)", f"post 7 {BODY} --prem={BODY}"),
        ("spec-post", "spec-post 7 --round fu1"),
        (
            "post of the write login's file",
            f"post 7 {WRITE_HOSTS} --dry-run",
        ),
        ("post of a link to it", f"post 7 {LINK} --dry-run"),
        (
            "post of a name that names nothing yet (made later, as a link)",
            f"post 7 {LATER} --dry-run",
        ),
    ):
        p.check("fail", what, "rkmon", f"sudo -n -u rkwrite {b} {args}")
    p.check(
        "is set from the host only",
        "a secret the writer keeps, set through sudo (the host sets it, as the writer)",
        "rkmon",
        f"sudo -n -u rkwrite {b} secret-set teams_webhook < /dev/null",
    )
    p.check(
        "fail",
        "rk post of the link, routed by rk",
        "rkmon",
        f"rk post 7 {LINK} --dry-run",
    )
    print("\n--- rk's write commands reach the broker")
    p.check(
        "reached",
        "rk post --dry-run (offline: rk's own guards answer)",
        "rkmon",
        f"rk post 7 {BODY} --dry-run",
    )
    p.check(
        "ok",
        "the broker logged it, as rkwrite, with the checked copy",
        "rkmon",
        "grep -E 'rk post 7 /home/rkwrite/.rk-write-args/run-[^/]+/body[.]md --dry-run \\(as rkwrite\\)' "
        + writer_run.LOG,
    )
    p.check(
        "fail",
        "the session cannot read the copies",
        "rkmon",
        "ls /home/rkwrite/.rk-write-args",
    )
    p.check(
        "ok",
        "the broker removed the run's copies when rk ended",
        "root",
        'test -d /home/rkwrite/.rk-write-args && test -z "$(ls -A /home/rkwrite/.rk-write-args)"',
    )
    p.check(
        "ok",
        "rk gh-logins --side write (the write probe through the broker)",
        "rkmon",
        "rk gh-logins --side write --json",
    )
    p.check(
        "ok",
        "the broker logged the probe",
        "rkmon",
        f"grep -F 'rk gh-logins --side write --json (as rkwrite)' {writer_run.LOG}",
    )
    print(
        "\n--- a configuration that would make rk write where the write user keeps its login"
    )
    edit = (
        f'python3 -I -c \'import json; p = "{CONFIG}"; d = json.load(open(p)); '
        'd["paths"]["posts"] = "{v}"; json.dump(d, open(p, "w"))\''
    )
    p.run("rkmon", edit.format(v=WRITE_LOGIN))
    p.check(
        "fail",
        f"rk post --dry-run with paths.posts in {L.WRITE_SECRETS_DIR}",
        "rkmon",
        f"rk post 7 {BODY} --dry-run",
    )
    p.run("rkmon", edit.format(v=""))


def pr_runners(p: Proof) -> None:
    group, slots_dir = L.PR_GROUP, L.SLOTS_DIR
    names = [u.name for u in L.PR_RUNNERS]
    print("\n--- the PR runners: one per slot, in a group of their own")
    ids = "".join(f"{u.uid}:{u.gid}:{u.name} {group};" for u in L.PR_RUNNERS)
    p.check(
        "ok",
        f"{names[0]} to {names[-1]} have the package's ids, each its own group as its primary one, and are in the "
        "runners' group and no other",
        "root",
        f'test "$(for u in {" ".join(names)}; do echo "$(id -u $u):$(id -g $u):$(id -Gn $u)"; done | tr "\\n" ";")" '
        f'= "{ids}"',
    )
    p.check(
        "ok",
        "the monitor's user is in that group (it gives the trees PR code reads that group); the writer is not",
        "root",
        f"id -Gn rkmon | grep -qw {group} && ! id -Gn rkwrite | grep -qw {group}",
    )
    p.check(
        "ok",
        "the start made the slots: each the runner's, group rkmon, 2751",
        "root",
        f"for u in {' '.join(names)}; do "
        f'test "$(stat -c \'%U %G %a\' {slots_dir}/$u)" = "$u rkmon 2751" || exit 1; done',
    )
    print(
        "\n--- PR code as a job script runs it (prslot, prc: the launcher, through its sudo rule)"
    )
    p.check(
        RAN,
        "it runs as the runner of the slot its job took",
        "rkmon",
        SLOT7 + "pc true",
    )
    for what, path in (
        ("the secrets volume", L.SECRETS_DIR),
        ("the read login", READ_HOSTS),
        ("the atlassian login", ATLASSIAN_FILE),
        ("the Claude session's key", CLAUDE_KEY),
        ("the write volume", L.WRITE_SECRETS_DIR),
        ("the write login", WRITE_HOSTS),
        ("Claude's login", CLAUDE_LOGIN),
        ("the broker's copies", "/home/rkwrite/.rk-write-args"),
    ):
        p.check(
            "Permission denied",
            f"it cannot read {what}",
            "rkmon",
            SLOT7 + f"pc ls -la {path}; pc cat {path}",
            need=RAN,
        )
    p.check(
        "Permission denied",
        "it cannot read the session's processes' environments",
        "rkmon",
        SLOT7 + f"pc cat /proc/$(cat {SESSION_PID})/environ",
        need=RAN,
    )
    p.check(
        "fail",
        "gh under the read login, as a runner",
        "rkmon",
        SLOT7 + f"pc env GH_CONFIG_DIR={READ_LOGIN} gh auth token",
        need=RAN,
    )
    p.check(
        "Permission denied",
        "python, as a runner, cannot open the read login",
        "rkmon",
        SLOT7 + f"pc python3 -I -c 'print(open(\"{READ_HOSTS}\").read())'",
        need=RAN,
    )
    for what, cmd in (
        ("the monitor's configuration", f"cat {CONFIG}"),
        ("the monitor's state", f"cat {STATE}"),
        ("the checks job's report", f"cat {REPORT}"),
        ("the logs", f"cat {writer_run.LOG}"),
        ("the monitor's home", f"ls -a {SESSION_ENV['HOME']}"),
        ("the container's first process's environment", "cat /proc/1/environ"),
    ):
        p.check(
            "Permission denied",
            f"it cannot read {what}",
            "rkmon",
            SLOT7 + f"pc {cmd}",
            need=RAN,
        )
    out = p.check(
        "VALUES 1 rkpr",
        "its environment: the listed variables, the slot's folders, git that fetches nothing",
        "rkmon",
        SLOT7 + "pc python3 -I -c " + shlex.quote(ENV_PROBE),
        need=RAN,
    )
    have = next(
        (ln.split()[1:] for ln in out.splitlines() if ln.startswith("NAMES ")), []
    )
    values = next(
        (ln.split()[1:] for ln in out.splitlines() if ln.startswith("VALUES ")), []
    )
    fixed = next(
        (ln.split()[1:] for ln in out.splitlines() if ln.startswith("FIXED ")), []
    )
    odd = [
        n
        for n in have
        if not (
            pr_env.listed(n)
            or n in pr_env.FIXED
            or n in LAUNCHER_NAMES
            or re.fullmatch(r"GIT_CONFIG_(KEY|VALUE)_\d+", n)
        )
    ]
    p.expect(
        bool(have) and not odd,
        f"every variable it has is listed, fixed or the launcher's ({odd or 'none other'})",
    )
    p.expect(
        fixed == [f"{k}={v}" for k, v in sorted(pr_env.FIXED.items())],
        f"what pr_env fixes is in it, as fixed ({' '.join(fixed) or 'none'})",
    )
    for name in NEVER:
        p.expect(
            bool(have) and not any(n.startswith(name) for n in have),
            f"its environment has no {name}",
        )
    slot = f"{slots_dir}/{values[1]}" if len(values) == 6 else "?"
    p.expect(
        values[2:] == [f"{slot}/home", f"{slot}/tmp", f"{slot}/out", f"{slot}/build"],
        f"HOME, TMPDIR, RKMON_OUT and RKMON_BUILD are its slot's folders ({' '.join(values[2:])})",
    )
    p.check(
        "fail", "sudo as a runner: none", "rkmon", SLOT7 + "pc sudo -n true", need=RAN
    )
    p.check(
        "fail",
        "sudo as a runner: not the broker either",
        "rkmon",
        SLOT7 + f"pc sudo -n -u rkwrite {BROKER} status",
        need=RAN,
    )
    p.check(
        "fail",
        "the launcher with sudo's -E",
        "rkmon",
        f"printf '' | sudo -n -E -u {names[0]} {RUNNER} run -- id -un",
    )
    p.check(
        "fail",
        "the launcher as the writer",
        "rkmon",
        f"printf '' | sudo -n -u rkwrite {RUNNER} run -- id -un",
    )
    print("\n--- what a runner writes, and what it reads of the slots")
    tries = [
        ("the work volume's top", f"{L.WORK_DIR}/x"),
        ("the monitor's home (its configuration and its state)", f"{L.MONITOR_HOME}/x"),
        ("the logs", f"{LOGS}/x"),
        ("the clone", f"{REPO}/x"),
        ("the PR's tree", f"{WT}/x"),
        ("the probes", f"{PROBES}/x"),
        ("the write user's home", "/home/rkwrite/x"),
        ("/tmp", "/tmp/x"),
        ("/var/tmp", "/var/tmp/x"),
        ("the RAM disk's top", f"{L.RAMDISK}/x"),
        ("the slots' folder", f"{slots_dir}/x"),
        ("the slots' records", f"{L.SLOT_LOCKS}/x"),
        ("the inputs the monitor hands its job", '"$SI/x"'),
        ("another slot", f"{slots_dir}/$(other)/x"),
        ("the clone's git configuration", f"{REPO}/.git/config"),
        ("the clone's hooks", f"{REPO}/.git/hooks/post-checkout"),
        ("the clone's attributes", f"{REPO}/.git/info/attributes"),
        ("the clone's objects", f"{REPO}/.git/objects/{TRY}"),
        ("the PR tree's git configuration", f"{WT}/.git/config"),
        ("the PR tree's hooks", f"{WT}/.git/hooks/pre-commit"),
        ("the monitor's state", f"{os.path.dirname(STATE)}/{TRY}"),
        ("the checks job's outputs", f"{CHECKS}/{TRY}"),
        ("Claude's folder", f"{CLAUDE}/{TRY}"),
        ("the monitor's home's start file", f"{SESSION_ENV['HOME']}/.bashrc"),
        ("the scratch folder", f"{os.path.dirname(RD)}/{TRY}"),
        ("the image's code", f"{PKG}/rkmon_lib/{TRY}.py"),
        ("the image's agent files", f"{PKG}/agentic/{TRY}.md"),
        ("a program on the search path", "/usr/local/bin/git"),
        ("the system's Python", f"/usr/lib/python3/dist-packages/{TRY}.pth"),
        ("the shells' start files", f"/etc/profile.d/{TRY}.sh"),
    ]
    tries += [(f"the {c} cache", f"{L.WORK_DIR}/cache/{c}/{TRY}") for c in paths.CACHES]
    # the image's virtual environments (docker/Dockerfile)
    for venv in sorted(glob.glob("/opt/rkmon/venv*")):
        site = glob.glob(f"{venv}/lib/python3*/site-packages") or [f"{venv}/lib"]
        tries += [
            (f"{venv}'s programs", f"{venv}/bin/{TRY}"),
            (f"{venv}'s packages", f"{site[0]}/{TRY}.pth"),
        ]
    before = {path: what_is_at(path) for _what, path in tries if "$" not in path}
    for what, path in tries:
        p.check(
            "Permission denied",
            f"it cannot write {what}",
            "rkmon",
            SLOT7 + f"pc touch {path}",
            need=RAN,
        )
    p.check(
        "Permission denied",
        "nor the clone's configuration through git's own command (a core.fsmonitor or a hooks path set there would run "
        "as the monitor)",
        "rkmon",
        SLOT7
        + f"pc -C {REPO} git config core.fsmonitor {shlex.quote(f'touch /tmp/{TRY}')}",
        need=RAN,
    )
    before[f"/tmp/{TRY}"] = None
    changed = [path for path, was in before.items() if what_is_at(path) != was]
    p.expect(
        len(before) > len(tries) // 2 and not changed,
        f"no try made or changed a file ({' '.join(changed) or 'none did'})",
    )
    for what, path in (
        ("the work volume", L.WORK_DIR),
        ("/tmp", "/tmp"),
        ("the slots' records", L.SLOT_LOCKS),
        ("another slot", f"{slots_dir}/$(other)"),
    ):
        p.check(
            "Permission denied",
            f"it cannot list {what}",
            "rkmon",
            SLOT7 + f"pc ls {path}",
            need=RAN,
        )
    p.check(
        "ok",
        "it writes its slot's four folders: out, build, tmp and home",
        "rkmon",
        SLOT7
        + 'pc sh -c \'touch "$RKMON_OUT/o" "$RKMON_BUILD/b" "$TMPDIR/t" "$HOME/h"\'',
        need=RAN,
    )
    p.check(
        "the job's result",
        "the monitor reads its results through the result reader (prr)",
        "rkmon",
        SLOT7
        + 'pc sh -c \'echo "the job\'"\'"\'s result" > "$RKMON_OUT/r.txt"\'; '
        + f"prr r.txt {CHECKS}; cat {CHECKS}/r.txt",
        need=RAN,
    )
    plant = (
        f'ln -s {CONFIG} "$RKMON_OUT/r.txt" && ln -s {L.MONITOR_HOME} "$RKMON_OUT/d" && '
        'echo "planted as $(id -un)"'
    )
    target = f"stat -c '%i %s %a %Y' {CONFIG} {STATE}"
    out = p.check(
        "untouched",
        "links planted in out, to a file and to a folder of the monitor's: the result reader refuses both and copies "
        "nothing, and emptying the slot removes the links, not what they name",
        "rkmon",
        SLOT7
        + f"b=$({target}); pc sh -c {shlex.quote(plant)}; "
        + f"prr r.txt {CHECKS}/links; prr d {CHECKS}/links; "
        + f'echo "copied: $(ls -A {CHECKS}/links 2>/dev/null)."; prfree; '
        + 'test -L "$SO/r.txt" || test -L "$SO/d" || echo "links gone"; '
        + f'test "$({target})" = "$b" && echo untouched',
        need=f"planted as {L.PR_GROUP}",
    )
    p.expect(
        out.count("a link, or not readable") == 2
        and "copied: ." in out
        and "links gone" in out,
        "(both refused, nothing copied, the links gone)",
        out,
    )
    p.check(
        "ok",
        "it writes its own entry in /dev/shm",
        "rkmon",
        SLOT7 + f"pc sh -c 'echo x > {L.SHM_DIR}/rk-isolation'",
        need=RAN,
    )
    p.check(
        "ok",
        "which is gone once its job's slot is freed",
        "root",
        f"! test -e {L.SHM_DIR}/rk-isolation",
    )
    print("\n--- its processes")
    leave = (
        "(setsid env -i /bin/bash -c 'trap \"\" TERM HUP INT; exec -a innocuous /bin/sleep 600' "
        ">/dev/null 2>&1 </dev/null &); sleep 0.5; "
        'pgrep -u "$(id -u)" -xf \'innocuous 600\' >/dev/null && echo "left one as $(id -un)"; '
        "echo 'the result' > \"$RKMON_OUT/left.txt\""
    )
    p.check(
        "read with 0 left",
        "every process its command left is stopped when the command ends, before its results are read: one in a "
        "session of its own, its environment cleared, deaf to SIGTERM, under another name",
        "rkmon",
        SLOT7
        + f"pc sh -c {shlex.quote(leave)}; "
        + f'n=$(pgrep -u "$SLOT" | wc -l); prr left.txt {CHECKS} && echo "read with $n left"',
        need=f"left one as {L.PR_GROUP}",
    )
    out = p.check(
        "none left",
        "the launcher's kill stops a runner's processes",
        "rkmon",
        SLOT7
        + "(prc sleep 600 >/dev/null 2>&1 &); "
        + 'for i in $(seq 20); do pgrep -u "$SLOT" -x sleep >/dev/null && break; sleep 0.5; done; '
        + 'echo "RUNS $(pgrep -u "$SLOT" -x sleep)"; '
        + 'kill $(pgrep -u "$SLOT" -x sleep) && echo "the session signalled it"; '
        + 'sudo -n -u "$SLOT" "$RKMON_PR_RUNNER" kill; sleep 1; '
        + 'pgrep -u "$SLOT" -x sleep || echo "none left"',
        need="stopped",
    )
    p.expect(
        re.search(r"RUNS \d+", out) is not None
        and "Operation not permitted" in out
        and "the session signalled it" not in out,
        "which the session user cannot signal itself",
        out,
    )
    print("\n--- named work and other work")
    p.check(
        "kept its home",
        "named work keeps its folders from one command to the next, until it ends",
        "rkmon",
        'rk pr-run --slot pr7-isolation -- sh -c \'echo "kept its home" > "$HOME/n"\' && '
        "rk pr-run --slot pr7-isolation -- sh -c 'cat \"$HOME/n\"' && rk pr-run --slot pr7-isolation --end && "
        "{ rk pr-slot status | grep 'named work' || echo 'no named work left'; }",
        need="no named work left",
    )
    p.check(
        "fresh",
        "other work gets fresh folders",
        "rkmon",
        "rk pr-run -- sh -c 'echo x > \"$HOME/f\"' && "
        'rk pr-run -- sh -c \'test -z "$(ls -A "$HOME")" && echo fresh\'',
    )
    p.check(
        "named work pr7-left",
        "named work left without its end keeps its slot while its owner runs",
        "rkmon",
        "rk pr-run --slot pr7-left -- sh -c 'echo x > \"$HOME/left\"' && rk pr-slot status",
    )
    p.check(
        "no named work left",
        "the review's end frees it (review.end_slots, at each advance of the reviews: no review of PR 7 is in "
        "flight, and its owner is gone)",
        "rkmon",
        f"python3 -B -c {shlex.quote(END_SLOTS)} && "
        "{ rk pr-slot status | grep 'named work' || echo 'no named work left'; }",
    )
    p.check(
        "ok",
        "its folder emptied",
        "root",
        f'test -z "$(find {slots_dir} -name left)"',
    )
    print("\n--- git in a slot: the job's trees only, and no fetch")
    p.check(
        "one",
        "git reads the tree its command runs in (prc -C: the launcher takes that tree as safe)",
        "rkmon",
        SLOT7 + f"pc -C {REPO} git log --format=%s -1",
        need=RAN,
    )
    p.check(
        "dubious ownership",
        "git refuses a tree the job did not name",
        "rkmon",
        SLOT7 + f"pc git -C {WT} log --format=%s -1",
        need=RAN,
    )
    p.check(
        "pr7",
        f"and reads it once the job names it ({prcode.TREES_VAR})",
        "rkmon",
        SLOT7 + f"{prcode.TREES_VAR}={WT} pc git -C {WT} log --format=%s -1",
        need=RAN,
    )
    out = p.check(
        "Not a valid object name",
        "a blob the clone lacks: git in a slot reads none",
        "rkmon",
        SLOT7 + f"pc -C {REPO} sh -c 'GIT_TRACE=1 git cat-file -p HEAD:README.md'",
        need=RAN,
    )
    p.expect(
        "trace: built-in: git cat-file" in out
        and not re.search(r"run_command:.*\bfetch\b", out),
        "and fetches nothing: git's trace shows no fetch tried (GIT_NO_LAZY_FETCH)",
        out,
    )
    p.check(
        "the clone",
        "the monitor fetches it (git as the session user)",
        "rkmon",
        f"git -C {REPO} cat-file -p HEAD:README.md",
    )
    p.check(
        "the clone",
        "then git in a slot reads it",
        "rkmon",
        SLOT7 + f"pc -C {REPO} git cat-file -p HEAD:README.md",
        need=RAN,
    )
    hook = "printf '%s-%s\\n' FSMONITOR RAN >&2; false"
    make = f"git init -q r && git -C r config core.fsmonitor {shlex.quote(hook)} && echo made as $(id -un)"
    out = p.check(
        "dubious ownership",
        "the monitor's git refuses a repository a runner made in its slot (its core.fsmonitor would run as the "
        "monitor)",
        "rkmon",
        SLOT7 + f'pc -C "$SB" sh -c {shlex.quote(make)}; git -C "$SB/r" status',
        need=f"made as {L.PR_GROUP}",
    )
    p.expect("FSMONITOR-RAN" not in out, "(its core.fsmonitor never ran)", out)
    print("\n--- what a runner shares with the others")
    p.check(
        "fail",
        "it makes no namespace (unshare)",
        "rkmon",
        SLOT7 + "pc unshare -U true",
        need=RAN,
    )
    p.check(
        "fail",
        "nor through bubblewrap",
        "rkmon",
        SLOT7 + "pc bwrap --ro-bind / / true",
        need=RAN,
    )
    out = p.check(
        "ok",
        "the folders and files every user writes, and the ones the runners' group writes",
        "root",
        "find / \\( -path /proc -o -path /sys \\) -prune -o \\( -perm -0002 ! -type l ! -type c \\) -print "
        f"-o \\( -group {group} -perm -0020 ! -type l \\) -print",
    )
    shared = sorted(set(out.split()))
    p.expect(
        set(shared) <= set(SHARED_KERNEL),
        f"none but the kernel's folders of shared memory and message queues ({' '.join(shared)})",
        out,
    )
    p.note("a mount namespace, as root", "root", "unshare -m true")
    p.note("a tmpfs, as root", "root", "mount -t tmpfs tmpfs /mnt")
    p.note("root's capabilities", "root", "grep -E '^Cap(Eff|Bnd)' /proc/self/status")
    seen = (
        "awk 'NR > 1 && $1 == "
        + str(SYSV_KEY)
        + ' {f = 1} END {print f ? "made" : "none"}\' /proc/sysvipc/shm'
    )
    p.check(
        "SHM made then none",
        "its System V shared memory, which every user's processes share, is gone once its job's slot is freed",
        "rkmon",
        SLOT7
        + f"pc python3 -I -c {shlex.quote(SHMGET)}; "
        + f'b=$({seen}); prfree; echo "SHM $b then $({seen})"',
        need=RAN,
    )
    queue = f"{L.MQUEUE_DIR}/rk-isolation"
    seen = f"test -e {queue} && echo made || echo none"
    p.check(
        "MQUEUE made then none",
        f"its POSIX message queue, in {L.MQUEUE_DIR}, is gone once its job's slot is freed",
        "rkmon",
        SLOT7
        + f"pc python3 -I -c {shlex.quote(MQ_OPEN)}; "
        + f'b=$({seen}); prfree; echo "MQUEUE $b then $({seen})"',
        need=RAN,
    )


# The folders the kernel's shared memory and message queues live in: every user's, sticky (docker/rk_prcode.py removes a
# runner's own entries in them when it begins, keeps or ends its slot, and its System V objects).
SHARED_KERNEL = (L.SHM_DIR, L.MQUEUE_DIR)
SYSV_KEY = 0x524B4953  # a System V shared memory segment's key: "RKIS"
SHMGET = (
    "import ctypes; l = ctypes.CDLL(None, use_errno=True); "
    f'print("shmget", l.shmget({SYSV_KEY}, 4096, 0o1666))'
)
MQ_OPEN = (
    "import ctypes, os; l = ctypes.CDLL(None, use_errno=True); "
    'print("mq_open", l.mq_open(b"/rk-isolation", os.O_CREAT | os.O_RDWR, 0o666, None))'
)


def scratch_folder(p: Proof) -> None:
    wt, data = f"{RD}/wt_v1", f"{RD}/data"
    print("\n--- the scratch folder: a review directory, and the trees made in it")
    p.check(
        f"{L.PR_GROUP} 2750 {L.PR_GROUP} 2750",
        "the session makes a review directory with its notes, a worktree and a folder of files PR code reads, each "
        "made a tree first, as its agents add theirs (`rk pr-tree` first, then `git worktree add --detach`)",
        "rkmon",
        f"umask 027; mkdir -p {RD} && echo 'the review notes' > {RD}/notes.md && rk pr-tree {wt} {data} && "
        f"git -C {REPO} worktree add -q --detach {wt} HEAD && echo 'echo the data ran' > {data}/step.sh && "
        f"stat -c '%G %a' {wt} {data} | tr '\\n' ' '",
    )
    for what, cmd in (
        ("list the review directory", f"ls {RD}"),
        ("read its notes", f"cat {RD}/notes.md"),
        ("write in it", f"touch {RD}/{TRY}"),
        ("write in the worktree", f"touch {wt}/{TRY}"),
        (
            "change the worktree's pointer to its git folder",
            f"sh -c 'echo gitdir: /tmp > {wt}/.git'",
        ),
        ("write in the folder of files", f"touch {data}/{TRY}"),
    ):
        p.check(
            "Permission denied",
            f"a runner cannot {what}",
            "rkmon",
            SLOT7 + f"pc {cmd}",
            need=RAN,
        )
    p.check(
        "the clone",
        "it reads the worktree's files",
        "rkmon",
        SLOT7 + f"pc cat {wt}/README.md",
        need=RAN,
    )
    p.check(
        "one",
        "and runs git there (prc -C: the launcher takes the worktree as safe)",
        "rkmon",
        SLOT7 + f"pc -C {wt} git log --format=%s -1",
        need=RAN,
    )
    p.check(
        "the data ran",
        "it runs what the session put in the folder of files",
        "rkmon",
        SLOT7 + f"pc sh {data}/step.sh",
        need=RAN,
    )
    where = "sh -c 'echo \"in $(pwd) as $(id -un)\"; cat {}'"
    p.check(
        f"in {wt} as {L.PR_GROUP}",
        "rk pr-run in the worktree runs there and reads it (the agents' runs of PR code)",
        "rkmon",
        f"cd {wt} && rk pr-run -- {where.format('README.md')}",
        need="the clone",
    )
    p.check(
        f"in {RD} as {L.PR_GROUP}",
        "rk pr-run in the review directory runs there, and reads none of it",
        "rkmon",
        f"cd {RD} && rk pr-run -- {where.format('notes.md')}",
        need="Permission denied",
    )
    p.check(
        "a scratch folder (",
        "the start's layout step walks the scratch folder again (given other permissions first, as on a volume an "
        "earlier package made)",
        "root",
        f"chmod 2770 {os.path.dirname(RD)} && python3 -I {PKG}/docker/rk_layout.py",
    )
    p.check(
        "ok",
        "and leaves the trees there as they were",
        "root",
        f"test \"$(stat -c '%G %a' {wt} {wt}/README.md {data}/step.sh | tr '\\n' ' ')\" = "
        f"'{L.PR_GROUP} 2750 {L.PR_GROUP} 640 {L.PR_GROUP} 640 '",
    )
    p.check(
        "the clone",
        "a runner still reads the worktree",
        "rkmon",
        SLOT7 + f"pc cat {wt}/README.md",
        need=RAN,
    )
    p.check(
        "Permission denied",
        "and still not the notes",
        "rkmon",
        SLOT7 + f"pc cat {RD}/notes.md",
        need=RAN,
    )


def kept_slot(p: Proof) -> None:
    slots_dir = L.SLOTS_DIR
    print("\n--- a kept slot: the checks' build, kept for its review")
    build = (
        'cmake -S {wt}/cm -B wt7m -G Ninja >/dev/null && cmake --build wt7m >/dev/null && echo s > "$HOME/h" && '
        'echo s > "$RKMON_OUT/o" && echo s > "$TMPDIR/t" && chmod 777 "$HOME" "$RKMON_OUT" "$TMPDIR" wt7m && '
        "cp /bin/true wt7m/suid && chmod 4755 wt7m/suid && echo built"
    ).format(wt=WT)
    keep = (
        SLOT7
        + f'pc -C "$SB" sh -c {shlex.quote(build)} || exit 98; '
        + f'echo "$SB/wt7m" > {RECORD}; prkeep 7 abc1234 "$SB/wt7m" {RECORD} && echo "KEPT $SLOT"'
    )
    out = p.check(
        "KEPT rkpr",
        "a job builds in its slot (opening its other folders and the build to every user), then keeps the slot for "
        "PR 7's review (prkeep)",
        "rkmon",
        keep,
        need="built",
    )
    m = re.search(r"KEPT (\S+)", out)
    kept = m.group(1) if m else "none"
    top = f"{slots_dir}/{kept}"
    p.check(
        "ok",
        "its other folders are made again: the runner's, 2750, empty",
        "root",
        f"for f in out tmp home; do test \"$(stat -c '%U %a' {top}/$f)\" = '{kept} 2750' && "
        f'test -z "$(ls -A {top}/$f)" || exit 1; done',
    )
    p.check(
        "ok",
        "its build is readable by every runner, writable by none but its own, with no set-user-ID file",
        "root",
        f'test -n "$(ls {top}/build/wt7m)" && test -z "$(find {top}/build -perm /0022 ! -type l)" && '
        f'test -z "$(find {top}/build -type f -perm /6000)" && test -z "$(find {top}/build ! -type l ! -perm -0004)"',
    )
    p.check(
        f"{kept}: kept for #7 at abc1234",
        "rk pr-slot status says so",
        "rkmon",
        "rk pr-slot status",
    )
    p.check(
        "add_test",
        "another slot's job reads the kept build",
        "rkmon",
        SLOT7 + f"pc cat {top}/build/wt7m/CTestTestfile.cmake",
        need=RAN,
    )
    p.check(
        "Permission denied",
        "and cannot write it",
        "rkmon",
        SLOT7 + f"pc touch {top}/build/wt7m/x {top}/build/x",
        need=RAN,
    )
    out = p.check(
        "Permission denied",
        "nor list the kept slot, nor read its other folders",
        "rkmon",
        SLOT7 + f"pc ls {top}; pc ls {top}/home {top}/out {top}/tmp",
        need=RAN,
    )
    p.expect(out.count("Permission denied") >= 4, "(each of the four refused)", out)
    out = p.check(
        "no job runs in a kept slot",
        "the launcher runs no job in it",
        "rkmon",
        f"printf '' | sudo -n -u {kept} {RUNNER} run -- true; echo \"exit $?\"",
    )
    p.expect("exit 3" in out, "(exit 3: the slot cannot be used)", out)
    listing = f"find {top} -printf '%P %s %T@ %m %U\\n' | sort | sha256sum"
    _rc, before = p.run("root", listing)
    p.check(
        "100% tests passed",
        "ctest on the kept build, as PR code in a slot of its own (tools/ctest_read_only.py through rk pr-run)",
        "rkmon",
        f"rk pr-run -- python3 -B {PKG}/tools/ctest_read_only.py {top}/build/wt7m",
    )
    _rc, after = p.run("root", listing)
    p.expect(before == after, "it wrote nothing in the kept slot", before + after)
    p.check(
        f"{kept}: free",
        "the review's end frees it (review.end_slots: no review of PR 7 is in flight), its shared build record removed",
        "rkmon",
        f"python3 -B -c {shlex.quote(END_SLOTS)} && ! test -e {RECORD} && rk pr-slot status",
    )
    p.check(
        "ok",
        "its folder emptied, its record gone",
        "root",
        f'test -z "$(ls -A {top})" && ! test -e {L.SLOT_LOCKS}/{kept}.json',
    )
    out = p.check(
        "KEPT rkpr",
        "kept again",
        "rkmon",
        SLOT7
        + 'prc -C "$SB" mkdir wt7m || exit 98; '
        + f'echo "$SB/wt7m" > {RECORD}; prkeep 7 abc1234 "$SB/wt7m" {RECORD} && echo "KEPT $SLOT"',
    )
    m = re.search(r"KEPT (\S+)", out)
    kept = m.group(1) if m else "none"
    others = len(L.PR_RUNNERS) - 1
    out = p.check(
        f"TOOK {kept}",
        f"with the {others} other slots taken, the next job takes the slot kept longest",
        "rkmon",
        f"pids=''; for i in $(seq {others}); do sleep 600 >/dev/null 2>&1 & pids=\"$pids $!\"; done; "
        'for p in $pids; do rk pr-slot take --pid $p > /dev/null || echo "none for $p"; done; '
        'sleep 600 >/dev/null 2>&1 & last=$!; a=$(rk pr-slot take --pid $last) && eval "$a" && echo "TOOK $SLOT"; '
        f"test -e {RECORD} && echo 'the shared build record stays'; "
        "for p in $pids $last; do rk pr-slot free --pid $p > /dev/null; done; kill $pids $last; rk pr-slot status",
    )
    p.expect(
        "the shared build record stays" not in out and "none for" not in out,
        "and removes its shared build record",
        out,
    )
    p.expect(
        out.count(": free") == len(L.PR_RUNNERS),
        "every slot is free after",
        out,
    )


# A PR's test file and a saved probe that try to read both secrets volumes: they pass (exit 0) only as a PR runner,
# which reads neither; as the session user the read login is readable, the test fails and prints it.
LEAKS = (READ_HOSTS, CLAUDE_KEY, WRITE_HOSTS)  # what they try to read
LEAK_TEST = f"""import os
import pwd
import re


def test_reads_no_secret():
    for path in {LEAKS!r}:
        try:
            print(open(path).read())
        except OSError:
            continue
        raise AssertionError("read " + path)
    assert re.fullmatch(r"rkpr[0-9]+", pwd.getpwuid(os.geteuid()).pw_name)
"""
LEAK_PROBE = f"""if cat {' '.join(LEAKS)} 2>/dev/null; then
    echo "read a secret as $(id -un)"; exit 1
fi
echo "read nothing as $(id -un)"; exit 0
"""
TOOLS_ENV = {
    "RKMON_WORKING_DIR": L.WORK_DIR,
    "RKMON_ROCKE_PATH": "rk",
    "RKMON_VENV": "/opt/rkmon/venv",
    "RKMON_TOOLKIT": PKG,
    "RKMON_PROBES": os.path.dirname(PROBES),
}


def tools_call(mod: str, body: str) -> str:
    """A Python one-liner, as the session user: a worker tools module with the job's RKMON_* paths, then `body`."""
    return f"import sys; sys.path.insert(0, {PKG + '/worker'!r}); import {mod} as rt; env = {TOOLS_ENV!r}; {body}"


def pr_code_callers(p: Proof) -> None:
    print(
        "\n--- every other run of PR code: a PR test file and a saved probe that read for secrets"
    )

    def reviewer(tool: str, args: dict) -> str:
        return "python3 -B -c " + shlex.quote(
            tools_call(
                "reviewer_tools",
                f"t = rt.Tree({PR7!r}, 7, get=lambda k, required=True: env.get(k, '')); "
                f"print(rt.run_tool(t, {tool!r}, {args!r})[0])",
            )
        )

    p.check(
        "1 passed",
        "run_pytest (the local worker's and the offload passes' tool) runs the PR's test as a PR runner",
        "rkmon",
        reviewer("run_pytest", {"paths": "rk/tests/test_secret.py"}),
    )
    p.check(
        f"read nothing as {L.PR_GROUP}",
        "run_probe (a saved probe, the offload passes' tool) runs as a PR runner",
        "rkmon",
        reviewer("run_probe", {"name": "leak", "tree": "head"}),
    )
    drafter = tools_call(
        "review_tools",
        f"t = rt.Tree({PR7!r}, 7, {L.WORK_DIR!r}, 'rk'); "
        "print(rt.run_tool(t, 'probe', {'name': 'leak', 'tree': 'head'})[0])",
    )
    p.check(
        f"read nothing as {L.PR_GROUP}",
        "the drafter's saved probe runs as a PR runner",
        "rkmon",
        "python3 -B -c " + shlex.quote(drafter),
    )
    p.check(
        "1 passed",
        "rk pr-run (the session's own run of a PR test file) runs it as a PR runner",
        "rkmon",
        f"cd {WT}/rk && rk pr-run -- /opt/rkmon/venv/bin/python -B -m pytest -q -p no:cacheprovider "
        "tests/test_secret.py",
    )
    p.check(
        "fail",
        "the same test as the session user fails: it reads the read login (the proof that the test can tell)",
        "rkmon",
        f"set -o pipefail; cd {WT}/rk && /opt/rkmon/venv/bin/python -B -m pytest -q -p no:cacheprovider "
        "tests/test_secret.py | grep -v FAKE-",
    )


def release_archive(name: str, folder: str = "/root") -> str:
    """The image's own code (PKG) as a package's tar archive, as `rkmon apply` hands one to root, with one change that
    tells this release from the image's code and from another release: its writer's runner (WRITER_RUNNER) says MARK
    with NAME first. Python's bytecode is left out, as root leaves it out; every other entry goes in as it is, a link
    as a link. The archive's path, in FOLDER (root's home)."""
    top = os.path.basename(PKG)
    runner = f"{top}/{WRITER_RUNNER}"
    said = MARK.format(name) + "\n"
    body = pathlib.Path(PKG, WRITER_RUNNER).read_bytes()
    body += f"\nimport sys as _isolation_sys\n\n_isolation_sys.stderr.write({said!r})\n".encode()

    def kept(m: tarfile.TarInfo):
        return None if m.name == runner or "__pycache__" in m.name.split("/") else m

    path = os.path.join(folder, f"rk-isolation-{name}.tar")
    with tarfile.open(path, "w") as tar:
        tar.add(PKG, arcname=top, filter=kept)
        info = tar.gettarinfo(os.path.join(PKG, WRITER_RUNNER), arcname=runner)
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    return path


def code_tries(folders, files, links) -> str:
    """A shell script that tries, as the user it runs as, to change the code volume: to make a file in each of FOLDERS,
    rename it and open it up; to write each of FILES and open it up; to point each of LINKS elsewhere and remove it. It
    says NOTHING CHANGED as that user when every try failed, else what each try that worked did. A link counts as
    removed only when rm removed it: `rm -f` also says done for a name that is not there.
    """

    def words(names) -> str:
        return " ".join(shlex.quote(n) for n in names)

    return (
        "did=''\n"
        f"for d in {words(folders)}; do\n"
        f'  touch "$d/{TRY}" 2>/dev/null && did="$did made:$d/{TRY}"\n'
        '  mv "$d" "$d.moved" 2>/dev/null && did="$did renamed:$d"\n'
        '  chmod a+w "$d" 2>/dev/null && did="$did opened:$d"\n'
        "done\n"
        f"for f in {words(files)}; do\n"
        '  (printf x >>"$f") 2>/dev/null && did="$did wrote:$f"\n'
        '  chmod a+w "$f" 2>/dev/null && did="$did opened:$f"\n'
        "done\n"
        f"for l in {words(links)}; do\n"
        '  ln -sfn /tmp "$l" 2>/dev/null && did="$did pointed:$l"\n'
        '  rm "$l" </dev/null 2>/dev/null && did="$did removed:$l"\n'
        "done\n"
        'if [ -z "$did" ]; then echo "NOTHING CHANGED as $(id -un)"; else echo "CHANGED as $(id -un):$did"; fi\n'
    )


def tree_state(top: str) -> dict:
    """What is at the folder TOP and at each path under it (what_is_at), links not followed: to compare before and
    after."""
    state = {top: what_is_at(top)}
    for folder, dirs, files in os.walk(top):
        for name in dirs + files:
            state[os.path.join(folder, name)] = what_is_at(os.path.join(folder, name))
    return state


def code_is_roots(p: Proof) -> None:
    code, b, writer = L.CODE_DIR, BROKER, L.WRITER.name
    print(
        "\n--- the code is root's: root installs releases, no other user changes them, and the broker runs root's"
    )
    for name in ("one", "two"):
        archive = release_archive(name)
        p.check(
            "INSTALLED: release",
            f"root installs release {name} over the image, as rkmon apply has it run (entrypoint.sh code-install, as "
            "root)",
            "root",
            f"{PKG}/docker/entrypoint.sh code-install < {archive}",
            env={"RKMON_AS": "root"},
        )
        os.remove(archive)
    releases = os.path.join(code, code_volume.RELEASES)
    current, kept = (
        os.path.realpath(os.path.join(code, n))
        for n in (code_volume.CURRENT, code_volume.KEPT)
    )

    def marked(release: str, name: str) -> bool:
        try:
            return MARK.format(name) in pathlib.Path(release, WRITER_RUNNER).read_text()
        except OSError:
            return False

    p.expect(
        {os.path.dirname(current), os.path.dirname(kept)} == {releases}
        and marked(current, "two")
        and marked(kept, "one"),
        f"{code_volume.CURRENT} names release two ({current}), {code_volume.KEPT} release one ({kept})",
    )
    folders = (code, releases, current, os.path.join(current, "rkmon_lib"), kept)
    files = (
        os.path.join(current, "rk"),
        os.path.join(current, WRITER_RUNNER),
        os.path.join(current, code_volume.RELEASE_FILE),
        os.path.join(kept, "rk"),
        os.path.join(code, code_volume.LOCK),
    )
    links = tuple(
        os.path.join(code, n) for n in (code_volume.CURRENT, code_volume.KEPT)
    )
    tries = code_tries(folders, files, links)
    before = tree_state(code)
    for how, user, cmd, want, need in (
        ("the monitor", L.MONITOR.name, tries, L.MONITOR.name, ""),
        ("the writer", writer, tries, writer, ""),
        (
            "a PR runner, as the job scripts run PR code",
            L.MONITOR.name,
            SLOT7 + f"pc sh -c {shlex.quote(tries)}",
            L.PR_GROUP,
            RAN,
        ),
    ):
        p.check(
            f"NOTHING CHANGED as {want}",
            f"{how} makes, writes, renames, opens up, points elsewhere or removes nothing of the code volume: its "
            "folder, the releases' folder, both releases, their files, the lock, and the links "
            f"{code_volume.CURRENT} and {code_volume.KEPT}",
            user,
            cmd,
            need=need,
        )
    after = tree_state(code)
    p.expect(
        after == before,
        "nothing in the code volume changed (each entry's inode, size, permissions and times) "
        f"({', '.join(sorted(x for x in set(before) | set(after) if before.get(x) != after.get(x))) or 'none'})",
    )
    p.check(
        current,
        "the launcher runs CURRENT's release (rk --which-code, as the monitor)",
        L.MONITOR.name,
        "rk --which-code",
    )
    p.check(
        "REFUSED: in the container root alone changes the code",
        "rk code install, as the monitor, refuses and names the host's commands",
        L.MONITOR.name,
        "rk code install < /dev/null",
        need="next: ",
    )
    p.check(
        MARK.format("two"),
        "the broker runs CURRENT's release, though its caller names the release kept (RKMON_CODE, and an "
        "argument)",
        L.MONITOR.name,
        f"RKMON_CODE={kept} sudo -n -u {writer} {b} status {kept}",
    )
    planted = os.path.join(current, "rkmon_lib", "isolation-link")
    runner = os.path.join(current, WRITER_RUNNER)
    for what, change, undo, says in (
        (
            "a file another user may write (root makes one group-writable)",
            f"chmod g+w {runner}",
            f"chmod g-w {runner}",
            "may be written by another user",
        ),
        (
            "a file another user owns (root gives one to the monitor)",
            f"chown {L.MONITOR.name} {runner}",
            f"chown root {runner}",
            "is another user's",
        ),
        (
            "a link that leaves it (root plants one)",
            f"ln -s {L.WORK_DIR} {planted}",
            f"rm {planted}",
            "is a link that leaves the code",
        ),
    ):
        p.check("ok", f"root changes release two: {what}", "root", change)
        out = p.check(
            "is not root's and read-only all the way",
            f"the broker refuses release two with {what}",
            L.MONITOR.name,
            f"sudo -n -u {writer} {b} status",
            need=says,
        )
        p.expect(MARK.format("two") not in out, "and runs none of it", out)
        p.check("ok", "root puts it back", "root", undo)
    p.check(
        MARK.format("two"),
        "the broker runs release two again",
        L.MONITOR.name,
        f"sudo -n -u {writer} {b} status",
    )


def acls(path: str, perms) -> tuple:
    """The entry's access ACL and default ACL, each its entries or None, from their extended attributes (what getfacl
    reads)."""
    out = []
    for name in (perms.ACCESS, perms.DEFAULT):
        try:
            out.append(
                perms.acl_entries(os.getxattr(path, name, follow_symlinks=False))
            )
        except OSError as exc:
            if exc.errno != errno.ENODATA:
                raise
            out.append(None)
    return tuple(out)


def runner_bits(st: os.stat_result, acl, runners) -> int:
    """The bits (r 4, w 2, x 1) any of `runners` ((uid, gids) each) gets on an entry it does not own, by the kernel's
    check of the entry's mode and its access ACL (`acl`: its entries, None when it has none).
    """
    bit = {"r": 4, "w": 2, "x": 1}
    perm = {(tag, who): sum(bit[c] for c in text) for tag, who, text in acl or ()}
    mode = stat.S_IMODE(st.st_mode)
    mask = perm.get(("mask", None), 7)
    group_obj = perm.get(("group_obj", None), (mode >> 3) & 7)
    got = 0
    for uid, gids in runners:
        if ("user", uid) in perm:
            got |= perm[("user", uid)] & mask
            continue
        groups = [group_obj] if st.st_gid in gids else []
        groups += [perm[("group", g)] for g in gids if ("group", g) in perm]
        for g in groups:
            got |= g & mask
        if not groups:
            got |= mode & 7
    return got


def after_the_run(p: Proof) -> None:
    """The check after the run, by modes and ACLs, as root: `find -perm` reads no ACL. ORIGIN is left out: it stands
    in for GitHub, and the stage makes it, never the monitor."""
    sys.path.insert(0, f"{PKG}/docker")
    import rk_layout  # the start's layout step: its kinds' names and modes

    perms = paths.worker_module("perms")
    mon, writer = pwd.getpwnam(L.MONITOR.name), pwd.getpwnam(L.WRITER.name)
    pr_gid = grp.getgrnam(L.PR_GROUP).gr_gid
    runners = [
        (pwd.getpwnam(u.name).pw_uid, {pwd.getpwnam(u.name).pw_gid, pr_gid})
        for u in L.PR_RUNNERS
    ]
    shared = (("SHARED", perms.SHARED), ("SCRATCH", perms.scratch(pr_gid)))
    owners = (mon.pw_uid, writer.pw_uid)

    def kind(entries) -> str:
        return next((n for n, a in shared if perms.same_acl(entries, a)), "none")

    def name(look, ident) -> str:
        try:
            return look(ident)[0]
        except KeyError:
            return str(ident)

    def entry(tag: str, ident, text: str) -> str:
        """An ACL entry as getfacl writes it."""
        look = grp.getgrgid if tag == "group" else pwd.getpwuid
        return f"{tag}:{'' if ident is None else name(look, ident)}:{text or '-'}"

    print(
        "\n--- after the run: what the monitor and the writer made, by permissions and ACLs"
    )
    census, examples, writable, odd = collections.Counter(), {}, [], []
    tops = (L.WORK_DIR, L.RAMDISK, *L.SYSTEM_TMP, mon.pw_dir, writer.pw_dir)
    for top in tops + (L.SECRETS_DIR, L.WRITE_SECRETS_DIR):
        for folder, dirs, files in os.walk(top):
            dirs[:] = [d for d in dirs if os.path.join(folder, d) != ORIGIN]
            in_shared = kind(acls(folder, perms)[1]) != "none"
            for path in [folder] * (folder == top) + [
                os.path.join(folder, n) for n in dirs + files
            ]:
                st = os.lstat(path)
                if st.st_uid not in owners or stat.S_ISLNK(st.st_mode):
                    continue
                access, default = acls(path, perms)
                mode, isdir = stat.S_IMODE(st.st_mode), stat.S_ISDIR(st.st_mode)
                named = " ".join(
                    entry(*e) for e in access or () if e[0] in ("user", "group", "mask")
                )
                key = (
                    "folder" if isdir else "file",
                    name(pwd.getpwuid, st.st_uid),
                    name(grp.getgrgid, st.st_gid),
                    f"{mode:04o}",
                    named or "no ACL",
                    f"default {kind(default)}" if default else "",
                )
                census[key] += 1
                examples.setdefault(key, path)
                said = f"{path} {mode:04o} {name(grp.getgrgid, st.st_gid)} {named}"
                if runner_bits(st, access, runners) & 2:
                    writable.append(said)
                if (
                    path == top
                    or not in_shared
                    or not mode & 0o070
                    or perms.is_tree(st, pr_gid)
                ):
                    continue  # a top folder, in no shared folder, a private one, a tree
                if isdir:  # SCRATCH and CACHE: a runner passes through them
                    passes = os.path.basename(path) in (
                        rk_layout.SCRATCH,
                        rk_layout.CACHE,
                    )
                    modes = (
                        perms.SHARED_DIR_MODE,
                        rk_layout.PASS_MODE if passes else None,
                    )
                    good = mode in modes and kind(default) != "none"
                else:
                    good = mode == perms.SHARED_FILE_MODE
                if st.st_gid != mon.pw_gid or not good:
                    odd.append(said + f" default {kind(default)}")
    for key, n in sorted(census.items()):
        print(f"     | {n:6d} {' '.join(k for k in key if k)}  (as {examples[key]})")
    for line in writable[:40]:
        print(f"     | writable by a runner: {line}")
    p.expect(
        bool(census) and not writable,
        f"no folder or file the monitor or the writer made is writable by a PR runner, by its permissions and ACLs "
        f"({len(writable)} of {sum(census.values())} are)",
    )
    for line in odd[:40]:
        print(f"     | not shared as it should be: {line}")
    p.expect(
        bool(census) and not odd,
        "every file they share is 0660, group rkmon, every folder 2770 (2771 where a runner passes through), group "
        f"rkmon, with a shared folder's default ACL ({len(odd)} not)",
    )


def isolation_stage(home: pathlib.Path, port: int) -> int:
    if os.geteuid() != 0:
        print(
            "the isolation stage sets up as root (image.py runs it with RKMON_AS=root)"
        )
        return 2
    p = Proof()
    setup(p)
    root_step(p)
    write_login(p)
    pr_runners(p)
    scratch_folder(p)
    kept_slot(p)
    pr_code_callers(p)
    code_is_roots(p)
    after_the_run(p)
    p.run("root", f"kill $(cat {SESSION_PID}) 2>/dev/null")
    print("\n--- the broker's log")
    print(pathlib.Path(writer_run.LOG).read_text(errors="replace"))
    print(f"RESULT: {p.n - p.failed} PASS, {p.failed} FAIL of {p.n} checks")
    print(f"STAGE isolation: {'FAIL' if p.failed else 'PASS'}", flush=True)
    return 1 if p.failed else 0


STAGES = {"isolation": isolation_stage}

if __name__ == "__main__":
    sys.exit(isolation_stage(pathlib.Path("/"), 0))
