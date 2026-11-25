# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pull requests' code in the PR runners' slots: the one way the monitor's Python runs it, and the slot
commands of the job scripts and of `rk pr-run`.

The checks, the gates, the develop sanity runs, the bisects, the drift checks, the probes and the session's own runs
build and run code from pull requests. In the container that code runs as a PR runner (container_layout.PR_RUNNERS),
in the runner's slot: rkmon_lib/slots.py says which job holds which slot, the launcher (docker/rk_prcode.py) does the
work in it as the slot's user. A runner reads no secret and nothing of the monitor's but the trees (the clone, the
worktrees, the virtual environments, the probes, the code) and what the monitor hands its job (the slot's inputs
folder), and it writes only its slot's folders: `out` (the job's results), `build`, `tmp` and `home`. What the monitor
keeps of `out` it copies into its own folders through the result reader (worker/pr_results.py), as data. PR code gets
only the environment worker/pr_env.py lists. On a host (no launcher: RKMON_PR_RUNNER unset) PR code runs as the session
user, with that environment and pr_env.HOST's variables as they are, in a folder of the session user's that holds the
same folders.

- `Slot`: a slot taken when entered (in the container the launcher's begin makes its folders fresh) and freed when
  left (every process of its user stopped, its folder emptied). Named work (`name`, `rk pr-run --slot NAME`) gets its
  slot back while it holds it, and frees it only when it ends (`end`; `end_work` ends it with no command). `run` runs a
  command there, `give` hands the job a file of the monitor's to read, `read` reads one of the job's results and
  `results` copies them into a folder of the monitor's.
- `run(argv, ...)`: one command in a slot of its own, its output captured; `call(argv, ...)`: one command in a slot of
  its own, its output the caller's, the slot's results copied into a folder after.
- `fetch_for(argv, env)`: a check's `--fetch`, run as the monitor before the check's job takes a slot: the blobs of the
  clone the check reads fetched (rkmon_lib/prtree.py fetch_main), since git in a slot fetches nothing.
- `take`, `keep`, `free` (`rk pr-slot take|keep|free`): a slot for a job script's process (its folders as shell
  assignments: `assignments`), kept for its review (no job runs in it; the other runners read its build), freed.
  `kill_job` frees the slots a killed job's processes took, `stop_all` stops every runner's processes (rk stop,
  rk kill), `status` lists the slots (`rk pr-slot status`), and `doctor_checks` is rk doctor's row for them: each
  broken slot, with its why.
- `tree(path)`: an empty folder made the kind a tree PR code reads goes into (worker/perms.py). `with_trees(env,
  ...)`: an environment that names trees the job's git reads (TREES_VAR): in the container git takes as safe only
  those of the monitor's and the one CMD runs in, never a tree a runner made.

How a job script runs PR code: with SHELL's functions (`shell(package)`: the same, with that package's tools), in
every such script.
- `prslot [PR]` takes a slot for the script's process (`rk pr-slot take`; it waits while every slot is taken) and sets
  SLOT (its runner), SD (its folder), SO (its `out`), SB (its `build`) and SI (the folder of what the monitor hands
  the job). A subshell that calls it takes a slot of its own. `prfree` frees it, and the script's exit does too.
- `prc [-C DIR] CMD...` runs CMD in that slot, in DIR (absolute, or a folder of the slot's; its `home` when none). In
  the image RKMON_PR_RUNNER names the root-owned launcher docker/rk_prcode.py, which one sudo rule lets the monitor's
  user run as a PR runner and nothing else; `prc` hands it what worker/pr_env.py lists of the job's environment on its
  stdin, temporary assignments included, and the launcher stops every process CMD left. A script exports TREES_VAR
  with the trees its PR code's git reads.
- `prr NAME DEST` copies the job's result NAME (a file or a folder in its `out`) into the monitor's folder DEST through
  the result reader; nothing when the job left none. `prin FILE [NAME]` hands the job a copy of the monitor's FILE in
  SI.
- `prkeep PR HEAD BUILD RECORD` keeps the slot for PR's review of HEAD (`rk pr-slot keep`): BUILD, in its `build`, is
  the review's shared build, which the monitor's file RECORD names.
- `prtree DIR...` makes each DIR (missing, or empty) a folder for a tree PR code reads (`tree`).
- `prf CMD...` runs CMD's `--fetch`, as the monitor (`rk pr-fetch`, which runs `fetch_for`), before the same CMD runs
  in the slot (`prc`): a check that reads the clone fetches the blobs it reads first.
Without RKMON_PR_RUNNER (a host, the tests) the slot is a folder of the session user's in the temporary folder, with
the same folders, removed when it is freed, and its `build` is the build folder ($BD, when the script sets it): `prc`
runs CMD as the session user, there, with pr_env's environment, pr_env.HOST's variables as they are, and RKMON_OUT
and RKMON_BUILD naming the slot's `out` and `build`. The re-run commands the checks print are this code's `rk pr-run`
(RERUN_VAR, which SHELL sets) with the run's environment file, in the folder of the monitor's that the tool's results
are copied to (RESULTS_VAR, which a script sets for the tool's `prc`: `RKMON_RESULTS=DEST prc ...; prr NAME DEST`).

Every caller of the package's that runs PR code from Python goes through `Slot`, `run` or `call`: `rk probe-check`
(rkmon_lib/probes.py), `rk clash --build`, `rk lane-of` and the fix-line premise, `rk winsim`, `rk mutants`,
`rk import-probe`, `rk ctest-labels`, `rk werror` and `rk pr-run`; the worker tools through `rk pr-run`
(worker/pr_user.py).
"""

from __future__ import annotations

import datetime as dt
import os
import posixpath
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading

from . import container_layout, paths, slots

# The worker modules this module reads, each when first asked (paths.worker_module), never as it loads: docker/rkmon's
# deploy path imports this module (rkmon_lib/control.py loads it), from a package folder that holds no worker/.
# worker/perms.py gives the folders' modes, worker/pr_env.py the environment PR code gets, worker/pr_results.py the
# result reader, and worker/pr_user.py the worker tools' call and the values an install writes beside them.
# The names this module gives their values (`__getattr__` reads each when first asked): what an install writes beside
# the worker tools (deploy.install_worker_tools), by which an installed pr_user.py knows it is one, the values
# (VALUES_FILE: `installed_values`) and the install's record of each copy (MANIFEST); and the variable that names the
# trees the job's git reads, colon-separated (TREES_VAR: `with_trees`).
FROM_WORKER = {
    "VALUES_FILE": ("pr_user", "VALUES_FILE"),
    "MANIFEST": ("pr_user", "MANIFEST"),
    "TREES_VAR": ("pr_env", "TREES"),
}
RUNNER_VAR = slots.RUNNER_VAR
OUT_VAR = "RKMON_OUT"  # where PR code writes its results: the slot's `out` (the launcher sets it in the container)
FETCH_S = 900  # a check's `--fetch` at most (fetch_for)
# A host slot's folder of what the monitor hands its job, named as the container's (container_layout.SLOT_INPUTS).
INPUTS = container_layout.INPUTS

# The re-run commands a PR-code tool prints (tools/pytest_vs_base.py, tools/installed_tests.py) start with this
# variable's value when it is set, then name the run's environment file (`--env`) and folder (`-C`), and the command
# after `--`: the session runs them as PR code too, each in a slot of its own.
RERUN_VAR = "RKMON_RERUN_VIA"
# The monitor's folder a job copies a tool's results into (`prr NAME DEST`), set for that tool's `prc`: the re-run
# commands it prints name its environment file there.
RESULTS_VAR = "RKMON_RESULTS"


def __getattr__(name: str):
    """A name of FROM_WORKER: its worker module's value, read when first asked."""
    if name not in FROM_WORKER:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, value = FROM_WORKER[name]
    return getattr(paths.worker_module(module), value)


def rerun_via(package=paths.PACKAGE) -> str:
    """RERUN_VAR's value: `package`'s rk pr-run."""
    return shlex.quote(str(package / "rk")) + " pr-run"


RERUN_VIA = rerun_via()  # this code's rk
# The job scripts' functions (the module's docstring says what each does). @NAME@ marks a value `shell` puts in; the
# slot's variables are ASSIGNED's, which `rk pr-slot take` prints, and SLOT_PID names the process that took the slot.
_SHELL = r"""prheld() { [ -n "${SLOT_PID:-}" ] && return 0
  echo "$1: no slot of a PR runner is taken here (prslot)" >&2; return 1; }
prslot() { local pid=${BASHPID:-$$} a; [ "${SLOT_PID:-}" = "$pid" ] && return 0
  if [ -n "${RKMON_PR_RUNNER:-}" ]; then a=$(@RK@ pr-slot take --pid "$pid" ${1:+--pr "$1"}) || return 3; eval "$a"
  else SD=$(mktemp -d "${TMPDIR:-/tmp}/rkslot-XXXXXX") || return 3; (cd "$SD" && mkdir @FOLDERS@) || return 3
    SLOT=''; SO=$SD/@OUT@; SB=${BD:-$SD/@BUILD@}; SI=$SD/@INPUTS@; fi
  SLOT_PID=$pid; trap prfree EXIT; }
prfree() { [ -n "${SLOT_PID:-}" ] && [ "$SLOT_PID" = "${BASHPID:-$$}" ] || return 0
  if [ -n "${RKMON_PR_RUNNER:-}" ]; then @RK@ pr-slot free --pid "$SLOT_PID" >&2; else rm -rf "$SD"; fi; SLOT_PID=''; }
prc() { local d=''; [ "${1:-}" = -C ] && { d=$2; shift 2; }; prheld prc || return 3
  if [ -n "${RKMON_PR_RUNNER:-}" ]; then
    python3 -I @PR_ENV@ --print0 | sudo -n -u "$SLOT" "$RKMON_PR_RUNNER" run ${d:+-C "$d"} -- "$@"
  else case $d in '') d=$SD/@HOME@ ;; /*) ;; *) d=$SD/$d ;; esac
    (cd "$d" && export @OUT_VAR@="$SO" RKMON_BUILD="$SB" && exec python3 -I @PR_ENV@ "$@") < /dev/null; fi; }
prr() { prheld prr || return 3; [ -e "$SO/$1" ] || [ -L "$SO/$1" ] || return 0
  python3 -I -B @PR_RESULTS@ copy "$SD" "@OUT@/$1" "$2"; }
prin() { prheld prin || return 3; cp -- "$1" "$SI/${2:-${1##*/}}"; }
prkeep() { prheld prkeep || return 3; [ -n "${RKMON_PR_RUNNER:-}" ] || return 0
  @RK@ pr-slot keep --pid "$SLOT_PID" --pr "$1" --head "$2" --build "$3" --record "$4" && SLOT_PID=''; }
prtree() { @TREE@ "$@"; }
prf() { @RK@ pr-fetch -- "$@" < /dev/null; }
export @RERUN_VAR@=@RERUN_VIA@"""


def tree_command(*folders: str, package=paths.PACKAGE) -> list[str]:
    """The command that makes each of `folders` (missing, or empty) a folder for a tree PR code reads: the PR runners'
    group's, which none of them writes (worker/perms.py tree). SHELL's `prtree` runs it.
    """
    perms = str(package / "worker" / "perms.py")
    return [
        "python3",
        "-I",
        perms,
        "tree",
        "--group",
        container_layout.PR_GROUP,
        *folders,
    ]


def shell(package=paths.PACKAGE) -> str:
    """The job scripts' functions (SHELL), with `package`'s rk and worker tools."""
    q = shlex.quote
    worker = package / "worker"
    values = {
        "RK": q(str(package / "rk")),
        "PR_ENV": q(str(worker / "pr_env.py")),
        "PR_RESULTS": q(str(worker / "pr_results.py")),
        "TREE": shlex.join(tree_command(package=package)),
        "FOLDERS": " ".join(q(f) for f in (*container_layout.SLOT_FOLDERS, INPUTS)),
        "OUT": container_layout.SLOT_OUT,
        "BUILD": container_layout.SLOT_BUILD,
        "HOME": container_layout.SLOT_HOME,
        "INPUTS": INPUTS,
        "OUT_VAR": OUT_VAR,
        "RERUN_VAR": RERUN_VAR,
        "RERUN_VIA": q(rerun_via(package)),
    }
    text = _SHELL
    for name, value in values.items():
        text = text.replace(f"@{name}@", value)
    return text


SHELL = shell()


def installed_values(package=paths.PACKAGE) -> str:
    """VALUES_FILE's text, which an install writes beside the worker tools it copies from `package`
    (deploy.install_worker_tools): the PR runners' group, a slot's home folder and `package`'s rk, all of the layout an
    installed worker/pr_user.py reads."""
    return paths.worker_module("pr_user").values_text(
        container_layout.PR_GROUP, container_layout.SLOT_HOME, str(package / "rk")
    )


def tree(path) -> str:
    """Make `path` (missing, or an empty folder) the folder a tree PR code reads goes into: the runners' group, mode
    2750, no ACL (worker/perms.py); a plain folder on a host. "" when done, else why not.
    """
    return paths.worker_module("perms").tree_folder(
        str(path), container_layout.PR_GROUP
    )


def with_trees(env: dict, *trees) -> dict:
    """`env` with TREES_VAR naming `trees` after the ones it names already: the trees the job's git reads, which the
    launcher lets git take as safe when they are the monitor's (docker/rk_prcode.py)."""
    var = paths.worker_module("pr_env").TREES  # TREES_VAR
    have = [t for t in env.get(var, "").split(":") if t]
    have += [str(t) for t in trees if t and str(t) not in have]
    return dict(env, **{var: ":".join(have)})


def fetch_for(argv: list, env: dict | None = None, timeout: float = FETCH_S) -> int:
    """A check's `--fetch` (rkmon_lib/prtree.py fetch_main), run as the monitor before the check's job takes a slot,
    where git fetches nothing: `argv` with `--fetch` before its `--` (the targets after it), in `env` (default this
    process's) with lazy fetches on. What it prints goes to stderr. Its exit code; 124 when it ran out of time, 127
    when it did not start."""
    argv = [str(a) for a in argv]
    i = argv.index("--") if "--" in argv else len(argv)
    cmd = [*argv[:i], "--fetch", *argv[i:]]
    e = os.environ if env is None else env
    e = {k: v for k, v in e.items() if k != "GIT_NO_LAZY_FETCH"}
    try:
        p = subprocess.run(
            cmd,
            env=e,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(
            f"fetch: the blobs the check reads were not fetched in {timeout:.0f} s",
            file=sys.stderr,
        )
        return 124
    except OSError as exc:
        print(f"fetch: {cmd[0]}: {exc.strerror or exc}", file=sys.stderr)
        return 127
    said = (p.stdout + p.stderr).strip()
    if said:
        print(said, file=sys.stderr)
    elif p.returncode:
        print(f"fetch: exit {p.returncode}", file=sys.stderr)
    return p.returncode


def _text(data) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8", "replace")
    return data or ""


class NoSlot(Exception):
    """No slot could be taken: every one is broken, or the launcher refused the ones it tried (`lines` say why)."""

    def __init__(self, lines):
        self.lines = list(lines)
        super().__init__(self.lines[-1] if self.lines else "no slot could be taken")


def _on_term(signum, _frame):
    raise SystemExit(128 + signum)


def _hold_term():
    """While a slot is held, SIGTERM ends this process through Python's exit, which frees the slot on the way out, not
    at once: in the main thread, where nothing else handles SIGTERM. The handler to put back after (None: none).
    """
    if threading.current_thread() is not threading.main_thread():
        return None
    if signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
        return None
    signal.signal(signal.SIGTERM, _on_term)
    return signal.SIG_DFL


def _let_term(prev) -> None:
    if prev is not None and signal.getsignal(signal.SIGTERM) is _on_term:
        signal.signal(signal.SIGTERM, prev)


class Slot:
    """A slot for PR code, taken when entered and freed when left (also when SIGTERM ends this process).

    In the container a PR runner's slot (slots.py): its record names `job` (default: this process's job, the
    environment's RKMON_JOB) and `pr`, and this process, whose end lets the slot go; or the named work `name`
    (pr<N>-<word>) for the review of `head`, started by `owner` (default: the process that started this one's work,
    slots.Records.owner_of), which gets the slot back while it holds it and frees it only when it ends (`end`). `say`
    tells the wait for a slot. On a host a folder of the session user's in `tmp` (default: the temporary folder) with
    the same folders, removed when left; named work keeps its folder, rkslot-<name>, until it ends. NoSlot when none
    could be taken.
    """

    def __init__(
        self,
        *,
        job=None,
        pr="",
        name: str = "",
        head: str = "",
        owner: dict | None = None,
        end: bool = True,
        environ=None,
        run=None,
        records=None,
        say=None,
        tmp: str | None = None,
    ):
        from .jobs import JOB_ENV

        self.environ = os.environ if environ is None else environ
        self.launcher = slots.launcher(self.environ)
        self.job = self.environ.get(JOB_ENV, "") if job is None else str(job)
        self.pr, self.name, self.head, self.owner = str(pr or ""), name, head, owner
        self.end = end
        self._run = run or subprocess.run
        self.records, self.say, self._tmp = records, say, tmp
        self.pid = os.getpid()
        self.runner = self.dir = self.inputs = ""
        self.lines: list[str] = []
        self._term = None

    def __enter__(self):
        self._term = _hold_term()
        try:
            self._take()
        except BaseException:
            _let_term(self._term)
            raise
        return self

    def _take(self) -> None:
        if self.launcher:
            self.records = self.records or slots.Records()
            if self.name and self.owner is None:
                self.owner = self.records.owner_of(self.pid)
            who = (
                {"name": self.name, "head": self.head, "owner": self.owner}
                if self.name
                else {"pid": self.pid}
            )
            self.runner, self.lines = slots.take_begun(
                self.records,
                lambda r: slots.begin(r, self.launcher, self._run),
                job=self.job,
                pr=self.pr,
                say=self.say,
                **who,
            )
            if not self.runner:
                raise NoSlot(self.lines)
            self.dir, self.inputs = slots.folder(self.runner), slots.inputs(self.runner)
            return
        base = self._tmp or tempfile.gettempdir()
        os.makedirs(base, exist_ok=True)
        if self.name:
            d = _host_work(base, self.name)
            try:
                os.mkdir(d, 0o700)
            except FileExistsError:
                s = os.lstat(d)
                if not stat.S_ISDIR(s.st_mode) or s.st_uid != os.getuid():
                    raise NoSlot(
                        [
                            f"PR_SLOT {d} is not this user's folder: named work {self.name} not begun"
                        ]
                    ) from None
            self.dir = d
        else:
            self.dir = tempfile.mkdtemp(prefix="rkslot-", dir=base)
        for f in (*container_layout.SLOT_FOLDERS, INPUTS):
            os.makedirs(os.path.join(self.dir, f), exist_ok=True)
        self.inputs = os.path.join(self.dir, INPUTS)

    def __exit__(self, *_exc) -> None:
        try:
            if not (self.name and not self.end):
                self.lines += self.free()
        finally:
            _let_term(self._term)

    def free(self) -> list[str]:
        """Free the slot now (the lines); nothing when it is not this one's any more."""
        if not self.dir:
            return []
        if not self.launcher:
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir = ""
            return []
        mine = (
            (lambda rec: slots.taken_by(rec, name=self.name))
            if self.name
            else (lambda rec: slots.taken_by(rec, pid=self.pid))
        )
        out = slots.free(
            self.records,
            self.runner,
            lambda r: slots.end(r, self.launcher, self._run),
            holds=mine,
        )
        self.dir = ""
        return out

    def folder(self, name: str) -> str:
        return os.path.join(self.dir, name)

    @property
    def out(self) -> str:
        return self.folder(container_layout.SLOT_OUT)

    @property
    def home(self) -> str:
        return self.folder(container_layout.SLOT_HOME)

    def run(
        self,
        argv: list,
        cwd=None,
        env: dict | None = None,
        timeout: float | None = None,
        capture: bool = True,
    ) -> subprocess.CompletedProcess:
        """`argv` in the slot, with what worker/pr_env.py lists of `env` (default: this process's environment), in
        `cwd` (absolute, or a folder of the slot's; default: its home). Its output captured as text, unless `capture`
        is false (the caller's). Exit 124 on a timeout, 127 when it cannot start; in the container, every process the
        command left is stopped when it ends."""
        argv = [str(a) for a in argv]
        e = self.environ if env is None else env
        pr_env = paths.worker_module("pr_env")
        if self.launcher:
            p = slots.call(
                self.runner,
                ["run", *(["-C", str(cwd)] if cwd else []), "--", *argv],
                self.launcher,
                data=pr_env.data(e),
                run=self._run,
                timeout=timeout,
                capture=capture,
            )
            if not capture:
                return subprocess.CompletedProcess(argv, p.returncode)
            return subprocess.CompletedProcess(
                argv, p.returncode, _text(p.stdout), _text(p.stderr)
            )
        kw = {
            "cwd": os.path.join(self.dir, str(cwd)) if cwd else self.home,
            "env": dict(pr_env.host(e), **{OUT_VAR: self.out}),
            "timeout": timeout,
            "stdin": subprocess.DEVNULL,
        }
        if capture:
            kw.update(capture_output=True, text=True, errors="replace")
        try:
            return self._run(argv, **kw)
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                argv, 124, "", f"timed out after {exc.timeout} s"
            )
        except OSError as exc:
            return subprocess.CompletedProcess(argv, 127, "", str(exc))

    def read_json(self, rel: str):
        """The job's result file `rel` under its `out`, parsed as JSON by the result reader (worker/pr_results.py
        read_json; Refused, an OSError, when it refuses it)."""
        return paths.worker_module("pr_results").read_json(
            self.dir, posixpath.join(container_layout.SLOT_OUT, rel)
        )

    def give(self, src, name: str = "") -> str:
        """A copy of the monitor's file `src` in the job's inputs folder, which the job reads and no runner writes: its
        path."""
        dest = os.path.join(self.inputs, name or os.path.basename(str(src)))
        shutil.copyfile(src, dest)
        return dest

    def results(self, rel: str, dest) -> list[str]:
        """The job's results (`rel` under its `out`; "" for all of them) copied into the monitor's folder `dest` by the
        result reader's rules: the lines that name what was refused."""
        try:
            return paths.worker_module("pr_results").copy(
                self.dir,
                (
                    posixpath.join(container_layout.SLOT_OUT, rel)
                    if rel
                    else container_layout.SLOT_OUT
                ),
                str(dest),
            )
        except OSError as exc:
            return [str(exc)]


def run(
    argv: list,
    cwd=None,
    env: dict | None = None,
    timeout: float | None = 600,
    environ=None,
    run=None,
    records=None,
) -> subprocess.CompletedProcess:
    """`argv` in a slot of its own (Slot.run): its output captured. Exit 3 when no slot could be taken (the lines on
    its stderr)."""
    try:
        with Slot(environ=environ, run=run, records=records) as s:
            return s.run(argv, cwd=cwd, env=env, timeout=timeout)
    except NoSlot as exc:
        return subprocess.CompletedProcess(
            [str(a) for a in argv], slots.REFUSED, "", "\n".join(exc.lines)
        )


def call(
    argv,
    env: dict | None = None,
    cwd=None,
    out=None,
    *,
    environ=None,
    run=None,
    records=None,
    say=None,
    **slot,
) -> int:
    """`argv` (a list, or a function of the slot that returns one: a command that names the slot's folders) in a slot
    of its own (Slot.run; `slot`: Slot's own arguments, such as named work's), its output the caller's; then, with
    `out`, the job's results copied there. Its exit code; 1 when it ended well but a result was refused, 3 when no slot
    could be taken (the lines on stderr)."""
    try:
        with Slot(environ=environ, run=run, records=records, say=say, **slot) as s:
            cmd = argv(s) if callable(argv) else argv
            code = s.run(cmd, cwd=cwd, env=env, capture=False).returncode
            if out:
                refused = s.results("", out)
                for line in refused:
                    print(f"pr_results: refused {line}", file=sys.stderr)
                if refused and code == 0:
                    code = 1
            return code
    except NoSlot as exc:
        for line in exc.lines:
            print(line, file=sys.stderr)
        return slots.REFUSED


def read_env(path: str) -> dict:
    """The environment in the file `path` (`rk pr-run --env`; /dev/stdin: this process's stdin), NUL-separated
    NAME=VALUE, read as data: at most worker/pr_env.py's ENV_MAX bytes. OSError or ValueError when it cannot be read.
    """
    pr_env = paths.worker_module("pr_env")
    with open(path, "rb") as f:
        return pr_env.read(f, pr_env.ENV_MAX)


def _host_work(base: str, name: str) -> str:
    """Named work's folder on a host."""
    return os.path.join(base, f"rkslot-{name}")


def end_work(
    name: str, environ=None, run=None, records=None, tmp: str | None = None
) -> list[str]:
    """End named work `name` (`rk pr-run --slot NAME --end` with no command): its slot freed (every process stopped,
    its folder emptied), on a host its folder removed. The lines; nothing when it holds no slot.
    """
    launcher = slots.launcher(environ)
    if not launcher:
        d = _host_work(tmp or tempfile.gettempdir(), name)
        if (
            os.path.isdir(d)
            and not os.path.islink(d)
            and os.lstat(d).st_uid == os.getuid()
        ):
            shutil.rmtree(d, ignore_errors=True)
        return []
    recs, run = records or slots.Records(), run or subprocess.run
    out = []
    for r, rec in recs.all().items():
        if slots.taken_by(rec, name=name):
            out += slots.free(
                recs,
                r,
                lambda x: slots.end(x, launcher, run),
                holds=lambda cur: slots.taken_by(cur, name=name),
            )
    return out


# --- the job scripts' slots (rk pr-slot) ----------------------------------------------------------------------------


def _calls(environ, run):
    """The launcher's begin, end and keep for a runner, as functions of it; None when there is no launcher here."""
    launcher, run = slots.launcher(environ), run or subprocess.run
    if not launcher:
        return None
    return (
        lambda r: slots.begin(r, launcher, run),
        lambda r: slots.end(r, launcher, run),
        lambda r: slots.keep_call(r, launcher, run),
    )


NO_SLOTS = "PR_SLOT no slots here (no launcher): PR code runs as the session user"
# `rk pr-slot take`'s assignments, for a job script's `eval`: the runner, the slot's folder, its out and build, and the
# folder of what the monitor hands the job.
ASSIGNED = ("SLOT", "SD", "SO", "SB", "SI")


def assignments(runner: str) -> list[str]:
    """`rk pr-slot take`'s stdout for the slot of `runner`: ASSIGNED as shell assignments."""
    values = (
        runner,
        slots.folder(runner),
        slots.folder(runner, container_layout.SLOT_OUT),
        slots.folder(runner, container_layout.SLOT_BUILD),
        slots.inputs(runner),
    )
    return [f"{k}={shlex.quote(v)}" for k, v in zip(ASSIGNED, values)]


def taken_by(records, pid: int) -> str:
    """The runner whose slot process `pid` took; "" when none."""
    return next(
        (r for r, rec in records.all().items() if slots.taken_by(rec, pid=int(pid))),
        "",
    )


def take(
    pid: int,
    job: str = "",
    pr="",
    environ=None,
    run=None,
    records=None,
    say=None,
) -> tuple[str, list[str]]:
    """A slot for the job script's process `pid` (its record names it: the slot is free again once that process is
    gone), begun: (its runner, the lines); "" when none could be taken."""
    c = _calls(environ, run)
    if c is None:
        return "", [NO_SLOTS]
    return slots.take_begun(
        records or slots.Records(), c[0], job=job, pr=pr, pid=int(pid), say=say
    )


def keep(
    pid: int,
    pr,
    head: str,
    build: str,
    record: str,
    environ=None,
    run=None,
    records=None,
) -> tuple[bool, list[str]]:
    """Keep the slot process `pid` took for PR `pr`'s review of `head` (slots.keep): its `build` is the review's
    shared build, which `record` names. (kept, the lines)."""
    c = _calls(environ, run)
    if c is None:
        return False, [NO_SLOTS]
    recs = records or slots.Records()
    runner = taken_by(recs, pid)
    if not runner:
        return False, [f"PR_SLOT no slot is taken by process {pid}: none kept"]
    return slots.keep(
        recs,
        runner,
        c[2],
        c[1],
        pid=int(pid),
        pr=pr,
        head=head,
        build=build,
        record=record,
    )


def free(pid: int, environ=None, run=None, records=None) -> list[str]:
    """Free the slot process `pid` took (slots.free): the lines; nothing when it took none."""
    c = _calls(environ, run)
    if c is None:
        return []
    recs = records or slots.Records()
    runner = taken_by(recs, pid)
    if not runner:
        return []
    return slots.free(
        recs, runner, c[1], holds=lambda rec: slots.taken_by(rec, pid=int(pid))
    )


def free_ended(live, environ=None, run=None, records=None) -> list[str]:
    """Free the slots of the reviews that ended (`live(pr, head)`: PR's review of head is in flight) and of the named
    work whose owner is gone (slots.free_ended): the lines. Nothing without the launcher.
    """
    c = _calls(environ, run)
    if c is None:
        return []
    return slots.free_ended(records or slots.Records(), live, c[1])


def kill_job(job_dir: str, environ=None, run=None, records=None) -> list[str]:
    """Free the slots the job in `job_dir` took (its processes are stopped, its slots emptied): the lines. A slot kept
    for its review stays kept. Nothing without the launcher."""
    c = _calls(environ, run)
    if c is None:
        return []
    recs = records or slots.Records()
    out = []
    for r, rec in recs.all().items():
        if rec.get("state") == slots.TAKEN and rec.get("job") == str(job_dir):
            out += slots.free(recs, r, c[1], holds=lambda cur, rec=rec: cur == rec)
    return out


def stop_all(now: bool = False, environ=None, run=None) -> list[str]:
    """Stop every process of every PR runner (rk stop; rk kill: `now`, SIGKILL at once), the runners at once: the
    lines of those that had any. The slots' records stay. Nothing without the launcher.
    """
    launcher, run = slots.launcher(environ), run or subprocess.run
    if not launcher:
        return []
    args = ["kill", *(["--now"] if now else [])]
    runners = [u.name for u in container_layout.PR_RUNNERS]

    def one(r: str) -> list[str]:
        p = slots.call(r, args, launcher, run=run, timeout=120)
        return [
            ln for ln in slots.lines_of(p, r, "kill") if "stopped 0 process" not in ln
        ]

    # Imported here: a tool run as a script from rkmon_lib/ would take its queue.py for the pool's queue.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(runners)) as ex:
        return [ln for lines in ex.map(one, runners) for ln in lines]


def _when(t) -> str:
    try:
        return dt.datetime.fromtimestamp(float(t), dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


def describe(records, runner: str, rec: dict | None) -> str:
    """One slot's line for `rk pr-slot status`."""
    if rec is None or not records.held(rec):
        return f"{runner}: free"
    since = _when(rec.get("since"))
    state = rec.get("state")
    pr = f" #{rec['pr']}" if rec.get("pr") else ""
    if state == slots.KEPT:
        return (
            f"{runner}: kept for{pr} at {str(rec.get('head') or '')[:10]} (its build {rec.get('build')}, read by the "
            f"other runners) since {since}"
        )
    if state == slots.BROKEN:
        return f"{runner}: broken until the container's next start ({rec.get('why')}) since {since}"
    if state == slots.FREEING:
        return f"{runner}: being freed by process {rec.get('pid')} since {since}"
    if rec.get("name"):
        head = str(rec.get("head") or "")[:10]
        return (
            f"{runner}: named work {rec['name']}"
            + (f" for the review at {head}" if head else "")
            + f" since {since}"
        )
    job = f" (job {rec['job']})" if rec.get("job") else ""
    return f"{runner}: taken by process {rec.get('pid')}{pr}{job} since {since}"


def status(environ=None, records=None) -> list[str]:
    """`rk pr-slot status`: where PR code runs here, and each slot: free, taken (by which process or named work), kept
    (for which review) or broken."""
    launcher = slots.launcher(environ)
    if not launcher:
        return [
            "PR code runs as the session user (no PR runners here: on a host, or without the image's launcher)"
        ]
    recs = records or slots.Records()
    have = recs.all()
    lines = [
        f"PR code runs in the PR runners' slots ({len(recs.runners)}; the launcher {launcher}):"
    ]
    return lines + ["  " + describe(recs, r, have.get(r)) for r in recs.runners]


SLOTS_ROW = "PR runners' slots"  # rk doctor's row


def doctor_checks(environ=None, records=None) -> list:
    """rk doctor's row (a checks row) for the PR runners' slots where PR code runs in them: none broken. A broken slot
    takes no job until the container's next start (slots.py): the row names each, with its why (`describe`), a WARN
    while other slots are left and a FAIL when none is, its fix that start. Nothing without the launcher.
    """
    from . import doctor
    from .checks import Check, Command, Steps

    if not slots.launcher(environ):
        return []
    recs = records or slots.Records()
    have = recs.all()
    bad = [
        r
        for r in recs.runners
        if have.get(r, {}).get("state") == slots.BROKEN and recs.held(have[r])
    ]
    n = len(recs.runners)
    if not bad:
        return [Check(SLOTS_ROW, "PASS", f"{n} slots, none broken")]
    w = doctor.wrapper_name(os.environ if environ is None else environ)
    return [
        Check(
            SLOTS_ROW,
            "FAIL" if len(bad) == n else "WARN",
            doctor.blocking(
                f"{len(bad)} of {n} broken: "
                + "; ".join(describe(recs, r, have[r]) for r in bad),
                "every job that runs PR code" if len(bad) == n else "",
            ),
            Steps(
                (
                    Command((w, "stop")),
                    ", then",
                    Command((w, "start")),
                    ": the container's next start makes them again",
                )
            ),
        )
    ]
