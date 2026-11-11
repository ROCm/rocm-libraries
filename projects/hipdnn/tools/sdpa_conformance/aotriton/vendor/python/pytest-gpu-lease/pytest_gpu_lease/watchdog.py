# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Detect a wedged pytest-xdist worker from outside its process, and kill it.

Run beside pytest, not inside it: ``python -m pytest_gpu_lease.watchdog --lockfile
... --workers N``. It polls the lease lock file that ``plugin.py`` already
maintains -- one 4096-byte page per GPU, write-locked by whichever worker owns
that GPU for the run -- and reads the 8-byte ``time.monotonic_ns()`` stamp
written there: an initial one set by ``gpu_id`` the moment it takes the lease,
refreshed before every test and zeroed after by a ``pytest_runtest_protocol``
hookwrapper. A zero means the worker is between tests and is skipped.

**This module owns the timeout.** The worker reports only when it was last
seen alive; ``--threshold`` here is the sole definition of how stale that may
get, so there is no second copy of the policy to fall out of sync with, and
changing the timeout means changing one command line rather than a command
line and an environment variable that have to agree.

Why the lock file is the right substrate, in one line: ``fcntl(F_GETLK)`` fills
``l_pid`` with the pid holding a byte range, and the kernel releases a record
lock automatically when its owner dies. So the lock answers "does this still
need killing?" on its own: a worker that died between one poll and the next is
simply unlocked by the time we look, and gets nothing sent to it.

It does not answer "who is this?", and pids are not stable identities -- once a
process is reaped its number can be reissued. Every signal therefore goes
through a pidfd (see ``_pidfd_open``), which names one specific process for as
long as the fd is open. The two questions stay separate: the lock decides
whether to signal, the pidfd decides who receives it.

Stopping it is a plain ``kill``: SIGTERM (also SIGINT, SIGHUP) unwinds
cleanly and removes the lock file on the way out. SIGKILL is the one case
that cannot, and leaves the file behind. It never stops on its own -- see
``watch``.

Escalation is SIGTERM, a grace period, then SIGKILL if the page is still locked.
SIGTERM is expected to end the worker: ``plugin.py`` registers it with
``faulthandler`` using ``chain=True``, so it dumps a stack -- C-level and
GIL-free, naming whatever frame the wedge is in -- and then terminates as it
normally would. SIGKILL is the last resort, for a process SIGTERM could not
end, such as one stuck in an uninterruptible driver call.

Shares ``PAGE_SIZE`` and ``STRUCT_FLOCK`` with ``plugin.py`` by importing them
rather than restating them, since a copy that drifts from the writer's layout
would silently misread every stamp.
"""

import argparse
import fcntl
import glob
import os
import signal
import struct
import sys
import time
from typing import NamedTuple

from .plugin import (HEARTBEAT_FMT, HEARTBEAT_SIZE, OWNER_FMT, OWNER_OFFSET,
                     PAGE_SIZE, STRUCT_FLOCK, dump_path)

# Every signal goes through a pidfd, so this is a hard requirement rather than
# an enhancement -- see _pidfd_open. Needs Linux 5.3 and Python 3.9.
#
# That is a statement about this harness, not about ROCm: ROCm still supports
# RHEL 8.10, whose kernel predates pidfd_open. This is Level-3 CI tooling,
# deployed on Ubuntu 22.04 or later (5.15, Python 3.10), where the requirement
# costs nothing. A suite that has to run on something older should not be
# using this module.
def _pidfd_supported() -> bool:
    """Whether pidfd actually works here, by using it rather than asking.

    `hasattr(os, 'pidfd_open')` is not the question: Python 3.9+ on a 4.18
    kernel has the function and fails at the syscall, which would surface as a
    crash mid-pass instead of a refusal at import. run-test.sh probes the same
    way before deciding whether to start a watchdog at all.
    """
    try:
        os.close(os.pidfd_open(os.getpid()))
        return True
    except (AttributeError, OSError):
        return False


assert _pidfd_supported(), (
    'pytest_gpu_lease.watchdog requires working pidfd support: Linux 5.3+ and '
    'Python 3.9+')

# The timeout, and the only one in the tree: how long a worker's page may go
# without a fresh stamp before it is treated as wedged. Because the worker
# rewrites its stamp before every test, this is exactly one thing -- the
# longest a single test may legitimately take. Not the suite length, not the
# worker count.
#
# A real pass ran 265,282 tests in 227,112 worker-seconds, 0.86s/test mean, so
# 600s is about the tail, not the average -- and the tail is plausibly the
# torch reference materialising an 8192x8192 score matrix, not the kernel
# under test, so sizing this off kernel cost alone would guess far too low.
# The cost of being wrong is asymmetric: too high and a wedge idles a worker
# for the excess (a handful of tests a pass); too low and a slow-but-passing
# test is recorded as a crash, a restart is burned, and the scheduler goes
# through the worker-teardown path again. Refine it with
# `pytest --durations=50 --durations-min=10` on an idle GPU.
_DEFAULT_THRESHOLD_S = 600

# Default cadence: frequent enough that a 600s threshold is caught within a
# small fraction of itself, infrequent enough that polling 4-8 pages is noise
# next to a 15-22h pass.
_DEFAULT_POLL_INTERVAL_S = 5.0

# Default grace between SIGTERM and SIGKILL: long enough for faulthandler to
# flush a stack dump to stderr (a syscall, not instant under load) and for a
# process that was already exiting on its own to finish doing so, short enough
# that a genuinely wedged worker is not left idle for long after being caught.
_DEFAULT_GRACE_S = 30.0



def _getlk(fd: int, page: int) -> tuple[bool, int]:
    """Whether `page` is currently write-locked, and by which pid.

    The pid is meaningless when the first element is False -- F_GETLK leaves
    only `l_type` defined in that case (see fcntl(2)) -- and callers must not
    use it.
    """
    probe = struct.pack(STRUCT_FLOCK, fcntl.F_WRLCK, os.SEEK_SET,
                        PAGE_SIZE * page, PAGE_SIZE, 0)
    result = fcntl.fcntl(fd, fcntl.F_GETLK, probe)
    lock_type, _, _, _, pid = struct.unpack(STRUCT_FLOCK, result)
    return lock_type != fcntl.F_UNLCK, pid


def _read_heartbeat(fd: int, page: int) -> tuple[int, int]:
    """The stamp at `page`'s base and the pid of the worker that wrote it.

    `(0, 0)` when the record is not fully there. The lock file is never
    pre-sized (see plugin.py's `_gpu_lease_lockfile`), so a page whose worker
    holds the lock but has not written yet is genuinely short of these bytes
    rather than zero-filled, and `os.pread` returns what exists. Treated the
    same as an explicit zero: nothing has expired.

    No torn-read handling, for the same reason the writer needs none -- see
    the layout note in plugin.py for why stamp-then-pid ordering makes a
    half-written pair safe to read.
    """
    raw = os.pread(fd, HEARTBEAT_SIZE, PAGE_SIZE * page)
    if len(raw) < HEARTBEAT_SIZE:
        return 0, 0
    stamp, = struct.unpack_from(HEARTBEAT_FMT, raw, 0)
    owner, = struct.unpack_from(OWNER_FMT, raw, OWNER_OFFSET)
    return stamp, owner


class _Staged(NamedTuple):
    """A page whose holder has been SIGTERMed and may still need SIGKILL."""

    pid: int
    sent_ns: int
    pidfd: int


def _pidfd_open(pid: int) -> int | None:
    """A handle to *this* process, or None if it is already gone.

    A pid is not a stable identity: the moment its process is reaped the
    number can be handed to somebody else, so every `os.kill(pid, ...)` is a
    bet that nothing has changed since whatever check justified it. A pidfd
    refers to one specific process for as long as the fd is open -- signalling
    through it after that process dies raises ProcessLookupError rather than
    reaching whoever inherited the number.

    Required, not best-effort: falling back to `os.kill` would silently give
    up that guarantee on the one path where the payload is SIGKILL. The module
    refuses to load without it; the assert at the top of this file records
    which systems that rules out and why they are out of scope here.

    None also for an `l_pid` that is not a local pid at all -- F_GETLK reports
    -1 for an open-file-description lock, and a lock held over NFS can report
    a pid meaningless here. `os.pidfd_open` would raise EINVAL, and the
    check has to happen before it because this is the first place a raw
    `l_pid` is used for anything.
    """
    if pid <= 0:
        print(f'pytest_gpu_lease.watchdog: ignoring a page held by pid {pid}; '
              f'l_pid is not a local pid (OFD lock, or a lock held over NFS)',
              file=sys.stderr, flush=True)
        return None
    try:
        return os.pidfd_open(pid)
    except ProcessLookupError:
        return None
    except OSError as exc:
        # Anything else -- a transient EMFILE, say. Skipping this page for one
        # poll costs a few seconds; letting it out of `_poll_once` kills the
        # loop, and the pass then runs the rest of its 22 hours unprotected
        # with one line in a log to show for it.
        print(f'pytest_gpu_lease.watchdog: no pidfd for {pid} ({exc}); retrying next poll',
              file=sys.stderr, flush=True)
        return None


def _discard(pending: dict[int, _Staged], page: int) -> None:
    """Drop a staged escalation, closing its pidfd."""
    staged = pending.pop(page, None)
    if staged is not None:
        os.close(staged.pidfd)


def _send(pid: int, sig: signal.Signals, reason: str, pidfd: int) -> None:
    """Signal `pid`, tolerating a process that is already gone.

    The F_GETLK check immediately before every call site is the liveness proof
    (see module docstring); the only race left is the process exiting in the
    interval between that check and this call, which is exactly what
    ProcessLookupError reports -- not a bug to guard against, just the same
    race resolving itself one step later.

    `pid` is only for the log line: the signal goes to `pidfd`, which
    `_pidfd_open` has already vetted.
    """
    try:
        signal.pidfd_send_signal(pidfd, sig)
        print(f'pytest_gpu_lease.watchdog: sent {sig.name} to pid {pid} ({reason})',
              file=sys.stderr, flush=True)
    except ProcessLookupError:
        print(f'pytest_gpu_lease.watchdog: pid {pid} already gone, no {sig.name} needed ({reason})',
              file=sys.stderr, flush=True)


def _emit_dump(lockfile: str, page: int, pid: int) -> None:
    """Relay a wedged worker's stack dump into our stderr, then delete it.

    The worker writes it to a file of its own rather than the shared stream
    (see `plugin.dump_path`); this process is the serialisation point. Deleted
    as soon as it is read: the owner is about to be SIGKILLed and will never
    reach the teardown that would otherwise remove it.
    """
    path = dump_path(lockfile, pid)
    try:
        with open(path) as f:
            dump = f.read().rstrip()
    except OSError:
        dump = ''
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    if dump:
        print(f'--- pytest_gpu_lease.watchdog: stack of pid {pid} on GPU {page} ---\n'
              f'{dump}\n--- end of stack ---', file=sys.stderr, flush=True)


def _sweep_dumps(lockfile: str, keep: set[int]) -> None:
    """Delete dump files whose worker is gone.

    A worker killed by a GPU fault reaches no teardown, so it never removes
    its own file, and the watchdog will never see its pid again -- F_GETLK
    reports live lock holders and nothing else. Sweeping every poll rather
    than at shutdown keeps /dev/shm, which is small in a container, from
    collecting one file per dead worker across a pass that restarts them
    dozens of times.
    """
    prefix, suffix = f'{lockfile}.', '.dump'
    for path in glob.glob(glob.escape(lockfile) + '.*' + suffix):
        try:
            pid = int(path[len(prefix):-len(suffix)])
        except ValueError:
            continue  # not one of ours
        if pid in keep:
            # Staged for escalation. Its owner dying is precisely when the
            # dump becomes worth reading, so it must outlive the process.
            continue
        try:
            os.kill(pid, 0)
            continue  # still alive: its dump is still wanted
        except PermissionError:
            continue  # alive, just not ours to signal
        except ProcessLookupError:
            pass
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _poll_once(fd: int, lockfile: str, workers: int, threshold_ns: int, grace_ns: int,
              pending: dict[int, _Staged]) -> bool:
    """One sweep of every page. Sends at most one signal -- see module docstring
    on staggering kills -- and returns whether any page is currently locked
    (unused by `watch`, which never stops; the tests assert on it).

    `pending` maps a page already SIGTERMed to the `_Staged` record for that
    kill, and is mutated in place across calls so escalation survives between
    polls without any state living outside this loop.
    """
    now = time.monotonic_ns()
    any_locked = False
    _sweep_dumps(lockfile, {staged.pid for staged in pending.values()})

    # Escalations in flight take priority over freshly-discovered ones: a kill
    # already staged should not be starved, poll after poll, by a steady trickle
    # of newly-expired pages elsewhere in the pool.
    for page in list(pending):
        staged = pending[page]
        locked, pid = _getlk(fd, page)
        # "Still locked" is not enough to escalate: the page has to still be
        # held by the *same* process. A SIGTERMed worker that dies anyway (the
        # HIP runtime aborting on a fault is the common way) is replaced by
        # xdist, and the replacement takes the lowest free page -- very often
        # the one just vacated -- well inside a 30s grace period. Escalating on
        # the page alone would SIGKILL that healthy replacement, then drop the
        # entry and be free to do it again. Dropping the entry here instead
        # hands the page back to the heartbeat scan below, where a replacement
        # is judged on its own stamp like any other worker.
        if not locked or pid != staged.pid:
            # It died inside the grace period -- normally because the SIGTERM
            # did its job. Relay its dump before the entry goes:
            # producing that was the entire point of the SIGTERM, and letting
            # this path discard silently threw it away in the common case.
            _emit_dump(lockfile, page, staged.pid)
            print(f'pytest_gpu_lease.watchdog: pid {staged.pid} exited during its '
                  f'grace period; no SIGKILL needed', file=sys.stderr, flush=True)
            _discard(pending, page)
            continue
        any_locked = True
        if now - staged.sent_ns >= grace_ns:
            # Reaching here means SIGTERM did not end it, which normally it
            # does. No reprieve for a page that came back to life meanwhile: a
            # worker past the threshold is a dead worker, and the test that ran
            # long enough to be signalled has already failed. Sparing it would
            # make the pass depend on how close to the threshold it landed.
            #
            # Dump first: producing it was the whole point of the SIGTERM, and
            # after the SIGKILL nobody is left to ask.
            _emit_dump(lockfile, page, staged.pid)
            # Through the pidfd, so this cannot land on anyone else. The lock
            # check above answers the policy question -- is it still wedged and
            # still leasing -- but not the identity one: between that F_GETLK
            # and this call the process may exit and its pid be reissued. That
            # window is small and the payload is SIGKILL, which is precisely
            # the combination not to leave to chance.
            _send(staged.pid, signal.SIGKILL,
                  'grace period expired, still holding its page', staged.pidfd)
            _discard(pending, page)
            return any_locked  # one signal per poll

    for page in range(workers):
        if page in pending:
            continue
        locked, pid = _getlk(fd, page)
        if not locked:
            continue
        any_locked = True
        last_activity, owner = _read_heartbeat(fd, page)
        if last_activity == 0:
            continue  # between tests -- see plugin.py's gpu_id and pytest_runtest_protocol
        if owner != pid:
            # The stamp belongs to a previous holder: this worker has the lock
            # but has not written yet. Judging it on someone else's stamp is
            # how a replacement gets killed for inheriting the expired one left
            # by the worker this watchdog had just SIGKILLed off that page.
            #
            # Not proof against pid rollover: were the dead worker's pid handed
            # straight back to its own replacement, the stale stamp would look
            # owned and the new worker would be signalled. That needs the pid
            # counter to wrap in the millisecond between the two, and the cost
            # is one spurious kill on a run that is already restarting workers.
            # Not fixed; a generation counter would not pay for itself.
            continue
        stale_ns = now - last_activity
        if stale_ns > threshold_ns:
            # Take the handle first, then re-read the lock. If the pid was
            # reissued between the F_GETLK above and the pidfd_open, the
            # original holder is dead, its lock is released, and this second
            # read sees either an unlocked page or a different holder -- so the
            # handle we are about to signal through is the process we measured.
            pidfd = _pidfd_open(pid)
            if pidfd is None:
                continue  # exited between the F_GETLK above and the open
            locked_now, pid_now = _getlk(fd, page)
            if not locked_now or pid_now != pid:
                os.close(pidfd)
                continue
            _send(pid, signal.SIGTERM,
                  f'no heartbeat for {stale_ns / 1e9:.1f}s, past the '
                  f'{threshold_ns / 1e9:.0f}s threshold', pidfd)
            pending[page] = _Staged(pid, now, pidfd)
            return any_locked  # one signal per poll

    return any_locked


def watch(lockfile: str, workers: int, threshold_s: float, grace_s: float,
         poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S) -> None:
    """Poll `lockfile` forever. Returns only when signalled.

    Deliberately has no idea whether a pass is running, finishing, or finished:
    it watches a file, and whoever started it decides when that stops being
    useful (`run-test.sh` does, in a trap). Every rule for inferring "the run
    is over" from the file itself is a guess, and a wrong guess is expensive
    and silent -- the watchdog exits, `main` unlinks the lock file, and the
    rest of a 22h pass runs unprotected with no line in any log saying so.

    An earlier revision guessed from consecutive polls with no page locked.
    That is not the same question: a worker this watchdog just killed is
    replaced, and the replacement holds no lease until it has re-imported torch
    and re-collected -- minutes, with the whole pool in that state at once
    after a multi-worker wedge. The first threshold was 60s and fired during
    exactly that; raising it only moved the guess.

    The cost of never stopping is an orphan if the starting shell is SIGKILLed
    and its trap never runs. That orphan is inert: `fd` is opened once here, so
    once the file is unlinked it polls a deleted inode, and a later run gets a
    new path and a new inode it can never see. No locks observed means no
    pidfds opened and nothing signalled. It shows up in `ps`, and its lock file
    is left behind in /dev/shm.
    """
    threshold_ns = int(threshold_s * 1_000_000_000)
    grace_ns = int(grace_s * 1_000_000_000)
    pending: dict[int, _Staged] = {}
    # O_CREAT so the watchdog can be started before pytest ever touches the
    # lock file: run-test.sh starts it first so that it is already watching by
    # the time the first worker takes a lease. plugin.py's own lockfile fixture
    # is equally permissive for the mirror-image reason.
    fd = os.open(lockfile, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        while True:
            _poll_once(fd, lockfile, workers, threshold_ns, grace_ns, pending)
            time.sleep(poll_interval_s)
    finally:
        for page in list(pending):
            _discard(pending, page)  # close any pidfd still staged
        os.close(fd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m pytest_gpu_lease.watchdog',
        description='Escalate SIGTERM -> SIGKILL on a pytest_gpu_lease worker whose '
                    'page has gone unstamped for too long.')
    parser.add_argument('--lockfile', required=True,
                        help='Shared lease lock file to watch. Required, rather than '
                             'defaulted from $GPU_LEASE_LOCKFILE, so that `ps` shows which '
                             'file each watchdog is on -- that is the only way to tell a '
                             'stale watchdog from a live one. Workers still find the same '
                             'file through the environment variable; only this process '
                             'insists on being told.')
    parser.add_argument('--workers', type=int, required=True,
                        help='Size of the GPU pool (the -n given to pytest); pages '
                             '0..workers-1 are watched.')
    parser.add_argument('--threshold', type=float, default=float(_DEFAULT_THRESHOLD_S),
                        help='Seconds a test may legitimately run before its worker is '
                             'treated as wedged (default %(default)s). This is the timeout: '
                             'workers only report when they were last alive, so nothing '
                             'else in the system has an opinion about it.')
    parser.add_argument('--grace', type=float, default=_DEFAULT_GRACE_S,
                        help='Seconds to wait after SIGTERM before re-checking the lock '
                             'and escalating to SIGKILL if it is still held.')
    parser.add_argument('--poll_interval', type=float, default=_DEFAULT_POLL_INTERVAL_S,
                        help='Seconds between sweeps of the lock file.')
    return parser


def _exit_on_signal(signum, _frame):
    """Turn a termination signal into a normal unwind.

    A plain `kill` -- SIGTERM -- is how this service is meant to be stopped,
    and with no handler it would kill the process outright, skipping the
    `finally` in `main` that removes the lock file. SIGHUP is the same story.
    SIGINT already unwinds and is handled only so all three behave alike.

    The signal is named in the log because this now shares a file with
    pytest's stderr, where "stopped" alone does not distinguish a shutdown
    from a crash.
    """
    print(f'pytest_gpu_lease.watchdog: caught {signal.Signals(signum).name}',
          file=sys.stderr, flush=True)
    raise SystemExit(128 + signum)


def _remove_run_files(lockfile: str) -> None:
    """Remove the lock file and any dumps left beside it."""
    for path in [lockfile, *glob.glob(glob.escape(lockfile) + '.*.dump')]:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _exit_on_signal)
    stopped_cleanly = False
    try:
        watch(args.lockfile, args.workers, args.threshold, args.grace,
             args.poll_interval)
        stopped_cleanly = True  # watch() does not return today
    except SystemExit:
        stopped_cleanly = True  # _exit_on_signal: told to stop, so the pass is over
        raise
    finally:
        # Only after a clean stop. On an unexpected failure the pass may still
        # be running, and unlinking then would hand replacement workers a fresh
        # inode while their peers still hold locks on this one -- two workers
        # leasing one GPU, which is the failure the lease exists to prevent.
        if stopped_cleanly:
            _remove_run_files(args.lockfile)
            print('pytest_gpu_lease.watchdog: stopped', file=sys.stderr, flush=True)
        else:
            print(f'pytest_gpu_lease.watchdog: crashed; leaving {args.lockfile} '
                  f'in place for any pass still using it', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
