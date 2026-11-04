# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lease exactly one GPU to each pytest-xdist worker.

Workers of a single pytest run coordinate through POSIX record locks on one shared
file: GPU *n* is represented by the byte range [n*PAGE_SIZE, (n+1)*PAGE_SIZE). A
worker walks the range round-robin and takes the first page it can write-lock; it
holds that lock for its whole session and releases it at teardown.

Three modes, selected by environment:

* ``GPU_LEASE_PIN=<n>`` -- every worker pinned to GPU n, no locking. Checked
  first, so it works with or without xdist. For single-GPU reruns and for
  bisecting a failure onto a known-good device.
* not running under xdist (no ``config.workerinput``) -- GPU 0, no locking.
* otherwise -- lease as described above, one page per worker.

The mapping is deliberately 1:1 worker-to-GPU with no oversubscription knob: running
several tests concurrently on one GPU invites memory pressure and runtime / driver /
firmware / VBIOS races.
"""

import errno
import fcntl
import faulthandler
import itertools
import os
import signal
import struct
import sys
import sysconfig
import time
from pathlib import Path
from types import MappingProxyType

import pytest

STRUCT_FLOCK = 'hhqqi'
PAGE_SIZE = 4096

# `struct flock` uses off_t, we only handle 64-bit off_t and fail loud for 32-bit off_t.
# No plan to support 32-bit off_t
_SIZEOF_OFF_T = sysconfig.get_config_var('SIZEOF_OFF_T')
assert _SIZEOF_OFF_T == 8, (
    f'pytest_gpu_lease requires a 64-bit off_t; this interpreter reports '
    f'SIZEOF_OFF_T={_SIZEOF_OFF_T!r}, so STRUCT_FLOCK={STRUCT_FLOCK!r} '
    f'({struct.calcsize(STRUCT_FLOCK)} bytes) does not describe this '
    f"platform's struct flock")

_RETRY_INTERVAL = 0.05

# Page layout, shared with watchdog.py: an 8-byte `monotonic_ns` stamp at the
# page base -- 0 meaning "between tests" -- then the pid of the worker that
# wrote it.
#
# The pid is what makes a stale stamp harmless. A page is freed still carrying
# an expired stamp in exactly one situation, and it is the one this feature
# creates: the watchdog SIGKILLs a wedged worker, whose stamp is by definition
# past the threshold, and xdist gives the replacement the page it just vacated.
# Between that F_SETLK and the replacement's first write, the page reads
# locked-and-ancient. Writing sooner only narrows that window; comparing the
# recorded pid against the current lock holder closes it, however wide it is.
#
# Stamp first, pid second, so a reader that catches the pair half written sees
# (new stamp, old pid) and skips it -- never (old stamp, new pid), the one
# combination that would kill a healthy worker. That ordering is why this needs
# no 16-byte atomic write.
HEARTBEAT_FMT = '<Q'
OWNER_FMT = '<q'
OWNER_OFFSET = struct.calcsize(HEARTBEAT_FMT)
HEARTBEAT_SIZE = OWNER_OFFSET + struct.calcsize(OWNER_FMT)


def dump_path(lockfile, pid: int) -> str:
    """Where a leasing worker's faulthandler writes its stack dump.

    One file per worker, not the shared stderr: a dump is many small writes, so
    two workers dumping at once would splice into each other. `watchdog.py`
    reads these back and relays them, being the only serial writer into the
    pass's stderr.

    Keyed by pid, not by GPU: a pass restarts workers many times, and a
    replacement taking the freed page would otherwise truncate its
    predecessor's dump before anyone had read it. pid is also the only handle
    the watchdog has -- `F_GETLK` gives it that and nothing else.
    """
    return f'{lockfile}.{pid}.dump'

# A plain global object to track current GPU lease. It is safe here because
# `gpu_id` is session-scoped and, under xdist, "session" means "per worker
# process": at most one lease is ever active in a given interpreter.
_active_lease: tuple[int, int] | None = None  # (fd, page_base)


# Stand-in for a process that is not an xdist worker: the controller, a plain
# `pytest` run, or xdist not loaded at all. All three are the same thing here, so
# one set of defaults covers them and callers never test for absence.
#
# 'master' is the label xdist's own worker_id fixture reports off-worker. A count
# of 0 is an unambiguous sentinel for "not distributed": a real worker always sees
# at least 1, and `-n 0` creates no workers and hence no workerinput.
_NO_XDIST = MappingProxyType({'workerid': 'master', 'workercount': 0})


def _workerinput(config):
    """xdist's per-worker payload, with off-worker defaults filled in.

    Always returns a mapping. Two keys matter, and both are deliberately sourced
    from here rather than from the more obvious places:

    ``workercount`` -- the size of the GPU pool. NOT
    ``PYTEST_XDIST_WORKER_COUNT``: xdist sets that inside the worker process, but
    this module is a ``pytest11`` entry-point plugin imported during
    ``Config._preparse``, long before. Read at module scope it saw 0 and silently
    put every worker on GPU 0. The value here is fixed at the original ``-n`` for
    the whole run -- a crashed worker is replaced from its own spec, so the pool
    never resizes and the page it freed stays in range for its replacement.

    ``workerid`` -- the label used in announcements. NOT xdist's ``worker_id``
    fixture: depending on it made ``gpu_id`` unresolvable whenever xdist was not
    loaded (``-p no:xdist``, ``PYTEST_DISABLE_PLUGIN_AUTOLOAD``), including on the
    pinned and no-xdist paths, which need nothing from xdist at all.
    """
    workerinput = getattr(config, 'workerinput', None)
    return _NO_XDIST if workerinput is None else workerinput


def _env_pin() -> int | None:
    """``GPU_LEASE_PIN`` as an int, or None. Read lazily, never at import."""
    raw = os.getenv('GPU_LEASE_PIN', default=None)
    return None if raw is None else int(raw)


def _announce(config, message: str) -> None:
    """Write `message` to stderr *now*, bypassing pytest's output capture.

    The lease is decided during fixture setup, and pytest captures setup output at
    the fd level, replaying it only in the report section -- and only for failing
    tests, unless ``-rA`` is given. On a green run the GPU assignment would never
    be shown until the run was over, which is the whole point of announcing it.

    ``capsys.disabled()`` is the documented way to suspend capture, but every
    capture fixture is function-scoped while ``gpu_id`` is session-scoped, so
    requesting one here would raise ``ScopeMismatch``. We therefore call the
    capture manager that ``capsys.disabled()`` itself delegates to.

    That is ``_pytest.capture`` internals, not public API. If a future pytest
    reorganises it, the lookup below degrades to a plain print rather than
    breaking the run -- the symptom is the line reappearing only in the replayed
    "Captured stderr setup" section, which is the cue to pin pytest and revisit.
    """
    capman = config.pluginmanager.getplugin('capturemanager')
    disabled = getattr(capman, 'global_and_fixture_disabled', None)
    if disabled is None:  # -p no:capture, or the internals moved
        print(message, file=sys.stderr, flush=True)
        return
    with disabled():
        print(message, file=sys.stderr, flush=True)


@pytest.fixture(scope='session')
def _gpu_lease_lockfile(tmp_path_factory):
    """Path to the run-wide lock file, created if absent.

    NOT autouse: this plugin auto-loads into every pytest run in the environment,
    including GPU-less suites (python/test, modules/flash/tests/test_gpu_targets.py),
    which must not touch the filesystem.

    The file is never sized or truncated. POSIX record locks may be placed beyond EOF,
    so pre-sizing buys nothing -- and the old open(..., 'wb') let a late-starting worker
    truncate a file its peers were already locking.

    ``GPU_LEASE_LOCKFILE``, when set, wins over the derived path. It exists for
    the watchdog, which is started separately and before pytest, and so cannot
    predict what ``getbasetemp()`` will resolve to.
    """
    override = os.getenv('GPU_LEASE_LOCKFILE', default=None)
    if override is not None:
        lockfile = Path(override)
    else:
        # getbasetemp().parent is shared by all workers of the run; getbasetemp() is per-worker.
        lockfile = tmp_path_factory.getbasetemp().parent / 'gpulock'
    fd = os.open(lockfile, os.O_RDWR | os.O_CREAT, 0o644)
    os.close(fd)
    return lockfile


@pytest.fixture(scope='session')  # under xdist, "session" scope is per-worker process
def gpu_id(request):
    """Index of the GPU this worker owns for the duration of its session.

    Every mode announces its choice, not just the leasing one: without it there is
    no way to confirm that GPU_LEASE_PIN actually took effect either.

    Depends on no xdist fixture, so it resolves even under ``-p no:xdist``.
    """
    workerinput = _workerinput(request.config)
    worker_id = workerinput['workerid']

    # GPU_LEASE_PIN wins over everything, distributed or not: "put all work on
    # GPU n" is a debugging override and should not depend on how pytest is run.
    pinned = _env_pin()
    if pinned is not None:
        _announce(request.config, f'{worker_id} uses GPU {pinned} (GPU_LEASE_PIN, no lease)')
        yield pinned
        return

    nworkers = int(workerinput['workercount'])
    if nworkers == 0:
        _announce(request.config, f'{worker_id} uses GPU 0 (no xdist, no lease)')
        yield 0
        return

    # Resolved lazily, NOT as a fixture parameter: pytest instantiates declared
    # params before the body runs, so naming _gpu_lease_lockfile in the signature
    # would create the file in the no-xdist and pinned modes too -- the very
    # side effect dropping `autouse` was meant to prevent.
    lockfile = request.getfixturevalue('_gpu_lease_lockfile')
    global _active_lease
    with open(lockfile, 'r+b') as f:
        for gpu in itertools.cycle(range(nworkers)):
            page_base = PAGE_SIZE * gpu
            claim = struct.pack(STRUCT_FLOCK, fcntl.F_WRLCK, os.SEEK_SET,
                                page_base, PAGE_SIZE, 0)
            try:
                fcntl.fcntl(f, fcntl.F_SETLK, claim)
            except BlockingIOError:
                # Every page is taken for the moment. Sleep instead of spinning --
                # the original loop pegged a core while waiting.
                if gpu == nworkers - 1:
                    time.sleep(_RETRY_INTERVAL)
                continue
            # Initialize the heartbeat value, before anything else that can
            # take time. Until this write the page is locked but still carries
            # the *previous* holder's stamp, and the one way a page is freed
            # with a stale stamp is the case this feature creates: the watchdog
            # SIGKILLed a wedged worker, whose stamp is by definition past the
            # threshold, and xdist's replacement takes the page it just
            # vacated. A poll landing in that gap would read locked-and-ancient
            # and SIGTERM a brand-new healthy worker -- which then gets
            # SIGKILLed regardless, since escalation offers no reprieve.
            os.pwrite(f.fileno(), struct.pack(HEARTBEAT_FMT, time.monotonic_ns()), page_base)
            os.pwrite(f.fileno(), struct.pack(OWNER_FMT, os.getpid()),
                      page_base + OWNER_OFFSET)
            _announce(request.config,
                      f'{worker_id} uses GPU {gpu} filelock = {lockfile}')
            # Handle SIGTERM from watchdog to print a full stack dump naming the frame.
            # Note signal handler itself cannot detect blocked GPU kernel:
            # CPython only runs a Python signal handler when the main thread
            # reaches a bytecode boundary.
            # Only under a watchdog: GPU_LEASE_LOCKFILE is how run-test.sh
            # hands one path to both sides, so without it nothing will ever
            # read a dump.
            #
            # NOT sys.stderr: at fixture setup that is pytest's fd-level
            # capture target, and a capture buffer is only replayed when the
            # test finishes -- which a wedged test never does. faulthandler
            # needs its descriptor now, since the wedged worker runs no Python
            # later, so it gets a file of its own (see dump_path).
            dumpfile = None
            if os.getenv('GPU_LEASE_LOCKFILE'):
                dumpfile = open(dump_path(lockfile, os.getpid()), 'w')
                # chain=True so SIGTERM still terminates: faulthandler dumps,
                # then restores the previous handler and re-raises, so the
                # default action runs. Without it SIGTERM is disarmed for the
                # rest of the session, which makes the watchdog's escalation
                # the only way to end a worker and breaks `docker stop` and CI
                # cancellation besides. SIGKILL stays what it should be -- the
                # last resort for a process SIGTERM could not end.
                faulthandler.register(signal.SIGTERM, file=dumpfile,
                                      all_threads=True, chain=True)
            _active_lease = (f.fileno(), page_base)
            try:
                yield gpu
            finally:
                _active_lease = None
                if dumpfile is not None:
                    faulthandler.unregister(signal.SIGTERM)  # before closing its file
                    dumpfile.close()
                    # The moment this worker can no longer wedge. /dev/shm is
                    # small in a container and a long pass restarts workers
                    # many times, so these must not outlive their owner.
                    try:
                        os.unlink(dump_path(lockfile, os.getpid()))
                    except FileNotFoundError:
                        pass
                release = struct.pack(STRUCT_FLOCK, fcntl.F_UNLCK, os.SEEK_SET,
                                      page_base, PAGE_SIZE, 0)
                fcntl.fcntl(f, fcntl.F_SETLK, release)
            return


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol():
    """Stamp this worker's page with `monotonic_ns()` before each test, 0 after.

    0 means "between tests" and the watchdog skips it. `watchdog.py` owns the
    timeout; this side only reports liveness.

    Wraps the whole protocol so a wedge in fixture setup or teardown counts
    too. Consequence: `gpu_id` resolves *inside* the first wrapped call, so
    `_active_lease` is still None here for a worker's first test -- `gpu_id`
    writes that initial stamp itself.

    No lease means no I/O. This hook runs in every pytest process in the
    environment, GPU-less suites included.

    Runs on the worker's MainThread deliberately: a background heartbeat
    thread would keep ticking through the very wedge this exists to detect.
    """
    lease = _active_lease
    if lease is None:
        try:
            yield
        finally:
            # `gpu_id` may have taken the lease inside this very call -- a
            # worker's first test -- and written the initial stamp with nothing
            # to clear it. Left alone that stamp keeps ageing while the worker
            # sits between its first and second test, so a slow first test plus
            # the gap after it can cross the threshold and get an idle worker
            # killed. Zero it here, as every later test does for itself.
            lease = _active_lease
            if lease is not None:
                fd, page_base = lease
                os.pwrite(fd, struct.pack(HEARTBEAT_FMT, 0), page_base)
        return
    fd, page_base = lease
    # Stamp only; the owner pid is written once, when the lease is taken.
    os.pwrite(fd, struct.pack(HEARTBEAT_FMT, time.monotonic_ns()), page_base)
    try:
        yield
    finally:
        # On the last test this same call tears down `gpu_id`, which clears
        # `_active_lease` and then closes `fd`. Re-reading the global is what
        # makes the write safe: `fd` is a bare integer, so once it is closed
        # the kernel is free to hand that number to the next file opened -- and
        # the rest of session teardown opens plenty. Writing to it then would
        # not raise, it would silently drop eight zero bytes at `page_base`
        # into somebody else's file. The watchdog needs nothing from this write
        # anyway: it only reads pages it found locked, and the lease is gone.
        if _active_lease is not None:
            os.pwrite(fd, struct.pack(HEARTBEAT_FMT, 0), page_base)


@pytest.fixture(scope='session')
def gpu_device_class() -> str:
    """Accelerator class used to build ``gpu_device``. Defaults to ``'cuda'``.

    The lease mechanism itself is device-agnostic -- it hands out an ordinal and
    never imports torch -- so retargeting a suite at another backend is purely a
    matter of how that ordinal is spelled. Override this fixture in a conftest.py
    (see "Extending to other device classes" in the plan) or set
    ``GPU_LEASE_DEVICE_CLASS`` in the environment.
    """
    return os.getenv('GPU_LEASE_DEVICE_CLASS', default='cuda')


@pytest.fixture(scope='session')
def gpu_device(gpu_id, gpu_device_class) -> str:
    """``gpu_id`` as a torch device string, e.g. ``'cuda:3'`` or ``'xpu:3'``."""
    return f'{gpu_device_class}:{gpu_id}'


@pytest.fixture(scope='session')
def torch_gpu(gpu_id) -> int:
    """Back-compat alias for :func:`gpu_id`."""
    return gpu_id


# Errnos a pipe write produces when the process on the other end is gone. The
# closed-channel case carries no errno at all, hence the isclosed() check too.
_PEER_GONE_ERRNOS = frozenset({errno.EPIPE, errno.EBADF, errno.ECONNRESET,
                               errno.ESHUTDOWN})


def _tolerate_closed_worker_channel() -> None:
    """Stop a second dying worker turning the first one's crash into an
    INTERNALERROR that kills the whole session.

    `worker_errordown(A)` -> `remove_node` -> `check_schedule` -> `_send_tests`
    tops up the *other* nodes' queues. If B is dying too -- its own crash
    report still in flight -- that send raises `OSError: cannot send (already
    closed?)`, uncaught, and the run ends.

    Usually triggered by killing several stalled GPU workers at once -- by
    hand, or by the watchdog, which staggers to one kill per poll for exactly
    this reason. Two ordinary crashes landing close enough together do it too.

    Patches `sendcommand` rather than one scheduler's `_send_tests` because
    every `--dist` mode funnels through it, and `WorkerController.shutdown`
    already wraps the same call in the same `except OSError` -- the schedulers
    were just missed.

    Swallowing is enough: B's own crash will `remove_node` it shortly, which
    requeues its whole pending list, phantom entry included. Removing the node
    from inside the send that node removal triggered would recurse.
    """
    try:
        from xdist.workermanage import WorkerController
    except ImportError:
        return  # xdist not installed in this environment, or this process never imports it

    original = WorkerController.sendcommand
    if getattr(original, '_gpu_lease_tolerates_closed_channel', False):
        return  # pytest_configure can run more than once in the same interpreter

    def sendcommand(self, name, **kwargs):
        try:
            original(self, name, **kwargs)
        except OSError as exc:
            # Only a peer that has gone. `sendcommand` reaches OSError by two
            # different routes and they are not interchangeable: execnet's
            # `Channel.send` raises one built from a bare string when the
            # channel is already closed (so `errno` is None -- this is the case
            # actually observed), while the gateway's own write below it can
            # fail for any reason a pipe write can.
            #
            # Swallowing all of them would file a full disk or a failing device
            # as "worker crashed" and quietly drop that node's tests, turning a
            # broken machine into a green-looking short run. Re-raise anything
            # that is not recognisably the peer being gone.
            if not (self.channel.isclosed() or exc.errno in _PEER_GONE_ERRNOS):
                raise
            print(f'pytest_gpu_lease: {self.gateway.id} channel already closed, '
                  f'dropping {name}({kwargs}) instead of crashing the session ({exc})',
                  file=sys.stderr, flush=True)

    sendcommand._gpu_lease_tolerates_closed_channel = True
    WorkerController.sendcommand = sendcommand


@pytest.hookimpl
def pytest_configure(config):
    """Apply the scheduler guard once per process. A no-op where it does not
    apply: only the controller schedules, and a non-distributed run never
    imports xdist at all.
    """
    _tolerate_closed_worker_channel()
