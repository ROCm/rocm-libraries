# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Kernel ids and applicability declines, read from the backend's INFO log.

hipDNN has no API that names the kernel an ingestor engine picked, so the id is taken from the
line the plan builder logs once a plan constructs:

* ``ingestor: engine '<name>' selected kernel <uuid> at rank N from M candidate(s) ...``
  (``GenericPlanBuilder.hpp:509``)
* ``ingestor: engine '<name>' slot '<slot>' selected kernel <uuid> at rank N from M
  candidate(s)`` (slotted engines, ``GenericPlanBuilder.hpp:1365``)

An engine's applicability declines are INFO lines too (``[<Builder>::isApplicable] <reason>``
from the ASM builders, ``ingestor: engine '<name>' declined ...`` from ingestor engines).

:func:`install_kernel_log` registers a synchronous user log callback on the backend library
already mapped into the process, then raises the global level to INFO. The callback runs on the
thread that logs, so it does nothing unless that thread is inside :func:`capture`, which
``_runtime.run`` opens around a plan build: every other INFO line of the backend (execute logs
one per API call) costs one attribute lookup. A capture is per thread; a line logged by another
thread never reaches it.

Backend entry points, through ``ctypes`` on the mapped library:

* ``hipdnnSetUserLogCallback_ext(hipdnnUserLogCallback_t, hipdnnSeverity_t,
  hipdnnLogCallbackMode_t, hipdnnUserLogCallbackHandle_t) -> hipdnnStatus_t``
  (``hipdnn_backend.h:736``); the callback is ``void (*)(void* userHandle, hipdnnSeverity_t,
  const char* message)`` (``HipdnnBackendCallbackTypes.h:55``). ``HIPDNN_LOG_CALLBACK_SYNC`` is
  0 (``hipdnn_backend.h:690``); a minimum severity of ``HIPDNN_SEV_OFF`` (4) removes the callback.
* ``hipdnnBackendSetGlobalLogLevel_ext(hipdnnSeverity_t) -> hipdnnStatus_t``
  (``hipdnn_backend.h:752``); ``HIPDNN_SEV_INFO`` is 0 (``HipdnnBackendCallbackTypes.h:21``).
"""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import json
import os
import re
import threading

import conformance_contract as contract

SEV_INFO = 0
SEV_OFF = 4
LOG_CALLBACK_SYNC = 0
STATUS_SUCCESS = 0

_SELECTED = re.compile(
    r"ingestor: engine '(?P<engine>[^']+)' (?:slot '(?P<slot>[^']+)' )?"
    r"selected kernel (?P<kernel>\S+) at rank"
)
_INGESTOR_DECLINE = re.compile(
    r"ingestor: engine '(?P<engine>[^']+)' .*(?:declined|cached decline|no kernel applies)"
)
_BUILDER_DECLINE = re.compile(r"\[(?P<engine>\w+)::isApplicable\]")

_LOCAL = threading.local()  # .capture: the open Capture of this thread, if any
_DECLINE_COUNTS: dict = {}  # engine -> declines written by this process
_DECLINE_LOCK = threading.Lock()
_INSTALLED = None  # (library, C callback, user handle) once registered

_CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p)


class Capture:
    """Kernel selections logged on this thread during one plan build."""

    def __init__(self):
        self._selected: list = []  # (engine, slot or None, kernel) in log order

    def kernel(self, plan_engine: str) -> str:
        """Kernel id of the engine that served the plan; empty when no selection was logged.

        ``plan_engine`` is the plan's UED name (``hipkernel:flydsl_sdpa_cdna``) while the log
        carries the engine's own name; when a build logged several engines (one fell back to
        another) the one whose name the plan name ends with wins, else the last.
        """
        if not self._selected:
            return ""
        engines = [engine for engine, _, _ in self._selected]
        served = next(
            (
                e
                for e in reversed(engines)
                if plan_engine == e or plan_engine.endswith(":" + e)
            ),
            engines[-1],
        )
        picked = [(slot, kernel) for e, slot, kernel in self._selected if e == served]
        if any(slot is not None for slot, _ in picked):
            return "|".join(f"{slot}={kernel}" for slot, kernel in picked)
        return picked[-1][1]


@contextlib.contextmanager
def capture():
    """Collect this thread's kernel selections and declines until the block ends."""
    previous = getattr(_LOCAL, "capture", None)
    current = _LOCAL.capture = Capture()
    try:
        yield current
    finally:
        _LOCAL.capture = previous


def on_line(message: str) -> None:
    """Handle one backend log line; a no-op unless this thread is inside :func:`capture`."""
    current = getattr(_LOCAL, "capture", None)
    if current is None:
        return
    selected = _SELECTED.search(message)
    if selected is not None:
        current._selected.append(
            (selected["engine"], selected["slot"], selected["kernel"])
        )
        return
    decline = _INGESTOR_DECLINE.search(message) or _BUILDER_DECLINE.search(message)
    if decline is not None:
        _keep_decline(decline["engine"], message)


def _keep_decline(engine: str, line: str) -> None:
    directory = os.environ.get(contract.DECLINES_DIR_ENV)
    if not directory:
        return
    with _DECLINE_LOCK:
        count = _DECLINE_COUNTS.get(engine, 0)
        if count >= contract.DECLINES_PER_ENGINE:
            return
        _DECLINE_COUNTS[engine] = count + 1
        nodeid = os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]
        worker = os.environ.get("PYTEST_XDIST_WORKER") or "main"
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"{worker}.jsonl"), "a") as out:
            out.write(json.dumps({"engine": engine, "line": line, "nodeid": nodeid}))
            out.write("\n")


def _c_callback(user_handle, severity, message):
    on_line(message.decode("utf-8", "replace") if message else "")


def _mapped_backend(maps_path: str = "/proc/self/maps") -> str | None:
    """Path of the ``libhipdnn_backend`` mapped into this process, if any."""
    try:
        with open(maps_path) as maps:
            for line in maps:
                if "libhipdnn_backend" in line:
                    return line.split()[-1]
    except OSError:
        return None
    return None


def _find_backend():
    path = _mapped_backend()
    if path is None:
        try:
            import hipdnn_frontend  # noqa: F401  (maps the backend it links against)
        except ImportError:
            return None
        path = _mapped_backend()
    return None if path is None else ctypes.CDLL(path)


def _install(lib) -> bool:
    global _INSTALLED
    callback = _CALLBACK_TYPE(_c_callback)
    handle = ctypes.c_void_p(
        1
    )  # non-null; with the callback it identifies the registration
    set_callback = lib.hipdnnSetUserLogCallback_ext
    set_callback.argtypes = [
        _CALLBACK_TYPE,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    set_callback.restype = ctypes.c_int
    set_level = lib.hipdnnBackendSetGlobalLogLevel_ext
    set_level.argtypes = [ctypes.c_int]
    set_level.restype = ctypes.c_int
    # The callback first: raising the level before it exists would let INFO lines reach the
    # sinks with nobody listening.
    if set_callback(callback, SEV_INFO, LOG_CALLBACK_SYNC, handle) != STATUS_SUCCESS:
        return False
    if set_level(SEV_INFO) != STATUS_SUCCESS:
        set_callback(callback, SEV_OFF, LOG_CALLBACK_SYNC, handle)
        return False
    _INSTALLED = (lib, callback, handle)
    return True


def _uninstall() -> None:
    global _INSTALLED
    if _INSTALLED is None:
        return
    lib, callback, handle = _INSTALLED
    _INSTALLED = None
    lib.hipdnnSetUserLogCallback_ext(callback, SEV_OFF, LOG_CALLBACK_SYNC, handle)


def install_kernel_log() -> bool:
    """Start reading the backend's INFO log; False when no backend library is mapped.

    Idempotent. Registers an exit hook that removes the callback.
    """
    if _INSTALLED is not None:
        return True
    lib = _find_backend()
    if lib is None or not _install(lib):
        return False
    atexit.register(_uninstall)
    return True
