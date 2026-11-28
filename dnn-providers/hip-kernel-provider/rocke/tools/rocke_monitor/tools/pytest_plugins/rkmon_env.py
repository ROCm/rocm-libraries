# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""pytest plugin for the review checks: a test that fails because this host's hipcc cannot compile HIP at all is skipped,
with the reason, instead of failing (a host with hipcc but no HIP headers fails every hipcc case at
`#include <hip/hip_runtime.h>`, which is no test failure).

Only a subprocess.CalledProcessError from a hipcc command counts, or another exception whose message says
`hip/hip_runtime.h` was not found (a helper that wraps hipcc's failure, e.g. in a RuntimeError), and only when a
probe that compiles `#include <hip/hip_runtime.h>` with the same hipcc (and --offload-arch) fails too (once per hipcc).

A test that fails with ModuleNotFoundError for an optional module this host lacks (RKMON_HOST_ABSENT, default torch;
CI's GPU lanes have no torch either) is skipped too, and listed on "rkmon-import:" lines: such a test should call
pytest.importorskip (a test that imports torch directly fails where torch is missing).

Every skipped test, at collection or in a run, is printed at the end on a "rkmon-skip:" line with its reason, and
marked when the reason names a tool CI's lanes have that this run lacks (cmake, PyYAML; rkmon_skips.py): a skip is never
silent, and a missing tool never reads as a pass. The checks keep those lines in tests_skipped.tsv, which the remote
GPU tests read for the cases this machine skipped for want of a GPU (rkmon_lib/gpu.py).
"""

import importlib.util
import os
import re
import shutil
import subprocess
import tempfile

import pytest

import rkmon_skips

REASON = "this host's hipcc cannot compile <hip/hip_runtime.h>"
HIP_HEADER_MISSING = re.compile(r"hip/hip_runtime\.h'? file not found")
HOST_ABSENT = tuple(
    m for m in os.environ.get("RKMON_HOST_ABSENT", "torch").split(",") if m
)
_PROBES: dict = {}
_SKIPPED: list = []
_NO_MODULE: list = []
_SKIPS: list = []  # (test id, reason)


def _absent(exc: ModuleNotFoundError):
    """The missing module's top-level name, when it is one this host is known to lack; else None."""
    name = (exc.name or "").split(".")[0]
    return (
        name if name in HOST_ABSENT and importlib.util.find_spec(name) is None else None
    )


def _argv(cmd) -> list:
    return [str(a) for a in cmd] if isinstance(cmd, (list, tuple)) else str(cmd).split()


def _host_cannot_compile_hip(argv: list) -> bool:
    key = (argv[0], *[a for a in argv if a.startswith("--offload-arch=")])
    if key not in _PROBES:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "probe.hip")
            with open(src, "w") as f:
                f.write(
                    "#include <hip/hip_runtime.h>\n__global__ void rkmon_probe() {}\n"
                )
            try:
                r = subprocess.run(
                    [*key, "--genco", "-o", os.path.join(td, "probe.co"), src],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                _PROBES[key] = r.returncode != 0
            except (OSError, subprocess.TimeoutExpired):
                _PROBES[key] = True
    return _PROBES[key]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when != "call" or not rep.failed or call.excinfo is None:
        return
    exc = call.excinfo.value
    mod = _absent(exc) if isinstance(exc, ModuleNotFoundError) else None
    if mod:
        rep.outcome = "skipped"
        rep.longrepr = (
            str(item.path),
            item.location[1] or 0,
            f"Skipped: rkmon: this host has no {mod}; the test should use pytest.importorskip",
        )
        _NO_MODULE.append((item.nodeid, mod))
        return
    if isinstance(exc, subprocess.CalledProcessError):
        argv = _argv(exc.cmd)
        if (
            not argv
            or not os.path.basename(argv[0]).startswith("hipcc")
            or not _host_cannot_compile_hip(argv)
        ):
            return
    else:
        # A helper that wraps hipcc's failure in its own exception, with hipcc's stderr in the message.
        hipcc = shutil.which("hipcc") if HIP_HEADER_MISSING.search(str(exc)) else None
        if not hipcc or not _host_cannot_compile_hip([hipcc]):
            return
    rep.outcome = "skipped"
    rep.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: rkmon: {REASON}")
    _SKIPPED.append(item.nodeid)


def _skip_reason(longrepr) -> str:
    """A skip's reason: pytest keeps it as (path, line, "Skipped: <reason>")."""
    text = (
        longrepr[2]
        if isinstance(longrepr, tuple) and len(longrepr) == 3
        else str(longrepr or "").strip()
    )
    text = str(text).splitlines()[-1] if str(text).strip() else ""
    return text[len("Skipped: ") :] if text.startswith("Skipped: ") else text


def pytest_runtest_logreport(report):
    if report.skipped and not hasattr(report, "wasxfail"):
        _SKIPS.append((report.nodeid, _skip_reason(report.longrepr)))


def pytest_collectreport(report):
    if report.skipped:
        _SKIPS.append(
            (report.nodeid or str(report.fspath), _skip_reason(report.longrepr))
        )


def _cwd_relative(config, nodeid) -> str:
    """A node id relative to the run's directory, as pytest shows it (a node id is relative to the rootdir, which an
    installed entry's pytest.ini can move). Computed here: pytest's own Config.cwd_relative_nodeid takes a string up
    to pytest 9.1 and a node id object from 9.2 on."""
    nodeid = str(nodeid)
    root = getattr(config, "rootpath", None)
    cwd = getattr(getattr(config, "invocation_params", None), "dir", None)
    if (
        not nodeid
        or root is None
        or cwd is None
        or os.path.abspath(root) == os.path.abspath(cwd)
    ):
        return nodeid
    path, sep, rest = nodeid.partition("::")
    return os.path.relpath(os.path.join(str(root), path), str(cwd)) + sep + rest


def pytest_terminal_summary(terminalreporter):
    missing = rkmon_skips.missing_tools() if _SKIPS else set()
    for nodeid, reason in _SKIPS:
        terminalreporter.write_line(
            rkmon_skips.line(
                _cwd_relative(terminalreporter.config, nodeid), reason, missing
            )
        )
    if _SKIPPED:
        terminalreporter.write_line(
            f"rkmon: {len(_SKIPPED)} hipcc failures skipped ({REASON})"
        )
    if _NO_MODULE:
        mods = ", ".join(sorted({m for _, m in _NO_MODULE}))
        terminalreporter.write_line(
            f"rkmon-import: {len(_NO_MODULE)} tests skipped: they import {mods}, which this "
            f"host lacks, without pytest.importorskip"
        )
        for nodeid, _ in _NO_MODULE[:10]:
            terminalreporter.write_line(f"rkmon-import:   {nodeid}")
