#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Gauntlet fuzz driver (RFC 0023, section 7).

Draws random legal samples from the spaces in ``spaces/``, writes each batch as a concrete
sweep under ``<work>/bundles/gauntlet/<Op>/<Space>/<dtype>_cases/`` (the template the space
names, copied as ``graph.template.json``, plus a generated ``sweep.json`` holding the samples
of that input dtype; ``dtype_tag`` says why the directory is named so), runs the unchanged
``hipdnn_integration_tests`` binary on it with ``--outcome-report``, and records exactly one
outcome per sample: ``PASS``, ``DECLINED``, ``WRONG``, ``CRASH``, ``UNCLAIMED_PASS`` or
``UNEXERCISED``. It does not change what the binary runs.

Synopsis::

    run_gauntlet_fuzz.py --binary BIN [--test-article A] [--test-engine E] [--test-config C]
        [--work-dir DIR] [--samples N] [--seed S] [--data-seed D] [--filter REGEX]
        [--time-budget 30m] [--case-timeout 10m] [--batch-size N] [--summary FILE]
        [--spaces-dir DIR] [--binary-arg ARG ...]

Control (a flag overrides its environment variable)::

    HIPDNN_FUZZ_SAMPLES      --samples       samples per space (default 32; 0 = unlimited)
    HIPDNN_FUZZ_SEED         --seed          shape seed of the first sample (default random, printed)
    HIPDNN_FUZZ_DATA_SEED    --data-seed     input-data seed for every sample (default derived)
    HIPDNN_FUZZ_FILTER       --filter        regex on <Op>/<Space> (default: every space)
    HIPDNN_FUZZ_TIME_BUDGET  --time-budget   wall-clock limit, e.g. 30m, 4h (default none)
    HIPDNN_FUZZ_CASE_TIMEOUT --case-timeout  longest one case may run (default 10m)

Sample ``i`` of a space has shape seed ``S + i``; batches are drawn round-robin over the
spaces. Without ``--samples`` and with ``--time-budget`` (or with ``--samples 0``) the run is
continuous: fresh seeds every batch until the time budget ends, the sample cap is hit or
SIGINT arrives (the current batch finishes, then the report is written). A crash costs one
case, not the run.

Outputs under the work directory: ``outcomes.jsonl`` (one record per sample),
``samples.jsonl``, ``summary.json``, ``reports/`` and ``logs/`` (per binary invocation) and
``failures/<case>/`` for every ``WRONG`` and ``CRASH``. The exit status is non-zero iff a
sample was ``WRONG`` or ``CRASH`` (2 for usage and space errors).
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import sampler  # noqa: E402
import space as space_lib  # noqa: E402

MIGRATION_DIR = HERE.parent / "migration-scripts"
SCRIPTS_DIR = HERE.parent / "scripts"
DRIVER_PATH = Path(__file__).resolve()

OUTCOMES = ("PASS", "DECLINED", "WRONG", "CRASH", "UNCLAIMED_PASS", "UNEXERCISED")
FAILING = ("WRONG", "CRASH")
DEFAULT_SAMPLES = 32
DEFAULT_BATCH_SIZE = 16
DEFAULT_CASE_TIMEOUT = "10m"
LOG_TAIL_LINES = 200
LOG_TAIL_BYTES = 32 * 1024
HOST_FILL_ENV = "HIPDNN_TEST_HOST_INPUT_FILL"
# The axes whose values the summary breaks the outcomes down by (``summary["axis_outcomes"]``).
AXIS_OUTCOME_AXES = ("alignment", "placement", "domain")
ZERO_TESTS_MARKER = "Error: zero tests ran"
NO_DEVICE_MARKER = "No HIP devices available"
GTEST_FILTER_LINE = re.compile(r"\s*gtest_filter:")
# The axes a space's input data type is read from, first match wins (see ``dtype_tag``).
DTYPE_TAG_AXES = ("x_dtype", "dtype")
POLL_SECONDS = 0.05
MAX_PRINTED_FINDINGS = 20


class DriverError(Exception):
    """A usage or environment error (exit status 2)."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def load_bundle_utils():
    """``migration-scripts/bundle_utils.py``, imported by path (no copy)."""
    module = sys.modules.get("bundle_utils")
    if module is not None:
        return module
    path = MIGRATION_DIR / "bundle_utils.py"
    spec = importlib.util.spec_from_file_location("bundle_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bundle_utils"] = module
    spec.loader.exec_module(module)
    return module


def load_support_matrix_units():
    """``scripts/support_matrix_units.py``, imported by path (no copy): the module whose
    ``apply_tensor_patches`` expands a sweep case's ``tensor_patches`` as the harness does.
    """
    module = sys.modules.get("support_matrix_units")
    if module is not None:
        return module
    path = SCRIPTS_DIR / "support_matrix_units.py"
    spec = importlib.util.spec_from_file_location("support_matrix_units", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["support_matrix_units"] = module
    spec.loader.exec_module(module)
    return module


def dtype_tag(sp, sample):
    """The dtype tag of a sample's input data type (``fp16``, ``bfp16``, ``fp32``), or None.

    An engine's TOML config matches its tolerance, validator and skip globs against the full
    GTest name ``<suite>.<case>`` (``TomlGuards.hpp`` ``currentTestName()``), and the quick
    bundles carry the input dtype tag in the case id, as in ``LayernormBackward*fp16_*``.
    Gauntlet case ids are ``s<seed>``, so the tag goes in the suite, which the harness derives
    from the sweep directory path (``BundleDiscovery.hpp`` ``deriveSuiteName``). The tag is
    the sample's ``x_dtype`` when its space has that axis (mixed precision keeps the other
    tensors apart from the input) and its ``dtype`` otherwise; a space with neither has no tag.
    Spelling is that of the bundle case ids (``bundle_utils.DTYPE_TOKEN``).
    """
    for axis in DTYPE_TAG_AXES:
        if axis in sp.axes:
            value = str(sample.axis_values[axis])
            token = load_bundle_utils().DTYPE_TOKEN.get(value)
            return token or re.sub(r"[^a-z0-9_]", "_", value.lower())
    return None


def dtype_directory(tag):
    """The sweep directory name of a dtype tag. The TOML globs read ``fp16_*``, so the
    directory name, which becomes part of the suite name, keeps an underscore after the tag.
    """
    return f"{tag}_cases"


_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)?")
_DURATION_UNITS = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    None: 1.0,
}


def parse_duration(text):
    """Seconds for ``30m``, ``4h``, ``1h30m``, ``90`` (seconds), ``0.5s``; None for ``0``/``none``."""
    text = str(text).strip().lower()
    if text in ("", "0", "none", "off"):
        return None
    pos = 0
    total = 0.0
    while pos < len(text):
        match = _DURATION_PART.match(text, pos)
        if match is None:
            raise ValueError(
                f"invalid duration {text!r} (examples: 90, 30m, 4h, 1h30m)"
            )
        total += float(match.group(1)) * _DURATION_UNITS[match.group(2)]
        pos = match.end()
    return total if total > 0 else None


def json_dump(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def resolve_binary(text):
    """argv prefix for ``--binary``: a path, a ``.py`` script, or a quoted command line."""
    path = Path(text)
    if path.is_file():
        return [sys.executable, str(path)] if path.suffix == ".py" else [str(path)]
    parts = shlex.split(text, posix=os.name != "nt")
    if not parts:
        raise DriverError("--binary is empty")
    if parts[0].lower().endswith(".py"):
        parts.insert(0, sys.executable)
    elif shutil.which(parts[0]) is None:
        raise DriverError(f"--binary '{text}': '{parts[0]}' is not an executable file")
    return parts


def kill_tree(proc):
    """Kill a process and everything it started."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # the child leads its own session
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def log_tail(path):
    """The last lines of a log file, bounded in lines and bytes."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - LOG_TAIL_BYTES))
            data = f.read()
    except OSError:
        return ""
    lines = data.decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-LOG_TAIL_LINES:]) + "\n"


_ROCM_VERSION_CACHE = []


def rocm_version():
    """Best-effort ROCm version: env, ``.info/version``, then ``rocm-sdk version``."""
    if _ROCM_VERSION_CACHE:
        return _ROCM_VERSION_CACHE[0]
    found = os.environ.get("ROCM_VERSION") or None
    if found is None:
        root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
        try:
            found = (root / ".info" / "version").read_text(
                encoding="utf-8"
            ).strip() or None
        except OSError:
            found = None
    if found is None and shutil.which("rocm-sdk"):
        try:
            done = subprocess.run(
                ["rocm-sdk", "version"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            found = (
                done.stdout.strip().splitlines()[0]
                if done.returncode == 0 and done.stdout.strip()
                else None
            )
        except (OSError, subprocess.TimeoutExpired):
            found = None
    _ROCM_VERSION_CACHE.append(found)
    return found


def git_sha():
    try:
        done = subprocess.run(
            ["git", "-C", str(HERE), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 and done.stdout.strip() else None


class _Raw(str):
    """A shell fragment that ``_quote`` emits unquoted."""


def _quote(token):
    return str(token) if isinstance(token, _Raw) else shlex.quote(token)


def host_input_fill(env=None):
    value = (os.environ if env is None else env).get(HOST_FILL_ENV, "")
    return bool(value) and value != "0"


# --------------------------------------------------------------------------
# Running the binary
# --------------------------------------------------------------------------


class ProcResult:
    """What one binary invocation did."""

    def __init__(self):
        self.returncode = None
        self.hung = False
        self.aborted = False
        self.records = []
        self.log_path = None

    @property
    def started(self):
        return [r["test"] for r in self.records if r.get("event") == "started"]

    @property
    def finished(self):
        return {r["test"]: r for r in self.records if r.get("event") == "finished"}


def parse_listing(text):
    """Full test names from ``--gtest_list_tests`` output."""
    names = set()
    suite = None
    for line in text.splitlines():
        if not line.strip():
            continue
        body = line.split("#", 1)[0].rstrip()
        if not body:
            continue
        if not line.startswith(" "):
            if body.endswith("."):
                suite = body
            continue
        if suite is not None:
            names.add(suite + body.strip())
    return names


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


class Fuzz:
    """One fuzz run: samples, batches, outcomes, fault artifacts, summary."""

    def __init__(self, args, spaces):
        self.args = args
        self.spaces = spaces
        self.binary_cmd = resolve_binary(args.binary)
        # Replays run from any directory, so a path to the binary is made absolute.
        self.binary_text = (
            str(Path(args.binary).resolve())
            if Path(args.binary).is_file()
            else args.binary
        )
        self.work = Path(args.work_dir).resolve()
        self.bundles = self.work / "bundles"
        self.failures = self.work / "failures"
        self.logs = self.work / "logs"
        self.reports = self.work / "reports"
        self.summary_path = (
            Path(args.summary).resolve() if args.summary else self.work / "summary.json"
        )
        self.results = []
        self.generated = {}
        self.engines_seen = {}
        self.arch_seen = None
        self.stop = threading.Event()
        self.abort = threading.Event()
        self.stop_reason = None
        self._invocation = 0
        self._started_at = None
        self._bu = None
        self._outcomes_file = None
        self._samples_file = None
        # (space id, case id) -> the values of the AXIS_OUTCOME_AXES the sample drew
        self._outcome_axes = {}

    # -- control ---------------------------------------------------------

    def request_stop(self, abort=False):
        """Stop after the current batch (or, with ``abort``, kill the binary now)."""
        self.stop.set()
        if abort:
            self.abort.set()

    def _install_signals(self):
        if threading.current_thread() is not threading.main_thread():
            return None
        previous = signal.getsignal(signal.SIGINT)

        def handler(_signum, _frame):
            if self.stop.is_set():
                print("\nSIGINT again: killing the running binary", flush=True)
                self.request_stop(abort=True)
            else:
                print(
                    "\nSIGINT: finishing the current batch, then writing the report",
                    flush=True,
                )
                self.request_stop()

        signal.signal(signal.SIGINT, handler)
        return previous

    # -- run -------------------------------------------------------------

    def run(self):
        for d in (self.work, self.logs, self.reports, self.failures):
            d.mkdir(parents=True, exist_ok=True)
        self._bu = load_bundle_utils()
        self._started_at = time.monotonic()
        self._started_wall = datetime.datetime.now(datetime.timezone.utc)
        previous = self._install_signals()
        self._outcomes_file = open(
            self.work / "outcomes.jsonl", "w", encoding="utf-8", newline="\n"
        )
        self._samples_file = open(
            self.work / "samples.jsonl", "w", encoding="utf-8", newline="\n"
        )
        error = None
        try:
            self._print_header()
            self._preflight()
            self._loop()
        except BaseException as e:  # noqa: BLE001 - report what we have, then re-raise
            error = e
            self.stop_reason = f"error: {e}"
            raise
        finally:
            self._outcomes_file.close()
            self._samples_file.close()
            if previous is not None:
                signal.signal(signal.SIGINT, previous)
            summary = self.summarize()
            if error is not None:
                summary["error"] = str(error)
            json_dump(self.summary_path, summary)
        self._print_report(summary)
        return summary["exit_code"]

    def _print_header(self):
        a = self.args
        limit = "unlimited" if not a.samples else str(a.samples)
        print(f"gauntlet fuzz: seed {a.seed} (replay with --seed {a.seed})", flush=True)
        print(f"  spaces: {', '.join(s.space_id for s in self.spaces)}", flush=True)
        print(
            f"  samples per space: {limit}; batch size {a.batch_size}; work dir {self.work}",
            flush=True,
        )
        if a.time_budget:
            print(f"  time budget: {a.time_budget:g}s", flush=True)
        if not a.samples and not a.time_budget:
            print(
                "  warning: unlimited run without a time budget; stop it with SIGINT",
                flush=True,
            )

    def _preflight(self):
        """Fail fast when the binary cannot run at all (RFC 0023 7.4).

        Lists the tests of the binary on an empty bundle root with the run's own flags and no
        filter. A healthy binary lists its compiled-in tests and exits 0, or says "Error: zero tests
        ran." and exits 1 when it has none to list (it got that far, so it is healthy); a bad
        ``--test-engine`` or ``--test-article``, a missing config or a crash at startup exits
        non-zero. Without this every batch would die the same way, be bisected down to single
        cases and be reported as ``load-crash`` with exit status 0.
        """
        empty = self.work / "preflight-bundles"
        empty.mkdir(parents=True, exist_ok=True)
        log_path, _ = self._next_invocation("preflight")
        result = self._spawn(self._command("preflight", None, bundles=empty), log_path)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if result.aborted:
            return
        # "Error: zero tests ran." (exit 1) is the binary telling us it started, loaded the
        # plugin and engine and registered nothing: an empty bundle root with no compiled-in
        # tests, or a GTEST_FILTER in the environment. Startup is proven, so it is healthy.
        started = result.returncode == 0 or (
            result.returncode == 1 and ZERO_TESTS_MARKER in text
        )
        if result.hung or not started:
            what = "hung" if result.hung else f"exited with status {result.returncode}"
            raise DriverError(
                f"the binary {what} when asked to list its tests with this run's flags, so no"
                f" sample could run (log {log_path}):\n{log_tail(log_path)}"
            )
        if NO_DEVICE_MARKER in text:
            raise DriverError(
                "the binary found no HIP device and exited 0 without running anything"
                f" (log {log_path}):\n{log_tail(log_path)}"
            )

    def _time_up(self):
        budget = self.args.time_budget
        return bool(budget) and time.monotonic() - self._started_at >= budget

    def _loop(self):
        a = self.args
        cursors = {s.space_id: 0 for s in self.spaces}
        while True:
            ran = False
            for sp in self.spaces:
                if self.stop.is_set():
                    self.stop_reason = self.stop_reason or "interrupted"
                    return
                if self._time_up():
                    self.stop_reason = "time-budget"
                    return
                done = cursors[sp.space_id]
                if a.samples and done >= a.samples:
                    continue
                size = (
                    a.batch_size
                    if not a.samples
                    else min(a.batch_size, a.samples - done)
                )
                samples = [
                    sampler.sample(sp, a.seed + done + i, a.data_seed)
                    for i in range(size)
                ]
                cursors[sp.space_id] = done + size
                self.generated[sp.space_id] = self.generated.get(sp.space_id, 0) + size
                for s in samples:
                    self._samples_file.write(
                        json.dumps(
                            {
                                "space": sp.space_id,
                                "case": s.case_id,
                                "shape_seed": s.shape_seed,
                                "data_seed": s.data_seed,
                                "axes": s.axis_values,
                            }
                        )
                        + "\n"
                    )
                self._samples_file.flush()
                before = len(self.results)
                self._execute(sp, samples)
                ran = True
                self._print_batch(sp, samples, self.results[before:])
            if not ran:
                self.stop_reason = self.stop_reason or "sample-cap"
                return

    # -- bundles ---------------------------------------------------------

    def _write_bundle(self, sp, samples):
        """One sweep directory per input dtype tag (``dtype_tag``), samples in draw order."""
        root = self.bundles / "gauntlet"
        if root.exists():
            shutil.rmtree(root)
        groups = {}
        for s in samples:
            groups.setdefault(dtype_tag(sp, s), []).append(s)
        for tag, group in groups.items():
            out = root / sp.operation / sp.name
            if tag is not None:
                out = out / dtype_directory(tag)
            out.mkdir(parents=True)
            shutil.copyfile(sp.template_path, out / "graph.template.json")
            json_dump(
                out / "sweep.json", {"version": 1, "cases": [s.row for s in group]}
            )

    def _test_name(self, sp, sample):
        """The GTest name the harness registers: the suite of the sweep directory (the
        sanitized path joined by ``_``, as in ``BundleDiscovery.hpp``) and the case id.
        """
        tag = dtype_tag(sp, sample)
        suite = (
            sp.suite_name if tag is None else f"{sp.suite_name}_{dtype_directory(tag)}"
        )
        return f"{suite}.{re.sub(r'[^A-Za-z0-9_]', '_', sample.case_id)}"

    # -- invoking the binary --------------------------------------------

    def _next_invocation(self, kind):
        self._invocation += 1
        stem = f"{self._invocation:05d}-{kind}"
        return self.logs / f"{stem}.log", self.reports / f"{stem}.jsonl"

    def _command(self, kind, names, report=None, bundles=None):
        root = self.bundles if bundles is None else bundles
        cmd = list(self.binary_cmd) + ["--golden-data-dir", str(root)]
        if report is not None:
            cmd += ["--outcome-report", str(report)]
        if names is not None:
            cmd.append("--gtest_filter=" + ":".join(names))
        if kind in ("list", "preflight"):
            cmd.append("--gtest_list_tests")
        a = self.args
        for flag, value in (
            ("--test-article", a.test_article),
            ("--test-engine", a.test_engine),
            ("--test-config", a.test_config),
        ):
            if value:
                cmd += [flag, value]
        return cmd + list(a.binary_arg)

    def _spawn(self, cmd, log_path, report_path=None):
        """Run ``cmd`` with a hang watchdog; ``report_path`` is tailed for started/finished."""
        result = ProcResult()
        result.log_path = log_path
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = (
                True  # a terminal SIGINT must not reach the binary
            )
        offset = 0
        pending = b""
        last_progress = time.monotonic()

        def drain():
            nonlocal offset, pending, last_progress
            if report_path is None:
                return
            try:
                with open(report_path, "rb") as f:
                    f.seek(offset)
                    data = f.read()
            except OSError:
                return
            offset += len(data)
            pending += data
            *lines, pending = pending.split(b"\n")
            for line in lines:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and isinstance(record.get("test"), str):
                    result.records.append(record)
                    last_progress = time.monotonic()

        with open(log_path, "wb") as log:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                **kwargs,
            )
            while True:
                rc = proc.poll()
                drain()
                if rc is not None:
                    result.returncode = rc
                    break
                if self.abort.is_set():
                    result.aborted = True
                    kill_tree(proc)
                    result.returncode = proc.returncode
                    break
                if time.monotonic() - last_progress > self.args.case_timeout:
                    result.hung = True
                    kill_tree(proc)
                    result.returncode = proc.returncode
                    break
                time.sleep(POLL_SECONDS)
        drain()
        return result

    # -- one batch -------------------------------------------------------

    def _execute(self, sp, samples):
        """Run ``samples`` of ``sp``; each ends in ``self.results`` exactly once."""
        if not samples:
            return
        if self.abort.is_set():
            for s in samples:
                self._finish(
                    sp, s, "UNEXERCISED", "setup", "interrupted before the case ran"
                )
            return
        self._write_bundle(sp, samples)
        names = [self._test_name(sp, s) for s in samples]
        log_path, _ = self._next_invocation("list")
        listing = self._spawn(self._command("list", names), log_path)
        listing_text = log_path.read_text(encoding="utf-8", errors="replace")
        if listing.aborted:
            for s in samples:
                self._finish(
                    sp, s, "UNEXERCISED", "setup", "interrupted before the case ran"
                )
            return
        # Nothing after the zero-tests diagnostic is a test name (it echoes the filter).
        listed = parse_listing(listing_text.split(ZERO_TESTS_MARKER, 1)[0])
        # A listing that selects nothing makes the real binary print "Error: zero tests
        # ran." and exit 1 (main.cpp). That is "nothing registered", answered below per case
        # with the load message; only a hang, a signal or any other non-zero exit is a crash
        # while registering, and that is bisected.
        nothing_registered = ZERO_TESTS_MARKER in listing_text and not listed
        if listing.hung or (listing.returncode != 0 and not nothing_registered):
            self._isolate(sp, samples, listing, "while listing tests")
            return
        remaining = []
        for s, name in zip(samples, names):
            if name in listed:
                remaining.append(s)
            else:
                self._finish(
                    sp,
                    s,
                    "UNEXERCISED",
                    "setup",
                    "load: " + self._load_message(listing_text, s),
                )
        while remaining:
            names = [self._test_name(sp, s) for s in remaining]
            log_path, report_path = self._next_invocation("run")
            attempt = self._spawn(
                self._command("run", names, report_path), log_path, report_path
            )
            resolved, unresolved = self._classify(sp, remaining, names, attempt)
            if self.abort.is_set() or attempt.aborted:
                for s in unresolved:
                    self._finish(
                        sp, s, "UNEXERCISED", "setup", "interrupted before the case ran"
                    )
                return
            if not unresolved:
                return
            if resolved == 0:
                if attempt.returncode == 0:
                    for s in unresolved:
                        self._finish(
                            sp, s, "UNEXERCISED", "setup",
                            "the binary exited 0 without an outcome record for this case",
                        )  # fmt: skip
                    return
                self._isolate(sp, unresolved, attempt, "before any case started")
                return
            if attempt.returncode == 0:
                for s in unresolved:
                    self._finish(
                        sp, s, "UNEXERCISED", "setup",
                        "the binary exited 0 without an outcome record for this case",
                    )  # fmt: skip
                return
            remaining = unresolved  # crash recovery: run the rest again

    def _classify(self, sp, remaining, names, attempt):
        """Record the outcome of every sample the attempt resolved; return (count, unresolved)."""
        finished = attempt.finished
        started = set(attempt.started)
        tail = None
        unresolved = []
        resolved = 0
        for s, name in zip(remaining, names):
            record = finished.get(name)
            if record is not None:
                self._finish_record(sp, s, record, attempt)
                resolved += 1
            elif name in started and not attempt.aborted:
                tail = tail if tail is not None else log_tail(attempt.log_path)
                if attempt.hung:
                    reason = (
                        f"hang: the case ran longer than the {self.args.case_timeout:g}s case"
                        " timeout; the process tree was killed"
                    )
                    kind = "hang"
                else:
                    reason = (
                        f"the binary died (exit code {attempt.returncode}) after starting the"
                        " case, before its finished record"
                    )
                    kind = "death"
                self._finish(
                    sp,
                    s,
                    "CRASH",
                    "execute",
                    reason,
                    crash_kind=kind,
                    log=tail,
                    attempt=attempt,
                )
                resolved += 1
            else:
                unresolved.append(s)
        return resolved, unresolved

    def _isolate(self, sp, samples, attempt, phase):
        """The binary died or hung with no case started: bisect to the case that does it."""
        if len(samples) == 1:
            what = "hung" if attempt.hung else f"died (exit code {attempt.returncode})"
            last = [
                ln.strip()
                for ln in log_tail(attempt.log_path).splitlines()
                if ln.strip()
            ][-3:]
            self._finish(
                sp, samples[0], "UNEXERCISED", "setup",
                f"load-crash: the binary {what} {phase} with only this case in the bundle"
                + (f"; last output: {' | '.join(last)[:300]}" if last else ""),
            )  # fmt: skip
            return
        mid = len(samples) // 2
        self._execute(sp, samples[:mid])
        self._execute(sp, samples[mid:])

    @staticmethod
    def _load_message(listing_text, sample):
        pattern = re.compile(
            r"(?<![A-Za-z0-9_])" + re.escape(sample.case_id) + r"(?![A-Za-z0-9_])"
        )
        # Only what the binary printed before its zero-tests diagnostic can be a load
        # message; the diagnostic itself echoes the whole --gtest_filter, case ids included.
        text = listing_text.split(ZERO_TESTS_MARKER, 1)[0]
        lines = [
            ln.strip()
            for ln in text.splitlines()
            if pattern.search(ln) and not GTEST_FILTER_LINE.match(ln)
        ]
        if lines:
            return " | ".join(lines[:3])[:400]
        return "absent from --gtest_list_tests (the harness skipped the case at load)"

    # -- outcomes --------------------------------------------------------

    def _finish_record(self, sp, sample, record, attempt):
        outcome = record.get("outcome")
        reason = record.get("reason") or ""
        stage = record.get("stage")
        if outcome not in OUTCOMES:
            reason = f"unrecognized outcome {outcome!r} in the outcome record. {reason}".strip()
            outcome = "UNEXERCISED"
        self._finish(sp, sample, outcome, stage, reason, record=record, attempt=attempt)

    def _finish(
        self,
        sp,
        sample,
        outcome,
        stage,
        reason,
        record=None,
        crash_kind=None,
        log=None,
        attempt=None,
    ):
        engine = (record or {}).get("engine")
        arch = (record or {}).get("arch")
        if engine:
            self.engines_seen[engine] = self.engines_seen.get(engine, 0) + 1
        if arch and self.arch_seen is None:
            self.arch_seen = arch
        result = {
            "space": sp.space_id,
            "case": sample.case_id,
            "test": self._test_name(sp, sample),
            "shape_seed": sample.shape_seed,
            "data_seed": sample.data_seed,
            "outcome": outcome,
            "stage": stage,
            "reason": reason,
            "engine": engine,
            "arch": arch,
            "source": "binary" if record is not None else "driver",
        }
        self._outcome_axes[(sp.space_id, sample.case_id)] = {
            name: sample.axis_values[name]
            for name in AXIS_OUTCOME_AXES
            if name in sample.axis_values
        }
        if record is not None:
            for key in ("purpose", "depth", "origin", "verifier", "platform"):
                if key in record:
                    result[key] = record[key]
        if crash_kind:
            result["crash_kind"] = crash_kind
        if outcome in FAILING:
            # The fault folder exists before the run goes on.
            result["failure_dir"] = str(self.failures / self._case_dir_name(sp, sample))
            result["replay"] = self._replay_text(sp, sample)
            if log is None:
                log = log_tail(attempt.log_path) if attempt is not None else ""
            self._write_failure(sp, sample, result, record, log)
            print(f"  {outcome}: {sp.space_id} {sample.case_id}: {reason}", flush=True)
            print(f"    artifacts: {result['failure_dir']}", flush=True)
            print(f"    replay: {result['replay']}", flush=True)
        self.results.append(result)
        self._outcomes_file.write(json.dumps(result) + "\n")
        self._outcomes_file.flush()

    # -- fault artifacts -------------------------------------------------

    def _run_engine(self):
        if self.args.test_engine:
            return self.args.test_engine
        if self.engines_seen:
            return max(self.engines_seen, key=self.engines_seen.get)
        return None

    @staticmethod
    def _freeze_command(graph):
        """``import_graph.py`` on a failure's graph; absolute paths, so it runs from anywhere."""
        return shlex.join(
            [
                sys.executable,
                str((MIGRATION_DIR / "import_graph.py").resolve()),
                "--graph",
                str(Path(graph).resolve()),
                "--bundle-dir",
                str((HERE.parent / "integration-test-bundles").resolve()),
                "--tier",
                "quick",
            ]
        )

    def _replay_argv(self, sp, sample, work_dir):
        a = self.args
        argv = [
            sys.executable, str(DRIVER_PATH),
            "--filter", f"^{sp.operation}/{sp.name}$",
            "--seed", str(sample.shape_seed),
            "--samples", "1",
            "--data-seed", str(sample.data_seed),
            "--time-budget", "0",
            "--case-timeout", f"{a.case_timeout:g}s",
            "--binary", self.binary_text,
            "--work-dir", str(work_dir),
        ]  # fmt: skip
        for flag, value in (
            ("--test-article", a.test_article),
            ("--test-engine", a.test_engine),
            ("--test-config", a.test_config),
        ):
            if value:
                argv += [flag, value]
        if Path(a.spaces_dir).resolve() != space_lib.DEFAULT_SPACES_DIR.resolve():
            argv += ["--spaces-dir", str(Path(a.spaces_dir).resolve())]
        for extra in a.binary_arg:
            argv.append(f"--binary-arg={extra}")
        return argv

    def _replay_env(self):
        env_set = {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith("HIPDNN_") and not k.startswith("HIPDNN_FUZZ_")
        }
        env_unset = [] if HOST_FILL_ENV in env_set else [HOST_FILL_ENV]
        return env_set, env_unset

    def _replay_command(self, argv, env_set, env_unset):
        prefix = []
        if env_set or env_unset:
            prefix = ["env"]
            for name in env_unset:
                prefix += ["-u", name]
            prefix += [f"{k}={v}" for k, v in env_set.items()]
        return " ".join(_quote(t) for t in prefix + argv)

    def _replay_text(self, sp, sample):
        env_set, env_unset = self._replay_env()
        folder = self.failures / self._case_dir_name(sp, sample)
        return self._replay_command(
            self._replay_argv(sp, sample, folder / "replay-work"), env_set, env_unset
        )

    @staticmethod
    def _case_dir_name(sp, sample):
        return f"{sp.operation}_{sp.name}_{sample.case_id}"

    def _write_failure(self, sp, sample, result, record, tail):
        """``<work>/failures/<case>/``: everything needed to replay or freeze a finding."""
        bu = self._bu
        case = self._case_dir_name(sp, sample)
        folder = self.failures / case
        folder.mkdir(parents=True, exist_ok=True)

        template = json.loads(sp.template_path.read_bytes())
        expanded = bu.expand(template, sample.row["values"])
        load_support_matrix_units().apply_tensor_patches(expanded, sample.row)
        uid_map = bu.canonical_uid_map_by_name(expanded)
        graph = bu.remap_graph(expanded, uid_map)
        json_dump(folder / f"{case}.json", graph)

        meta = dict(sample.row["metadata"])
        meta["purpose"] = "regression"
        meta["reference_source"] = (
            f"gauntlet-fuzz: space {sp.space_id}, shape seed {sample.shape_seed}"
        )
        meta["notes"] = (
            f"{result['outcome']} found by gauntlet-fuzz. {sample.row['metadata'].get('notes', '')}".strip()
        )
        meta = bu.remap_meta_inputs(meta, uid_map)
        json_dump(folder / f"{case}.meta.json", meta)

        host_fill = host_input_fill()
        env_set, env_unset = self._replay_env()
        argv = self._replay_argv(sp, sample, folder / "replay-work")
        command = self._replay_command(argv, env_set, env_unset)
        engine = result["engine"] or self._run_engine()
        a = self.args
        repro = {
            "tool": "run_gauntlet_fuzz.py",
            "space": sp.space_id,
            "space_file": str(sp.path),
            "space_sha256": sp.space_sha256,
            "template_file": str(sp.template_path),
            "template_sha256": sp.template_sha256,
            "case_id": sample.case_id,
            "gtest_name": result["test"],
            "shape_seed": sample.shape_seed,
            "data_seed": sample.data_seed,
            "axis_values": sample.axis_values,
            "sweep_row": sample.row,
            "uids_canonical": all(k == v for k, v in uid_map.items()),
            "input_fill": {
                "data_seed": sample.data_seed,
                "recipes": sample.row["metadata"].get("inputs"),
                "path": "host" if host_fill else "device",
                HOST_FILL_ENV: os.environ.get(HOST_FILL_ENV),
            },
            "engine": engine,
            "test_article": a.test_article,
            "test_config": a.test_config,
            "arch": result["arch"] or self.arch_seen,
            "rocm_version": rocm_version(),
            "git_sha": git_sha(),
            "binary": a.binary,
            "driver_flags": {
                k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()
            },
            "driver_argv": list(sys.argv),
            "hipdnn_env": {
                k: v for k, v in sorted(os.environ.items()) if k.startswith("HIPDNN_")
            },
            "outcome": result["outcome"],
            "replay": {
                "argv": argv,
                "env_set": env_set,
                "env_unset": env_unset,
                "command": command,
            },
        }
        json_dump(folder / "repro.json", repro)
        json_dump(folder / "outcome.json", {"result": result, "record": record})
        write_text(folder / "log_tail.txt", tail)

        script_argv = [
            _Raw('"$D/replay-work"') if t == str(folder / "replay-work") else t
            for t in argv
        ]
        script_command = self._replay_command(script_argv, env_set, env_unset)
        script = (
            "#!/bin/sh\n"
            f"# Replay {case}: {result['outcome']} on {engine or 'unknown engine'}.\n"
            f"# Regenerates the same sweep row (shape seed {sample.shape_seed}, data seed"
            f" {sample.data_seed}) and runs it alone.\n"
            f"# Input fill path: {'host' if host_fill else 'device'} ({HOST_FILL_ENV}).\n"
            'D=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
            f"exec {script_command}\n"
        )
        write_text(folder / "replay.sh", script)
        try:
            os.chmod(folder / "replay.sh", 0o755)
        except OSError:
            pass
        write_text(
            folder / "replay.txt",
            f"{command}\n"
            f"input fill path: {'host' if host_fill else 'device'} ({HOST_FILL_ENV}="
            f"{os.environ.get(HOST_FILL_ENV)!r})\n"
            f"freeze: {self._freeze_command(folder / (case + '.json'))}\n",
        )
        return folder

    # -- summary ---------------------------------------------------------

    def _print_batch(self, sp, samples, results):
        counts = {}
        for r in results:
            counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
        text = ", ".join(f"{k} {counts[k]}" for k in OUTCOMES if k in counts)
        lo, hi = samples[0].shape_seed, samples[-1].shape_seed
        print(f"batch {sp.space_id} seeds {lo}..{hi}: {text}", flush=True)

    def _reason_key(self, reason):
        """A reason with the work directory, dtype directories and case ids masked, so equal
        causes group."""
        text = (reason or "").replace(str(self.work), "<work>")
        text = re.sub(
            r"(?<![A-Za-z0-9])[a-z0-9]+_cases(?![A-Za-z0-9_])", "<dtype>_cases", text
        )
        return re.sub(r"(?<![A-Za-z0-9_])s\d+(?![A-Za-z0-9_])", "<case>", text)[:300]

    def summarize(self):
        run_engine = self._run_engine() or "unknown"
        engines = {}
        totals = dict.fromkeys(OUTCOMES, 0)
        for r in self.results:
            engine = r["engine"] or run_engine
            e = engines.setdefault(
                engine,
                {"samples": 0, "outcomes": dict.fromkeys(OUTCOMES, 0), "spaces": {}},
            )
            sp = e["spaces"].setdefault(
                r["space"],
                {
                    "samples": 0,
                    "outcomes": dict.fromkeys(OUTCOMES, 0),
                    "declined_reasons": {},
                    "unexercised_reasons": {},
                    "crash_reasons": {},
                    "wrong": [],
                    "crash": [],
                },
            )
            for bucket in (e, sp):
                bucket["samples"] += 1
                bucket["outcomes"][r["outcome"]] += 1
            totals[r["outcome"]] += 1
            reason = self._reason_key(r["reason"])
            if r["outcome"] == "DECLINED":
                sp["declined_reasons"][reason] = (
                    sp["declined_reasons"].get(reason, 0) + 1
                )
            elif r["outcome"] == "UNEXERCISED":
                sp["unexercised_reasons"][reason] = (
                    sp["unexercised_reasons"].get(reason, 0) + 1
                )
            elif r["outcome"] == "CRASH":
                sp["crash_reasons"][reason] = sp["crash_reasons"].get(reason, 0) + 1
                sp["crash"].append(r["case"])
            elif r["outcome"] == "WRONG":
                sp["wrong"].append(r["case"])
        axis_outcomes = {}
        for r in self.results:
            axes = self._outcome_axes.get((r["space"], r["case"]), {})
            for name, value in axes.items():
                counts = (
                    axis_outcomes.setdefault(r["space"], {})
                    .setdefault(name, {})
                    .setdefault(str(value), dict.fromkeys(OUTCOMES, 0))
                )
                counts[r["outcome"]] += 1
        per_space_results = {}
        for r in self.results:
            per_space_results[r["space"]] = per_space_results.get(r["space"], 0) + 1
        reconciled = per_space_results == {
            k: v for k, v in self.generated.items() if v
        } and sum(totals.values()) == sum(self.generated.values())
        findings = []
        for r in self.results:
            if r["outcome"] in FAILING:
                findings.append(
                    {
                        "space": r["space"],
                        "case": r["case"],
                        "outcome": r["outcome"],
                        "reason": r["reason"],
                        "failure_dir": r.get("failure_dir"),
                        "replay": r.get("replay"),
                    }
                )
        a = self.args
        failed = bool(totals["WRONG"] or totals["CRASH"])
        return {
            "seed": a.seed,
            "data_seed": a.data_seed,
            "samples_per_space": a.samples or None,
            "continuous": not a.samples,
            "time_budget_seconds": a.time_budget,
            "stopped_because": self.stop_reason,
            "elapsed_seconds": (
                round(time.monotonic() - self._started_at, 3) if self._started_at else 0
            ),
            "started_at": self._started_wall.isoformat() if self._started_at else None,
            "binary": a.binary,
            "test_article": a.test_article,
            "test_engine": a.test_engine,
            "test_config": a.test_config,
            "arch": self.arch_seen,
            "engine": run_engine,
            "spaces": [s.space_id for s in self.spaces],
            "samples": sum(self.generated.values()),
            "samples_by_space": dict(self.generated),
            "outcomes": totals,
            "reconciled": reconciled,
            "axis_outcomes": axis_outcomes,
            "engines": engines,
            "findings": findings,
            "work_dir": str(self.work),
            "exit_code": 1 if failed else 0,
        }

    def _print_report(self, summary):
        print("", flush=True)
        print(
            f"gauntlet fuzz report (seed {summary['seed']}, stopped: {summary['stopped_because']})"
        )
        for engine, e in summary["engines"].items():
            for space_id, sp in e["spaces"].items():
                o = sp["outcomes"]
                print(
                    f"  {engine} {space_id}: {o['PASS'] + o['UNCLAIMED_PASS']} of {sp['samples']}"
                    f" passed; {o['DECLINED']} declined; {o['WRONG']} wrong; {o['CRASH']} crashed;"
                    f" {o['UNEXERCISED']} unexercised"
                )
                for title, key in (
                    ("declined", "declined_reasons"),
                    ("unexercised", "unexercised_reasons"),
                    ("crash", "crash_reasons"),
                ):
                    for reason, n in sorted(sp[key].items(), key=lambda kv: -kv[1]):
                        print(f"      {title} x{n}: {reason}")
        if not summary["reconciled"]:
            print("  WARNING: outcomes do not add up to samples")
        shown = summary["findings"][:MAX_PRINTED_FINDINGS]
        for f in shown:
            print(f"  {f['outcome']} {f['space']} {f['case']}")
            print(f"    replay: {f['replay']}")
        if len(summary["findings"]) > len(shown):
            print(
                f"  ... {len(summary['findings']) - len(shown)} more findings in the summary"
            )
        print(f"summary: {self.summary_path}")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _env_default(name, convert, parser):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return convert(raw)
    except ValueError as e:
        parser.error(f"{name}={raw!r}: {e}")


def build_parser():
    p = argparse.ArgumentParser(
        prog="run_gauntlet_fuzz.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--binary",
        required=True,
        help="hipdnn_integration_tests (path, .py script or quoted command)",
    )
    p.add_argument("--test-article", help="passed to the binary as --test-article")
    p.add_argument("--test-engine", help="passed to the binary as --test-engine")
    p.add_argument("--test-config", help="passed to the binary as --test-config")
    p.add_argument(
        "--binary-arg",
        action="append",
        default=[],
        help="extra argument for the binary (repeatable)",
    )
    p.add_argument(
        "--work-dir",
        default="gauntlet-fuzz-work",
        help="bundles, logs, reports, failures, summary",
    )
    p.add_argument(
        "--spaces-dir",
        default=str(space_lib.DEFAULT_SPACES_DIR),
        help="directory of *.space.json",
    )
    p.add_argument(
        "--samples",
        type=int,
        help="samples per space (env HIPDNN_FUZZ_SAMPLES; 0 = unlimited)",
    )
    p.add_argument(
        "--seed", type=int, help="shape seed of the first sample (env HIPDNN_FUZZ_SEED)"
    )
    p.add_argument(
        "--data-seed",
        type=int,
        help="input-data seed for every sample (env HIPDNN_FUZZ_DATA_SEED)",
    )
    p.add_argument("--filter", help="regex on <Op>/<Space> (env HIPDNN_FUZZ_FILTER)")
    p.add_argument(
        "--time-budget",
        help="wall-clock limit, e.g. 30m, 4h (env HIPDNN_FUZZ_TIME_BUDGET)",
    )
    p.add_argument(
        "--case-timeout",
        help=f"hang limit per case (env HIPDNN_FUZZ_CASE_TIMEOUT; default {DEFAULT_CASE_TIMEOUT})",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="samples per binary invocation",
    )
    p.add_argument(
        "--summary", help="summary.json path (default <work-dir>/summary.json)"
    )
    return p


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    def pick(flag_value, env_name, convert):
        return (
            flag_value
            if flag_value is not None
            else _env_default(env_name, convert, parser)
        )

    args.samples = pick(args.samples, "HIPDNN_FUZZ_SAMPLES", int)
    args.seed = pick(args.seed, "HIPDNN_FUZZ_SEED", int)
    args.data_seed = pick(args.data_seed, "HIPDNN_FUZZ_DATA_SEED", int)
    args.filter = pick(args.filter, "HIPDNN_FUZZ_FILTER", str)
    try:
        budget = (
            args.time_budget
            if args.time_budget is not None
            else os.environ.get("HIPDNN_FUZZ_TIME_BUDGET")
        )
        args.time_budget = parse_duration(budget) if budget else None
        timeout = (
            args.case_timeout
            if args.case_timeout is not None
            else os.environ.get("HIPDNN_FUZZ_CASE_TIMEOUT")
        )
        args.case_timeout = parse_duration(timeout or DEFAULT_CASE_TIMEOUT)
    except ValueError as e:
        parser.error(str(e))
    if args.case_timeout is None:
        parser.error("--case-timeout must be positive")
    if args.samples is None:
        args.samples = 0 if args.time_budget else DEFAULT_SAMPLES
    if args.samples < 0:
        parser.error("--samples must be >= 0")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    try:
        if args.seed is None:
            args.seed = random.SystemRandom().randrange(0, 2**31)
        sampler.check_shape_seed(args.seed)
        if args.data_seed is not None:
            sampler.check_data_seed(args.data_seed)
    except ValueError as e:
        parser.error(str(e))
    return args


def select_spaces(args):
    spaces = space_lib.load_spaces(args.spaces_dir)
    if args.filter:
        try:
            pattern = re.compile(args.filter)
        except re.error as e:
            raise DriverError(
                f"--filter {args.filter!r} is not a valid regex: {e}"
            ) from None
        spaces = [s for s in spaces if pattern.search(s.space_id)]
    if not spaces:
        raise DriverError(
            f"no space in {args.spaces_dir} matches"
            + (f" --filter {args.filter!r}" if args.filter else "")
        )
    return spaces


def main(argv=None):
    args = parse_args(argv)
    try:
        spaces = select_spaces(args)
        return Fuzz(args, spaces).run()
    except (DriverError, space_lib.SpaceError, sampler.InfeasibleSpaceError) as e:
        print(f"run_gauntlet_fuzz.py: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
