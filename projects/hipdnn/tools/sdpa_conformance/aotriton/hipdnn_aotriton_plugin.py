# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""pytest plugin that records what hipDNN did for every case of the AOTriton port.

Load it with ``-p hipdnn_aotriton_plugin`` (``aotriton/`` on PYTHONPATH). It

* refuses the vendored suite's calibration and backend-pinning knobs at configure time;
* keeps a process-global *mode record*: wrappers installed around ``hipdnn_sdpa.sdpa_fwd`` /
  ``sdpa_bwd`` append ``hipdnn:<engine>`` or ``declined:<reason>`` per call, and the shim appends
  ``ref:<reason>`` when it falls back to the reference forward, all from any thread; at report
  time the entries fold into the case's forward and backward state;
* applies the outcome table: a case ends as ``pass``, ``fail``, ``crash``, ``not-run``,
  ``declined``, ``native-skip`` or ``no-hipdnn`` (the test passed but hipDNN computed nothing),
  and any result that rests on reference tensors alone becomes a skip;
* writes one CSV row per case, in the controller process only, flushed per row;
* supports ``--shard i/N`` (1-based) and ``--fake always-decline``.
"""

import json
import os
import threading
from pathlib import Path

import pytest

import aotriton_layout as layout
import reconcile

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
ROW_KEY = "sdpa_conformance_row"
_BOOL_KNOBS = {
    "SKIP_BWD",
    "SMALL_HEADDIM_ONLY",
    "LARGE_HEADDIM_ONLY",
    "AOTRITON_TORCH_ONLY_USE_CPU",
    "PROBE_UNSUPPORTED",
    "IGNORE_BACKWARD_IMPORT",
}
_FORWARDED_KNOBS = (
    "FOR_RELEASE",
    "SMALL_VRAM",
    "AOTRITON_REF_DEVICE_OPTION",
    "HIPDNN_FWD_ENGINE",
    "HIPDNN_BWD_ENGINE",
)
_REASON_LIMIT = 400

# ---------------------------------------------------------------------------- mode record

_LOCK = threading.Lock()
_RECORDS = {}
_CURRENT = None


def record(direction, kind, detail=""):
    """Append one hipDNN call outcome for the running case; thread-safe.

    ``direction`` is ``'fwd'`` or ``'bwd'``; ``kind`` is ``'hipdnn'`` (``detail`` the engine),
    ``'ref'`` (the reference forward ran; ``detail`` the decline reason) or ``'declined'``. A
    ``'ref'`` entry replaces the ``'declined'`` entry of the call it falls back from.
    """
    with _LOCK:
        entries = _RECORDS.setdefault(_CURRENT, [])
        if kind == "ref":
            for i in range(len(entries) - 1, -1, -1):
                if entries[i][:2] == (direction, "declined"):
                    del entries[i]
                    break
        entries.append((direction, kind, str(detail)))


def _fold(entries):
    """Per direction: ``None`` (never called) or ``(kind, detail)``.

    A direction is hipDNN only if every call ran hipDNN; ``ref`` if any call ran the reference;
    ``declined`` if any call declined (declined wins).
    """
    folded = {}
    for direction in ("fwd", "bwd"):
        calls = [(kind, detail) for d, kind, detail in entries if d == direction]
        if not calls:
            folded[direction] = None
            continue
        for kind in ("declined", "ref"):
            hit = next((c for c in calls if c[0] == kind), None)
            if hit:
                folded[direction] = hit
                break
        else:
            folded[direction] = ("hipdnn", "+".join(sorted({d for _, d in calls})))
    return folded


def _state_text(state):
    return "" if state is None else f"{state[0]}:{state[1]}"


def _declined_exception(excinfo):
    """The ``hipdnn_sdpa.EngineDeclined`` behind ``excinfo`` (through ``__cause__``), if any."""
    import sys

    module = sys.modules.get("hipdnn_sdpa")
    declined = getattr(module, "EngineDeclined", None)
    if declined is None or excinfo is None:
        return None
    exc, seen = excinfo.value, set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, declined):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


def _one_line(text):
    return " ".join(str(text).split())[:_REASON_LIMIT]


def _skip_location_and_reason(report):
    longrepr = report.longrepr
    if isinstance(longrepr, tuple):
        path, _, message = longrepr
        return Path(path), str(message).removeprefix("Skipped: ")
    return None, str(longrepr)


def _is_native_skip(path, config):
    if path is None:
        return False
    if not path.is_absolute():
        path = Path(config.rootpath) / path
    try:
        path.resolve().relative_to(Path(VENDOR_DIR).resolve())
    except ValueError:
        return False
    return True


def _make_row(item, call, report):
    """Classify one setup/call report; rewrites ``report`` where the outcome table says so."""
    nodeid = item.nodeid
    with _LOCK:
        entries = list(_RECORDS.get(nodeid, ()))
    exc = _declined_exception(call.excinfo)
    if exc is not None and not any(
        d == exc.direction and k == "declined" for d, k, _ in entries
    ):
        entries.append((exc.direction, "declined", exc.reason))
    folded = _fold(entries)
    fwd, bwd = folded["fwd"], folded["bwd"]
    row = {
        "nodeid": nodeid,
        "outcome": "",
        "mode": "",
        "fwd": _state_text(fwd),
        "bwd": _state_text(bwd),
        "reason": "",
    }

    declined = []
    if fwd and fwd[0] == "declined":
        declined.append(f"fwd {fwd[1]}")
    elif fwd and fwd[0] == "ref" and not (bwd and bwd[0] == "hipdnn"):
        declined.append(f"fwd {fwd[1]}")
    if bwd and bwd[0] == "declined":
        declined.append(f"bwd {bwd[1]}")

    def rewrite(outcome, reason):
        """Turn the report into a skip (``declined``) or a failure, and record the row."""
        row.update(outcome=outcome, reason=_one_line(reason), mode="")
        if outcome == "declined":
            report.outcome = "skipped"
            report.longrepr = (
                str(item.path),
                item.reportinfo()[1] or 0,
                f"Skipped: declined: {reason}",
            )
        else:
            report.outcome = "failed"
            report.longrepr = reason

    if hasattr(report, "wasxfail"):
        del report.wasxfail  # pytest only counts a failure that is not an expected one
        rewrite("fail", "xfail/xpass is not an AOTriton outcome the port records")
    elif declined:
        rewrite("declined", "; ".join(declined))
    elif report.skipped:
        path, reason = _skip_location_and_reason(report)
        if reason.startswith("declined:"):
            row.update(
                outcome="declined",
                reason=_one_line(reason.removeprefix("declined:").strip()),
            )
        elif _is_native_skip(path, item.config):
            row.update(outcome="native-skip", reason=_one_line(reason))
        else:
            rewrite("fail", f"unmarked skip outside vendor/: {reason}")
    else:
        row["outcome"] = "pass" if report.passed else "fail"
        if fwd and fwd[0] == "ref":
            row["mode"] = "ref-fwd"
        elif (fwd and fwd[0] == "hipdnn") or (bwd and bwd[0] == "hipdnn"):
            row["mode"] = "hipdnn"
        elif report.passed:
            row["outcome"] = (
                "no-hipdnn"  # the result stands, but hipDNN computed nothing
            )
        if report.failed:
            row["reason"] = (
                _one_line(call.excinfo.exconly())
                if call.excinfo
                else _one_line(report.longrepr)
            )
    return row


# ---------------------------------------------------------------------------- hooks


def pytest_addoption(parser):
    group = parser.getgroup("sdpa-conformance")
    group.addoption(
        "--results-dir", metavar="DIR", help="write DIR/results.csv (controller only)"
    )
    group.addoption(
        "--shard", metavar="I/N", help="run every N-th node id (sorted), 1 <= I <= N"
    )
    group.addoption(
        "--fake",
        choices=["always-decline"],
        help="replace hipdnn_sdpa.sdpa_fwd/sdpa_bwd with functions that decline",
    )


def _parse_shard(text):
    try:
        i, n = (int(part) for part in text.split("/"))
        assert 1 <= i <= n
    except (ValueError, AssertionError):
        raise pytest.UsageError(
            f"--shard expects I/N with 1 <= I <= N, got {text!r}"
        ) from None
    return i, n


def _knob_active(name):
    value = os.environ.get(name)
    if value is None or value == "":
        return False
    if name in _BOOL_KNOBS:
        try:
            return bool(int(value))
        except ValueError:
            return True
    return True


def _install_wrappers(fake):
    """Wrap ``hipdnn_sdpa.sdpa_fwd`` / ``sdpa_bwd`` so every call lands in the mode record.

    With ``fake`` the declining stand-ins become the inner functions. Re-installing starts from
    the unwrapped originals, so a module shared by several sessions is never wrapped twice.
    """
    try:
        import hipdnn_sdpa
    except ImportError:
        return

    declined_type = hipdnn_sdpa.EngineDeclined

    def make(direction, inner):
        def wrapper(*args, **kwargs):
            try:
                engine = inner(*args, **kwargs)
            except declined_type as declined:
                record(direction, "declined", declined.reason)
                raise
            record(direction, "hipdnn", engine)
            return engine

        wrapper._sdpa_conformance_inner = inner
        return wrapper

    for direction in ("fwd", "bwd"):
        name = f"sdpa_{direction}"
        inner = getattr(hipdnn_sdpa, name)
        inner = getattr(inner, "_sdpa_conformance_inner", inner)
        if fake:

            def inner(*args, _direction=direction, **kwargs):
                raise declined_type(_direction, "always-decline fake")

        setattr(hipdnn_sdpa, name, make(direction, inner))


class _CsvController:
    """Writes the CSV from reports the controller receives (rows and xdist crash reports)."""

    def __init__(self, path):
        self.seen = set()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a")
        if self.path.stat().st_size == 0:
            self.file.write(",".join(reconcile.FIELDS) + "\n")
            self.file.flush()

    def pytest_runtest_logreport(self, report):
        rows = [json.loads(v) for k, v in report.user_properties if k == ROW_KEY]
        if rows:
            row = rows[0]
        elif report.when == "???":
            row = {
                "nodeid": report.nodeid,
                "outcome": "crash",
                "reason": _one_line(str(report.longrepr)),
            }
        else:
            return
        if row["nodeid"] in self.seen:
            return
        self.seen.add(row["nodeid"])
        self.file.write(reconcile.format_row(row))
        self.file.flush()

    def close(self):
        self.file.close()


def pytest_configure(config):
    refused = [k for k in layout.REFUSED_KNOBS if _knob_active(k)]
    if refused:
        raise pytest.UsageError(
            "the AOTriton port refuses these environment variables (they change what the "
            f"vendored suite measures): {', '.join(refused)}"
        )
    if config.getoption("shard"):
        _parse_shard(config.getoption("shard"))
    _install_wrappers(config.getoption("fake") == "always-decline")
    results = config.getoption("results_dir")
    if results and not hasattr(config, "workerinput"):
        config._sdpa_csv = _CsvController(Path(results) / reconcile.CSV_NAME)
        config.pluginmanager.register(config._sdpa_csv, "sdpa-conformance-csv")


def pytest_unconfigure(config):
    csv_controller = getattr(config, "_sdpa_csv", None)
    if csv_controller is not None:
        config.pluginmanager.unregister(csv_controller)
        csv_controller.close()


def pytest_report_header(config):
    forwarded = [f"{k}={os.environ[k]}" for k in _FORWARDED_KNOBS if k in os.environ]
    return "sdpa-conformance: " + (", ".join(forwarded) or "no forwarded knobs")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    spec = config.getoption("shard")
    if not spec:
        return
    index, count = _parse_shard(spec)
    chosen = set(sorted(item.nodeid for item in items)[index - 1 :: count])
    kept = [item for item in items if item.nodeid in chosen]
    dropped = [item for item in items if item.nodeid not in chosen]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = kept


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    global _CURRENT
    with _LOCK:
        _CURRENT = item.nodeid
        _RECORDS[item.nodeid] = []


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    global _CURRENT
    if call.when == "teardown":
        with _LOCK:
            _RECORDS.pop(item.nodeid, None)
            _CURRENT = None
        return report
    if call.when == "setup" and report.passed:
        return report
    row = _make_row(item, call, report)
    report.user_properties.append((ROW_KEY, json.dumps(row)))
    return report


@pytest.fixture(autouse=True)
def _hipdnn_cuda_device(request):
    """Bind the worker's leased GPU as torch's current device (a no-op without CUDA)."""
    try:
        import torch
    except ImportError:
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(request.getfixturevalue("gpu_id"))
    yield
