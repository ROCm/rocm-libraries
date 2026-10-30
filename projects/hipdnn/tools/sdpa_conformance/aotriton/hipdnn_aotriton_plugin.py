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
* writes one CSV row per case, in the controller process only, flushed per row; ``fwd_engines`` /
  ``bwd_engines`` hold the applicable engines (``;``-joined, heuristic rank order) of a direction
  hipDNN planned unpinned, empty otherwise;
* fills ``fwd_kernel`` / ``bwd_kernel`` (the kernel id ``hipdnn_sdpa.last_call`` reports; kind
  ``kernel`` entries, ignored by the fold), ``fail_side`` and ``stage`` of a ``fail`` row, and
  ``repro`` once ``hipdnn_sdpa.save_repro`` wrote the case's repro directory (``--repro-dir``,
  capped per engine and test function and per shard);
* turns a worker death into a ``crash`` row naming the engine, direction and stage left in the
  worker's in-flight marker (``conformance_contract``): ``fwd``/``bwd`` hold ``hipdnn:<engine>``
  for the direction in flight and ``fail_side`` names it; the marker is deleted once read, and
  removed by a case's teardown otherwise;
* supports ``--shard i/N`` (1-based), ``--hipdnn-nodeids-file`` and ``--fake always-decline``.
"""

import json
import os
import re
import threading
from pathlib import Path

import pytest

import aotriton_layout as layout
import conformance_contract as contract
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
    "HIPDNN_SDPA_CONFORMANCE_FWD_NO_STATS",
)
_REASON_LIMIT = 400

# ---------------------------------------------------------------------------- mode record

_LOCK = threading.Lock()
_RECORDS = {}
_CURRENT = None


def record(direction, kind, detail=""):
    """Append one hipDNN call outcome for the running case; thread-safe.

    ``direction`` is ``'fwd'`` or ``'bwd'``; ``kind`` is ``'hipdnn'`` (``detail`` the engine),
    ``'ref'`` (the reference forward ran; ``detail`` the decline reason), ``'declined'`` or
    ``'engines'`` (``detail`` the ``;``-joined ranked engine names of the preceding ``'hipdnn'``
    call) or ``'kernel'`` (the kernel id of that call; ignored by the fold). A ``'ref'`` entry replaces the ``'declined'`` entry of the call it falls back from.
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
        calls = [
            (kind, detail)
            for d, kind, detail in entries
            if d == direction and kind not in ("engines", "kernel")
        ]
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


def _engine_list(entries, folded, direction):
    """``;``-joined ranked engines of ``direction``'s first call that has a list; '' unless hipDNN served it."""
    state = folded[direction]
    if state is None or state[0] != "hipdnn":
        return ""
    return next(
        (e for d, k, e in entries if (d, k) == (direction, "engines") and e), ""
    )


def _kernel_text(entries, direction):
    """``+``-joined distinct kernel ids recorded for ``direction`` (kind ``kernel``)."""
    return "+".join(
        sorted({d for dr, k, d in entries if (dr, k) == (direction, "kernel") and d})
    )


def _names(value):
    """A marker's engine list (a list, or already ``;``-joined) as ``;``-joined text."""
    return value if isinstance(value, str) else ";".join(value or ())


_FORWARD_COMPARISON = re.compile(r"\bForward pass\b")


def _fail_side(nodeid, excinfo, folded):
    """``fwd`` when the failing assertion is the vendored forward comparison, the case is a forward
    case, or the backward never ran; else ``bwd``."""
    if reconcile.direction_of(nodeid) == "fwd" or folded["bwd"] is None:
        return "fwd"
    text = excinfo.exconly() if excinfo else ""
    return "fwd" if _FORWARD_COMPARISON.search(text) else "bwd"


def _inflight_marker_path(worker):
    """The worker's in-flight marker of the current pass; None unless a directory is configured."""
    root = os.environ.get(contract.INFLIGHT_DIR_ENV)
    if not root:
        return None
    slug = os.environ.get(contract.PASS_SLUG_ENV, reconcile.MAIN_PASS)
    return Path(root) / slug / contract.marker_name(worker)


def _read_marker(path, nodeid):
    """The marker's fields when ``path`` holds one written for ``nodeid``, else None."""
    try:
        marker = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return marker if marker.get("nodeid") == nodeid else None


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
        "fwd_engines": _engine_list(entries, folded, "fwd"),
        "bwd_engines": _engine_list(entries, folded, "bwd"),
        "fwd_kernel": _kernel_text(entries, "fwd"),
        "bwd_kernel": _kernel_text(entries, "bwd"),
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
    if row["outcome"] == "fail":
        _describe_failure(item, call, row, folded)
    return row


def _describe_failure(item, call, row, folded):
    """Fill ``fail_side``, ``stage`` and ``repro`` of a ``fail`` row."""
    row["fail_side"] = _fail_side(item.nodeid, call.excinfo, folded)
    path = _inflight_marker_path(os.environ.get("PYTEST_XDIST_WORKER"))
    marker = _read_marker(path, item.nodeid) if path else None
    row["stage"] = marker["stage"] if marker else ""
    row["repro"] = _save_repro(item, row)


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
        "--hipdnn-nodeids-file",
        metavar="PATH",
        help="run only the collected items whose node id is listed in PATH (one per line)",
    )
    group.addoption(
        "--repro-dir",
        metavar="DIR",
        help="save a repro of every failing case under DIR (the runner passes <out>/repro)",
    )
    group.addoption(
        "--fake",
        choices=["always-decline"],
        help="replace hipdnn_sdpa.sdpa_fwd/sdpa_bwd with functions that decline",
    )
    group.addoption(
        "--hipdnn-prefix",
        metavar="DIR",
        help=(
            "hipDNN install prefix whose libhipdnn_backend must be the one mapped into the "
            "process (the runner passes $ROCM_PATH); unset leaves the backend unchecked"
        ),
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


def _mapped_backends(maps_path="/proc/self/maps"):
    """Paths of every ``libhipdnn_backend`` mapped into this process."""
    with open(maps_path) as maps:
        return sorted(
            {line.split()[-1] for line in maps if "libhipdnn_backend" in line}
        )


def backend_problem(mapped, prefix):
    """Why the loaded hipDNN backend is not ``prefix``'s, or None when it is.

    ``mapped`` lists the mapped ``libhipdnn_backend`` paths; ``prefix`` is where
    dnn-benchmarking's ``setup_env.py`` installed hipDNN (``ROCM_PATH``).
    """
    if len(mapped) != 1:
        return f"expected one mapped libhipdnn_backend, found {len(mapped)}: {mapped}"
    root = os.path.realpath(prefix) + os.sep
    if not os.path.realpath(mapped[0]).startswith(root):
        return f"mapped {mapped[0]}, which is not under {prefix}"
    return None


def _bind_backend_first(prefix):
    """Load hipDNN before any test module imports torch, then verify which backend bound.

    dnn-benchmarking's recipe (``docs/troubleshooting.md``, "Mixed ROCm torch and hipDNN
    prefixes"): the ROCm devel wheel ships an older ``libhipdnn_backend.so`` beside the one
    ``setup_env.py`` builds, ``$ROCM_PATH/lib`` comes first on ``LD_LIBRARY_PATH``, importing
    ``hipdnn_frontend`` first gives it the first bind, and ``/proc/self/maps`` is the proof.
    Returns the mapped backend path; None when no prefix is given (CPU-only test runs).
    """
    if not prefix:
        return None
    try:
        import hipdnn_frontend  # noqa: F401
    except ImportError as exc:
        raise pytest.UsageError(
            f"--hipdnn-prefix given but hipdnn_frontend: {exc}"
        ) from None
    mapped = _mapped_backends()
    problem = backend_problem(mapped, prefix)
    if problem:
        raise pytest.UsageError(f"hipDNN backend check failed: {problem}")
    return mapped[0]


_CALLS = {}  # nodeid -> {"records": {direction: CallRecord}, "tensors": {role: tensor}}
_POSITIONAL_ROLES = {"fwd": ("q", "k", "v"), "bwd": ("q", "k", "v", "o", "do", "stats")}


def _remember_call(direction, call, args, kwargs):
    """Keep what a repro of the running case needs: its calls' records and tensors by role."""
    import torch

    roles = dict(zip(_POSITIONAL_ROLES[direction], args))
    roles.update(kwargs)
    dropout, varlen = roles.pop("dropout", None), roles.pop("varlen", None)
    for owner, names in (
        (dropout, ("seed", "offset")),
        (varlen, ("seq_len_q", "seq_len_kv", "pos_q", "pos_kv")),
    ):
        roles.update({n: getattr(owner, n, None) for n in names} if owner else {})
    tensors = {r: t for r, t in roles.items() if isinstance(t, torch.Tensor)}
    with _LOCK:
        saved = _CALLS.setdefault(_CURRENT, {"records": {}, "tensors": {}})
        if call:
            saved["records"][direction] = call
        saved["tensors"].update(tensors)


def _repro_root(config):
    option = config.getoption("repro_dir")
    return Path(option) if option else None


def _save_repro(item, row):
    """Save the case's repro through ``hipdnn_sdpa.save_repro`` within the caps; its path relative
    to the shard directory, or '' when nothing was saved."""
    import sys

    root = _repro_root(item.config)
    module = sys.modules.get("hipdnn_sdpa")
    with _LOCK:
        saved = _CALLS.get(item.nodeid)
    if (
        root is None
        or saved is None
        or not saved["records"]
        or not hasattr(module, "save_repro")
    ):
        return ""
    engine = reconcile.served_engines(row[row["fail_side"]])
    if not reconcile.claim_repro(root, "+".join(sorted(engine)), item.nodeid):
        return ""
    case = contract.repro_case(
        item.nodeid, os.environ.get(contract.PASS_SLUG_ENV, reconcile.MAIN_PASS)
    )
    module.save_repro(root / case, dict(saved["records"]), dict(saved["tensors"]))
    return f"{contract.REPRO_DIR}/{case}"


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
            call = hipdnn_sdpa.last_call(direction)
            record(direction, "engines", ";".join(call.ranked) if call else "")
            if call and call.kernel:
                record(direction, "kernel", call.kernel)
            _remember_call(direction, call, args, kwargs)
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


def _crash_row(report):
    """The row of a case whose worker died: the engine, direction and stage come from the marker
    ``<in-flight dir>/<pass>/<gateway id>.json`` when it was written for this case; it is deleted
    once read. xdist gives a replacement worker a new gateway id, so a marker is never reused.
    """
    row = {
        "nodeid": report.nodeid,
        "outcome": "crash",
        "reason": _one_line(str(report.longrepr)),
    }
    gateway = getattr(getattr(report, "node", None), "gateway", None)
    path = _inflight_marker_path(getattr(gateway, "id", None))
    marker = _read_marker(path, report.nodeid) if path else None
    if marker is None:
        return row
    path.unlink(missing_ok=True)
    direction = marker.get("direction", "")
    engine = marker.get("engine") or marker.get(f"pin_{direction}", "")
    if engine == contract.REFERENCE_PIN:
        engine = ""
    stage = marker.get("stage", "")
    row.update(
        fail_side=direction,
        stage=stage,
        fwd_engines=_names(marker.get("fwd_engines")),
        bwd_engines=_names(marker.get("bwd_engines")),
        reason=_one_line(
            f"worker died in the {direction} {stage} stage"
            + (f" on {engine}" if engine else "")
            + f" (pid {marker.get('pid', '?')})"
        ),
    )
    if direction in ("fwd", "bwd"):
        row[direction] = f"hipdnn:{engine}" if engine else ""
        row[f"{direction}_kernel"] = marker.get("kernel", "")
    return row


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
            row = _crash_row(report)
        else:
            return
        if row["nodeid"] in self.seen:
            return
        self.seen.add(row["nodeid"])
        self.file.write(reconcile.format_row(row))
        self.file.flush()

    def close(self):
        self.file.close()


def _install_kernel_log():
    """Register ``hipdnn_sdpa``'s kernel-id log callback in a process that runs cases."""
    import sys

    module = sys.modules.get("hipdnn_sdpa")
    if hasattr(module, "install_kernel_log"):
        module.install_kernel_log()


def pytest_configure(config):
    refused = [k for k in layout.REFUSED_KNOBS if _knob_active(k)]
    if refused:
        raise pytest.UsageError(
            "the AOTriton port refuses these environment variables (they change what the "
            f"vendored suite measures): {', '.join(refused)}"
        )
    if config.getoption("shard"):
        _parse_shard(config.getoption("shard"))
    config._sdpa_backend = _bind_backend_first(config.getoption("hipdnn_prefix"))
    fake = config.getoption("fake") == "always-decline"
    _install_wrappers(fake)
    runs_cases = (
        hasattr(config, "workerinput") or config.getoption("dist", "no") == "no"
    )
    if runs_cases and not fake:
        _install_kernel_log()
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
    backend = getattr(config, "_sdpa_backend", None) or "not loaded"
    return (
        "sdpa-conformance: "
        + (", ".join(forwarded) or "no forwarded knobs")
        + f"; hipDNN backend: {backend}"
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    listed = config.getoption("hipdnn_nodeids_file")
    if listed:
        wanted = {
            line.strip()
            for line in Path(listed).read_text().splitlines()
            if line.strip()
        }
        _keep(config, items, wanted)
    spec = config.getoption("shard")
    if spec:
        index, count = _parse_shard(spec)
        _keep(
            config,
            items,
            set(sorted(item.nodeid for item in items)[index - 1 :: count]),
        )


def _keep(config, items, chosen):
    """Deselect every item whose node id is not in ``chosen``."""
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
            _CALLS.pop(item.nodeid, None)
            _CURRENT = None
        marker = _inflight_marker_path(os.environ.get("PYTEST_XDIST_WORKER"))
        if marker:
            marker.unlink(missing_ok=True)
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
