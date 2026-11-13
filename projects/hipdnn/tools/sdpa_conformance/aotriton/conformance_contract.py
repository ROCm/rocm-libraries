# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Names and formats shared by the runner, the plugin, the shim and ``hipdnn_sdpa``.

Every on-disk file and environment variable that crosses a process or module boundary in a run is
named here, so the writer and every reader agree on one spelling.

Engine pins and run modes
    ``HIPDNN_FWD_ENGINE`` / ``HIPDNN_BWD_ENGINE`` (``ENGINE_ENV``) pin a direction's engine;
    ``HIPDNN_FWD_ENGINE=reference`` (``REFERENCE_PIN``) runs the reference forward without asking
    hipDNN. ``FWD_NO_STATS_ENV`` runs the forward without Stats and ``LAYOUT_MODE_ENV`` names the
    layout mode; the runner sets both for every pass. ``IMAGE_ENV`` and ``SHAS_ENV`` are set by the
    job that launches the runner (the container image and the source SHAs it built) and are copied
    into each repro's ``meta.json``; the tool never sets them.

Pass kinds
    ``pass.json`` ``kind`` of an ``--all-engines`` or ``--exhaustive`` pass: ``PASS_SWEEP`` (one
    direction pinned), ``PASS_PAIR`` (both pinned), ``PASS_DISCOVERY`` and ``PASS_REFWD`` (backward
    on the reference forward, unpinned and pinned), ``PASS_CANDIDATE`` and ``PASS_CANDIDATE_PAIR``.

Call errors
    A ``hipdnn_sdpa`` call that fails with anything other than ``EngineDeclined`` (a build or
    execute error) re-raises with attribute ``sdpa_call`` (``CALL_ERROR_ATTR``) set to the failing
    call's :class:`CallRecord`, or ``None`` when it failed before its graph was built. The plugin
    records that direction as kind ``error`` with the engine as detail, so the row names the
    direction that failed and its repro holds the failing call.

In-flight marker
    ``$HIPDNN_SDPA_INFLIGHT_DIR/<pass-slug>/<worker>.json``, where ``<worker>`` is
    ``PYTEST_XDIST_WORKER`` (``gw0``...; ``main`` without xdist). ``hipdnn_sdpa`` rewrites it
    atomically at the start of each ``run`` (stage ``build``), again once the ranked engine list
    is known (stage ``build``), and before ``execute`` (stage ``execute``). The plugin removes it
    at teardown. A worker that dies leaves it behind; the controller reads the marker named by
    the crash report's gateway id whose ``nodeid`` matches the report, then deletes it. xdist
    gives a replacement worker a new gateway id, so a marker is never overwritten by one.

Kernel records
    ``hipdnn_sdpa.last_call(direction)`` returns the :class:`CallRecord` of the calling thread's
    most recent ``sdpa_fwd``/``sdpa_bwd``. The plugin records its kernel with
    ``record(direction, "kernel", kernel)``; the plugin's fold and ``reconcile.served_engines``
    ignore entries of kind ``kernel`` as they ignore ``engines``. Kernel ids come from the backend's
    INFO log, and installing the callback that reads it raises the backend log level and forces
    ingestor applicability inline, which changes how the product runs; so the plugin installs it
    only when ``HIPDNN_SDPA_KERNEL_LOG`` (``KERNEL_LOG_ENV``) is ``1``, which the runner sets in
    every pass for ``--kernel-ids`` and ``--exhaustive`` (candidate honoring compares kernel ids).
    Without it the kernel columns stay empty.

Candidates
    ``--exhaustive`` pins kernels inside an engine. ``HIPDNN_FWD_CANDIDATE`` / ``HIPDNN_BWD_CANDIDATE``
    (``CANDIDATE_ENV``) hold a candidate index ``k`` and take effect only beside the same
    direction's engine pin; ``hipdnn_sdpa`` builds the ``k``-th entry of ``hipdnn_bench enumerate``
    for that engine and declines ``candidate_not_honored`` when the kernel the backend logged is
    not the candidate's. ``HIPDNN_SDPA_EXHAUSTIVE`` makes an unpinned call count the candidates of
    every engine it ranked, and ``HIPDNN_SDPA_PREFIX`` names the hipDNN prefix whose ``bin/``
    holds ``hipdnn_bench``. ``fwd_candidate`` / ``bwd_candidate`` of a ``results.csv`` row hold the
    pinned index in a candidate pass and, in pass 1, a JSON object ``{engine: count | message}``
    (``count`` an integer, ``message`` why the engine cannot enumerate). Candidate passes live
    under ``<out>/cand/``.

Repro directories
    ``<out>/repro/<case>/`` with ``case = repro_case(nodeid, pass_slug)``, holding
    ``meta.json``, ``<direction>.graph.json`` (``repro_graph_name``), ``tensors.pt`` and, for a
    failure, ``outputs.pt`` (the ``OUTPUT_ROLES`` of the last direction). ``HIPDNN_SDPA_REPRO_CAPTURE``
    names the repro root; when set, ``hipdnn_sdpa`` captures a call's inputs, graph and problem
    scalars before building its plan, so a crash in build or execute still leaves a replayable
    directory.

Inventory
    ``<out>/engine-inventory.txt``: one engine name per line, as ``hipdnn_list_engines`` prints
    it (``ASM_SDPA_ENGINE``, ``hipkernel:flydsl_sdpa_cdna``...), written by the job.
"""

import dataclasses
import hashlib
import os

ENGINE_ENV = {"fwd": "HIPDNN_FWD_ENGINE", "bwd": "HIPDNN_BWD_ENGINE"}
FWD_NO_STATS_ENV = "HIPDNN_SDPA_CONFORMANCE_FWD_NO_STATS"
LAYOUT_MODE_ENV = "HIPDNN_SDPA_CONFORMANCE_LAYOUT_MODE"
IMAGE_ENV = "HIPDNN_SDPA_IMAGE"
SHAS_ENV = "HIPDNN_SDPA_SHAS"

# The vendored forward test file: its cases are forward cases, every other case is a backward case.
FORWARD_TESTS = "test_forward.py"

PASS_SWEEP = "sweep"
PASS_PAIR = "pair"
PASS_DISCOVERY = "discovery"
PASS_REFWD = "refwd"
PASS_CANDIDATE = "candidate"
PASS_CANDIDATE_PAIR = "candidate-pair"

CALL_ERROR_ATTR = "sdpa_call"

# Value of HIPDNN_FWD_ENGINE that runs the reference forward without asking hipDNN.
REFERENCE_PIN = "reference"

# Candidate pins (see "Candidates" above).
CANDIDATE_ENV = {"fwd": "HIPDNN_FWD_CANDIDATE", "bwd": "HIPDNN_BWD_CANDIDATE"}
EXHAUSTIVE_ENV = "HIPDNN_SDPA_EXHAUSTIVE"
# "1" makes the plugin install ``hipdnn_sdpa``'s kernel-id log callback (see "Kernel records").
KERNEL_LOG_ENV = "HIPDNN_SDPA_KERNEL_LOG"
PREFIX_ENV = "HIPDNN_SDPA_PREFIX"
BENCH_RELATIVE = "bin/hipdnn_bench"
BENCH_PLUGIN_DIR = "lib/hipdnn_plugins/engines"
CANDIDATES_DIR = "cand"
NOT_ENUMERABLE = "not-enumerable"  # outcome of a case whose engine lists no candidates
CANDIDATE_NOT_HONORED = "candidate_not_honored"  # decline reason prefix

INFLIGHT_DIR_ENV = "HIPDNN_SDPA_INFLIGHT_DIR"
PASS_SLUG_ENV = "HIPDNN_SDPA_PASS_SLUG"
REPRO_CAPTURE_ENV = "HIPDNN_SDPA_REPRO_CAPTURE"
DECLINES_DIR_ENV = "HIPDNN_SDPA_DECLINES_DIR"
PLAN_CACHE_ENV = "HIPDNN_SDPA_PLAN_CACHE"
# Test only: "<direction>:<engine>" makes hipdnn_sdpa exit the process in place of executing a
# plan of that engine and direction, after writing the execute-stage marker.
TEST_CRASH_ENV = "HIPDNN_SDPA_TEST_CRASH"
TEST_CRASH_EXIT_CODE = 134

STAGE_BUILD = "build"
STAGE_EXECUTE = "execute"
STAGES = (STAGE_BUILD, STAGE_EXECUTE)

MARKER_FIELDS = (
    "nodeid",
    "direction",
    "stage",
    "engine",
    "pin_fwd",
    "pin_bwd",
    "fwd_engines",
    "bwd_engines",
    "kernel",
    "pid",
    "graph_json",
)

REPRO_DIR = "repro"
REPRO_META = "meta.json"
REPRO_TENSORS = "tensors.pt"
REPRO_OUTPUTS = "outputs.pt"
# Output tensor roles per direction; a repro's outputs.pt holds those of its last direction.
OUTPUT_ROLES = {"fwd": ("o", "stats"), "bwd": ("dq", "dk", "dv", "dbias")}
REPRO_MAX_PER_KEY = 3  # per (engine, test function)
REPRO_MAX_BYTES = 2 << 30  # per shard

INVENTORY_NAME = "engine-inventory.txt"
DECLINES_PER_ENGINE = 20

# Columns appended to reconcile.FIELDS and reconcile.ENGINES_FIELDS; older CSVs read them empty.
RESULTS_EXTRA_FIELDS = ("fwd_kernel", "bwd_kernel", "fail_side", "stage", "repro")
ENGINES_EXTRA_FIELDS = ("pin_fwd", "pin_bwd", "kernel", "stage")
# Columns appended after RESULTS_EXTRA_FIELDS (results.csv) and ENGINES_EXTRA_FIELDS (engines.csv).
CANDIDATE_FIELDS = ("fwd_candidate", "bwd_candidate")
FAIL_SIDES = ("fwd", "bwd", "")


@dataclasses.dataclass(frozen=True)
class CallRecord:
    """What one ``sdpa_fwd``/``sdpa_bwd`` call ran.

    ``kernel`` is the ingestor kernel id (``prep=<id>|dkdv=<id>|dq=<id>`` for a slotted engine);
    empty for engines that report none. ``ranked`` is the applicable engines in heuristic rank
    order, empty for a pinned call. ``problem_meta`` holds the scalars a replay needs (scale,
    mask, dropout probability, varlen maxima) and the role-to-uid map. ``candidate`` is the pinned
    candidate as ``{"index": k, "id": ..., "knobs": ...}`` (an entry of ``hipdnn_bench
    enumerate`` plus its index), ``None`` when the call pinned no candidate.
    """

    direction: str
    engine: str
    kernel: str
    ranked: tuple
    graph_json: str
    problem_meta: dict
    candidate: dict = None


def marker_name(worker):
    """File name of a worker's in-flight marker inside the pass's in-flight directory."""
    return f"{worker or 'main'}.json"


def repro_graph_name(direction):
    """File name of a direction's graph JSON inside a repro directory."""
    return f"{direction}.graph.json"


def current_nodeid():
    """Node id of the test pytest is running in this process ('' outside a test)."""
    return os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]


def repro_case(nodeid, pass_slug):
    """Directory name of one case's repro under ``<out>/repro/``."""
    digest = hashlib.sha1(nodeid.encode()).hexdigest()[:12]
    return f"{digest}-{pass_slug}"


def parse_test_crash(value):
    """``(direction, engine)`` from a ``TEST_CRASH_ENV`` value; engine names contain ``:``."""
    direction, _, engine = (value or "").partition(":")
    return (direction, engine) if direction and engine else None
