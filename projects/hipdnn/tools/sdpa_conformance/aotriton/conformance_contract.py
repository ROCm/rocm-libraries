# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Names and formats shared by the runner, the plugin, the shim and ``hipdnn_sdpa``.

Every on-disk file and environment variable that crosses a process or module boundary in an
``--all-engines`` run is named here, so the writer and every reader agree on one spelling.

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
    ignore entries of kind ``kernel`` as they ignore ``engines``.

Repro directories
    ``<out>/repro/<case>/`` with ``case = repro_case(nodeid, pass_slug)``, holding
    ``meta.json``, ``fwd.graph.json``, ``bwd.graph.json``, ``tensors.pt`` and, for a failure,
    ``outputs.pt``. ``HIPDNN_SDPA_REPRO_CAPTURE`` names the repro root; when set, ``hipdnn_sdpa``
    captures a call's inputs, graph and problem scalars before building its plan, so a crash in
    build or execute still leaves a replayable directory.

Inventory
    ``<out>/engine-inventory.txt``: one engine name per line, as ``hipdnn_list_engines`` prints
    it (``ASM_SDPA_ENGINE``, ``hipkernel:flydsl_sdpa_cdna``...), written by the job.
"""

import dataclasses
import hashlib

# Value of HIPDNN_FWD_ENGINE that runs the reference forward without asking hipDNN.
REFERENCE_PIN = "reference"

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
REPRO_FILES = (
    "meta.json",
    "fwd.graph.json",
    "bwd.graph.json",
    "tensors.pt",
    "outputs.pt",
)
REPRO_MAX_PER_KEY = 3  # per (engine, test function)
REPRO_MAX_BYTES = 2 << 30  # per shard

INVENTORY_NAME = "engine-inventory.txt"
DECLINES_PER_ENGINE = 20

# Columns appended to reconcile.FIELDS and reconcile.ENGINES_FIELDS; older CSVs read them empty.
RESULTS_EXTRA_FIELDS = ("fwd_kernel", "bwd_kernel", "fail_side", "stage", "repro")
ENGINES_EXTRA_FIELDS = ("pin_fwd", "pin_bwd", "kernel", "stage")
FAIL_SIDES = ("fwd", "bwd", "")


@dataclasses.dataclass(frozen=True)
class CallRecord:
    """What one ``sdpa_fwd``/``sdpa_bwd`` call ran.

    ``kernel`` is the ingestor kernel id (``prep=<id>|dkdv=<id>|dq=<id>`` for a slotted engine);
    empty for engines that report none. ``ranked`` is the applicable engines in heuristic rank
    order, empty for a pinned call. ``problem_meta`` holds the scalars a replay needs (scale,
    mask, dropout probability, varlen maxima) and the role-to-uid map.
    """

    direction: str
    engine: str
    kernel: str
    ranked: tuple
    graph_json: str
    problem_meta: dict


def marker_name(worker):
    """File name of a worker's in-flight marker inside the pass's in-flight directory."""
    return f"{worker or 'main'}.json"


def repro_case(nodeid, pass_slug):
    """Directory name of one case's repro under ``<out>/repro/``."""
    digest = hashlib.sha1(nodeid.encode()).hexdigest()[:12]
    return f"{digest}-{pass_slug}"


def parse_test_crash(value):
    """``(direction, engine)`` from a ``TEST_CRASH_ENV`` value; engine names contain ``:``."""
    direction, _, engine = (value or "").partition(":")
    return (direction, engine) if direction and engine else None
