# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A file-backed stand-in for the ``hipdnn_sdpa`` package, importable by pytest subprocesses.

Engines: ``FAKE_FWD_ENGINES`` / ``FAKE_BWD_ENGINES`` (``;``-joined, ranked). An unpinned call
runs the first; ``HIPDNN_FWD_ENGINE`` / ``HIPDNN_BWD_ENGINE`` pin one (``reference`` declines the
forward, as the shim's reference pin does). Like the real runtime it leaves the contract's
in-flight marker (build, then execute), honours ``HIPDNN_SDPA_TEST_CRASH`` and, with
``HIPDNN_SDPA_REPRO_CAPTURE`` set, writes the case's repro directory before executing.
"""

import json
import os
import threading

from conformance_contract import (
    INFLIGHT_DIR_ENV,
    PASS_SLUG_ENV,
    REPRO_CAPTURE_ENV,
    TEST_CRASH_ENV,
    TEST_CRASH_EXIT_CODE,
    CallRecord,
    marker_name,
    parse_test_crash,
    repro_case,
)


class EngineDeclined(Exception):
    def __init__(self, direction, reason):
        super().__init__(f"{direction}: {reason}")
        self.direction, self.reason = direction, reason


_last = threading.local()


def _nodeid():
    return os.environ["PYTEST_CURRENT_TEST"].rsplit(" ", 1)[0]


def _marker(direction, stage, engine, ranked):
    root = os.environ.get(INFLIGHT_DIR_ENV)
    if not root:
        return
    path = os.path.join(
        root,
        os.environ.get(PASS_SLUG_ENV, "main"),
        marker_name(os.environ.get("PYTEST_XDIST_WORKER")),
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = dict(
        nodeid=_nodeid(),
        direction=direction,
        stage=stage,
        engine=engine,
        pin_fwd=os.environ.get("HIPDNN_FWD_ENGINE", ""),
        pin_bwd=os.environ.get("HIPDNN_BWD_ENGINE", ""),
        fwd_engines=list(ranked) if direction == "fwd" else [],
        bwd_engines=list(ranked) if direction == "bwd" else [],
        kernel=f"kernel-{engine}",
        pid=os.getpid(),
        graph_json="{}",
    )
    with open(path + ".tmp", "w") as f:
        json.dump(fields, f)
    os.replace(path + ".tmp", path)


def _capture(direction, engine):
    root = os.environ.get(REPRO_CAPTURE_ENV)
    if root:
        case = os.path.join(
            root, repro_case(_nodeid(), os.environ.get(PASS_SLUG_ENV, "main"))
        )
        os.makedirs(case, exist_ok=True)
        with open(os.path.join(case, "meta.json"), "w") as f:
            json.dump(
                {"nodeid": _nodeid(), "direction": direction, "engine": engine}, f
            )


def _run(direction, engine):
    ranked = [
        e
        for e in os.environ.get(f"FAKE_{direction.upper()}_ENGINES", "").split(";")
        if e
    ]
    pin = engine or os.environ.get(f"HIPDNN_{direction.upper()}_ENGINE") or ""
    setattr(_last, direction, None)
    if pin == "reference":
        raise EngineDeclined(direction, "pinned reference")
    chosen = pin or ranked[0]
    _marker(direction, "build", "", ranked)
    if chosen not in ranked:
        raise EngineDeclined(direction, f"engine_not_applicable: {chosen}")
    _marker(direction, "build", chosen, ranked)
    _capture(direction, chosen)
    _marker(direction, "execute", chosen, ranked)
    crash = parse_test_crash(os.environ.get(TEST_CRASH_ENV))
    if crash == (direction, chosen):
        os._exit(TEST_CRASH_EXIT_CODE)
    setattr(
        _last,
        direction,
        CallRecord(
            direction,
            chosen,
            f"kernel-{chosen}",
            () if pin else tuple(ranked),
            "{}",
            {},
        ),
    )
    return chosen


def sdpa_fwd(q, k, v, *, o, stats, scale, engine=None, **_):
    return _run("fwd", engine)


def sdpa_bwd(q, k, v, o, do, stats, *, dq, dk, dv, scale, engine=None, **_):
    return _run("bwd", engine)


def last_call(direction):
    return getattr(_last, direction, None)


def install_kernel_log():
    return True


def save_repro(dir, direction_records, tensors_by_role):
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, "meta.json"), "w") as f:
        json.dump(
            {"directions": sorted(direction_records), "roles": sorted(tensors_by_role)},
            f,
        )
