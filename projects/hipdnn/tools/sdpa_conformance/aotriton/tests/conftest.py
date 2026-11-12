# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fixtures shared by the CPU tests of the AOTriton port.

`fake_hipdnn_sdpa` installs a recording stand-in for the `hipdnn_sdpa` package: the Contract's
dataclasses implemented locally, `sdpa_fwd` / `sdpa_bwd` that record their arguments (and decline
on request), and a deterministic `philox_mask`. The real package is never imported.
"""

import dataclasses
import json
import sys
import threading
import types
from pathlib import Path

import pytest
import torch

AOTRITON_DIR = Path(__file__).resolve().parents[1]
for entry in (AOTRITON_DIR, AOTRITON_DIR / "shim"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from conformance_contract import CALL_ERROR_ATTR, CallRecord  # noqa: E402

pytest_plugins = ["pytester"]


class EngineDeclined(Exception):
    def __init__(self, direction, reason):
        super().__init__(f"{direction}: {reason}")
        self.direction = direction
        self.reason = reason


@dataclasses.dataclass(frozen=True)
class Mask:
    alignment: str
    left: "int | None"
    right: "int | None"


@dataclasses.dataclass(frozen=True)
class Varlen:
    seq_len_q: torch.Tensor
    seq_len_kv: torch.Tensor
    pos_q: "torch.Tensor | None"
    pos_kv: "torch.Tensor | None"
    max_sq: int
    max_sk: int


@dataclasses.dataclass(frozen=True)
class Dropout:
    p: float
    seed: torch.Tensor
    offset: torch.Tensor


def keep_pattern(B, Hq, max_sq, max_sk, p, seed, offset, device):
    """Deterministic stand-in for the Philox keep mask: depends on every argument."""
    b, h, i, j = torch.meshgrid(
        torch.arange(B),
        torch.arange(Hq),
        torch.arange(max_sq),
        torch.arange(max_sk),
        indexing="ij",
    )
    return ((b * 7 + h * 5 + i * 3 + j + seed + offset) % 3 != 0).to(device)


@dataclasses.dataclass(frozen=True)
class _FakeCall(CallRecord):
    """What the real runtime's call record adds to the contract's: the exhaustive pass-1 counts."""

    candidates: dict = dataclasses.field(default_factory=dict)


@pytest.fixture
def fake_sdpa(monkeypatch):
    """The fake `hipdnn_sdpa` module; `.calls` lists (name, kwargs) of every sdpa_* call and
    `.decline` (a dict direction -> reason) makes a direction raise `EngineDeclined`;
    `.engine_lists` (direction -> tuple of names), `.kernels` (direction -> kernel id) and `.candidates`
    (direction -> {engine: count | message}) fill the
    `CallRecord` that `last_call(direction)` returns on the calling thread after a successful call
    (None after a decline). `.log_installs` counts `install_kernel_log()` calls; `.repros` lists the
    `(dir, directions, roles)` of every `save_repro` call, which also writes `<dir>/saved.json`.
    `.errors` (direction -> message) makes a direction raise `RuntimeError` carrying the failing
    call's record as `sdpa_call` (None for a direction in `.unbuilt`, a failure before the graph);
    `.missing` is what `missing_bindings()` returns; `.log_ok` is what `install_kernel_log()` returns;
    `.save_repro_error`, when set, is raised by `save_repro`.
    """
    module = types.ModuleType("hipdnn_sdpa")
    module.EngineDeclined, module.Mask, module.Varlen, module.Dropout = (
        EngineDeclined,
        Mask,
        Varlen,
        Dropout,
    )
    module.philox_mask = keep_pattern
    module.calls = []
    module.decline = {}
    module.engine_lists = {}
    module.kernels = {}
    module.candidates = {}
    module.errors = {}
    module.unbuilt = set()
    module.missing = []
    module.log_ok = True
    module.save_repro_error = None
    module.log_installs = 0
    module.repros = []
    module.CallRecord = CallRecord
    last = threading.local()

    def last_call(direction):
        return getattr(last, direction, None)

    def call_record(direction, engine):
        return _FakeCall(
            direction,
            engine,
            module.kernels.get(direction, ""),
            tuple(module.engine_lists.get(direction, ())),
            "{}",
            {},
            None,
            dict(module.candidates.get(direction, {})),
        )

    def finish(direction, engine):
        setattr(last, direction, call_record(direction, engine))
        return engine

    def fail_if_asked(direction):
        if direction in module.errors:
            exc = RuntimeError(module.errors[direction])
            built = direction not in module.unbuilt
            setattr(
                exc,
                CALL_ERROR_ATTR,
                call_record(direction, f"fake_{direction}") if built else None,
            )
            raise exc

    def install_kernel_log():
        module.log_installs += 1
        return module.log_ok

    def missing_bindings():
        return list(module.missing)

    def save_repro(dir, direction_records, tensors_by_role):
        if module.save_repro_error:
            raise module.save_repro_error
        module.repros.append((dir, sorted(direction_records), sorted(tensors_by_role)))
        dir.mkdir(parents=True, exist_ok=True)
        (dir / "saved.json").write_text(
            json.dumps(
                {
                    "directions": sorted(direction_records),
                    "roles": sorted(tensors_by_role),
                }
            )
        )

    module.last_call, module.install_kernel_log = last_call, install_kernel_log
    module.save_repro, module.missing_bindings = save_repro, missing_bindings

    def sdpa_fwd(
        q,
        k,
        v,
        *,
        o,
        stats,
        scale,
        bias=None,
        mask=None,
        dropout=None,
        varlen=None,
        engine=None,
    ):
        module.calls.append(
            (
                "fwd",
                dict(
                    q=q,
                    k=k,
                    v=v,
                    o=o,
                    stats=stats,
                    scale=scale,
                    bias=bias,
                    mask=mask,
                    dropout=dropout,
                    varlen=varlen,
                    engine=engine,
                ),
            )
        )
        last.fwd = None
        if "fwd" in module.decline:
            raise EngineDeclined("fwd", module.decline["fwd"])
        fail_if_asked("fwd")
        o.fill_(1.0)
        if stats is not None:
            stats.fill_(2.0)
        return finish("fwd", "fake_fwd")

    def sdpa_bwd(
        q,
        k,
        v,
        o,
        do,
        stats,
        *,
        dq,
        dk,
        dv,
        scale,
        dbias=None,
        bias=None,
        mask=None,
        dropout=None,
        varlen=None,
        engine=None,
    ):
        module.calls.append(
            (
                "bwd",
                dict(
                    q=q,
                    k=k,
                    v=v,
                    o=o,
                    do=do,
                    stats=stats,
                    dq=dq,
                    dk=dk,
                    dv=dv,
                    scale=scale,
                    dbias=dbias,
                    bias=bias,
                    mask=mask,
                    dropout=dropout,
                    varlen=varlen,
                    engine=engine,
                ),
            )
        )
        last.bwd = None
        if "bwd" in module.decline:
            raise EngineDeclined("bwd", module.decline["bwd"])
        fail_if_asked("bwd")
        return finish("bwd", "fake_bwd")

    module.sdpa_fwd, module.sdpa_bwd = sdpa_fwd, sdpa_bwd
    monkeypatch.setitem(sys.modules, "hipdnn_sdpa", module)
    return module


@pytest.fixture
def mode_record(monkeypatch):
    """Stand-in for the plugin's mode record: a list of (direction, kind, detail)."""
    module = types.ModuleType("hipdnn_aotriton_plugin")
    module.entries = []
    module.record = lambda direction, kind, detail="": module.entries.append(
        (direction, kind, str(detail))
    )
    monkeypatch.setitem(sys.modules, "hipdnn_aotriton_plugin", module)
    return module.entries
