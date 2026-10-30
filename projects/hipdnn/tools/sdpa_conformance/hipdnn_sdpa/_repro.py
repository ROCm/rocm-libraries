# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Repro directories: a failing or crashing call's inputs, graph and problem scalars on disk.

Layout (``conformance_contract`` names the files)::

    meta.json        node id, pass, pins, directions, per-direction engine, kernel, scale, mask,
                     dropout p, varlen maxima, role -> uid and slot layout, per-tensor layout
    fwd.graph.json   the forward graph (``Graph.to_json()``), when a forward is part of the repro
    bwd.graph.json   the backward graph
    tensors.pt       role -> compact CPU tensor, every role the replay reads
    outputs.pt       role -> tensor hipDNN wrote (failures only)

A backward repro holds the forward's ``o`` and ``stats`` (the backward's inputs) and ``do`` in
``tensors.pt``; its ``outputs.pt`` holds ``dq``/``dk``/``dv`` (``dbias``). A forward-only repro
holds ``o``/``stats`` in ``outputs.pt``. ``aotriton/replay.py <dir>`` replays it.

Two writers share the format: :func:`save_repro` (the plugin, after a failure) and
:func:`capture` (``HIPDNN_SDPA_REPRO_CAPTURE``, from ``run`` before any hipDNN call that could
crash, so a build or execute crash still leaves the directory).
"""

from __future__ import annotations

import json
import os

import conformance_contract as contract
import torch

_OUTPUT_ROLES = {"fwd": ("o", "stats"), "bwd": ("dq", "dk", "dv", "dbias")}
_DIRECTIONS = ("fwd", "bwd")

# The capture state of the case being run in this process: records and tensors accumulate over
# its forward and backward calls and the directory is rewritten from them.
_CAPTURED: dict = {}


def problem_meta(prob) -> dict:
    """The scalars and layout a replay needs, JSON-safe (``CallRecord.problem_meta``)."""
    mask = prob.mask
    return {
        "scale": prob.scale,
        "mask_alignment": None if mask is None else mask.alignment,
        "mask_left": None if mask is None else mask.left,
        "mask_right": None if mask is None else mask.right,
        "dropout_p": prob.dropout_p,
        "padding": prob.padding,
        "packed_q": prob.packed_q,
        "packed_kv": prob.packed_kv,
        "max_sq": prob.slots["q"].dims[2] if prob.padding else None,
        "max_sk": prob.slots["k"].dims[2] if prob.padding else None,
        "uids": prob.uids(),
        "slots": {
            role: {
                "dims": list(slot.dims),
                "strides": list(slot.strides),
                "dtype": str(slot.dtype),
            }
            for role, slot in prob.slots.items()
        },
    }


def _nodeid() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]


def _atomic(path: str, write) -> None:
    tmp = f"{path}.tmp"
    write(tmp)
    os.replace(tmp, path)


def _compact(tensor: torch.Tensor) -> torch.Tensor:
    """A dense CPU copy: saving a view would save its whole storage."""
    return tensor.detach().clone(memory_format=torch.contiguous_format).cpu()


def _write(directory: str, records: dict, tensors: dict, with_outputs: bool) -> None:
    directions = [d for d in _DIRECTIONS if d in records]
    out_roles = _OUTPUT_ROLES[directions[-1]]
    os.makedirs(directory, exist_ok=True)

    def save(name, roles):
        _atomic(
            os.path.join(directory, name),
            lambda tmp: torch.save({r: _compact(tensors[r]) for r in roles}, tmp),
        )

    inputs = [r for r in tensors if r not in out_roles]
    save("tensors.pt", inputs)
    outputs = [r for r in tensors if r in out_roles]
    if with_outputs and outputs:
        save("outputs.pt", outputs)

    for direction in directions:
        _atomic(
            os.path.join(directory, f"{direction}.graph.json"),
            lambda tmp, d=direction: open(tmp, "w").write(records[d].graph_json),
        )

    meta = {
        "nodeid": _nodeid(),
        "pass": os.environ.get(contract.PASS_SLUG_ENV, ""),
        "pin_fwd": os.environ.get("HIPDNN_FWD_ENGINE", ""),
        "pin_bwd": os.environ.get("HIPDNN_BWD_ENGINE", ""),
        "directions": directions,
        "image": os.environ.get("HIPDNN_SDPA_IMAGE", ""),
        "shas": os.environ.get("HIPDNN_SDPA_SHAS", ""),
        "calls": {
            d: {
                "engine": records[d].engine,
                "kernel": records[d].kernel,
                "ranked": list(records[d].ranked),
                **records[d].problem_meta,
            }
            for d in directions
        },
        "tensors": {
            role: {
                "shape": list(t.shape),
                "strides": list(t.stride()),
                "dtype": str(t.dtype),
            }
            for role, t in tensors.items()
        },
    }
    _atomic(
        os.path.join(directory, "meta.json"),
        lambda tmp: json.dump(meta, open(tmp, "w"), indent=1),
    )


def save_repro(directory, direction_records, tensors_by_role=None) -> None:
    """Write the repro of a failed case into ``directory`` (created if missing).

    ``direction_records`` maps ``"fwd"``/``"bwd"`` to the :func:`hipdnn_sdpa.last_call` records of
    the calls the case made. Tensors come from the records themselves (every slot of the
    problem, derived ones included); ``tensors_by_role`` adds roles the records lack (extra keys, such
    as raw varlen arguments, are kept as given). Outputs
    (``dq``/``dk``/``dv``/``dbias`` for a backward, else ``o``/``stats``) go to ``outputs.pt``.
    """
    records = {d: r for d, r in direction_records.items() if r is not None}
    tensors: dict = {}
    for direction in _DIRECTIONS:
        if direction in records:
            tensors.update(getattr(records[direction], "tensors", None) or {})
    for role, tensor in (tensors_by_role or {}).items():
        tensors.setdefault(role, tensor)
    _write(str(directory), records, tensors, with_outputs=True)


def capture(prob, graph_json: str, engine: str | None) -> None:
    """Save this call's inputs and graph under ``HIPDNN_SDPA_REPRO_CAPTURE``; off when unset."""
    root = os.environ.get(contract.REPRO_CAPTURE_ENV)
    if not root:
        return
    case = contract.repro_case(_nodeid(), os.environ.get(contract.PASS_SLUG_ENV, ""))
    if _CAPTURED.get("case") != case:
        _CAPTURED.clear()
        _CAPTURED.update(case=case, records={}, tensors={})
    _CAPTURED["records"][prob.direction] = contract.CallRecord(
        prob.direction, engine or "", "", (), graph_json, problem_meta(prob)
    )
    _CAPTURED["tensors"].update(prob.tensors)
    _write(
        os.path.join(root, case),
        _CAPTURED["records"],
        _CAPTURED["tensors"],
        with_outputs=False,
    )
