# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Replay a repro directory: ``python replay.py <repro-dir> [--device cuda|cpu]``.

Needs torch and ``hipdnn_frontend``; neither pytest nor AOTriton. For each direction the repro
holds, a child process rebuilds the graph (``Graph()`` then ``from_json(handle, json)``), pins the
engine named in ``meta.json``, binds the saved tensors by ``meta.json``'s role-to-uid map and
executes. The child runs apart so a hardware fault ends it, not this report: a crash prints
``reproduced: died with signal N`` (``reproduced: exited with status N`` for a plain exit) and the
exit status is 1. A completed replay prints, per output role, the maximum absolute error against
the hipDNN output the repro saved and against the FP32 reference attention (backward through
autograd), and exits 0.

The repro layout is documented in ``hipdnn_sdpa/_repro.py``.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
for entry in (
    HERE.parent,
    HERE / "shim",
):  # hipdnn_sdpa, then the reference in the shim
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch  # noqa: E402

REPRODUCED_EXIT = 1
SETUP_ERROR_EXIT = 70
OUTPUT_ROLES = {"fwd": ("o", "stats"), "bwd": ("dq", "dk", "dv", "dbias")}


def _dtype(name):
    return getattr(torch, name.split(".")[-1])


def _load(directory):
    root = Path(directory)
    meta = json.loads((root / "meta.json").read_text())
    tensors = torch.load(root / "tensors.pt")
    outputs_path = root / "outputs.pt"
    outputs = torch.load(outputs_path) if outputs_path.exists() else {}
    return root, meta, tensors, outputs


def _materialize(layout, saved, device):
    """A tensor with the original shape and strides holding ``saved`` (zeros when absent)."""
    t = torch.empty_strided(
        layout["shape"], layout["strides"], dtype=_dtype(layout["dtype"]), device=device
    ).zero_()
    if saved is not None:
        t.copy_(saved)
    return t


def _bind(meta, tensors, outputs, direction, device):
    """role -> tensor for every slot of ``direction``; outputs start zeroed."""
    bound = {}
    for role in meta["calls"][direction]["uids"]:
        saved = tensors.get(role)
        if saved is None and role not in OUTPUT_ROLES[direction]:
            saved = outputs.get(role)
        bound[role] = _materialize(meta["tensors"][role], saved, device)
    return bound


def replay_child(directory, out_path, device):
    """Execute every saved direction on hipDNN; write ``{direction: {role: output}}``."""
    import hipdnn_frontend as hipdnn
    from hipdnn_sdpa import _runtime

    _, meta, tensors, outputs = _load(directory)
    stream = torch.cuda.current_stream().cuda_stream if device == "cuda" else 0
    handle = hipdnn.Handle(stream)
    results = {}
    for direction in meta["directions"]:
        call = meta["calls"][direction]
        graph = hipdnn.Graph()
        err = graph.from_json(
            handle, (Path(directory) / f"{direction}.graph.json").read_text()
        )
        if err.is_bad():
            raise RuntimeError(f"from_json: {err.get_message()}")
        _, workspace_size, _ = _runtime._select(
            graph, direction, call["engine"] or None, handle
        )
        bound = _bind(meta, tensors, outputs, direction, device)
        workspace = (
            torch.empty(workspace_size, dtype=torch.uint8, device=device)
            if workspace_size > 0
            else None
        )
        variant_pack = {call["uids"][role]: t.data_ptr() for role, t in bound.items()}
        err = graph.execute(
            handle, variant_pack, workspace.data_ptr() if workspace is not None else 0
        )
        if err.is_bad():
            raise RuntimeError(f"execute: {err.get_message()}")
        if device == "cuda":
            torch.cuda.synchronize()
        results[direction] = {
            role: bound[role].cpu() for role in OUTPUT_ROLES[direction] if role in bound
        }
    torch.save(results, out_path)


# ---------------------------------------------------------------------------- reference


def _reference_inputs(call, tensors):
    import hipdnn_sdpa

    mask = (
        None
        if call["mask_alignment"] is None
        else hipdnn_sdpa.Mask(
            call["mask_alignment"], call["mask_left"], call["mask_right"]
        )
    )
    varlen = None
    if call["padding"]:
        pos = {}
        for side in ("q", "kv"):
            if call[f"packed_{side}"]:
                role = next(
                    r for r in tensors if r.startswith(f"ragged_offset_{side}_")
                )
                pos[side] = tensors[role].reshape(-1) // int(role.rsplit("_", 1)[1])
        varlen = hipdnn_sdpa.Varlen(
            tensors["seq_len_q"].reshape(-1),
            tensors["seq_len_kv"].reshape(-1),
            pos.get("q"),
            pos.get("kv"),
            call["max_sq"],
            call["max_sk"],
        )
    return mask, varlen


def _reference(direction, call, tensors):
    """Reference outputs (FP32 accumulation) for ``direction``; backward through autograd."""
    import aotriton_flash
    import hipdnn_sdpa

    mask, varlen = _reference_inputs(call, tensors)
    q, k, v = (tensors[r].float().requires_grad_() for r in ("q", "k", "v"))
    bias = tensors["bias"].float().requires_grad_() if "bias" in tensors else None
    keep = None
    p = call["dropout_p"]
    if p is not None:
        batch = q.shape[0] if varlen is None else varlen.seq_len_q.shape[0]
        max_sq = q.shape[2] if varlen is None else varlen.max_sq
        max_sk = k.shape[2] if varlen is None else varlen.max_sk
        seed, offset = int(tensors["seed"].item()), int(tensors["offset"].item())
        keep = hipdnn_sdpa.philox_mask(
            batch, q.shape[1], max_sq, max_sk, p, seed, offset, "cpu"
        )
    o = torch.zeros(q.shape, dtype=torch.float32)
    stats = torch.zeros(*q.shape[:3], 1, dtype=torch.float32)
    aotriton_flash._reference_forward(
        q, k, v, bias, call["scale"], mask, keep, p, o, stats, varlen
    )
    if direction == "fwd":
        return {"o": o.detach(), "stats": stats.detach()}
    leaves = [q, k, v] + ([bias] if bias is not None and "dbias" in tensors else [])
    grads = torch.autograd.grad(o, leaves, tensors["do"].float(), allow_unused=True)
    names = ["dq", "dk", "dv"] + (["dbias"] if len(leaves) == 4 else [])
    return dict(zip(names, grads))


def _max_error(a, b):
    return (a.float() - b.float().reshape(a.shape)).abs().max().item()


def report(directory, results):
    _, meta, tensors, outputs = _load(directory)
    last = meta["directions"][-1]
    for direction in meta["directions"]:
        call = meta["calls"][direction]
        print(
            f"{direction}: engine {call['engine'] or '(heuristic)'} kernel {call['kernel'] or '-'}"
        )
        try:
            reference = _reference(direction, call, tensors)
        except (
            Exception
        ) as exc:  # the reference is a convenience; the replay stands without it
            reference = {}
            print(f"  reference unavailable: {type(exc).__name__}: {exc}")
        for role, got in results[direction].items():
            saved = outputs.get(role) if direction == last else tensors.get(role)
            if saved is None:
                print(f"  {role}: no saved hipDNN output")
            else:
                print(
                    f"  {role}: max |replay - saved hipDNN| = {_max_error(got, saved):.6g}"
                )
            if role in reference:
                print(
                    f"  {role}: max |replay - reference|   = {_max_error(got, reference[role]):.6g}"
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--child", metavar="OUT", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        try:
            replay_child(args.directory, args.child, args.device)
        except Exception:
            traceback.print_exc()
            return SETUP_ERROR_EXIT
        return 0
    with tempfile.TemporaryDirectory() as scratch:
        out_path = os.path.join(scratch, "results.pt")
        status = subprocess.run(
            [
                sys.executable,
                __file__,
                args.directory,
                "--device",
                args.device,
                "--child",
                out_path,
            ]
        ).returncode
        if status < 0:
            print(f"reproduced: died with signal {-status}")
            return REPRODUCED_EXIT
        if status == SETUP_ERROR_EXIT:
            print("replay failed before the graph executed; see the traceback above")
            return SETUP_ERROR_EXIT
        if status != 0:
            print(f"reproduced: exited with status {status}")
            return REPRODUCED_EXIT
        report(args.directory, torch.load(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
