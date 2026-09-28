# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Freeze one fp32-SDPA ``attention_dense`` oracle into committed ``.npz`` blobs.

Why this exists
---------------
Every numeric test in this tree computes its reference with torch, so on a CI
node without torch the test does not fail -- it *skips*, and a skipped GPU test
is indistinguishable from a passing one. torch is deliberately optional in rocke
(numpy is the only hard runtime dep) and is not installed on the CI test nodes,
so the numeric lanes have never actually run there.

This script moves the torch dependency *offline*. It runs once, here, on a
developer box, and writes the inputs plus the fp32 reference into ``.npz`` files
that are committed to git. The replay test then reconstructs the exact same
comparison with numpy alone. torch stays in the loop as the oracle; it just
stops being a runtime dependency of CI.

The oracle is CPU-computable and arch-independent, so this runs anywhere torch
is importable -- no GPU is required to *generate*, only to *replay*.

``ref_meta["provisional"]``
--------------------------
The artifact layout is only trustworthy once the replay half of the contract --
load these arrays, launch the kernel, compare -- has actually run on the target
architecture. ``ref_meta["provisional"]`` records whether it has.

A plain run writes ``provisional: true``: the arrays, dtypes, layout, and the
inputs/ref split are unvalidated, and a second family's artifact must not be
built on this shape yet. After
[`test_attention_dense_gfx942_replay.py`](../test_attention_dense_gfx942_replay.py)
has passed on real gfx942 silicon (the ``mkm-gfx942`` rocke-slurm lane), rerun
this script with ``--validated`` to clear the flag. Everything else is
byte-identical across the two runs -- the seed is fixed and the oracle is CPU
torch -- so ``--validated`` rewrites the metadata and nothing else.

Shape and byte budget
---------------------
``B=1 S=512 Hq=1 Hkv=1 D=64`` fp16, reference stored as fp32::

    q       1*512*1*64*2 =  65536 B    \\
    k       1*512*1*64*2 =  65536 B     >  inputs blob = 196608 B = 192 KiB
    v       1*512*1*64*2 =  65536 B    /
    ref_out 1*512*1*64*4 = 131072 B    -> ref blob    = 131072 B = 128 KiB

Two blobs, not one. ``S=512`` is not negotiable -- it is the size at which the
dense kernel's grid-stride loop actually iterates, so a smaller sequence would
leave the loop untested and the artifact would gate nothing interesting. At that
sequence length no single file holding both the inputs and an fp32 reference
fits the 256 KiB per-file budget, so the payload is split: 192 KiB + 128 KiB,
each comfortably inside 256 KiB, 320 KiB together against an 8 MiB per-directory
budget. Splitting costs one extra ``np.load`` and keeps both budgets and the
fp32 reference; shrinking the sequence would have cost the coverage the artifact
exists for.

Compression is not what buys the fit: every input is ``torch.randn``, i.e.
near-maximum-entropy IEEE floats, and deflate recovers only a few percent on
those. The shape does.

``Hq == Hkv == 1`` -- so this artifact does **not** exercise the GQA head-
expansion path. That is a deliberate loss. GQA needs ``Hq > Hkv``, and the
smallest such case at ``S=512`` is ``Hq=2, Hkv=1``, whose inputs alone are
exactly 256 KiB before ``.npz`` header overhead -- over budget, and its fp32
reference is another 256 KiB. GQA coverage stays with the torch-based cohort in
[`test_attention_dense_gfx942_numeric.py`](../test_attention_dense_gfx942_numeric.py)
and with the golden-IR tests, neither of which is byte-budgeted.

The reference is stored at **fp32**, not at the input dtype. That is lossless
with respect to what is actually compared -- the assertion is
``|ref - out| < 2e-2`` and fp32 carries ~1e-7 relative error against a 2e-2
band -- while costing half of what fp64 would.

fp16, not bf16: stock numpy has native ``float16`` but no ``bfloat16``, and
``ml_dtypes`` is deliberately not a dependency of the torch-free host path. A
bf16 artifact would reintroduce exactly the optional-dependency problem this
script exists to remove.

Not collected by pytest: ``library/tests/pytest.ini`` sets
``python_files = test_*.py``, so this module's name keeps it out of collection
even though it lives under the test tree.

Usage::

    python library/tests/artifacts/gen_attention_dense_reference.py
    python library/tests/artifacts/gen_attention_dense_reference.py --validated
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

# Bumped whenever the array names, dtypes, layout, the inputs/ref split, or the
# set of ``ref_meta`` keys the replay test relies on changes. The replay test
# asserts on it, so a stale artifact fails loudly instead of being compared
# under the wrong assumptions.
#
# 1 -> 2: added ``tolerance`` (the replay test reads its threshold from the
#         artifact instead of hardcoding one) and made ``provisional`` a bool
#         rather than a free-text string.
FORMAT_VERSION = 2

# The comparison threshold, frozen into the artifact so the replay test has a
# single source of truth for it. bf16 is listed for the day an artifact needs
# it; this generator only emits fp16 (stock numpy has no bfloat16 -- see the
# module docstring).
TOL_BY_DTYPE = {"fp16": 2e-2, "bf16": 4e-2}

# Per-file and per-directory budgets, asserted by the replay test. Both sit
# under pre-commit's ``check-added-large-files`` 500 KB default; rocke is not in
# that hook's opt-out anchor, so an oversized artifact would be rejected at
# commit time rather than at review time.
MAX_FILE_BYTES = 256 * 1024
MAX_DIR_BYTES = 8 * 1024 * 1024

# The frozen config. S=512 is load-bearing -- see the byte budget above.
CONFIG = {
    "dtype": "fp16",
    "batch": 1,
    "seqlen": 512,
    "nhead_q": 1,
    "nhead_kv": 1,
    "head_size": 64,
    "causal": True,
    "persistent": False,
    "arch": "gfx942",
    "seed": 0,
}

_STEM = "attention_dense_gfx942_fp16_b1s512h1x1d64"
_HERE = Path(__file__).resolve().parent
INPUTS_ARTIFACT = _HERE / f"{_STEM}_inputs.npz"
REF_ARTIFACT = _HERE / f"{_STEM}_ref.npz"


def _git_commit() -> str:
    """Best-effort rocke commit this artifact was generated from.

    Provenance only: it tells a future reader which tree produced the reference
    when a regeneration disagrees with the committed one. Never load-bearing, so
    a detached/exported tree degrades to ``"unknown"`` rather than failing.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(_HERE), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _engine_stamp() -> dict:
    """Engine provenance, or ``"unknown"`` when the C++ engine is not built.

    Reports ``"unknown"`` rather than omitting the keys: the difference between
    "generated against an unbuilt engine" and "generated before these fields
    existed" is exactly what a consumer needs when an artifact goes stale.
    """
    try:
        from rocke.helpers.manifest import engine_build_id, engine_version

        return {"engine_build_id": engine_build_id(), "engine_version": engine_version()}
    except Exception:
        return {"engine_build_id": "unknown", "engine_version": "unknown"}


def build(*, validated: bool = False) -> tuple[dict, dict]:
    """Compute the oracle with torch; return ``(inputs, ref)`` array dicts.

    torch is imported inside the function, not at module scope, so importing
    this file to read :data:`CONFIG` or the artifact paths -- which the replay
    test does -- never requires torch.
    """
    import torch
    import torch.nn.functional as F

    c = CONFIG
    b, s, hq, hkv, d = (
        c["batch"],
        c["seqlen"],
        c["nhead_q"],
        c["nhead_kv"],
        c["head_size"],
    )
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(c["seed"])

    # run_attention_dense_torch ABI: q/out [B,S,Hq,D], k/v [B,S,Hkv,D].
    q = torch.randn(b, s, hq, d, dtype=torch.float16)
    k = torch.randn(b, s, hkv, d, dtype=torch.float16)
    v = torch.randn(b, s, hkv, d, dtype=torch.float16)

    # fp32 SDPA oracle in [B,H,S,D] layout, verbatim from
    # test_attention_dense_gfx942_numeric.py. GQA is expanded here by repeating
    # each kv head to its query heads rather than via the ``enable_gqa=`` kwarg:
    # that kwarg is recent, and on an older ROCm torch passing it raises
    # TypeError. ``repeat_interleave`` along the head axis is exactly what
    # ``enable_gqa`` does internally and is the mapping the kernel itself uses
    # (hkv = hq // gqa), so the frozen reference is unchanged by the choice. At
    # this config rep == 1 and the expansion is a no-op -- it is kept so that a
    # future GQA artifact regenerates through the identical code path.
    rep = hq // hkv
    qf = q.transpose(1, 2).float()
    kf = k.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    vf = v.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    ref = F.scaled_dot_product_attention(
        qf, kf, vf, is_causal=c["causal"], scale=scale
    ).transpose(
        1, 2
    )  # -> [B,S,Hq,D]

    meta = dict(c)
    meta["format_version"] = FORMAT_VERSION
    meta["scale"] = scale
    meta["tolerance"] = TOL_BY_DTYPE[c["dtype"]]
    meta["oracle"] = "torch.nn.functional.scaled_dot_product_attention, fp32"
    meta["torch_version"] = torch.__version__
    meta["rocke_commit"] = _git_commit()
    # True until a gfx942 replay has passed against this layout; see the module
    # docstring. Cleared by rerunning with --validated, which changes nothing
    # else in the artifact.
    meta["provisional"] = not validated
    meta.update(_engine_stamp())

    inputs = {
        "q": q.numpy(),
        "k": k.numpy(),
        "v": v.numpy(),
        # A 0-d unicode array, so np.load without allow_pickle can read it. It
        # rides with the inputs because the replay test needs the spec fields to
        # build the kernel before it ever touches the reference.
        "ref_meta": np.array(json.dumps(meta, sort_keys=True)),
    }
    ref_arrays = {
        # fp32, not fp16: this is the reference, and rounding it to the input
        # dtype would fold a 5e-4 quantization into a 2e-2 tolerance check.
        "ref_out": ref.float().numpy(),
    }
    return inputs, ref_arrays


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--validated",
        action="store_true",
        help=(
            "clear ref_meta['provisional'] -- pass this ONLY after the gfx942 "
            "replay test has passed against the artifact this run reproduces"
        ),
    )
    args = ap.parse_args()
    inputs, ref_arrays = build(validated=args.validated)
    total = 0
    for path, arrays in ((INPUTS_ARTIFACT, inputs), (REF_ARTIFACT, ref_arrays)):
        # Uncompressed on purpose: the payload is near-maximum-entropy IEEE
        # floats, so deflate buys a few percent for a slower, less inspectable
        # file. The budget is met by the shape and the split, not by compression.
        np.savez(path, **arrays)
        size = path.stat().st_size
        total += size
        status = "OK" if size <= MAX_FILE_BYTES else "OVER BUDGET"
        print(f"wrote {path.name}: {size} B ({size / 1024:.1f} KiB) [{status}]")
        for name, a in arrays.items():
            if a.ndim:
                print(f"  {name:8s} {str(a.dtype):8s} {a.shape}")
    print(f"total {total} B ({total / 1024:.1f} KiB), per-file budget {MAX_FILE_BYTES} B")


if __name__ == "__main__":
    main()
