# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free gfx942 replay of the frozen ``attention_dense`` reference.

This is the functional GPU-CI lane. Every other numeric test in this tree
computes its reference with torch, so on a CI node without torch it does not
fail -- it *skips*, and a skipped GPU test is indistinguishable from a passing
one. torch is optional in rocke and is not installed on the CI test nodes, so
those lanes have never actually run there.

Here the oracle is precomputed offline by
[`artifacts/gen_attention_dense_reference.py`](artifacts/gen_attention_dense_reference.py)
and committed as ``.npz``. **This module imports numpy and nothing else that is
optional** -- no torch on any path, including the launch path, which goes
through ``DeviceMem`` + ``Runtime.memcpy_*`` (ctypes over libamdhip64) rather
than through ``run_attention_dense_torch``.

What this file asserts, and why each one is load-bearing:

* **The artifact is the one this test was written against.** ``format_version``
  and the array shapes are checked before anything is compared. A regenerated
  artifact with a different layout must fail loudly, not be silently compared
  under the old assumptions.
* **The tolerance comes from ``ref_meta``**, not from a literal here. Two copies
  of a threshold drift; one of them then gates nothing.
* **The resolved LLVM flavor**, via
  :func:`rocke.core.lower_llvm.resolve_llvm_flavor`, not the env var.
  ``ROCKE_LLVM_FLAVOR`` fails *open*: an unrecognised value silently falls back
  to autodetection, so the env var and the bytes actually emitted can disagree.
  Only the resolved value catches that.
* **The artifact directory stays inside its byte budget** -- and that check is
  deliberately *not* GPU-gated, so it runs on every CPU box too.

Run standalone::

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_attention_dense_gfx942_replay.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ._archgate import requires_gfx942_gpu

# Bump in lockstep with FORMAT_VERSION in the generator. Asserted, not read:
# the point is to notice when the artifact moved and this file did not.
EXPECTED_FORMAT_VERSION = 2

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_STEM = "attention_dense_gfx942_fp16_b1s512h1x1d64"
INPUTS_ARTIFACT = ARTIFACT_DIR / f"{_STEM}_inputs.npz"
REF_ARTIFACT = ARTIFACT_DIR / f"{_STEM}_ref.npz"

# Mirrors the generator's budget constants. Duplicated rather than imported so
# the budget test does not depend on a module that imports torch in one of its
# functions -- and so a generator edit that relaxes the budget does not silently
# relax the assertion that is supposed to catch it.
MAX_FILE_BYTES = 256 * 1024
MAX_DIR_BYTES = 8 * 1024 * 1024

# numpy dtype per artifact dtype tag. bf16 is absent on purpose: stock numpy has
# no bfloat16 and ``ml_dtypes`` is not a dependency of this torch-free path, so
# a bf16 artifact must fail here rather than pull one in.
_NP_DTYPE = {"fp16": np.float16}


def _load_artifact():
    """``(meta, q, k, v, ref_out)``, validated against what this test expects.

    Validation happens here, once, before any kernel is built: a shape or
    version mismatch is a broken *artifact*, and diagnosing it from a numeric
    assertion failure several hundred milliseconds of compilation later is
    strictly worse than failing at load.
    """
    for path in (INPUTS_ARTIFACT, REF_ARTIFACT):
        if not path.exists():
            raise AssertionError(
                f"missing reference artifact {path.name}; regenerate with "
                f"`python {ARTIFACT_DIR.name}/gen_attention_dense_reference.py`"
            )

    # allow_pickle stays off: these files are committed to git and loaded by CI,
    # so they must remain plain data. ref_meta is a 0-d unicode array precisely
    # to keep that true.
    inputs = np.load(INPUTS_ARTIFACT, allow_pickle=False)
    ref = np.load(REF_ARTIFACT, allow_pickle=False)

    meta = json.loads(str(inputs["ref_meta"]))
    got_version = meta.get("format_version")
    assert got_version == EXPECTED_FORMAT_VERSION, (
        f"{INPUTS_ARTIFACT.name} is format_version {got_version!r}, this test "
        f"expects {EXPECTED_FORMAT_VERSION}. The artifact layout moved: reread "
        f"gen_attention_dense_reference.py and update this module rather than "
        f"relaxing the check."
    )

    assert meta["dtype"] in _NP_DTYPE, (
        f"artifact dtype {meta['dtype']!r} has no numpy dtype on the torch-free "
        f"path (supported: {sorted(_NP_DTYPE)})"
    )
    np_dt = _NP_DTYPE[meta["dtype"]]

    b, s = meta["batch"], meta["seqlen"]
    hq, hkv, d = meta["nhead_q"], meta["nhead_kv"], meta["head_size"]
    want_q = (b, s, hq, d)
    want_kv = (b, s, hkv, d)

    q, k, v, ref_out = inputs["q"], inputs["k"], inputs["v"], ref["ref_out"]
    for name, arr, want, want_dt in (
        ("q", q, want_q, np_dt),
        ("k", k, want_kv, np_dt),
        ("v", v, want_kv, np_dt),
        ("ref_out", ref_out, want_q, np.float32),
    ):
        assert arr.shape == want and arr.dtype == want_dt, (
            f"{name} in the artifact is {arr.shape}/{arr.dtype}, but ref_meta "
            f"describes {want}/{np.dtype(want_dt)} -- the arrays and their "
            f"metadata disagree, so the artifact is corrupt"
        )

    return meta, q, k, v, ref_out


def _spec_from_meta(meta):
    """The SHIPPED gfx942 dense spec for the artifact's config.

    Built through the dispatch factory, never hand-rolled: hand-rolling pins
    every tuned lever to the shared dataclass default, so the lane would assert
    on a binary that does not ship (``waves_per_eu`` alone makes it a different
    kernel). Same reasoning as ``_spec`` in
    ``test_attention_dense_gfx942_numeric.py``.
    """
    from dispatch.attention import AttentionRequest
    from dispatch.attention.gfx942 import _dense_spec

    return _dense_spec(
        AttentionRequest(
            batch=meta["batch"],
            nhead_q=meta["nhead_q"],
            nhead_k=meta["nhead_kv"],
            seqlen_q=meta["seqlen"],
            seqlen_k=meta["seqlen"],
            hdim_q=meta["head_size"],
            hdim_v=meta["head_size"],
            arch="gfx942",
            mask_type=1 if meta["causal"] else 0,
            dtype=meta["dtype"],
            sliding_window=0,
            algorithm="attention_dense",
            dense_persistent="on" if meta["persistent"] else "off",
        )
    )


def _run_on_device(spec, q, k, v, *, scale):
    """Compile, launch on device 0, and return the output as a numpy array.

    The torch-free twin of ``run_attention_dense_torch``: same spec, same
    signature, same grid/block helpers, but host<->device I/O goes through
    ``DeviceMem`` and ``Runtime.memcpy_*`` (ctypes over libamdhip64). Imports
    are function-local so that collecting this module on a CPU box -- where the
    device gate skips it -- never needs comgr or a HIP runtime.

    ``fence=True`` makes the launch stream-synchronize before returning, so the
    D2H readback observes a finished kernel.
    """
    from kernels.gfx942.attention_dense import (
        attention_dense_block,
        attention_dense_grid,
        attention_dense_signature,
        build_attention_dense,
        supports_attention_dense,
    )
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import DeviceMem, KernelLauncher, LaunchConfig

    ok, why = supports_attention_dense(spec, arch="gfx942")
    assert ok, f"the frozen artifact's config is not supported on gfx942: {why}"

    art = compile_kernel(
        build_attention_dense(spec, arch="gfx942"),
        arch="gfx942",
        backend="python",
        capture_ir_text=False,
    )
    assert art.kernel_name == spec.kernel_name(), (art.kernel_name, spec.kernel_name())

    out = np.zeros_like(q)
    rt = Runtime()
    q_dev = DeviceMem(q.nbytes)
    k_dev = DeviceMem(k.nbytes)
    v_dev = DeviceMem(v.nbytes)
    o_dev = DeviceMem(out.nbytes)
    rt.memcpy_h2d(q_dev.ptr(), as_u8_buffer(q), q.nbytes)
    rt.memcpy_h2d(k_dev.ptr(), as_u8_buffer(k), k.nbytes)
    rt.memcpy_h2d(v_dev.ptr(), as_u8_buffer(v), v.nbytes)
    rt.memset(o_dev.ptr(), 0, out.nbytes)

    vals = {
        "q_ptr": q_dev,
        "k_ptr": k_dev,
        "v_ptr": v_dev,
        "o_ptr": o_dev,
        "scale": float(scale),
    }
    # Mirrors run_attention_dense_torch: on the runtime-shape path these three
    # are kernel params rather than baked constants, and omitting them would
    # pack a short kernarg buffer.
    if spec.runtime_shape:
        vals["batch"] = int(spec.batch)
        vals["seqlen_q"] = int(spec.seqlen_q)
        vals["seqlen_kv"] = int(spec.seqlen_kv)

    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=attention_dense_signature(spec),
    )
    launcher(
        vals,
        config=LaunchConfig(
            grid=attention_dense_grid(spec),
            block=attention_dense_block(spec),
            fence=True,
        ),
    )
    rt.memcpy_d2h(as_u8_buffer(out), o_dev.ptr(), out.nbytes)
    return out


@requires_gfx942_gpu
@pytest.mark.gpu
def test_dense_replay_matches_frozen_reference():
    """Launch the shipped gfx942 dense kernel on the frozen inputs and compare.

    This is the assertion the whole lane exists for, and the one a deliberate
    kernel perturbation must break.
    """
    meta, q, k, v, ref_out = _load_artifact()

    # The flavor actually in force, not the env var -- see the module docstring.
    from rocke.core.lower_llvm import LLVM_FLAVORS, resolve_llvm_flavor

    flavor = resolve_llvm_flavor()
    assert flavor in LLVM_FLAVORS, (
        f"resolved LLVM flavor {flavor!r} is not one of {LLVM_FLAVORS}; the "
        f"engine would emit IR this tree does not know how to gate"
    )
    env_flavor = os.environ.get("ROCKE_LLVM_FLAVOR", "").strip().lower()
    if env_flavor:
        # ROCKE_LLVM_FLAVOR fails open on an unrecognised value. If the caller
        # asked for a flavor and got a different one, the run is not testing
        # what the caller thinks it is.
        assert flavor == env_flavor, (
            f"ROCKE_LLVM_FLAVOR={env_flavor!r} but the engine resolved "
            f"{flavor!r} -- resolution fell back to autodetection instead of "
            f"honouring the request"
        )

    tol = meta["tolerance"]
    out = _run_on_device(_spec_from_meta(meta), q, k, v, scale=meta["scale"])

    max_abs = float(np.abs(ref_out.astype(np.float32) - out.astype(np.float32)).max())
    assert max_abs < tol, (
        f"attention_dense gfx942 replay drifted from the frozen reference: "
        f"max_abs={max_abs:.3e} >= tol={tol:.3e} "
        f"({meta['dtype']} B{meta['batch']} S{meta['seqlen']} "
        f"Hq{meta['nhead_q']}/Hkv{meta['nhead_kv']} D{meta['head_size']}, "
        f"causal={meta['causal']}, flavor={flavor}, "
        f"artifact from rocke commit {meta.get('rocke_commit', 'unknown')})"
    )


def test_artifact_dir_within_byte_budget():
    """The artifact directory stays inside its budget, in CI and not only at commit.

    ``check-added-large-files`` in pre-commit catches an oversized *new file* on
    the machine that commits it. It does not catch a directory that grew to
    hundreds of small ``.npz`` blobs, and it does not run in CI at all. File
    count is the long-term risk here; this asserts the total that count feeds.

    Not GPU-gated on purpose -- a budget regression should be caught on every
    CPU box, not only on the one runner that has a gfx942.
    """
    assert ARTIFACT_DIR.is_dir(), f"{ARTIFACT_DIR} is missing"

    sizes = {p.name: p.stat().st_size for p in ARTIFACT_DIR.rglob("*") if p.is_file()}
    oversized = {
        n: s for n, s in sizes.items() if n.endswith(".npz") and s > MAX_FILE_BYTES
    }
    assert not oversized, (
        f"artifact(s) over the {MAX_FILE_BYTES} B per-file budget: {oversized}. "
        f"Shrink the config or split the payload; do not raise the budget."
    )

    total = sum(sizes.values())
    assert total <= MAX_DIR_BYTES, (
        f"{ARTIFACT_DIR.name}/ is {total} B across {len(sizes)} files, over the "
        f"{MAX_DIR_BYTES} B directory budget. Reference artifacts are committed "
        f"to git; prune before raising this."
    )
