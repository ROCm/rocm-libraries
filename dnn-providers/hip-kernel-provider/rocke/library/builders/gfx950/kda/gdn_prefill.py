#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host driver + fp64 parity for the GDN mode of the KDA chunkwise prefill kernel.

GDN is KDA with a scalar forget-gate. This targets the **split (raw-prep) path**:
the raw prep kernel fuses q/k L2-norm, the GDN softplus gate, and beta=sigmoid, so
inputs here are RAW (un-cooked). The oracle cooks them identically and expands q/k
from Hk key-heads to Hv value-heads (value head vh reads key head vh // kv_group).

Run (on a gfx950 box)::

    python gdn_prefill.py            # parity sweep, split path, MHA + GQA
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys

try:
    import rocke  # noqa: F401
except ImportError:  # running as a bare script outside the editable install
    _HERE = os.path.dirname(__file__)
    _RK = os.path.abspath(os.path.join(_HERE, "../../../.."))
    sys.path[:0] = [_RK + "/library", _RK + "/platform/python"]

import numpy as np  # noqa: E402
from rocke.core.dtypes import dtype_info  # noqa: E402
from rocke.helpers.activations import LOG2E  # noqa: E402
from rocke.numeric import storage  # noqa: E402
from rocke.numeric.gated_delta import (  # noqa: E402
    gated_delta_recurrence,
    gdn_log_decay,
    l2_normalize,
    sigmoid,
)
from rocke.runtime.device_array import DeviceArray  # noqa: E402
from rocke.runtime.hip_module import get_device_count  # noqa: E402

from builders.common.kda_host import initial_state, score  # noqa: E402
from kernels.gfx950.kda_chunkwise import (  # noqa: E402
    EXP2_CLAMP,
    KdaChunkScanSpec,
    KdaTileSpec,
)

if __package__:
    from . import kda_chunk_prep as prep_mod  # noqa: E402
    from . import kda_chunk_split as split_mod  # noqa: E402
    from .kda_chunk_fused import device_state, read_state  # noqa: E402
else:
    import kda_chunk_prep as prep_mod  # noqa: E402
    import kda_chunk_split as split_mod  # noqa: E402
    from kda_chunk_fused import device_state, read_state  # noqa: E402

TOL = 3e-2


# The generator cannot see the spec. chunk=32 is the only dispatchable chunk
# today and carries the TIGHTEST limit (a smaller chunk permits more decay), so
# sizing against it keeps every draw inside every reachable bound.
_GENERATOR_CHUNK = 32
# Stay clear of the boundary rather than on it -- see make_gdn_inputs.
_GENERATOR_MARGIN = 0.9


def make_gdn_inputs(B, Hv, Hk, T, DK, DV, gate_low=-0.5, seed=0):
    """RAW GDN host inputs: un-normalized bf16 q/k [B,T,Hk,DK], bf16 v
    [B,T,Hv,DV], raw f32 a/beta [B,T,Hv], f32 a_log/dt_bias [Hv]."""
    rng = np.random.default_rng(seed)
    q = storage.encode(rng.standard_normal((B, T, Hk, DK)), "bf16")
    k = storage.encode(rng.standard_normal((B, T, Hk, DK)), "bf16")
    v = storage.encode(0.2 * rng.standard_normal((B, T, Hv, DV)), "bf16")
    a = storage.encode(gate_low * rng.random((B, T, Hv)), "fp32")
    beta = storage.encode(rng.random((B, T, Hv)), "fp32")
    dt_bias = np.zeros(Hv, dtype=np.float32)
    # Draw the decay rate INSIDE the bound launch_gdn enforces, so no shape can
    # generate an input the guard then refuses. peak = exp(a_log) * softplus(a +
    # dt_bias), and softplus saturates at ln2 for the a <= 0 drawn here, so the
    # rate ceiling is limit / ln2. _GENERATOR_MARGIN keeps the draw clear of the
    # boundary: the guard rejects on peak > limit, and landing exactly on it
    # would make rejection a rounding accident -- intermittent, and only on the
    # GPU-gated shapes where Hv=32 draws enough samples to reach the ceiling.
    #
    # This SUPERSEDES the earlier exp(a_log) <= 8 bound (vault note 2026-09-04,
    # job 265), which was measured where the output leaves a 3e-2 tolerance --
    # the knee, peak ~5.545. The guard promises exactness, not a tolerance, so
    # its bound is the clamp onset (5.4585 at chunk=32) and the generator
    # follows it. Two definitions of "supported" cannot both stand; this is the
    # operative one. randn() alone draws exp>20 in the tail, which made the
    # parity latently flaky as head count grew.
    rate_ceiling = _GENERATOR_MARGIN * decay_limit_for_chunk(_GENERATOR_CHUNK)
    a_log = np.minimum(
        rng.standard_normal(Hv), math.log(rate_ceiling / math.log(2.0))
    ).astype(np.float32)
    return q, k, v, a, beta, a_log, dt_bias


def decay_limit_for_chunk(chunk: int) -> float:
    """Largest per-token decay exponent a chunk of ``chunk`` tokens can carry.

    The kernel does not apply the decay token by token: it factors a whole
    chunk, so the exponents ACCUMULATE and the quantity that can overflow is
    decay x tokens-per-chunk. The midpoint factoring puts the worst case at
    ``chunk / 2`` tokens, and the hardware clamp is ``EXP2_CLAMP`` in base 2,
    hence ``EXP2_CLAMP / (log2(e) * chunk/2)``.

    That gives 5.46 at chunk=32 and 10.92 at chunk=16, measured on gfx950: the
    knee sits at 5.5-5.6 for chunk=32 and 11.5-12.0 for chunk=16, the same
    relationship, so the bound is derived from the clamp rather than fitted.

    The bound is the point the clamp starts to saturate -- where the answer
    stops being exact, NOT where it leaves a tolerance. That is the contract
    this guard enforces.

    ``chunk=16`` is not reachable through the dispatcher today (no
    ``value_splits`` band produces a valid chunk=16 scan spec), so the
    chunk-awareness is currently defensive rather than load-bearing.
    """
    return EXP2_CLAMP / (LOG2E * (chunk / 2.0))


def reject_if_decay_out_of_range(a_log, a, dt_bias, *, chunk):
    """Raise if the intra-chunk decay exponent exceeds what the clamp can carry.

    Past this bound ``exp2`` saturates and the kernel returns a finite, bounded,
    WRONG answer -- no NaN, no fault. Measured on gfx950: at chunk=32 the output
    relative error is 7.6e-3 at decay 5.0 and 0.93 at 6.0, while the final STATE
    error stays near 3e-3, so a serving loop that validates only its carried
    state sees nothing wrong.

    Raises rather than warns: ``warnings.warn`` is filtered by default in most
    serving stacks, and a silent wrong answer is exactly what this exists to
    stop. Cheap host reduction -- call once per problem, not inside a timed
    launch loop. Returns the observed peak decay exponent.
    """
    limit = decay_limit_for_chunk(chunk)
    peak = float(np.abs(gdn_log_decay(a, dt_bias, a_log)).max())
    if peak > limit:
        raise ValueError(
            f"GDN decay exponent peak {peak:.3f} exceeds what a {chunk}-token "
            f"chunk can carry ({limit:.3f}); past this the exp2 clamp saturates "
            f"and the kernel returns a bounded but wrong result"
        )
    return peak


def ref_gdn_raw(q, k, v, a, beta, a_log, dt_bias, scale, kv_group, h0=None):
    """float64 oracle on the stored host inputs: cook like the kernel (q/k
    L2-normalised and rounded to bf16, the scalar softplus gate, beta through
    a sigmoid), expand Hk->Hv, then walk the recurrence token by token.
    Returns ``(o [B,Hv,T,DV], final state [B,Hv,DK,DV])``."""
    Hv = v.shape[2]
    kidx = np.arange(Hv) // kv_group  # value head -> key head
    qn = storage.round_to(l2_normalize(storage.decode(q, "bf16")), "bf16")
    kn = storage.round_to(l2_normalize(storage.decode(k, "bf16")), "bf16")
    gate = gdn_log_decay(a, dt_bias, a_log)[..., None]  # [B,T,Hv,1]

    def head_major(x):
        return np.swapaxes(x, 1, 2)

    return gated_delta_recurrence(
        head_major(qn[:, :, kidx]),
        head_major(kn[:, :, kidx]),
        head_major(storage.decode(v, "bf16")),
        head_major(gate),
        head_major(sigmoid(beta)),
        scale=scale,
        initial_state=h0,
    )


def gdn_specs(DK, DV, kv_group, with_h0):
    """A plain scan spec and the raw GDN prep spec derived from it (flags added)."""
    scan = KdaChunkScanSpec(
        head_k=DK,
        head_v=DV,
        tile=KdaTileSpec(chunk=32),
        value_splits=1,
        token_major_io=True,
        has_initial_state=with_h0,
    )
    prep = dataclasses.replace(
        split_mod.prep_spec_of(scan, raw=True), gate_kind="gdn", kv_group=kv_group
    )
    return scan, prep


def _validate_prefill_inputs(scan, prep, q, k, v, a, beta, a_log, dt_bias, h0):
    """Reject tensors whose shape or dtype disagrees with the spec.

    Every address this kernel computes comes from SPEC constants -- head counts,
    head dims, chunk -- never from the tensors' own shapes, and it emits no
    buffer descriptor, so there is no ``num_records`` to clamp an over-reach. A
    tensor narrower than the spec is read past its end with nothing to catch it.

    dtype matters as much as shape and no device-side check can see it: bf16 and
    f16 are both 16 bits, so a mismatched array computes every address
    identically and merely decodes the bytes under the wrong rule. Mirrors
    ``validate_inputs`` in the GDN decode driver. The inputs are host arrays in
    their storage types (bf16 as ``uint16`` bits); shape and dtype only, so it
    is cheap to call on every launch.
    """
    # Take the head DIMS from the spec, not from the tensors: the kernel is
    # compiled against spec constants, so "what the tensor says" is the thing
    # under test, not the reference. Deriving DK/DV from q/v would make their
    # rows compare each tensor to itself -- a check that cannot fire.
    #
    # B, T and Hv have no spec source (the request carries them, the spec does
    # not), so they come from the tensors and the rows below check every OTHER
    # tensor agrees with them.
    B, T = q.shape[0], q.shape[1]
    Hk, Hv = q.shape[2], v.shape[2]
    DK, DV = prep.head_k, scan.head_v
    io_dt = storage.host_dtype(dtype_info(prep.dtype).name)
    st_dt = np.dtype(np.float32)
    want = {
        "q": ((B, T, Hk, DK), io_dt),
        "k": ((B, T, Hk, DK), io_dt),
        "v": ((B, T, Hv, DV), io_dt),
        "a": ((B, T, Hv), st_dt),
        "beta": ((B, T, Hv), st_dt),
        "a_log": ((Hv,), st_dt),
        "dt_bias": ((Hv,), st_dt),
    }
    got = dict(q=q, k=k, v=v, a=a, beta=beta, a_log=a_log, dt_bias=dt_bias)
    for name, (want_shape, want_dt) in want.items():
        t = got[name]
        if tuple(t.shape) != want_shape:
            raise ValueError(
                f"{name} must be {want_shape} to agree with q/v; got "
                f"{tuple(t.shape)} -- the kernel addresses it from spec "
                f"constants and would read past its end"
            )
        if t.dtype != want_dt:
            raise ValueError(
                f"{name} dtype {t.dtype} != {want_dt}; the kernel is compiled "
                f"with a {want_dt} pointer and would reinterpret these bytes"
            )
    if h0 is not None and tuple(h0.shape) != (B, Hv, DK, DV):
        raise ValueError(f"h0 must be {(B, Hv, DK, DV)}; got {tuple(h0.shape)}")
    if prep.kv_group != max(Hv // Hk, 1):
        raise ValueError(
            f"kv_group {prep.kv_group} disagrees with the tensors "
            f"(Hv {Hv} / Hk {Hk}); the GQA gather would read the wrong key-head"
        )


class GdnPrefillLaunch:
    """One raw split GDN prefill problem, checked and uploaded: heads=Hv (value
    heads); q/k carry Hk key-heads. :meth:`run` launches the prep and scan
    kernels (repeatable: the scan rewrites the outputs from the inputs), and
    :meth:`results` reads the outputs back.

    Construction rejects what the kernels cannot run correctly, before
    anything is uploaded: a sequence that is not a whole number of chunks,
    inputs that disagree with the specs (:func:`_validate_prefill_inputs`), and
    a decay steeper than the chunk can carry (past that bound the exp2 clamp
    saturates and the kernel returns a bounded, wrong answer that the final
    state does not reveal).
    """

    def __init__(self, scan, prep, q, k, v, a, beta, a_log, dt_bias, h0=None):
        B, T, _, DK = q.shape
        Hv = v.shape[2]
        C = scan.tile.chunk
        # T % C != 0 would silently process floor(T/C) chunks and return the
        # tail rows of `o` uninitialised -- wrong numbers, no fault. The
        # dispatcher rejects this (dispatch/gdn/prefill_common.py), but callers
        # reaching the builder directly bypass that, so the launch boundary
        # enforces it too.
        if T % C:
            raise ValueError(
                f"seqlen {T} must be a multiple of chunk {C}; "
                f"this family has no varlen path"
            )
        _validate_prefill_inputs(scan, prep, q, k, v, a, beta, a_log, dt_bias, h0)
        reject_if_decay_out_of_range(a_log, a, dt_bias, chunk=C)
        self.scan, self.prep = scan, prep
        self.batch, self.tokens, self.heads = B, T, Hv
        self.scale = DK**-0.5
        self.num_chunks = T // C
        self.tiles = prep_mod.alloc_tiles(B * Hv * self.num_chunks, prep)
        self.q, self.k, self.v = (DeviceArray(x, prep.dtype) for x in (q, k, v))
        self.a, self.beta = DeviceArray(a), DeviceArray(beta)
        self.a_log, self.dt_bias = DeviceArray(a_log), DeviceArray(dt_bias)
        self.o = DeviceArray.zeros(v.shape, prep.dtype)
        self.ht = device_state(scan, B, Hv)
        self.h0 = None if h0 is None else device_state(scan, B, Hv, h0)

    def run(self) -> None:
        """Launch the raw prep kernel, then the scan, on the default stream."""
        B, Hv, T, NC = self.batch, self.heads, self.tokens, self.num_chunks
        prep_mod.run_prep(
            self.prep,
            self.q,
            self.k,
            self.a,
            self.beta,
            self.tiles,
            self.scale,
            batch=B,
            heads=Hv,
            tseq=T,
            nc=NC,
            a_log=self.a_log,
            dt_bias=self.dt_bias,
        )
        split_mod.run_scan(
            self.scan,
            self.tiles,
            self.v,
            self.o,
            self.ht,
            B * Hv,
            NC,
            h0=self.h0,
            batch=B,
            heads=Hv,
            tseq=T,
        )

    def results(self):
        """Host ``(o [B,T,Hv,DV] bf16 bits, final state [B,Hv,DK,DV] fp32)``."""
        return self.o.values(), read_state(self.ht, self.batch, self.heads)


def launch_gdn(scan, prep, q, k, v, a, beta, a_log, dt_bias, h0=None):
    """Run one raw split GDN prefill on host inputs (:class:`GdnPrefillLaunch`)
    and return its host results."""
    launch = GdnPrefillLaunch(scan, prep, q, k, v, a, beta, a_log, dt_bias, h0)
    launch.run()
    return launch.results()


def check_gdn(
    B,
    Hv,
    Hk,
    T,
    DK,
    DV,
    gate_low=-0.5,
    with_h0=False,
    seed=0,
    specs=None,
):
    # Hv % Hk != 0 would floor to a kv_group the kernel then strides q/k by,
    # reading past the end of a tensor that has fewer key-heads than the stride
    # implies -- an out-of-bounds read, not a wrong answer. Hv=6/Hk=4 gives
    # kv_group=1 and visits khead up to 5 where 3 is the maximum.
    if Hv % Hk:
        raise ValueError(
            f"num_v_heads {Hv} must be a multiple of num_k_heads {Hk}; "
            f"GQA gathers key-head head//kv_group"
        )
    kv_group = Hv // Hk
    q, k, v, a, beta, a_log, dt_bias = make_gdn_inputs(
        B, Hv, Hk, T, DK, DV, gate_low=gate_low, seed=seed
    )
    h0 = initial_state(B, Hv, DK, DV) if with_h0 else None
    scan, prep = gdn_specs(DK, DV, kv_group, with_h0) if specs is None else specs
    o, ht = launch_gdn(scan, prep, q, k, v, a, beta, a_log, dt_bias, h0=h0)
    o_ref, s_ref = ref_gdn_raw(
        q, k, v, a, beta, a_log, dt_bias, DK**-0.5, kv_group, h0=h0
    )
    out = np.swapaxes(storage.decode(o, "bf16"), 1, 2)
    return score((("out", out, o_ref), ("final_state", ht, s_ref)), TOL, verbose=False)


def main() -> int:
    if not get_device_count():
        print("no HIP device", file=sys.stderr)
        return 2
    worst = 0.0
    for Hv, Hk in ((4, 4), (8, 4)):
        for gate_low in (-0.5, -5.0):
            for with_h0 in (False, True):
                w = check_gdn(
                    2,
                    Hv,
                    Hk,
                    256,
                    128,
                    128,
                    gate_low=gate_low,
                    with_h0=with_h0,
                )
                worst = max(worst, w)
                tag = f"Hv{Hv}/Hk{Hk} gate[{gate_low},0] h0={with_h0}"
                print(f"  {tag:32s} rel={w:.3e} {'PASS' if w <= TOL else 'FAIL'}")
    print(f"worst={worst:.3e} tol={TOL:.1e}")
    return 0 if worst <= TOL else 1


if __name__ == "__main__":
    raise SystemExit(main())
