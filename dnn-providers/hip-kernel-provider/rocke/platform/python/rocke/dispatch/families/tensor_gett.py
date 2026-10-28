# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tensor contraction (GETT) dispatcher family, f16 / bf16 on gfx950.

A request carries one :class:`rocke.instances.common.tensor.TensorGettSpec`
problem (mode lists, extents, strides, dtype, ``bilinear``) and the target.
Two candidates are registered, in preference order:

* ``tensor_gett_mn`` -- the native GETT on the GEMM dispatcher's own body for
  the same ``M x N x K`` with both operands MN-contiguous (layout ``CRR``):
  the dispatcher pick first, then the 256x256x32 AGPR interleave ring. The
  GETT hook replaces only the operand addressing (per-lane mode offsets) and
  the epilogue (permuted alpha / beta, 16-byte ``v_permlane32_swap`` stores
  when D's fastest run allows), so the contraction runs on whatever body the
  GEMM dispatcher currently picks. Needs 16-byte operand runs.
* ``tensor_gett_t32s3`` -- the ring-body GETT (its own K-outer LDS layout,
  4-byte DTL lanes for operand runs that are not a multiple of 8) with
  ``tile_k`` 32 and three direct-to-LDS stages.

Problems the GETT planner cannot map (batch / Hadamard modes, an operand whose
fastest free run is odd) are not admitted; a caller falls back to a permute +
GEMM composition for those. K modes that do not split into the K tile are
planned with one in-tile K mode padded (padded rows read zeros).

:func:`plan_tensor_gett` is the product route. It compares, with the GEMM
dispatcher's time model (``dispatch.gemm.plan.predicted_us``) plus a pack cost
model:

* ``single``: one MN-layout GETT launch (the operands read in place);
* ``pack_hybrid``: each operand whose fastest free run is not 16-byte
  DTL-able is first copied by a ``tensor_pack`` kernel so its free modes form
  one contiguous run, then the MN-layout GETT reads the copy;
* ``kc``: both operands are packed K-contiguous ([rows, K]) and the GETT runs
  on the GEMM dispatcher's RCR pick for the same problem (the row-interleave
  / hb_sched bodies, whose vector epilogue then writes the permuted D through
  the GETT epilogue hook).

The ring-body candidate serves what none of these admits.

Autotune mode: :func:`plan_tensor_gett_candidates` lists the candidate routes
of a problem (the cost-model route, the MN-layout GETT on the selector's best
both-MN-contiguous bodies, one-step body mutations, the K-contiguous route on
its best RCR bodies) and :func:`tune_tensor_gett` measures them with the
search engine. Measured winners ship in the tuned route table
(:data:`TENSOR_GETT_TUNED`, sidecar ``tensor_gett_tuned.json``), which
:func:`plan_tensor_gett` consults before the cost model.

The candidate's ``sweep_space`` is its knob neighbourhood (K tile, stage
count, persistent grid, paired 16-byte epilogue stores), for the tuning
engine to search.

AOT route table: :data:`TENSOR_GETT_AOT` (sidecar ``tensor_gett_aot.json``)
freezes the product route (route and GEMM body) of fixed problems for
ahead-of-time kernel packs; :func:`aot_route_plan` rebuilds a plan from an
entry, and :func:`plan_tensor_gett` never reads it
(``tools/gen_tensor_gett_aot.py`` writes and refreshes it).

The C++ engine mirrors the route key and rebuilds the launch programs of both
tables (``rocke::gett::table_plan``: single-launch MN, the pack hybrid, and the
K-contiguous route with its operand packs; the tables are compiled into the
engine); the ``tensor_gett_aot`` / ``tensor_gett_kc`` / ``tensor_pack`` parity
families keep the two engines byte-identical. The cost-model routes and the
tuning flow are Python engine only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Optional

from ...instances.common.tensor import (
    TENSOR_GETT_ARCHS,
    TENSOR_GETT_DTYPES,
    TensorGettSpec,
    build_tensor_gett,
    is_valid_spec,
    tensor_gett_block,
    tensor_gett_grid,
    tensor_gett_signature,
)
from ..core import (
    CandidateRegistry,
    Capability,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    selector_matches,
    stable_json_hash,
)

_FAMILY = "tensor_gett"
TENSOR_GETT_ABI_VERSION = "rocke-tensor-gett/v1"


@dataclass(frozen=True)
class TensorGettRequest(OperatorRequest):
    """One f16 / bf16 GETT launch. ``problem`` carries the modes, extents,
    strides, dtype and ``knobs.bilinear``; its body levers are ignored (the
    candidate picks them)."""

    problem: TensorGettSpec
    arch: str
    op: str = "contraction"
    algorithm: str = "auto"
    spec_id: str = "auto"

    @property
    def dtype(self) -> str:
        return "fp16" if self.problem.dtype in ("f16", "fp16") else self.problem.dtype

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = self.dtype
        return d

    def dims(self) -> dict[str, int]:
        ext = dict(self.problem.extents)
        p = self.problem
        size = lambda ms: _prod(ext[m] for m in ms)
        k_modes = [m for m in p.modes_a if m in p.modes_b and m not in p.modes_d]
        m_modes = [m for m in p.modes_d if m in p.modes_a]
        n_modes = [m for m in p.modes_d if m in p.modes_b and m not in p.modes_a]
        return {"M": size(m_modes), "N": size(n_modes), "K": size(k_modes)}


def _prod(xs) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


TENSOR_GETT_DIM_VOCABULARY = ("M", "N", "K")


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, TensorGettRequest):
        return [f"expected TensorGettRequest, got {type(req).__name__}"]
    errors = []
    if req.op != "contraction":
        errors.append(f"unsupported op {req.op!r}")
    if req.dtype not in TENSOR_GETT_DTYPES:
        errors.append(f"unsupported dtype {req.dtype!r}")
    return errors


def _spec_t32s3(req: TensorGettRequest) -> TensorGettSpec:
    return replace(
        req.problem,
        dtype=req.dtype,
        tile_k=32,
        dtl_stages=3,
        persistent_ctas=0,
        knobs=replace(req.problem.knobs, bilinear=req.problem.bilinear),
    )


#: MN-layout candidate: the 16-byte permlane epilogue where D's fastest run
#: allows it (False: the 8-byte epilogue).
TENSOR_GETT_MN_EPI_V16 = True
#: MN-layout candidate, bilinear: C loads issued this many atom rows ahead
#: (2: hides the C load latency behind the epilogue math on the bilinear
#: doc rows; 0 issues each load where it is consumed).
TENSOR_GETT_MN_C_PREFETCH = 2
#: K-contiguous candidate, bilinear: C loads issued this many rows ahead in
#: the vector epilogue hook (8: the C load latency overlaps the alpha / beta
#: math of the previous rows; measured faster than 0, 2 and 4 on the
#: bilinear canonical rows this route serves; 0 issues each load where it is
#: consumed).
TENSOR_GETT_KC_C_PREFETCH = 8
#: K-contiguous candidate on a vector-epilogue body: D's contiguous run on
#: the column side must be at least this long, unless K is at least
#: :data:`TENSOR_GETT_KC_SHORT_RUN_MIN_K`. A wave's store instruction covers
#: 16 lanes x 8 consecutive columns; shorter runs scatter it, which makes
#: this route slower than the MN-layout GETT on permuted-D rows unless the
#: run is long or K is large enough for the main loop to dominate.
TENSOR_GETT_KC_MIN_D_RUN = 128
TENSOR_GETT_KC_SHORT_RUN_MIN_K = 4096
#: MN-layout candidate: run the 256x256 bodies persistent (one workgroup per
#: CU walking the tiles) when the grid has at least this many tiles per CU
#: (helps at short K, where the per-tile prologue is a larger share; neutral
#: at long K); 0: the dispatcher pick's grid.
TENSOR_GETT_MN_PERSISTENT_MIN_WAVES = 2
#: MN-layout candidate: prefer a body whose K tile splits the K modes exactly
#: over an earlier body that needs a padded K mode (False: the first valid
#: body, padded or not). Used when the GEMM time model has no prediction;
#: otherwise the body with the least predicted time for its (padded) K wins
#: (:data:`TENSOR_GETT_MN_BY_MODEL`).
TENSOR_GETT_MN_PREFER_EXACT_K = True
#: MN-layout candidate: choose among the valid bodies by the GEMM time model
#: (padded K included); False: the first valid body (exact K preferred).
TENSOR_GETT_MN_BY_MODEL = True


def _mn_bodies(req: TensorGettRequest) -> list:
    """(tile, trait) of the both-MN-contiguous GEMM bodies to try, in order:
    the GEMM dispatcher's CRR pick for the problem's M x N x K, then the
    256x256x32 AGPR interleave ring (one workgroup per tile)."""
    from ..gemm import GemmRequest, dispatch_gemm_bf16, dispatch_gemm_fp16
    from ..gemm.common import agpr_ilv_big_grid_tile_trait

    p = req.problem
    try:
        plan = replace(p, tile_k=32, mn_tile=None, mn_trait=None).plan()
    except ValueError:
        return []
    out = []
    disp = dispatch_gemm_bf16 if req.dtype == "bf16" else dispatch_gemm_fp16
    try:
        r = disp(
            GemmRequest(
                M=plan.M,
                N=plan.N,
                K=plan.K,
                arch=req.arch,
                dtype=req.dtype,
                layout="CRR",
                trans_a=True,
                trans_b=False,
            )
        )
        if hasattr(r.spec, "tile") and hasattr(r.spec, "trait"):
            out.append((r.spec.tile, r.spec.trait))
    except (ValueError, KeyError):
        pass
    try:
        tile, trait = agpr_ilv_big_grid_tile_trait(req.arch)
        out.append((tile, replace(trait, persistent_ctas=0)))
    except ValueError:
        pass
    return out


def _mn_persistent(req: TensorGettRequest, tile, trait):
    """The MN body trait with the persistent tile walk where it pays (see
    :data:`TENSOR_GETT_MN_PERSISTENT_MIN_WAVES`)."""
    from ...helpers.split_k import arch_num_cus

    waves = int(TENSOR_GETT_MN_PERSISTENT_MIN_WAVES)
    if not waves or trait.persistent_ctas or (tile.tile_m, tile.tile_n) != (256, 256):
        return trait
    try:
        plan = replace(req.problem, tile_k=32, mn_tile=None, mn_trait=None).plan()
        cus = int(arch_num_cus(req.arch))
    except (ValueError, KeyError):
        return trait
    tiles = -(-plan.M // 256) * -(-plan.N // 256)
    if tiles < waves * cus:
        return trait
    return replace(trait, persistent_ctas=cus)


def _spec_mn(req: TensorGettRequest, bodies=None):
    """The first valid MN-layout GETT spec over :func:`_mn_bodies` (with the
    16-byte epilogue when it applies), or None. ``bodies``: explicit
    ``(tile, trait)`` pairs to try instead (used as given, no persistent
    rewrite; the candidate routes of :func:`plan_tensor_gett_candidates`)."""
    p = req.problem
    valid = []
    for tile, trait in _mn_bodies(req) if bodies is None else bodies:
        if bodies is None:
            trait = _mn_persistent(req, tile, trait)
        kn = replace(
            p.knobs,
            bilinear=p.bilinear,
            c_prefetch=TENSOR_GETT_MN_C_PREFETCH if p.bilinear else 0,
        )
        sp = replace(
            p,
            dtype=req.dtype,
            tile_k=tile.tile_k,
            dtl_stages=3,
            persistent_ctas=0,
            chiplet_wgm=8,
            chiplet_chunk_size=64,
            mn_tile=tile,
            mn_trait=trait,
            knobs=kn,
        )
        got = None
        if TENSOR_GETT_MN_EPI_V16:
            v16 = replace(sp, knobs=replace(kn, epi_v16=True))
            if is_valid_spec(v16, arch=req.arch)[0]:
                got = v16
        if got is None and is_valid_spec(sp, arch=req.arch)[0]:
            got = sp
        if got is None:
            continue
        valid.append(got)
    if not valid:
        return None
    if TENSOR_GETT_MN_BY_MODEL and len(valid) > 1:
        timed = []
        for s in valid:
            pl = s.plan()
            t = _predicted(_gemm_req(req, pl.M, pl.N, pl.K, "CRR"), s.body())
            if t is None:
                timed = None
                break
            timed.append((t, s))
        if timed:
            return min(timed, key=lambda x: x[0])[1]
    for s in valid:
        if not TENSOR_GETT_MN_PREFER_EXACT_K or s.plan().k_pad is None:
            return s
    return valid[0]


def _neighbourhood(spec: TensorGettSpec) -> tuple[TensorGettSpec, ...]:
    """The default spec, then one-lever moves the tuner starts from."""
    out = [spec]
    if spec.mn_body:
        tr = spec.mn_trait
        moves = [
            {"knobs": replace(spec.knobs, epi_v16=not spec.knobs.epi_v16)},
            {"mn_trait": replace(tr, persistent_ctas=0 if tr.persistent_ctas else 256)},
            {"mn_trait": replace(tr, mn_pad=0 if tr.mn_pad else 32)},
            {"mn_trait": replace(tr, chiplet_wgm=8 if tr.chiplet_wgm != 8 else 4)},
        ]
        if spec.bilinear:
            moves.append({"knobs": replace(spec.knobs, c_prefetch=2)})
        for kw in moves:
            cand = replace(spec, **kw)
            if cand != spec and cand not in out:
                out.append(cand)
        return tuple(out)
    for kw in (
        {"dtl_stages": 2},
        {"dtl_stages": 4},
        {"tile_k": 64},
        {"persistent_ctas": 256},
        {"knobs": replace(spec.knobs, epi_pair=True)},
    ):
        cand = replace(spec, **kw)
        if cand != spec and cand not in out:
            out.append(cand)
    return tuple(out)


def _make_candidate(*, name, spec_id, priority, spec_fn) -> KernelCandidate:
    def support(req: OperatorRequest) -> tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, TensorGettRequest)
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        sp = spec_fn(req)
        if sp is None:
            return False, "no valid body for this problem"
        return is_valid_spec(sp, arch=req.arch)

    def select(req: OperatorRequest) -> TensorGettSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, TensorGettRequest)
        return spec_fn(req)

    def sweep(req: OperatorRequest):
        if not candidate.admits(req)[0]:
            return ()
        assert isinstance(req, TensorGettRequest)
        return tuple(
            s
            for s in _neighbourhood(spec_fn(req))
            if is_valid_spec(s, arch=req.arch)[0]
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm="gett",
        spec_id=spec_id,
        abi_version=TENSOR_GETT_ABI_VERSION,
        priority=priority,
        capability=Capability(arches=TENSOR_GETT_ARCHS, dtypes=TENSOR_GETT_DTYPES),
        _supports=support,
        select_spec=select,
        signature=tensor_gett_signature,
        grid=lambda spec, _req: tensor_gett_grid(spec),
        block=tensor_gett_block,
        sweep_space=sweep,
        build=lambda spec, arch: build_tensor_gett(spec, arch=arch),
    )
    return candidate


TENSOR_GETT_REGISTRY = CandidateRegistry(
    _FAMILY, dim_vocabulary=TENSOR_GETT_DIM_VOCABULARY, require_build=True
)
TENSOR_GETT_REGISTRY.extend(
    (
        _make_candidate(
            name="tensor_gett_mn",
            spec_id="mn",
            priority=5,
            spec_fn=_spec_mn,
        ),
        _make_candidate(
            name="tensor_gett_t32s3",
            spec_id="t32s3",
            priority=10,
            spec_fn=_spec_t32s3,
        ),
    )
)


def tensor_gett_candidates() -> tuple[KernelCandidate, ...]:
    return TENSOR_GETT_REGISTRY.candidates()


def _kernel_id(req: TensorGettRequest, candidate: KernelCandidate, spec) -> KernelId:
    return KernelId(
        op="contraction",
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=stable_json_hash(req.normalized(), n=16),
        spec_hash=stable_json_hash(asdict(spec), n=16),
    )


def tensor_gett_sweep_space(req: OperatorRequest) -> Sequence[TensorGettSpec]:
    if _request_errors(req):
        return ()
    return TENSOR_GETT_REGISTRY.sweep_space(req)


def dispatch_tensor_gett_all(req: TensorGettRequest) -> tuple[DispatchResult, ...]:
    if _request_errors(req):
        return ()
    return TENSOR_GETT_REGISTRY.dispatch_all(req, kernel_id=_kernel_id)


def dispatch_tensor_gett(
    req: TensorGettRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select a registered tensor GETT candidate for ``req``."""
    candidate = TENSOR_GETT_REGISTRY.select(req, ranker=ranker)
    spec = candidate.select_spec(req)
    kid = _kernel_id(req, candidate, spec)
    return DispatchResult(
        request=req,
        candidate=candidate,
        spec=spec,
        kernel_id=kid,
        grid=candidate.grid(spec, req),
        block=candidate.block(spec),
        signature=tuple(candidate.signature(spec)),
        explanation=(
            f"selected {candidate.name} for {req.dtype} GETT on {req.arch}",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )


# ---------------------------------------------------------------------------
# product route: single launch or pack hybrid
# ---------------------------------------------------------------------------

#: Pack operands with short runs and run the MN-layout GETT on the packed
#: copies (False: the ring-body GETT reads them with 4-byte DTL lanes).
TENSOR_GETT_PACK_HYBRID = True
#: Route policy of :func:`plan_tensor_gett`: "auto" (cost model), "mn" (the
#: MN-layout routes only), "kc" (the K-contiguous route wherever it is valid).
#: Env override ``ROCKE_TENSOR_GETT_ROUTE``.
TENSOR_GETT_ROUTE = "auto"
TENSOR_GETT_ROUTE_ENV = "ROCKE_TENSOR_GETT_ROUTE"
#: tensor_pack cost model (gfx950) per form: (launch floor in microseconds,
#: sustained bytes moved (read + write) per microsecond), fitted on the doc-row
#: operand packs
TENSOR_PACK_COST = {
    "pairs": (2.0, 4.2e6),
    "tiled": (5.0, 5.5e6),
    "elements": (3.0, 1.3e6),
    "permute_tiled": (2.5, 5.8e6),
}
#: ratio of the GETT kernel time to the GEMM model's prediction of its body (the
#: model predicts the plain GEMM): the MN-layout GETT (operand addressing +
#: permuted epilogue) and the K-contiguous GETT on the RCR bodies (its
#: epilogue hook replaces the bodies' guarded vector epilogue)
TENSOR_GETT_MN_TIME_FACTOR = 1.02
TENSOR_GETT_KC_TIME_FACTOR = 0.96
#: log-time margin the K-contiguous route must be predicted faster by
TENSOR_GETT_KC_MARGIN = 0.01


def _route_policy() -> str:
    import os

    v = os.environ.get(TENSOR_GETT_ROUTE_ENV, TENSOR_GETT_ROUTE).strip().lower()
    return v if v in ("auto", "mn", "kc") else "auto"


def _pack_cost_us(spec) -> float:
    from ...instances.common.permute_tiled import PermuteTiledSpec

    if isinstance(spec, PermuteTiledSpec):
        c0, bw = TENSOR_PACK_COST["permute_tiled"]
        n = 1
        for e in spec.x_shape:
            n *= int(e)
        return c0 + 2.0 * spec.elem_bytes * n / bw
    c0, bw = TENSOR_PACK_COST[spec.resolved_form()]
    return c0 + 4.0 * spec.numel() / bw


#: Pack kernel of the pack steps: "tensor_pack" (the GETT family's own pack)
#: or "permute_tiled" (the multi-axis tiled permute: 16-byte accesses on both
#: sides with an in-register / LDS transpose; used for contiguous sources,
#: strided sources keep tensor_pack)
TENSOR_GETT_PACK_KERNEL = "tensor_pack"


def _pack_step(name: str, ps, arch: str) -> "TensorPackStep":
    """The pack step copying operand ``name`` per ``ps`` (a TensorPackSpec)
    with the :data:`TENSOR_GETT_PACK_KERNEL` kernel."""
    from ...instances.common.tensor.pack import tensor_pack_block, tensor_pack_grid

    if TENSOR_GETT_PACK_KERNEL == "permute_tiled" and ps.strides_src is None:
        from ...instances.common.permute_tiled import (
            permute_tiled_grid,
            plan_permute_tiled,
        )

        ext = dict(ps.extents)
        shape = [int(ext[m]) for m in ps.modes_src]
        perm = [list(ps.modes_src).index(m) for m in ps.modes_dst]
        dt = "f16" if ps.dtype in ("f16", "fp16") else ps.dtype
        try:
            pt = plan_permute_tiled(shape, perm, dt, arch)
        except ValueError:
            pt = None
        if pt is not None:
            return TensorPackStep(
                name, pt, permute_tiled_grid(pt), (pt.block_size, 1, 1), 2 * ps.numel()
            )
    return TensorPackStep(
        name, ps, tensor_pack_grid(ps), tensor_pack_block(ps), 2 * ps.numel()
    )


def build_pack_step(step: "TensorPackStep", arch: str):
    """The kernel of a pack step (params ``X`` = source, ``Y`` = workspace)."""
    from ...instances.common.permute_tiled import PermuteTiledSpec, build_permute_tiled
    from ...instances.common.tensor.pack import build_tensor_pack

    if isinstance(step.spec, PermuteTiledSpec):
        return build_permute_tiled(step.spec, arch=arch)
    return build_tensor_pack(step.spec, arch=arch)


def _gemm_req(req, M, N, K, layout):
    from ..gemm import GemmRequest

    return GemmRequest(
        M=M,
        N=N,
        K=K,
        arch=req.arch,
        dtype=req.dtype,
        layout=layout,
        trans_a=layout[0] == "C",
        trans_b=layout[1] == "C",
    )


def _gemm_dispatch(req):
    from ..gemm import dispatch_gemm_bf16, dispatch_gemm_fp16

    return dispatch_gemm_bf16 if req.dtype == "bf16" else dispatch_gemm_fp16


def _predicted(greq, spec):
    from ..gemm.plan import predicted_us

    try:
        return predicted_us(greq, spec)
    except Exception:  # noqa: BLE001 - no model: no estimate
        return None


def _spec_kc(req: TensorGettRequest, body=None):
    """The K-contiguous GETT spec on the GEMM dispatcher's RCR pick, or None.
    The pick's epilogue decides which GEMM operand owns D's fastest mode (the
    row-interleave vector epilogue stores 8 columns per lane). ``body``: an
    explicit RCR GEMM spec to use instead of the dispatcher pick."""
    from ...instances.common.gemm_universal import UniversalGemmSpec

    p = req.problem
    disp = _gemm_dispatch(req)
    for role in ("B", "A"):
        try:
            plan = _kc_plan(p, role)
        except ValueError:
            continue
        if body is not None:
            r_spec = body
        else:
            try:
                r_spec = disp(_gemm_req(req, plan.M, plan.N, plan.K, "RCR")).spec
            except (ValueError, KeyError):
                continue
        if not isinstance(r_spec, UniversalGemmSpec):
            continue
        tr = r_spec.trait
        if bool(tr.rilv_vec_epilogue) != (role == "B"):
            continue
        if (
            role == "B"
            and plan.d_col_digits[0][0] < TENSOR_GETT_KC_MIN_D_RUN
            and plan.K < TENSOR_GETT_KC_SHORT_RUN_MIN_K
        ):
            continue
        sp = replace(
            p,
            dtype=req.dtype,
            tile_k=32,
            dtl_stages=3,
            persistent_ctas=0,
            chiplet_wgm=8,
            chiplet_chunk_size=64,
            mn_tile=None,
            mn_trait=None,
            kc_tile=r_spec.tile,
            kc_trait=tr,
            knobs=replace(
                p.knobs,
                epi_v16=False,
                store_nt=False,
                epi_pair=False,
                c_prefetch=TENSOR_GETT_KC_C_PREFETCH if p.bilinear else 0,
            ),
        )
        if is_valid_spec(sp, arch=req.arch)[0]:
            return sp
    return None


def _kc_plan(p, role):
    from ...instances.common.tensor.gett_plan import plan_gett

    return plan_gett(
        p.modes_a,
        p.modes_b,
        p.modes_d,
        dict(p.extents),
        strides_a=None if p.strides_a is None else dict(p.strides_a),
        strides_b=None if p.strides_b is None else dict(p.strides_b),
        strides_d=None if p.strides_d is None else dict(p.strides_d),
        kc=True,
        d_fast_role=role,
    )


# the K-contiguous route as a candidate (after the MN-layout and ring bodies in
# auto selection; plan_tensor_gett chooses it by the cost model)
TENSOR_GETT_REGISTRY.extend(
    (
        _make_candidate(
            name="tensor_gett_kc",
            spec_id="kc",
            priority=20,
            spec_fn=_spec_kc,
        ),
    )
)


@dataclass(frozen=True)
class TensorPackStep:
    """Copy the user operand ``operand`` (``"A"`` / ``"B"``) into a packed
    workspace of ``workspace_bytes`` with the ``tensor_pack`` kernel
    ``spec`` (params X = the user tensor, Y = the workspace)."""

    operand: str
    spec: object
    grid: tuple
    block: tuple
    workspace_bytes: int


@dataclass(frozen=True)
class TensorGettPlan:
    """The launches of one contraction: ``packs`` (in order), then ``gett``
    (whose A / B params take the packed workspace for every packed operand
    and the user tensor otherwise; see :func:`gett_operand_sources`)."""

    packs: tuple
    gett: DispatchResult
    workspace_bytes: int
    explanation: tuple

    @property
    def kind(self) -> str:
        return "pack_hybrid" if self.packs else "single"


#: Pack order: "local" moves only the K modes that split the operand's
#: innermost free modes (just enough free modes for a run that is a multiple
#: of 8 elements; the copy stays local in memory), "full" moves every K mode
#: outside every free mode.
TENSOR_GETT_PACK_ORDER = "local"


def _pack_order(modes, k_modes, ext):
    """The packed mode order (row-major) of an operand: its free modes form an
    innermost contiguous run that is a multiple of 8 elements."""
    full = tuple(m for m in modes if m in k_modes) + tuple(
        m for m in modes if m not in k_modes
    )
    if TENSOR_GETT_PACK_ORDER != "local":
        return full
    prod, first = 1, None
    for i in range(len(modes) - 1, -1, -1):
        if modes[i] in k_modes:
            continue
        prod *= ext[modes[i]]
        first = i
        if prod % 8 == 0:
            break
    if first is None or prod % 8:
        return full
    inner = modes[first:]
    return (
        tuple(modes[:first])
        + tuple(m for m in inner if m in k_modes)
        + tuple(m for m in inner if m not in k_modes)
    )


def plan_tensor_gett(req: TensorGettRequest) -> TensorGettPlan:
    """The product route for ``req`` (see the module doc). A problem with an
    entry in the tuned route table (:data:`TENSOR_GETT_TUNED`, measured by
    :func:`tune_tensor_gett`) takes that route under the ``auto`` policy."""
    policy = _route_policy()
    if policy == "auto" and TENSOR_GETT_USE_TUNED:
        tuned = _tuned_plan(req)
        if tuned is not None:
            return tuned
    mn_plan = _plan_mn(req)
    if policy == "mn":
        return mn_plan
    kc_plan = _plan_kc(req)
    if kc_plan is None:
        return mn_plan
    if policy == "kc":
        return kc_plan
    t_mn = _plan_cost(req, mn_plan)
    t_kc = _plan_cost(req, kc_plan)
    if t_mn is None or t_kc is None:
        return mn_plan
    import math

    if math.log(t_mn) - math.log(t_kc) > TENSOR_GETT_KC_MARGIN:
        return replace(
            kc_plan,
            explanation=kc_plan.explanation
            + (f"predicted {t_kc:.1f} us vs {t_mn:.1f} us ({mn_plan.kind})",),
        )
    return replace(
        mn_plan,
        explanation=mn_plan.explanation
        + (f"predicted {t_mn:.1f} us vs {t_kc:.1f} us (kc)",),
    )


def _plan_cost(req: TensorGettRequest, plan: "TensorGettPlan"):
    """Predicted time (us): the GEMM model's time of the plan's body for the
    plan's M x N x K plus the pack cost model."""
    sp = plan.gett.spec
    if not (sp.mn_body or sp.kc_body):
        return None
    pl = sp.plan()
    body = sp.body()
    t = _predicted(_gemm_req(req, pl.M, pl.N, pl.K, body.data.layout), body)
    if t is None:
        return None
    t *= TENSOR_GETT_KC_TIME_FACTOR if sp.kc_body else TENSOR_GETT_MN_TIME_FACTOR
    return t + sum(_pack_cost_us(s.spec) for s in plan.packs)


def _plan_kc(req: TensorGettRequest, body=None):
    """The K-contiguous route (both operands packed [rows, K]), or None
    (``body``: an explicit RCR GEMM spec, see :func:`_spec_kc`)."""
    from ...instances.common.tensor import kc_operand_modes, launch_operands
    from ...instances.common.tensor.pack import TensorPackSpec
    from ...instances.common.tensor.pack import is_valid_spec as _pack_valid

    sp = _spec_kc(req, body)
    if sp is None:
        return None
    p = req.problem
    ext = dict(p.extents)
    packs = []
    ga, gb = launch_operands(sp)
    for nm, order in zip((ga, gb), kc_operand_modes(sp)):
        modes = p.modes_a if nm == "A" else p.modes_b
        strides = p.strides_a if nm == "A" else p.strides_b
        ps = TensorPackSpec.make(
            modes,
            order,
            {m: ext[m] for m in modes},
            strides_src=None if strides is None else dict(strides),
            dtype=req.dtype,
        )
        if not _pack_valid(ps, arch=req.arch)[0]:
            return None
        packs.append(_pack_step(nm, ps, req.arch))
    cand = TENSOR_GETT_REGISTRY.get("tensor_gett_kc")
    kid = _kernel_id(req, cand, sp)
    r = DispatchResult(
        request=req,
        candidate=cand,
        spec=sp,
        kernel_id=kid,
        grid=cand.grid(sp, req),
        block=cand.block(sp),
        signature=tuple(cand.signature(sp)),
        explanation=(f"selected {cand.name} (K-contiguous packed operands)",),
    )
    return TensorGettPlan(
        tuple(packs),
        r,
        sum(s.workspace_bytes for s in packs),
        ("kc: both operands packed K-contiguous, then tensor_gett_kc",),
    )


def _mn_result(req: TensorGettRequest, bodies) -> Optional[DispatchResult]:
    """A ``tensor_gett_mn`` dispatch result on the explicit ``bodies`` (None:
    no valid body)."""
    if _request_errors(req):
        return None
    sp = _spec_mn(req, bodies)
    if sp is None or not is_valid_spec(sp, arch=req.arch)[0]:
        return None
    cand = TENSOR_GETT_REGISTRY.get("tensor_gett_mn")
    return DispatchResult(
        request=req,
        candidate=cand,
        spec=sp,
        kernel_id=_kernel_id(req, cand, sp),
        grid=cand.grid(sp, req),
        block=cand.block(sp),
        signature=tuple(cand.signature(sp)),
        explanation=(f"selected {cand.name} on an explicit body",),
    )


def _plan_mn(req: TensorGettRequest, bodies=None) -> "TensorGettPlan":
    """The MN-layout routes: single launch, else the pack hybrid, else the
    ring body. ``bodies``: explicit MN bodies (see :func:`_spec_mn`); then the
    ring-body fallback is not taken (None when no body serves)."""
    from ...instances.common.tensor.pack import TensorPackSpec
    from ...instances.common.tensor.pack import is_valid_spec as _pack_valid

    p = req.problem
    mn = TENSOR_GETT_REGISTRY.get("tensor_gett_mn")
    if bodies is not None:
        r = _mn_result(req, bodies)
        if r is not None:
            return TensorGettPlan((), r, 0, (f"single launch: {r.candidate.name}",))
    elif mn.admits(req)[0] or not TENSOR_GETT_PACK_HYBRID:
        r = dispatch_tensor_gett(req)
        return TensorGettPlan((), r, 0, (f"single launch: {r.candidate.name}",))
    try:
        plan = replace(p, tile_k=32, mn_tile=None, mn_trait=None).plan()
    except ValueError:
        plan = None
    packs = []
    newp = p
    why = []
    if plan is not None:
        ext = dict(p.extents)
        k_modes = [m for m in p.modes_a if m in p.modes_b and m not in p.modes_d]
        for role in ("A", "B"):
            op = plan.ops[role]
            if op.vec >= 8:
                continue
            nm = op.name
            modes = p.modes_a if nm == "A" else p.modes_b
            strides = p.strides_a if nm == "A" else p.strides_b
            order = _pack_order(modes, k_modes, ext)
            ps = TensorPackSpec.make(
                modes,
                order,
                {m: ext[m] for m in modes},
                strides_src=None if strides is None else dict(strides),
                dtype=req.dtype,
            )
            ok, w = _pack_valid(ps, arch=req.arch)
            if not ok:
                why.append(f"operand {nm}: no pack ({w})")
                packs = None
                break
            packs.append(_pack_step(nm, ps, req.arch))
            if nm == "A":
                newp = replace(newp, modes_a=order, strides_a=None)
            else:
                newp = replace(newp, modes_b=order, strides_b=None)
    if packs:
        preq = replace(req, problem=newp)
        r = None
        if bodies is not None:
            r = _mn_result(preq, bodies)
        elif mn.admits(preq)[0]:
            r = dispatch_tensor_gett(preq)
        if r is not None:
            return TensorGettPlan(
                tuple(packs),
                r,
                sum(s.workspace_bytes for s in packs),
                (
                    "pack hybrid: "
                    + ", ".join(f"operand {s.operand} packed" for s in packs),
                    f"then {r.candidate.name}",
                ),
            )
        why.append("the packed problem has no MN-layout body")
    if bodies is not None:
        return None
    r = dispatch_tensor_gett(req)
    return TensorGettPlan(
        (), r, 0, tuple(why) + (f"single launch: {r.candidate.name}",)
    )


# ---------------------------------------------------------------------------
# candidate routes, autotune mode and the tuned route table
# ---------------------------------------------------------------------------

#: candidate routes of :func:`plan_tensor_gett_candidates`: the MN-layout GETT
#: on the best ``TENSOR_GETT_TUNE_MN_TOP`` both-MN-contiguous GEMM bodies of the
#: native selector's ranking, and the K-contiguous route on the best
#: ``TENSOR_GETT_TUNE_KC_TOP`` RCR bodies of the RCR selector's ranking (one
#: body per geometry, see :func:`_diverse_bodies`)
TENSOR_GETT_TUNE_MN_TOP = 6
TENSOR_GETT_TUNE_KC_TOP = 6
#: one-step body mutations :func:`plan_tensor_gett_candidates` adds around its
#: first ``neighbours`` MN-layout bodies (the autotune search space beyond the
#: selector's ranking): grid raster, ring depth, scheduling priority and
#: interleave barrier, occupancy hint, persistent grid, the MN-operand LDS
#: block layout, the LDS swizzle, AGPR accumulators, fragment prefetch
TENSOR_GETT_TUNE_LEVERS = (
    ("chiplet_swizzle", (False,)),
    ("chiplet_wgm", (4, 8, 16)),
    ("dtl_stages", (2, 3, 4)),
    ("interwave_setprio", (True, False)),
    ("ilv_barrier_after", (0, 2)),
    ("waves_per_eu", (1, 2)),
    ("persistent_ctas", (0, 256)),
    ("chiplet_chunk_size", (4, 8, 32)),
    ("mn_chunk", (64, 128, 256)),
    ("mn_pad", (0, 32, 64)),
    ("lds_swizzle", (True, False)),
    ("acc_agpr", (True, False)),
    ("dtl_frag_prefetch", (True, False)),
)
#: consult :data:`TENSOR_GETT_TUNED` in :func:`plan_tensor_gett` (False: the
#: cost-model routes only)
TENSOR_GETT_USE_TUNED = True
#: Tuned routes: ``{arch: {tensor_gett_route_key(req): (route, gemm config)}}``
#: with route ``"mn"`` (MN-layout GETT, pack hybrid when an operand run needs
#: it) or ``"kc"`` (K-contiguous packed operands) and the GEMM body as a
#: selector config string (``selector.config_string``). Entries come from
#: :func:`tune_tensor_gett` runs (each winner passed the full numeric check
#: and was confirmed in an interleaved final pass). The shipped entries live in
#: the ``tensor_gett_tuned.json`` sidecar next to this module (each with the
#: problem it was tuned on, see :func:`tuned_entry_problem`).
TENSOR_GETT_TUNED_FILE = "tensor_gett_tuned.json"


#: AOT route table: the product route (:func:`plan_tensor_gett`) of fixed
#: problems, frozen for ahead-of-time kernel packs (hipTensor's rocKE kernel
#: pack carries one launch program per entry: the pack steps and the GETT
#: launch). Same format as the tuned table; each entry is
#: :func:`tensor_gett_aot_entry` of its problem, which the tests check, so the
#: table follows the product route. :func:`plan_tensor_gett` never reads it.
#: Regenerate with ``platform/tools/gen_tensor_gett_aot.py``.
TENSOR_GETT_AOT_FILE = "tensor_gett_aot.json"


def _read_table(fname: str) -> dict:
    import json
    from pathlib import Path

    path = Path(__file__).with_name(fname)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("entries", {})


def _load_table(fname: str) -> dict:
    return {
        arch: {k: (e["route"], e["body"]) for k, e in table.items()}
        for arch, table in _read_table(fname).items()
    }


def _load_shipped_tuned() -> dict:
    return _load_table(TENSOR_GETT_TUNED_FILE)


_SHIPPED_TUNED: dict = _load_shipped_tuned()
TENSOR_GETT_TUNED: dict = {a: dict(t) for a, t in _SHIPPED_TUNED.items()}
#: ``{arch: {tensor_gett_route_key(req): (route, gemm config)}}`` of the AOT
#: route table (:data:`TENSOR_GETT_AOT_FILE`)
TENSOR_GETT_AOT: dict = _load_table(TENSOR_GETT_AOT_FILE)


def tuned_entry_problem(arch: str, key: str) -> Optional[dict]:
    """The problem (``modes_a`` / ``modes_b`` / ``modes_d`` / ``extents`` /
    ``dtype`` / ``bilinear``) a shipped tuned entry was measured on, or None."""
    e = _read_table(TENSOR_GETT_TUNED_FILE).get(arch, {}).get(key)
    return None if e is None else e.get("problem")


def aot_entry_problem(arch: str, key: str) -> Optional[dict]:
    """The problem of an AOT route table entry (as :func:`tuned_entry_problem`),
    or None."""
    e = _read_table(TENSOR_GETT_AOT_FILE).get(arch, {}).get(key)
    return None if e is None else e.get("problem")


@dataclass(frozen=True)
class TensorGettRoute:
    """One candidate route: ``route`` ("mn" / "kc"), the GEMM ``body`` spec
    it runs on, where it came from (``origin``) and its launches (``plan``)."""

    origin: str
    route: str
    body: object
    plan: "TensorGettPlan"


@dataclass
class TensorGettTuneResult:
    route: TensorGettRoute
    evaluations: int
    routes: tuple
    search: object = None

    def table_entry(self, req: "TensorGettRequest") -> tuple:
        """``(key, (route, config))`` for :data:`TENSOR_GETT_TUNED`."""
        from ..gemm.selector import config_string

        return tensor_gett_route_key(req), (
            self.route.route,
            config_string(self.route.body),
        )


def _canonical_modes(p) -> dict:
    """The problem's modes renamed by first appearance (D, then A, then B):
    two problems that differ only in mode names get the same key."""
    names: dict = {}
    for m in tuple(p.modes_d) + tuple(p.modes_a) + tuple(p.modes_b):
        names.setdefault(m, f"x{len(names)}")
    return names


def tensor_gett_route_key(req: "TensorGettRequest") -> str:
    """The tuned-table key of ``req``: dtype, bilinear and the problem's
    modes / extents / strides with mode names canonicalised."""
    p = req.problem
    nm = _canonical_modes(p)
    ext = dict(p.extents)

    def strides(st):
        if st is None:
            return None
        return sorted((nm[k], int(v)) for k, v in dict(st).items())

    body = {
        "dtype": req.dtype,
        "bilinear": bool(p.bilinear),
        "a": [nm[m] for m in p.modes_a],
        "b": [nm[m] for m in p.modes_b],
        "d": [nm[m] for m in p.modes_d],
        "ext": sorted((nm[m], int(e)) for m, e in ext.items() if m in nm),
        "sa": strides(p.strides_a),
        "sb": strides(p.strides_b),
        "sd": strides(p.strides_d),
    }
    return stable_json_hash(body, n=20)


def _diverse_bodies(ranked, k: int) -> list:
    """The best spec of each distinct body geometry (tile, waves, atom,
    persistent grid, ring depth, row interleave), best first, at most ``k``."""
    out, seen = [], set()
    if k <= 0:
        return out
    for _sc, c in ranked:
        t, tr = c.spec.tile, c.spec.trait
        key = (
            t.tile_m,
            t.tile_n,
            t.tile_k,
            t.warp_m,
            t.warp_n,
            t.warp_tile_m,
            t.warp_tile_n,
            t.warp_tile_k,
            tr.persistent_ctas > 0,
            tr.dtl_stages,
            bool(tr.lds_row_interleave),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(c.spec)
        if len(out) >= k:
            break
    return out


def _route_body(sp):
    """The GEMM body of a GETT spec with its tile / trait as the route stores
    them (``mn_*`` / ``kc_*``, before the GETT normalisation of
    :meth:`TensorGettSpec.body`), so a tuned-table entry rebuilds the same
    spec."""
    b = sp.body()
    if sp.kc_body:
        return replace(b, tile=sp.kc_tile, trait=sp.kc_trait)
    return replace(b, tile=sp.mn_tile, trait=sp.mn_trait)


def _ranked_bodies(req: "TensorGettRequest", M: int, N: int, K: int, layout: str):
    from ..gemm import selector as _S
    from ..gemm.autotune import _registry

    greq = _S._auto(_gemm_req(req, M, N, K, layout))
    registry = _registry(greq)
    try:
        cand = registry.select(greq)
        rule = (cand, cand.select_spec(greq))
    except Exception:  # noqa: BLE001 - no rule pick: rank without it
        rule = None
    try:
        return _S.rank(greq, registry, rule_pick=rule)
    except Exception:  # noqa: BLE001 - no model for this arch / layout
        return []


def plan_tensor_gett_candidates(
    req: "TensorGettRequest",
    *,
    mn_top: Optional[int] = None,
    kc_top: Optional[int] = None,
    neighbours: int = 0,
) -> list:
    """The candidate routes of ``req`` for autotuning, the cost-model product
    route first: the MN-layout GETT on each of the selector's best
    both-MN-contiguous bodies (with the pack hybrid where an operand run
    needs it), one-step mutations (:data:`TENSOR_GETT_TUNE_LEVERS`) of the
    first ``neighbours`` of those bodies, and the K-contiguous route on each
    of the selector's best RCR bodies. Every route's GETT spec passed
    :func:`is_valid_spec`; duplicates count once."""
    mn_top = TENSOR_GETT_TUNE_MN_TOP if mn_top is None else mn_top
    kc_top = TENSOR_GETT_TUNE_KC_TOP if kc_top is None else kc_top
    out: list = []
    seen: set = set()

    def add(origin, route, body, plan):
        if plan is None:
            return
        key = (route, stable_json_hash(asdict(plan.gett.spec), n=16))
        if key in seen:
            return
        seen.add(key)
        out.append(TensorGettRoute(origin, route, body, plan))

    global TENSOR_GETT_USE_TUNED
    saved = TENSOR_GETT_USE_TUNED
    TENSOR_GETT_USE_TUNED = False
    try:
        dflt = plan_tensor_gett(req)
    finally:
        TENSOR_GETT_USE_TUNED = saved
    sp = dflt.gett.spec
    if sp.mn_body or sp.kc_body:
        add("default", "kc" if sp.kc_body else "mn", _route_body(sp), dflt)
    try:
        base = replace(req.problem, tile_k=32, mn_tile=None, mn_trait=None).plan()
    except ValueError:
        base = None
    mn_bodies = [_route_body(sp)] if sp.mn_body else []
    if base is not None and mn_top > 0:
        ranked = _ranked_bodies(req, base.M, base.N, base.K, "CRR")
        for i, body in enumerate(_diverse_bodies(ranked, mn_top)):
            mn_bodies.append(body)
            add(f"mn:{i}", "mn", body, _plan_mn(req, [(body.tile, body.trait)]))
    for r in tensor_gett_neighbour_routes(req, mn_bodies[: max(0, int(neighbours))]):
        add(r.origin, r.route, r.body, r.plan)
    if kc_top > 0:
        kp = None
        for role in ("B", "A"):
            try:
                kp = _kc_plan(req.problem, role)
                break
            except ValueError:
                kp = None
        if kp is not None:
            ranked = _ranked_bodies(req, kp.M, kp.N, kp.K, "RCR")
            for i, body in enumerate(_diverse_bodies(ranked, kc_top)):
                add(f"kc:{i}", "kc", body, _plan_kc(req, body))
    return out


def tensor_gett_neighbour_routes(req: "TensorGettRequest", bodies) -> list:
    """MN-layout routes on the one-step mutations
    (:data:`TENSOR_GETT_TUNE_LEVERS`) of each MN body in ``bodies`` (origin
    ``nb<i>:<lever>=<value>``); invalid mutations are dropped."""
    from ...helpers.split_k import arch_num_cus

    out = []
    for j, body in enumerate(bodies):
        for lever, values in TENSOR_GETT_TUNE_LEVERS:
            for v in values:
                if lever == "persistent_ctas" and v:
                    v = arch_num_cus(req.arch)
                if getattr(body.trait, lever) == v:
                    continue
                tr = replace(body.trait, **{lever: v})
                if lever == "chiplet_wgm":
                    tr = replace(tr, chiplet_swizzle=True)
                nb = replace(body, trait=tr)
                pl = _plan_mn(req, [(nb.tile, nb.trait)])
                if pl is not None:
                    out.append(TensorGettRoute(f"nb{j}:{lever}={v}", "mn", nb, pl))
    return out


def _route_flat(r: TensorGettRoute) -> dict:
    from ..gemm.selector import flatten_spec as _flat

    d = {"route": r.route, "packs": len(r.plan.packs)}
    d.update(_flat(r.body))
    return d


def tune_tensor_gett(
    req: "TensorGettRequest",
    *,
    measure,
    check,
    mode: str = "search",
    budget: int = 12,
    strategy: str = "portfolio",
    seed: int = 0,
    mn_top: Optional[int] = None,
    kc_top: Optional[int] = None,
    routes=None,
    **tune_kwargs,
) -> TensorGettTuneResult:
    """Autotune mode of the product route: measure the candidate routes of
    :func:`plan_tensor_gett_candidates` (or the given ``routes``) with the
    search engine (:func:`rocke.dispatch.tuning.search.api.search_configs`)
    and return the fastest one that passed ``check``.

    ``measure(plan, fidelity)`` times one :class:`TensorGettPlan` (packs +
    GETT launch); ``check(plan)`` runs the full-output numeric check (a
    route that fails it can never win). ``mode="lite"`` times every
    candidate once plus an interleaved confirmation of the finalists;
    ``mode="search"`` spends ``budget`` evaluations (the cost-model route is
    the warm start)."""
    from ..tuning.search import Budget, WarmStart
    from ..tuning.search.api import search_configs

    if routes is None:
        routes = plan_tensor_gett_candidates(req, mn_top=mn_top, kc_top=kc_top)
    routes = list(routes)
    if not routes:
        raise ValueError(f"no GETT route for {req}")
    rows = [_route_flat(r) for r in routes]
    by_id = {id(r.plan): i for i, r in enumerate(routes)}
    confirm = int(tune_kwargs.pop("confirm", min(3, len(routes))))
    lite = mode == "lite"
    n = len(routes) + confirm if lite else max(1, int(budget))
    warm = () if lite else [WarmStart(rows[0], "default")]
    winner, result = search_configs(
        [r.plan for r in routes],
        measure=measure,
        check=check,
        flatten=lambda pl: rows[by_id[id(pl)]],
        strategy="exhaustive" if lite else strategy,
        budget=Budget(evaluations=n),
        seed=seed,
        warm=warm,
        confirm=confirm,
        **tune_kwargs,
    )
    if winner is None:
        raise ValueError("autotune found no route that passed the numeric check")
    return TensorGettTuneResult(
        route=routes[winner],
        evaluations=len(getattr(result, "history", []) or []) or n,
        routes=tuple(routes),
        search=result,
    )


def _tuned_plan(req: "TensorGettRequest") -> Optional["TensorGettPlan"]:
    """The plan of ``req``'s tuned-table entry, or None (no entry, or the
    entry no longer builds a valid route)."""
    return _table_plan(req, TENSOR_GETT_TUNED, "tuned route table")


def aot_route_plan(req: "TensorGettRequest") -> Optional["TensorGettPlan"]:
    """The plan of ``req``'s AOT route table entry (:data:`TENSOR_GETT_AOT`),
    or None. It equals :func:`plan_tensor_gett` of ``req`` (the tests check
    every entry)."""
    return _table_plan(req, TENSOR_GETT_AOT, "AOT route table")


def tensor_gett_aot_entry(req: "TensorGettRequest") -> Optional[tuple]:
    """``(key, entry)`` of ``req`` for the AOT route table: the route and GEMM
    body of :func:`plan_tensor_gett` and the problem, or None when the product
    route runs no GEMM body (the ring-body fallback) or the problem has
    explicit strides (table problems are packed)."""
    from ..gemm.selector import config_string

    p = req.problem
    if p.strides_a is not None or p.strides_b is not None or p.strides_d is not None:
        return None
    sp = plan_tensor_gett(req).gett.spec
    if not (sp.mn_body or sp.kc_body):
        return None
    problem = {
        "bilinear": bool(p.bilinear),
        "dtype": req.dtype,
        "extents": {m: int(e) for m, e in sorted(dict(p.extents).items())},
        "modes_a": list(p.modes_a),
        "modes_b": list(p.modes_b),
        "modes_d": list(p.modes_d),
    }
    entry = {
        "body": config_string(_route_body(sp)),
        "problem": problem,
        "route": "kc" if sp.kc_body else "mn",
    }
    return tensor_gett_route_key(req), entry


def _table_plan(
    req: "TensorGettRequest", tables: dict, source: str
) -> Optional["TensorGettPlan"]:
    """The plan of ``req``'s entry in ``tables`` (``{arch: {key: (route,
    config)}}``), or None (no entry, or the entry no longer builds a valid
    route)."""
    table = tables.get(req.arch)
    if not table:
        return None
    ent = table.get(tensor_gett_route_key(req))
    if ent is None:
        return None
    route, cfg = ent
    from ...instances.common.gemm_universal import TileSpec, TraitSpec
    from ..gemm.selector import parse_config

    try:
        tl, tr = parse_config(cfg)
        tile, trait = TileSpec(**tl), TraitSpec(**tr)
    except (ValueError, TypeError):
        return None
    if route == "mn":
        pl = _plan_mn(req, [(tile, trait)])
    elif route == "kc":
        try:
            kp = _kc_plan(req.problem, "B")
            pick = _gemm_dispatch(req)(_gemm_req(req, kp.M, kp.N, kp.K, "RCR")).spec
        except (ValueError, KeyError):
            return None
        pl = _plan_kc(req, replace(pick, tile=tile, trait=trait))
    else:
        return None
    if pl is None:
        return None
    return replace(pl, explanation=pl.explanation + (source,))


def gett_operand_sources(plan: TensorGettPlan) -> tuple:
    """For the GETT launch's (A, B) params: ``("user", "A" | "B")`` or
    ``("pack", step index)``."""
    from ...instances.common.tensor import launch_operands

    ga, gb = launch_operands(plan.gett.spec)
    idx = {s.operand: i for i, s in enumerate(plan.packs)}
    return tuple(("pack", idx[n]) if n in idx else ("user", n) for n in (ga, gb))


__all__ = [
    "TENSOR_GETT_PACK_HYBRID",
    "TENSOR_GETT_PACK_KERNEL",
    "build_pack_step",
    "TENSOR_GETT_PACK_ORDER",
    "TensorGettPlan",
    "TensorGettRoute",
    "TensorGettTuneResult",
    "TensorPackStep",
    "TENSOR_GETT_AOT",
    "TENSOR_GETT_AOT_FILE",
    "TENSOR_GETT_TUNED",
    "aot_entry_problem",
    "aot_route_plan",
    "tensor_gett_aot_entry",
    "plan_tensor_gett_candidates",
    "tensor_gett_neighbour_routes",
    "tensor_gett_route_key",
    "tuned_entry_problem",
    "tune_tensor_gett",
    "gett_operand_sources",
    "plan_tensor_gett",
    "TENSOR_GETT_REGISTRY",
    "TensorGettRequest",
    "dispatch_tensor_gett",
    "dispatch_tensor_gett_all",
    "tensor_gett_candidates",
    "tensor_gett_sweep_space",
]
