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

The candidate's ``sweep_space`` is its knob neighbourhood (K tile, stage
count, persistent grid, paired 16-byte epilogue stores), for the tuning
engine to search. Python engine only (no C++ builder).
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
#: the vector epilogue hook.
TENSOR_GETT_KC_C_PREFETCH = 0
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


def _spec_mn(req: TensorGettRequest):
    """The first valid MN-layout GETT spec over :func:`_mn_bodies` (with the
    16-byte epilogue when it applies), or None."""
    p = req.problem
    valid = []
    for tile, trait in _mn_bodies(req):
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
    c0, bw = TENSOR_PACK_COST[spec.resolved_form()]
    return c0 + 4.0 * spec.numel() / bw


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


def _spec_kc(req: TensorGettRequest):
    """The K-contiguous GETT spec on the GEMM dispatcher's RCR pick, or None.
    The pick's epilogue decides which GEMM operand owns D's fastest mode (the
    row-interleave vector epilogue stores 8 columns per lane)."""
    from ...instances.common.gemm_universal import UniversalGemmSpec

    p = req.problem
    disp = _gemm_dispatch(req)
    for role in ("B", "A"):
        try:
            plan = _kc_plan(p, role)
        except ValueError:
            continue
        try:
            r = disp(_gemm_req(req, plan.M, plan.N, plan.K, "RCR"))
        except (ValueError, KeyError):
            continue
        if not isinstance(r.spec, UniversalGemmSpec):
            continue
        tr = r.spec.trait
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
            kc_tile=r.spec.tile,
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
    """The product route for ``req`` (see the module doc)."""
    p = req.problem
    mn = TENSOR_GETT_REGISTRY.get("tensor_gett_mn")
    mn_plan = _plan_mn(req)
    policy = _route_policy()
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


def _plan_kc(req: TensorGettRequest):
    """The K-contiguous route (both operands packed [rows, K]), or None."""
    from ...instances.common.tensor import kc_operand_modes, launch_operands
    from ...instances.common.tensor.pack import (
        TensorPackSpec,
        tensor_pack_block,
        tensor_pack_grid,
    )
    from ...instances.common.tensor.pack import is_valid_spec as _pack_valid

    sp = _spec_kc(req)
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
        packs.append(
            TensorPackStep(
                nm, ps, tensor_pack_grid(ps), tensor_pack_block(ps), 2 * ps.numel()
            )
        )
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


def _plan_mn(req: TensorGettRequest) -> "TensorGettPlan":
    """The MN-layout routes: single launch, else the pack hybrid, else the
    ring body."""
    from ...instances.common.tensor.pack import (
        TensorPackSpec,
        tensor_pack_block,
        tensor_pack_grid,
    )
    from ...instances.common.tensor.pack import is_valid_spec as _pack_valid

    p = req.problem
    mn = TENSOR_GETT_REGISTRY.get("tensor_gett_mn")
    if mn.admits(req)[0] or not TENSOR_GETT_PACK_HYBRID:
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
            packs.append(
                TensorPackStep(
                    nm, ps, tensor_pack_grid(ps), tensor_pack_block(ps), 2 * ps.numel()
                )
            )
            if nm == "A":
                newp = replace(newp, modes_a=order, strides_a=None)
            else:
                newp = replace(newp, modes_b=order, strides_b=None)
    if packs:
        preq = replace(req, problem=newp)
        if mn.admits(preq)[0]:
            r = dispatch_tensor_gett(preq)
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
    r = dispatch_tensor_gett(req)
    return TensorGettPlan(
        (), r, 0, tuple(why) + (f"single launch: {r.candidate.name}",)
    )


def gett_operand_sources(plan: TensorGettPlan) -> tuple:
    """For the GETT launch's (A, B) params: ``("user", "A" | "B")`` or
    ``("pack", step index)``."""
    from ...instances.common.tensor import launch_operands

    ga, gb = launch_operands(plan.gett.spec)
    idx = {s.operand: i for i, s in enumerate(plan.packs)}
    return tuple(("pack", idx[n]) if n in idx else ("user", n) for n in (ga, gb))


__all__ = [
    "TENSOR_GETT_PACK_HYBRID",
    "TENSOR_GETT_PACK_ORDER",
    "TensorGettPlan",
    "TensorPackStep",
    "gett_operand_sources",
    "plan_tensor_gett",
    "TENSOR_GETT_REGISTRY",
    "TensorGettRequest",
    "dispatch_tensor_gett",
    "dispatch_tensor_gett_all",
    "tensor_gett_candidates",
    "tensor_gett_sweep_space",
]
