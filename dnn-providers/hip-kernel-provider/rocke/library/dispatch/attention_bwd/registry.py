# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Candidates, ranker and entry points of the attention backward dispatcher.

The backward follows the forward dispatcher's shape
(:mod:`dispatch.attention`, on :mod:`rocke.dispatch.core`): one
:class:`~rocke.dispatch.core.KernelCandidate` per (family, arch) in
:data:`ATTENTION_BWD_REGISTRY`, a ranker, and :func:`dispatch_attention_bwd`
returning a :class:`~rocke.dispatch.core.DispatchResult`.

* **Request.** :class:`AttentionBwdRequest` wraps the family-level request
  (:class:`~kernels.common.attention_bwd_plan.AttnBwdRequest`, the hipDNN
  backward; or :class:`~kernels.common.attention_bwd_unified_plan.AttnBwdUnifiedRequest`,
  the rocKE paged-KV entry, which carries the ``paged_kv`` feature) with the
  target ``arch`` and the ``algorithm`` / ``spec_id`` pins.
* **Candidates** (:func:`register`): ``general`` on every declared backward
  target, ``dense`` on the targets with a dense family, ``unified`` on the
  targets the unified suite validated. Each declares its coverage as a
  :class:`~rocke.dispatch.core.Capability` (arch, dtype, head size, the
  ``paged_kv`` feature) and keeps the exact predicate in ``_supports``:
  the family's own (``attn_bwd_support`` under the routed policy,
  ``dense_bwd_eligibility`` plus the dense plan, ``attn_bwd_unified_support``)
  and the known-wrong values (:data:`~.tuning_common.BWD_KNOWN_WRONG_KNOBS`)
  of the plan it would serve. A candidate's spec is the plan of its family
  (:class:`~kernels.common.attention_bwd_plan.AttnBwdPlan`: every launch, the
  runtime values, the workspace); ``grid`` / ``block`` are the main launch's.
  ``build`` is not declared (a plan is several kernels):
  :func:`launch_attention_bwd` compiles and launches a result.
* **Ranker** (:func:`attention_bwd_ranker`): the tuned tables first, then the
  defaults. A candidate whose family has a tuned entry for the request comes
  first, the most specific table first (the dense pack is keyed by the exact
  shape, the general table by class and buckets, the unified page tiles by
  head size and page); every other candidate follows in registered
  ``(priority, name)`` order, where the general family precedes the dense
  one. So a dense-eligible request goes to the dense family exactly when its
  dense key is in the tuned dense pack, as :func:`.routing.attention_bwd_route`
  decides; without the ranker the registered order serves the general family.
* **Entry points.** :func:`dispatch_attention_bwd` (selection; a declined
  request raises ``ValueError`` with the predicate's reason code),
  :func:`dispatch_attention_bwd_all` (every admitting candidate),
  :func:`launch_attention_bwd` (compile and launch a result on caller-owned
  buffers, after :func:`check_attention_bwd_tensors` when the caller passes
  its tensors' metadata) and :func:`run_attention_bwd` (both). The hipDNN provider's
  backward engine is the follow-up that calls this entry from the provider;
  the provider path is C++ and does not serve the backward yet.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, MutableMapping
from types import SimpleNamespace
from typing import Any, Optional, Sequence, Tuple, Union

from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    ShapeRange,
    normalize_selector,
    selector_matches,
    stable_json_hash,
)

from kernels.common.attention_bwd import (
    ATTN_BWD_ABI,
    BWD_DTYPES,
    BWD_HEAD_SIZES,
)
from kernels.common.attention_bwd_plan import (
    DECLARED_ARCHS,
    AttnBwdPlan,
    AttnBwdRequest,
    _dense_key,
    _table_knobs,
    attn_bwd_support,
    attn_bwd_tuning_key,
    dense_bwd_eligibility,
)
from kernels.common.attention_bwd_unified import ATTN_BWD_UNIFIED_ABI
from kernels.common.attention_bwd_unified_plan import (
    VALIDATED_ARCHS,
    AttnBwdUnifiedRequest,
    attn_bwd_unified_support,
)
from kernels.common.attention_dense_bwd_spec import (
    ATTN_DENSE_BWD_ABI,
    DENSE_BWD_ARCHS,
)

from . import routing
from .tuning_common import known_wrong_findings
from .unified_tuning import UNIFIED_TUNED_PAGE_TILES

__all__ = [
    "ALGORITHMS",
    "ATTENTION_BWD_DIM_VOCABULARY",
    "ATTENTION_BWD_REGISTRY",
    "FAMILY",
    "PAGED_KV",
    "AttentionBwdRequest",
    "attention_bwd_candidates",
    "attention_bwd_ranker",
    "check_attention_bwd_tensors",
    "dispatch_attention_bwd",
    "dispatch_attention_bwd_all",
    "launch_attention_bwd",
    "plan_known_wrong",
    "priority_ranker",
    "register",
    "run_attention_bwd",
]

FAMILY = "attention_bwd"
# The families the dispatcher selects among (a candidate's algorithm).
ALGORITHMS = ("general", "dense", "unified")
# The feature of a rocKE unified (paged KV) request.
PAGED_KV = "paged_kv"
_ABI = {
    "general": ATTN_BWD_ABI,
    "dense": ATTN_DENSE_BWD_ABI,
    "unified": ATTN_BWD_UNIFIED_ABI,
}
# Registered order without a ranker: the general family before the dense one
# (the dense family serves a request only through its tuned dense key).
_PRIORITY = {"general": 10, "dense": 20, "unified": 10}
# The main launch of each family's plan.
_MAIN_STAGE = {"general": "main", "dense": "dense", "unified": "main"}
_GENERAL_DIMS = ("b", "h_q", "h_k", "h_v", "s_q", "s_kv", "d_qk", "d_v")
_UNIFIED_DIMS = (
    "num_seqs",
    "h_q",
    "h_kv",
    "d_qk",
    "d_v",
    "total_q",
    "max_seqlen_q",
    "max_seqlen_kv",
    "page",
    "num_blocks",
    "max_blocks_per_seq",
)
ATTENTION_BWD_DIM_VOCABULARY = frozenset(_GENERAL_DIMS + _UNIFIED_DIMS)


def _json(x):
    """Plain JSON form of a request or plan field (for hashes)."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _json(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, Mapping):
        return {
            str(k): _json(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(x, (tuple, list)):
        return [_json(v) for v in x]
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return repr(x)


@dataclasses.dataclass(frozen=True)
class AttentionBwdRequest(OperatorRequest):
    """One attention backward problem on one target.

    ``problem``: the general request (hipDNN backward) or the unified (paged
    KV) request; ``arch``: the target; ``algorithm`` / ``spec_id``: pins
    (``"auto"`` or one candidate's value; :data:`ALGORITHMS`).
    """

    problem: Union[AttnBwdRequest, AttnBwdUnifiedRequest]
    arch: str
    algorithm: str = "auto"
    spec_id: str = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.problem, (AttnBwdRequest, AttnBwdUnifiedRequest)):
            raise TypeError(
                f"problem must be an AttnBwdRequest or AttnBwdUnifiedRequest, got "
                f"{type(self.problem).__name__}"
            )
        for name in ("arch", "algorithm", "spec_id"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a str")

    @property
    def paged(self) -> bool:
        """A rocKE unified (paged KV) request."""
        return isinstance(self.problem, AttnBwdUnifiedRequest)

    def normalized(self) -> dict:
        return {
            "op": FAMILY,
            "arch": self.arch,
            "algorithm": normalize_selector(self.algorithm),
            "spec_id": normalize_selector(self.spec_id),
            "entry": "unified" if self.paged else "general",
            "dtype": self.problem.dtype,
            "problem": _json(self.problem),
        }

    def dims(self) -> Mapping[str, int]:
        names = _UNIFIED_DIMS if self.paged else _GENERAL_DIMS
        return {n: getattr(self.problem, n) for n in names}

    def features(self) -> frozenset:
        return frozenset({PAGED_KV}) if self.paged else frozenset()


def plan_known_wrong(plan: AttnBwdPlan) -> Tuple[str, ...]:
    """Known-wrong findings (:data:`~.tuning_common.BWD_KNOWN_WRONG_KNOBS`,
    ``"knob=value"``) of every knob a plan carries: its main spec (and the
    dense family's resolved body spec), its dQ kernel spec (as ``dq_*``
    knobs) and its runtime values."""
    arch = plan.arch
    out = set(known_wrong_findings(arch, dict(plan.runtime)))
    for stage, spec in plan.specs.items():
        if not dataclasses.is_dataclass(spec):
            continue
        fields = {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec)}
        if stage in ("dq", "dense_dq"):
            fields = {f"dq_{k}": v for k, v in fields.items()}
        elif stage not in ("main", "dense"):
            continue  # aux kernels: their knobs are runtime values
        out.update(known_wrong_findings(arch, fields))
        if stage == "dense":
            body = spec.body_spec().resolved(arch)
            body_fields = {
                f.name: getattr(body, f.name) for f in dataclasses.fields(body)
            }
            out.update(known_wrong_findings(arch, body_fields))
    return tuple(sorted(out))


def _plan(req: AttentionBwdRequest, algorithm: str) -> AttnBwdPlan:
    """The plan of ``algorithm``'s family for ``req`` (``ValueError`` when
    the family cannot serve it)."""
    if algorithm == "unified":
        return routing.plan_attention_bwd_unified(req.problem, req.arch)
    return routing.plan_attention_bwd(req.problem, req.arch, route=algorithm)


def _verdict(req: AttentionBwdRequest, algorithm: str) -> Tuple[bool, str]:
    """The family's exact verdict for ``req`` (after the capability)."""
    if algorithm == "unified":
        v = attn_bwd_unified_support(req.problem, req.arch)
        if not v.ok:
            return False, f"{v.code}: {v.detail}"
    else:
        v = attn_bwd_support(req.problem, req.arch, policy=routing.routed_policy())
        if not v.ok:
            return False, f"{v.code}: {v.detail}"
        if algorithm == "dense":
            ok, tag = dense_bwd_eligibility(req.problem, req.arch)
            if not ok:
                return False, f"not dense-eligible: {tag}"
    try:
        plan = _plan(req, algorithm)
    except ValueError as ex:
        return False, str(ex)
    wrong = plan_known_wrong(plan)
    if wrong:
        return False, f"known wrong on {req.arch}: {', '.join(wrong)}"
    return True, "ok"


def _main_step(plan: AttnBwdPlan, algorithm: str):
    stage = _MAIN_STAGE[algorithm]
    for step in plan.launches:
        if step.stage == stage:
            return step
    raise AssertionError(f"{algorithm} plan without a {stage!r} launch")


def _signature(plan: AttnBwdPlan) -> Tuple[dict, ...]:
    """One entry per launch of the plan, in launch order."""
    return tuple(
        {
            "stage": s.stage,
            "kernel": s.kernel,
            "grid": list(s.grid),
            "block": int(s.block),
            "scalars": sorted(s.scalars),
            "pointers": sorted(s.pointers),
        }
        for s in plan.launches
    )


def _make_candidate(algorithm: str, arch: str) -> KernelCandidate:
    spec_id = f"attn_bwd_{algorithm}"
    pins = SimpleNamespace(algorithm=algorithm, spec_id=spec_id)
    paged = algorithm == "unified"

    def supports(req: OperatorRequest) -> Tuple[bool, str]:
        if not isinstance(req, AttentionBwdRequest):
            return False, f"expected AttentionBwdRequest, got {type(req).__name__}"
        ok, why = selector_matches(req, pins)
        if not ok:
            return False, why
        return _verdict(req, algorithm)

    def select_spec(req: AttentionBwdRequest) -> AttnBwdPlan:
        return _plan(req, algorithm)

    return KernelCandidate(
        name=f"attention_bwd_{algorithm}_{arch}",
        family=FAMILY,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=_ABI[algorithm],
        priority=_PRIORITY[algorithm],
        _supports=supports,
        select_spec=select_spec,
        signature=_signature,
        grid=lambda plan, _req: tuple(_main_step(plan, algorithm).grid),
        block=lambda plan: (int(_main_step(plan, algorithm).block), 1, 1),
        sweep_space=lambda req: (select_spec(req),),
        capability=Capability(
            arches=(arch,),
            dtypes=tuple(BWD_DTYPES),
            shapes=(ShapeRange("d_qk", allowed=tuple(BWD_HEAD_SIZES)),),
            supports_features=frozenset({PAGED_KV}) if paged else frozenset(),
            requires_features=frozenset({PAGED_KV}) if paged else frozenset(),
        ),
    )


def register(registry: CandidateRegistry) -> None:
    """Register one candidate per (family, arch): ``general`` on
    :data:`~kernels.common.attention_bwd_plan.DECLARED_ARCHS`, ``dense`` on
    :data:`~kernels.common.attention_dense_bwd_spec.DENSE_BWD_ARCHS`,
    ``unified`` on the unified family's validated targets."""
    for arch in DECLARED_ARCHS:
        registry.register(_make_candidate("general", arch))
    for arch in DENSE_BWD_ARCHS:
        registry.register(_make_candidate("dense", arch))
    for arch in sorted(VALIDATED_ARCHS):
        registry.register(_make_candidate("unified", arch))


ATTENTION_BWD_REGISTRY = CandidateRegistry(
    FAMILY, dim_vocabulary=ATTENTION_BWD_DIM_VOCABULARY
)
register(ATTENTION_BWD_REGISTRY)


def attention_bwd_candidates() -> Tuple[KernelCandidate, ...]:
    return ATTENTION_BWD_REGISTRY.candidates()


def _tuned_rank(req: AttentionBwdRequest, candidate: KernelCandidate) -> Optional[int]:
    """The rank of ``candidate``'s tuned entry for ``req`` (lower: more
    specific), or ``None`` when its family's tables have no entry."""
    prob, arch = req.problem, req.arch
    if candidate.algorithm == "unified":
        if (arch, prob.d_qk, prob.page) in UNIFIED_TUNED_PAGE_TILES:
            return 1
        return None
    if req.paged:
        return None
    policy = routing.routed_policy()
    if candidate.algorithm == "dense":
        return 0 if (arch, _dense_key(prob)) in policy.dense_pack else None
    if _table_knobs(policy.tuned_table, attn_bwd_tuning_key(prob, arch)):
        return 1
    return None


def attention_bwd_ranker(
    request: OperatorRequest, candidates: Sequence[KernelCandidate]
) -> Sequence[KernelCandidate]:
    """The tuned tables first, then the defaults (see the module docstring).

    ``candidates`` are the supported ones in registered order; a candidate
    whose family has a tuned entry for the request moves to the front, the
    most specific table first (the dense pack, then the general table and the
    unified page tiles). The order is otherwise kept.
    """
    if not isinstance(request, AttentionBwdRequest):
        return candidates
    tuned, rest = [], []
    for c in candidates:
        rank = _tuned_rank(request, c)
        (rest if rank is None else tuned).append((rank, c))
    tuned.sort(key=lambda rc: rc[0])  # stable: registered order within a rank
    return [c for _r, c in tuned] + [c for _r, c in rest]


def priority_ranker(
    request: OperatorRequest, candidates: Sequence[KernelCandidate]
) -> Sequence[KernelCandidate]:
    """Registered ``(priority, name)`` order (no table consulted)."""
    return candidates


def _kernel_id(
    req: AttentionBwdRequest, candidate: KernelCandidate, plan: AttnBwdPlan
) -> KernelId:
    """Stable identity of one pick; the spec hash covers the plan's launches
    (salted kernel names, grids, kernarg values), runtime values and
    workspace."""
    ident = {
        "route": plan.route,
        "launches": [
            [s.stage, s.kernel, list(s.grid), int(s.block), _json(s.scalars)]
            for s in plan.launches
        ],
        "runtime": _json(plan.runtime),
        "workspace_bytes": int(plan.workspace_bytes),
    }
    return KernelId(
        op=FAMILY,
        family=candidate.family,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=stable_json_hash(req.normalized(), n=16),
        spec_hash=stable_json_hash(ident, n=16),
    )


def _result(
    req: AttentionBwdRequest, candidate: KernelCandidate, what: str
) -> DispatchResult:
    plan = candidate.select_spec(req)
    kid = _kernel_id(req, candidate, plan)
    return DispatchResult(
        request=req,
        candidate=candidate,
        spec=plan,
        kernel_id=kid,
        grid=candidate.grid(plan, req),
        block=candidate.block(plan),
        signature=tuple(candidate.signature(plan)),
        explanation=(
            f"{what} {candidate.name} ({candidate.algorithm}) on {req.arch}",
            f"route={plan.route} ({plan.route_tag})",
            f"tuning_key={plan.tuning_key!r}",
            f"perf_status={plan.perf_status}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )


def _decline_error(req: AttentionBwdRequest, reasons: Sequence[str]) -> ValueError:
    """The predicate's reason code first (the hipDNN decline), then every
    candidate's reason."""
    if req.paged:
        v = attn_bwd_unified_support(req.problem, req.arch)
    else:
        v = attn_bwd_support(req.problem, req.arch, policy=routing.routed_policy())
    head = f"{v.code}: {v.detail}" if not v.ok else "no candidate admits the request"
    tail = "; ".join(reasons) if reasons else "no candidate registered"
    return ValueError(f"{head} ({tail})")


def dispatch_attention_bwd(
    req: AttentionBwdRequest,
    *,
    ranker: Ranker | None = None,
    registry: CandidateRegistry | None = None,
) -> DispatchResult:
    """Select the backward family for ``req`` and plan it.

    ``ranker`` (default :func:`attention_bwd_ranker`) orders the supported
    candidates; ``registry`` defaults to :data:`ATTENTION_BWD_REGISTRY`. A
    request no candidate admits raises ``ValueError`` led by the predicate's
    reason code. The result's ``spec`` is the plan (every launch, the runtime
    values and the workspace size the caller provides).
    """
    if not isinstance(req, AttentionBwdRequest):
        raise TypeError(f"expected AttentionBwdRequest, got {type(req).__name__}")
    registry = ATTENTION_BWD_REGISTRY if registry is None else registry
    try:
        candidate = registry.select(req, ranker=ranker or attention_bwd_ranker)
    except ValueError:
        if registry.supported(req):
            raise  # the ranker broke the registry's contract
        reasons = [f"{c.name}: {c.admits(req)[1]}" for c in registry.for_arch(req.arch)]
        raise _decline_error(req, reasons) from None
    return _result(req, candidate, "selected")


def dispatch_attention_bwd_all(
    req: AttentionBwdRequest, *, registry: CandidateRegistry | None = None
) -> Tuple[DispatchResult, ...]:
    """One result per candidate that admits ``req``, ranked by
    :func:`attention_bwd_ranker` (the first is :func:`dispatch_attention_bwd`'s
    choice). For gate runs that time every family on the same request."""
    if not isinstance(req, AttentionBwdRequest):
        raise TypeError(f"expected AttentionBwdRequest, got {type(req).__name__}")
    registry = ATTENTION_BWD_REGISTRY if registry is None else registry
    ranked = attention_bwd_ranker(req, registry.supported(req))
    return tuple(_result(req, c, "candidate") for c in ranked)


def check_attention_bwd_tensors(
    result: DispatchResult, tensor_meta: Mapping[str, Any]
) -> None:
    """Check the caller's tensors against the dispatched request
    (``ValueError`` with the check's code): ``tensor_meta`` maps the plan's
    tensor roles to :class:`kernels.common.attention_bwd_run.TensorMeta`
    (shape, strides, dtype, addressable bytes; ``TensorMeta.of(tensor)`` for a
    torch tensor). General and dense: :func:`check_attn_bwd_tensors`;
    unified: :func:`check_attn_bwd_unified_inputs`."""
    req = result.request
    if req.paged:
        from kernels.common.attention_bwd_unified_run import (
            check_attn_bwd_unified_inputs,
        )

        check_attn_bwd_unified_inputs(req.problem, tensors=tensor_meta)
        return
    from kernels.common.attention_bwd_run import check_attn_bwd_tensors

    check_attn_bwd_tensors(req.problem, tensor_meta)


def launch_attention_bwd(
    result: DispatchResult,
    *,
    tensors: Mapping[str, int],
    workspace: int,
    workspace_size: int,
    stream: int = 0,
    cache: MutableMapping | None = None,
    tensor_meta: Optional[Mapping[str, Any]] = None,
) -> list:
    """Compile (cached) and launch a dispatched backward on ``stream``.

    The caller owns every buffer: ``tensors`` maps the plan's pointer roles
    to device addresses and ``workspace`` is a ``WORKSPACE_ALIGN``-aligned
    buffer of ``workspace_size >= result.spec.workspace_bytes`` bytes
    (see :func:`kernels.common.attention_bwd_run.run_attn_bwd` /
    :func:`kernels.common.attention_bwd_unified_run.run_attn_bwd_unified`).
    The pointers are trusted; a caller that holds the tensors passes their
    ``tensor_meta`` and they are checked against the request first
    (:func:`check_attention_bwd_tensors`), before anything is compiled or
    launched. Returns the launched kernel names in order.
    """
    if tensor_meta is not None:
        check_attention_bwd_tensors(result, tensor_meta)
    plan = result.spec
    kw = dict(
        tensors=tensors,
        workspace=workspace,
        workspace_size=workspace_size,
        stream=stream,
        cache=cache,
    )
    if result.candidate.algorithm == "unified":
        from kernels.common.attention_bwd_unified_run import run_attn_bwd_unified

        return run_attn_bwd_unified(plan, **kw)
    from kernels.common.attention_bwd_run import run_attn_bwd

    return run_attn_bwd(plan, **kw)


def run_attention_bwd(
    req: AttentionBwdRequest,
    *,
    tensors: Mapping[str, int],
    workspace: int,
    workspace_size: int,
    stream: int = 0,
    cache: MutableMapping | None = None,
    ranker: Ranker | None = None,
    tensor_meta: Optional[Mapping[str, Any]] = None,
) -> Tuple[DispatchResult, list]:
    """Dispatch ``req`` and launch the selection
    (:func:`dispatch_attention_bwd` then :func:`launch_attention_bwd`, with
    the optional ``tensor_meta`` check). Returns ``(result, launched kernel
    names)``; a workspace smaller than the plan's or a tensor that does not
    match the request raises before anything is compiled."""
    result = dispatch_attention_bwd(req, ranker=ranker)
    if workspace_size < result.spec.workspace_bytes:
        raise ValueError(
            f"workspace of {workspace_size} bytes is smaller than the plan's "
            f"{result.spec.workspace_bytes}"
        )
    names = launch_attention_bwd(
        result,
        tensors=tensors,
        workspace=workspace,
        workspace_size=workspace_size,
        stream=stream,
        cache=cache,
        tensor_meta=tensor_meta,
    )
    return result, names
