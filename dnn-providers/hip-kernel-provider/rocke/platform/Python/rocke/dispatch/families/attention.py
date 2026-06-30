# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Attention / FMHA dispatcher family (path-level selection).

Worked implementation mirroring :mod:`rocke.dispatch.gemm.bf16_rcr`, backed by
:mod:`rocke.instances.common.attention_unified` (the unified tiled FMHA
emitter).

SCOPE -- what this dispatcher decides
-------------------------------------
The load-bearing dispatch decision for unified attention is the *kernel path*:
the 2D-tiled (per-(kv_head, q_block) CTA) kernel vs the 3D split-KV kernel. That
decision is a PURE function of the problem
(:func:`rocke.helpers.attention.use_2d_kernel`, surfaced as
``UnifiedAttentionProblem.select_path``), so it can be mirrored byte-faithfully
on the C++ side. Backend coverage is gated by
:func:`attention_unified.supports_native_unified_attention` (head_size /
block_size / dtype / feature gate -- also pure).

The structural identity used for selection parity is therefore::

    (path, head_size, block_size)

where ``path`` is ``"2d"`` or ``"3d"``.

DEFERRED -- arch-tuned block geometry
-------------------------------------
The 2D-tiled kernel's exact CTA geometry (``num_warps`` / ``block_m_per_warp`` /
``tile_size``) is chosen by heuristics in ``attention_unified`` that query the
running device arch (``_resolve_attention_arch``) and encode many measured,
shape-specific thresholds (see ``_select_2d_num_warps`` et al.). Those are
downstream PERFORMANCE-TUNING knobs, not a "which kernel family" decision, and
they are not reproducible CPU-only without a device. They are intentionally OUT
of the parity identity here; modelling them faithfully across C++/Python is a
separate, larger effort. This dispatcher selects the path + backend; geometry is
left to the instance builder at launch time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence, Tuple

from ...core.arch import ArchTarget
from ...instances.common.attention_unified import (
    UnifiedAttentionProblem,
    supports_native_unified_attention,
)
from ..core import (
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    stable_json_hash,
)

_FAMILY = "attention_unified"
ATTENTION_ABI_VERSION = "hipkg-attention-unified/v1"


@dataclass(frozen=True)
class AttentionRequest(OperatorRequest):
    """Normalized scaled-dot-product-attention request."""

    batch: int
    nhead_q: int
    nhead_k: int
    seqlen_q: int
    seqlen_k: int
    hdim_q: int
    hdim_v: int
    arch: str
    mask_type: int = 0  # 0=none, 1=causal/top-left, ...
    use_sinks: bool = False
    sliding_window: int = 0
    kv_block_size: int = 16  # paged KV block_size (modulus); {16,32,64}
    num_sms: int = 120
    op: str = "attention"
    dtype: str = "fp16"
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = self.dtype.lower()
        return d


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, AttentionRequest):
        return [f"expected AttentionRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "attention":
        errors.append(f"unsupported op {req.op!r}")
    for field in ("batch", "nhead_q", "nhead_k", "seqlen_q", "seqlen_k", "hdim_q"):
        if int(getattr(req, field)) <= 0:
            errors.append(f"{field} must be positive")
    if req.hdim_q != req.hdim_v:
        errors.append("only hdim_q == hdim_v is supported")
    if int(req.nhead_q) % int(req.nhead_k):
        errors.append("nhead_q must be divisible by nhead_k (GQA grouping)")
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    return errors


def _problem(req: AttentionRequest) -> UnifiedAttentionProblem:
    # total_q = batch * seqlen_q (the flattened query rows). num_seqs = batch.
    return UnifiedAttentionProblem(
        total_q=int(req.batch) * int(req.seqlen_q),
        num_seqs=int(req.batch),
        num_query_heads=int(req.nhead_q),
        num_kv_heads=int(req.nhead_k),
        head_size=int(req.hdim_q),
        block_size=int(req.kv_block_size),
        max_seqlen_q=int(req.seqlen_q),
        max_seqlen_k=int(req.seqlen_k),
        dtype=req.dtype.lower(),
        sliding_window=int(req.sliding_window),
        use_sinks=bool(req.use_sinks),
        num_sms=int(req.num_sms),
    )


def _selector_matches(
    req: AttentionRequest, candidate: KernelCandidate
) -> Tuple[bool, str]:
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"


@dataclass(frozen=True)
class AttentionSpec:
    """The selected attention path + the dims that determine the kernel family."""

    path: str  # "2d" | "3d"
    head_size: int
    block_size: int
    dtype: str
    num_query_heads: int
    num_kv_heads: int
    name: str = "rocke_attention_unified"

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            self.path,
            self.dtype,
            f"hd{self.head_size}",
            f"bs{self.block_size}",
            f"gqa{self.num_query_heads}x{self.num_kv_heads}",
        )


def _make_candidate(*, path: str, priority: int) -> KernelCandidate:
    spec_id = f"unified_{path}"
    name = f"attention_unified_{path}"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        problem = _problem(req)
        ok, why = supports_native_unified_attention(problem)
        if not ok:
            return False, why
        if problem.select_path() != path:
            return False, (
                f"problem routes to {problem.select_path()!r} path, not {path!r}"
            )
        return True, "ok"

    def select(req: OperatorRequest) -> AttentionSpec:
        ok, why = support(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, AttentionRequest)
        problem = _problem(req)
        return AttentionSpec(
            path=path,
            head_size=problem.head_size,
            block_size=problem.block_size,
            dtype=problem.dtype,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=f"unified_{path}",
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=priority,
        supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=lambda spec, req: (0, 0, 0),  # geometry deferred (see module doc)
        block=lambda spec: (0, 0, 0),
        sweep_space=lambda req: (select(req),) if support(req)[0] else (),
    )
    return candidate


ATTENTION_REGISTRY = CandidateRegistry(_FAMILY)
ATTENTION_REGISTRY.extend(
    (
        # 2d and 3d are mutually exclusive per problem (select_path returns one),
        # so priority only orders the two when both could match -- which they
        # cannot. Equal priority keeps the registry order stable.
        _make_candidate(path="2d", priority=10),
        _make_candidate(path="3d", priority=10),
    )
)


def attention_candidates() -> Tuple[KernelCandidate, ...]:
    return ATTENTION_REGISTRY.candidates()


def _kernel_id(
    req: AttentionRequest, candidate: KernelCandidate, spec: AttentionSpec
) -> KernelId:
    request_hash = stable_json_hash(req.normalized(), n=16)
    spec_hash = stable_json_hash(asdict(spec), n=16)
    return KernelId(
        op="attention",
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=request_hash,
        spec_hash=spec_hash,
    )


def attention_sweep_space(req: OperatorRequest) -> Sequence[AttentionSpec]:
    if _request_errors(req):
        return ()
    specs = []
    seen = set()
    for candidate in ATTENTION_REGISTRY.supported(req):
        spec = candidate.select_spec(req)
        h = stable_json_hash(asdict(spec), n=16)
        if h not in seen:
            seen.add(h)
            specs.append(spec)
    return tuple(specs)


def dispatch_attention(
    req: AttentionRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select the unified attention kernel PATH for ``req``.

    Returns the 2D-tiled or 3D split-KV path (a pure function of the problem),
    gated by the native-backend coverage predicate. The CTA geometry is left to
    the instance builder (see module docstring -- deferred from parity).
    """
    candidate = ATTENTION_REGISTRY.select(req, ranker=ranker)
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
            f"selected {candidate.name} ({spec.path} path) on {req.arch}",
            f"algorithm={candidate.algorithm}",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )
