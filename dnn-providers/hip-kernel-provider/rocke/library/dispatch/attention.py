# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Attention / FMHA dispatcher family (path-level selection).

Worked implementation mirroring :mod:`rocke.dispatch.gemm.bf16_rcr`, backed by
:mod:`kernels.common.attention_unified` (the unified tiled FMHA
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

from rocke.core.arch import ArchTarget
from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    supports_native_unified_attention,
)
from rocke.dispatch.core import (
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
    # --- gfx950 attention_dense knobs (only consumed by the opt-in
    #     ``attention_dense`` candidate; ignored by the unified 2D/3D paths).
    #     Defaults deliver the persistent ~970-TFLOPS prefill path for large Sq:
    #     ``dense_persistent="auto"`` turns on the grid-stride variant once there
    #     is enough work to fill the persistent grid, and ``persist_decode="auto"``
    #     picks the L2-locality hkv-major decode where it is balance-safe. ---
    dense_persistent: str = "auto"  # "auto" | "on" | "off"
    dense_num_persistent: int = 256
    dense_persist_decode: str = "auto"  # "auto" | "qb_major" | "hkv_major"

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
    # When set, this verbatim kernel_name is returned by :meth:`kernel_name`
    # instead of the composed unified name. Used by the dense candidate to
    # surface the concrete persistent/ragged/decode kernel it will launch.
    kernel_name_override: str = ""
    # Optional pinned codegen knobs a specialized candidate wants the builder to
    # apply (sorted (key, value) pairs; empty for generic candidates). See
    # ``attention_unified._d256_gfx950_spec_overrides``; the builder consumes them
    # via ``_tiled_spec_from_problem(problem, overrides=...)``.
    tiled_overrides: Tuple[Tuple[str, object], ...] = ()

    def kernel_name(self) -> str:
        if self.kernel_name_override:
            return self.kernel_name_override
        from rocke.helpers.spec import kernel_name_join

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


def _make_gfx942_dense_pipe_candidate() -> KernelCandidate:
    """Fast gfx942 fp16 prefill kernel — transposed-x8 flash with ring-sliced K.

    Registered at priority 5 so it outranks the generic unified_2d candidate
    (priority 10) whenever both would match the same gfx942 fp16 2D problem.
    The registry sorts ascending (lower = higher precedence).
    Callers can also force this path explicitly via algorithm="dense_pipe".
    """
    spec_id = "gfx942_dense_pipe"
    name = "attention_gfx942_dense_pipe"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        if req.arch != "gfx942":
            return False, f"dense_pipe requires gfx942 (got {req.arch!r})"
        if req.dtype != "fp16":
            return False, f"dense_pipe is fp16-only (got {req.dtype!r})"
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        problem = _problem(req)
        ok, why = supports_native_unified_attention(problem)
        if not ok:
            return False, why
        if problem.select_path() != "2d":
            return False, "problem routes to 3D, not 2D"
        from kernels.common.attention_unified import _enable_gfx942_fp16_flash

        if not _enable_gfx942_fp16_flash(problem):
            return False, "gfx942 fp16 flash not eligible for this shape"
        return True, "ok"

    def select(req: OperatorRequest) -> AttentionSpec:
        ok, why = support(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, AttentionRequest)
        problem = _problem(req)
        return AttentionSpec(
            path="2d",
            head_size=problem.head_size,
            block_size=problem.block_size,
            dtype=problem.dtype,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            name="rocke_attention_gfx942_dense_pipe",
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm="dense_pipe",
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=5,
        supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=lambda spec, req: (0, 0, 0),
        block=lambda spec: (0, 0, 0),
        sweep_space=lambda req: (select(req),) if support(req)[0] else (),
    )
    return candidate


ATTENTION_REGISTRY.register(_make_gfx942_dense_pipe_candidate())


# block_n (KV tile) the dense candidate ships; 64 is the resource-efficient
# peak (see AttentionDenseSpec.block_n).
_DENSE_BLOCK_N = 64


def _dense_spec(req: OperatorRequest):
    """Build the ``AttentionDenseSpec`` for a request at its best-performing
    config. Persistent ("auto") turns on the grid-stride ~970-TFLOPS variant once
    there is enough work to fill the persistent grid (``nqb*Hq*B >= num_persistent``
    -- the large-Sq prefill regime); ``persist_decode`` / ``lazy_rescale`` default
    to the L2-locality hkv-major decode and always-on lazy rescale. Non-tile-
    multiple self-attention lengths use the on-chip ragged path (no host pad)."""
    from kernels.gfx950.attention_dense import AttentionDenseSpec, _BLOCK_M

    assert isinstance(req, AttentionRequest)
    sq, sk = int(req.seqlen_q), int(req.seqlen_k)
    bn = _DENSE_BLOCK_N
    # on-chip ragged padding for ragged self-attention lengths (seqlen_q==seqlen_kv,
    # not a 256/block_n multiple). Cross-attention ragged is left to the validator.
    ragged = (sq == sk) and ((sq % _BLOCK_M != 0) or (sk % bn != 0))
    nqb = (sq + _BLOCK_M - 1) // _BLOCK_M
    work = nqb * int(req.nhead_q) * int(req.batch)
    np = int(req.dense_num_persistent)
    mode = req.dense_persistent.strip().lower()
    if mode == "on":
        persistent = True
    elif mode == "off":
        persistent = False
    elif mode == "auto":
        persistent = work >= np  # enough work to fill the persistent grid
    else:
        raise ValueError(
            f"dense_persistent must be 'auto'/'on'/'off', got {req.dense_persistent!r}"
        )
    return AttentionDenseSpec(
        batch=int(req.batch),
        seqlen_q=sq,
        seqlen_kv=sk,
        num_query_heads=int(req.nhead_q),
        num_kv_heads=int(req.nhead_k),
        head_size=int(req.hdim_q),
        causal=(int(req.mask_type) != 0),
        dtype=req.dtype.lower(),
        block_n=bn,
        persistent=persistent,
        num_persistent=np,
        persist_decode=req.dense_persist_decode.strip().lower(),
        ragged=ragged,
    )


def dense_spec_for_request(req: AttentionRequest):
    """Public builder: the launch-ready ``AttentionDenseSpec`` for ``req`` at its
    best config (see :func:`_dense_spec`). Pair with ``run_attention_dense_torch``
    to execute the dispatched dense candidate."""
    return _dense_spec(req)


def _make_gfx950_attention_dense_candidate() -> KernelCandidate:
    """Dense CK-1 persistent flash-attn prefill on gfx950 (bf16/fp16, causal/full).

    OPT-IN ONLY: matches solely when the request explicitly names
    ``algorithm="attention_dense"`` (or ``spec_id="gfx950_attention_dense"``), so it
    never auto-overrides the generic unified_2d path (no change to default routing).
    When selected it uses the persistent best-config from :func:`_dense_spec`
    (grid-stride + hkv-major + lazy for large Sq); the concrete kernel_name (incl.
    ``persist``/``hkvmaj``/``lazyrs``/``ragged``) is surfaced on the spec so the
    launched kernel is the fast path, not the default grid. End-to-end launch is
    ``run_attention_dense_torch(spec=dense_spec_for_request(req), ...)``.
    """
    spec_id = "gfx950_attention_dense"
    name = "attention_gfx950_dense"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        # Opt-in: never selected under algorithm/spec_id "auto".
        if req.algorithm.strip().lower() != "attention_dense" and (
            req.spec_id.strip().lower() != spec_id
        ):
            return False, "attention_dense is opt-in (algorithm='attention_dense')"
        if req.arch != "gfx950":
            return False, f"attention_dense requires gfx950 (got {req.arch!r})"
        if req.dtype.lower() not in ("bf16", "fp16"):
            return False, f"attention_dense is bf16/fp16 only (got {req.dtype!r})"
        if int(req.sliding_window) or bool(req.use_sinks):
            return False, "attention_dense is dense (no sliding-window / sinks)"
        from kernels.gfx950.attention_dense import supports_attention_dense

        try:
            spec = _dense_spec(req)
        except ValueError as e:
            return False, str(e)
        ok, why = supports_attention_dense(spec, arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> AttentionSpec:
        ok, why = support(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, AttentionRequest)
        problem = _problem(req)
        dense_spec = _dense_spec(req)
        return AttentionSpec(
            path="2d",
            head_size=problem.head_size,
            block_size=problem.block_size,
            dtype=problem.dtype,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            name="rocke_attention_dense",
            # surface the concrete persistent/hkvmaj/lazyrs/ragged kernel so the
            # dispatched spec names the fast path it will actually launch.
            kernel_name_override=dense_spec.kernel_name(),
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm="attention_dense",
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=3,
        supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=lambda spec, req: (0, 0, 0),
        block=lambda spec: (0, 0, 0),
        sweep_space=lambda req: (select(req),) if support(req)[0] else (),
    )
    return candidate


def _make_gfx950_d256_candidate() -> KernelCandidate:
    """Fast gfx950 bf16 head_size-256 prefill kernel — 32x32 transposed stack
    with FA3-style softmax<->MFMA interleave (mode2/g4) + slab-padded K_lds.

    Registered at priority 5 so it outranks the generic unified_2d candidate
    (priority 10) for the gfx950 bf16 D256 prefill cohort. The registry sorts
    ascending (lower = higher precedence); gfx950-only, so it never competes
    with the gfx942 dense_pipe candidate. Callers can also force this path
    explicitly via algorithm="d256_gfx950".

    The cohort is the single source of truth
    ``kernels.common.attention_unified._d256_gfx950_cohort`` — the same predicate
    the orchestrator's ``_d256_gfx950_fast`` override uses — so dispatch selection
    and the built spec cannot drift. Only the arch gate differs (request arch
    here vs resolved device arch there).
    """
    spec_id = "gfx950_d256"
    name = "attention_gfx950_d256"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        if req.arch != "gfx950":
            return False, f"d256_gfx950 requires gfx950 (got {req.arch!r})"
        if req.dtype != "bf16":
            return False, f"d256_gfx950 is bf16-only (got {req.dtype!r})"
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        problem = _problem(req)
        ok, why = supports_native_unified_attention(problem)
        if not ok:
            return False, why
        if problem.select_path() != "2d":
            return False, "problem routes to 3D, not 2D"
        from kernels.common.attention_unified import _d256_gfx950_cohort

        if not _d256_gfx950_cohort(problem):
            return False, "not the gfx950 bf16 D256 prefill fast-path cohort"
        return True, "ok"

    def select(req: OperatorRequest) -> AttentionSpec:
        ok, why = support(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, AttentionRequest)
        problem = _problem(req)
        from kernels.common.attention_unified import _d256_gfx950_spec_overrides

        return AttentionSpec(
            path="2d",
            head_size=problem.head_size,
            block_size=problem.block_size,
            dtype=problem.dtype,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            name="rocke_attention_gfx950_d256",
            tiled_overrides=tuple(sorted(_d256_gfx950_spec_overrides().items())),
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm="d256_gfx950",
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=5,
        supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=lambda spec, req: (0, 0, 0),
        block=lambda spec: (0, 0, 0),
        sweep_space=lambda req: (select(req),) if support(req)[0] else (),
    )
    return candidate


def _make_d256_decode_candidate() -> KernelCandidate:
    """D256 bf16 decode candidate for gfx950 and gfx942 — 3D split-KV path.

    Registered at priority 5 so it outranks the generic unified_3d candidate
    (priority 10) for eligible D256 bf16 decode shapes. Callers can also force
    this path explicitly via algorithm="d256_decode".
    """
    from kernels.common.attention_unified import _d256_decode_cohort

    spec_id = "d256_decode"
    name = "attention_d256_decode"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        if req.arch not in ("gfx942", "gfx950"):
            return False, f"d256_decode requires gfx942 or gfx950 (got {req.arch!r})"
        if req.dtype != "bf16":
            return False, f"d256_decode is bf16-only (got {req.dtype!r})"
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        problem = _problem(req)
        ok, why = supports_native_unified_attention(problem)
        if not ok:
            return False, why
        if not _d256_decode_cohort(problem):
            return False, "problem is not in the D256 bf16 decode cohort"
        if problem.select_path() != "3d":
            return False, (
                f"problem routes to {problem.select_path()!r} path, not '3d'"
            )
        return True, "ok"

    def select(req: OperatorRequest) -> AttentionSpec:
        ok, why = support(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, AttentionRequest)
        problem = _problem(req)
        return AttentionSpec(
            path="3d",
            head_size=problem.head_size,
            block_size=problem.block_size,
            dtype=problem.dtype,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            name="rocke_attention_d256_decode",
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm="d256_decode",
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=5,
        supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=lambda spec, req: (0, 0, 0),
        block=lambda spec: (0, 0, 0),
        sweep_space=lambda req: (select(req),) if support(req)[0] else (),
    )
    return candidate


ATTENTION_REGISTRY.register(_make_gfx950_attention_dense_candidate())
ATTENTION_REGISTRY.register(_make_gfx950_d256_candidate())
ATTENTION_REGISTRY.register(_make_d256_decode_candidate())


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
