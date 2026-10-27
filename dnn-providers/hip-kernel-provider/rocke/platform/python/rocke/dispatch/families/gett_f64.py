# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""f64 / complex-f64 GETT (tensor contraction / strided GEMM) dispatcher family.

A request carries one :class:`rocke.instances.common.gett_f64.GettF64Problem`
(one launch of :func:`rocke.instances.common.gett_f64_plan.plan_contraction`, or
a strided GEMM from :func:`rocke.instances.common.gett_f64.gemm_problem`) and the
target. Four candidates are registered on gfx90a / gfx942 / gfx950:

* ``gett_f64_pipe`` (first choice where it applies) -- the pipelined
  configuration of the request's pipe class
  (:func:`~rocke.instances.common.gett_f64.pipe_spec`: an instruction-level
  interleaved K loop, K-major LDS tiles for K-contiguous operands and AGPR
  accumulators; ``PIPE_CONFIGS``). These levers are built by the Python
  engine only; requests with ``runtime_sizes``, complex data, small grids and
  problems the tuned path splits along K skip it;
* ``gett_f64_tuned`` (first choice) -- the tuned configuration of the
  request's shape class (:func:`~rocke.instances.common.gett_f64.tuned_spec`:
  grid size per CU, K length, and on gfx90a the operands read along K),
  found with the search engine and reduced to a per-arch portfolio by
  cross-evaluation (``TUNED_CONFIGS``);
* ``gett_f64_t128`` -- 128x128 CTA tiles (64x64 for complex), the reference
  defaults of :func:`~rocke.instances.common.gett_f64.arch_spec`;
* ``gett_f64_t64`` -- 64x64 tiles, when the 128-tile grid would leave CUs
  idle.

Each candidate's ``sweep_space`` starts at its spec, then the other tuned
configurations of the arch and dtype, then one-knob moves, for the tuning
engine to search.

The shape classes need the target's CU count: pass ``num_cus``, or run where
the visible device is the request's arch (it is queried then). A request with
``runtime_sizes`` selects a kernel that takes the sizes and strides as
arguments (strided batched GEMMs): its spec carries the layout template, so
every problem of one layout and shape class maps to one kernel.

The complex dtype is named ``c128`` (two f64 per element, numpy / torch
``complex128``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Optional, Sequence, Tuple

from ...instances.common.gett_f64 import (
    F64_GETT_ARCHS,
    PIPE_CONFIGS,
    TUNED_CONFIGS,
    GettF64Problem,
    GettF64Spec,
    arch_spec,
    build_gett_f64,
    gett_f64_grid,
    gett_f64_signature,
    is_valid_spec,
    pipe_spec,
    resolve_num_cus,
    runtime_compatible,
    runtime_template,
    tuned_spec,
)
from ..core import (
    Capability,
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    stable_json_hash,
    selector_matches,
)

_FAMILY = "gett_f64"
GETT_F64_ABI_VERSION = "rocke-gett-f64/v2"


@dataclass(frozen=True)
class GettF64Request(OperatorRequest):
    """One f64 / cf64 GETT launch."""

    problem: GettF64Problem
    arch: str
    bilinear: bool = True
    num_cus: Optional[int] = None
    op: str = "contraction"
    algorithm: str = "auto"
    spec_id: str = "auto"
    runtime_sizes: bool = False

    @property
    def dtype(self) -> str:
        return "c128" if self.problem.complex_ else "f64"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = self.dtype
        return d

    def dims(self) -> dict[str, int]:
        p = self.problem
        return {"M": int(p.M), "N": int(p.N), "K": int(p.K), "L": int(p.L)}

    def cus(self) -> int:
        return resolve_num_cus(self.arch, self.num_cus)


GETT_F64_DIM_VOCABULARY = ("M", "N", "K", "L")


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, GettF64Request):
        return [f"expected GettF64Request, got {type(req).__name__}"]
    errors = []
    if req.op != "contraction":
        errors.append(f"unsupported op {req.op!r}")
    ok, why = req.problem.check()
    if not ok:
        errors.append(why)
    if req.runtime_sizes and runtime_template(req.problem) is None:
        errors.append("runtime_sizes needs one digit per index space (a strided GEMM)")
    return errors


def _tiles(req: GettF64Request, tile: int) -> int:
    p = req.problem
    return -(-p.M // tile) * -(-p.N // tile) * p.L


def _rt(req: GettF64Request, spec: GettF64Spec) -> GettF64Spec:
    """A runtime-size request's spec carries the layout template."""
    if req.runtime_sizes:
        return replace(spec, runtime_sizes=True, problem=runtime_template(req.problem))
    return spec


def _spec_tuned(req: GettF64Request) -> GettF64Spec:
    knobs = {"runtime_sizes": True} if req.runtime_sizes else {}
    return tuned_spec(
        req.problem, req.arch, bilinear=req.bilinear, num_cus=req.cus(), **knobs
    )


def _spec_pipe(req: GettF64Request) -> GettF64Spec:
    spec = pipe_spec(req.problem, req.arch, bilinear=req.bilinear, num_cus=req.cus())
    if spec is None:
        raise ValueError("no pipelined configuration applies")
    return spec


def _fits_pipe(req: GettF64Request) -> Tuple[bool, str]:
    if req.runtime_sizes:
        return False, "runtime_sizes: the pipelined loop needs compile-time sizes"
    spec = pipe_spec(req.problem, req.arch, bilinear=req.bilinear, num_cus=req.cus())
    if spec is None:
        return False, "no pipelined configuration applies to this problem"
    return True, "ok"


def _spec_t128(req: GettF64Request) -> GettF64Spec:
    return _rt(req, arch_spec(req.problem, req.arch, bilinear=req.bilinear))


def _spec_t64(req: GettF64Request) -> GettF64Spec:
    return _rt(
        req,
        arch_spec(req.problem, req.arch, bilinear=req.bilinear, tile_m=64, tile_n=64),
    )


def _supports_problem(spec: GettF64Spec, req: GettF64Request) -> Tuple[bool, str]:
    ok, why = is_valid_spec(spec, arch=req.arch)
    if ok and spec.runtime_sizes:
        ok, why = runtime_compatible(spec, req.problem)
    return ok, why


def _neighbourhood(spec: GettF64Spec, arch: str = "") -> Tuple[GettF64Spec, ...]:
    """The spec, the arch's other tuned configurations of the same dtype, then
    one-knob moves the tuner starts from."""
    out = [spec]
    kind = "complex" if spec.problem.complex_ else "real"
    for (a, k, _c), cfg in TUNED_CONFIGS.items():
        if a == arch and k == kind:
            cand = arch_spec(
                spec.problem,
                arch,
                bilinear=spec.bilinear,
                runtime_sizes=spec.runtime_sizes,
                **cfg,
            )
            if cand not in out:
                out.append(cand)
    for kw in (
        {"tile_k": 8},
        {"tile_k": 16},
        {"group_m": 1},
        {"group_m": 16},
        {"lds_swizzle": False},
        {"xcds": 8},
        {"waves_per_eu": None},
        {"lds_stages": 1 if spec.lds_stages == 2 else 2},
        {"prefetch_depth": 2 if spec.prefetch_depth == 1 else 1},
        {"k_block": 0 if spec.k_block else 16},
    ):
        cand = replace(spec, **kw)
        if cand != spec and cand not in out:
            out.append(cand)
    return tuple(out)


def _make_candidate(
    *, name, spec_id, priority, spec_fn, fits, arches=F64_GETT_ARCHS
) -> KernelCandidate:
    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, GettF64Request)
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        try:
            ok, why = fits(req)
            if not ok:
                return False, why
            spec = spec_fn(req)
        except ValueError as e:  # e.g. no CU count for the shape classes
            return False, str(e)
        return _supports_problem(spec, req)

    def select(req: OperatorRequest) -> GettF64Spec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, GettF64Request)
        return spec_fn(req)

    def sweep(req: OperatorRequest):
        if not candidate.admits(req)[0]:
            return ()
        assert isinstance(req, GettF64Request)
        return tuple(
            s
            for s in _neighbourhood(spec_fn(req), req.arch)
            if _supports_problem(s, req)[0]
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm="gett",
        spec_id=spec_id,
        abi_version=GETT_F64_ABI_VERSION,
        priority=priority,
        capability=Capability(arches=tuple(arches), dtypes=("f64", "c128")),
        _supports=support,
        select_spec=select,
        signature=gett_f64_signature,
        grid=lambda spec, req: gett_f64_grid(spec, req.problem),
        block=lambda spec: (int(spec.threads), 1, 1),
        sweep_space=sweep,
        build=lambda spec, arch: build_gett_f64(spec, arch=arch),
    )
    return candidate


def _fits_t128(req: GettF64Request) -> Tuple[bool, str]:
    if req.problem.complex_:
        return True, "ok"  # complex defaults to 64x64 tiles already
    if _tiles(req, 128) < req.cus():
        return (
            False,
            f"{_tiles(req, 128)} 128-tiles leave CUs idle (num_cus={req.cus()})",
        )
    return True, "ok"


def _fits_any(req: GettF64Request) -> Tuple[bool, str]:
    return True, "ok"


GETT_F64_REGISTRY = CandidateRegistry(
    _FAMILY, dim_vocabulary=GETT_F64_DIM_VOCABULARY, require_build=True
)
GETT_F64_REGISTRY.extend(
    (
        _make_candidate(
            name="gett_f64_pipe",
            spec_id="pipe",
            priority=4,
            spec_fn=_spec_pipe,
            fits=_fits_pipe,
            # the targets with pipelined configurations
            arches=tuple(
                a for a in F64_GETT_ARCHS if any(k[0] == a for k in PIPE_CONFIGS)
            ),
        ),
        _make_candidate(
            name="gett_f64_tuned",
            spec_id="tuned",
            priority=5,
            spec_fn=_spec_tuned,
            fits=_fits_any,
        ),
        _make_candidate(
            name="gett_f64_t128",
            spec_id="t128",
            priority=10,
            spec_fn=_spec_t128,
            fits=_fits_t128,
        ),
        _make_candidate(
            name="gett_f64_t64",
            spec_id="t64",
            priority=20,
            spec_fn=_spec_t64,
            fits=_fits_any,
        ),
    )
)


def gett_f64_candidates() -> Tuple[KernelCandidate, ...]:
    return GETT_F64_REGISTRY.candidates()


def _kernel_id(req: GettF64Request, candidate: KernelCandidate, spec) -> KernelId:
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


def gett_f64_sweep_space(req: OperatorRequest) -> Sequence[GettF64Spec]:
    if _request_errors(req):
        return ()
    return GETT_F64_REGISTRY.sweep_space(req)


def dispatch_gett_f64_all(req: GettF64Request) -> Tuple[DispatchResult, ...]:
    if _request_errors(req):
        return ()
    return GETT_F64_REGISTRY.dispatch_all(req, kernel_id=_kernel_id)


def dispatch_gett_f64(
    req: GettF64Request, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select a registered f64 GETT candidate for ``req``."""
    candidate = GETT_F64_REGISTRY.select(req, ranker=ranker)
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


__all__ = [
    "GETT_F64_REGISTRY",
    "GettF64Request",
    "dispatch_gett_f64",
    "dispatch_gett_f64_all",
    "gett_f64_candidates",
    "gett_f64_sweep_space",
]
