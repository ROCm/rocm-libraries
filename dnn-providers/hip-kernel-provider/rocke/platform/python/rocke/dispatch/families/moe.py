# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fused MoE dispatcher family (single-launch mega-kernel).

Worked implementation mirroring :mod:`rocke.dispatch.gemm.bf16_rcr`, backed by
:mod:`rocke.instances.common.moe_fused_mega` (f16/bf16) and
:mod:`rocke.instances.common.moe_fused_mega_fp8` (fp8 e4m3 block-scale).

SCOPE -- what this dispatcher decides
-------------------------------------
The fused-MoE mega-kernel has a STATIC tile geometry (locked by BUILD_SPEC:
``tile_m=16, tile_n_inter=256, tile_k_gu=32``); the MoE problem dims
(num_tokens / hidden / intermediate / num_experts / top_k) are RUNTIME kernel
args, not selection knobs. The load-bearing dispatch decision is therefore the
*element path*: the f16/bf16 mega-kernel vs the fp8 block-scale mega-kernel.

The candidate set is two element-path kernels, selected by request dtype:

* ``mega_f16``  : f16/bf16 mega-kernel,
* ``mega_fp8``  : fp8 e4m3 block-scale mega-kernel.

Arch coverage: MoE is CDNA-only (the mega-kernel atoms are MFMA), and each
element path spans gfx950 (CDNA4) and gfx942 (CDNA3) as ONE candidate whose
spec function branches on ``req.arch`` -- the shape
:mod:`rocke.dispatch.gemm.fp16_rcr` already uses for the same split. The two
arches need different MFMA atoms, so the support predicate validates the atom
the selected spec actually names against the per-arch MMA catalog, rather than
asserting a literal shape that only one arch has. See :data:`_GFX942_F16_RETILE`
for what gfx942 changes and why.

The fp8 hero atom is the one case the catalog cannot answer: ``16x16x128`` is a
gfx950-only scaled intrinsic that is not a generic MMA catalog row anywhere, and
the instance builder skips the catalog guard for ``atom.k == 128`` and raises its
own arch error instead. The predicate therefore leaves K=128 to the builder and
checks only the atoms the catalog can express.

DEFERRED -- the MoE component pipeline
--------------------------------------
The non-fused MoE component kernels (``moe_sorting``, ``moe_gemm_fused``,
``moe_smoothquant``) and the multi-launch ``fused_moe`` path are separate
algorithms; only the single-launch mega-kernel is dispatched here. Adding them
is a candidate-registration follow-on (same recipe).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence, Tuple

from ...core.arch import ArchTarget
from ...instances.common.moe_fused_mega import (
    FusedMegaKernelSpec,
    build_moe_fused_mega_gemm,
)
from ...instances.common.moe_fused_mega_fp8 import (
    FusedMegaKernelSpecFp8,
    build_moe_fused_mega_gemm_fp8,
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
)

_FAMILY = "moe_fused_mega"
MOE_ABI_VERSION = "hipkg-moe-fused-mega/v1"

# Explicit gfx targets rather than a "cdna" family label: family does not imply
# wave size (gfx1250 is cdna at wave32) and would admit a wave32 target into
# these wave64 MFMA kernels. An arch absent here is one the mega has never been
# built against.
_SUPPORTED_ARCHES = ("gfx942", "gfx950")


@dataclass(frozen=True)
class MoeRequest(OperatorRequest):
    """Normalized fused mixture-of-experts request."""

    num_tokens: int
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int
    arch: str
    op: str = "moe"
    dtype: str = "fp16"
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = _moe_dtype(self.dtype)
        return d

    def dims(self) -> dict[str, int]:
        return {
            "num_tokens": int(self.num_tokens),
            "hidden": int(self.hidden),
            "intermediate": int(self.intermediate),
            "num_experts": int(self.num_experts),
            "top_k": int(self.top_k),
        }


MOE_DIM_VOCABULARY = (
    "num_tokens",
    "hidden",
    "intermediate",
    "num_experts",
    "top_k",
)


def _moe_dtype(dtype: str) -> str:
    d = dtype.lower()
    if d in ("f16", "half"):
        return "fp16"
    if d in ("fp8", "f8", "fp8e4m3", "e4m3"):
        return "fp8e4m3"
    return d


_F16_DTYPES = ("fp16", "bf16")
_FP8_DTYPES = ("fp8e4m3",)


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, MoeRequest):
        return [f"expected MoeRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "moe":
        errors.append(f"unsupported op {req.op!r}")
    for field in ("num_tokens", "hidden", "intermediate", "num_experts", "top_k"):
        if int(getattr(req, field)) <= 0:
            errors.append(f"{field} must be positive")
    if int(req.top_k) > int(req.num_experts):
        errors.append("top_k must be <= num_experts")
    dt = _moe_dtype(req.dtype)
    if dt not in _F16_DTYPES + _FP8_DTYPES:
        errors.append(f"unsupported dtype {req.dtype!r}; one of fp16/bf16/fp8")
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    return errors


def _selector_matches(req: MoeRequest, candidate: KernelCandidate) -> Tuple[bool, str]:
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"


# gfx942 (CDNA3) departures from the shipped gfx950 geometry. Neither is a
# tuning preference; both are the arch refusing to run the shipped spec:
#
# * ``warp_tile_k=16``: the shipped f16/bf16 atom is ``16x16x32``, a CDNA4
#   catalog row. gfx942's widest f16/bf16 MFMA is ``16x16x16``.
# * ``tile_n_down=128``: the shipped tiling's whole-kernel LDS pool is 74,752 B.
#   That fits gfx950's 163,840 B and not gfx942's 65,536 B, and the mega's LDS
#   guard rejects it. Halving the down output tile halves ``Bd_smem``
#   (32,768 -> 16,384 B) for a 58,368 B pool with 7,168 B of headroom. It is the
#   smallest departure that fits: ``tile_k_down=32`` costs exactly the same
#   bytes, and ``tile_n_inter=128`` saves more but also halves the gate/up N
#   extent, which is the dimension grid.x already splits.
_GFX942_F16_RETILE = {"warp_tile_k": 16, "tile_n_down": 128}

# The fp8 mirror image: the shipped ``16x16x128`` hero atom is a gfx950-only
# scaled intrinsic, so gfx942 takes the ``16x16x32`` fp8 atom both arches carry.
# ``down_k`` is set with it for consistency even though the builder currently
# drives BOTH the gate/up and the down MFMAs off ``gate_up_atom()`` and never
# reads ``down_atom()`` -- ``gate_up_k`` is what actually selects the atom. The
# K=32 path also auto-bypasses direct-to-LDS staging (which needs
# ``atoms_per_group == 1``), so ``use_dtla`` is left at its default.
#
# The distinct ``name`` is load-bearing: ``FusedMegaKernelSpecFp8.kernel_name()``
# encodes only ``tile_m``/``tile_n_inter``/``tile_k_gu``, so without it the K=32
# kernel and the K=128 hero kernel would emit the same entry-point symbol.
_GFX942_FP8_RETILE = {"name": "moe_fp8_k32", "gate_up_k": 32, "down_k": 32}


def _spec_f16(req: MoeRequest) -> FusedMegaKernelSpec:
    dt = _moe_dtype(req.dtype)
    if req.arch == "gfx942":
        return FusedMegaKernelSpec(name=f"moe_{dt}", dtype=dt, **_GFX942_F16_RETILE)
    return FusedMegaKernelSpec(name=f"moe_{dt}", dtype=dt)


def _spec_fp8(req: MoeRequest) -> FusedMegaKernelSpecFp8:
    if req.arch == "gfx942":
        return FusedMegaKernelSpecFp8(**_GFX942_FP8_RETILE)
    return FusedMegaKernelSpecFp8(name="moe_fp8")


def _atom(spec) -> Tuple[str, int] | None:
    """``(dtype, atom_k)`` of the MFMA the builder will drive off ``spec``.

    ``None`` when the atom has no generic MMA catalog row to check it against:
    the fp8 K=128 hero atom lowers to the scaled f8f6f4 instruction, which the
    catalog does not model and the instance builder gates on itself.
    """
    if isinstance(spec, FusedMegaKernelSpecFp8):
        return None if spec.gate_up_k == 128 else ("fp8e4m3", spec.gate_up_k)
    return (spec.dtype, spec.warp_tile_k)


def _tile_covers_shape(req: MoeRequest, spec) -> Tuple[bool, str]:
    """Can ``spec``'s static tile geometry cover this request's H and I?

    The mega-kernel has no N-edge predication on either GEMM. ``grid.x`` is
    ``ceil(I / tile_n_inter)`` but the down k-loop contracts a compile-time
    constant ``tile_n_inter``, and the H_out loop steps ``tile_n_down`` with an
    epilogue whose atomic add is unguarded (the mega runs ``pad_n=False``). A
    ragged ``I`` or ``H`` therefore reads weights past the end of the tensor AND
    atomically adds into ``Y`` past the end of the row, silently and on device.

    Refusing is the honest answer rather than retiling to fit: a tile that
    divides the shape is easy to find (2880 takes 192 on gfx942, 320 on gfx950),
    but picking one here would be an untuned guess at a geometry nothing has
    measured, and the shapes that need it -- the gpt-oss rows at H = I = 2880 --
    also want a clamped SwiGLU this kernel does not implement, so they would
    trade a silent out-of-bounds access for a silently wrong activation. A
    shape-aware selector is a separate, tuned change.
    """
    if int(req.intermediate) % spec.tile_n_inter:
        return False, (
            f"intermediate={req.intermediate} is not a multiple of "
            f"tile_n_inter={spec.tile_n_inter}; the gate/up N extent and the "
            "down contraction are unpredicated"
        )
    if int(req.hidden) % spec.tile_n_down:
        return False, (
            f"hidden={req.hidden} is not a multiple of "
            f"tile_n_down={spec.tile_n_down}; the down output tile and its "
            "atomic epilogue are unpredicated"
        )
    return True, "ok"


def _build(spec, arch: str):
    """Route to the builder that matches the spec ``select_spec`` produced.

    The family carries two spec types, so the fp8/f16 split that ``_struct``
    already makes for identity has to be made here too. Keyed on the spec type
    rather than on the request dtype, so the two can never disagree.
    """
    if isinstance(spec, FusedMegaKernelSpecFp8):
        return build_moe_fused_mega_gemm_fp8(spec, arch)
    return build_moe_fused_mega_gemm(spec, arch)


def _make_candidate(*, name, spec_id, dtypes, spec_fn, priority) -> KernelCandidate:
    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, MoeRequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        # Gate on the spec this candidate would actually return, not on a
        # literal geometry: the spec is arch-dependent, so a hard-coded shape
        # here could disagree with what gets built.
        spec = spec_fn(req)
        atom = _atom(spec)
        if atom is not None:
            dt, atom_k = atom
            target = ArchTarget.from_gfx(req.arch)
            if not target.mma.has_shape(
                family="mma",
                a_dtype=dt,
                b_dtype=dt,
                c_dtype="fp32",
                m=16,
                n=16,
                k=atom_k,
            ):
                return False, f"unsupported {dt} 16x16x{atom_k} MoE atom on {req.arch}"
        return _tile_covers_shape(req, spec)

    def select(req: OperatorRequest):
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, MoeRequest)
        return spec_fn(req)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=spec_id,
        spec_id=spec_id,
        abi_version=MOE_ABI_VERSION,
        priority=priority,
        capability=Capability(arches=_SUPPORTED_ARCHES, dtypes=dtypes),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=lambda spec, req: (0, 0, 0),  # grid is runtime (num_m_blocks, inter)
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=_build,
    )
    return candidate


MOE_REGISTRY = CandidateRegistry(
    _FAMILY, dim_vocabulary=MOE_DIM_VOCABULARY, require_build=True
)
MOE_REGISTRY.extend(
    (
        _make_candidate(
            name="moe_fused_mega_f16",
            spec_id="mega_f16",
            dtypes=_F16_DTYPES,
            spec_fn=_spec_f16,
            priority=10,
        ),
        _make_candidate(
            name="moe_fused_mega_fp8",
            spec_id="mega_fp8",
            dtypes=_FP8_DTYPES,
            spec_fn=_spec_fp8,
            priority=10,
        ),
    )
)


def moe_candidates() -> Tuple[KernelCandidate, ...]:
    return MOE_REGISTRY.candidates()


def _struct(spec) -> dict:
    """The structural identity of a MoE mega spec.

    This is what ``KernelId.spec_hash`` -- and therefore ``compile_key``, the
    key an HSACO cache is documented to use -- is taken over, so it has to
    separate every spec that compiles to a different binary. It carries the
    element dtype and the down tile as well as the gate/up tile because both are
    things this family varies: the f16 path serves fp16 and bf16 through
    different MFMA intrinsics, and gfx942 halves ``tile_n_down``.
    """
    if isinstance(spec, FusedMegaKernelSpecFp8):
        atom_k = spec.gate_up_k
        path = "fp8"
    else:
        atom_k = spec.warp_tile_k
        path = "f16"
    return {
        "path": path,
        "dtype": spec.dtype,
        "tile_m": spec.tile_m,
        "tile_n_inter": spec.tile_n_inter,
        "tile_k_gu": spec.tile_k_gu,
        "tile_n_down": spec.tile_n_down,
        "tile_k_down": spec.tile_k_down,
        "atom_k": atom_k,
        "block_size": int(spec.block_size),
    }


def _kernel_id(req: MoeRequest, candidate: KernelCandidate, spec) -> KernelId:
    request_hash = stable_json_hash(req.normalized(), n=16)
    spec_hash = stable_json_hash(_struct(spec), n=16)
    return KernelId(
        op="moe",
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=request_hash,
        spec_hash=spec_hash,
    )


def moe_sweep_space(req: OperatorRequest) -> Sequence[object]:
    if _request_errors(req):
        return ()
    specs = []
    seen = set()
    for candidate in MOE_REGISTRY.supported(req):
        spec = candidate.select_spec(req)
        h = stable_json_hash(_struct(spec), n=16)
        if h not in seen:
            seen.add(h)
            specs.append(spec)
    return tuple(specs)


def dispatch_moe(req: MoeRequest, *, ranker: Ranker | None = None) -> DispatchResult:
    """Select the fused-MoE mega-kernel element path for ``req``."""
    candidate = MOE_REGISTRY.select(req, ranker=ranker)
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
            f"selected {candidate.name} for {req.dtype} fused MoE on {req.arch}",
            f"algorithm={candidate.algorithm}",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )
