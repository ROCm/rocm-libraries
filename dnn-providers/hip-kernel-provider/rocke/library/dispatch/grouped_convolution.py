# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Grouped convolution dispatcher over a registered list of AOT instances.

WHAT IS REGISTERED
------------------
Exactly the instances in the per-arch, per-direction lists under
``grouped_convolution_instances/`` (``<arch>_<direction>.json``) -- currently
forward and backward-data (dgrad) convolutions on gfx950. Weight-gradient
(wgrad), gfx942 forward and gfx1250 forward have no list registered at all:
this rewrite moved fwd/dgrad onto the instance-list machinery but has not yet
ported those onto it, so those
directions/arches are an explicit, declared gap rather than a silent
regression -- a request for either is rejected with a reason, not routed to a
default. Every
record there spells the :class:`KernelIdentity` of one AOT kernel (format:
:class:`InstanceList`; edit with
``benchmarks/common/grouped_conv/instance_list.py``) and becomes one
:class:`KernelCandidate` of its direction's registry, whose ``name`` and
``spec_id`` are the identity's lossless name (:meth:`KernelIdentity.label`,
format in :mod:`kernels.conv.kernel_label`) and whose ``algorithm`` is
the kernel family (``implicit_gemm``, ``direct_nongrouped``,
``direct_depthwise_dgrad``, ...). So a request can pin one family
(``algorithm=``) or one instance (``spec_id=<name>``).

A dgrad request describes the same convolution as the forward one (its
``Hi``/``Wi`` are dX's extents, ``Ho``/``Wo`` dY's); the instance computes dX
from dY and the filter.

WHAT DISPATCH DECIDES
---------------------
*Whether* an instance can run a request -- and, when several can, which one.

* **Admission** is the AOT binary's own rule,
  :func:`kernels.conv.kernel_identity.identity_supports_problem` (the
  one the AOT cache applies before it offers a binary), plus, for the direct
  families, a re-validation of the instance's tuning knobs against the real
  shape (:func:`kernels.conv.direct_variants.direct_spec_for_problem`).
  The instances are shape-generic binaries built against a probe problem, so
  admission never builds a kernel. The LLVM flavor is part of an instance's
  name, and admission checks it against ``ConvGroupedRequest.llvm_flavor``
  (default :data:`DEFAULT_LLVM_FLAVOR`, the flavor the registered lists are
  built for): a request for another flavor admits none of them. Selection is
  a pure function of the request -- it never probes the process
  (``ROCKE_LLVM_FLAVOR``, COMGR) for a flavor.
* **Selection** among admitted instances is the list order: the first listed
  wins. That is a placeholder policy until measured rankings exist; the
  ``ranker`` hook of :func:`dispatch_conv_grouped` is where one plugs in.

Compiling the instances is not done here: their binaries come from the AOT
build (``benchmarks/common/grouped_conv/benchmark_grouped_conv.py
--compile-all``), keyed by the same identity. Every candidate can still build
its kernel IR (``DispatchResult.build()``): from the job ``--compile-all``
builds its identity from (:func:`kernels.conv.aot_jobs.job_for_identity`,
:func:`kernels.conv.direct_aot_jobs.direct_job_for_identity`), so it is
the IR of the cached binary. What dispatch hands a launcher is
a :class:`ConvLaunchPlan` (:func:`launch_plan_for`): the kernels to launch in
order -- each one's identity, launch geometry, kernarg signature and buffers
-- plus the scratch and host tables they need. Most instances are one kernel;
the MFMA direct dgrad first runs the weight transforms named in its spec's
``helpers``, and implicit-GEMM dgrad reads a tilde sub-GEMM table the caller
uploads.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from rocke.core.arch import ArchTarget
from kernels.conv.aot_jobs import job_for_identity, kernel_for_job
from kernels.conv.direct_aot_jobs import (
    direct_job_for_identity,
    direct_kernel_for_job,
)
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.conv.direct_variants import (
    direct_dgrad_weight_transforms,
    direct_dgrad_workspaces,
    direct_spec_for_problem,
    make_direct_spec,
)
from kernels.common.conv_implicit_gemm import ConvProblem
from kernels.conv.kernel_identity import (
    KernelIdentity,
    dgrad_sub_gemms,
    identity_supports_problem,
)
from kernels.conv.kernel_label import field_baseline
from rocke.core.lower_llvm import LLVM_FLAVORS
from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    make_dispatch_result,
    make_kernel_id,
    normalize_selector,
    selector_matches,
)

__all__ = [
    "CONV_DGRAD_EXECUTION_REGISTRY",
    "CONV_DGRAD_REGISTRY",
    "CONV_DIRECT_ABI_VERSION",
    "CONV_FWD_EXECUTION_REGISTRY",
    "CONV_FWD_REGISTRY",
    "CONV_GROUPED_ABI_VERSION",
    "DEFAULT_LLVM_FLAVOR",
    "ConvGroupedRequest",
    "ConvInstanceSpec",
    "ConvLaunchPlan",
    "ConvLaunchStep",
    "InstanceList",
    "InstanceListError",
    "conv_grouped_candidates",
    "conv_grouped_sweep_space",
    "dispatch_conv_grouped",
    "dispatch_conv_grouped_all",
    "format_instance_list",
    "iter_dispatch_conv_grouped_all",
    "launch_plan_for",
    "load_instance_list",
    "registered_conv_grouped_combos",
    "registered_conv_grouped_instances",
]

# ---------------------------------------------------------------------------
# Family / ABI constants
# ---------------------------------------------------------------------------

# Registry family of each direction that has registered instances.
_FAMILIES = {"fwd": "conv_fwd", "dgrad": "conv_dgrad"}

# Kernarg layouts: implicit GEMM takes the conv_args_signature ABI, every
# direct variant the conv_direct_args_signature one.
CONV_GROUPED_ABI_VERSION = "hipkg-conv-grouped/v1"
CONV_DIRECT_ABI_VERSION = "hipkg-conv-direct/v1"

# The registered instance lists, by (arch, direction). Relative to this file.
_INSTANCE_DIR = Path(__file__).resolve().with_name("grouped_convolution_instances")
_INSTANCE_LISTS: dict[tuple[str, str], str] = {
    ("gfx950", "fwd"): "gfx950_fwd.json",
    ("gfx950", "dgrad"): "gfx950_dgrad.json",
}

# The LLVM flavor a request is admitted against unless it names its own: the
# one every registered list is built for (pinned by
# test_default_llvm_flavor_is_the_registered_one).
DEFAULT_LLVM_FLAVOR = "llvm22"

# The buffers bound to an instance's A, B and D operands, by direction.
_OPERANDS = {"fwd": ("x", "w", "y"), "dgrad": ("dy", "w", "dx")}

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvGroupedRequest(OperatorRequest):
    """Normalized grouped convolution request (NHWC activations, KYXC filter)."""

    N: int
    C: int
    K: int
    Hi: int
    Wi: int
    Y: int
    X: int
    arch: str
    G: int = 1
    stride_h: int = 1
    stride_w: int = 1
    pad_h: int = 0
    pad_w: int = 0
    dilation_h: int = 1
    dilation_w: int = 1
    dtype: str = "fp16"
    layout: str = "NHWC"
    # "fwd" or "dgrad" (the directions with registered instances).
    direction: str = "fwd"
    # 3D depth fields -- all must be set together or all left as None (2D).
    Di: int | None = None
    Z: int | None = None
    stride_d: int | None = None
    pad_d: int | None = None
    dilation_d: int | None = None
    op: str = "conv_grouped"
    algorithm: str = "auto"
    spec_id: str = "auto"
    # LLVM flavor the caller lowers to; instances built for another flavor are
    # not admitted. Part of the request, never read from the process.
    llvm_flavor: str = DEFAULT_LLVM_FLAVOR

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = d["dtype"].lower()
        d["layout"] = d["layout"].upper()
        d["direction"] = d["direction"].lower()
        return d

    def dims(self) -> dict[str, int]:
        d = {
            name: int(getattr(self, name))
            for name in (
                "N",
                "C",
                "K",
                "Hi",
                "Wi",
                "Y",
                "X",
                "G",
                "stride_h",
                "stride_w",
                "pad_h",
                "pad_w",
                "dilation_h",
                "dilation_w",
            )
        }
        if self.Di is not None:
            for name in ("Di", "Z", "stride_d", "pad_d", "dilation_d"):
                v = getattr(self, name)
                if v is not None:
                    d[name] = int(v)
        p = _conv_problem(self)
        if p is not None:
            d.update(Ho=int(p.Ho), Wo=int(p.Wo))
            if p.is_3d:
                d["Do"] = int(p.Do)
        return d


# ---------------------------------------------------------------------------
# Request -> problem
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=256)
def _conv_problem(req: ConvGroupedRequest) -> ConvProblem | None:
    """The implicit-GEMM view of ``req``, or ``None`` when it is not a shape.

    Memoised: every registered candidate asks for it on every request.
    """
    try:
        return ConvProblem(
            N=int(req.N),
            Hi=int(req.Hi),
            Wi=int(req.Wi),
            C=int(req.C),
            K=int(req.K),
            Y=int(req.Y),
            X=int(req.X),
            sH=int(req.stride_h),
            sW=int(req.stride_w),
            pH=int(req.pad_h),
            pW=int(req.pad_w),
            dH=int(req.dilation_h),
            dW=int(req.dilation_w),
            groups=int(req.G),
            Di=int(req.Di) if req.Di is not None else None,
            Z=int(req.Z) if req.Z is not None else None,
            sD=int(req.stride_d) if req.stride_d is not None else None,
            pD=int(req.pad_d) if req.pad_d is not None else None,
            dD=int(req.dilation_d) if req.dilation_d is not None else None,
        )
    except (ValueError, ZeroDivisionError):
        return None


@functools.lru_cache(maxsize=256)
def _direct_problem(req: ConvGroupedRequest) -> tuple[DirectConvProblem | None, str]:
    """The direct-conv view of ``req``: ``(problem, "ok")`` or ``(None, why)``.

    Direct conv describes a 2-D convolution with one stride and one padding
    for both axes, so a request outside that has no direct view at all.
    """
    if req.Di is not None:
        return None, "direct conv is 2-D only"
    if req.stride_h != req.stride_w:
        return None, (
            f"direct conv takes one stride (got {req.stride_h}x{req.stride_w})"
        )
    if req.pad_h != req.pad_w:
        return None, f"direct conv takes one padding (got {req.pad_h}x{req.pad_w})"
    return (
        DirectConvProblem(
            N=int(req.N),
            H=int(req.Hi),
            W=int(req.Wi),
            groups=int(req.G),
            cpg=int(req.C) // int(req.G),
            kpg=int(req.K) // int(req.G),
            KH=int(req.Y),
            KW=int(req.X),
            PAD=int(req.pad_h),
            stride=int(req.stride_h),
            dtype=req.dtype.lower(),
            dil_h=int(req.dilation_h),
            dil_w=int(req.dilation_w),
        ),
        "ok",
    )


@functools.lru_cache(maxsize=256)
def _request_errors(req: OperatorRequest) -> tuple[str, ...]:
    """Why ``req`` is not a well-formed grouped convolution (empty if it is)."""
    if not isinstance(req, ConvGroupedRequest):
        return (f"expected ConvGroupedRequest, got {type(req).__name__}",)
    errors: list[str] = []
    if req.op != "conv_grouped":
        errors.append(f"unsupported op {req.op!r}")
    if req.direction.lower() not in _FAMILIES:
        errors.append(
            f"no {req.direction!r} instances are registered "
            f"(directions: {', '.join(_FAMILIES)})"
        )
    for field_name in ("N", "C", "K", "Hi", "Wi", "Y", "X", "G"):
        if int(getattr(req, field_name)) <= 0:
            errors.append(f"{field_name} must be positive")
    if req.dtype.lower() not in ("fp16", "bf16"):
        errors.append(f"unsupported dtype {req.dtype!r}; fp16 or bf16 only")
    if req.layout.upper() != "NHWC":
        errors.append(f"unsupported layout {req.layout!r}; NHWC only")
    if req.llvm_flavor not in LLVM_FLAVORS:
        errors.append(
            f"unknown llvm_flavor {req.llvm_flavor!r}; one of {', '.join(LLVM_FLAVORS)}"
        )
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    fields_3d = {
        "Di": req.Di,
        "Z": req.Z,
        "stride_d": req.stride_d,
        "pad_d": req.pad_d,
        "dilation_d": req.dilation_d,
    }
    set_3d = {k for k, v in fields_3d.items() if v is not None}
    if set_3d and set_3d != set(fields_3d):
        missing = sorted(set(fields_3d) - set_3d)
        errors.append(f"3D fields must all be set or all be None; missing: {missing}")
    if errors:
        return tuple(errors)
    if req.C % req.G or req.K % req.G:
        return (f"C={req.C} and K={req.K} must both be divisible by G={req.G}",)
    p = _conv_problem(req)
    if p is None:
        return ("not a valid convolution shape",)
    if p.Ho <= 0 or p.Wo <= 0:
        errors.append(
            f"degenerate output spatial dims Ho={p.Ho} Wo={p.Wo} "
            "(filter larger than padded input)"
        )
    if p.is_3d and p.Do <= 0:
        errors.append(
            f"degenerate output depth Do={p.Do} (filter larger than padded input depth)"
        )
    return tuple(errors)


# ---------------------------------------------------------------------------
# Selected spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvInstanceSpec:
    """A registered instance, selected for a request.

    ``name`` is the instance's lossless kernel name and ``identity`` the AOT
    identity it decodes to -- the key its binary is built and cached under.
    ``block`` is the workgroup size the binary is launched with; it is a
    function of the instance's tuning alone, never of the shape. ``helpers``
    are the binaries a run launches before it (the MFMA dgrad's weight
    transforms), built and cached alongside it.
    """

    name: str
    identity: KernelIdentity
    block: tuple[int, int, int]
    helpers: tuple[KernelIdentity, ...] = ()

    @property
    def algorithm(self) -> str:
        return self.identity.algorithm

    @property
    def direction(self) -> str:
        """``fwd`` or ``dgrad``, for implicit GEMM and direct conv alike."""
        return self.identity.direction.removeprefix("direct_")

    @property
    def dtype(self) -> str:
        return self.identity.dtype_a

    @property
    def is_direct(self) -> bool:
        return self.identity.is_direct

    def kernel_name(self) -> str:
        return self.name


def _instance_knobs(identity: KernelIdentity) -> dict:
    return json.loads(identity.knobs) if identity.knobs else {}


@functools.lru_cache(maxsize=4096)
def _direct_launch_spec(
    identity: KernelIdentity, req: ConvGroupedRequest
) -> tuple[object | None, str]:
    """The direct kernel spec of ``identity`` on ``req``'s shape, or ``(None, why)``."""
    problem, why = _direct_problem(req)
    if problem is None:
        return None, why
    try:
        return direct_spec_for_problem(
            identity.algorithm, problem, _instance_knobs(identity), identity.arch
        )
    except (ValueError, ZeroDivisionError) as e:
        # A spec constructor that refuses the shape outright is a refusal too,
        # not a reason to fail the whole dispatch.
        return None, str(e)


def _admits(identity: KernelIdentity, req: ConvGroupedRequest) -> tuple[bool, str]:
    """Can the AOT binary of ``identity`` run ``req``?"""
    if not identity.is_direct:
        return identity_supports_problem(identity, _conv_problem(req), req.llvm_flavor)
    if identity.algorithm == "direct_nongrouped" and req.G != 1:
        # The non-grouped family only exists for one group (the AOT run path,
        # direct_kernel_sweep.direct_plans, applies the same rule).
        return False, "direct_nongrouped serves groups == 1 only"
    problem, why = _direct_problem(req)
    if problem is None:
        return False, why
    ok, why = identity_supports_problem(identity, problem, req.llvm_flavor)
    if not ok:
        return False, why
    spec, why = _direct_launch_spec(identity, req)
    return spec is not None, why


def _instance_block(identity: KernelIdentity) -> tuple[int, int, int]:
    """Workgroup size of an instance, from its tuning alone."""
    if not identity.is_direct:
        return (identity.warp_m * identity.warp_n * identity.wave_size, 1, 1)
    # threads_per_block reads only the knobs, so any shape the spec accepts
    # at construction answers it.
    probe = DirectConvProblem(
        N=1,
        H=16,
        W=16,
        groups=1,
        cpg=max(1, identity.cpg),
        kpg=max(1, identity.kpg),
        KH=identity.filter_h,
        KW=identity.filter_w,
        PAD=identity.pad_h,
        stride=identity.stride_h,
        dtype=identity.dtype_a,
        dil_h=identity.dilation_h,
        dil_w=identity.dilation_w,
    )
    spec = make_direct_spec(identity.algorithm, probe, _instance_knobs(identity))
    return (spec.threads_per_block, 1, 1)


def _instance_helpers(identity: KernelIdentity) -> tuple[KernelIdentity, ...]:
    """The weight-transform binaries an MFMA dgrad instance launches first.

    They read the filter and channel counts alone, so the AOT cache keys them
    with padding and stride zeroed: every instance sharing a filter shares
    them.
    """
    if identity.algorithm != "direct_grouped_dgrad_mfma":
        return ()
    return tuple(
        dataclasses.replace(
            identity,
            algorithm=helper,
            direction="direct_dgrad_helper",
            pad_h=0,
            pad_w=0,
            stride_h=0,
            stride_w=0,
            knobs=json.dumps(helper_knobs, sort_keys=True),
        )
        for helper, helper_knobs, _, _ in direct_dgrad_weight_transforms(
            _instance_knobs(identity)
        )
    )


# ---------------------------------------------------------------------------
# Launch contract
# ---------------------------------------------------------------------------


def _signature(spec: ConvInstanceSpec) -> tuple[dict, ...]:
    from kernels.common.conv_abi import (
        conv_args_signature,
        conv_direct_args_signature,
    )

    if not spec.is_direct:
        return tuple(
            conv_args_signature(
                spec.dtype, direction=spec.direction, is_3d=spec.identity.is_3d
            )
        )
    # The MFMA dgrad's own kernel is a forward pass over the transposed problem.
    direction = "fwd" if spec.helpers else spec.direction
    return tuple(conv_direct_args_signature(spec.dtype, direction=direction))


@functools.lru_cache(maxsize=4096)
def _sub_gemms(identity: KernelIdentity, req: ConvGroupedRequest) -> tuple:
    return tuple(dgrad_sub_gemms(identity, _conv_problem(req)))


def _grid(spec: ConvInstanceSpec, req: OperatorRequest) -> tuple[int, int, int]:
    assert isinstance(req, ConvGroupedRequest)
    if spec.is_direct:
        from kernels.common.conv_direct_grouped import direct_launch_geometry
        from kernels.common.conv_direct_nongrouped import DirectNongroupedConvSpec

        launch_spec, why = _direct_launch_spec(spec.identity, req)
        if launch_spec is None:
            raise ValueError(f"{spec.name} cannot run this request: {why}")
        if isinstance(launch_spec, DirectNongroupedConvSpec):
            return launch_spec.grid()
        return direct_launch_geometry(launch_spec)[0]
    ident = spec.identity
    if spec.direction == "dgrad":
        # One workgroup per tile of every tilde sub-GEMM (the kernel finds its
        # sub-GEMM by binary search over the table); the group rides y.
        tiles = sum(
            -(-sg.gemm_m // ident.tile_m) * -(-sg.gemm_n // ident.tile_n)
            for sg in _sub_gemms(ident, req)
        )
        return (tiles, max(1, req.G), max(1, ident.split_k))
    from kernels.common.conv_args import ConvArgs

    return ConvArgs.from_problem(
        _conv_problem(req), tile_m=ident.tile_m, tile_n=ident.tile_n
    ).grid()


def _block(spec: ConvInstanceSpec) -> tuple[int, int, int]:
    return spec.block


@dataclass(frozen=True)
class ConvLaunchStep:
    """One kernel launch of a :class:`ConvLaunchPlan`.

    ``identity`` is the binary to launch (its AOT cache key). ``buffers``
    names the buffers bound to its A, B and D kernarg slots; the weight
    transforms take no B and no extents (``args`` is ``None``), a convolution
    kernel takes its extents from ``args``.
    """

    identity: KernelIdentity
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    signature: tuple[dict, ...]
    buffers: tuple[str, str | None, str]
    args: object = None


@dataclass(frozen=True)
class ConvLaunchPlan:
    """Everything a launcher needs to run one instance on one request.

    The steps run in order on one stream; the last is the instance's own
    kernel. Buffers are named: the operands ``x``/``w``/``y`` (fwd) or
    ``dy``/``w``/``dx`` (dgrad), the caller-allocated scratch in
    ``workspaces`` (name -> bytes, no initial contents), and for implicit-GEMM
    dgrad ``sub_gemms``, a device copy of the i32 table ``sub_gemm_table``
    (``num_sub_gemms`` records).
    """

    steps: tuple[ConvLaunchStep, ...]
    workspaces: Mapping[str, int] = field(default_factory=dict)
    sub_gemm_table: tuple[int, ...] = ()
    num_sub_gemms: int = 0

    def values(
        self, step: ConvLaunchStep, ptrs: Mapping[str, int], sizes: Mapping[str, int]
    ) -> dict:
        """The full kernarg ``values`` of ``step``, its buffers at ``ptrs``."""
        a, b, d = step.buffers
        if step.args is None:
            return {
                "A": ptrs[a],
                "D": ptrs[d],
                "A_bytes": sizes[a],
                "D_bytes": sizes[d],
            }
        extras = {}
        if self.num_sub_gemms:
            extras = {
                "sub_gemm_buf": ptrs["sub_gemms"],
                "num_sub_gemms": self.num_sub_gemms,
            }
        return step.args.to_launch_values(
            ptrs[a], ptrs[b], ptrs[d], sizes[a], sizes[b], sizes[d], **extras
        )


def launch_plan_for(req: ConvGroupedRequest, spec: ConvInstanceSpec) -> ConvLaunchPlan:
    """How to run the admitted instance ``spec`` on ``req``.

    A = dY / activation (NHWK / NHWC), B = filter (KYXC, per-group channels),
    D = dX / output. Implicit GEMM takes its tile counts as kernargs (the
    chiplet-swizzle path decodes the workgroup id against them), so the
    instance's tile is part of the extents.
    """
    from kernels.common.conv_args import ConvArgs

    ident = spec.identity
    step = functools.partial(
        ConvLaunchStep,
        ident,
        _grid(spec, req),
        spec.block,
        _signature(spec),
        _OPERANDS[spec.direction],
    )
    if not ident.is_direct:
        args = ConvArgs.from_problem(
            _conv_problem(req),
            direction=spec.direction,
            tile_m=ident.tile_m,
            tile_n=ident.tile_n,
        )
        if spec.direction != "dgrad":
            return ConvLaunchPlan((step(args),))
        from kernels.common.conv_implicit_gemm_dgrad import pack_sub_gemm_buffer

        sub_gemms = list(_sub_gemms(ident, req))
        return ConvLaunchPlan(
            (step(args),),
            sub_gemm_table=tuple(
                pack_sub_gemm_buffer(sub_gemms, ident.tile_m, ident.tile_n)
            ),
            num_sub_gemms=len(sub_gemms),
        )
    problem, _ = _direct_problem(req)
    if not spec.helpers:
        args = ConvArgs.from_problem(problem, direction=spec.direction)
        return ConvLaunchPlan((step(args),))

    # MFMA dgrad: the weight transforms, then a forward pass over the
    # transposed problem reading the transformed weights.
    from kernels.common.conv_abi import conv_direct_transform_signature
    from kernels.common.conv_direct_grouped import direct_launch_geometry

    knobs = _instance_knobs(ident)
    transforms = direct_dgrad_weight_transforms(knobs)
    steps = []
    for helper, (variant, helper_knobs, src, dst) in zip(spec.helpers, transforms):
        grid, block = direct_launch_geometry(
            make_direct_spec(variant, problem, helper_knobs)
        )
        steps.append(
            ConvLaunchStep(
                helper,
                grid,
                block,
                tuple(conv_direct_transform_signature(spec.dtype)),
                (src, None, dst),
            )
        )
    launch_spec, _ = _direct_launch_spec(ident, req)
    steps.append(
        dataclasses.replace(
            step(ConvArgs.from_problem(launch_spec.problem, direction="fwd")),
            buffers=("dy", transforms[-1][3], "dx"),
        )
    )
    return ConvLaunchPlan(tuple(steps), direct_dgrad_workspaces(problem, knobs))


# ---------------------------------------------------------------------------
# Instance candidates
# ---------------------------------------------------------------------------


def _build_instance(spec: ConvInstanceSpec, arch: str):
    """``build(spec, arch)`` of every instance candidate: its kernel IR.

    Built from the job ``--compile-all`` builds the identity from, so this is
    the IR of the cached binary. Only the instance's own kernel: the helpers a
    run launches before it (``spec.helpers``) are binaries of their own.
    """
    identity = spec.identity
    if arch != identity.arch:
        raise ValueError(f"{spec.name} is a {identity.arch} instance, not {arch}")
    if identity.is_direct:
        job = direct_job_for_identity(identity)
        return direct_kernel_for_job(job, arch, identity.dtype_a)[0]
    return kernel_for_job(job_for_identity(identity), arch, identity.dtype_a)[0]


def _make_instance_candidate(spec: ConvInstanceSpec, priority: int) -> KernelCandidate:
    name, identity = spec.name, spec.identity

    def support(req: OperatorRequest) -> tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        return _admits(identity, req)

    def select(req: OperatorRequest) -> ConvInstanceSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        return spec

    candidate = KernelCandidate(
        name=name,
        family=_FAMILIES[spec.direction],
        algorithm=identity.algorithm,
        spec_id=name,
        abi_version=(
            CONV_DIRECT_ABI_VERSION if identity.is_direct else CONV_GROUPED_ABI_VERSION
        ),
        priority=priority,
        capability=Capability(
            arches=(identity.arch,),
            dtypes=(identity.dtype_a,),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=_signature,
        grid=_grid,
        block=_block,
        sweep_space=lambda req: (spec,) if candidate.admits(req)[0] else (),
        build=_build_instance,
    )
    return candidate


# ---------------------------------------------------------------------------
# Instance list files
# ---------------------------------------------------------------------------

# Record keys that spell several identity fields as one JSON array, in the
# order the fields are listed. ``dtype`` is also a plain string when all
# three operands share it.
_RECORD_GROUPS: dict[str, tuple[str, ...]] = {
    "dtype": ("dtype_a", "dtype_b", "dtype_d"),
    "tile": ("tile_m", "tile_n", "tile_k"),
    "warps": ("warp_m", "warp_n"),
    "warp_tile": ("warp_tile_m", "warp_tile_n", "warp_tile_k"),
    "vector": ("vector_size_a", "vector_size_b", "vector_size_c"),
    "filter": ("filter_h", "filter_w"),
    "pad": ("pad_h", "pad_w"),
    "stride": ("stride_h", "stride_w"),
    "dilation": ("dilation_h", "dilation_w"),
    "async_chunk": ("async_chunk_a", "async_chunk_b"),
}
_GROUP_OF = {f: key for key, fields in _RECORD_GROUPS.items() for f in fields}

# Spelled in every record of the family, even at the baseline -- the head of
# the kernel's name (kernels.conv.kernel_label), and for direct conv
# every one of them is baked into the binary.
_RECORD_HEAD = {
    False: ("tile", "warps", "warp_tile", "vector", "pipeline", "epilogue"),
    True: ("filter", "pad", "stride", "dilation", "cpg", "kpg"),
}

_LIST_KEYS = ("comment", "arch", "direction", "defaults", "instances")

# Python type of every identity field's JSON value ("int | None" admits null).
_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    f.name: {
        "int": (int,),
        "bool": (bool,),
        "str": (str,),
        "int | None": (int, type(None)),
    }[str(f.type)]
    for f in fields(KernelIdentity)
}


class InstanceListError(ValueError):
    """An instance list file that does not hold a valid list."""


@dataclass(frozen=True)
class InstanceList:
    """One instance list file: ``grouped_convolution_instances/<arch>_<dir>.json``.

    ``identities`` are in priority order. ``defaults`` are field values every
    record shares unless it spells its own (``llvm_flavor``, ``wave_size``);
    a record spells only the fields that leave the family's baseline
    (:func:`kernels.conv.kernel_label.field_baseline`) or ``defaults``,
    plus its head. ``direction`` is the list's; a direct-conv record's identity
    carries ``direct_<direction>``.
    """

    arch: str
    direction: str
    identities: tuple[KernelIdentity, ...]
    defaults: dict = field(default_factory=dict)
    comment: tuple[str, ...] = ()

    def identity_direction(self, is_direct: bool) -> str:
        return f"direct_{self.direction}" if is_direct else self.direction

    def record(self, identity: KernelIdentity) -> dict:
        """The sparse JSON record of ``identity`` in this list."""
        direction = self.identity_direction(identity.is_direct)
        if (identity.arch, identity.direction) != (self.arch, direction):
            raise InstanceListError(
                f"{identity.label()} is not a {self.arch} {direction} instance"
            )
        values = asdict(identity)
        baseline = field_baseline(identity.algorithm)
        baseline.update(self.defaults)
        baseline["arch"] = self.arch
        baseline["direction"] = direction
        head = _RECORD_HEAD[identity.is_direct]

        def at_baseline(f: str) -> bool:
            return (
                f in baseline
                and values[f] == baseline[f]
                and type(values[f]) is type(baseline[f])
            )

        record: dict = {"algorithm": identity.algorithm}
        dtypes = [values[f] for f in _RECORD_GROUPS["dtype"]]
        record["dtype"] = dtypes[0] if len(set(dtypes)) == 1 else dtypes
        # The head first, in name order; then the rest in field order.
        for key in (*head, *(_GROUP_OF.get(f, f) for f in values)):
            if key in record:
                continue
            if key in _RECORD_GROUPS:
                group = _RECORD_GROUPS[key]
                if key in head or not all(at_baseline(g) for g in group):
                    record[key] = [values[g] for g in group]
            elif key == "knobs":
                if not at_baseline(key):
                    record[key] = json.loads(values[key])
            elif key in head or not at_baseline(key):
                record[key] = values[key]
        if self.identity(record, "record") != identity:
            raise InstanceListError(f"{identity.label()} has no lossless record")
        return record

    def identity(self, record: object, where: str) -> KernelIdentity:
        """The identity ``record`` spells; ``where`` prefixes every error."""
        if not isinstance(record, dict):
            raise InstanceListError(
                f"{where}: a record is a JSON object, got {record!r}"
            )
        algorithm = record.get("algorithm")
        if not isinstance(algorithm, str):
            raise InstanceListError(f"{where}: record has no algorithm")
        values = field_baseline(algorithm)
        values.update(self.defaults)
        values["arch"] = self.arch
        values["direction"] = self.identity_direction(algorithm.startswith("direct"))
        for key, value in record.items():
            if key == "dtype" and isinstance(value, str):
                value = [value] * 3
            if key in _RECORD_GROUPS:
                group = _RECORD_GROUPS[key]
                if not isinstance(value, list) or len(value) != len(group):
                    raise InstanceListError(
                        f"{where}: {key} is a list of {len(group)} "
                        f"({', '.join(group)}), got {value!r}"
                    )
                values.update(zip(group, value))
            elif key == "knobs":
                if not isinstance(value, dict):
                    raise InstanceListError(
                        f"{where}: knobs is an object, got {value!r}"
                    )
                values[key] = json.dumps(value, sort_keys=True)
            elif key in ("arch", "direction"):
                raise InstanceListError(f"{where}: {key} is the list's, not a record's")
            elif key in _FIELD_TYPES:
                values[key] = value
            else:
                raise InstanceListError(f"{where}: unknown key {key!r}")
        for f, value in values.items():
            if f not in _FIELD_TYPES:
                raise InstanceListError(f"{where}: unknown field {f!r}")
            types = _FIELD_TYPES[f]
            # bool is an int to isinstance; a count spelled true is a typo.
            if type(value) not in types:
                names = " or ".join(
                    "null" if t is type(None) else t.__name__ for t in types
                )
                raise InstanceListError(f"{where}: {f} must be {names}, got {value!r}")
        missing = sorted(set(_FIELD_TYPES) - set(values))
        if missing:
            raise InstanceListError(
                f"{where}: {algorithm} record lacks {', '.join(missing)}"
            )
        return KernelIdentity(**values)


def load_instance_list(path: Path) -> InstanceList:
    """The instance list in ``path``, its identities in priority order.

    Fails on a broken list rather than skipping an entry: a record that does
    not spell an identity of the list's arch and direction, or one that repeats an
    earlier record's kernel, is an error naming its position.
    """
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise InstanceListError(f"{path.name}: not JSON: {e}") from None
    if not isinstance(doc, dict):
        raise InstanceListError(f"{path.name}: expected a JSON object")
    unknown = sorted(set(doc) - set(_LIST_KEYS))
    if unknown:
        raise InstanceListError(f"{path.name}: unknown keys {unknown}")
    for key in ("arch", "direction"):
        if not isinstance(doc.get(key), str):
            raise InstanceListError(f"{path.name}: {key} must be a string")
    records = doc.get("instances")
    if not isinstance(records, list):
        raise InstanceListError(f"{path.name}: instances must be a list")
    defaults = doc.get("defaults", {})
    if not isinstance(defaults, dict) or not set(defaults) <= set(_FIELD_TYPES):
        raise InstanceListError(
            f"{path.name}: defaults must map identity fields to values"
        )
    for key in ("arch", "direction"):
        if key in defaults:
            raise InstanceListError(f"{path.name}: {key} is the list's, not a default")
    comment = doc.get("comment", [])
    if not isinstance(comment, list) or not all(isinstance(c, str) for c in comment):
        raise InstanceListError(f"{path.name}: comment must be a list of strings")
    lst = InstanceList(
        arch=doc["arch"],
        direction=doc["direction"],
        identities=(),
        defaults=dict(defaults),
        comment=tuple(comment),
    )
    identities: list[KernelIdentity] = []
    seen: dict[KernelIdentity, int] = {}
    for i, record in enumerate(records):
        where = f"{path.name}: instances[{i}]"
        identity = lst.identity(record, where)
        if identity.pipeline == "wavelet":
            # Wavelet appends load waves to the workgroup, which _instance_block
            # does not model; no registered arch builds it.
            raise InstanceListError(f"{where}: wavelet instances are not supported")
        if identity in seen:
            raise InstanceListError(
                f"{where}: {identity.label()} is already instances[{seen[identity]}]"
            )
        seen[identity] = i
        identities.append(identity)
    return dataclasses.replace(lst, identities=tuple(identities))


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def format_instance_list(lst: InstanceList) -> str:
    """``lst`` as its file's text: canonical records, one per line."""
    lines = ["{"]
    if lst.comment:
        lines.append('  "comment": [')
        lines += [f"    {_json(c)}," for c in lst.comment]
        lines[-1] = lines[-1].rstrip(",")
        lines.append("  ],")
    lines.append(f'  "arch": {_json(lst.arch)},')
    lines.append(f'  "direction": {_json(lst.direction)},')
    if lst.defaults:
        lines.append(f'  "defaults": {_json(lst.defaults)},')
    if lst.identities:
        lines.append('  "instances": [')
        lines += [f"    {_json(lst.record(i))}," for i in lst.identities]
        lines[-1] = lines[-1].rstrip(",")
        lines.append("  ]")
    else:
        lines.append('  "instances": []')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _read_instance_list(arch: str, direction: str) -> tuple[KernelIdentity, ...]:
    """The registered instances of ``(arch, direction)``, in priority order.

    Fails at import rather than at a first request: a list that does not
    load, or that declares another arch or direction than it is registered
    under, is a broken list, not a missing kernel.
    """
    path = _INSTANCE_DIR / _INSTANCE_LISTS[(arch, direction)]
    lst = load_instance_list(path)
    if (lst.arch, lst.direction) != (arch, direction):
        raise InstanceListError(
            f"{path.name}: declares {lst.arch} {lst.direction}, "
            f"registered as {arch} {direction}"
        )
    return lst.identities


_CONV_DIM_VOCABULARY = (
    "N",
    "C",
    "K",
    "Hi",
    "Wi",
    "Y",
    "X",
    "G",
    "stride_h",
    "stride_w",
    "pad_h",
    "pad_w",
    "dilation_h",
    "dilation_w",
    "Ho",
    "Wo",
    # 3D depth dims (absent when 2D)
    "Di",
    "Z",
    "stride_d",
    "pad_d",
    "dilation_d",
    "Do",
)

_REGISTRIES = {
    direction: CandidateRegistry(family, dim_vocabulary=_CONV_DIM_VOCABULARY)
    for direction, family in _FAMILIES.items()
}
CONV_FWD_REGISTRY = _REGISTRIES["fwd"]
CONV_DGRAD_REGISTRY = _REGISTRIES["dgrad"]

# A second registry per direction, holding the same candidates as the route
# registry above but meant for sweeping/benchmarking rather than production
# routing (see :func:`iter_dispatch_conv_grouped_all`). Kept distinct from
# ``_REGISTRIES`` so routing policy (e.g. a future ranker) can diverge from
# what a sweep walks, mirroring ``ATTENTION_ROUTE_REGISTRY`` /
# ``ATTENTION_EXECUTION_REGISTRY`` (:mod:`dispatch.attention`). Like
# attention's, it refuses a candidate without a ``build``, so everything a
# sweep walks can be built (:func:`_build_instance`).
_EXECUTION_REGISTRIES = {
    direction: CandidateRegistry(
        family, dim_vocabulary=_CONV_DIM_VOCABULARY, require_build=True
    )
    for direction, family in _FAMILIES.items()
}
CONV_FWD_EXECUTION_REGISTRY = _EXECUTION_REGISTRIES["fwd"]
CONV_DGRAD_EXECUTION_REGISTRY = _EXECUTION_REGISTRIES["dgrad"]

# Registered instances by name; a candidate's spec does not depend on the request.
_INSTANCE_SPECS: dict[str, ConvInstanceSpec] = {}
for (_arch, _direction), _ in _INSTANCE_LISTS.items():
    for _priority, _identity in enumerate(_read_instance_list(_arch, _direction)):
        _spec = ConvInstanceSpec(
            name=_identity.label(),
            identity=_identity,
            block=_instance_block(_identity),
            helpers=_instance_helpers(_identity),
        )
        _INSTANCE_SPECS[_spec.name] = _spec
        _candidate = _make_instance_candidate(_spec, _priority)
        _REGISTRIES[_direction].register(_candidate)
        _EXECUTION_REGISTRIES[_direction].register(_candidate)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _kernel_id(
    req: OperatorRequest, candidate: KernelCandidate, spec: ConvInstanceSpec
) -> KernelId:
    return make_kernel_id(req, candidate, spec, op=_FAMILIES[spec.direction])


def _registry(req: ConvGroupedRequest) -> CandidateRegistry:
    """The registry of a request :func:`_request_errors` accepted."""
    return _REGISTRIES[req.direction.lower()]


def conv_grouped_candidates(direction: str = "fwd") -> tuple[KernelCandidate, ...]:
    """Every registered candidate of ``direction``, in priority order."""
    registry = _REGISTRIES.get(direction)
    return registry.candidates() if registry is not None else ()


def registered_conv_grouped_instances(
    arch: str | None = None, direction: str = "fwd"
) -> tuple[ConvInstanceSpec, ...]:
    """Every registered instance (optionally of one ``arch``), in priority order.

    What an AOT build compiles: request-independent, no GPU needed.
    """
    return tuple(
        _INSTANCE_SPECS[c.name]
        for c in conv_grouped_candidates(direction)
        if arch is None or arch in c.capability.arches
    )


def registered_conv_grouped_combos(
    req: OperatorRequest,
) -> tuple[tuple[KernelCandidate, ConvInstanceSpec], ...]:
    """Every registered ``(candidate, spec)`` that can run ``req``, in priority order."""
    if _request_errors(req):
        return ()
    return _registry(req).combos(req)


def conv_grouped_sweep_space(req: OperatorRequest) -> Sequence[ConvInstanceSpec]:
    """Every registered instance that can run ``req``, in priority order."""
    if _request_errors(req):
        return ()
    return _registry(req).sweep_space(req, spec_key=lambda spec: spec.name)


def dispatch_conv_grouped_all(req: ConvGroupedRequest) -> tuple[DispatchResult, ...]:
    """A :class:`DispatchResult` for every instance that can run ``req``.

    In priority order, each with its own launch geometry and signature -- the
    benchmark primitive.
    """
    if _request_errors(req):
        return ()
    return _registry(req).dispatch_all(req, kernel_id=_kernel_id)


def iter_dispatch_conv_grouped_all(req: ConvGroupedRequest) -> Iterable[DispatchResult]:
    """Stream a :class:`DispatchResult` for every instance that can run ``req``.

    Reads :data:`CONV_FWD_EXECUTION_REGISTRY` / :data:`CONV_DGRAD_EXECUTION_REGISTRY`
    rather than the production route registries -- the home for sweep/tuning
    consumers, mirroring :func:`dispatch.attention.iter_dispatch_attention_all`.
    Every result builds (``result.build()`` is the instance's kernel IR, see
    :func:`_build_instance`); today the candidates are the same as
    :func:`dispatch_conv_grouped_all`'s.
    """
    if _request_errors(req):
        return
    yield from _EXECUTION_REGISTRIES[req.direction.lower()].iter_dispatch_all(
        req, kernel_id=_kernel_id
    )


def _no_instance_reason(req: ConvGroupedRequest) -> str:
    """Why no registered instance admits ``req``, summarised.

    The registry's own message lists every candidate's refusal, which for a
    list of a thousand instances is unreadable; the distinct reasons with
    their counts say the same thing.
    """
    errors = _request_errors(req)
    if errors:
        return "; ".join(errors)
    reasons: dict[str, int] = {}
    for candidate in _registry(req).candidates():
        ok, why = candidate.admits(req)
        if not ok:
            reasons[why] = reasons.get(why, 0) + 1
    if not reasons:
        return "no instances registered"
    top = sorted(reasons.items(), key=lambda kv: -kv[1])[:5]
    return "; ".join(f"{n}x {why}" for why, n in top) + (
        f" (+{len(reasons) - len(top)} more reasons)" if len(reasons) > len(top) else ""
    )


def dispatch_conv_grouped(
    req: ConvGroupedRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """The instance that runs ``req``: the first admitted one, unless ``ranker``
    reorders them.

    Raises ``ValueError`` when no registered instance can run ``req`` (a
    ``PinRefused`` when ``spec_id`` pins one that cannot).
    """
    if _request_errors(req):
        raise ValueError(
            f"no registered grouped-convolution instance can run the request: "
            f"{'; '.join(_request_errors(req))}"
        )
    registry = _registry(req)
    if not registry.supported(req):
        if normalize_selector(getattr(req, "spec_id", "auto")) != "auto":
            # The registry's pin refusal names the pinned instance's reason.
            registry.select(req, ranker=ranker)
        raise ValueError(
            f"no registered grouped-convolution instance can run the request: "
            f"{_no_instance_reason(req)}"
        )
    candidate = registry.select(req, ranker=ranker)
    spec = candidate.select_spec(req)
    return make_dispatch_result(
        req,
        candidate,
        spec,
        kernel_id=_kernel_id(req, candidate, spec),
        headline=f"selected {candidate.name} on {req.arch}",
    )
