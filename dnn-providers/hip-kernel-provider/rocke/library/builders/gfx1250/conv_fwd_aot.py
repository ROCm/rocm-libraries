# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Packaging adapter of the gfx1250 forward-convolution AOT engine.

``hipkernel:Gfx1250ConvFwdAot`` ships the instances the gfx1250 forward instance
list registers (``dispatch/grouped_convolution_instances/gfx1250_fwd.json``). An
instance is one shape-generic rocKE ``KernelIdentity``: a compiled configuration
with no problem extents, which serves every problem its fields admit.

:class:`Gfx1250ConvFwdAotSpec` is the instance as the catalog carries it: every
identity field the packaged launch reads, and the family's knobs, each a
descriptor ``int`` or ``string``. :func:`spec_from_identity` builds it,
:meth:`Gfx1250ConvFwdAotSpec.to_identity` restores the identity (its
``stable_hash`` is the spec's ``instance_id``), and
:func:`build_gfx1250_conv_fwd_aot` is the ``(spec, *, arch)`` entry point
``hkp_pack`` builds each kernel through.

Whether an instance serves a problem and how it launches is the dispatcher's
(:func:`dispatch.grouped_convolution._admits`,
:func:`~dispatch.grouped_convolution.launch_plan_for`);
:func:`gfx1250_conv_fwd_aot_plan` answers it for a spec, and the engine's C++
port is tested against it.

Knob encoding: a family knob the identity leaves unset is 0. Every int knob the
list sets is positive, so 0 is free to mean "the family's default". A bool knob
is 1 when set true, 2 when set false (an identity may state ``False``), 0 when
unset, so the restored identity keeps its ``stable_hash``.
"""

from __future__ import annotations

import dataclasses
import functools
import json

from dispatch import grouped_convolution as gc
from kernels.conv.aot_jobs import job_for_identity, kernel_for_job
from kernels.conv.direct_aot_jobs import direct_job_for_identity, direct_kernel_for_job
from kernels.conv.kernel_identity import KernelIdentity

ARCH = "gfx1250"
ENGINE = "hipkernel:Gfx1250ConvFwdAot"
PACKAGED_LLVM_FLAVOR = "llvm24"
DTYPES = ("fp16", "bf16")

IMPLICIT_FAMILY = "implicit_gemm"
#: A bool knob's descriptor values (0 is unset).
BOOL_TRUE, BOOL_FALSE = 1, 2
#: Per family, the knob fields (``KernelIdentity.knobs`` keys) its descriptors carry,
#: and their kind. The direct families are the ones the engine's C++ port reads.
#: This is the descriptor layout, not the builder's contract: a family's rocKE spec
#: may take fewer (gstream takes no ``ck``; smallc no ``ck``, ``stages``,
#: ``tile_w`` or ``prefetch``; smallk no ``tile_w``, ``tiles_k`` or ``waves_k``),
#: and :func:`supports_gfx1250_conv_fwd_aot` refuses an identity whose job rocKE
#: cannot build, so such a knob is refused there rather than in the build.
_INT, _BOOL, _STRING = "int", "bool", "string"
_DIRECT_WMMA_COMMON = {
    "ck": _INT,
    "stages": _INT,
    "tile_w": _INT,
    "tiles_k": _INT,
    "tiles_p": _INT,
    "waves_k": _INT,
    "waves_p": _INT,
    "prefetch": _BOOL,
}
FAMILY_KNOBS = {
    IMPLICIT_FAMILY: {},
    "direct_fwd_wmma": {
        **_DIRECT_WMMA_COMMON,
        "tile_n": _INT,
        "pack_groups": _INT,
        "pp_group": _INT,
        "sched": _STRING,
        "split_barrier": _BOOL,
        "wave_epilogue": _BOOL,
        "phase_split": _BOOL,
        "row_ring": _BOOL,
        "tap_ring": _BOOL,
        "pingpong": _BOOL,
        "tap_reuse": _BOOL,
        "tap_shift": _BOOL,
        "strided_view": _BOOL,
        "fold_n": _BOOL,
        "poly_view": _BOOL,
    },
    "direct_fwd_gstream_wmma": {
        **_DIRECT_WMMA_COMMON,
        "walk": _INT,
        "pf_depth": _INT,
        "sched": _STRING,
        "phase_split": _BOOL,
    },
    "direct_fwd_tapouter_wmma": {**_DIRECT_WMMA_COMMON, "tdm_split": _INT},
    "direct_fwd_smallc_wmma": {
        **_DIRECT_WMMA_COMMON,
        "out_h": _INT,
        "out_w": _INT,
        "tiles_per_block": _INT,
        "tdm_store": _BOOL,
    },
    "direct_fwd_smallk_wmma": {
        **_DIRECT_WMMA_COMMON,
        "out_h": _INT,
        "out_w": _INT,
        "tiles_c": _INT,
        "waves_c": _INT,
        "sched": _STRING,
    },
    "direct_dw_fwd_wmma": {
        "block_h": _INT,
        "block_waves": _INT,
        "prefetch_rows": _INT,
        "w_fold": _INT,
    },
    "direct_depthwise": {"block_w": _INT, "block_waves": _INT, "cpl": _INT},
    "direct_depthwise_tiled": {
        "block_w": _INT,
        "block_h": _INT,
        "block_waves": _INT,
        "cpl": _INT,
        "unroll_rows": _BOOL,
    },
    "direct_depthwise_col": {
        "block_w": _INT,
        "block_h": _INT,
        "block_waves": _INT,
        "max_live_f32": _INT,
    },
    "direct_depthwise_spatial": {"block_waves": _INT},
}
FAMILIES = tuple(FAMILY_KNOBS)

#: The identity fields the catalog carries, after ``dtype``: the family, the
#: implicit-GEMM tile (a direct instance holds the direct sweep's placeholders) and
#: the baked geometry (0 where the kernel reads it at run time).
CARRIED_STRINGS = ("algorithm", "pipeline", "epilogue")
CARRIED_INTS = (
    "wave_size",
    "tile_m",
    "tile_n",
    "tile_k",
    "warp_m",
    "warp_n",
    "warp_tile_m",
    "warp_tile_n",
    "warp_tile_k",
    "vector_size_a",
    "vector_size_b",
    "vector_size_c",
    "group_merge",
    "grouped",
    "filter_h",
    "filter_w",
    "stride_h",
    "stride_w",
    "dilation_h",
    "dilation_w",
    "pad_h",
    "pad_w",
    "cpg",
    "kpg",
)
_BOOL_CARRIED = ("grouped",)
#: Knobs carried in the identity field of the same name. A direct identity leaves
#: that field 0 (``tile_n`` is the implicit GEMM's tile, ``block_h`` unused), so the
#: field holds the knob for a direct instance.
_SHARED_FIELDS = ("tile_n", "block_h")


def _knob_fields() -> dict:
    """Every family knob, in a stable order, with its kind."""
    fields = {}
    for knobs in FAMILY_KNOBS.values():
        for name, kind in knobs.items():
            if name in CARRIED_INTS:
                continue
            if fields.setdefault(name, kind) != kind:
                raise AssertionError(f"knob {name} has two kinds")
    return dict(sorted(fields.items()))


KNOB_FIELDS = _knob_fields()
#: The descriptor fields of a spec, in order: its spec order and KMD.
SPEC_FIELDS = (
    ("dtype", "instance_id") + CARRIED_STRINGS + CARRIED_INTS + tuple(KNOB_FIELDS)
)
STRING_FIELDS = (
    ("dtype", "instance_id")
    + CARRIED_STRINGS
    + tuple(name for name, kind in KNOB_FIELDS.items() if kind == _STRING)
)

#: Identity fields every packaged instance holds at one value: anything else selects
#: emission the packaged launch does not model.
_CONSTANT = {
    "arch": ARCH,
    "llvm_flavor": PACKAGED_LLVM_FLAVOR,
    "async_dma": False,
    "unroll_k": False,
    "chiplet_swizzle": False,
    "lds_layout": "default",
    "lds_k_pad": 0,
    "lds_k_outer": False,
    "waves_per_eu": None,
    "acc_epilogue": "none",
    "split_k": 1,
    "split_k_cluster": 0,
    "two_stage": False,
    "ws_replicas": 0,
    "streamk": "off",
    "streamk_reduction": "",
    "prefetch_addr": False,
    "uniform_k_decode": False,
    "patch": False,
    "tap_outer_k": False,
    "dy_halo": 0,
    "dy_halo_setprio": 0,
    "dy_halo_kouter_pad": 0,
    "num_load_waves": 0,
    "cshuffle_no_alias": False,
    "is_3d": False,
    "is_pointwise": False,
    "filter_d": 0,
    "block_h": 0,
    "max_sub_gemms": 0,
    "halo_max_wo": 0,
    "async_chunk_a": 0,
    "async_chunk_b": 0,
}


def _direction(algorithm: str) -> str:
    return "fwd" if algorithm == IMPLICIT_FAMILY else "direct_fwd"


@dataclasses.dataclass(frozen=True)
class _SpecBase:
    @property
    def is_direct(self) -> bool:
        return self.algorithm != IMPLICIT_FAMILY

    def knobs(self) -> dict:
        """The identity's ``knobs``: the family knobs this spec states."""
        out = {}
        for name, kind in FAMILY_KNOBS[self.algorithm].items():
            value = getattr(self, name)
            if value == (0 if kind != _STRING else ""):
                continue
            if kind == _BOOL:
                out[name] = value == BOOL_TRUE
            else:
                out[name] = value
        return out

    def to_identity(self) -> KernelIdentity:
        """The rocKE identity this spec was built from."""
        fields = {name: getattr(self, name) for name in CARRIED_STRINGS + CARRIED_INTS}
        for name in _BOOL_CARRIED:
            fields[name] = bool(fields[name])
        if self.is_direct:
            for name in _SHARED_FIELDS:
                fields[name] = 0
        knobs = self.knobs()
        return KernelIdentity(
            **{**_CONSTANT, **fields},
            direction=_direction(self.algorithm),
            dtype_a=self.dtype,
            dtype_b=self.dtype,
            dtype_d=self.dtype,
            knobs=json.dumps(knobs, sort_keys=True) if knobs else "",
        )

    def label(self) -> str:
        return self.to_identity().label()


def _spec_field(name: str):
    if name in STRING_FIELDS:
        return (name, str, dataclasses.field(default=""))
    return (name, int, dataclasses.field(default=0))


Gfx1250ConvFwdAotSpec = dataclasses.make_dataclass(
    "Gfx1250ConvFwdAotSpec",
    [_spec_field(name) for name in SPEC_FIELDS],
    bases=(_SpecBase,),
    frozen=True,
)
Gfx1250ConvFwdAotSpec.__doc__ = """One shape-generic gfx1250 forward instance (see the
module docstring). Every field is a descriptor ``int`` or ``string``."""
Gfx1250ConvFwdAotSpec.__module__ = __name__


def _identity_error(identity: KernelIdentity) -> str:
    if identity.algorithm not in FAMILY_KNOBS:
        return f"algorithm {identity.algorithm!r} has no packaged launch"
    if identity.direction != _direction(identity.algorithm):
        return f"{identity.algorithm} is not a {identity.direction} algorithm"
    if not identity.dtype_a == identity.dtype_b == identity.dtype_d:
        return "mixed operand dtypes are not packaged"
    if identity.dtype_a not in DTYPES:
        return f"dtype {identity.dtype_a} is not packaged"
    for name, value in _CONSTANT.items():
        if name in _SHARED_FIELDS and identity.algorithm != IMPLICIT_FAMILY:
            continue
        if getattr(identity, name) != value:
            return (
                f"{name}={getattr(identity, name)!r} is not packaged (only {value!r})"
            )
    knobs = json.loads(identity.knobs) if identity.knobs else {}
    allowed = FAMILY_KNOBS[identity.algorithm]
    for name, value in knobs.items():
        kind = allowed.get(name)
        if kind is None:
            return f"knob {name} is not {identity.algorithm}'s (it takes {sorted(allowed)})"
        if kind == _BOOL and type(value) is not bool:
            return f"knob {name} must be a bool, got {value!r}"
        if kind == _INT and (type(value) is not int or value <= 0):
            return f"knob {name} must be a positive integer, got {value!r}"
        if kind == _STRING and (type(value) is not str or not value):
            return f"knob {name} must be a non-empty string, got {value!r}"
    if identity.is_direct:
        for name in _SHARED_FIELDS:
            if getattr(identity, name) != 0:
                return f"a direct identity's {name} field is {getattr(identity, name)}, not 0"
    return ""


def spec_from_identity(identity: KernelIdentity) -> Gfx1250ConvFwdAotSpec:
    """The catalog spec of ``identity``; ``ValueError`` when it is not packaged."""
    error = _identity_error(identity)
    if error:
        raise ValueError(f"{identity.label()}: {error}")
    fields = {name: getattr(identity, name) for name in CARRIED_STRINGS + CARRIED_INTS}
    for name in _BOOL_CARRIED:
        fields[name] = int(fields[name])
    knobs = json.loads(identity.knobs) if identity.knobs else {}
    for name, value in knobs.items():
        if FAMILY_KNOBS[identity.algorithm][name] == _BOOL:
            value = BOOL_TRUE if value else BOOL_FALSE
        fields[name] = value
    return Gfx1250ConvFwdAotSpec(
        dtype=identity.dtype_a, instance_id=identity.stable_hash(), **fields
    )


def _build_job(identity: KernelIdentity):
    """The job rocKE's sweep builds ``identity`` from (``--compile-all``'s)."""
    if identity.is_direct:
        return direct_job_for_identity(identity)
    return job_for_identity(identity)


@functools.lru_cache(maxsize=1)
def registered_instances() -> dict[str, KernelIdentity]:
    """Label -> identity of every instance the gfx1250 forward list registers."""
    return {
        s.name: s.identity
        for s in gc.registered_conv_grouped_instances(ARCH, direction="fwd")
    }


def supports_gfx1250_conv_fwd_aot(
    spec: Gfx1250ConvFwdAotSpec, arch: str = ARCH
) -> tuple[bool, str]:
    """Whether ``spec`` is a packaged instance for ``arch``, and why not."""
    if arch != ARCH:
        return False, f"gfx1250 instances, not {arch}"
    if spec.dtype not in DTYPES:
        return False, f"dtype {spec.dtype!r} is not packaged"
    if spec.algorithm not in FAMILY_KNOBS:
        return False, f"algorithm {spec.algorithm!r} has no packaged launch"
    try:
        identity = spec.to_identity()
    except (TypeError, ValueError) as exc:
        return False, str(exc)
    error = _identity_error(identity)
    if error:
        return False, error
    if identity.stable_hash() != spec.instance_id:
        return False, (
            f"instance {spec.instance_id} is not the identity its fields describe "
            f"(hash {identity.stable_hash()})"
        )
    try:
        # The family's spec decides which knobs it takes and which values build.
        _build_job(identity)
    except (LookupError, TypeError, ValueError) as exc:
        return False, f"rocKE cannot build it: {exc}"
    return True, "ok"


def build_gfx1250_conv_fwd_aot(spec: Gfx1250ConvFwdAotSpec, *, arch: str = ARCH):
    """The typed ``(spec, *, arch)`` entry point hkp_pack builds an instance by: the
    job rocKE's sweep builds the identity from, as ``--compile-all`` does."""
    ok, reason = supports_gfx1250_conv_fwd_aot(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid packaged gfx1250 fwd instance: {reason}")
    identity = spec.to_identity()
    job = _build_job(identity)
    if identity.is_direct:
        return direct_kernel_for_job(job, arch, spec.dtype)[0]
    return kernel_for_job(job, arch, spec.dtype)[0]


# ---------------------------------------------------------------------------
# Admission and launch plan
# ---------------------------------------------------------------------------

#: The problem fields of :func:`gfx1250_conv_fwd_aot_plan`: extents, groups, then
#: stride, padding and dilation as (h, w).
PROBLEM_FIELDS = (
    "N",
    "C",
    "K",
    "H",
    "W",
    "Y",
    "X",
    "G",
    "sh",
    "sw",
    "ph",
    "pw",
    "dh",
    "dw",
)


def problem_request(problem: dict, dtype: str):
    """The dispatcher's request for a 2-D grouped forward problem (``PROBLEM_FIELDS``)."""
    p = problem
    return gc.ConvGroupedRequest(
        N=p["N"], C=p["C"], K=p["K"], Hi=p["H"], Wi=p["W"], Y=p["Y"], X=p["X"],
        arch=ARCH, G=p["G"], stride_h=p["sh"], stride_w=p["sw"],
        pad_h=p["ph"], pad_w=p["pw"], dilation_h=p["dh"], dilation_w=p["dw"],
        dtype=dtype,
    )  # fmt: skip


def gfx1250_conv_fwd_aot_serves(spec, problem: dict) -> tuple[bool, str]:
    """Whether ``spec`` admits ``problem``, and the dispatcher's reason if not."""
    return gc._admits(spec.to_identity(), problem_request(problem, spec.dtype))


def gfx1250_conv_fwd_aot_plan(spec, problem: dict):
    """The dispatcher's launch plan of ``spec`` on ``problem``, which it must admit."""
    ok, why = gfx1250_conv_fwd_aot_serves(spec, problem)
    if not ok:
        raise ValueError(f"{spec.instance_id} does not serve the problem: {why}")
    instance = gc._INSTANCE_SPECS[spec.to_identity().label()]
    return gc.launch_plan_for(problem_request(problem, spec.dtype), instance)
