# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Arch-neutral vocabulary for the GDN decode dispatcher family.

Holds everything that describes a *problem* rather than a chip: the request
dataclass, the tuned spec dispatch hands out, the family identity, the
dimension vocabulary the registry may gate on, and the shared request/selector
validation. Arch-specific capability and tile selection live in the per-arch
modules (``gfx950.py``), and the knob space describes the gfx950 kernel's
knobs (``axes.py``, ``rules.py``). The tuned spec builds and grids through its
own arch's kernel module (:func:`_decode_kernel_module`), so a new arch adds
one branch there and nothing else in this file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Tuple

from kernels.gfx950.gdn_decode import (
    # Re-exported, never redeclared. The kernel owns what it covers; dispatch's
    # job is to state that coverage, not to restate it -- a copy drifts in the
    # direction that fails silently, rejecting a shape the kernel has since
    # learned to run.
    GDN_DTYPES,
    STATE_DTYPES,
    fuse_conv_heads_error,
)
from rocke.dispatch.core import (
    OperatorRequest,
    normalize_selector,
    selector_matches as _shared_selector_matches,
)
from rocke.dispatch.tuning import Knobs, normalize_knobs

FAMILY = "gdn_decode"

# Bumped when the kernel's argument contract changes, so a cached compile from
# an older layout can never be reused against a newer launcher.
GDN_ABI_VERSION = "rocke-gdn-decode/v1"

# Shared by both GDN families. The f32 spellings exist for decode's
# ``state_dtype``; GDN prefill normalizes its ``dtype`` through the same map
# and still rejects f32 (bf16 only), so for prefill they change only how
# ``normalized()`` spells a dtype that is rejected either way.
_DTYPE_ALIASES = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "f16": "f16",
    "fp16": "f16",
    "float16": "f16",
    "f32": "f32",
    "fp32": "f32",
    "float32": "f32",
}


def normalize_dtype(dtype: str) -> str:
    d = str(dtype).strip().lower()
    return _DTYPE_ALIASES.get(d, d)


@dataclass(frozen=True)
class GdnDecodeRequest(OperatorRequest):
    """One gated-delta-rule single-token decode step.

    ``batch`` is the number of *active* sequences this step. It always
    contributes to the launch grid.

    For ``gate_kind="gdn"``, ``auto`` uses the static ``DEFAULT_TILE`` whenever
    it is legal; otherwise dispatch chooses a validator-admitted fallback.
    Neither ``batch`` nor ``num_v_heads`` chooses GDN's auto tile.

    For ``gate_kind="kda"``, ``auto`` is keyed on ``batch * num_v_heads`` -- the
    "work" -- so tensor parallelism selects the tile for heads local to a rank.

    ``state_dtype`` may be ``"f32"`` as well as a 16-bit dtype, for both gate
    kinds; only the recurrent state widens, so ``dtype`` stays 16-bit.

    ``fuse_conv`` / ``fuse_out_norm`` select the optional fused modes: a
    causal conv1d + SiLU on the packed q/k/v row (in-place conv-state shift)
    and a gated RMSNorm on the output. Fused requests run one workgroup per
    head, so only ``blocks_per_v_dim == 1`` tiles are legal, and ``auto`` uses
    the static ``FUSED_DEFAULT_TILE`` for both gate kinds whenever it is legal.
    ``fuse_conv`` also requires ``num_k_heads == num_v_heads``. Both flags
    change the kernel ABI, so they (and ``use_qk_l2norm``) must be real bools,
    and ``batch`` and the head counts and dims must be real ints: a request is
    rejected, never coerced into a different one.

    ``gate_kind`` selects the forget-gate granularity: ``"gdn"`` (one scalar
    decay per head) or ``"kda"`` (a per-channel decay). It reaches the spec and
    therefore the kernel name, so the two never share a compile-cache entry.

    ``seq_len`` is not a field. This family is decode-only -- one token per
    sequence -- and a request carrying any other sequence length would be a
    different operator, so it is rejected rather than silently accepted.

    Pinning a configuration. Every registered (gate kind, tile) is a tuned
    candidate whose knob space holds the kernel's performance knobs; the
    knobs are never request fields.

    * ``spec_id`` names a registered tile: ``nw{N}_wtk{K}_bpv{B}`` for
      ``gate_kind="gdn"`` and ``kda_nw{N}_wtk{K}_bpv{B}`` for ``"kda"``.
      ``algorithm`` may stay ``"auto"`` (the family has one algorithm,
      ``"warp_tiled"``) or name it.
    * ``tuning_id`` / ``tuning_knobs`` (only with a ``spec_id``) name one
      configuration of that tile: ``tuning_knobs`` is the knob dict a sweep
      recorded next to the id (``{"dpp_reduce": True, ...}``). ``tuning_id``
      ``"auto"`` with no knobs is the tile's base configuration (every knob
      off), which is what ``auto`` runs. A pin that does not resolve raises
      ``PinRefused``; it is never mapped to another configuration.
    """

    batch: int
    arch: str
    num_k_heads: int = 16
    num_v_heads: int = 32
    head_k_dim: int = 128
    head_v_dim: int = 128
    op: str = "gdn_decode"
    dtype: str = "bf16"
    state_dtype: str = "bf16"
    use_qk_l2norm: bool = True
    algorithm: str = "auto"
    gate_kind: str = "gdn"
    spec_id: str = "auto"
    fuse_conv: bool = False
    fuse_out_norm: bool = False
    tuning_id: str = "auto"
    tuning_knobs: Knobs = ()

    def __post_init__(self):
        object.__setattr__(self, "tuning_knobs", normalize_knobs(self.tuning_knobs))

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = normalize_dtype(self.dtype)
        d["state_dtype"] = normalize_dtype(self.state_dtype)
        # Canonicalize the dims for the identity payload: a validated request
        # may carry numpy integers (Integral), which are real ints but do not
        # serialize to JSON. Only an Integral that is not a bool converts, so
        # malformed values (``batch=None``, ``batch=2.5``) survive raw for
        # request_errors to name -- Capability.check() calls this before
        # request_errors, and the identity hash only runs on a request that
        # passed it.
        for name in GDN_DIM_VOCABULARY:
            value = d.get(name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                continue
            d[name] = int(value)
        return d

    def dims(self) -> dict:
        # Raw values: the capability prefilter reads these before
        # request_errors, which rejects a non-int one by name.
        return {name: getattr(self, name) for name in GDN_DIM_VOCABULARY}


def _decode_kernel_module(arch: str):
    """The GDN decode kernel module for ``arch``."""
    if arch == "gfx950":
        from kernels.gfx950 import gdn_decode

        return gdn_decode
    raise ValueError(f"no GDN decode kernel for arch {arch!r}")


@dataclass(frozen=True)
class GdnDecodeTunedSpec:
    """What dispatch hands out: one registered tile plus a canonical knob set.

    ``tuning_id`` / ``config_key`` / ``knobs`` / ``variant_id`` name the
    configuration and are the same on every problem the tile admits
    (``rocke.dispatch.tuning``). ``kernel_spec`` is the concrete kernel spec of
    ``arch``'s kernel module, which the builder, launcher and kernel name read.
    :meth:`identity` is the kernel spec's own field dict, so
    ``KernelId.spec_hash`` (and ``compile_key``) names the compiled kernel
    exactly as it did when dispatch handed out the kernel spec itself.
    """

    arch: str
    candidate_name: str
    tuning_id: str
    variant_id: str
    config_key: str
    knobs: Knobs
    kernel_spec: Any

    def identity(self) -> dict:
        return asdict(self.kernel_spec)

    def kernel_name(self) -> str:
        return self.kernel_spec.kernel_name()

    def build(self, arch: str | None = None):
        return _decode_kernel_module(self.arch).build_gdn_decode(
            self.kernel_spec, arch=arch or self.arch
        )

    def launch_grid(self, batch: int) -> Tuple[int, int, int]:
        # As given: request_errors rejects a non-int batch.
        kernels = _decode_kernel_module(self.arch)
        return kernels.gdn_decode_grid(batch, self.kernel_spec)

    def launch_block(self) -> Tuple[int, int, int]:
        return (int(self.kernel_spec.block_size), 1, 1)


GDN_DIM_VOCABULARY = (
    "batch",
    "num_k_heads",
    "num_v_heads",
    "head_k_dim",
    "head_v_dim",
)


def request_errors(req: OperatorRequest) -> list:
    """Shape-level rejections that are independent of any candidate or arch."""
    if not isinstance(req, GdnDecodeRequest):
        return [f"expected GdnDecodeRequest, got {type(req).__name__}"]
    errors = []
    # Type checks first, and stop on any: every rule below compares these
    # fields, and the spec builder copies them onto the spec as given. Reject,
    # never normalize -- ``batch=2.5`` would launch 2 sequences, and a truthy
    # non-bool (``fuse_conv="false"``) would select a different kernel ABI.
    for field in GDN_DIM_VOCABULARY:
        value = getattr(req, field)
        # Integral, not int: an ``np.int64(4)`` is an integer and the old
        # ``int()`` coercion accepted it. ``bool`` is an Integral subclass but
        # a flag, never a count or a dim.
        if isinstance(value, bool) or not isinstance(value, Integral):
            errors.append(f"{field} must be an int, got {value!r}")
    for field in ("use_qk_l2norm", "fuse_conv", "fuse_out_norm"):
        value = getattr(req, field)
        if type(value) is not bool:
            errors.append(f"{field} must be a bool, got {value!r}")
    if errors:
        return errors
    if req.batch <= 0:
        errors.append(f"batch must be positive, got {req.batch}")
    if req.num_k_heads <= 0 or req.num_v_heads <= 0:
        errors.append("head counts must be positive")
    elif req.num_v_heads % req.num_k_heads:
        errors.append(
            f"num_v_heads {req.num_v_heads} must be a multiple of "
            f"num_k_heads {req.num_k_heads}"
        )
    if req.head_k_dim <= 0 or req.head_v_dim <= 0:
        errors.append("head dims must be positive")
    if normalize_dtype(req.dtype) not in GDN_DTYPES:
        errors.append(f"unsupported dtype {req.dtype!r}")
    if normalize_dtype(req.state_dtype) not in STATE_DTYPES:
        errors.append(f"unsupported state_dtype {req.state_dtype!r}")
    if req.gate_kind not in ("gdn", "kda"):
        errors.append(
            f"unsupported gate_kind {req.gate_kind!r} (expected 'gdn' or 'kda')"
        )
    if req.gate_kind == "kda" and (req.head_k_dim != 128 or req.head_v_dim != 128):
        errors.append(
            "NOT_YET_IMPLEMENTED: KDA decode currently requires "
            "head_k_dim == head_v_dim == 128"
        )
    if req.fuse_conv:
        # The kernel's own rule, shared so the two cannot drift.
        why = fuse_conv_heads_error(req.num_k_heads, req.num_v_heads)
        if why is not None:
            errors.append(why)
    # A tuning pin names a configuration of one tile, so it needs the tile.
    if type(req.tuning_id) is not str:
        errors.append(f"tuning_id must be a str, got {req.tuning_id!r}")
    elif normalize_selector(req.spec_id) == "auto" and (
        (req.tuning_id.strip() or "auto") != "auto" or req.tuning_knobs
    ):
        errors.append(
            "tuning_id / tuning_knobs name a configuration of one tile; pin its "
            "spec_id too"
        )
    return errors


# Shared pin-selector, re-exported under this family's name (see dispatch core).
selector_matches = _shared_selector_matches
