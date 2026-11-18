# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared contracts for the MLA dispatcher family.

MLA is a *separate* family from ``attention_unified``, not another candidate in
it. The reason is mechanical rather than stylistic:

* :meth:`rocke.dispatch.core.CandidateRegistry.register` rejects a candidate
  whose ``family`` differs from the registry's, so an MLA candidate could only
  join the attention registry by declaring ``family="attention_unified"`` --
  i.e. by lying about what it is.
* :class:`dispatch.attention.common.AttentionSpec` composes its kernel name from
  a *single* ``head_size``. MLA's query and value widths differ (192 vs 128), so
  a symmetric spec cannot name an MLA kernel without a collision.
* ``AttentionRequest.normalized()`` is ``asdict``. Adding MLA fields to it would
  change ``request_hash`` for every *existing* attention request and invalidate
  anything keyed on it.

So MLA gets its own registry, its own request type, and its own dim vocabulary.
Nothing in :mod:`rocke.dispatch.core` changes: ``CandidateRegistry`` is already
family-parameterized and ``OperatorRequest``/``Capability`` are family-neutral.

The design note at ``builders/mla/DESIGN.md`` contradicts
itself on this point -- section 7.3 first says to add an MLA module to the
attention ``__init__`` module tuple, then eight lines later says MLA "wants its
own registry" because ``register`` enforces family match. Only the second is
implementable; this module follows it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List

from rocke.dispatch.core import OperatorRequest, selector_matches
from rocke.dispatch.tuning.identity import normalize_knobs

FAMILY = "mla"
MLA_ABI_VERSION = "hipkg-mla/v1"

#: Features an MLA candidate may declare. ``causal`` is the bottom-right runtime
#: mask the prefill kernel implements; ``absorbed_query`` is the unfused-projection
#: ABI (:attr:`MLARequest.absorbed_query`). A request asking for anything outside
#: a candidate's set is refused by ``Capability`` rather than silently ignored.
MLA_FEATURES = frozenset({"causal", "absorbed_query"})

#: Integer request fields: exact ``int`` (not ``bool``, not ``float``/``str``).
_POSITIVE_INT_FIELDS = (
    "num_heads",
    "total_q",
    "num_seqs",
    "max_seqlen_k",
    "d_nope",
    "d_rope",
    "d_v",
    "kv_lora_rank",
    "page_block_size",
)
_NONNEGATIVE_INT_FIELDS = ("num_kv_blocks",)
# Type-checked only: <= 0 means "unknown" (see ``MLARequest.num_cus``).
_ANY_INT_FIELDS = ("num_cus",)
_BOOL_FIELDS = ("causal", "absorbed_query")

#: Every gateable integer an MLA capability may constrain. The registry rejects a
#: ``ShapeRange`` naming a dim outside this set, so the vocabulary is the single
#: place a typo in a capability declaration is caught.
MLA_DIM_VOCABULARY = (
    "num_heads",
    "d_nope",
    "d_rope",
    "d_v",
    "hdim_qk",
    "kv_lora_rank",
    "page_block_size",
    "total_q",
    "max_seqlen_k",
    "num_seqs",
)


@dataclass(frozen=True)
class MLARequest(OperatorRequest):
    """Normalized multi-head latent attention request.

    Deliberately carries the *packed varlen* quantities (``total_q``,
    ``num_seqs``) rather than a rectangular ``(batch, seqlen_q)``: the kernel's
    launch geometry is ``total_q // block_q + num_seqs`` in the AITER block
    numbering its in-kernel binary search inverts, which a rectangular request
    cannot express for ragged inputs.

    ``mla_prefill_fwd`` takes the query already projected by ``W_UQ`` and
    RoPE-rotated (``[total_q, num_heads, d_nope + d_rope]``). Neither step is
    in-tree; the caller applies both, and a pre-kernel for them is a follow-up.
    """

    num_heads: int
    total_q: int
    num_seqs: int
    max_seqlen_k: int
    arch: str
    d_nope: int = 128
    d_rope: int = 64
    d_v: int = 128
    kv_lora_rank: int = 512
    page_block_size: int = 16
    causal: bool = True
    # Device CU count; 0 resolves the live device's when the build box is the
    # requested arch (see ``dispatch.mla.gfx942._resolve_num_cus``). It only
    # picks the XCD-aware workgroup order, so a wrong value costs speed.
    num_cus: int = 0
    # Blocks in the paged KV pool (``c_kv`` / ``k_rope`` first dimension).
    # Required: the candidate refuses 0 rather than skip the 32-bit paged
    # extent admission (``check_mla_prefill_extents``). Callers holding the
    # tensors pass ``c_kv.shape[0]``; callers without tensors pass the pool
    # size they would allocate. Negative is malformed.
    num_kv_blocks: int = 0
    op: str = "mla_prefill_fwd"
    dtype: str = "bf16"
    algorithm: str = "auto"
    spec_id: str = "auto"
    # Same fields as ``AttentionRequest`` so the shared pin/tuning plumbing in
    # :mod:`rocke.dispatch.core` treats every family alike. MLA has no knob
    # axes yet, so ``_request_errors`` refuses anything but the defaults.
    tuning_id: str = "auto"
    tuning_knobs: tuple[tuple[str, Any], ...] = ()
    # Unfused head projections: the kernel takes the absorbed query
    # ``[total_q, H, kv_lora_rank + d_rope]`` and writes the normalized latent
    # output plus the LSE (``MlaPrefillSpec.absorbed_query``). Two-heads
    # layouts only, so an odd head count is refused.
    absorbed_query: bool = False

    def __post_init__(self) -> None:
        # Callers rebuild pins from stored JSON; a bad value fails here, with
        # its name, rather than as an unhashable key inside a cache.
        object.__setattr__(self, "tuning_knobs", normalize_knobs(self.tuning_knobs))

    @property
    def hdim_qk(self) -> int:
        """Score-side query width: the nope half plus the RoPE half."""
        return self.d_nope + self.d_rope

    def normalized(self) -> dict:
        d = asdict(self)
        # ``Capability.check`` normalizes before any predicate runs; a non-str
        # dtype is refused there ("dtype ... not in"), not raised here.
        if isinstance(self.dtype, str):
            d["dtype"] = self.dtype.lower()
        return d

    def _dim(self, name: str) -> int:
        """``name`` if it is an exact ``int``, else a stand-in.

        ``Capability.check`` reads :meth:`dims` before the candidate predicate,
        and every MLA predicate starts with ``_request_errors``, which names a
        mistyped field ("num_heads must be an int, got NoneType None"). So a
        mistyped value is replaced with one that passes the shipped capability
        -- the field's default, or 1 for the required fields -- rather than
        coerced (which raises on ``None`` / ``"abc"`` / ``[1]``) or dropped
        (which would refuse it as a missing dim). The stand-in never reaches a
        spec: the predicate refuses the request first.
        """
        value = getattr(self, name)
        if type(value) is int:
            return value
        default = type(self).__dataclass_fields__[name].default
        return default if type(default) is int else 1

    def dims(self) -> dict[str, int]:
        d_nope, d_rope = self._dim("d_nope"), self._dim("d_rope")
        return {
            "num_heads": self._dim("num_heads"),
            "d_nope": d_nope,
            "d_rope": d_rope,
            "d_v": self._dim("d_v"),
            "hdim_qk": d_nope + d_rope,
            "kv_lora_rank": self._dim("kv_lora_rank"),
            "page_block_size": self._dim("page_block_size"),
            "total_q": self._dim("total_q"),
            "max_seqlen_k": self._dim("max_seqlen_k"),
            "num_seqs": self._dim("num_seqs"),
        }

    def features(self) -> frozenset[str]:
        active = set()
        # ``is not False``: a non-bool ``causal`` (0, None) must pass the
        # capability prefilter so ``_request_errors`` names the type error.
        if self.causal is not False:
            active.add("causal")
        if self.absorbed_query:
            active.add("absorbed_query")
        return frozenset(active)


def _request_errors(req: OperatorRequest) -> List[str]:
    """Family-level validity, independent of any candidate.

    Only things that make a request *malformed* -- a spec that cannot be
    constructed from it, or a launch count that cannot be computed. Whether any
    kernel can serve a well-formed request is the candidate's question, not
    this one's.
    """
    if not isinstance(req, MLARequest):
        return [f"expected an MLARequest, got {type(req).__name__}"]
    errors: List[str] = []
    # Exact types, never coerced: "false" is truthy, True == 1 and 1.5 would
    # truncate, so a loose value would silently select a different kernel.
    for name in _BOOL_FIELDS:
        value = getattr(req, name)
        if type(value) is not bool:
            errors.append(
                f"{name} must be a bool, got {type(value).__name__} {value!r}"
            )
    for name in _POSITIVE_INT_FIELDS + _NONNEGATIVE_INT_FIELDS + _ANY_INT_FIELDS:
        value = getattr(req, name)
        if type(value) is not int:
            errors.append(
                f"{name} must be an int, got {type(value).__name__} {value!r}"
            )
        elif name in _POSITIVE_INT_FIELDS and value <= 0:
            errors.append(f"{name} must be positive, got {value}")
        elif name in _NONNEGATIVE_INT_FIELDS and value < 0:
            errors.append(f"{name} must be >= 0, got {value}")
    if not isinstance(req.arch, str) or not req.arch.strip():
        errors.append("arch must be a non-empty architecture name")
    if not isinstance(req.tuning_id, str):
        errors.append(f"tuning_id must be a str, got {type(req.tuning_id).__name__}")
    elif (req.tuning_id.strip() or "auto") != "auto" or req.tuning_knobs:
        errors.append(
            "MLA prefill has no tuning knobs yet: tuning_id must be 'auto' and "
            f"tuning_knobs empty, got tuning_id={req.tuning_id!r}, "
            f"tuning_knobs={req.tuning_knobs!r}"
        )
    return errors


# Shared pin-selector: identical across families, so it lives in the dispatch
# core and each family re-exports it under its own name.
_selector_matches = selector_matches


def num_q_blocks_for(req: MLARequest, block_q: int) -> int:
    """Packed-varlen q-block count in the AITER numbering.

    Sequence ``i``'s first global block is ``cu_q[i] // block_q + i``, so the
    total is ``total_q // block_q + num_seqs``. This is **not**
    ``sum(ceil(S_q / block_q))``: the two disagree whenever a ``cu_q`` entry is
    an exact multiple of ``block_q``, and the difference is a sequence's tail
    rows never being written. See ``mla_prefill_fwd_grid``.
    """
    if block_q <= 0:
        raise ValueError(f"block_q must be positive, got {block_q}")
    return int(req.total_q) // int(block_q) + int(req.num_seqs)


__all__ = [
    "FAMILY",
    "MLA_ABI_VERSION",
    "MLA_DIM_VOCABULARY",
    "MLA_FEATURES",
    "MLARequest",
    "_request_errors",
    "_selector_matches",
    "num_q_blocks_for",
]
