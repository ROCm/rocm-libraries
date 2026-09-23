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

The design note at ``builders/mla/design/07-hipdnn-exposure.md`` contradicts
itself on this point -- section 7.3 first says to add an MLA module to the
attention ``__init__`` module tuple, then eight lines later says MLA "wants its
own registry" because ``register`` enforces family match. Only the second is
implementable; this module follows it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Tuple

from rocke.dispatch.core import KernelCandidate, OperatorRequest

FAMILY = "mla"
MLA_ABI_VERSION = "hipkg-mla/v1"

#: Features an MLA candidate may declare. ``causal`` is the bottom-right runtime
#: mask the prefill kernel implements; a request asking for anything outside this
#: set is refused by ``Capability`` rather than silently ignored.
MLA_FEATURES = frozenset({"causal"})

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
    op: str = "mla_prefill_fwd"
    dtype: str = "bf16"
    algorithm: str = "auto"
    spec_id: str = "auto"

    @property
    def hdim_qk(self) -> int:
        """Score-side query width: the nope half plus the RoPE half."""
        return self.d_nope + self.d_rope

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = self.dtype.lower()
        return d

    def dims(self) -> dict[str, int]:
        return {
            "num_heads": int(self.num_heads),
            "d_nope": int(self.d_nope),
            "d_rope": int(self.d_rope),
            "d_v": int(self.d_v),
            "hdim_qk": int(self.hdim_qk),
            "kv_lora_rank": int(self.kv_lora_rank),
            "page_block_size": int(self.page_block_size),
            "total_q": int(self.total_q),
            "max_seqlen_k": int(self.max_seqlen_k),
            "num_seqs": int(self.num_seqs),
        }

    def features(self) -> frozenset[str]:
        return frozenset({"causal"}) if self.causal else frozenset()


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
    for name in (
        "num_heads",
        "total_q",
        "num_seqs",
        "max_seqlen_k",
        "d_nope",
        "d_rope",
        "d_v",
        "kv_lora_rank",
        "page_block_size",
    ):
        value = getattr(req, name)
        if int(value) <= 0:
            errors.append(f"{name} must be positive, got {value}")
    if not str(req.arch).strip():
        errors.append("arch must be a non-empty architecture name")
    return errors


def _selector_matches(req: MLARequest, candidate: KernelCandidate) -> Tuple[bool, str]:
    """Honour an explicit ``algorithm``/``spec_id`` pin on the request.

    ``"auto"`` matches anything; anything else must name this candidate. Copied
    rather than imported from the attention family: importing it would couple
    two registries that are deliberately independent, and the function is five
    lines.
    """
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"


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
