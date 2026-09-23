# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Loader for the MLA benchmark shape files.

The MLA shape files (``mla_prefill_shapes.json`` and ``mla_shapes.json`` under
``benchmarks/<arch>/attention/{prefill,decode}/``) shipped as a *specification*
with no runner: their ``_harness_note`` records that the existing prefill harness
reads ndjson ``UAShape`` records and would parse zero shapes from this nested
JSON, silently. This module is the loader that makes them real input.

Why MLA needs its own shape type rather than ``UAShape``: ``UAShape`` carries a
single ``head_size``, and MLA has no single value for it. The score side is
``d_nope + d_rope = 192`` wide, the output side is ``d_v = 128``, and the cache
stores neither -- it stores a ``kv_lora_rank = 512`` latent that the kernel
expands per head. A single ``head_size`` cannot describe that, which is exactly
why the note says a new shape type is the concrete change.

Placed in ``benchmarks/common/`` rather than beside the prefill driver because
the decode shape file has the same schema and will want the same loader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Sequence

#: Keys in a shape file that are prose, not data. Every one starts with ``_``,
#: so the rule is mechanical rather than a list to keep in sync.
_DOC_KEY_PREFIX = "_"


@dataclass(frozen=True)
class MLAGeometry:
    """The latent geometry shared by every shape in one file.

    Declared once at file scope rather than per shape because it is a property
    of the *model architecture*, not of the benchmark point: every DeepSeek-V3
    and Kimi-K2 shape uses the same widths and differs only in sequence lengths
    and head count.
    """

    qk_nope_dim: int = 128
    qk_rope_dim: int = 64
    v_head_dim: int = 128
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    num_kv_heads: int = 1

    @property
    def head_dim_qk(self) -> int:
        """Score-side width. The value ``UAShape.head_size`` cannot express."""
        return self.qk_nope_dim + self.qk_rope_dim


@dataclass(frozen=True)
class MLAShape:
    """One benchmark point.

    ``regime`` describes the *shape* -- whether ``seqlen_q == seqlen_k`` -- and
    not which kernel runs. The shape files are explicit that the two must not be
    conflated: a ``chunked`` shape at ``seqlen_q=512`` still stays on the in-loop
    path on footprint grounds.
    """

    model: str
    label: str
    batch: int
    seqlen_q: int
    seqlen_k: int
    num_query_heads: int
    num_kv_heads: int
    block_size: int
    dtype: str
    regime: str
    geometry: MLAGeometry

    @property
    def total_q(self) -> int:
        """Packed query-token count: the axis the kernel actually indexes."""
        return self.batch * self.seqlen_q

    @property
    def signature(self) -> str:
        return (
            f"{self.model}/{self.label}/b{self.batch}"
            f"_q{self.seqlen_q}k{self.seqlen_k}"
            f"_h{self.num_query_heads}_{self.dtype}"
        )

    def to_request(self, *, arch: str, causal: bool = True):
        """Build the dispatcher request for this shape.

        Imported lazily so that loading and inspecting shapes does not require
        the dispatch package -- listing a shape file is useful even where the
        kernel cannot be built.
        """
        from dispatch.mla import MLARequest

        return MLARequest(
            num_heads=self.num_query_heads,
            total_q=self.total_q,
            num_seqs=self.batch,
            max_seqlen_k=self.seqlen_k,
            arch=arch,
            d_nope=self.geometry.qk_nope_dim,
            d_rope=self.geometry.qk_rope_dim,
            d_v=self.geometry.v_head_dim,
            kv_lora_rank=self.geometry.kv_lora_rank,
            page_block_size=self.block_size,
            causal=causal,
            dtype=self.dtype,
        )


def _geometry_from(doc: Mapping[str, Any]) -> MLAGeometry:
    raw = doc.get("mla_geometry") or {}
    fields = MLAGeometry.__dataclass_fields__
    unknown = set(raw) - set(fields)
    if unknown:
        # Loud rather than ignored: a misspelled geometry key would otherwise
        # silently fall back to the DeepSeek default and benchmark the wrong
        # model.
        raise ValueError(
            f"unknown mla_geometry keys {sorted(unknown)}; "
            f"expected a subset of {sorted(fields)}"
        )
    return MLAGeometry(**{k: int(v) for k, v in raw.items()})


def load_mla_shapes(path: str | Path) -> List[MLAShape]:
    """Parse one MLA shape file into shapes, model defaults merged in.

    Raises rather than returning an empty list when the file holds no shapes:
    the failure this loader exists to prevent is a harness silently measuring
    nothing, which is what the previous ndjson path did with these files.
    """
    path = Path(path)
    doc = json.loads(path.read_text())
    geometry = _geometry_from(doc)

    models = doc.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"{path}: no 'models' list; got keys {sorted(doc)}")

    shapes: List[MLAShape] = []
    for model in models:
        defaults = {
            k: v
            for k, v in model.items()
            if k not in ("shapes", "model") and not k.startswith(_DOC_KEY_PREFIX)
        }
        name = model.get("model", path.stem)
        for entry in model.get("shapes", []):
            merged = {**defaults, **entry}
            try:
                shapes.append(
                    MLAShape(
                        model=name,
                        label=merged.get("label", ""),
                        batch=int(merged["batch"]),
                        seqlen_q=int(merged["seqlen_q"]),
                        seqlen_k=int(merged["seqlen_k"]),
                        num_query_heads=int(merged["num_query_heads"]),
                        num_kv_heads=int(
                            merged.get("num_kv_heads", geometry.num_kv_heads)
                        ),
                        block_size=int(merged["block_size"]),
                        dtype=str(merged.get("dtype", "bf16")),
                        regime=str(merged.get("regime", "")),
                        geometry=geometry,
                    )
                )
            except KeyError as missing:
                raise ValueError(
                    f"{path}: shape {merged.get('label', entry)!r} is missing "
                    f"required key {missing}"
                ) from None

    if not shapes:
        raise ValueError(f"{path}: parsed zero shapes")
    return shapes


def filter_shapes(
    shapes: Sequence[MLAShape],
    *,
    regime: str | None = None,
    model: str | None = None,
    limit: int | None = None,
) -> List[MLAShape]:
    """Narrow a shape list. Substring match on ``model``, exact on ``regime``."""
    out = list(shapes)
    if regime:
        out = [s for s in out if s.regime == regime]
    if model:
        out = [s for s in out if model.lower() in s.model.lower()]
    if limit is not None:
        out = out[:limit]
    return out


__all__ = ["MLAGeometry", "MLAShape", "filter_shapes", "load_mla_shapes"]
