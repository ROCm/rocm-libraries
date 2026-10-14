# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 (CDNA4) MLA decode: the shared decode kernel with gfx950's operands.

:func:`kernels.mla.mla_decode.build_mla_decode` emits the kernel for both
arches from one body; this subclass of the shared spec is what makes it emit
the gfx950 form. Two choices differ from gfx942, both fixed by the class:

* **Score atom** ``mfma_f32_16x16x32_bf16``. The score is a K = 576
  reduction, so the K-packed atom halves its MFMA count (18 k-steps, not 36)
  and reads each operand fragment as one 16 B load: the absorbed query from
  global memory into registers, the key tile and the parked ``Q_rope`` from
  LDS. 16 B LDS reads need 16 B-aligned rows, so both buffers pad by 8 bf16.
* **PV latent operand** via ``ds_read_tr16_b64``. The PV contracts over keys
  while ``kv_lds`` stores keys as rows; the transpose read hands each lane its
  column fragment from that natural tile (``builders/mla/DESIGN.md`` section
  5.2), where gfx942 gathers it with a row permutation the epilogue has to
  undo.

The PV keeps the ``16x16x16`` atom: a key tile is one 16-row page, so a
32-deep PV would need two pages per tile. Everything else -- heads on the
MFMA n axis, 16 heads per wave, the lazy-rescale online softmax, split-KV and
the arch-neutral combine -- is the gfx942 kernel's. A bring-up configuration:
the 32-head ``32x32x16`` layout, multi-page tiles and LDS DMA are tuning work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from rocke.core.ir import KernelDef

from .mla_decode import MlaDecodeSpec, build_mla_decode

__all__ = ["MlaDecodeGfx950Spec", "build_mla_decode_gfx950"]


@dataclass(frozen=True)
class MlaDecodeGfx950Spec(MlaDecodeSpec):
    """:class:`MlaDecodeSpec` for gfx950."""

    ARCH: ClassVar[str] = "gfx950"
    SCORE_K: ClassVar[int] = 32
    PV_TRANSPOSE_READ: ClassVar[bool] = True
    LDS_BYTES: ClassVar[int] = 160 * 1024


def build_mla_decode_gfx950(
    spec: MlaDecodeGfx950Spec, *, arch: str = "gfx950"
) -> KernelDef:
    """Emit the gfx950 MLA decode latent kernel: the shared body, gfx950 operands.

    A builder of its own so that a kernel descriptor naming it hydrates
    :class:`MlaDecodeGfx950Spec`: the arch's emission choices are class
    attributes, which no field-based descriptor can carry.
    """
    return build_mla_decode(spec, arch=arch)
