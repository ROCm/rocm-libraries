# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""hipdnn_sdpa -- scaled dot-product attention on hipDNN, for conformance testing.

All tensors are caller-allocated torch tensors; hipDNN reads and writes them in
place on ``torch.cuda.current_stream()``. Call ``hipdnn_sdpa.sdpa_fwd`` and
``hipdnn_sdpa.sdpa_bwd`` as module attributes at call time, never through
``from hipdnn_sdpa import sdpa_*``, so a patch of those attributes takes effect.

``sdpa_fwd(q, k, v, *, o, stats, scale, bias=None, mask=None, dropout=None,
varlen=None, engine=None)`` and ``sdpa_bwd(q, k, v, o, do, stats, *, dq, dk, dv,
scale, dbias=None, bias=None, mask=None, dropout=None, varlen=None, engine=None)``
return the engine name used and raise :class:`EngineDeclined` when hipDNN cannot
serve the call. There is no native fallback.

Tensor shapes:

* Dense and padded varlen: 4-D BHSD views with real strides; ``stats`` is a 4-D
  ``[B, Hq, Sq, 1]`` FP32 view.
* Packed varlen (``Varlen.pos_*`` set): 4-D ``[1, H, T, D]`` views (``stats``
  ``[1, H, T, 1]``). A tensor that breaks RFC-0014's sequence-outermost rule
  raises :class:`EngineDeclined` before any graph is built; otherwise it is
  declared ``[B, H, max_S, D]`` with a ``[B+1, 1, 1, 1]`` INT32 ragged offset.
* Varlen always sets ``padding_mask`` with both sequence lengths as
  ``[B, 1, 1, 1]`` INT32.

``philox_mask`` reproduces the FlyDSL backward's dropout mask in pure torch.

The process must already provide a working hipDNN (``hipdnn_frontend`` importable,
engine plugins loadable); this package selects an engine only by name, per call.
"""

from ._api import sdpa_bwd, sdpa_fwd
from ._philox import philox_mask
from ._types import Dropout, EngineDeclined, Mask, Varlen

__all__ = [
    "Dropout",
    "EngineDeclined",
    "Mask",
    "Varlen",
    "philox_mask",
    "sdpa_bwd",
    "sdpa_fwd",
]
