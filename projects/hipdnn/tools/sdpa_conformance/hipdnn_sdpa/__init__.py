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

Conformance support (see ``aotriton/conformance_contract.py``): ``last_call(direction)`` returns the
calling thread's latest successful call as a ``CallRecord`` (engine, kernel id, ranked engines,
graph JSON, problem scalars); ``install_kernel_log()`` starts reading kernel ids and applicability
declines from the backend's INFO log; ``save_repro(dir, records, tensors)`` writes a replayable
repro directory (``aotriton/replay.py``).

``philox_mask`` reproduces the FlyDSL backward's dropout mask in pure torch.

The process must already provide a working hipDNN (``hipdnn_frontend`` importable,
engine plugins loadable); this package selects an engine only by name, per call.
"""

import importlib.util
import sys
from pathlib import Path


def _load_contract():
    """Register ``aotriton/conformance_contract.py`` as ``conformance_contract``.

    The runner puts ``aotriton/`` on ``PYTHONPATH`` so the module is already importable there;
    elsewhere (the package's own tests) it is loaded from the sibling directory, once, so every
    importer shares one module and one ``CallRecord`` class.
    """
    if "conformance_contract" in sys.modules:
        return
    path = (
        Path(__file__).resolve().parent.parent / "aotriton" / "conformance_contract.py"
    )
    spec = importlib.util.spec_from_file_location("conformance_contract", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["conformance_contract"] = module
    spec.loader.exec_module(module)


_load_contract()

from ._api import sdpa_bwd, sdpa_fwd
from ._kernel_log import install_kernel_log
from ._philox import philox_mask
from ._repro import save_repro
from ._runtime import last_call
from ._types import Dropout, EngineDeclined, Mask, Varlen

__all__ = [
    "Dropout",
    "EngineDeclined",
    "Mask",
    "Varlen",
    "install_kernel_log",
    "last_call",
    "philox_mask",
    "save_repro",
    "sdpa_bwd",
    "sdpa_fwd",
]
