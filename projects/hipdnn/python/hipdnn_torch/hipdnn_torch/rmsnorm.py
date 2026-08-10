# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.rmsnorm -- route ``F.rms_norm`` (every ``nn.RMSNorm`` and many
hand-rolled norms) onto the hipDNN engine's 2-D RMSNorm.

The N-D activation ``[..., N]`` is flattened to ``[M, N]``. Two model-friendly
touches make it catch the norms real models actually emit:

  * **weightless norms** (``weight=None``, e.g. adaLN / block norms) are served by
    synthesising a cached ones-weight ``[1, N]``, since the adapter needs a
    per-column scale. This is what lets the layer capture the majority of a real
    model's RMSNorm calls.
  * **eps=None** is resolved the way torch does (``torch.finfo(dtype).eps``) so the
    baked epsilon matches native output exactly.

Constraint: single-axis (last-dim) norms on cuda f16/bf16 only; anything else falls
back to native and is logged.
"""

import logging

from .base import NotApplicable, OpOverride


class RmsNormOverride(OpOverride):
    op_name = "rms_norm"

    def __init__(self):
        super().__init__()
        self._ones_cache = {}

    def _ones(self, n, dtype, device):
        key = (n, dtype, str(device))
        w = self._ones_cache.get(key)
        if w is None:
            w = self.state.torch.ones(n, dtype=dtype, device=device)
            self._ones_cache[key] = w
        return w

    def _gate(self, input, weight, ns, n):
        torch = self.state.torch
        if not input.is_cuda:
            return False, "input not on cuda"
        if input.dtype not in (torch.float16, torch.bfloat16):
            return False, f"dtype {self._tok(input.dtype)} (need f16/bf16)"
        if weight is not None and weight.dtype != input.dtype:
            # The graph declares the weight tensor as the input dtype and
            # execute() passes its raw pointer, so a differing weight dtype
            # (e.g. an fp32 norm scale) would be reinterpreted byte-for-byte ->
            # silently wrong. Decline instead. (weight=None synthesises a
            # matching-dtype ones-weight, so it is always fine.)
            return False, f"weight dtype {self._tok(weight.dtype)} != input {self._tok(input.dtype)}"
        if input.dim() < 2:
            return False, "input rank < 2"
        if len(ns) != 1:
            return False, f"normalized_shape rank {len(ns)} (need 1)"
        if int(ns[0]) != n:
            return False, "normalized_shape != last dim"
        return True, ""

    def _graph(self, m, n, eps, dtype):
        st = self.state
        hipdnn = st.hipdnn
        hf = st.dtype_map[dtype]
        g = hipdnn.Graph()
        g.set_io_data_type(hf)
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        x_t = g.tensor(
            hipdnn.Tensor().set_name("x").set_dim([m, n]).set_stride([n, 1])
            .set_data_type(hf).set_uid(1)
        )
        w_t = g.tensor(
            hipdnn.Tensor().set_name("weight").set_dim([1, n]).set_stride([n, 1])
            .set_data_type(hf).set_uid(2)
        )
        eps_t = g.tensor(
            hipdnn.Tensor().set_name("eps").set_dim([1]).set_stride([1])
            .set_data_type(hipdnn.DataType.FLOAT).set_uid(3).set_value(float(eps))
        )
        attrs = hipdnn.RMSNormAttributes()
        attrs.set_forward_phase(hipdnn.NormFwdPhase.INFERENCE)
        attrs.set_epsilon(eps_t)

        y_t = g.rmsnorm(x_t, w_t, attrs)[0]
        y_t.set_dim([m, n])
        y_t.set_stride([n, 1])
        y_t.set_data_type(hf)
        y_t.set_uid(4)
        return g

    def _call(self, real, input, normalized_shape, weight=None, eps=None):
        torch = self.state.torch
        n = int(input.shape[-1])
        dtype = input.dtype
        try:
            ns = tuple(normalized_shape)
        except TypeError:
            ns = (int(normalized_shape),)
        key = f"N={n},dtype={self._tok(dtype)}"

        ok, reason = self._gate(input, weight, ns, n)
        if not ok:
            self.note_native(key, reason)
            return real(input, normalized_shape, weight, eps)

        # eps=None -> torch's own default (finfo eps), so the baked value matches.
        e = float(eps) if eps is not None else float(torch.finfo(dtype).eps)
        weightless = weight is None
        try:
            x2d = input.reshape(-1, n).contiguous()
            w = weight if not weightless else self._ones(n, dtype, input.device)
            w2d = w.reshape(1, n).contiguous()
            m = int(x2d.shape[0])
            entry = self._cached_graph(
                (m, n, e, dtype),
                lambda: self._graph(m, n, e, dtype),
                f"[{m},{n}] {dtype}",
            )
            y = torch.empty_like(x2d)
            self._execute(entry, {1: x2d.data_ptr(), 2: w2d.data_ptr(), 4: y.data_ptr()},
                          input.device)
            self.note_aot(key, weightless=1 if weightless else 0)
            return y.reshape(input.shape)
        except NotApplicable as na:  # engine can't serve this shape -> native
            self.note_native(key, str(na))
            return real(input, normalized_shape, weight, eps)
        except Exception as ex:  # noqa: BLE001 -- any failure -> native, never break the model
            self.note_native(key, f"exception: {type(ex).__name__}: {ex}",
                             level=logging.WARNING)
            return real(input, normalized_shape, weight, eps)
