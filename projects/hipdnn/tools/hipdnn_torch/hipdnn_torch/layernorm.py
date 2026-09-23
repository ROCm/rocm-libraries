# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.layernorm -- route ``F.layer_norm`` (every ``nn.LayerNorm``) onto a
hipDNN LayerNorm graph.

**Pure passthrough.** The LayerNorm graph node is N-D (it infers the normalized dim
count from the scale shape). So the override builds it at the input's *true rank* --
no flatten, no ``.contiguous()`` -- from the input's actual dims + strides. Scale and
bias are described logically as ``[1,…,1, *normalized_shape]`` at x's rank: free
rank-expanding views of the contiguous parameters (leading batch dims strided 0).
Multi-axis and non-2-D inputs are built and submitted directly; a mismatched
``normalized_shape`` is submitted and rejected *at hipDNN*, not pre-declined.

Two model-friendly touches:

  * **weightless / biasless norms** (``weight=None`` / ``bias=None``) synthesise a
    cached ones-weight and zeros-bias of shape ``normalized_shape`` -- LayerNorm is
    a 3-input op.
  * **eps** defaults to torch's ``1e-5`` and is baked into the graph so the output
    matches native exactly.

Dtype is set per tensor (a mixed-dtype scale/bias is described faithfully). The
output is ``torch.empty_like(input)`` (preserving layout). Nothing about size, rank,
axis, or dtype is pre-judged -- hipDNN decides, and a call no loaded engine serves
falls back to native and is counted.
"""

import logging

from .base import NotApplicable, OpOverride
from .rmsnorm import _scale_view

_X_UID, _W_UID, _B_UID, _EPS_UID, _Y_UID = 1, 2, 3, 4, 5


class LayerNormOverride(OpOverride):
    op_name = "layer_norm"

    def __init__(self):
        super().__init__()
        self._ones_cache = {}
        self._zeros_cache = {}

    def _ones(self, shape, dtype, device):
        key = (tuple(shape), dtype, str(device))
        w = self._ones_cache.get(key)
        if w is None:
            w = self.state.torch.ones(shape, dtype=dtype, device=device)
            self._ones_cache[key] = w
        return w

    def _zeros(self, shape, dtype, device):
        key = (tuple(shape), dtype, str(device))
        b = self._zeros_cache.get(key)
        if b is None:
            b = self.state.torch.zeros(shape, dtype=dtype, device=device)
            self._zeros_cache[key] = b
        return b

    def _gate(self, input):
        # The only two legitimate stops: a device pointer to execute against and a
        # dtype the graph builder can express. Rank, axis, multi-axis, mixed dtype
        # -- all built and left to hipDNN.
        if not input.is_cuda:
            return False, "input not on cuda"  # execute() needs a device pointer
        if input.dtype not in self.state.dtype_map:
            return False, f"dtype {self._tok(input.dtype)} not graph-mappable"
        return True, ""

    def _graph(
        self,
        x_dims,
        x_str,
        x_dtype,
        w_dims,
        w_str,
        w_dtype,
        b_dims,
        b_str,
        b_dtype,
        y_dims,
        y_str,
        y_dtype,
        eps,
    ):
        st = self.state
        hipdnn = st.hipdnn
        g = hipdnn.Graph()
        g.set_io_data_type(st.dtype_map[x_dtype])
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        x_t = g.tensor(
            hipdnn.Tensor()
            .set_name("x")
            .set_dim(list(x_dims))
            .set_stride(list(x_str))
            .set_data_type(st.dtype_map[x_dtype])
            .set_uid(_X_UID)
        )
        w_t = g.tensor(
            hipdnn.Tensor()
            .set_name("weight")
            .set_dim(list(w_dims))
            .set_stride(list(w_str))
            .set_data_type(st.dtype_map[w_dtype])
            .set_uid(_W_UID)
        )
        b_t = g.tensor(
            hipdnn.Tensor()
            .set_name("bias")
            .set_dim(list(b_dims))
            .set_stride(list(b_str))
            .set_data_type(st.dtype_map[b_dtype])
            .set_uid(_B_UID)
        )
        eps_t = g.tensor(
            hipdnn.Tensor()
            .set_name("eps")
            .set_dim([1])
            .set_stride([1])
            .set_data_type(hipdnn.DataType.FLOAT)
            .set_uid(_EPS_UID)
            .set_value(float(eps))
        )
        attrs = hipdnn.LayernormAttributes()
        attrs.set_forward_phase(hipdnn.NormFwdPhase.INFERENCE)
        attrs.set_epsilon(eps_t)

        y_t = g.layernorm(x_t, w_t, b_t, attrs)[0]
        y_t.set_dim(list(y_dims))
        y_t.set_stride(list(y_str))
        y_t.set_data_type(st.dtype_map[y_dtype])
        y_t.set_uid(_Y_UID)
        y_t.set_output(True)  # terminal output must be non-virtual (MIOpen requires it)
        return g

    def _call(self, real, input, normalized_shape, weight=None, bias=None, eps=1e-5):
        torch = self.state.torch
        dtype = input.dtype

        def _native():
            return real(input, normalized_shape, weight, bias, eps)

        try:
            n = int(input.shape[-1])
            key = f"N={n},dtype={self._tok(dtype)}"
        except Exception:  # noqa: BLE001
            return _native()

        ok, reason = self._gate(input)
        if not ok:
            self.note_native(key, reason)
            return _native()

        e = float(eps) if eps is not None else 1e-5
        weightless = weight is None
        biasless = bias is None
        try:
            try:
                ns = tuple(int(s) for s in normalized_shape)
            except TypeError:
                ns = (int(normalized_shape),)
            w = weight if not weightless else self._ones(ns, dtype, input.device)
            b = bias if not biasless else self._zeros(ns, dtype, input.device)
            r = input.dim()
            w_dims, w_str = _scale_view(w, r)
            b_dims, b_str = _scale_view(b, r)
            y = torch.empty_like(input)
            entry = self._cached_graph(
                (
                    tuple(input.shape),
                    tuple(input.stride()),
                    dtype,
                    tuple(w_dims),
                    tuple(w_str),
                    w.dtype,
                    tuple(b_dims),
                    tuple(b_str),
                    b.dtype,
                    tuple(y.stride()),
                    e,
                ),
                lambda: self._graph(
                    list(input.shape),
                    list(input.stride()),
                    dtype,
                    w_dims,
                    w_str,
                    w.dtype,
                    b_dims,
                    b_str,
                    b.dtype,
                    list(y.shape),
                    list(y.stride()),
                    dtype,
                    e,
                ),
                f"{list(input.shape)} ns={list(ns)} {dtype}",
            )
            self._execute(
                entry,
                {
                    _X_UID: input.data_ptr(),
                    _W_UID: w.data_ptr(),
                    _B_UID: b.data_ptr(),
                    _Y_UID: y.data_ptr(),
                },
                input.device,
            )
            self.note_aot(
                key, weightless=1 if weightless else 0, biasless=1 if biasless else 0
            )
            return y
        except NotApplicable as na:  # engine can't serve this call -> native
            self.note_native(key, str(na))
            return _native()
        except (
            Exception
        ) as ex:  # noqa: BLE001 -- any failure -> native, never break the model
            self.note_native(
                key, f"exception: {type(ex).__name__}: {ex}", level=logging.WARNING
            )
            return _native()
