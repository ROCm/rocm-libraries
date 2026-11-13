# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.rmsnorm -- route ``F.rms_norm`` (every ``nn.RMSNorm`` and many
hand-rolled norms) onto a hipDNN RMSNorm graph.

**Pure passthrough.** The RMSNorm graph node is N-D: the normalized dims are the
trailing suffix where ``scale[i] == x[i]``, leading (batch) dims are preserved. So
the override builds it at the input's *true rank* -- no flatten, no ``.contiguous()``
-- from the input's actual dims + strides. The scale (weight) is described logically
as ``[1,…,1, *normalized_shape]`` at x's rank: a free rank-expanding view of the
contiguous parameter (leading batch dims strided 0). Multi-axis and non-2-D inputs
are built and submitted directly; a ``normalized_shape`` that does not match x's
trailing dims is still submitted and rejected *at hipDNN*, not pre-declined.

Two model-friendly touches:

  * **weightless norms** (``weight=None``, e.g. adaLN / block norms) synthesise a
    cached ones-weight of shape ``normalized_shape`` -- the node needs a scale
    tensor. This captures the majority of a real model's RMSNorm calls.
  * **eps=None** is resolved the way torch does (``torch.finfo(dtype).eps``) so the
    baked epsilon matches native output exactly.

Dtype is set per tensor (a mixed-dtype scale is described faithfully). The output is
``torch.empty_like(input)`` (preserving layout). Nothing about size, rank, axis, or
dtype is pre-judged -- hipDNN decides, and a call no loaded engine serves falls back
to native and is counted.
"""

import logging

from .base import NotApplicable, OpOverride

_X_UID, _W_UID, _EPS_UID, _Y_UID = 1, 2, 3, 4


def _scale_view(param, r):
    """Describe ``param`` (physical shape = the normalized dims) as a logical
    ``[1,…,1, *param.shape]`` view at rank ``r``: leading batch dims strided 0, the
    trailing dims carrying the parameter's own strides. Pure description, no copy."""
    lead = r - param.dim()
    dims = [1] * lead + [int(s) for s in param.shape]
    strides = [0] * lead + [int(s) for s in param.stride()]
    return dims, strides


class RmsNormOverride(OpOverride):
    op_name = "rms_norm"

    def __init__(self):
        super().__init__()
        self._ones_cache = {}

    def _ones(self, shape, dtype, device):
        key = (tuple(shape), dtype, str(device))
        w = self._ones_cache.get(key)
        if w is None:
            w = self.state.torch.ones(shape, dtype=dtype, device=device)
            self._ones_cache[key] = w
        return w

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
        eps_t = g.tensor(
            hipdnn.Tensor()
            .set_name("eps")
            .set_dim([1])
            .set_stride([1])
            .set_data_type(hipdnn.DataType.FLOAT)
            .set_uid(_EPS_UID)
            .set_value(float(eps))
        )
        attrs = hipdnn.RMSNormAttributes()
        attrs.set_forward_phase(hipdnn.NormFwdPhase.INFERENCE)
        attrs.set_epsilon(eps_t)

        y_t = g.rmsnorm(x_t, w_t, attrs)[0]
        y_t.set_dim(list(y_dims))
        y_t.set_stride(list(y_str))
        y_t.set_data_type(st.dtype_map[y_dtype])
        y_t.set_uid(_Y_UID)
        y_t.set_output(True)  # terminal output must be non-virtual (MIOpen requires it)
        return g

    def _call(self, real, input, normalized_shape, weight=None, eps=None):
        torch = self.state.torch
        dtype = input.dtype

        def _native():
            return real(input, normalized_shape, weight, eps)

        try:
            n = int(input.shape[-1])
            key = f"N={n},dtype={self._tok(dtype)}"
        except Exception:  # noqa: BLE001
            return _native()

        ok, reason = self._gate(input)
        if not ok:
            self.note_native(key, reason)
            return _native()

        # eps=None -> torch's own default (finfo eps), so the baked value matches.
        e = float(eps) if eps is not None else float(torch.finfo(dtype).eps)
        weightless = weight is None
        try:
            try:
                ns = tuple(int(s) for s in normalized_shape)
            except TypeError:
                ns = (int(normalized_shape),)
            w = weight if not weightless else self._ones(ns, dtype, input.device)
            r = input.dim()
            w_dims, w_str = _scale_view(w, r)
            y = torch.empty_like(input)
            entry = self._cached_graph(
                (
                    tuple(input.shape),
                    tuple(input.stride()),
                    dtype,
                    tuple(w_dims),
                    tuple(w_str),
                    w.dtype,
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
                    _Y_UID: y.data_ptr(),
                },
                input.device,
            )
            self.note_aot(key, weightless=1 if weightless else 0)
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
