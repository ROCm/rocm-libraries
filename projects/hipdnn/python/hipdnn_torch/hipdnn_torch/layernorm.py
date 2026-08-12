# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.layernorm -- route ``F.layer_norm`` (every ``nn.LayerNorm``) onto the
hipDNN engine's 2-D LayerNorm.

The N-D activation ``[..., N]`` is flattened to ``[M, N]``. Like the RMSNorm
override, two model-friendly touches make it catch the norms real models emit:

  * **weightless / biasless norms** (``weight=None`` / ``bias=None``) are served by
    synthesising a cached ones-weight and zeros-bias ``[1, N]``, since the adapter
    needs a per-column scale *and* bias (LayerNorm is a 3-input op).
  * **eps** defaults to torch's ``1e-5`` and is baked into the graph so the output
    matches native exactly.

Constraint: single-axis (last-dim) norms on cuda f16/bf16 only; anything else falls
back to native and is logged.
"""

import logging

from .base import NotApplicable, OpOverride


class LayerNormOverride(OpOverride):
    op_name = "layer_norm"

    def __init__(self):
        super().__init__()
        self._ones_cache = {}
        self._zeros_cache = {}

    def _ones(self, n, dtype, device):
        key = (n, dtype, str(device))
        w = self._ones_cache.get(key)
        if w is None:
            w = self.state.torch.ones(n, dtype=dtype, device=device)
            self._ones_cache[key] = w
        return w

    def _zeros(self, n, dtype, device):
        key = (n, dtype, str(device))
        b = self._zeros_cache.get(key)
        if b is None:
            b = self.state.torch.zeros(n, dtype=dtype, device=device)
            self._zeros_cache[key] = b
        return b

    def _gate(self, input, weight, bias, ns, n):
        torch = self.state.torch
        if not input.is_cuda:
            return False, "input not on cuda"
        if input.dtype not in (torch.float16, torch.bfloat16):
            return False, f"dtype {self._tok(input.dtype)} (need f16/bf16)"
        # The graph declares weight/bias as the input dtype and execute() passes
        # their raw pointers, so a differing dtype (e.g. an fp32 norm scale) would
        # be reinterpreted byte-for-byte -> silently wrong. Decline instead.
        # (weight/bias=None synthesise matching-dtype tensors, so they are fine.)
        if weight is not None and weight.dtype != input.dtype:
            return (
                False,
                f"weight dtype {self._tok(weight.dtype)} != input {self._tok(input.dtype)}",
            )
        if bias is not None and bias.dtype != input.dtype:
            return (
                False,
                f"bias dtype {self._tok(bias.dtype)} != input {self._tok(input.dtype)}",
            )
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
            hipdnn.Tensor()
            .set_name("x")
            .set_dim([m, n])
            .set_stride([n, 1])
            .set_data_type(hf)
            .set_uid(1)
        )
        w_t = g.tensor(
            hipdnn.Tensor()
            .set_name("weight")
            .set_dim([1, n])
            .set_stride([n, 1])
            .set_data_type(hf)
            .set_uid(2)
        )
        b_t = g.tensor(
            hipdnn.Tensor()
            .set_name("bias")
            .set_dim([1, n])
            .set_stride([n, 1])
            .set_data_type(hf)
            .set_uid(3)
        )
        eps_t = g.tensor(
            hipdnn.Tensor()
            .set_name("eps")
            .set_dim([1])
            .set_stride([1])
            .set_data_type(hipdnn.DataType.FLOAT)
            .set_uid(4)
            .set_value(float(eps))
        )
        attrs = hipdnn.LayernormAttributes()
        attrs.set_forward_phase(hipdnn.NormFwdPhase.INFERENCE)
        attrs.set_epsilon(eps_t)

        y_t = g.layernorm(x_t, w_t, b_t, attrs)[0]
        y_t.set_dim([m, n])
        y_t.set_stride([n, 1])
        y_t.set_data_type(hf)
        y_t.set_uid(5)
        return g

    def _call(self, real, input, normalized_shape, weight=None, bias=None, eps=1e-5):
        torch = self.state.torch
        n = int(input.shape[-1])
        dtype = input.dtype
        try:
            ns = tuple(normalized_shape)
        except TypeError:
            ns = (int(normalized_shape),)
        key = f"N={n},dtype={self._tok(dtype)}"

        ok, reason = self._gate(input, weight, bias, ns, n)
        if not ok:
            self.note_native(key, reason)
            return real(input, normalized_shape, weight, bias, eps)

        e = float(eps) if eps is not None else 1e-5
        weightless = weight is None
        biasless = bias is None
        try:
            x2d = input.reshape(-1, n).contiguous()
            w = weight if not weightless else self._ones(n, dtype, input.device)
            b = bias if not biasless else self._zeros(n, dtype, input.device)
            w2d = w.reshape(1, n).contiguous()
            b2d = b.reshape(1, n).contiguous()
            m = int(x2d.shape[0])
            entry = self._cached_graph(
                (m, n, e, dtype),
                lambda: self._graph(m, n, e, dtype),
                f"[{m},{n}] {dtype}",
            )
            y = torch.empty_like(x2d)
            self._execute(
                entry,
                {
                    1: x2d.data_ptr(),
                    2: w2d.data_ptr(),
                    3: b2d.data_ptr(),
                    5: y.data_ptr(),
                },
                input.device,
            )
            self.note_aot(
                key, weightless=1 if weightless else 0, biasless=1 if biasless else 0
            )
            return y.reshape(input.shape)
        except NotApplicable as na:  # engine can't serve this shape -> native
            self.note_native(key, str(na))
            return real(input, normalized_shape, weight, bias, eps)
        except (
            Exception
        ) as ex:  # noqa: BLE001 -- any failure -> native, never break the model
            self.note_native(
                key, f"exception: {type(ex).__name__}: {ex}", level=logging.WARNING
            )
            return real(input, normalized_shape, weight, bias, eps)
