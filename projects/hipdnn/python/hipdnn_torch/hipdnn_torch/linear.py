# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.linear -- route ``F.linear`` (every ``nn.Linear``) onto the hipDNN
engine's RCR matmul.

``F.linear(x, W, b)`` computes ``y = x @ W^T + b`` with ``W`` physically ``[N, K]``
row-major -- which is exactly the RCR layout the matmul path serves: A ``[M, K]``
strides ``{K, 1}``, B logical ``[K, N]`` strides ``{1, K}`` (the same ``[N, K]``
weight buffer), C ``[M, N]`` strides ``{N, 1}``. The N-D activation ``[..., K]`` is
flattened to ``[M, K]``. Bias is added natively *after* the matmul (the kernel ABI
has no epilogue), so a bias no longer forces a fallback -- it is just counted.

Constraint: the reference wmma matmul serves multiple-of-16 M/N/K only; anything
else falls back to native and is logged.
"""

import logging

from .base import OpOverride


class LinearOverride(OpOverride):
    op_name = "linear"

    def _gate(self, input, weight):
        torch = self.state.torch
        if not input.is_cuda:
            return False, "input not on cuda"
        if input.dtype not in (torch.float16, torch.bfloat16):
            return False, f"dtype {self._tok(input.dtype)} (need f16/bf16)"
        if weight.dim() != 2:
            return False, "weight not 2-D"
        if input.dim() < 2:
            return False, "input rank < 2"
        k = int(input.shape[-1])
        if int(weight.shape[1]) != k:
            return False, "weight/input K mismatch"
        n = int(weight.shape[0])
        m = 1
        for s in input.shape[:-1]:
            m *= int(s)
        if m % 16 or n % 16 or k % 16:
            return False, f"M/N/K not all %16 (M={m},N={n},K={k})"
        return True, ""

    def _graph(self, m, n, k, dtype):
        st = self.state
        hipdnn = st.hipdnn
        hf = st.dtype_map[dtype]
        g = hipdnn.Graph()
        g.set_io_data_type(hf)
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        a_t = g.tensor(
            hipdnn.Tensor().set_name("A").set_dim([m, k]).set_stride([k, 1])
            .set_data_type(hf).set_uid(1)
        )
        # B: logical [K,N] strides {1,K} -- the physical [N,K] nn.Linear weight.
        b_t = g.tensor(
            hipdnn.Tensor().set_name("B").set_dim([k, n]).set_stride([1, k])
            .set_data_type(hf).set_uid(2)
        )
        attrs = hipdnn.MatmulAttributes()
        attrs.set_compute_data_type(hipdnn.DataType.FLOAT)
        c_t = g.matmul(a_t, b_t, attrs)
        c_t.set_dim([m, n])
        c_t.set_stride([n, 1])
        c_t.set_data_type(hf)
        c_t.set_uid(3)
        return g

    def _call(self, real, input, weight, bias=None):
        torch = self.state.torch
        k = int(input.shape[-1])
        n = int(weight.shape[0]) if weight.dim() == 2 else -1
        key = f"K={k},N={n},dtype={self._tok(input.dtype)}"

        ok, reason = self._gate(input, weight)
        if not ok:
            self.note_native(key, reason)
            return real(input, weight, bias)

        try:
            x2d = input.reshape(-1, k).contiguous()
            w = weight.contiguous()
            m = int(x2d.shape[0])
            entry = self._cached_graph(
                (m, n, k, input.dtype),
                lambda: self._graph(m, n, k, input.dtype),
                f"[{m},{n},{k}] {input.dtype}",
            )
            y = torch.empty(m, n, dtype=input.dtype, device=input.device)
            self._execute(entry, {1: x2d.data_ptr(), 2: w.data_ptr(), 3: y.data_ptr()},
                          input.device)
            extras = {}
            if bias is not None:
                y = y + bias  # native epilogue; kernel ABI has no bias
                extras["biased"] = 1
            self.note_aot(key, **extras)
            return y.reshape(*input.shape[:-1], n)
        except Exception as e:  # noqa: BLE001 -- any failure -> native, never break the model
            self.note_native(key, f"exception: {type(e).__name__}: {e}",
                             level=logging.WARNING)
            return real(input, weight, bias)
