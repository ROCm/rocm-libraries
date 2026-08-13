# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.linear -- route ``F.linear`` (every ``nn.Linear``) onto the hipDNN
engine's RCR matmul.

``F.linear(x, W, b)`` computes ``y = x @ W^T + b`` with ``W`` physically ``[N, K]``
row-major -- which is exactly the RCR layout the matmul path serves: A ``[M, K]``
strides ``{K, 1}``, B logical ``[K, N]`` strides ``{1, K}`` (the same ``[N, K]``
weight buffer), C ``[M, N]`` strides ``{N, 1}``. The N-D activation ``[..., K]`` is
flattened to ``[M, K]``.

**Bias fusion (fallback ladder).** When a bias is present we express it as a
second graph node -- a ``PointwiseMode.ADD`` on the (virtual) matmul output and a
per-column bias vector -- so a capable engine (hipBLASLt) folds it into the matmul
as a single ``HIPBLASLT_EPILOGUE_BIAS`` call instead of a separate elementwise
kernel. The provider only fuses bias/activation that appear as nodes in the *same*
graph, so a matmul-only graph + Python bias-add can never fuse. We therefore try,
in order:
  1. the **fused** ``[matmul, bias-ADD]`` graph (one epilogue call);
  2. on decline, the **matmul-only** graph + a native Python bias-add, counted as
     ``fused_declined`` and logged as a finding (hipBLASLt lacked a bias-epilogue
     solution for this shape/arch -- e.g. the thin gfx1151 tuned set);
  3. on decline of that too, full native ``F.linear``.
The two graphs use distinct cache keys, so each shape probes each path once.
Activation that follows a linear is a *separate* downstream ``F.gelu``/``F.silu``
call, invisible here, so activation fusion (``GELU_BIAS`` etc.) is deferred -- it
needs cross-call detection the functional monkeypatch does not have.

Shape is not pre-filtered: the override builds the graph and lets hipDNN rank the
loaded engines. rocKE's wmma matmul serves multiple-of-16 M/N/K only, but
hipBLASLt serves arbitrary sizes -- so under pick-best a non-%16 shape routes to
hipBLASLt, and only a shape no loaded engine claims falls back to native.
"""

import logging

from .base import NotApplicable, OpOverride


class LinearOverride(OpOverride):
    op_name = "linear"

    def _gate(self, input, weight):
        # Structural facts (needed to build/address the graph) + what the matmul
        # graph cannot represent. No shape/size pre-filter: build and let hipDNN's
        # engine ranking decide (rocKE wants %16 M/N/K, hipBLASLt serves any size,
        # so a non-%16 shape routes to hipBLASLt under pick-best; only a shape no
        # loaded engine claims falls back to native).
        if not input.is_cuda:
            return False, "input not on cuda"  # execute() needs a device pointer
        if input.dtype not in self.state.dtype_map:
            return False, f"dtype {self._tok(input.dtype)} not graph-mappable"
        if weight.dim() != 2:
            return False, "weight not 2-D"  # builder addresses B as [N,K]
        if input.dim() < 2:
            return False, "input rank < 2"  # need a trailing K to flatten to [M,K]
        k = int(input.shape[-1])
        if int(weight.shape[1]) != k:
            return False, "weight/input K mismatch"  # ill-formed matmul
        # --- not representable in the matmul graph -> catch here ---
        if weight.dtype != input.dtype:
            # The graph declares B as the input dtype and execute() passes the
            # weight's raw pointer, so a differing weight dtype would be
            # reinterpreted byte-for-byte -> silently wrong. Decline instead.
            return (
                False,
                f"weight dtype {self._tok(weight.dtype)} != input {self._tok(input.dtype)}",
            )
        return True, ""

    def _graph(self, m, n, k, dtype):
        st = self.state
        hipdnn = st.hipdnn
        hf = st.dtype_map[dtype]
        g = hipdnn.Graph()
        g.set_io_data_type(hf)
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        a_t = g.tensor(
            hipdnn.Tensor()
            .set_name("A")
            .set_dim([m, k])
            .set_stride([k, 1])
            .set_data_type(hf)
            .set_uid(1)
        )
        # B: logical [K,N] strides {1,K} -- the physical [N,K] nn.Linear weight.
        b_t = g.tensor(
            hipdnn.Tensor()
            .set_name("B")
            .set_dim([k, n])
            .set_stride([1, k])
            .set_data_type(hf)
            .set_uid(2)
        )
        attrs = hipdnn.MatmulAttributes()
        attrs.set_compute_data_type(hipdnn.DataType.FLOAT)
        c_t = g.matmul(a_t, b_t, attrs)
        c_t.set_dim([m, n])
        c_t.set_stride([n, 1])
        c_t.set_data_type(hf)
        c_t.set_uid(3)
        c_t.set_output(True)  # terminal output must be non-virtual (MIOpen requires it)
        return g

    def _graph_biased(self, m, n, k, dtype):
        """``[matmul, bias-ADD]`` graph: the matmul output is a *virtual*
        intermediate (uid 4) whose uid the bias-ADD consumes -- exactly the pattern
        the hipBLASLt provider folds into ``HIPBLASLT_EPILOGUE_BIAS``. The bias is a
        per-column vector ``[1, N]`` broadcast over the M rows (stride ``[0, 1]``).
        uids: A=1, B=2, Y=3 (terminal), matmul-out=4 (virtual), bias=5."""
        st = self.state
        hipdnn = st.hipdnn
        hf = st.dtype_map[dtype]
        g = hipdnn.Graph()
        g.set_io_data_type(hf)
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        a_t = g.tensor(
            hipdnn.Tensor()
            .set_name("A")
            .set_dim([m, k])
            .set_stride([k, 1])
            .set_data_type(hf)
            .set_uid(1)
        )
        b_t = g.tensor(
            hipdnn.Tensor()
            .set_name("B")
            .set_dim([k, n])
            .set_stride([1, k])
            .set_data_type(hf)
            .set_uid(2)
        )
        mm_attrs = hipdnn.MatmulAttributes()
        mm_attrs.set_compute_data_type(hipdnn.DataType.FLOAT)
        mm_t = g.matmul(a_t, b_t, mm_attrs)
        mm_t.set_dim([m, n])
        mm_t.set_stride([n, 1])
        mm_t.set_data_type(hf)
        mm_t.set_uid(4)
        mm_t.set_output(False)  # virtual: its uid feeds the bias-ADD (fusion match)

        bias_t = g.tensor(
            hipdnn.Tensor()
            .set_name("bias")
            .set_dim([1, n])
            .set_stride([0, 1])
            .set_data_type(hf)
            .set_uid(5)
        )
        add_attrs = hipdnn.PointwiseAttributes()
        add_attrs.set_mode(
            hipdnn.PointwiseMode.ADD
        )  # PointwiseAttributes has no compute-dtype setter
        y_t = g.pointwise(mm_t, bias_t, add_attrs)
        y_t.set_dim([m, n])
        y_t.set_stride([n, 1])
        y_t.set_data_type(hf)
        y_t.set_uid(3)
        y_t.set_output(True)  # terminal output must be non-virtual
        return g

    def _fusible_bias(self, bias, n):
        """True if this bias can be expressed as the fused ``[matmul, bias-ADD]``
        graph: a 1-D per-column vector of length N in the input dtype (what the
        hipBLASLt bias epilogue requires)."""
        return bias is not None and bias.dim() == 1 and int(bias.shape[0]) == n

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

            # -- rung 1: fused matmul+bias (one hipBLASLt EPILOGUE_BIAS call) -----
            fuse = self._fusible_bias(bias, n) and bias.dtype == input.dtype
            if fuse:
                try:
                    entry = self._cached_graph(
                        ("biased", m, n, k, input.dtype),
                        lambda: self._graph_biased(m, n, k, input.dtype),
                        f"[{m},{n},{k}]+bias {input.dtype}",
                    )
                    y = torch.empty(m, n, dtype=input.dtype, device=input.device)
                    self._execute(
                        entry,
                        {
                            1: x2d.data_ptr(),
                            2: w.data_ptr(),
                            5: bias.contiguous().data_ptr(),
                            3: y.data_ptr(),
                        },
                        input.device,
                    )
                    self.note_aot(key, biased=1, fused=1)
                    return y.reshape(*input.shape[:-1], n)
                except NotApplicable as na:
                    # No engine served the fused epilogue for this shape/arch (e.g.
                    # hipBLASLt's thin gfx1151 tuned set) -> degrade to GEMM + native
                    # bias. Record it as a finding: this is a real coverage gap.
                    self.note_native(
                        key, f"bias-fusion declined -> GEMM+native bias ({na})"
                    )

            # -- rung 2: matmul-only graph + native bias-add ----------------------
            entry = self._cached_graph(
                (m, n, k, input.dtype),
                lambda: self._graph(m, n, k, input.dtype),
                f"[{m},{n},{k}] {input.dtype}",
            )
            y = torch.empty(m, n, dtype=input.dtype, device=input.device)
            self._execute(
                entry,
                {1: x2d.data_ptr(), 2: w.data_ptr(), 3: y.data_ptr()},
                input.device,
            )
            extras = {}
            if bias is not None:
                y = y + bias  # native epilogue; this matmul graph has no bias node
                extras["biased"] = 1
                if fuse:  # we tried to fuse and fell back to GEMM here
                    extras["fused_declined"] = 1
            self.note_aot(key, **extras)
            return y.reshape(*input.shape[:-1], n)
        except NotApplicable as na:  # engine can't serve this shape -> native
            self.note_native(key, str(na))
            return real(input, weight, bias)
        except (
            Exception
        ) as e:  # noqa: BLE001 -- any failure -> native, never break the model
            self.note_native(
                key, f"exception: {type(e).__name__}: {e}", level=logging.WARNING
            )
            return real(input, weight, bias)
