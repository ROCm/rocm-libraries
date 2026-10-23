# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.linear -- route ``F.linear`` (every ``nn.Linear``) onto a hipDNN
matmul graph.

**Pure passthrough.** ``F.linear(x, W, b)`` computes ``y = x @ W^T + b`` with ``W``
physically ``[N, K]`` row-major. The matmul graph node is N-D (operands rank ≥ 2,
equal rank, leading batch dims broadcast), so the override builds it at the input's
*true rank* -- no flatten, no ``.contiguous()`` -- from the input's actual dims +
strides:

  * **A** = ``x`` described by ``x.shape`` / ``x.stride()`` (``[batch…, M, K]``).
  * **B** = the ``[N, K]`` weight described *logically* as ``[1,…,1, K, N]`` -- dims
    ``[1]*(r-2)+[K,N]``, strides ``[0]*(r-2)+[W.stride(1), W.stride(0)]``. This is a
    pure description of the existing buffer (the batch strides are 0, the trailing
    two express the transpose); no copy.
  * **C** = ``[batch…, M, N]`` contiguous (we own it), its strides fed to the graph.

A degenerate rank-1 input ``[K]`` gets a free ``[1, K]`` view to meet the node's
rank-2 minimum. Dtype is set per tensor, so a mixed-dtype linear is described
faithfully and left for hipDNN to accept or reject. Nothing about size, rank, or
K-agreement is pre-judged -- hipDNN's ``check_support`` decides, and a shape no
loaded engine claims falls back to native and is counted.

**Bias fusion (fallback ladder).** When a bias is present it is expressed as a
second graph node -- a ``PointwiseMode.ADD`` on the (virtual) matmul output and a
bias described logically as ``[1,…,1, N]`` (batch + M strides 0) -- so a capable
engine may fold it into the matmul as a single epilogue rather than a separate
kernel. Fusion only happens when bias and matmul are nodes of the *same* graph, so
we try, in order:
  1. the **fused** ``[matmul, bias-ADD]`` graph (one epilogue);
  2. on decline, the **matmul-only** graph + a native bias-add, counted as
     ``fused_declined`` (a real coverage gap);
  3. on decline of that too, full native ``F.linear``.
The two graphs use distinct cache keys, so each shape probes each path once.
Activation that follows a linear is a *separate* downstream ``F.gelu``/``F.silu``
call, invisible here, so activation fusion is deferred -- it needs cross-call
detection the per-op functional monkeypatch does not have.
"""

import logging

from .base import NotApplicable, OpOverride

_A_UID, _B_UID, _C_UID, _MM_UID, _BIAS_UID = 1, 2, 3, 4, 5


class LinearOverride(OpOverride):
    op_name = "linear"

    def _gate(self, input):
        # The only two legitimate stops: we need a device pointer to execute
        # against, and a dtype the graph builder can express. Size, rank,
        # K-agreement, mixed dtype -- all built and left to hipDNN.
        if not input.is_cuda:
            return False, "input not on cuda"  # execute() needs a device pointer
        if input.dtype not in self.state.dtype_map:
            return False, f"dtype {self._tok(input.dtype)} not graph-mappable"
        return True, ""

    def _tensor(self, g, name, dims, strides, dtype, uid, virtual=False):
        st = self.state
        hipdnn = st.hipdnn
        t = g.tensor(
            hipdnn.Tensor()
            .set_name(name)
            .set_dim(list(dims))
            .set_stride(list(strides))
            .set_data_type(st.dtype_map[dtype])
            .set_uid(uid)
        )
        return t

    def _graph(
        self, a_dims, a_str, a_dtype, b_dims, b_str, b_dtype, c_dims, c_str, c_dtype
    ):
        st = self.state
        hipdnn = st.hipdnn
        g = hipdnn.Graph()
        g.set_io_data_type(st.dtype_map[a_dtype])
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        a_t = self._tensor(g, "A", a_dims, a_str, a_dtype, _A_UID)
        b_t = self._tensor(g, "B", b_dims, b_str, b_dtype, _B_UID)
        attrs = hipdnn.MatmulAttributes()
        attrs.set_compute_data_type(hipdnn.DataType.FLOAT)
        c_t = g.matmul(a_t, b_t, attrs)
        c_t.set_dim(list(c_dims))
        c_t.set_stride(list(c_str))
        c_t.set_data_type(st.dtype_map[c_dtype])
        c_t.set_uid(_C_UID)
        c_t.set_output(True)  # terminal output must be non-virtual (MIOpen requires it)
        return g

    def _graph_biased(
        self,
        a_dims,
        a_str,
        a_dtype,
        b_dims,
        b_str,
        b_dtype,
        c_dims,
        c_str,
        c_dtype,
        bias_dims,
        bias_str,
        bias_dtype,
    ):
        """``[matmul, bias-ADD]`` graph: the matmul output is a *virtual*
        intermediate (uid 4) whose uid the bias-ADD consumes -- the pattern a
        provider folds into a single bias epilogue. uids: A=1, B=2, Y=3 (terminal),
        matmul-out=4 (virtual), bias=5."""
        st = self.state
        hipdnn = st.hipdnn
        g = hipdnn.Graph()
        g.set_io_data_type(st.dtype_map[a_dtype])
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        a_t = self._tensor(g, "A", a_dims, a_str, a_dtype, _A_UID)
        b_t = self._tensor(g, "B", b_dims, b_str, b_dtype, _B_UID)
        mm_attrs = hipdnn.MatmulAttributes()
        mm_attrs.set_compute_data_type(hipdnn.DataType.FLOAT)
        mm_t = g.matmul(a_t, b_t, mm_attrs)
        mm_t.set_dim(list(c_dims))
        mm_t.set_stride(list(c_str))
        mm_t.set_data_type(st.dtype_map[c_dtype])
        mm_t.set_uid(_MM_UID)
        mm_t.set_output(False)  # virtual: its uid feeds the bias-ADD (fusion match)

        bias_t = self._tensor(g, "bias", bias_dims, bias_str, bias_dtype, _BIAS_UID)
        add_attrs = hipdnn.PointwiseAttributes()
        add_attrs.set_mode(
            hipdnn.PointwiseMode.ADD
        )  # PointwiseAttributes has no compute-dtype setter
        y_t = g.pointwise(mm_t, bias_t, add_attrs)
        y_t.set_dim(list(c_dims))
        y_t.set_stride(list(c_str))
        y_t.set_data_type(st.dtype_map[c_dtype])
        y_t.set_uid(_C_UID)
        y_t.set_output(True)  # terminal output must be non-virtual
        return g

    def _operands(self, input, weight, n, k):
        """Describe A/B/C dims + strides at the input's true rank (bumping a
        rank-1 input to a free ``[1, K]`` view). Returns
        ``(a_dims, a_str, b_dims, b_str, c_dims, rank1)``."""
        if input.dim() >= 2:
            a_dims = [int(s) for s in input.shape]
            a_str = [int(s) for s in input.stride()]
            rank1 = False
        else:  # [K] -> logical [1, K]; leading unit dim's stride is irrelevant
            a_dims = [1, k]
            a_str = [0, int(input.stride(0))]
            rank1 = True
        r = len(a_dims)
        # B: the [N,K] weight described logically as [1,...,1, K, N].
        b_dims = [1] * (r - 2) + [k, n]
        b_str = [0] * (r - 2) + [int(weight.stride(1)), int(weight.stride(0))]
        c_dims = a_dims[:-1] + [n]
        return a_dims, a_str, b_dims, b_str, c_dims, rank1

    def _call(self, real, input, weight, bias=None):
        torch = self.state.torch

        def _native():
            return real(input, weight, bias)

        try:
            k = int(input.shape[-1])
            n = int(weight.shape[0])
            key = f"K={k},N={n},dtype={self._tok(input.dtype)}"
        except Exception:  # noqa: BLE001 -- unparseable call site -> native
            return _native()

        ok, reason = self._gate(input)
        if not ok:
            self.note_native(key, reason)
            return _native()

        try:
            a_dims, a_str, b_dims, b_str, c_dims, rank1 = self._operands(
                input, weight, n, k
            )
            r = len(a_dims)

            def _finish(y):
                # y is [..., M, N]; a rank-1 input produced [1, N] -> [N].
                return y.reshape(n) if rank1 else y

            # -- rung 1: fused matmul+bias (one epilogue call) -------------------
            fuse = bias is not None and bias.dim() == 1 and int(bias.shape[0]) == n
            if fuse:
                bias_dims = [1] * (r - 1) + [n]
                bias_str = [0] * (r - 1) + [int(bias.stride(0))]
                try:
                    y = torch.empty(c_dims, dtype=input.dtype, device=input.device)
                    entry = self._cached_graph(
                        (
                            "biased",
                            tuple(a_dims),
                            tuple(a_str),
                            input.dtype,
                            tuple(b_dims),
                            tuple(b_str),
                            weight.dtype,
                            tuple(y.stride()),
                            tuple(bias_dims),
                            tuple(bias_str),
                            bias.dtype,
                        ),
                        lambda: self._graph_biased(
                            a_dims,
                            a_str,
                            input.dtype,
                            b_dims,
                            b_str,
                            weight.dtype,
                            c_dims,
                            list(y.stride()),
                            input.dtype,
                            bias_dims,
                            bias_str,
                            bias.dtype,
                        ),
                        f"{c_dims}+bias {input.dtype}",
                    )
                    self._execute(
                        entry,
                        {
                            _A_UID: input.data_ptr(),
                            _B_UID: weight.data_ptr(),
                            _BIAS_UID: bias.data_ptr(),
                            _C_UID: y.data_ptr(),
                        },
                        input.device,
                    )
                    self.note_aot(key, biased=1, fused=1)
                    return _finish(y)
                except NotApplicable as na:
                    # No engine served the fused epilogue for this shape -> degrade
                    # to matmul + native bias. Record it as a finding.
                    self.note_native(
                        key, f"bias-fusion declined -> matmul+native bias ({na})"
                    )

            # -- rung 2: matmul-only graph + native bias-add --------------------
            y = torch.empty(c_dims, dtype=input.dtype, device=input.device)
            entry = self._cached_graph(
                (
                    tuple(a_dims),
                    tuple(a_str),
                    input.dtype,
                    tuple(b_dims),
                    tuple(b_str),
                    weight.dtype,
                    tuple(y.stride()),
                ),
                lambda: self._graph(
                    a_dims,
                    a_str,
                    input.dtype,
                    b_dims,
                    b_str,
                    weight.dtype,
                    c_dims,
                    list(y.stride()),
                    input.dtype,
                ),
                f"{c_dims} {input.dtype}",
            )
            self._execute(
                entry,
                {
                    _A_UID: input.data_ptr(),
                    _B_UID: weight.data_ptr(),
                    _C_UID: y.data_ptr(),
                },
                input.device,
            )
            extras = {}
            if bias is not None:
                y = y + bias  # native epilogue; this matmul graph has no bias node
                extras["biased"] = 1
                if fuse:  # we tried to fuse and fell back to matmul here
                    extras["fused_declined"] = 1
            self.note_aot(key, **extras)
            return _finish(y)
        except NotApplicable as na:  # engine can't serve this shape -> native
            self.note_native(key, str(na))
            return _native()
        except (
            Exception
        ) as e:  # noqa: BLE001 -- any failure -> native, never break the model
            self.note_native(
                key, f"exception: {type(e).__name__}: {e}", level=logging.WARNING
            )
            return _native()
