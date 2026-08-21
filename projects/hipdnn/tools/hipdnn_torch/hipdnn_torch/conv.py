# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.conv -- route ``F.conv2d``/``F.conv3d`` (every ``nn.Conv2d`` /
``nn.Conv3d``) onto a hipDNN forward-convolution graph.

**Pure passthrough.** The override builds the conv graph from the input's and
weight's *actual* dims + strides -- whatever layout they arrive in -- and submits it.
It imposes no layout (no ``channels_last`` conversion, no ``.contiguous()``) and
pre-judges nothing about what any engine can serve; hipDNN's ``check_support`` /
``execute`` decides, and a shape no loaded engine claims falls back to native and is
counted. The output is allocated to mirror the input's layout (channels-last in →
channels-last out, contiguous in → contiguous out), matching native's contract, and
its actual strides are fed to the graph.

Every torch conv parameter is translated into the graph:

  * **groups** need no attribute -- the graph infers them from the weight's own
    ``Cin/groups`` channel count (weight dims ``[K, Cin/groups, *kernel]``).
  * **padding** ``int``/tuple is symmetric (``pre == post``); ``'valid'`` → 0;
    ``'same'`` → per-axis total ``dilation*(k-1)`` split ``pre = total//2``,
    ``post = total - pre`` (torch pads the trailing side at least as much;
    ``'same'`` requires stride 1). Expressed via ``set_pre_padding`` /
    ``set_post_padding``.
  * **stride** / **dilation** pass straight through.
  * **dtype** is set per tensor, so a mixed-dtype conv is described faithfully and
    left for hipDNN to accept or reject.

Bias is added as a native epilogue after the conv (a same-graph bias fusion is a
possible future enhancement) and is simply counted.
"""

import logging

from .base import NotApplicable, OpOverride

_X_UID, _W_UID, _Y_UID = 1, 2, 3


def _ntuple(v, n):
    """Normalise an int-or-tuple hyperparameter to an ``n``-int tuple, or ``None``
    if it is neither (e.g. a ``'same'``/``'valid'`` padding string). A length-1
    tuple is broadcast to ``n`` (matches torch's own handling)."""
    if isinstance(v, int):
        return (v,) * n
    try:
        t = tuple(int(x) for x in v)
    except (TypeError, ValueError):
        return None
    if len(t) == 1:
        return t * n
    if len(t) == n:
        return t
    return None


def _resolve_pads(padding, dil_t, ksp, n):
    """Translate torch's ``padding`` argument to ``(pre, post)`` ``n``-int tuples.

    Numeric padding is symmetric (``pre == post``). ``'valid'`` → all-zero.
    ``'same'`` → per-axis total ``dilation*(k-1)``, split ``pre = total//2``,
    ``post = total - pre``. Returns ``None`` if the argument is unrecognised."""
    if isinstance(padding, str):
        if padding == "valid":
            z = (0,) * n
            return z, z
        if padding == "same":
            pre = tuple(dil_t[i] * (ksp[i] - 1) // 2 for i in range(n))
            post = tuple(dil_t[i] * (ksp[i] - 1) - pre[i] for i in range(n))
            return pre, post
        return None
    p = _ntuple(padding, n)
    if p is None:
        return None
    return p, p


class _ConvFpropOverride(OpOverride):
    """Rank-generic conv forward. Concrete subclasses set :attr:`op_name`
    (``conv2d``/``conv3d``) and :attr:`spatial_rank` (2/3)."""

    #: number of spatial axes (2 for conv2d, 3 for conv3d)
    spatial_rank = None

    def _out_mem_fmt(self, input):
        """Pick the output memory format to mirror the input's layout (native's
        contract): channels-last in → channels-last out, else contiguous."""
        torch = self.state.torch
        cl = torch.channels_last if self.spatial_rank == 2 else torch.channels_last_3d
        if input.is_contiguous(memory_format=cl):
            return cl
        return torch.contiguous_format

    def _gate(self, input, weight):
        # The only two legitimate stops: we need a device pointer to execute
        # against, and a dtype the graph builder can express. Everything else --
        # groups, padding mode, mixed dtype, size, rank -- is built (or, if it
        # cannot be constructed, cleanly declined in _call) and left to hipDNN.
        if not input.is_cuda:
            return False, "input not on cuda"  # execute() needs a device pointer
        if input.dtype not in self.state.dtype_map:
            return False, f"dtype {self._tok(input.dtype)} not graph-mappable"
        return True, ""

    def _graph(
        self,
        x_dims,
        x_strides,
        x_dtype,
        w_dims,
        w_strides,
        w_dtype,
        y_dims,
        y_strides,
        y_dtype,
        stride,
        pre_pad,
        post_pad,
        dilation,
    ):
        st = self.state
        hipdnn = st.hipdnn
        hx = st.dtype_map[x_dtype]
        hw = st.dtype_map[w_dtype]
        hy = st.dtype_map[y_dtype]
        g = hipdnn.Graph()
        g.set_io_data_type(hx)
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        x_t = g.tensor(
            hipdnn.Tensor()
            .set_name("x")
            .set_dim(list(x_dims))
            .set_stride(list(x_strides))
            .set_data_type(hx)
            .set_uid(_X_UID)
        )
        w_t = g.tensor(
            hipdnn.Tensor()
            .set_name("w")
            .set_dim(list(w_dims))
            .set_stride(list(w_strides))
            .set_data_type(hw)
            .set_uid(_W_UID)
        )
        attrs = hipdnn.ConvFpropAttributes()
        # Padding API differs across frontend versions: newer builds bind the
        # asymmetric ``set_pre_padding``/``set_post_padding`` pair (needed for torch
        # ``'same'``), older ones bind only the symmetric ``set_padding(seq)``. Use
        # whichever this build exposes. If only the symmetric API exists but the
        # requested padding is asymmetric, it genuinely cannot be expressed -- decline
        # cleanly to native. Version skew, not a capability pre-judgement.
        if hasattr(attrs, "set_pre_padding"):
            attrs.set_pre_padding(list(pre_pad))
            attrs.set_post_padding(list(post_pad))
        elif list(pre_pad) == list(post_pad):
            attrs.set_padding(list(pre_pad))
        else:
            raise NotApplicable(
                "asymmetric conv padding needs set_pre_padding/set_post_padding, "
                "absent in this frontend build"
            )
        attrs.set_stride(list(stride))
        attrs.set_dilation(list(dilation))

        y_t = g.conv_fprop(x_t, w_t, attrs)
        y_t.set_dim(list(y_dims))
        y_t.set_stride(list(y_strides))
        y_t.set_data_type(hy)
        y_t.set_uid(_Y_UID)
        y_t.set_output(True)  # terminal output must be non-virtual (MIOpen requires it)
        return g

    def _call(
        self, real, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
    ):
        torch = self.state.torch
        sr = self.spatial_rank
        rank = sr + 2

        def _native():
            return real(input, weight, bias, stride, padding, dilation, groups)

        # -- translate params + derive shapes (all guarded: a call we cannot even
        #    describe falls back cleanly, never crashes the model) --------------
        try:
            st_t = _ntuple(stride, sr)
            dil_t = _ntuple(dilation, sr)
            rank_ok = input.dim() == rank and weight.dim() == rank
            ksp = tuple(int(weight.shape[2 + i]) for i in range(sr)) if rank_ok else ()
            k = int(weight.shape[0]) if rank_ok else -1
            cpg = int(weight.shape[1]) if rank_ok else -1  # Cin / groups
            pads = (
                _resolve_pads(padding, dil_t, ksp, sr)
                if (rank_ok and dil_t is not None)
                else None
            )
            ksp_str = "x".join(str(x) for x in ksp) if ksp else "?"
            key = f"Cpg={cpg},K={k},ksp={ksp_str},dtype={self._tok(input.dtype)}"
        except Exception:  # noqa: BLE001 -- unparseable call site -> native
            self.note_native(f"conv{sr}d", "unparseable conv args")
            return _native()

        ok, reason = self._gate(input, weight)
        if not ok:
            self.note_native(key, reason)
            return _native()

        # A wrong-rank input or an unrecognised padding string cannot be turned
        # into conv tensors at all -- a construction failure, not a capability
        # judgement -- so it falls back cleanly.
        if not rank_ok or st_t is None or dil_t is None or pads is None:
            self.note_native(key, "conv args not expressible as a graph -> native")
            return _native()

        pre_t, post_t = pads
        try:
            n = int(input.shape[0])
            cin = int(input.shape[1])
            in_sp = [int(input.shape[2 + i]) for i in range(sr)]
            out_sp = [
                (in_sp[i] + pre_t[i] + post_t[i] - dil_t[i] * (ksp[i] - 1) - 1)
                // st_t[i]
                + 1
                for i in range(sr)
            ]
            # Output mirrors the input's layout; we own it, so its strides are known.
            y = torch.empty(
                (n, k, *out_sp),
                dtype=input.dtype,
                device=input.device,
                memory_format=self._out_mem_fmt(input),
            )

            entry = self._cached_graph(
                (
                    tuple(input.shape),
                    tuple(input.stride()),
                    tuple(weight.shape),
                    tuple(weight.stride()),
                    tuple(y.stride()),
                    input.dtype,
                    weight.dtype,
                    st_t,
                    pre_t,
                    post_t,
                    dil_t,
                ),
                lambda: self._graph(
                    [n, cin, *in_sp],
                    list(input.stride()),
                    input.dtype,
                    [k, cpg, *ksp],
                    list(weight.stride()),
                    weight.dtype,
                    [n, k, *out_sp],
                    list(y.stride()),
                    input.dtype,
                    st_t,
                    pre_t,
                    post_t,
                    dil_t,
                ),
                f"[{n},{cin},{in_sp}]*[{k},{cpg},{list(ksp)}] {input.dtype}",
            )
            self._execute(
                entry,
                {
                    _X_UID: input.data_ptr(),
                    _W_UID: weight.data_ptr(),
                    _Y_UID: y.data_ptr(),
                },
                input.device,
            )
            extras = {}
            if bias is not None:
                # native epilogue; reshape to [1,K,1,...] to broadcast over spatial.
                y = y + bias.reshape(1, k, *([1] * sr))
                extras["biased"] = 1
            self.note_aot(key, **extras)
            return y
        except NotApplicable as na:  # hipDNN could not serve this shape -> native
            self.note_native(key, str(na))
            return _native()
        except (
            Exception
        ) as ex:  # noqa: BLE001 -- any failure -> native, never break the model
            self.note_native(
                key, f"exception: {type(ex).__name__}: {ex}", level=logging.WARNING
            )
            return _native()


class Conv2dFpropOverride(_ConvFpropOverride):
    op_name = "conv2d"
    spatial_rank = 2


class Conv3dFpropOverride(_ConvFpropOverride):
    op_name = "conv3d"
    spatial_rank = 3
