# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.conv -- route ``F.conv2d``/``F.conv3d`` (every ``nn.Conv2d`` /
``nn.Conv3d``) onto a hipDNN forward-convolution engine.

The graph is addressed channels-last on every operand: input NHWC/NDHWC, weight
KRSC/KTRSC, output NHWK/NDHWK. torch defaults to contiguous NCHW/NCDHW, so the
override converts input/weight to ``channels_last`` (2-D) or ``channels_last_3d``
(3-D) and allocates a matching output; the graph carries the canonical NCHW(D)
logical dims with those channels-last strides. Symmetric padding is required (the
runtime ABI carries one pad per spatial axis). Bias is added natively after the
conv (the kernel ABI has no epilogue), so a bias is just counted.

Shape is not pre-filtered beyond genuine correctness constraints (cuda f16/bf16,
correct rank, ``groups==1``, integer symmetric hyperparameters): the override
builds the graph and lets hipDNN rank the loaded engines. rocKE's WMMA
implicit-GEMM conv serves 2-D only, while MIOpen serves both 2-D and 3-D -- so a
conv3d graph routes to MIOpen under pick-best, and only a shape no loaded engine
claims falls back to native.
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


class _ConvFpropOverride(OpOverride):
    """Rank-generic conv forward. Concrete subclasses set :attr:`op_name`
    (``conv2d``/``conv3d``) and :attr:`spatial_rank` (2/3)."""

    #: number of spatial axes (2 for conv2d, 3 for conv3d)
    spatial_rank = None

    def _mem_fmt(self):
        torch = self.state.torch
        return torch.channels_last if self.spatial_rank == 2 else torch.channels_last_3d

    def _gate(self, input, weight, groups, stride, padding, dilation):
        # Structural facts (needed to build the graph) + features the conv graph
        # cannot represent. No shape/size gating: build and let hipDNN decide.
        rank = self.spatial_rank + 2
        if not input.is_cuda:
            return False, "input not on cuda"  # execute() needs a device pointer
        if input.dtype not in self.state.dtype_map:
            return False, f"dtype {self._tok(input.dtype)} not graph-mappable"
        if input.dim() != rank or weight.dim() != rank:
            return False, f"input/weight not rank-{rank}"  # builder unpacks NCHW(D)
        # --- not representable in the conv-fprop graph -> catch here ---
        if weight.dtype != input.dtype:
            # The graph declares one io dtype and execute() passes raw pointers, so a
            # differing weight dtype would be reinterpreted byte-for-byte. A per-tensor
            # mixed-dtype conv is a builder extension, not something hipDNN can rescue.
            return (
                False,
                f"weight dtype {self._tok(weight.dtype)} != input {self._tok(input.dtype)}",
            )
        if groups != 1:
            # ConvFpropAttributes has no group setter -> grouped/depthwise conv
            # cannot be expressed in the graph.
            return False, f"groups={groups} not representable (no group attr)"
        if stride is None or padding is None or dilation is None:
            # 'same'/'valid' string padding has no symmetric-int ABI representation.
            return False, "non-integer stride/padding/dilation (e.g. 'same'/'valid')"
        return True, ""

    def _graph(
        self,
        x_dims,
        x_strides,
        w_dims,
        w_strides,
        y_dims,
        y_strides,
        stride,
        padding,
        dilation,
        dtype,
    ):
        st = self.state
        hipdnn = st.hipdnn
        hf = st.dtype_map[dtype]
        g = hipdnn.Graph()
        g.set_io_data_type(hf)
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        x_t = g.tensor(
            hipdnn.Tensor()
            .set_name("x")
            .set_dim(x_dims)
            .set_stride(x_strides)
            .set_data_type(hf)
            .set_uid(_X_UID)
        )
        w_t = g.tensor(
            hipdnn.Tensor()
            .set_name("w")
            .set_dim(w_dims)
            .set_stride(w_strides)
            .set_data_type(hf)
            .set_uid(_W_UID)
        )
        attrs = hipdnn.ConvFpropAttributes()
        attrs.set_padding(list(padding))
        attrs.set_stride(list(stride))
        attrs.set_dilation(list(dilation))

        y_t = g.conv_fprop(x_t, w_t, attrs)
        y_t.set_dim(y_dims)
        y_t.set_stride(y_strides)
        y_t.set_data_type(hf)
        y_t.set_uid(_Y_UID)
        y_t.set_output(True)  # terminal output must be non-virtual (MIOpen requires it)
        return g

    def _call(
        self, real, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
    ):
        torch = self.state.torch
        sr = self.spatial_rank

        def _native():
            return real(input, weight, bias, stride, padding, dilation, groups)

        st_t = _ntuple(stride, sr)
        pad_t = _ntuple(padding, sr)
        dil_t = _ntuple(dilation, sr)

        rank_ok = weight.dim() == sr + 2
        c = int(weight.shape[1]) if rank_ok else -1
        k = int(weight.shape[0]) if rank_ok else -1
        ksp = tuple(int(weight.shape[2 + i]) for i in range(sr)) if rank_ok else ()
        ksp_str = "x".join(str(x) for x in ksp) if ksp else "?"
        key = f"C={c},K={k},ksp={ksp_str},dtype={self._tok(input.dtype)}"

        ok, reason = self._gate(input, weight, groups, st_t, pad_t, dil_t)
        if not ok:
            self.note_native(key, reason)
            return _native()

        mem_fmt = self._mem_fmt()
        try:
            # channels-last packed on all three operands: canonical NCHW(D) logical
            # dims, channels-last strides -- exactly what the adapters address.
            x = input.contiguous(memory_format=mem_fmt)
            w = weight.contiguous(memory_format=mem_fmt)
            n = int(x.shape[0])
            cin = int(x.shape[1])
            in_sp = [int(x.shape[2 + i]) for i in range(sr)]
            out_sp = [
                (in_sp[i] + 2 * pad_t[i] - dil_t[i] * (ksp[i] - 1) - 1) // st_t[i] + 1
                for i in range(sr)
            ]
            y = torch.empty(
                (n, k, *out_sp),
                dtype=input.dtype,
                device=input.device,
                memory_format=mem_fmt,
            )

            entry = self._cached_graph(
                (n, cin, tuple(in_sp), k, ksp, st_t, pad_t, dil_t, input.dtype),
                lambda: self._graph(
                    [n, cin, *in_sp],
                    list(x.stride()),
                    [k, cin, *ksp],
                    list(w.stride()),
                    [n, k, *out_sp],
                    list(y.stride()),
                    st_t,
                    pad_t,
                    dil_t,
                    input.dtype,
                ),
                f"[{n},{cin},{in_sp}]*[{k},{cin},{list(ksp)}] {input.dtype}",
            )
            self._execute(
                entry,
                {_X_UID: x.data_ptr(), _W_UID: w.data_ptr(), _Y_UID: y.data_ptr()},
                input.device,
            )
            extras = {}
            if bias is not None:
                # native epilogue; kernel ABI has no bias. reshape to [1,K,1,...].
                y = y + bias.reshape(1, k, *([1] * sr))
                extras["biased"] = 1
            self.note_aot(key, **extras)
            return y
        except NotApplicable as na:  # engine can't serve this shape -> native
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
