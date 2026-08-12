# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.conv -- route ``F.conv2d`` (every ``nn.Conv2d``) onto the hipDNN
engine's WMMA implicit-GEMM forward convolution.

The runtime kernel addresses input as NHWC, weight as KRSC, output as NHWK -- i.e.
channels-last packed on every operand. torch defaults to NCHW-contiguous, so the
override converts input/weight to ``channels_last`` and allocates a ``channels_last``
output; the graph carries the canonical NCHW logical dims with the NHWC/KRSC/NHWK
strides the adapter's ``isPackedChannelsLast`` check expects. Symmetric padding is
required (the runtime ABI carries one pad per axis). Bias is added natively after
the conv (the kernel ABI has no epilogue), so a bias is just counted.

Constraint: cuda f16/bf16, rank-4, ``groups==1``, symmetric integer padding; anything
else falls back to native and is logged.
"""

import logging

from .base import NotApplicable, OpOverride

_X_UID, _W_UID, _Y_UID = 1, 2, 3


def _pair(v):
    """Normalise an int-or-2-tuple hyperparameter to a ``(h, w)`` int tuple, or
    ``None`` if it is neither (e.g. a ``'same'``/``'valid'`` padding string)."""
    if isinstance(v, int):
        return (v, v)
    try:
        t = tuple(int(x) for x in v)
    except (TypeError, ValueError):
        return None
    if len(t) == 1:
        return (t[0], t[0])
    if len(t) == 2:
        return t
    return None


class Conv2dFpropOverride(OpOverride):
    op_name = "conv2d"

    def _gate(self, input, weight, groups, stride, padding, dilation):
        torch = self.state.torch
        if not input.is_cuda:
            return False, "input not on cuda"
        if input.dtype not in (torch.float16, torch.bfloat16):
            return False, f"dtype {self._tok(input.dtype)} (need f16/bf16)"
        if weight.dtype != input.dtype:
            # The graph declares the weight tensor as the input dtype and
            # execute() passes its raw pointer, so a differing weight dtype would
            # be reinterpreted byte-for-byte -> silently wrong. Decline instead.
            return (
                False,
                f"weight dtype {self._tok(weight.dtype)} != input {self._tok(input.dtype)}",
            )
        if input.dim() != 4 or weight.dim() != 4:
            return False, "input/weight not rank-4"
        if groups != 1:
            return False, f"groups={groups} (only groups==1)"
        if stride is None or padding is None or dilation is None:
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
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
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
        attrs.set_padding([ph, pw])
        attrs.set_stride([sh, sw])
        attrs.set_dilation([dh, dw])

        y_t = g.conv_fprop(x_t, w_t, attrs)
        y_t.set_dim(y_dims)
        y_t.set_stride(y_strides)
        y_t.set_data_type(hf)
        y_t.set_uid(_Y_UID)
        return g

    def _call(
        self, real, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
    ):
        torch = self.state.torch

        def _native():
            return real(input, weight, bias, stride, padding, dilation, groups)

        st_hw = _pair(stride)
        pad_hw = _pair(padding)
        dil_hw = _pair(dilation)

        c = int(weight.shape[1]) if weight.dim() == 4 else -1
        k = int(weight.shape[0]) if weight.dim() == 4 else -1
        r = int(weight.shape[2]) if weight.dim() == 4 else -1
        s = int(weight.shape[3]) if weight.dim() == 4 else -1
        key = f"C={c},K={k},R={r},S={s},dtype={self._tok(input.dtype)}"

        ok, reason = self._gate(input, weight, groups, st_hw, pad_hw, dil_hw)
        if not ok:
            self.note_native(key, reason)
            return _native()

        sh, sw = st_hw
        ph, pw = pad_hw
        dh, dw = dil_hw
        try:
            # channels-last packed on all three operands: logical NCHW dims,
            # NHWC/KRSC/NHWK strides -- exactly what the adapter addresses.
            x = input.contiguous(memory_format=torch.channels_last)
            w = weight.contiguous(memory_format=torch.channels_last)
            n, cin, hin, win = (int(d) for d in x.shape)
            ho = (hin + 2 * ph - dh * (r - 1) - 1) // sh + 1
            wo = (win + 2 * pw - dw * (s - 1) - 1) // sw + 1
            y = torch.empty(
                (n, k, ho, wo),
                dtype=input.dtype,
                device=input.device,
                memory_format=torch.channels_last,
            )

            entry = self._cached_graph(
                (n, cin, hin, win, k, r, s, sh, sw, ph, pw, dh, dw, input.dtype),
                lambda: self._graph(
                    [n, cin, hin, win],
                    list(x.stride()),
                    [k, cin, r, s],
                    list(w.stride()),
                    [n, k, ho, wo],
                    list(y.stride()),
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                    input.dtype,
                ),
                f"[{n},{cin},{hin},{win}]*[{k},{cin},{r},{s}] {input.dtype}",
            )
            self._execute(
                entry,
                {_X_UID: x.data_ptr(), _W_UID: w.data_ptr(), _Y_UID: y.data_ptr()},
                input.device,
            )
            extras = {}
            if bias is not None:
                y = y + bias.reshape(
                    1, k, 1, 1
                )  # native epilogue; kernel ABI has no bias
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
