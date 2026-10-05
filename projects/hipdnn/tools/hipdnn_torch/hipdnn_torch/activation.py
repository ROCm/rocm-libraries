# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.activation -- route ``F.silu`` and ``F.gelu`` onto hipDNN's elementwise
activation graph.

**Pure passthrough.** Both are unary pointwise ops. The override builds the graph
from the input's *actual* dims + strides -- whatever layout it arrives in, no
flatten, no ``.contiguous()`` -- so the graph carries ``A[dims] -> C[dims]`` with the
activation mode on a ``PointwiseAttributes``:

  * ``F.silu`` -> ``PointwiseMode.SWISH_FWD`` (SiLU is Swish with beta == 1).
  * ``F.gelu(approximate="tanh")`` -> ``PointwiseMode.GELU_APPROX_TANH_FWD``.
  * ``F.gelu()`` (exact erf) -> ``PointwiseMode.GELU_FWD``.

Both GELU flavours are built and submitted; hipDNN decides whether a loaded engine
serves the mode (an engine with no builder for it declines -> native). The output is
``torch.empty_like(input)`` (preserving layout). The only two stops are a device
pointer to execute against and a graph-mappable dtype; anything else falls back to
native and is counted.
"""

import logging

from .base import NotApplicable, OpOverride


class _ActivationOverride(OpOverride):
    """Shared machinery for the unary-pointwise overrides. Subclasses set
    :attr:`op_name` and implement :meth:`_mode` (returns the ``PointwiseMode`` or a
    fallback reason string)."""

    #: uid layout for the single-input pointwise graph.
    _A_UID, _C_UID = 1, 2

    def _mode(self, **kwargs):
        raise NotImplementedError

    def _gate(self, input):
        # Structural preconditions only: a device pointer to execute against and a
        # dtype the graph builder can map. No shape/size gating -- build and submit.
        if not input.is_cuda:
            return False, "input not on cuda"
        if input.dtype not in self.state.dtype_map:
            return False, f"dtype {self._tok(input.dtype)} not graph-mappable"
        return True, ""

    def _graph(self, dims, strides, mode, dtype):
        st = self.state
        hipdnn = st.hipdnn
        hf = st.dtype_map[dtype]
        g = hipdnn.Graph()
        g.set_io_data_type(hf)
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        a_t = g.tensor(
            hipdnn.Tensor()
            .set_name("A")
            .set_dim(list(dims))
            .set_stride(list(strides))
            .set_data_type(hf)
            .set_uid(self._A_UID)
        )
        attrs = hipdnn.PointwiseAttributes()
        attrs.set_mode(mode)
        c_t = g.pointwise(a_t, attrs)
        c_t.set_dim(list(dims))
        c_t.set_stride(list(strides))
        c_t.set_data_type(hf)
        c_t.set_uid(self._C_UID)
        c_t.set_output(True)  # terminal output must be non-virtual (MIOpen requires it)
        return g

    def _run(self, real, input, native, **mode_kwargs):
        torch = self.state.torch
        dtype = input.dtype
        try:
            numel = int(input.numel())
        except Exception:  # noqa: BLE001
            return native()
        key = f"numel={numel},dtype={self._tok(dtype)}"

        ok, reason = self._gate(input)
        if not ok:
            self.note_native(key, reason)
            return native()

        mode = self._mode(**mode_kwargs)
        if isinstance(mode, str):  # unsupported variant -> native, with a reason
            self.note_native(key, mode)
            return native()

        try:
            # Build at the input's actual dims/strides; output mirrors its layout.
            y = torch.empty_like(input)
            entry = self._cached_graph(
                # str(mode), not int(mode): the bound PointwiseMode enum isn't
                # int-convertible, and its repr is a stable per-value key.
                (tuple(input.shape), tuple(input.stride()), str(mode), dtype),
                lambda: self._graph(
                    list(input.shape), list(input.stride()), mode, dtype
                ),
                f"{list(input.shape)} {dtype}",
            )
            self._execute(
                entry,
                {self._A_UID: input.data_ptr(), self._C_UID: y.data_ptr()},
                input.device,
            )
            self.note_aot(key)
            return y
        except NotApplicable as na:  # engine can't serve this shape -> native
            self.note_native(key, str(na))
            return native()
        except (
            Exception
        ) as ex:  # noqa: BLE001 -- any failure -> native, never break the model
            self.note_native(
                key, f"exception: {type(ex).__name__}: {ex}", level=logging.WARNING
            )
            return native()


class SiluOverride(_ActivationOverride):
    op_name = "silu"

    def _mode(self, **kwargs):
        return self.state.hipdnn.PointwiseMode.SWISH_FWD

    def _call(self, real, input, inplace=False):
        return self._run(real, input, lambda: real(input, inplace=inplace))


class GeluOverride(_ActivationOverride):
    op_name = "gelu"

    def _mode(self, approximate="none"):
        pm = self.state.hipdnn.PointwiseMode
        # Both flavours are representable; hipDNN decides if an engine serves them.
        if approximate == "tanh":
            return pm.GELU_APPROX_TANH_FWD
        return pm.GELU_FWD  # exact erf GELU

    def _call(self, real, input, approximate="none"):
        return self._run(
            real,
            input,
            lambda: real(input, approximate=approximate),
            approximate=approximate,
        )
