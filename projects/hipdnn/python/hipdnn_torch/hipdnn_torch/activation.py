# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.activation -- route ``F.silu`` and ``F.gelu`` onto the hipDNN engine's
elementwise activation kernels.

Both are unary pointwise ops; the tensor is flattened to a single contiguous run of
``numel`` elements (the kernel walks a flat buffer), so the graph carries just
``A[numel] -> C[numel]`` with the activation mode on a ``PointwiseAttributes``:

  * ``F.silu`` -> ``PointwiseMode.SWISH_FWD`` (SiLU is Swish with beta == 1).
  * ``F.gelu(approximate="tanh")`` -> ``PointwiseMode.GELU_APPROX_TANH_FWD``.
    The default exact-erf ``F.gelu`` has no rocKE builder op yet, so it falls back
    to native and is logged (reason ``erf-gelu unsupported``).

Constraint: cuda f16/bf16 only; anything else falls back to native and is logged.
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
        torch = self.state.torch
        if not input.is_cuda:
            return False, "input not on cuda"
        if input.dtype not in (torch.float16, torch.bfloat16):
            return False, f"dtype {self._tok(input.dtype)} (need f16/bf16)"
        return True, ""

    def _graph(self, numel, mode, dtype):
        st = self.state
        hipdnn = st.hipdnn
        hf = st.dtype_map[dtype]
        g = hipdnn.Graph()
        g.set_io_data_type(hf)
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        a_t = g.tensor(
            hipdnn.Tensor()
            .set_name("A")
            .set_dim([numel])
            .set_stride([1])
            .set_data_type(hf)
            .set_uid(self._A_UID)
        )
        attrs = hipdnn.PointwiseAttributes()
        attrs.set_mode(mode)
        c_t = g.pointwise(a_t, attrs)
        c_t.set_dim([numel])
        c_t.set_stride([1])
        c_t.set_data_type(hf)
        c_t.set_uid(self._C_UID)
        return g

    def _run(self, real, input, native, **mode_kwargs):
        torch = self.state.torch
        dtype = input.dtype
        numel = int(input.numel())
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
            x1d = input.reshape(-1).contiguous()
            entry = self._cached_graph(
                (numel, int(mode), dtype),
                lambda: self._graph(numel, mode, dtype),
                f"[{numel}] {dtype}",
            )
            y = torch.empty_like(x1d)
            self._execute(
                entry,
                {self._A_UID: x1d.data_ptr(), self._C_UID: y.data_ptr()},
                input.device,
            )
            self.note_aot(key)
            return y.reshape(input.shape)
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
        if approximate == "tanh":
            return self.state.hipdnn.PointwiseMode.GELU_APPROX_TANH_FWD
        # Exact erf GELU has no rocKE builder op yet -> native fallback.
        return "erf-gelu unsupported"

    def _call(self, real, input, approximate="none"):
        return self._run(
            real,
            input,
            lambda: real(input, approximate=approximate),
            approximate=approximate,
        )
