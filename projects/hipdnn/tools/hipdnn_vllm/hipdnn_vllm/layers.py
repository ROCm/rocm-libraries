# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Out-of-tree ``CustomOp`` overrides for vLLM layers (LayerNorm, Conv2d/Conv3d).

* ``HipdnnLayerNorm``   → ``torch.ops.hipdnn_vllm.layer_norm``
* ``HipdnnConv2dLayer`` / ``HipdnnConv3dLayer`` → ``torch.ops.hipdnn_vllm.conv_fwd``

The ops decide per call whether hipDNN serves the signature; on a decline
they count ``route=fallback`` and return the parent's math (``F.layer_norm``
in the promoted dtype / ``F.conv*d``), or raise in strict mode. Keeping that
decision inside the op makes ``forward_hip`` a single opaque op call, which is
what torch.compile (fullgraph) and HIP-graph capture need.

Enablement. ``CustomOp.register_oot`` renames the override to the parent's
*class* name (``LayerNorm``, ``Conv2dLayer``), so vLLM's ``custom_ops`` switch
would look for ``+LayerNorm`` instead of ``+layer_norm``, and under Inductor's
``custom_ops=["none"]`` default the override would never run. ``enabled()`` is
therefore overridden: an op listed in ``HIPDNN_VLLM_OPS`` is on unless
``custom_ops`` disables it (``-layer_norm`` or ``-LayerNorm``); an op not
listed keeps vLLM's own rule under either name, and ``forward_hip`` defers to
the parent.

Conv routing. Only the parent's true convolution path is routed.
``Conv3dLayer.forward_cuda`` (which ``forward_hip`` inherits) takes an
unfold + linear shortcut when kernel == stride, padding is 0 and groups is 1
on torch >= 2.9; that call is left to the parent.
``Conv2dLayer.forward_cuda`` always convolves.
"""

from __future__ import annotations

from . import _bootstrap
from .config import Config

__all__ = ["register_layers"]

# hipdnn_vllm op name -> names vLLM's custom_ops may use for the layer.
_OP_NAMES = {
    "layer_norm": ("layer_norm", "LayerNorm"),
    "conv2d": ("conv2d", "Conv2dLayer"),
    "conv3d": ("conv3d", "Conv3dLayer"),
}

_config: Config | None = None


def _plugin_op(op: str) -> str:
    return "conv" if op.startswith("conv") else op


def _enabled(op: str) -> bool:
    from vllm.config import get_cached_compilation_config
    from vllm.model_executor.custom_op import CustomOp

    custom_ops = get_cached_compilation_config().custom_ops
    names = _OP_NAMES[op]
    if any(f"-{n}" in custom_ops for n in names):
        return False
    if _config is not None and _config.op_enabled(_plugin_op(op)):
        return True
    return CustomOp.default_on() or any(f"+{n}" in custom_ops for n in names)


def _routed(op: str) -> bool:
    """Whether calls go to hipDNN (else the parent's forward_hip). Loads ``_C``
    at layer construction so traced forwards only see the registered op."""
    if _config is None or not _config.op_enabled(_plugin_op(op)):
        return False
    _bootstrap.load(_config)
    return True


def _strict() -> bool:
    return bool(_config is not None and _config.strict)


def _make_classes():
    import torch

    from vllm.model_executor.layers.conv import Conv2dLayer, Conv3dLayer
    from vllm.model_executor.layers.layernorm import LayerNorm
    from vllm.utils.torch_utils import is_torch_equal_or_newer

    conv3d_shortcut = is_torch_equal_or_newer("2.9.0")

    class HipdnnLayerNorm(LayerNorm):
        @classmethod
        def enabled(cls) -> bool:
            return _enabled("layer_norm")

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._hipdnn = _routed("layer_norm")
            self._hipdnn_strict = _strict()

        def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
            if not self._hipdnn:
                return super().forward_hip(x)
            return torch.ops.hipdnn_vllm.layer_norm(
                x, self.weight, self.bias, self.eps, self._hipdnn_strict
            )

    def _conv(layer, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.hipdnn_vllm.conv_fwd(
            x,
            layer.weight,
            layer.bias,
            list(layer.stride),
            list(layer.padding),
            list(layer.dilation),
            layer.groups,
            layer._hipdnn_strict,
        )

    class HipdnnConv2dLayer(Conv2dLayer):
        @classmethod
        def enabled(cls) -> bool:
            return _enabled("conv2d")

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._hipdnn = _routed("conv2d")
            self._hipdnn_strict = _strict()

        def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
            if not self._hipdnn:
                return super().forward_hip(x)
            return _conv(self, x)

    class HipdnnConv3dLayer(Conv3dLayer):
        @classmethod
        def enabled(cls) -> bool:
            return _enabled("conv3d")

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            # The parent's unfold + linear shortcut is not a convolution call.
            self._hipdnn = _routed("conv3d") and not (self.enable_linear and conv3d_shortcut)
            self._hipdnn_strict = _strict()

        def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
            if not self._hipdnn:
                return super().forward_hip(x)
            return _conv(self, x)

    return (
        (LayerNorm, HipdnnLayerNorm),
        (Conv2dLayer, HipdnnConv2dLayer),
        (Conv3dLayer, HipdnnConv3dLayer),
    )


def register_layers(config: Config) -> list[str]:
    """Register the OOT overrides for every enabled op. Returns the parent
    class names registered. Called once from :func:`hipdnn_vllm.register`."""
    global _config
    _config = config
    if not (config.op_enabled("layer_norm") or config.op_enabled("conv")):
        return []
    from vllm.model_executor.custom_op import op_registry_oot

    names = []
    for parent, cls in _make_classes():
        op = "layer_norm" if parent.__name__ == "LayerNorm" else "conv"
        if not config.op_enabled(op):
            continue
        existing = op_registry_oot.get(parent.__name__)
        if existing is not None:
            if existing.__module__ == __name__:
                continue  # this module registered it earlier in the process
            raise RuntimeError(
                f"hipdnn_vllm: vLLM layer {parent.__name__} already has an OOT override "
                f"({existing.__module__}.{existing.__qualname__})"
            )
        parent.register_oot(cls)
        names.append(parent.__name__)
    return names
