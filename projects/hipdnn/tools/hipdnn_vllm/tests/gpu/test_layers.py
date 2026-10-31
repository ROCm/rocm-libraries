# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The vLLM LayerNorm / Conv2dLayer / Conv3dLayer OOT overrides inside the
pinned vLLM: construction picks the override, forward goes to
hipDNN (or to the parent's shortcut), and results match the parent."""

import os
import subprocess
import sys

import pytest
import torch

pytest.importorskip("vllm")

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.custom_op import op_registry_oot  # noqa: E402
from vllm.model_executor.layers.conv import Conv2dLayer, Conv3dLayer  # noqa: E402
from vllm.model_executor.layers.layernorm import LayerNorm  # noqa: E402

from hipdnn_vllm import layers, stats  # noqa: E402
from hipdnn_vllm.config import load_config  # noqa: E402


@pytest.fixture(scope="module")
def registered(hipdnn_state):
    """Register with this process's HIPDNN_VLLM_OPS (strict). LayerNorm is a
    default op; the conv overrides register only when conv is enabled."""
    config = load_config({**os.environ, "HIPDNN_VLLM_STRICT": "1"})
    expected = {"LayerNorm"} if config.op_enabled("layer_norm") else set()
    if config.op_enabled("conv"):
        expected |= {"Conv2dLayer", "Conv3dLayer"}
    previous = layers._config
    names = layers.register_layers(config)
    assert expected <= set(op_registry_oot)
    assert set(names) <= expected
    yield config
    layers._config = previous
    for n in ("LayerNorm", "Conv2dLayer", "Conv3dLayer"):
        cls = op_registry_oot.get(n)
        if cls is not None and cls.__module__ == layers.__name__:
            del op_registry_oot[n]


@pytest.fixture
def layer_norm_registered(registered):
    if not registered.op_enabled("layer_norm"):
        pytest.skip("layer_norm is not in HIPDNN_VLLM_OPS for this process")


def test_default_ops_register_layer_norm_only():
    """Conv is opt-in: the default HIPDNN_VLLM_OPS overrides LayerNorm, not the conv layers."""
    code = (
        "from hipdnn_vllm import layers\n"
        "from hipdnn_vllm.config import load_config\n"
        "print(sorted(layers.register_layers(load_config({}))))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "['LayerNorm']", out.stdout + out.stderr


def _vllm_config(custom_ops):
    # "none" is vLLM's default base mode under Inductor.
    return set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=custom_ops)))


def _routes(op):
    return {
        (r["route"], r["reason"]): r["count"]
        for r in stats.snapshot()["counters"]
        if r["op"] == op
    }


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_layer_norm_override(layer_norm_registered, bias):
    with _vllm_config(["none"]):
        layer = LayerNorm(1152, eps=1e-6, bias=bias, dtype=torch.bfloat16).cuda()
    assert type(layer).__name__ == "HipdnnLayerNorm"
    assert layer._forward_method == layer.forward_hip
    with torch.no_grad():
        layer.weight.normal_()
        if bias:
            layer.bias.normal_()
    x = torch.randn(300, 1152, dtype=torch.bfloat16, device="cuda")
    stats.reset()
    y = layer(x)
    assert sum(c for (route, _), c in _routes("layer_norm").items() if route == "hipdnn") >= 1
    assert not any(route == "fallback" for route, _ in _routes("layer_norm"))
    torch.testing.assert_close(y, layer.forward_native(x), atol=2e-2, rtol=2e-2)


def test_custom_ops_minus_disables_override(layer_norm_registered):
    for spelling in ("-layer_norm", "-LayerNorm"):
        with _vllm_config(["none", spelling]):
            layer = LayerNorm(64)
        assert layer._forward_method != layer.forward_hip


@pytest.mark.conv
def test_conv2d_override(registered):
    with _vllm_config(["none"]):
        layer = Conv2dLayer(3, 768, kernel_size=16, stride=16, params_dtype=torch.float16).cuda()
    assert type(layer).__name__ == "HipdnnConv2dLayer" and layer._hipdnn
    with torch.no_grad():
        layer.weight.normal_(std=0.05)
        layer.bias.normal_()
    x = torch.randn(2, 3, 224, 224, dtype=torch.float16, device="cuda")
    stats.reset()
    y = layer(x)
    assert _routes("conv_fwd") in ({("hipdnn", ""): 1}, {("hipdnn", "PARTIAL"): 1})
    torch.testing.assert_close(y, layer._forward_conv(x), atol=1e-2, rtol=5e-3)


@pytest.mark.conv
def test_conv3d_linear_shortcut_left_to_parent(registered):
    with _vllm_config(["none"]):
        shortcut = Conv3dLayer(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False,
                               params_dtype=torch.bfloat16).cuda()
        plain = Conv3dLayer(3, 64, kernel_size=3, stride=1, padding=1,
                            params_dtype=torch.bfloat16).cuda()
    assert type(shortcut).__name__ == "HipdnnConv3dLayer"
    assert shortcut.enable_linear and not shortcut._hipdnn
    assert plain._hipdnn
    x = torch.randn(16, 3, 2, 14, 14, dtype=torch.bfloat16, device="cuda")
    stats.reset()
    shortcut(x)
    assert _routes("conv_fwd") == {}
    with torch.no_grad():
        plain.weight.normal_(std=0.05)
        plain.bias.normal_()
    v = torch.randn(1, 3, 4, 16, 16, dtype=torch.bfloat16, device="cuda")
    y = plain(v)
    assert _routes("conv_fwd") in ({("hipdnn", ""): 1}, {("hipdnn", "PARTIAL"): 1})
    torch.testing.assert_close(y, plain._forward_conv(v), atol=5e-2, rtol=2e-2)
