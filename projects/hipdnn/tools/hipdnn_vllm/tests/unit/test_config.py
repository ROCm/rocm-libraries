# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import os

import pytest

from hipdnn_vllm.config import ALL_OPS, DEFAULT_OPS, ConfigError, load_config


def test_defaults():
    c = load_config({})
    assert not c.disable
    assert c.ops == frozenset(DEFAULT_OPS)
    assert not c.strict and not c.attn_debug_route
    assert c.report_dir is None
    assert c.report_interval == 60.0
    assert c.plan_cache_size == 4096
    assert c.plugin_mode == "auto"
    assert c.provider_dirs == ()
    assert "linear" not in c.ops


def test_full_env(tmp_path):
    c = load_config(
        {
            "HIPDNN_VLLM_DISABLE": "1",
            "HIPDNN_VLLM_OPS": " attention , linear ,linear_all",
            "HIPDNN_VLLM_ATTN_DEBUG_ROUTE": "true",
            "HIPDNN_VLLM_STRICT": "yes",
            "HIPDNN_VLLM_REPORT_DIR": str(tmp_path),
            "HIPDNN_VLLM_REPORT_INTERVAL": "0.5",
            "HIPDNN_VLLM_PLAN_CACHE_SIZE": "8",
            "HIPDNN_VLLM_PLUGIN_MODE": "ABSOLUTE",
            "HIPDNN_VLLM_PROVIDER_DIRS": f"/a/b{os.pathsep}/c,/d",
        }
    )
    assert c.disable and c.strict and c.attn_debug_route
    assert c.ops == frozenset({"attention", "linear", "linear_all"})
    assert c.report_dir == str(tmp_path)
    assert c.report_interval == 0.5
    assert c.plan_cache_size == 8
    assert c.plugin_mode == "absolute"
    assert c.provider_dirs == ("/a/b", "/c", "/d")


def test_disable_turns_every_op_off():
    c = load_config({"HIPDNN_VLLM_DISABLE": "1"})
    assert not any(c.op_enabled(op) for op in ALL_OPS)
    assert load_config({}).op_enabled("attention")


def test_empty_ops_list_enables_nothing():
    c = load_config({"HIPDNN_VLLM_OPS": ""})
    assert c.ops == frozenset()
    assert not c.op_enabled("rms_norm")


@pytest.mark.parametrize(
    "env",
    [
        {"HIPDNN_VLLM_DISABLE": "maybe"},
        {"HIPDNN_VLLM_OPS": "attention,softmax"},
        {"HIPDNN_VLLM_PLAN_CACHE_SIZE": "0"},
        {"HIPDNN_VLLM_PLAN_CACHE_SIZE": "lots"},
        {"HIPDNN_VLLM_REPORT_INTERVAL": "-1"},
        {"HIPDNN_VLLM_PLUGIN_MODE": "replace"},
    ],
)
def test_invalid_values_raise(env):
    with pytest.raises(ConfigError):
        load_config(env)


def test_unknown_op_query_raises():
    with pytest.raises(ConfigError):
        load_config({}).op_enabled("softmax")
