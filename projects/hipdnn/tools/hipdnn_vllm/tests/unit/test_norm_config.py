# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""HIPDNN_VLLM_NORM_EXACT_ROWS parsing (exact-row rms_norm plans)."""

import pytest

from hipdnn_vllm.config import ConfigError, load_config


@pytest.mark.parametrize(
    "raw, want",
    [
        (None, None),
        ("capture", None),
        ("CAPTURE", None),
        ("", None),
        ("none", ()),
        ("24,1, 8,24", (1, 8, 24)),
    ],
)
def test_norm_exact_rows(raw, want):
    env = {} if raw is None else {"HIPDNN_VLLM_NORM_EXACT_ROWS": raw}
    assert load_config(env).norm_exact_rows == want


@pytest.mark.parametrize("raw", ["0", "8,-1", "eight", "8;16"])
def test_norm_exact_rows_invalid(raw):
    with pytest.raises(ConfigError):
        load_config({"HIPDNN_VLLM_NORM_EXACT_ROWS": raw})
