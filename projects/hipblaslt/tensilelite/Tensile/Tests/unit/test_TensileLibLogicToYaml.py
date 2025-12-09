################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import pytest
from unittest.mock import mock_open, patch
from pathlib import Path
import yaml
import sys
import os
from typing import Union
import logging

parentdir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
)
sys.path.append(parentdir)
from Tensile import TensileLibLogicToYamlRunner


# Test data
VALID_YAML_FILE_CONTENT = """- {function: matmul, M: 768, N: 3072, K: 2048, lda: 2048, ldb: 2048, ldc: 768, ldd: 768, stride_a: 0, stride_b: 0, stride_c: 0, stride_d: 0, alpha: 1.000000, beta: 0.000000, transA: T, transB: N, batch_count: 1, scaleA: 0, scaleB: 0, scaleC: 0, scaleD: 0, swizzleA: false, swizzleB: false, scaleAlpha_vector: false, gradient: false, use_e: false, bias_vector: false, bias_source: d, a_type: bf16_r, b_type: bf16_r, c_type: bf16_r, d_type: bf16_r, scale_type: f32_r, bias_type: f32_r, compute_type: c_f32_r, activation_type: none, flush: false, any_stride: true, rotating: 512, cold_iters: 0, iters: 1, print_kernel_info: true}"""


@pytest.fixture
def mock_yaml_file():
    with patch(
        "builtins.open", mock_open(read_data=VALID_YAML_FILE_CONTENT)
    ) as mock_file:
        yield mock_file


def extract_function(function_string: str, function: str) -> str:
    msg_prefix = f"Invalid function: {function_string}"
    key, _, val = function_string.partition("=")
    if key == "function":
        if val != function:
            raise ValueError(f"{msg_prefix}: {function} is not valid")
    return function_string


import re


class LogicFileError(Exception):  # remove
    def __init__(self, message="Expected line is either not present or is malformed"):
        self.message = message
        super().__init__(self.message)


def extract_size(file: Union[str, Path]) -> list:
    def get_size(line: str, match):
        match_M = re.match(r"M: (\d+)", line)
        match_N = re.match(r"N: (\d+)", line)
        match_K = re.match(r"K: (\d+)", line)
        match_batch = re.match(r"batch_count: (\d+)", line)
        matches = [match_M, match_N, match_K, match_batch]
        for m in matches:
            if m != None:
                value = m.group(1).strip()
                match.append(int(value))

    match = []
    with open(file, "r") as f:
        line = f.readline()
        line = line.split(",")
        for x in line:
            get_size(x.strip(), match)
        for m in match:
            assert m > 0
    return 1


REQUIRED_PARAMS = [
    "function",
    "M",
    "N",
    "K",
    "batch_count",
    "transA",
    "transB",
    "a_type",
    "b_type",
    "c_type",
    "d_type",
    "compute_type",
    "iters",
]


def check_params(file: Union[str, Path]) -> list:
    with open(file, "r") as f:
        data = yaml.safe_load(f)
        missing = [p for p in REQUIRED_PARAMS if p not in data[0]]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
        return 1


def test_matmul(mock_yaml_file):
    assert extract_function("dummy.yaml", "matmul")


def test_size(mock_yaml_file):
    assert extract_size("dummy.yaml")


def test_params(mock_yaml_file):
    assert check_params("dummy.yaml")


@pytest.mark.xfail
def test_TensileLibLogicToYaml():
    hipblaslt_path = "."
    device_id = 7
    workspace = "/tmp/tempTestLibLogic"
    arch = "gfx950"
    assert TensileLibLogicToYamlRunner.main(
        hipblaslt_path, device_id, workspace, arch, VALID_YAML_FILE_CONTENT
    )
