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
import tempfile
import re


parentDir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
)
sys.path.append(parentDir)
from Tensile import TensileLibLogicToYamlRunner


# Test data
VALID_YAML_FILE_CONTENT = """- {function: matmul, M: 768, N: 3072, K: 2048, lda: 2048, ldb: 2048, ldc: 768, ldd: 768, stride_a: 0, stride_b: 0, stride_c: 0, stride_d: 0, alpha: 1.000000, beta: 0.000000, transA: T, transB: N, batch_count: 1, scaleA: 0, scaleB: 0, scaleC: 0, scaleD: 0, swizzleA: false, swizzleB: false, scaleAlpha_vector: false, gradient: false, use_e: false, bias_vector: false, bias_source: d, a_type: bf16_r, b_type: bf16_r, c_type: bf16_r, d_type: bf16_r, scale_type: f32_r, bias_type: f32_r, compute_type: c_f32_r, activation_type: none, flush: false, any_stride: true, rotating: 512, cold_iters: 0, iters: 1, print_kernel_info: true}"""


@pytest.fixture
def mockYamlFile():
    with patch(
        "builtins.open", mock_open(read_data=VALID_YAML_FILE_CONTENT)
    ) as mockFile:
        yield mockFile


def extractFunction(file: str, function: str) -> bool:
    msgPrefix = f"Invalid function: {file}"
    with open(file, "r") as f:
        data = yaml.safe_load(f)[0]
        if "function" not in data:
            raise ValueError(f"{msgPrefix}: function not present!")
        if data["function"] != function:
                raise ValueError(f"{msgPrefix}: {data["function"]} is not the same as {function}")
    return True

def extractSize(file: Union[str, Path]) -> bool:
    def getSize(line: str, match):
        matchM = re.match(r"M: (\d+)", line)
        matchN = re.match(r"N: (\d+)", line)
        matchK = re.match(r"K: (\d+)", line)
        matchBatch = re.match(r"batch_count: (\d+)", line)
        matches = [matchM, matchN, matchK, matchBatch]
        for m in matches:
            if m != None:
                value = m.group(1).strip()
                match.append(int(value))

    match = []
    with open(file, "r") as f:
        line = f.readline()
        line = line.split(",")
        for x in line:
            getSize(x.strip(), match)
        for m in match:
            assert m > 0
    return True


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


def checkParams(file: Union[str, Path]) -> bool:
    with open(file, "r") as f:
        data = yaml.safe_load(f)
        missing = [p for p in REQUIRED_PARAMS if p not in data[0]]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
        return True


def test_matmul(mockYamlFile):
    assert extractFunction("dummy.yaml", "matmul")


def test_size(mockYamlFile):
    assert extractSize("dummy.yaml")


def test_params(mockYamlFile):
    assert checkParams("dummy.yaml")


@pytest.mark.xfail
def test_TensileLibLogicToYaml():
    hipblasltPath = "/workspace/rocm-libraries/projects/hipblaslt/"
    deviceId = 0

    with tempfile.TemporaryDirectory() as workspace:
        arch = "gfx950"
        assert TensileLibLogicToYamlRunner.main(
            hipblasltPath, deviceId, workspace, arch, VALID_YAML_FILE_CONTENT
        )
