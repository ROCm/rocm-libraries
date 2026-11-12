# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Checks the skinny MXFP8 kernels against their own custom.config.

The custom.config block embedded in each .s file is the single source of truth for the kernel
metadata. The Tests/custom YAML forks repeat it, so they are checked against it here instead of
against a second retyped table. The host sizes the launch from this metadata alone (the kernel
symbol is not parsed).
"""

import os
import re
from pathlib import Path

import pytest
import yaml

import tensilelite
from tensilelite.CustomKernels import readCustomKernelConfig

pytestmark = pytest.mark.unit

KERNEL_DIR = os.path.join(os.path.dirname(tensilelite.__file__), "CustomKernels", "skinny_mxfp8")
TESTS_CUSTOM_DIR = Path(tensilelite.__file__).parent / "Tests" / "custom"

ARG_BYTES = {"address": 8, "uint64": 8, "int32": 4, "float32": 4}
DEPTH = 4096


def kernel_names():
    return sorted(f[:-2] for f in os.listdir(KERNEL_DIR) if f.endswith(".s"))


KERNELS = kernel_names()


def is_mfma(name):
    return "MFMA1" in name


def assert_items(mapping):
    return {key: value for key, value in mapping.items() if key.startswith("Assert")}


def kernarg_segment_size(name):
    """The .kernarg_segment_size of the kernel, read from its .amdgpu_metadata block."""
    text = Path(KERNEL_DIR, name + ".s").read_text()
    block = re.search(r"^\.amdgpu_metadata\n(.*?)^\.end_amdgpu_metadata", text, re.S | re.M)
    return yaml.safe_load(block.group(1))["amdhsa.kernels"][0][".kernarg_segment_size"]


# ---------------------------------------------------------------------------
# The kernel arguments agree with the kernel metadata.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", KERNELS)
def test_kernargs_match_the_kernel_metadata(name):
    args = readCustomKernelConfig(name)["CustomKernel"]["args"]
    assert sum(ARG_BYTES[entry["type"]] for entry in args) == kernarg_segment_size(name)


@pytest.mark.parametrize("name", KERNELS)
def test_features_reject_bias_activation_and_scale_alpha(name):
    # LibraryIO restricts the solution's ProblemType from these, so the kernels must declare them.
    features = readCustomKernelConfig(name)["Features"]
    assert features["SupportsBias"] is False
    assert features["SupportsActivation"] is False
    assert features["SupportsScaleAlpha"] is False


@pytest.mark.parametrize("name", KERNELS)
def test_name_fields_agree_with_the_metadata(name):
    # The host never parses the name, so this cross-checks the two independently.
    descriptor = readCustomKernelConfig(name)["CustomKernel"]
    waves = int(name.split("_WAVES")[1].split("_")[0])
    tile_n = int(name.split("_TN")[1].split("_")[0])
    assert descriptor["threads"][0] == 64 * waves
    # An MFMA wave covers 16 columns; the macrotile is square so that the host launch matches the
    # kernel's tile count in both orientations up to a skinny extent of 16.
    if is_mfma(name):
        assert descriptor["macrotile"] == [16 * tile_n, 16 * tile_n, DEPTH]
        return
    # A VALU wave covers 64 / lanesK columns per thread group.
    block_m = int(name.split("gemm_BM")[1].split("_")[0])
    lanes_k = int(name.split("_LANES_K")[1].split("_")[0])
    assert descriptor["macrotile"] == [waves * (64 // lanes_k) * tile_n, block_m, DEPTH]


# ---------------------------------------------------------------------------
# The copies of the metadata agree with custom.config.
# ---------------------------------------------------------------------------


def load_custom_yaml_for(name):
    """The one Tests/custom YAML whose fork names this kernel, and its parsed pieces."""
    found = []
    for path in sorted(TESTS_CUSTOM_DIR.glob("custom_skinny_mxfp8_tn_sel*.yaml")):
        with open(path) as f:
            doc = yaml.safe_load(f)
        problem_type, benchmark = doc["BenchmarkProblems"][0]
        forks = {k: v for fork in benchmark["ForkParameters"] for k, v in fork.items()}
        if forks["CustomKernel"][0]["name"] == name:
            found.append((path.name, doc, problem_type, forks))
    assert len(found) == 1, f"expected one Tests/custom YAML for {name}, found {len(found)}"
    return found[0]


@pytest.mark.parametrize("name", KERNELS)
def test_load_custom_yaml_for_matches_custom_config(name):
    config = readCustomKernelConfig(name)
    _, doc, problem_type, forks = load_custom_yaml_for(name)

    assert forks["CustomKernel"] == [{**config["CustomKernel"], "name": name}]
    asserts = assert_items(forks)
    assert asserts == {key: [value] for key, value in assert_items(config).items()}
    assert forks["MXScaleFormat"] == [config["MXScaleFormat"]]
    assert forks["StaggerU"] == [config["StaggerU"]]
    assert forks["MatrixInstruction"][0][: len(config["MatrixInstruction"])] == config["MatrixInstruction"]
    for key, value in config["ProblemType"].items():
        assert problem_type[key] == value, key
    assert doc["LibraryLogic"]["ArchitectureName"] == "gfx950"


# ---------------------------------------------------------------------------
# The .s header is comment-only and names the constraints of the embedded custom.config.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", KERNELS)
def test_header_is_comment_only_and_lists_every_assert_key(name):
    header = Path(KERNEL_DIR, name + ".s").read_text().split(".amdgcn_target", 1)[0]
    assert all(line.startswith("//") for line in header.splitlines())
    missing = [key for key in assert_items(readCustomKernelConfig(name)) if key not in header]
    assert not missing, missing
