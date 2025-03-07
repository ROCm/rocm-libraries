#!/usr/bin/env python3

################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

"""Test basic functionality of rocRoller's GEMM client."""

import contextlib
import functools
import itertools
import os
import pathlib
import pytest
import subprocess

gemm = (
    pathlib.Path(__file__).parent.parent / "build" / "bin" / "client" / "rocRoller_gemm"
).resolve()


# Python 3.11 has contextlib.chdir but 3.10 doesn't
@contextlib.contextmanager
def chdir(directory):
    current_directory = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(current_directory)


@functools.cache
def rocm_gfx():
    output = None
    try:
        output = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        return None

    for line in output.splitlines():
        if line.startswith("  Name:"):
            _, arch, *_ = list(map(lambda x: x.strip(), line.split()))
            if arch.startswith("gfx"):
                return arch

    return None


#
# sample configs
#

DP_GEMM = """\
---
architecture:
  ArchString: gfxunknown
  Xnack: false
  Sramecc: false
mac_m: 64
mac_n: 64
mac_k: 64
wave_m: 32
wave_n: 32
wave_k: 2
wave_b: 1
workgroup_size_x: 128
workgroup_size_y: 2
unroll_x: 0
unroll_y: 0
loadLDS_A: true
loadLDS_B: true
storeLDS_D: true
direct2LDS_A: false
direct2LDS_B: false
prefetch: false
prefetchInFlight: 0
prefetchLDSFactor: 0
betaInFma: true
scheduler: Priority
trans_A: N
trans_B: N
type_A: float
type_B: float
type_C: float
type_D: float
type_acc: float
streamK: false
streamKTwoTile: false
matchMemoryAccess: true
scale_A: None
scale_B: None
loadScaleLDS_A: false
loadScaleLDS_B: false
...

"""


#
# example subcommand
#


def test_gemm_example(tmp_path):
    # "gemm example" with no output file should fail
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([gemm, "example"], check=True)

    # "gemm example example.yaml" should write to example.yaml
    example = tmp_path / "example.yaml"
    subprocess.run([gemm, "example", example], check=True)
    assert example.exists()


#
# generate subcommand basics
#


def test_gemm_generate(tmp_path):
    with chdir(tmp_path):
        # "gemm generate" should pass
        subprocess.run([gemm, "generate"], check=True)

        # "gemm generate --asm" should write an assembly+yaml file in the current directory
        before = list(tmp_path.glob("*.s")) + list(tmp_path.glob("*.yaml"))
        subprocess.run([gemm, "generate", "--asm"], check=True)
        after = list(tmp_path.glob("*.s")) + list(tmp_path.glob("*.yaml"))
        assert len(after) == len(before) + 2

        # "gemm generate --asm test.s" should write .s+.yaml pair
        asm_path = pathlib.Path("test_asm.s")
        yaml_path = asm_path.with_suffix(".yaml")
        subprocess.run([gemm, "generate", "--asm", asm_path], check=True)
        assert asm_path.exists()
        assert yaml_path.exists()

        # possible to not write a pair?
        # "gemm generate --co" should write an co+yaml file in the current directory
        before = list(tmp_path.glob("*.co")) + list(tmp_path.glob("*.yaml"))
        subprocess.run([gemm, "generate", "--co"], check=True)
        after = list(tmp_path.glob("*.co")) + list(tmp_path.glob("*.yaml"))
        assert len(after) == len(before) + 2

        # "gemm generate --co test.co" should write .co+.yaml pair
        co_path = pathlib.Path("test_co.co")
        yaml_path = asm_path.with_suffix(".yaml")
        subprocess.run([gemm, "generate", "--co", co_path], check=True)
        assert co_path.exists()
        assert yaml_path.exists()

        # "gemm generate --config" should fail
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.run([gemm, "generate", "--config"], check=True)


#
# generate and validate
#


def check_returncode(p):
    """Checks return code of GEMM client.

    Returns True if the GEMM client returned SOLUTION_NOT_SUPPORTED_ON_ARCH.

    Raises an error if the GEMM client returned any other non-zero
    code.  This usually means a correctness failure.

    Returns False if the GEMM client returned 0 (ie, OK).
    """
    SOLUTION_NOT_SUPPORTED_ON_ARCH = 3
    if p.returncode != 0:
        if p.returncode == SOLUTION_NOT_SUPPORTED_ON_ARCH:
            return True
        else:
            raise RuntimeError("Client failure: correctness or general failure")
    return False


def write_solution_config_if_present(tmp_path, solution_params):
    """Write contents of '--config' option to .yaml configuration file (if present).

    This helper looks for '--config' in the solution params.  It then
    takes the text of the following argument and writes it to a file
    named 'config.yaml', and replaces the argument.
    """
    if not ("--config" in solution_params):
        return solution_params

    solution_params = list(solution_params)  # copy

    config_index = solution_params.index("--config") + 1
    config = solution_params[config_index]

    config_path = tmp_path / "config.yaml"
    config_path.write_text(config)

    solution_params[config_index] = config_path

    return solution_params


def match_architecture(solution_params):
    """Look for '--arch' option and check if it matches the local device.

    Returns True if either:
    1. No --arch flag present.
    2. The --arch flag matches the local device.
    """

    if not ("--arch" in solution_params):
        return True

    arch_index = solution_params.index("--arch") + 1
    arch = solution_params[arch_index]

    return arch == rocm_gfx()


def gemm_validate_single_stage(tmp_path, solution_params, problem_params):
    solution_params = write_solution_config_if_present(tmp_path, solution_params)

    if not match_architecture(solution_params):
        return

    cmd = [gemm]
    cmd.extend(["generate"])
    cmd.extend(solution_params)
    cmd.extend(["validate"])
    cmd.extend(problem_params)
    check_returncode(subprocess.run(cmd))


def gemm_validate_two_stage_codeobject(tmp_path, solution_params, problem_params):
    solution_params = write_solution_config_if_present(tmp_path, solution_params)

    co = tmp_path / "test.co"

    cmd = [gemm]
    cmd.extend(["generate", "--co", co])
    cmd.extend(solution_params)
    skip = check_returncode(subprocess.run(cmd))
    if skip:
        return

    if not match_architecture(solution_params):
        return

    cmd = [gemm]
    cmd.extend(["validate", "--load", co])
    cmd.extend(problem_params)
    subprocess.run(cmd, check=True)


def gemm_validate_two_stage_assembly(tmp_path, solution_params, problem_params):
    solution_params = write_solution_config_if_present(tmp_path, solution_params)

    asm = tmp_path / "test.s"

    # get these working; load command from .co

    cmd = [gemm]
    cmd.extend(["generate", "--asm", asm])
    cmd.extend(solution_params)
    skip = check_returncode(subprocess.run(cmd))
    if skip:
        return

    if not match_architecture(solution_params):
        return

    cmd = [gemm]
    cmd.extend(["validate", "--load", asm])
    cmd.extend(problem_params)
    subprocess.run(cmd, check=True)


solution_params = [
    # data-parallel gemm, float, params from command line
    [],
    # data-parallel gemm, float, params from config file
    ["--config", DP_GEMM],
    # streamk gemm, float, params from command line
    ["--streamk"],
    # data-parallel gemm, mxfp4, params from command line
    [
        "--arch",
        "gfx950",
        "--mac_M",
        "256",
        "--mac_N",
        "256",
        "--mac_K",
        "64",
        "--mi",
        "32x32x64x1",
        "--type_A",
        "fp4",
        "--type_B",
        "fp4",
        "--scale_A",
        "Separate",
        "--scale_B",
        "Separate",
    ],
    [
        "--arch",
        "gfx950",
        "--mac_M",
        "256",
        "--mac_N",
        "256",
        "--mac_K",
        "64",
        "--mi",
        "32x32x64x1",
        "--type_A",
        "fp4",
        "--type_B",
        "fp4",
        "--type_C",
        "half",
        "--type_D",
        "half",
        "--scale_A",
        "Separate",
        "--scale_B",
        "Separate",
    ],
    [
        "--arch",
        "gfx950",
        "--mac_M",
        "256",
        "--mac_N",
        "256",
        "--mac_K",
        "64",
        "--mi",
        "32x32x64x1",
        "--type_A",
        "fp4",
        "--type_B",
        "fp4",
        "--scale_A",
        "Separate",
        "--scale_B",
        "Separate",
        "--loadLDSScale_A",
        "--loadLDSScale_B",
    ],
]

problem_params = [["--m", "512", "--n", "512", "--k", "256", "--numWGs", "4"]]

validation_parameters = itertools.product(solution_params, problem_params)


@pytest.mark.parametrize("solution_params,problem_params", validation_parameters)
def test_gemm_validate(tmp_path, solution_params, problem_params):
    gemm_validate_single_stage(tmp_path, solution_params, problem_params)
    gemm_validate_two_stage_codeobject(tmp_path, solution_params, problem_params)
    gemm_validate_two_stage_assembly(tmp_path, solution_params, problem_params)


if __name__ == "__main__":
    test_gemm_example()
