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
import yaml

from dataclasses import dataclass

build = pathlib.Path(__file__).parent.parent / "build"
if os.getenv("ROCROLLER_BUILD_DIR") is not None:
    build = pathlib.Path(os.getenv("ROCROLLER_BUILD_DIR"))

gemm = (build / "client" / "rocroller-gemm").resolve()


# Python 3.11 has contextlib.chdir but 3.10 doesn't
@contextlib.contextmanager
def chdir(directory):
    current_directory = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(current_directory)


#
# Helpers
#


@functools.cache
def rocm_gfx():
    """Return GPU architecture (gfxXXXX) for local GPU device."""
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


#
# Solution parameter helpers
#


@dataclass
class Types:
    """Container for GEMM type parameters."""

    A: str
    B: str
    C: str
    D: str

    def client_arguments(self):
        return [
            "--type_A",
            self.A,
            "--type_B",
            self.B,
            "--type_C",
            self.C,
            "--type_D",
            self.D,
        ]


@dataclass
class Scale:
    """Container for MX GEMM scaling parameters."""

    argument: str  # A or B
    mode: str  # Separate, SingleScale, etc
    lds: bool  # load through LDS
    value: float  # for SingleScale, the value
    blockSize: int  # for scale block size
    scaleType: str  # data type of the scale values

    def client_arguments(self):
        params = []
        if self.mode is not None:
            params.extend(["--scale_" + self.argument, self.mode])
            if self.value is not None:
                params.extend(["--scaleValue_" + self.argument, str(self.value)])
            if self.lds:
                params.append("--loadLDSScale_" + self.argument)

            if self.mode == "Separate" or self.mode == "SingleScale":
                if self.scaleType is not None:
                    params.extend(["--scaleType_" + self.argument, self.scaleType])
                else:
                    params.extend(["--scaleType_" + self.argument, "E8M0"])

        return params

    def maybe_add_block_size(self, params):
        if self.blockSize is None:
            return

        scaleBlockSizeStr = "--scaleBlockSize"
        if scaleBlockSizeStr not in params:
            params.extend([scaleBlockSizeStr, str(self.blockSize)])


@dataclass
class Prefetch:
    """Container for GEMM prefetching parameters."""

    in_flight: int
    lds_factor: int

    def client_arguments(self):
        params = []
        if self.in_flight != 0:
            params.extend(
                [
                    "--prefetch",
                    "--prefetchInFlight",
                    str(self.in_flight),
                    "--prefetchLDSFactor",
                    str(self.lds_factor),
                ]
            )
        return params


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
workgroupMapping: [-1, -1]
workgroupRemapXCC: false
workgroupRemapXCCValue: -1
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
prefetchMixMemOps: false
betaInFma: true
scheduler: Priority
types:
  trans_A: N
  trans_B: N
  type_A: float
  type_B: float
  type_C: float
  type_D: float
  type_acc: float
  scale_A: None
  scaleType_A: None
  scale_B: None
  scaleType_B: None
  scaleBlockSize: 0
  scaleSkipPermlane: false
streamK: false
streamKTwoTile: false
matchMemoryAccess: true
loadLDSScale_A: false
loadLDSScale_B: false
swizzleScale: false
prefetchScale: false
...

"""

DP_HGEMM = """\
---
architecture:
  ArchString: gfx90a
  Xnack: false
  Sramecc: false
mac_m: 64
mac_n: 64
mac_k: 64
wave_m: 32
wave_n: 32
wave_k: 8
wave_b: 1
workgroup_size_x: 128
workgroup_size_y: 2
workgroupMapping: [-1, -1]
workgroupRemapXCC: false
workgroupRemapXCCValue: -1
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
prefetchMixMemOps: false
betaInFma: true
scheduler: Priority
matchMemoryAccess: true
types:
  trans_A: N
  trans_B: N
  type_A: half
  type_B: half
  type_C: half
  type_D: half
  type_acc: float
  scale_A: None
  scaleType_A: None
  scale_B: None
  scaleType_B: None
  scaleBlockSize: 0
  scaleSkipPermlane: false
loadLDSScale_A: false
loadLDSScale_B: false
swizzleScale: false
prefetchScale: false
streamK: false
streamKTwoTile: false
...
"""

DP_HGEMM_GFX120X = """\
---
architecture:
  ArchString: gfx1201
  Xnack: false
  Sramecc: false
mac_m: 64
mac_n: 64
mac_k: 64
wave_m: 16
wave_n: 16
wave_k: 16
wave_b: 1
workgroup_size_x: 64
workgroup_size_y: 2
workgroupMapping: [-1, -1]
workgroupRemapXCC: false
workgroupRemapXCCValue: -1
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
prefetchMixMemOps: false
betaInFma: true
scheduler: Priority
matchMemoryAccess: true
types:
  trans_A: N
  trans_B: N
  type_A: half
  type_B: half
  type_C: half
  type_D: half
  type_acc: float
  scale_A: None
  scaleType_A: None
  scale_B: None
  scaleType_B: None
  scaleBlockSize: 0
  scaleSkipPermlane: false
loadLDSScale_A: false
loadLDSScale_B: false
swizzleScale: false
prefetchScale: false
streamK: false
streamKTwoTile: false
...
"""


def type_configurations():
    """Return list of type combinations to test."""
    typeAs = ["fp4"]
    typeBs = ["fp4", "fp8"]
    typeDs = ["half", "float"]
    return [Types(A, B, D, D) for A, B, D in itertools.product(typeAs, typeBs, typeDs)]


def scale_configurations(argument):
    """Return list of MX scale modes to test for each of A and B."""
    modes = [None, "None", "Separate", "SingleScale"]
    ldss = [True, False]
    values = [0.5, 1.0]
    blockSize = 32
    scaleType = "E8M0"

    rv = []
    for mode in modes:
        if mode is not None and mode == "Separate":
            rv.extend(
                [Scale(argument, mode, lds, None, blockSize, scaleType) for lds in ldss]
            )
        elif mode is not None and mode == "SingleScale":
            rv.extend(
                [
                    Scale(argument, mode, False, value, None, scaleType)
                    for value in values
                ]
            )
        else:
            rv.append(Scale(argument, mode, False, None, None, None))
    return rv


def prefetch_configurations():
    """Return list of prefetching modes to test."""
    rv = [Prefetch(0, 0)]
    for lds_factor in [0, 2]:
        rv.append(Prefetch(2, lds_factor))
    return rv


def build_solution_params():
    """Build giant list of solution parameter combinations to test."""

    solution_params = [
        # data-parallel gemm, float, params from command line
        [],
        # data-parallel gemm, float, params from config file
        ["--config", DP_GEMM],
        # streamk gemm, float, params from command line
        ["--streamk"],
    ]

    for type, prefetch, scaleA, scaleB in itertools.product(
        type_configurations(),
        prefetch_configurations(),
        scale_configurations("A"),
        scale_configurations("B"),
    ):
        # XXXX Mixing and outputting to half precision fails correctness checks.
        if (type.A != type.B) and (type.D == "half"):
            continue
        params = [
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
        ]
        for x in [type, prefetch, scaleA, scaleB]:
            params.extend(x.client_arguments())

        scaleA.maybe_add_block_size(params)
        scaleB.maybe_add_block_size(params)

        solution_params.append(params)

    return solution_params


def build_problem_params():
    """Build list of problem parameters to test."""
    # Should consider making this a container
    return [["--m", "512", "--n", "512", "--k", "256", "--numWGs", "4"]]


def build_wgm_params():
    """Build list of workgroup-mapping parameters to test."""
    size = [1024, 1024, 512]
    tile_size = [64, 64, 64]
    num_tiles = [size[i] // tile_size[i] for i in range(len(size))]

    params = []
    for dimension in [0, 1]:
        # the last entry will have no "tail block"
        for wgm in [1, 3, 5, 16, num_tiles[dimension]]:
            solution_params = [
                f"--mac_m={tile_size[0]}",
                f"--mac_n={tile_size[1]}",
                f"--mac_k={tile_size[2]}",
                f"--workgroupMapping={dimension},{wgm}",
            ]
            problem_params = [f"--m={size[0]}", f"--n={size[1]}", f"--k={size[2]}"]
            params.append([solution_params, problem_params])
    return params


#
# GEMM 'generate' and 'validate' helpers
#


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


#
# PyTest tests!
#


def test_gemm_example(tmp_path):
    """GEMM 'example' subcommand."""

    # "gemm example" with no output file should fail
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([gemm, "example"], check=True)

    # "gemm example example.yaml" should write to example.yaml
    example = tmp_path / "example.yaml"
    subprocess.run([gemm, "example", example], check=True)
    assert example.exists()


def test_gemm_config(tmp_path):
    """GEMM load from config file."""

    solution_params = ["--arch", "gfx90a", "--config", DP_HGEMM]
    solution_params = write_solution_config_if_present(tmp_path, solution_params)

    co_path = tmp_path / "test_config.co"
    yaml_path = co_path.with_suffix(".yaml")

    cmd = [gemm]
    cmd.extend(["generate", "--co", co_path])
    cmd.extend(solution_params)
    subprocess.run(cmd, check=True)

    yaml_contents = yaml_path.read_text()
    client = yaml.load(yaml_contents, Loader=yaml.Loader)
    del client["version"]

    reference = yaml.load(DP_HGEMM, Loader=yaml.Loader)

    assert client == reference


def test_gemm_generate(tmp_path):
    """GEMM 'generate' basics."""

    with chdir(tmp_path):
        # "gemm generate" should pass
        subprocess.run([gemm, "generate"], check=True)

        # "gemm generate --asm" should write an assembly+yaml file in the current directory
        before = list(tmp_path.glob("*.s")) + list(tmp_path.glob("*.yaml"))
        subprocess.run([gemm, "generate", "--asm"], check=True)
        after = list(tmp_path.glob("*.s")) + list(tmp_path.glob("*.yaml"))
        assert len(after) == len(before) + 2

        # "gemm generate --asm test.s" should write .s+.yaml pair
        asm_path = tmp_path / "test_asm.s"
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
        co_path = tmp_path / "test_co.co"
        yaml_path = asm_path.with_suffix(".yaml")
        subprocess.run([gemm, "generate", "--co", co_path], check=True)
        assert co_path.exists()
        assert yaml_path.exists()

        # "gemm generate --config" should fail
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.run([gemm, "generate", "--config"], check=True)


def test_gemm_validate(tmp_path):
    """GEMM generate and validate using one and two stages.

    This runs each problem/solution three times.
    """

    isGFX120X = rocm_gfx().startswith("gfx120")

    problem_params = [["--m", "512", "--n", "512", "--k", "256", "--numWGs", "4"]]
    solution_params = [
        # data-parallel gemm, float, params from command line
        # [],
        # data-parallel gemm, float, params from config file
        ["--config", DP_HGEMM_GFX120X if isGFX120X else DP_GEMM],
        # streamk gemm, float, params from command line
        # ["--streamk"],
    ]

    for problem, solution in itertools.product(problem_params, solution_params):
        gemm_validate_single_stage(tmp_path, solution, problem)
        gemm_validate_two_stage_codeobject(tmp_path, solution, problem)
        gemm_validate_two_stage_assembly(tmp_path, solution, problem)


@pytest.mark.parametrize(
    "solution_params,problem_params",
    itertools.product(build_solution_params(), build_problem_params()),
)
def test_gemm_validate_once(tmp_path, solution_params, problem_params):
    """GEMM generate (always) and validate (if arch matches)."""

    gemm_validate_two_stage_codeobject(tmp_path, solution_params, problem_params)


@pytest.mark.parametrize("solution_params,problem_params", build_wgm_params())
def test_gemm_wgm(tmp_path, solution_params, problem_params):
    # TODO This is a temporary fix to enable GFX12 CI
    if rocm_gfx().startswith("gfx12"):
        return

    gemm_validate_single_stage(tmp_path, solution_params, problem_params)


if __name__ == "__main__":
    print("Solution params")
    for p in build_solution_params():
        print(p)
    print("Problem params")
    for p in build_problem_params():
        print(p)
