#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test basic functionality of rocRoller's GEMM client."""

import contextlib
import functools
import itertools
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

SOLUTION_NOT_SUPPORTED_ON_ARCH = 3

build = Path(__file__).parent.parent / "build"
if os.getenv("ROCROLLER_BUILD_DIR") is not None:
    build = Path(os.getenv("ROCROLLER_BUILD_DIR"))

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
    path: bool  # load through LDS
    value: float  # for SingleScale, the value
    blockSize: int  # for scale block size
    scaleType: str  # data type of the scale values

    def client_arguments(self):
        params = []
        if self.mode is not None:
            params.extend(["--scale_" + self.argument, self.mode])
            if self.value is not None:
                params.extend(["--scaleValue_" + self.argument, str(self.value)])
            if self.path is not None:
                params.extend(["--loadScale_" + self.argument, self.path])

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
workgroupMappingDim: -1
workgroupRemapXCC: false
workgroupRemapXCCValue: -1
load_A: BufferToLDSViaVGPR
load_B: BufferToLDSViaVGPR
padLDS_A: [0, 0]
padLDS_B: [0, 0]
store: VGPRToGlobalMemoryViaLDSWithBuffer
prefetch: false
prefetchInFlight: 0
prefetchLDSFactor: 0
prefetchMixMemOps: false
betaInFma: true
scheduler: Priority
schedulerCost: LinearWeighted
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
  scalePreTileA: []
  scalePreTileB: []
  scaleShuffleTileA: []
  scaleShuffleTileB: []
  scaleSkipPermlane: None
  pretileA: []
  pretileB: []
tailLoops: true
streamK: None
loadScale_A: BufferToVGPR
loadScale_B: BufferToVGPR
swizzleScale: false
swizzleTileSize:
  m: 0
  k: 0
  n: 0
  l: 0
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
workgroupMappingDim: -1
workgroupRemapXCC: false
workgroupRemapXCCValue: -1
load_A: BufferToLDSViaVGPR
load_B: BufferToLDSViaVGPR
padLDS_A: [0, 0]
padLDS_B: [0, 0]
store: VGPRToGlobalMemoryViaLDSWithBuffer
prefetch: false
prefetchInFlight: 0
prefetchLDSFactor: 0
prefetchMixMemOps: false
betaInFma: true
scheduler: Priority
schedulerCost: LinearWeighted
tailLoops: true
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
  scalePreTileA: []
  scalePreTileB: []
  scaleShuffleTileA: []
  scaleShuffleTileB: []
  scaleSkipPermlane: None
  pretileA: []
  pretileB: []
loadScale_A: BufferToVGPR
loadScale_B: BufferToVGPR
swizzleScale: false
swizzleTileSize:
  m: 0
  k: 0
  n: 0
  l: 0
prefetchScale: false
streamK: None
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
workgroupMappingDim: -1
workgroupRemapXCC: false
workgroupRemapXCCValue: -1
load_A: BufferToLDSViaVGPR
load_B: BufferToLDSViaVGPR
padLDS_A: [0, 0]
padLDS_B: [0, 0]
store: VGPRToGlobalMemoryViaLDSWithBuffer
prefetch: false
prefetchInFlight: 0
prefetchLDSFactor: 0
prefetchMixMemOps: false
betaInFma: true
scheduler: Priority
schedulerCost: LinearWeighted
tailLoops: true
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
  scalePreTileA: []
  scalePreTileB: []
  scaleShuffleTileA: []
  scaleShuffleTileB: []
  scaleSkipPermlane: None
  pretileA: []
  pretileB: []
loadScale_A: BufferToVGPR
loadScale_B: BufferToVGPR
swizzleScale: false
swizzleTileSize:
  m: 0
  k: 0
  n: 0
  l: 0
prefetchScale: false
streamK: None
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
    paths = ["BufferToVGPR", "BufferToLDSViaVGPR"]
    values = [0.5, 1.0]
    blockSize = 32
    scaleType = "E8M0"

    rv = []
    for mode in modes:
        if mode is not None and mode == "Separate":
            rv.extend(
                [
                    Scale(argument, mode, path, None, blockSize, scaleType)
                    for path in paths
                ]
            )
        elif mode is not None and mode == "SingleScale":
            rv.extend(
                [
                    Scale(argument, mode, None, value, None, scaleType)
                    for value in values
                ]
            )
        else:
            rv.append(Scale(argument, mode, None, None, None, None))
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
        ["--streamK", "Standard"],
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

        if scaleA.mode == "Separate" or scaleB.mode == "Separate":
            params.extend(["--sts", "64x4/64x4"])

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
                f"--workgroupMappingDim={dimension}",
                f"--workgroupMappingValue={wgm}",
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


def test_gemm_options(tmp_path):
    """GEMM options."""

    example = tmp_path / "example.yaml"
    example_problem = tmp_path / "example_problem.yaml"

    def run_and_load_example_yaml(cmd):
        subprocess.run(cmd, check=True)
        yaml_contents = example.read_text()
        return yaml.load(yaml_contents, Loader=yaml.Loader)

    def run_and_load_example_problem_yaml(cmd):
        subprocess.run(cmd, check=True)
        yaml_contents = example_problem.read_text()
        return yaml.load(yaml_contents, Loader=yaml.Loader)

    # fails
    with pytest.raises(subprocess.CalledProcessError):
        # overspecify tile size is bad
        subprocess.run(
            [
                gemm,
                "example",
                example,
                "--arch=gfx950",
                "--wgts=128x128x128",
                "--mac_M=256",
            ],
            check=True,
        )

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                gemm,
                "example",
                example,
                "--arch=gfx950",
                "--wgs=128x2",
                "--workgroup_size_x=256",
            ],
            check=True,
        )

    # PreSwizzleScaleGFX950 requires pretileScale; client must assert and exit non-zero
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                gemm,
                "example",
                example,
                "--arch=gfx950",
                "--scaleSkipPermlane=PreSwizzleScaleGFX950",
                "--scale_A=Separate",
                "--scale_B=Separate",
                "--scaleBlockSize=32",
                "--sts=32x8/32x8",
            ],
            check=True,
        )

    # PreSwizzleScaleGFX950 requires swizzleTileSize (m=32, n=32, k=8); wrong sts must fail
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                gemm,
                "example",
                example,
                "--arch=gfx950",
                "--scaleSkipPermlane=PreSwizzleScaleGFX950",
                "--scale_A=Separate",
                "--scale_B=Separate",
                "--scaleBlockSize=32",
                "--sts=64x8/64x8",
                "--pretileScale",
            ],
            check=True,
        )

    # setting tile size via shortcut
    post = run_and_load_example_yaml(
        [gemm, "example", example, "--arch=gfx950", "--wgts=1024x2048x4096"]
    )
    assert post["mac_m"] == 1024
    assert post["mac_n"] == 2048
    assert post["mac_k"] == 4096

    # setting wg size via shortcut
    post = run_and_load_example_yaml(
        [gemm, "example", example, "--arch=gfx950", "--wgs=64x4"]
    )
    assert post["workgroup_size_x"] == 64
    assert post["workgroup_size_y"] == 4

    # setting mi via shortcut
    post = run_and_load_example_yaml(
        [gemm, "example", example, "--arch=gfx950", "--mi=2x4x8"]
    )
    assert post["wave_m"] == 2
    assert post["wave_n"] == 4
    assert post["wave_k"] == 8
    assert post["wave_b"] == 1

    post = run_and_load_example_yaml(
        [gemm, "example", example, "--arch=gfx950", "--mi=4x8x16x2"]
    )
    assert post["wave_m"] == 4
    assert post["wave_n"] == 8
    assert post["wave_k"] == 16
    assert post["wave_b"] == 2

    # setting lds options
    post = run_and_load_example_yaml(
        [gemm, "example", example, "--arch=gfx950", "--lds=AB"]
    )
    assert post["load_A"] == "BufferToLDSViaVGPR"
    assert post["load_B"] == "BufferToLDSViaVGPR"
    assert post["store"] == "VGPRToGlobalMemoryWithBuffer"

    post = run_and_load_example_yaml(
        [gemm, "example", example, "--arch=gfx950", "--lds=BD"]
    )
    assert post["load_A"] == "BufferToVGPR"
    assert post["load_B"] == "BufferToLDSViaVGPR"
    assert post["store"] == "VGPRToGlobalMemoryViaLDSWithBuffer"

    # setting d2l options
    post = run_and_load_example_yaml(
        [gemm, "example", example, "--arch=gfx950", "--d2lds=AB"]
    )
    assert post["load_A"] == "BufferToLDS"
    assert post["load_B"] == "BufferToLDS"

    post = run_and_load_example_yaml(
        [gemm, "example", example, "--arch=gfx950", "--d2lds=A"]
    )
    assert post["load_A"] == "BufferToLDS"
    assert post["load_B"] == "BufferToVGPR"

    post = run_and_load_example_yaml(
        [
            gemm,
            "example",
            example,
            "--arch=gfx950",
            "--padLDS_A=22,33",
            "--padLDS_B=44,55",
        ]
    )
    assert post["padLDS_A"] == [22, 33]
    assert post["padLDS_B"] == [44, 55]

    # setting mxlds options
    post = run_and_load_example_yaml(
        [gemm, "example", example, "--arch=gfx950", "--mxlds=AB"]
    )
    assert post["loadScale_A"] == "BufferToLDSViaVGPR"
    assert post["loadScale_B"] == "BufferToLDSViaVGPR"

    post = run_and_load_example_yaml(
        [gemm, "example", example, "--arch=gfx950", "--mxlds=B"]
    )
    assert post["loadScale_A"] == "BufferToVGPR"
    assert post["loadScale_B"] == "BufferToLDSViaVGPR"

    post = run_and_load_example_yaml(
        [gemm, "example", example, "--arch=gfx950", "--mxd2lds=AB"]
    )
    assert post["loadScale_A"] == "BufferToLDS"
    assert post["loadScale_B"] == "BufferToLDS"

    # setting swizzle tile size
    post = run_and_load_example_yaml(
        [gemm, "example", example, "--arch=gfx950", "--sts=5x7/11x13"]
    )
    assert post["swizzleTileSize"]["m"] == 5
    assert post["swizzleTileSize"]["k"] == 7
    assert post["swizzleTileSize"]["n"] == 11
    assert post["swizzleTileSize"]["l"] == 13

    # can also use a big X
    post = run_and_load_example_yaml(
        [gemm, "example", example, "--arch=gfx950", "--sts=5x7X11x13"]
    )
    assert post["swizzleTileSize"]["m"] == 5
    assert post["swizzleTileSize"]["k"] == 7
    assert post["swizzleTileSize"]["n"] == 11
    assert post["swizzleTileSize"]["l"] == 13

    # pretileA
    post = run_and_load_example_yaml(
        [gemm, "example", example, "--wgts=256x256x256", "--pretileA=64x128"]
    )
    assert post["types"]["pretileA"] == [64, 128]

    # pretileB
    post = run_and_load_example_yaml(
        [gemm, "example", example, "--wgts=256x256x256", "--pretileB=64x128"]
    )
    assert post["types"]["pretileB"] == [64, 128]

    # setting data initialization modes
    post = run_and_load_example_problem_yaml(
        [
            gemm,
            "exampleProblem",
            example_problem,
            "--arch=gfx950",
            "--initMode_A=Bounded",
            "--initMode_B=BoundedAlternatingSign",
            "--initMode_C=Unbounded",
        ]
    )
    assert post["initMode_A"] == "DataInitMode(Bounded)"
    assert post["initMode_B"] == "DataInitMode(BoundedAlternatingSign)"
    assert post["initMode_C"] == "DataInitMode(Unbounded)"

    post = run_and_load_example_problem_yaml(
        [
            gemm,
            "exampleProblem",
            example_problem,
            "--arch=gfx950",
            "--initMode_A=Identity",
            "--initMode_B=Ones",
            "--initMode_C=Zeros",
        ]
    )
    assert post["initMode_A"] == "DataInitMode(Identity)"
    assert post["initMode_B"] == "DataInitMode(Ones)"
    assert post["initMode_C"] == "DataInitMode(Zeros)"

    mean_B = 0.0
    std_dev_B = 1.0
    mean_C = 2.0
    std_dev_C = 3.0
    post = run_and_load_example_problem_yaml(
        [
            gemm,
            "exampleProblem",
            example_problem,
            "--arch=gfx950",
            "--initMode_A=TrigonometricFromFloat",
            f"--initMode_B=NormalFromFloat({mean_B}, {std_dev_B})",
            f"--initMode_C=NormalFromFloat({mean_C}, {std_dev_C})",
        ]
    )
    assert post["initMode_A"] == "DataInitMode(TrigonometricFromFloat)"

    initMode_B = post["initMode_B"]
    assert initMode_B.startswith("DataInitMode(NormalFromFloat(")
    mean, std_dev = initMode_B.split("(")[-1][:-2].split(", ")
    assert float(mean) == mean_B
    assert float(std_dev) == std_dev_B

    initMode_C = post["initMode_C"]
    assert initMode_C.startswith("DataInitMode(NormalFromFloat(")
    mean, std_dev = initMode_C.split("(")[-1][:-2].split(", ")
    assert float(mean) == mean_C
    assert float(std_dev) == std_dev_C


def test_gemm_generate_from_example(tmp_path):
    """GEMM 'generate' from the output of 'example'."""

    example = tmp_path / "example.yaml"
    subprocess.run([gemm, "example", example], check=True)

    # We should be able to generate a kernel from the config file
    subprocess.run([gemm, "generate", "--config", example], check=True)


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

        # "gemm generate --asm" should write an assembly and two yaml files in the current directory
        before = list(tmp_path.glob("*.s")) + list(tmp_path.glob("*.yaml"))
        subprocess.run([gemm, "generate", "--asm"], check=True)
        after = list(tmp_path.glob("*.s")) + list(tmp_path.glob("*.yaml"))
        assert len(after) == len(before) + 3

        # "gemm generate --asm test.s" should write an .s and .yaml pair
        asm_path = tmp_path / "test_asm.s"
        yaml_path = asm_path.with_suffix(".yaml")
        subprocess.run([gemm, "generate", "--asm", asm_path], check=True)
        assert asm_path.exists()
        assert yaml_path.exists()

        # possible to not write a pair?
        # "gemm generate --co" should write an co and two yaml files in the current directory
        before = list(tmp_path.glob("*.co")) + list(tmp_path.glob("*.yaml"))
        subprocess.run([gemm, "generate", "--co"], check=True)
        after = list(tmp_path.glob("*.co")) + list(tmp_path.glob("*.yaml"))
        assert len(after) == len(before) + 3

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


def test_kernel_graph_dot_truncation(tmp_path):
    """Validate Graphviz DOT rendering succeeds when node labels are truncated.
    - With truncation enabled (small max label length), kgraph.py should succeed and produce non-empty outputs.
    - With truncation disabled (0), kgraph.py should report a parse error.
    """
    arch = rocm_gfx()
    if arch is not None and arch.startswith("gfx12"):
        pytest.skip("Skipping KernelGraph DOT truncation test on gfx12")

    if not gemm.exists():
        pytest.skip("rocroller-gemm binary not found")

    kgraph = (Path(__file__).parent.parent / "scripts" / "kgraph.py").resolve()
    if not kgraph.exists():
        pytest.skip("kgraph.py script not found")

    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' not available in PATH")

    def run_cmd(cmd, env=None, cwd=None):
        return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)

    def assert_non_empty(path: Path):
        assert path.exists(), f"Expected file to exist: {path}"
        assert path.stat().st_size > 0, f"Expected file to be non-empty: {path}"

    def generate_asm(asm_path: Path, extra_env: dict):
        env = os.environ.copy()
        env.update(extra_env)

        cmd = [
            str(gemm),
            "--workgroupMappingDim=0",
            "--workgroupMappingValue=6",
            "generate",
            "--asm",
            str(asm_path),
        ]

        p = run_cmd(cmd, env=env, cwd=tmp_path)

        if p.returncode == SOLUTION_NOT_SUPPORTED_ON_ARCH:
            pytest.skip("GEMM solution not supported on this architecture")

        assert p.returncode == 0, (
            "rocroller-gemm failed\n"
            f"cmd: {cmd}\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        )
        assert_non_empty(asm_path)

    def run_kgraph(asm_path: Path, pdf_path: Path):
        cmd = [str(kgraph), str(asm_path), "-o", str(pdf_path)]
        p = run_cmd(cmd, env=os.environ.copy(), cwd=tmp_path)
        combined = (p.stdout or "") + "\n" + (p.stderr or "")
        return p, combined

    # Case 1 : truncation enabled(should succeed)
    asm_trunc = tmp_path / "workgroupmapping_truncated5.s"
    pdf_trunc = tmp_path / "workgroupmapping_truncated5.pdf"

    generate_asm(
        asm_trunc,
        {
            "ROCROLLER_SERIALIZE_KERNEL_GRAPH_DOT": "1",
            "ROCROLLER_KGRAPH_NODE_LABEL_MAX_LENGTH": "5",
        },
    )

    p, combined = run_kgraph(asm_trunc, pdf_trunc)
    assert p.returncode == 0, (
        "kgraph.py expected to succeed with truncation enabled\n"
        f"output:\n{combined}\n"
    )

    dot_trunc = pdf_trunc.with_suffix(".dot")
    norm_trunc = pdf_trunc.with_name(pdf_trunc.stem + "_normalized.s")
    assert_non_empty(dot_trunc)
    assert_non_empty(pdf_trunc)
    assert_non_empty(norm_trunc)

    # Case 2 : truncation disabled(should error in kgraph parse)
    asm_untrunc = tmp_path / "workgroupmapping_untruncated.s"
    pdf_untrunc = tmp_path / "workgroupmapping_untruncated.pdf"

    generate_asm(
        asm_untrunc,
        {
            "ROCROLLER_SERIALIZE_KERNEL_GRAPH_DOT": "1",
            "ROCROLLER_KGRAPH_NODE_LABEL_MAX_LENGTH": "0",
        },
    )

    p2, combined2 = run_kgraph(asm_untrunc, pdf_untrunc)

    assert (
        p2.returncode != 0
        or "syntax error" in combined2
        or "longer than 16384" in combined2
    ), (
        "Expected Graphviz parse error when truncation is disabled\n"
        f"returncode: {p2.returncode}\n"
        f"output:\n{combined2}\n"
    )

    dot_untrunc = pdf_untrunc.with_suffix(".dot")
    assert_non_empty(dot_untrunc)
    max_line2 = max(
        len(line)
        for line in dot_untrunc.read_text(errors="ignore").splitlines() or [""]
    )
    assert (
        max_line2 >= 16384
    ), f"Expected an extremely long DOT line without truncation, got max {max_line2}"


if __name__ == "__main__":
    print("Solution params")
    for p in build_solution_params():
        print(p)
    print("Problem params")
    for p in build_problem_params():
        print(p)
