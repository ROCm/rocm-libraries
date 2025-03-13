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

import pathlib
from dataclasses import dataclass, field, fields, asdict
from typing import List
import yaml

repo_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent


def field_dict(cls, obj):
    return {f.name: getattr(obj, f.name) for f in fields(cls)}


def to_bool(val):
    if val in ["True", "true", "T", "t", "1"]:
        return True
    elif val in ["False", "false", "F", "f", "0"]:
        return False
    else:
        return bool(val)


def convert_class_params(cls, obj):
    for f in fields(cls):
        if f.type == bool:
            setattr(obj, f.name, to_bool(getattr(obj, f.name)))
        else:
            setattr(obj, f.name, f.type(getattr(obj, f.name)))


@dataclass
class RRPerfResult:
    """Base class for timing results.

    Result classes should be hashable, but the hashes should not
    contain timers or counters.
    """

    client: str = field(repr=False)
    path: pathlib.Path = field(repr=False, hash=False)

    kernelGenerate: int = field(repr=False, hash=False)
    kernelAssemble: int = field(repr=False, hash=False)
    kernelExecute: List[int] = field(repr=False, hash=False)

    device: int = field(repr=False, hash=False, compare=False, default=0)

    checked: bool = field(repr=False, hash=False, compare=False, default=False)
    correct: bool = field(repr=False, hash=False, compare=False, default=True)


#
# GEMM
#
@dataclass(unsafe_hash=True)
class GEMMProblem:
    """GEMM arguments that are part of the problem description."""

    M: int = 1024
    N: int = 1024
    K: int = 1024

    alpha: float = 2.0
    beta: float = 0.5

    type_A: str = "float"
    type_B: str = "float"
    type_C: str = "float"
    type_D: str = "float"
    type_acc: str = "float"

    trans_A: str = "N"
    trans_B: str = "N"

    def __post_init__(self):
        convert_class_params(GEMMProblem, self)


@dataclass(unsafe_hash=True)
class GEMMSolution:
    """GEMM arguments that are part of the selected solution."""

    mac_m: int = 16
    mac_n: int = 16
    mac_k: int = 16

    wave_m: int = -1
    wave_n: int = -1
    wave_k: int = -1
    wave_b: int = -1

    workgroup_size_x: int = 64 * 2
    workgroup_size_y: int = 2

    unroll_x: int = 0
    unroll_y: int = 0

    loadLDS_A: bool = True
    loadLDS_B: bool = True
    storeLDS_D: bool = True
    betaInFma: bool = True

    direct2LDS_A: bool = False
    direct2LDS_B: bool = False

    scheduler: str = "Priority"

    prefetch: bool = True
    prefetchInFlight: int = 2
    prefetchLDSFactor: int = 0

    scale_A: str = "None"
    scale_B: str = "None"

    loadLDSScale_A: bool = False
    loadLDSScale_B: bool = False

    streamK: bool = False
    numWGs: int = 0
    streamKTwoTile: bool = False

    def __post_init__(self):
        convert_class_params(GEMMSolution, self)


@dataclass(unsafe_hash=True)
class GEMM(GEMMProblem, GEMMSolution):
    """GEMM base problem description."""

    numWarmUp: int = 1
    numOuter: int = 1
    numInner: int = 10

    visualize: bool = False

    match_memory_access: bool = True

    def __post_init__(self):
        convert_class_params(GEMM, self)

    @property
    def run_invariant_token(self):
        return repr(GEMMProblem(**field_dict(GEMMProblem, self))) + repr(
            GEMMSolution(**field_dict(GEMMSolution, self))
        )

    @property
    def token(self):
        return repr(GEMM(**field_dict(GEMM, self)))

    def problem_token(self, priority_problems):
        label = ""
        self_dict = field_dict(GEMMProblem, self)
        for label, priority_problem in priority_problems.items():
            if all(
                [
                    arg in self_dict and self_dict[arg] == priority_problem[arg]
                    for arg in priority_problem
                ]
            ):
                label = str(label) + ": "
                break
        return (
            label
            + "GEMM("
            + ", ".join(
                [
                    key + ": " + str(val)
                    for key, val in field_dict(GEMMProblem, self).items()
                ]
            )
            + ")"
        )

    @property
    def solution_token(self):
        return (
            "GEMM("
            + ", ".join(
                [
                    key + ": " + str(val)
                    for key, val in field_dict(GEMMSolution, self).items()
                ]
            )
            + ")"
        )


@dataclass(unsafe_hash=True)
class GEMMRun(GEMM):
    """GEMM run interface."""

    output: pathlib.Path = field(repr=False, default=None, hash=False)

    @property
    def group(self):
        return "gemm"

    def set_output(self, path: pathlib.Path):
        self.output = path

    def command(self, **extra_args) -> List[str]:
        specialNames = {
            "output": "yaml",
            "numWarmUp": "num_warmup",
            "numOuter": "num_outer",
            "numInner": "num_inner",
        }

        command = "bin/client/rocRoller_gemm"

        def argName(key):
            if key in specialNames:
                return specialNames[key]
            return key

        arg_dict = {argName(key): value for key, value in asdict(self).items()}
        for key, value in extra_args.items():
            arg_dict[key] = value

        args = list([f"--{key}={value}" for key, value in arg_dict.items()])
        retval = [command] + args

        return retval


@dataclass
class GEMMResult(GEMM, RRPerfResult):
    """GEMM result interface."""

    rnorm: float = field(repr=False, default=None, hash=False)

    def compact(self):
        def TF(x):
            return {True: "T", False: "F"}[x]

        return {
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "PREC": "".join(
                [
                    {"half": "h", "float": "f"}[getattr(self, "type_" + x)]
                    for x in ["A", "B", "C", "D", "acc"]
                ]
            ),
            "AB": self.trans_A + self.trans_B,
            "m": self.mac_m,
            "n": self.mac_n,
            "k": self.mac_k,
            "WG": str(self.workgroup_size_x) + "/" + str(self.workgroup_size_y),
            "LDS": TF(self.loadLDS_A) + TF(self.loadLDS_B) + TF(self.storeLDS_D),
            "Direct2LDS": TF(self.direct2LDS_A) + TF(self.direct2LDS_B),
            "PF": TF(self.prefetch)
            + "/"
            + str(self.prefetchInFlight)
            + "/"
            + str(self.prefetchLDSFactor),
            "SCH": self.scheduler[0],
            "SK": TF(self.streamK) + "/" + self.numWGs,
            "2TSK": TF(self.streamKTwoTile),
            "iters": "/".join(
                [str(getattr(self, "num" + x)) for x in ["WarmUp", "Outer", "Inner"]]
            ),
        }


#
# CodeGen
#


@dataclass(unsafe_hash=True)
class CodeGen:
    """CodeGen base problem description."""

    instCount: int = 0
    instructions: str = "simple_mfma"

    numWarmUp: int = 2
    numRuns: int = 10

    @property
    def run_invariant_token(self):
        return self.problem_token(None)

    @property
    def token(self):
        return repr(CodeGen(**field_dict(CodeGen, self)))

    @staticmethod
    def problem_args():
        return {"instCount", "instructions"}

    @staticmethod
    def solution_args():
        return {}

    def problem_token(self, priority_problems):
        return (
            "CodeGen("
            + ", ".join(
                [
                    key + ": " + str(val)
                    for key, val in field_dict(CodeGen, self).items()
                    if key in CodeGen.problem_args()
                ]
            )
            + ")"
        )

    @property
    def solution_token(self):
        return (
            "CodeGen("
            + ", ".join(
                [
                    key + ": " + str(val)
                    for key, val in field_dict(CodeGen, self).items()
                    if key in CodeGen.solution_args()
                ]
            )
            + ")"
        )


@dataclass(unsafe_hash=True)
class CodeGenRun(CodeGen):
    """CodeGen run interface."""

    output: pathlib.Path = field(repr=False, default=None, hash=False)

    @property
    def group(self):
        return "codegen"

    def set_output(self, path: pathlib.Path):
        self.output = path

    def command(self) -> List[str]:
        retval = [
            "bin/client/rocRoller_codegen_stress",
            "--inst_count=" + str(self.instCount),
            "--instructions=" + str(self.instructions),
            "--yaml=" + str(self.output),
            "--num_warmup=" + str(self.numWarmUp),
            "--num_runs=" + str(self.numRuns),
        ]
        print(" ".join(retval))
        return retval


@dataclass
class CodeGenResult(CodeGen, RRPerfResult):
    """CodeGen result interface."""

    pass


@dataclass(unsafe_hash=True)
class TensileRun(GEMM):
    """Tensile run interface."""

    config: pathlib.Path = field(repr=False, default=None, hash=False)
    output: pathlib.Path = field(repr=False, default=None, hash=False)
    tensile_commit: str = "rocm-6.0.0"

    @property
    def group(self):
        return "gemm"

    def set_output(self, path: pathlib.Path):
        self.output = path

    def command(self, **extra_args) -> List[str]:
        command = str(repo_dir / "scripts" / "benchmark_tensile")

        arg_dict = asdict(self)
        for key, value in extra_args.items():
            arg_dict[key] = value

        for non_gemm_arg in ["config", "output", "tensile_commit"]:
            arg_dict.pop(non_gemm_arg, None)

        args = list([f"{key}={value}" for key, value in arg_dict.items()])

        retval = [
            command,
            str(self.config),
            f"--yaml={str(self.output)}",
            f"--tensile_commit={self.tensile_commit}",
            "--kwargs",
        ] + args

        return retval


#
# Up/down cast from BASE classes to RUN and RESULT classes.
#

_client_to_result_class = {
    "GEMMv00": GEMMResult,
    "CodeGenv00": CodeGenResult,
}

_base_to_run_class = {
    GEMM: GEMMRun,
    CodeGen: CodeGenRun,
}


def load_results(path: pathlib.Path):
    """
    Load results from a YAML file `path` and return an array of RESULT objects.
    """
    rv = []
    for r in yaml.load_all(path.read_text(), Loader=yaml.FullLoader):
        ResultClass = _client_to_result_class[r["client"]]
        r.pop("path", None)
        rv.append(ResultClass(path=path, **r))
    return rv


def upcast_to_run(obj):
    """Upcast a BASE object to a RUN object."""
    DownClass = type(obj)
    UpClass = _base_to_run_class[DownClass]
    return UpClass(**field_dict(DownClass, obj))
