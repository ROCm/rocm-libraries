import pathlib
from dataclasses import dataclass, field, fields, asdict
from typing import List

import yaml


def field_dict(cls, obj):
    return {f.name: getattr(obj, f.name) for f in fields(cls)}


@dataclass(frozen=True)
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


@dataclass(unsafe_hash=True)
class GEMMSolution:
    """GEMM arguments that are part of the selected solution."""

    mac_m: int = 16
    mac_n: int = 16
    mac_k: int = 16

    workgroup_size_x: int = 64 * 2
    workgroup_size_y: int = 2

    unroll_x: int = 0
    unroll_y: int = 0

    loadLDS_A: bool = True
    loadLDS_B: bool = True
    storeLDS_D: bool = True
    betaInFma: bool = True

    scheduler: str = "Priority"

    prefetch: bool = True


@dataclass(unsafe_hash=True)
class GEMM(GEMMProblem, GEMMSolution):
    """GEMM base problem description."""

    numWarmUp: int = 2
    numOuter: int = 10
    numInner: int = 1

    visualize: bool = False

    match_memory_access: bool = True

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

    def command(self):
        specialNames = {
            "output": "yaml",
            "numWarmUp": "num_warmup",
            "numOuter": "num_outer",
            "numInner": "num_inner",
        }

        command = "client/gemm"

        def argName(key):
            if key in specialNames:
                return specialNames[key]
            return key

        args = list(
            [f"--{argName(key)}={value}" for key, value in asdict(self).items()]
        )
        retval = [command] + args

        return retval


@dataclass(frozen=True)
class GEMMResult(GEMM, RRPerfResult):
    """GEMM result interface."""

    def compact(self):
        def TF(x):
            return {True: "T", False: "F"}[x]

        return {
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "prec": "".join(
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
            "scheduler": self.scheduler,
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

    instCount: int
    instructions: str = "simple_mfma"

    numWarmUp: int = 2
    numRuns: int = 10

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

    def command(self):
        retval = [
            "client/codegen_stress",
            "--inst_count=" + str(self.instCount),
            "--instructions=" + str(self.instructions),
            "--yaml=" + str(self.output),
            "--num_warmup=" + str(self.numWarmUp),
            "--num_runs=" + str(self.numRuns),
        ]
        print(" ".join(retval))
        return retval


@dataclass(frozen=True)
class CodeGenResult(CodeGen, RRPerfResult):
    """CodeGen result interface."""

    pass


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
        rv.append(ResultClass(path=path, **r))
    return rv


def upcast_to_run(obj):
    """Upcast a BASE object to a RUN object."""
    DownClass = type(obj)
    UpClass = _base_to_run_class[DownClass]
    return UpClass(**field_dict(DownClass, obj))
