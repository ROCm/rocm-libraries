import pathlib
from dataclasses import dataclass, field, fields
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
class GEMM:
    """GEMM base problem description."""

    M: int
    N: int
    K: int

    mac_m: int
    mac_n: int
    mac_k: int

    alpha: float = 2.0
    beta: float = 0.5

    workgroup_size_x: int = 64 * 2
    workgroup_size_y: int = 2

    unroll_x: int = 0
    unroll_y: int = 0

    type_A: str = "float"
    type_B: str = "float"
    type_C: str = "float"
    type_D: str = "float"
    type_acc: str = "float"

    numWarmUp: int = 2
    numOuter: int = 10
    numInner: int = 1

    trans_A: str = "N"
    trans_B: str = "N"

    loadLDS_A: bool = True
    loadLDS_B: bool = True
    storeLDS_D: bool = False

    @property
    def token(self):
        return repr(GEMM(**field_dict(GEMM, self)))


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
        retval = [
            "client/gemm",
            "--M=" + str(self.M),
            "--N=" + str(self.N),
            "--K=" + str(self.K),
            "--mac_m=" + str(self.mac_m),
            "--mac_n=" + str(self.mac_n),
            "--mac_k=" + str(self.mac_k),
            "--workgroup_size_x=" + str(self.workgroup_size_x),
            "--workgroup_size_y=" + str(self.workgroup_size_y),
            "--unroll_x=" + str(self.unroll_x),
            "--unroll_y=" + str(self.unroll_y),
            "--alpha=" + str(self.alpha),
            "--beta=" + str(self.beta),
            "--type_A=" + str(self.type_A),
            "--type_B=" + str(self.type_B),
            "--type_C=" + str(self.type_C),
            "--type_D=" + str(self.type_D),
            "--type_acc=" + str(self.type_acc),
            "--loadLDS_A=" + str(self.loadLDS_A),
            "--loadLDS_B=" + str(self.loadLDS_B),
            "--storeLDS_D=" + str(self.storeLDS_D),
            "--yaml=" + str(self.output),
            "--num_warmup=" + str(self.numWarmUp),
            "--num_outer=" + str(self.numOuter),
            "--num_inner=" + str(self.numInner),
            "--trans_A=" + str(self.trans_A),
            "--trans_B=" + str(self.trans_B),
        ]
        print(" ".join(retval))
        return retval


@dataclass(frozen=True)
class GEMMResult(GEMM, RRPerfResult):
    """GEMM result interface."""

    pass


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
