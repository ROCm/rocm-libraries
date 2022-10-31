import pathlib
import yaml

from dataclasses import dataclass, field, fields
from typing import List


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

    type_A: str = "float"
    type_B: str = "float"
    type_C: str = "float"
    type_D: str = "float"
    type_acc: str = "float"

    numWarmUp: int = 2
    numOuter: int = 10
    numInner: int = 1

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
            "--alpha=" + str(self.alpha),
            "--beta=" + str(self.beta),
            "--type_A=" + str(self.type_A),
            "--type_B=" + str(self.type_B),
            "--type_C=" + str(self.type_C),
            "--type_D=" + str(self.type_D),
            "--type_acc=" + str(self.type_acc),
            "--yaml=" + str(self.output),
            "--num_warmup=" + str(self.numWarmUp),
            "--num_outer=" + str(self.numOuter),
            "--num_inner=" + str(self.numInner),
        ]
        print(" ".join(retval))
        return retval


@dataclass(frozen=True)
class GEMMResult(GEMM, RRPerfResult):
    """GEMM result interface."""

    pass


#
# Up/down cast from BASE classes to RUN and RESULT classes.
#

_client_to_result_class = {
    "GEMMv00": GEMMResult,
}

_base_to_run_class = {
    GEMM: GEMMRun,
}


def load_results(path: pathlib.Path):
    """Load results from a YAML file `path` and return an array of RESULT objects."""
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
