# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml
from rrperf.utils import get_dataclass_id

repo_dir = Path(__file__).resolve().parent.parent.parent.parent


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
            setattr(obj, f.name, getattr(obj, f.name))


@dataclass(unsafe_hash=True)
class MNKTuple:
    """M/N/K tuple"""

    m: int
    n: int
    k: int

    def __str__(self):
        return f"{self.m}x{self.n}x{self.k}"


@dataclass(unsafe_hash=True)
class MKNLTuple:
    """MxK/NxL tuple"""

    m: int
    k: int
    n: int
    l: int

    def __str__(self):
        return f"{self.m}x{self.k}/{self.n}x{self.l}"


@dataclass
class RRPerfResult:
    """Base class for timing results.

    Result classes should be hashable, but the hashes should not
    contain timers or counters.
    """

    resultType: str = field(repr=False)
    path: Path = field(repr=False, hash=False)

    kernelGenerate: int = field(repr=False, hash=False)
    kernelAssemble: int = field(repr=False, hash=False)
    kernelExecute: list[int] = field(repr=False, hash=False)

    device: int = field(repr=False, hash=False, compare=False, default=0)

    checked: bool = field(repr=False, hash=False, compare=False, default=False)
    correct: bool = field(repr=False, hash=False, compare=False, default=True)

    sgprCount: int = field(repr=False, hash=False, compare=False, default=0)
    vgprCount: int = field(repr=False, hash=False, compare=False, default=0)
    agprCount: int = field(repr=False, hash=False, compare=False, default=0)

    ldsBytes: int = field(repr=False, hash=False, compare=False, default=0)


@dataclass(unsafe_hash=True)
class TypeParameters:
    """All types that are part of the problem description"""

    type_A: str = "float"
    type_B: str = "float"
    type_C: str = "float"
    type_D: str = "float"
    type_acc: str = "float"

    trans_A: str = "N"
    trans_B: str = "N"

    scale_A: str = "None"
    scaleType_A: str = "None"
    scale_B: str = "None"
    scaleType_B: str = "None"

    # If scale_A or scale_B is Separate, scaleBlockSize
    # needs to be set to a valid block size (e.g. 32)
    scaleBlockSize: int = -1
    scaleSkipPermlane: str = "None"  # None, PreSwizzleScale, PreSwizzleScaleGFX950

    def __init__(self, typeParams: Any | None = None, **kwargs):
        if isinstance(typeParams, TypeParameters):
            for f in fields(self):
                setattr(self, f.name, getattr(typeParams, f.name))
        elif typeParams is not None:
            raise TypeError(
                f"Expected TypeParameters or None, got {type(typeParams).__name__}"
            )

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Unknown field: {key}")

    def asArgs(self) -> list[str]:
        rv: list[str] = []
        for f in fields(self):
            rv.append(f"--{f.name}={self.__getattribute__(f.name)}")
        return rv


@dataclass(unsafe_hash=True)
class GPUArchitectureTarget:
    """GPUArchitectureTarget"""

    ArchString: str = ""
    Xnack: bool = False
    Sramecc: bool = False

    def asArgs(self) -> list[str]:
        if len(self.ArchString) == 0:
            return []
        arch = self.ArchString
        arch += ":xnack+" if self.Xnack else ""
        arch += ":sramecc+" if self.Sramecc else ""
        return ["--arch=" + arch]


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

    types: TypeParameters = TypeParameters()

    scaleValue_A: float = 1.0
    scaleValue_B: float = 1.0

    initMode_A: str = "Bounded"
    initMode_B: str = "Bounded"
    initMode_C: str = "Bounded"

    workgroupMappingDim: int = -1
    workgroupMappingValue: int = -1

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

    workgroup_size_x: int = -1
    workgroup_size_y: int = -1
    workgroupRemapXCC: bool = False
    workgroupRemapXCCValue: int = -1

    load_A: str = "BufferToLDSViaVGPR"
    load_B: str = "BufferToLDSViaVGPR"
    store: str = "VGPRToGlobalMemoryViaLDSWithBuffer"
    betaInFma: bool = True

    padLDS_A: tuple[int, int] = (0, 0)
    padLDS_B: tuple[int, int] = (0, 0)

    scheduler: str = "Priority"
    schedulerCost: str = "LinearWeighted"

    prefetch: bool = True
    prefetchInFlight: int = 2
    prefetchLDSFactor: int = 0
    prefetchMixMemOps: bool = False

    loadScale_A: str = "BufferToVGPR"
    loadScale_B: str = "BufferToVGPR"

    swizzleScale: bool = False
    swizzleTileSize: MKNLTuple = MKNLTuple(0, 0, 0, 0)
    prefetchScale: bool = False
    pretileScale: bool = False

    ldsBankSwizzle: str = "None"

    streamK: str = "None"
    numWGs: int = 0

    architecture: GPUArchitectureTarget = GPUArchitectureTarget()
    tailLoops: bool = True

    version: str = ""

    def __post_init__(self):
        convert_class_params(GEMMSolution, self)


@dataclass(unsafe_hash=True)
class GEMM(GEMMProblem, GEMMSolution):
    """GEMM base problem description."""

    numWarmUp: int = 1
    numOuter: int = 1
    numInner: int = 10

    noCheck: bool = False

    visualize: bool = False

    def __post_init__(self):
        convert_class_params(GEMM, self)

    @property
    def id(self):
        return get_dataclass_id(self)

    @property
    def run_invariant_token(self):
        # Build metadata (e.g. git commit tag) should not affect cross-run matching.
        # Excluding version keeps GEMM comparisons stable across different commits.
        solution_fields = field_dict(GEMMSolution, self)
        solution_fields.pop("version", None)
        return repr(GEMMProblem(**field_dict(GEMMProblem, self))) + repr(
            GEMMSolution(**solution_fields)
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

    output: Path = field(repr=False, default=None, hash=False, compare=False)

    @property
    def group(self):
        return "gemm"

    def set_output(self, path: Path):
        self.output = path

    def command(
        self, generate_only=False, architecture=None, **extra_args
    ) -> list[str]:

        specialNames = {
            "output": "yaml",
            "numWarmUp": "num_warmup",
            "numOuter": "num_outer",
            "numInner": "num_inner",
            "swizzleTileSize": "sts",
        }

        command = "client/rocroller-gemm"

        def argName(key):
            if key in specialNames:
                return specialNames[key]
            return key

        if architecture is not None:
            self.architecture.ArchString = architecture

        arg_dict = {
            argName(key): value
            for key, value in (
                (f.name, getattr(self, f.name)) for f in fields(self.__class__)
            )
        }
        arg_dict.update(extra_args)

        if generate_only:
            for attr in ["yaml", "num_warmup", "num_inner", "num_outer"]:
                arg_dict.pop(attr)
        arg_dict.pop("version")

        args = []
        for key, value in arg_dict.items():
            if hasattr(value, "asArgs"):
                args.extend(value.asArgs())
            elif isinstance(value, tuple):
                args.append(f"--{key}={','.join(map(str, value))}")
            else:
                args.append(f"--{key}={str(value)}")

        if generate_only:
            args.append("generate")

        return [command] + args


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
                    {"half": "h", "float": "f"}[getattr(self.types, "type_" + x)]
                    for x in ["A", "B", "C", "D", "acc"]
                ]
            ),
            "AB": self.types.trans_A + self.types.trans_B,
            "m": self.mac_m,
            "n": self.mac_n,
            "k": self.mac_k,
            "WG": str(self.workgroup_size_x) + "/" + str(self.workgroup_size_y),
            "Load_A": TF(self.load_A),
            "Load_B": TF(self.load_B),
            "Store_D": self.store,
            "PF": TF(self.prefetch)
            + "/"
            + str(self.prefetchInFlight)
            + "/"
            + str(self.prefetchLDSFactor),
            "SCH": self.scheduler[0],
            "SK": self.streamK + "/" + str(self.numWGs),
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
    instructions: str = "simple_mi"

    numWarmUp: int = 2
    numRuns: int = 10

    @property
    def id(self):
        return get_dataclass_id(self)

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

    output: Path = field(repr=False, default=None, hash=False, compare=False)

    @property
    def group(self):
        return "codegen"

    def set_output(self, path: Path):
        self.output = path

    def command(self) -> list[str]:
        retval = [
            "client/rocroller-codegen-stress",
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

    config: Path = field(repr=False, default=None, hash=False, compare=False)
    output: Path = field(repr=False, default=None, hash=False, compare=False)
    tensile_commit: str = "rocm-6.0.0"

    @property
    def group(self):
        return "gemm"

    def set_output(self, path: Path):
        self.output = path

    def command(self, **extra_args) -> list[str]:
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
    "GEMM": GEMMResult,
    "CodeGen": CodeGenResult,
}

_base_to_run_class = {
    GEMM: GEMMRun,
    CodeGen: CodeGenRun,
}


def _is_nested_gemm_result(result: dict) -> bool:
    return any(key in result for key in ("problem", "solution", "benchmark"))


def _flatten_nested_gemm_result(result: dict) -> dict:
    """
    Flatten nested GEMM result sections into legacy top-level keys.

    Merge order matches the old writer's effective behavior:
    problem -> solution -> benchmark, where later sections override earlier values.
    """
    flattened = {
        key: value
        for key, value in result.items()
        if key not in ("problem", "solution", "benchmark")
    }

    for key in ("problem", "solution", "benchmark"):
        section = result.get(key)
        if isinstance(section, dict):
            flattened.update(section)

    return flattened


def cast_missing_parameters(result):
    """
    Cast parameters in previous GEMMResult version into existing parameters

    Args:
        result: a dictionary with parameters (keys) and their values

    """
    if "workgroupMapping" in result:

        assert (
            len(result["workgroupMapping"]) == 2
        ), "workgroupMapping should contain a dimension and a value"

        wgmDim = result["workgroupMapping"][0]
        wgmValue = result["workgroupMapping"][1]

        del result["workgroupMapping"]

        result["workgroupMappingDim"] = wgmDim
        result["workgroupMappingValue"] = wgmValue

    # Remove old/deprecated
    for attr in ["unroll_x", "unroll_y"]:
        if attr in result:
            del result[attr]

    if "storeLDS_D" in result:
        storeLDS_D = result["storeLDS_D"]
        del result["storeLDS_D"]
        result["store"] = (
            "VGPRToGlobalMemoryViaLDSWithBuffer"
            if storeLDS_D
            else "VGPRToGlobalMemoryWithBuffer"
        )

    # Convert old streamK bool fields to new streamK string enum
    if "streamKTwoTile" in result or "streamKTwoTileDPFirst" in result:
        old_streamK = result.get("streamK", False)
        old_twoTile = result.pop("streamKTwoTile", False)
        old_dpFirst = result.pop("streamKTwoTileDPFirst", False)

        if old_twoTile:
            result["streamK"] = "TwoTile"
        elif old_dpFirst:
            result["streamK"] = "TwoTileDPFirst"
        elif old_streamK:
            result["streamK"] = "Standard"
        else:
            result["streamK"] = "None"

    if "matchMemoryAccess" in result:
        del result["matchMemoryAccess"]


def load_results(path: Path):
    """
    Load results from a YAML file `path` and return an array of RESULT objects.
    """
    rv = []
    for r in yaml.load_all(path.read_text(), Loader=yaml.FullLoader):
        if r is None:
            continue

        r = dict(r)
        ResultClass = _client_to_result_class[r["resultType"]]
        if r.get("resultType") == "GEMM" and _is_nested_gemm_result(r):
            r = _flatten_nested_gemm_result(r)
        r.pop("path", None)
        cast_missing_parameters(r)
        rv.append(ResultClass(path=path, **r))
    return rv


def upcast_to_run(obj):
    """Upcast a BASE object to a RUN object."""
    DownClass = type(obj)
    UpClass = _base_to_run_class[DownClass]
    return UpClass(**field_dict(DownClass, obj))
