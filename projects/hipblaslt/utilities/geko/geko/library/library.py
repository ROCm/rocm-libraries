# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""
Library module.

This module provides classes for managing library data loaded from
YAML library logic files. It supports creation, modification,
benchmark input generation, and serialization of individual libraries and
collections of libraries.

Classes:
    Library
        Represents a single library with architecture, problem, solutions,
        sizes, indexing order, performance metric and type.
        Provides methods for adding epilogues, generating benchmark inputs,
        and dumping to YAML.

    LibraryCollection
        A container for multiple Library instances. Supports iteration,
        indexing, appending libraries, and forwarding operations like
        'add_epilogues', 'create_bench_input', 'trim' and 'dump' to all
        contained libraries.

Usage example:
    >>> from library import Library, LibraryCollection, operations
    >>> lib = operations.load_library(path/to/gfx950_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml)
    >>> collection = LibraryCollection([lib])
    >>> collection.append(another_lib)
    >>> collection.add_epilogues()
    >>> collection.dump()
"""

import yaml
import copy
import math
import logging
import re

from pathlib import Path
from typing import List, Tuple, Iterator

from geko import bench
from geko.constants import INDEX_TYPE_MAP
from geko.concurrency import parallel_for

__all__ = ["Library", "LibraryCollection"]

BIAS_MAP = {0: "S", 4: "H", 7: "B"}
SAB_MAP = {"Scalar": "SAB", "Vector": "SABV"}
SUPPORTED_INITIALIZATIONS = ("trig_float", "rand_int", "hpl")

logger = logging.getLogger("GEKO")


try:
    DEFAULT_YAML_DUMPER = yaml.CSafeDumper
except (ModuleNotFoundError, AttributeError):
    DEFAULT_YAML_DUMPER = yaml.SafeDumper


class Library:
    """Represents a single Library loaded from a data structure.

    The Library supports both legacy list-format and dict-format logic payloads.
    It provides access to architecture, problem description, solutions, sizes,
    and type, with validation on setters to maintain data integrity.
    """

    def __init__(self, data: List | dict, name: str):
        """Initialize a Library instance with validated data.

        Args:
            data (List | dict): Library logic payload in one of two formats.
                Legacy list format must have at least 9 elements with required
                positions:
                    - data[2]: architecture
                    - data[4]: problem
                    - data[5]: solutions (non-empty list)
                    - data[6]: indexing order
                    - data[7]: sizes
                    - data[10]: performance metric
                    - data[11]: type
                Dict format must contain required keys:
                    - ArchitectureName
                    - ProblemType
                    - DefaultSolution
                    - Solutions (non-empty list)
                    - ExactLogic (sizes)
                    - LibraryType
            name (str): The filename of the library. Must be a non-empty string.

        Raises:
            ValueError: If 'data' is neither list nor dict, is missing required
                        fields for its format, or contains empty solutions.
            ValueError: If 'name' is not a non-empty string.
        """
        if isinstance(data, list):
            self.format = "list"
            if len(data) < 9:
                raise ValueError(f"Library data must have at least 9 elements, got {len(data)}")
            if data[5] is None or len(data[5]) == 0:
                raise ValueError(f"Library logic does not have any Solutions")
        elif isinstance(data, dict):
            self.format = "dict"
            required_keys = ["ArchitectureName", "ProblemType", "DefaultSolution", "Solutions", "LibraryType"]
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Library data must contain key '{key}'")
            if data["Solutions"] is None or len(data["Solutions"]) == 0:
                raise ValueError(f"Library logic does not have any Solutions")
        else:
            raise ValueError(f"Library logic must be of type 'list' or 'dict', got {type(data)}")
        
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError(f"'name' must be of type 'str' and non-empty")
        
        self.data = data
        self.name = name

    def _get(self, idx: int, dict_key: str):
        if self.format == "dict":
            return self.data.get(dict_key)
        if len(self.data) <= idx:
            return None
        return self.data[idx]

    def _set(self, idx: int, dict_key: str, value) -> None:
        if self.format == "dict":
            self.data[dict_key] = value
        else:
            if len(self.data) <= idx:
                self.data.extend([None] * (idx - len(self.data) + 1))
            self.data[idx] = value

    @property
    def arch(self) -> str:
        """Get the architecture of the library.

        Returns:
            str: Target GPU architecture (e.g., "gfx950", "gfx942").
        """
        if self.format == "dict":
            return self.data.get("ArchitectureName")
        if isinstance(self.data[2], dict):
            return self.data[2].get("Architecture")
        return self.data[2]

    @property
    def problem(self) -> dict:
        """Get the problem description of the library.

        Returns:
            dict: Problem specification including GEMM parameters and configuration.
        """
        return self._get(4, "ProblemType")

    @property
    def solutions(self) -> List[dict]:
        """Get the list of solutions in the library.

        Returns:
            List[dict]: List of solution dictionaries with kernel configurations.
        """
        return self._get(5, "Solutions")

    @solutions.setter
    def solutions(self, val: List) -> None:
        """Set the solutions list.

        Args:
            val (List): New solutions list.

        Raises:
            TypeError: If 'val' is not a list.
        """
        if not isinstance(val, list):
            raise TypeError("Must be a list")
        self._set(5, "Solutions", val)

    @property
    def order(self) -> List[int]:
        """Get the indexing order.

        Returns:
            List: List of integers containing the indexing order.
        """
        return self._get(6, "IndexOrder")

    @order.setter
    def order(self, val: List[int]) -> None:
        """Set the indexing order list.

        Args:
            val (List[int]): New indexing order list.

        Raises:
            TypeError: If 'val' is not a list.
        """
        if not isinstance(val, list):
            raise TypeError("Must be a list")
        self._set(6, "IndexOrder", val)

    @property
    def sizes(self) -> List:
        """Get the list of problem sizes in the library.

        Returns:
            List: List of tuples containing size specifications and performance data.
        """
        return self._get(7, "ExactLogic")

    @sizes.setter
    def sizes(self, val: List) -> None:
        """Set the sizes list.

        Args:
            val (List): New sizes list.

        Raises:
            TypeError: If 'val' is not a list.
        """
        if not isinstance(val, list):
            raise TypeError("Must be a list")
        self._set(7, "ExactLogic", val)

    @property
    def metric(self) -> str:
        """Get the library performance metric identifier.

        Returns:
            str: Library performance  specification.
        """
        return self._get(10, "PerfMetric")

    @metric.setter
    def metric(self, val: str) -> None:
        """Set the performance metric of the library.

        Args:
            val (str): New performance metric.

        Raises:
            TypeError: If 'val' is not a string.
        """
        if not isinstance(val, str):
            raise TypeError("Must be a string")
        self._set(10, "PerfMetric", val)

    @property
    def type(self) -> str:
        """Get the library type identifier.

        Returns:
            str: Library type specification.
        """
        return self._get(11, "LibraryType")

    @type.setter
    def type(self, val: str) -> None:
        """Set the type of the library.

        Args:
            val (str): New type.

        Raises:
            TypeError: If 'val' is not a string.
        """
        if not isinstance(val, str):
            raise TypeError("Must be a string")
        self._set(11, "LibraryType", val)
    
    @property
    def default_solution(self) -> str:
        """Get the library default solution.

        Returns:
            dict: Library default solution.
        """
        return self._get(12, "DefaultSolution")

    @default_solution.setter
    def default_solution(self, val: dict) -> None:
        """Set the default solution of the library.

        Args:
            val (dict): New default solution.

        Raises:
            TypeError: If 'val' is not a dict.
        """
        if not isinstance(val, dict):
            raise TypeError("Must be a dict")
        self._set(12, "DefaultSolution", val)

    def add_epilogues(self) -> None:
        """Add epilogue support (bias, activation, scaling) to a library.

        Modifies the library to support activation functions, bias vectors,
        and scaling operations by updating solution names and library metadata.

        Note:
            Assumes no epilogues were previously added. Updates all solution
            names with appropriate suffixes and sets library flags.
        """
        _NO_EPILOGUE_DTYPES = (1, 2, 3)  # f64_r, f32_c, f64_c
        if self.problem.get("DataType") in _NO_EPILOGUE_DTYPES:
            return

        # This assumes no epilogues were added before, otherwise this may create inconsistencies

        self.problem["Activation"] = True
        self.problem["ActivationType"] = "hipblaslt_all"
        self.problem["UseBias"] = 1

        # If DestDataType is f8/b8 use [0, 4, 7]
        bias_list = [0, 4, 7] if self.problem["DestDataType"] > 10 else [0, self.problem["DestDataType"]]
        self.problem["BiasDataTypeList"] = sorted(set(bias_list))
        self.problem["UseScaleAlphaVec"] = 1

        # If f8/b8
        if self.problem["DataType"] > 10 or self.problem["DestDataType"] > 10:
            self.problem["UseScaleAB"] = "Scalar"

        sab = SAB_MAP.get(self.problem["UseScaleAB"], None)
        btype = "".join([BIAS_MAP[bt] for bt in self.problem["BiasDataTypeList"]])

        epilogue_suffix = f"_Bias{btype}_HAS{('_' + sab) if sab is not None else ''}_SAV_UserArgs"

        def update_epilogue_string(s: str, suffix: str) -> str:
            if "_UserArgs" not in s:
                return s
            left, right = s.split("_UserArgs", 1)
            pattern = r'_(?:Bias[SBH]*|HAS|HA_S|SABV?|SAV)'
            left = re.sub(pattern, '', left)
            s = left + "_UserArgs" + right
            return s.replace("_UserArgs", suffix)


        for sol in self.solutions:
            sol["ActivationFuncCall"] = False
            for key in ["BaseName", "KernelNameMin", "SolutionNameMin"]:
                if key in sol and epilogue_suffix not in sol[key]:
                    sol[key] = update_epilogue_string(sol[key], epilogue_suffix)

        if epilogue_suffix not in self.name:
            self.name = update_epilogue_string(self.name, epilogue_suffix)

    def create_bench_input(
        self,
        output_dir: str | Path,
        verify: bool = False,
        duration: float = 0.5,
        iters: int = 100,
        cold_iters: int = 100,
        rotating: int = 512,
        beta: bool = True,
        flush: bool = True,
        print_kernel_info: bool = True,
        initialization: str = "trig_float",
    ) -> Tuple[str, str | None]:
        """Generate hipBLASLt benchmark input files for the library.

        Args:
            output_dir (str | Path): Directory to store benchmark files.
            verify (bool, optional): Whether to generate a verification file.
                Defaults to False.
            duration (float, optional): Target benchmark duration in seconds.
                Defaults to 0.5.
            iters (int, optional): Number of benchmark iterations.
                Defaults to 100.
            cold_iters (int, optional): Number of warm-up iterations.
                Defaults to 20.
            rotating (int, optional): Memory rotation parameter.
                Defaults to 512.
            beta (bool, optional): Whether to use non-zero beta values.
                Defaults to True.
            flush (bool, optional): Whether to flush GPU caches.
                Defaults to True.
            print_kernel_info (bool, optional): Whether to print solution information.
                Defaults to True.
            initialization (str, optional): Initialization method (trig_float, rand_int, hpl).
                Defaults to "trig_float".

        Returns:
            Tuple[str, str | None]: Paths to benchmark and verification files
                (verification file is None if 'verify=False').

        Raises:
            ValueError: If 'initialization' is not supported.
        """
        if initialization not in SUPPORTED_INITIALIZATIONS:
            raise ValueError(f"Must be on of {SUPPORTED_INITIALIZATIONS}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ccA = self.problem.get("ComplexConjugateA", False)
        ccB = self.problem.get("ComplexConjugateB", False)
        transA = "C" if (self.problem["TransposeA"] and ccA) else ("T" if self.problem["TransposeA"] else "N")
        transB = "C" if (self.problem["TransposeB"] and ccB) else ("T" if self.problem["TransposeB"] else "N")

        if initialization != "rand_int" and INDEX_TYPE_MAP[self.problem["DataType"]] == "i8_r":
            initialization = "rand_int"
            logger.warning(f"Initialization {initialization} is not allowed for int8 datatype. Changed to rand_int")

        common = dict(
            function="matmul",
            transA=transA,
            transB=transB,
            a_type=INDEX_TYPE_MAP[self.problem["DataType"]],
            b_type=INDEX_TYPE_MAP[self.problem["DataType"]],
            c_type=INDEX_TYPE_MAP[self.problem["DestDataType"]],
            d_type=INDEX_TYPE_MAP[self.problem["DestDataType"]],
            initialization=initialization,
            alpha=1.0,
        )

        if "ComputeDataType" in self.problem:
            compute_type = INDEX_TYPE_MAP[self.problem["ComputeDataType"]]

            _COMPLEX_TO_REAL_COMPUTE = {"f32_c": "f32_r", "f64_c": "f64_r"}
            compute_type = _COMPLEX_TO_REAL_COMPUTE.get(compute_type, compute_type)

            common["scale_type"] = compute_type

            if "F32XdlMathOp" in self.problem and self.problem["F32XdlMathOp"] == 10:  # TF32
                compute_type = "x" + compute_type

            if INDEX_TYPE_MAP[self.problem["DataType"]] == "f8b8":
                common["a_type"] = "f8_r"
                common["b_type"] = "bf8_r"
            elif INDEX_TYPE_MAP[self.problem["DataType"]] == "b8f8":
                common["a_type"] = "bf8_r"
                common["b_type"] = "f8_r"

            common["compute_type"] = "c_" + compute_type
        else:
            if common["a_type"] == "f16_r" and self.problem["HighPrecisionAccumulate"]:
                common["compute_type"] = "c_f32_r"
            elif common["a_type"] == "i8_r": 
                common["compute_type"] = "i32_r"
            else:
                common["compute_type"] = common["a_type"]

        if "F32XdlMathOp" in self.problem and self.problem["F32XdlMathOp"] == 9:  # TF32
            common["math_mode"] = 1

        gemms = []
        latency = []
        for size in self.sizes:
            dims, (_, gflops) = size
            gemm = copy.deepcopy(common)
            gemm.update({"batch_count": dims[2], "M": dims[0], "N": dims[1], "K": dims[3]})
            if len(dims) >= 8:
                gemm.update({"ldc": dims[4], "ldd": dims[5], "lda": dims[6], "ldb": dims[7]})
            gemms.append(gemm)
            latency.append(0 if gflops <= 0 else (2 * math.prod(dims[:4])) / gflops / 1000)  # us

        gemms = bench.log.update(
            gemms,
            latency,
            duration=duration,
            iters=iters,
            cold_iters=cold_iters,
            rotating=rotating,
            beta=beta,
            flush=flush,
            aux=False,
            print_kernel_info=print_kernel_info,
        )[0]

        bench_file = output_dir / (Path(self.name).stem + "_bench.yaml")
        bench.log.dump(gemms, bench_file)

        if not verify:
            return str(bench_file), None

        for gemm in gemms:
            gemm.update({"norm_check": 1, "norm_check_assert": 0, "allclose_check": 1, "iters": 1, "cold_iters": 0})
            if "flush" in gemm:
                del gemm["flush"]
            if "rotating" in gemm:
                del gemm["rotating"]

        verif_file = str(bench_file).replace("_bench.yaml", "_verify.yaml")
        bench.log.dump(gemms, verif_file)

        return str(bench_file), str(verif_file)

    def trim(self) -> None:
        """Remove duplicate solutions for the same problem size, keeping the best performing one.

        For each unique problem size, if multiple solutions exist, only the solution
        with the highest GFLOPS performance is retained. Also removes unused solutions
        that are not referenced by any size and renumbers solution indices.

        Side effects:
            Modifies 'self.sizes' and 'self.solutions' in place.
        """
        # Remove duplicate sizes
        size_map = {
            tuple(size[0]): [sidx for sidx, sz in enumerate(self.sizes) if sz[0] == size[0]] for size in self.sizes
        }
        keep = set()
        for size, indices in size_map.items():
            if len(indices) == 1:
                keep.add(indices[0])
                continue
            keep_idx = max(indices, key=lambda sidx: self.sizes[sidx][1][1])
            keep.add(keep_idx)
            perf = self.sizes[keep_idx][1][1]
            logger.info(
                f"{self.name} | {len(indices)} solutions found for {size}, keeping solution with highest performance: {perf}"
            )
        self.sizes = [self.sizes[keep_idx] for keep_idx in keep]

        # Trim unused solutions and invalid sizes
        sol_indices = set([size[1][0] for size in self.sizes])
        if len(sol_indices) != len(self.solutions) or max(sol_indices) > len(self.solutions):
            new_sols = []
            new_sizes = []
            for i, sol in enumerate(self.solutions):
                sizes = [sz for sz in self.sizes if sz[1][0] == i]
                if len(sizes) == 0:
                    continue
                kidx = len(new_sols)
                sol = copy.deepcopy(sol)
                sol["SolutionIndex"] = kidx
                sizes = copy.deepcopy(sizes)
                for size in sizes:
                    size[1][0] = kidx
                new_sols.append(sol)
                new_sizes.extend(sizes)

            if len(new_sizes) < len(self.sizes):
                logger.info(
                    f"{self.name} | Removed {len(self.sizes) - len(new_sizes)} sizes that point to non-existing solutions"
                )
            if len(new_sols) < len(self.solutions):
                logger.info(f"{self.name} | Removed {len(self.solutions) - len(new_sols)} unused solutions")

            self.solutions = new_sols
            self.sizes = new_sizes

    def dump(self, output_dir, name: str | None = None) -> None:
        """Write the library data to a YAML file.

        Args:
            output_dir (str | Path): Directory to store the YAML file.
            name (str | None, optional): File name. Defaults to the library's 'name'.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = name or self.name
        
        with open(output_dir / name, "w") as f:
            yaml.dump(self.data, f, default_flow_style=None, Dumper=DEFAULT_YAML_DUMPER, sort_keys=False)


class LibraryCollection:
    """A collection of Library objects.

    Provides iteration, indexing, length checking, and bulk operations
    like adding epilogues, generating benchmark inputs, or dumping to YAML.

    Example:
        >>> collection = LibraryCollection([lib1, lib2])
        >>> collection.add_epilogues()
        >>> collection.create_bench_input(output_dir="./benchmarks")
    """

    def __init__(self, libs: List[Library] = None):
        """Initialize a LibraryCollection.

        Args:
            libs (List[Library], optional): Initial list of Library objects.
                Defaults to empty list.

        Raises:
            TypeError: If 'libs' is not a list or contains non-Library elements.
        """
        if not libs:
            libs = []

        if not isinstance(libs, list):
            raise TypeError(f"Must be of type 'list'")

        if len(libs) > 0 and not all(isinstance(lib, Library) for lib in libs):
            raise TypeError(f"All libraries must be of type 'Library'")

        self.libs = libs

    def __iter__(self) -> Iterator[Library]:
        return iter(self.libs)

    def __getitem__(self, index) -> Library:
        return self.libs[index]

    def __len__(self) -> int:
        return len(self.libs)

    def append(self, lib: Library) -> None:
        """Append a Library to the collection.

        Args:
            lib (Library): Library to add.

        Raises:
            ValueError: If 'lib' is not a Library instance.
        """
        if not isinstance(lib, Library):
            raise ValueError(f"Must be of type 'Library'")

        self.libs.append(lib)

    def add_epilogues(self) -> None:
        """Call 'add_epilogues' on each Library in the collection."""
        for lib in self.libs:
            lib.add_epilogues()

    def create_bench_input(self, *args, **kwargs) -> None:
        """Call 'create_bench_input' on each Library in the collection.

        All positional and keyword arguments are forwarded to
        'Library.create_bench_input'.

        Args:
            *args: Positional arguments forwarded to 'Library.create_bench_input'.
            **kwargs: Keyword arguments forwarded to 'Library.create_bench_input'.
        """
        for lib in self.libs:
            lib.create_bench_input(*args, **kwargs)

    def trim(self) -> None:
        """Call 'trim' on each Library in the collection."""
        for lib in self.libs:
            lib.trim()

    def dump(self, output_dir: str | Path) -> None:
        """Call 'dump' on each Library in the collection.

        Args:
            output_dir (str | Path): Directory to store YAML files for all libraries.
        """
        parallel_for(lambda lib: lib.dump(output_dir), self.libs)
