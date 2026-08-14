# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""
Library operations module.

This module provides tools to manage and manipulate Tensile solution libraries including
loading, merging, and creating optimized GEMM solution libraries. It handles YAML
manipulation, solution library operations, and integration with the Tensile framework.

The operations module enables the final step of the optimization workflow by merging
individual optimized solutions into hipBLASLt libraries.

Functions:
    load_library(path) -> Library
        Load a single library from a YAML file.
    load_collection(lib_dir) -> LibraryCollection
        Load all YAML libraries in a folder into a collection.
    merge_solutions(input_dir) -> LibraryCollection
        Merge optimized solutions from multiple build directories into single libraries.
    extract_solutions(df, match_table_path) -> LibraryCollection
        Build library from DataFrame with GEMMs and solution indices.
    merge(hipblaslt_path, orig_dir, inc_dir, output_dir) -> None
        Merge incremental library into original using TensileMergeLibrary.
    create(hipblaslt_path, input_dir, output_dir) -> None
        Create a Tensile library from YAML library logic files using TensileCreateLibrary.
    from_dataframe(df, lib_dir) -> LibraryCollection
        Create filtered libraries from a DataFrame of selected solutions.
    prune_library(hipblaslt_path, base_lib) -> Library
        Prune a library to its minimum required solutions, without losing performance.

Example:
    >>> from library import operations
    >>> lib = operations.load_library("path/to/lib.yaml")
    >>> collection = operations.load_collection("path/to/libs")
    >>> lib = operations.from_dataframe(df, "path/to/libs")
"""

import yaml
import logging
import copy
import pandas as pd
import glob

from pathlib import Path
from typing import List, Tuple, Sequence
from threading import Lock

from geko.utils import run_silent_command, parse_devices
from geko.concurrency import parallel_for
from geko.concurrency.runner import Runner, Worker
from geko.library import Library, LibraryCollection
from geko.constants import GEMM_LOG_FIELDS, INDEX_TYPE_MAP
from geko.library import _bank

logger = logging.getLogger("GEKO")

try:
    SafeLoader = yaml.CSafeLoader
except (ModuleNotFoundError, AttributeError):
    logger.warning("yaml.CSafeLoader not found, using yaml.SafeLoader")
    logger.warning("Please update the yaml package to the latest version")
    logger.warning("You can do this by running: pip install --upgrade pyyaml")
    SafeLoader = yaml.SafeLoader


__all__ = [
    "load_library",
    "load_collection",
    "merge_solutions",
    "extract_solutions",
    "merge",
    "create",
    "normalize",
    "from_dataframe",
    "from_full_dataframe",
    "prune_library",
]


def load_library(path: str | Path) -> Library:
    """Load a YAML library logic file.

    Args:
        path (str | Path): Path to YAML library logic file.

    Returns:
        Library: Library data structure.
    """
    logger.debug(f"Loading library file: {path}")
    with open(path) as f:
        data = yaml.load(f, Loader=SafeLoader)
    return Library(data, Path(path).name)


def load_collection(lib_dir: str | Path) -> LibraryCollection:
    """Load all YAML library logic files in a directory.

    Args:
        lib_dir (str | Path): Path to the directory containing YAML library logic files.

    Returns:
        LibraryCollection: LibraryCollection with all loaded libraries.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    lib_dir = Path(lib_dir)
    if not lib_dir.is_dir():
        raise FileNotFoundError(f"Folder '{lib_dir}' does not exist")

    paths = sorted(lib_dir.glob("*.yaml"))
    logger.debug(f"Loading library collection from {lib_dir} (n_files={len(paths)})")
    libs = parallel_for(load_library, paths)
    return LibraryCollection(libs)


def _library_supports_epilogues(lib: Library) -> bool:
    """Return False for f64 (DGEMM) libraries; Tensile epilogues are not supported."""
    data_type = lib.problem.get("DataType")
    return INDEX_TYPE_MAP.get(data_type) != "f64_r"


def merge_solutions(
    input_dir: str | Path,
    epilogues: bool = True,
    pattern: str = "**/3_LibraryLogic/*.yaml",
    trim_size: bool = True,
) -> LibraryCollection:
    """Merge optimized solutions from multiple build directories into single libraries.

    Processes all 3_LibraryLogic/*.yaml files in build_* subdirectories,
    merging solutions by library type and deduplicating based on performance.

    Args:
        input_dir (str | Path): Directory containing build_* subdirectories with optimized solutions.
        epilogues (bool, optional): Whether to add epilogue support to merged libraries.
            Defaults to True. DGEMM (f64) libraries are always skipped.
        pattern (str, optional): Search pattern for libraries in the given input dir.
            Defaults to Tensile format.
        trim_size (bool, optional): Whether to use [M, N, B, K] format or to keep leading dimensions.
            Defaults to True.

    Returns:
        LibraryCollection: LibraryCollection with all merged solutions.

    Raises:
        ValueError: If no valid YAML library logic files found.

    Note:
        For duplicate problem sizes, keeps the solution with highest performance.
        Updates solution indices and trims unused solutions from final libraries.
    """
    logger.info(f"Merging solutions in '{input_dir}' into a single library")

    input_dir = Path(input_dir)

    lib_paths = sorted(input_dir.glob(pattern))
    if not lib_paths:
        raise ValueError(f"No valid YAML libraries found in '{input_dir}'")
    logger.debug(f"Merging solutions, discovered {len(lib_paths)} library files")

    # update dependant parameters if StaggerU == 0
    def _sanitize_solution(sol):
        if "ProblemType" in sol:
            del sol["ProblemType"] # remove ProblemType from solution
        if sol.get("StaggerU") == 0:
            sol["StaggerUMapping"] = 0
            sol["StaggerUStride"] = 0
            sol["_staggerStrideShift"] = 0

    libs: dict[str, Library] = {}
    for lib in sorted(parallel_for(load_library, lib_paths), key=lambda k: k.name):
        if lib.name not in libs:
            for size in lib.sizes:
                if trim_size:
                    size[0] = size[0][:4]
            libs[lib.name] = lib
            continue

        if libs[lib.name].default_solution != lib.default_solution:
            raise NotImplementedError(f"Default solution mismatch for library '{lib.name}'. "
                                      f"Use TensileMergeLibrary instead.")

        base_sols = libs[lib.name].solutions
        base_sizes = libs[lib.name].sizes
        for i, sol in enumerate(lib.solutions):
            sizes = [sz for sz in lib.sizes if sz[1][0] == i]
            sol = copy.deepcopy(sol)
            _sanitize_solution(sol)
            sol["SolutionIndex"] = len(base_sols)
            base_sols.append(sol)
            for size in sizes:
                size = copy.deepcopy(size)
                if trim_size:
                    size[0] = size[0][:4]
                size[1][0] = sol["SolutionIndex"]
                base_sizes.append(size)

    if len(libs) == 0:
        raise ValueError(f"No valid YAML libraries found in '{input_dir}'")

    merged = LibraryCollection()
    for _, lib in libs.items():  # If this is ever slow, we can use parallel_for
        lib.trim()  # Remove duplicate sizes/solutions
        if epilogues and _library_supports_epilogues(lib):
            lib.add_epilogues()
        merged.append(lib)
        logger.info(f"Processed {len(lib.solutions)} solutions and {len(lib.sizes)} sizes for '{lib.name}'")

    return merged


def extract_solutions(df: pd.DataFrame, match_table_path: str | Path) -> LibraryCollection:
    """Extract solutions by solution index from a DataFrame.

    Processes all GEMMs and their corresponding 'winning' solutions,
    merging solutions by library type.

    Args:
        df (DataFrame): DataFrame containing GEMM and winning solution index pairs.
        match_table_path (str | Path): Path to MatchTable.yaml to extract the solutions.

    Returns:
        LibraryCollection: LibraryCollection with all exctractes solutions.

    Raises:
        ValueError: If no valid YAML library logic files found.

    Note:
        All new libraries are exported as 'Equality'.
    """
    logger.info(f"Extracting solutions into a single library")
    if "solutionIdx" not in df.columns:
        raise ValueError(f"Missing 'solutionIdx' field in input DataFrame")

    df["solutionIdx"] = df["solutionIdx"].astype(int)
    df = df.sort_values(["m", "n", "batch_count", "k"])

    with open(match_table_path) as f:
        match_table = yaml.load(f, Loader=SafeLoader)

    lib_paths = set()
    for idx in df["solutionIdx"]:
        lib_paths.add(match_table[idx][0])
    logger.debug(f"Extracting from {len(lib_paths)} source libraries")

    lib_paths = sorted(lib_paths)
    libs: dict[str, Library] = {}
    for lib, lib_path in zip(parallel_for(load_library, lib_paths), lib_paths):
        mapping = {}
        for _, row in df.iterrows():
            if match_table[row["solutionIdx"]][0] != lib_path:
                continue
            kidx = match_table[row["solutionIdx"]][1]
            if kidx not in mapping:
                mapping[kidx] = []
            mapping[kidx].append([row["m"], row["n"], row["batch_count"], row["k"]])

        if lib.name not in libs:  # First library for the given GEMM type
            libs[lib.name] = lib
            lib.order = [2, 3, 0, 1]
            lib.metric = "DeviceEfficiency"
            lib.type = "Equality"
            new_sols = []
            new_sizes = []
        else:
            if libs[lib.name].default_solution != lib.default_solution:
                raise NotImplementedError(f"Default solution mismatch for library '{lib.name}'. "
                                          f"Use TensileMergeLibrary instead.")
            new_sols = libs[lib.name].solutions
            new_sizes = libs[lib.name].sizes

        for kidx, sizes in mapping.items():
            sol = copy.deepcopy(lib.solutions[kidx])
            sol["SolutionIndex"] = len(new_sols)
            sizes = [[size, [sol["SolutionIndex"], 0.0]] for size in sizes]
            new_sols.append(sol)
            new_sizes.extend(sizes)
        logger.debug(
            f"Extracted library '{lib.name}': n_solutions={len(new_sols)} "
            f"n_sizes={len(new_sizes)}"
        )

        lib.solutions = new_sols
        lib.sizes = new_sizes

    return LibraryCollection(list(libs.values()))


def merge(
    hipblaslt_path: str | Path,
    orig_dir: str | Path,
    inc_dir: str | Path,
    output_dir: str | Path,
    eff: bool = False,
    force: bool = True,
) -> None:
    """Merge incremental library into original using TensileMergeLibrary.

    Args:
        hipblaslt_path (str | Path): Path to hipBLASLt installation.
        orig_dir (str | Path): Directory containing original Tensile library.
        inc_dir (str | Path): Directory containing incremental/tuned library.
        output_dir (str | Path): Output directory for merged library.
        eff (bool, optional): Whether to set efficiency calculations.
            Defaults to False for no_eff.
        force (bool, optional): Whether to force merge of duplicated sizes.
            Defaults to True.

    Raises:
        FileNotFoundError: If hipBLASLt path does not exist
    """
    hipblaslt_path = Path(hipblaslt_path)
    if not hipblaslt_path.is_dir():
        raise FileNotFoundError(f"hipBLASLt path not found: '{hipblaslt_path}'")

    logger.info(f"Calling TensileMergeLibrary on '{inc_dir}'")
    cmd = [str(hipblaslt_path / "tensilelite/Tensile/bin/TensileMergeLibrary")]
    if not eff:
        cmd += ["--no_eff"]
    cmd += ["--force_merge", str(force), orig_dir, inc_dir, output_dir]
    run_silent_command(cmd)


def create(hipblaslt_path: str | Path, library_dir: str | Path, output_dir: str | Path, version: str = "5") -> None:
    """Create a Tensile library from YAML library logic files using TensileCreateLibrary.

    Args:
        hipblaslt_path (str | Path): Path to hipBLASLt installation.
        library_dir (str | Path): Directory containing YAML library logic files.
        output_dir (str | Path): Output directory for created library.
        version (str, optional): Code object version to use. Defaults to "5".

    Raises:
        ValueError: If no valid YAML library logic files found.
        FileNotFoundError: If hipBLASLt path does not exist.
    """
    hipblaslt_path = Path(hipblaslt_path)
    if not hipblaslt_path.is_dir():
        raise FileNotFoundError(f"hipBLASLt path not found: '{hipblaslt_path}'")

    library_dir = Path(library_dir)
    lib_paths = list(library_dir.glob("**/*.yaml"))
    if len(lib_paths) == 0:
        raise ValueError(f"No valid libraries found in '{library_dir}'")

    arch = load_library(lib_paths[0]).arch

    logger.info(f"Calling TensileCreateLibrary on '{library_dir}'")
    run_silent_command(
        [
            str(hipblaslt_path / "tensilelite/Tensile/bin/TensileCreateLibrary"),
            "--code-object-version",
            version,
            "--library-format",
            "msgpack",
            "--architecture",
            arch,
            library_dir.resolve(),
            output_dir,
            "HIP",
        ]
    )

def normalize(library_path: str | Path, output_path: str | Path, hipblaslt_path: ( str | Path) | None = None) -> None:
    """Normalize a Tensile library using TensileNormalizeLibrary.

    Args:
        library_path (str | Path): Path to the input library.
        output_path (str | Path): Path to the output normalized library.
        hipblaslt_path (str | Path, optional): Path to hipBLASLt installation. 
            If set, will append the path to sys.path to find Tensile.

    Raises:
        FileNotFoundError: If library path does not exist.
    """
    library_path = Path(library_path)
    if not library_path.is_file():
        raise FileNotFoundError(f"Library path not found: '{library_path}'")

    if hipblaslt_path is not None:
        hipblaslt_path = Path(hipblaslt_path)
        if not hipblaslt_path.is_dir():
            raise FileNotFoundError(f"hipBLASLt path not found: '{hipblaslt_path}'")
        import sys
        sys.path.append(str(hipblaslt_path / "tensilelite"))

    try:
        from Tensile import LibraryIO
        from Tensile.CustomYamlLoader import load_yaml_stream
        from Tensile.TensileMergeLibrary import convertToDict, normalizeDictLibraryLayout
    except ImportError as e:
        raise ImportError(f"Failed to import Tensile. Install it or pass the correct path to hipBLASLt. Error: {e}. ")
    
    data = load_yaml_stream(library_path, SafeLoader)
    if not isinstance(data, list):
        raise ValueError(f"Library file '{library_path}' is not in list format.")

    converted = convertToDict(copy.deepcopy(data), str(library_path))
    normalizeDictLibraryLayout(converted)

    LibraryIO.writeYAML(
        str(output_path),
        converted,
        explicit_start=False,
        explicit_end=False,
        sort_keys=False,
    )
    logger.info(f"Library {library_path.name} normalized and saved to {output_path}")


def from_dataframe(
    df: pd.DataFrame,
    lib_dir: str | Path,
    type_override: str = None,
) -> LibraryCollection:
    """Create filtered libraries from a DataFrame of selected solutions.

    Args:
        df (pd.DataFrame): DataFrame with solution selection results, must include 'lib' column.
        lib_dir (str | Path): Directory containing source library files.
        output_dir (str | Path): Output directory for filtered libraries.
        type_override (str, optional): Library type to override (e.g., "Equality").
            Defaults to None.

    Raises:
        ValueError: If DataFrame missing necessary columns or solution count mismatch.

    Returns:
        LibraryCollection: LibraryCollection with all solutions / sizes in the DataFrame.

    Note:
        Groups solutions by GEMM type and creates separate library files
        with only the selected solutions and their corresponding sizes.
    """
    if "lib" not in df.columns:
        raise ValueError(f"Each GEMM must contain the lib (file) it belongs to")

    df = df.rename({"m": "M", "n": "N", "k": "K"}, axis=1)
    if not all(c in df.columns for c in GEMM_LOG_FIELDS):
        raise ValueError(f"Input DataFrame has missing fields")

    lib_dir = Path(lib_dir)
    n_sols = len(df)

    libs = LibraryCollection()
    for _, group in df.groupby("lib"):  # Each library
        lib = None
        sols = []
        sizes = []
        tuned_sols = {}
        for _, row in group.iterrows():  # Each size
            lib_path = lib_dir / row["lib"]  # We need the lib.name to load the lib

            if lib is None:
                lib = load_library(lib_path)

            size_ = [row["M"], row["N"], row["batch_count"], row["K"]]
            size = [size for size in lib.sizes if size[0] == size_]
            if len(size) == 0:
                raise ValueError(f"{size_} not found in '{lib_path}'")

            size = copy.deepcopy(size[0])
            src_kidx = size[1][0]
            if src_kidx in tuned_sols:
                kidx = tuned_sols[src_kidx]
            else:
                sol = copy.deepcopy(lib.solutions[src_kidx])
                kidx = len(sols)
                sol["SolutionIndex"] = kidx
                sols.append(sol)
                tuned_sols[src_kidx] = kidx
            size[1][0] = kidx
            sizes.append(size)

        lib.solutions = sols
        lib.sizes = sizes

        if type_override:
            lib.type = type_override

        libs.append(lib)

        logger.info(f"Processed {len(sols)} solutions and {len(sizes)} sizes for '{lib.name}'")

    processed = sum(len(lib.sizes) for lib in libs)
    if processed != n_sols:
        raise ValueError(f"Number of processed solutions mismatch: {n_sols} vs {processed}")

    return libs


def from_full_dataframe(
    df: pd.DataFrame,
    lib_dir: str | Path,
    match_table_path: str | Path,
    type_override: str = None,
) -> LibraryCollection:
    """Create full libraries from a DataFrame of solutions.
    Args:
        df (pd.DataFrame): DataFrame with solution selection results, must include 'lib' column.
        lib_dir (str | Path): Directory containing source library files.
        match_table_path (str | Path): Path to the MatchTable.yaml file.
        type_override (str, optional): Library type to override (e.g., "Equality").
            Defaults to None.
    Raises:
        ValueError: If DataFrame missing necessary columns or solution count mismatch.
    Returns:
        LibraryCollection: LibraryCollection with all solutions / sizes in the DataFrame.
    Note:
        Groups solutions by GEMM type and creates separate library files
        with the best of tuned, reference solutions and their corresponding sizes.
    """
    if "lib" not in df.columns:
        raise ValueError(f"Each GEMM must contain the lib (file) it belongs to")

    if "solutionIdx_reference" not in df.columns:
        raise ValueError(f"Each GEMM must contain the solutionIdx_reference column")

    df = df.rename({"m": "M", "n": "N", "k": "K"}, axis=1)
    if not all(c in df.columns for c in GEMM_LOG_FIELDS):
        raise ValueError(f"Input DataFrame has missing fields")

    with open(match_table_path) as f:
        match_table = yaml.load(f, Loader=DEFAULT_YAML_LOADER)

    lib_dir = Path(lib_dir)
    n_sols = len(df)

    ref_libs = {}
    libs = LibraryCollection()
    for _, group in df.groupby("lib"):  # Each library
        lib = None
        sols = []
        sizes = []
        tuned_sols = {}
        ref_sols = {}
        for _, row in group.iterrows():  # Each size
            lib_path = lib_dir / row["lib"]  # We need the lib.name to load the lib

            if lib is None:
                lib = load_library(lib_path)

            size_ = [row["M"], row["N"], row["batch_count"], row["K"]]
            kidx = len(sols)
            if row["winner"] == "tuned":
                size = [size for size in lib.sizes if size[0] == size_]
                if len(size) == 0:
                    raise ValueError(f"{size_} not found in '{lib_path}'")

                size = copy.deepcopy(size[0])
                src_kidx = size[1][0]
                if src_kidx in tuned_sols:
                    kidx = tuned_sols[src_kidx]
                else:
                    kidx = len(sols)
                    tuned_sols[src_kidx] = kidx
                    sol = copy.deepcopy(lib.solutions[src_kidx])
                    sol["SolutionIndex"] = kidx
                    sols.append(sol)
            elif row["winner"] == "reference":
                ref_gsi = int(row["solutionIdx_reference"])
                if ref_gsi not in match_table:
                    raise ValueError(f"Reference solution index {ref_gsi} not found in MatchTable")

                size = [copy.copy(size_), [0, 0.0]]
                if ref_gsi not in ref_sols:  # New solution, add to the list
                    kidx = len(sols)
                    ref_sols[ref_gsi] = kidx

                    ref_lib_path, ref_sol_idx = match_table[ref_gsi]
                    if ref_lib_path not in ref_libs:
                        ref_libs[ref_lib_path] = load_library(ref_lib_path)
                    ref_lib = ref_libs[ref_lib_path]

                    sol = copy.deepcopy(ref_lib.solutions[ref_sol_idx])
                    sol["SolutionIndex"] = kidx

                    sols.append(sol)
                else:  # Existing solution, retrieve its index
                    kidx = ref_sols[ref_gsi]
            else:
                raise ValueError(f"Unknown winner type '{row['winner']}' for size {size_}")

            size[1][0] = kidx
            sizes.append(size)

        lib.solutions = sols
        lib.sizes = sizes

        if type_override:
            lib.type = type_override

        libs.append(lib)
        logger.info(f"Processed {len(sols)} solutions and {len(sizes)} sizes for '{lib.name}'")

    processed = sum(len(lib.sizes) for lib in libs)
    if processed != n_sols:
        raise ValueError(f"Number of processed solutions mismatch: {n_sols} vs {processed}")

    return libs
    

def prune_library(
    hipblaslt_path: str | Path,
    base_lib: Library,
    workdir: str | Path = ".",
    cluster: bool = True,
    devices: Sequence[int] | None = None,
    tol: float = 0.02,
    scale_tol: bool = True,
    other_keys: Sequence[str] = None,
):
    """Prune a library to its minimum required solutions, without losing performance 
       (remove similar solutions to reduce the library sizes).

    Clusters the solutions by MacroTile and DepthU, then performs a dense benchmark of solution-size
    for each cluster (all sizes with the same MacroTile and DepthU against all other solutions with the 
    same MacroTile and DepthU in the cluster). Trims the solutions while guaranteeing that no
    size loses performance higher than 'tol' parameter.

    Args:
        hipblaslt_path (str | Path): Path to hipBLASLt installation.
        base_lib (Library): Input Library logic. The library type should be equality or gridbased, 
            with solutions and sizes.
        workdir (str | Path, optional): Working directory for intermediate files and configs.
            Defaults to '.'.
        cluster (bool, optional): Whether to cluster the solutions by MacroTile and DepthU. 
            If false, each size will be benchmarked against all other solutions in the library.
            Potentially slower, but more accurate.
            Defaults to True.
        devices (Sequence[int]): List of GPU device IDs to use. 
            Defaults to None, which is interpreted as all 8 devices if not specified.
        tol (float, optional): Performance error threshold to consider a solution 'optimal' for a size.
            Defaults to 0.02 (2%).
        scale_tol (bool, optional): Whether to scale ``tol`` using the cluster's
            average FLOPs (inverse-saturating behavior). Defaults to True.
        other_keys (Sequence[str], optional): Additional parameters to cluster on (if cluster is True).
            Defaults to None.

    Returns:
        Library: Trimmed Library data structure.

    Raises:
        FileNotFoundError: If hipBLASLt path or log file doesn't exist.
        TypeError: If base_lib is not of type 'Library'.
        ValueError: If base_lib does not have any sizes.

    Note:
        - If clustering is not enabled, only one GPU will be used.
        - Will re-use benchmark results if they exist.
    """
    hipblaslt_path = Path(hipblaslt_path)
    if not hipblaslt_path.is_dir():
        raise FileNotFoundError(f"hipBLASLt path not found: '{hipblaslt_path}'")

    if devices is None:
        devices = list(range(8))  # Default to all 8 devices if not specified
    devices = parse_devices(devices)
    workdir = Path(workdir)

    if not isinstance(base_lib, Library):
        raise TypeError(f"Must be of type 'Library'")

    if base_lib.sizes is None:
        raise ValueError(f"Only libraries with size-solution mappings are supported (equality or gridbased library logic)")

    logger.info(f"Working on library '{base_lib.name}'")

    if cluster:
        MTDU = _bank.cluster_solutions(base_lib, other_keys=other_keys)
    else:
        logger.warning(f"Clustering is disabled, benchmarking all solutions against each other. This may be very slow for large libraries.")
        MTDU = {"all": {"solutions": base_lib.solutions, "sizes": base_lib.sizes}}

    logger.info(f"Found {len(MTDU)} clusters")
    logger.info(f"Running with tolerance of {100 * (tol):.2f}%")

    workdir = workdir / Path(base_lib.name).stem
    workdir.mkdir(parents=True, exist_ok=True)

    keys = sorted(list(MTDU.keys()))
    use_single_device_per_worker = len(keys) > 1

    if use_single_device_per_worker:
        logger.info("Multiple clusters detected: each worker will benchmark on its assigned single device")
    else:
        logger.info("Single cluster detected: benchmarking will use all selected devices")

    results_lock = Lock()
    result_counts: dict[str, tuple[int, int]] = {}
    failed_errors: dict[str, Exception] = {}

    def _cluster_name(key: str | Tuple) -> str:
        return f"MT{'x'.join(str(kv) for kv in key)}" if isinstance(key, Tuple) else "all"

    class PruneWorker(Worker[str | Tuple]):
        """Worker that prunes a single solution cluster on one device."""

        def setup(self) -> None:
            self.key = self.item
            self.name = _cluster_name(self.key)
            self.lib = copy.deepcopy(base_lib)
            self.sols = MTDU[self.key]["solutions"]
            self.sizes = MTDU[self.key]["sizes"]
            self.n_before = len(self.sols)

            self.lib.solutions = self.sols
            self.lib.sizes = self.sizes

            self.cluster_dir = workdir / self.name
            self.lib_dir = self.cluster_dir / "lib"
            self.custom_lib_dir = self.cluster_dir / "build"
            self.lib.dump(self.lib_dir)

        def run(self) -> bool:
            if self.n_before < 2:
                with results_lock:
                    result_counts[self.name] = (self.n_before, self.n_before)
                return True

            try:
                if len(list(self.custom_lib_dir.glob("library/**/TensileLibrary_lazy_gfx*.dat"))) == 0:
                    create(hipblaslt_path, self.lib_dir, self.custom_lib_dir)

                new_sols, new_sizes = _bank.min_assigment(
                    hipblaslt_path,
                    self.lib,
                    self.cluster_dir,
                    self.custom_lib_dir,
                    devices=[self.device] if use_single_device_per_worker else devices,
                    tol=tol,
                    scale_tol=scale_tol,
                )

                self.lib.solutions = new_sols
                self.lib.sizes = new_sizes
                self.lib.dump(self.lib_dir)
                with results_lock:
                    result_counts[self.name] = (self.n_before, len(new_sols))
                return True
            except Exception as e:
                with results_lock:
                    failed_errors[self.name] = e
                return False

        def teardown(self) -> None:
            pass

    prev_level = logger.level
    mute_for_progress = logger.getEffectiveLevel() > logging.DEBUG
    if mute_for_progress:
        logger.setLevel(logging.WARNING)

    runner = Runner(
        items=keys,
        worker_impl=PruneWorker,
        devices=devices,
        n_slots=1,
    )

    try:
        completed_keys = runner(workdir)
    finally:
        if mute_for_progress:
            logger.setLevel(prev_level)

    if len(completed_keys) != len(keys):
        failed_names = [_cluster_name(key) for key in keys if key not in set(completed_keys)]
        if len(failed_names) == 1 and failed_names[0] in failed_errors:
            raise failed_errors[failed_names[0]]
        if len(failed_errors) > 0:
            details = "; ".join(f"{name}: {err}" for name, err in failed_errors.items())
            raise RuntimeError(f"Cluster pruning failed for {failed_names}: {details}")
        raise RuntimeError(f"Cluster pruning failed for {failed_names}")

    n_before_all = n_after_all = 0
    for key in keys:
        cn = _cluster_name(key)
        n_before, n_after = result_counts[cn]
        n_before_all += n_before
        n_after_all += n_after
        logger.info(f"'{cn}': removed {n_before - n_after} solutions out of {n_before}")
    logger.info(f"Final lib: removed {n_before_all - n_after_all} solutions out of {n_before_all}")

    merged = merge_solutions(workdir, epilogues=False, pattern="*/lib/*.yaml")
    if len(merged) != 1:
        raise RuntimeError(f"Found {len(merged)} libraries after merging, check the results in {workdir}")

    return merged[0]
