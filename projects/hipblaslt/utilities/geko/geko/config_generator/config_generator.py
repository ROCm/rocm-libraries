# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import sys
import copy
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

from geko.config_generator import get_optimization_params, get_post_processor
from geko.config_generator.cluster_sizes import do_cluster
from geko.config_generator.config_merger import do_merge
from geko.config_generator.config_sections_generator import ConfigSectionGenerator
from geko.config_generator.fork_param_generator import generate_fork_params
from geko.config_generator.mi_designer import MIDesign
from geko.config_generator.output_writer import EntityOutputWriter
from geko.config_generator.shared_utils import ConfigEntry
from geko.utils import build_tensilelite_client
from geko.concurrency import parallel_for

logger = logging.getLogger("GEKO")


def run(
    config: Dict[str, Any],
    hipblaslt_path: str | Path,
    output_path: str | Path,
    *,
    write_shell_scripts: bool = True,
) -> None:
    """Run the full generation pipeline for every GEMM problem in the config.

    Iterates over each GemmConfig in
    config["GemmProblems"], sets it as the active config["GemmProblem"],
    and dispatches to _run_per_gemm_type. The tensilelite client is
    built once up front and reused across problems when shell scripts are being
    written.

    Args:
        config: Prepared config dict (already passed through
            apply_input_config_defaults).
            Must contain the GemmProblems list along with ARCH-related fields.
        hipblaslt_path: Path to the local hipBLASLt repository
            (tensilelite is added to sys.path).
        output_path: Output directory for generated artifacts.
        write_shell_scripts: If false, emit only YAML and config log
            (no .sh or run-all script).
    """
    hipblaslt_path, output_path = map(Path, (hipblaslt_path, output_path))

    sys.path.insert(0, str(hipblaslt_path / "tensilelite"))

    client_path = None
    if write_shell_scripts:
        client_path = build_tensilelite_client(
            hipblaslt_path, config.get("BUILD_DIR", None)
        )

    for gp in config["GemmProblems"]:
        _config = copy.deepcopy(config)
        _config["GemmProblem"] = gp
        _run_per_gemm_type(
            _config,
            hipblaslt_path,
            output_path,
            write_shell_scripts=write_shell_scripts,
            client_path=client_path,
        )


def _run_per_gemm_type(
    config: Dict[str, Any],
    hipblaslt_path: Path,
    output_path: Path,
    *,
    write_shell_scripts: bool = True,
    client_path: Any = None,
) -> None:
    """Resolve sizes and emit tuning artifacts for the active GEMM problem.

    Runs MI design, fork-parameter generation, clustering / merging, and
    output writing for the GEMM described by config["GemmProblem"].
    Each GEMM type writes its own MI_finder_log/<gemm_name>/ subdirectory
    under output_path.

    Args:
        config: Prepared config dict; config["GemmProblem"] must be set
            to the GemmConfig currently being processed.
        hipblaslt_path: Path to the local hipBLASLt repository.
        output_path: Output directory for generated artifacts.
        write_shell_scripts: If false, emit only YAML and config log
            (no .sh or run-all script).
        client_path: Pre-built tensilelite client path; if None and
            write_shell_scripts is true, the client is built on demand.
    """
    gp = config["GemmProblem"]

    mi_finder_log_path = output_path / "MI_finder_log" / gp.gemm_type.gemm_name
    mi_finder_log_path.mkdir(parents=True, exist_ok=True)

    size_list = gp.sizes
    logger.info(" Total number of sizes: %s", len(size_list))

    # --- Create MI designer, optimization params, and post-processor ---
    mi_designer = MIDesign(mi_finder_log_path, config)
    opt_params = get_optimization_params(config)
    post_processor = get_post_processor(config)

    config["LOG_LEVEL"] = logger.getEffectiveLevel()

    # --- Generate fork parameters for each size -----------------------
    fork_one = partial(
        _fork_params_entry_for_size,
        mi_designer=mi_designer,
        opt_params=opt_params,
        config=config,
        post_processor=post_processor,
    )
    entries: List[ConfigEntry] = parallel_for(fork_one, size_list)

    # --- Cluster sizes and merge fork params across clusters -----------
    clusters = do_cluster(
        entries, config["CUs"],
        config["CLUSTER"], config["ONE_SIZE_PER_CONFIG"],
    )

    entries = do_merge(
        clusters, entries,
        config['MAX_NUM_KERNELS_PER_CONFIG'],
    )

    GEMM_type = gp.gemm_type.gemm_name

    if write_shell_scripts and client_path is None:
        client_path = build_tensilelite_client(
            hipblaslt_path, config.get("BUILD_DIR", None)
        )

    csg = ConfigSectionGenerator(config)

    cur_working_dir = output_path
    output_writer = EntityOutputWriter(
        cur_working_dir,
        GEMM_type,
        hipblaslt_path,
        client_path=client_path,
        write_shell_scripts=write_shell_scripts,
    )

    # --- Write output configs and scripts (parallel), then ordered logs --
    indexed_entries = list(enumerate(entries))
    n_out = len(entries)
    progress_denom = n_out - 1
    emit_one = partial(
        _emit_entity_files,
        csg=csg,
        output_writer=output_writer,
        gemm_type=GEMM_type,
        config=config,
    )
    parallel_for(emit_one, indexed_entries)
    # Serial pass: shared run-all script and Config_*.log must append in index order.
    for idx, entry in indexed_entries:
        output_writer.append_aggregate_metadata(
            GEMM_type + f"_{idx}",
            entry,
            progress=f"{idx}/{progress_denom}",
        )

# --- Parallel worker targets (used from _run_per_gemm_type) ---

Size4 = Tuple[int, int, int, int]


def _fork_params_entry_for_size(
    size: Size4,
    mi_designer: MIDesign,
    opt_params,
    config: dict,
    post_processor,
) -> ConfigEntry:
    """Build one :class:`~geko.config_generator.shared_utils.ConfigEntry` for a problem size.

    Intended for use with geko.utils.parallel_for (e.g. loky workers).

    Raises:
        ValueError: If no kernels exist for the given size and the GEMM type
            named by config["GemmProblem"].gemm_type.
    """
    logger.setLevel(config.get("LOG_LEVEL", logging.WARNING))
    fork_params, num_mis, nkernels = generate_fork_params(
        mi_designer,
        opt_params,
        config,
        size,
        post_processor=post_processor,
    )
    if nkernels == 0:
        raise ValueError(
            f"No kernels found for GEMM type "
            f"{config['GemmProblem'].gemm_type.gemm_name} "
            f" and size {size}."
        )
    return ConfigEntry(
        sizes=[size],
        fork_params=fork_params,
        nkernels=nkernels,
        mis_per_size={tuple(size): num_mis},
    )


def _emit_entity_files(
    idx_entry: Tuple[int, ConfigEntry],
    csg: ConfigSectionGenerator,
    output_writer: EntityOutputWriter,
    gemm_type: str,
    config: dict,
) -> None:
    """Build tuning YAML (and optional shell script) for one merged cluster entry.

    Writes only per-entity artifacts via :meth:`EntityOutputWriter.write_entity_files_only`.
    Run-all and config log lines are appended serially by the driver after all workers finish.
    """
    logger.setLevel(config.get("LOG_LEVEL", logging.WARNING))
    
    idx, entry = idx_entry
    cat_name = f"_{idx}"
    entity_name = gemm_type + cat_name
    built = csg.build_config(
        entry,
        is_ga=config["GA"],
        config_name=entity_name,
        cms_priority=config["CMS_PRIORITY"],
        soo=config["MACROTILE_OPT"],
    )
    header = csg.generate_comment(entry.nkernels)
    output_writer.write_entity_files_only(entry, built, header, entity_name)
