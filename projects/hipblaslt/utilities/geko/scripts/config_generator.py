# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import logging
from pathlib import Path

from geko import logger
from geko.config_generator.config_generator import run
from geko.config_generator.load_input_config import load_prepared_config_from_yaml
from geko.paths import resolve_hipblaslt_path


def main(
    hipblaslt_path: str | None = None,
    output_path: str = "./",
    verbose: int = 1,
    *,
    config: str | None = None,
    arch: str | None = None,
    gemm_log_path: str | None = None,
    write_shell_scripts: bool = True,
) -> None:
    """Generate tuning YAML (and optional shell scripts) for hipBLASLt.

    Pass a tuning config YAML path, or omit it and pass arch plus gemm_log_path
    for workload-only mode (SIZE_OPTION 2, parse-only GemmProblems).

    Args:
        hipblaslt_path: Path to local hipBLASLt repository. When None, it is
            auto-detected from this script's location (and $GEKO_HIPBLASLT_PATH).
        output_path: Output directory for generated configs. Defaults to "./".
        verbose: Logging verbosity (0=WARNING, 1=INFO).
        config: Tuning YAML path; required unless both arch and gemm_log_path are set.
        arch: ARCH string; with gemm_log_path replaces a tuning YAML.
        gemm_log_path: Workload YAML path (sets GEMM_LOG_PATH / SIZE_OPTION 2 from CLI).
        write_shell_scripts: If false, emit YAML and config log only (no .sh).

    Raises:
        FileNotFoundError: If a required path does not exist.
        ValueError: If arguments are inconsistent or configuration is invalid.
    """
    have_config_path = config is not None
    have_arch_and_gemm_log = arch is not None and gemm_log_path is not None
    if not (have_config_path or have_arch_and_gemm_log):
        raise ValueError(
            "Provide --config PATH to a tuning YAML, or both --arch and --gemm-log-path."
        )

    if config is not None:
        config_path = Path(config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: '{config_path}'")
    else:
        config_path = None

    if hipblaslt_path is None:
        hipblaslt_path = resolve_hipblaslt_path(anchor=__file__)
    hipblaslt_path = str(Path(hipblaslt_path))
    if not Path(hipblaslt_path).is_dir():
        raise FileNotFoundError(f"hipBLASLt path not found: '{hipblaslt_path}'")

    if gemm_log_path is not None and not Path(gemm_log_path).is_file():
        raise FileNotFoundError(f"GEMM log / workload file not found: '{gemm_log_path}'")

    Path(output_path).mkdir(parents=True, exist_ok=True)

    if verbose == 0:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)

    config_dict = load_prepared_config_from_yaml(
        config_path,
        arch=arch,
        gemm_log_path=gemm_log_path,
    )
    run(
        config_dict,
        hipblaslt_path,
        output_path,
        write_shell_scripts=write_shell_scripts,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Tensile tuning configurations from a YAML config file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        metavar="PATH",
        help="Tuning YAML (or omit and pass --arch with --gemm-log-path)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="ARCH (required with --gemm-log-path when no tuning YAML; optional override)",
    )
    parser.add_argument(
        "--gemm-log-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Workload or hipBLASLt-style GEMM YAML (implies SIZE_OPTION=2)",
    )
    parser.add_argument(
        "--outputPath",
        "-o",
        type=str,
        default="./",
        help="Output directory for generated configs",
    )
    parser.add_argument(
        "--hipblaslt",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "hipBLASLt checkout root (overrides auto-detection and $GEKO_HIPBLASLT_PATH). "
            "Auto-detected from this script's location when omitted."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=1,
        choices=[0, 1],
        help="Logging verbosity: 0=WARNING, 1=INFO",
    )
    parser.add_argument(
        "--no-shell-scripts",
        action="store_true",
        help="Skip per-entity .sh and run_*_all.sh; emit YAML and config log only",
    )

    args = parser.parse_args()

    main(
        hipblaslt_path=args.hipblaslt,
        output_path=args.outputPath,
        verbose=args.verbose,
        config=args.config,
        arch=args.arch,
        gemm_log_path=args.gemm_log_path,
        write_shell_scripts=not args.no_shell_scripts,
    )
