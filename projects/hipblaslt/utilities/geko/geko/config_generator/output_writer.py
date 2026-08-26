# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""Output writer for tuning configuration artifacts.

Use :class:`EntityOutputWriter` once per Driver run. For each entity:

  1. :meth:`~EntityOutputWriter.write_entity_files_only` — ``<entity>.yaml`` and optional
     ``<entity>.sh``
  2. :meth:`~EntityOutputWriter.append_aggregate_metadata` — append to ``run_<type>_all.sh``
     (if shell scripts are enabled) and ``Config_<type>.log``

The driver may call (1) in parallel, then (2) in index order so aggregate files stay consistent.

Serializes ForkParameter objects directly to YAML in a single pass with no post-processing.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from geko.config_generator.shared_utils import ConfigEntry, ForkParameter

logger = logging.getLogger("GEKO")


# =====================================================================
# Tuning config YAML writer
# =====================================================================

class TuningConfigWriter:
    """Serialize a config dict (with :class:`ForkParameter` values) to Tensile YAML.

    Emits GlobalParameters, BenchmarkProblems (including ForkParameters and Groups),
    LibraryLogic, optional ductile section, in one pass without post-processing.
    """

    @staticmethod
    def _format_scalar(val: Any) -> str:
        """Format a scalar for YAML output.

        Booleans render as ``true`` / ``false``; other values use ``str()``.
        """
        if isinstance(val, bool):
            return 'true' if val else 'false'
        return str(val)

    @staticmethod
    def _quote_if_string(val: Any) -> str:
        """Format a value for inclusion inside a YAML list literal.

        Strings are wrapped in double quotes; other types use ``str()``.
        """
        if isinstance(val, str):
            return f'"{val}"'
        return str(val)

    @staticmethod
    def _format_fork_values(values: list) -> str:
        """Render a fork-parameter value list as a YAML-style ``[v1, v2, ...]`` string."""
        return "[" + ", ".join(TuningConfigWriter._quote_if_string(v) for v in values) + "]"

    @staticmethod
    def _format_group_entry_value(fp: ForkParameter) -> str:
        """Render one parameter inside a group entry for YAML.

        Single-element lists are unwrapped to the scalar string. Appends
        ``# comment`` when ``fp.comment`` is set.
        """
        values = fp.values
        if isinstance(values, list) and len(values) == 1:
            val_str = str(values[0])
        else:
            val_str = str(values)
        if fp.comment:
            return f"{val_str} #{fp.comment}"
        return val_str

    def write(self, config: Dict[str, Any], header: str, filepath: str | Path) -> None:
        """Write the full tuning YAML for one entity.

        Args:
            config: Built config from ``ConfigSectionGenerator`` (nested dicts and
                ``ForkParameter`` / group structures).
            header: Leading comment block (e.g. version and kernel count).
            filepath: Destination ``.yaml`` path (opened with ``'w'``).
        """
        lines: List[str] = [header]
        lines.extend(self._write_section('GlobalParameters', config['GlobalParameters']))
        lines.extend(self._write_benchmark_problems(config['BenchmarkProblems']))
        lines.extend(self._write_section('LibraryLogic', config['LibraryLogic'], format_val=str))
        if '#LibraryClient' in config:
            lines.append('#LibraryClient: \n')
        if 'Backend' in config:
            lines.extend(self._write_backend(config['Backend']))
        lines.append(f'# End of {Path(filepath).name} \n')

        with open(filepath, 'w') as f:
            f.writelines(lines)
        logger.info("output path: %s, configYaml: %s", filepath, filepath)

    # ------------------------------------------------------------------
    # Reusable helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_mapping(
        data: Dict[str, Any],
        indent: int = 2,
        format_val: Any = None,
        first_prefix: Optional[str] = None,
    ) -> List[str]:
        """Write a mapping as indented ``key: value`` lines.

        Args:
            data: Keys and values to serialize.
            indent: Leading spaces for every line (except possibly the first).
            format_val: Callable to stringify each value.
            first_prefix: If set, used instead of indent for the first key only
                (e.g. ``'- - '`` to continue a YAML list item).
        """
        if format_val is None:
            format_val = TuningConfigWriter._format_scalar
        lines: List[str] = []
        prefix = ' ' * indent
        first = True
        for key, val in data.items():
            formatted = format_val(val)
            if first and first_prefix is not None:
                lines.append(f"{first_prefix}{key}: {formatted}\n")
                first = False
            else:
                lines.append(f"{prefix}{key}: {formatted}\n")
        return lines

    def _write_section(
        self,
        name: str,
        data: Dict[str, Any],
        indent: int = 2,
        format_val: Any = None,
    ) -> List[str]:
        """Write ``name:\\n`` followed by a mapping at the given indent."""
        if format_val is None:
            format_val = TuningConfigWriter._format_scalar
        return [f'{name}:\n'] + self._write_mapping(data, indent, format_val)

    # ------------------------------------------------------------------
    # Benchmark sections (structurally unique)
    # ------------------------------------------------------------------

    def _write_benchmark_problems(self, problems: List[list]) -> List[str]:
        """Serialize ``BenchmarkProblems`` from ``[problem_type, benchmark_common], ...`` pairs."""
        lines = ['BenchmarkProblems:\n']
        for problem_type, benchmark_common in problems:
            lines.extend(self._write_mapping(
                problem_type, indent=4, first_prefix='- - '))
            lines.extend(self._write_benchmark_common(benchmark_common))
        return lines

    def _write_benchmark_common(self, bc: Dict[str, Any]) -> List[str]:
        """Serialize the inner benchmark-common block (fork params, final params, etc.)."""
        lines: List[str] = []
        for key, val in bc.items():
            if key == 'InitialSolutionParameters':
                lines.append(f"  - {key}: \n")
            elif key == 'BenchmarkCommonParameters':
                lines.append(f"    {key}:\n")
                for item in val:
                    for k, v in item.items():
                        lines.append(f"    - {k}: {v}\n")
            elif key == 'ForkParameters':
                lines.extend(self._write_fork_params(val))
            elif key == 'BenchmarkJoinParameters':
                lines.append(f"    {key}: \n")
            elif key == 'BenchmarkFinalParameters':
                lines.extend(self._write_benchmark_final(val))
        return lines

    def _write_fork_params(self, fork_params: Dict[str, ForkParameter]) -> List[str]:
        """Serialize ``ForkParameters``; inactive params are commented out."""
        lines = ['    ForkParameters:\n']
        for name, fp in fork_params.items():
            if name == 'Groups':
                lines.extend(self._write_groups(fp.values))
                continue
            val_str = self._format_fork_values(fp.values)
            if fp.comment:
                val_str = f"{val_str} #{fp.comment}"
            if fp.active:
                lines.append(f"    - {name}: {val_str}\n")
            else:
                lines.append(f"#     - {name}: {val_str}\n")
        return lines

    def _write_groups(self, group_dimensions: list) -> List[str]:
        """Serialize ``Groups`` nested list structure (MI and fork group dimensions)."""
        lines = ['    - Groups:\n']
        for dim in group_dimensions:
            if not dim:
                continue
            for entry_idx, entry in enumerate(dim):
                first_key = True
                for param_name, fp in entry.items():
                    val_str = self._format_group_entry_value(fp)
                    if entry_idx == 0 and first_key:
                        lines.append(f"      - - {param_name}: {val_str}\n")
                    elif first_key:
                        lines.append(f"        - {param_name}: {val_str}\n")
                    else:
                        lines.append(f"          {param_name}: {val_str}\n")
                    first_key = False
        return lines

    def _write_benchmark_final(self, final_params: list) -> List[str]:
        """Serialize ``BenchmarkFinalParameters`` (e.g. ProblemSizes)."""
        lines = ['    BenchmarkFinalParameters:\n']
        for item in final_params:
            for key, val in item.items():
                if key == 'ProblemSizes':
                    lines.append(f"    - {key}:\n")
                    for size_entry in val:
                        for sk, sv in size_entry.items():
                            lines.append(f"      - {sk}: {sv}\n")
                else:
                    lines.append(f"    - {key}: {val}\n")
        return lines

    def _write_backend(self, backend: Dict[str, Any]) -> List[str]:
        """Serialize ``Backend`` block including optional Ductile config weights."""
        simple = {k: v for k, v in backend.items() if k != 'Config'}
        lines = ['Backend:\n'] + self._write_mapping(simple)
        if "Config" not in backend:
            return lines
        
        simple = {k: v for k, v in backend["Config"].items() if k != 'weights'}
        lines.extend(['  Config:\n'] + self._write_mapping(simple, indent=4))

        if "weights" not in backend["Config"]:
            return lines
        lines.append('    weights:\n')
        for weight_dict in backend["Config"].get('weights', []):
            for key, val in weight_dict.items():
                lines.append(f"      - {key}: {val}\n")
        return lines


# =====================================================================
# Run script generator
# =====================================================================

_RUN_SCRIPT_TEMPLATE = """\
#!/bin/bash

NAME="{entity_name}"
YAML="$NAME.yaml"
OUT="$NAME-tensilelite.log"
TUNING_DIR="build_$NAME"

WORK_DIR="WDirDevice_id"

echo "running $NAME ..."
{run_command}
mkdir $TUNING_DIR
cp  $YAML $TUNING_DIR
mv $WORK_DIR/2\\_BenchmarkData $WORK_DIR/3\\_LibraryLogic $OUT $TUNING_DIR
rm -rf $WORK_DIR/1\\_BenchmarkProblems 
echo " ---- $NAME Done!"
"""


def write_run_script(
    filepath: str | Path,
    entity_name: str,
    hipblaslt_path: str | Path,
    client_path: Optional[str | Path] = None,
) -> None:
    """Write an executable bash script that runs Tensile for one YAML.
    
    Args:
        filepath: Path for the ``.sh`` file (created with mode ``0o755``).
        entity_name: Base name matching ``{entity_name}.yaml`` in the working directory.
        hipblaslt_path: Root of the hipBLASLt checkout (for ``tensilelite`` paths).
        client_path: Optional path passed as ``--prebuilt-client`` when set.
    """
    hip_s = str(Path(hipblaslt_path).resolve())
    client_path_str = ''
    if client_path:
        client_path_str = f'--prebuilt-client {Path(client_path).resolve()}'

    run_command = (
        f'PYTHONPATH={hip_s}/tensilelite/ '
        f'{hip_s}/tensilelite/Tensile/bin/Tensile '
        f'$YAML $WORK_DIR {client_path_str} 2>&1 | tee $OUT'
    )

    content = _RUN_SCRIPT_TEMPLATE.format(
        entity_name=entity_name,
        run_command=run_command,
    )

    with open(filepath, 'w') as f:
        f.write(content)
    os.chmod(filepath, 0o755)


# =====================================================================
# Run-all aggregator and config log
# =====================================================================

def init_run_all_script(filepath: str | Path) -> None:
    """Create or truncate ``run_*_all.sh`` with shebang and a short comment.

    Args:
        filepath: Aggregated script path (mode ``0o755`` after write).

    Called from :class:`EntityOutputWriter` construction so :func:`append_run_all`
    does not append to a stale file from a previous run.
    """
    with open(filepath, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Auto-generated run-all script; regenerated each Driver run.\n')
    os.chmod(filepath, 0o755)


def append_run_all(filepath: str | Path, entity_script: str, progress: str) -> None:
    """Append lines to run all per-entity scripts in sequence.

    Args:
        filepath: Aggregated ``run_*_all.sh`` path (opened append).
        entity_script: Name of the per-entity script (e.g. ``foo.sh``); emitted as ``./foo.sh``.
        progress: Label echoed after the script (e.g. ``0/3``).
    """
    with open(filepath, 'a') as f:
        f.write(f'./{entity_script}\n')
        f.write(f'echo "PROGRESS - {progress}"\n')
    os.chmod(filepath, 0o755)


def write_config_log(
    filepath: str | Path,
    entity_name: str,
    nkernels: int,
    mis_per_size: Dict[Any, int],
) -> None:
    """Append a summary block for one entity to ``Config_<type>.log``.

    Args:
        filepath: Log file path (parent of ``output_dir`` in the Driver layout).
        entity_name: Config / script basename for this entity.
        nkernels: Total kernel count for the entity.
        mis_per_size: Map of problem size tuple to MI count.
    """
    with open(filepath, 'a') as f:
        f.write(f' ==== {entity_name}\n')
        f.write(f' #kernels {nkernels}\n')
        for size, count in mis_per_size.items():
            f.write(f' # MIs for {size}: {count}\n')


# =====================================================================
# Orchestrator
# =====================================================================

class EntityOutputWriter:
    """Writes YAML, optional per-entity run scripts and run-all, and config log.

    Instantiate once per Driver run (same ``output_dir`` / ``gemm_type``).
    When ``write_shell_scripts`` is true, construction creates a fresh
    ``run_<gemm_type>_all.sh``; :meth:`append_aggregate_metadata` appends to it.
    """

    def __init__(
        self,
        output_dir: str | Path,
        gemm_type: str,
        hipblaslt_path: str | Path,
        *,
        client_path: Optional[str | Path] = None,
        write_shell_scripts: bool = True,
    ) -> None:
        """Store paths and options for this run, and optionally reset run-all script.

        Args:
            output_dir: Directory for YAML and ``.sh`` files (and ``run_<gemm_type>_all.sh``).
            gemm_type: GEMM string used in run-all and config log basenames.
            hipblaslt_path: hipBLASLt root for per-entity run scripts.
            client_path: Optional prebuilt Tensile client for run scripts.
            write_shell_scripts: If false, skip ``.sh`` and ``run_*_all.sh`` (YAML and log only).
        """
        self._output_dir = Path(output_dir)
        self._gemm_type = gemm_type
        self._hipblaslt_path = Path(hipblaslt_path)
        self._client_path = Path(client_path) if client_path is not None else None
        self._write_shell_scripts = write_shell_scripts
        self._tuning_writer = TuningConfigWriter()
        if write_shell_scripts:
            init_run_all_script(self._run_all_path)

    @property
    def _run_all_path(self) -> Path:
        """Path to ``run_<gemm_type>_all.sh`` under ``output_dir``."""
        return self._output_dir / f'run_{self._gemm_type}_all.sh'

    @property
    def _config_log_path(self) -> Path:
        """Path to ``Config_<gemm_type>.log`` next to ``output_dir``'s parent."""
        return self._output_dir.parent / f'Config_{self._gemm_type}.log'

    def write_entity_files_only(
        self,
        entry: ConfigEntry,
        config: Dict[str, Any],
        header: str,
        entity_name: str,
    ) -> None:
        """Write YAML and optional per-entity ``.sh`` only (no run-all or config log).

        Use with :meth:`append_aggregate_metadata` when emitting entities in parallel
        so shared log/run-all files are updated in deterministic order afterward.
        """
        yaml_path = self._output_dir / f'{entity_name}.yaml'
        self._tuning_writer.write(config, header, yaml_path)

        if self._write_shell_scripts:
            script_path = self._output_dir / f'{entity_name}.sh'
            write_run_script(
                script_path,
                entity_name,
                self._hipblaslt_path,
                self._client_path,
            )

    def append_aggregate_metadata(
        self,
        entity_name: str,
        entry: ConfigEntry,
        progress: str = '',
    ) -> None:
        """Append this entity to ``run_*_all.sh`` (if enabled) and ``Config_*.log``."""
        if self._write_shell_scripts:
            append_run_all(
                self._run_all_path, f'{entity_name}.sh', progress)
        write_config_log(
            self._config_log_path, entity_name, entry.nkernels, entry.mis_per_size,
        )
