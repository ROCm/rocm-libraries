# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""Generate all config sections for tuning YAML other than fork parameters.

Size-agnostic sections (ProblemType, LibraryLogic, base GlobalParameters)
are computed once at init. Per-size sections are built via build_config().
"""

import math
import numpy as np

from geko.config_generator.constants import *
from geko.config_generator.mi_designer import MIDesign
from geko.config_generator.shared_utils import ConfigEntry
from typing import Any, Dict, Optional, Sequence


class ConfigSectionGenerator:
    """Generate config sections for tuning configuration YAML.

    Size-agnostic sections (ProblemType, LibraryLogic, base GlobalParameters)
    are computed once at init. Per-size sections are built via build_config().
    """

    @staticmethod
    def _is_tf32(dtype: str) -> bool:
        """Whether *dtype* is TF32 (X / X1)."""
        return dtype in ("X", "X1")

    @staticmethod
    def _convert_type(dtype: str) -> str:
        """Map config datatype to Tensile datatype."""
        if dtype == "X1":
            return "B"
        if dtype == "X":
            return "S"
        return dtype

    @staticmethod
    def _calc_iters(m: int, n: int, b: int, k: int) -> int:
        """Estimate iteration count for benchmarking based on problem size."""
        return max(round((-(m + n + k) * 0.015 + 431) / b), 5)

    def _use_epilogues(self) -> bool:
        """Whether to emit epilogue fields for this GEMM type."""
        gt = self._gt
        no_epilogue = (gt.data_type == "D" and gt.dest_data_type == "D") or gt.data_type in ("C", "Z")
        return self.config["EPILOGUES"] and not no_epilogue

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._gt = config["GemmProblem"].gemm_type
        self._problem_type = self._build_problem_type()
        self._library_logic = self._build_library_logic()
        self._global_params_base = self._build_global_params_base()
        self._bias_type_args = self._resolve_bias_type()

    # ------------------------------------------------------------------
    # Size-agnostic sections (computed once at init)
    # ------------------------------------------------------------------

    def _build_problem_type(self) -> Dict[str, Any]:
        """Generate ProblemType section for the config."""
        val_HighPrecisionAccumulate = self._gt.data_type != self._gt.compute_data_type
        if self._is_tf32(self._gt.data_type):
            val_HighPrecisionAccumulate = False

        val_transA = "True" if self._gt.transA in ("T", "C") else "False"
        val_transB = "True" if self._gt.transB in ("T", "C") else "False"

        pt = {}
        pt['OperationType'] = 'GEMM'
        pt['DataType'] = self._convert_type(self._gt.data_type)
        if self._gt.data_type == "X1":
            pt['DataTypeA'] = "S"
            pt['DataTypeB'] = "S"
        pt['DestDataType'] = self._convert_type(self._gt.dest_data_type)
        pt['ComputeDataType'] = self._convert_type(self._gt.compute_data_type)
        pt['HighPrecisionAccumulate'] = val_HighPrecisionAccumulate
        # fp4 inputs use MX block scaling (block size 32 on A and B).
        if self._gt.data_type == "F4":
            pt['MXBlockA'] = 32
            pt['MXBlockB'] = 32
        pt['TransposeA'] = val_transA
        pt['TransposeB'] = val_transB
        if self._gt.data_type in ("C", "Z"):
            pt['ComplexConjugateA'] = "True" if self._gt.transA == "C" else "False"
            pt['ComplexConjugateB'] = "True" if self._gt.transB == "C" else "False"
        pt['UseBeta'] = "True"

        epi_tag = "" if self._use_epilogues() else "#"
        pt[f'{epi_tag}Activation'] = "True"
        pt[f'{epi_tag}ActivationHPA'] = "True"
        pt[f'{epi_tag}ActivationType'] = "hipblaslt_all"
        pt[f'{epi_tag}UseScaleAlphaVec'] = "1"
        pt[f'{epi_tag}UseBias'] = "1"
        if "8" in pt["DataType"] or "8" in pt["DestDataType"]:
            pt[f'{epi_tag}UseScaleAB'] = "Scalar"

        pt['Batched'] = "True"

        if self._is_tf32(self._gt.data_type) and self._gt.data_type != "X1":
            pt['F32XdlMathOp'] = self._gt.data_type

        return pt

    def _build_library_logic(self) -> Dict[str, Any]:
        """Generate LibraryLogic section from ``ARCH`` (``HARDWARE_MAP``)."""
        return dict(HARDWARE_MAP[self.config["ARCH"]]["LibraryLogic"])

    def _build_global_params_base(self) -> Dict[str, Any]:
        """Build the size-agnostic portion of GlobalParameters.

        EnqueuesPerSync and NumWarmups are placeholders here; actual values
        are set per-size in build_config().  They must appear in this dict
        so emitted YAML keeps stable key ordering before per-size overrides.
        """
        is_i8 = self._gt.data_type == 'I8'
        is_fp4 = self._gt.data_type == 'F4'
        params = {
            'MinimumRequiredVersion': '5.0.0',
            'SleepPercent': 0,
            'EnqueuesPerSync': 0,
            'NumWarmups': 0,
            'KernelTime': True,
            'NumElementsToValidate': 0,
            'DataInitTypeBeta': 1,
            'DataInitTypeAlpha': 1,
            'DataInitTypeA': 3 if is_i8 else 12,
            'DataInitTypeB': 3 if is_i8 else 13,
            'DataInitTypeC': 3 if is_i8 else 12,
            'DataInitTypeD': 3 if is_i8 else 12,
            'DataInitTypeScaleAlphaVec': 3 if is_i8 else 12,
            'CSVExportWinner': True,
            'CSVMergeSameProblemID': True,
            'PreciseKernelTime': False,
            'Device': 0,
            'SkipSlowSolutionRatio': 0.0,
            '#PrintSolutionRejectionReason': True,
            'KeepBuildTmp': False,
            'RotatingBufferSize': 1024,
            'UseEffLike': False,
        }
        if is_fp4:
            params['DataInitTypeMXSA'] = 3
            params['DataInitTypeMXSB'] = 3
            params['MXScaleFormat'] = 1
        return params

    def _resolve_bias_type(self) -> Optional[str]:
        """Compute BiasTypeArgs string if epilogues are enabled, else None."""
        if not self._use_epilogues():
            return None
        if self._gt.data_type in ("C", "Z"):
            return None
        bias_type = self._convert_type(self._gt.data_type)
        if "8" in bias_type:
            bias_type = self._convert_type(self._gt.dest_data_type)
        if "8" in bias_type:
            bias_type = "S"
        if self._gt.data_type == "X1":
            bias_type = "S"
        if self._gt.data_type == "F4":
            # fp4 cannot be used as bias type.
            bias_type = self._convert_type(self._gt.dest_data_type)
        return f"[{bias_type}]"

    # ------------------------------------------------------------------
    # Size-dependent helpers
    # ------------------------------------------------------------------

    def _get_enqueues_per_sync(self, size: Sequence[int]) -> int:
        """Compute EnqueuesPerSync for a single size."""
        M_dim, N_dim, B_dim, K_dim = size
        mnk = M_dim * N_dim * K_dim * B_dim
        for i in range(len(stepValue_EnqueuesPerSync)):
            if mnk < stepValue_EnqueuesPerSync[i][0] or i == len(stepValue_EnqueuesPerSync) - 1:
                break
        val = stepValue_EnqueuesPerSync[i][1]

        ds = dataSize[self._gt.data_type]
        mn_area = M_dim * N_dim * B_dim
        mt_max = get_list_of_mt_max_size(self.config.get("search_space"))
        threshold = mt_max[self._gt.data_type] * self.config['CUs'] * 10

        if ds == 1 and mn_area < threshold:
            val = math.ceil(val * 3.5)
        elif ds == 2 and mn_area < threshold:
            val *= 2

        return int(val)

    def _compute_enqueues(self, sizes: Sequence[Sequence[int]]) -> int:
        """Compute EnqueuesPerSync across all sizes in a group (takes max).

        TODO: When clustering is enabled, sizes may contain multiple entries
        from different clusters. Review whether max() is the right aggregation
        strategy for the merged set.
        """
        return max(self._get_enqueues_per_sync(sz) for sz in sizes)

    def _apply_enqueue_and_warmup_params(
        self,
        global_params: Dict[str, Any],
        sizes: Sequence[Sequence[int]],
        backend: str,
    ) -> None:
        """Set EnqueuesPerSync, NumWarmups, and (Ductile-only) SleepPercent."""
        enqueues = self._compute_enqueues(sizes)
        if backend != "ductile":
            global_params['EnqueuesPerSync'] = enqueues
            global_params['NumWarmups'] = enqueues
        else:
            global_params['SleepPercent'] = 50
            iters = max(self._calc_iters(*sz) for sz in sizes)
            global_params['EnqueuesPerSync'] = int(0.75 * max(iters, enqueues))
            global_params['NumWarmups'] = int(
                0.75 * max(iters, global_params['EnqueuesPerSync']))

    def _build_ductile(
        self,
        fork_params: Dict[str, Any],
        sizes: Sequence[Sequence[int]],
        config_name: str,
        cms_priority: bool,
        soo: bool,
    ) -> Dict[str, Any]:
        """Build the ductile YAML section. Computes sampling costs from MI groups and problem sizes.

        Cost priority (lower is better):
        1) Lower ceil(TilesPerCU) bucket
        2) Higher TilesPerCU within the same bucket
        3) Higher totalGranularity
        4) Higher GSU

        Group cost is averaged across sizes.
        """

        val_profile = self.config.get("DUCTILE_VALIDATION_PROFILE", 1)
        if val_profile not in DUCTILE_VALIDATION_PROFILE_MAP:
            raise ValueError(f"Invalid DUCTILE_VALIDATION_PROFILE {val_profile}; must be one of {DUCTILE_VALIDATION_PROFILE_MAP.keys()}")

        n_elements_to_validate = DUCTILE_VALIDATION_PROFILE_MAP[val_profile]
        mi_groups = fork_params["Groups"].values[0]

        has_priority = lambda grp: "UseCustomMainLoopSchedule" in grp and int(grp["UseCustomMainLoopSchedule"].values[0]) and cms_priority

        # pop_size must be <= SearchSpace.n_perms in Ductile GA.
        n_perms = self._compute_n_perms(fork_params)
        pop_size = self._safe_pop_size(n_perms)

        non_cms_mask = np.array([not has_priority(grp) for grp in mi_groups], dtype=bool)
        if non_cms_mask.sum() <= 1 or len(sizes) == 0:
            d = dict(soo=soo, n_elements_to_validate=n_elements_to_validate)
            if pop_size:
                d["pop_size"] = pop_size
            return d
        
        gsu_values = [float(grp["MatrixInstruction"].metadata.get("GSU", 1)) for grp in mi_groups if not has_priority(grp)]
        gsu_min = min(gsu_values) if gsu_values else 1.0
        gsu_max = max(gsu_values) if gsu_values else 1.0
        gsu_span = max(gsu_max - gsu_min, 1e-12)

        # eps values adjusted for accurate lexicographical sorting.
        def _compute_cost(
            grp: Dict[str, Any], 
            size: Sequence[int],
            eps_t: float = 1,
            eps_g: float = 1e-3,
            eps_s: float = 1e-11
        ) -> float:
            metadata = grp["MatrixInstruction"].metadata
            MT0, MT1 = metadata["MT"]
            LSU = metadata.get("LSU", 1)
            GSU = metadata.get("GSU", 1)
            wave = metadata["wave"]

            granular_metrics = MIDesign.calculate_granularities(
                MT0, MT1, size[0], size[1], size[2],
                self.config['CUs'], LSU, GSU, wave)

            ceil_tpcu = math.ceil(granular_metrics.TilesPerCU)
            tile_term = ceil_tpcu - granular_metrics.TilesPerCU
            gran_term = 1.0 - granular_metrics.totalGranularity
            # GSU term is normalized across groups to prevent scale issues with tile/granularity terms.
            gsu_term = 1.0 - (GSU - gsu_min) / gsu_span

            return ceil_tpcu + eps_t * tile_term + eps_g * gran_term + eps_s * gsu_term
        
        cost_matrix = []
        for size in sizes:
            size_cost = np.array([_compute_cost(grp, size) if not has_priority(grp) else np.nan for grp in mi_groups], dtype=np.float32)
            size_cost[non_cms_mask] = size_cost[non_cms_mask] / size_cost[non_cms_mask].min()
            cost_matrix.append(size_cost)
        cost_matrix = np.array(cost_matrix, dtype=np.float32)

        # Map CMS groups to the best non-CMS average cost.
        cost = np.empty(len(mi_groups), dtype=np.float32)
        cost[non_cms_mask] = cost_matrix[:, non_cms_mask].mean(axis=0)
        cost[~non_cms_mask] = cost[non_cms_mask].min()
        
        d = dict(
            soo=soo,
            n_elements_to_validate=n_elements_to_validate,
            weights=[{"group_0": f"{cost.tolist()}"}],
        )
        if pop_size:
            d["pop_size"] = pop_size
        return d

    @staticmethod
    def _compute_n_perms(fork_params: Dict[str, Any]) -> int:
        """Approximate Ductile search-space permutations from active fork params."""
        n_perms = 1
        for name, fp in fork_params.items():
            if not getattr(fp, "active", True):
                continue
            if name == "Groups":
                for group_dim in fp.values:
                    if len(group_dim) > 1:
                        n_perms *= len(group_dim)
                continue
            if len(fp.values) > 1:
                n_perms *= len(fp.values)
        return n_perms

    @staticmethod
    def _safe_pop_size(n_perms: int, default: int = 512) -> int:
        """Return a valid pop_size satisfying 2 < pop_size <= n_perms, else 0."""
        if n_perms >= default:
            return default
        if n_perms < 3:
            return 0
        return min(n_perms, max(3, n_perms - 1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_config(
        self,
        entry: ConfigEntry,
        backend: str = "tensile",
        cms_priority: bool = False,
        config_name: Optional[str] = None,
        soo: bool = False,
    ) -> Dict[str, Any]:
        """Build the complete config dict for a size group.

        Args:
            entry: ConfigEntry holding sizes, fork_params, nkernels, and mis_per_size.
            backend: Tuning backend ("ductile" or "tensile").
            config_name: Config name for log files (required for Ductile).
            cms_priority: Whether to prioritize CMS tiles (Ductile only).
            soo: Whether single-objective optimization is enabled (Ductile only).

        Returns:
            Complete config dict ready for YAML serialization.
        """
        sizes = entry.sizes
        fork_params = entry.fork_params
        is_ductile = backend == "ductile"

        global_params = dict(self._global_params_base)
        self._apply_enqueue_and_warmup_params(global_params, sizes, backend)

        # K is rounded up to a multiple of 32 only for MXFP4 (F4).
        problem_sizes = []
        is_mxfp4 = self._gt.data_type == "F4"
        for M, N, batch, K in sizes:
            if is_mxfp4:
                K = ((K + 31) // 32) * 32
            problem_sizes.append({"Exact": f'[ {M}, {N}, {batch}, {K} ]'})

        benchmark_final = [{"ProblemSizes": problem_sizes}]
        if self._bias_type_args:
            benchmark_final.append({"BiasTypeArgs": self._bias_type_args})

        benchmark_common = {
            "InitialSolutionParameters": '',
            "BenchmarkCommonParameters": [{"KernelLanguage": "[Assembly]"}],
            "ForkParameters": fork_params,
            "BenchmarkJoinParameters": '',
            "BenchmarkFinalParameters": benchmark_final,
        }

        tuning_config = {
            'GlobalParameters': global_params,
            'BenchmarkProblems': [[self._problem_type, benchmark_common]],
            'LibraryLogic': self._library_logic,
            '#LibraryClient': '',
            'Backend': {"Name": "Ductile"} if is_ductile else {"Name": "Tensile"}
        }

        if is_ductile:
            tuning_config['Backend']["Config"] = self._build_ductile(
                fork_params, 
                sizes, 
                config_name,
                cms_priority,
                soo
            )

        return tuning_config

    def generate_comment(self, nkernels: int) -> str:
        """Generate YAML header comment for a config."""
        header = '\n#==================================\n'
        header += (
            f"# This yaml is auto-generated by geko.config_generator.\n"
            f"# Version: {VERSION}\n"
            f"# GEMM Type: "
            f"{self._gt.gemm_name}\n"
            f"# Total #kernels: {nkernels}\n"
        )
        header += '#==================================\n\n'
        return header
