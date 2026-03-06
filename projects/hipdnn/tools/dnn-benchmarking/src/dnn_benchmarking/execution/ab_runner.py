# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""A/B testing runner for comparing plugin/engine configurations."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from ..common.exceptions import ExecutionError
from ..config.benchmark_config import ABTestConfig, BenchmarkConfig, ValidationConfig
from ..graph.loader import GraphLoader
from ..graph.tensor_info import TensorInfo
from ..reporting.statistics import BenchmarkResult
from ..validation.comparison import ArrayComparator, ComparisonResult
from .buffer_manager import BufferManager
from .executor import Executor


@dataclass
class ValidationResult:
    """Result of reference validation for a single configuration.

    Attributes:
        passed: Whether validation passed.
        max_abs_diff: Maximum absolute difference from reference.
        max_rel_diff: Maximum relative difference from reference.
        provider_name: Name of the reference provider used.
    """

    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    provider_name: str


@dataclass
class ABTestResult:
    """Result of A/B test comparison.

    Attributes:
        result_a: Full benchmark result for configuration A (includes E2E and kernel timings).
        result_b: Full benchmark result for configuration B (includes E2E and kernel timings).
        init_time_a_ms: Graph initialization time for A in milliseconds.
        init_time_b_ms: Graph initialization time for B in milliseconds.
        passed: Whether outputs match within tolerance.
        max_abs_diff: Maximum absolute difference between outputs.
        max_rel_diff: Maximum relative difference between outputs.
        validation_a: Optional reference validation result for configuration A.
        validation_b: Optional reference validation result for configuration B.
    """

    result_a: BenchmarkResult
    result_b: BenchmarkResult
    init_time_a_ms: float
    init_time_b_ms: float
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    validation_a: Optional[ValidationResult] = None
    validation_b: Optional[ValidationResult] = None


class ABRunner:
    """Runs A/B comparison between two plugin/engine configurations.

    This class handles:
    - Setting plugin paths for each configuration
    - Executing the same graph with different engines
    - Comparing outputs using np.allclose
    - Collecting timing statistics for both configurations (E2E and kernel)
    - Optional reference validation for each configuration
    """

    def __init__(
        self,
        graph_json: Dict[str, Any],
        config: BenchmarkConfig,
        ab_config: ABTestConfig,
        gpu_backend: Literal["torch", "auto", "none"] = "auto",
        validation_config: Optional[ValidationConfig] = None,
    ) -> None:
        """Initialize A/B runner.

        Args:
            graph_json: The graph as a parsed JSON dictionary.
            config: Benchmark configuration (warmup/iters).
            ab_config: A/B test configuration (paths, engine IDs, tolerances).
            gpu_backend: GPU timer backend to use (torch, auto, none).
            validation_config: Optional validation configuration for reference checking.
        """
        self._graph_json = graph_json
        self._config = config
        self._ab_config = ab_config
        self._gpu_backend = gpu_backend
        self._validation_config = validation_config

    def _set_plugin_path(self, plugin_path: Optional[Path]) -> None:
        """Set plugin path using hipdnn_frontend API.

        Args:
            plugin_path: Path to plugin directory, or None for default.
        """
        import hipdnn_frontend as hipdnn

        if plugin_path is not None:
            # Use ABSOLUTE mode to ensure only this plugin is used
            hipdnn.set_engine_plugin_paths(
                [str(plugin_path)], hipdnn.PluginLoadingMode.ABSOLUTE
            )

    def _run_single(
        self,
        plugin_path: Optional[Path],
        engine_id: int,
        buffer_manager: BufferManager,
        config_name: str = "",
    ) -> Tuple[Dict[int, np.ndarray], BenchmarkResult, float]:
        """Execute graph with specific plugin/engine configuration.

        Args:
            plugin_path: Path to plugin directory, or None for default.
            engine_id: Engine ID to use.
            buffer_manager: Buffer manager with allocated tensors.
            config_name: Name for this configuration (e.g., "A" or "B").

        Returns:
            Tuple of (outputs_dict, benchmark_result, init_time_ms) where
            outputs_dict maps tensor UID to numpy array for all output tensors.
        """
        import hipdnn_frontend as hipdnn

        # Set plugin path before creating Handle
        self._set_plugin_path(plugin_path)

        handle = hipdnn.Handle()
        executor = Executor(
            json.dumps(self._graph_json), self._config, gpu_backend=self._gpu_backend
        )
        executor.prepare(handle, engine_id=engine_id)
        init_time_ms = executor.init_time_ms

        variant_pack = buffer_manager.create_variant_pack()
        executor.warmup(handle, variant_pack)
        result = executor.benchmark(handle, variant_pack, graph_name=config_name)

        # Get all output data - copy to avoid overwriting
        output_tensors = buffer_manager.get_output_tensors()
        if not output_tensors:
            raise ExecutionError("No output tensors found in graph")

        outputs: Dict[int, np.ndarray] = {}
        for tensor in output_tensors:
            data = buffer_manager.get_output_data(tensor.uid)
            if data is None:
                raise ExecutionError(
                    f"Failed to retrieve output data for tensor uid={tensor.uid}"
                )
            outputs[tensor.uid] = data.copy()

        return outputs, result, init_time_ms

    def _validate_output(
        self,
        outputs: Dict[int, np.ndarray],
        tensor_infos: List[TensorInfo],
        buffer_manager: BufferManager,
        config_name: str,
    ) -> Optional[ValidationResult]:
        """Validate output against reference provider.

        Args:
            outputs: Dict mapping tensor UID to output data from execution.
            tensor_infos: List of tensor info objects.
            buffer_manager: Buffer manager with input data.
            config_name: Name of the configuration being validated.

        Returns:
            ValidationResult if validation was performed, None otherwise.
        """
        if self._validation_config is None or not self._validation_config.enabled:
            return None

        from ..validation import ReferenceProviderRegistry

        try:
            provider = ReferenceProviderRegistry.get_provider(
                self._validation_config.provider
            )

            if not provider.is_available():
                return None

            if not provider.supports_graph(self._graph_json):
                return None

            # Collect input data
            input_data: Dict[int, np.ndarray] = {}
            for tensor_info in tensor_infos:
                if not tensor_info.is_virtual and not tensor_info.is_output:
                    data = buffer_manager.get_input_data(tensor_info.uid)
                    if data is not None:
                        input_data[tensor_info.uid] = data

            # Compute reference
            reference_outputs = provider.compute_reference(self._graph_json, input_data)

            # Compare all output tensors, track worst-case diffs
            comparator = ArrayComparator(
                rtol=self._validation_config.rtol, atol=self._validation_config.atol
            )
            all_passed = True
            worst_abs = 0.0
            worst_rel = 0.0

            for uid, output_data in outputs.items():
                ref_output = reference_outputs.get(uid)
                if ref_output is None:
                    continue
                comparison = comparator.compare(
                    output_data,
                    ref_output.data,
                    config_name,
                    self._validation_config.provider,
                )
                if not comparison.passed:
                    all_passed = False
                worst_abs = max(worst_abs, comparison.max_abs_diff)
                worst_rel = max(worst_rel, comparison.max_rel_diff)

            return ValidationResult(
                passed=all_passed,
                max_abs_diff=worst_abs,
                max_rel_diff=worst_rel,
                provider_name=self._validation_config.provider,
            )

        except (ValueError, NotImplementedError, ImportError):
            return None

    def run(self, seed: Optional[int] = 42) -> ABTestResult:
        """Run A/B comparison.

        Args:
            seed: Random seed for reproducible input data.

        Returns:
            ABTestResult with full benchmark results and comparison.
        """
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(self._graph_json)

        validation_a: Optional[ValidationResult] = None
        validation_b: Optional[ValidationResult] = None

        with BufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=seed)

            # Run configuration A
            buffer_manager.zero_outputs()
            outputs_a, result_a, init_a = self._run_single(
                self._ab_config.a_path, self._ab_config.a_id, buffer_manager, "A"
            )

            # Validate A if configured
            validation_a = self._validate_output(
                outputs_a, tensor_infos, buffer_manager, "A"
            )

            # Synchronize GPU to ensure Config A's work is complete before B starts
            # This prevents stream state contamination in timing measurements
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass

            # Run configuration B (same inputs)
            buffer_manager.zero_outputs()
            outputs_b, result_b, init_b = self._run_single(
                self._ab_config.b_path, self._ab_config.b_id, buffer_manager, "B"
            )

            # Validate B if configured
            validation_b = self._validate_output(
                outputs_b, tensor_infos, buffer_manager, "B"
            )

        # Compare all output tensors, track worst-case diffs across all outputs
        comparator = ArrayComparator(
            rtol=self._ab_config.rtol, atol=self._ab_config.atol
        )
        all_passed = True
        worst_abs = 0.0
        worst_rel = 0.0

        for uid in outputs_a:
            if uid not in outputs_b:
                all_passed = False
                worst_abs = float("inf")
                worst_rel = float("inf")
                continue
            comparison = comparator.compare(outputs_a[uid], outputs_b[uid], "A", "B")
            if not comparison.passed:
                all_passed = False
            worst_abs = max(worst_abs, comparison.max_abs_diff)
            worst_rel = max(worst_rel, comparison.max_rel_diff)

        return ABTestResult(
            result_a=result_a,
            result_b=result_b,
            init_time_a_ms=init_a,
            init_time_b_ms=init_b,
            passed=all_passed,
            max_abs_diff=worst_abs,
            max_rel_diff=worst_rel,
            validation_a=validation_a,
            validation_b=validation_b,
        )
