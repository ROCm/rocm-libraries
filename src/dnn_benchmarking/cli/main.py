# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Main entry point for dnn-benchmark CLI."""

import json
import sys
from pathlib import Path
from typing import Literal, Optional

from ..common.exceptions import ExecutionError, GraphLoadError
from ..config.benchmark_config import ABTestConfig, BenchmarkConfig, ValidationConfig
from ..execution.ab_runner import ABRunner
from ..execution.buffer_manager import BufferManager
from ..execution.executor import Executor
from ..graph.loader import GraphLoader
from ..reporting.reporter import Reporter
from ..reporting.statistics import BenchmarkStats, CombinedBenchmarkStats
from ..validation import ArrayComparator, ReferenceProviderRegistry
from ..validation.validator import Validator
from .parser import create_parser


def run_benchmark(
    config: BenchmarkConfig,
    seed: Optional[int] = None,
    validation_config: Optional[ValidationConfig] = None,
    output_path: Optional[Path] = None,
    gpu_backend: Literal["torch", "auto", "none"] = "auto",
) -> int:
    """Run the benchmark workflow.

    Args:
        config: Benchmark configuration.
        seed: Optional random seed for reproducibility.
        validation_config: Optional validation configuration.
        output_path: Optional path to export benchmark results as JSON.
        gpu_backend: GPU timer backend to use (torch, auto, none).

    Returns:
        Exit code (0 for success, 1 for error, 2 for validation failure).
    """
    reporter = Reporter()
    validation_passed = True

    try:
        # Load and validate graph
        loader = GraphLoader()
        graph_json = loader.load_json(config.graph_path)
        loader.validate(graph_json)

        graph_name = loader.get_graph_name(graph_json)
        tensor_infos = loader.extract_tensor_info(graph_json)

        # Print header
        reporter.print_header(config, graph_name)

        # Import hipdnn after validation to give better error messages
        try:
            import hipdnn_frontend as hipdnn
        except ImportError:
            reporter.print_error(
                "hipdnn_frontend not available. "
                "Install hipDNN Python bindings first."
            )
            return 1

        # Create handle
        handle = hipdnn.Handle()

        # Prepare executor
        graph_json_str = json.dumps(graph_json)
        executor = Executor(graph_json_str, config, gpu_backend=gpu_backend)
        executor.prepare(handle)

        reporter.print_init_time(executor.init_time_ms)

        # Allocate buffers
        with BufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=seed)
            buffer_manager.zero_outputs()

            variant_pack = buffer_manager.create_variant_pack()

            # Run warmup
            executor.warmup(handle, variant_pack)

            # Run benchmark
            result = executor.benchmark(handle, variant_pack, graph_name=graph_name)

            # Calculate statistics
            stats = CombinedBenchmarkStats.from_result(result)
            reporter.print_combined_stats(stats)

            # Export results if requested
            if output_path:
                result.save_json(str(output_path))
                print(f"Results exported to: {output_path}")

            # Validation
            if validation_config is not None and validation_config.enabled:
                validation_passed = _run_reference_validation(
                    graph_json=graph_json,
                    buffer_manager=buffer_manager,
                    tensor_infos=tensor_infos,
                    validation_config=validation_config,
                    reporter=reporter,
                )

        reporter.print_footer()
        return 0 if validation_passed else 2

    except GraphLoadError as e:
        reporter.print_error(f"Graph load error: {e}")
        return 1

    except ExecutionError as e:
        reporter.print_error(f"Execution error: {e}")
        return 1

    except Exception as e:
        reporter.print_error(f"Unexpected error: {e}")
        return 1


def _run_reference_validation(
    graph_json: dict,
    buffer_manager: BufferManager,
    tensor_infos: list,
    validation_config: ValidationConfig,
    reporter: Reporter,
) -> bool:
    """Run reference validation against a provider.

    Args:
        graph_json: The graph as a parsed JSON dictionary.
        buffer_manager: Buffer manager with allocated tensors.
        tensor_infos: List of TensorInfo objects.
        validation_config: Validation configuration.
        reporter: Reporter for output.

    Returns:
        True if validation passed, False otherwise.
    """
    try:
        # Get reference provider
        provider = ReferenceProviderRegistry.get_provider(validation_config.provider)

        if not provider.is_available():
            reporter.print_error(
                f"Reference provider '{validation_config.provider}' is not available. "
                f"Available providers: {ReferenceProviderRegistry.list_available()}"
            )
            return False

        # Check if provider supports all operations in graph
        if not provider.supports_graph(graph_json):
            unsupported = provider.get_unsupported_operations(graph_json)
            reporter.print_error(
                f"Reference provider '{validation_config.provider}' does not support "
                f"operations: {unsupported}"
            )
            return False

        # Collect input data from buffer manager
        input_data = {}
        for tensor_info in tensor_infos:
            if not tensor_info.is_virtual and not tensor_info.is_output:
                data = buffer_manager.get_input_data(tensor_info.uid)
                if data is not None:
                    input_data[tensor_info.uid] = data

        # Compute reference outputs
        reference_outputs = provider.compute_reference(graph_json, input_data)

        # Compare each output tensor
        comparator = ArrayComparator(
            rtol=validation_config.rtol, atol=validation_config.atol
        )

        all_passed = True
        for tensor_info in tensor_infos:
            if not tensor_info.is_output:
                continue

            actual_data = buffer_manager.get_output_data(tensor_info.uid)
            if actual_data is None:
                reporter.print_error(
                    f"Failed to get output data for tensor {tensor_info.uid}"
                )
                all_passed = False
                continue

            ref_output = reference_outputs.get(tensor_info.uid)
            if ref_output is None:
                reporter.print_error(
                    f"Reference provider did not produce output for tensor {tensor_info.uid}"
                )
                all_passed = False
                continue

            comparison = comparator.compare(
                actual_data, ref_output.data, "hipDNN", validation_config.provider
            )

            reporter.print_reference_validation(
                provider_name=validation_config.provider,
                passed=comparison.passed,
                max_abs_diff=comparison.max_abs_diff,
                max_rel_diff=comparison.max_rel_diff,
                rtol=validation_config.rtol,
                atol=validation_config.atol,
            )

            if not comparison.passed:
                all_passed = False

        return all_passed

    except ValueError as e:
        reporter.print_error(f"Validation error: {e}")
        return False
    except NotImplementedError as e:
        reporter.print_error(f"Validation error: {e}")
        return False
    except ImportError as e:
        reporter.print_error(f"Validation error: {e}")
        return False


def run_pytorch_benchmark(
    config: BenchmarkConfig,
    seed: Optional[int] = None,
    output_path: Optional[Path] = None,
    device: str = "cuda:0",
) -> int:
    """Run PyTorch CUDA benchmark workflow.

    Args:
        config: Benchmark configuration.
        seed: Optional random seed for reproducibility.
        output_path: Optional path to export benchmark results as JSON.
        device: CUDA device to use.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from ..execution.pytorch_buffer_manager import PyTorchCudaBufferManager
    from ..execution.pytorch_executor import PyTorchCudaExecutor, PyTorchExecutionError

    reporter = Reporter()

    try:
        # Load graph (skip hipDNN-specific validation)
        loader = GraphLoader()
        graph_json = loader.load_json(config.graph_path)

        graph_name = loader.get_graph_name(graph_json)
        tensor_infos = loader.extract_tensor_info(graph_json)

        # Print header
        reporter.print_pytorch_header(config, graph_name, device)

        # Check PyTorch CUDA availability
        try:
            import torch

            if not torch.cuda.is_available():
                reporter.print_error(
                    "PyTorch GPU not available. "
                    "Install PyTorch with CUDA or ROCm support."
                )
                return 1
        except ImportError:
            reporter.print_error(
                "PyTorch not available. Install with: pip install torch"
            )
            return 1

        # Create executor
        executor = PyTorchCudaExecutor(graph_json, config, device=device)
        executor.prepare()

        reporter.print_init_time(executor.init_time_ms)

        # Allocate buffers
        with PyTorchCudaBufferManager(tensor_infos, device=device) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=seed)
            buffer_manager.zero_outputs()

            tensors = buffer_manager.get_tensors()

            # Run warmup
            executor.warmup(tensors)

            # Run benchmark
            result = executor.benchmark(tensors, graph_name=graph_name)

            # Calculate statistics
            stats = CombinedBenchmarkStats.from_result(result)
            reporter.print_combined_stats(stats)

            # Export results if requested
            if output_path:
                result.save_json(str(output_path))
                print(f"Results exported to: {output_path}")

        reporter.print_footer()
        return 0

    except GraphLoadError as e:
        reporter.print_error(f"Graph load error: {e}")
        return 1

    except PyTorchExecutionError as e:
        reporter.print_error(f"PyTorch execution error: {e}")
        return 1

    except Exception as e:
        reporter.print_error(f"Unexpected error: {e}")
        return 1


def run_ab_test(
    config: BenchmarkConfig,
    ab_config: ABTestConfig,
    seed: Optional[int] = None,
    gpu_backend: Literal["torch", "auto", "none"] = "auto",
    validation_config: Optional[ValidationConfig] = None,
) -> int:
    """Run A/B comparison workflow.

    Args:
        config: Benchmark configuration.
        ab_config: A/B test configuration.
        seed: Optional random seed for reproducibility.
        gpu_backend: GPU timer backend to use (torch, auto, none).
        validation_config: Optional validation configuration for reference checking.

    Returns:
        Exit code (0 for success, 1 for error, 2 for comparison failure).
    """
    reporter = Reporter()

    try:
        # Validate plugin paths if specified
        ab_config.validate_paths()

        # Load and validate graph
        loader = GraphLoader()
        graph_json = loader.load_json(config.graph_path)
        loader.validate(graph_json)

        graph_name = loader.get_graph_name(graph_json)

        # Print header
        reporter.print_ab_header(config, ab_config, graph_name)

        # Run A/B comparison
        runner = ABRunner(
            graph_json,
            config,
            ab_config,
            gpu_backend=gpu_backend,
            validation_config=validation_config,
        )
        result = runner.run(seed=seed)

        # Compute combined stats from results
        stats_a = CombinedBenchmarkStats.from_result(result.result_a)
        stats_b = CombinedBenchmarkStats.from_result(result.result_b)

        # Print results with both E2E and kernel stats
        reporter.print_ab_combined_stats(
            stats_a,
            stats_b,
            result.init_time_a_ms,
            result.init_time_b_ms,
        )

        reporter.print_ab_comparison(
            result.passed,
            result.max_abs_diff,
            result.max_rel_diff,
            ab_config.rtol,
            ab_config.atol,
        )

        # Print validation results if available
        if validation_config is not None and validation_config.enabled:
            reporter.print_ab_validation(
                result.validation_a,
                result.validation_b,
                validation_config.rtol,
                validation_config.atol,
            )

        reporter.print_footer()

        # Check validation results
        validation_passed = True
        if result.validation_a is not None and not result.validation_a.passed:
            validation_passed = False
        if result.validation_b is not None and not result.validation_b.passed:
            validation_passed = False

        # Return 0 for pass, 2 for comparison or validation failure
        return 0 if (result.passed and validation_passed) else 2

    except GraphLoadError as e:
        reporter.print_error(f"Graph load error: {e}")
        return 1

    except ExecutionError as e:
        reporter.print_error(f"Execution error: {e}")
        return 1

    except ValueError as e:
        reporter.print_error(f"Configuration error: {e}")
        return 1

    except Exception as e:
        reporter.print_error(f"Unexpected error: {e}")
        return 1


def main() -> int:
    """CLI entry point.

    Returns:
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args()

    try:
        config = BenchmarkConfig(
            graph_path=args.graph,
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
            engine_id=args.engine_id,
        )
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    # Check if A/B testing mode is enabled (either AId or BId specified)
    if args.AId is not None or args.BId is not None:
        # Both AId and BId should be specified for A/B testing
        if args.AId is None or args.BId is None:
            print(
                "A/B testing requires both --AId and --BId to be specified",
                file=sys.stderr,
            )
            return 1

        try:
            ab_config = ABTestConfig(
                a_path=args.APath,
                a_id=args.AId,
                b_path=args.BPath,
                b_id=args.BId,
                rtol=args.rtol,
                atol=args.atol,
            )
        except ValueError as e:
            print(f"A/B configuration error: {e}", file=sys.stderr)
            return 1

        # Create validation config if validation is enabled for A/B test
        ab_validation_config = None
        if args.validate != "none":
            try:
                ab_validation_config = ValidationConfig(
                    provider=args.validate,
                    rtol=args.validate_rtol,
                    atol=args.validate_atol,
                )
            except ValueError as e:
                print(f"Validation configuration error: {e}", file=sys.stderr)
                return 1

        return run_ab_test(
            config,
            ab_config,
            seed=args.seed,
            gpu_backend=args.gpu_backend,
            validation_config=ab_validation_config,
        )

    # Route based on execution backend
    if args.backend == "pytorch":
        return run_pytorch_benchmark(
            config,
            seed=args.seed,
            output_path=args.output,
        )

    # Create validation config if validation is enabled
    validation_config = None
    if args.validate != "none":
        try:
            validation_config = ValidationConfig(
                provider=args.validate,
                rtol=args.validate_rtol,
                atol=args.validate_atol,
            )
        except ValueError as e:
            print(f"Validation configuration error: {e}", file=sys.stderr)
            return 1

    return run_benchmark(
        config,
        seed=args.seed,
        validation_config=validation_config,
        output_path=args.output,
        gpu_backend=args.gpu_backend,
    )


if __name__ == "__main__":
    sys.exit(main())
