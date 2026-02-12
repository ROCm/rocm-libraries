# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Origami: Analytical GEMM Solution Selection

Python bindings for the Origami C++ library.
"""

try:
    # Import the compiled extension module
    from .origami import (
        # Enums
        architecture_t,
        data_type_t,
        transpose_t,
        grid_selection_t,
        reduction_t,
        prediction_modes_t,
        # Data structures
        dim3_t,
        dim4_t,
        tensile_params_t,
        config_t,
        prediction_result_t,
        workgroup_mapping_t,
        staggerU_t,
        problem_t,
        hardware_t,
        context_t,
        # Hardware functions
        get_hardware_for_device,
        get_hardware_for_arch,
        # Data type functions
        int_to_data_type,
        datatype_to_bits,
        string_to_datatype,
        datatype_to_string,
        # Configuration selection functions
        select_config,
        rank_configs,
        select_config_mnk,
        select_topk_configs,
        # Performance functions
        compute_perf_gflops,
        compute_total_latency,
        compute_number_matrix_instructions,
        compute_mt_compute_latency,
        # Memory functions
        check_lds_capacity,
        estimate_l2_hit,
        estimate_mall_hit,
        compute_memory_latency,
        compute_l2_tiles,
        compute_mall_tiles,
        predict_workgroup_mapping,
        wgm_to_grid,
        count_unique_tiles,
        count_unique_tiles_timestep,
        estimate_cache_hit_rates,
        # Latency functions
        compute_tile_latency,
        compute_timestep_latency,
        # StreamK functions
        select_grid_size,
        select_reduction,
        select_workgroup_mapping,
        compute_number_of_output_tiles,
        # Reduction functions
        int_to_reduction_t,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import origami extension module: {e}. "
        "Please ensure the package is properly installed."
    ) from e

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Enums
    "architecture_t",
    "data_type_t",
    "transpose_t",
    "grid_selection_t",
    "reduction_t",
    "prediction_modes_t",
    # Data structures
    "dim3_t",
    "dim4_t",
    "tensile_params_t",
    "config_t",
    "prediction_result_t",
    "workgroup_mapping_t",
    "problem_t",
    "hardware_t",
    "context_t",
    # Hardware functions
    "get_hardware_for_device",
    "get_hardware_for_arch",
    # Data type functions
    "int_to_data_type",
    "datatype_to_bits",
    "string_to_datatype",
    "datatype_to_string",
    # Configuration selection functions
    "select_config",
    "rank_configs",
    "select_config_mnk",
    "select_topk_configs",
    # Performance functions
    "compute_perf_gflops",
    "compute_total_latency",
    "compute_number_matrix_instructions",
    "compute_mt_compute_latency",
    # Memory functions
    "wgm_to_grid",
    "compute_l2_tiles",
    "compute_mall_tiles",
    "count_unique_tiles",
    "count_unique_tiles_timestep",
    "estimate_cache_hit_rates",
    "check_lds_capacity",
    "estimate_l2_hit",
    "estimate_mall_hit",
    "compute_memory_latency",
    # Latency functions
    "compute_tile_latency",
    "compute_timestep_latency",
    # StreamK functions
    "select_grid_size",
    "select_reduction",
    "select_workgroup_mapping",
    "compute_number_of_output_tiles",
    # Reduction functions
    "int_to_reduction_t",
]

try:
    # Import the python selector if possible (requires torch)
    from .selector import OrigamiMatmulSelector
    __all__.append("OrigamiMatmulSelector")
except ImportError:
    # Do not raise this error if import fails - compiled Origami bindings still
    # work without the dedicated Python selector
    pass

