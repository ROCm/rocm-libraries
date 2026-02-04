# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import math
from typing import Iterable, Optional

import torch

import origami

"""
Origami: Analytical GEMM Solution Selection

Python bindings for the Origami C++ library.
"""


class OrigamiMatmulSelector:
    """
    Analytical GEMM configuration selector for GPUs.
    
    This class uses the Origami analytical model to select optimal GEMM kernel
    configurations based on problem dimensions, data types, and hardware characteristics.
    It provides a high-level interface for integrating with PyTorch-based frameworks.
    
    The selector analyzes a set of candidate configurations and predicts their performance
    using analytical models of compute, memory, and synchronization costs. It returns
    the configuration with the lowest predicted latency.
    
    Attributes:
        dtype_to_str (dict): Mapping from PyTorch dtypes to Origami dtype strings.
    """
    # https://docs.pytorch.org/docs/stable/tensors.html
    dtype_to_str = {
        torch.float32: "f32",
        torch.complex64: "c32",
        torch.complex128: "c64",
        torch.float64: "f64",
        torch.float16: "f16",
        torch.int32: "i32",
        torch.bfloat16: "bf16",
        torch.int8: "i8",
        torch.float8_e5m2: "f8",
        torch.float8_e4m3fn: "f8",
    }
    # Add FP8 FNUZ variants if available (for non-gfx950 architectures)
    if hasattr(torch, "float8_e5m2fnuz"):
        dtype_to_str[torch.float8_e5m2fnuz] = "f8"
    if hasattr(torch, "float8_e4m3fnuz"):
        dtype_to_str[torch.float8_e4m3fnuz] = "f8"


    def __init__(
        self,
        config_gen: Iterable,
        m: int,
        n: int,
        k: int,
        a_dtype: torch.dtype,
        b_dtype: torch.dtype,
        out_dtype: torch.dtype,
        device: torch.device,
        a_stride: Optional[tuple[int, ...]] = None,
        b_stride: Optional[tuple[int, ...]] = None,
        batch: int = 1,
        mx_block_size=0,
        streamk=False
    ):
        """
        Initialize the Origami matmul configuration selector.
        
        Args:
            config_gen: Iterable of Triton-style config objects with kwargs containing
                       'BLOCK_M', 'BLOCK_N', 'BLOCK_K', and 'waves_per_eu'.
            m: M dimension of the GEMM (number of rows in A and output).
            n: N dimension of the GEMM (number of columns in B and output).
            k: K dimension of the GEMM (shared dimension between A and B).
            a_dtype: PyTorch dtype for matrix A.
            b_dtype: PyTorch dtype for matrix B.
            out_dtype: PyTorch dtype for output matrix.
            device: PyTorch CUDA device (must be ROCm-capable).
            a_stride: Tuple of PyTorch tensor strides for matrix A (default: None).
            b_stride: Tuple of PyTorch tensor strides for matrix B (default: None).
            batch: Batch size of the GEMM (default: 1).
            mx_block_size: Block size for MX format dtypes (default: 0 for non-MX types).
            streamk: Whether to use StreamK scheduling (default: False).
        
        Raises:
            RuntimeError: If no ROCm-capable device is detected.
            ValueError: If the hardware architecture is unsupported or data types are incompatible.
        """
        # Save tensor sizes
        self._m = m
        self._n = n
        self._k = k

        # Save tensor strides
        self._a_stride = a_stride
        self._b_stride = b_stride

        # Save batch size
        self._batch = batch

        # Save tensor dtypes as strings
        self._a_dtype_str   = OrigamiMatmulSelector.dtype_to_str.get(a_dtype, a_dtype)
        self._b_dtype_str   = OrigamiMatmulSelector.dtype_to_str.get(b_dtype, b_dtype)
        self._out_dtype_str = OrigamiMatmulSelector.dtype_to_str.get(out_dtype, out_dtype)
        
        # Save MX block size
        self._mx_block_size = mx_block_size

        # Helper function to get bits for both float, int, and MX dtypes
        mx_types = ["f4"]
        def get_dtype_bits(dtype):
            # Handle MX types (string-based)
            if dtype in mx_types:
                return origami.datatype_to_bits(origami.string_to_datatype(dtype))

            # Handle torch dtypes
            try:
                return torch.finfo(dtype).bits
            except TypeError:
                return torch.iinfo(dtype).bits
        self._a_dtype_bitsize = get_dtype_bits(a_dtype)
        self._b_dtype_bitsize = get_dtype_bits(b_dtype)
        self._out_dtype_bitsize = get_dtype_bits(out_dtype)

        # For matrix instruction latency lookup, use input dtype (not output dtype)
        # because the matrix instruction type is determined by input operand types
        # Example: FP8 inputs with BF16 output still uses FP8 matrix instructions
        # Set MI dtype - use string for MX types, otherwise lookup from dict
        if a_dtype in mx_types:
            self.mi_dtype = a_dtype
        else:
            input_dtype_for_mi = (
                a_dtype
                if get_dtype_bits(a_dtype) <= get_dtype_bits(b_dtype)
                else b_dtype
            )
            self.mi_dtype = OrigamiMatmulSelector.dtype_to_str.get(
                input_dtype_for_mi, OrigamiMatmulSelector.dtype_to_str.get(out_dtype)
            )

        # Get hardware info from Origami
        self._hardware = origami.get_hardware_for_device(device.index)
        self._N_CU = self._hardware.N_CU
        
        # Create Origami problem_t based on problem metadata
        self._problem = self._make_problem()

        # Create list of Origami config_t objects based on generator.
        self._configs = self._generate_configs(config_gen)

        # Run Origami solution selection
        self._result = origami.select_config(self._problem,
                                             self._hardware,
                                             self._configs)

        if streamk:
            self._grid = origami.select_grid_size(self._problem,
                                                  self._hardware,
                                                  self._result.config,
                                                  origami.grid_selection_t.k_split_aware,
                                                  self._hardware.N_CU)
        else:
            self._grid = origami.select_grid_size(self._problem,
                                                  self._hardware,
                                                  self._result.config,
                                                  origami.grid_selection_t.data_parallel,
                                                  self._hardware.N_CU)

        self._workgroup_mapping = (
            origami.select_workgroup_mapping(self._problem,
                                             self._hardware,
                                             self._result.config,
                                             self._grid)
        )


    @property
    def macrotile_m(self):
        """
        M dimension of the selected macrotile (block size in M dimension).
        
        Returns:
            int: Number of rows processed per workgroup.
        """
        return self._result.config.mt.m


    @property
    def macrotile_n(self):
        """
        N dimension of the selected macrotile (block size in N dimension).
        
        Returns:
            int: Number of columns processed per workgroup.
        """
        return self._result.config.mt.n


    @property
    def macrotile_k(self):
        """
        K dimension of the selected macrotile (block size in K dimension).
        
        Returns:
            int: Number of elements in the reduction dimension processed per iteration.
        """
        return self._result.config.mt.k


    @property
    def wgm(self):
        """
        Workgroup mapping parameter for tiling in the M-N plane.
        
        This controls how output tiles are grouped together for better cache locality.
        
        Returns:
            int: Workgroup mapping size.
        """
        return self._workgroup_mapping.wgm
    
    @property
    def wgmxcc(self):
        """
        Workgroup mapping size across XCCs (chiplets).
        
        For multi-chiplet GPUs (e.g., MI300 series), this controls how work is
        distributed across the different chiplets.
        
        Returns:
            int: Number of workgroups mapped per XCC.
        """
        return self._workgroup_mapping.wgmxcc
    
    @property
    def wgmxccchunk(self):
        """
        Workgroup mapping chunk size for XCC distribution.
        
        Controls the granularity of work distribution across chiplets.
        
        Returns:
            int: Chunk size for XCC workgroup mapping.
        """
        return self._workgroup_mapping.wgmxccchunk

    @property
    def number_of_cus(self):
        """
        Number of compute units (CUs) available on the target GPU.
        
        This is a hardware property that indicates the total parallelism available.
        Common values:
        - MI200 (gfx90a): 104 CUs
        - MI300A (gfx942): 228 CUs
        - MI300X (gfx942): 304 CUs
        - MI350 (gfx950): 256 CUs
        
        Returns:
            int: Total number of compute units on the device.
        """
        return self._hardware.N_CU


    @property
    def occupancy(self):
        """
        Number of wavefronts resident per SIMD unit (occupancy).
        
        Higher occupancy can hide memory latency but reduces register availability.
        Typical values are 1-4 for efficient schedules, though hardware supports more.
        
        Returns:
            int: Number of concurrent wavefronts per EU (1-64, typically 1-2 for performance).
        """
        return self._result.config.occupancy


    @property
    def even_k(self):
        """
        Whether the K dimension is evenly divisible by the macrotile K size.
        
        When True, the kernel can avoid bounds checking in the K loop, improving performance.
        When False, the kernel must handle the remainder iteration.
        
        Returns:
            bool: True if K is evenly divisible by macrotile_k, False otherwise.
        """
        return math.gcd(self._k, self.macrotile_k) == self.macrotile_k


    @property
    def grid_size(self):
        """
        Grid size (number of workgroups) to launch for the kernel.
        
        For StreamK scheduling, this is analytically determined to balance
        load across CUs while minimizing synchronization overhead.
        
        Returns:
            int: Number of workgroups to launch.
        """
        return self._grid


    def _generate_configs(self, config_gen) -> [origami.config_t]:
        """
        Convert Triton-style configs to Origami config_t objects.
        
        Takes an iterable of Triton autotuner configs and converts them to
        Origami's internal config_t representation, inferring matrix instruction
        dimensions based on hardware and data types.
        
        Args:
            config_gen: Iterable of config objects with kwargs containing
                       'BLOCK_M', 'BLOCK_N', 'BLOCK_K', and 'waves_per_eu'.
        
        Returns:
            list[origami.config_t]: List of Origami configuration objects.
        """
        configs_list = []

        # Get recommended matrix instruction dimensions
        mi = self._hardware.get_recommended_matrix_instruction(self._problem.mi_dtype)

        for config in config_gen:
            # config is type triton.runtime.autotuner.Config

            # Create special dim3_t object for BLK_* sizes (macrotile dimensions)
            mt = origami.dim3_t(config.kwargs['BLOCK_M'],
                                config.kwargs['BLOCK_N'],
                                config.kwargs['BLOCK_K'])

            # Create and set new config_t values
            new_config           = origami.config_t()
            new_config.mt        = mt
            new_config.mi        = mi
            new_config.occupancy = config.kwargs['waves_per_eu']

            configs_list.append(new_config)

        return configs_list


    def _make_problem(self) -> origami.problem_t:
        """
        Create an Origami problem_t object from the GEMM problem specification.
        
        Converts PyTorch-style problem parameters (dimensions, dtypes) into
        Origami's internal problem representation. Assumes standard GEMM
        operation: D = A^T @ B + C (with A transposed).
        
        Returns:
            origami.problem_t: Problem specification for Origami analytical model.
        """
        # Create special dim3_t object for problem sizes
        size = origami.dim3_t(self._m, self._n, self._k)

        # Convert torch dtypes to Origami dtypes based on problem metadata
        a_origami_dtype = origami.string_to_datatype(self._a_dtype_str)
        b_origami_dtype = origami.string_to_datatype(self._b_dtype_str)
        c_origami_dtype = origami.string_to_datatype(self._out_dtype_str)

        # Calculate transpose types based on tensor strides
        a_transpose = self._check_transpose_type((self._m, self._k),
                                                 self._a_stride,
                                                 default=origami.transpose_t.T)
        b_transpose = self._check_transpose_type((self._k, self._n),
                                                 self._b_stride,
                                                 default=origami.transpose_t.N)

        # Create and set new problem_t values
        problem = origami.problem_t()
        problem.size            = size
        problem.batch           = self._batch
        problem.a_transpose     = a_transpose
        problem.b_transpose     = b_transpose
        problem.a_dtype         = a_origami_dtype
        problem.b_dtype         = b_origami_dtype
        problem.c_dtype         = c_origami_dtype
        problem.d_dtype         = c_origami_dtype
        problem.mi_dtype        = c_origami_dtype
        problem.a_mx_block_size = self._mx_block_size
        problem.b_mx_block_size = self._mx_block_size
    
        return problem


    def _check_transpose_type(
        self,
        shape: tuple[int, int],
        strides: Optional[tuple[int, ...]] = None,
        default: Optional[origami.transpose_t] = origami.transpose_t.N
    ) -> origami.transpose_t:
        """
        Determine transpose type from tensor shape and stride pattern.

        Analyzes the memory layout of a matrix by examining its strides to determine
        whether it is stored in row-major (non-transposed) or column-major (transposed)
        format. If strides are not provided, returns the default transpose type.

        Args:
            shape: Tuple of (m, n) dimensions for the matrix.
            strides: Optional tuple of PyTorch tensor strides. If None, the default
                    transpose type is returned without analysis.
            default: Transpose type to use when strides is None or as a starting point
                    (default: origami.transpose_t.N).

        Returns:
            origami.transpose_t: The determined transpose type.

        Raises:
            ValueError: If the matrix has a non-dense memory layout or an unsupported 
                       stride pattern.
        """
        # Start with supplied default transpose type as fallback
        transpose_t = default

        if strides is not None:
            m, n = shape
            # Make sure we check the last 2 values of the stride in case this
            # is a high-order tensor
            stride_m, stride_n = strides[-2], strides[-1]

            # Degenerate cases
            if m == 1 or n == 1:
                if stride_n == 1:
                    transpose_t = origami.transpose_t.N
                elif stride_m == 1:
                    transpose_t = origami.transpose_t.T
                else:
                    raise ValueError(
                        f"[ROCm:ORIGAMI]: A matrix with strides ({stride_m}, "
                        f"{stride_n}) appears to be a non-dense vector in memory."
                    )
            # Row-major dense case
            elif stride_n == 1:
                transpose_t = origami.transpose_t.N
            # Tranposed-view of row-major dense case
            elif stride_m == 1:
                transpose_t = origami.transpose_t.T
            # Unknown stride pattern
            else:
                raise ValueError(
                    f"[ROCm:ORIGAMI] A matrix with strides ({stride_m}, "
                    f"{stride_n}) is an unsupported layout." 
                )

        return transpose_t

