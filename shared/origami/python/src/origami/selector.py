
import math
from typing import Iterable

import torch

import origami

"""
Origami: Analytical GEMM Solution Selection

Python bindings for the Origami C++ library.
"""


class OrigamiMatmulSelector:
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
        mx_block_size=0,
        streamk=False
    ):
        # Save tensor sizes
        self._m = m
        self._n = n
        self._k = k

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
        
        # Create list of Origami config_t objects based on generator.
        self._configs = self._generate_configs(config_gen)

        # Create Origami problem_t based on problem metadata
        self._problem = self._make_problem()

        # Run Origami solution selection
        self._result = origami.select_config(self._problem,
                                             self._hardware,
                                             self._configs)

        if streamk:
            self._grid = origami.select_grid_size(self._problem,
                                                  self._hardware,
                                                  self._result.config,
                                                  origami.grid_selection_t.data_parallel,
                                                  self._hardware.N_CU)
        else:
            self._grid = self._hardware.N_CU

        self._workgroup_mapping = (
            origami.select_workgroup_mapping(self._problem,
                                             self._hardware,
                                             self._result.config,
                                             self._grid)
        )


    @property
    def block_m(self):
        return self._result.config.mt.m


    @property
    def block_n(self):
        return self._result.config.mt.n


    @property
    def block_k(self):
        return self._result.config.mt.k


    @property
    def group_m(self):
        return self._workgroup_mapping.wgm


    @property
    def num_sms(self):
        return self._xcc_workgroup_mapping.wgmxcc


    @property
    def waves_per_eu(self):
        return self._result.config.occupancy


    @property
    def even_k(self):
        return math.gcd(self._k, self.block_k) == self.block_k


    @property
    def sk_grid(self):
        return self._grid


    def _generate_configs(self, config_gen):
        configs_list = []

        for config in config_gen:
            # config is type triton.runtime.autotuner.Config

            # Create special dim3_t object for BLK_* sizes
            mt = origami.dim3_t(config.kwargs['BLOCK_M'],
                                config.kwargs['BLOCK_N'],
                                config.kwargs['BLOCK_K'])
            # Get matrix instruction dimentions, also in dim3_t object
            mi = self._infer_matrix_instruction_dimensions()

            # Create and set new config_t values
            new_config           = origami.config_t()
            new_config.mt        = mt
            new_config.mi        = mi
            new_config.occupancy = config.kwargs['waves_per_eu']

            configs_list.append(new_config)

        return configs_list


    def _make_problem(self) -> origami.problem_t:
        # Create special dim3_t object for problem sizes
        size = origami.dim3_t(self._m, self._n, self._k)

        # Convert torch dtypes to Origami dtypes based on problem metadata
        a_origami_dtype = origami.string_to_datatype(self._a_dtype_str)
        b_origami_dtype = origami.string_to_datatype(self._b_dtype_str)
        c_origami_dtype = origami.string_to_datatype(self._out_dtype_str)

        # Create and set new problem_t values
        problem = origami.problem_t()
        problem.size            = size
        problem.batch           = 1
        problem.a_transpose     = origami.transpose_t.T
        problem.b_transpose     = origami.transpose_t.N
        problem.a_dtype         = a_origami_dtype
        problem.b_dtype         = b_origami_dtype
        problem.c_dtype         = c_origami_dtype
        problem.d_dtype         = c_origami_dtype
        problem.mi_dtype        = c_origami_dtype
        problem.a_mx_block_size = self._mx_block_size
        problem.b_mx_block_size = self._mx_block_size
    
        return problem


    def _infer_matrix_instruction_dimensions(self):
        """
        Infers the matrix instruction dimensions based on the hardware configuration
        and the sizes of the input data types.  The input dtype sizes are retrieved
        from local object variables.

        Returns:
            origami.dim3_t: An Origami dimension trio containing the matrixinstruction
                dimensions [M, N, K].

        Raises:
            ValueError: If the hardware architecture is unsupported or if the data type
                sizes are not compatible with the detected hardware.
        """
        largest_bitsize = max(self._a_dtype_bitsize, self._b_dtype_bitsize)

        mi_dim = None
        # gfx950
        if self._hardware.N_CU == 256:
            # FP32
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            # FP16/BF16
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 32)
            # F4F6F8
            if largest_bitsize <= 8:
                mi_dim = origami.dim3_t(16, 16, 128)
        # gfx942
        if self._hardware.N_CU == 304:
            # FP32
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            # FP16/BF16
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 16)
            # F8
            if largest_bitsize == 8:
                mi_dim = origami.dim3_t(16, 16, 32)
            # F4F6 -> Unsupported on MI300X
            if largest_bitsize < 8:
                raise ValueError("MI300X doesn't support F4/F6")
        if self._hardware.N_CU == 228:
            # FP32
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            # FP16/BF16
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 16)
            # F8
            if largest_bitsize == 8:
                mi_dim = origami.dim3_t(16, 16, 32)
            # F4F6 -> Unsupported on MI300A
            if largest_bitsize < 8:
                raise ValueError("MI300A doesn't support F4/F6")
        # gfx90a
        if self._hardware.N_CU == 104:
            # FP32
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            # FP16/BF16
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 16)
            if largest_bitsize == 8:
                raise ValueError("MI200 doesn't support F8")
            if largest_bitsize < 8:
                raise ValueError("MI200 doesn't support F4/F6")
        # Architecture Detected is not valid
        if mi_dim == None:
            raise ValueError(
                f"No Valid Matrix Instruction integrated for {element_size_A}-bit or {element_size_B}-bit datatypes"
            )

        return mi_dim

