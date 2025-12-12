
import functools
import itertools
import math
import torch
from typing import Iterable
import origami


class TorchMatmulHeuristic:
    def __init__(self,
                 torch_configs: Iterable,
                 m: int,
                 n: int,
                 k: int,
                 a_dtype: torch.dtype,
                 b_dtype: torch.dtype,
                 c_dtype: torch.dtype,
                 device: torch.device,
                 streamk=False):
        # Save tensor sizes
        self._m = m
        self._n = n
        self._k = k

        # Save tensor dtypes
        self._a_dtype = a_dtype
        self._b_dtype = b_dtype
        self._c_dtype = c_dtype

        # Save tensor dtype byte sizes
        self._a_dtype_bytes = a_dtype.itemsize
        self._b_dtype_bytes = b_dtype.itemsize
        self._c_dtype_bytes = c_dtype.itemsize

        # Get hardware info from Origami
        self._hardware = origami.get_hardware_for_device(device.index)
        self._N_CU = self._hardware.N_CU

        # Create list of Origami config_t objects based on generator from torch.
        self._configs = self._generate_configs(torch_configs)

        # Create Origami problem_t based on problem metadata
        self._problem = self._make_problem()

        # Run Origami solution selection
        self._result = origami.select_config(self._problem,
                                            self._hardware,
                                            self._configs)

        if streamk:
            self._grid = self._compute_sk_grid()
        else:
            self._grid = self._hardware.N_CU

        _, self._workgroup_mapping = origami.select_workgroup_mapping(self._problem,
                                                                      self._hardware,
                                                                      self._result.config,
                                                                      self._grid)

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
        return self._workgroup_mapping


    @property
    def waves_per_eu(self):
        return self._result.config.occupancy


    @property
    def even_k(self):
        return math.gcd(self._k, self.block_k) == self.block_k


    @property
    def sk_grid(self):
        return self._grid


    def _compute_sk_grid(self):
        # Grid model constants for StreamK
        split_factors  = [8, 6, 4, 3, 2, 1]
        tile_fractions = [0.0,
                          1.0 / 2.0,
                          1.0 / 8.0,
                          1.0 / 5.0,
                          1.0 / 4.0,
                          1.0 / 3.0]
        max_workspace  = 128 * 1024 * 1024

        M, N, K = self._m, self._n, self._k
        BLK_M, BLK_N, BLK_K = self._result.config.mt.m, self._result.config.mt.n, self._result.config.mt.k
        cu_count = self._hardware.N_CU

        # Fallback if no better fractional split is found
        tiles = ceil(M / BLK_M) * ceil(N / BLK_N)
        sk_grid = tiles
        iters_per_tile = max(1, ceil(K / BLK_K))

        # More tiles than CUs: try fractional splits to distribute work
        if tiles > cu_count:
            virt_cu_count = cu_count
            # if size_mapping.CUOccupancy > 1:
            # virt_cu_count *= size_mapping.CUOccupancy

            # Try these fractional denominators in order
            min_even_tiles = tiles / virt_cu_count

            for frac in tile_fractions:
                # Compute candidate grid with rounding
                frac_grid = int((tiles / (min_even_tiles + frac)) + 0.5)

                # Skip if this split leaves a remainder AND workspace is too large
                if (
                    tiles % frac_grid != 0
                    and self._partial_tile_size(frac_grid) > max_workspace
                ):
                    continue

                # Accept the first grid no larger than the virtual CU count
                if frac_grid <= virt_cu_count:
                    sk_grid = frac_grid
                    break

        # Fewer tiles than CUs: split along k-dimension up to some factor
        elif tiles < cu_count:
            for factor in split_factors:
                split_grid = tiles * factor
                iters_per_cu = iters_per_tile // factor

                if split_grid <= cu_count and iters_per_cu >= 8:
                    sk_grid = split_grid
                    break

        # Final check: if the chosen grid leaves a remainder AND
        # workspace exceeds what the problem allows, fall back to no split
        if tiles % sk_grid != 0:
            sk_grid = tiles

        if tiles >= cu_count:
            last_wave_remainder = tiles % cu_count
            last_wave_occupancy = last_wave_remainder / cu_count

            # Really bad last wave, which would have originally been compensated for
            # by changing tile size, but triton tile sizes are limited
            if last_wave_remainder < 128 and last_wave_remainder > 0 and cu_count == 304:
                sk_grid = 256
        return sk_grid


    def _partial_tile_size(self, sk_grid: int) -> int:
        """
        Python equivalent of ContractionSolution::partialTileSize.

        workspaceSizePerElemC = (element_size_out bits) / 8 → bytes per output element

        tileSize = BLK_M * BLK_N * workspaceSizePerElemC
        return tileSize * sk_grid
        """
        # get the macro-tile dims you already compute
        BLK_M, BLK_N, GSIZE = self._result.config.mt.m, self._result.config.mt.n, self._result.config.workgroup_mapping

        # bytes per C element
        bytes_per_elem = self._c_dtype_bytes

        # size of one partial tile per WG
        tile_size = BLK_M * BLK_N * bytes_per_elem

        # scale by the number of partial‑tiles per WG
        return tile_size * sk_grid


    def _generate_configs(self, torch_configs):
        configs_list = []

        for tconfig in torch_configs:
            # tconfig is type triton.runtime.autotuner.Config

            # Create special dim3_t object for BLK_* sizes
            mt = origami.dim3_t(tconfig.kwargs['BLOCK_M'],
                                tconfig.kwargs['BLOCK_N'],
                                tconfig.kwargs['BLOCK_K'])
            # Get matrix instruction dimentions, also in dim3_t object
            mi = self._infer_matrix_instruction_dimensions()

            # Create and set new config_t values
            new_config           = origami.config_t()
            new_config.mt        = mt
            new_config.mi        = mi
            new_config.occupancy = tconfig.kwargs['waves_per_eu']

            configs_list.append(new_config)

        return configs_list


    def _make_problem(self) -> origami.problem_t:
        # Create special dim3_t object for problem sizes
        size = origami.dim3_t(self._m, self._n, self._k)

        # Convert torch dtypes to Origami dtypes based on problem metadata
        a_origami_dtype = TorchMatmulHeuristic.torch_dtype_to_origami_dtype(self._a_dtype)
        b_origami_dtype = TorchMatmulHeuristic.torch_dtype_to_origami_dtype(self._b_dtype)
        c_origami_dtype = TorchMatmulHeuristic.torch_dtype_to_origami_dtype(self._c_dtype)

        # Create and set new problem_t values
        problem = origami.problem_t()
        problem.size        = size
        problem.batch       = 1
        problem.a_transpose = origami.transpose_t.T
        problem.b_transpose = origami.transpose_t.N
        problem.a_dtype     = a_origami_dtype
        problem.b_dtype     = b_origami_dtype
        problem.c_dtype     = c_origami_dtype
        problem.d_dtype     = c_origami_dtype
        problem.mi_dtype    = c_origami_dtype
    
        return problem


    def _infer_matrix_instruction_dimensions(self):
        """
        Infers the matrix instruction dimensions based on the hardware configuration
        and the sizes of the input data types.

        Parameters:
            element_bitsize_A (int): The size (in bits) of the elements in matrix A.
            element_bitsize_B (int): The size (in bits) of the elements in matrix B.

        Returns:
            list[int]: A list representing the matrix instruction dimensions [M, N, K].

        Raises:
            ValueError: If the hardware architecture is unsupported or if the data type
            sizes are not compatible with the detected hardware.
        """
        element_bitsize_A = self._a_dtype_bytes * 8
        element_bitsize_B = self._b_dtype_bytes * 8
        largest_bitsize = max(element_bitsize_A, element_bitsize_B)

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


    @staticmethod
    def torch_dtype_to_origami_dtype(torch_dtype: torch.dtype) -> origami.data_type_t:
        origami_dtype = None
        
        #TODO: Add the rest of the types after resolving type differences
        match torch_dtype:
            case torch.float32:
                origami_dtype = origami.data_type_t.Float
            case torch.float64:
                origami_dtype = origami.data_type_t.Double
            case torch.complex32:
                origami_dtype = origami.data_type_t.ComplexFloat
            case torch.complex64:
                origami_dtype = origami.data_type_t.ComplexDouble
            case torch.float16:
                origami_dtype = origami.data_type_t.Half
            case torch.bfloat16:
                origami_dtype = origami.data_type_t.BFloat16
            case torch.int32:
                origami_dtype = origami.data_type_t.Int32
            case torch.int64:
                origami_dtype = origami.data_type_t.Int64
            case torch.int8:
                origami_dtype = origami.data_type_t.Int8
            case _:
                raise RuntimeError(f'Conversion from {torch_dtype} to respective origami dtype not mapped out')

        return origami_dtype

