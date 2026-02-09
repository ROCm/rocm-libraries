#!/usr/bin/env python3

# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from typing import List, Optional, OrderedDict
import numpy as np
from hip import hip
import sys

sys.path.append("../")
from utils import hip_check, TYPE_CONFIGS, Parser
from tuning_scripts.base_tuner import BaseTuner

KEY_TYPES = ["rocprim::int128_t", "int64_t", "int", "short", "int8_t", "double", "float", "rocprim::half"]
VALUE_TYPES = ["rocprim::int128_t", "int64_t", "int", "short", "int8_t"]

DEFAULT_BYTES = 1024 * 1024 * 32 * 4
DEFAULT_ITERATIONS = 32
DEFAULT_MAX_FEVAL = 100
DEFAULT_OUTPUT_DIR = "../output"
DEFAULT_STRATEGY = "dual_annealing"


class Tuner(BaseTuner):
    def __init__(self, **kwargs):
        args = {
            "algo_full_name": "device_merge",
            "bytes_size": DEFAULT_BYTES,
            "iterations": DEFAULT_ITERATIONS,
            "output_dir": DEFAULT_OUTPUT_DIR,
            "include_default_config": True,
            "max_fevals": DEFAULT_MAX_FEVAL,
            "save_metadata": False,
            "strategy": DEFAULT_STRATEGY,
            "simulation_mode": False,
            "arch_name": None,
        }

        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        args.update(filtered_kwargs)
        super().__init__(**args)

    def _get_tune_params(self) -> OrderedDict:
        """Returns tuning parameters and their possible values as an OrderedDict.
        Each parameter maps to a list of valid values to explore during tuning."""
        params = OrderedDict()
        params["block_size_x"] = [64 * i for i in range(1, 17)]
        params["ipt"] = [1, 2] + [4 * i for i in range(1, 65)]
        return params

    def _get_restrictions(
        self, key_type: str, value_type: Optional[str] = None
    ) -> List[str]:
        """Constraints for what parameter combinations are valid during tuning"""
        size = self.bytes_size // TYPE_CONFIGS[key_type].size
        element_size = TYPE_CONFIGS[key_type].size
        if value_type:
            element_size += TYPE_CONFIGS[value_type].size

        def validate(params):
            block_size = params['block_size_x']
            ipt = params['ipt']

            # Total size constraint
            if block_size * ipt > size:
                return False

            # Memory size constraint
            if block_size * ipt * element_size > 65536:
                return False

            # Block size constraint
            if block_size > 1024:
                return False

            # Items per thread constraint - high ipts don't perform well
            if ipt >= block_size:
                return False

            # High ipts on gfx1030 cause HSA_STATUS_ERROR_INVALID_ISA
            if params.get('arch_name') == 'gfx1030' and ipt > 28:
                return False

            return True

        return validate

    def _get_grid_div_x(self):
        """Return the grid_div_x parameter for kernel_tuner.tune_kernel()"""
        return ["block_size_x"]

    def _allocate_device_memory(self, key_type: str, value_type: Optional[str] = None):
        """
        Allocates device memory for merge operation.

        Args:
            key_type: Type of the keys
            value_type: Optional type of the values

        Returns:
            Tuple containing:
            - List of device pointers and other arguments for kernel
            - Dictionary of allocated memory pointers for cleanup
            - Total bytes allocated
            - Total size of arrays
        """
        key_info = TYPE_CONFIGS[key_type]
        size = self.bytes_size // key_info.size
        size1 = size // 2
        size2 = size - size1

        allocated_mem = {}
        if self.simulation_mode:
            kernel_args = [
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
            ]
            if value_type:
                kernel_args.extend([np.array([]), np.array([]), np.array([])])

        else:
            # Generate and sort input data
            if key_type in ["rocprim::int128_t", "rocprim::uint128_t"]:
                keys_input1 = np.zeros(size1, dtype=key_info.numpy_type)
                keys_input2 = np.zeros(size2, dtype=key_info.numpy_type)
                keys_input1['low'] = np.sort(np.random.rand(size1)).astype(keys_input1['low'][0])
                keys_input2['low'] = np.sort(np.random.rand(size2)).astype(keys_input2['low'][0])
            else:
                keys_input1 = np.sort(np.random.rand(size1)).astype(key_info.numpy_type)
                keys_input2 = np.sort(np.random.rand(size2)).astype(key_info.numpy_type)

            # Allocate device memory for keys and configure arrays
            allocated_mem["d_keys_input1"] = hip_check(
                hip.hipMalloc(size1 * key_info.size)
            )
            allocated_mem["d_keys_input2"] = hip_check(
                hip.hipMalloc(size2 * key_info.size)
            )
            allocated_mem["d_keys_output"] = hip_check(
                hip.hipMalloc(size * key_info.size)
            )

            # Copy key data to device
            hip_check(
                hip.hipMemcpy(
                    allocated_mem["d_keys_input1"],
                    keys_input1,
                    size1 * key_info.size,
                    hip.hipMemcpyKind.hipMemcpyHostToDevice,
                )
            )
            hip_check(
                hip.hipMemcpy(
                    allocated_mem["d_keys_input2"],
                    keys_input2,
                    size2 * key_info.size,
                    hip.hipMemcpyKind.hipMemcpyHostToDevice,
                )
            )

            # Create initial argument list for size query
            kernel_args = [
                allocated_mem["d_keys_input1"],
                allocated_mem["d_keys_input2"],
                allocated_mem["d_keys_output"],
            ]

            # Handle value arrays if needed
            if value_type:
                value_info = TYPE_CONFIGS[value_type]

                # Generate and sort input data
                if value_type in ["rocprim::int128_t", "rocprim::uint128_t"]:
                    values_input1 = np.zeros(size1, dtype=value_info.numpy_type)
                    values_input2 = np.zeros(size2, dtype=value_info.numpy_type)
                    values_input1['low'] = np.sort(np.random.rand(size1)).astype(values_input1['low'][0])
                    values_input2['low'] = np.sort(np.random.rand(size2)).astype(values_input2['low'][0])
                else:
                    values_input1 = np.sort(np.random.rand(size1)).astype(value_info.numpy_type)
                    values_input2 = np.sort(np.random.rand(size2)).astype(value_info.numpy_type)

                allocated_mem["d_values_input1"] = hip_check(
                    hip.hipMalloc(size1 * value_info.size)
                )
                allocated_mem["d_values_input2"] = hip_check(
                    hip.hipMalloc(size2 * value_info.size)
                )
                allocated_mem["d_values_output"] = hip_check(
                    hip.hipMalloc(size * value_info.size)
                )

                # Copy value data to device
                hip_check(
                    hip.hipMemcpy(
                        allocated_mem["d_values_input1"],
                        values_input1,
                        size1 * value_info.size,
                        hip.hipMemcpyKind.hipMemcpyHostToDevice,
                    )
                )
                hip_check(
                    hip.hipMemcpy(
                        allocated_mem["d_values_input2"],
                        values_input2,
                        size2 * value_info.size,
                        hip.hipMemcpyKind.hipMemcpyHostToDevice,
                    )
                )

                kernel_args.extend(
                    [
                        allocated_mem["d_values_input1"],
                        allocated_mem["d_values_input2"],
                        allocated_mem["d_values_output"],
                    ]
                )

        # Add sizes to arguments
        kernel_args.extend([np.uint64(size1), np.uint64(size2)])

        return kernel_args, allocated_mem

    def tune_all(self) -> None:
        """Tune for all key type and value type combinations"""
        for key_type in KEY_TYPES:
            self.tune_type(key_type)
            for value_type in VALUE_TYPES:
                self.tune_type(key_type, value_type)


if __name__ == "__main__":
    parser = Parser.get_parser(
        DEFAULT_BYTES,
        DEFAULT_ITERATIONS,
        DEFAULT_MAX_FEVAL,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_STRATEGY,
    )
    args = parser.parse_args()

    tuner = Tuner(
        algo_full_name="device_merge",
        bytes_size=args.size,
        iterations=args.iterations,
        output_dir=args.output_dir,
        include_default_config=args.include_default_config,
        max_fevals=args.max_fevals,
        save_metadata=args.save_metadata,
        strategy=args.strategy,
        simulation_mode=args.simulation_mode,
        arch_name=args.arch_name,
        seed=args.seed,
    )
    tuner.tune_all()
