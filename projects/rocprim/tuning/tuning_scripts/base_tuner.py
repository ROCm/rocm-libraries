#!/usr/bin/env python3

# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

from abc import ABC, abstractmethod
from typing import List, Optional, OrderedDict, Dict
from kernel_tuner.file_utils import store_metadata_file, store_output_file
import kernel_tuner
import json
from pathlib import Path
import numpy as np
from jinja2 import Environment, FileSystemLoader
from utils import TYPE_CONFIGS, ConfigParser
from hip import hip
import warnings
from dataclasses import dataclass
from utils import TYPE_CONFIGS, Parser
import pathlib

"""
The following base class is used when implementing the tuning for new algorithms
using Kernel Tuner. For a detailed example, check out the tuning of device_merge.

The Tuner class manages the complete tuning workflow, handling parameter configurations,
device memory, result collection, and creates the necessary C++
wrapper code that interfaces between Kernel Tuner and rocPRIM templates. Since we are
tuning templated functions in rocPRIM rather than direct kernels, we need to create C
extern wrappers that return a float representing execution time. This wrapping is
necessary because we cannot know the C++ mangled name of the compiled templated function
ahead of time. For each combination of key-value types, the Tuner creates a
unique header file. We then use Kernel Tuner's C compiler functionality to compile and
benchmark these wrappers, rather than using its HIP backend. This is because we're
tuning complete algorithm functions that may internally launch one or more kernels,
rather than tuning individual kernels directly. The wrapper files serve as the
interface between Kernel Tuner and the rocPRIM algorithm implementations, allowing us to
measure and optimize performance across the entire algorithm execution. A base template
is available to use with the different algorithms. This template can be constomized
by implementing the different jinja blocks when extending the base template.

For each type combination, the tuning workflow in the Tuner class prepares the specific
parameters needed for tune_kernel, including tuning parameters like block sizes and
items per thread, function arguments from the wrapper, problem size specifications, and
parameter space restrictions. It also handles compiler configuration and strategy
options. A key feature integrated into the tuning process is the ability to
incorporate previous best configurations from older tuning runs. This ensures
continuity and improvement - new tuning results will be either equivalent to or better
than previous ones. This is achieved by parsing the existing configuration header file
to extract relevant configs for the current type combination being tuned. All this
happens with the ConfigParser class in config_parser.py. These configurations are added
to the parameter space, and if the chosen strategy doesn't naturally explore them, they
are benchmarked separately to ensure inclusion in the final results.

The final output consists of JSON files containing the tuning results, organized by
architecture, algorithm name, and data type combinations. These files include time,
performance metrics, parameter configurations, and optional metadata for analysis. One
particularly useful aspect of using Kernel Tuner is its ability to pause and resume
tuning sessions without losing previous progress, thanks to the caching system that
utilizes these JSON output files.

To adapt this framework for a new algorithm, you'll need to implement the Tuner and a
extend the jinja base template for your specific algorithm requirements. The device_merge
implementation serves as a comprehensive example.

For detailed information:
Kernel Tuner documentation: https://kerneltuner.github.io/kernel_tuner/stable/contents.html
ROCm HIP Python Wrapper: https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html
"""

@dataclass
class TunerArgs:
    algo_full_name: Optional[str] = None
    """The full algorithm name, with type and name.
    """

    size: int = 1024 * 1024 * 32 * 4
    """Size in bytes for the problem/data to tune with.
    """

    max_fevals: int = 100
    """Maximum number of function evaluations for the tuning strategy
    """

    strategy = "dual_annealing"
    """Tuning strategy to use (e.g. "dual_annealing", "brute_force").
    """

    output_dir: str = "../output"
    """Directory path to store tuning results and metadata.
    """

    include_default_config: bool = True
    """Whether to also test default configurations.
    """

    save_metadata: bool = False
    """Whether to save additional tuning metadata.
    """

    simulation_mode: bool = False
    seed: int = -1
    arch_name: Optional[str] = None

    def update_with_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__dict__.keys() and v:
                self.__dict__[k] = v

class BaseTuner(ABC):
    def __init__(
        self,
        args: TunerArgs
    ):
        """Initialize the tuner with configuration parameters.
        """
        self.algo_full_name = args.algo_full_name
        algo_types = ["warp", "block", "device"]
        algo_full_name_parts = args.algo_full_name.split("_")
        if algo_full_name_parts[0] in algo_types:
            self.algo_type = algo_full_name_parts[0]
            self.algo_name = "_".join(algo_full_name_parts[1:])
        else:
            raise ValueError(
                f"Invalid algo_type in {args.algo_full_name}. Must start with one of {algo_types}"
            )
        self.device_id = 0
        self.bytes_size = args.size
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.simulation_mode = args.simulation_mode
        if not self.simulation_mode:
            # TODO: remove or replace this to get rid of hip-python
            self.device_properties = hip.hipDeviceProp_t()
            hip.hipGetDeviceProperties(self.device_properties, self.device_id)
            self.arch_name = self.device_properties.gcnArchName.decode().split(":")[0]
        else:
            assert args.arch_name, "Simulation mode requires --arch-name to be specified"
            self.arch_name = args.arch_name
        self.include_default_config = args.include_default_config
        self.max_fevals = args.max_fevals
        self.save_metadata = args.save_metadata
        self.strategy = args.strategy
        self.seed = None if args.seed == -1 else args.seed

        if self.include_default_config:
            self.config_parser = ConfigParser(
                self.algo_name, self._get_tune_params().keys()
            )
            with open(
                f"../../rocprim/include/rocprim/device/detail/config/{self.algo_full_name}.hpp"
            ) as f:
                self.default_configs = self.config_parser.parse_header(f.read())
        else:
            self.config_parser = None
            self.default_configs = None

    @abstractmethod
    def _get_default_args() -> TunerArgs:
        """Returns the default arguments. Used to generate default values for the CLI."""
        pass

    @classmethod
    def cli(cls):
        defaults: TunerArgs = cls._get_default_args()
        args = Parser.get_parser(
            default_bytes=defaults.size,
            default_max_feval=defaults.max_feval,
            default_output_dir=defaults.output_dir,
            default_strategy=defaults.strategy,
        )
        cls(args).tune_all()

    @abstractmethod
    def _get_tune_params(self) -> OrderedDict:
        """Returns tuning parameters and their possible values as an OrderedDict.
        Each parameter maps to a list of valid values to explore during tuning."""
        pass

    @abstractmethod
    def _get_restrictions(
        self, key_type: str, value_type: Optional[str] = None
    ) -> List[str]:
        """Define constraints for what parameter combinations are valid during tuning.

        Two options:
        1. Return list of string expressions that must all evaluate to True
            Example: ["block_size_x * ipt <= size", "block_size_x <= 1024"]
            These expressions are evaluated as boolean conditions by the Kernel Tuner.
            Each string must be a valid Python expression using the parameter names.

        2. Return a callable (function/lambda) that takes a params dictionary as input
            and returns True/False to validate the configuration.

            See Kernel Tuner's documentation for more details
        """
        pass

    def _get_grid_div_x(self):
        """Return the grid_div_x parameter for kernel_tuner.tune_kernel()"""
        return ["block_size_x"]

    @abstractmethod
    def tune_all(self) -> None:
        """Call tune_type for all key type and value type combinations"""
        pass

    def _get_problem_size(self, key_type: str, value_type: Optional[str] = None):
        return self.bytes_size // (
            TYPE_CONFIGS[key_type].size + (TYPE_CONFIGS[value_type].size if value_type else 0)
        )

    def _get_metrics(self, key_type: str, value_type: Optional[str] = None) -> Dict:
        """Default metrics for performance measurement."""
        total_bytes = self._get_problem_size(key_type, value_type) * (
            TYPE_CONFIGS[key_type].size
            + (TYPE_CONFIGS[value_type].size if value_type else 0)
        )
        # p["time"] is actually measured in bytes per second
        return {"GB/s": lambda p: (p["time"] / 1e6)}

    def _run_default_config(
        self, tune_kernel_args: Dict, key_type: str, value_type: Optional[str] = None
    ) -> None:
        """Runs the default configuration if enabled."""
        if not self.include_default_config or self.simulation_mode:
            return

        print("Running default config")
        default_tune_params = self.config_parser.get_default_config(
            self.arch_name, key_type, value_type
        )
        if default_tune_params is None:
            warnings.warn("No tuning parameters extracted from default config. Skipping!")
            return

        default_args = tune_kernel_args.copy()
        default_args.update(
            {"tune_params": default_tune_params, "strategy": "brute_force"}
        )

        kernel_tuner.tune_kernel(**default_args)

    def tune_type(
        self,
        key_type: str,
        value_type: Optional[str] = None,
    ) -> None:
        """Performs auto-tuning for a specific key type and optional value type combination."""
        print(f"\nTuning for {key_type} {value_type if value_type else ''}")
        print(f"Using size: {self.bytes_size} bytes")
        strategy_print_message = (
            f"Using strategy: {self.strategy if self.strategy else 'brute_force'}"
        )
        strategy_print_message += (
            f", max fevals: {self.max_fevals}" if self.strategy != "brute_force" else ""
        )
        print(strategy_print_message)

        try:
            tune_kernel_args = self._get_base_tune_kernel_args(key_type, value_type)
            # Run main tuning
            results, _ = kernel_tuner.tune_kernel(**tune_kernel_args)
            # Run default config if enabled
            self._run_default_config(tune_kernel_args, key_type, value_type)

            cache_file_path = self._get_cache_file_path(key_type, value_type)
            self._save_output(cache_file_path, key_type, value_type, results)

        except Exception as e:
            print(f"Failed tuning for {key_type} {value_type if value_type else ''}")
            print(f"Error: {str(e)}")
            raise

    def generate_wrapper(
        self,
        tune_params: OrderedDict,
        key_type: str,
        value_type: Optional[str] = None,
        output_file: bool = False,
    ) -> str:
        """Generate wrapper code using Jinja2 template inheritance."""
        template_dir = pathlib.Path("templates")
        env = Environment(
            loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True
        )

        # We first check if a specialized template exists, e.g.
        # 'device_merge_wrapper_template' if it doesn't exist, we 
        # fallback to 'base_wrapper_template'.
        template_name = f"{self.algo_full_name}_wrapper_template"
        if not template_dir.joinpath(template_name).exists():
            template_name = "base_wrapper_template"

        template = env.get_template(template_name)
        context = {
            "algo_type": self.algo_type,
            "algo_name": self.algo_name,
            "config_params": ", ".join(tune_params.keys()),
            "key_type": key_type,
            "value_type": value_type,
        }

        content = template.render(**context)

        if output_file:
            base_name = f"device_{self.algo_name}_wrapper"
            file_name = (
                f"{base_name}_{value_type}.hpp" if value_type else f"{base_name}.hpp"
            )
            with open(file_name, "w") as f:
                f.write(content)

        return content

    def _get_base_tune_kernel_args(
        self,
        key_type: str,
        value_type: Optional[str] = None,
    ) -> Dict:
        """Returns base arguments for kernel_tuner.tune_kernel()."""
        np.random.seed(self.seed)

        wrapper_string = self.generate_wrapper(
            tune_params=self._get_tune_params(),
            key_type=key_type,
            value_type=value_type,
        )

        tune_kernel_args = {
            "kernel_name": f"{self.algo_full_name}_wrapper",
            "kernel_source": wrapper_string,
            "problem_size": self._get_problem_size(key_type, value_type),
            "arguments": [np.uint64(self.bytes_size)],
            "tune_params": self._get_tune_params(),
            "strategy": self.strategy,
            "grid_div_x": self._get_grid_div_x(),
            "cache": str(self._get_cache_file_path(key_type, value_type)),
            "lang": "C",
            "compiler": "hipcc",
            "compiler_options": self._get_compiler_options(),
            "restrictions": self._get_restrictions(key_type, value_type),
            "verbose": True,
            "iterations": 1,
            "log": False,
            "device": self.device_id,
            "simulation_mode": self.simulation_mode,
            "objective_higher_is_better": True,
        }

        if self.strategy != "brute_force":
            tune_kernel_args["strategy_options"] = {"max_fevals": self.max_fevals}

        return tune_kernel_args

    def _get_cache_file_name(self, key_type: str, value_type: str = None):
        """Return the name of the cache file based on algo name, arch name and key value types"""
        cache_file_path = f'{self.algo_full_name}_{self.arch_name}_{key_type.replace("rocprim::", "")}'
        if value_type:
            cache_file_path += f"_{value_type.replace("rocprim::", "")}"
        cache_file_path += "_cache.json"

        return cache_file_path

    def _get_cache_file_path(self, key_type: str, value_type: str = None):
        "Return the path of the cache file"
        return self.output_dir / self._get_cache_file_name(key_type, value_type)

    def _save_output(
        self,
        cache_file: str,
        key_type,
        value_type,
        results: Dict = None,
    ):
        """Save tuning results and metadata to files."""
        if self.simulation_mode:
            assert results
            cache_file = Path(
                f"../simulated_output/{self.strategy}_fevals{self.max_fevals}"
            )
            cache_file.mkdir(parents=True, exist_ok=True)
            cache_file = cache_file / self._get_cache_file_name(
                self.algo_full_name, key_type, value_type
            )
            store_output_file(str(cache_file), results, self.tune_params)

        with open(cache_file, "r") as f:
            cache_dict = json.load(f)
        if "arch_name" not in cache_dict:
            with open(cache_file, "r") as f:
                content = f.read()

            new_content = "{\n"
            new_content += f'"arch_name": "{self.arch_name}",\n'
            new_content += f'"algo_name": "{self.algo_full_name}",\n'
            new_content += f'"key_type": "{key_type}",\n'
            value_type_string = f"{value_type}" if value_type else "empty_type"
            new_content += f'"value_type": "{value_type_string}",'
            new_content += content.lstrip()[1:]

            with open(cache_file, "w") as f:
                f.write(new_content)

        if self.save_metadata:
            store_metadata_file(cache_file.replace("cache", "metadata"))

    def _get_compiler_options(self) -> List[str]:
        """Returns a list with all compiler options to pass to Kernel Tuner"""
        monorepo_dir = pathlib.Path('../../../..').resolve()
        rocprim_dir = monorepo_dir / 'projects/rocprim'
        return [
            "-fPIC",
            "-std=c++17",
            f"-I{rocprim_dir / 'rocprim/include'}",
            f"-I{rocprim_dir / 'build/rocprim/include/rocprim'}",
            f"-I{rocprim_dir / 'build/release/rocprim/include/rocprim'}",
            f"-I{rocprim_dir / 'benchmark'}",
            f"-I{monorepo_dir / 'shared/primbench'}",
            "-Wno-#pragma-messages",
            f"--offload-arch={self.arch_name}",
            "-lamd_smi",
        ]
