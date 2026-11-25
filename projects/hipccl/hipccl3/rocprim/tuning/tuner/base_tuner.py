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
from typing import Any, Callable, List, Optional, OrderedDict, Dict
from kernel_tuner.file_utils import store_metadata_file, store_output_file
import kernel_tuner
import json
from pathlib import Path
import numpy as np
from jinja2 import Environment, FileSystemLoader
from utils import TYPE_CONFIGS, Parser, BASE_DIR
from hip import hip  # type: ignore (pyright doesn't detect hip-python correctly)
import warnings
from dataclasses import dataclass
import confgen.parse
import confgen.generate
import pathlib
import re

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

COMMON_KEY_TYPES = [
    "rocprim::int128_t",
    "int64_t",
    "int",
    "short",
    "int8_t",
    "double",
    "float",
    "rocprim::half",
]
COMMON_VALUE_TYPES = [
    "rocprim::int128_t",
    "int64_t",
    "int",
    "short",
    "int8_t",
]


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

    strategy: str = "brute_force"
    """Tuning strategy to use (e.g. "brute_force", "dual_annealing"). Tuners whose
    search space is too large to enumerate override this in _get_default_args.
    """

    output_dir: str = "../output"
    """Directory path to store tuning results and metadata.
    """

    tune_baseline: str = "current"
    """Which existing configs to also benchmark: "current" (the config the library
    picks for this arch), "fallback" (the config of the generic target), "both" or
    "skip".
    """

    save_metadata: bool = False
    """Whether to save additional tuning metadata.
    """

    simulation_mode: bool = False
    seed: int = -1
    arch_name: Optional[str] = None

    param_overrides: Optional[str] = None
    """JSON object that replaces the values of tune parameters, used to narrow or widen
    the search (e.g. for staged tuning). Each value is a single value, a list of values,
    or "best" to pin the parameter to the best entry in that type's existing cache.
    """

    def update_with_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__dict__.keys() and not v is None:
                self.__dict__[k] = v


class BaseTuner(ABC):
    def __init__(self, args: TunerArgs):
        """Initialize the tuner with configuration parameters."""
        self.algo_full_name = args.algo_full_name
        algo_types = ["warp", "block", "device"]
        algo_full_name_parts = (
            args.algo_full_name.split("_") if args.algo_full_name is not None else []
        )
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
            self.device_properties: hip.hipDeviceProp_t = hip.hipDeviceProp_t()
            (_, self.device_properties) = hip.hipGetDeviceProperties(self.device_id)
            self.arch_name = self.device_properties.gcnArchName.split(":")[0]
            # Detect the GPU like the library does, so configs are generated for the
            # right GPU without --map-gfx.
            device_name = self.device_properties.name
            if isinstance(device_name, bytes):
                device_name = device_name.decode(errors="ignore")
            with open(
                f"{BASE_DIR}/../rocprim/include/rocprim/device/config_types.hpp"
            ) as f:
                gpu_names = confgen.parse.parse_gpu_names(f.read())
            self.gpu_name = confgen.parse.gpu_from_device_name(device_name, gpu_names)
        else:
            assert (
                args.arch_name
            ), "Simulation mode requires --arch-name to be specified"
            self.arch_name = args.arch_name
            self.gpu_name = "generic"
        self.tune_baseline = args.tune_baseline
        self.max_fevals = args.max_fevals
        self.save_metadata = args.save_metadata
        self.strategy = args.strategy
        self.seed = None if args.seed == -1 else args.seed
        self.param_overrides = (
            json.loads(args.param_overrides) if args.param_overrides else {}
        )

        if self.tune_baseline != "skip":
            (self.existing_config, self.fallback_target) = self._read_config_header(
                self.algo_full_name
            )
        else:
            self.existing_config = None
            self.fallback_target = None

    @classmethod
    @abstractmethod
    def _get_default_args(cls) -> TunerArgs:
        """Returns the default arguments. Used to generate default values for the CLI."""
        pass

    @classmethod
    def cli(cls):
        defaults: TunerArgs = cls._get_default_args()
        args = dict(
            Parser.get_parser(
                default_bytes=defaults.size,
                default_max_feval=defaults.max_fevals,
                default_output_dir=defaults.output_dir,
                default_strategy=defaults.strategy,
                default_tune_baseline=defaults.tune_baseline,
            )
            .parse_args()
            ._get_kwargs()
        )
        defaults.update_with_kwargs(**args)
        cls(defaults).tune_all()

    @abstractmethod
    def _get_tune_params(self, types: Dict[str, Any]) -> OrderedDict:
        """Returns tuning parameters and their possible values as an OrderedDict.
        Each parameter maps to a list of valid values to explore during tuning."""
        pass

    @abstractmethod
    def _get_restrictions(
        self, types: Dict[str, Any]
    ) -> Callable[[dict], bool] | List[str]:
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
        """Call tune_type for all type combinations"""
        pass

    def _get_problem_size(self, types: Dict[str, Any]):
        return self.bytes_size // sum(
            TYPE_CONFIGS[t].size for t in types.values() if t in TYPE_CONFIGS
        )

    def _get_metrics(self, types: Dict[str, Any]) -> Dict:
        """Default metrics for performance measurement."""
        # p["time"] is measured in ms/GiB
        return {"GiB/s": lambda p: (1000.0 / p["time"])}

    @staticmethod
    def _read_config_header(algo_full_name: str) -> tuple[Dict, Optional[frozenset]]:
        """Returns the parsed configs of a config header in rocPRIM, and the target its
        generic target forwards to (see confgen.parse.parse_fallback_target)."""
        with open(
            f"{BASE_DIR}/../rocprim/include/rocprim/device/detail/config/{algo_full_name}.hpp"
        ) as f:
            header = f.read()
        return (
            confgen.parse.parse_lines(header.splitlines()),
            confgen.parse.parse_fallback_target(header),
        )

    def _get_target_info(self) -> Dict[str, str]:
        """Returns the target being tuned, as in the TARGET comments of config headers."""
        return {
            "gen": confgen.generate.derive_gen(self.arch_name),
            "arch": self.arch_name,
            "gpu": self.gpu_name,
            "rep": "amdgcn",
        }

    def _get_baseline_targets(self) -> Dict[str, frozenset]:
        """Returns the targets selected by --tune-baseline, whose configs the library
        uses on the current arch: 'current' is the target most_common_config() picks
        (config_types.hpp), and 'fallback' is the target the generic target forwards to
        (usually gfx908). They are the same target when the header has no entry for
        this arch or generation."""
        targets = {}
        if self.tune_baseline in ("fallback", "both"):
            if self.fallback_target is not None:
                targets["fallback"] = self.fallback_target
            else:
                warnings.warn(
                    "The generic target returns the base config, which has no values to "
                    "benchmark. Skipping the fallback baseline."
                )
        if self.tune_baseline not in ("current", "both"):
            return targets

        current = confgen.parse.select_target(
            self.existing_config.keys(), self._get_target_info()
        )
        if current is None:
            current = self.fallback_target
        if current is not None:
            targets["current"] = current
        return targets

    def _run_baseline_configs(self, types: Dict[str, Any]) -> Dict[str, Dict]:
        """Benchmarks the configs the library uses on the current arch today (see
        _get_baseline_targets), so the config generator can keep them when the tuned
        winner isn't clearly faster. Returns the baselines that were found."""
        if (
            self.tune_baseline == "skip"
            or self.simulation_mode
            or self.existing_config is None
        ):
            return {}

        if self.arch_name is None:
            warnings.warn(f"Could not detect current architecture!")
            return {}

        required_keys = list(self._get_tune_params(types).keys())
        baselines = {}
        for name, target in self._get_baseline_targets().items():
            # Get first matching config
            config = next(
                (
                    c
                    for c in self.existing_config.get(target, [])
                    if all(
                        str(c.get(k, "empty_type")) == str(v) for k, v in types.items()
                    )
                ),
                None,
            )
            target_arch = dict(target)["arch"]
            if config is None:
                warnings.warn(
                    f"No {name} configuration ({target_arch}) found for types '{types}'"
                )
                continue

            # If any tunable parameter is not in the existing config, we can't run it.
            if any(k not in config for k in required_keys):
                warnings.warn(
                    f"The {name} configuration ({target_arch}) does not match the tunable parameters."
                )
                continue

            params = {k: config[k] for k in required_keys}
            baselines[name] = {"target": dict(target), "config": params}
            if any(b["config"] == params for n, b in baselines.items() if n != name):
                # Same config as the other baseline, it's already in the cache.
                continue

            # Force the tune parameters to the existing config, and run it with
            # bruteforce. The library already builds this config, so the restrictions
            # of the search grid don't apply to it.
            tune_kernel_args = self._get_base_tune_kernel_args(types)
            tune_kernel_args.update(
                {
                    "tune_params": {k: [v] for k, v in params.items()},
                    "strategy": "brute_force",
                    "strategy_options": {},
                    "restrictions": None,
                }
            )

            print(f"Benchmarking {name} configuration ({target_arch}): {params}")
            kernel_tuner.tune_kernel(**tune_kernel_args)

        return baselines

    def tune_type(
        self,
        types: Dict[str, Any],
    ) -> None:
        """Performs auto-tuning for a specific type combination."""
        print(f"\nTuning for {' '.join([str(x) for x in types.values()])}")
        print(f"Using size: {self.bytes_size} bytes")
        self._get_case_results_dir(types).mkdir(parents=True, exist_ok=True)
        strategy_print_message = (
            f"Using strategy: {self.strategy if self.strategy else 'brute_force'}"
        )
        strategy_print_message += (
            f", max fevals: {self.max_fevals}" if self.strategy != "brute_force" else ""
        )
        print(strategy_print_message)

        try:
            tune_kernel_args = self._get_base_tune_kernel_args(types)
            # Run main tuning
            results, _ = kernel_tuner.tune_kernel(**tune_kernel_args)
            # Run the configs the library currently uses, if enabled
            baselines = self._run_baseline_configs(types)

            cache_file_path = self._get_cache_file_path(types)
            self._save_output(cache_file_path, types, results, baselines)

        except Exception as e:
            print(f"Failed tuning for {' '.join([str(x) for x in types.values()])}")
            print(f"Error: {str(e)}")
            raise

    def _get_search_tune_params(self, types: Dict[str, Any]) -> OrderedDict:
        """Returns the tune parameters with param_overrides applied. Parameter names and
        order are kept, so existing cache files stay compatible."""
        params = self._get_tune_params(types)
        best = None
        for name, value in self.param_overrides.items():
            if name not in params:
                raise ValueError(
                    f"Unknown tune parameter '{name}' in param_overrides, expected one of {list(params.keys())}"
                )
            if value == "best":
                if best is None:
                    best = self._get_best_cached_config(types)
                if best is None:
                    warnings.warn(
                        f"No cached results for {types}, '{name}' is not pinned"
                    )
                    continue
                value = best[name]
            params[name] = value if isinstance(value, list) else [value]
        return params

    def _get_best_cached_config(self, types: Dict[str, Any]) -> Optional[Dict]:
        """Returns the fastest finite entry of this type's cache file, if any."""
        cache_file = self._get_cache_file_path(types)
        if not cache_file.exists():
            return None
        content = cache_file.read_text()
        try:
            cache = json.loads(content)
        except json.JSONDecodeError:
            # An interrupted Kernel Tuner run leaves the cache file unterminated.
            cache = json.loads(content.rstrip().rstrip(",") + "}}")
        timed = [
            e
            for e in cache.get("cache", {}).values()
            if isinstance(e.get("time"), (int, float)) and np.isfinite(e["time"])
        ]
        return min(timed, key=lambda e: e["time"]) if timed else None

    def generate_wrapper(
        self,
        config: OrderedDict,
        types: Dict[str, Any],
    ) -> str:
        """Generate wrapper code using Jinja2 template inheritance."""
        template_dir = pathlib.Path(f"{BASE_DIR}/tuner/templates")
        env = Environment(
            loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True
        )

        template_name = f"{self.algo_full_name}.cpp.jinja2"
        template = env.get_template(template_name)
        # The parameter values are defined using #define by kernel tuner.
        # We only need the right symbol name, so 'config.ipt' becomes 'ipt'.
        # In the templates, the full '{{ config.ipt }}'-syntax is preferred,
        # since this makes the templates more consistent with confgen templates.
        context = {
            "algo_type": self.algo_type,
            "algo_name": self.algo_name,
            "config": config,
            "case_results_dir": str(self._get_case_results_dir(types).resolve()),
            **types,
        }

        content = template.render(**context)

        return content

    def _get_base_tune_kernel_args(
        self,
        types: Dict[str, Any],
    ) -> Dict:
        """Returns base arguments for kernel_tuner.tune_kernel()."""
        np.random.seed(self.seed)

        wrapper_string = lambda config: self.generate_wrapper(
            config=config,
            types=types,
        )

        tune_kernel_args = {
            "defines": {},
            "kernel_name": f"{self.algo_full_name}_wrapper",
            "kernel_source": wrapper_string,
            "problem_size": self._get_problem_size(types),
            "arguments": [np.uint64(self.bytes_size)],
            "tune_params": self._get_search_tune_params(types),
            "strategy": self.strategy,
            "grid_div_x": self._get_grid_div_x(),
            "cache": str(self._get_cache_file_path(types)),
            "lang": "C",
            "compiler": "hipcc",
            "compiler_options": self._get_compiler_options(),
            "restrictions": self._get_restrictions(types),
            "verbose": False,
            "iterations": 1,
            "log": False,
            "device": self.device_id,
            "simulation_mode": self.simulation_mode,
            "objective_higher_is_better": False,
            "quiet": True,
        }

        if self.strategy != "brute_force":
            tune_kernel_args["strategy_options"] = {"max_fevals": self.max_fevals}

        return tune_kernel_args

    def _get_cache_file_name(self, types: Dict[str, Any]):
        """Return the name of the cache file based on algo name, arch name and types"""
        parts = [self.algo_full_name, self.arch_name]
        parts += [str(v).replace("rocprim::", "") for _, v in sorted(types.items())]
        return "_".join(parts) + "_cache.json"

    def _get_cache_file_path(self, types: Dict[str, Any]):
        """Return the path of the cache file"""
        return self.output_dir / self._get_cache_file_name(types)

    @staticmethod
    def _write_cache_annotation(cache_file: str | pathlib.Path, name: str, value: Any):
        """Adds a top-level entry for the config generator to a cache file. It's inserted
        as a single line (replacing the one from an earlier run), so the rest of kernel
        tuner's cache file stays as it wrote it."""
        with open(cache_file, "r") as f:
            content = f.read()
        content = re.sub(rf'^"{name}": .*,\n', "", content, flags=re.M)
        new_content = "{\n" + f'"{name}": {json.dumps(value)},'
        new_content += content.lstrip()[1:]
        with open(cache_file, "w") as f:
            f.write(new_content)

    def _find_noise_timeouts(
        self, cache_file: str | pathlib.Path, types: Dict[str, Any]
    ) -> List[str]:
        """Returns the cache keys of the configs whose per-case results (written by the
        wrapper, see _get_case_results_dir) show primbench's noise timeout."""
        with open(cache_file, "r") as f:
            cache = json.load(f)["cache"]
        keys = list(self._get_tune_params(types).keys())
        case_results_dir = self._get_case_results_dir(types)
        noise_timeouts = []
        for cache_key, entry in cache.items():
            # Same file names as the wrapper: <name>=<value>_..._size<bytes>.json
            prefix = "_".join(f"{k}={entry.get(k)}" for k in keys)
            path = case_results_dir / f"{prefix}_size{self.bytes_size}.json"
            if path.exists() and '"noise_timeout":true' in path.read_text():
                noise_timeouts.append(cache_key)
        return noise_timeouts

    def _get_case_results_dir(self, types: Dict[str, Any]) -> Path:
        """Return the directory for the per-case primbench JSON of each candidate.
        The cache only stores the combined score, so selection rules that need each
        case (worst case against develop) read these."""
        cache_name = self._get_cache_file_name(types).removesuffix("_cache.json")
        return self.output_dir / "cases" / cache_name

    def _save_output(
        self,
        cache_file: str | pathlib.Path,
        types: Dict[str, Any],
        results: List | object | Any | None = None,
        baselines: Optional[Dict[str, Dict]] = None,
    ):
        """Save tuning results and metadata to files."""
        if self.simulation_mode:
            assert results
            cache_file = Path(
                f"../simulated_output/{self.strategy}_fevals{self.max_fevals}"
            )
            cache_file.mkdir(parents=True, exist_ok=True)
            cache_file = cache_file / self._get_cache_file_name(types)
            store_output_file(str(cache_file), results, self._get_tune_params(types))

        with open(cache_file, "r") as f:
            cache_dict = json.load(f)
        if "arch_name" not in cache_dict:
            with open(cache_file, "r") as f:
                content = f.read()

            new_content = "{\n"
            new_content += f'"arch_name": "{self.arch_name}",\n'
            new_content += f'"algo_name": "{self.algo_full_name}",\n'
            for i, (k, v) in enumerate(types.items()):
                new_content += f'"{k}": "{v}",'
                if i < len(types) - 1:
                    new_content += "\n"
            new_content += content.lstrip()[1:]

            with open(cache_file, "w") as f:
                f.write(new_content)

        if not self.simulation_mode:
            # Record the GPU, so the config generator doesn't need --map-gfx.
            if self.gpu_name != "generic":
                self._write_cache_annotation(cache_file, "gpu", self.gpu_name)

            # Record which cache entries are the baselines, for the config generator.
            if baselines is not None:
                self._write_cache_annotation(cache_file, "baselines", baselines)

            # Record the configs that hit the noise timeout, so the config generator
            # drops them.
            noise_timeouts = self._find_noise_timeouts(cache_file, types)
            if noise_timeouts:
                print(
                    f"Dropping {len(noise_timeouts)} config(s) that hit the noise "
                    f"timeout: {noise_timeouts}"
                )
            self._write_cache_annotation(cache_file, "noise_timeouts", noise_timeouts)

        if self.save_metadata:
            store_metadata_file(str(cache_file).replace("cache", "metadata"))

    def _get_compiler_options(self) -> List[str]:
        """Returns a list with all compiler options to pass to Kernel Tuner"""
        monorepo_dir = (pathlib.Path(BASE_DIR) / "../../../../..").resolve()
        rocprim_dir = (pathlib.Path(BASE_DIR) / "..").resolve()
        return [
            # Match the Release build of the benchmarks (rocPRIM's default build type).
            "-O3",
            "-DNDEBUG",
            "-fPIC",
            "-std=c++17",
            f"-I{rocprim_dir / 'rocprim/include'}",
            f"-I{rocprim_dir / 'benchmark'}",
            f"-I{monorepo_dir / 'shared/primbench'}",
            "-Wno-#pragma-messages",
            f"--offload-arch={self.arch_name}",
            # Ensure we can find libamd_smi on systems/containers that do not expose /opt/rocm/lib
            # in LD_LIBRARY_PATH.
            "-Wl,-rpath=/opt/rocm/lib",
            # Required for primbench.
            "-lamd_smi",
        ]
