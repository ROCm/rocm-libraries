#!/usr/bin/env python3

# Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

"""
This Python script is intended for the creation of autotuned configurations
for the supported rocPRIM algorithms based on benchmark results. The script
does not update the configurations automatically, the user is responsible for 
installation and the correctness of the files
"""

import json
import re
import argparse
import os
import sys
import collections
import copy
import math
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Callable, Optional, Tuple
from jinja2 import Environment, PackageLoader, select_autoescape

TARGET_ARCHITECTURES = ['gfx803', 'gfx900', 'gfx906', 'gfx908', 'gfx90a', 'gfx1030', 'gfx1100', 'gfx1102']
# C++ typename used for optional types
EMPTY_TYPENAME = "empty_type"

env = Environment(
    loader=PackageLoader("create_optimization"),
    lstrip_blocks=True,
    trim_blocks=True

)

class NotSupportedError(Exception):
    """Exception raised for algorithms that are not supported
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


@dataclass
class SelectionType:
    """
    Data class describing a type used to select a configuration.
    """
    name: str
    # True if rocprim::empty_type is a valid type in the algorithm's configuration.
    is_optional: bool
    # True if only the size of the type is relevant. If false, selection also happens based on rocprim::is_floating.
    #   Only selecting on size will limit the number of tuning benchmarks required and configs generated.
    # For example, a sort by key will not perform any logic on the value type and only move the data.
    #   Hence it can select the value type on size only.
    select_on_size_only: bool

@dataclass
class SelectionConst:
    """
    Data class describing a constant (integral, enum) used to select a configuration.
    """
    name: str

@dataclass
class FallbackCase:
    """
    Data class describing a fallback case: a description for a set of types that
    use the same tuned configuration.
    """
    # Optional regex filter for the algorithm name.
    algo_regex: Optional[str]
    # C++ type for which to take the best tuned configuration.
    based_on_type: str
    # Inclusive maximum size of the type.
    sizeof_max: int
    # Exclusive minimum size of the type.
    sizeof_min: int
    # Whether the type has a floating point.
    is_floating_point: bool

def translate_settings_to_cpp_metaprogramming(
        fallback_configuration: List[Tuple[SelectionType, FallbackCase]], 
        const_configuration: List[SelectionConst]) -> str:
    """
    Translates a list of named fallback configuration entries to C++ metaprogramming idioms.
    """

    setting_list: List[str] = []
    for (config_selection_type, fallback_entry) in fallback_configuration:
        typename: str = config_selection_type.name

        if fallback_entry.based_on_type == EMPTY_TYPENAME:
            # If the entry is based on the empty type
            # (which is not present in the fallback file, but separately inserted)
            setting_list.append(f"(std::is_same<{typename}, rocprim::{EMPTY_TYPENAME}>::value)")
        else:
            # Only add a floating-point check on the first selection type. For the remaining selection types, a limited
            # number of fallbacks are generated, which are based on the integral types.
            if not config_selection_type.select_on_size_only:
                negation: str = "" if fallback_entry.is_floating_point else "!"
                output: str = negation + f"bool(rocprim::is_floating_point<{typename}>::value)"
                setting_list.append(output)

            setting_list.append(f"(sizeof({typename}) <= {fallback_entry.sizeof_max})")
            # sizeof(type) will always be greater than zero, can omit the check if min is zero or smaller
            if fallback_entry.sizeof_min > 0:
                setting_list.append(f"(sizeof({typename}) > {fallback_entry.sizeof_min})")

            # If the fallback entry has a sizeof of one (only true for int8_t) and the associated type is optional,
            # we need an additional check since the empty type also has a sizeof of one
            if fallback_entry.based_on_type == "int8_t" and config_selection_type.is_optional:
                setting_list.append(f"(!std::is_same<{typename}, rocprim::{EMPTY_TYPENAME}>::value)")
    for name, value in const_configuration.items():
        setting_list.append(f"({name} == {value})")
    return "std::enable_if_t<(" + " && ".join(setting_list) + ")>"

class BenchmarksOfArchitecture:
    """
    Stores the benchmark results for a specific architecture and algorithm.
    """

    def __init__(self, arch_name: str, config_selection_params, fallback_entries: List[FallbackCase], config_get_best, algorithm_name):
        self.config_selection_params = config_selection_params
        self.fallback_entries: List[FallbackCase] = fallback_entries
        self.arch_name: str = arch_name
        self.config_get_best: Callable[[Dict], Dict[str, str]] = config_get_best
        self.algorithm_name: str = algorithm_name
        # Dictionary storing the benchmarks
        # Key is an instantiation of the configuration selection types
        # Value is a list of all benchmark runs corresponding to that instantiation,
        # these benchmarks in this list vary in the actual configuration used to run the benchmark
        self.benchmarks = defaultdict(list)

    def __get_instance_key(self, instanced_types):
        """
        Takes in a list of instantiated types
        in the form of (name, value)-pairs for some 'name' in the selection types.

        Returns a hashable named tuple type where the names are based on the configuration selection types
        and the values on the instantiated types. If an instanced type is not present for a selection type
        a None object will be assigned as value.

        The created key can be used to access the specific benchmark results for a given combination of instantiated 
        selection types in the benchmarks member variable   
        """
        Instance = collections.namedtuple(typename='Instance', field_names=[cfg_param.name for cfg_param in self.config_selection_params])
        return Instance(**{field : instanced_types[field] if field in instanced_types.keys() else EMPTY_TYPENAME for field in Instance._fields})

    def add_measurement(self, benchmark_data: Dict[str, str]):
        """
        Adds a single benchmark run.
        """
        instance_key = self.__get_instance_key(benchmark_data)
        self.benchmarks[instance_key].append(benchmark_data)

    @property
    def name(self) -> str:
        return self.arch_name

    def __get_best_benchmark(self, instance_key) -> Dict[str, str]:
        """
        Returns the best performing benchmark from a list of benchmarks.
        For now, use the items per second as metric. in case the benchmark with the 
        given configuration is not present None is returned
        """
        if instance_key in self.benchmarks.keys():
            return self.config_get_best(self.benchmarks[instance_key])
        else:
            return None

    @property
    def best_config_by_selection_types(self):
        """
        Returns a dictionary containing each instantion of the selection configuration as a key
        and the single best performing benchmark run as a value.
        """
        output = {}
        for instance, benchmarks in self.benchmarks.items():
            output[instance] = self.__get_best_benchmark(instance)
        return output

    def __add_fallback_to_output(
            self, 
            output, 
            fallback_configuration: List[Tuple[SelectionType, FallbackCase]], 
            const_configurations: List[SelectionConst]):
        """
        searches for specific fallback configuration in benchmark results and adds it to the output
        :param output: list of strings to append to
        :param fallback_configuration: the configuration to look for
        :param const_configurations: list of constant selection parameters
        :return:
        """
        search_key: Dict[str, str] = {
            config_selection_type.name : fallback_entry.based_on_type for (config_selection_type, fallback_entry) in fallback_configuration}
        for const_configuration in (const_configurations or [{}]):
            for k, v in const_configuration.items():
                search_key[k] = v
            best_benchmark_result: Dict[str, str] = self.__get_best_benchmark(self.__get_instance_key(search_key))
            print_config: str = ', '.join([f'{k} = {v}' for k, v in search_key.items()])
            if best_benchmark_result is None:
                print(
                    f'WARNING {self.name}: No {self.algorithm_name} measurement found for creating fallback configuration '
                    f'entry for \"{print_config}\"')
            else:
                output.append((print_config,
                               translate_settings_to_cpp_metaprogramming(fallback_configuration,
                                                                         const_configuration),
                               best_benchmark_result))

    @property
    def fallback_types(self):
        """
        Provides a fallback triplet of (string describing the type used for generating the fallback,
        C++ enable if statement, benchmark containing the selected parameters for the algorithm).

        This function only supports algorithms with at most two types.
        """

        output = []

        # Collect all combination of constant selection parameters from avaliable benchmark results
        const_configurations = []
        config_selection_const_names = [cfg_param.name for cfg_param in self.config_selection_params if isinstance(cfg_param, SelectionConst)]
        if config_selection_const_names:
            Consts = collections.namedtuple(typename='Consts', field_names=config_selection_const_names)
            all_consts = [Consts(**{k : v for k, v in instance_key._asdict().items() if k in config_selection_const_names}) for instance_key in self.benchmarks.keys()]
            const_configurations = [c._asdict() for c in sorted(set(all_consts))]

        config_selection_types = [cfg_param for cfg_param in self.config_selection_params if isinstance(cfg_param, SelectionType)]

        # If there are more than two selection types, do not generate fallback cases
        # Otherwise, too many benchmarks would be needed support for the full product of fallback entries
        if len(config_selection_types) > 2:
            print(f"INFO: not generating fallbacks for {self.algorithm_name} as it has too many types.")
            return output

        # If the type requires selection on size only, the floating-point fallback entries as skipped.
        #   The decision to skip floating-point entries instead of integral ones is arbitrary.
        def skip_entry(config_selection_type: SelectionType, fallback_entry: FallbackCase) -> bool:
            return config_selection_type.select_on_size_only and fallback_entry.is_floating_point

        # If there is exactly one type, generate full fallbacks. Note that this type will never be optional
        if len(config_selection_types) == 1:
            config_selection_type = config_selection_types[0]

            if config_selection_type.is_optional:
                raise (ValueError(f'Algorithm "{self.algorithm_name}" has a single type that is optional'))

            for entry in self.fallback_entries:
                if skip_entry(config_selection_type, entry):
                    continue

                # Let the single selection type be based on the current fallback entry
                fallback_configuration = [(config_selection_type, entry)]

                # Find the closest measurement and create the config line
                self.__add_fallback_to_output(output, fallback_configuration, const_configurations)

        # If there are two types, generate full fallbacks for the first type but limited fallbacks for the second type
        if len(config_selection_types) == 2:
            # Assume that the first type is not optional
            if config_selection_types[0].is_optional:
                raise (ValueError(f'Algorithm "{self.algorithm_name}" two types but the first is optional'))

            # Enforce that only one type selects both on size and number representation (integral or floating)
            #   to limit the number of generated configs.
            if not config_selection_types[0].select_on_size_only and not config_selection_types[1].select_on_size_only:
                raise (ValueError(f'Algorithm "{self.algorithm_name}" two types but neither only selects on size'))

            # Only based_on_type is relevant
            empty_fallback = FallbackCase(None, EMPTY_TYPENAME, 0, 0, False)

            # If a type is optional, also generate the fallbacks where the type is empty.
            fallback_entries_0: List[FallbackCase] = self.fallback_entries
            if config_selection_types[0].is_optional:
                fallback_entries_0.append(empty_fallback)
            fallback_entries_1: List[FallbackCase] = self.fallback_entries
            if config_selection_types[1].is_optional:
                fallback_entries_1.append(empty_fallback)

            fallback_configuration: List[Tuple[SelectionType, FallbackCase]] = []

            for fallback_0 in fallback_entries_0:
                if skip_entry(config_selection_types[0], fallback_0):
                    continue

                fallback_configuration.append((config_selection_types[0], fallback_0))

                for fallback_1 in fallback_entries_1:
                    if skip_entry(config_selection_types[1], fallback_1):
                        continue

                    fallback_configuration.append((config_selection_types[1], fallback_1))
                    self.__add_fallback_to_output(output, fallback_configuration, const_configurations)
                    fallback_configuration.pop()

                fallback_configuration.pop()

        return output

# Default formula to pick the best configuration, only look at items_per_second.
def default_config_get_best(input: Dict) -> Dict[str, str]:
    return max(input, key=lambda x: x.get('items_per_second', 0.0))

# If we can double the sorted items_per_block and items_per_second does not degrade more than ~10%, consider it superior.
def block_sort_config_get_best(input: Dict) -> Dict[str, str]:
    return max(input, key=lambda x: x.get('items_per_second', 0.0)*((float(x['cfg']['bs'])*float(x['cfg']['ipt']))**(1/4)))

# Best configuration is a combination between best oddeven and best mergepath impl.
# We use oddeven only for small input sizes (< ~200K), so it is a hardcoded value which is the best for almost all cases.
# You can find this value in the tuning template
def merge_sort_block_merge_config_get_best(input: Dict) -> Dict[str, str]:
    input_mergepath = list(filter(lambda x: (int(x.get('cfg').get('oddeven_size_limit')) == 0), input))
    # Since merge_sort_block_merge is used after radix_sort_block_sort<256, 4>, and
    # mergepath_block_size * mergepath_items_per_thread >= 256*4 should hold (TODO: this will be solved in the near future):
    input_mergepath = list(filter(lambda x: (int(x.get('cfg').get('mergepath_bs'))*int(x.get('cfg').get('mergepath_ipt')) <= 1024), input_mergepath))

    best_mergepath = max(input_mergepath, key=lambda x: x.get('items_per_second', 0.0))
    return best_mergepath

class Algorithm:
    """
    Aggregates the data for an algorithm, including the generation of the configuration file.
    """

    def __init__(self, fallback_entries: List[FallbackCase], config_get_best = default_config_get_best):
        self.architectures: Dict[str, BenchmarksOfArchitecture] = {}
        self.fallback_entries: List[FallbackCase] = fallback_entries
        self.config_get_best = config_get_best

    def add_measurement(self, single_benchmark_data: Dict[str, str], architecture: str):
        """
        Adds a single benchmark execution for a given architecture
        """
        if architecture not in self.architectures:
            self.architectures[architecture] = BenchmarksOfArchitecture(architecture, self.config_selection_params,
                                                                        self.fallback_entries, self.config_get_best,
                                                                        self.algorithm_name)
        self.architectures[architecture].add_measurement(single_benchmark_data)

    def create_config_file_content(self) -> str:
        """
        Generate the content of the configuration file, including license
        and header guards, based on general template file.
        """
        if 'target_arch::gfx908' in self.architectures:
            self.architectures['target_arch::unknown'] = copy.deepcopy(self.architectures['target_arch::gfx908'])
            self.architectures['target_arch::unknown'].arch_name = 'target_arch::unknown'
            if 'target_arch::gfx90a' not in self.architectures:
                self.architectures['target_arch::gfx90a'] = copy.deepcopy(self.architectures['target_arch::gfx908'])
                self.architectures['target_arch::gfx90a'].arch_name = 'target_arch::gfx90a'

        algorithm_template = env.get_template(self.cpp_configuration_template_name)
        rendered_template = algorithm_template.render(all_architectures=self.architectures.values())

        return rendered_template


"""
Each algorithm uses ninja templates to generate C++ configuration specification.
The generated configuration file contains configs for two cases:
- No architecture or instantiation of configuration selection types is provided (general base case).
- The architecture and configuration selection types are provided, the configuration is based on the
  benchmark results of similar types (fallback case).

config_selection_params is a list of parameters (types, constants) that are used to select a configuration.
The fallback file will be used to generate the fallback cases. If there are two types, one of these will only select 
on the size of the type (defined by the config selection parameters), limiting the number of fallback cases. 

If the type is optional, additional fallback configurations will be generated that match the case when the optional 
selection type passed by the user is rocprim::empty_type. The config_selection_params should specify at least 
one non-optional type. The optional type should not be the first type.

The 'name' fields should correspond to a named capturing group in the regex field of the benchmark,
these names should be valid C++ identifiers. The matched values in the name field of
the benchmark should also be valid C++ typenames. This is required as these names will be in the 
generated C++ code.
"""


class AlgorithmDeviceHistogram(Algorithm):
    algorithm_name = "device_histogram"
    cpp_configuration_template_name = "histogram_config_template"
    config_selection_params = [
        SelectionType(name="value_type", is_optional=False, select_on_size_only=False),
        SelectionConst(name="channels"),
        SelectionConst(name="active_channels")]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)


class AlgorithmDeviceMergeSortBlockSort(Algorithm):
    algorithm_name = "device_merge_sort_block_sort"
    cpp_configuration_template_name = "mergesort_block_sort_config_template"
    config_selection_params = [
        SelectionType(name="key_type", is_optional=False, select_on_size_only=False),
        SelectionType(name="value_type", is_optional=True, select_on_size_only=True)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries, block_sort_config_get_best)


class AlgorithmDeviceMergeSortBlockMerge(Algorithm):
    algorithm_name = "device_merge_sort_block_merge"
    cpp_configuration_template_name = "mergesort_block_merge_config_template"
    config_selection_params = [
        SelectionType(name="key_type", is_optional=False, select_on_size_only=False),
        SelectionType(name="value_type", is_optional=True, select_on_size_only=True)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(
            self, fallback_entries, merge_sort_block_merge_config_get_best)


class AlgorithmDeviceRadixSortBlockSort(Algorithm):
    algorithm_name = "device_radix_sort_block_sort"
    cpp_configuration_template_name = "radixsort_block_sort_config_template"
    config_selection_params = [
        SelectionType(name="key_type", is_optional=False, select_on_size_only=False),
        SelectionType(name="value_type", is_optional=True, select_on_size_only=True)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries, block_sort_config_get_best)


class AlgorithmDeviceRadixSortOnesweep(Algorithm):
    algorithm_name = "device_radix_sort_onesweep"
    cpp_configuration_template_name = "radixsort_onesweep_config_template"
    config_selection_params = [
        SelectionType(name="key_type", is_optional=False, select_on_size_only=False),
        SelectionType(name="value_type", is_optional=True, select_on_size_only=True)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDeviceReduce(Algorithm):
    algorithm_name = "device_reduce"
    config_selection_params = [
        SelectionType(name="key_type", is_optional=False, select_on_size_only=False)]
    cpp_configuration_template_name = "reduce_config_template"
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDeviceScan(Algorithm):
    algorithm_name = "device_scan"
    cpp_configuration_template_name = "scan_config_template"
    config_selection_params = [
        SelectionType(name="value_type", is_optional=False, select_on_size_only=False)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

# TODO select_on_size_only may need to get re-evaluated, it is likely that swapping the value
#   of select_on_size_only for key and value gives better results
class AlgorithmDeviceScanByKey(Algorithm):
    algorithm_name = "device_scan_by_key"
    cpp_configuration_template_name = "scanbykey_config_template"
    config_selection_params = [
        SelectionType(name="key_type", is_optional=False, select_on_size_only=False),
        SelectionType(name="value_type", is_optional=False, select_on_size_only=True)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDeviceBinarySearch(Algorithm):
    algorithm_name = "device_binary_search"
    cpp_configuration_template_name = "binary_search_config_template"
    config_selection_params = [
        SelectionType(name="value_type", is_optional=False, select_on_size_only=False),
        SelectionType(name="output_type", is_optional=False, select_on_size_only=True)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDeviceUpperBound(Algorithm):
    algorithm_name = "device_upper_bound"
    cpp_configuration_template_name = "upper_bound_config_template"
    config_selection_params = [
        SelectionType(name="value_type", is_optional=False, select_on_size_only=False),
        SelectionType(name="output_type", is_optional=False, select_on_size_only=True)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDeviceLowerBound(Algorithm):
    algorithm_name = "device_lower_bound"
    cpp_configuration_template_name = "lower_bound_config_template"
    config_selection_params = [
        SelectionType(name="value_type", is_optional=False, select_on_size_only=False),
        SelectionType(name="output_type", is_optional=False, select_on_size_only=True)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDeviceAdjacentDifference(Algorithm):
    algorithm_name = "device_adjacent_difference"
    cpp_configuration_template_name = "adjacent_difference_config_template"
    config_selection_params = [
        SelectionType(name="value_type", is_optional=False, select_on_size_only=False)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDeviceAdjacentDifferenceInplace(Algorithm):
    algorithm_name = "device_adjacent_difference_inplace"
    cpp_configuration_template_name = "adjacent_difference_inplace_config_template"
    config_selection_params = [
        SelectionType(name="value_type", is_optional=False, select_on_size_only=False)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDeviceSegmentedRadixSort(Algorithm):
    algorithm_name = "device_segmented_radix_sort"
    cpp_configuration_template_name = "segmented_radix_sort_config_template"
    config_selection_params = [
        SelectionType(name="key_type", is_optional=False, select_on_size_only=False),
        SelectionType(name="value_type", is_optional=True, select_on_size_only=True)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDeviceTransform(Algorithm):
    algorithm_name = "device_transform"
    cpp_configuration_template_name = "transform_config_template"
    config_selection_params = [
        SelectionType(name="value_type", is_optional=False, select_on_size_only=False)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDevicePartitionTwoWayPredicate(Algorithm):
    algorithm_name = "device_partition_two_way_predicate"
    cpp_configuration_template_name = "partition_two_way_predicate_config_template"
    config_selection_params = [
        SelectionType(name="data_type", is_optional=False, select_on_size_only=False)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDevicePartitionTwoWayFlag(Algorithm):
    algorithm_name = "device_partition_two_way_flag"
    cpp_configuration_template_name = "partition_two_way_flag_config_template"
    config_selection_params = [
        SelectionType(name="data_type", is_optional=False, select_on_size_only=False)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDevicePartitionFlag(Algorithm):
    algorithm_name = "device_partition_flag"
    cpp_configuration_template_name = "partition_flag_config_template"
    config_selection_params = [
        SelectionType(name="data_type", is_optional=False, select_on_size_only=False)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDevicePartitionPredicate(Algorithm):
    algorithm_name = "device_partition_predicate"
    cpp_configuration_template_name = "partition_predicate_config_template"
    config_selection_params = [
        SelectionType(name="data_type", is_optional=False, select_on_size_only=False)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDevicePartitionThreeWay(Algorithm):
    algorithm_name = "device_partition_three_way"
    cpp_configuration_template_name = "partition_three_way_config_template"
    config_selection_params = [
        SelectionType(name="data_type", is_optional=False, select_on_size_only=False)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDeviceSelectFlag(Algorithm):
    algorithm_name = "device_select_flag"
    cpp_configuration_template_name = "select_flag_config_template"
    config_selection_params = [
        SelectionType(name="data_type", is_optional=False, select_on_size_only=False)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDeviceSelectPredicate(Algorithm):
    algorithm_name = "device_select_predicate"
    cpp_configuration_template_name = "select_predicate_config_template"
    config_selection_params = [
        SelectionType(name="data_type", is_optional=False, select_on_size_only=False)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDeviceSelectUnique(Algorithm):
    algorithm_name = "device_select_unique"
    cpp_configuration_template_name = "select_unique_config_template"
    config_selection_params = [
        SelectionType(name="data_type", is_optional=False, select_on_size_only=False)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDeviceSelectUniqueByKey(Algorithm):
    algorithm_name = "device_select_unique_by_key"
    cpp_configuration_template_name = "select_unique_by_key_config_template"
    config_selection_params = [
        SelectionType(name="key_type", is_optional=False, select_on_size_only=False),
        SelectionType(name="value_type", is_optional=False, select_on_size_only=True)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDeviceReduceByKey(Algorithm):
    algorithm_name = "device_reduce_by_key"
    cpp_configuration_template_name = "reduce_by_key_config_template"
    config_selection_params = [
        SelectionType(name="key_type", is_optional=False, select_on_size_only=True),
        SelectionType(name="value_type", is_optional=False, select_on_size_only=False)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

class AlgorithmDeviceFindFirstOf(Algorithm):
    algorithm_name = "device_find_first_of"
    cpp_configuration_template_name = "find_first_of_config_template"
    config_selection_params = [
        SelectionType(name="value_type", is_optional=False, select_on_size_only=True)]
    def __init__(self, fallback_entries):
        Algorithm.__init__(self, fallback_entries)

def filt_algo_regex(e: FallbackCase, algorithm_name):
    if e.algo_regex:
        return re.match(e.algo_regex, algorithm_name) is not None
    return True

def create_algorithm(algorithm_name: str, fallback_entries: List[FallbackCase]):
    fallback_entries = list(filter(lambda e: filt_algo_regex(e, algorithm_name), fallback_entries))
    if algorithm_name == 'device_histogram':
        return AlgorithmDeviceHistogram(fallback_entries)
    elif algorithm_name == 'device_merge_sort_block_sort':
        return AlgorithmDeviceMergeSortBlockSort(fallback_entries)
    elif algorithm_name == 'device_merge_sort_block_merge':
        return AlgorithmDeviceMergeSortBlockMerge(fallback_entries)
    elif algorithm_name == 'device_radix_sort_block_sort':
        return AlgorithmDeviceRadixSortBlockSort(fallback_entries)
    elif algorithm_name == 'device_radix_sort_onesweep':
        return AlgorithmDeviceRadixSortOnesweep(fallback_entries)
    elif algorithm_name == 'device_reduce':
        return AlgorithmDeviceReduce(fallback_entries)
    elif algorithm_name == 'device_scan':
        return AlgorithmDeviceScan(fallback_entries)
    elif algorithm_name == 'device_scan_by_key':
        return AlgorithmDeviceScanByKey(fallback_entries)
    elif algorithm_name == 'device_binary_search':
        return AlgorithmDeviceBinarySearch(fallback_entries)
    elif algorithm_name == 'device_upper_bound':
        return AlgorithmDeviceUpperBound(fallback_entries)
    elif algorithm_name == 'device_lower_bound':
        return AlgorithmDeviceLowerBound(fallback_entries)
    elif algorithm_name == 'device_adjacent_difference':
        return AlgorithmDeviceAdjacentDifference(fallback_entries)
    elif algorithm_name == 'device_adjacent_difference_inplace':
        return AlgorithmDeviceAdjacentDifferenceInplace(fallback_entries)
    elif algorithm_name == 'device_segmented_radix_sort':
        return AlgorithmDeviceSegmentedRadixSort(fallback_entries)
    elif algorithm_name == 'device_transform':
        return AlgorithmDeviceTransform(fallback_entries)
    elif algorithm_name == 'device_partition_two_way_predicate':
        return AlgorithmDevicePartitionTwoWayPredicate(fallback_entries)
    elif algorithm_name == 'device_partition_two_way_flag':
        return AlgorithmDevicePartitionTwoWayFlag(fallback_entries)
    elif algorithm_name == 'device_partition_flag':
        return AlgorithmDevicePartitionFlag(fallback_entries)
    elif algorithm_name == 'device_partition_predicate':
        return AlgorithmDevicePartitionPredicate(fallback_entries)
    elif algorithm_name == 'device_partition_three_way':
        return AlgorithmDevicePartitionThreeWay(fallback_entries)
    elif algorithm_name == 'device_select_flag':
        return AlgorithmDeviceSelectFlag(fallback_entries)
    elif algorithm_name == 'device_select_predicate':
        return AlgorithmDeviceSelectPredicate(fallback_entries)
    elif algorithm_name == 'device_select_unique':
        return AlgorithmDeviceSelectUnique(fallback_entries)
    elif algorithm_name == 'device_select_unique_by_key':
        return AlgorithmDeviceSelectUniqueByKey(fallback_entries)
    elif algorithm_name == 'device_reduce_by_key':
        return AlgorithmDeviceReduceByKey(fallback_entries)
    elif algorithm_name == 'device_find_first_of':
        return AlgorithmDeviceFindFirstOf(fallback_entries)
    else:
        raise(NotSupportedError(f'Algorithm "{algorithm_name}" is not supported (yet)'))

class BenchmarkDataManager:
    """
    Aggregates the data from multiple benchmark files containing single benchmark runs
    with different configurations. One file may contain data for multiple algorithms
    """

    def __init__(self, fallback_config_file: str):
        self.algorithms: Dict[str, Algorithm] = {}
        abs_path_to_script_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.abs_path_to_template: str = os.path.join(abs_path_to_script_dir, 'config_template')
        self.fallback_config_file: str = fallback_config_file
        self.fallback_entries: List[FallbackCase] = self.__load_fallback_entries()

    def __load_fallback_entries(self) -> List[FallbackCase]:
        """
        Reads in fallback json file to list of dictionaries
        """

        raw_fallback_entries = json.load(self.fallback_config_file)['fallback_cases']
        fallback_entries: List[FallbackCase] = []
        for fallback_settings_entry in raw_fallback_entries:
            if "based_on_type" not in fallback_settings_entry \
                or "sizeof_max_inclusive" not in fallback_settings_entry \
                or "sizeof_min_exclusive" not in fallback_settings_entry \
                or "is_floating_point" not in fallback_settings_entry:
                raise (ValueError(f'Fallback entry "{raw_fallback_entries}" does not have all required fields'))

            fallback_entries.append(FallbackCase(
                    None if "algo_regex" not in fallback_settings_entry else fallback_settings_entry["algo_regex"],
                    fallback_settings_entry["based_on_type"],
                    fallback_settings_entry["sizeof_max_inclusive"],
                    fallback_settings_entry["sizeof_min_exclusive"],
                    fallback_settings_entry["is_floating_point"]))

        return fallback_entries

    def __get_target_architecture_from_context(self, benchmark_run):
        """
        Uses the benchmark run context embedded into the benchmark output json to retrieve the targeted architecture
        """

        name_from_context = benchmark_run['context']['hdp_gcn_arch_name'].split(":")[0]
        if name_from_context in TARGET_ARCHITECTURES:
            return f'target_arch::{name_from_context}'
        else:
            raise RuntimeError(f"ERROR: unknown hdp_gcn_arch_name: {name_from_context}")

    def __get_single_benchmark(self, single_benchmark):
        """
        Enriches the benchmark the data in single_benchmark with the information stored in the actual name of the particular benchmark run

        This information contains the different settings the benchmark has been executed with which will be used to create the customized 
        configuration case.
        """
        # google benchmark may postfix the JSON name: extract the '{...}' substring
        tokenized_name = re.match(r"{.*}", single_benchmark['name']).group(0)
        tokenized_name = json.loads(tokenized_name)
        if not tokenized_name:
            raise RuntimeError(f"ERROR: cannot parse JSON from: \"{single_benchmark['name']}\"")
        return dict(single_benchmark, **tokenized_name)

    def __add_benchmark_to_algorithm(self, single_benchmark, arch):
        """
        Adds a single_benchmark execution of a given Algorithm on a given architecture, to the Algorithm object

        In case the Algorithm object does not exist, a new object will be created.
        """
        algorithm_name: str = single_benchmark['lvl'] + "_" + single_benchmark['algo']
        if 'subalgo' in single_benchmark:
            algorithm_name += "_" + single_benchmark['subalgo']
        if algorithm_name not in self.algorithms:
            self.algorithms[algorithm_name] = create_algorithm(algorithm_name, self.fallback_entries)
        self.algorithms[algorithm_name].add_measurement(single_benchmark, arch)

    def add_run(self, benchmark_run_file_path: str):
        """
        Adds a single file containing the results of benchmarks executed on a single architecture.
        The benchmarks within the file may belong to different algorithms.
        """

        with open(benchmark_run_file_path, "r") as file_handle:
            benchmark_run_data = json.load(file_handle)

        try:
            print(f'INFO: Processing "{benchmark_run_file_path}"')
            arch = self.__get_target_architecture_from_context(benchmark_run_data)
            for raw_single_benchmark in benchmark_run_data['benchmarks']:
                single_benchmark = self.__get_single_benchmark(raw_single_benchmark)
                self.__add_benchmark_to_algorithm(single_benchmark, arch)
            print(f'INFO: Successfully processed file "{benchmark_run_file_path}"')
        except NotSupportedError as error:
            print(f'WARNING: Could not process file "{benchmark_run_file_path}": {error}', file=sys.stderr, flush=True)

    def write_configs_to_files(self, base_dir: str):
        """
        For each algorithm, creates a file containing configurations and places these in base_dir.
        """
        if len(self.algorithms) == 0:
            raise(KeyError('No suitable files to process'))

        for algo_name, algo in self.algorithms.items():
            config: str = algo.create_config_file_content()
            path_str: str = os.path.join(base_dir, f"{algo_name}.hpp")
            with open(path_str, "w") as outfile:
                outfile.write(config)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Tool for generating optimized launch parameters for rocPRIM based on benchmark results")
    parser.add_argument('-b','--benchmark_files', nargs='+', help="Benchmark files listed in the form <path_to_benchmark>.json")
    parser.add_argument("-p", "--out_basedir", type=str, help="Base dir for the output files, for each algorithm a new file will be created in this directory", required=True)
    parser.add_argument("-c", "--fallback_configuration", type=argparse.FileType('r'), default=os.path.join(current_dir, "fallback_config.json"), help="Configuration for fallbacks for not tested datatypes")
    args = parser.parse_args()

    benchmark_manager = BenchmarkDataManager(args.fallback_configuration)

    for benchmark_run in args.benchmark_files:
        benchmark_manager.add_run(benchmark_run)

    benchmark_manager.write_configs_to_files(args.out_basedir)

if __name__ == '__main__':
    main()
