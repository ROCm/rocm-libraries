#!/usr/bin/env python3

# Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
import collections
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List

@dataclass
class SelectionType:
    """
    Data class describing a type used to select a configuration.
    """
    name: str
    is_optional: bool

# C++ typename used for optional types
EMPTY_TYPENAME = "rocprim::empty_type"

def tokenize_benchmark_name(input_name: str, name_regex: str) -> Dict[str, str]:
    match = re.search(name_regex, input_name)
    if match:
        data_dict = match.groupdict()
        return data_dict
    else:
        return None

def translate_settings_to_cpp_metaprogramming(fallback_configuration) -> str:
    """
    Translates a list of named fallback configuration entries to 
    C++ metaprogramming idioms.
    """
    
    setting_list: List[str] = []
    for typename, entry in fallback_configuration.items():
        if entry["based_on"]["datatype"] == EMPTY_TYPENAME:
            # If the entry is based on the empty type
            # (which is not present in the fallback file, but separately inserted)
            setting_list.append(f"(std::is_same<{typename}, {EMPTY_TYPENAME}>::value)")
        else:
            if "floating_point" in entry.keys():
                negation: str = "" if entry['floating_point'] else "!"
                output: str = negation + f"bool(rocprim::is_floating_point<{typename}>::value)"
                setting_list.append(output)

            for config_setting, value in entry['sizeof'].items():
                if config_setting == "min_exclusive":
                    setting_list.append(f"(sizeof({typename}) > {value})")
                elif config_setting == "max_inclusive":
                    setting_list.append(f"(sizeof({typename}) <= {value})")
                else:
                    print(f"WARNING: {config_setting} is not known")
    
    return "std::enable_if_t<(" + " && ".join(setting_list) + ")>"

class BenchmarksOfArchitecture:
    """
    Stores the benchmark results for a specific architecture and algorithm.
    """

    def __init__(self, arch_name: str):
        self.arch_name: str = arch_name
        # Dictionary storing the benchmarks
        # Key is an instantiation of the configuration selection types
        # Value is a list of all benchmark runs corresponding to that instantiation,
        # these benchmarks in this list vary in the actual configuration used to run the benchmark
        self.benchmarks = defaultdict(list)

    def __get_instance_key(self, config_selection_types, instanced_types):
        """
        Takes in a list of selection types and a dict of instantiated types
        in the form of (name, value)-pairs for each 'name' in the selection types.

        Returns a named tuple type where the names are based on the configuration selection types
        and the values on the instantiated types.
        """
    
        Instance = collections.namedtuple(typename='Instance', field_names=[cfg_type.name for cfg_type in config_selection_types])
        return Instance(**{field : instanced_types[field] for field in Instance._fields})

    def add_measurement(self, benchmark_data: Dict[str, str], config_selection_types):
        """
        Adds a single benchmark run with a specific instance and configuration.
        """
        # Supply the empty typename if no type was read in due to a type being optional
        instanced_types: Dict[str, str] = {cfg_type.name: benchmark_data[cfg_type.name] or EMPTY_TYPENAME for cfg_type in config_selection_types}
        # Get a hashable key based on the instantiated selection types
        instance_key = self.__get_instance_key(config_selection_types, instanced_types)
        self.benchmarks[instance_key].append(benchmark_data)

    @property
    def name(self) -> str:
        return self.arch_name

    def __find_best_benchmark(self, benchmarks: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Returns the best performing benchmark from a list of benchmarks.
        For now, use the items per second as metric.
        """

        return max(benchmarks, key=lambda x: x['items_per_second'])

    def base_config_case(self, config_selection_types, algo_name) -> Dict[str, str]:
        """
        Finds a suitable configuration if only the architecture is specified.
        """
        
        # For now, return the best performing configuration that has:
        # int in non-optional types, the empty type in optional types
        instanced_types: Dict[str, str] = {cfg_type.name: EMPTY_TYPENAME if cfg_type.is_optional else 'int' for cfg_type in config_selection_types}
        instance_key = self.__get_instance_key(config_selection_types, instanced_types)
        if instance_key not in self.benchmarks:
            print(f"WARNING {algo_name}: requested base config case key \"{instance_key}\" not in list of benchmarks, falling back to first key")
            return self.__find_best_benchmark(next(iter(self.benchmarks.values())))

        return self.__find_best_benchmark(self.benchmarks[instance_key])

    @property
    def best_config_by_selection_types(self):
        """
        Returns for each type selection instance the single best performing benchmark run.
        """
        output = {}
        for instance, benchmarks in self.benchmarks.items():
            output[instance] = self.__find_best_benchmark(benchmarks)
        return output

class Algorithm:
    """
    Aggregates the data for a algorithm, including the generation of
    the configuration file.
    """

    def __init__(self, algorithm_name: str, fallback_entries):
        self.name: str = algorithm_name
        self.architectures: Dict(str, BenchmarksOfArchitecture) = {}
        self.fallback_entries = fallback_entries
    
    def add_measurement(self, single_benchmark_data: Dict[str, str]):
        """
        Adds a single benchmark run with a specific configuration and selected types.
        """
        architecture_name: str = single_benchmark_data['arch']
        if architecture_name not in self.architectures:
            self.architectures[architecture_name] = BenchmarksOfArchitecture(architecture_name)
        self.architectures[architecture_name].add_measurement(
            single_benchmark_data, self.config_selection_types)

    def create_config_file_content(self, abs_path_to_template: str) -> str:
        """
        Generate the content of the configuration file, including license
        and header guards, based on general template file.
        """
 
        configuration_lines: List[str] = self.__get_configurations()
        configuration: str = '\n'.join(configuration_lines)

        with open(abs_path_to_template) as template_file:
            template_file_content: str = template_file.read()
            generated_config_file_content: str = template_file_content.format(guard=self.name.upper(), config_body=configuration)

        return generated_config_file_content

    def __create_fallback_cases(self, fallback_entries, benchmarks_of_architecture) -> List[str]:
        """
        Creates fallback cases based on entries in a fallback file.
        """

        out_lines = []

        # If there are multiple non-optional selection types, do not generate fallback cases
        # Otherwise, too many benchmarks would be needed support for the full product of fallback entries
        non_optional_selection_types: List[str] = \
                [cfg_type.name for cfg_type in self.config_selection_types if not cfg_type.is_optional]
        if len(non_optional_selection_types) != 1:
            return out_lines
        non_optional_selection_type: str = non_optional_selection_types[0]

        # Fill all optional types with a special case, indicating these are based on the empty type
        # Insert None for the singular non-optional type to maintain order
        fallback_configuration = {cfg_type.name : {'based_on' : {'datatype' : EMPTY_TYPENAME}} if cfg_type.is_optional else None for cfg_type in self.config_selection_types}

        for entry in fallback_entries:
            # Let the single non-optional type be based on the current fallback entry
            fallback_configuration[non_optional_selection_type] = entry

            # Find the closest measurement and create the config line
            measurement: Dict[str, str] = self.__get_fallback_match(benchmarks_of_architecture, fallback_configuration)
            print_config: str = ', '.join([key + ' = ' + value['based_on']['datatype'] for key, value in fallback_configuration.items()])
            if measurement is None:
                print(f'WARNING {self.name}: No measurement found for creating fallback configuration entry for \"{print_config}\"')
            else:
                # Add a line with a comment describing the fallback case
                out_lines.append(f'// Based on {print_config}')
                out_lines.append(self._create_fallback_case(benchmarks_of_architecture, fallback_configuration, measurement))
        
        return out_lines

    def __get_configurations(self) -> List[str]:
        """
        Generate each line of configuration, where configuration
        is a valid cpp template instantiation.
        """

        configuration_lines: List[str] = []

        # Hardcoded configuration in case none of the specializations can be instantiated
        configuration_lines.append(self._create_general_base_case())

        for benchmarks_of_architecture in self.architectures.values():
            # Per-architecture configuration
            configuration_lines.append(self._create_base_case_for_arch(
                benchmarks_of_architecture.base_config_case(self.config_selection_types, self.name), benchmarks_of_architecture))
            for configuration, measurement in benchmarks_of_architecture.best_config_by_selection_types.items():
                # Per-architecture and per-data-key configuration
                configuration_lines.append(self._create_specialized_case_for_arch(
                    measurement, benchmarks_of_architecture, configuration))

            # Fallback cases
            configuration_lines += self.__create_fallback_cases(
                self.fallback_entries, benchmarks_of_architecture)
        
        return configuration_lines

    def __get_fallback_match(self, benchmarks_of_architecture, fallback_configuration) -> Dict[str, str]:
        """
        fallback_configuration is a dict of (name, fallback entry) pairs. 
        Returns the configuration that matches the entry for each pair.
        """
        for benchmark in benchmarks_of_architecture.best_config_by_selection_types.values():
            for cfg_type in self.config_selection_types:
                # The typename of the benchmark for this config selection type
                instanced_typename: str = benchmark[cfg_type.name]

                # The typename the fallback is based on
                fallback_typename: str = fallback_configuration[cfg_type.name]['based_on']['datatype']
                
                # The typenames match if they are equal or if the fallback is based on the empty type
                # and the benchmark's typename is None (meaning that the tokenizer found no type)

                match: bool = \
                    instanced_typename == fallback_typename or \
                    (fallback_typename == EMPTY_TYPENAME and not instanced_typename)
                if not match:
                    break
            else:
                # if all typenames of the benchmark correspond to the typenames of the fallback config
                return benchmark
        return None

"""
Each algorithm class specifies methods to generate each C++ configuration specification.
The generated configuration file contains configs for four cases:
- No architecture or instantiation of configuration selection types is provided 
  (general base case).
- Only the architecture is specified, no instantiation of configuration selection 
  types is provided (base case for arch).
- The architecture and an instantiation of configuration selection types is 
  provided (specialized case for arch). 
- The architecture and an instantiation of configuration selection types is 
  provided, but there is no benchmark with the same instantiation of types.
  The configuration is based on a fallback (fallback case). 

config_selection_types is a list of types that are used to select a configuration.
The fallback file will be used to generate fallback cases, in addition
to the typenames specified in the benchmark runs. Generating fallbacks only happens
when there is only a single non-optional type.

If the type is optional, the generated fallback cases will use the empty type instead
of the full list of fallback entries. The config_selection_types should specify at 
least one non-optional type.

The 'name' fields should correspond to a named capturing group in the regex field of the benchmark,
these names should be valid C++ identifiers. The matched values in the name field of
the benchmark should also be valid C++ typenames. This is required as these names will be in the 
generated C++ code.
"""

class AlgorithmDeviceMergeSort(Algorithm):
    config_selection_types = [
        SelectionType(name='key_type', is_optional=False),
        SelectionType(name='value_type', is_optional=True)]

    def __init__(self, algorithm_name, fallback_entries):
        Algorithm.__init__(self, algorithm_name, fallback_entries)

    def _create_general_base_case(self):
        return "template<unsigned int arch, class key_type, class value_type, class enable = void> struct default_merge_sort_config :\n" \
            "default_merge_sort_config_base<key_type, value_type> { };"

    def _create_base_case_for_arch(self, measurement, arch):
        return f"template<class key_type, class value_type> struct default_merge_sort_config<{arch.name}, key_type, value_type> :\n" + \
            self.__create_device_merge_sort_configuration_template(measurement)

    def _create_specialized_case_for_arch(self, measurement, arch, configuration):
        return f"template<> struct default_merge_sort_config<{arch.name}, {configuration.key_type}, {configuration.value_type}> :\n" + \
            self.__create_device_merge_sort_configuration_template(measurement)

    def _create_fallback_case(self, arch, fallback_configuration, measurement):
        return f"template<class key_type, class value_type> struct default_merge_sort_config<{arch.name}, key_type, value_type, "\
            f"{translate_settings_to_cpp_metaprogramming(fallback_configuration)}> :\n" + \
            self.__create_device_merge_sort_configuration_template(measurement)

    def __create_device_merge_sort_configuration_template(self, measurement):
        return f"merge_sort_config<{measurement['merge_block_size']}, {measurement['sort_block_size']}, {measurement['sort_items_per_thread']}> {{ }};"

class AlgorithmDeviceRadixSort(Algorithm):
    config_selection_types = [
        SelectionType(name='key_type', is_optional=False),
        SelectionType(name='value_type', is_optional=True)]

    def __init__(self, algorithm_name, fallback_entries):
        Algorithm.__init__(self, algorithm_name, fallback_entries)

    def _create_general_base_case(self):
        return "template<unsigned int arch, class key_type, class value_type, class enable = void> struct default_radix_sort_config :\n" \
            "default_radix_sort_config_base<key_type, value_type> { };"

    def _create_base_case_for_arch(self, measurement, arch):
        return f"template<class key_type, class value_type> struct default_radix_sort_config<{arch.name}, key_type, value_type> :\n" + \
            self.__create_device_radix_sort_configuration_template(measurement)

    def _create_specialized_case_for_arch(self, measurement, arch, configuration):
        return f"template<> struct default_radix_sort_config<{arch.name}, "\
            f"{configuration.key_type}, {(configuration.value_type)}> :\n" + \
            self.__create_device_radix_sort_configuration_template(measurement)

    def _create_fallback_case(self, arch, fallback_configuration, measurement):
        return f"template<class key_type, class value_type> struct default_radix_sort_config<{arch.name}, key_type, value_type, " \
            f"{translate_settings_to_cpp_metaprogramming(fallback_configuration)}> :\n" + \
            self.__create_device_radix_sort_configuration_template(measurement)

    def __create_device_radix_sort_configuration_template(self, measurement):
        return f"radix_sort_config<{measurement['long_radix_bits']}, {measurement['short_radix_bits']}, "\
            f"::rocprim::kernel_config<{measurement['scan_block_size']}, {measurement['scan_items_per_thread']}>, "\
            f"::rocprim::kernel_config<{measurement['sort_block_size']}, {measurement['sort_items_per_thread']}>, "\
            f"::rocprim::kernel_config<{measurement['sort_single_block_size']}, {measurement['sort_single_items_per_thread']}>, "\
            f"::rocprim::kernel_config<{measurement['sort_merge_block_size']}, {measurement['sort_merge_items_per_thread']}>, "\
            f"{measurement['force_single_kernel_config']}> {{ }};"

class AlgorithmDeviceReduce(Algorithm):
    config_selection_types = [SelectionType(name='datatype', is_optional=False)]

    def __init__(self, algorithm_name, fallback_entries):
        Algorithm.__init__(self, algorithm_name, fallback_entries)

    def _create_general_base_case(self):
        return "template<unsigned int arch, class datatype, class enable = void> struct default_reduce_config :\n" \
            "default_reduce_config_base<datatype> { };"

    def _create_base_case_for_arch(self, measurement, arch):
        return f"template<class datatype> struct default_reduce_config<{arch.name}, datatype> :\n" + \
            self.__create_device_reduce_configuration_template(measurement)

    def _create_specialized_case_for_arch(self, measurement, arch, configuration):
        return f"template<> struct default_reduce_config<{arch.name}, {configuration.datatype}> :\n" + \
            self.__create_device_reduce_configuration_template(measurement)

    def _create_fallback_case(self, arch, fallback_configuration, measurement):
        return f"template<class datatype> struct default_reduce_config<{arch.name}, datatype, "\
            f"{translate_settings_to_cpp_metaprogramming(fallback_configuration)}> :\n" + \
            self.__create_device_reduce_configuration_template(measurement)

    def __create_device_reduce_configuration_template(self, measurement):
        return f"reduce_config<{measurement['block_size']}, {measurement['items_per_thread']}, ::rocprim::block_reduce_algorithm::using_warp_reduce> {{ }};"

class AlgorithmDeviceScan(Algorithm):
    config_selection_types = [SelectionType(name='value_type', is_optional=False)]

    def __init__(self, algorithm_name, fallback_entries):
        Algorithm.__init__(self, algorithm_name, fallback_entries)

    def _create_general_base_case(self):
        return "template<unsigned int arch, class value_type, class enable = void> struct default_scan_config :\n" \
            "default_scan_config_base<value_type> { };"

    def _create_base_case_for_arch(self, measurement, arch):
        return f"template<class value_type> struct default_scan_config<{arch.name}, value_type> :\n" + \
            self.__create_device_scan_configuration_template(measurement)

    def _create_specialized_case_for_arch(self, measurement, arch, configuration):
        return f"template<> struct default_scan_config<{arch.name}, {configuration.value_type}> :\n" + \
            self.__create_device_scan_configuration_template(measurement)

    def _create_fallback_case(self, arch, fallback_configuration, measurement):
        return f"template<class value_type> struct default_scan_config<{arch.name}, value_type, "\
            f"{translate_settings_to_cpp_metaprogramming(fallback_configuration)}> :\n" + \
            self.__create_device_scan_configuration_template(measurement)

    def __create_device_scan_configuration_template(self, measurement):
        return f"scan_config<{measurement['block_size']}, {measurement['items_per_thread']}, true, "\
            "::rocprim::block_load_method::block_load_transpose, "\
            "::rocprim::block_store_method::block_store_transpose, " \
            f"{measurement['block_scan_algo']}> {{ }};"

class AlgorithmDeviceScanByKey(Algorithm):
    config_selection_types = [
        SelectionType(name='key_type', is_optional=False),
        SelectionType(name='value_type', is_optional=False)]

    def __init__(self, algorithm_name, fallback_entries):
        Algorithm.__init__(self, algorithm_name, fallback_entries)

    def _create_general_base_case(self):
        return "template<unsigned int arch, class key_type, class value_type, class enable = void> struct default_scan_by_key_config :\n" \
            "default_scan_by_key_config_base<key_type, value_type> { };"

    def _create_base_case_for_arch(self, measurement, arch):
        return f"template<class key_type, class value_type> struct default_scan_by_key_config<{arch.name}, key_type, value_type> :\n" + \
            self.__create_device_scan_by_key_configuration_template(
                measurement)

    def _create_specialized_case_for_arch(self, measurement, arch, configuration):
        return f"template<> struct default_scan_by_key_config<{arch.name}, {configuration.key_type}, {configuration.value_type}> :\n" + \
            self.__create_device_scan_by_key_configuration_template(
                measurement)

    def _create_fallback_case(self, arch, fallback_configuration, measurement):
        return f"template<class key_type, class value_type> struct default_scan_by_key_config<{arch.name}, key_type, value_type, "\
            f"{translate_settings_to_cpp_metaprogramming(fallback_configuration)}> :\n" + \
            self.__create_device_scan_by_key_configuration_template(measurement)

    def __create_device_scan_by_key_configuration_template(self, measurement):
        return f"scan_by_key_config<{measurement['block_size']}, {measurement['items_per_thread']}, true, "\
            "::rocprim::block_load_method::block_load_transpose, "\
            "::rocprim::block_store_method::block_store_transpose, " \
            f"{measurement['block_scan_algo']}> {{ }};"

def create_algorithm(algorithm_name: str, fallback_entries):
    if algorithm_name == 'device_merge_sort':
        return AlgorithmDeviceMergeSort(algorithm_name, fallback_entries)
    elif algorithm_name == 'device_radix_sort':
        return AlgorithmDeviceRadixSort(algorithm_name, fallback_entries)
    elif algorithm_name == 'device_reduce':
        return AlgorithmDeviceReduce(algorithm_name, fallback_entries)
    elif algorithm_name == 'device_scan':
        return AlgorithmDeviceScan(algorithm_name, fallback_entries)
    elif algorithm_name == 'device_scan_by_key':
        return AlgorithmDeviceScanByKey(algorithm_name, fallback_entries)
    else:
        raise(KeyError)

class BenchmarkDataManager:
    """
    Aggregates the data from multiple benchmark files containing single benchmark runs
    with different configurations.
    """
    
    def __init__(self, fallback_config_file: str):
        self.algorithms: Dict[str, Algorithm] = {}
        abs_path_to_script_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.abs_path_to_template: str = os.path.join(abs_path_to_script_dir, 'config_template')

        # Read in fallback json file to list of dictionaries
        # Remove all fallback cases that are not based on a datatype
        fallback_entries = json.load(fallback_config_file)['fallback_cases']
        processed_fallback_entries: List[Dict] = []
        for fallback_settings_entry in fallback_entries:
            if('datatype' in fallback_settings_entry['based_on'].keys()):
                processed_fallback_entries.append(fallback_settings_entry)
            else:
                print(f"WARNING: Currently only fallbacks based on datatype are implemented, ignoring \"{fallback_settings_entry['based_on']}\"")
        self.fallback_entries: List[Dict] = processed_fallback_entries

    def add_run(self, benchmark_run_file_path: str):
        """
        Adds a single file containing the results of benchmarks executed on a single architecture.
        The benchmarks within the file may belong to different algorithms.
        """

        with open(benchmark_run_file_path) as file_handle:
            benchmark_run_data = json.load(file_handle)
        # these are mirrored from the target_arch enum in device/config_types.hpp
        target_architectures = ['gfx803', 'gfx900', 'gfx906', 'gfx908', 'gfx90a', 'gfx1030']
        name_from_context = benchmark_run_data['context']['hdp_gcn_arch_name'].split(":")[0]
        if name_from_context in target_architectures:
            arch = f'target_arch::{name_from_context}'
        else:
            raise RuntimeError(f"ERROR: unknown hdp_gcn_arch_name: {name_from_context}")
        name_regex = benchmark_run_data['context']['autotune_config_pattern']
        for single_benchmark in benchmark_run_data['benchmarks']:
            tokenized_name = tokenize_benchmark_name(single_benchmark['name'], name_regex)
            if not tokenized_name:
                raise RuntimeError(f"ERROR: cannot tokenize \"{single_benchmark['name']}\" with regex:\n{name_regex}")
            single_benchmark: Dict[str, str] = dict(single_benchmark, **tokenized_name)
            single_benchmark['arch'] = arch

            algorithm_name: str = single_benchmark['algo']
            if algorithm_name not in self.algorithms:
                self.algorithms[algorithm_name] = create_algorithm(algorithm_name, self.fallback_entries)
            self.algorithms[algorithm_name].add_measurement(single_benchmark)

    def write_configs_to_files(self, base_dir: str):
        """
        For each algorithm, creates a file containing configurations and places these in base_dir.
        """

        for algo_name, algo in self.algorithms.items():
            config: str = algo.create_config_file_content(self.abs_path_to_template)
            path_str: str = os.path.join(base_dir, algo_name)
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

    for benchmark_run_file in args.benchmark_files:
        bench_path = benchmark_run_file
        benchmark_manager.add_run(bench_path)
    
    benchmark_manager.write_configs_to_files(args.out_basedir)

if __name__ == '__main__':
    main()
