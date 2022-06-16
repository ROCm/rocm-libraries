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


def tokenize_test_name(input_name, name_regex):
    match = re.search(name_regex, input_name)
    data_dict = match.groupdict()
    return data_dict

def translate_settings_to_cpp_metaprogramming(config_dict):
    """
    Translates the entry of the fallback configuration to cpp metaprograming idioms
    """
    
    setting_list = []
    begin = "std::enable_if<("
    end = ")>"
    if "floating_point" in config_dict.keys():
        negation = "" if config_dict['floating_point'] else "!"
        output = negation + "bool(rocprim::is_floating_point<Value>::value)"
        setting_list.append(output)

    for config_setting, value in config_dict['sizeof'].items():

        if config_setting == "min_exclusive":
            setting_list.append(f"(sizeof(Value) > {value})")
        elif config_setting == "max_inclusive":
            setting_list.append(f"(sizeof(Value) <= {value})")
        else:
            print(f"WARNING: {config_setting} is not known")
    
    return begin + " && ".join(setting_list) + end


class BenchmarksOfArchitecture:
    def __init__(self, arch_name):
        # Stores datatype as keys, measurement data as values
        self.datatypes = {}
        self.arch_name = arch_name

    def add_measurement(self, data_entry):
        datatype = data_entry['datatype']
        if datatype not in self.datatypes.keys():
            self.datatypes[datatype] = []
        self.datatypes[datatype].append(data_entry)

    @property
    def name(self):
        return self.arch_name

    @property
    def base_config_case(self):
        # For now lets just return the best int performance as a fallback if the dtype is not found
        return max(self.datatypes['int'], key=lambda x: x['items_per_second'])

    @property
    def specialized_config_cases(self):
        # return a dict
        output = {}
        for key, value in self.datatypes.items():
            output[key] = max(value, key=lambda x: x['items_per_second'])
        return output


class Algorithm:
    """
    Aggregates the data for a algorithm, including the generation of
    the configuration file
    """

    def __init__(self, algorithm_name, abs_path_to_script_dir, abs_path_to_fallback):
        self.name = algorithm_name
        self.architectures = {}
        self.configuration_lines = []
        self.abs_path_to_script_dir = abs_path_to_script_dir
        self.abs_path_to_fallback = abs_path_to_fallback
    
    def architecture_exists(self, architecture_name):
        return architecture_name in self.architectures.keys()

    def add_new_architecture(self, architecture_name):
        benchmarks_of_architecture = BenchmarksOfArchitecture(architecture_name)
        self.architectures[architecture_name] = benchmarks_of_architecture

    def get_architecture(self, architecture_name):
        return self.architectures[architecture_name]

    def add_measurement(self, single_benchmark_data):
        architecture_name = single_benchmark_data['arch']
        if not self.architecture_exists(architecture_name):
            self.add_new_architecture(architecture_name)
        self.get_architecture(architecture_name).add_measurement(single_benchmark_data)


    def create_config_file_content(self):
        """
        Generate the content of the configuration file, including license
        and header guards, based on general template file
        """

        generated_config_file_content=""
        self.set_configurations()        

        configuration= '\n'.join(self.configuration_lines)

        with open(self.abs_path_to_template) as template_file:
            template_file_content = template_file.read()
            generated_config_file_content=template_file_content.format(guard=self.name.upper(), config_body=configuration)
        return generated_config_file_content

    def set_configurations(self):
        """
        Generate each line of configuration, where configuration
        is a valid cpp template instantiation
        """

        self.configuration_lines.append(self._create_general_base_case())
        for benchmarks_of_architecture in self.architectures.values():
            self.configuration_lines.append(self._create_base_case_for_arch(benchmarks_of_architecture))
            self.configuration_lines += self._create_specialized_cases_for_arch(benchmarks_of_architecture)
            self.configuration_lines += self._create_fallback_cases(benchmarks_of_architecture)

    def get_best_case_of_datatype(self, arch, datatype):
        for best_configuration_case in arch.specialized_config_cases.values():
            if best_configuration_case['datatype']  == datatype:
                return best_configuration_case

class AlgorithmDeviceReduce(Algorithm):
    def __init__(self, algorithm_name, abs_path_to_script_dir, abs_path_to_fallback):
        Algorithm.__init__(self, algorithm_name, abs_path_to_script_dir, abs_path_to_fallback)
        self.abs_path_to_template=os.path.join(self.abs_path_to_script_dir, "config_template")


    def _create_general_base_case(self):
        #Hardcode some configurations in case non of the specializations can be instantiated
        return "template<unsigned int arch, class Value, class enable = void> struct default_reduce_config  : reduce_config<256, 4, ::rocprim::block_reduce_algorithm::using_warp_reduce> { };"

    def _create_base_case_for_arch(self, arch):
        measurement = arch.base_config_case
        return f"template<class Value> struct default_reduce_config<{arch.name}, Value>  :" +  self.__create_device_reduce_configuration_template(measurement)

    def _create_specialized_cases_for_arch(self, arch):
        out = []
        for data_type, measurement in arch.specialized_config_cases.items():
            out.append(f"template<> struct default_reduce_config<{arch.name}, {data_type}> :" + self.__create_device_reduce_configuration_template(measurement) )
        return out

    def _create_fallback_cases(self, arch):
        """
        Generates fallback configuration based on the input present in the fallback_config.json
        """
        
        out=[]
        data = json.load(self.abs_path_to_fallback)
        
        data = data['fallback_cases']
        for fallback_settings_entry in data:
            config_line = f"template<class Value> struct default_reduce_config<{arch.name}, Value, {translate_settings_to_cpp_metaprogramming(fallback_settings_entry)}> :"
            if('datatype' in fallback_settings_entry['based_on'].keys()):
                measurement_entry = self.get_best_case_of_datatype(arch, fallback_settings_entry['based_on']['datatype'])
                if(measurement_entry != None):
                    config_line += self.__create_device_reduce_configuration_template(measurement_entry)
                    out.append(config_line)
                else:
                    print(f"WARNING: No measurement found for creating fallback configuration entry for {fallback_settings_entry['based_on']['datatype']}")
            else:
                print(f"WARNING: Currently only fallbacks based on datatype are implemented")
        return out

    def __create_device_reduce_configuration_template(self, measurement):
        return f" reduce_config<{measurement['block_size']}, {measurement['items_per_thread']}, ::rocprim::block_reduce_algorithm::using_warp_reduce> {{ }};"

class AlgorithmFactory:
    def create_algorithm(self, algorithm_name, abs_path_to_script_dir, abs_path_to_fallback):
        if algorithm_name == 'device_reduce':
            return AlgorithmDeviceReduce(algorithm_name, abs_path_to_script_dir, abs_path_to_fallback)
        else:
            raise(KeyError)

class BenchmarkDataManager:
    """
    Aggregates the data from multiple benchmark files containing single benchmark runs
    with different configurations.
    """
    
    def __init__(self, fallback_config_file):
        self.algorithms={}
        self.algo_factory = AlgorithmFactory()
        self.abs_path_to_script_dir=os.path.dirname(os.path.abspath(__file__))
        self.fallback_config_file=fallback_config_file


    def add_run(self, benchmark_run_file_path, arch):
        benchmark_run_data = {}
        with open(benchmark_run_file_path) as file_handle:
            benchmark_run_data = json.load(file_handle)
        name_regex = benchmark_run_data['context']['autotune_config_pattern']
        for single_benchmark in benchmark_run_data['benchmarks']:
            tokenized_name = tokenize_test_name(single_benchmark['name'], name_regex)
            single_benchmark=dict(single_benchmark, **tokenized_name)
            single_benchmark['arch'] = arch
            self.__add_measurement(single_benchmark)

    def write_configs_to_files(self, base_dir):
        data = self.__generate_configuration()
        for algo_name, config in data.items():
            path_str=os.path.join(base_dir, algo_name)
            with open(path_str, "w") as outfile:
                outfile.write(config)

    def add_new_algorithm(self, algo_name):
        self.algorithms[algo_name] = self.algo_factory.create_algorithm(algo_name, self.abs_path_to_script_dir, self.fallback_config_file)

    def algorithm_exists(self, algo_name):
        return algo_name in self.algorithms.keys()

    def get_algorithm(self, algo_name):
        return self.algorithms[algo_name]
    
    @property
    def path_to_script_dir(self):
        return self.abs_path_to_script_dir

    def __add_measurement(self, single_benchmark_data):
        algorithm_name = single_benchmark_data['algo']
        if not self.algorithm_exists(algorithm_name):
            self.add_new_algorithm(algorithm_name)
        self.get_algorithm(algorithm_name).add_measurement(single_benchmark_data)
    
    def __generate_configuration(self):
        out = {}
        for key, algo in self.algorithms.items():
            out[key] = algo.create_config_file_content()
        return out

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Tool for generating optimized launch parameters for rocPRIM based on benchmark results")
    parser.add_argument('-b','--benchmark_files', nargs='+', help="Benchmarked architectures listed int the form <arch-id>:<path_to_benchmark>.json")
    parser.add_argument("-p", "--out_basedir", type=str, help="Base dir for the output files, for each algorithm a new file will be created in this directory", required=True)
    parser.add_argument("-c", "--fallback_configuration", type=argparse.FileType('r'), default=os.path.join(current_dir, "fallback_config.json"), help="Configuration for fallbacks for not tested datatypes")
    args = parser.parse_args()

    benchmark_manager = BenchmarkDataManager(args.fallback_configuration)

    for benchmark_run_file_and_arch in args.benchmark_files:
        arch_id, bench_path = benchmark_run_file_and_arch.split(":")
        benchmark_manager.add_run(bench_path, arch_id)
    
    benchmark_manager.write_configs_to_files(args.out_basedir)

if __name__ == '__main__':
    main()
