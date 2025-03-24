#!/usr/bin/python3

import argparse
import itertools
import os
import re
import subprocess
import sys

###############################################################################
############################ TEST SUITE VARIABLES #############################
###############################################################################
algorithms = {
    0:  'get',
    1:  'get_nbi',
    2:  'put',
    3:  'put_nbi',
    4:  'get_swarm',
    5:  'reduction',
    6:  'amo_fadd',
    7:  'amo_finc',
    8:  'amo_fetch',
    9:  'amo_fcswap',
    10: 'amo_add',
    11: 'amo_inc',
    12: 'amo_cswap',
    13: 'init',
    14: 'ping_pong',
    15: 'barrier',
    16: 'random',
    17: 'barrier_all',
    18: 'sync_all',
    19: 'sync_test',
    20: 'broadcast',
    21: 'collect',
    22: 'fcollect',
    23: 'all_to_all',
    24: 'all_to_alls'
}

primitives = [0, 1, 2, 3]
threaded = [4]
atomics = [6, 7, 8, 9, 10, 11, 12]
inits = [13]
point_to_points = [14]
collectives = [5, 15, 17, 18, 19, 20, 21, 22, 23, 24]

single_thread_algorithms = primitives + \
                           atomics + \
                           inits + \
                           point_to_points + \
                           [5] # reduction

multi_thread_algorithms = primitives + \
                          threaded + \
                          atomics + \
                          inits + \
                          point_to_points + \
                          [5] # reduction

reverse_offload_algorithms = primitives + \
                             threaded + \
                             inits + \
                             point_to_points + \
                             [5]

###############################################################################
############################## PARSER FUNCTIONS ###############################
###############################################################################
def parse_command_line():
    parser = argparse.ArgumentParser(
        description='Execute rocshmem integration tests.')

    parser.add_argument('--mpirun_procs',
                        dest='mpirun_procs',
                        type=int,
                        nargs='+',
                        default=[2, 4, 8])

    parser.add_argument('--mpirun_machines',
                        dest='mpirun_machines',
                        type=str,
                        nargs='+',
                        default=['ast-rocm1',
                                 'ast-rocm2',
                                 'ast-rocm3',
                                 'ast-rocm4'])

    parser.add_argument('--library_build_config_type',
                        dest='library_build_config_type',
                        type=str,
                        default='rc_single')

    parser.add_argument('--client_binary_path',
                        dest='client_binary_path',
                        type=str,
                        default=os.getcwd()+'/build/rocshmem_example_driver')

    parser.add_argument('--output_directory_path',
                        dest='output_directory_path',
                        type=str,
                        default=os.getcwd()+'/build')

    parser.add_argument('--number_threads',
                        dest='number_threads',
                        type=int,
                        nargs='+',
                        default=1)

    parser.add_argument('--number_workgroups',
                        dest='number_workgroups',
                        type=int,
                        nargs='+',
                        default=1)

    parser.add_argument('--workgroup_size',
                        dest='workgroup_size',
                        type=int,
                        nargs='+',
                        default=[64, 1024])

    parser.add_argument('--workgroup_context_type',
                        dest='workgroup_context_type',
                        type=int,
                        nargs='+',
                        default=[0, 8])

    parser.add_argument('--max_message_size',
                        dest='max_message_size',
                        type=int,
                        default=1048576)

    parser.add_argument('--algorithms',
                        dest='algorithms',
                        type=int,
                        nargs='*',
                        default=None)

    return parser.parse_args()

def convert_arguments_to_dictionary(args):
    return vars(args)

def determine_algos_from_library_config_type(config):
    if config['algorithms']:
        return config

    thread_single = re.match('.*single.*', config['library_build_config_type'])

    if thread_single:
        config['algorithms'] = single_thread_algorithms
    else:
        config['algorithms'] = multi_thread_algorithms

    return config

def process_arguments(args):
    config = convert_arguments_to_dictionary(args)
    config = determine_algos_from_library_config_type(config)
    return config

###############################################################################
######################## OPTION CONVERSION DICTIONARY #########################
###############################################################################
option_keyword_dictionary = {
    'mpirun_procs' : '-np',
    'mpirun_machines' : '--host',
    'number_threads' : '-t',
    'number_workgroups' : '-w',
    'max_message_size' : '-s',
    'algorithms' : '-a',
    'workgroup_size' : '-z',
    'workgroup_context_type' : '-x',
}

###############################################################################
########################## TEST GENERATION FUNCTIONS ##########################
###############################################################################
def separate_dictionaries(dictionary):
    library_build_config = {}
    mpirun = {}
    path = {}
    driver = {}

    library_build_config_keys = [
        'library_build_config_type'
    ]
    mpirun_keys = [
        'mpirun_machines',
        'mpirun_procs'
    ]
    path_keys = [
        'output_directory_path',
        'client_binary_path'
    ]

    for key, value in dictionary.items():
        if key in library_build_config_keys:
            library_build_config[key] = value
        elif key in mpirun_keys:
            mpirun[key] = value
        elif key in path_keys:
            path[key] = value
        else:
            driver[key] = value

    return library_build_config, mpirun, path, driver

def convert_dict_values_to_list_of_lists(dictionary):
    values = dictionary.values()
    list_of_lists = []
    for element in values:
        if isinstance(element, list):
            list_of_lists.append(element)
        else:
            single_element_list = [element]
            list_of_lists.append(single_element_list)
    return list_of_lists

def cartesian_product(list_of_lists):
    return list(itertools.product(*list_of_lists))

def create_config_combinations(dictionary):
    keys = []
    for k,v in dictionary.items():
        keys.append(k)

    values_list_of_lists = convert_dict_values_to_list_of_lists(dictionary)
    combinations = cartesian_product(values_list_of_lists)
    return keys, combinations

def create_all_mpirun_machine_combinations(mpirun_dict):
    mpirun_machine_combinations = []
    number_of_machines = len(mpirun_dict['mpirun_machines'])

    for i in range(1, number_of_machines + 1):
        combinations_object = \
            itertools.combinations(mpirun_dict['mpirun_machines'], i)
        combinations_list = list(combinations_object)
        mpirun_machine_combinations += combinations_list

    machine_combo_strings = stringify_machines(mpirun_machine_combinations)
    mpirun_dict['mpirun_machines'] = machine_combo_strings

    return mpirun_dict

def stringify_machines(machine_list_of_tuples):
    list_of_strings = []
    for tup in machine_list_of_tuples:
        list_of_strings.append(','.join(map(str, tup)))
    return list_of_strings

def stringify_config(keys, combo_list):
    string_list = []
    for combo_tuple in combo_list:
        one_string = ''
        for position in range(len(keys)):
            substr = '{key} {combo}'.format(key=keys[position],
                                            combo=combo_tuple[position])
            one_string += substr + ' '
        string_list.append(one_string)
    return string_list

def generate_test_commands(config_templates):
    (build_config, mpirun, path, driver) = \
        separate_dictionaries(config_templates)

    driver_keys, driver_combos = create_config_combinations(driver)
    driver_commands = stringify_config(driver_keys, driver_combos)

    mpirun = create_all_mpirun_machine_combinations(mpirun)
    mpirun_keys, mpirun_combos = create_config_combinations(mpirun)
    mpirun_commands = stringify_config(mpirun_keys, mpirun_combos)

    commands = []
    for mpirun_command in mpirun_commands:
        for driver_command in driver_commands:
            mpirun_bin = 'mpirun '
            driver_bin = path['client_binary_path'] + ' '
            command = mpirun_bin + \
                      mpirun_command + \
                      driver_bin + \
                      driver_command

            commands.append(command)

    return commands

def replace_option_keywords(commands):
    altered_commands = []
    for command in commands:
        for k,v in option_keyword_dictionary.items():
            command = command.replace(k, v)
        altered_commands.append(command)
    return altered_commands

def generate_tests():
    command_line_arguments = parse_command_line()
    config_template = process_arguments(command_line_arguments)
    commands = generate_test_commands(config_template)
    commands = replace_option_keywords(commands)
    return commands

###############################################################################
############################ FILE HELPER FUNCTIONS ############################
###############################################################################
def open_output_file(filepath):
    try:
        f = open(filepath, 'r')
    except IOError:
        sys.exit('failed to open: ' + filepath)
    return f

def create_file_name():
    return

def write_output():
    return

###############################################################################
########################## TEST EXECUTION FUNCTIONS ###########################
###############################################################################
def single_test(test):
    try:
        subprocess.run(test, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e)
    return

def all_tests():
    for test in generate_tests():
        print(test)
        #single_test(test)
    return

###############################################################################
############################## SCRIPT MAIN BODY ###############################
###############################################################################
all_tests()
