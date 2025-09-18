#!/usr/bin/env python3

import argparse
import subprocess
import os
from pathlib import Path
import yaml
import sys

script_dir = os.path.dirname( __file__ )
getspec_module_dir = os.path.join(script_dir, '..', 'blas')
sys.path.append(getspec_module_dir)
import getspecs

def run_command(bench_command, input_yaml_file, output_csv_file):

    with open(input_yaml_file, 'r') as file:
        problems = file.read()

    output = subprocess.check_output([bench_command,
                                      '--log_function_name',
                                      '--log_datatype',
                                      '--yaml', input_yaml_file])
    output = output.decode('utf-8')

    with open(output_csv_file, 'w') as f:
        f.write(output)

if __name__ =='__main__':

    parser = argparse.ArgumentParser(
            description='run rocblas-bench for multiple yaml files',
            epilog='Example usage: python3 benchmark.py ' +
                    '-l blas1 -t gfx906  -f scal -f axpy ' +
                    '-b ../../../build_tensile/release/clients/staging/rocblas-bench ')

    parser.add_argument('-l', help='level.',          dest='level',          default="blas1")
    parser.add_argument('-t', help='tag.',            dest='tag',            default="gfx906")
    parser.add_argument('-f', help='function names.', dest='function_names', required=True, action='append')
    parser.add_argument('-b', help='bench command.',  dest='bench_command',  default="../../../build/release/clients/staging/rocblas-bench")

    args = parser.parse_args()

    device_number = 1
    cuda = False
    arch = getspecs.getgfx(device_number, cuda)
    if arch not in args.tag:
        print("Warning: tag does not contain architecture, it is recommended that tag contain the architecture")
        print("         tag, architecture = ", args.tag, ", ", arch)

    output_dir = os.path.join(args.level,args.tag)

    if os.path.exists(output_dir):
        print("Warning: directory exists: ", output_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for function_name in args.function_names:

        input_filename = os.path.join(args.level, function_name+".yaml")
        output_filename = os.path.join(args.level, args.tag, function_name+".csv")

        run_command(args.bench_command, input_filename, output_filename)
