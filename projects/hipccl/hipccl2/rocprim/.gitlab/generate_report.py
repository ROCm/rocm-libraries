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

import json
import argparse
import os
import re
import stat
import sys
class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def load_benchmarks(benchmark_dir):
    def is_benchmark_json(filename):
        if not re.match(r'.*\.json$', filename):
            return False
        path = os.path.join(benchmark_dir, filename)
        st_mode = os.stat(path).st_mode

        # we are not interested in permissions, just whether it is a regular file (S_IFREG)
        return (st_mode & stat.S_IFREG)

    def add_results(results, file_path: str):
        """
        Adds a single file to the results. The file contains the results of benchmarks executed on a single architecture.
        The benchmarks within the file may belong to different algorithms.
        """
        with open(file_path, "r+") as file_handle:
            # Fix Google Benchmark comma issue
            contents = file_handle.read()
            contents = re.sub(r"(\s*\"[^\"]*\"[^,])(^\s*\"[^\"]*\":)", "\\1,\\2", contents, 0, re.MULTILINE)
            file_handle.seek(0)
            file_handle.write(contents)
            file_handle.truncate()

        with open(file_path) as file_handle:
            benchmark_run_data = json.load(file_handle)

        try:
            arch = benchmark_run_data['context']['hdp_gcn_arch_name'].split(":")[0]
            results.setdefault(arch, {})
            for single_benchmark in benchmark_run_data['benchmarks']:
                name = single_benchmark['name'].replace('/manual_time','')
                name = re.sub(r"(^device.*?)(,\s[A-z_]*_config.*>)$", "\\1>", name, 0, re.MULTILINE)
                results[arch][name] = single_benchmark['bytes_per_second']
        except KeyError as err:
            print(f'KeyError: {err}, while reading file: {file_path}', file=sys.stderr, flush=True)

    benchmark_names = [name for name in os.listdir(benchmark_dir) if is_benchmark_json(name)]
    print('The following benchmark results will be reported:\n{}'.format('\n'.join(benchmark_names)))
    # Results is: {arch : {algorithm : bytes_per_second}, ...}
    results = {}
    for benchmark_name in benchmark_names:
        path = os.path.join(benchmark_dir, benchmark_name)
        add_results(results, path)

    return results

def compare_results(old, new):
    results = []
    incomparable = 0
    for (arch, names) in new.items():
        if arch in old:
            for (name, value_new) in names.items():
                if name in old[arch]:
                    results.append((f'{name} ({arch})', ((value_new - old[arch][name]) / old[arch][name]) * 100))
                else:
                    incomparable = incomparable + 1

    if(incomparable > 0):
        print(f'Could not compare {incomparable} benchmarks.')
    print(f'----------------------------------------')

    success = True
    results.sort(key = lambda x: x[0])
    for (name, difference) in results:
        if difference < -10:
            success = False
            print(f'{bcolors.FAIL}X {bcolors.ENDC} {name}: {bcolors.FAIL}{difference:.0f}{bcolors.ENDC}%')
        elif difference < -2:
            success = False
            print(f'{bcolors.WARNING}! {bcolors.ENDC} {name}: {bcolors.WARNING}{difference:.0f}{bcolors.ENDC}%')
        else:
            print(f'{bcolors.OKGREEN}OK{bcolors.ENDC} {name}: {bcolors.OKGREEN}{difference:.0f}{bcolors.ENDC}%')

    return success

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old',
                        help='The local directory that contains the old benchmark json files',
                        required=True)
    parser.add_argument('--new',
                        help='The local directory that contains the new benchmark json files',
                        required=True)
    args = parser.parse_args()

    old_benchmarks = load_benchmarks(args.old)
    new_benchmarks = load_benchmarks(args.new)
    return compare_results(old_benchmarks, new_benchmarks)

if __name__ == '__main__':
    success = main()
    if success:
        exit(0)
    else:
        exit(1)
