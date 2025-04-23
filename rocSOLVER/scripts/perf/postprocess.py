# ##########################################################################
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# ##########################################################################

import argparse
import pandas as pd

def append_units(label):
     if label.endswith('_time'):
         return label + '_us'
     else:
         return label

def operation_count(n, function):
    """Returns the number of operations required for a given function with an n-by-n matrix."""
    if function.startswith('getrf'):
        return (2 * n ** 3 / 3) - (n ** 2 / 2) - (n / 6)
    elif function.startswith('geqrf'):
        return (4 * n ** 3 / 3) + (2 * n ** 2) + (4 * n)
    elif function.startswith('getri'):
        return (4 * n ** 3 / 3) - (n ** 2) + (5 * n / 3)
    else: 
        raise ValueError(f'Could not calculate operation count for {function}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Processes the rocsolver-benchmark-suite results.')
    parser.add_argument('input_path',
            help='the input file name for the raw benchmark results')
    parser.add_argument('output_path',
            help='the output file name for the processed benchmark results')
    args = parser.parse_args()

    # load the data and filter on the function name
    df = pd.read_csv(args.input_path, encoding='utf-8')

    # rename columns for nicer labels
    df.rename(append_units, axis='columns', inplace=True)

    # prepend name column
    df.insert(0, 'name', df['precision'].str.cat(df['function']))

    # the op_count derived column calculation can't handle a mix of functions
    if not df['function'].empty and (df['function'] != df['function'][0]).all():
        raise ValueError('This script can only handle one function at a time.')
    function = df['function'][0]

    # append derived columns
    df['mean_gpu_time_us_per_matrix'] = df['gpu_time_us'].astype(float) / df.get('batch_c', 1)
    df['op_count'] = operation_count(df['n'], function).round().astype(int) * df.get('batch_c', 1)
    df['performance_gflops'] = df['op_count'] / df['gpu_time_us'] / 1000

    # save results
    df.to_csv(args.output_path, encoding='utf-8', index=False)
