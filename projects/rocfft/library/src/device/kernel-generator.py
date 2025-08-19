#!/usr/bin/env python3
# Copyright (C) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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
"""rocFFT kernel generator.

It accept two sub-commands:
1. list - lists names of kernels that will be generated
2. generate - generate them!

"""

import argparse
import functools
import subprocess
import sys
import json
import threading

import kernels.configs.config_sbrc as config_sbrc
import kernels.configs.config_sbrr as config_sbrr
import kernels.configs.config_sbcc as config_sbcc
import kernels.configs.config_sbcr as config_sbcr
import kernels.configs.config_2d_single as config_2d_single
import kernels.configs.config_pp_3d as config_pp_3d

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from operator import mul

from generator import (ArgumentList, BaseNode, Call, CommentBlock, Function,
                       Include, LineBreak, Map, StatementList, Variable,
                       Assign, name_args, write, ForwardDeclaration)

from collections import namedtuple

LaunchParams = namedtuple(
    'LaunchParams',
    [
        'transforms_per_block',
        'workgroup_size',
        'threads_per_transform',
        'half_lds',  # load real and imag part separately with half regular lds resouce to increase occupancy
        'direct_to_from_reg'
    ]
)  # load from global mem to registers directly and store from registers to global mem.

#
# CMake helpers
#


def scjoin(xs):
    """Join 'xs' with semi-colons."""
    return ';'.join(str(x) for x in xs)


def scprint(xs):
    """Print 'xs', joined by semi-colons, on a single line.  CMake friendly."""
    print(scjoin(xs), end='', flush=True)


def cjoin(xs):
    """Join 'xs' with commas."""
    return ','.join(str(x) for x in xs)


#
# Helpers
#
def unique(kernels):
    """Merge kernel lists without duplicated meta.length; ignore later ones."""
    r, s = list(), set()
    for kernel in kernels:
        if isinstance(kernel.length, list):
            key = tuple(kernel.length) + (kernel.scheme, kernel.lds_size_bytes)
        else:
            key = (kernel.length, kernel.scheme, kernel.lds_size_bytes)
        if key not in s:
            s.add(key)
            r.append(kernel)
    return r


def is_aot_rtc(meta):
    return not meta.runtime_compile


#
# Prototype generators
#


@name_args(['function'])
class FFTKernel(BaseNode):

    def __str__(self):
        aot_rtc = is_aot_rtc(self.function.meta)
        f = 'FFTKernel('
        use_3steps_large_twd = getattr(self.function.meta,
                                       'use_3steps_large_twd', None)
        # assume half-precision needs the same thing as single
        precision = 'sp' if self.function.meta.precision == 'half' else self.function.meta.precision
        if use_3steps_large_twd is not None:
            f += str(use_3steps_large_twd[precision])
        else:
            f += 'false'
        factors = getattr(self.function.meta, 'factors', None)
        if factors is not None:
            f += ', {' + cjoin(factors) + '}'
        transforms_per_block = getattr(self.function.meta,
                                       'transforms_per_block', None)
        if transforms_per_block is not None:
            f += ', ' + str(transforms_per_block)
        workgroup_size = getattr(self.function.meta, 'workgroup_size', None)
        if workgroup_size is not None:
            f += ', ' + str(workgroup_size)
        f += ', {' + ','.join(
            [str(s) for s in self.function.meta.threads_per_transform]) + '}'
        direct_to_from_reg = None
        half_lds = None
        if hasattr(self.function.meta, 'params'):
            half_lds = getattr(self.function.meta.params, 'half_lds', None)
            direct_to_from_reg = getattr(self.function.meta.params,
                                         'direct_to_from_reg', None)
        if half_lds is not None:
            f += ', ' + str(half_lds).lower()
        if direct_to_from_reg is not None:
            f += ', ' + str(direct_to_from_reg).lower()
        f += ', '

        f += 'true' if aot_rtc else 'false'
        f += ', ' + str(self.function.meta.pp_child_scheme)
        f += ', ' + str(self.function.meta.pp_current_dim)
        f += ', ' + str(self.function.meta.pp_off_dim)
        pp_factors_curr = getattr(self.function.meta, 'pp_factors_curr', None)
        if pp_factors_curr is not None:
            f += ', {' + cjoin(pp_factors_curr) + '}'
        pp_factors_other = getattr(self.function.meta, 'pp_factors_other',
                                   None)
        if pp_factors_other is not None:
            f += ', {' + cjoin(pp_factors_other) + '}'
        f += ')'
        return f


def generate_cpu_function_pool_main(num_files):
    """Generate main function_pool.cpp file which calls each function_pool_init function"""
    fwd_declarations = StatementList()
    for i in range(num_files):
        fwd_declarations += ForwardDeclaration(
            type='void',
            name=f'function_pool_init_{i}',
            value=
            'std::tuple<FPKeyMap, PPFPKeyMap>& def_keys, std::tuple<FPMap, PPFPMap>& function_maps'
        )

    call_list = StatementList()
    call_args = ArgumentList('def_keys', 'function_maps')
    for i in range(num_files):
        call_list += Call(name=f'function_pool_init_{i}', arguments=call_args)
    return StatementList(
        Include('"../include/function_pool.h"'), fwd_declarations,
        Function(name='function_pool_data::function_pool_data',
                 value=False,
                 arguments=ArgumentList(),
                 body=call_list))


def generate_cpu_function_pool_pieces(functions, pp_functions, num_files):
    """Generate function(s) to populate the kernel function pool."""

    function_map = Map('function_map')
    precisions = {
        'sp': 'rocfft_precision_single',
        'dp': 'rocfft_precision_double',
        'half': 'rocfft_precision_half',
    }
    var_kernel = Variable('kernel', 'FFTKernel')
    var_pp_kernel_1 = Variable('pp_kernel_1', 'FFTKernel')
    var_pp_kernel_2 = Variable('pp_kernel_2', 'FFTKernel')

    # Init list to store contents of function_pool_init function per file being generated
    piece_contents = [
        StatementList() + var_kernel.declaration() +
        var_pp_kernel_1.declaration() + var_pp_kernel_2.declaration()
        for _ in range(num_files)
    ]

    # Cycles through each file per loop execution to distribute regular kernels work amongst N files
    curr_func, curr_file = 0, 0
    while curr_func < len(functions):
        f = functions[curr_func]
        length, precision, scheme, transpose = f.meta.length, f.meta.precision, f.meta.scheme, f.meta.transpose

        if isinstance(length, (int, str)):
            length = [length, 0]
        piece_contents[curr_file] += Assign(var_kernel, FFTKernel(f))
        key = Call(
            name='FMKey',
            arguments=ArgumentList(length[0], length[1], precisions[precision],
                                   scheme, transpose or 'NONE',
                                   'kernel.get_kernel_config()')).inline()
        piece_contents[curr_file] += function_map.insert(
            key, var_kernel, 'std::get<0>(def_keys)',
            'std::get<0>(function_maps)', f.meta.lds_size_bytes)

        curr_func = curr_func + 1
        curr_file = (curr_file + 1) % num_files

    # Partial-pass kernels are handled separately.
    if len(pp_functions) > 0:
        counter_f_pp_1 = 0
        skip_to_next_iter = False
        # Cycles through each file per loop execution to distribute partial-pass kernels work amongst N files
        while True:
            if counter_f_pp_1 >= len(pp_functions):
                break
            # get first pp kernel
            f_pp_1 = pp_functions[counter_f_pp_1]

            # PPFMKey entry needs two kernels with same length and precision, but different pp_current_dim
            counter_f_pp_2 = counter_f_pp_1 + 1
            if counter_f_pp_2 >= len(pp_functions):
                break
            # loop to get the second pp kernel
            while counter_f_pp_2 < len(pp_functions):
                f_pp_2 = pp_functions[counter_f_pp_2]
                if (f_pp_1.meta.length == f_pp_2.meta.length
                        and f_pp_1.meta.precision == f_pp_2.meta.precision
                        and f_pp_1.meta.pp_current_dim !=
                        f_pp_2.meta.pp_current_dim):
                    break
                if (f_pp_1.meta.length != f_pp_2.meta.length):
                    # we hit a new kernel with different length
                    # start next iteration looking for the next pair
                    counter_f_pp_1 = counter_f_pp_2
                    skip_to_next_iter = True
                    break
                counter_f_pp_2 = counter_f_pp_2 + 1

            if skip_to_next_iter:
                skip_to_next_iter = False
                continue
            if counter_f_pp_2 >= len(pp_functions):
                break
            # get second pp kernel
            f_pp_2 = pp_functions[counter_f_pp_2]

            piece_contents[curr_file] += Assign(var_pp_kernel_1,
                                                FFTKernel(f_pp_1))
            piece_contents[curr_file] += Assign(var_pp_kernel_2,
                                                FFTKernel(f_pp_2))

            length = f_pp_1.meta.length
            precision = f_pp_1.meta.precision
            scheme = f_pp_1.meta.scheme
            key = Call(name='PPFMKey',
                       arguments=ArgumentList(
                           length[0], length[1], length[2],
                           precisions[precision], scheme,
                           'pp_kernel_1.get_kernel_config()',
                           'pp_kernel_2.get_kernel_config()')).inline()
            piece_contents[curr_file] += function_map.insert_pp(
                key, var_pp_kernel_1, var_pp_kernel_2, 'std::get<1>(def_keys)',
                'std::get<1>(function_maps)', f_pp_1.meta.lds_size_bytes)

            counter_f_pp_1 = counter_f_pp_1 + 1
            curr_file = (curr_file + 1) % num_files

    # Assemble contents of each file to return in a list
    pieces = [None] * num_files
    piece_args = ArgumentList('std::tuple<FPKeyMap, PPFPKeyMap>& def_keys',
                              'std::tuple<FPMap, PPFPMap>& function_maps')
    for k in range(num_files):
        pieces[k] = StatementList(
            Include('"../include/function_pool.h"'),
            Function(name=f'void function_pool_init_{k}',
                     value=False,
                     arguments=piece_args,
                     body=piece_contents[k]))

    return pieces


def kernel_name(ns):
    """Given kernel info namespace, return reasonable file name."""

    assert hasattr(ns, 'length')
    length = ns.length

    if isinstance(length, (tuple, list)):
        length = 'x'.join(str(x) for x in length)

    postfix = ''
    if ns.scheme == 'CS_KERNEL_STOCKHAM_BLOCK_CC':
        postfix = '_sbcc'
    elif ns.scheme == 'CS_KERNEL_STOCKHAM_BLOCK_RC':
        postfix = '_sbrc'
    elif ns.scheme == 'CS_KERNEL_STOCKHAM_BLOCK_CR':
        postfix = '_sbcr'

    if hasattr(ns, 'lds_size_bytes'):
        postfix += f'_lds{ns.lds_size_bytes}'

    return f'rocfft_len{length}{postfix}'


def list_small_kernels():
    """Return list of small kernels to generate."""
    kernels1d = config_sbrr.sbrr_kernels

    kernels = [
        NS(**kernel.__dict__,
           scheme='CS_KERNEL_STOCKHAM',
           precision=['sp', 'dp'] if not hasattr(kernel, 'double_precision')
           or kernel.double_precision else ['sp']) for kernel in kernels1d
    ]

    return kernels


def list_large_kernels():
    """Return list of large kernels to generate."""

    # for SBCC kernel, increase desired workgroup_size so that columns per
    # thread block is also increased. currently targeting for 16 columns
    block_width = 16
    sbcc_kernels = config_sbcc.sbcc_kernels
    for k in sbcc_kernels:
        k.scheme = 'CS_KERNEL_STOCKHAM_BLOCK_CC'
        if not hasattr(k, 'workgroup_size'):
            k.workgroup_size = block_width * \
                functools.reduce(mul, k.factors, 1) // min(k.factors)
        if hasattr(k, 'half_lds') and k.half_lds is True:
            k.workgroup_size = min(1024, k.workgroup_size * 2)
        if not hasattr(k, 'length'):
            k.length = functools.reduce(lambda a, b: a * b, k.factors)

    block_width = 16
    sbcr_kernels = config_sbcr.sbcr_kernels
    for k in sbcr_kernels:
        k.scheme = 'CS_KERNEL_STOCKHAM_BLOCK_CR'
        k.half_lds = False
        if not hasattr(k, 'workgroup_size'):
            k.workgroup_size = block_width * \
                functools.reduce(mul, k.factors, 1) // min(k.factors)
        if not hasattr(k, 'length'):
            k.length = functools.reduce(lambda a, b: a * b, k.factors)

    sbrc_kernels = config_sbrc.sbrc_kernels
    for k in sbrc_kernels:
        k.half_lds = False

    return config_sbcc.sbcc_kernels + config_sbcr.sbcr_kernels + config_sbrc.sbrc_kernels


def list_2d_kernels():
    """Return list of fused 2D kernels to generate."""

    fused_2d_kernels = config_2d_single.fused_2d_kernels

    expanded = []
    expanded.extend(
        NS(**kernel.__dict__,
           scheme='CS_KERNEL_2D_SINGLE',
           runtime_compile=True,
           precision=['sp', 'dp'] if not hasattr(kernel, 'double_precision')
           or kernel.double_precision else ['sp'])
        for kernel in fused_2d_kernels)

    return expanded


def list_3d_partial_pass_kernels():
    """Return list of partial-pass 3D kernels to generate."""

    pp_3d_kernels = config_pp_3d.pp_3d_kernels

    expanded = []
    expanded.extend(
        NS(**kernel.__dict__, scheme='CS_3D_PP', runtime_compile=True)
        for kernel in pp_3d_kernels)

    return expanded


def default_runtime_compile(kernels, default_val):
    '''Returns a copy of input kernel list with a default value for runtime_compile.'''

    return [
        k if hasattr(k, 'runtime_compile') else NS(**k.__dict__,
                                                   runtime_compile=default_val)
        for k in kernels
    ]


def generate_kernel_functions(kernels, precisions, launchers_json):
    """Generate CPU functions used to populate function pool with
    each kernel in `kernels`, and its variations.
    """

    kernel_functions = []
    pp_kernel_functions = []
    data = Variable('data_p', 'const void *')
    back = Variable('back_p', 'void *')
    # launchers_json has kernel names as keys to a list of launchers for each kernel variant
    for kernel in kernels:
        launchers = launchers_json[kernel_name(kernel)]
        for launcher_dict in launchers:
            launcher = NS(**launcher_dict)

            factors = launcher.factors

            if len(launcher.lengths) == 1:
                length = launcher.lengths[0]
            elif len(launcher.lengths) == 2:
                length = (launcher.lengths[0], launcher.lengths[1])
            elif len(launcher.lengths) == 3:
                length = (launcher.lengths[0], launcher.lengths[1],
                          launcher.lengths[2])

            transforms_per_block = launcher.transforms_per_block
            workgroup_size = launcher.workgroup_size
            threads_per_transform = workgroup_size // transforms_per_block
            half_lds = launcher.half_lds
            direct_to_from_reg = launcher.direct_to_from_reg
            scheme = launcher.scheme
            pp_child_scheme = launcher.pp_child_scheme
            pp_factors_curr = launcher.pp_factors_curr
            pp_factors_other = launcher.pp_factors_other
            pp_current_dim = launcher.pp_current_dim
            pp_off_dim = launcher.pp_off_dim
            sbrc_transpose_type = launcher.sbrc_transpose_type
            precision = 'dp' if launcher.double_precision else 'sp'
            runtime_compile = kernel.runtime_compile
            use_3steps_large_twd = getattr(kernel, 'use_3steps_large_twd',
                                           None)

            params = LaunchParams(transforms_per_block, workgroup_size,
                                  threads_per_transform, half_lds,
                                  direct_to_from_reg)

            # make 2D list of threads_per_transform to populate FFTKernel
            tpt_list = kernel.threads_per_transform if scheme == 'CS_KERNEL_2D_SINGLE' else [
                threads_per_transform, 0
            ]

            precisions = [precision]
            if precision == 'sp':
                precisions.append('half')
            for p in precisions:
                f = Function(arguments=ArgumentList(data, back),
                             meta=NS(factors=factors,
                                     length=length,
                                     params=params,
                                     precision=p,
                                     runtime_compile=runtime_compile,
                                     scheme=scheme,
                                     workgroup_size=workgroup_size,
                                     transforms_per_block=transforms_per_block,
                                     threads_per_transform=tpt_list,
                                     transpose=sbrc_transpose_type,
                                     use_3steps_large_twd=use_3steps_large_twd,
                                     lds_size_bytes=kernel.lds_size_bytes,
                                     pp_child_scheme=pp_child_scheme,
                                     pp_factors_curr=pp_factors_curr,
                                     pp_factors_other=pp_factors_other,
                                     pp_current_dim=pp_current_dim,
                                     pp_off_dim=pp_off_dim))

                if (scheme == 'CS_3D_PP'):
                    pp_kernel_functions.append(f)
                else:
                    kernel_functions.append(f)

    return kernel_functions, pp_kernel_functions


def read_subprocess(proc_output, output):
    """To be run in another thread to keep reading in stdout from a subprocess"""
    json_string = ""
    for line in proc_output:
        json_string += line
    output[0] = json_string


def generate_kernels(kernels, precisions, stockham_gen):
    """Generate and write kernels from the kernel list.

    Entries in the kernel list are simple namespaces.  These are
    passed as keyword arguments to the Stockham generator.

    A list of CPU functions is returned.
    """

    pre_enum = {'sp': 0, 'dp': 1}

    # run stockham_gen to retrieve JSON output via stdout, used for additional kernel details
    proc = subprocess.Popen(args=[stockham_gen],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            universal_newlines=True)

    # read from subprocess' stdout through another thread
    json_result = [None]
    reader = threading.Thread(target=read_subprocess,
                              args=(proc.stdout, json_result))
    reader.start()

    kernel_idx = 0
    done_writing = False
    num_kernels = len(kernels)

    while True:
        # Send input if any remaining
        if kernel_idx < num_kernels:
            k = kernels[kernel_idx]

            kernel_precisions = k.precision if hasattr(
                k, 'precision') else precisions

            # 2D single kernels always specify threads per transform
            if isinstance(k.length, list):
                proc.stdin.write(','.join([str(f)
                                           for f in k.factors[0]]) + " ")
                proc.stdin.write(','.join([str(f)
                                           for f in k.factors[1]]) + " ")
                proc.stdin.write(
                    ','.join([str(pre_enum[pre])
                              for pre in kernel_precisions]) + " ")
                proc.stdin.write(','.join(
                    [str(f) for f in k.threads_per_transform]))
            else:
                proc.stdin.write(','.join([str(f) for f in k.factors]) + " ")
                proc.stdin.write(','.join(
                    [str(pre_enum[pre]) for pre in kernel_precisions]))
                # 1D kernels might not, and need to default to 'uwide'
                threads_per_transform = getattr(
                    k, 'threads_per_transform', {
                        'uwide': k.length // min(k.factors),
                        'wide': k.length // max(k.factors),
                        'tall': 0,
                        'consolidated': 0
                    }[getattr(k, 'flavour', 'uwide')])
                proc.stdin.write(f' {str(threads_per_transform)}')

            # default half_lds to True only for CS_KERNEL_STOCKHAM
            half_lds = getattr(k, 'half_lds', k.scheme == 'CS_KERNEL_STOCKHAM')
            # but we don't use LDS for single-radix kernels, so half_lds is meaningless there
            if len(k.factors) == 1:
                half_lds = False

            # Send data over to subprocess

            if isinstance(k.workgroup_size, list):
                proc.stdin.write(" " +
                                 ','.join([str(f) for f in k.workgroup_size]))
            else:
                proc.stdin.write(f' {str(k.workgroup_size)}')

            proc.stdin.write(' 1' if half_lds else ' 0')

            direct_to_from_reg = getattr(k, 'direct_to_from_reg', True)

            if isinstance(direct_to_from_reg, list):
                proc.stdin.write(
                    " " +
                    ','.join(['1' if f else '0' for f in direct_to_from_reg]))
            else:
                # for unspecified direct_to_from_reg, default is True only for CS_KERNEL_STOCKHAM and SBCC
                direct_to_from_reg = getattr(k, 'direct_to_from_reg', True)
                proc.stdin.write(' 1' if direct_to_from_reg else ' 0')

            # check for data specific to partial-pass 3D kernels
            if hasattr(k, 'dims'):
                proc.stdin.write(" " + ','.join([str(f) for f in k.dims]))
                proc.stdin.write(" " +
                                 ','.join([str(f)
                                           for f in k.factors_pp[0]]) + " ")
                proc.stdin.write(','.join([str(f)
                                           for f in k.factors_pp[1]]) + " ")
                proc.stdin.write(','.join([str(f) for f in k.length]))

            proc.stdin.write(f' {k.scheme}')
            proc.stdin.write(f' {kernel_name(k)}')
            proc.stdin.write(f' {k.lds_size_bytes}')
            proc.stdin.write('\n')

            kernel_idx += 1
            proc.stdin.flush()
        elif not done_writing:
            proc.stdin.close()
            done_writing = True

        # Check if subprocess has exited
        proc.poll()
        if proc.returncode != None:
            break

    # Load string from reader thread as json mapping kernel name to list of kernel launchers
    reader.join()
    json_string = json_result[0]
    kernel_launchers = json.loads(json_string)

    return generate_kernel_functions(kernels, precisions, kernel_launchers)


def cli():
    """Command line interface..."""
    parser = argparse.ArgumentParser(prog='kernel-generator')
    subparsers = parser.add_subparsers(dest='command')
    parser.add_argument('--runtime-compile-default',
                        type=str,
                        help='Compile kernels at runtime by default.')
    parser.add_argument(
        '--num-files',
        type=int,
        help='Number of files to generate for parallel compilation.')

    list_parser = subparsers.add_parser(
        'list', help='List names of kernels that will be generated.')

    generate_parser = subparsers.add_parser('generate',
                                            help='Generate kernels.')
    generate_parser.add_argument('stockham_gen',
                                 type=str,
                                 help='Stockham gen executable.')

    args = parser.parse_args()
    if args.num_files:
        assert (args.num_files >
                0), 'Number of files for function_pool should be positive'

    precisions = ['dp', 'sp']

    #
    # kernel list
    #

    kernels = []
    # move 2d out from all, no need to iterate the 2d-kernels for non-2d patterns
    kernels_2d = list_2d_kernels()
    kernel_3d_pp = list_3d_partial_pass_kernels()
    all_kernels = list_small_kernels() + list_large_kernels()

    kernels += all_kernels + kernels_2d + kernel_3d_pp

    # set default lds size (64k) on kernels if not specified
    for k in kernels:
        if not hasattr(k, 'lds_size_bytes'):
            k.lds_size_bytes = 65536

    kernels = unique(kernels)

    #
    # set runtime compile
    #

    kernels = default_runtime_compile(kernels,
                                      args.runtime_compile_default == 'ON')

    #
    # sub commands
    #

    if args.command == 'generate':
        functions, pp_functions = generate_kernels(kernels, precisions,
                                                   args.stockham_gen)
        func_files = generate_cpu_function_pool_pieces(functions, pp_functions,
                                                       args.num_files)
        for i in range(args.num_files):
            write(f'function_pool_init_{i}.cpp', func_files[i], format=False)
        write('function_pool.cpp',
              generate_cpu_function_pool_main(args.num_files),
              format=False)


if __name__ == '__main__':
    cli()
