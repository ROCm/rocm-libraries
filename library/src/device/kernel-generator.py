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
            key = tuple(kernel.length) + (kernel.scheme, )
        else:
            key = (kernel.length, kernel.scheme)
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
        if self.function.meta.runtime_compile or aot_rtc:
            f += 'nullptr'
        else:
            f += str(self.function.address())
        use_3steps_large_twd = getattr(self.function.meta,
                                       'use_3steps_large_twd', None)
        # assume half-precision needs the same thing as single
        precision = 'sp' if self.function.meta.precision == 'half' else self.function.meta.precision
        if use_3steps_large_twd is not None:
            f += ', ' + str(use_3steps_large_twd[precision])
        else:
            f += ', false'
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
            'std::unordered_map<FMKey, FMKey, SimpleHash>& def_key_pool, std::unordered_map<FMKey, FFTKernel, SimpleHash>& function_map'
        )

    call_list = StatementList()
    call_args = ArgumentList('def_key_pool', 'function_map')
    for i in range(num_files):
        call_list += Call(name=f'function_pool_init_{i}', arguments=call_args)
    return StatementList(
        Include('"../include/function_pool.h"'), fwd_declarations,
        Function(name='function_pool::function_pool',
                 value=False,
                 arguments=ArgumentList(),
                 body=call_list))


def generate_cpu_function_pool_pieces(functions, num_files):
    """Generate function(s) to populate the kernel function pool."""
    function_map = Map('function_map')
    precisions = {
        'sp': 'rocfft_precision_single',
        'dp': 'rocfft_precision_double',
        'half': 'rocfft_precision_half',
    }
    var_kernel = Variable('kernel', 'FFTKernel')
    initial_statement = StatementList()
    initial_statement += var_kernel.declaration()

    # Init list to store contents of function_pool_init function per file being generated
    piece_contents = [
        StatementList() + var_kernel.declaration() for _ in range(num_files)
    ]

    # Cycles through each file per loop execution to distribute work amongst N files
    curr_file = 0

    for i, f in enumerate(functions):
        length, precision, scheme, transpose = f.meta.length, f.meta.precision, f.meta.scheme, f.meta.transpose
        if isinstance(length, (int, str)):
            length = [length, 0]
        piece_contents[curr_file] += Assign(var_kernel, FFTKernel(f))
        key = Call(
            name='FMKey',
            arguments=ArgumentList(length[0], length[1], precisions[precision],
                                   scheme, transpose or 'NONE',
                                   'kernel.get_kernel_config()')).inline()
        piece_contents[curr_file] += function_map.assert_insert(
            key, var_kernel, 'def_key_pool', 'function_map')
        curr_file = (curr_file + 1) % num_files

    # Assemble contents of each file to return in a list
    pieces = [None] * num_files
    piece_args = ArgumentList(
        'std::unordered_map<FMKey, FMKey, SimpleHash>& def_key_pool',
        'std::unordered_map<FMKey, FFTKernel, SimpleHash>& function_map')
    for i in range(num_files):
        pieces[i] = StatementList(
            Include('"../include/function_pool.h"'),
            Function(name=f'void function_pool_init_{i}',
                     value=False,
                     arguments=piece_args,
                     body=piece_contents[i]))

    return pieces


def list_generated_kernels(kernels):
    """Return list of kernel names."""
    return [
        kernel_name(x) for x in kernels
        if not x.runtime_compile and not is_aot_rtc(x)
    ]


#
# Main!
#


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

    return f'rocfft_len{length}{postfix}'


# yapf: disable
def list_small_kernels():
    """Return list of small kernels to generate."""

    # Note: Default half_lds is True and default direct_to_from_reg is True as well.
    # TODO- Currently, if half_lds is True, then direct_to_from_reg must be True
    #       but if half_lds is False, direct_to_from_reg can be either (still can be True).
    kernels1d = [
        NS(length=   1, workgroup_size= 64, threads_per_transform=  1, factors=(1,), runtime_compile=True),
        NS(length=   2, workgroup_size= 64, threads_per_transform=  1, factors=(2,), runtime_compile=True),
        NS(length=   3, workgroup_size= 64, threads_per_transform=  1, factors=(3,), runtime_compile=True),
        NS(length=   4, workgroup_size=128, threads_per_transform=  1, factors=(4,), runtime_compile=True),
        NS(length=   5, workgroup_size=128, threads_per_transform=  1, factors=(5,), runtime_compile=True),
        NS(length=   6, workgroup_size=128, threads_per_transform=  1, factors=(6,), runtime_compile=True),
        NS(length=   7, workgroup_size= 64, threads_per_transform=  1, factors=(7,), runtime_compile=True),
        NS(length=   8, workgroup_size= 64, threads_per_transform=  4, factors=(4, 2), runtime_compile=True),
        NS(length=   9, workgroup_size= 64, threads_per_transform=  3, factors=(3, 3), runtime_compile=True),
        NS(length=  10, workgroup_size= 64, threads_per_transform=  1, factors=(10,), runtime_compile=True),
        NS(length=  11, workgroup_size=128, threads_per_transform=  1, factors=(11,), runtime_compile=True),
        NS(length=  12, workgroup_size=128, threads_per_transform=  6, factors=(6, 2), runtime_compile=True),
        NS(length=  13, workgroup_size= 64, threads_per_transform=  1, factors=(13,), runtime_compile=True),
        NS(length=  14, workgroup_size=128, threads_per_transform=  7, factors=(7, 2), runtime_compile=True),
        NS(length=  15, workgroup_size=128, threads_per_transform=  5, factors=(3, 5), runtime_compile=True),
        NS(length=  16, workgroup_size= 64, threads_per_transform=  4, factors=(4, 4), runtime_compile=True),
        NS(length=  17, workgroup_size=256, threads_per_transform=  1, factors=(17,), runtime_compile=True),
        NS(length=  18, workgroup_size= 64, threads_per_transform=  6, factors=(3, 6), runtime_compile=True),
        NS(length=  20, workgroup_size=256, threads_per_transform= 10, factors=(5, 4), runtime_compile=True),
        NS(length=  21, workgroup_size=128, threads_per_transform=  7, factors=(3, 7), runtime_compile=True),
        NS(length=  22, workgroup_size= 64, threads_per_transform=  2, factors=(11, 2), runtime_compile=True),
        NS(length=  24, workgroup_size=256, threads_per_transform=  8, factors=(8, 3), runtime_compile=True),
        NS(length=  25, workgroup_size=256, threads_per_transform=  5, factors=(5, 5), runtime_compile=True),
        NS(length=  26, workgroup_size= 64, threads_per_transform=  2, factors=(13, 2), runtime_compile=True),
        NS(length=  27, workgroup_size=256, threads_per_transform=  9, factors=(3, 3, 3), runtime_compile=True),
        NS(length=  28, workgroup_size= 64, threads_per_transform=  4, factors=(7, 4), runtime_compile=True),
        NS(length=  30, workgroup_size=128, threads_per_transform= 10, factors=(10, 3), runtime_compile=True),
        NS(length=  32, workgroup_size=128, threads_per_transform= 16, factors=(8, 4)),
        NS(length=  33, workgroup_size=256, threads_per_transform= 11, factors=(11, 3), runtime_compile=True),
        NS(length=  34, workgroup_size=256, threads_per_transform= 17, factors=(17, 2), runtime_compile=True),
        NS(length=  35, workgroup_size=256, threads_per_transform=  7, factors=(5, 7), half_lds=False, runtime_compile=True),
        NS(length=  36, workgroup_size= 64, threads_per_transform=  6, factors=(6, 6)),
        NS(length=  39, workgroup_size=256, threads_per_transform= 13, factors=(13, 3), runtime_compile=True),
        NS(length=  40, workgroup_size=128, threads_per_transform= 10, factors=(10, 4)),
        NS(length=  42, workgroup_size=256, threads_per_transform=  7, factors=(7, 6)),
        NS(length=  44, workgroup_size= 64, threads_per_transform=  4, factors=(11, 4)),
        NS(length=  45, workgroup_size=128, threads_per_transform= 15, factors=(5, 3, 3)),
        NS(length=  48, workgroup_size= 64, threads_per_transform= 16, factors=(4, 3, 4)),
        NS(length=  49, workgroup_size= 64, threads_per_transform=  7, factors=(7, 7)),
        NS(length=  50, workgroup_size=256, threads_per_transform= 10, factors=(10, 5)),
        NS(length=  51, workgroup_size=256, threads_per_transform= 17, factors=(17, 3), runtime_compile=True),
        NS(length=  52, workgroup_size= 64, threads_per_transform=  4, factors=(13, 4)),
        NS(length=  54, workgroup_size=256, threads_per_transform= 18, factors=(6, 3, 3)),
        NS(length=  55, workgroup_size=256, threads_per_transform= 11, factors=(5, 11), half_lds=False, runtime_compile=True),
        NS(length=  56, workgroup_size=128, threads_per_transform=  8, factors=(7, 8)),
        NS(length=  60, workgroup_size= 64, threads_per_transform= 10, factors=(6, 10)),
        NS(length=  63, workgroup_size=256, threads_per_transform= 21, factors=(3, 3, 7), half_lds=False, runtime_compile=True),
        NS(length=  64, workgroup_size= 64, threads_per_transform= 16, factors=(4, 4, 4), half_lds=False, direct_to_from_reg=True),
        NS(length=  65, workgroup_size=256, threads_per_transform= 13, factors=(13, 5), runtime_compile=True),
        NS(length=  66, workgroup_size=256, threads_per_transform= 11, factors=(6, 11), half_lds=False, runtime_compile=True),
        NS(length=  68, workgroup_size=256, threads_per_transform= 17, factors=(17, 4), runtime_compile=True),
        NS(length=  70, workgroup_size=256, threads_per_transform= 14, factors=(2, 5, 7), runtime_compile=True),
        NS(length=  72, workgroup_size= 64, threads_per_transform=  9, factors=(8, 3, 3)),
        NS(length=  75, workgroup_size=256, threads_per_transform= 25, factors=(5, 5, 3)),
        NS(length=  77, workgroup_size=256, threads_per_transform= 11, factors=(7, 11), runtime_compile=True),
        NS(length=  78, workgroup_size=256, threads_per_transform= 13, factors=(6, 13), half_lds=False, runtime_compile=True),
        NS(length=  80, workgroup_size= 64, threads_per_transform= 10, factors=(5, 2, 8)),
        NS(length=  81, workgroup_size=128, threads_per_transform= 27, factors=(3, 3, 3, 3)),
        NS(length=  84, workgroup_size=128, threads_per_transform= 12, factors=(7, 2, 6)),
        NS(length=  85, workgroup_size=256, threads_per_transform= 17, factors=(17, 5), runtime_compile=True),
        NS(length=  88, workgroup_size=128, threads_per_transform= 11, factors=(11, 8)),
        NS(length=  90, workgroup_size= 64, threads_per_transform=  9, factors=(3, 3, 10)),
        NS(length=  91, workgroup_size=256, threads_per_transform= 13, factors=(7, 13), half_lds=False, runtime_compile=True),
        NS(length=  96, workgroup_size=128, threads_per_transform= 16, factors=(6, 16), half_lds=False, direct_to_from_reg=False),
        NS(length=  98, workgroup_size= 256, threads_per_transform= 14, factors=(2, 7, 7), half_lds=False, runtime_compile=True),
        NS(length=  99, workgroup_size= 256, threads_per_transform= 11, factors=(3, 3, 11), half_lds=False, runtime_compile=True),
        NS(length= 100, workgroup_size= 64, threads_per_transform= 10, factors=(10, 10)),
        NS(length= 102, workgroup_size=128, threads_per_transform= 17, factors=(17, 6), runtime_compile=True),
        NS(length= 104, workgroup_size= 64, threads_per_transform=  8, factors=(13, 8)),
        NS(length= 105, workgroup_size=256, threads_per_transform= 21, factors=(7, 3, 5), half_lds=False, runtime_compile=True),
        NS(length= 108, workgroup_size=256, threads_per_transform= 36, factors=(6, 6, 3)),
        NS(length= 110, workgroup_size=256, threads_per_transform= 11, factors=(2, 5, 11), half_lds=False, runtime_compile=True),
        NS(length= 112, workgroup_size=256, threads_per_transform= 16, factors=(16, 7), half_lds=False, direct_to_from_reg=False),
        NS(length= 117, workgroup_size= 64, threads_per_transform= 13, factors=(13, 9), runtime_compile=True),
        NS(length= 119, workgroup_size=256, threads_per_transform= 17, factors=(17, 7), runtime_compile=True),
        NS(length= 120, workgroup_size= 64, threads_per_transform= 12, factors=(6, 10, 2), runtime_compile=True),
        NS(length= 121, workgroup_size=128, threads_per_transform= 11, factors=(11, 11), runtime_compile=True),
        NS(length= 125, workgroup_size=256, threads_per_transform= 25, factors=(5, 5, 5), half_lds=False, direct_to_from_reg=False),
        NS(length= 126, workgroup_size= 256, threads_per_transform= 42, factors=(6, 7, 3), half_lds=False, runtime_compile=True),
        NS(length= 128, workgroup_size=256, threads_per_transform= 16, factors=(16, 8)),
        NS(length= 130, workgroup_size= 64, threads_per_transform= 13, factors=(13, 10), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 132, workgroup_size=128, threads_per_transform= 22, factors=(11, 6, 2), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 135, workgroup_size=128, threads_per_transform=  9, factors=(5, 3, 3, 3), runtime_compile=True),
        NS(length= 136, workgroup_size=128, threads_per_transform=17, factors=(17, 8), runtime_compile=True),
        NS(length= 140, workgroup_size= 64, threads_per_transform= 28, factors=(7, 5, 4), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 143, workgroup_size=256, threads_per_transform= 13, factors=(13, 11), half_lds=False, runtime_compile=True),
        NS(length= 144, workgroup_size=128, threads_per_transform= 12, factors=(6, 6, 4)),
    NS(length= 147, workgroup_size= 64, threads_per_transform= 21, factors=(7, 7, 3), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 150, workgroup_size= 64, threads_per_transform=  5, factors=(10, 5, 3), runtime_compile=True),
    NS(length= 153, workgroup_size=128, threads_per_transform= 17, factors=(17, 9), runtime_compile=True),
    NS(length= 154, workgroup_size=128, threads_per_transform= 22, factors=(11, 7, 2), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
    NS(length= 156, workgroup_size= 128, threads_per_transform=13, factors=(3, 4, 13), half_lds=False, runtime_compile=True),
        NS(length= 160, workgroup_size=256, threads_per_transform= 16, factors=(16, 10)),
        NS(length= 162, workgroup_size=256, threads_per_transform= 27, factors=(6, 3, 3, 3), runtime_compile=True),
    NS(length= 165, workgroup_size= 64, threads_per_transform= 11, factors=(11, 5, 3), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 168, workgroup_size=256, threads_per_transform= 56, factors=(8, 7, 3), half_lds=False, direct_to_from_reg=False),
        NS(length= 169, workgroup_size=256, threads_per_transform= 13, factors=(13, 13), runtime_compile=True),
    NS(length= 170, workgroup_size=128, threads_per_transform= 17, factors=(17, 10), runtime_compile=True),
    NS(length= 175, workgroup_size=256, threads_per_transform= 35, factors=(5, 5, 7), half_lds=False, runtime_compile=True),
        NS(length= 176, workgroup_size= 64, threads_per_transform= 16, factors=(11, 16), runtime_compile=True),
        NS(length= 180, workgroup_size=256, threads_per_transform= 60, factors=(10, 6, 3), half_lds=False, direct_to_from_reg=False),
    NS(length= 182, workgroup_size= 64, threads_per_transform= 13, factors=(13, 2, 7), half_lds=False, runtime_compile=True),
    NS(length= 187, workgroup_size=128, threads_per_transform= 17, factors=(17, 11), runtime_compile=True),
    NS(length= 189, workgroup_size= 64, threads_per_transform= 21, factors=(7, 3, 3, 3), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 192, workgroup_size=128, threads_per_transform= 16, factors=(6, 4, 4, 2)),
    NS(length= 195, workgroup_size= 64, threads_per_transform= 13, factors=(13, 5, 3), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
    NS(length= 196, workgroup_size= 64, threads_per_transform= 28, factors=(4, 7, 7), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
    NS(length= 198, workgroup_size=128, threads_per_transform= 22, factors=(11, 2, 9), half_lds=False, runtime_compile=True),
        NS(length= 200, workgroup_size= 64, threads_per_transform= 20, factors=(10, 10, 2)),
    NS(length= 204, workgroup_size=128, threads_per_transform= 17, factors=(17, 4, 3), runtime_compile=True),
        NS(length= 208, workgroup_size= 64, threads_per_transform= 16, factors=(13, 16)),
    NS(length= 210, workgroup_size= 64, threads_per_transform= 30, factors=(10, 7, 3), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 216, workgroup_size=256, threads_per_transform= 36, factors=(6, 6, 6)),
    NS(length= 220, workgroup_size=128, threads_per_transform= 22, factors=(10, 2, 11), half_lds=False, runtime_compile=True),
    NS(length= 221, workgroup_size=128, threads_per_transform= 17, factors=(17, 13), runtime_compile=True),
        NS(length= 224, workgroup_size= 64, threads_per_transform= 16, factors=(7, 2, 2, 2, 2, 2)),
        NS(length= 225, workgroup_size=256, threads_per_transform= 75, factors=(5, 5, 3, 3), runtime_compile=True),
    NS(length= 231, workgroup_size=256, threads_per_transform= 33, factors=(11, 7, 3), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
    NS(length= 234, workgroup_size= 64, threads_per_transform= 26, factors=(13, 9, 2), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
    NS(length= 238, workgroup_size= 64, threads_per_transform= 17, factors=(17, 7, 2), runtime_compile=True),
        NS(length= 240, workgroup_size=128, threads_per_transform= 48, factors=(8, 5, 6)),
    NS(length= 242, workgroup_size=128, threads_per_transform= 22, factors=(11, 2, 11), half_lds=False, runtime_compile=True),
        NS(length= 243, workgroup_size=256, threads_per_transform= 81, factors=(3, 3, 3, 3, 3)),
    NS(length= 245, workgroup_size=256, threads_per_transform= 35, factors=(7, 5, 7), half_lds=False, runtime_compile=True),
        NS(length= 250, workgroup_size=128, threads_per_transform= 25, factors=(10, 5, 5), runtime_compile=True),
    NS(length= 252, workgroup_size= 64, threads_per_transform= 63, factors=(7, 3, 3, 4), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
    NS(length= 255, workgroup_size= 64, threads_per_transform= 17, factors=(17, 5, 3), runtime_compile=True),
        NS(length= 256, workgroup_size= 64, threads_per_transform= 64, factors=(4, 4, 4, 4)),
    NS(length= 260, workgroup_size= 64, threads_per_transform= 26, factors=(13, 10, 2), half_lds=False, runtime_compile=True),
    NS(length= 264, workgroup_size=256, threads_per_transform= 33, factors=(8, 3, 11), half_lds=False, runtime_compile=True),
        NS(length= 270, workgroup_size=128, threads_per_transform= 27, factors=(10, 3, 3, 3)),
        NS(length= 272, workgroup_size=128, threads_per_transform= 17, factors=(16, 17), runtime_compile=True),
    NS(length= 273, workgroup_size= 64, threads_per_transform= 13, factors=(13, 3, 7), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
    NS(length= 275, workgroup_size= 64, threads_per_transform= 55, factors=(11, 5, 5), half_lds=False, runtime_compile=True),
    NS(length= 280, workgroup_size= 64, threads_per_transform= 56, factors=(8, 7, 5), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
    NS(length= 286, workgroup_size= 64, threads_per_transform= 26, factors=(13, 11, 2), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 288, workgroup_size=128, threads_per_transform= 24, factors=(6, 6, 4, 2), runtime_compile=True),
        NS(length= 289, workgroup_size=128, threads_per_transform= 17, factors=(17, 17), runtime_compile=True),
    NS(length= 294, workgroup_size=128, threads_per_transform= 42, factors=(6, 7, 7), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
    NS(length= 297, workgroup_size=256, threads_per_transform= 33, factors=(9, 3, 11), runtime_compile=True),
        NS(length= 300, workgroup_size= 64, threads_per_transform= 30, factors=(10, 10, 3), runtime_compile=True),
    NS(length= 306, workgroup_size=256, threads_per_transform= 34, factors=(17, 2, 9), runtime_compile=True),
    NS(length= 308, workgroup_size= 64, threads_per_transform= 44, factors=(11, 7, 4), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
    NS(length= 312, workgroup_size= 64, threads_per_transform= 26, factors=(13, 4, 3, 2), half_lds=False, runtime_compile=True),
    NS(length= 315, workgroup_size= 64, threads_per_transform= 63, factors=(7, 3, 3, 5), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 320, workgroup_size= 64, threads_per_transform= 16, factors=(10, 4, 4, 2), runtime_compile=True),
        NS(length= 324, workgroup_size= 64, threads_per_transform= 54, factors=(3, 6, 6, 3), runtime_compile=True),
        NS(length= 325, workgroup_size= 64, threads_per_transform= 13, factors=(13, 5, 5), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 330, workgroup_size=128, threads_per_transform= 33, factors=(11, 10, 3), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 336, workgroup_size=128, threads_per_transform= 56, factors=(8, 7, 6)),
        NS(length= 338, workgroup_size= 64, threads_per_transform= 26, factors=(13, 2, 13), runtime_compile=True),
        NS(length= 340, workgroup_size=128, threads_per_transform= 34, factors=(17, 2, 10), runtime_compile=True),
        NS(length= 343, workgroup_size=256, threads_per_transform= 49, factors=(7, 7, 7), runtime_compile=True),
        NS(length= 350, workgroup_size= 64, threads_per_transform= 50, factors=(5, 7, 10), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 351, workgroup_size=128, threads_per_transform= 39, factors=(13, 3, 9), half_lds=False, runtime_compile=True),
        NS(length= 352, workgroup_size= 64, threads_per_transform= 32, factors=(11, 2, 16), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 357, workgroup_size=256, threads_per_transform= 17, factors=(17, 3, 7), runtime_compile=True),
        NS(length= 360, workgroup_size=256, threads_per_transform= 60, factors=(10, 6, 6), runtime_compile=True),
        NS(length= 363, workgroup_size=128, threads_per_transform= 33, factors=(11, 3, 11), runtime_compile=True),
        NS(length= 364, workgroup_size= 64, threads_per_transform= 52, factors=(13, 7, 4), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 374, workgroup_size=256, threads_per_transform= 34, factors=(17, 2, 11), runtime_compile=True),
        NS(length= 375, workgroup_size=128, threads_per_transform= 25, factors=(5, 5, 5, 3), runtime_compile=True),
        NS(length= 378, workgroup_size=128, threads_per_transform=126, factors=(6, 3, 3, 7), half_lds=False, runtime_compile=True),
        NS(length= 384, workgroup_size=128, threads_per_transform= 32, factors=(6, 4, 4, 4), runtime_compile=True),
        NS(length= 385, workgroup_size= 64, threads_per_transform= 55, factors=(11, 7, 5), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 390, workgroup_size=128, threads_per_transform= 39, factors=(13, 3, 10), half_lds=False, runtime_compile=True),
        NS(length= 392, workgroup_size= 64, threads_per_transform= 56, factors=(8, 7, 7), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 396, workgroup_size= 64, threads_per_transform= 44, factors=(11, 9, 4), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 400, workgroup_size=128, threads_per_transform= 40, factors=(4, 10, 10), runtime_compile=True),
        NS(length= 405, workgroup_size=128, threads_per_transform= 27, factors=(5, 3, 3, 3, 3), runtime_compile=True),
        NS(length= 408, workgroup_size= 64, threads_per_transform= 17, factors=(17, 3, 8), runtime_compile=True),
        NS(length= 416, workgroup_size= 64, threads_per_transform= 32, factors=(13, 2, 16), half_lds=False, runtime_compile=True),
        NS(length= 420, workgroup_size= 64, threads_per_transform= 60, factors=(10, 7, 6), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 425, workgroup_size= 64, threads_per_transform= 17, factors=(17, 5, 5), runtime_compile=True),
        NS(length= 429, workgroup_size=128, threads_per_transform= 39, factors=(13, 3, 11), half_lds=False, runtime_compile=True),
        NS(length= 432, workgroup_size= 64, threads_per_transform= 27, factors=(3, 16, 3, 3), runtime_compile=True),
        NS(length= 440, workgroup_size= 64, threads_per_transform= 55, factors=(11, 8, 5), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 441, workgroup_size= 64, threads_per_transform= 63, factors=(9, 7, 7), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 442, workgroup_size=256, threads_per_transform= 34, factors=(17, 2, 13), runtime_compile=True),
        NS(length= 448, workgroup_size=128, threads_per_transform= 64, factors=(8, 7, 8), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 450, workgroup_size=128, threads_per_transform= 30, factors=(10, 5, 3, 3), runtime_compile=True),
        NS(length= 455, workgroup_size=256, threads_per_transform= 65, factors=(13, 5, 7), half_lds=False, runtime_compile=True),
        NS(length= 459, workgroup_size=256, threads_per_transform= 51, factors=(17, 3, 9), runtime_compile=True),
        NS(length= 462, workgroup_size=256, threads_per_transform= 77, factors=(11, 6, 7), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 468, workgroup_size= 64, threads_per_transform= 52, factors=(13, 9, 4), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 476, workgroup_size=128, threads_per_transform= 34, factors=(17, 2, 7, 2), runtime_compile=True),
        NS(length= 480, workgroup_size= 64, threads_per_transform= 16, factors=(10, 8, 6), runtime_compile=True),
        NS(length= 484, workgroup_size= 64, threads_per_transform= 44, factors=(4, 11, 11), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 486, workgroup_size=256, threads_per_transform=162, factors=(6, 3, 3, 3, 3), runtime_compile=True),
        NS(length= 490, workgroup_size=256, threads_per_transform= 70, factors=(10, 7, 7), half_lds=False, runtime_compile=True),
        NS(length= 495, workgroup_size= 64, threads_per_transform= 55, factors=(11, 9, 5), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 500, workgroup_size=128, threads_per_transform=100, factors=(10, 5, 10), runtime_compile=True),
        NS(length= 504, workgroup_size= 64, threads_per_transform= 63, factors=(7, 9, 4, 2), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 507, workgroup_size=128, threads_per_transform= 39, factors=(13, 3, 13), runtime_compile=True),
        NS(length= 510, workgroup_size=256, threads_per_transform= 34, factors=(17, 2, 3, 5), runtime_compile=True),
        NS(length= 512, workgroup_size= 64, threads_per_transform= 64, factors=(8, 8, 8)),
        NS(length= 520, workgroup_size= 64, threads_per_transform= 52, factors=(13, 10, 4), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 525, workgroup_size= 128, threads_per_transform=105, factors=(7, 3, 5, 5), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 528, workgroup_size= 64, threads_per_transform= 48, factors=(4, 4, 3, 11), runtime_compile=True),
        NS(length= 539, workgroup_size=256, threads_per_transform= 77, factors=(11, 7, 7), runtime_compile=True),
        NS(length= 540, workgroup_size=256, threads_per_transform= 54, factors=(3, 10, 6, 3), runtime_compile=True),
        NS(length= 544, workgroup_size=128, threads_per_transform= 34, factors=(17, 2, 16), runtime_compile=True),
        NS(length= 546, workgroup_size=128, threads_per_transform= 39, factors=(13, 3, 7, 2), runtime_compile=True),
        NS(length= 550, workgroup_size= 64, threads_per_transform= 55, factors=(11, 10, 5), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 560, workgroup_size= 64, threads_per_transform= 56, factors=(8, 7, 5, 2), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 561, workgroup_size=256, threads_per_transform= 51, factors=(17, 3, 11), runtime_compile=True),
        NS(length= 567, workgroup_size= 64, threads_per_transform= 63, factors=(7, 9, 3, 3), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 572, workgroup_size= 64, threads_per_transform= 52, factors=(13, 11, 4), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 576, workgroup_size=128, threads_per_transform= 96, factors=(16, 6, 6), runtime_compile=True),
        NS(length= 578, workgroup_size= 256, threads_per_transform=34, factors=(17, 17, 2), runtime_compile=True),
        NS(length= 585, workgroup_size= 256, threads_per_transform=65, factors=(13, 5, 9), half_lds=False, runtime_compile=True),
        NS(length= 588, workgroup_size= 256, threads_per_transform=84, factors=(7, 3, 4, 7), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 594, workgroup_size=128, threads_per_transform= 99, factors=(11, 3, 6, 3), half_lds=False, runtime_compile=True),
        NS(length= 595, workgroup_size= 64, threads_per_transform= 17, factors=(7, 17, 5), runtime_compile=True),
        NS(length= 600, workgroup_size= 64, threads_per_transform= 60, factors=(10, 6, 10), runtime_compile=True),
        NS(length= 605, workgroup_size= 64, threads_per_transform= 55, factors=(11, 5, 11), half_lds=False, runtime_compile=True),
        NS(length= 612, workgroup_size= 64, threads_per_transform= 51, factors=(17, 3, 6, 2), runtime_compile=True),
        NS(length= 616, workgroup_size=128, threads_per_transform= 88, factors=(11, 7, 8), half_lds=False, runtime_compile=True),
        NS(length= 624, workgroup_size= 64, threads_per_transform= 52, factors=(13, 4, 6, 2), half_lds=False, runtime_compile=True),
        NS(length= 625, workgroup_size=128, threads_per_transform=125, factors=(5, 5, 5, 5), runtime_compile=True),
        NS(length= 630, workgroup_size= 64, threads_per_transform= 63, factors=(3, 3, 5, 7, 2), runtime_compile=True),
        NS(length= 637, workgroup_size=128, threads_per_transform= 91, factors=(13, 7, 7), runtime_compile=True),
        NS(length= 640, workgroup_size=128, threads_per_transform= 64, factors=(8, 10, 8), runtime_compile=True),
        NS(length= 648, workgroup_size=256, threads_per_transform=216, factors=(8, 3, 3, 3, 3), runtime_compile=True),
        NS(length= 650, workgroup_size= 256, threads_per_transform=65, factors=(10, 5, 13), half_lds=False, runtime_compile=True),
        NS(length= 660, workgroup_size=128, threads_per_transform=110, factors=(11, 6, 10), runtime_compile=True),
        NS(length= 663, workgroup_size= 64, threads_per_transform= 51, factors=(17, 13, 3), half_lds=False, runtime_compile=True),
        NS(length= 672, workgroup_size= 64, threads_per_transform= 56, factors=(2, 2, 2, 2, 2, 3, 7), runtime_compile=True),
        NS(length= 675, workgroup_size=256, threads_per_transform=225, factors=(5, 5, 3, 3, 3), runtime_compile=True),
        NS(length= 676, workgroup_size= 64, threads_per_transform= 52, factors=(13, 13, 4), half_lds=False, runtime_compile=True),
        NS(length= 680, workgroup_size=256, threads_per_transform= 68, factors=(17, 4, 10), runtime_compile=True),
        NS(length= 686, workgroup_size= 64, threads_per_transform= 49, factors=(7, 7, 7, 2), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 693, workgroup_size=128, threads_per_transform= 99, factors=(11, 7, 9), runtime_compile=True),
        NS(length= 700, workgroup_size= 128, threads_per_transform=100, factors=(10, 7, 10), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 702, workgroup_size= 128, threads_per_transform=117, factors=(13, 3, 6, 3), runtime_compile=True),
        NS(length= 704, workgroup_size=256, threads_per_transform=88, factors=(2, 2, 2, 2, 11, 2, 2), runtime_compile=True),
        NS(length= 714, workgroup_size=64, threads_per_transform=51, factors=(3, 17, 7, 2), runtime_compile=True),
        NS(length= 715, workgroup_size=256, threads_per_transform= 65, factors=(13, 5, 11), runtime_compile=True),
        NS(length= 720, workgroup_size=256, threads_per_transform=120, factors=(10, 3, 8, 3), runtime_compile=True),
        NS(length= 726, workgroup_size=256, threads_per_transform= 66, factors=(11, 6, 11), half_lds=False, runtime_compile=True),
        NS(length= 728, workgroup_size=128, threads_per_transform=104, factors=(13, 7, 8), runtime_compile=True),
        NS(length= 729, workgroup_size=256, threads_per_transform=243, factors=(3, 3, 3, 3, 3, 3), runtime_compile=True),
        NS(length= 735, workgroup_size= 256, threads_per_transform=147, factors=(7, 3, 5, 7), half_lds=False, runtime_compile=True),
        NS(length= 748, workgroup_size= 256, threads_per_transform=68, factors=(17, 4, 11), runtime_compile=True),
        NS(length= 750, workgroup_size=256, threads_per_transform=250, factors=(10, 5, 3, 5), runtime_compile=True),
        NS(length= 756, workgroup_size= 64, threads_per_transform= 63, factors=(2, 2, 3, 3, 3, 7), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 765, workgroup_size=256, threads_per_transform= 51, factors=(17, 3, 5, 3), runtime_compile=True),
        NS(length= 768, workgroup_size= 64, threads_per_transform= 48, factors=(16, 3, 16), runtime_compile=True),
        NS(length= 770, workgroup_size=256, threads_per_transform=110, factors=(11, 10, 7), half_lds=False, runtime_compile=True),
        NS(length= 780, workgroup_size=256, threads_per_transform= 78, factors=(2, 3, 13, 5, 2), runtime_compile=True),
        NS(length= 784, workgroup_size= 64, threads_per_transform= 56, factors=(2, 2, 2, 2, 7, 7), runtime_compile=True),
        NS(length= 792, workgroup_size=256, threads_per_transform= 88, factors=(2, 2, 2, 3, 3, 11), half_lds=False, runtime_compile=True),
        NS(length= 800, workgroup_size=256, threads_per_transform=160, factors=(16, 5, 10), runtime_compile=True),
        NS(length= 810, workgroup_size=128, threads_per_transform= 81, factors=(3, 10, 3, 3, 3), runtime_compile=True),
        NS(length= 816, workgroup_size= 64, threads_per_transform= 51, factors=(17, 2, 3, 2, 2, 2), runtime_compile=True),
        NS(length= 819, workgroup_size=128, threads_per_transform=117, factors=(9, 7, 13), half_lds=False, runtime_compile=True),
        NS(length= 825, workgroup_size= 64, threads_per_transform= 55, factors=(11, 5, 5, 3), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 832, workgroup_size=128, threads_per_transform=104, factors=(13, 2, 2, 2, 2, 2, 2), runtime_compile=True),
        NS(length= 833, workgroup_size=128, threads_per_transform=119, factors=(17, 7, 7), runtime_compile=True),
        NS(length= 840, workgroup_size= 64, threads_per_transform= 56, factors=(2, 2, 2, 3, 5, 7), runtime_compile=True),
        NS(length= 845, workgroup_size= 256, threads_per_transform=65, factors=(13, 5, 13), runtime_compile=True),
        NS(length= 847, workgroup_size= 256, threads_per_transform=77, factors=(11, 7, 11), runtime_compile=True),
        NS(length= 850, workgroup_size= 128, threads_per_transform=85, factors=(10, 5, 17), half_lds=False, runtime_compile=True),
        NS(length= 858, workgroup_size= 256, threads_per_transform=78, factors=(13, 11, 6), runtime_compile=True),
        NS(length= 864, workgroup_size= 64, threads_per_transform= 54, factors=(3, 6, 16, 3), runtime_compile=True),
        NS(length= 867, workgroup_size= 64, threads_per_transform=51, factors=(17, 17, 3), runtime_compile=True),
        NS(length= 875, workgroup_size= 256, threads_per_transform=175, factors=(7, 5, 5, 5), half_lds=False, runtime_compile=True),
        NS(length= 880, workgroup_size=256, threads_per_transform= 88, factors=(2, 2, 2, 2, 11, 5), runtime_compile=True),
        NS(length= 882, workgroup_size= 64, threads_per_transform=63, factors=(9, 7, 7, 2), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 884, workgroup_size= 256, threads_per_transform=68, factors=(13, 4, 17), runtime_compile=True),
        NS(length= 891, workgroup_size= 256, threads_per_transform=99, factors=(9, 11, 3, 3), runtime_compile=True),
        NS(length= 896, workgroup_size=128, threads_per_transform=112, factors=(2, 2, 2, 2, 2, 2, 2, 7), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 900, workgroup_size=256, threads_per_transform= 90, factors=(10, 10, 3, 3), runtime_compile=True),
        NS(length= 910, workgroup_size=256, threads_per_transform= 91, factors=(13, 2, 7, 5), half_lds=False, runtime_compile=True),
        NS(length= 918, workgroup_size=128, threads_per_transform=102, factors=(17, 9, 2, 3), runtime_compile=True),
        NS(length= 924, workgroup_size= 64, threads_per_transform= 44, factors=(2, 2, 3, 7, 11), runtime_compile=True),
        NS(length= 935, workgroup_size= 256, threads_per_transform= 85, factors=(17, 11, 5), runtime_compile=True),
        NS(length= 936, workgroup_size=256, threads_per_transform= 78, factors=(2, 2, 13, 2, 3, 3), runtime_compile=True),
        NS(length= 945, workgroup_size= 64, threads_per_transform= 63, factors=(3, 3, 3, 5, 7), runtime_compile=True),
        NS(length= 952, workgroup_size=256, threads_per_transform= 68, factors=(17, 4, 2, 7), runtime_compile=True),
        NS(length= 960, workgroup_size=256, threads_per_transform=160, factors=(16, 10, 6), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 968, workgroup_size=256, threads_per_transform= 88, factors=(2, 2, 2, 11, 11), half_lds=False, runtime_compile=True),
        NS(length= 972, workgroup_size=256, threads_per_transform=162, factors=(3, 6, 3, 6, 3), runtime_compile=True),
        NS(length= 975, workgroup_size=128, threads_per_transform= 39, factors=(13, 5, 3, 5), runtime_compile=True),
        NS(length= 980, workgroup_size= 256, threads_per_transform=196, factors=(7, 5, 7, 4), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length= 990, workgroup_size=128, threads_per_transform=110, factors=(2, 3, 3, 5, 11), half_lds=False, runtime_compile=True),
        NS(length=1000, workgroup_size=128, threads_per_transform=100, factors=(10, 10, 10), runtime_compile=True),
        NS(length=1001, workgroup_size=256, threads_per_transform= 91, factors=(13, 7, 11), runtime_compile=True),
        NS(length=1008, workgroup_size= 64, threads_per_transform= 56, factors=(2, 2, 2, 2, 3, 3, 7), runtime_compile=True),
        NS(length=1014, workgroup_size=256, threads_per_transform= 78, factors=(13, 6, 13), half_lds=False, runtime_compile=True),
        NS(length=1020, workgroup_size=256, threads_per_transform= 68, factors=(2, 17, 2, 3, 5), runtime_compile=True),
        NS(length=1024, workgroup_size=128, threads_per_transform=128, factors=(8, 8, 4, 4)),
        NS(length=1040, workgroup_size=256, threads_per_transform=208, factors=(13, 16, 5), runtime_compile=True),
        NS(length=1050, workgroup_size=256, threads_per_transform=210, factors=(2, 3, 5, 5, 7), half_lds=False, runtime_compile=True),
        NS(length=1053, workgroup_size=128, threads_per_transform=117, factors=(3, 3, 13, 3, 3), runtime_compile=True),
        NS(length=1056, workgroup_size=256, threads_per_transform=176, factors=(2, 2, 2, 2, 11, 6), runtime_compile=True),
        NS(length=1071, workgroup_size=128, threads_per_transform=119, factors=(17, 7, 9), runtime_compile=True),
        NS(length=1078, workgroup_size=256, threads_per_transform= 77, factors=(2, 11, 7, 7), runtime_compile=True),
        NS(length=1080, workgroup_size=256, threads_per_transform=108, factors=(6, 10, 6, 3), runtime_compile=True),
        NS(length=1088, workgroup_size=256, threads_per_transform= 68, factors=(17, 4, 4, 2, 2), runtime_compile=True),
        NS(length=1089, workgroup_size=128, threads_per_transform=121, factors=(3, 11, 3, 11), half_lds=False, runtime_compile=True),
        NS(length=1092, workgroup_size= 64, threads_per_transform= 52, factors=(2, 2, 13, 7, 3), runtime_compile=True),
        NS(length=1100, workgroup_size=128, threads_per_transform=110, factors=(2, 2, 11, 5, 5), half_lds=False, runtime_compile=True),
        NS(length=1105, workgroup_size=256, threads_per_transform= 85, factors=(17, 13, 5), runtime_compile=True),
        NS(length=1120, workgroup_size=256, threads_per_transform=224, factors=(2, 2, 2, 2, 2, 5, 7), runtime_compile=True),
        NS(length=1122, workgroup_size=256, threads_per_transform=102, factors=(17, 11, 6), runtime_compile=True),
        NS(length=1125, workgroup_size=256, threads_per_transform=225, factors=(5, 5, 3, 3, 5), runtime_compile=True),
        NS(length=1134, workgroup_size=128, threads_per_transform=126, factors=(2, 3, 3, 3, 3, 7), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length=1144, workgroup_size=128, threads_per_transform=104, factors=(13, 11, 8), half_lds=False, direct_to_from_reg=False, runtime_compile=True),
        NS(length=1152, workgroup_size=256, threads_per_transform=144, factors=(4, 3, 8, 3, 4), runtime_compile=True),
        NS(length=1155, workgroup_size= 64, threads_per_transform= 55, factors=(11, 5, 7, 3), runtime_compile=True),
        NS(length=1156, workgroup_size=256, threads_per_transform= 68, factors=(17, 2, 17, 2), runtime_compile=True),
        NS(length=1170, workgroup_size=256, threads_per_transform=117, factors=(2, 13, 3, 5, 3), half_lds=False, runtime_compile=True),
        NS(length=1176, workgroup_size= 64, threads_per_transform= 56, factors=(2, 2, 2, 3, 7, 7), runtime_compile=True),
        NS(length=1183, workgroup_size=256, threads_per_transform= 91, factors=(7, 13, 13), runtime_compile=True),
        NS(length=1188, workgroup_size=256, threads_per_transform= 66, factors=(6, 11, 2, 3, 3), runtime_compile=True),
        NS(length=1190, workgroup_size=256, threads_per_transform= 85, factors=(17, 2, 5, 7), runtime_compile=True),
        NS(length=1200, workgroup_size=256, threads_per_transform= 75, factors=(5, 5, 16, 3), runtime_compile=True),
        NS(length=1210, workgroup_size=128, threads_per_transform=110, factors=(2, 5, 11, 11), runtime_compile=True),
        NS(length=1215, workgroup_size=256, threads_per_transform=243, factors=(5, 3, 3, 3, 3, 3), runtime_compile=True),
        NS(length=1224, workgroup_size=256, threads_per_transform=102, factors=(17, 3, 4, 6), runtime_compile=True),
        NS(length=1225, workgroup_size=256, threads_per_transform=175, factors=(5, 5, 7, 7), runtime_compile=True),
        NS(length=1232, workgroup_size=256, threads_per_transform=176, factors=(2, 2, 2, 2, 11, 7), runtime_compile=True),
        NS(length=1248, workgroup_size= 64, threads_per_transform= 52, factors=(2, 2, 13, 2, 3, 2, 2), runtime_compile=True),
        NS(length=1250, workgroup_size=256, threads_per_transform=250, factors=(5, 10, 5, 5), runtime_compile=True),
        NS(length=1260, workgroup_size= 64, threads_per_transform= 63, factors=(2, 2, 3, 3, 5, 7), runtime_compile=True),
        NS(length=1274, workgroup_size=256, threads_per_transform=182, factors=(2, 13, 7, 7), runtime_compile=True),
        NS(length=1275, workgroup_size=256, threads_per_transform= 85, factors=(17, 3, 5, 5), runtime_compile=True),
        NS(length=1280, workgroup_size=128, threads_per_transform= 80, factors=(16, 5, 16), runtime_compile=True),
        NS(length=1287, workgroup_size=128, threads_per_transform=117, factors=(3, 13, 3, 11), half_lds=False, runtime_compile=True),
        NS(length=1296, workgroup_size=128, threads_per_transform=108, factors=(6, 6, 6, 6), runtime_compile=True),
        NS(length=1300, workgroup_size=256, threads_per_transform=130, factors=(10, 10, 13), half_lds=False, runtime_compile=True),
        NS(length=1309, workgroup_size=128, threads_per_transform=119, factors=(17, 7, 11), runtime_compile=True),
        NS(length=1320, workgroup_size=256, threads_per_transform=165, factors=(11, 2, 3, 5, 4), half_lds=False, runtime_compile=True),
        NS(length=1323, workgroup_size=256, threads_per_transform=189, factors=(3, 3, 3, 7, 7), half_lds=False, runtime_compile=True),
        NS(length=1326, workgroup_size=256, threads_per_transform=102, factors=(17, 6, 13), runtime_compile=True),
        NS(length=1331, workgroup_size=256, threads_per_transform=121, factors=(11, 11, 11), runtime_compile=True),
        NS(length=1344, workgroup_size=256, threads_per_transform=224, factors=(2, 2, 2, 2, 2, 2, 3, 7), runtime_compile=True),
        NS(length=1350, workgroup_size=256, threads_per_transform=135, factors=(5, 10, 3, 3, 3), runtime_compile=True),
        NS(length=1352, workgroup_size= 64, threads_per_transform= 52, factors=(2, 13, 13, 4), runtime_compile=True),
        NS(length=1360, workgroup_size=256, threads_per_transform= 85, factors=(17, 5, 16), runtime_compile=True),
        NS(length=1365, workgroup_size=256, threads_per_transform= 91, factors=(13, 7, 5, 3), runtime_compile=True),
        NS(length=1372, workgroup_size=256, threads_per_transform= 98, factors=(2, 2, 7, 7, 7), runtime_compile=True),
        NS(length=1375, workgroup_size= 64, threads_per_transform= 55, factors=(11, 5, 5, 5), runtime_compile=True),
        NS(length=1377, workgroup_size= 64, threads_per_transform= 51, factors=(17, 3, 9, 3), runtime_compile=True),
        NS(length=1386, workgroup_size=256, threads_per_transform=231, factors=(2, 7, 3, 11, 3), runtime_compile=True),
        NS(length=1400, workgroup_size= 64, threads_per_transform= 56, factors=(2, 2, 2, 5, 7, 5), runtime_compile=True),
        NS(length=1404, workgroup_size=128, threads_per_transform=117, factors=(2, 2, 3, 13, 3, 3), runtime_compile=True),
        NS(length=1408, workgroup_size=256, threads_per_transform=176, factors=(2, 2, 2, 2, 2, 2, 11, 2), runtime_compile=True),
        NS(length=1428, workgroup_size=128, threads_per_transform=119, factors=(17, 2, 7, 6), runtime_compile=True),
        NS(length=1430, workgroup_size=256, threads_per_transform=143, factors=(13, 11, 10), half_lds=False, runtime_compile=True),
        NS(length=1440, workgroup_size=128, threads_per_transform= 90, factors=(10, 16, 3, 3), runtime_compile=True),
        NS(length=1445, workgroup_size=128, threads_per_transform= 85, factors=(17, 5, 17), runtime_compile=True),
        NS(length=1452, workgroup_size=256, threads_per_transform=132, factors=(11, 3, 11, 4), runtime_compile=True),
        NS(length=1456, workgroup_size=256, threads_per_transform=182, factors=(13, 4, 7, 2, 2), runtime_compile=True),
        NS(length=1458, workgroup_size=256, threads_per_transform=243, factors=(6, 3, 3, 3, 3, 3), runtime_compile=True),
        NS(length=1470, workgroup_size=256, threads_per_transform=210, factors=(2, 3, 5, 7, 7), runtime_compile=True),
        NS(length=1485, workgroup_size=256, threads_per_transform=165, factors=(3, 5, 11, 3, 3), half_lds=False, runtime_compile=True),
        NS(length=1496, workgroup_size=256, threads_per_transform=187, factors=(17, 8, 11), runtime_compile=True),
        NS(length=1500, workgroup_size=256, threads_per_transform=150, factors=(5, 10, 10, 3), runtime_compile=True),
        NS(length=1512, workgroup_size= 64, threads_per_transform= 63, factors=(2, 2, 2, 3, 3, 3, 7), runtime_compile=True),
        NS(length=1521, workgroup_size=128, threads_per_transform=117, factors=(13, 3, 3, 13), runtime_compile=True),
        NS(length=1530, workgroup_size=128, threads_per_transform=102, factors=(17, 3, 6, 5), runtime_compile=True),
        NS(length=1536, workgroup_size=256, threads_per_transform=256, factors=(16, 16, 6), runtime_compile=True),
        NS(length=1540, workgroup_size=256, threads_per_transform=154, factors=(11, 2, 7, 5, 2), runtime_compile=True),
        NS(length=1547, workgroup_size=128, threads_per_transform=119, factors=(17, 7, 13), runtime_compile=True),
        NS(length=1560, workgroup_size=256, threads_per_transform=156, factors=(13, 2, 2, 10, 3), half_lds=False, runtime_compile=True),
        NS(length=1568, workgroup_size=256, threads_per_transform=224, factors=(2, 2, 2, 2, 2, 7, 7), runtime_compile=True),
        NS(length=1573, workgroup_size=256, threads_per_transform=143, factors=(13, 11, 11), half_lds=False, runtime_compile=True),
        NS(length=1575, workgroup_size= 64, threads_per_transform= 63, factors=(3, 3, 5, 7, 5), runtime_compile=True),
        NS(length=1584, workgroup_size=256, threads_per_transform=176, factors=(4, 2, 2, 11, 3, 3), runtime_compile=True),
        NS(length=1600, workgroup_size=256, threads_per_transform=100, factors=(10, 16, 10), runtime_compile=True),
        NS(length=1617, workgroup_size=256, threads_per_transform=231, factors=(3, 7, 7, 11), half_lds=False, runtime_compile=True),
        NS(length=1620, workgroup_size=256, threads_per_transform=162, factors=(10, 3, 3, 6, 3), runtime_compile=True),
        NS(length=1625, workgroup_size=256, threads_per_transform= 65, factors=(13, 5, 5, 5), runtime_compile=True),
        NS(length=1632, workgroup_size=128, threads_per_transform=102, factors=(17, 2, 2, 3, 8), runtime_compile=True),
        NS(length=1638, workgroup_size=256, threads_per_transform=182, factors=(13, 2, 3, 7, 3), runtime_compile=True),
        NS(length=1650, workgroup_size=128, threads_per_transform=110, factors=(11, 2, 3, 5, 5), runtime_compile=True),
        NS(length=1664, workgroup_size=256, threads_per_transform=208, factors=(13, 2, 2, 4, 2, 2, 2), runtime_compile=True),
        NS(length=1666, workgroup_size=128, threads_per_transform=119, factors=(17, 2, 7, 7), runtime_compile=True),
        NS(length=1680, workgroup_size=128, threads_per_transform=112, factors=(2, 2, 2, 2, 3, 7, 5), runtime_compile=True),
        NS(length=1683, workgroup_size= 64, threads_per_transform= 51, factors=(17, 3, 11, 3), runtime_compile=True),
        NS(length=1690, workgroup_size=256, threads_per_transform=169, factors=(13, 10, 13), half_lds=False, runtime_compile=True),
        NS(length=1694, workgroup_size=256, threads_per_transform=154, factors=(11, 2, 11, 7), runtime_compile=True),
        NS(length=1700, workgroup_size=256, threads_per_transform=170, factors=(17, 10, 10), runtime_compile=True),
        NS(length=1701, workgroup_size= 64, threads_per_transform= 63, factors=(3, 3, 3, 3, 3, 7), runtime_compile=True),
        NS(length=1715, workgroup_size=256, threads_per_transform=245, factors=(5, 7, 7, 7), runtime_compile=True),
        NS(length=1716, workgroup_size=256, threads_per_transform=156, factors=(13, 2, 6, 11), half_lds=False, runtime_compile=True),
        NS(length=1728, workgroup_size=128, threads_per_transform=108, factors=(3, 6, 6, 16), runtime_compile=True),
        NS(length=1734, workgroup_size=128, threads_per_transform=102, factors=(17, 17, 6), runtime_compile=True),
        NS(length=1750, workgroup_size=256, threads_per_transform=175, factors=(2, 5, 5, 7, 5), runtime_compile=True),
        NS(length=1755, workgroup_size=128, threads_per_transform=117, factors=(13, 3, 3, 3, 5), runtime_compile=True),
        NS(length=1760, workgroup_size=256, threads_per_transform=176, factors=(2, 2, 2, 2, 2, 11, 5), runtime_compile=True),
        NS(length=1764, workgroup_size=128, threads_per_transform=126, factors=(2, 2, 3, 3, 7, 7), runtime_compile=True),
        NS(length=1768, workgroup_size=256, threads_per_transform=136, factors=(17, 13, 8), runtime_compile=True),
        NS(length=1782, workgroup_size=128, threads_per_transform= 99, factors=(11, 3, 3, 3, 3, 2), runtime_compile=True),
        NS(length=1785, workgroup_size=128, threads_per_transform=119, factors=(17, 3, 5, 7), runtime_compile=True),
        NS(length=1792, workgroup_size=256, threads_per_transform=224, factors=(4, 4, 4, 4, 7), runtime_compile=True),
        NS(length=1800, workgroup_size=256, threads_per_transform=180, factors=(10, 6, 10, 3), runtime_compile=True),
        NS(length=1815, workgroup_size=256, threads_per_transform=165, factors=(11, 3, 5, 11), half_lds=False, runtime_compile=True),
        NS(length=1820, workgroup_size=256, threads_per_transform=182, factors=(10, 13, 7, 2), runtime_compile=True),
        NS(length=1836, workgroup_size=256, threads_per_transform=153, factors=(17, 3, 3, 2, 6), runtime_compile=True),
        NS(length=1848, workgroup_size=256, threads_per_transform=231, factors=(3, 11, 7, 4, 2), runtime_compile=True),
        NS(length=1859, workgroup_size=256, threads_per_transform=169, factors=(13, 11, 13), runtime_compile=True),
        NS(length=1870, workgroup_size=256, threads_per_transform=187, factors=(17, 10, 11), runtime_compile=True),
        NS(length=1872, workgroup_size=256, threads_per_transform=156, factors=(13, 3, 4, 6, 2), runtime_compile=True),
        NS(length=1875, workgroup_size=256, threads_per_transform=125, factors=(5, 5, 5, 5, 3), runtime_compile=True),
        NS(length=1890, workgroup_size=128, threads_per_transform=126, factors=(2, 3, 3, 3, 7, 5), runtime_compile=True),
        NS(length=1904, workgroup_size=128, threads_per_transform=119, factors=(17, 2, 2, 7, 4), runtime_compile=True),
        NS(length=1911, workgroup_size=128, threads_per_transform= 91, factors=(13, 7, 7, 3), runtime_compile=True),
        NS(length=1920, workgroup_size=256, threads_per_transform=120, factors=(10, 6, 16, 2), runtime_compile=True),
        NS(length=1925, workgroup_size= 64, threads_per_transform= 55, factors=(7, 11, 5, 5), runtime_compile=True),
        NS(length=1936, workgroup_size=256, threads_per_transform=176, factors=(2, 2, 4, 11, 11), half_lds=False, runtime_compile=True),
        NS(length=1944, workgroup_size=256, threads_per_transform=243, factors=(3, 3, 3, 3, 8, 3), runtime_compile=True),
        NS(length=1950, workgroup_size=256, threads_per_transform=195, factors=(13, 5, 10, 3), half_lds=False, runtime_compile=True),
        NS(length=1960, workgroup_size= 64, threads_per_transform= 56, factors=(4, 7, 2, 7, 5), runtime_compile=True),
        NS(length=1980, workgroup_size=256, threads_per_transform=198, factors=(11, 2, 3, 3, 5, 2), runtime_compile=True),
        NS(length=1989, workgroup_size=256, threads_per_transform=153, factors=(17, 13, 9), runtime_compile=True),
        NS(length=2000, workgroup_size=128, threads_per_transform=125, factors=(5, 5, 5, 16), runtime_compile=True),
        NS(length=2002, workgroup_size=256, threads_per_transform=182, factors=(2, 13, 7, 11), runtime_compile=True),
        NS(length=2016, workgroup_size=256, threads_per_transform=112, factors=(2, 2, 2, 2, 2, 3, 3, 7), runtime_compile=True),
        NS(length=2023, workgroup_size=128, threads_per_transform=119, factors=(17, 7, 17), runtime_compile=True),
        NS(length=2025, workgroup_size=256, threads_per_transform=135, factors=(3, 3, 5, 5, 3, 3), runtime_compile=True),
        NS(length=2028, workgroup_size=256, threads_per_transform=156, factors=(13, 4, 3, 13), half_lds=False, runtime_compile=True),
        NS(length=2040, workgroup_size=256, threads_per_transform=170, factors=(17, 4, 3, 10), runtime_compile=True),
        NS(length=2048, workgroup_size=256, threads_per_transform=256, factors=(16, 16, 8), runtime_compile=True),
        NS(length=2160, workgroup_size=256, threads_per_transform= 60, factors=(10, 6, 6, 6), runtime_compile=True),
        NS(length=2187, workgroup_size=256, threads_per_transform=243, factors=(3, 3, 3, 3, 3, 3, 3), runtime_compile=True),
        NS(length=2197, workgroup_size=256, threads_per_transform=169, factors=(13, 13, 13), runtime_compile=True),
        NS(length=2250, workgroup_size=256, threads_per_transform= 90, factors=(10, 3, 5, 3, 5), runtime_compile=True),
        NS(length=2304, workgroup_size=256, threads_per_transform=192, factors=(6, 6, 4, 4, 4), runtime_compile=True),
        NS(length=2400, workgroup_size=256, threads_per_transform=240, factors=(4, 10, 10, 6), runtime_compile=True),
        NS(length=2401, workgroup_size=256, threads_per_transform= 49, factors=(7, 7, 7, 7), runtime_compile=True),
        NS(length=2430, workgroup_size=256, threads_per_transform= 81, factors=(10, 3, 3, 3, 3, 3), runtime_compile=True),
        NS(length=2500, workgroup_size=256, threads_per_transform=250, factors=(10, 5, 10, 5), runtime_compile=True),
        NS(length=2560, workgroup_size=128, threads_per_transform=128, factors=(4, 4, 4, 10, 4), runtime_compile=True),
        NS(length=2592, workgroup_size=256, threads_per_transform=216, factors=(6, 6, 6, 6, 2), runtime_compile=True),
        NS(length=2700, workgroup_size=128, threads_per_transform= 90, factors=(3, 10, 10, 3, 3), runtime_compile=True),
        NS(length=2880, workgroup_size=256, threads_per_transform= 96, factors=(10, 6, 6, 2, 2, 2), runtime_compile=True),
        NS(length=2916, workgroup_size=256, threads_per_transform=243, factors=(6, 6, 3, 3, 3, 3), runtime_compile=True),
        NS(length=3000, workgroup_size=128, threads_per_transform=100, factors=(10, 3, 10, 10), runtime_compile=True),
        NS(length=3072, workgroup_size=256, threads_per_transform=256, factors=(6, 4, 4, 4, 4, 2), runtime_compile=True),
        NS(length=3125, workgroup_size=128, threads_per_transform=125, factors=(5, 5, 5, 5, 5), runtime_compile=True),
        NS(length=3200, workgroup_size=256, threads_per_transform=160, factors=(10, 10, 4, 4, 2), runtime_compile=True),
        NS(length=3240, workgroup_size=128, threads_per_transform=108, factors=(3, 3, 10, 6, 6), runtime_compile=True),
        NS(length=3375, workgroup_size=256, threads_per_transform=225, factors=(5, 5, 5, 3, 3, 3), runtime_compile=True),
        NS(length=3456, workgroup_size=256, threads_per_transform=144, factors=(6, 6, 6, 4, 4), runtime_compile=True),
        NS(length=3600, workgroup_size=256, threads_per_transform=120, factors=(10, 10, 6, 6), runtime_compile=True),
        NS(length=3645, workgroup_size=256, threads_per_transform=243, factors=(5, 3, 3, 3, 3, 3, 3), runtime_compile=True),
        NS(length=3750, workgroup_size=256, threads_per_transform=125, factors=(3, 5, 5, 10, 5), runtime_compile=True),
        NS(length=3840, workgroup_size=256, threads_per_transform=128, factors=(10, 6, 2, 2, 2, 2, 2, 2), runtime_compile=True),
        NS(length=3888, workgroup_size=512, threads_per_transform=324, factors=(16, 3, 3, 3, 3, 3), runtime_compile=True),
        NS(length=4000, workgroup_size=256, threads_per_transform=200, factors=(10, 10, 10, 4), runtime_compile=True),
        NS(length=4050, workgroup_size=256, threads_per_transform=135, factors=(10, 5, 3, 3, 3, 3), runtime_compile=True),
        NS(length=4096, workgroup_size=256, threads_per_transform=256, factors=(16, 16, 16), runtime_compile=True),
    ]

    kernels = [NS(**kernel.__dict__,
                  scheme='CS_KERNEL_STOCKHAM',
                  precision=['sp', 'dp']) for kernel in kernels1d]

    return kernels


def list_2d_kernels():
    """Return list of fused 2D kernels to generate."""

    fused_kernels = [
        NS(length=[4,4], factors=[[2,2],[2,2]], threads_per_transform=[2,2], workgroup_size=8),
        NS(length=[4,8], factors=[[2,2],[4,2]], threads_per_transform=[2,2], workgroup_size=16),
        NS(length=[4,9], factors=[[2,2],[3,3]], threads_per_transform=[2,3], workgroup_size=18),
        NS(length=[4,16], factors=[[2,2],[4,4]], threads_per_transform=[2,4], workgroup_size=32),
        NS(length=[4,25], factors=[[2,2],[5,5]], threads_per_transform=[2,5], workgroup_size=50),
        NS(length=[4,27], factors=[[2,2],[3,3,3]], threads_per_transform=[2,9], workgroup_size=54),
        NS(length=[4,32], factors=[[2,2],[8,4]], threads_per_transform=[2,4], workgroup_size=64),
        NS(length=[4,64], factors=[[2,2],[4,4,4]], threads_per_transform=[2,16], workgroup_size=128),
        NS(length=[4,81], factors=[[2,2],[3,3,3,3]], threads_per_transform=[2,27], workgroup_size=162),
        NS(length=[4,125], factors=[[2,2],[5,5,5]], threads_per_transform=[2,25], workgroup_size=250),
        NS(length=[4,128], factors=[[2,2],[8,4,4]], threads_per_transform=[2,16], workgroup_size=256),
        NS(length=[4,243], factors=[[2,2],[3,3,3,3,3]], threads_per_transform=[2,81], workgroup_size=486),
        NS(length=[4,256], factors=[[2,2],[4,4,4,4]], threads_per_transform=[2,64], workgroup_size=512),
        NS(length=[8,4], factors=[[4,2],[2,2]], threads_per_transform=[2,2], workgroup_size=16),
        NS(length=[8,8], factors=[[4,2],[4,2]], threads_per_transform=[2,2], workgroup_size=16),
        NS(length=[8,9], factors=[[4,2],[3,3]], threads_per_transform=[2,3], workgroup_size=24),
        NS(length=[8,16], factors=[[4,2],[4,4]], threads_per_transform=[2,4], workgroup_size=32),
        NS(length=[8,25], factors=[[4,2],[5,5]], threads_per_transform=[2,5], workgroup_size=50),
        NS(length=[8,27], factors=[[4,2],[3,3,3]], threads_per_transform=[2,9], workgroup_size=72),
        NS(length=[8,32], factors=[[4,2],[8,4]], threads_per_transform=[2,4], workgroup_size=64),
        NS(length=[8,64], factors=[[4,2],[4,4,4]], threads_per_transform=[2,16], workgroup_size=128),
        NS(length=[8,81], factors=[[4,2],[3,3,3,3]], threads_per_transform=[2,27], workgroup_size=216),
        NS(length=[8,125], factors=[[4,2],[5,5,5]], threads_per_transform=[2,25], workgroup_size=250),
        NS(length=[8,128], factors=[[4,2],[8,4,4]], threads_per_transform=[2,16], workgroup_size=256),
        NS(length=[8,243], factors=[[4,2],[3,3,3,3,3]], threads_per_transform=[2,81], workgroup_size=648),
        NS(length=[8,256], factors=[[4,2],[4,4,4,4]], threads_per_transform=[2,64], workgroup_size=512),
        NS(length=[9,4], factors=[[3,3],[2,2]], threads_per_transform=[3,2], workgroup_size=18),
        NS(length=[9,8], factors=[[3,3],[4,2]], threads_per_transform=[3,2], workgroup_size=24),
        NS(length=[9,9], factors=[[3,3],[3,3]], threads_per_transform=[3,3], workgroup_size=27),
        NS(length=[9,16], factors=[[3,3],[4,4]], threads_per_transform=[3,4], workgroup_size=48),
        NS(length=[9,25], factors=[[3,3],[5,5]], threads_per_transform=[3,5], workgroup_size=75),
        NS(length=[9,27], factors=[[3,3],[3,3,3]], threads_per_transform=[3,9], workgroup_size=81),
        NS(length=[9,32], factors=[[3,3],[8,4]], threads_per_transform=[3,4], workgroup_size=96),
        NS(length=[9,64], factors=[[3,3],[4,4,4]], threads_per_transform=[3,16], workgroup_size=192),
        NS(length=[9,81], factors=[[3,3],[3,3,3,3]], threads_per_transform=[3,27], workgroup_size=243),
        NS(length=[9,125], factors=[[3,3],[5,5,5]], threads_per_transform=[3,25], workgroup_size=375),
        NS(length=[9,128], factors=[[3,3],[8,4,4]], threads_per_transform=[3,16], workgroup_size=384),
        NS(length=[9,243], factors=[[3,3],[3,3,3,3,3]], threads_per_transform=[3,81], workgroup_size=729),
        NS(length=[9,256], factors=[[3,3],[4,4,4,4]], threads_per_transform=[3,64], workgroup_size=768),
        NS(length=[16,4], factors=[[4,4],[2,2]], threads_per_transform=[4,2], workgroup_size=32),
        NS(length=[16,8], factors=[[4,4],[4,2]], threads_per_transform=[4,2], workgroup_size=32),
        NS(length=[16,9], factors=[[4,4],[3,3]], threads_per_transform=[4,3], workgroup_size=48),
        NS(length=[16,16], factors=[[4,4],[4,4]], threads_per_transform=[4,4], workgroup_size=64),
        NS(length=[16,25], factors=[[4,4],[5,5]], threads_per_transform=[4,5], workgroup_size=100),
        NS(length=[16,27], factors=[[4,4],[3,3,3]], threads_per_transform=[4,9], workgroup_size=144),
        NS(length=[16,32], factors=[[4,4],[8,4]], threads_per_transform=[4,4], workgroup_size=128),
        NS(length=[16,64], factors=[[4,4],[4,4,4]], threads_per_transform=[4,16], workgroup_size=256),
        NS(length=[16,81], factors=[[4,4],[3,3,3,3]], threads_per_transform=[4,27], workgroup_size=432),
        NS(length=[16,125], factors=[[4,4],[5,5,5]], threads_per_transform=[4,25], workgroup_size=500),
        NS(length=[16,128], factors=[[4,4],[8,4,4]], threads_per_transform=[4,16], workgroup_size=512),
        NS(length=[25,4], factors=[[5,5],[2,2]], threads_per_transform=[5,2], workgroup_size=50),
        NS(length=[25,8], factors=[[5,5],[4,2]], threads_per_transform=[5,2], workgroup_size=50),
        NS(length=[25,9], factors=[[5,5],[3,3]], threads_per_transform=[5,3], workgroup_size=75),
        NS(length=[25,16], factors=[[5,5],[4,4]], threads_per_transform=[5,4], workgroup_size=100),
        NS(length=[25,25], factors=[[5,5],[5,5]], threads_per_transform=[5,5], workgroup_size=125),
        NS(length=[25,27], factors=[[5,5],[3,3,3]], threads_per_transform=[5,9], workgroup_size=225),
        NS(length=[25,32], factors=[[5,5],[8,4]], threads_per_transform=[5,4], workgroup_size=160),
        NS(length=[25,64], factors=[[5,5],[4,4,4]], threads_per_transform=[5,16], workgroup_size=400),
        NS(length=[25,81], factors=[[5,5],[3,3,3,3]], threads_per_transform=[5,27], workgroup_size=675),
        NS(length=[25,125], factors=[[5,5],[5,5,5]], threads_per_transform=[5,25], workgroup_size=625),
        NS(length=[25,128], factors=[[5,5],[8,4,4]], threads_per_transform=[5,16], workgroup_size=640),
        NS(length=[27,4], factors=[[3,3,3],[2,2]], threads_per_transform=[9,2], workgroup_size=54),
        NS(length=[27,8], factors=[[3,3,3],[4,2]], threads_per_transform=[9,2], workgroup_size=72),
        NS(length=[27,9], factors=[[3,3,3],[3,3]], threads_per_transform=[9,3], workgroup_size=81),
        NS(length=[27,16], factors=[[3,3,3],[4,4]], threads_per_transform=[9,4], workgroup_size=144),
        NS(length=[27,25], factors=[[3,3,3],[5,5]], threads_per_transform=[9,5], workgroup_size=225),
        NS(length=[27,27], factors=[[3,3,3],[3,3,3]], threads_per_transform=[9,9], workgroup_size=243),
        NS(length=[27,32], factors=[[3,3,3],[8,4]], threads_per_transform=[9,4], workgroup_size=288),
        NS(length=[27,64], factors=[[3,3,3],[4,4,4]], threads_per_transform=[9,16], workgroup_size=576),
        NS(length=[27,81], factors=[[3,3,3],[3,3,3,3]], threads_per_transform=[9,27], workgroup_size=729),
        NS(length=[32,4], factors=[[8,4],[2,2]], threads_per_transform=[4,2], workgroup_size=64),
        NS(length=[32,8], factors=[[8,4],[4,2]], threads_per_transform=[4,2], workgroup_size=64),
        NS(length=[32,9], factors=[[8,4],[3,3]], threads_per_transform=[4,3], workgroup_size=96),
        NS(length=[32,16], factors=[[8,4],[4,4]], threads_per_transform=[4,4], workgroup_size=128),
        NS(length=[32,25], factors=[[8,4],[5,5]], threads_per_transform=[4,5], workgroup_size=160),
        NS(length=[32,27], factors=[[8,4],[3,3,3]], threads_per_transform=[4,9], workgroup_size=288),
        NS(length=[32,32], factors=[[8,4],[8,4]], threads_per_transform=[4,4], workgroup_size=128),
        NS(length=[32,64], factors=[[8,4],[4,4,4]], threads_per_transform=[4,16], workgroup_size=512),
        NS(length=[32,81], factors=[[8,4],[3,3,3,3]], threads_per_transform=[4,27], workgroup_size=864),
        NS(length=[32,125], factors=[[8,4],[5,5,5]], threads_per_transform=[4,25], workgroup_size=800),
        NS(length=[32,128], factors=[[8,4],[8,4,4]], threads_per_transform=[4,16], workgroup_size=512),
        NS(length=[64,4], factors=[[4,4,4],[2,2]], threads_per_transform=[16,2], workgroup_size=128),
        NS(length=[64,8], factors=[[4,4,4],[4,2]], threads_per_transform=[16,2], workgroup_size=128),
        NS(length=[64,9], factors=[[4,4,4],[3,3]], threads_per_transform=[16,3], workgroup_size=192),
        NS(length=[64,16], factors=[[4,4,4],[4,4]], threads_per_transform=[16,4], workgroup_size=256),
        NS(length=[64,25], factors=[[4,4,4],[5,5]], threads_per_transform=[16,5], workgroup_size=400),
        NS(length=[64,27], factors=[[4,4,4],[3,3,3]], threads_per_transform=[16,9], workgroup_size=576),
        NS(length=[64,32], factors=[[4,4,4],[8,4]], threads_per_transform=[16,4], workgroup_size=512),
        NS(length=[81,4], factors=[[3,3,3,3],[2,2]], threads_per_transform=[27,2], workgroup_size=162),
        NS(length=[81,8], factors=[[3,3,3,3],[4,2]], threads_per_transform=[27,2], workgroup_size=216),
        NS(length=[81,9], factors=[[3,3,3,3],[3,3]], threads_per_transform=[27,3], workgroup_size=243),
        NS(length=[81,16], factors=[[3,3,3,3],[4,4]], threads_per_transform=[27,4], workgroup_size=432),
        NS(length=[81,25], factors=[[3,3,3,3],[5,5]], threads_per_transform=[27,5], workgroup_size=675),
        NS(length=[81,27], factors=[[3,3,3,3],[3,3,3]], threads_per_transform=[27,9], workgroup_size=729),
        NS(length=[81,32], factors=[[3,3,3,3],[8,4]], threads_per_transform=[27,4], workgroup_size=864),
        NS(length=[125,4], factors=[[5,5,5],[2,2]], threads_per_transform=[25,2], workgroup_size=250),
        NS(length=[125,8], factors=[[5,5,5],[4,2]], threads_per_transform=[25,2], workgroup_size=250),
        NS(length=[125,9], factors=[[5,5,5],[3,3]], threads_per_transform=[25,3], workgroup_size=375),
        NS(length=[125,16], factors=[[5,5,5],[4,4]], threads_per_transform=[25,4], workgroup_size=500),
        NS(length=[125,25], factors=[[5,5,5],[5,5]], threads_per_transform=[25,5], workgroup_size=625),
        NS(length=[125,32], factors=[[5,5,5],[8,4]], threads_per_transform=[25,4], workgroup_size=800),
        NS(length=[128,4], factors=[[8,4,4],[2,2]], threads_per_transform=[16,2], workgroup_size=256),
        NS(length=[128,8], factors=[[8,4,4],[4,2]], threads_per_transform=[16,2], workgroup_size=256),
        NS(length=[128,9], factors=[[8,4,4],[3,3]], threads_per_transform=[16,3], workgroup_size=384),
        NS(length=[128,16], factors=[[8,4,4],[4,4]], threads_per_transform=[16,4], workgroup_size=512),
        NS(length=[128,25], factors=[[8,4,4],[5,5]], threads_per_transform=[16,5], workgroup_size=640),
        NS(length=[128,32], factors=[[8,4,4],[8,4]], threads_per_transform=[16,4], workgroup_size=512),
        NS(length=[243,4], factors=[[3,3,3,3,3],[2,2]], threads_per_transform=[81,2], workgroup_size=486),
        NS(length=[243,8], factors=[[3,3,3,3,3],[4,2]], threads_per_transform=[81,2], workgroup_size=648),
        NS(length=[243,9], factors=[[3,3,3,3,3],[3,3]], threads_per_transform=[81,3], workgroup_size=729),
        NS(length=[256,4], factors=[[4,4,4,4],[2,2]], threads_per_transform=[64,2], workgroup_size=512),
        NS(length=[256,8], factors=[[4,4,4,4],[4,2]], threads_per_transform=[64,2], workgroup_size=512),
        NS(length=[256,9], factors=[[4,4,4,4],[3,3]], threads_per_transform=[64,3], workgroup_size=768),
        # ----- new for r2c/c2r
        NS(length=[7,84], factors=[[7],[7,2,6]], threads_per_transform=[1,12], workgroup_size=84),
        NS(length=[84,7], factors=[[7,2,6],[7]], threads_per_transform=[12,1], workgroup_size=84),
        NS(length=[10,20], factors=[[10],[5,4]], threads_per_transform=[1,5], workgroup_size=50),
        NS(length=[20,10], factors=[[5,4],[10]], threads_per_transform=[5,1], workgroup_size=50),
        NS(length=[26,64], factors=[[13,2],[4,4,4]], threads_per_transform=[2,16], workgroup_size=416),
        NS(length=[64,26], factors=[[4,4,4],[13,2]], threads_per_transform=[16,2], workgroup_size=416),
        NS(length=[26,72], factors=[[13,2],[8,3,3]], threads_per_transform=[2,9], workgroup_size=234),
        NS(length=[72,26], factors=[[8,3,3],[13,2]], threads_per_transform=[9,2], workgroup_size=234),
        NS(length=[30,60], factors=[[10,3],[6,10]], threads_per_transform=[3,10], workgroup_size=300),
        NS(length=[60,30], factors=[[6,10],[10,3]], threads_per_transform=[10,3], workgroup_size=300),
        NS(length=[36,72], factors=[[6,6],[8,3,3]], threads_per_transform=[6,9], workgroup_size=432),
        NS(length=[72,36], factors=[[8,3,3],[6,6]], threads_per_transform=[9,6], workgroup_size=432),
        NS(length=[36,80], factors=[[6,6],[5,2,8]], threads_per_transform=[6,10], workgroup_size=480),
        NS(length=[80,36], factors=[[5,2,8],[6,6]], threads_per_transform=[10,6], workgroup_size=480),
        NS(length=[36,84], factors=[[6,6],[7,2,6]], threads_per_transform=[6,12], workgroup_size=504),
        NS(length=[84,36], factors=[[7,2,6],[6,6]], threads_per_transform=[12,6], workgroup_size=504),
        NS(length=[40,80], factors=[[10,4],[5,2,8]], threads_per_transform=[4,10], workgroup_size=400),
        NS(length=[80,40], factors=[[5,2,8],[10,4]], threads_per_transform=[10,4], workgroup_size=400),
        NS(length=[42,84], factors=[[7,6],[7,2,6]], threads_per_transform=[6,12], workgroup_size=504),
        NS(length=[84,42], factors=[[7,2,6],[7,6]], threads_per_transform=[12,6], workgroup_size=504),
        NS(length=[42,96], factors=[[7,6],[6,16]], threads_per_transform=[6,6], workgroup_size=576),
        NS(length=[96,42], factors=[[6,16],[7,6]], threads_per_transform=[6,6], workgroup_size=576),
    ]

    expanded = []
    expanded.extend(NS(**kernel.__dict__,
                       scheme='CS_KERNEL_2D_SINGLE', runtime_compile=True) for kernel in fused_kernels)

    return expanded


def list_large_kernels():
    """Return list of large kernels to generate."""

    # Note: Default direct_to_from_reg is True
    sbcc_kernels = [
        NS(length=50,  factors=[10, 5],      use_3steps_large_twd={
           'sp': 'true',  'dp': 'true'}, workgroup_size=256),
        NS(length=52,  factors=[13, 4],      use_3steps_large_twd={
           'sp': 'true',  'dp': 'true'}),
        NS(length=60,  factors=[6, 10],      use_3steps_large_twd={
           'sp': 'false',  'dp': 'false'}),
        NS(length=64,  factors=[8, 8],       use_3steps_large_twd={
           'sp': 'true',  'dp': 'false'}, workgroup_size=256),
        NS(length=72,  factors=[8, 3, 3],    use_3steps_large_twd={
           'sp': 'true',  'dp': 'false'}),
        NS(length=80,  factors=[10, 8],      use_3steps_large_twd={
           'sp': 'false',  'dp': 'false'}),
        # 9,9 is good when direct-to-reg, but bad for Navi, so still uses radix-3
        NS(length=81,  factors=[3, 3, 3, 3], use_3steps_large_twd={
           'sp': 'true',  'dp': 'true'}),
        NS(length=84,  factors=[7, 2, 6],    use_3steps_large_twd={
           'sp': 'true',  'dp': 'true'}, threads_per_transform=14),
        NS(length=96,  factors=[8, 3, 4],    use_3steps_large_twd={
           'sp': 'false',  'dp': 'false'}, workgroup_size=256),
        NS(length=100, factors=[5, 5, 4],    use_3steps_large_twd={
           'sp': 'true',  'dp': 'false'}, workgroup_size=100, half_lds=True),
        NS(length=104, factors=[13, 8],      use_3steps_large_twd={
           'sp': 'true',  'dp': 'false'}),
        NS(length=108, factors=[6, 6, 3],    use_3steps_large_twd={
           'sp': 'true',  'dp': 'false'}),
        NS(length=112, factors=[4, 7, 4],    use_3steps_large_twd={
           'sp': 'false',  'dp': 'false'}),
        NS(length=121, factors=[11, 11],    use_3steps_large_twd={
           'sp': 'true',  'dp': 'true'}, workgroup_size=128, runtime_compile=True),
        NS(length=125, factors=[5, 5, 5],    use_3steps_large_twd={
           'sp': 'true',  'dp': 'false'}),
        NS(length=128, factors=[16, 8],    use_3steps_large_twd={
           'sp': 'true',  'dp': 'true'}, workgroup_size=256, threads_per_transform= 16),
        NS(length=160, factors=[4, 10, 4],   use_3steps_large_twd={
           'sp': 'false', 'dp': 'false'}, flavour='wide'),
        NS(length=168, factors=[7, 6, 4],    use_3steps_large_twd={
           'sp': 'true', 'dp': 'false'}, workgroup_size=128, half_lds=True),
        NS(length=169, factors=[13, 13],    use_3steps_large_twd={
           'sp': 'true', 'dp': 'false'}, workgroup_size=256, runtime_compile=True),
        NS(length=192, factors=[8, 6, 4],    use_3steps_large_twd={
           'sp': 'true', 'dp': 'true'}),
        NS(length=200, factors=[5, 8, 5],    use_3steps_large_twd={
           'sp': 'false', 'dp': 'false'}),
        NS(length=208, factors=[13, 16],     use_3steps_large_twd={
           'sp': 'false', 'dp': 'false'}),
        NS(length=216, factors=(6, 6, 6), use_3steps_large_twd={
           'sp': 'false', 'dp': 'false'}, threads_per_transform=36),
        NS(length=224, factors=[8, 7, 4],    use_3steps_large_twd={
           'sp': 'true', 'dp': 'false'}),
        NS(length=240, factors=[8, 5, 6],    use_3steps_large_twd={
           'sp': 'false', 'dp': 'false'}),
        # 9,9,3 isn't better on all archs, some are much better, some get regressions
        NS(length=243, factors=[3, 3, 3, 3, 3],    use_3steps_large_twd={
           'sp': 'true', 'dp': 'false'}, workgroup_size=243),
        NS(length=256, factors=[8, 4, 8], use_3steps_large_twd={
           'sp': 'true',  'dp': 'false'}, flavour='wide'),
        NS(length=280, factors=[8, 5, 7], use_3steps_large_twd={
           'sp': 'false',  'dp': 'false'}, runtime_compile=True),
        NS(length=289, factors=[17, 17],    use_3steps_large_twd={
           'sp': 'true', 'dp': 'true'}, runtime_compile=True),
        NS(length=336, factors=[6, 7, 8],    use_3steps_large_twd={
           'sp': 'false', 'dp': 'false'}),
        NS(length=343, factors=[7, 7, 7],    use_3steps_large_twd={
           'sp': 'true', 'dp': 'true'}),
        NS(length=512, factors=[8, 8, 8],    use_3steps_large_twd={
           'sp': 'true', 'dp': 'false'}),
    ]

    # for SBCC kernel, increase desired workgroup_size so that columns per
    # thread block is also increased. currently targeting for 16 columns
    block_width = 16
    for k in sbcc_kernels:
        k.scheme = 'CS_KERNEL_STOCKHAM_BLOCK_CC'
        if not hasattr(k, 'workgroup_size'):
            k.workgroup_size = block_width * \
                functools.reduce(mul, k.factors, 1) // min(k.factors)
        if hasattr(k, 'half_lds') and k.half_lds is True:
            k.workgroup_size = min(1024, k.workgroup_size * 2)
        if not hasattr(k, 'length'):
            k.length = functools.reduce(lambda a, b: a * b, k.factors)

    # for SBRC, if direct_to_from_reg is True, we do store-from-reg, but will not do load-to-reg
    #           And since SBRC is is dir-from-lds but NOT dir-to-reg, the global load part requires full LDS
    #           So, SBRC is able to use half-lds.
    sbrc_kernels = [
        NS(length=17,  factors=[17], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=1, runtime_compile=True),
        NS(length=49,  factors=[7, 7], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=196, threads_per_transform=7), # block_width=28
        NS(length=50,  factors=[10, 5], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=50, threads_per_transform=5, direct_to_from_reg=False), # block_width=10
        # SBRC64: wgs=256 poor in MI50
        NS(length=64,  factors=[4, 4, 4], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=128, threads_per_transform=16), # block_width=8
        # 9,9 not good by experiments
        NS(length=81,  factors=[3, 3, 3, 3], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=243, threads_per_transform=27), # block_width=9
        NS(length=100, factors=[5, 5, 4], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=100, threads_per_transform=25), # block_width=4
        NS(length=112, factors=[4, 7, 4], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=448, threads_per_transform=28), # block_width=16
        NS(length=121, factors=[11, 11], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=128, threads_per_transform=11, runtime_compile=True),
        NS(length=125, factors=[5, 5, 5], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=250, threads_per_transform=25), # block_width=10
        NS(length=128, factors=[8, 4, 4], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=128, threads_per_transform=16), # block_width=8
        NS(length=169, factors=[13, 13], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=13, runtime_compile=True),
        NS(length=192, factors=[6, 4, 4, 2], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=32), # block_width=8
        NS(length=200, factors=[8, 5, 5], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=400, threads_per_transform=40), # block_width=10
        NS(length=243, factors=[3, 3, 3, 3, 3], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=27, runtime_compile=True), # block_width=10
        NS(length=256, factors=[4, 4, 4, 4], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=32), # block_width=8
        NS(length=289, factors=[17, 17], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=128, threads_per_transform=17, runtime_compile=True),
        NS(length=343, factors=[7, 7, 7], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=49, runtime_compile=True),
        NS(length=512, factors=[8, 8, 8], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=512, threads_per_transform=128),
        NS(length=625, factors=[5, 5, 5, 5], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=128, threads_per_transform=125, runtime_compile=True),
        NS(length=1331, factors=[11, 11, 11], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=121, runtime_compile=True),
    ]

    for k in sbrc_kernels:
        k.half_lds = False

    # NB:
    # Technically, we could have SBCR kernels the same amount as SBCC.
    #
    # sbcr_kernels = copy.deepcopy(sbcc_kernels)
    # for k in sbcr_kernels:
    #     k.scheme = 'CS_KERNEL_STOCKHAM_BLOCK_CR'
    #

    # for SBCR, if direct_to_from_reg is True, we do load-to-reg, but will not do store-from-reg
    #           And since sbcr is dir-to-reg BUT NOT dir-from-reg, the global store part requires full LDS
    #           So, we can't satifly half_lds in SBCR !
    sbcr_kernels = [
        NS(length=56,  factors=[7, 8], direct_to_from_reg=False),
        NS(length=100, factors=[10, 10], workgroup_size=100),
        NS(length=200, factors=[8, 5, 5]),
        NS(length=336, factors=[6, 7, 8])
    ]

    block_width = 16
    for k in sbcr_kernels:
        k.scheme = 'CS_KERNEL_STOCKHAM_BLOCK_CR'
        k.half_lds = False
        if not hasattr(k, 'workgroup_size'):
            k.workgroup_size = block_width * \
                functools.reduce(mul, k.factors, 1) // min(k.factors)
        if not hasattr(k, 'length'):
            k.length = functools.reduce(lambda a, b: a * b, k.factors)

    return sbcc_kernels + sbcr_kernels + sbrc_kernels
# yapf: enable


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

    cpu_functions = []
    data = Variable('data_p', 'const void *')
    back = Variable('back_p', 'void *')
    # launchers_json has kernel names as keys to a list of launchers for each kernel variant
    for kernel in kernels:
        launchers = launchers_json[kernel_name(kernel)]
        for launcher_dict in launchers:
            launcher = NS(**launcher_dict)

            factors = launcher.factors
            length = launcher.lengths[0] if len(
                launcher.lengths) == 1 else (launcher.lengths[0],
                                             launcher.lengths[1])
            transforms_per_block = launcher.transforms_per_block
            workgroup_size = launcher.workgroup_size
            threads_per_transform = workgroup_size // transforms_per_block
            half_lds = launcher.half_lds
            direct_to_from_reg = launcher.direct_to_from_reg
            scheme = launcher.scheme
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
                             meta=NS(
                                 factors=factors,
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
                             ))

                cpu_functions.append(f)

    return cpu_functions


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
            # 2D single kernels always specify threads per transform
            if isinstance(k.length, list):
                proc.stdin.write(','.join([str(f)
                                           for f in k.factors[0]]) + " ")
                proc.stdin.write(','.join([str(f)
                                           for f in k.factors[1]]) + " ")
                proc.stdin.write(
                    ','.join([str(pre_enum[pre]) for pre in precisions]) + " ")
                proc.stdin.write(','.join(
                    [str(f) for f in k.threads_per_transform]))
            else:
                proc.stdin.write(','.join([str(f) for f in k.factors]) + " ")
                proc.stdin.write(','.join(
                    [str(pre_enum[pre]) for pre in precisions]))
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

            # for unspecified direct_to_from_reg, default is True only for CS_KERNEL_STOCKHAM and SBCC
            direct_to_from_reg = getattr(k, 'direct_to_from_reg', True)

            # Send data over to subprocess
            proc.stdin.write(f' {str(k.workgroup_size)}')
            proc.stdin.write(' 1' if half_lds else ' 0')
            proc.stdin.write(' 1' if direct_to_from_reg else ' 0')
            proc.stdin.write(f' {k.scheme}')
            proc.stdin.write(f' {kernel_name(k)}\n')

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
    parser.add_argument('--pattern',
                        type=str,
                        help='Kernel pattern to generate.',
                        default='all')
    parser.add_argument('--precision',
                        type=str,
                        help='Precision to generate.',
                        default='all')
    parser.add_argument('--manual-small',
                        type=str,
                        help='Small kernel sizes to generate.')
    parser.add_argument('--manual-large',
                        type=str,
                        help='Large kernel sizes to generate.')
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

    patterns = args.pattern.split(',')
    precisions = args.precision.split(',')
    if 'all' in precisions:
        precisions = ['dp', 'sp']
    precisions = [{
        'single': 'sp',
        'double': 'dp'
    }.get(p, p) for p in precisions]

    #
    # kernel list
    #

    kernels = []
    # move 2d out from all, no need to iterate the 2d-kernels for non-2d patterns
    kernels_2d = list_2d_kernels()
    all_kernels = list_small_kernels() + list_large_kernels()

    manual_small, manual_large = [], []
    if args.manual_small:
        manual_small = list(map(int, args.manual_small.split(',')))
    if args.manual_large:
        manual_large = list(map(int, args.manual_large.split(',')))

    if 'all' in patterns and not manual_small and not manual_large:
        kernels += all_kernels + kernels_2d
    if 'pow2' in patterns:
        lengths = [2**x for x in range(13)]
        kernels += [k for k in all_kernels if k.length in lengths]
    if 'pow3' in patterns:
        lengths = [3**x for x in range(8)]
        kernels += [k for k in all_kernels if k.length in lengths]
    if 'pow5' in patterns:
        lengths = [5**x for x in range(6)]
        kernels += [k for k in all_kernels if k.length in lengths]
    if 'pow7' in patterns:
        lengths = [7**x for x in range(5)]
        kernels += [k for k in all_kernels if k.length in lengths]
    if 'small' in patterns:
        schemes = ['CS_KERNEL_STOCKHAM']
        kernels += [k for k in all_kernels if k.scheme in schemes]
    if 'large' in patterns:
        schemes = [
            'CS_KERNEL_STOCKHAM_BLOCK_CC', 'CS_KERNEL_STOCKHAM_BLOCK_RC',
            'CS_KERNEL_STOCKHAM_BLOCK_CR'
        ]
        kernels += [k for k in all_kernels if k.scheme in schemes]
    if '2D' in patterns:
        kernels += kernels_2d
    if manual_small:
        schemes = ['CS_KERNEL_STOCKHAM']
        kernels += [
            k for k in all_kernels
            if k.length in manual_small and k.scheme in schemes
        ]
    if manual_large:
        schemes = [
            'CS_KERNEL_STOCKHAM_BLOCK_CC', 'CS_KERNEL_STOCKHAM_BLOCK_RC',
            'CS_KERNEL_STOCKHAM_BLOCK_CR'
        ]
        kernels += [
            k for k in all_kernels
            if k.length in manual_large and k.scheme in schemes
        ]

    kernels = unique(kernels)

    #
    # set runtime compile
    #

    kernels = default_runtime_compile(kernels,
                                      args.runtime_compile_default == 'ON')

    #
    # sub commands
    #

    if args.command == 'list':
        scprint(set(['function_pool.cpp'] + list_generated_kernels(kernels)))

    if args.command == 'generate':
        cpu_functions = generate_kernels(kernels, precisions,
                                         args.stockham_gen)
        func_files = generate_cpu_function_pool_pieces(cpu_functions,
                                                       args.num_files)
        for i in range(args.num_files):
            write(f'function_pool_init_{i}.cpp', func_files[i], format=False)
        write('function_pool.cpp',
              generate_cpu_function_pool_main(args.num_files),
              format=False)


if __name__ == '__main__':
    cli()
