# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

from copy import deepcopy
from generator import Problem, ProblemSet

amax_def_args = {'--type'  : 'H',
                 '--dtype' : 'S',
                 '--init'  : 'hpl',
                 }

lengths = {

    'amax_example': [
        (64, 64),
        (128, 128),
    ],

    'amax_set_1': [
        (16,    1024  ),
        (16,    8192  ),
        (16,    65536 ),
        (2048,  1024  ),
        (2048,  8192  ),
        (2048,  65536 ),
        (8192,  1024  ),
        (8192,  8192  ),
        (8192,  65536 ),
        (16,    16384 ),
        (32,    16384 ),
        (16,    2048  ),
        (32,    2048  ),
        (16,    4096  ),
        (32,    4096  ),
    ],
}

# Suite definitions
def amax_example():
    """AMAX example."""

    problemlist = []

    for length in lengths['amax_example']:
        args = deepcopy(amax_def_args)
        args.update({'--m': str(length[0]),
                     '--n': str(length[1])})
        problemlist.append(Problem(args=args))

    yield ProblemSet(benchType="amax", name="example", problems=problemlist)

def amax_set_1():
    """AMAX benchset 1."""

    problemlist = []

    for length in lengths['amax_set_1']:
        args = deepcopy(amax_def_args)
        args.update({'--m': str(length[0]),
                     '--n': str(length[1])})
        problemlist.append(Problem(args=args))

    yield ProblemSet(benchType="amax", name="benchset_1", problems=problemlist)

def matmul_set_1():
    """gemm benchset 1"""

    problemlist = [Problem(args={"--log_function_name" : "" , "--yaml" : "matmul_probset1_bench.yaml"})]
    yield ProblemSet(benchType="matmul", name="benchset_1", problems=problemlist)


def all():
    """all routine benchmarks"""

    yield from amax_set_1()
    yield from matmul_set_1()