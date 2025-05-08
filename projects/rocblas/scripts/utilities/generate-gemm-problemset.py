#!/usr/bin/python

"""Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
   ies of the Software, and to permit persons to whom the Software is furnished
   to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
   PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
   CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import itertools
import random
import sys
import getopt
import yaml

def yamlArguments(problemDesc, iterations):
    [dataType,
     [transA, transB],
     alpha,
     beta,
     [m_start, m_end],
     [n_start, n_end],
     [k_start, k_end]] = problemDesc

    m = random.randrange(m_start, m_end)
    n = random.randrange(n_start, n_end)
    k = random.randrange(k_start, k_end)

    return ('function: gemm_ex, <<: *{0}, alpha: {1}, beta: {2}, '
            'M: {3}, N: {4}, K: {5}, transA: {6}, transB: {7}, '
            'iters: {8}').format(
            dataType, alpha, beta,
            m, n, k, transA, transB,
            iterations)

def main(args):

    #Default values no configuration file provided
    realDataTypes  = [('f16_r',  'f16_r'),
                      ('f16_r',  'f32_r'),
                      ('bf16_r', 'f32_r'),
                      ('f32_r',  'f32_r'),
                      ('f64_r',  'f64_r'),
                      ('i8_r',   'i32_r')]

    dataTypes = ['half_precision',
                 'hpa_half_precision',
                 'single_precision',
                 'double_precision',
                 'int8_precision',
                 'hpa_bf16_precision']

    transposes = [['N', 'N'],
                  ['N', 'T'],
                  ['T', 'N'],
                  ['T', 'T']]

    dimensionRanges = [[20, 80],
                       [200, 900],
                       [2000, 7000]]

    alphas         = [1.0, 2.0]
    betas          = [1.0, 2.0, 0.0]

    (opts, rem) = getopt.getopt(args, '', ['filename=', 'yaml=', 'seed=', 'iters='])
    optDict = dict(opts)
    filename    = optDict.get('--filename', 'benchmark_problems.yaml')
    iterations  = optDict.get('--iters', 10)

    if '--seed' in optDict:
        random.seed(optDict['--seed'])

    if '--yaml' in optDict:
        contents = []
        with open(optDict['--yaml'], 'r') as f:
            try:
                contents = yaml.safe_load(f)
            except:
                print('Failed to read file: {}'.format(filename))
                return

        if type(contents) is dict:
            dataTypes       = contents.get('dataTypes', dataTypes)
            transposes      = contents.get('transposes', transposes)
            alphas          = contents.get('alphas', alphas)
            betas           = contents.get('betas', betas)
            dimensionRanges = contents.get('dimensionRanges', dimensionRanges)
        else:
            print('Top level of yaml file is not a dictionary')
            return

    problems = itertools.product(dataTypes,
                                 transposes,
                                 alphas,
                                 betas,
                                 dimensionRanges,
                                 dimensionRanges,
                                 dimensionRanges)

    with open(filename, 'w') as f:
        for problem in problems:
            f.write('- {{ {} }}\n'.format(yamlArguments(problem, iterations)))

if __name__=="__main__":
    main(sys.argv[1:])
