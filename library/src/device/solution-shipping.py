#!/usr/bin/env python3
# Copyright (C) 2021 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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
"""rocFFT solution map library builder.

"""

import argparse
import json
from os import listdir
from os.path import isfile, join
from pathlib import Path
from types import SimpleNamespace as NS

from generator import (ArgumentList, BaseNode, Call, CommentLines, Function,
                       Include, LineBreak, Map, StatementList, Throw, Variable,
                       Assign, If, ReturnStatement, name_args, write)

#
# CMake helpers
#


def cjoin(xs):
    """Join 'xs' with commas."""
    return ','.join(str(x) for x in xs)


# get gfx___ which is the prefix of the solution map
def get_local_gpu_gfx(archs):
    archs_gfx = []
    for arch in archs:
        gfx_target = arch.split(':')[0]
        if gfx_target not in archs_gfx:
            archs_gfx.append(gfx_target)

    return archs_gfx


#
# Prototype generators
#


@name_args(['config', 'meta'])
class KernelConfig(BaseNode):

    def __str__(self):
        kc = 'KernelConfig('

        use_3steps = self.config['use_3steps']
        kc += 'true' if (use_3steps is not None
                         and use_3steps == True) else 'false'

        factors = self.config['factors']
        kc += ', {' + cjoin(
            factors) + '}' if factors is not None else ', { 0 }'

        tpb = self.config['tpb']
        kc += ', ' + str(tpb) if tpb is not None else ', 0'

        workgroup_size = self.config['wgs']
        kc += ', ' + str(
            workgroup_size) if workgroup_size is not None else ', 0'

        tpt = self.config['tpt']
        kc += ', {' + ','.join([str(s) for s in tpt
                                ]) + '}' if tpt is not None else ', { 0 }'

        half_lds = self.config['half_lds']
        kc += ', true' if (half_lds is not None
                           and half_lds == True) else ', false'

        dir_reg = self.config['dir_reg']
        kc += ', true' if (dir_reg is not None
                           and dir_reg == True) else ', false'

        buffer_inst = self.config['buffer_inst']
        kc += ', true' if (buffer_inst is not None
                           and buffer_inst == True) else ', false'

        kc += ', StrToEBType("' + str(self.config['ebtype']) + '")'
        kc += ', ' + str(self.config['direction'])
        kc += ', ' + str(self.config['static_dim'])
        kc += ', StrToPlacementCode("' + str(self.config['placement']) + '")'
        kc += ', StrToArrayType("' + str(self.config['iAryType']) + '")'
        kc += ', StrToArrayType("' + str(self.config['oAryType']) + '")'

        kc += ')'
        return kc


@name_args(['key', 'meta'])
class FMKey(BaseNode):

    def __str__(self):
        k = 'FMKey('
        lengths = self.key['lengths']
        k += str(lengths[0])
        if lengths[1] != 0:
            k += ', ' + str(lengths[1])
        k += ', StrToPrecision("' + str(self.key['precision']) + '")'
        k += ', StrToComputeScheme("' + str(self.key['scheme']) + '")'
        k += ', StrToSBRCTransType("' + str(self.key['sbrc_trans']) + '")'
        k += ', ' + str(KernelConfig(self.key['kernelConfig']))
        k += ')'
        return k


@name_args(['key', 'meta'])
class SolutionPtr(BaseNode):

    def __str__(self):
        k = '{'
        k += '"' + str(self.key['child_token']) + '"'
        k += ',' + str(self.key['child_option'])
        k += '}'
        return k


@name_args(['meta'])
class SolutionNode(BaseNode):

    def __str__(self):
        sol = 'SolutionNode'
        return sol


def generate_solution_map(solutions):
    """Generate function to populate the solution map."""

    #
    # add the solutions in the solution_map constructor
    #
    solution_nodes = Map('solution_nodes')
    var_solution = Variable('solution', 'SolutionNode')

    populate = StatementList()
    populate += If('!rocfft_getenv("ROCFFT_USE_EMPTY_SOL_MAP").empty()',
                   ReturnStatement())

    if len(solutions) > 0:
        populate += var_solution.declaration()
        populate += LineBreak()

    for sol in solutions:
        arch, token, sol_node_type, scheme, kernel_key, childrens = \
            sol.meta.arch, sol.meta.token, sol.meta.sol_type, \
                sol.meta.scheme, sol.meta.kernel, sol.meta.childnodes

        populate += LineBreak()
        populate += CommentLines("add new solution")

        # assigning solution data
        populate += Assign(
            str(var_solution) + '.arch_name', '"' + str(arch) + '"')

        populate += Assign(
            str(var_solution) + '.sol_node_type',
            'StrToSolutionNodeType("' + sol_node_type + '")')

        # Check if it has .kernel_key field
        populate += Assign(
            str(var_solution) + '.kernel_key',
            FMKey(kernel_key)
            if kernel_key is not None else 'FMKey::EmptyFMKey()')

        # SOL_INTERNAL_NODE or SOL_LEAF_NODE or SOL_DUMMY
        if scheme is not None:
            populate += Assign(
                str(var_solution) + '.using_scheme',
                'StrToComputeScheme("' + str(scheme) + '")')
        # SOL_KERNEL_ONLY
        elif kernel_key is not None:
            populate += Assign(
                str(var_solution) + '.using_scheme',
                str(var_solution) + '.kernel_key.scheme')
        # SOL_BUILTIN_KERNEL
        else:
            populate += Assign(str(var_solution) + '.using_scheme', 'CS_NONE')

        populate += Assign(
            str(var_solution) + '.solution_childnodes',
            '{' + cjoin([SolutionPtr(s) for s in childrens]) + '}')

        # ready to add to map
        probKey = Call(name='ProblemKey',
                       arguments=ArgumentList('"' + arch + '"',
                                              '"' + token + '"')).inline()
        populate += Call(name='add_solution_private',
                         arguments=ArgumentList(probKey, var_solution))

    return StatementList(
        Include('"solution_map.h"'), Include('"../../shared/environment.h"'),
        Function(name='solution_map::solution_map',
                 value=False,
                 arguments=ArgumentList(),
                 body=populate), LineBreak())


#
# Main!
#


def generate_solutions(archs, folder):
    solutions = []
    path = Path(folder)

    if not path.exists():
        return solutions

    all_files = [f for f in listdir(folder) if isfile(join(folder, f))]
    # solution map file
    target_files = [
        join(folder, f) for f in all_files if f.split('_')[0] in archs
    ]

    # get solutions part
    for file in target_files:
        solution_map_file = open(file, 'r')
        solution_map_data = json.load(solution_map_file)

        # handle version
        version = 'no version num'
        if 'Version' in solution_map_data:
            version = solution_map_data['Version']
            all_solutions = solution_map_data['Data']
        else:
            all_solutions = solution_map_data

        print('-- format version=' + str(version))

        for entry_dict in all_solutions:

            entry = NS(**entry_dict)
            problemKey = entry.Problem
            solutionVec = entry.Solutions

            problem = NS(**problemKey)
            arch = problem.arch
            token = problem.token

            # list of {}, {}, {}.... {} = solutionNode which is a dict
            for solution_dict in solutionVec:
                solution = NS(**solution_dict)
                sol_node_type = solution.sol_node_type
                using_scheme = getattr(solution, 'using_scheme', None)
                kernel_key = getattr(solution, 'kernel_key', None)
                childrens = getattr(solution, 'solution_childnodes', [])
                s = SolutionNode(meta=NS(
                    arch=str(arch),
                    token=str(token),
                    sol_type=str(sol_node_type),
                    scheme=using_scheme,
                    kernel=kernel_key,
                    childnodes=childrens,
                ))
                solutions.append(s)

    return solutions


def cli():
    """Command line interface..."""
    parser = argparse.ArgumentParser(prog='solution-shipping')
    parser.add_argument('--gpu-arch',
                        type=str,
                        help='Solutions of specific gpu arch')
    parser.add_argument('--data-folder',
                        type=str,
                        help='Folder containing the solution map text files.')

    args = parser.parse_args()

    print('-- gpu_arch=' + args.gpu_arch)
    print('-- data-folder=' + args.data_folder)

    archs = args.gpu_arch.split(' ')
    if 'all' in archs:
        archs = [
            'gfx900', 'gfx906', 'gfx908', 'gfx90a', 'gfx942', 'gfx1030',
            'gfx1100', 'gfx1101', 'gfx1102', 'gfx1200', 'gfx1201'
        ]

    # remove xnack and sramecc
    archs = get_local_gpu_gfx(archs)
    archs.append('any')

    solutions = generate_solutions(archs, args.data_folder)

    write('solutions.cpp', generate_solution_map(solutions), format=True)


if __name__ == '__main__':
    cli()
