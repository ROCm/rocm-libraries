################################################################################
#
# Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import pytest
import itertools
import copy
from Tensile.Properties import Predicate
from Tensile.SolutionLibrary import PredicateLibrary


def test_perf_metric_predicate_comparison():
    cu = Predicate('CUEfficiency')
    dv = Predicate('TruePred')

    assert cu < dv
    assert not dv < cu

def perf_metric_library_objects_order():
    objs = [PredicateLibrary('Problem', [{'predicate': Predicate('CUEfficiency')}]),
            PredicateLibrary('Problem', [{'predicate': Predicate('TruePred')}])
    ]

    return [copy.deepcopy(libs) for libs in itertools.permutations(objs)]

@pytest.mark.parametrize("libraries", perf_metric_library_objects_order())
def test_perf_metric_library_merge_order(libraries):
    lib = libraries[0]
    for lib2 in libraries[1:]:
        lib.merge(lib2)

    assert lib.rows[0]['predicate'] == Predicate('CUEfficiency')
    assert lib.rows[1]['predicate'] == Predicate('TruePred')

def perf_metric_library_objects_dups():
    objs = [PredicateLibrary('Problem', [{'predicate': Predicate('CUEfficiency'), 'library': PredicateLibrary()}]),
            PredicateLibrary('Problem', [{'predicate': Predicate('CUEfficiency'), 'library': PredicateLibrary()}]),
            PredicateLibrary('Problem', [{'predicate': Predicate('TruePred'),     'library': PredicateLibrary()}]),
            PredicateLibrary('Problem', [{'predicate': Predicate('TruePred'),     'library': PredicateLibrary()}])
    ]

    return [copy.deepcopy(libs) for libs in itertools.permutations(objs)]

@pytest.mark.parametrize("libraries", perf_metric_library_objects_dups())
def test_perf_metric_library_merge_dups(libraries):
    lib = libraries[0]
    for lib2 in libraries[1:]:
        lib.merge(lib2)

    assert len(lib.rows) == 2

    def getPred(row):
        return row['predicate']
    rowPreds = map(getPred, lib.rows)

    assert Predicate('CUEfficiency') in rowPreds
    assert Predicate('TruePred') in rowPreds
