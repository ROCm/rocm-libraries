################################################################################
#
# Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
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
from Tensile.Hardware import HardwarePredicate
from Tensile.SolutionLibrary import PredicateLibrary


def test_hardware_predicate_comparison():
    a = HardwarePredicate.FromISA((9,0,0))
    b = HardwarePredicate.FromISA((9,0,6))
    c = HardwarePredicate("TruePred")
    d = HardwarePredicate.FromHardware((9,0,8), 60)
    e = HardwarePredicate.FromHardware((9,0,8), 64)
    f = HardwarePredicate.FromHardware((9,4,2))
    g = HardwarePredicate.FromHardware((9,4,2), isAPU=0)
    h = HardwarePredicate.FromHardware((9,4,2), isAPU=1)

    assert a < b
    assert a < c
    assert b < c

    assert d < a
    assert d < b
    assert d < c

    assert e < a
    assert e < b
    assert e < c
    assert e < d

    assert g < a
    assert g < b
    assert g < c
    assert g < d
    assert g < e
    assert g < f

    assert h < a
    assert h < b
    assert h < c
    assert h < d
    assert h < e
    assert h < f
    assert h < g

    assert not b < a
    assert not c < a
    assert not c < b
    assert not a < e
    assert not b < e
    assert not c < e
    assert not d < e
    assert not f < g
    assert not g < h

    assert not a < a
    assert not b < b
    assert not c < c
    assert not d < d
    assert not e < e
    assert not f < f
    assert not g < g
    assert not h < h

def hardware_library_objects_order():
    objs = [PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromISA((9,0,0))}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromISA((9,0,6))}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromISA((9,0,8))}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromISA((9,0,10))}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromHardware((9,0,8), 60)}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromHardware((9,0,8), 64)}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate('TruePred')}])
    ]

    return [copy.deepcopy(libs) for libs in itertools.permutations(objs)]

@pytest.mark.parametrize("libraries", hardware_library_objects_order())
def test_hardware_library_merge_order(libraries):
    lib = libraries[0]
    for lib2 in libraries[1:]:
        lib.merge(lib2)

    assert lib.rows[-1]['predicate'] == HardwarePredicate('TruePred')
    assert lib.rows[0]['predicate'] == HardwarePredicate.FromHardware((9,0,8), 64)
    assert lib.rows[1]['predicate'] == HardwarePredicate.FromHardware((9,0,8), 60)
    for r in lib.rows[:-1]:
        assert r['predicate'] != HardwarePredicate('TruePred')

def hardware_library_objects_order2():
    objs = [PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromISA((9,0,6))}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromHardware((9,4,2))}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromHardware((9,4,2), isAPU=0)}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromHardware((9,4,2), isAPU=1)}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate('TruePred')}])
    ]

    return [copy.deepcopy(libs) for libs in itertools.permutations(objs)]

@pytest.mark.parametrize("libraries", hardware_library_objects_order2())
def test_hardware_library_merge_order2(libraries):
    lib = libraries[0]
    for lib2 in libraries[1:]:
        lib.merge(lib2)

    assert lib.rows[-1]['predicate'] == HardwarePredicate('TruePred')
    assert lib.rows[0]['predicate'] == HardwarePredicate.FromHardware((9,4,2), isAPU=1)
    assert lib.rows[1]['predicate'] == HardwarePredicate.FromHardware((9,4,2), isAPU=0)
    assert lib.rows[3]['predicate'] == HardwarePredicate.FromHardware((9,4,2))
    for r in lib.rows[:-1]:
        assert r['predicate'] != HardwarePredicate('TruePred')

def hardware_library_objects_dups():
    objs = [PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromISA((9,0,0)), 'library': PredicateLibrary()}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromISA((9,0,6)), 'library': PredicateLibrary()}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromISA((9,0,6)), 'library': PredicateLibrary()}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromISA((9,0,8)), 'library': PredicateLibrary()}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate('TruePred'),      'library': PredicateLibrary()}])
    ]

    return [copy.deepcopy(libs) for libs in itertools.permutations(objs)]

@pytest.mark.parametrize("libraries", hardware_library_objects_dups())
def test_hardware_library_merge_dups(libraries):
    lib = libraries[0]
    for lib2 in libraries[1:]:
        lib.merge(lib2)

    assert len(lib.rows) == 4

    def getPred(row):
        return row['predicate']
    rowPreds = map(getPred, lib.rows)

    assert HardwarePredicate.FromISA((9,0,0)) in rowPreds
    assert HardwarePredicate.FromISA((9,0,6)) in rowPreds
    assert HardwarePredicate.FromISA((9,0,8)) in rowPreds
    assert HardwarePredicate('TruePred') in rowPreds
