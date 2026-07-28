import importlib

import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")

pytestmark = pytest.mark.unit

fn = S._subtileGRKPartitionIsBuggy

def test_all_three_conditions_true_returns_true():

    assert fn(2, [3, 2, 5]) is True

def test_load_ratio_not_greater_than_one_returns_false():

    assert fn(1, [3, 2, 5]) is False

def test_boundary_load_ratio_two_kills_gt2_mutant():

    assert fn(2, [3, 2, 2]) is True

def test_grid_index_one_used_not_index_two_kills_index_mutant():

    assert fn(2, [3, 2, 0]) is True

def test_grid_index_one_boundary_kills_ge_and_multi_mutants():

    assert fn(2, [3, 1, 5]) is False

def test_grid_index_one_gt2_kills_mutant():

    assert fn(2, [3, 2, 5]) is True

def test_modulo_not_division_kills_div_mutant():

    assert fn(2, [4, 2, 5]) is False

def test_modulo_uses_index_zero_not_one_kills_index_mutant():

    assert fn(2, [3, 4, 5]) is True

def test_int_of_load_ratio_not_none_kills_none_mutant():

    assert fn(2, [3, 2, 5]) is True
