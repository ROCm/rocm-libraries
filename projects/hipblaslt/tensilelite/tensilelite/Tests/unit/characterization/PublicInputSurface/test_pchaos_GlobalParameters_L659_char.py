################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

import pytest

pytestmark = pytest.mark.unit


def min_required_version_present(config: dict) -> bool:
    return "MinimumRequiredVersion" in config


def test_key_present_version_6_0_0():
    config = {"MinimumRequiredVersion": "6.0.0"}
    assert min_required_version_present(config) is True


def test_key_present_version_5_0_0():
    config = {"MinimumRequiredVersion": "5.0.0"}
    assert min_required_version_present(config) is True


def test_key_present_version_4_99_0():
    config = {"MinimumRequiredVersion": "4.99.0"}
    assert min_required_version_present(config) is True


def test_key_present_alongside_other_keys():
    config = {"MinimumRequiredVersion": "5.0.0", "GlobalParameters": {}}
    assert min_required_version_present(config) is True


def test_key_absent_empty_config():
    config = {}
    assert min_required_version_present(config) is False


def test_key_absent_other_key_only():
    config = {"OtherKey": "x"}
    assert min_required_version_present(config) is False


def test_key_absent_multiple_unrelated_keys():
    config = {"GlobalParameters": {}, "BenchmarkProblems": []}
    assert min_required_version_present(config) is False


def test_predicate_ignores_value_content():
    for version_string in ("6.0.0", "5.0.0", "4.99.0", "0.0.1", "99.99.99"):
        config = {"MinimumRequiredVersion": version_string}
        assert min_required_version_present(config) is True


def test_similar_key_name_does_not_trigger():
    config = {"minimumrequiredversion": "5.0.0"}
    assert min_required_version_present(config) is False
    config2 = {"MinimumRequired": "5.0.0"}
    assert min_required_version_present(config2) is False
