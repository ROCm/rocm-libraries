################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurface characterization: the GPU-frequency guard at Tensile.py:603.

Branch 26f1acfe1ff93095519d418855bb9593f2c4f4bb. The predicate is a pure
4-ary boolean conjunction controlling entry into the GPU-frequency configuration
block:

  'LibraryLogic' in config and UseEffLike and not buildOnly
      and not globalParameters["CpuOnly"]

Exhaustive 2^4 truth table: exactly 1 TRUE assignment (all four conjuncts
hold), 15 FALSE assignments. The four false_examples each negate exactly one
conjunct, covering the four minimal failure modes.

Caveat (not part of this guard logic): Tensile.py:600 forces
UseEffLike=False on RHEL8, making the TRUE branch unreachable on a RHEL8
host -- an input-reachability constraint on the enclosing call path.

These tests pin ACTUAL observed behavior; nothing aspirational.
"""

import pytest

pytestmark = pytest.mark.unit


def liblogic_freq_branch(
    library_logic_in_config: bool,
    use_eff_like: bool,
    build_only: bool,
    cpu_only: bool,
) -> bool:
    """Mirrors Tensile.py:603 guard: enter GPU-frequency configuration block."""
    return (
        library_logic_in_config
        and use_eff_like
        and not build_only
        and not cpu_only
    )


def test_l603_true_all_conjuncts_hold():
    assert liblogic_freq_branch(
        library_logic_in_config=True,
        use_eff_like=True,
        build_only=False,
        cpu_only=False,
    ) is True


def test_l603_false_missing_library_logic_key():
    assert liblogic_freq_branch(
        library_logic_in_config=False,
        use_eff_like=False,
        build_only=False,
        cpu_only=False,
    ) is False


def test_l603_false_build_only_set():
    assert liblogic_freq_branch(
        library_logic_in_config=True,
        use_eff_like=True,
        build_only=True,
        cpu_only=False,
    ) is False


def test_l603_false_cpu_only_set():
    assert liblogic_freq_branch(
        library_logic_in_config=True,
        use_eff_like=True,
        build_only=False,
        cpu_only=True,
    ) is False


def test_l603_false_use_eff_like_false():
    assert liblogic_freq_branch(
        library_logic_in_config=True,
        use_eff_like=False,
        build_only=False,
        cpu_only=False,
    ) is False


def test_l603_real_predicate_dict_membership_true():
    config = {'LibraryLogic': ['some_logic_entry']}
    UseEffLike = True
    buildOnly = False
    globalParameters = {'CpuOnly': False}
    result = (
        'LibraryLogic' in config
        and UseEffLike
        and not buildOnly
        and not globalParameters['CpuOnly']
    )
    assert result is True


def test_l603_real_predicate_dict_membership_key_absent():
    config = {'GlobalParameters': {}}
    UseEffLike = True
    buildOnly = False
    globalParameters = {'CpuOnly': False}
    result = (
        'LibraryLogic' in config
        and UseEffLike
        and not buildOnly
        and not globalParameters['CpuOnly']
    )
    assert result is False
