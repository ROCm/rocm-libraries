# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The test session's garbage collector setting (``rocke.tests.session_gc``)."""

from __future__ import annotations

import gc

from rocke.tests.session_gc import TEST_SESSION_YOUNG_GC_THRESHOLD, use_test_session_gc


def test_the_session_collects_the_young_generation_at_the_raised_threshold():
    """The tests' ``conftest.py`` applied the setting before this test ran."""
    assert gc.get_threshold()[0] == TEST_SESSION_YOUNG_GC_THRESHOLD


def test_the_setting_keeps_the_older_generations_thresholds():
    session = gc.get_threshold()
    try:
        gc.set_threshold(700, 11, 12)
        use_test_session_gc()
        assert gc.get_threshold() == (TEST_SESSION_YOUNG_GC_THRESHOLD, 11, 12)
    finally:
        gc.set_threshold(*session)
