# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Copy-paste overlap (SLO106)."""
import slo_overlap as O


def test_plain_overlap_uses_the_smaller_file():
    a = "\n".join(f"value_number_{i} = compute({i})" for i in range(100))
    b = a + "\n" + "\n".join(f"other_value_{i} = compute({i})" for i in range(100))
    ps = O.pairs({"a.py": a, "b.py": b}, {"a.py": 300, "b.py": 300})
    assert ps[0]["overlap"] == 1.0 and ps[0]["shared"] == 100


def test_renamed_copy_is_caught_only_by_the_normalized_score():
    a = "\n".join(f"lane_{i} = b.add(tid, b.mul(row, {i}))" for i in range(100))
    b = a.replace("lane_", "q_").replace("tid", "t").replace("row", "r")
    ps = O.pairs({"a.py": a, "b.py": b}, {"a.py": 300, "b.py": 300})
    assert ps[0]["overlap"] == 0.0 and ps[0]["overlap_norm"] == 1.0


def test_small_files_are_not_compared():
    a = "\n".join(f"value_number_{i} = compute({i})" for i in range(100))
    assert O.pairs({"a.py": a, "b.py": a}, {"a.py": 100, "b.py": 300}) == []
