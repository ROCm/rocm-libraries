# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocke.tests.arch_catalog: inside the block every catalog lookup reads only the
synthetic catalog, and after it none reads it, including caches the block filled
as a side effect."""

from __future__ import annotations

from rocke.core.arch.target import _op_id_dst_dtype, _op_id_family
from rocke.tests.arch_catalog import patched_arch_specs

_SYNTHETIC_ROW = {
    "family": "mma",
    "op_id": "synthetic_fixture_atom",
    "m": 16,
    "n": 16,
    "k": 4,
    "srcs": [{"dtype": "fp32"}, {"dtype": "fp32"}, {"dtype": "fp32"}],
    "dst": {"dtype": "fp32"},
}


def test_patched_arch_specs_serves_only_its_catalog_and_leaves_no_trace():
    real_families = dict(_op_id_family())  # filled from the real catalog
    real_dst_dtypes = dict(_op_id_dst_dtype())
    assert _SYNTHETIC_ROW["op_id"] not in real_families

    with patched_arch_specs({"synthetic": {"mma": [_SYNTHETIC_ROW]}}):
        assert _op_id_family() == {"synthetic_fixture_atom": "mma"}
        assert set(_op_id_dst_dtype()) == {"synthetic_fixture_atom"}

    assert _op_id_family() == real_families
    assert _op_id_dst_dtype() == real_dst_dtypes
