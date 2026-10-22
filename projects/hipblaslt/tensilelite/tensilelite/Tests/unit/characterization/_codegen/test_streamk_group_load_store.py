# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check real workspace-store operands and ordering, without executing instructions."""

from pathlib import Path
import re

import pytest
from tensilelite import __file__ as tensile_file

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit


def test_grouped_partial_stores_advance_workspace_offset():
    config = Path(tensile_file).parent / "Tests/common/streamk/gfx1250/core/sk_group_load_store_partials.yaml"
    kernels = emit_kernels_from_config(str(config), arch="gfx1250", canonical=False)
    assert len(kernels) == 2
    for name, assembly, error in kernels:
        assert error == 0
        assert_assembles(assembly, name)
        stores = list(re.finditer(
            r"^buffer_store_b128 .*s\[sgprSrdWS:sgprSrdWS\+3\], (s\d+) offen offset:0\b.*$",
            assembly, re.M))
        # MT256x256 / 128 threads / 8 FP32 values per vector = 64 vectors.
        assert len(stores) == 64
        initializers = list(re.finditer(rf"^s_mov_b32 {stores[0][1]}, 0\b",
                                       assembly[:stores[0].start()], re.M))
        assert initializers
        previous = initializers[-1].start()
        for index, store in enumerate(stores):
            offset = store[1]
            updates = re.findall(rf"^s_(?:mov_b32|add_u32) {offset}, ([^/\n]+)",
                                 assembly[previous:store.start()], re.M)
            assert [u.strip() for u in updates] == (["0"] if index == 0 else [f"{offset}, 4096"])
            # Both 16-byte halves must consume the same scalar offset.
            second = re.search(
                rf"^buffer_store_b128 .*s\[sgprSrdWS:sgprSrdWS\+3\], {offset} offen offset:16\b.*$",
                assembly[store.end():], re.M)
            assert second
            end = store.end() + second.end()
            assert not re.search(rf"^s_(?:mov_b32|add_u32) {offset},", assembly[store.end():end], re.M)
            previous = end
