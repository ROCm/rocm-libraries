# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""IR ordering and invalid-row publication for shared segment bounds."""

import pytest

from tests.parity.attention_paged_kv_emit import build, spec_for_index


@pytest.mark.parametrize("index", [66, 75, 77, 82, 91, 93])
def test_checked_segment_guards_and_invalid_row_publication(index):
    spec, arch = spec_for_index(index)
    kernel = build(spec, arch)
    ops = kernel.body.ops
    definitions = {value.name: op for op in ops for value in op.results}
    branches = [op for op in ops if op.name == "scf.if"]
    guard, invalid = branches[:2]
    assert [op.name for op in guard.regions[0].ops] == ["cf.return"]
    loads = [op for op in ops if op.name == "memref.global_load_typed"]
    assert ops.index(guard) < ops.index(loads[0])
    initialization = next(op for op in ops if op.name == "tile.smem_store")
    assert ops.index(guard) < ops.index(initialization) < ops.index(loads[0])
    assert definitions[initialization.operands[-1].name].attrs["value"] == 0

    # The sticky failure slot and safe length must use the same invalid predicate.
    predicate = invalid.operands[0].name
    selects = [op for op in ops if op.name == "arith.select"]
    checked = [op for op in selects if op.operands[0].name == predicate]
    assert len(checked) == 2
    assert definitions[checked[0].operands[1].name].attrs["value"] == 1
    assert definitions[checked[1].operands[1].name].attrs["value"] == 0
    assert checked[1].operands[2].name == loads[2].result.name

    def walk(region):
        for op in region.ops:
            yield op
            for child in op.regions:
                yield from walk(child)

    rejected = list(walk(invalid.regions[0]))
    assert invalid.regions[0].ops[-1].name == "cf.return"
    assert not any("global_load" in op.name for op in rejected)
    stores = [op for op in rejected if op.name == "memref.global_store_typed"]
    assert {op.operands[0].name for op in stores} == {
        "%segm_output_ptr",
        "%segm_max_ptr",
        "%segm_expsum_ptr",
        "%metadata_status_ptr",
    }
    assert stores[-1].operands[0].name == "%metadata_status_ptr"
    assert any(op.name == "tile.wave_any" for op in rejected)
