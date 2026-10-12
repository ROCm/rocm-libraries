# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from rocisa.instruction import GirActionKind, SSchedulingFence


def test_gir_action_getter_returns_complete_record():
    inst = SSchedulingFence()
    assert inst.getGirActionData() is None

    accesses = [{
        "is_write": False,
        "operand": 3,
        "ring": 2,
        "gen": 1,
        "gdelta": -1,
        "absolute": -1,
        "cross_agent": True,
    }]
    inst.setGirActionData(17, 9, GirActionKind.Fence, accesses)

    assert inst.getGirActionData() == {
        "action_id": 17,
        "anchor_action": 9,
        "kind": 3,
        "accesses": accesses,
    }
