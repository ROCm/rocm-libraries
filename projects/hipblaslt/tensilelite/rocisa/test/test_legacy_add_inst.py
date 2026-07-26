# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import rocisa
from rocisa.code import Module


def test_add_inst_formats_legacy_instruction():
    module = Module("legacy")

    module.addInst("v_add_u32", "v0", "v1", 2, "add operands")

    assert str(module) == "v_add_u32 v0, v1, 2".ljust(50) + " // add operands\n"


def test_add_inst_instruction_survives_no_comment_output():
    module = Module("legacy")
    module.addInst("s_nop", 0, "wait")

    target = rocisa.rocIsa.getInstance()
    options = target.getOutputOptions()
    original = options.outputNoComment
    try:
        options.outputNoComment = True
        target.setOutputOptions(options)
        assert str(module) == "s_nop 0\n"
    finally:
        options.outputNoComment = original
        target.setOutputOptions(options)
