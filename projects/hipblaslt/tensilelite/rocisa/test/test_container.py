################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import rocisa
from copy import deepcopy
import pickle
import os

isa = (9, 0, 10)
rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
global_isa = rocisa.rocIsa.getInstance()
global_isa.init(isa, rocm_path + "/bin/amdclang++", False)
global_isa.setKernel(isa, 64)

def test_containers():
    # Test Container
    container = rocisa.container.Container()
    assert isinstance(container, rocisa.container.Container)

    # Test DSModifiers
    ds_modifiers = rocisa.container.DSModifiers(1, 2, gds=True)
    assert str(ds_modifiers) == " offset:2 gds"

    # Test FLATModifiers
    flat_modifiers = rocisa.container.FLATModifiers(8, True, False, True, False)
    assert str(flat_modifiers) == " offset:8 glc lds"

    # Test MUBUFModifiers
    mubuf_modifiers = rocisa.container.MUBUFModifiers(True, 12, True, False, True, False, True)
    assert str(mubuf_modifiers) == " offen offset:12, glc"

    # Test SMEMModifiers
    smem_modifiers = rocisa.container.SMEMModifiers(True, False, 8)
    assert str(smem_modifiers) == " offset:8 glc"

    # Test SDWAModifiers
    sdwa_modifiers = rocisa.container.SDWAModifiers(
        dst_sel=rocisa.enum.SelectBit.WORD_0,
        src0_sel=rocisa.enum.SelectBit.WORD_0,
        src1_sel=rocisa.enum.SelectBit.WORD_1
    )
    assert str(sdwa_modifiers) == " dst_sel:WORD_0 src0_sel:WORD_0 src1_sel:WORD_1"

    # Test VOP3PModifiers
    vop3p_modifiers = rocisa.container.VOP3PModifiers([0, 0], [0, 1], [0, 0])
    assert str(vop3p_modifiers) == " op_sel:[0,0] op_sel_hi:[0,1] byte_sel:[0,0]"

    # Test EXEC
    exec_modifiers = rocisa.container.EXEC(True)
    assert str(exec_modifiers) == "exec"

    # Test VCC
    vcc_modifiers = rocisa.container.VCC(True)
    assert str(vcc_modifiers) == "vcc"

    # Test HWRegContainer
    hwreg_container = rocisa.container.HWRegContainer("reg", [1, 1])
    assert str(hwreg_container) == "hwreg(reg,1,1)"

    # Test RegName
    reg_name = rocisa.container.RegName("GSUIndex", [4, 4])
    assert reg_name.getTotalOffsets() == 8
    assert str(reg_name) == "GSUIndex+4+4"
    reg_name.addOffset(4)
    assert reg_name.getTotalOffsets() == 12
    reg_name.setOffset(0, 8)
    assert reg_name.getTotalOffsets() == 16

    # Test RegisterContainer
    reg_container = rocisa.container.RegisterContainer("v", reg_name, 4, 2)
    assert reg_container.getRegNameWithType() == "vgprGSUIndex"
    reg_container.setAbs(True)
    assert str(reg_container) == "abs(v[vgprGSUIndex+8+4+4:vgprGSUIndex+8+4+4+1])"
    reg_container.setAbs(False)
    assert str(reg_container) == "v[vgprGSUIndex+8+4+4:vgprGSUIndex+8+4+4+1]"
    assert reg_container != None
    reg1, reg2 = reg_container.splitRegContainer()
    assert str(reg1) == "v[vgprGSUIndex+8+4+4]"
    assert str(reg2) == "v[vgprGSUIndex+8+4+4+1]"

    # Test HolderContainer
    holder_container = rocisa.container.HolderContainer("v", 4, 2)
    assert holder_container.getCopiedRC() == holder_container
    assert str(holder_container) == "v[4:5]"

    # Test Holder
    from rocisa.container import vgpr, sgpr, accvgpr, mgpr, Holder
    holder = Holder("holder");
    assert holder.idx == -1
    assert str(holder.name) == "holder"
    assert isinstance(holder, Holder)

    # Test xgpr
    testGpr = vgpr("TestGpr", 2)
    assert str(testGpr) == "v[vgprTestGpr:vgprTestGpr+1]"
    testGpr.setAbs(True)
    assert str(testGpr) == "abs(v[vgprTestGpr:vgprTestGpr+1])"
    testGpr = sgpr(2, 2)
    assert str(testGpr) == "s[2:3]"
    testGpr = accvgpr(Holder(4))
    assert str(testGpr) == "acc4"
    testGpr = mgpr(holder)
    assert str(testGpr) == "m[mgprholder]"

def test_containers_copy():
    def copy_test(name, obj):
        print("Copy test:", name)
        obj2 = deepcopy(obj)
        print("    Copied using deepcopy:", str(obj2))
        obj2 = pickle.loads(pickle.dumps(obj))
        print("    Copied using pickle:", str(obj2))

    ds = rocisa.container.DSModifiers(gds=True)
    copy_test("DSModifiers", ds)
    flat = rocisa.container.FLATModifiers(lds=True)
    copy_test("FLATModifiers", flat)
    mubuf = rocisa.container.MUBUFModifiers(True, 12, True, False, True, False, True)
    copy_test("MUBUFModifiers", mubuf)
    smem = rocisa.container.SMEMModifiers(True, False, 8)
    copy_test("SMEMModifiers", smem)
    sdwa = rocisa.container.SDWAModifiers(
        dst_sel=rocisa.enum.SelectBit.WORD_0,
        src0_sel=rocisa.enum.SelectBit.WORD_0,
        src1_sel=rocisa.enum.SelectBit.WORD_1
    )
    copy_test("SDWAModifiers", sdwa)
    vop3p = rocisa.container.VOP3PModifiers([0, 0], [0, 1], [0, 0])
    copy_test("VOP3PModifiers", vop3p)
    exec = rocisa.container.EXEC(True)
    copy_test("EXEC", exec)
    vcc = rocisa.container.VCC(True)
    copy_test("VCC", vcc)
    hwreg = rocisa.container.HWRegContainer("reg", [1, 1])
    copy_test("HWRegContainer", hwreg)
    reg_name = rocisa.container.RegName("GSUIndex", [4, 4])
    copy_test("RegName", reg_name)
    reg = rocisa.container.RegisterContainer("v", reg_name, 4, 2)
    copy_test("RegisterContainer", reg)
    holder = rocisa.container.HolderContainer("v", 4, 2)
    copy_test("HolderContainer", holder)

test_containers()
test_containers_copy()
