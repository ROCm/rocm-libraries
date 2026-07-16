# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from copy import deepcopy

from rocisa.asmpass import rocIsaPass, rocIsaPassOption
from rocisa.code import KernelBody, Label, Module
from rocisa.container import sgpr
from rocisa.instruction import SSwapPCB64


def test_swappc_callee_funcs_survive_deepcopy():
    inst = SSwapPCB64(dst=sgpr(26, 2), src=sgpr(12, 2))
    inst.calleeFuncs = ["label_Activation_Relu_VW1", "label_Activation_Gelu_VW1"]

    copied = deepcopy(inst)
    inst.calleeFuncs[0] = "changed"

    assert copied.calleeFuncs == ["label_Activation_Relu_VW1", "label_Activation_Gelu_VW1"]


def test_module_callable_metadata_survives_deepcopy():
    mod = Module("label_Activation_Relu_VW1")
    mod.isCallable = True
    mod.callableName = "label_Activation_Relu_VW1"

    copied = deepcopy(mod)

    assert copied.isCallable
    assert copied.callableName == "label_Activation_Relu_VW1"


def test_remove_duplicated_function_canonicalizes_swappc_callees():
    body = Module("body")

    for label in ("Activation_Relu_VW1_0", "Activation_Relu_VW1_1"):
        act_func = Module("ActFunc_VW1_Sgpr0_Tmp0_0")
        label_module = Module(f"{label}_module")
        label_module.add(Label(label, ""))
        act_func.add(label_module)
        body.add(act_func)

    swappc = SSwapPCB64(dst=sgpr(26, 2), src=sgpr(12, 2))
    swappc.calleeFuncs = ["label_Activation_Relu_VW1_1"]
    body.add(swappc)

    kernel = KernelBody("test")
    kernel.addBody(body)
    options = rocIsaPassOption()
    options.insertDelayAlu = False
    options.removeDupFunc = True
    options.removeDupAssign = False
    options.getCycles = False

    rocIsaPass(kernel, options)

    assert swappc.calleeFuncs == ["label_Activation_Relu_VW1_0"]


def test_remove_duplicated_function_preserves_non_duplicate_swappc_callees():
    body = Module("body")

    for label in ("Activation_Relu_VW1_0", "Activation_Relu_VW1_1"):
        act_func = Module("ActFunc_VW1_Sgpr0_Tmp0_0")
        label_module = Module(f"{label}_module")
        label_module.add(Label(label, ""))
        act_func.add(label_module)
        body.add(act_func)

    swappc = SSwapPCB64(dst=sgpr(26, 2), src=sgpr(12, 2))
    swappc.calleeFuncs = [
        "label_Activation_Relu_VW1_1",
        "label_Activation_Gelu_VW1_1",
    ]
    body.add(swappc)

    kernel = KernelBody("test")
    kernel.addBody(body)
    options = rocIsaPassOption()
    options.insertDelayAlu = False
    options.removeDupFunc = True
    options.removeDupAssign = False
    options.getCycles = False

    rocIsaPass(kernel, options)

    assert swappc.calleeFuncs == [
        "label_Activation_Relu_VW1_0",
        "label_Activation_Gelu_VW1_1",
    ]
