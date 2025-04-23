#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
import migraphx, array, sys


def create_buffer(t, data, shape):
    a = array.array(t, data)
    m = memoryview(a.tobytes())
    return m.cast(t, shape)


def test_add_op():
    p = migraphx.program()
    mm = p.get_main_module()
    x = mm.add_literal(create_buffer('f', [1.0] * 9, (3, 3)))
    y = mm.add_literal(create_buffer('f', [2.0] * 9, (3, 3)))
    add_op = mm.add_instruction(migraphx.op("add"), [x, y])
    mm.add_return([add_op])
    p.compile(migraphx.get_target("ref"))
    params = {}
    output = p.run(params)[-1].tolist()
    assert output == list([3.0] * 9)


def test_if_then_else():
    param_shape = migraphx.shape(lens=[3, 3], type="float")
    cond_shape = migraphx.shape(type="bool", lens=[1], strides=[0])

    def create_program():
        p = migraphx.program()
        mm = p.get_main_module()
        cond = mm.add_parameter("cond", cond_shape)
        x = mm.add_parameter("x", param_shape)
        y = mm.add_parameter("y", param_shape)
        then_mod = p.create_module("If_0_if")
        x_identity = then_mod.add_instruction(migraphx.op("identity"), [x])
        then_mod.add_return([x_identity])

        else_mod = p.create_module("If_0_else")
        y_identity = else_mod.add_instruction(migraphx.op("identity"), [y])
        else_mod.add_return([y_identity])

        if_ins = mm.add_instruction(migraphx.op("if"), [cond],
                                    [then_mod, else_mod])
        ret = mm.add_instruction(migraphx.op("get_tuple_elem", **{"index": 0}),
                                 [if_ins])
        mm.add_return([ret])
        return p

    params = {}
    params["x"] = migraphx.generate_argument(param_shape)
    params["y"] = migraphx.generate_argument(param_shape)

    def run_prog(cond):
        p = create_program()
        p.compile(migraphx.get_target("ref"))
        params["cond"] = migraphx.fill_argument(cond_shape, cond)
        output = p.run(params)[-1]
        return output

    assert run_prog(True) == params["x"]
    assert run_prog(False) == params["y"]


if __name__ == "__main__":
    if sys.version_info >= (3, 0):
        test_add_op()
    test_if_then_else()
