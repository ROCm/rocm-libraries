#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
import migraphx


def test_instruction_shape():
    p = migraphx.program()
    mm = p.get_main_module()
    input_shape = migraphx.shape(lens=[4, 4, 64], type="half_type")
    i = mm.add_parameter("x", input_shape)
    i2 = mm.add_instruction(migraphx.op("reshape", dims=[16, 64]), [i])
    out_shape = i2.shape()

    assert out_shape.lens() == [16, 64]
    assert out_shape.strides() == [64, 1]
    assert out_shape.type_string() == "half_type"


def test_instruction_op():
    p = migraphx.program()
    mm = p.get_main_module()
    input_shape = migraphx.shape(lens=[2, 24])
    i = mm.add_parameter("x", input_shape)
    i2 = mm.add_instruction(migraphx.op("relu"), [i])
    out_op = i2.op()

    assert out_op.name() == "relu"


if __name__ == "__main__":
    test_instruction_shape()
    test_instruction_op()
