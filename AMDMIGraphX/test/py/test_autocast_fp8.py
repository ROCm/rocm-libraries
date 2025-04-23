#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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


def test_autocast_fp8_1():
    p1 = migraphx.program()
    m1 = p1.get_main_module()
    x = m1.add_parameter("x", shape=migraphx.shape(type='fp8e4m3fnuz_type'))
    y = m1.add_parameter("y", shape=migraphx.shape(type='fp8e4m3fnuz_type'))
    sum_op = m1.add_instruction(migraphx.op("add"), [x, y])
    m1.add_return([sum_op])

    m1 = migraphx.autocast_fp8(p1)

    p2 = migraphx.program()
    m2 = p2.get_main_module()
    y_fp32 = m2.add_parameter("y", shape=migraphx.shape(type='float_type'))
    x_fp32 = m2.add_parameter("x", shape=migraphx.shape(type='float_type'))

    y_fp8 = m2.add_instruction(
        migraphx.op("convert",
                    target_type=int(migraphx.shape.type_t.fp8e4m3fnuz_type)),
        [y_fp32])
    x_fp8 = m2.add_instruction(
        migraphx.op("convert",
                    target_type=int(migraphx.shape.type_t.fp8e4m3fnuz_type)),
        [x_fp32])

    sum_fp8 = m2.add_instruction(migraphx.op("add"), [x_fp8, y_fp8])
    sum_fp32 = m2.add_instruction(
        migraphx.op("convert",
                    target_type=int(migraphx.shape.type_t.float_type)),
        [sum_fp8])

    m2.add_return([sum_fp32])
    assert p1 == p2


def test_autocast_fp8_2():
    p1 = migraphx.program()
    m1 = p1.get_main_module()
    x = m1.add_parameter("x", shape=migraphx.shape(type='float_type'))
    y = m1.add_parameter("y", shape=migraphx.shape(type='float_type'))
    sum = m1.add_instruction(migraphx.op("add"), [x, y])
    m1.add_return([sum])

    m1 = migraphx.autocast_fp8(p1)

    p2 = p1
    assert p1 == p2


if __name__ == "__main__":
    test_autocast_fp8_1()
    test_autocast_fp8_2()
