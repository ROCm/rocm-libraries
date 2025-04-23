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


def test_add_fill():
    p = migraphx.program()
    mm = p.get_main_module()
    x = mm.add_literal(
        migraphx.fill_argument(migraphx.shape(type='float_type', lens=[3, 3]),
                               1))
    y = mm.add_literal(
        migraphx.fill_argument(migraphx.shape(type='float_type', lens=[3, 3]),
                               2))
    add_op = mm.add_instruction(migraphx.op("add"), [x, y])
    mm.add_return([add_op])
    p.compile(migraphx.get_target("ref"))
    params = {}
    output = p.run(params)[-1].tolist()
    assert output == list([3.0] * 9)


def test_add_create():
    p = migraphx.program()
    mm = p.get_main_module()
    x = mm.add_literal(
        migraphx.create_argument(
            migraphx.shape(type='float_type', lens=[2, 2]), [1, 2, 3, 4]))
    y = mm.add_literal(
        migraphx.create_argument(
            migraphx.shape(type='float_type', lens=[2, 2]), [5, 6, 7, 8]))
    add_op = mm.add_instruction(migraphx.op("add"), [x, y])
    mm.add_return([add_op])
    p.compile(migraphx.get_target("ref"))
    params = {}
    output = p.run(params)[-1].tolist()
    assert output == list([6, 8, 10, 12])


if __name__ == "__main__":
    test_add_fill()
    test_add_create()
