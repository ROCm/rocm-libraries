#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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


def test_create_shape():
    s = migraphx.shape(lens=[1, 64, 3, 3])
    assert s.standard()
    assert s.packed()
    assert s.lens() == [1, 64, 3, 3]
    assert s.ndim() == 4


def test_create_shape_transposed():
    s = migraphx.shape(lens=[1, 64, 3, 3], permutation=[0, 2, 3, 1])
    assert not s.standard()
    assert s.packed()
    assert s.transposed()
    assert s.lens() == [1, 64, 3, 3]
    assert s.strides() == [64 * 3 * 3, 1, 64 * 3, 64]
    assert s.ndim() == 4


def test_create_shape_broadcast():
    s = migraphx.shape(lens=[1, 64, 3, 3], strides=[0, 1, 0, 0])
    assert s.broadcasted()
    assert s.lens() == [1, 64, 3, 3]
    assert s.strides() == [0, 1, 0, 0]


def test_create_shape_type():
    s = migraphx.shape(type='int64_t')
    assert s.type_string() == 'int64_type'
    assert s.type_size() == 8
    s = migraphx.shape(type='uint8_t')
    assert s.type_string() == "uint8_type"
    assert s.type_size() == 1
    s = migraphx.shape(type='float')
    assert s.type_size() == 4


def test_create_dyn_dims():
    a = migraphx.shape.dynamic_dimension()
    assert a.is_fixed()
    assert a.min == 0
    b = migraphx.shape.dynamic_dimension(4, 4)
    assert b.is_fixed()
    assert b.max == 4
    c = migraphx.shape.dynamic_dimension(1, 4, {2, 4})
    assert not c.is_fixed()
    assert c.min == 1
    assert c.max == 4
    assert c.optimals == {2, 4}

    dyn_dims = [a, b]
    dyn_dims.append(c)
    assert dyn_dims[1] == b


def test_create_dyn_shape():
    a = migraphx.shape.dynamic_dimension(1, 4, {2, 4})
    b = migraphx.shape.dynamic_dimension(4, 4)
    dds = [a, b]
    dyn_shape = migraphx.shape(type='float', dyn_dims=dds)
    assert dyn_shape.dynamic()
    assert dyn_shape.dyn_dims()[0].min == dds[0].min
    assert dyn_shape.dyn_dims()[0].max == dds[0].max
    assert dyn_shape.dyn_dims()[0].optimals == dds[0].optimals


def test_type_enum():
    mgx_types = [
        'bool_type', 'double_type', 'float_type', 'half_type', 'int16_type',
        'int32_type', 'int64_type', 'int8_type', 'uint16_type', 'uint32_type',
        'uint64_type', 'uint8_type'
    ]
    for t in mgx_types:
        assert hasattr(migraphx.shape.type_t, t)


if __name__ == "__main__":
    test_create_shape()
    test_create_shape_broadcast()
    test_create_shape_type()
    test_create_dyn_dims()
    test_create_dyn_shape()
