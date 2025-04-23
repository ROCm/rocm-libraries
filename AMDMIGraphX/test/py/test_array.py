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
import migraphx, struct, array, sys
try:
    from functools import reduce
except:
    pass


def assert_eq(x, y):
    if x == y:
        pass
    else:
        raise Exception(str(x) + " != " + str(y))


def read_float(b, index):
    return struct.unpack_from('f', b, index * 4)[0]


def write_float(b, index):
    struct.pack_into('f', b, index * 4)


def nelements(lens):
    return reduce(lambda x, y: x * y, lens, 1)


def create_buffer(t, data, shape):
    a = array.array(t, data)
    if sys.version_info >= (3, 0):
        m = memoryview(a.tobytes())
        return m.cast(t, shape)
    else:
        m = memoryview(a.tostring())
        return m


def check_argument(a):
    l = a.tolist()
    for i in range(len(l)):
        assert_eq(l[i], read_float(a, i))


def check_shapes(r, m):
    lens = list(m.shape)
    strides = [int(s / m.itemsize) for s in m.strides]
    elements = nelements(lens)
    assert_eq(r.get_shape().elements(), elements)
    assert_eq(r.get_shape().lens(), lens)
    assert_eq(r.get_shape().strides(), strides)


def run(p):
    params = {}
    for key, value in p.get_parameter_shapes().items():
        params[key] = migraphx.generate_argument(value)

    return p.run(params)


def test_shape(shape):
    data = list(range(nelements(shape)))
    m = create_buffer('f', data, shape)
    a = migraphx.argument(m)
    check_shapes(a, m)
    assert_eq(a.tolist(), data)


def test_input():
    if sys.version_info >= (3, 0):
        test_shape([4])
        test_shape([2, 3])
    else:
        data = list(range(4))
        m = create_buffer('f', data, [4])
        a1 = migraphx.argument(m)
        a2 = migraphx.argument(bytearray(a1))
        check_shapes(a2, m)
        assert_eq(a1.tolist(), m.tolist())


def test_output():
    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx")
    p.compile(migraphx.get_target("gpu"))

    r1 = run(p)[-1]
    r2 = run(p)[-1]
    assert_eq(r1, r2)
    assert_eq(r1.tolist(), r2.tolist())

    check_argument(r1)
    check_argument(r2)

    m1 = memoryview(r1)
    m2 = memoryview(r2)

    check_shapes(r1, m1)
    check_shapes(r2, m2)


test_input()
test_output()
