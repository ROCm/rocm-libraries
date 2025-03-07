
################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
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
#
################################################################################

import matrix_utils
import msgpack


def test_input_basic(tmp_path_factory):
    x = 16
    y = 128
    test_val = {"sizes": [x, y], "data": list(range(x * y))}

    out_dir = tmp_path_factory.mktemp("test_matrixIO_input")
    f = out_dir / "data.msgpack"
    f.write_bytes(msgpack.packb(test_val))

    test = matrix_utils.loadMatrix(f)

    assert test.shape == (16, 128)
    assert test[0, 0] == 0
    assert test[0, 1] == 16
    assert test[1, 0] == 1
