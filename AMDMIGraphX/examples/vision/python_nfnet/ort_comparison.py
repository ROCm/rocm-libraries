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
import numpy
import onnxruntime as rt

sess = rt.InferenceSession("dm_nfnet_f0.onnx")

input_name = sess.get_inputs()[0].name
print("input name", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape", input_shape)
input_type = sess.get_inputs()[0].type
print("input type", input_type)

output_name = sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
output_type = sess.get_outputs()[0].type
print("output type", output_type)

x = numpy.random.random((1, 3, 192, 192))
x = x.astype(numpy.float32)

import migraphx

model = migraphx.parse_onnx("dm_nfnet_f0.onnx")
model.compile(migraphx.get_target("gpu"))
print(model.get_parameter_names())
print(model.get_parameter_shapes())
print(model.get_output_shapes())

result_migraphx = model.run({"inputs": x})
result_ort = sess.run([output_name], {input_name: x})

result_migraphx = result_migraphx[0].tolist()

for i in range(10):
    x = numpy.random.random((1, 3, 192, 192))
    x = x.astype(numpy.float32)

    result_migraphx = model.run({"inputs": x})
    result_ort = sess.run([output_name], {input_name: x})

    try:
        numpy.testing.assert_allclose(result_migraphx[0].tolist(),
                                      result_ort[0][0],
                                      rtol=1e-02)
        print(f"Test #{i} completed.")
    except AssertionError as e:
        print(e)
