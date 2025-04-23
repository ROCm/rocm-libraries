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
#
#####################################################################################
import os
import onnxruntime as ort

from ..utils import get_model_io, numpy_to_pb


def _inference(session, input_data_map, outputs, folder_name):
    input_pb_name = "input_{}.pb"
    output_pb_name = "output_{}.pb"
    os.makedirs(folder_name, exist_ok=True)
    for input_idx, (input_name,
                    input_data) in enumerate(input_data_map.items()):
        numpy_to_pb(input_name, input_data,
                    f"{folder_name}/{input_pb_name.format(input_idx)}")

    ort_result = session.run(outputs, input_data_map)
    output_data_map = {
        output_name: result_data
        for (output_name, result_data) in zip(outputs, ort_result)
    }
    for output_idx, (output_name,
                     result_data) in enumerate(output_data_map.items()):
        numpy_to_pb(output_name, result_data,
                    f"{folder_name}/{output_pb_name.format(output_idx)}")

    return output_data_map


def generate_test_dataset(model,
                          dataset,
                          output_folder_prefix=None,
                          sample_limit=None,
                          decode_limit=None):
    assert model.is_diffuser() == False, "Only base and decoder supported"
    output_path = f"{output_folder_prefix or 'generated'}/{dataset.name()}/{model.name()}"
    folder_name_prefix = f"{output_path}/test_data_set"

    model_path = model.get_model(output_path)
    inputs, outputs = get_model_io(model_path)

    sess = ort.InferenceSession(model_path)
    print(f"Creating {folder_name_prefix}s...")
    test_idx = 0
    for idx, data in enumerate(dataset):
        input_data_map = dataset.transform(inputs, data, model.preprocess)
        is_eos, decode_idx = False, 0
        while not is_eos and not (decode_limit and decode_limit <= decode_idx):
            folder_name = f"{folder_name_prefix}_{test_idx}"

            output_data_map = _inference(sess, input_data_map, outputs,
                                         folder_name)

            is_eos = not model.is_decoder() or model.decode_step(
                input_data_map, output_data_map)
            test_idx += 1
            decode_idx += 1

        if sample_limit and sample_limit - 1 <= idx:
            break
