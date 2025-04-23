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
import onnx
import logging
import os
import requests


def get_model_io(model_path):
    model = onnx.load(model_path)
    outputs = [node.name for node in model.graph.output]

    input_all = [node.name for node in model.graph.input]
    input_initializer = [node.name for node in model.graph.initializer]
    inputs = list(set(input_all) - set(input_initializer))

    return inputs, outputs


def numpy_to_pb(name, np_data, out_filename):
    """Convert numpy data to a protobuf file."""

    tensor = onnx.numpy_helper.from_array(np_data, name)
    onnx.save_tensor(tensor, out_filename)


def download(url, filename, quiet=False):
    try:
        from tqdm import tqdm
    except:
        quiet = True

    logging.debug(f"Download {filename} from {url}")
    response = requests.get(url, stream=True)

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    if not quiet:
        progress_bar = tqdm(total=total_size_in_bytes,
                            unit="iB",
                            unit_scale=True)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            if not quiet:
                progress_bar.update(len(data))
            f.write(data)
    if not quiet:
        progress_bar.close()


def get_imagenet_classes():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    return response.text.split("\n")
