#  The MIT License (MIT)
#
#  Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the 'Software'), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

from argparse import ArgumentParser
import onnx
import os


def argparser():
    parser = ArgumentParser()
    parser.add_argument("-i", "--model_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    return parser.parse_args()


def add_nodes_as_output(model, nodes):
    shape_info = onnx.shape_inference.infer_shapes(model)
    output_nodes = [
        node for node in shape_info.graph.value_info if node.name in nodes
    ]
    model.graph.output.extend(output_nodes)
    onnx.checker.check_model(model)
    return model


def modify_model(model_path, output_path):
    model = onnx.load(model_path)
    model = add_nodes_as_output(model, ["hidden_states"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save(model, output_path)


if __name__ == "__main__":
    args = argparser()
    modify_model(**vars(args))
