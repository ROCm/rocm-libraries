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
import argparse
import onnx
from onnx import version_converter


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'MIGraphX Onnx Model Convertion. Use to convert the opset of the input model to MIGraphX\'s'
    )
    req_args = parser.add_argument_group(title='required arguments')
    req_args.add_argument('--model',
                          type=str,
                          required=True,
                          help='path to onnx file')
    req_args.add_argument('--output',
                          type=str,
                          required=True,
                          help='path to output onnx file')
    req_args.add_argument('--opset',
                          type=int,
                          required=True,
                          help='The output opset')
    req_args.add_argument('--infer_shapes',
                          action='store_true',
                          help='Infer shapes for output model')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='show verbose information (for debugging)')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    model_path = args.model
    out_model_path = args.output
    target_opset = args.opset
    verbose = args.verbose
    infer_shapes = args.infer_shapes

    original_model = onnx.load(model_path)
    if verbose:
        print(f"The model before conversion:\n{original_model}")

    # A full list of supported adapters can be found here:
    # https://github.com/onnx/onnx/blob/main/onnx/version_converter.py#L21
    # Apply the version conversion on the original model
    converted_model = version_converter.convert_version(
        original_model, target_opset)

    if infer_shapes:
        converted_model = onnx.shape_inference.infer_shapes(converted_model)

    if verbose:
        print(f"The model after conversion:\n{converted_model}")

    # Save the ONNX model
    onnx.save(converted_model, out_model_path)


if __name__ == '__main__':
    main()
