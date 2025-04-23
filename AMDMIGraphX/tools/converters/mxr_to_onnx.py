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
import onnx
from onnx import helper, TensorProto, checker
import numpy as np
import os
import argparse


# Utility function to map MIGraphX types to ONNX data types
def get_dtype(instruction):
    type_mapping = {
        'float_type': TensorProto.FLOAT,
        'bf16_type': TensorProto.BFLOAT16,
        'half_type': TensorProto.FLOAT16
    }
    return type_mapping[instruction.shape().type_string()]


# Utility function to get the shape of an instruction
def get_shape(instruction):
    if isinstance(instruction, list):
        raise ValueError("Expected instruction, got a list.")
    return instruction.shape().lens()


# Utility function to map MIGraphX operations to ONNX operations
def map_operation(operation):
    mxr_to_onnx_op = {
        "dot": "MatMul",
        "mul": "Mul",
        "add": "Add",
        "multibroadcast": "Expand",
        "erf": "Erf",
        "tanh": "Tanh",
        "exp": "Exp",
        "div": "Div",
        "relu": "Relu",
        "pow": "Pow"
    }

    if operation not in mxr_to_onnx_op:
        raise NotImplementedError(f"Operation '{operation}' is not supported.")
    return mxr_to_onnx_op[operation]


# Helper function to create ONNX nodes for specific operations
def create_node(instruction, parameters, node_name, n, initializers):
    if node_name == "multibroadcast" or node_name == "reshape":
        shape_key = "out_lens" if node_name == "multibroadcast" else "dims"
        shape_array = np.array(parameters[shape_key], dtype=np.int64)
        initializer_name = f"{node_name}_shape_{n}"

        initializers.append(
            helper.make_tensor(name=initializer_name,
                               data_type=TensorProto.INT64,
                               dims=shape_array.shape,
                               vals=shape_array.flatten().tolist()))
        return helper.make_node(
            map_operation(node_name),
            inputs=[str(hash(i))
                    for i in instruction.inputs()] + [initializer_name],
            outputs=[str(hash(instruction))])

    elif node_name == "transpose":
        return helper.make_node(
            "Transpose",
            inputs=[str(hash(i)) for i in instruction.inputs()],
            outputs=[str(hash(instruction))],
            perm=parameters["permutation"])

    elif node_name == "convolution":
        return helper.make_node(
            "Conv",
            inputs=[str(hash(i)) for i in instruction.inputs()],
            outputs=[str(hash(instruction))
                     ],  #[str(hash(i)) for i in instruction.outputs()],
            dilations=parameters["dilation"],
            group=parameters["group"],
            pads=parameters["padding"],
            strides=parameters["stride"])

    return helper.make_node(
        map_operation(node_name),
        inputs=[str(hash(i)) for i in instruction.inputs()],
        outputs=[str(hash(instruction))])


# Main function to convert MIGraphX module to ONNX model
def generate_onnx(module):
    inputs = {}
    operations = []
    initializers = []
    n = 0  # Node counter
    output = None

    for instruction in module:
        op_name = instruction.op().name()

        # Handle input nodes
        if op_name in ["@literal", "@param"]:

            inputs[str(hash(instruction))] = helper.make_tensor_value_info(
                str(hash(instruction)), get_dtype(instruction),
                get_shape(instruction))

        # Handle computational nodes
        elif "@" not in op_name:
            n += 1
            parameters = instruction.op().values()

            operations.append(
                create_node(instruction, parameters, op_name, n, initializers))

        # Handle return node
        elif op_name == "@return":

            output = [
                helper.make_tensor_value_info(str(hash(i)), get_dtype(i),
                                              get_shape(i))
                for i in instruction.inputs()
            ]

    # Create the ONNX graph
    graph = helper.make_graph(nodes=operations,
                              name="Graph",
                              inputs=list(inputs.values()),
                              initializer=initializers,
                              outputs=output if output else [])

    return helper.make_model(graph, producer_name="onnx-dot-add-example")


# Main function to process MIGraphX files and generate ONNX models
def main(mxr_directory_path, onnx_directory_path):
    for file_name in os.listdir(mxr_directory_path):
        file_path = os.path.join(mxr_directory_path, file_name)
        if ".mxr" in file_path:
            try:
                program = migraphx.load(file_path)
                module = program.get_main_module()
                model = generate_onnx(module)

                # Validate the generated ONNX model
                try:
                    checker.check_model(model)
                    print(f"ONNX model for {file_path} is valid.")
                except onnx.checker.ValidationError as e:
                    print(f"Validation failed for {file_path}: {e}")
                except Exception as e:
                    print(
                        f"Unexpected error during validation for {file_path}: {e}"
                    )

                os.makedirs(onnx_directory_path, exist_ok=True)
                onnx_file_path = os.path.join(onnx_directory_path,
                                              file_name.replace("mxr", "onnx"))
                onnx.save(model, onnx_file_path)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process MXR files and generate ONNX models.")
    parser.add_argument("mxr_directory_path",
                        type=str,
                        help="Path to the directory containing MXR files.")
    parser.add_argument(
        "onnx_directory_path",
        type=str,
        help="Path to the directory where ONNX models will be saved.")

    args = parser.parse_args()
    main(args.mxr_directory_path, args.onnx_directory_path)
