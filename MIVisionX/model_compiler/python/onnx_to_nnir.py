# Copyright (c) 2018 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import *
from builtins import range
from past.utils import old_div
import os, sys
import onnx
from onnx import onnx_pb
from onnx import numpy_helper
import nnir as ir

onnx2ir_attr = {
    'axis' : 'axis',
    'axes'  : 'axes',
    'perm' : 'axes',
    'broadcast' : 'broadcast',
    'keepdims' : 'keepdims',
    'kernel_shape' : 'kernel_shape',
    'pads' : 'pads',
    'strides' : 'strides',
    'dilations' : 'dilations',
    'group' : 'group',
    'epsilon' : 'epsilon',
    'alpha' : 'alpha',
    'beta' : 'beta',
    'transA' : 'transA',
    'transB' : 'transB',
    'bias' : 'bias',
    'size' : 'size',
    'split' : 'split',
    'shape' : 'shape',
    'min' : 'min',
    'max' : 'max',
    'to' : 'to', 
    'center_point_box' : 'center_point_box',
    'value' : 'value',
    'largest' : 'largest',
    'sorted' : 'sorted',
}

onnx2ir_op_type = { 
    'Conv'               : 'conv',
    'ConvTranspose'      : 'conv_transpose',
    'BatchNormalization' : 'batch_norm',
    'AveragePool'        : 'avg_pool',
    'MaxPool'            : 'max_pool',
    'Relu'               : 'relu',
    'Sum'                : 'sum',
    'Add'                : 'add',
    'Sub'                : 'sub',
    'Mul'                : 'mul',
    'MatMul'             : 'matmul',
    'Gemm'               : 'gemm',
    'LRN'                : 'lrn',
    'Concat'             : 'concat',
    'LeakyRelu'          : 'leaky_relu',
    'Sigmoid'            : 'sigmoid',
    'GlobalAveragePool'  : 'global_avg_pool',
    'Softmax'            : 'softmax',
    'Reshape'            : 'reshape',
    'Squeeze'            : 'squeeze',
    'Unsqueeze'          : 'unsqueeze',
    'Transpose'          : 'transpose',
    'Flatten'            : 'flatten',
    'Identity'           : 'copy',
    'Min'                : 'min',
    'Max'                : 'max',
    'Div'                : 'div',
    'Exp'                : 'exp',
    'Log'                : 'log',
    'ReduceMean'         : 'global_avg_pool',
    'Clip'               : 'clamp',
    'Cast'               : 'cast',
    'Shape'              : 'shape',  
    'ArgMax'             : 'argmax',
    'NonMaxSuppression'  : 'nms',
    'Constant'           : 'constant',
    'Gather'             : 'gather',
    'TopK'               : 'topk',
    'ReduceMin'          : 'reduce_min',
    'Tile'               : 'tile',
    'Less'               : 'less',
    'Greater'            : 'greater',
    'LessOrEqual'        : 'less_equal',
    'GreaterOrEqual'     : 'greater_equal',
    'Equal'              : 'equal'
}

onnx2ir_data_type = [
    "UND_", "F032", "U008", "I008", "U016", "I016", "I032", "I064",
    "STR_", "BOOL", "F016", "F064", "U032", "U064", "C064", "C128"
]

def onnx_name_to_ir_name(name):
    return '_'.join('_'.join(('_'.join(('_'.join(name.split('/')).split('-')))).split(':')).split('.'))

def onnx_node_to_ir_attr(node):
    global onnx2ir_attr
    attr = ir.IrAttr()
    for item in node.attribute:
        if item.name in onnx2ir_attr:
            name = onnx2ir_attr[item.name]
            if item.HasField('f'):
                attr.set(name,float(item.f))
            elif item.HasField('i'):
                attr.set(name,int(item.i))
            elif item.HasField('s'):
                attr.set(name,item.s)
            elif item.HasField('t'):
                attr.set(name,numpy_helper.to_array(item.t))
            elif len(item.floats):
                attr.set(name,list(item.floats))
            elif len(item.ints):
                attr.set(name,[int(v) for v in list(item.ints)])
            elif len(item.strings):
                attr.set(name,list(item.strings))
            else:
                raise ValueError("Unsupported ONNX attribute: {}".format(item))
    if attr.is_set('output_padding'):
        output_padding = attr.get('output_padding')
        kernel_shape = attr.get('kernel_shape')
        if (kernel_shape[0] <= 1) or (kernel_shape[1] <= 1) or \
           ((output_padding[0] % (kernel_shape[0] - 1)) != 0) or \
           ((output_padding[1] % (kernel_shape[1] - 1)) != 0):
            raise ValueError("Unsupported ONNX value for output_padding attribute")
        dilations = [old_div(output_padding[0], (kernel_shape[0] - 1)) + 1, old_div(output_padding[1], (kernel_shape[1] - 1)) + 1]
        attr.set('dilations', dilations)       
    if node.op_type == 'MatMul':
        attr.set('beta', 0.0)
    return attr

def onnx_node_to_ir_node(onnx_node):
    global onnx2ir_op_type
    node = ir.IrNode()
    if onnx_node.op_type in onnx2ir_op_type:
        type = onnx2ir_op_type[onnx_node.op_type]
    else:
        print('ERROR: ONNX operation "%s" not supported yet' % (onnx_node.op_type))
        sys.exit(1)
    node.set(type, [onnx_name_to_ir_name(name) for name in onnx_node.input], \
                   [onnx_name_to_ir_name(name) for name in onnx_node.output], \
                   onnx_node_to_ir_attr(onnx_node))
    return node

def onnx_tensor_info_to_data(info, dims):
    tensor = ir.IrTensor()
    tensor.setName(onnx_name_to_ir_name(info.name))
    tensor.setInfo(onnx2ir_data_type[info.data_type], [int(x) for x in dims])
    return tensor

def onnx_value_info_to_data(info, dims):
    tensor = ir.IrTensor()
    tensor.setName(onnx_name_to_ir_name(info.name))
    tensor.setInfo(onnx2ir_data_type[info.type.tensor_type.elem_type], [int(x) for x in dims])
    return tensor

def onnx_graph_to_ir_graph(onnx_graph):
    graph = ir.IrGraph(False)
    initializerList = []
    shapeList = []
    inputUser = False
    tensorAliasList = {}

    for onnx_node in onnx_graph.node:
        if onnx_node.op_type == 'Reshape' and len(onnx_node.input) == 2:
            for tensor in onnx_graph.initializer:
                tensorName = onnx_name_to_ir_name(tensor.name)
                if tensor.name == onnx_node.input[1] and tensorName not in shapeList:
                    shapeList.append(tensorName)
                    graph.addVariable(onnx_tensor_info_to_data(tensor,numpy_helper.to_array(tensor)))
                    raw_data = numpy_helper.to_array(tensor)
                    graph.addBinary(tensorName, raw_data)
        if onnx_node.op_type == 'Dropout':
            tensorAliasList[onnx_node.output[0]] = onnx_node.input[0]
        else:
            for i in range(len(onnx_node.input)):
                if onnx_node.input[i] in tensorAliasList:
                    onnx_node.input[i] = tensorAliasList[onnx_node.input[i]]
            node = onnx_node_to_ir_node(onnx_node)
            graph.addNode(node)
    for tensor in onnx_graph.initializer:
        tensorName = onnx_name_to_ir_name(tensor.name)
        if tensorName not in shapeList:
            initializerList.append(tensorName)
            graph.addVariable(onnx_tensor_info_to_data(tensor, tensor.dims))
            raw_data = numpy_helper.to_array(tensor)
            graph.addBinary(tensorName, raw_data)
    for tensor in onnx_graph.input:
        if onnx_name_to_ir_name(tensor.name) not in initializerList and onnx_name_to_ir_name(tensor.name) not in shapeList:
            input_dims = [int(x.dim_value) for x in tensor.type.tensor_type.shape.dim]
            if (len(sys.argv) > 3) and (sys.argv[3] == "--input_dims"):
                if (x == 0 or x is None or x == '?' for x in input_dims):
                    input_dims = sys.argv[4].split(',')
                    inputUser = True
            graph.addInput(onnx_value_info_to_data(tensor, input_dims))
    for tensor in onnx_graph.output:
        output_dims = [int(x.dim_value) for x in tensor.type.tensor_type.shape.dim]
        if (x == 0 or x is None or x == '?' for x in output_dims):
            if inputUser is True:
                output_dims[0] = input_dims[0]
        while len(output_dims) != 4:
            output_dims.append(1)
        graph.addOutput(onnx_value_info_to_data(tensor, output_dims))
    graph.updateLocals()
    return graph

def onnx2ir(model, output_folder, node_type_append):
    # get graph from ONNX model
    if isinstance(model, str):
        onnx_model = onnx.load(model)
    elif isinstance(model, onnx.ModelProto):
        onnx_model = model
    else:
        raise TypeError("Model must be file path to .onnx file or onnx loaded model")
    graph = onnx_graph_to_ir_graph(onnx_model.graph)
    graph.toFile(output_folder, node_type_append)

def main():
    if len(sys.argv) < 3:
        print('Usage: python onnx_to_nnir.py <onnxModel> <nnirOutputFolder> [--input_dims n,c,h,w (optional)] [--node_type_append 0/1 (optional: appends node type to output tensor name)]')
        sys.exit(1)
    onnxFileName = sys.argv[1]
    outputFolder = sys.argv[2]
    #appends node type to output tensor name. 
    node_type_append = 0
    pos = 4
    while pos < len(sys.argv)  and len(sys.argv) > 3 and sys.argv[pos][:2] == '--':
        if sys.argv[pos] == '--node_type_append':
            node_type_append = int(sys.argv[pos+1])
            pos = pos + 2
        elif sys.argv[pos] == '--input_dims':
            #input_dims = sys.argv[pos+1]
            pos = pos + 2
    print('loading ONNX model from %s ...' % (onnxFileName))
    onnx_model_proto = onnx_pb.ModelProto()
    if not os.path.isfile(onnxFileName):
        print('ERROR: unable to open: ' + onnxFileName)
        sys.exit(1)
    onnx_model_proto.ParseFromString(open(onnxFileName, 'rb').read())
    print('converting to IR model in %s ...' % (outputFolder))
    onnx2ir(onnx_model_proto, outputFolder, node_type_append)

if __name__ == '__main__':
    main()