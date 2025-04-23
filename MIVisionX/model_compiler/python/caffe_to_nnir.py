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
from builtins import str
from builtins import range
from past.utils import old_div
import os
import caffe_pb2
import nnir as ir
import sys
#import argparse
import struct
import math
import collections

# mapping from caffe layer types to nnir operators.
# pooling is mapped to either avg_pool or max_pool
# scale is fused to batch_norm if its previous layer is batch_norm or fused to mul or muladd in nnir.
caffe2ir_op_type = {
    'Convolution': 'conv',
    'Deconvolution': 'conv_transpose',
    'BatchNorm' : 'batch_norm',
    'InnerProduct' : 'gemm',
    'ReLU' : 'relu',
    'LRN' : 'lrn',
    'Eltwise' : 'sum',
    'Concat' : 'concat',
    'Softmax' : 'softmax',
    'SoftmaxWithLoss' : 'softmax',
    'Interp' : 'upsample',
    'Crop' : 'crop',
    'Permute' : 'permute',
    'PriorBox' : 'prior_box',
    'Flatten' : 'flatten',
    'Reshape' : 'reshape',
    'DetectionOutput' : 'detection_output',
}

# convert caffename to ir names.
def caffe_name_to_ir_name(name):
    return '_'.join(('_'.join(name.split('/')).split('-')))

# convert caffe blobs to ir tensor.
def caffe_blob_to_ir_tensor(blob_name, blob_data_type, blob_shape):
    tensor = ir.IrTensor()
    tensor.setName(caffe_name_to_ir_name(blob_name))
    tensor.setInfo(blob_data_type, [int(x) for x in blob_shape])
    return tensor

# convert caffe bin formats to ir bin formats.
def convert_caffe_bin_to_ir_bin(floatlist):
    buf = struct.pack('%sf' % len(floatlist), *floatlist)
    return buf

# map caffe attr to ir attr.
def caffe_attr_to_ir_attr(attribute_map):
    attr = ir.IrAttr()
    attr_names = list(attribute_map.keys())
    for i in range(len(attr_names)):
        attributeInfo = attribute_map[attr_names[i]]
        if type(attributeInfo) is float:
            attr.set(attr_names[i], float(attributeInfo))
        elif type(attributeInfo) is int:
            attr.set(attr_names[i], int(attributeInfo))
        elif type(attributeInfo) is str:
            attr.set(attr_names[i], str(attributeInfo)) 
        elif type(attributeInfo) == type([]):
            if (isinstance(attributeInfo[0], int)):
                attr.set(attr_names[i], [int(v) for v in (attributeInfo)])
            elif (isinstance(attributeInfo[0], float)):
                attr.set(attr_names[i], [float(v) for v in (attributeInfo)])
            elif (isinstance(attributeInfo[0], int)):
                attr.set(attr_names[i], [int(v) for v in (attributeInfo)])
            else:
                print ("ERROR: unsupported list attribute")
                sys.exit(1)
        else:
            print ("ERROR: Unsupported type of caffe attribute %s" % attr_names[i])
            sys.exit(1)
    return attr

# map caffe node to ir node.
def caffe_node_to_ir_node(layer_type, layer_info_map):
    node = ir.IrNode()
    input_map = layer_info_map["inputs"]
    output_map = layer_info_map["outputs"]
    weight_map = {}
    scale_map_w = {}
    scale_map_b = {}

    if ("scale_weights" in layer_info_map):
        scale_map_w = layer_info_map["scale_weights"]
    if ("scale_bias" in layer_info_map):
        scale_map_b = layer_info_map["scale_bias"]
    if "weights" in layer_info_map:
        weight_map = layer_info_map["weights"]
    bias_map = {}
    if "biases" in layer_info_map:
        bias_map = layer_info_map["biases"]
    attribute_map = layer_info_map["attributes"]

    inputs = []
    for i in range(len(list(input_map.keys()))):
        inputs.append(list(input_map.keys())[i])
    for i in range(len(scale_map_w)):
        inputs.append(list(scale_map_w.keys())[i])
    for i in range(len(scale_map_b)):
        inputs.append(list(scale_map_b.keys())[i])
    for i in range(len(list(weight_map.keys()))):
        inputs.append(list(weight_map.keys())[i])
    for i in range(len(list(bias_map.keys()))):
        inputs.append(list(bias_map.keys())[i])

    outputs = []
    for i in range(len(list(output_map.keys()))):
        outputs.append(list(output_map.keys())[i])

    node.set(layer_type, [caffe_name_to_ir_name(name) for name in inputs],\
                         [caffe_name_to_ir_name(name) for name in outputs],\
                         caffe_attr_to_ir_attr(attribute_map))
    return node

# extract binary data from caffe layers if present.
def extractBinary(layer_parameter, graph, verbose):
    layer_name = caffe_name_to_ir_name(layer_parameter.name)
    if (verbose):
        print ("Extracting binaries from : "  + layer_name)

    ## dump weights and biases if present.
    blob_size = len(layer_parameter.blobs)
    if blob_size > 0:
        weight_blob_proto = layer_parameter.blobs[0]
        weight_blob_name = caffe_name_to_ir_name(layer_name + '_w')
        if (verbose):
            print (weight_blob_name)
        buf = convert_caffe_bin_to_ir_bin(weight_blob_proto.data)
        graph.addBinary(weight_blob_name, buf)

    if blob_size > 1:
        bias_blob_proto = layer_parameter.blobs[1]
        bias_blob_name = caffe_name_to_ir_name(layer_name + '_b')
        if (verbose):
            print (bias_blob_name)
        buf = convert_caffe_bin_to_ir_bin(bias_blob_proto.data)
        graph.addBinary(bias_blob_name, buf)

# extracting input from caffe network and converting into ir input.
def extractInput(net_parameter, graph, input_dims):
    inputList = {}
    if (len(net_parameter.layer) == 0):
        layers = net_parameter.layers
    else:
        layers = net_parameter.layer
    first_layer_param = layers[0]
    first_layer_param_type = first_layer_param.type
    input_name = ""
    if len(net_parameter.input) != 0:
        input_name = caffe_name_to_ir_name(net_parameter.input[0])
    elif (first_layer_param_type == "Data" or first_layer_param_type == "Input" or first_layer_param_type == "ImageData"):
        top_list = first_layer_param.top
        if (len(top_list) == 0):
            input_name = caffe_name_to_ir_name(first_layer_param.name)
        else:
            input_name = caffe_name_to_ir_name(top_list[0])
    else:
        bottom_list = first_layer_param.bottom
        if (len(bottom_list) == 0):
            top_list = first_layer_param.top
            input_name = caffe_name_to_ir_name(top_list[0])
        else:
            input_name = caffe_name_to_ir_name(bottom_list[0])

    inputList[str(input_name)] = input_dims
    graph.addInput(caffe_blob_to_ir_tensor(input_name, "F032", input_dims))
    return inputList

# extraction of output from caffe network to ir output.
def extractOutput(graph, inputOutputLayers, output_list, verbose):
    outputList = {}
    if (len(output_list) == 1):
        last_layer_index = len(inputOutputLayers) - 1
        last_layer_info = inputOutputLayers[last_layer_index]
        output_map = last_layer_info["outputs"]
        output_name = list(output_map.keys())[0]
        if (verbose):
            print ("output name is : " + output_name)
        output_dims = output_map[output_name]
        graph.addOutput(caffe_blob_to_ir_tensor(output_name, "F032", output_dims))
        outputList[output_name] = output_dims
    else:
        for i in range(len(output_list)):
            output_name = output_list[i]
            if (verbose):
                print ("output name at index: "+ str(i) + " " + output_name)
            for j in range(len(inputOutputLayers)):
                if (output_name in inputOutputLayers[j]["layer_name"]):
                    output_map = inputOutputLayers[j]["outputs"]
                    output_dims = output_map[output_name]
                    graph.addOutput(caffe_blob_to_ir_tensor(output_name, "F032", output_dims))
                    outputList[output_name] = output_dims
                    break
    return outputList

# extract layer attribute information from caffe layers.
def extractCaffeAttrInfo(layer_param):
    if(type(layer_param) == caffe_pb2.V1LayerParameter):
        layer_type = convertV1LayerTypeToString(layer_param)
    else:
        layer_type = layer_param.type
    attribute_map = {}
    if (layer_type == "Convolution" or layer_type == "Deconvolution"):
        conv = layer_param.convolution_param
        pad_h = conv.pad_h if (conv.HasField('pad_h')) else (int(conv.pad[0]) if (len(conv.pad) > 0) else 0)
        pad_w = conv.pad_w if (conv.HasField('pad_w')) else (int(conv.pad[1]) if (len(conv.pad) > 1) else pad_h)
        stride_h = conv.stride_h if (conv.HasField('stride_h')) else (int(conv.stride[0]) if (len(conv.stride) > 0) else 1)
        stride_w = conv.stride_w if (conv.HasField('stride_w')) else (int(conv.stride[1]) if (len(conv.stride) > 1) else stride_h)
        kernel_h = conv.kernel_h if (conv.HasField('kernel_h')) else (int(conv.kernel_size[0]) if (len(conv.kernel_size) > 0) else 0)
        kernel_w = conv.kernel_w if (conv.HasField('kernel_w')) else (int(conv.kernel_size[1]) if (len(conv.kernel_size) > 1) else kernel_h)
        dilation_h = conv.dilation[0] if (len(conv.dilation) > 0) else 1
        dilation_w = conv.dilation[1] if (len(conv.dilation) > 1) else dilation_h
        groups = conv.group if (conv.HasField('group')) else 1

        attribute_map["strides"] = [stride_w, stride_h]
        attribute_map["kernel_shape"] = [kernel_w, kernel_h]
        attribute_map["group"] = groups
        attribute_map["pads"] = [pad_w, pad_h, pad_w, pad_h]
        attribute_map["dilations"] = [dilation_w, dilation_h]

    elif (layer_type == "Pooling"):
        pooling = layer_param.pooling_param
        pad_h = int(pooling.pad_h) if (pooling.HasField('pad_h')) else int(pooling.pad)
        pad_w = int(pooling.pad_w) if (pooling.HasField('pad_w')) else int(pooling.pad)
        stride_h = int(pooling.stride_h) if (pooling.HasField('stride_h')) else int(pooling.stride)
        stride_w = int(pooling.stride_w) if (pooling.HasField('stride_w')) else int(pooling.stride)
        kernel_h = int(pooling.kernel_h) if (pooling.HasField('kernel_h')) else int(pooling.kernel_size)
        kernel_w = int(pooling.kernel_w) if (pooling.HasField('kernel_w')) else int(pooling.kernel_size)

        attribute_map["strides"] = [stride_w, stride_h]
        attribute_map["kernel_shape"] = [kernel_w, kernel_h]
        attribute_map["pads"] = [pad_w, pad_h, pad_w, pad_h]
        attribute_map["dim_round_mode"] = "ceil"
        #attribute_map["dilations"] = [1,1]

    elif (layer_type == "LRN"):
        lrn = layer_param.lrn_param
        local_size = int(lrn.local_size)
        alpha = float(lrn.alpha)
        beta = float(lrn.beta)
        k = float(lrn.k)
        norm_region = lrn.norm_region

        attribute_map["alpha"] = alpha
        attribute_map["beta"] = beta
        attribute_map["size"] = local_size
        attribute_map["bias"] = k
        if (norm_region == caffe_pb2.LRNParameter.ACROSS_CHANNELS):
            attribute_map["mode"] = 1
        elif (norm_region == caffe_pb2.LRNParameter.WITHIN_CHANNEL):
            attribute_map["mode"] = 0

    elif (layer_type == "BatchNorm"):
        attribute_map["epsilon"] = float(layer_param.batch_norm_param.eps)

    elif (layer_type == "InnerProduct"):
        attribute_map["broadcast"] = 1
        attribute_map["transB"] = 1
    elif (layer_type == "ReLU"):
        relu = layer_param.relu_param
        slope = relu.negative_slope
        attribute_map["alpha"] = slope
    elif (layer_type == "Interp"):
        if layer_param.python_param.param_str != '':
            python_param_str = eval(layer_param.python_param.param_str)
            zoom_factor = int(python_param_str["zoom_factor"])
        else:
            zoom_factor = 2  #default value
        attribute_map["zoom_factor"] = zoom_factor
    elif (layer_type == "Crop"):
        crop = layer_param.crop_param
        axis = crop.axis if (crop.HasField('axis')) else 2
        offset = crop.offset
        new_offset = []

        for i in range(4):
            if (i < axis):
                new_offset.append(0)
            else:
                if (len(offset) == 1):
                    new_offset.append(offset[0])    
                else:
                    new_offset.append(offset[i-axis])

        attribute_map["axis"] = axis
        attribute_map["offset"] = new_offset

    elif (layer_param.type == "Reshape"):
        reshape = layer_param.reshape_param
        shape = reshape.shape.dim
        new_shape = [int(z) for z in shape]
        attribute_map["shape"] = new_shape

    elif (layer_param.type == "Concat"):
        concat = layer_param.concat_param
        axis = concat.axis
        attribute_map["axis"] = axis

    elif (layer_param.type == "DetectionOutput"):
        detection_output = layer_param.detection_output_param
        num_classes = detection_output.num_classes
        share_location = detection_output.share_location
        background_label_id = detection_output.background_label_id
        nms_threshold = detection_output.nms_param.nms_threshold
        top_k = detection_output.nms_param.top_k
        code_type = detection_output.code_type
        variance_encoded_in_target = detection_output.variance_encoded_in_target
        keep_top_k = detection_output.keep_top_k
        confidence_threshold = detection_output.confidence_threshold
        attribute_map["num_classes"] = num_classes
        attribute_map["share_location"] = 1 if share_location == True else 0
        attribute_map["background_label_id"] = background_label_id
        attribute_map["nms_threshold"] = nms_threshold
        attribute_map["top_k"] = top_k
        attribute_map["code_type"] = code_type
        attribute_map["variance_encoded_in_target"] = 1 if variance_encoded_in_target == True else 0
        attribute_map["keep_top_k"] = keep_top_k
        attribute_map["confidence_threshold"] = confidence_threshold
        
    elif (layer_param.type == "Softmax"):
        softmax = layer_param.softmax_param
        axis = softmax.axis
        attribute_map["axis"] = axis
        
    return attribute_map

# calculate dimensions of the output of each layer.
def calculateTensorDims(layer_param, input_map, attribute_map):
    dimList = {}
    output_dims = [0, 0, 0, 0]
    inputs = list(input_map.keys())
    if(type(layer_param) == caffe_pb2.V1LayerParameter):
        layer_type = convertV1LayerTypeToString(layer_param)
    else:
        layer_type = layer_param.type
    if(layer_type == "Convolution"):
        strides = attribute_map["strides"]
        pads = attribute_map["pads"]
        dilations = attribute_map["dilations"]
        kernel_shape = attribute_map["kernel_shape"]
        group = attribute_map["group"]
        n,c,h,w = input_map[inputs[0]]
        output_dims[3] = ((int(w) + 2 * pads[0] - kernel_shape[0] - (kernel_shape[0] - 1) * (dilations[0] - 1))// strides[0]) + 1
        output_dims[2] = ((int(h) + 2 * pads[1] - kernel_shape[1] - (kernel_shape[1] - 1) * (dilations[1] - 1))// strides[1]) + 1
        output_dims[1] = layer_param.convolution_param.num_output
        output_dims[0] = n
        weight_dims = [output_dims[1], old_div(int(c),group), kernel_shape[1], kernel_shape[0]]
        dimList["weights"] = weight_dims
        if (layer_param.convolution_param.bias_term):
            bias_dims = [weight_dims[0]]
            dimList["bias"] = bias_dims

    elif (layer_type == "Deconvolution"):
        strides = attribute_map["strides"]
        pads = attribute_map["pads"]
        dilations = attribute_map["dilations"]
        kernel_shape = attribute_map["kernel_shape"]
        n,c,h,w = input_map[str(inputs[0])]

        output_dims[3] = strides[0] * (w - 1) + dilations[0] * (kernel_shape[0] - 1) + 1 - (2 * pads[0])
        output_dims[2] = strides[1] * (h - 1) + dilations[1] * (kernel_shape[1] - 1) + 1 - (2 * pads[1])
        output_dims[1] = layer_param.convolution_param.num_output
        output_dims[0] = n
        weight_dims = [output_dims[1], c, kernel_shape[1] , kernel_shape[0]]
        dimList["weights"] = weight_dims
        if (layer_param.convolution_param.bias_term):
            bias_dims = [weight_dims[0]]
            dimList["bias"] = bias_dims

    elif (layer_type == "Pooling"):
        strides = attribute_map["strides"]
        pads = attribute_map["pads"]
        kernel_shape = attribute_map["kernel_shape"]
        n,c,h,w = input_map[str(inputs[0])]
        if (layer_param.pooling_param.global_pooling):
            kernel_shape[1] = h
            kernel_shape[0] = w
            pads[0] = 0
            pads[1] = 0
            strides[0] = 1
            strides[1] = 1

        output_dims[3] = int(math.ceil(float(w + 2 * pads[0] + strides[0] - kernel_shape[0])/strides[0]))
        output_dims[2] = int(math.ceil(float(h + 2 * pads[1] + strides[1] - kernel_shape[1])/strides[1]))
        if (pads[1] > 0):
            if (output_dims[2] - 1) * strides[1] >= (h + pads[1]):
                output_dims[2] = output_dims[2] - 1
        if (pads[0] > 0):
            if (output_dims[3] - 1) * strides[0] >= (w + pads[0]):
                output_dims[3] = output_dims[3] - 1
        output_dims[1] = c
        output_dims[0] = n

    elif (layer_type == "InnerProduct"):
        n,c,h,w = input_map[str(inputs[0])]
        output_dims[3] = 1
        output_dims[2] = 1
        output_dims[1] = layer_param.inner_product_param.num_output
        output_dims[0] = n
        weight_dims = [output_dims[1], c, h, w]
        dimList["weights"] = weight_dims
        if (layer_param.inner_product_param.bias_term):
            dimList["bias"] = [weight_dims[0]]

    elif (layer_type == "Concat"):
        inputs = list(input_map.keys())
        axis = attribute_map["axis"]
        if axis == 1:
            for i in range(len(inputs)):
                n,c,h,w = input_map[inputs[i]]
                output_dims[1] += c
            n,c,h,w = input_map[inputs[0]]
            output_dims[0] = n
            output_dims[2] = h
            output_dims[3] = w
        elif axis == 2:
            for i in range(len(inputs)):
                n,c,h,w = input_map[inputs[i]]
                output_dims[2] += h
            n,c,h,w = input_map[inputs[0]]
            output_dims[0] = n
            output_dims[1] = c
            output_dims[3] = w

    elif (layer_type == "Interp"):
        inputs = list(input_map.keys())
        zoom_factor = attribute_map["zoom_factor"]
        for i in range(len(inputs)):
            n,c,h,w = input_map[inputs[i]]
        n,c,h,w = input_map[inputs[0]]
        output_dims[0] = n
        output_dims[1] = c
        output_dims[2] = h*zoom_factor
        output_dims[3] = w*zoom_factor
        #print('INFO: Found Layertype Interp with zoom '+ str(zoom_factor))

    elif (layer_type == "BatchNorm" or layer_param.type == "Scale"):
        output_dims[0], output_dims[1], output_dims[2], output_dims[3] = input_map[str(inputs[0])]
        if (len(layer_param.blobs) > 0):
            weight_dims = [output_dims[1]]
            dimList["weights"] = weight_dims
        if (len(layer_param.blobs) > 1):
            bias_dims = [output_dims[1]]
            dimList["bias"] = bias_dims
    
    elif (layer_type == "Crop"):
        inputs = list(input_map.keys())
        axis = attribute_map["axis"]
        new_axis = 3 - axis

        for i in range(4):
            if (i <= new_axis):
                output_dims[i] = input_map[inputs[0]][i]
            else:
                output_dims[i] = input_map[inputs[1]][i]

    elif (layer_type == "Permute"):
        permute = layer_param.permute_param
        order = permute.order        
        order = [int(i) for i in order]
        attribute_map["order"] = order
        n,c,h,w = input_map[str(inputs[0])]
        if order == [0, 2, 3, 1]:
            output_dims[0] = n
            output_dims[1] = h
            output_dims[2] = w
            output_dims[3] = c
        if order == [0, 1, 2, 3]:
            output_dims[0] = n
            output_dims[1] = c
            output_dims[2] = h
            output_dims[3] = w

    elif (layer_type == "PriorBox"):
        n,c,h,w = input_map[str(inputs[0])]
        prior_box = layer_param.prior_box_param
        min_size = prior_box.min_size[0]
        attribute_map["min_size"] = min_size
        max_size = prior_box.max_size[0] if prior_box.max_size else 0.0
        attribute_map["max_size"] = max_size
        aspect_ratio = []
        for i in range(len(prior_box.aspect_ratio)):
            aspect_ratio.append(prior_box.aspect_ratio[i])
        attribute_map["aspect_ratio"] = aspect_ratio
        flip = int(prior_box.flip)
        attribute_map["flip"] = flip
        clip = int(prior_box.clip)
        attribute_map["clip"] = clip
        variance = []
        for i in range(len(prior_box.variance)):
            variance.append(prior_box.variance[i])
        attribute_map["variance"] = variance
        offset = float(prior_box.offset)
        attribute_map["prior_offset"] = offset
        dim = 1 #for min_size
        dim += len(aspect_ratio)
        if max_size > 0:
            dim += 1
        if flip == 1:
            dim += len(aspect_ratio)
        output_dims[0] = 1
        output_dims[1] = 2 #for mean and variance values
        output_dims[2] = h * w * dim * 4 
        output_dims[3] = 1
        
    elif (layer_type == "Flatten"):
        flatten = layer_param.flatten_param 
        axis = flatten.axis
        attribute_map["axis"] = axis
        n,c,h,w = input_map[str(inputs[0])]
        output_dims[0] = n
        output_dims[1] = c*h*w
        output_dims[2] = 1
        output_dims[3] = 1
    elif (layer_type == "Reshape"):
        shape = attribute_map["shape"]
        input_shape = input_map[str(inputs[0])]
        input_shape = [int(z) for z in input_shape]
        
        icount = 1
        ocount = 1

        for dim in range(len(input_shape)):
            icount *= input_shape[dim]
        for dim in range(len(shape)):
            if shape[dim] > 0:
                output_dims[dim] = shape[dim]
                ocount *= output_dims[dim]
            elif shape[dim] == 0:
                output_dims[dim] = input_shape[dim]
                ocount *= output_dims[dim]
            
        
        for dim in range(len(shape)):
            if shape[dim] == -1:
                output_dims[dim] = icount// ocount
                ocount *= output_dims[dim]

        for i in range(len(output_dims)):       
            if output_dims[i] == 0:     
                output_dims[i] = 1
    elif (layer_param.type == "DetectionOutput"):
        output_dims[0] = 1
        output_dims[1] = 1
        output_dims[2] = 1
        output_dims[3] = 7
    else:
        output_dims[0],output_dims[1],output_dims[2],output_dims[3] = input_map[str(inputs[0])]

    dimList["output"] = output_dims

    return dimList


def convertV1LayerTypeToString(layer_param):
    EnumDescriptor = list(caffe_pb2.V1LayerParameter.LayerType.items())
    for item in EnumDescriptor:
        if layer_param.type == item[1]:
            layer_type_V1 = item[0]
    if layer_type_V1 == "CONCAT":
        layer_type = "Concat"
    elif layer_type_V1 == "CONVOLUTION":
        layer_type = "Convolution"
    elif layer_type_V1 == "DATA":
        layer_type = "Data"
    elif layer_type_V1 == "DECONVOLUTION":
        layer_type = "Deconvolution"
    elif layer_type_V1 == "DROPOUT":
        layer_type = "Dropout"   
    elif layer_type_V1 == "ELTWISE":
        layer_type = "Eltwise"
    elif layer_type_V1 == "FLATTEN":
        layer_type = "Flatten"
    elif layer_type_V1 == "IMAGE_DATA":
        layer_type = "ImageData"
    elif layer_type_V1 == "INNER_PRODUCT":
        layer_type = "InnerProduct" 
    elif layer_type_V1 == "LRN":
        layer_type = "LRN"
    elif layer_type_V1 == "POOLING":
        layer_type = "Pooling" 
    elif layer_type_V1 == "RELU":
        layer_type = "ReLU" 
    elif layer_type_V1 == "SOFTMAX":
        layer_type = "Softmax"    
    elif layer_type_V1 == "SOFTMAX_LOSS":
        layer_type = "SoftmaxWithLoss"
    elif layer_type_V1 == "SPLIT":
        layer_type = "Split"
    elif layer_type_V1 == "SLICE":
        layer_type = "Slice"
    elif layer_type_V1 == "SCALE":
        layer_type = "Scale"
    else:
        layer_type = "Unknown V1 Layer Type"
    return layer_type

# extract caffe node information into ir nodes.
def extractCaffeNodeInfo(net_parameter, graph, inputsInfo, verbose):
    inputOutputMap = collections.OrderedDict()
    dropoutLayerMap = {}
    splitLayerMap = {}
    outputNameAliasMap = {}
    inputsMap = {}
    outputsMap = {}
    count = 0
    _output_name = {}

    if (len(net_parameter.layer) == 0):
        layers = net_parameter.layers
    else:
        layers = net_parameter.layer

    for i in range(len(layers)):
        layer_param = layers[i]
        layer_name = caffe_name_to_ir_name(str(layer_param.name))
        if(type(layer_param) == caffe_pb2.V1LayerParameter):
            layer_type = convertV1LayerTypeToString(layer_param)
        else:
            layer_type = str(layer_param.type)
    
        inputs = layer_param.bottom
        outputs = layer_param.top
        # ignoring the input/data layer as input is already obtained in previous step.
        if (layer_type == "Data" or layer_type == "ImageData" or layer_type == "Input"):
            continue
        # find out all the outputs and store names
        for k in range(len(layer_param.bottom)):
            if layer_param.bottom[k] in _output_name:
                _output_name[layer_param.bottom[k]]['count'] = _output_name[layer_param.bottom[k]]['count']+1
            else:
                _output_name[layer_param.bottom[k]] = {'count':0}
        for k in range(len(layer_param.top)):
            if layer_param.top[k] in _output_name:
                _output_name[layer_param.top[k]]['count'] = _output_name[layer_param.top[k]]['count']+1
            else:
                _output_name[layer_param.top[k]] = {'count':0, 'name':layer_name}

        # dropout layer is copy layer in inference, hence aliasing the input for dropout layer for next layer.
        if (layer_type == "Dropout"):
            in_name = caffe_name_to_ir_name(str(inputs[0]))
            if in_name in outputNameAliasMap:
                in_name = outputNameAliasMap[in_name]
            dropoutLayerMap[caffe_name_to_ir_name(str(outputs[0]))] = in_name
            continue

        # split layer optimization.
        if (layer_type == "Split"):
            in_name = caffe_name_to_ir_name(str(inputs[0]))
            if (in_name in outputNameAliasMap):
                in_name = outputNameAliasMap[in_name]
            for k in range(len(outputs)):
                splitLayerMap[caffe_name_to_ir_name(outputs[k])] = in_name
            continue

        layer_info_map = {}
        input_info_map = collections.OrderedDict()
        output_info_map = collections.OrderedDict()
        layer_info_map["layer_name"] = layer_name
        if layer_type in caffe2ir_op_type:
            layer_info_map["layer_type"] = caffe2ir_op_type[layer_type]
        elif layer_type == "Pooling":
            pool_type = layer_param.pooling_param.pool
            layer_info_map["layer_type"] = "max_pool" if (pool_type == caffe_pb2.PoolingParameter.MAX) else "avg_pool"

        #fusing scale layer to batchnorm layer.
        #adding scale weights and biases into the batchnorm, else fusing scale to mul or muladd operator.
        elif layer_type == "Scale":
            scale_fused = 0
            if (verbose):
                print ("Info: Found scale layer  " + str(layer_name))
            if (count > 0 and (count < len(layers))):
                in_name = caffe_name_to_ir_name(str(inputs[0]))
                for j in range(count-1, 0, -1):
                    prev_layer_info = inputOutputMap[j]
                    prev_layer_type = prev_layer_info["layer_type"]
                    prev_output_map = prev_layer_info["outputs"]
                    if (verbose):
                        print("prev_type " + str(prev_layer_type) + " " + str(in_name) + " " + str(prev_output_map))
                    if (prev_layer_type == "batch_norm" and ((in_name in prev_output_map) or (in_name in outputNameAliasMap))):
                        modified_out_info_map = {}
                        scale_weights_map = {}
                        scale_bias_map = {}
                        extractBinary(layer_param, graph, verbose)
                        prev_input_map = prev_layer_info["inputs"]
                        prev_attribute_map = prev_layer_info["attributes"]
                        dimList = calculateTensorDims(layer_param, prev_input_map, prev_attribute_map)
                        modified_out_info_map[layer_name] = dimList["output"]
                        outputsMap.update(modified_out_info_map)
                        prev_layer_info["outputs"] = modified_out_info_map
                        if ("weights" in dimList):
                            scale_weights = layer_name + "_w"
                            scale_weights_map[scale_weights] = dimList["weights"]
                            prev_layer_info["scale_weights"] = scale_weights_map
                            graph.addVariable(caffe_blob_to_ir_tensor(scale_weights, "F032", dimList["weights"]))
                        if ("bias" in dimList):
                            scale_bias = layer_name + "_b"
                            scale_bias_map[scale_bias] = dimList["bias"]
                            prev_layer_info["scale_bias"] = scale_bias_map
                            graph.addVariable(caffe_blob_to_ir_tensor(scale_bias, "F032", dimList["bias"]))
                        if(layer_name != caffe_name_to_ir_name(str(outputs[0]))):
                            outputNameAliasMap[caffe_name_to_ir_name(str(outputs[0]))] = layer_name
                        prev_layer_name = prev_layer_info["layer_name"]
                        prev_layer_info["layer_name"] = layer_name
                        inputOutputMap[j] = prev_layer_info
                        if (verbose):
                            print (prev_layer_info)
                        node = caffe_node_to_ir_node(prev_layer_info["layer_type"], prev_layer_info)
                        graph.addNode(node)
                        if (verbose):
                            print ("OK: fusing scale" + str(layer_name) + "to batch_norm" + str(prev_layer_name))
                        scale_fused = 1
                        break
                if scale_fused == 0:
                    scale_layer_type = 'mul' if len(layer_param.blobs) == 1 else 'muladd'
                    if (verbose):
                        print ("OK: Fusing scale to : " + scale_layer_type)
                    layer_info_map["layer_type"] = scale_layer_type
                continue
        else:
            print ("ERROR: caffe operation %s is not supported yet." % (layer_type))
            sys.exit(1)

        # extract attributes of the layer.
        attribute_map = extractCaffeAttrInfo(layer_param)
        layer_info_map["attributes"] = attribute_map
        if (layer_type == "ReLU" and attribute_map["alpha"] != 0):
            layer_info_map["layer_type"] = "leaky_relu"

        #extract input information.
        if (count == 0):
            for k in range(len(inputs)):
                in_name = caffe_name_to_ir_name(str(inputs[k]))
                if str(inputs[k]) in inputsInfo:
                    input_info_map[in_name] = inputsInfo[in_name]
                elif str(inputs[k]) in splitLayerMap:
                    inp_name = splitLayerMap[in_name]
                    input_info_map[inp_name] = inputsInfo[inp_name]
                else:
                    print ("ERROR: unable to get the input dimensions for the layer %s" % (layer_name))
                    sys.exit(1)
        else:
            for k in range(len(inputs)):
                previous_layer_info = inputOutputMap[count - 1]
                prevOutMap = previous_layer_info["outputs"]
                input_name = str(caffe_name_to_ir_name(str(inputs[k])))

                # changing the name of the input based on alias name for top==bottom in previous layer.
                if (input_name in outputNameAliasMap):
                    input_name = outputNameAliasMap[input_name]

                if (input_name in splitLayerMap):
                    input_name = splitLayerMap[input_name]

                if (input_name in dropoutLayerMap):
                    input_name = dropoutLayerMap[input_name]

                # get the input dimensions.
                if input_name in prevOutMap:
                    input_info_map[input_name] = prevOutMap[input_name]
                elif input_name in outputsMap:
                    input_info_map[input_name] = outputsMap[input_name]
                elif input_name in inputsMap:
                    input_info_map[input_name] = inputsMap[input_name]
                elif input_name in dropoutLayerMap:
                    input_info_map[dropoutLayerMap[input_name]] = outputsMap[dropoutLayerMap[input_name]]
                elif input_name in splitLayerMap:
                    input_info_map[splitLayerMap[input_name]] = prevOutMap[splitLayerMap[input_name]]
                elif input_name in inputsInfo:
                    input_info_map[input_name] = inputsInfo[input_name]
                else:
                    if (((layer_type == "Softmax") or (layer_type == "SoftmaxWithLoss")) and k != 0):
                        break
                    elif input_name in outputNameAliasMap:
                        input_info_map[outputNameAliasMap[input_name]] = prevOutMap[outputNameAliasMap[input_name]]
                    else:
                        print ("ERROR: unknown dimensions for %s in the layer %s " % (input_name, layer_name))
                        sys.exit(1)

        inputsMap.update(input_info_map)
        #calculate output,weight and bias dimensions.
        dimList = calculateTensorDims(layer_param, input_info_map, attribute_map)
        if (len(outputs) > 0) and caffe_name_to_ir_name(str(layer_name)) != caffe_name_to_ir_name(str(outputs[0])):
            outputNameAliasMap[caffe_name_to_ir_name(str(outputs[0]))] = caffe_name_to_ir_name(str(layer_name))

        output_info_map[layer_name] = dimList["output"]
        outputsMap.update(output_info_map)

        # add inputs and outputs to layer info.
        layer_info_map["inputs"] = input_info_map
        layer_info_map["outputs"] = output_info_map

        #add weights and biases info if present into the layer info.
        extractBinary(layer_param, graph, verbose)
        weights_map = {}
        bias_map = {}
        if "weights" in dimList:
            weights = layer_name + '_w'
            weight_dims = dimList["weights"]
            weights_map[weights] = weight_dims
            graph.addVariable(caffe_blob_to_ir_tensor(weights, "F032", weight_dims))
            layer_info_map["weights"] = weights_map
        if "bias" in dimList:
            biases = layer_name + "_b"
            bias_dims = dimList["bias"]
            bias_map[biases] = bias_dims
            graph.addVariable(caffe_blob_to_ir_tensor(biases, "F032", bias_dims))
            layer_info_map["biases"] = bias_map

        inputOutputMap[count] = layer_info_map
        count += 1
        if(layer_type == "BatchNorm" and (i < len(layers) - 1)):
            for j in range(i+1, len(layers)-1):
                next_layer_param = layers[j]
                if(next_layer_param.type == "Scale" and str(next_layer_param.bottom[0]) ==  str(outputs[0])):
                    #scaleLayerInputMap[caffe_name_to_ir_name(str(next_layer_param.name))] = caffe_name_to_ir_name(str(outputs[0]))
                    break
            continue

        if (verbose):
            print (layer_info_map)
        node = caffe_node_to_ir_node(layer_info_map["layer_type"], layer_info_map)        
        graph.addNode(node)
    #add all outputs to graph    
    output_name = []
    for i in _output_name:
        if 'name' in _output_name[i] and _output_name[i]['count'] == 0:
            output_name.append(_output_name[i]['name']) 

    return inputOutputMap, output_name


# convert caffe graph to ir graph.
def caffe_graph_to_ir_graph(net_parameter, input_dims, verbose):
    graph = ir.IrGraph(False)
    inputMap = extractInput(net_parameter, graph, input_dims)
    inputOutputMap, output_name = extractCaffeNodeInfo(net_parameter, graph, inputMap, verbose)
    outputList = extractOutput(graph, inputOutputMap, output_name, verbose)
    graph.updateLocals()
    return graph

# convert caffe representation to ir representation.
def caffe2ir(net_parameter, input_dims, outputFolder, verbose,node_type_append):
    graph = caffe_graph_to_ir_graph(net_parameter, input_dims, verbose)
    graph.toFile(outputFolder,node_type_append)
    print ("OK: graph successfully formed.")

def main():
    if len(sys.argv) < 4:
        print ("Usage : python caffe_to_nnir.py <caffeModel> <nnirOutputFolder> --input-dims n,c,h,w [--verbose 0|1] [--node_type_append 0/1 (optional: appends node type to output tensor name)]")
        sys.exit(1)
    caffeFileName = sys.argv[1]
    outputFolder = sys.argv[2]
    input_dims = sys.argv[4].split(',')

    verbose = 0
    """if(len(sys.argv) > 5):
        verbose = 1 if int(sys.argv[6]) else 0
        if (verbose):
            print ("OK: verbose enabled.")
    """
    #appends node type to output tensor name.
    node_type_append = 0
    pos = 5
    while pos < len(sys.argv) and len(sys.argv) >= 5 and sys.argv[pos][:2] == '--':
        if sys.argv[pos] == '--node_type_append':
            node_type_append = int(sys.argv[pos+1])
            pos = pos + 2
        elif sys.argv[pos] == '--verbose':
            verbose = int(sys.argv[pos+1])
            pos = pos + 2
            if (verbose):
                print ("OK: verbose enabled.")
    print ("OK: loading caffemodel from %s ..." % (caffeFileName))
    net_parameter = caffe_pb2.NetParameter()
    if not os.path.isfile(caffeFileName):
        print ("ERROR: unable to open : " + caffeFileName)
        sys.exit(1)

    if (verbose):
        print ("parsing the caffemodel from : " + str(caffeFileName))
    net_parameter.ParseFromString(open(caffeFileName, 'rb').read())
    print ("OK: caffemodel read successful")
    print ("converting to AMD NNIR format in %s folder ... " % (outputFolder))
    if (verbose):
        print ("input parameters obtained are : " + str(input_dims[0]) + " " + str(input_dims[1]) + " " + str(input_dims[2]) + " " + str(input_dims[3]))

    caffe2ir(net_parameter, input_dims, outputFolder, verbose, node_type_append)

if __name__ == '__main__':
    main()
