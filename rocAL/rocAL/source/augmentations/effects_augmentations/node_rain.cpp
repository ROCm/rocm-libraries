/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "augmentations/effects_augmentations/node_rain.h"

#include <vx_ext_rpp.h>

#include "pipeline/exception.h"

RainNode::RainNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs),
                                                                                                _rain_value(RAIN_VALUE_RANGE[0], RAIN_VALUE_RANGE[1]),
                                                                                                _rain_width(RAIN_WIDTH_RANGE[0], RAIN_WIDTH_RANGE[1]),
                                                                                                _rain_height(RAIN_HEIGHT_RANGE[0], RAIN_HEIGHT_RANGE[0]),
                                                                                                _rain_transparency(RAIN_TRANSPARENCY_RANGE[0], RAIN_TRANSPARENCY_RANGE[1]) {}

void RainNode::create_node() {
    if (_node)
        return;

    _rain_value.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _rain_transparency.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _rain_width.create_array(_graph, VX_TYPE_UINT32, _batch_size);
    _rain_height.create_array(_graph, VX_TYPE_UINT32, _batch_size);
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppRain(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _rain_value.default_array(), _rain_width.default_array(), _rain_height.default_array(), _rain_transparency.default_array(), input_layout_vx, output_layout_vx, roi_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the rain (vxExtRppRain) node failed: " + TOSTR(status))
}

void RainNode::init(float rain_value, int rain_width, int rain_height, float rain_transparency) {
    _rain_value.set_param(rain_value);
    _rain_width.set_param(rain_width);
    _rain_height.set_param(rain_height);
    _rain_transparency.set_param(rain_transparency);
}

void RainNode::init(FloatParam *rain_value, IntParam *rain_width, IntParam *rain_height, FloatParam *rain_transparency) {
    _rain_value.set_param(core(rain_value));
    _rain_width.set_param(core(rain_width));
    _rain_height.set_param(core(rain_height));
    _rain_transparency.set_param(core(rain_transparency));
}

void RainNode::update_node() {
    _rain_height.update_array();
    _rain_width.update_array();
    _rain_value.update_array();
    _rain_transparency.update_array();
}
