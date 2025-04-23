/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "augmentations/geometry_augmentations/node_slice.h"

#include <vx_ext_rpp.h>

#include "pipeline/exception.h"

SliceNode::SliceNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void SliceNode::create_node() {
    if (_node)
        return;

    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);
    vx_array fill_values_array_vx = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size);
    auto status = vxAddArrayItems(fill_values_array_vx, _batch_size, _fill_values_vec.data(), sizeof(vx_float32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the slice (vxExtRppSlice) node: " + TOSTR(status));
    vx_scalar policy_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_policy);
    _node = vxExtRppSlice(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(),
                          _outputs[0]->get_roi_tensor(), _anchor->handle(), _shape->handle(),
                          fill_values_array_vx, policy_vx, input_layout_vx, roi_type_vx);

    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the slice node (vxRppSlice) failed: " + TOSTR(status))
}

void SliceNode::update_node() {}

void SliceNode::init(Tensor *anchor, Tensor *shape, std::vector<float> &fill_values, RocalOutOfBoundsPolicy policy) {
    _policy = policy;
    _anchor = anchor;
    _shape = shape;
    _fill_values = fill_values;
    _fill_values_vec.resize(_batch_size);
    // if fill values passed by user less than what is required, replicate the values
    if (_fill_values.size() == 1)
        std::fill(_fill_values_vec.begin(), _fill_values_vec.end(), _fill_values[0]);
}
