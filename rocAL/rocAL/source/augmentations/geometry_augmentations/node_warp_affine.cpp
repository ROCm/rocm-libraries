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

#include "augmentations/geometry_augmentations/node_warp_affine.h"

#include <vx_ext_rpp.h>

#include "pipeline/exception.h"

WarpAffineNode::WarpAffineNode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) : Node(inputs, outputs),
                                                                                                          _x0(COEFFICIENT_RANGE_1[0], COEFFICIENT_RANGE_1[1]),
                                                                                                          _x1(COEFFICIENT_RANGE_0[0], COEFFICIENT_RANGE_0[1]),
                                                                                                          _y0(COEFFICIENT_RANGE_0[0], COEFFICIENT_RANGE_0[1]),
                                                                                                          _y1(COEFFICIENT_RANGE_1[0], COEFFICIENT_RANGE_1[1]),
                                                                                                          _o0(COEFFICIENT_RANGE_OFFSET[0], COEFFICIENT_RANGE_OFFSET[1]),
                                                                                                          _o1(COEFFICIENT_RANGE_OFFSET[0], COEFFICIENT_RANGE_OFFSET[1]) {}

void WarpAffineNode::create_node() {
    if (_node)
        return;
    _affine.resize(6 * _batch_size);
    uint batch_size = _batch_size;
    for (uint i = 0; i < batch_size; i++) {
        _affine[i * 6 + 0] = _x0.renew();
        _affine[i * 6 + 1] = _y0.renew();
        _affine[i * 6 + 2] = _x1.renew();
        _affine[i * 6 + 3] = _y1.renew();
        _affine[i * 6 + 4] = _o0.renew();
        _affine[i * 6 + 5] = _o1.renew();
    }
    vx_status status;
    _affine_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size * 6);
    status = vxAddArrayItems(_affine_array, _batch_size * 6, _affine.data(), sizeof(vx_float32));
    vx_scalar interpolation_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_interpolation_type);
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppWarpAffine(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _affine_array,
                               interpolation_vx, input_layout_vx, output_layout_vx, roi_type_vx);
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the warp affine (vxExtRppWarpAffine) node failed: " + TOSTR(status))
}

void WarpAffineNode::update_affine_array() {
    for (uint i = 0; i < _batch_size; i++) {
        _affine[i * 6 + 0] = _x0.renew();
        _affine[i * 6 + 1] = _y0.renew();
        _affine[i * 6 + 2] = _x1.renew();
        _affine[i * 6 + 3] = _y1.renew();
        _affine[i * 6 + 4] = _o0.renew();
        _affine[i * 6 + 5] = _o1.renew();
    }
    vx_status affine_status;
    affine_status = vxCopyArrayRange((vx_array)_affine_array, 0, _batch_size * 6, sizeof(vx_float32), _affine.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if (affine_status != 0)
        THROW(" vxCopyArrayRange failed in the WarpAffine(vxExtRppWarpAffinePD) node: " + TOSTR(affine_status))
}

void WarpAffineNode::init(float x0, float x1, float y0, float y1, float o0, float o1, RocalResizeInterpolationType interpolation_type) {
    _x0.set_param(x0);
    _x1.set_param(x1);
    _y0.set_param(y0);
    _y1.set_param(y1);
    _o0.set_param(o0);
    _o1.set_param(o1);
    _interpolation_type = static_cast<int>(interpolation_type);
}

void WarpAffineNode::init(FloatParam* x0, FloatParam* x1, FloatParam* y0, FloatParam* y1,
                          FloatParam* o0, FloatParam* o1, RocalResizeInterpolationType interpolation_type) {
    _x0.set_param(core(x0));
    _x1.set_param(core(x1));
    _y0.set_param(core(y0));
    _y1.set_param(core(y1));
    _o0.set_param(core(o0));
    _o1.set_param(core(o1));
    _interpolation_type = static_cast<int>(interpolation_type);
}

void WarpAffineNode::update_node() {
    update_affine_array();
}
