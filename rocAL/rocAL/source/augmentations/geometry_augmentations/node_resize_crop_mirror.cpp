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

#include <vx_ext_rpp.h>
#include "pipeline/graph.h"
#include "augmentations/geometry_augmentations/node_resize_crop_mirror.h"
#include "pipeline/exception.h"

ResizeCropMirrorNode::ResizeCropMirrorNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : CropNode(inputs, outputs),
                                                                                                                        _mirror(MIRROR_RANGE[0], MIRROR_RANGE[1]) {
    _crop_param = std::make_shared<RocalCropParam>(_batch_size);
}

void ResizeCropMirrorNode::create_node() {
    if (_node)
        return;

    _crop_param->create_array(_graph);

    std::vector<uint32_t> dst_roi_width(_batch_size, _outputs[0]->info().max_shape()[0]);
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().max_shape()[1]);
    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);

    vx_status width_status, height_status;
    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    if (width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the crop resize node (vxExtRppResizeCropMirror)  node: " + TOSTR(width_status) + "  " + TOSTR(height_status))
    _mirror.create_array(_graph, VX_TYPE_UINT32, _batch_size);
    create_crop_tensor();
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);
    vx_scalar interpolation_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_interpolation_type);

    _node = vxExtRppResizeCropMirror(_graph->get(), _inputs[0]->handle(), _crop_tensor, _outputs[0]->handle(), _dst_roi_width,
                                     _dst_roi_height, _mirror.default_array(), interpolation_vx, input_layout_vx, output_layout_vx, roi_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the resize crop mirror node (vxExtRppResizeCropMirror) failed: " + TOSTR(status))
}

void ResizeCropMirrorNode::update_node() {
    _crop_param->set_image_dimensions(_inputs[0]->info().roi().get_2D_roi());
    _crop_param->update_array();
    std::vector<uint32_t> crop_h_dims, crop_w_dims;
    _crop_param->get_crop_dimensions(crop_w_dims, crop_h_dims);
    _outputs[0]->update_tensor_roi(crop_w_dims, crop_h_dims);
    _mirror.update_array();

    // Obtain the crop coordinates and update the roi
    auto x1 = _crop_param->get_x1_arr_val();
    auto y1 = _crop_param->get_y1_arr_val();
    Roi2DCords *crop_dims = static_cast<Roi2DCords *>(_crop_coordinates);
    for (unsigned i = 0; i < _batch_size; i++) {
        crop_dims[i].xywh.x = x1[i];
        crop_dims[i].xywh.y = y1[i];
        crop_dims[i].xywh.w = crop_w_dims[i];
        crop_dims[i].xywh.h = crop_h_dims[i];
    }
}

void ResizeCropMirrorNode::init(unsigned int crop_h, unsigned int crop_w, IntParam *mirror, RocalResizeInterpolationType interpolation_type) {
    _crop_param->crop_w = crop_w;
    _crop_param->crop_h = crop_h;
    _crop_param->x1 = 0;
    _crop_param->y1 = 0;
    _mirror.set_param(core(mirror));
    _interpolation_type = static_cast<int>(interpolation_type);
}

void ResizeCropMirrorNode::init(FloatParam *crop_h_factor, FloatParam *crop_w_factor, IntParam *mirror, RocalResizeInterpolationType interpolation_type) {
    _crop_param->set_crop_height_factor(core(crop_h_factor));
    _crop_param->set_crop_width_factor(core(crop_w_factor));
    _crop_param->set_random();
    _mirror.set_param(core(mirror));
    _interpolation_type = static_cast<int>(interpolation_type);
}
