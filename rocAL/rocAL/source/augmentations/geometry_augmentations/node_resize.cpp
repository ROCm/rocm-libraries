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

#include "augmentations/geometry_augmentations/node_resize.h"

#include <vx_ext_rpp.h>

#include "pipeline/exception.h"

ResizeNode::ResizeNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void ResizeNode::create_node() {
    if (_node)
        return;

    std::vector<uint32_t> dst_roi_width(_batch_size, _outputs[0]->info().max_shape()[0]);
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().max_shape()[1]);

    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);

    vx_status width_status, height_status;

    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    if (width_status != 0 || height_status != 0)
        THROW("vxAddArrayItems failed in the resize (vxExtRppResize) node: " + TOSTR(width_status) + "  " + TOSTR(height_status));

    vx_scalar interpolation_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_interpolation_type);
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppResize(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _dst_roi_width,
                           _dst_roi_height, interpolation_vx, input_layout_vx, output_layout_vx, roi_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the resize (vxExtRppResize) node failed: " + TOSTR(status))
}

void ResizeNode::update_node() {
    Roi2DCords *src_dims = _inputs[0]->info().roi().get_2D_roi();
    for (unsigned i = 0; i < _batch_size; i++) {
        _src_width = src_dims[i].xywh.w;
        _src_height = src_dims[i].xywh.h;
        _dst_width = _out_width;
        _dst_height = _out_height;
        adjust_out_roi_size();
        _dst_width = std::min(_dst_width, static_cast<unsigned>(_outputs[0]->info().max_shape()[0]));
        _dst_height = std::min(_dst_height, static_cast<unsigned>(_outputs[0]->info().max_shape()[1]));
        _dst_roi_width_vec.push_back(_dst_width);
        _dst_roi_height_vec.push_back(_dst_height);
    }
    vx_status width_status, height_status;
    width_status = vxCopyArrayRange((vx_array)_dst_roi_width, 0, _batch_size, sizeof(vx_uint32), _dst_roi_width_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    height_status = vxCopyArrayRange((vx_array)_dst_roi_height, 0, _batch_size, sizeof(vx_uint32), _dst_roi_height_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if (width_status != 0 || height_status != 0)
        WRN("ERROR: vxCopyArrayRange _dst_roi_width or _dst_roi_height failed " + TOSTR(width_status) + "  " + TOSTR(height_status));
    _outputs[0]->update_tensor_roi(_dst_roi_width_vec, _dst_roi_height_vec);
    _dst_roi_width_vec.clear();
    _dst_roi_height_vec.clear();
}

void ResizeNode::init(unsigned dest_width, unsigned dest_height, RocalResizeScalingMode scaling_mode,
                      const std::vector<unsigned> &max_size, RocalResizeInterpolationType interpolation_type) {
    _interpolation_type = (int)interpolation_type;
    _scaling_mode = scaling_mode;
    _out_width = dest_width;
    _out_height = dest_height;
    if (max_size.size() > 0) {
        _max_width = max_size[0];
        _max_height = max_size[1];
    }
}

void ResizeNode::adjust_out_roi_size() {
    bool has_max_size = (_max_width | _max_height) > 0;

    if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_STRETCH) {
        if (!_dst_width) _dst_width = _src_width;
        if (!_dst_height) _dst_height = _src_height;

        if (has_max_size) {
            if (_max_width) _dst_width = std::min(_dst_width, _max_width);
            if (_max_height) _dst_height = std::min(_dst_height, _max_height);
        }
    } else if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_DEFAULT) {
        if ((!_dst_width) & _dst_height) {  // Only height is passed
            _dst_width = std::lround(_src_width * (static_cast<float>(_dst_height) / _src_height));
        } else if ((!_dst_height) & _dst_width) {  // Only width is passed
            _dst_height = std::lround(_src_height * (static_cast<float>(_dst_width) / _src_width));
        }

        if (has_max_size) {
            if (_max_width) _dst_width = std::min(_dst_width, _max_width);
            if (_max_height) _dst_height = std::min(_dst_height, _max_height);
        }
    } else {
        float scale = 1.0f;
        float scale_w = static_cast<float>(_dst_width) / _src_width;
        float scale_h = static_cast<float>(_dst_height) / _src_height;
        if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_SMALLER) {
            scale = std::max(scale_w, scale_h);
        } else if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_LARGER) {
            scale = (scale_w > 0 && scale_h > 0) ? std::min(scale_w, scale_h) : ((scale_w > 0) ? scale_w : scale_h);
        }

        if (has_max_size) {
            if (_max_width) scale = std::min(scale, static_cast<float>(_max_width) / _src_width);
            if (_max_height) scale = std::min(scale, static_cast<float>(_max_height) / _src_height);
        }

        if ((scale_h != scale) || (!_dst_height)) _dst_height = std::lround(_src_height * scale);
        if ((scale_w != scale) || (!_dst_width)) _dst_width = std::lround(_src_width * scale);
    }
}
