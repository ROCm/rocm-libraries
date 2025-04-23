/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include "augmentations/geometry_augmentations/node_transpose.h"

#include <vx_ext_rpp.h>

#include "pipeline/exception.h"

TransposeNode::TransposeNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void TransposeNode::create_node() {
    if (_node)
        return;

    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);
    _perm_array_vx = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _perm.size());
    vx_status status = VX_SUCCESS;
    status |= vxAddArrayItems(_perm_array_vx, _perm.size(), _perm.data(), sizeof(vx_uint32));

    _node = vxExtRppTranspose(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(),
                              _perm_array_vx, input_layout_vx, output_layout_vx, roi_type_vx);
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the transpose (vxExtRppTranspose) node failed: " + TOSTR(status))
}

void TransposeNode::update_node() {
    auto data_layout = _inputs[0]->info().layout();
    // For NHWC/NCHW layouts, channel can only be transposed at the front or at the back
    if (data_layout == RocalTensorlayout::NHWC || data_layout == RocalTensorlayout::NCHW) {
        // For NHWC/NCHW layouts, we only update the ROI if the transpose order switches the height and width
        std::vector<unsigned> perm1 = {2, 1, 0};
        std::vector<unsigned> perm2 = {1, 0, 2};
        if (_perm == perm1 || _perm == perm2) {
            Roi2DCords *src_dims = _inputs[0]->info().roi().get_2D_roi();
            std::vector<unsigned> _dst_roi_width_vec, _dst_roi_height_vec;
            for (unsigned i = 0; i < _batch_size; i++) {
                _dst_roi_width_vec.push_back(src_dims[i].xywh.w);
                _dst_roi_height_vec.push_back(src_dims[i].xywh.h);
            }
            _outputs[0]->update_tensor_roi(_dst_roi_height_vec, _dst_roi_width_vec);
        }
    } else {
        auto nDim = _inputs[0]->info().num_of_dims() - 1;
        std::vector<std::vector<unsigned>> output_roi_tensor;
        output_roi_tensor.resize(_batch_size);
        for (unsigned i = 0; i < _batch_size; i++) {
            unsigned *input_roi_tensor = _inputs[0]->info().roi()[i].end;
            std::vector<unsigned> transposed_roi;
            for (uint j = 0; j < nDim; j++)
                transposed_roi.push_back(input_roi_tensor[_perm[j]]);
            output_roi_tensor[i] = transposed_roi;
        }
        _outputs[0]->update_tensor_roi(output_roi_tensor);
    }
}

void TransposeNode::init(std::vector<unsigned> perm) {
    _perm = perm;
}