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

#include "augmentations/effects_augmentations/node_normalize.h"

#include <vx_ext_rpp.h>

#include "pipeline/exception.h"

NormalizeNode::NormalizeNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void NormalizeNode::create_node() {
    if (_node)
        return;
    uint compute_mean_and_stddev = NormalizeModes::DO_NOT_COMPUTE;
    if (!_mean.size() && !_std_dev.size()) {
        compute_mean_and_stddev = NormalizeModes::COMPUTE_MEAN_STDDEV;
    } else if (!_mean.size() && _std_dev.size()) {
        compute_mean_and_stddev = NormalizeModes::COMPUTE_MEAN;
    } else if (!_std_dev.size() && _mean.size()) {
        compute_mean_and_stddev = NormalizeModes::COMPUTE_STDDEV;
    }

    int mean_stddev_array_size = 1;
    auto nDim = _inputs[0]->info().num_of_dims() - 1;
    std::vector<unsigned> axis(nDim);
    auto data_layout = _inputs[0]->info().layout();
    auto tensor_dims = _inputs[0]->info().dims();
    if (data_layout == RocalTensorlayout::NHWC) {
        for (unsigned i = 0; i < _batch_size; i++) {
            int totalElements = 1;
            unsigned *tensor_shape = _inputs[0]->info().roi()[i].end;
            totalElements *= ((_axis_mask & (1 << 0)) >= 1) ? 1 : tensor_shape[1];
            totalElements *= ((_axis_mask & (1 << 1)) >= 1) ? 1 : tensor_shape[0];
            totalElements *= ((_axis_mask & (1 << 2)) >= 1) ? 1 : tensor_dims[nDim];
            mean_stddev_array_size = std::max(mean_stddev_array_size, totalElements);
        }
    } else if (data_layout == RocalTensorlayout::NCHW) {
        for (unsigned i = 0; i < _batch_size; i++) {
            int totalElements = 1;
            unsigned *tensor_shape = _inputs[0]->info().roi()[i].end;
            totalElements *= ((_axis_mask & (1 << 0)) >= 1) ? 1 : tensor_dims[1];
            totalElements *= ((_axis_mask & (1 << 1)) >= 1) ? 1 : tensor_shape[1];
            totalElements *= ((_axis_mask & (1 << 2)) >= 1) ? 1 : tensor_shape[0];
            mean_stddev_array_size = std::max(mean_stddev_array_size, totalElements);
        }
    } else {
        for (unsigned i = 0; i < _batch_size; i++) {
            int totalElements = 1;
            unsigned *tensor_shape = _inputs[0]->info().roi()[i].end;
            for (uint j = 0; j < nDim; j++)
                totalElements *= ((_axis_mask & (1 << j)) >= 1) ? 1 : tensor_shape[j];
            mean_stddev_array_size = std::max(mean_stddev_array_size, totalElements);
        }
    }
    std::vector<float> mean_vec, stddev_vec;
    mean_vec.resize(_batch_size * mean_stddev_array_size, 0);
    stddev_vec.resize(_batch_size * mean_stddev_array_size, 0);

    if (!compute_mean_and_stddev) {
        for (uint i = 0; i < _batch_size; i++) {
            for (int j = 0; j < mean_stddev_array_size; j += _mean.size()) {
                for (size_t k = 0; k < _mean.size(); k++) {
                    mean_vec[i * mean_stddev_array_size + j + k] = _mean[k];
                    stddev_vec[i * mean_stddev_array_size + j + k] = _std_dev[k];
                }
            }
        }
    }
    vx_status status = VX_SUCCESS;
    _mean_vx_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, mean_vec.size());
    status |= vxAddArrayItems(_mean_vx_array, mean_vec.size(), mean_vec.data(), sizeof(vx_float32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the normalize node (vxExtRppNormalize)  node: " + TOSTR(status) + "  " + TOSTR(status))

    _stddev_vx_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, stddev_vec.size());
    status |= vxAddArrayItems(_stddev_vx_array, stddev_vec.size(), stddev_vec.data(), sizeof(vx_float32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the normalize node (vxExtRppNormalize)  node: " + TOSTR(status) + "  " + TOSTR(status))

    vx_scalar axis_mask_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_axis_mask);
    vx_scalar scale_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_scale);
    vx_scalar shift_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_shift);
    vx_scalar compute_mean_and_stddev_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT8, &compute_mean_and_stddev);
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppNormalize(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _outputs[0]->get_roi_tensor(), axis_mask_vx,
                              _mean_vx_array, _stddev_vx_array, compute_mean_and_stddev_vx, scale_vx, shift_vx, input_layout_vx, roi_type_vx);
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the normalize (vxExtRppNormalize) node failed: " + TOSTR(status))
}

void NormalizeNode::init(std::vector<unsigned> &axes, std::vector<float> &mean, std::vector<float> &std_dev, float scale, float shift) {
    _mean = mean;
    _std_dev = std_dev;
    _scale = scale;
    _shift = shift;
    for (unsigned d = 0; d < axes.size(); d++)
        _axis_mask |= (1 << axes[d]);
}
