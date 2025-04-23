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

#include "augmentations/audio_augmentations/node_resample.h"

#include <vx_ext_rpp.h>

#include "pipeline/exception.h"

ResampleNode::ResampleNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void ResampleNode::create_node() {
    if (_node)
        return;
    _src_sample_rate_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size);
    vx_status status;
    status = vxAddArrayItems(_src_sample_rate_array, _batch_size, _inputs[0]->info().get_sample_rates()->data(), sizeof(vx_float32));
    if (status != 0)
        THROW("vxAddArrayItems for _src_sample_rate_array failed in the Resample Node (vxExtRppResample) :" + TOSTR(status))
    vx_scalar quality_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_quality);
    _node = vxExtRppResample(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->get_roi_tensor(),
                             _src_sample_rate_array, _output_resample_rate->handle(), quality_vx);
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the Resample (vxExtRppResample) node failed: " + TOSTR(status))
}

void ResampleNode::update_node() {
    vx_status status = vxCopyArrayRange((vx_array)_src_sample_rate_array, 0, _batch_size, sizeof(vx_float32), _inputs[0]->info().get_sample_rates()->data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if ((status) != 0)
        THROW(" Failed calling vxCopyArrayRange for _src_sample_rate_array with status in Resample Node (vxExtRppResample) :" + TOSTR(status))
}

void ResampleNode::init(Tensor *output_resample_rate, float quality) {
    _output_resample_rate = output_resample_rate;
    _quality = quality;
}
