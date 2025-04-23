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

#include <vx_ext_rpp.h>
#include "augmentations/audio_augmentations/node_preemphasis_filter.h"
#include "pipeline/exception.h"

PreemphasisFilterNode::PreemphasisFilterNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs),
                                                                                                                          _preemph_coeff(PREEMPH_COEFF_RANGE[0], PREEMPH_COEFF_RANGE[1]) {}

void PreemphasisFilterNode::create_node() {
    if (_node)
        return;
    _preemph_coeff.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    vx_scalar border_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_preemph_border);
    _node = vxExtRppPreemphasisFilter(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _preemph_coeff.default_array(), border_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the vxExtRppPreemphasisFilter node failed: " + TOSTR(status))
}

void PreemphasisFilterNode::update_node() {
    _preemph_coeff.update_array();
}

void PreemphasisFilterNode::init(FloatParam *preemph_coeff, RocalAudioBorderType preemph_border) {
    if (preemph_coeff == nullptr)
        ERR("Invalid pre-Emphasis co-efficient passed")
    _preemph_coeff.set_param(core(preemph_coeff));
    _preemph_border = preemph_border;
}
