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

#include "augmentations/node_sequence_rearrange.h"

#include <VX/vx_compatibility.h>
#include <vx_ext_rpp.h>

#include "pipeline/exception.h"

SequenceRearrangeNode::SequenceRearrangeNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void SequenceRearrangeNode::create_node() {
    if (_node)
        return;

    vx_status status;
    vx_array sequence_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _new_order.size());
    status = vxAddArrayItems(sequence_array, _new_order.size(), _new_order.data(), sizeof(vx_uint32));
    if (status != VX_SUCCESS)
        THROW("Adding array items failed: " + TOSTR(status));
    int input_layout = (int)_inputs[0]->info().layout();
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    _node = vxExtRppSequenceRearrange(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), sequence_array, input_layout_vx);

    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the sequence rearrange (vxExtRppSequenceRearrange) node failed: " + TOSTR(status))
}

void SequenceRearrangeNode::init(std::vector<unsigned int> &new_order) {
    _new_order = new_order;
}

void SequenceRearrangeNode::update_node() {}
