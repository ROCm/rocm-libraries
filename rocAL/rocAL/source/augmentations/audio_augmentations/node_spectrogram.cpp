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

#include "augmentations/audio_augmentations/node_spectrogram.h"

#include <vx_ext_rpp.h>

#include "pipeline/exception.h"

SpectrogramNode::SpectrogramNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void SpectrogramNode::create_node() {
    if (_node)
        return;

    vx_status status = VX_SUCCESS;
    vx_array window_fn_vx_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _window_length);
    status |= vxAddArrayItems(window_fn_vx_array, _window_length, _window_fn.data(), sizeof(vx_float32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the spectrogram node (vxRppSpectrogram)  node: " + TOSTR(status) + "  " + TOSTR(status))
    vx_scalar center_windows_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_BOOL, &_is_center_windows);
    vx_scalar reflect_padding_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_BOOL, &_is_reflect_padding);
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    vx_scalar spectrogram_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar power_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_power);
    vx_scalar nfft_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_nfft);
    vx_scalar window_length_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_window_length);
    vx_scalar window_step_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_window_step);
    _node = vxExtRppSpectrogram(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _outputs[0]->get_roi_tensor(), window_fn_vx_array,
                                center_windows_vx, reflect_padding_vx, spectrogram_layout_vx, power_vx, nfft_vx, window_length_vx, window_step_vx);

    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the spectrogram node (vxRppSpectrogram) failed: " + TOSTR(status))
}

void SpectrogramNode::update_node() {}

void SpectrogramNode::init(bool is_center_windows, bool is_reflect_padding, int power, int nfft,
                           int window_length, int window_step, std::vector<float> &window_fn) {
    _is_center_windows = is_center_windows;
    _is_reflect_padding = is_reflect_padding;
    _power = power;
    _nfft = nfft;
    _window_length = window_length;
    _window_step = window_step;
    if (window_fn.empty()) {
        _window_fn.resize(_window_length);
        hann_window(_window_fn.data(), _window_length);
    }
}
