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

#pragma once
#include "pipeline/graph.h"
#include "pipeline/node.h"
#include "rocal_api_types.h"

/// @brief Generates hann window for spectrogram
/// @param output 
/// @param window_size 
inline void hann_window(float *output, int window_size) {
    if (window_size <= 0)
        THROW("Invalid window size, for Hann window")
    double a = (2.0 * M_PI) / window_size;
    for (int t = 0; t < window_size; t++) {
        double phase = a * (t + 0.5);
        output[t] = (0.5 * (1.0 - std::cos(phase)));
    }
}

class SpectrogramNode : public Node {
   public:
    SpectrogramNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    SpectrogramNode() = delete;
    void init(bool is_center_windows, bool is_reflect_padding, int power, int nfft,
              int window_length, int window_step, std::vector<float> &window_fn);

   protected:
    void create_node() override;
    void update_node() override;

   private:
    std::vector<float> _window_fn;
    int _power = 2;
    int _nfft = 2048;
    int _window_length = 512;
    int _window_step = 256;
    bool _is_center_windows = true;
    bool _is_reflect_padding = true;
};
