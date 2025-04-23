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

#pragma once
#include <random>
#include <thread>

#include "parameters/parameter_factory.h"

struct CropWindow {
    unsigned x, y, H, W;
    CropWindow() {
        x = 0;
        y = 0;
        H = 1;
        W = 1;
    }
    CropWindow(unsigned x1, unsigned y1, unsigned h, unsigned w) { x = x1, y = y1, H = h, W = w; }
    void set_shape(unsigned h, unsigned w) { H = h, W = w; }
};

typedef std::vector<size_t> Shape;
using AspectRatioRange = std::pair<float, float>;
using AreaRange = std::pair<float, float>;

class RocalRandomCropDecParam {
   public:
    explicit RocalRandomCropDecParam(
        AspectRatioRange aspect_ratio_range = {3.0f / 4, 4.0f / 3},
        AreaRange area_range = {0.08, 1},
        int64_t seed = time(0),
        int num_attempts = 10,
        int batch_size = 256);
    CropWindow generate_crop_window(const Shape& shape, const int instance);
    void generate_random_seeds();

   private:
    CropWindow generate_crop_window_implementation(const Shape& shape);
    AspectRatioRange _aspect_ratio_range;
    // Aspect ratios are uniformly distributed on logarithmic scale.
    // This provides natural symmetry and smoothness of the distribution.
    std::uniform_real_distribution<float> _aspect_ratio_log_dis;
    std::uniform_real_distribution<float> _area_dis;
    // thread_local is needed to call it from multiple threads async, so each thread will have its own copy
    static thread_local std::mt19937 _rand_gen;
    std::vector<int> _seeds;
    int _num_attempts;
    int _batch_size;
};
