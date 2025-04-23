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
#include "parameters/parameter_crop.h"

class RocalRandomCropParam : public CropParam {
   public:
    RocalRandomCropParam() = delete;
    ~RocalRandomCropParam() = default;
    RocalRandomCropParam(unsigned int batch_size) : CropParam(batch_size) {
        area_factor = default_area_factor();
        aspect_ratio = default_aspect_ratio();
    }
    void set_area_factor(Parameter<float>* crop_h_factor);
    void set_aspect_ratio(Parameter<float>* crop_w_factor);
    Parameter<float>* get_area_factor() { return area_factor; }
    Parameter<float>* get_aspect_ratio() { return aspect_ratio; }
    void update_array() override;
    void fill_crop_dims() override;

   private:
    constexpr static float AREA_FACTOR_RANGE[2] = {0.08, 0.99};
    constexpr static float ASPECT_RATIO_RANGE[2] = {0.7500, 1.333};
    Parameter<float>* default_area_factor();
    Parameter<float>* default_aspect_ratio();
    Parameter<float>*area_factor, *aspect_ratio;
};
