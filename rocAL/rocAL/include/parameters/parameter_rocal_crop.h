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

class RocalCropParam : public CropParam {
   public:
    RocalCropParam() = delete;
    ~RocalCropParam() = default;
    RocalCropParam(unsigned int batch_size) : CropParam(batch_size) {
        crop_height_factor = default_crop_height_factor();
        crop_width_factor = default_crop_width_factor();
    }
    unsigned int crop_w, crop_h, crop_d;
    void set_crop_height_factor(Parameter<float>* crop_h_factor);
    void set_crop_width_factor(Parameter<float>* crop_w_factor);
    void update_array() override;

   private:
    constexpr static float CROP_HEIGHT_FACTOR_RANGE[2] = {0.55, 0.95};
    constexpr static float CROP_WIDTH_FACTOR_RANGE[2] = {0.55, 0.95};
    Parameter<float>* default_crop_height_factor();
    Parameter<float>* default_crop_width_factor();
    Parameter<float>*crop_height_factor, *crop_width_factor;
    void fill_crop_dims() override;
};
