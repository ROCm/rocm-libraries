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

#include <cmath>
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include "pipeline/graph.h"
#include "parameters/parameter_random_crop.h"
#include "pipeline/commons.h"

void RocalRandomCropParam::set_area_factor(Parameter<float>* crop_area_factor) {
    if (!crop_area_factor)
        return;
    ParameterFactory::instance()->destroy_param(area_factor);
    area_factor = crop_area_factor;
}

void RocalRandomCropParam::set_aspect_ratio(Parameter<float>* crop_aspect_ratio) {
    if (!crop_aspect_ratio)
        return;
    ParameterFactory::instance()->destroy_param(aspect_ratio);
    aspect_ratio = crop_aspect_ratio;
}

void RocalRandomCropParam::update_array() {
    fill_crop_dims();
    update_crop_array();
}

void RocalRandomCropParam::fill_crop_dims() {
    float crop_area_factor = 1.0;
    float crop_aspect_ratio = 1.0;
    float in_ratio;
    unsigned short num_of_attempts = 5;
    float x_drift, y_drift;
    double target_area;
    auto is_valid_crop = [](uint h, uint w, uint height, uint width) {
        return (h < height && w < width);
    };
    for (uint img_idx = 0; img_idx < batch_size; img_idx++) {
        // Try for num_of_attempts time to get a good crop
        for (int i = 0; i < num_of_attempts; i++) {
            area_factor->renew();
            aspect_ratio->renew();
            crop_area_factor = area_factor->get();
            crop_aspect_ratio = aspect_ratio->get();
            target_area = crop_area_factor * in_roi[img_idx].xywh.h * in_roi[img_idx].xywh.w;
            cropw_arr_val[img_idx] = static_cast<size_t>(std::sqrt(target_area * crop_aspect_ratio));
            croph_arr_val[img_idx] = static_cast<size_t>(std::sqrt(target_area * (1 / crop_aspect_ratio)));
            if (is_valid_crop(croph_arr_val[img_idx], cropw_arr_val[img_idx], in_roi[img_idx].xywh.h, in_roi[img_idx].xywh.w)) {
                x_drift_factor->renew();
                y_drift_factor->renew();
                y_drift_factor->renew();
                x_drift = x_drift_factor->get();
                y_drift = y_drift_factor->get();
                x1_arr_val[img_idx] = static_cast<size_t>(x_drift * (in_roi[img_idx].xywh.w - cropw_arr_val[img_idx]));
                y1_arr_val[img_idx] = static_cast<size_t>(y_drift * (in_roi[img_idx].xywh.h - croph_arr_val[img_idx]));
                break;
            }
        }
        // Fallback on Central Crop
        if (!is_valid_crop(croph_arr_val[img_idx], cropw_arr_val[img_idx], in_roi[img_idx].xywh.h, in_roi[img_idx].xywh.w)) {
            in_ratio = static_cast<float>(in_roi[img_idx].xywh.w) / in_roi[img_idx].xywh.h;
            if (in_ratio < ASPECT_RATIO_RANGE[0]) {
                cropw_arr_val[img_idx] = in_roi[img_idx].xywh.w;
                croph_arr_val[img_idx] = cropw_arr_val[img_idx] / ASPECT_RATIO_RANGE[0];
            } else if (in_ratio > ASPECT_RATIO_RANGE[1]) {
                croph_arr_val[img_idx] = in_roi[img_idx].xywh.h;
                cropw_arr_val[img_idx] = croph_arr_val[img_idx] * ASPECT_RATIO_RANGE[1];
            } else {
                croph_arr_val[img_idx] = in_roi[img_idx].xywh.h;
                cropw_arr_val[img_idx] = in_roi[img_idx].xywh.w;
            }
            x1_arr_val[img_idx] = (in_roi[img_idx].xywh.w - cropw_arr_val[img_idx]) / 2;
            y1_arr_val[img_idx] = (in_roi[img_idx].xywh.h - croph_arr_val[img_idx]) / 2;
        }
        x2_arr_val[img_idx] = x1_arr_val[img_idx] + cropw_arr_val[img_idx];
        y2_arr_val[img_idx] = y1_arr_val[img_idx] + croph_arr_val[img_idx];
    }
}

Parameter<float>* RocalRandomCropParam::default_area_factor() {
    return ParameterFactory::instance()->create_uniform_float_rand_param(AREA_FACTOR_RANGE[0],
                                                                         AREA_FACTOR_RANGE[1])
        ->core;
}

Parameter<float>* RocalRandomCropParam::default_aspect_ratio() {
    return ParameterFactory::instance()->create_uniform_float_rand_param(ASPECT_RATIO_RANGE[0],
                                                                         ASPECT_RATIO_RANGE[1])
        ->core;
}
