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
#include <memory>
#include <set>

#include "meta_data/bounding_box_graph.h"
#include "meta_data/meta_data.h"
#include "pipeline/node.h"
#include "augmentations/node_ssd_random_crop.h"
#include "parameters/parameter_vx.h"

class SSDRandomCropMetaNode : public MetaNode {
   public:
    SSDRandomCropMetaNode(){};
    void update_parameters(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data) override;
    std::shared_ptr<SSDRandomCropNode> _node = nullptr;
    std::vector<uint32_t> in_width, in_height;
    void set_threshold(float threshold) { _threshold = threshold; }
    void set_num_of_attempts(int num_of_attempts) { _num_of_attempts = num_of_attempts; }
    Parameter<float> *x_drift_factor;
    Parameter<float> *y_drift_factor;

   private:
    std::shared_ptr<RocalRandomCropParam> _meta_crop_param;
    vx_array _crop_width, _crop_height, _x1, _y1;
    std::vector<uint> _crop_width_val, _crop_height_val, _x1_val, _y1_val;
    unsigned int _dst_width, _dst_height;
    float _threshold = 0.5;
    int _num_of_attempts = 20;
    void initialize();
};