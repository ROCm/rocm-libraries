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
#include <list>

#include "loaders/circular_buffer.h"
#include "meta_data/meta_data.h"
#include "meta_data/meta_node.h"
#include "pipeline/node.h"
#include "parameters/parameter_factory.h"
#include "meta_data/randombboxcrop_meta_data_reader.h"

typedef struct {
    std::vector<float> *anchors;
    float high_threshold;
    float low_threshold;
    bool allow_low_quality_matches;
} BoxIouMatcherInfo;

class MetaDataGraph {
   public:
    virtual ~MetaDataGraph() = default;
    virtual void process(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data) = 0;
    virtual void update_meta_data(pMetaDataBatch meta_data, DecodedDataInfo decoded_data_info) = 0;
    virtual void update_random_bbox_meta_data(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data, DecodedDataInfo decoded_data_info, CropImageInfo crop_image_info) = 0;
    virtual void update_box_encoder_meta_data(std::vector<float> *anchors, pMetaDataBatch full_batch_meta_data, float criteria, bool offset, float scale, std::vector<float> &means, std::vector<float> &stds, float *encoded_boxes_data, int *encoded_labels_data) = 0;
    virtual void update_box_iou_matcher(BoxIouMatcherInfo &iou_matcher_info, int *matches_idx_buffer, pMetaDataBatch full_batch_meta_data) = 0;
    std::list<std::shared_ptr<MetaNode>> _meta_nodes;
};
