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

#include "meta_data/meta_data_graph.h"
#include "meta_data/meta_node.h"

typedef  struct { float xc; float yc; float w; float h; } BoundingBoxCord_xcycwh;

class BoundingBoxGraph : public MetaDataGraph {
   public:
    void process(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data) override;
    void update_meta_data(pMetaDataBatch meta_data, DecodedDataInfo decode_image_info) override;
    void update_random_bbox_meta_data(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data, DecodedDataInfo decoded_image_info, CropImageInfo crop_image_info) override;
    void update_box_encoder_meta_data(std::vector<float> *anchors, pMetaDataBatch full_batch_meta_data, float criteria, bool offset, float scale, std::vector<float> &means, std::vector<float> &stds, float *encoded_boxes_data, int *encoded_labels_data) override;
    void update_box_iou_matcher(BoxIouMatcherInfo &iou_matcher_info, int *matches_idx_buffer, pMetaDataBatch full_batch_meta_data) override;
};
