/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "meta_data/meta_node_resize.h"

void ResizeMetaNode::update_parameters(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data) {
    if (_batch_size != input_meta_data->size()) {
        _batch_size = input_meta_data->size();
    }
    auto input_roi = _node->get_src_roi();
    auto output_roi = _node->get_dst_roi();
    for (int i = 0; i < _batch_size; i++) {
        float _dst_to_src_width_ratio = static_cast<float>(output_roi[i].xywh.w) / static_cast<float>(input_roi[i].xywh.w);
        float _dst_to_src_height_ratio = static_cast<float>(output_roi[i].xywh.h) / static_cast<float>(input_roi[i].xywh.h);
        unsigned bb_count = input_meta_data->get_labels_batch()[i].size();
        BoundingBoxCords coords_buf = input_meta_data->get_bb_cords_batch()[i];
        Labels labels_buf = input_meta_data->get_labels_batch()[i];
        BoundingBoxCords bb_coords;
        BoundingBoxCord temp_box;
        Labels bb_labels;
        temp_box.l = temp_box.t = temp_box.r = temp_box.b = 0;
        for (uint j = 0; j < bb_count; j++) {
            coords_buf[j].l *= _dst_to_src_width_ratio;
            coords_buf[j].t *= _dst_to_src_height_ratio;
            coords_buf[j].r *= _dst_to_src_width_ratio;
            coords_buf[j].b *= _dst_to_src_height_ratio;
            bb_coords.push_back(coords_buf[j]);
            bb_labels.push_back(labels_buf[j]);
        }
        if (bb_coords.size() == 0) {
            bb_coords.push_back(temp_box);
        }
        output_meta_data->get_bb_cords_batch()[i] = bb_coords;
        output_meta_data->get_labels_batch()[i] = bb_labels;
    }
}
