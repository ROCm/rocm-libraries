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

#include "meta_data/meta_node_crop_resize.h"
void CropResizeMetaNode::initialize() {
    _x1_val.resize(_batch_size);
    _y1_val.resize(_batch_size);
    _x2_val.resize(_batch_size);
    _y2_val.resize(_batch_size);
}

void CropResizeMetaNode::update_parameters(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data) {
    initialize();
    if (_batch_size != input_meta_data->size()) {
        _batch_size = input_meta_data->size();
    }
    _meta_crop_param = _node->get_crop_param();
    _x1 = _meta_crop_param->x1_arr;
    _y1 = _meta_crop_param->y1_arr;
    _x2 = _meta_crop_param->x2_arr;
    _y2 = _meta_crop_param->y2_arr;
    auto resize_w = _node->get_dst_width();
    auto resize_h = _node->get_dst_height();
    vxCopyArrayRange((vx_array)_x1, 0, _batch_size, sizeof(uint), _x1_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_y1, 0, _batch_size, sizeof(uint), _y1_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_x2, 0, _batch_size, sizeof(uint), _x2_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_y2, 0, _batch_size, sizeof(uint), _y2_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    BoundingBoxCord temp_box = {0, 0, static_cast<float>(resize_w), static_cast<float>(resize_h)};

    for (int i = 0; i < _batch_size; i++) {
        auto bb_count = input_meta_data->get_labels_batch()[i].size();
        Labels labels_buf = input_meta_data->get_labels_batch()[i];
        BoundingBoxCords coords_buf = input_meta_data->get_bb_cords_batch()[i];
        BoundingBoxCords bb_coords;
        Labels bb_labels;
        BoundingBoxCord crop_box;
        auto _crop_w = _x2_val[i] - _x1_val[i];
        auto _crop_h = _y2_val[i] - _y1_val[i];
        crop_box.l = _x1_val[i];
        crop_box.t = _y1_val[i];
        crop_box.r = _x1_val[i] + _crop_w;
        crop_box.b = _y1_val[i] + _crop_h;
        float _dst_to_src_width_ratio = static_cast<float>(resize_w) / _crop_w;
        float _dst_to_src_height_ratio = static_cast<float>(resize_h) / _crop_h;
        for (uint j = 0; j < bb_count; j++) {
            BoundingBoxCord box = coords_buf[j];
            if (BBoxIntersectionOverUnion(box, crop_box) >= _iou_threshold) {
                float xA = std::max(crop_box.l, box.l);
                float yA = std::max(crop_box.t, box.t);
                float xB = std::min(crop_box.r, box.r);
                float yB = std::min(crop_box.b, box.b);
                box.l = (xA - crop_box.l) * _dst_to_src_width_ratio;
                box.t = (yA - crop_box.t) * _dst_to_src_height_ratio;
                box.r = (xB - crop_box.l) * _dst_to_src_width_ratio;
                box.b = (yB - crop_box.t) * _dst_to_src_height_ratio;
                bb_coords.push_back(box);
                bb_labels.push_back(labels_buf[j]);
            }
        }
        if (bb_coords.size() == 0) {
            bb_coords.push_back(temp_box);
            bb_labels.push_back(0);
        }
        output_meta_data->get_bb_cords_batch()[i] = bb_coords;
        output_meta_data->get_labels_batch()[i] = bb_labels;
    }
}
