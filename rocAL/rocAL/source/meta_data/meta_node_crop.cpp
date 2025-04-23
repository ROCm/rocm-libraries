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

#include "meta_data/meta_node_crop.h"
void CropMetaNode::initialize() {
    _crop_width_val.resize(_batch_size);
    _crop_height_val.resize(_batch_size);
    _x1_val.resize(_batch_size);
    _y1_val.resize(_batch_size);
}
void CropMetaNode::update_parameters(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data) {
    initialize();
    if (_batch_size != input_meta_data->size()) {
        _batch_size = input_meta_data->size();
    }
    _meta_crop_param = _node->get_crop_param();
    _crop_width = _meta_crop_param->cropw_arr;
    _crop_height = _meta_crop_param->croph_arr;
    _x1 = _meta_crop_param->x1_arr;
    _y1 = _meta_crop_param->y1_arr;
    vxCopyArrayRange((vx_array)_crop_width, 0, _batch_size, sizeof(uint), _crop_width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_crop_height, 0, _batch_size, sizeof(uint), _crop_height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_x1, 0, _batch_size, sizeof(uint), _x1_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_y1, 0, _batch_size, sizeof(uint), _y1_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    for (int i = 0; i < _batch_size; i++) {
        auto bb_count = input_meta_data->get_labels_batch()[i].size();
        Labels labels_buf = input_meta_data->get_labels_batch()[i];
        BoundingBoxCords box_coords_buf = input_meta_data->get_bb_cords_batch()[i];
        BoundingBoxCords bb_coords;
        BoundingBoxCord temp_box;
        Labels bb_labels;
        BoundingBoxCord crop_box;
        crop_box.l = static_cast<float>(_x1_val[i]);
        crop_box.t = static_cast<float>(_y1_val[i]);
        crop_box.r = static_cast<float>((_x1_val[i]) + _crop_width_val[i]);
        crop_box.b = static_cast<float>((_y1_val[i]) + _crop_height_val[i]);

        for (uint j = 0; j < bb_count; j++) {
            if (BBoxIntersectionOverUnion(box_coords_buf[j], crop_box) >= _iou_threshold) {
                float xA = std::max(crop_box.l, box_coords_buf[j].l);
                float yA = std::max(crop_box.t, box_coords_buf[j].t);
                float xB = std::min(crop_box.r, box_coords_buf[j].r);
                float yB = std::min(crop_box.b, box_coords_buf[j].b);
                box_coords_buf[j].l = (xA - crop_box.l);
                box_coords_buf[j].t = (yA - crop_box.t);
                box_coords_buf[j].r = (xB - crop_box.l);
                box_coords_buf[j].b = (yB - crop_box.t);
                bb_coords.push_back(box_coords_buf[j]);
                bb_labels.push_back(labels_buf[j]);
            }
        }
        if (bb_coords.size() == 0) {
            temp_box.l = 0;
            temp_box.t = 0;
            temp_box.r = _crop_width_val[i];
            temp_box.b = _crop_height_val[i];
            bb_coords.push_back(temp_box);
            bb_labels.push_back(0);
        }
        output_meta_data->get_bb_cords_batch()[i] = bb_coords;
        output_meta_data->get_labels_batch()[i] = bb_labels;
    }
}