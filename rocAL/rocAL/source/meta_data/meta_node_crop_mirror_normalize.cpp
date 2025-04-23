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

#include "meta_data/meta_node_crop_mirror_normalize.h"
void CropMirrorNormalizeMetaNode::initialize() {
    _width_val.resize(_batch_size);
    _height_val.resize(_batch_size);
    _x1_val.resize(_batch_size);
    _y1_val.resize(_batch_size);
    _mirror_val.resize(_batch_size);
}
void CropMirrorNormalizeMetaNode::update_parameters(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data) {
    initialize();
    if (_batch_size != input_meta_data->size()) {
        _batch_size = input_meta_data->size();
    }
    _mirror = _node->return_mirror();
    _meta_crop_param = _node->return_crop_param();
    _dst_img_width = _meta_crop_param->cropw_arr;
    _dst_img_height = _meta_crop_param->croph_arr;
    _x1 = _meta_crop_param->x1_arr;
    _y1 = _meta_crop_param->y1_arr;
    vxCopyArrayRange((vx_array)_dst_img_width, 0, _batch_size, sizeof(uint), _width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_dst_img_height, 0, _batch_size, sizeof(uint), _height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_x1, 0, _batch_size, sizeof(uint), _x1_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_y1, 0, _batch_size, sizeof(uint), _y1_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_mirror, 0, _batch_size, sizeof(uint), _mirror_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    for (int i = 0; i < _batch_size; i++) {
        auto bb_count = input_meta_data->get_labels_batch()[i].size();
        Labels labels_buf = input_meta_data->get_labels_batch()[i];
        BoundingBoxCords coords_buf = input_meta_data->get_bb_cords_batch()[i];
        BoundingBoxCords bb_coords;
        BoundingBoxCord temp_box = {0, 0, static_cast<float>(_width_val[i]), static_cast<float>(_height_val[i])};
        Labels bb_labels;
        BoundingBoxCord crop_box;
        crop_box.l = (_x1_val[i]);
        crop_box.t = (_y1_val[i]);
        crop_box.r = (_x1_val[i] + _width_val[i]);
        crop_box.b = (_y1_val[i] + _height_val[i]);
        for (uint j = 0; j < bb_count; j++) {
            if (BBoxIntersectionOverUnion(coords_buf[j], crop_box) >= _iou_threshold) {
                float xA = std::max(crop_box.l, coords_buf[j].l);
                float yA = std::max(crop_box.t, coords_buf[j].t);
                float xB = std::min(crop_box.r, coords_buf[j].r);
                float yB = std::min(crop_box.b, coords_buf[j].b);
                coords_buf[j].l = (xA - crop_box.l);
                coords_buf[j].t = (yA - crop_box.t);
                coords_buf[j].r = (xB - crop_box.l);
                coords_buf[j].b = (yB - crop_box.t);
                if (_mirror_val[i] == 1) {
                    auto l = coords_buf[j].l;
                    coords_buf[j].l = _width_val[i] - coords_buf[j].r;
                    coords_buf[j].r = _width_val[i] - l;
                }
                bb_coords.push_back(coords_buf[j]);
                bb_labels.push_back(labels_buf[j]);
            }
        }
        // the following shouldn't happen since all crops should atleast have one bbox
        if (bb_coords.size() == 0) {
            std::cerr << "Crop mirror Normalize - Zero Bounding boxes" << std::endl;
            bb_coords.push_back(temp_box);
            bb_labels.push_back(0);
        }
        output_meta_data->get_bb_cords_batch()[i] = bb_coords;
        output_meta_data->get_labels_batch()[i] = bb_labels;
    }
}
