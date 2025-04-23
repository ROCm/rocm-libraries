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

#include "meta_data/meta_node_rotate.h"
void RotateMetaNode::initialize() {
    _angle_val.resize(_batch_size);
}
void RotateMetaNode::update_parameters(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data) {
    initialize();
    if (_batch_size != input_meta_data->size()) {
        _batch_size = input_meta_data->size();
    }
    auto input_roi = _node->get_src_roi();
    _dst_width = _node->get_dst_width();
    _dst_height = _node->get_dst_height();
    _angle = _node->get_angle();
    vxCopyArrayRange((vx_array)_angle, 0, _batch_size, sizeof(float), _angle_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    BoundingBoxCord temp_box = {0, 0, static_cast<float>(_dst_width), static_cast<float>(_dst_height)};
    for (int i = 0; i < _batch_size; i++) {
        auto bb_count = input_meta_data->get_labels_batch()[i].size();
        BoundingBoxCords coords_buf = input_meta_data->get_bb_cords_batch()[i];
        Labels labels_buf = input_meta_data->get_labels_batch()[i];
        BoundingBoxCords bb_coords;
        Labels bb_labels;
        BoundingBoxCord dest_image;
        dest_image.l = dest_image.t = 0;
        dest_image.r = _dst_width;
        dest_image.b = _dst_height;
        for (uint j = 0; j < bb_count; j++) {
            BoundingBoxCord box;
            float src_bb_x, src_bb_y, bb_w, bb_h;
            float dest_cx, dest_cy, src_cx, src_cy;
            float x1, y1, x2, y2, x3, y3, x4, y4, min_x, min_y;
            float rotate[4];
            float radian = RAD(_angle_val[i]);
            rotate[0] = rotate[3] = cos(radian);
            rotate[1] = sin(radian);
            rotate[2] = -1 * rotate[1];
            dest_cx = _dst_width / 2;
            dest_cy = _dst_height / 2;

            src_cx = static_cast<float>(input_roi[i].xywh.w) / 2;
            src_cy = static_cast<float>(input_roi[i].xywh.h) / 2;
            src_bb_x = coords_buf[j].l;
            src_bb_y = coords_buf[j].t;
            bb_w = coords_buf[j].r - coords_buf[j].l;
            bb_h = coords_buf[j].b - coords_buf[j].t;
            x1 = (rotate[0] * (src_bb_x - src_cx)) + (rotate[1] * (src_bb_y - src_cy)) + dest_cx;
            y1 = (rotate[2] * (src_bb_x - src_cx)) + (rotate[3] * (src_bb_y - src_cy)) + dest_cy;
            x2 = (rotate[0] * ((src_bb_x + bb_w) - src_cx)) + (rotate[1] * (src_bb_y - src_cy)) + dest_cx;
            y2 = (rotate[2] * ((src_bb_x + bb_w) - src_cx)) + (rotate[3] * (src_bb_y - src_cy)) + dest_cy;
            x3 = (rotate[0] * (src_bb_x - src_cx)) + (rotate[1] * ((src_bb_y + bb_h) - src_cy)) + dest_cx;
            y3 = (rotate[2] * (src_bb_x - src_cx)) + (rotate[3] * ((src_bb_y + bb_h) - src_cy)) + dest_cy;
            x4 = (rotate[0] * ((src_bb_x + bb_w) - src_cx)) + (rotate[1] * ((src_bb_y + bb_h) - src_cy)) + dest_cx;
            y4 = (rotate[2] * ((src_bb_x + bb_w) - src_cx)) + (rotate[3] * ((src_bb_y + bb_h) - src_cy)) + dest_cy;
            min_x = std::min(x1, std::min(x2, std::min(x3, x4)));
            min_y = std::min(y1, std::min(y2, std::min(y3, y4)));
            box.l = std::min(0.0f, min_x);
            box.t = std::min(0.0f, min_y);
            box.r = std::max(x1, std::max(x2, std::max(x3, x4)));
            box.b = std::max(y1, std::max(y2, std::max(y3, y4)));
            if (BBoxIntersectionOverUnion(box, dest_image) >= _iou_threshold) {
                box.l = std::max(dest_image.l, box.l);
                box.t = std::max(dest_image.t, box.t);
                box.r = std::min(dest_image.r, box.r);
                box.b = std::min(dest_image.b, box.b);
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
