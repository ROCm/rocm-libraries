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

#include "meta_data/meta_node_resize_mirror_normalize.h"

void ResizeMirrorNormalizeMetaNode::initialize() {
    _mirror_val.resize(_batch_size);
}

void ResizeMirrorNormalizeMetaNode::update_parameters(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data) {
    initialize();
    if (_batch_size != input_meta_data->size()) {
        _batch_size = input_meta_data->size();
    }
    _mirror = _node->get_mirror();
    auto input_roi = _node->get_src_roi();
    auto output_roi = _node->get_dst_roi();
    vxCopyArrayRange((vx_array)_mirror, 0, _batch_size, sizeof(uint), _mirror_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    for (int i = 0; i < _batch_size; i++) {
        float _dst_to_src_width_ratio = static_cast<float>(output_roi[i].xywh.w) / static_cast<float>(input_roi[i].xywh.w);
        float _dst_to_src_height_ratio = static_cast<float>(output_roi[i].xywh.h) / static_cast<float>(input_roi[i].xywh.h);

        auto bb_count = input_meta_data->get_labels_batch()[i].size();
        BoundingBoxCords coords_buf = input_meta_data->get_bb_cords_batch()[i];
        Labels labels_buf = input_meta_data->get_labels_batch()[i];
        BoundingBoxCords bb_coords;
        Labels bb_labels;
        if (input_meta_data->get_metadata_type() == MetaDataType::PolygonMask) {
            auto mask_data_ptr = input_meta_data->get_mask_cords_batch()[i].data();
            int mask_size = input_meta_data->get_mask_cords_batch()[i].size();
            for (int idx = 0; idx < mask_size; idx += 2) {
                if (_mirror_val[i] == 1) {
                    mask_data_ptr[idx] = output_roi[i].xywh.w - (mask_data_ptr[idx] * _dst_to_src_width_ratio) - 1;
                    mask_data_ptr[idx + 1] = mask_data_ptr[idx + 1] * _dst_to_src_height_ratio;
                } else {
                    mask_data_ptr[idx] = mask_data_ptr[idx] * _dst_to_src_width_ratio;
                    mask_data_ptr[idx + 1] = mask_data_ptr[idx + 1] * _dst_to_src_height_ratio;
                }
            }
            output_meta_data->get_mask_cords_batch()[i] = input_meta_data->get_mask_cords_batch()[i];
            output_meta_data->get_mask_polygons_count_batch()[i] = input_meta_data->get_mask_polygons_count_batch()[i];
            output_meta_data->get_mask_vertices_count_batch()[i] = input_meta_data->get_mask_vertices_count_batch()[i];
        }

        for (uint j = 0; j < bb_count; j++) {
            coords_buf[j].l *= _dst_to_src_width_ratio;
            coords_buf[j].t *= _dst_to_src_height_ratio;
            coords_buf[j].r *= _dst_to_src_width_ratio;
            coords_buf[j].b *= _dst_to_src_height_ratio;
            if (_mirror_val[i] == 1) {
                auto l = coords_buf[j].l;
                coords_buf[j].l = output_roi[i].xywh.w - coords_buf[j].r - 1;
                coords_buf[j].r = output_roi[i].xywh.w - l - 1;
            }
            bb_coords.push_back(coords_buf[j]);
            bb_labels.push_back(labels_buf[j]);
        }
        // get roi width and height of output image
        auto img_roi_size = input_meta_data->get_img_roi_sizes_batch()[i];
        img_roi_size.w = output_roi[i].xywh.w;
        img_roi_size.h = output_roi[i].xywh.h;
        output_meta_data->get_img_roi_sizes_batch()[i] = img_roi_size;
        output_meta_data->get_bb_cords_batch()[i] = bb_coords;
        output_meta_data->get_labels_batch()[i] = bb_labels;
    }
}
