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
#include "meta_data/bounding_box_graph.h"

void BoundingBoxGraph::process(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data) {
    size_t num_meta_nodes = _meta_nodes.size();
    for (auto &meta_node : _meta_nodes) {
        meta_node->update_parameters(input_meta_data, output_meta_data);
        if (--num_meta_nodes > 0)
            input_meta_data = output_meta_data->clone();
    }
}

// This function is used to rescale the metadata values w.r.t the decoded image sizes
void BoundingBoxGraph::update_meta_data(pMetaDataBatch input_meta_data, DecodedDataInfo decode_image_info) {
    std::vector<uint32_t> original_height = decode_image_info._original_height;
    std::vector<uint32_t> original_width = decode_image_info._original_width;
    std::vector<uint32_t> roi_width = decode_image_info._roi_width;
    std::vector<uint32_t> roi_height = decode_image_info._roi_height;
    for (int i = 0; i < input_meta_data->size(); i++) {
        float _dst_to_src_width_ratio = roi_width[i] / static_cast<float>(original_width[i]);
        float _dst_to_src_height_ratio = roi_height[i] / static_cast<float>(original_height[i]);
        unsigned bb_count = input_meta_data->get_labels_batch()[i].size();
        BoundingBoxCords coords_buf = input_meta_data->get_bb_cords_batch()[i];
        BoundingBoxCords bb_coords;
        BoundingBoxCord temp_box;
        temp_box.l = temp_box.t = temp_box.r = temp_box.b = 0;
        for (uint j = 0; j < bb_count; j++) {
            coords_buf[j].l *= _dst_to_src_width_ratio;
            coords_buf[j].t *= _dst_to_src_height_ratio;
            coords_buf[j].r *= _dst_to_src_width_ratio;
            coords_buf[j].b *= _dst_to_src_height_ratio;
            bb_coords.push_back(coords_buf[j]);
        }
        if (bb_coords.size() == 0) {
            bb_coords.push_back(temp_box);
        }
        input_meta_data->get_bb_cords_batch()[i] = bb_coords;
    }
}

inline float ssd_BBoxIntersectionOverUnion(const BoundingBoxCord &box1, const float &box1_area, const BoundingBoxCord &box2) {
    float xA = std::max(static_cast<float>(box1.l), box2.l);
    float yA = std::max(static_cast<float>(box1.t), box2.t);
    float xB = std::min(static_cast<float>(box1.r), box2.r);
    float yB = std::min(static_cast<float>(box1.b), box2.b);
    float intersection_area = std::max((float)0.0, xB - xA) * std::max((float)0.0, yB - yA);
    float box2_area = (box2.b - box2.t) * (box2.r - box2.l);
    return (float)(intersection_area / (box1_area + box2_area - intersection_area));
}

void BoundingBoxGraph::update_random_bbox_meta_data(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data, DecodedDataInfo decode_image_info, CropImageInfo crop_image_info) {
    std::vector<uint32_t> original_height = decode_image_info._original_height;
    std::vector<uint32_t> original_width = decode_image_info._original_width;
    std::vector<uint32_t> roi_width = decode_image_info._roi_width;
    std::vector<uint32_t> roi_height = decode_image_info._roi_height;
    auto crop_cords = crop_image_info._crop_image_coords;
    for (int i = 0; i < input_meta_data->size(); i++) {
        auto bb_count = input_meta_data->get_labels_batch()[i].size();
        BoundingBoxCords coords_buf;
        Labels labels_buf;
        coords_buf.resize(bb_count);
        labels_buf.resize(bb_count);
        memcpy(labels_buf.data(), input_meta_data->get_labels_batch()[i].data(), sizeof(int) * bb_count);
        memcpy((void *)coords_buf.data(), input_meta_data->get_bb_cords_batch()[i].data(), input_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        BoundingBoxCords bb_coords;
        Labels bb_labels;
        BoundingBoxCord crop_box;
        crop_box.l = crop_cords[i][0];
        crop_box.t = crop_cords[i][1];
        crop_box.r = crop_box.l + crop_cords[i][2];
        crop_box.b = crop_box.t + crop_cords[i][3];
        float w_factor = 1.0 / crop_cords[i][2];
        float h_factor = 1.0 / crop_cords[i][3];
        for (uint j = 0; j < bb_count; j++) {
            // Mask Criteria
            auto x_c = 0.5 * (coords_buf[j].l + coords_buf[j].r);
            auto y_c = 0.5 * (coords_buf[j].t + coords_buf[j].b);
            if ((x_c > crop_box.l) && (x_c < crop_box.r) && (y_c > crop_box.t) && (y_c < crop_box.b)) {
                float xA = std::max(crop_box.l, coords_buf[j].l);
                float yA = std::max(crop_box.t, coords_buf[j].t);
                float xB = std::min(crop_box.r, coords_buf[j].r);
                float yB = std::min(crop_box.b, coords_buf[j].b);
                coords_buf[j].l = (xA - crop_box.l) * w_factor;
                coords_buf[j].t = (yA - crop_box.t) * h_factor;
                coords_buf[j].r = (xB - crop_box.l) * w_factor;
                coords_buf[j].b = (yB - crop_box.t) * h_factor;
                bb_coords.push_back(coords_buf[j]);
                bb_labels.push_back(labels_buf[j]);
            }
        }
        if (bb_coords.size() == 0) {
            THROW("Bounding box co-ordinates not found in the image ");
        }
        output_meta_data->get_bb_cords_batch()[i] = bb_coords;
        output_meta_data->get_labels_batch()[i] = bb_labels;
    }
}

inline void calculate_ious_for_box(float *ious, BoundingBoxCord &box, BoundingBoxCord *anchors, unsigned int num_anchors) {
    float box_area = (box.b - box.t) * (box.r - box.l);
    ious[0] = ssd_BBoxIntersectionOverUnion(box, box_area, anchors[0]);

    int best_idx = 0;
    float best_iou = ious[0];
    for (unsigned int anchor_idx = 1; anchor_idx < num_anchors; anchor_idx++) {
        ious[anchor_idx] = ssd_BBoxIntersectionOverUnion(box, box_area, anchors[anchor_idx]);
        if (ious[anchor_idx] > best_iou) {
            best_iou = ious[anchor_idx];
            best_idx = anchor_idx;
        }
    }
    // For best default box matched with current object let iou = 2, to make sure there is a match,
    // as this object will be the best (highest IoU), for this default box
    ious[best_idx] = 2.;
}

inline int find_best_box_for_anchor(unsigned anchor_idx, const std::vector<float> &ious, unsigned num_boxes, unsigned anchors_size) {
    unsigned best_idx = 0;
    float best_iou = ious[anchor_idx];
    for (unsigned bbox_idx = 1; bbox_idx < num_boxes; ++bbox_idx) {
        if (ious[bbox_idx * anchors_size + anchor_idx] >= best_iou) {
            best_iou = ious[bbox_idx * anchors_size + anchor_idx];
            best_idx = bbox_idx;
        }
    }
    return best_idx;
}

void BoundingBoxGraph::update_box_encoder_meta_data(std::vector<float> *anchors, pMetaDataBatch full_batch_meta_data, float criteria, bool offset, float scale, std::vector<float> &means, std::vector<float> &stds, float *encoded_boxes_data, int *encoded_labels_data) {
#pragma omp parallel for
    for (int i = 0; i < full_batch_meta_data->size(); i++) {
        BoundingBoxCord *bbox_anchors = reinterpret_cast<BoundingBoxCord *>(anchors->data());
        auto bb_count = full_batch_meta_data->get_labels_batch()[i].size();
        int *bb_labels = full_batch_meta_data->get_labels_batch()[i].data();
        BoundingBoxCord *bb_coords = reinterpret_cast<BoundingBoxCord *>(full_batch_meta_data->get_bb_cords_batch()[i].data());
        unsigned anchors_size = anchors->size() / 4;  // divide the anchors_size by 4 to get the total number of anchors
        int *encoded_labels = encoded_labels_data + (i * anchors_size);
        BoundingBoxCord_xcycwh *encoded_bb = reinterpret_cast<BoundingBoxCord_xcycwh *>(encoded_boxes_data + (i * anchors_size * 4));
        // Calculate Ious
        // ious size - bboxes count x anchors count
        std::vector<float> ious(bb_count * anchors_size);
        for (uint bb_idx = 0; bb_idx < bb_count; bb_idx++) {
            auto iou_rows = ious.data() + (bb_idx * (anchors_size));
            calculate_ious_for_box(iou_rows, bb_coords[bb_idx], bbox_anchors, anchors_size);
        }
        float inv_stds[4] = {(float)(1. / stds[0]), (float)(1. / stds[1]), (float)(1. / stds[2]), (float)(1. / stds[3])};
        float half_scale = 0.5 * scale;
        // Depending on the matches ->place the best bbox instead of the corresponding anchor_idx in anchor
        for (unsigned anchor_idx = 0; anchor_idx < anchors_size; anchor_idx++) {
            BoundingBoxCord_xcycwh box_bestidx, anchor_xcyxwh;
            BoundingBoxCord *p_anchor = &bbox_anchors[anchor_idx];
            const auto best_idx = find_best_box_for_anchor(anchor_idx, ious, bb_count, anchors_size);
            // Filter matches by criteria
            if (ious[(best_idx * anchors_size) + anchor_idx] > criteria)  // Its a match
            {
                // Convert the "ltrb" format to "xcycwh"
                if (offset) {
                    box_bestidx.xc = (bb_coords[best_idx].l + bb_coords[best_idx].r) * half_scale;  // xc
                    box_bestidx.yc = (bb_coords[best_idx].t + bb_coords[best_idx].b) * half_scale;  // yc
                    box_bestidx.w = (bb_coords[best_idx].r - bb_coords[best_idx].l) * scale;        // w
                    box_bestidx.h = (bb_coords[best_idx].b - bb_coords[best_idx].t) * scale;        // h
                    // Convert the "ltrb" format to "xcycwh"
                    anchor_xcyxwh.xc = (p_anchor->l + p_anchor->r) * half_scale;  // xc
                    anchor_xcyxwh.yc = (p_anchor->t + p_anchor->b) * half_scale;  // yc
                    anchor_xcyxwh.w = (p_anchor->r - p_anchor->l) * scale;        // w
                    anchor_xcyxwh.h = (p_anchor->b - p_anchor->t) * scale;        // h
                    // Reference for offset calculation between the Ground Truth bounding boxes & anchor boxes in <xc,yc,w,h> format
                    // https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#predictions-vis-%C3%A0-vis-priors
                    box_bestidx.xc = ((box_bestidx.xc - anchor_xcyxwh.xc) / anchor_xcyxwh.w - means[0]) * inv_stds[0];
                    box_bestidx.yc = ((box_bestidx.yc - anchor_xcyxwh.yc) / anchor_xcyxwh.h - means[1]) * inv_stds[1];
                    box_bestidx.w = (std::log(box_bestidx.w / anchor_xcyxwh.w) - means[2]) * inv_stds[2];
                    box_bestidx.h = (std::log(box_bestidx.h / anchor_xcyxwh.h) - means[3]) * inv_stds[3];
                    encoded_bb[anchor_idx] = box_bestidx;
                    encoded_labels[anchor_idx] = bb_labels[best_idx];
                } else {
                    box_bestidx.xc = 0.5 * (bb_coords[best_idx].l + bb_coords[best_idx].r);  // xc
                    box_bestidx.yc = 0.5 * (bb_coords[best_idx].t + bb_coords[best_idx].b);  // yc
                    box_bestidx.w = bb_coords[best_idx].r - bb_coords[best_idx].l;           // w
                    box_bestidx.h = bb_coords[best_idx].b - bb_coords[best_idx].t;           // h
                    encoded_bb[anchor_idx] = box_bestidx;
                    encoded_labels[anchor_idx] = bb_labels[best_idx];
                }
            } else {
                // Not a match
                if (offset) {
                    encoded_bb[anchor_idx] = {0, 0, 0, 0};
                    encoded_labels[anchor_idx] = 0;
                } else {
                    // Convert the "ltrb" format to "xcycwh"
                    encoded_bb[anchor_idx].xc = 0.5 * (p_anchor->l + p_anchor->r);  // xc
                    encoded_bb[anchor_idx].yc = 0.5 * (p_anchor->t + p_anchor->b);  // yc
                    encoded_bb[anchor_idx].w = (-p_anchor->l + p_anchor->r);        // w
                    encoded_bb[anchor_idx].h = (-p_anchor->t + p_anchor->b);        // h
                    encoded_labels[anchor_idx] = 0;
                }
            }
        }
    }
}

void BoundingBoxGraph::update_box_iou_matcher(BoxIouMatcherInfo &iou_matcher_info, int *matches_idx_buffer, pMetaDataBatch full_batch_meta_data) {
    auto bb_coords_batch = full_batch_meta_data->get_bb_cords_batch();
    unsigned anchors_size = iou_matcher_info.anchors->size() / 4;  // divide the anchors_size by 4 to get the total number of anchors
    BoundingBoxCord *bbox_anchors = reinterpret_cast<BoundingBoxCord *>(iou_matcher_info.anchors->data());

    std::vector<int *> matches(full_batch_meta_data->size());
    for (int i = 0; i < full_batch_meta_data->size(); i++) {
        matches[i] = reinterpret_cast<int *>(matches_idx_buffer + i * anchors_size);
    }

#pragma omp parallel for
    for (int i = 0; i < full_batch_meta_data->size(); i++) {
        auto bb_coords = bb_coords_batch[i];
        auto bb_count = bb_coords.size();

        std::vector<float> matched_vals(anchors_size, -1.0);
        std::vector<int> low_quality_preds(anchors_size, -1);

        // Calculate IoU's, The number of IoU Values calculated will be (bb_count x anchors_size)
        for (unsigned bb_idx = 0; bb_idx < bb_count; bb_idx++) {
            BoundingBoxCord box = bb_coords[bb_idx];
            float box_area = (box.b - box.t) * (box.r - box.l);
            float best_bbox_iou = -1.0f;
            std::vector<float> bbox_iou(anchors_size);  // IoU value for bbox mapped with each anchor
            for (unsigned int anchor_idx = 0; anchor_idx < anchors_size; anchor_idx++) {
                float iou_val = ssd_BBoxIntersectionOverUnion(box, box_area, bbox_anchors[anchor_idx]);
                bbox_iou[anchor_idx] = iou_val;

                // Find col maximum in (bb_count x anchors_size) IoU values calculated
                if (iou_val > matched_vals[anchor_idx]) {
                    matched_vals[anchor_idx] = iou_val;
                    matches[i][anchor_idx] = static_cast<int>(bb_idx);
                }

                // Find row maximum in (bb_count x anchors_size) IoU values calculated
                if (iou_matcher_info.allow_low_quality_matches) {
                    if (iou_val > best_bbox_iou) best_bbox_iou = iou_val;
                }
            }

            if (iou_matcher_info.allow_low_quality_matches) {
                for (unsigned int anchor_idx = 0; anchor_idx < anchors_size; anchor_idx++) {  // if the element is found
                    if (fabs(bbox_iou[anchor_idx] - best_bbox_iou) < 1e-6)                    // Compare the IOU values and check if they are equal with a tolerance of 1e-6
                        low_quality_preds[anchor_idx] = anchor_idx;
                }
            }
        }

        // Update matched indices based on thresholds and low quality matches
        for (uint pred_idx = 0; pred_idx < anchors_size; pred_idx++) {
            if (!(iou_matcher_info.allow_low_quality_matches && low_quality_preds[pred_idx] != -1)) {
                if (matched_vals[pred_idx] < iou_matcher_info.low_threshold) {
                    matches[i][pred_idx] = -1;
                } else if ((matched_vals[pred_idx] < iou_matcher_info.high_threshold)) {
                    matches[i][pred_idx] = -2;
                }
            }
        }

        matched_vals.clear();
        low_quality_preds.clear();
    }
}
