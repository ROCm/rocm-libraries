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

#include "augmentations/node_ssd_random_crop.h"

#include "pipeline/graph.h"
#include <vx_ext_rpp.h>

#include "pipeline/exception.h"

SSDRandomCropNode::SSDRandomCropNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : CropNode(inputs, outputs),
                                                                                                                  _dest_width(_outputs[0]->info().max_shape()[0]),
                                                                                                                  _dest_height(_outputs[0]->info().max_shape()[1]) {
    _crop_param = std::make_shared<RocalRandomCropParam>(_batch_size);
    _is_ssd = true;
}

void SSDRandomCropNode::create_node() {
    _crop_width_val.resize(_batch_size);
    _crop_height_val.resize(_batch_size);
    _x1_val.resize(_batch_size);
    _y1_val.resize(_batch_size);
    _iou_range.resize(_batch_size);
    if (_node)
        return;

    if (_dest_width == 0 || _dest_height == 0)
        THROW("Uninitialized destination dimension")

    _crop_param->create_array(_graph);
    create_crop_tensor();
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppCrop(_graph->get(), _inputs[0]->handle(), _crop_tensor, _outputs[0]->handle(), input_layout_vx, output_layout_vx, roi_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the crop resize node (vxExtrppNode_ResizeCropbatchPD) failed: " + TOSTR(status))
}

inline double ssd_BBoxIntersectionOverUnion(const BoundingBoxCord &box1, const BoundingBoxCord &box2, bool is_iou = false) {
    double iou;
    float xA = std::max(box1.l, box2.l);
    float yA = std::max(box1.t, box2.t);
    float xB = std::min(box1.r, box2.r);
    float yB = std::min(box1.b, box2.b);
    float intersection_area = std::max((float)0.0, xB - xA) * std::max((float)0.0, yB - yA);
    float box1_h, box1_w, box2_h, box2_w;
    box1_w = box1.r - box1.l;
    box2_w = box2.r - box2.l;
    box1_h = box1.b - box1.t;
    box2_h = box2.b - box2.t;
    float box1_area = box1_h * box1_w;
    float box2_area = box2_h * box2_w;
    if (is_iou)
        iou = intersection_area / float(box1_area + box2_area - intersection_area);
    else
        iou = intersection_area / float(box1_area);

    return iou;
}

void SSDRandomCropNode::update_node() {
    _crop_param->set_image_dimensions(_inputs[0]->info().roi().get_2D_roi());
    _crop_param->update_array();
    Roi2DCords *crop_dims = static_cast<Roi2DCords *>(_crop_coordinates);  // ROI to be cropped from source
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 6);
    const std::vector<std::pair<float, float>> IOU = {std::make_pair(0.0f, 1.0f), std::make_pair(0.1f, 1.0f), std::make_pair(0.3f, 1.0f),
                                                      std::make_pair(0.5f, 1.0f), std::make_pair(0.45f, 1.0f), std::make_pair(0.35f, 1.0f), std::make_pair(0.0f, 1.0f)};
    int sample_option;
    std::pair<float, float> iou;
    float min_iou, max_iou;
    float w_factor = 0.0f, h_factor = 0.0f;
    auto input_roi = _crop_param->in_roi;
    bool invalid_bboxes = true;
    _entire_iou = true;
    BoundingBoxCord crop_box;
    _x1_val = _crop_param->get_x1_arr_val();
    _y1_val = _crop_param->get_y1_arr_val();
    _crop_width_val = _crop_param->get_cropw_arr_val();
    _crop_height_val = _crop_param->get_croph_arr_val();
    std::uniform_real_distribution<float> _float_dis(0.3, 1.0);
    size_t sample = 0;
    for (uint i = 0; i < _batch_size; i++) {
        int bb_count = _meta_data_info->get_labels_batch()[i].size();
        std::vector<int> labels_buf = _meta_data_info->get_labels_batch()[i];
        BoundingBoxCords coords_buf = _meta_data_info->get_bb_cords_batch()[i];

        crop_box.b = _y1_val[i] + _crop_height_val[i];
        crop_box.r = _x1_val[i] + _crop_width_val[i];
        crop_box.l = _x1_val[i];
        crop_box.t = _y1_val[i];
        while (true) {
            sample_option = dis(gen);
            iou = IOU[sample_option];
            _iou_range[i] = iou;
            min_iou = iou.first;
            max_iou = iou.second;
            // sample_option = 0;
            if (!sample_option) {
                crop_box.l = 0;
                crop_box.t = 0;
                crop_box.r = 1;
                crop_box.b = 1;
                break;
            }
            float aspect_ratio = 0.0f;
            for (int j = 0; j < _num_of_attempts; j++) {
                // Setting width and height factor btw 0.3 and 1.0";
                float w_factor = _float_dis(_rngs[sample]);
                float h_factor = _float_dis(_rngs[sample]);
                // aspect ratio check
                aspect_ratio = w_factor / h_factor;
                if ((aspect_ratio < 0.5) || (aspect_ratio > 2.))
                    continue;
            }
            if ((aspect_ratio < 0.5) || (aspect_ratio > 2.))
                continue;

            // Setting width factor btw 0 and 1 - width_factor and height factor btw 0 and 1 - height_factor
            std::uniform_real_distribution<float> l_dis(0.0, 1.0 - w_factor), t_dis(0.0, 1.0 - h_factor);
            float x_factor = l_dis(_rngs[sample]);
            float y_factor = t_dis(_rngs[sample]);
            // Got the crop
            crop_box.l = x_factor;
            crop_box.t = y_factor;
            crop_box.r = crop_box.l + w_factor;
            crop_box.b = crop_box.t + h_factor;

            invalid_bboxes = false;

            for (int j = 0; j < bb_count; j++) {
                float bb_iou = ssd_BBoxIntersectionOverUnion(coords_buf[j], crop_box, _entire_iou);
                if (bb_iou < min_iou || bb_iou > max_iou) {
                    invalid_bboxes = true;
                    break;
                }
            }

            if (invalid_bboxes)
                continue;
            int valid_bbox_count = 0;
            auto left = crop_box.l, top = crop_box.t, right = crop_box.r, bottom = crop_box.b;
            for (int j = 0; j < bb_count; j++) {
                auto x_c = 0.5f * (2 * coords_buf[j].l + coords_buf[j].r);
                auto y_c = 0.5f * (2 * coords_buf[j].t + coords_buf[j].b);
                if ((x_c >= left) && (x_c <= right) && (y_c >= top) && (y_c <= bottom))
                    valid_bbox_count++;
            }
            if (valid_bbox_count == 0)
                continue;
            break;
        }  // while loop

        if (_inputs[0]->info().roi_type() == RocalROIType::XYWH) {
            crop_dims[i].xywh.x = (crop_box.l) * input_roi[i].xywh.w;
            crop_dims[i].xywh.y = (crop_box.t) * input_roi[i].xywh.h;
            crop_dims[i].xywh.w = (crop_box.r - crop_box.l) * input_roi[i].xywh.w;
            crop_dims[i].xywh.h = (crop_box.b - crop_box.t) * input_roi[i].xywh.h;
        } else if (_inputs[0]->info().roi_type() == RocalROIType::LTRB) {
            crop_dims[i].xywh.x = (crop_box.l) * input_roi[i].ltrb.l;
            crop_dims[i].xywh.y = (crop_box.t) * input_roi[i].ltrb.t;
            crop_dims[i].xywh.w = (crop_box.r - crop_box.l) * (input_roi[i].ltrb.r - input_roi[i].ltrb.l + 1);
            crop_dims[i].xywh.h = (crop_box.b - crop_box.t) * (input_roi[i].ltrb.b - input_roi[i].ltrb.t + 1);
        }
    }
    _outputs[0]->update_tensor_roi(_crop_width_val, _crop_height_val);
}

void SSDRandomCropNode::init(FloatParam *crop_area_factor, FloatParam *crop_aspect_ratio, FloatParam *x_drift, FloatParam *y_drift, int num_of_attempts) {
    _crop_param->set_x_drift_factor(core(x_drift));
    _crop_param->set_y_drift_factor(core(y_drift));
    _crop_param->set_area_factor(core(crop_area_factor));
    _crop_param->set_aspect_ratio(core(crop_aspect_ratio));
    _num_of_attempts = num_of_attempts;
}
