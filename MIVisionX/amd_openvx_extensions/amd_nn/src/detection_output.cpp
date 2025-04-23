/*
Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "kernels.h"
#include <float.h>
#include <string.h>
#include <map>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>
#include <assert.h>
using namespace std;

class NormalizedBBox
{
    float size_;

    public:
    bool has_size;
    float xmin, ymin,xmax,ymax;
    bool set_size(float value){size_ = value; return true;}
    float size() const {return size_;}
};

typedef std::map<int, std::vector<NormalizedBBox> > LabelBBox;

static vx_status VX_CALLBACK validate(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[])
{
    // check tensor dims.
    vx_enum type;
    vx_size num_dims;
    vx_size input_dims[4],  output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type!= VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type!= VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type!= VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    vx_enum scalar_type;
    vx_int32 num_classes;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &num_classes, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(num_classes < 0) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #4 scalar type=%d (num classes must be greater than 0)\n", num_classes);

    vx_int32 share_location;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &share_location, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(share_location < 0 || share_location > 1) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #5 scalar type=%d (must be 1(true)/0(false))\n", share_location);

    vx_int32 background_label_id;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[5], &background_label_id, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(background_label_id < 0) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #6 scalar type=%d (must be greater than 0)\n", background_label_id);

    vx_float32 nms_threshold;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[6], &nms_threshold, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(nms_threshold < 0) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #7 scalar type=%f (must be greater than 0)\n", nms_threshold);

    vx_int32 code_type; 
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[7], &code_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(code_type < 1 || code_type > 3) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #8 code type=%d \n", code_type);

    vx_int32 keep_top_k;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[8], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[8], &keep_top_k, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(keep_top_k < 0) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #9 scalar type=%d (must be greater than 0)\n", keep_top_k);

    vx_int32 variance_encoded_in_target;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[9], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[9], &variance_encoded_in_target, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(variance_encoded_in_target < 0 || variance_encoded_in_target > 1) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #10 scalar type=%d (must be 1(true)/0(false))\n", variance_encoded_in_target);

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[10], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[10], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[10], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    // output tensor configuration
    type = VX_TYPE_FLOAT32;
    num_dims = 4;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[10], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[10], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[10], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    if(parameters[11])
    {
        vx_float32 eta;
        ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[11], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
        if(scalar_type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[11], &eta, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        if(eta <= 0 || eta > 1) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #12 scalar type=%f (must be greater than 0)\n", eta);        
    }

    if(parameters[12])
    {
        vx_int32 top_k;
        ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[12], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
        if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[12], &top_k, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        if(top_k < 0) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #13 scalar type=%d (must be greater than 0)\n", top_k);    
    }
    if(parameters[13])
    {
        vx_float32 confidence_threshold;
        ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[13], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
        if(scalar_type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[13], &confidence_threshold, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        if(confidence_threshold < 0) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #14 scalar type=%f (must be greater than 0)\n", confidence_threshold);        
    }
    return VX_SUCCESS;
}

static float BBoxSize(const NormalizedBBox& bbox, bool normalized)
{
    if (bbox.xmax < bbox.xmin || bbox.ymax < bbox.ymin)
    {
        return 0; 
    }
    else
    {
        if (bbox.has_size == true)
        {
            return bbox.size();
        }
        else
        {
            float width = bbox.xmax - bbox.xmin;
            float height = bbox.ymax - bbox.ymin;
            if (normalized)
            {
                return width * height;
            }
            else
            {
                // If bbox is not within range [0, 1].
                return (width + 1) * (height + 1);
            }
        }
    }
}



static void GetLocPredictions(const float* locData, const int num,
                           const int numPriors, const int num_loc_classes,
                           const bool share_location, std::vector<LabelBBox>* locPreds)
{
    locPreds->clear();
    if (share_location) 
    {
        assert(num_loc_classes == 1);
    }
    locPreds->resize(num);
    for (int i = 0; i < num; ++i) 
    {
        LabelBBox& label_bbox = (*locPreds)[i];
        for (int p = 0; p < numPriors; ++p) 
        {
            int startIdx = p * num_loc_classes * 4;
            for (int c = 0; c < num_loc_classes; ++c) 
            {
                int label = share_location ? -1 : c;
                if (label_bbox.find(label) == label_bbox.end()) 
                {
                    label_bbox[label].resize(numPriors);
                }
                label_bbox[label][p].xmin = locData[startIdx + c * 4];
                label_bbox[label][p].ymin = locData[startIdx + c * 4 + 1];
                label_bbox[label][p].xmax = locData[startIdx + c * 4 + 2];
                label_bbox[label][p].ymax = locData[startIdx + c * 4 + 3];
            }
        }
        locData += numPriors * num_loc_classes * 4;
    }
}

static void GetConfidenceScores(float* confData, const int num, const int numPriors, const int num_classes, vector<map<int, vector<float> > >* allConfidenceScores)
{
    allConfidenceScores->clear();
    allConfidenceScores->resize(num);
    for (int i = 0; i < num; i++)
    {
        map<int, vector<float> >&label_scores = (*allConfidenceScores)[i];
        for(int p = 0; p < numPriors; p++)
        {
           
            int start_index = p * num_classes;
            for(int c = 0; c < num_classes; c++)
            {   
                label_scores[c].push_back(confData[start_index+c]);
            }
            
        }
        confData += num_classes * numPriors;
    }
}

static void GetPriorBBoxes(float* priorData, const int numPriors, vector<NormalizedBBox>* priorBBoxes, vector<vector<float> >* priorVariances)
{
    priorBBoxes->clear();
    priorVariances->clear();

    for (int i = 0; i < numPriors; ++i)
    {
        int startIdx = i * 4;
        NormalizedBBox bbox;
        bbox.xmin = priorData[startIdx];
        bbox.ymin = priorData[startIdx + 1];
        bbox.xmax = priorData[startIdx + 2];
        bbox.ymax = priorData[startIdx + 3];
        float bbox_size = BBoxSize(bbox, false);
        bbox.set_size(bbox_size);
        priorBBoxes->push_back(bbox);        
    }

    for (int i = 0; i < numPriors; ++i)
    {
        int startIdx = (numPriors + i) * 4;
        vector<float> var;
        for (int j = 0; j < 4; ++j)
        {
            var.push_back(priorData[startIdx + j]);
        }
        priorVariances->push_back(var);
    }

}

void ClipBBox(const NormalizedBBox& bbox, NormalizedBBox* clip_bbox)
{
    clip_bbox->xmin = std::max(std::min(bbox.xmin, 1.f), 0.f);
    clip_bbox->ymin = std::max(std::min(bbox.ymin, 1.f), 0.f);
    clip_bbox->xmax = std::max(std::min(bbox.xmax, 1.f), 0.f);
    clip_bbox->ymax = std::max(std::min(bbox.ymax, 1.f), 0.f);
    float clip_bbox_size = BBoxSize(*clip_bbox, false);
    clip_bbox->set_size(clip_bbox_size);

}

void DecodeBBox(const NormalizedBBox& priorBBox, const vector<float>& priorVariance, string code_type, const bool variance_encoded_in_target, const bool clip_bbox, 
                const NormalizedBBox& bbox, NormalizedBBox* decode_bbox)
{
    if(code_type == "CORNER")
    {
        if(variance_encoded_in_target)
        {
            decode_bbox->xmin = priorBBox.xmin + bbox.xmin;
            decode_bbox->ymin = priorBBox.ymin + bbox.ymin;
            decode_bbox->xmax = priorBBox.xmax + bbox.xmax;
            decode_bbox->ymax = priorBBox.ymax + bbox.ymax;
        }
        else
        {
            decode_bbox->xmin = priorBBox.xmin + priorVariance[0]*bbox.xmin;
            decode_bbox->ymin = priorBBox.ymin + priorVariance[1]*bbox.ymin;
            decode_bbox->xmax = priorBBox.xmax + priorVariance[2]*bbox.xmax;
            decode_bbox->ymax = priorBBox.ymax + priorVariance[3]*bbox.ymax;
        }
    }
    else if(code_type == "CENTER_SIZE")
    {
        float prior_width = priorBBox.xmax - priorBBox.xmin;
        float prior_height = priorBBox.ymax - priorBBox.ymin;
        assert(prior_width > 0);
        assert(prior_height > 0);
        float prior_center_x = (priorBBox.xmin + priorBBox.xmax)/2;
        float prior_center_y = (priorBBox.ymin + priorBBox.ymax)/2;

        float decode_bbox_center_x, decode_bbox_center_y;
        float decode_bbox_width, decode_bbox_height;
        if(variance_encoded_in_target)
        {
            decode_bbox_center_x = bbox.xmin * prior_width + prior_center_x;
            decode_bbox_center_y = bbox.ymin * prior_height + prior_center_y;
            decode_bbox_width = exp(bbox.xmax) * prior_width;
            decode_bbox_height = exp(bbox.ymax) * prior_height;
        }
        else
        {
            decode_bbox_center_x = priorVariance[0] * bbox.xmin * prior_width + prior_center_x;
            decode_bbox_center_y = priorVariance[1] * bbox.ymin * prior_height + prior_center_y;
            decode_bbox_width = exp(priorVariance[2] * bbox.xmax) * prior_width;
            decode_bbox_height = exp(priorVariance[3] * bbox.ymax) * prior_height;
        }

        decode_bbox->xmin = decode_bbox_center_x - decode_bbox_width/2;
        decode_bbox->ymin = decode_bbox_center_y - decode_bbox_height/2;
        decode_bbox->xmax = decode_bbox_center_x + decode_bbox_width/2;
        decode_bbox->ymax = decode_bbox_center_y + decode_bbox_height/2;
        float bbox_size = BBoxSize(*decode_bbox, false);
        decode_bbox->set_size(bbox_size);
        if(clip_bbox)
        {
            ClipBBox(*decode_bbox, decode_bbox);
        }
    }
}

void DecodeBBoxes(const vector<NormalizedBBox>& priorBBoxes, const vector<vector<float> >& priorVariances, string code_type, const bool variance_encoded_in_target, const bool clip_bbox, 
                    const vector<NormalizedBBox>& bboxes, vector<NormalizedBBox>* decode_bboxes)
{
    assert(priorBBoxes.size() == priorVariances.size());
    assert(priorBBoxes.size() == bboxes.size());

    int numBBoxes = priorBBoxes.size();
    if(numBBoxes >= 1)
    {
        assert(priorVariances[0].size() == 4);
    }

    decode_bboxes->clear();
    for(int i = 0; i < numBBoxes; i++)
    {
        NormalizedBBox decode_bbox;
        DecodeBBox(priorBBoxes[i], priorVariances[i], code_type, variance_encoded_in_target, clip_bbox, bboxes[i], &decode_bbox);
        decode_bboxes->push_back(decode_bbox);
    }

}

void DecodeBBoxesAll(const vector<LabelBBox>& allLocPreds, const vector<NormalizedBBox>& priorBBoxes, const vector<vector<float> >& priorVariances, const int num, const bool share_location,
                    const int num_loc_classes, const int background_label_id, string code_type, const bool variance_encoded_in_target, const bool clip, vector<LabelBBox>* all_decode_bboxes)
{
    assert(allLocPreds.size() == num);
    all_decode_bboxes->clear();
    all_decode_bboxes->resize(num);

    for(int i = 0; i < num; i++)
    {
        // Decode predictions into bboxes.
        LabelBBox& decode_bboxes = (*all_decode_bboxes)[i];
        for (int c = 0; c < num_loc_classes; ++c) 
        {
            int label = share_location ? -1 : c;
            if (label == background_label_id) 
            {
                // Ignore background class.
                continue;
            }

            if(allLocPreds[i].find(label) == allLocPreds[i].end())
            {
                ERRMSG(VX_ERROR_INVALID_VALUE, "decodeBBoxesAll: could not find location predictions for label %d\n", label);
            }
            const vector<NormalizedBBox>& label_loc_preds = allLocPreds[i].find(label)->second;
            DecodeBBoxes(priorBBoxes, priorVariances, code_type, variance_encoded_in_target, clip, label_loc_preds, &(decode_bboxes[label]));
        }
    }
}

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k, std::vector<std::pair<float, int> >* score_index_vec)
{
    for(int i = 0; i < scores.size(); i++)
    {
        if(scores[i] > threshold)
        {
            score_index_vec->push_back(std::make_pair(scores[i], i));
        }
    }

    std::stable_sort(score_index_vec->begin() , score_index_vec->end(), SortScorePairDescend<int>);

    if (top_k > -1 && top_k < score_index_vec->size())
    {
        score_index_vec->resize(top_k);
    }
}



static float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2, const bool normalized)
{
    NormalizedBBox intersect_bbox;
    intersect_bbox.xmin = std::max(bbox1.xmin, bbox2.xmin);
    intersect_bbox.ymin = std::max(bbox1.ymin, bbox2.ymin);
    intersect_bbox.xmax = std::min(bbox1.xmax, bbox2.xmax);
    intersect_bbox.ymax = std::min(bbox1.ymax, bbox2.ymax);

    float intersect_size = BBoxSize(intersect_bbox, normalized);
    if (intersect_size > 0)
    {
        float bbox1_size = BBoxSize(bbox1, normalized);
        float bbox2_size = BBoxSize(bbox2, normalized);
        return intersect_size / (bbox1_size + bbox2_size - intersect_size);
    }
    else
    {
        return 0.;
    }
}

void ApplyNMSFast(const vector<NormalizedBBox>& bboxes, const vector<float>& scores, const float score_threshold, const float nms_threshold, const int top_k, vector<int>* indices, const float eta)
{
    assert(bboxes.size() == scores.size());
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);
    float adaptive_threshold = nms_threshold;
    indices->clear();
    while(score_index_vec.size() != 0)
    {
        const int idx = score_index_vec.front().second;
        bool keep = true;
        for(int k = 0; k < indices->size() && keep; k++)
        {
            if(keep)
            {
                const int kept_idx = (*indices)[k];
                float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx], false);
                keep = overlap <= adaptive_threshold;
            }
            else
            {
                break;
            }
        }

        if (keep)
            indices->push_back(idx);

        score_index_vec.erase(score_index_vec.begin());
        if(keep && adaptive_threshold > 0.5 && eta < 1)
            adaptive_threshold *= eta;
    }
}

static vx_status VX_CALLBACK processDetectionOutput(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    //get tensor dimensions
    vx_size input_dims_0[4], input_dims_1[4], input_dims_2[4], output_dims[4];
    vx_size num_of_dims;
    vx_enum type;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims_0, sizeof(input_dims_0)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims_1, sizeof(input_dims_1)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, input_dims_2, sizeof(input_dims_2)));
    
    vx_int32 num_classes, share_location, background_label_id, top_k, keep_top_k, variance_encoded_in_target;
    vx_float32 nms_threshold, confidence_threshold, eta;
    //vx_scalar code_type;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &num_classes, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &share_location, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[5], &background_label_id, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[6], &nms_threshold, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    vx_int32 code_type;
    string s_code_type; 
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[7], &code_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(code_type == 1) s_code_type = "CORNER";
    else if(code_type == 2) s_code_type = "CENTER_SIZE";
    else ERRMSG(VX_ERROR_INVALID_VALUE, "processDetectionOutput: code_type not supported %d\n", code_type);

    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[8], &keep_top_k, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[9], &variance_encoded_in_target, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[10], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[10], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));

    if(parameters[11])
    {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[11], &eta, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));    
    }
    else
    {
        eta = 1;
    }

    if(parameters[12])
    {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[12], &top_k, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }
    else
    {
        top_k = -1;
    }
    
    if(parameters[13])
    {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[13], &confidence_threshold, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));    
    }
    else
    {
        confidence_threshold = -FLT_MAX;
    }
 
    vx_int32 num_loc_classes = share_location ? 1:num_classes;
    vx_int32 num_priors = input_dims_2[1] / 4;
    if ((num_priors * num_loc_classes * 4) != input_dims_0[2])
    {
        printf("processDetectionOutput: detection_output: Number of priors must match number of location predictions\n");
        exit(0);
    }            
    if((num_priors * num_classes) != input_dims_1[2])   
    {
        printf("processDetectionOutput: detection_output: Number of priors must match number of confidence predictions\n");  
        exit(0);
    }  
    int num_batches = input_dims_0[3];
    int numPriors = input_dims_2[1]/4;

    //get memory pointers for all inputs
    vx_map_id map_id;
    vx_size stride[4];
    float * ptr;
    vx_enum usage = VX_READ_ONLY;
    vx_status status;
    vx_size count_tensor_loc = input_dims_0[0]*input_dims_0[1]*input_dims_0[2]*input_dims_0[3];
    vx_size count_tensor_conf = input_dims_1[0]*input_dims_1[1]*input_dims_1[2]*input_dims_1[3];
    vx_size count_tensor_prior = input_dims_2[0]*input_dims_2[1]*input_dims_2[2]*input_dims_2[3];
    
    float *locData = new float[count_tensor_loc];
    float *confData = new float[count_tensor_conf];
    float *priorData = new float[count_tensor_prior];

    status = vxMapTensorPatch((vx_tensor)parameters[0], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST);
    if(status)
    {
        std::cerr << "ERROR: vxMapTensorPatch() failed for input#1 (" << status << ")" << std::endl;
        return -1;
    }

    memcpy(locData, ptr, (count_tensor_loc*sizeof(float)));

    status = vxUnmapTensorPatch((vx_tensor)parameters[0], map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for input#1 (" << status << ")" << std::endl;
        return -1;
    }

    status = vxMapTensorPatch((vx_tensor)parameters[1], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST);
    if(status)
    {
        std::cerr << "ERROR: vxMapTensorPatch() failed for input#2(" << status << ")" << std::endl;
        return -1;
    }

    memcpy(confData, ptr, (count_tensor_conf*sizeof(float)));

    status = vxUnmapTensorPatch((vx_tensor)parameters[1], map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for input#2(" << status << ")" << std::endl;
        return -1;
    }

    status = vxMapTensorPatch((vx_tensor)parameters[2], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST);
    if(status)
    {
        std::cerr << "ERROR: vxMapTensorPatch() failed for input#3(" << status << ")" << std::endl;
        return -1;
    }

    memcpy(priorData, ptr, (count_tensor_prior*sizeof(float)));

    status = vxUnmapTensorPatch((vx_tensor)parameters[2], map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for input#3(" << status << ")" << std::endl;
    }
    
    // Retrieve all location predictions.
    vector<LabelBBox> allLocPreds;
    GetLocPredictions(locData, num_batches, numPriors, num_loc_classes,
                    share_location, &allLocPreds);

    // Retrieve all confidences.
    vector<map<int, vector<float> > > allConfidenceScores;
    GetConfidenceScores(confData, num_batches, numPriors, num_classes, &allConfidenceScores);

    // Retrieve all prior bboxes. It is same within a batch since we assume all
    // images in a batch are of same dimension.
    vector<NormalizedBBox> priorBBoxes;
    vector<vector<float> > priorVariances;
    GetPriorBBoxes(priorData, numPriors, &priorBBoxes, &priorVariances);

    std::vector<LabelBBox> allDecodedBBoxes;
    const bool clip_bbox = false;
    DecodeBBoxesAll(allLocPreds, priorBBoxes, priorVariances, num_batches,
                            share_location, num_loc_classes, background_label_id,
                            s_code_type, variance_encoded_in_target, clip_bbox, &allDecodedBBoxes);

    int numKept = 0;
    std::vector<std::map<int, std::vector<int> > > allIndices;
    for (int i = 0; i < num_batches; i++)
    {
        const LabelBBox &decode_bboxes = allDecodedBBoxes[i];
        std::map<int, std::vector<float> > &confidenceScores = allConfidenceScores[i];
        std::map<int, std::vector<int> > indices;
        int num_det = 0;
        for(int c = 0; c < num_classes; c++)
        {
            if(c == background_label_id)
                continue;
            if(confidenceScores.find(c) == confidenceScores.end())
                ERRMSG(VX_ERROR_INVALID_VALUE, "processDetectionOutput: could not find confidence predictions for label %d\n", c);

            const vector<float>& scores = confidenceScores.find(c)->second;
            int label = share_location ? -1 : c;
            if (decode_bboxes.find(label) == decode_bboxes.end()) {
                ERRMSG(VX_ERROR_INVALID_VALUE, "processDetectionOutput: could not find location predictions for label %d\n", label);
                continue;
            }
            const vector<NormalizedBBox> &bboxes = decode_bboxes.find(label)->second;
            ApplyNMSFast(bboxes, scores, confidence_threshold, nms_threshold, top_k, &(indices[c]), eta);

            num_det += indices[c].size();
        }
      
        if (keep_top_k > -1 && num_det > keep_top_k)
        {
            std::vector<std::pair<float, std::pair<int, int> > > scoreIndexPairs;
            for (std::map<int, std::vector<int> >::iterator it = indices.begin(); it != indices.end(); ++it)
            {
                int label = it->first;
                const std::vector<int>& labelIndices = it->second;
                if (confidenceScores.find(label) == confidenceScores.end())
                {
                    ERRMSG(VX_ERROR_INVALID_VALUE, "processDetectionOutput: could not find location predictions for label %d", label);
                    continue;
                }   
                const std::vector<float>& scores = confidenceScores.find(label)->second;
                for (size_t j = 0; j < labelIndices.size(); ++j)
                {
                    size_t idx = labelIndices[j];
                    assert(idx < scores.size());
                    scoreIndexPairs.push_back(std::make_pair(scores[idx], std::make_pair(label, idx)));
                }
            }
            // Keep top k results per image.
            std::sort(scoreIndexPairs.begin(), scoreIndexPairs.end(), SortScorePairDescend<pair<int, int> >);
            scoreIndexPairs.resize(keep_top_k);
            // Store the new indices.
            map<int, vector<int> > newIndices;
            for (int j = 0; j < scoreIndexPairs.size(); ++j)
            {
                int label = scoreIndexPairs[j].second.first;
                int idx = scoreIndexPairs[j].second.second;
                newIndices[label].push_back(idx);
            }
            allIndices.push_back(newIndices);
            numKept += keep_top_k;
        } 
        else
        {
            allIndices.push_back(indices);
            numKept += num_det;
        }
    }

    output_dims[3] = 1;
    output_dims[2] = 1;
    output_dims[1] = numKept;
    output_dims[0] = 7;
    int count_output_final = output_dims[0]*output_dims[1]*output_dims[2]*output_dims[3];
    vx_size stride_output_final[4] = {sizeof(float), output_dims[0]*sizeof(float), output_dims[0]*output_dims[1]*sizeof(float), output_dims[0]*output_dims[1]*output_dims[2]*sizeof(float) }; 
    float * outputData = new float[count_output_final];
    if(numKept == 0)
    {
        output_dims[2] = num_batches;
        for(int i = 0; i < num_batches; i++)
        {
            outputData[0] = i;
            outputData += 7;
        }
        return VX_SUCCESS;
    }

    int count = 0;
    for (int i = 0; i < num_batches; i++)
    {
        const map<int, vector<float> >& confScores = allConfidenceScores[i];
        const LabelBBox& decodeBBoxes = allDecodedBBoxes[i];

        for (map<int, vector<int> >::iterator it = allIndices[i].begin(); it != allIndices[i].end(); ++it)
        {
            int label = it->first;
            if (confScores.find(label) == confScores.end()) 
            {
                ERRMSG(VX_ERROR_INVALID_VALUE,"Could not find confidence predictions for %d\n",label);
                continue;
            }

            const vector<float>& scores = confScores.find(label)->second;
            int locLabel = share_location ? -1 : label;
            if (decodeBBoxes.find(locLabel) == decodeBBoxes.end())
            {
                ERRMSG(VX_ERROR_INVALID_VALUE,"Could not find location predictions for %d\n",locLabel);
                continue;
            }

            const vector<NormalizedBBox>& bboxes = decodeBBoxes.find(locLabel)->second;
            vector<int>& indices = it->second;

            for (int j = 0; j < indices.size(); ++j) 
            {
                int idx = indices[j];
                outputData[count * 7] = i;
                outputData[count * 7 + 1] = label;
                outputData[count * 7 + 2] = scores[idx];
                const NormalizedBBox& bbox = bboxes[idx];
                outputData[count * 7 + 3] = bbox.xmin;
                outputData[count * 7 + 4] = bbox.ymin;
                outputData[count * 7 + 5] = bbox.xmax;
                outputData[count * 7 + 6] = bbox.ymax;
                ++count;
            }
        }           
    }
    assert(count == numKept);

    //printf("output = %f %f %f %f %f %f %f\n", outputData[0],outputData[1],outputData[2],outputData[3],outputData[4], outputData[5],outputData[6]);
    status =  vxCopyTensorPatch((vx_tensor)parameters[10], 4, nullptr, nullptr, stride_output_final, outputData, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status)
    {
        std::cerr << "ERROR: vxCopyTensorPatch() failed for output tensor"  << std::endl;
        return -1;
    }
    delete [] locData;
    delete [] confData;
    delete [] priorData;
  
    /*DUMP LAYER BUFFER*/
    #if ENABLE_DEBUG_DUMP_NN_LAYER_BUFFERS
        //dump the output layer
        nn_layer_test_dumpBuffer("detection_output_%04d.bin", (vx_tensor)parameters[10]);
    #endif 
  
    return VX_SUCCESS;

}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
    vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
    vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
    )
{
    supported_target_affinity = AGO_TARGET_AFFINITY_CPU;
    return VX_SUCCESS;
}


//! \brief The kernel publisher.
vx_status publishDetectionOutputLayer(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.detection_output", VX_KERNEL_DETECTION_OUTPUT_LAYER_AMD, processDetectionOutput, 14, validate, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
    
    //set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 9, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 10, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 11, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 12, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 13, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));

    //finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    return VX_SUCCESS;
}


VX_API_ENTRY vx_node VX_API_CALL vxDetectionOutputLayer(vx_graph graph, vx_tensor input1, vx_tensor input2, vx_tensor input3, vx_int32 num_classes, vx_int32 share_location, vx_int32 background_label_id, 
                                                        vx_float32 nms_threshold, vx_int32 code_type, vx_int32 keep_top_k, vx_int32 variance_encoded_in_target, vx_tensor output)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_num_classes = vxCreateScalarWithSize(context, VX_TYPE_INT32, &num_classes, sizeof(num_classes));
        vx_scalar s_share_location = vxCreateScalarWithSize(context, VX_TYPE_INT32, &share_location, sizeof(share_location));
        vx_scalar s_background_label_id = vxCreateScalarWithSize(context, VX_TYPE_INT32, &background_label_id, sizeof(background_label_id));
        vx_scalar s_nms_threshold = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &nms_threshold, sizeof(nms_threshold));
        vx_scalar s_code_type = vxCreateScalarWithSize(context, VX_TYPE_INT32, &code_type, sizeof(code_type));
        vx_scalar s_keep_top_k = vxCreateScalarWithSize(context, VX_TYPE_INT32, &keep_top_k, sizeof(keep_top_k));
        vx_scalar s_variance_encoded_in_target = vxCreateScalarWithSize(context, VX_TYPE_INT32, &variance_encoded_in_target, sizeof(variance_encoded_in_target));

        vx_reference params[] = {
            (vx_reference)input1,
            (vx_reference)input2,
            (vx_reference)input3,
            (vx_reference)s_num_classes,
            (vx_reference)s_share_location,
            (vx_reference)s_background_label_id,
            (vx_reference)s_nms_threshold,
            (vx_reference)s_code_type,
            (vx_reference)s_keep_top_k,
            (vx_reference)s_variance_encoded_in_target,
            (vx_reference)output,
        };
        node = createNode(graph, VX_KERNEL_DETECTION_OUTPUT_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}
