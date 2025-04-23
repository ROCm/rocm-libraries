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

typedef struct normalizedBox
{
    float y1; //y_center for center_type = 1
    float x1; //x_center
    float y2; //height
    float x2; //width
} bboxes;



static vx_status VX_CALLBACK validate(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[])
{
    // check tensor dims.
    vx_enum type;
    vx_size num_dims;
    vx_size input_dims_1[4], input_dims_2[4], output_dims[4];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4 /*&& num_dims != 3*/) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: NMS: #1 input tensor data type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims_1, sizeof(input_dims_1)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4 /*&& num_dims != 3*/) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: NMS: #2 input tensor data type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims_2, sizeof(input_dims_2)));

    vx_int32 center_point_box;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &center_point_box, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(center_point_box != 0 && center_point_box != 1) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: NMS: #3 scalar type=%d ('center_point_box' must be between 0/1)\n", center_point_box);

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4 /*&& num_dims != 3*/) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_INT64) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: NMS: #4 output tensor data type=%d (must be int64)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    // output tensor configuration
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    if(parameters[4])
    {
        vx_size max_output_boxes_per_class_dims[1];
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
        if (num_dims != 1) return VX_ERROR_INVALID_DIMENSION;
        if (type != VX_TYPE_INT64) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: NMS: #5 input tensor data type=%d (must be int64)\n", type);
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, max_output_boxes_per_class_dims, sizeof(max_output_boxes_per_class_dims)));
    }

    if(parameters[5])
    {
        vx_size iou_threshold_dims[1];
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
        if (num_dims != 1) return VX_ERROR_INVALID_DIMENSION;
        if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: NMS: #6 input tensor data type=%d (must be float)\n", type);
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DIMS, iou_threshold_dims, sizeof(iou_threshold_dims)));
    }

    if(parameters[6])
    {
        vx_size score_threshold_dims[1];
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
        if (num_dims != 1) return VX_ERROR_INVALID_DIMENSION;
        if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: NMS: #7 input tensor data type=%d (must be float)\n", type);
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DIMS, score_threshold_dims, sizeof(score_threshold_dims)));
    }
    return VX_SUCCESS;
}

template <typename T>
static inline bool sortScorePairDescend(const std::pair<float, T>& pair1, const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

void getMaxScoreIndex(const std::vector<float>& scores, const float score_thresh, const int max_output_boxes_per_class,
                      std::vector<std::pair<float, int>>* score_index_vec)
{
    if(!score_index_vec->empty())
        score_index_vec->clear();

    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > score_thresh)
        {
            score_index_vec->push_back(std::make_pair(scores[i], i));
        }

    }

    std::stable_sort(score_index_vec->begin(), score_index_vec->end(), sortScorePairDescend<int>);
}

float computeOverlapCoordinates(bboxes& box1, bboxes &box2)
{
    float area1, area2, area12;
    float top, bottom, left, right;
    float ymin_1, xmin_1, ymax_1, xmax_1, ymin_2, xmin_2, ymax_2, xmax_2;

    ymin_1 = std::min(box1.y1, box1.y2);
    xmin_1 = std::min(box1.x1, box1.x2);
    ymax_1 = std::max(box1.y1, box1.y2);
    xmax_1 = std::max(box1.x1, box1.x2);

    ymin_2 = std::min(box2.y1, box2.y2);
    xmin_2 = std::min(box2.x1, box2.x2);
    ymax_2 = std::max(box2.y1, box2.y2);
    xmax_2 = std::max(box2.x1, box2.x2);

    area1 = (ymax_1 - ymin_1)*(xmax_1 - xmin_1);
    area2 = (ymax_2 - ymin_2)*(xmax_2 - xmin_2);
    if (area1 <= 0 || area2 <= 0)
        return 0.0;

    top = std::max(ymin_1, ymin_2);
    bottom = std::max(ymax_1, ymax_2);
    left = std::max(xmin_1, xmin_2);
    right = std::max(xmax_1, xmax_2);

    area12 = std::max((bottom-top), (float)0)*std::max((right-left), (float)0);
    if(bottom < top || left > right)
        return 0;
    if((std::min(xmax_1,xmax_2) < left) || (std::min(ymax_1,ymax_2) < top))
        return 0;
    
    return (area12/(area1+area2-area12));
}

float computeOverlapCenter(bboxes& box1, bboxes &box2)
{
    //look at struct definition for variable names
    float area1, area2, area12;
    float top, bottom, right, left, r11, r12, c11, c12, r21, r22, c21, c22;

    c11 = box1.x1 - (box1.y2/2);
    c12 = box1.x1 + (box1.y2/2);
    r11 = box1.y1 - (box1.x2/2);
    r12 = box1.y1 + (box1.x2/2);
        
    c21 = box2.x1 - (box2.y2/2);
    c22 = box2.x1 + (box2.y2/2);
    r21 = box2.y1 - (box2.x2/2);
    r22 = box2.y1 + (box2.x2/2);
    
    top = std::max(r11, r21);
    bottom = std::max(r12, r22);
    left = std::max(c11, c21);
    right = std::max(c12, c22);

    if(bottom < top || left > right)
        return 0;
    if((std::min(c12, c22) < left) || (std::min(r12, r22) < top))
        return 0;
    area1 = box1.x2*box1.y2;
    area2 = box2.x2*box2.y2;
    area12 = (bottom-top)*(right-left);
    return (area12/(area1+area2-area12));

}

static vx_status VX_CALLBACK processNMSLayer(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    //get tensor dimensions
    vx_size input_dims_0[4], input_dims_1[4], output_dims[4];
    vx_size num_of_dims;   
    vx_enum type;

    //get memory pointers for all inputs
    vx_map_id map_id;
    vx_size stride[4];
    float * ptr;
    vx_enum usage = VX_READ_ONLY;
    vx_status status;

    //query and copy boxes(tensor) 
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims_0, sizeof(input_dims_0)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims_1, sizeof(input_dims_1)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));

    //make sure input dimensions match for boxes ans scores
    if (input_dims_0[0] != input_dims_1[0])
    {
        printf("processNMSLayer: nms_layer: num_batches for scores(%lu) must match num_batches for boxes(%lu)\n", input_dims_0[1], input_dims_0[0]);  
        exit(0);
    }

    if (input_dims_0[2] != input_dims_1[1])
    {
        printf("processNMSLayer: nms_layer: spatial_dimension for scores(%lu) must match spatial_dimension for boxes(%lu)\n", input_dims_1[2], input_dims_0[1]);  
        exit(0);
    }

    int num_batches = input_dims_1[3];
    int num_classes = input_dims_1[2];
    const int spatial_dimension = input_dims_1[1];

    std::vector<bboxes> boxes(spatial_dimension);
    std::vector<std::vector<std::vector<float>>> scores(num_batches,std::vector<std::vector<float>>(num_classes, std::vector<float>(spatial_dimension)));
    
    //map openvx boxes tensor to vector
    status = vxMapTensorPatch((vx_tensor)parameters[0], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST);
    if(status)
    {
        std::cerr << "ERROR: vxMapTensorPatch() failed for input#1 (" << status << ")" << std::endl;
        return -1;
    }
    //memcpy(bboxes, ptr, (count_tensor_bboxes*sizeof(float)));
    for (size_t b = 0; b < num_batches; ++b)
    {
        int idx = b * spatial_dimension * 4;
        for(size_t i = 0; i < spatial_dimension; ++i)
        {
            boxes[i].y1 = ptr[idx + i*4];
            boxes[i].x1 = ptr[idx + i*4 + 1];
            boxes[i].y2 = ptr[idx + i*4 + 2];
            boxes[i].x2 = ptr[idx + i*4 + 3];
        }
    }
    status = vxUnmapTensorPatch((vx_tensor)parameters[0], map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for input#5 (" << status << ")" << std::endl;
        return -1;
    }

    //map openvx scores tensors to vector
    status = vxMapTensorPatch((vx_tensor)parameters[1], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST);
    if(status)
    {
        std::cerr << "ERROR: vxMapTensorPatch() failed for input#1 (" << status << ")" << std::endl;
        return -1;
    }

    //memcpy(scores, ptr, (count_tensor_scores*sizeof(float)));
    for (size_t b = 0; b < num_batches; ++b)
    {
        for (size_t c = 0; c < num_classes; ++c)
        {
            int idx = b * num_classes * spatial_dimension + c * spatial_dimension;
            for(size_t i = 0; i < spatial_dimension; ++i)
            {
                scores[b][c][i] = ptr[idx + i];
            }
        } 
    }

    status = vxUnmapTensorPatch((vx_tensor)parameters[1], map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for input#2 (" << status << ")" << std::endl;
        return -1;
    }

    vx_int32 center_point_box;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &center_point_box, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));

    int64_t *max_output_boxes_per_class = new int64_t[1];
    if(parameters[4])
    {
        vx_size max_output_dims[1];
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, max_output_dims, sizeof(max_output_dims)));
    
        //get memory pointers for all inputs
        vx_map_id map_id;
        vx_size stride[1];
        int * ptr;
        vx_enum usage = VX_READ_ONLY;
        vx_status status;
        status = vxMapTensorPatch((vx_tensor)parameters[4], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST);
        if(status)
        {
            std::cerr << "ERROR: vxMapTensorPatch() failed for input#5 (" << status << ")" << std::endl;
            return -1;
        }
    
        memcpy(max_output_boxes_per_class, ptr, (max_output_dims[0]*sizeof(int64_t)));
    
        status = vxUnmapTensorPatch((vx_tensor)parameters[4], map_id);
        if(status) {
            std::cerr << "ERROR: vxUnmapTensorPatch() failed for input#5 (" << status << ")" << std::endl;
            return -1;
        }
    }
    else
    {
        printf("processNMSLayer: nms_layer: returning no ouput since max_output_boxes_per_class = 0\n");
        return VX_SUCCESS;
    }

    float *iou_thresh = new float[1];
    if(parameters[5])
    {
        vx_size iou_thresh_dims[1];
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DIMS, iou_thresh_dims, sizeof(iou_thresh_dims)));
    
        //get memory pointers for all inputs
        vx_map_id map_id;
        vx_size stride[1];
        float * ptr;
        vx_enum usage = VX_READ_ONLY;
        vx_status status;
        status = vxMapTensorPatch((vx_tensor)parameters[5], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST);
        if(status)
        {
            std::cerr << "ERROR: vxMapTensorPatch() failed for input#5 (" << status << ")" << std::endl;
            return -1;
        }
    
        memcpy(iou_thresh, ptr, (iou_thresh_dims[0]*sizeof(float)));
    
        status = vxUnmapTensorPatch((vx_tensor)parameters[5], map_id);
        if(status) {
            std::cerr << "ERROR: vxUnmapTensorPatch() failed for input#5 (" << status << ")" << std::endl;
            return -1;
        }
    }
    else
    {
        iou_thresh[0] = 0.0;
    }
    
    float *score_thresh = new float[1];
    if(parameters[6])
    {
        vx_size score_thresh_dims[1];
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DIMS, score_thresh_dims, sizeof(score_thresh_dims)));
    
        //get memory pointers for all inputs
        vx_map_id map_id;
        vx_size stride[1];
        float * ptr;
        vx_enum usage = VX_READ_ONLY;
        vx_status status;
        status = vxMapTensorPatch((vx_tensor)parameters[6], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST);
        if(status)
        {
            std::cerr << "ERROR: vxMapTensorPatch() failed for input#5 (" << status << ")" << std::endl;
            return -1;
        }
    
        memcpy(score_thresh, ptr, (score_thresh_dims[0]*sizeof(float)));
    
        status = vxUnmapTensorPatch((vx_tensor)parameters[6], map_id);
        if(status) {
            std::cerr << "ERROR: vxUnmapTensorPatch() failed for input#5 (" << status << ")" << std::endl;
            return -1;
        }
    }
    else
    {
        score_thresh[0] = 0.0;
    }

    std::vector<int64_t> final_selected_indices;
    //get top_k scores with indices per batch per class. Common for both center point types.
    for (int b = 0; b < num_batches; ++b)
    {
        for (int c = 0; c < num_classes; ++c)
        {
            std::vector<std::pair<float, int>> score_index_vec;
            getMaxScoreIndex(scores[b][c], score_thresh[0], max_output_boxes_per_class[0], &score_index_vec);

            std::vector<int> selected_indices;
            selected_indices.clear();

            for(int i = 0; i < score_index_vec.size(); ++i)
            {
                const int idx = score_index_vec[i].second;
                bool keep = true;
                for(int k = 0; k < (int)selected_indices.size() && keep; ++k)
                {
                    const int prev_idx = selected_indices[k];
                    float overlap = 0.0;
                    if (center_point_box == 0) /*indicates box data = [y1,x1,y2,x2] - mostly TF models */
                    {
                        overlap = computeOverlapCoordinates(boxes[idx], boxes[prev_idx]);
                    }
                    else if(center_point_box == 1) /*indicates box data = [x_center,y_center,width,height] - mostly PyTorch models*/
                    {
                        overlap = computeOverlapCenter(boxes[idx], boxes[prev_idx]);
                    }
                    if(overlap <= iou_thresh[0])
                        keep = true;
                    else
                        keep = false;
                    //keep = overlap <= iou_thresh[0];
                }
                if(keep)
                    selected_indices.push_back(idx);
            }
            if(max_output_boxes_per_class[0] < selected_indices.size())
                selected_indices.resize(max_output_boxes_per_class[0]);

            for(int f = 0; f < selected_indices.size(); f++)
            {
                final_selected_indices.push_back((int64_t)b);
                final_selected_indices.push_back((int64_t)c);
                final_selected_indices.push_back((int64_t)selected_indices[f]);
            }
        }
    }    

    // ptr for copying back to tensor
    int64_t *final_selected_indices_ptr = &final_selected_indices[0];
        
    //finding size of nms output and assigning stride
    output_dims[3] = 1; 
    output_dims[2] = 1;
    output_dims[1] = final_selected_indices.size(); //number of boxes found;
    output_dims[0] = 3; //3 values per index

    vx_size stride_output_final[4] = {sizeof(int64_t), output_dims[0]*sizeof(int64_t), output_dims[0]*output_dims[1]*sizeof(int64_t), output_dims[0]*output_dims[1]*output_dims[2]*sizeof(int64_t) };
    status =  vxCopyTensorPatch((vx_tensor)parameters[3], 4, nullptr, nullptr, stride_output_final, final_selected_indices_ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status)
    {
        std::cerr << "ERROR: vxCopyTensorPatch() failed for output tensor"  << std::endl;
        return -1;
    }

    delete [] iou_thresh;
    delete [] score_thresh;
    delete [] max_output_boxes_per_class;

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
vx_status publishNMSLayer(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.nms_layer", VX_KERNEL_NMS_LAYER_AMD, processNMSLayer, 7, validate, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));

    //set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));    
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));

    //finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxNMSLayer(vx_graph graph, vx_tensor boxes, vx_tensor scores, vx_int32 center_point_box, vx_tensor output, vx_tensor max_output_boxes_per_class,
                                             vx_tensor iou_threshold, vx_tensor score_threshold)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
    	vx_scalar s_center_point_box = vxCreateScalarWithSize(context, VX_TYPE_INT32, &center_point_box, sizeof(center_point_box));
        vx_reference params[] = {
            (vx_reference)boxes,
            (vx_reference)scores,
            (vx_reference)s_center_point_box,
            (vx_reference)output,
        };
        node = createNode(graph, VX_KERNEL_NMS_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}
