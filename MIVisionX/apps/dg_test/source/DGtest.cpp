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

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vx_ext_amd.h>
#include "DGtest.h"

using namespace cv;
using namespace std;

#if USE_OPENCV_4
#define CV_BGR2GRAY COLOR_BGR2GRAY
#endif

#define ERROR_CHECK_OBJECT(obj) { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS) { vxAddLogEntry((vx_reference)context, status     , "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; } }
#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS) { printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); exit(-1); } }

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    size_t len = strlen(string);
    if (len > 0) {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}

DGtest::DGtest(const char* weights) {
    // create contex
    mContext = vxCreateContext();
    vx_status status;
    status = vxGetStatus((vx_reference)mContext);
    if(status) {
        printf("ERROR: vxCreateContext() failed\n");
        exit(-1);
    }
    vxRegisterLogCallback(mContext, log_callback, vx_false_e);

    // create graph
    mGraph = vxCreateGraph(mContext);
    status = vxGetStatus((vx_reference)mGraph);
    if(status) {
        printf("ERROR: vxCreateGraph(...) failed (%d)\n", status);
        exit(-1);
    }

    // create and initialize input tensor data
    vx_size input_dims[4] = { 28, 28, 1, 1 };
    mInputTensor = vxCreateTensor(mContext, 4, input_dims, VX_TYPE_FLOAT32, 0);
    if(vxGetStatus((vx_reference)mInputTensor)) {
        printf("ERROR: vxCreateTensor() failed\n");
        exit(-1);
    }
    
    // create and initialize output tensor data
    vx_size output_dims[4] = { 1, 1, 10, 1 };
    mOutputTensor = vxCreateTensor(mContext, 4, output_dims, VX_TYPE_FLOAT32, 0);
    if(vxGetStatus((vx_reference)mOutputTensor)) {
        printf("ERROR: vxCreateTensor() failed for mOutputTensor\n");
        exit(-1);
    }

    //add input tensor, output tensor, and weights to the graph
    status = annAddToGraph(mGraph, mInputTensor, mOutputTensor, weights);
    if(status) {
         printf("ERROR: annAddToGraph() failed (%d)\n", status);
         exit(-1);
    }
    
    //verify the graph
    status = vxVerifyGraph(mGraph);
    if(status) {
        printf("ERROR: vxVerifyGraph(...) failed (%d)\n", status);
        exit(-1);
    }
};

DGtest::~DGtest(){
    //release the tensors
    ERROR_CHECK_STATUS(vxReleaseTensor(&mInputTensor));
    ERROR_CHECK_STATUS(vxReleaseTensor(&mOutputTensor));
    //release the graph
    ERROR_CHECK_STATUS(vxReleaseGraph(&mGraph));
    // release context
    ERROR_CHECK_STATUS(vxReleaseContext(&mContext));
};

int DGtest::runInference(Mat &image) {
    
    Mat img = image.clone();
    
    // convert to grayscale image
    cvtColor(img, img, CV_BGR2GRAY);

    // resize to 24 x 24 
	resize(img, img, Size(24, 24));
    
    // dilate image
	dilate(img, img, Mat::ones(2,2,CV_8U)); 
    
    // add border to the image so that the digit will go center and become 28 x 28 image
    copyMakeBorder(img, img, 2, 2, 2, 2, BORDER_CONSTANT, Scalar(0,0,0));
		
    vx_size dims[4] = { 1, 1, 1, 1 }, stride[4];
    vx_status status;
    vx_map_id map_id;
    float * ptr;

    // query tensor for the dimension    
    vxQueryTensor(mInputTensor, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*4);
 
    // copy image to input tensor
    status = vxMapTensorPatch(mInputTensor, 4, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for " <<  std::endl;
        return -1;
    }
    
    for(vx_size y = 0; y < dims[1]; y++) {
        unsigned char * src = img.data + y*dims[0]*dims[2];
        float * dst = ptr + ((y * stride[1]) >> 2);
        for(vx_size x = 0; x < dims[0]; x++, src++) {
            *dst++ = src[0];
        }
    }

    status = vxUnmapTensorPatch(mInputTensor, map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for " <<  std::endl;
        return -1;
    }

    //process the graph
    status = vxProcessGraph(mGraph);
    if(status != VX_SUCCESS) {
        std::cerr << "ERROR: vxProcessGraph() failed" <<  std::endl;
        return -1;
    }

    // get the output result from output tensor
    status = vxMapTensorPatch(mOutputTensor, 4, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for "  << std::endl;
        return -1;
    }

    mDigit = std::distance(ptr, std::max_element(ptr, ptr + 10));

    status = vxUnmapTensorPatch(mOutputTensor, map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for "  << std::endl;
        return -1;
    }
    return 0;
}

int DGtest::getResult() {
    return mDigit;
}
