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

#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include "opencv2/opencv.hpp"
#include "vx_ext_pop.h"
#include <string>

using namespace cv;
using namespace std;

#if USE_OPENCV_4
#define CV_BGR2RGB COLOR_BGR2RGB
#endif

#define ERROR_CHECK_STATUS(status)                                                              \
    {                                                                                           \
        vx_status status_ = (status);                                                           \
        if (status_ != VX_SUCCESS)                                                              \
        {                                                                                       \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1);                                                                            \
        }                                                                                       \
    }

#define ERROR_CHECK_OBJECT(obj)                                                                 \
    {                                                                                           \
        vx_status status_ = vxGetStatus((vx_reference)(obj));                                   \
        if (status_ != VX_SUCCESS)                                                              \
        {                                                                                       \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1);                                                                            \
        }                                                                                       \
    }

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    size_t len = strlen(string);
    if (len > 0)
    {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage:\n"
               "./vxPop --donut\n"
               "./vxPop --bubble \n");
        return 0;
    }

    // check option
    string optionCheck = argv[1];
    if (optionCheck != "--bubble" && optionCheck != "--donut")
    {
        printf("Usage:\n"
               "./vxPop --donut\n"
               "./vxPop --bubble \n");
        return 0;
    }

    int width = 720, height = 480;

    // create OpenVX Context
    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);
    vxRegisterLogCallback(context, log_callback, vx_false_e);

    // load vx_pop kernels
    ERROR_CHECK_STATUS(vxLoadKernels(context, "vx_pop"));

    // create OpenVX Graph
    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph);

    // create OpenVX Images
    vx_image input_rgb_image = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
    vx_image output_pop_image = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(input_rgb_image);
    ERROR_CHECK_OBJECT(output_pop_image);

    // create intermediate images
    vx_image yuv_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_IYUV);
    vx_image luma_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image output_canny_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image output_skinTone_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image output_canny_skinTone_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(yuv_image);
    ERROR_CHECK_OBJECT(luma_image);
    ERROR_CHECK_OBJECT(output_canny_image);
    ERROR_CHECK_OBJECT(output_skinTone_image);
    ERROR_CHECK_OBJECT(output_canny_skinTone_image);

    // create threshold variable
    vx_threshold hyst = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    vx_int32 lower = 80, upper = 100;
    vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, &lower, sizeof(vx_int32));
    vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, &upper, sizeof(vx_int32));
    ERROR_CHECK_OBJECT(hyst);
    vx_int32 gradient_size = 3;

    // create intermediate images which are not accessed by the user to be mem optimized
    vx_image R_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image G_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image B_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image RmG_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image RmB_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image R95_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image G40_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image B20_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image RmG15_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image RmB0_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image and1_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image and2_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image and3_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(R_image);
    ERROR_CHECK_OBJECT(G_image);
    ERROR_CHECK_OBJECT(B_image);
    ERROR_CHECK_OBJECT(RmG_image);
    ERROR_CHECK_OBJECT(RmB_image);
    ERROR_CHECK_OBJECT(R95_image);
    ERROR_CHECK_OBJECT(G40_image);
    ERROR_CHECK_OBJECT(B20_image);
    ERROR_CHECK_OBJECT(RmG15_image);
    ERROR_CHECK_OBJECT(RmB0_image);
    ERROR_CHECK_OBJECT(and1_image);
    ERROR_CHECK_OBJECT(and2_image);
    ERROR_CHECK_OBJECT(and3_image);

    // create threshold values
    vx_threshold thresh95 = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    vx_int32 thresValue95 = 95;
    vxSetThresholdAttribute(thresh95, VX_THRESHOLD_THRESHOLD_VALUE, &thresValue95, sizeof(vx_int32));
    ERROR_CHECK_OBJECT(thresh95);
    vx_threshold thresh40 = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    vx_int32 thresValue40 = 40;
    vxSetThresholdAttribute(thresh40, VX_THRESHOLD_THRESHOLD_VALUE, &thresValue40, sizeof(vx_int32));
    ERROR_CHECK_OBJECT(thresh40);
    vx_threshold thresh20 = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    vx_int32 thresValue20 = 20;
    vxSetThresholdAttribute(thresh20, VX_THRESHOLD_THRESHOLD_VALUE, &thresValue20, sizeof(vx_int32));
    ERROR_CHECK_OBJECT(thresh20);
    vx_threshold thresh15 = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    vx_int32 thresValue15 = 15;
    vxSetThresholdAttribute(thresh15, VX_THRESHOLD_THRESHOLD_VALUE, &thresValue15, sizeof(vx_int32));
    ERROR_CHECK_OBJECT(thresh15);
    vx_threshold thresh0 = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    vx_int32 thresValue0 = 0;
    vxSetThresholdAttribute(thresh0, VX_THRESHOLD_THRESHOLD_VALUE, &thresValue0, sizeof(vx_int32));
    ERROR_CHECK_OBJECT(thresh0);

    // add nodes to the graph
    string option = argv[1];
    if (option == "--bubble")
    {
        vx_node nodes[] =
            {
                // extract R,G,B channels and compute R-G and R-B
                vxChannelExtractNode(graph, input_rgb_image, VX_CHANNEL_R, R_image),
                vxChannelExtractNode(graph, input_rgb_image, VX_CHANNEL_G, G_image),
                vxChannelExtractNode(graph, input_rgb_image, VX_CHANNEL_B, B_image),
                vxSubtractNode(graph, R_image, G_image, VX_CONVERT_POLICY_SATURATE, RmG_image),
                vxSubtractNode(graph, R_image, B_image, VX_CONVERT_POLICY_SATURATE, RmB_image),
                // compute threshold
                vxThresholdNode(graph, R_image, thresh95, R95_image),
                vxThresholdNode(graph, G_image, thresh40, G40_image),
                vxThresholdNode(graph, B_image, thresh20, B20_image),
                vxThresholdNode(graph, RmG_image, thresh15, RmG15_image),
                vxThresholdNode(graph, RmB_image, thresh0, RmB0_image),
                // aggregate all thresholded values to produce SKIN pixels
                vxAndNode(graph, R95_image, G40_image, and1_image),
                vxAndNode(graph, and1_image, B20_image, and2_image),
                vxAndNode(graph, RmG15_image, RmB0_image, and3_image),
                vxAndNode(graph, and2_image, and3_image, output_skinTone_image),
                // create canny edge
                vxColorConvertNode(graph, input_rgb_image, yuv_image),
                vxChannelExtractNode(graph, yuv_image, VX_CHANNEL_Y, luma_image),
                vxCannyEdgeDetectorNode(graph, luma_image, hyst, gradient_size, VX_NORM_L1, output_canny_image),
                // or - canny & skintone images
                vxOrNode(graph, output_canny_image, output_skinTone_image, output_canny_skinTone_image),
                // vx pop - bubble pop
                vxExtPopNode_bubblePop(graph, output_canny_skinTone_image, output_pop_image)};
        for (vx_size i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
        {
            ERROR_CHECK_OBJECT(nodes[i]);
            ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
        }
    }
    else
    {
        vx_node nodes[] =
            {
                // extract R,G,B channels and compute R-G and R-B
                vxChannelExtractNode(graph, input_rgb_image, VX_CHANNEL_R, R_image),
                vxChannelExtractNode(graph, input_rgb_image, VX_CHANNEL_G, G_image),
                vxChannelExtractNode(graph, input_rgb_image, VX_CHANNEL_B, B_image),
                vxSubtractNode(graph, R_image, G_image, VX_CONVERT_POLICY_SATURATE, RmG_image),
                vxSubtractNode(graph, R_image, B_image, VX_CONVERT_POLICY_SATURATE, RmB_image),
                // compute threshold
                vxThresholdNode(graph, R_image, thresh95, R95_image),
                vxThresholdNode(graph, G_image, thresh40, G40_image),
                vxThresholdNode(graph, B_image, thresh20, B20_image),
                vxThresholdNode(graph, RmG_image, thresh15, RmG15_image),
                vxThresholdNode(graph, RmB_image, thresh0, RmB0_image),
                // aggregate all thresholded values to produce SKIN pixels
                vxAndNode(graph, R95_image, G40_image, and1_image),
                vxAndNode(graph, and1_image, B20_image, and2_image),
                vxAndNode(graph, RmG15_image, RmB0_image, and3_image),
                vxAndNode(graph, and2_image, and3_image, output_skinTone_image),
                // create canny edge
                vxColorConvertNode(graph, input_rgb_image, yuv_image),
                vxChannelExtractNode(graph, yuv_image, VX_CHANNEL_Y, luma_image),
                vxCannyEdgeDetectorNode(graph, luma_image, hyst, gradient_size, VX_NORM_L1, output_canny_image),
                // or - canny & skintone images
                vxOrNode(graph, output_canny_image, output_skinTone_image, output_canny_skinTone_image),
                // vx pop - donut pop
                vxExtPopNode_donutPop(graph, output_canny_skinTone_image, output_pop_image)};
        for (vx_size i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
        {
            ERROR_CHECK_OBJECT(nodes[i]);
            ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
        }
    }

    // verify graph - only once
    ERROR_CHECK_STATUS(vxVerifyGraph(graph));

    Mat input, input_rgb;
    cv::namedWindow("VX POP - LIVE", cv::WINDOW_GUI_EXPANDED);
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        printf("Unable to open camera\n");
        return 0;
    }
    for (;;)
    {
        cap >> input;
        resize(input, input, Size(width, height));
        cvtColor(input, input_rgb, CV_BGR2RGB);
        if (waitKey(30) >= 0)
            break;
        vx_rectangle_t cv_rgb_image_region;
        cv_rgb_image_region.start_x = 0;
        cv_rgb_image_region.start_y = 0;
        cv_rgb_image_region.end_x = width;
        cv_rgb_image_region.end_y = height;
        vx_imagepatch_addressing_t cv_rgb_image_layout;
        cv_rgb_image_layout.stride_x = 3;
        cv_rgb_image_layout.stride_y = input_rgb.step;
        vx_uint8 *cv_rgb_image_buffer = input_rgb.data;
        ERROR_CHECK_STATUS(vxCopyImagePatch(input_rgb_image, &cv_rgb_image_region, 0,
                                            &cv_rgb_image_layout, cv_rgb_image_buffer,
                                            VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        ERROR_CHECK_STATUS(vxProcessGraph(graph));
        vx_rectangle_t rect = {0, 0, (vx_uint32)width, (vx_uint32)height};
        vx_map_id map_id;
        vx_imagepatch_addressing_t addr;
        void *ptr;
        ERROR_CHECK_STATUS(vxMapImagePatch(output_pop_image, &rect, 0, &map_id, &addr, &ptr,
                                           VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
        Mat mat(height, width, CV_8U, ptr, addr.stride_y);
        imshow("VX POP - LIVE", mat);
        if (waitKey(30) >= 0)
            break;
        ERROR_CHECK_STATUS(vxUnmapImagePatch(output_pop_image, map_id));
    }

    // release objects
    ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
    ERROR_CHECK_STATUS(vxReleaseThreshold(&hyst));
    ERROR_CHECK_STATUS(vxReleaseThreshold(&thresh95));
    ERROR_CHECK_STATUS(vxReleaseThreshold(&thresh40));
    ERROR_CHECK_STATUS(vxReleaseThreshold(&thresh20));
    ERROR_CHECK_STATUS(vxReleaseThreshold(&thresh15));
    ERROR_CHECK_STATUS(vxReleaseThreshold(&thresh0));
    ERROR_CHECK_STATUS(vxReleaseImage(&yuv_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&luma_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&output_canny_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&input_rgb_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&R_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&G_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&B_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&RmG_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&RmB_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&R95_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&G40_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&B20_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&RmG15_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&RmB0_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&and1_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&and2_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&and3_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&output_skinTone_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&output_canny_skinTone_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&output_pop_image));
    ERROR_CHECK_STATUS(vxReleaseContext(&context));
    return 0;
}
