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

#include <cstring>
#include <iostream>
#include <chrono>

#include <VX/vx.h>
#include <VX/vx_compatibility.h>

using namespace std;

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

    int width = 480, height = 360;

    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);
    vxRegisterLogCallback(context, log_callback, vx_false_e);

    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph);

    vx_image input_rgb_image = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
    vx_image output_filtered_image = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(input_rgb_image);
    ERROR_CHECK_OBJECT(output_filtered_image);

    vx_image yuv_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_IYUV);
    vx_image luma_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(yuv_image);
    ERROR_CHECK_OBJECT(luma_image);

    vx_threshold hyst = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    vx_int32 lower = 80, upper = 100;
    vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, &lower, sizeof(lower));
    vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, &upper, sizeof(upper));
    ERROR_CHECK_OBJECT(hyst);
    vx_int32 gradient_size = 3;

    vx_node nodes[] =
        {
            vxColorConvertNode(graph, input_rgb_image, yuv_image),
            vxChannelExtractNode(graph, yuv_image, VX_CHANNEL_Y, luma_image),
            vxCannyEdgeDetectorNode(graph, luma_image, hyst, gradient_size, VX_NORM_L1, output_filtered_image)};

    for (vx_size i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
    {
        ERROR_CHECK_OBJECT(nodes[i]);
        ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
    }

    ERROR_CHECK_STATUS(vxVerifyGraph(graph));

    auto start = std::chrono::steady_clock::now();
    ERROR_CHECK_STATUS(vxProcessGraph(graph));
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "STATUS: vxProcessGraph() took " << (elapsed_seconds.count()*1000.0f) << "msec (1st iteration)\n";

    start = std::chrono::steady_clock::now();
    for(int i = 0; i < 100; i++){
        ERROR_CHECK_STATUS(vxProcessGraph(graph));
    }
    end = std::chrono::steady_clock::now();
    elapsed_seconds = (end-start)/100;
    std::cout << "STATUS: vxProcessGraph() took " << (elapsed_seconds.count()*1000.0f) << "msec (AVG)\n";

    ERROR_CHECK_STATUS(vxReleaseImage(&yuv_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&luma_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&input_rgb_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&output_filtered_image));
    ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
    ERROR_CHECK_STATUS(vxReleaseContext(&context));

    return 0;
}
