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
#include <vx_ext_opencv.h>

using namespace cv;
using namespace std;

#define ERROR_CHECK_STATUS( status ) { \
        vx_status status_ = (status); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
}

#define ERROR_CHECK_OBJECT( obj ) { \
        vx_status status_ = vxGetStatus((vx_reference)(obj)); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
}

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

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Usage:\n"
                "./cannyDetect --image <imageName>\n"
                "./cannyDetect --live \n");
        return 0;
    }

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

    vx_image yuv_image  = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_IYUV);
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
        vxCannyEdgeDetectorNode(graph, luma_image, hyst, gradient_size, VX_NORM_L1, output_filtered_image)
    };

    for( vx_size i = 0; i < sizeof( nodes ) / sizeof( nodes[0] ); i++ )
    {
        ERROR_CHECK_OBJECT( nodes[i] );
        ERROR_CHECK_STATUS( vxReleaseNode( &nodes[i] ) );
    }

    ERROR_CHECK_STATUS( vxVerifyGraph( graph ) );
    
    string option = argv[1];
    Mat input;

    if (option == "--image") {
        input = imread(argv[2]);
        if (input.empty()) {
           printf("Image not found\n");
           return 0;
        }
        resize(input, input, Size(width, height));
        imshow("inputWindow", input);
        vx_rectangle_t cv_rgb_image_region;
        cv_rgb_image_region.start_x    = 0;
        cv_rgb_image_region.start_y    = 0;
        cv_rgb_image_region.end_x      = width;
        cv_rgb_image_region.end_y      = height;
        vx_imagepatch_addressing_t cv_rgb_image_layout;
        cv_rgb_image_layout.stride_x   = 3;
        cv_rgb_image_layout.stride_y   = input.step;
        vx_uint8 * cv_rgb_image_buffer = input.data;
        ERROR_CHECK_STATUS( vxCopyImagePatch( input_rgb_image, &cv_rgb_image_region, 0,
                                            &cv_rgb_image_layout, cv_rgb_image_buffer,
                                            VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST ) );
        ERROR_CHECK_STATUS( vxProcessGraph( graph ) );
        vx_rectangle_t rect = { 0, 0, (vx_uint32)width, (vx_uint32)height };
        vx_map_id map_id;
        vx_imagepatch_addressing_t addr;
        void * ptr;
        ERROR_CHECK_STATUS( vxMapImagePatch( output_filtered_image, &rect, 0, &map_id, &addr, &ptr,
                                            VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X ) );
        Mat mat( height, width, CV_8U, ptr, addr.stride_y );
        imshow( "CannyDetect", mat );
        waitKey(0);
        ERROR_CHECK_STATUS( vxUnmapImagePatch( output_filtered_image, map_id ) );
    }
    else if (option == "--live") {
        VideoCapture cap(0);
        if (!cap.isOpened()) {
            printf("Unable to open camera\n");
            return 0;
        }
        for(;;) {
            cap >> input;
            resize(input, input, Size(width, height));
            imshow("inputWindow", input);
            if(waitKey(30) >= 0) break;
            vx_rectangle_t cv_rgb_image_region;
            cv_rgb_image_region.start_x    = 0;
            cv_rgb_image_region.start_y    = 0;
            cv_rgb_image_region.end_x      = width;
            cv_rgb_image_region.end_y      = height;
            vx_imagepatch_addressing_t cv_rgb_image_layout;
            cv_rgb_image_layout.stride_x   = 3;
            cv_rgb_image_layout.stride_y   = input.step;
            vx_uint8 * cv_rgb_image_buffer = input.data;
            ERROR_CHECK_STATUS( vxCopyImagePatch( input_rgb_image, &cv_rgb_image_region, 0,
                                                &cv_rgb_image_layout, cv_rgb_image_buffer,
                                                VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST ) );
            ERROR_CHECK_STATUS( vxProcessGraph( graph ) );
            vx_rectangle_t rect = { 0, 0, (vx_uint32)width, (vx_uint32)height };
            vx_map_id map_id;
            vx_imagepatch_addressing_t addr;
            void * ptr;
            ERROR_CHECK_STATUS( vxMapImagePatch( output_filtered_image, &rect, 0, &map_id, &addr, &ptr,
                                            VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X ) );
            Mat mat( height, width, CV_8U, ptr, addr.stride_y );
            imshow( "CannyDetect", mat );
            if(waitKey(30) >= 0) break;
            ERROR_CHECK_STATUS( vxUnmapImagePatch( output_filtered_image, map_id ) );
        }
    }
    else {
        printf("Usage:\n"
                "./cannyDetect --image <imageName>\n"
                "./cannyDetect --live \n");
        return 0;
    }

    ERROR_CHECK_STATUS( vxReleaseGraph( &graph ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &yuv_image ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &luma_image ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &input_rgb_image ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &output_filtered_image ) );
    ERROR_CHECK_STATUS( vxReleaseContext( &context ) );
    return 0;
}
