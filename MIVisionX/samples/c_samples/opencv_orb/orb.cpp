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
    if (argc < 1) {
        printf("Usage: ./orbDetect\n");
        return 0;
    }

    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);
    vxRegisterLogCallback(context, log_callback, vx_false_e);
    
    vxLoadKernels(context, "vx_opencv");
    
    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph);
    
    int width = 1280, height = 720;
    vx_image inter_luma = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(inter_luma);

    VideoCapture cap(0);
    Mat input, output;
    cap >> input;
    cap >> output;
    cvtColor(input, input, COLOR_RGB2GRAY);
    
    if (input.empty()) {
        printf("Image not found\n");
    }
    cv::resize(input, input, Size(width, height));
    cv::resize(output, output, Size(width, height));
    
    vx_rectangle_t cv_image_region;
    cv_image_region.start_x    = 0;
    cv_image_region.start_y    = 0;
    cv_image_region.end_x      = width;
    cv_image_region.end_y      = height;
    vx_imagepatch_addressing_t cv_image_layout;
    cv_image_layout.stride_x   = 1;
    cv_image_layout.stride_y   = input.step;
    vx_uint8 * cv_image_buffer = input.data;
    
    ERROR_CHECK_STATUS( vxCopyImagePatch( inter_luma, &cv_image_region, 0,
                                          &cv_image_layout, cv_image_buffer,
                                          VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST ) );

    vx_array keypoints = vxCreateArray( context, VX_TYPE_KEYPOINT, 10000 );
    ERROR_CHECK_OBJECT( keypoints );
    
    vx_int32 nFeatures = 1000;
    vx_float32 scaleFactor = 1.2;
    vx_int32 nlevels = 2;
    vx_int32 edgeThreshold = 31;
    vx_int32 firstLevel = 0;
    vx_int32 WTA_K = 2;
    vx_int32 scoreType = 0;
    vx_int32 patchSize = 31;

    vx_node nodes[] =
    {
        vxExtCvNode_orbDetect(graph, inter_luma, inter_luma, keypoints, nFeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize)
    };

    for( vx_size i = 0; i < sizeof( nodes ) / sizeof( nodes[0] ); i++ )
    {
        ERROR_CHECK_OBJECT( nodes[i] );
        ERROR_CHECK_STATUS( vxReleaseNode( &nodes[i] ) );
    }

    ERROR_CHECK_STATUS( vxVerifyGraph( graph ) );
    ERROR_CHECK_STATUS( vxProcessGraph( graph ) );

    vx_size num_corners = 0;
    ERROR_CHECK_STATUS(vxQueryArray (keypoints, VX_ARRAY_NUMITEMS, &num_corners, sizeof(num_corners)));
    if (num_corners > 0) {
        vx_size kp_stride;
        vx_map_id kp_map;
        vx_uint8 * kp_buf;
        ERROR_CHECK_STATUS(vxMapArrayRange (keypoints, 0, num_corners, &kp_map, &kp_stride, (void **)&kp_buf, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
        for (vx_size i = 0; i < num_corners; i++) {
            vx_keypoint_t * kp = (vx_keypoint_t *) (kp_buf + i*kp_stride);
            cv::Point center (kp->x, kp->y);
            cv::circle (output, center, 1, cv::Scalar(0,255, 0),2);
        }
    }

    imshow( "OrbDetect", output );
    waitKey(0);

    ERROR_CHECK_STATUS( vxReleaseGraph( &graph ) );
    ERROR_CHECK_STATUS( vxReleaseArray (&keypoints));
    ERROR_CHECK_STATUS( vxReleaseImage( &inter_luma ) );
    ERROR_CHECK_STATUS( vxReleaseContext( &context ) );
    return 0;
}
