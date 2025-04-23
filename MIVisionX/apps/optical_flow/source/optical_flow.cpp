/*
Copyright (c) 2022 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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
#include "opencv2/opencv.hpp"

#ifndef DEFAULT_WAITKEY_DELAY
#define DEFAULT_WAITKEY_DELAY 1 /* waitKey delay time in milliseconds after each frame processing */
#endif

//   ERROR_CHECK_STATUS     - check whether the status is VX_SUCCESS
#define ERROR_CHECK_STATUS(status)                                                              \
    {                                                                                           \
        vx_status status_ = (status);                                                           \
        if (status_ != VX_SUCCESS)                                                              \
        {                                                                                       \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1);                                                                            \
        }                                                                                       \
    }

//   ERROR_CHECK_OBJECT     - check whether the object creation is successful
#define ERROR_CHECK_OBJECT(obj)                                                                 \
    {                                                                                           \
        vx_status status_ = vxGetStatus((vx_reference)(obj));                                   \
        if (status_ != VX_SUCCESS)                                                              \
        {                                                                                       \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1);                                                                            \
        }                                                                                       \
    }

// log_callback() function implements a mechanism to print log messages from OpenVX framework onto console.
void VX_CALLBACK log_callback(vx_context context,
                              vx_reference ref,
                              vx_status status,
                              const vx_char string[])
{
    printf("LOG: [ status = %d ] %s\n", status, string);
    fflush(stdout);
}

// OpenCV Draw on image functions
void drawPoint(cv::Mat m_imgBGR, int x, int y)
{
    cv::Point center(x, y);
    cv::circle(m_imgBGR, center, 1, cv::Scalar(0, 0, 255), 2);
}

void drawArrow(cv::Mat m_imgBGR, int x0, int y0, int x1, int y1)
{
    drawPoint(m_imgBGR, x0, y0);
    float dx = (float)(x1 - x0), dy = (float)(y1 - y0), arrow_len = sqrtf(dx * dx + dy * dy);
    if ((arrow_len >= 3.0f) && (arrow_len <= 50.0f))
    {
        cv::Scalar color = cv::Scalar(0, 255, 255);
        float tip_len = 5.0f + arrow_len * 0.1f, angle = atan2f(dy, dx);
        cv::line(m_imgBGR, cv::Point(x0, y0), cv::Point(x1, y1), color, 1);
        cv::line(m_imgBGR, cv::Point(x1, y1), cv::Point(x1 - (int)(tip_len * cosf(angle + (float)CV_PI / 6)), y1 - (int)(tip_len * sinf(angle + (float)CV_PI / 6))), color, 1);
        cv::line(m_imgBGR, cv::Point(x1, y1), cv::Point(x1 - (int)(tip_len * cosf(angle - (float)CV_PI / 6)), y1 - (int)(tip_len * sinf(angle - (float)CV_PI / 6))), color, 1);
    }
}

void drawText(cv::Mat m_imgBGR, int x, int y, const char *text)
{
    cv::putText(m_imgBGR, text, cv::Point(x, y),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(128, 0, 0), 1, cv::LineTypes::LINE_AA);
    printf("text: %s\n", text);
}

// end application
bool abortRequested()
{
    char key = cv::waitKey(DEFAULT_WAITKEY_DELAY);
    if (key == ' ')
    {
        key = cv::waitKey(0);
    }
    if ((key == 'q') || (key == 'Q') || (key == 27) /*ESC*/)
    {
        return true;
    }
    return false;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Usage:\n"
               "\t./opticalFlow --video <Video File>\n"
               "\t./opticalFlow --live  <Capture Device ID>\n");
        return 0;
    }

    vx_uint32 widthSet = 640, heightSet = 480;

    // Set the application configuration parameters
    vx_uint32 width = widthSet;                        // image width
    vx_uint32 height = heightSet;                      // image height
    vx_size maxKeypointCount = 10000;                  // maximum number of keypoints to track
    vx_float32 harrisStrengthThresh = 0.0005f;         // minimum corner strength to keep a corner
    vx_float32 harrisMinDistance = 5.0f;               // radial L2 distance for non-max suppression
    vx_float32 harrisSensitivity = 0.04f;              // multiplier k in det(A) - k * trace(A)^2
    vx_int32 harrisGradientSize = 3;                   // window size for gradient computation
    vx_int32 harrisBlockSize = 3;                      // block window size for Harris corner score
    vx_uint32 lkPyramidLevels = 6;                     // number of pyramid levels for optical flow
    vx_float32 lkPyramidScale = VX_SCALE_PYRAMID_HALF; // pyramid levels scale by factor of two
    vx_enum lkTermination = VX_TERM_CRITERIA_BOTH;     // iteration termination criteria (eps & iterations)
    vx_float32 lkEpsilon = 0.01f;                      // convergence criterion
    vx_uint32 lkNumIterations = 5;                     // maximum number of iterations
    vx_bool lkUseInitialEstimate = vx_false_e;         // don't use initial estimate
    vx_uint32 lkWindowDimension = 6;                   // window size for evaluation
    vx_float32 trackableKpRatioThr = 0.8f;             // threshold for the ration of tracked keypoints to all

    // Create the OpenVX context and check if returned context is valid
    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);

    // Register the log_callback
    vxRegisterLogCallback(context, log_callback, vx_false_e);
    vxAddLogEntry((vx_reference)context, VX_SUCCESS, "OpenVX Sample - Optical Flow\n\n");

    // Create OpenVX image object for input RGB image
    vx_image inputRgbImage = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
    ERROR_CHECK_OBJECT(inputRgbImage);

    // create a OpenVX pyramid and delay objects
    vx_pyramid pyramidExemplar = vxCreatePyramid(context, lkPyramidLevels,
                                                 lkPyramidScale, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(pyramidExemplar);
    vx_delay pyramidDelay = vxCreateDelay(context, (vx_reference)pyramidExemplar, 2);
    ERROR_CHECK_OBJECT(pyramidDelay);
    ERROR_CHECK_STATUS(vxReleasePyramid(&pyramidExemplar));
    vx_array keypointsExemplar = vxCreateArray(context, VX_TYPE_KEYPOINT, maxKeypointCount);
    ERROR_CHECK_OBJECT(keypointsExemplar);
    vx_delay keypointsDelay = vxCreateDelay(context, (vx_reference)keypointsExemplar, 2);
    ERROR_CHECK_STATUS(vxReleaseArray(&keypointsExemplar));

    vx_pyramid currentPyramid = (vx_pyramid)vxGetReferenceFromDelay(pyramidDelay, 0);
    vx_pyramid previousPyramid = (vx_pyramid)vxGetReferenceFromDelay(pyramidDelay, -1);
    vx_array currentKeypoints = (vx_array)vxGetReferenceFromDelay(keypointsDelay, 0);
    vx_array previousKeypoints = (vx_array)vxGetReferenceFromDelay(keypointsDelay, -1);
    ERROR_CHECK_OBJECT(currentPyramid);
    ERROR_CHECK_OBJECT(previousPyramid);
    ERROR_CHECK_OBJECT(currentKeypoints);
    ERROR_CHECK_OBJECT(previousKeypoints);

    // Harris and optical flow algorithms require their own graph objects
    vx_graph graphHarris = vxCreateGraph(context);
    vx_graph graphTrack = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graphHarris);
    ERROR_CHECK_OBJECT(graphTrack);

    // Harris and pyramid computation expect input to be an 8-bit image
    vx_image harrisYuvImage = vxCreateVirtualImage(graphHarris, width, height, VX_DF_IMAGE_IYUV);
    vx_image harrisGrayImage = vxCreateVirtualImage(graphHarris, width, height, VX_DF_IMAGE_U8);
    vx_image opticalflowYuvImage = vxCreateVirtualImage(graphTrack, width, height, VX_DF_IMAGE_IYUV);
    vx_image opticalFlowGrayImage = vxCreateVirtualImage(graphTrack, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(harrisYuvImage);
    ERROR_CHECK_OBJECT(harrisGrayImage);
    ERROR_CHECK_OBJECT(opticalflowYuvImage);
    ERROR_CHECK_OBJECT(opticalFlowGrayImage);

    // The Harris corner detector and optical flow nodes scalar objects as parameters
    vx_scalar strengthThresh = vxCreateScalar(context, VX_TYPE_FLOAT32, &harrisStrengthThresh);
    vx_scalar minDistance = vxCreateScalar(context, VX_TYPE_FLOAT32, &harrisMinDistance);
    vx_scalar sensitivity = vxCreateScalar(context, VX_TYPE_FLOAT32, &harrisSensitivity);
    vx_scalar epsilon = vxCreateScalar(context, VX_TYPE_FLOAT32, &lkEpsilon);
    vx_scalar numIterations = vxCreateScalar(context, VX_TYPE_UINT32, &lkNumIterations);
    vx_scalar useInitialEstimate = vxCreateScalar(context, VX_TYPE_BOOL, &lkUseInitialEstimate);
    ERROR_CHECK_OBJECT(strengthThresh);
    ERROR_CHECK_OBJECT(minDistance);
    ERROR_CHECK_OBJECT(sensitivity);
    ERROR_CHECK_OBJECT(epsilon);
    ERROR_CHECK_OBJECT(numIterations);
    ERROR_CHECK_OBJECT(useInitialEstimate);

    // Graph to perform Harris corner detection and initial pyramid computation
    vx_node nodesHarris[] =
        {
            vxColorConvertNode(graphHarris, inputRgbImage, harrisYuvImage),
            vxChannelExtractNode(graphHarris, harrisYuvImage, VX_CHANNEL_Y, harrisGrayImage),
            vxGaussianPyramidNode(graphHarris, harrisGrayImage, currentPyramid),
            vxHarrisCornersNode(graphHarris, harrisGrayImage, strengthThresh, minDistance, sensitivity, harrisGradientSize, harrisBlockSize, currentKeypoints, NULL)};
    for (vx_size i = 0; i < sizeof(nodesHarris) / sizeof(nodesHarris[0]); i++)
    {
        ERROR_CHECK_OBJECT(nodesHarris[i]);
        ERROR_CHECK_STATUS(vxReleaseNode(&nodesHarris[i]));
    }
    ERROR_CHECK_STATUS(vxReleaseImage(&harrisYuvImage));
    ERROR_CHECK_STATUS(vxReleaseImage(&harrisGrayImage));
    ERROR_CHECK_STATUS(vxVerifyGraph(graphHarris));

    // Graph to compute image pyramid for the next frame, and tracks features using optical flow
    vx_node nodesTrack[] =
        {
            vxColorConvertNode(graphTrack, inputRgbImage, opticalflowYuvImage),
            vxChannelExtractNode(graphTrack, opticalflowYuvImage, VX_CHANNEL_Y, opticalFlowGrayImage),
            vxGaussianPyramidNode(graphTrack, opticalFlowGrayImage, currentPyramid),
            vxOpticalFlowPyrLKNode(graphTrack, previousPyramid, currentPyramid,
                                   previousKeypoints, previousKeypoints, currentKeypoints,
                                   lkTermination, epsilon, numIterations,
                                   useInitialEstimate, lkWindowDimension)};
    for (vx_size i = 0; i < sizeof(nodesTrack) / sizeof(nodesTrack[0]); i++)
    {
        ERROR_CHECK_OBJECT(nodesTrack[i]);
        ERROR_CHECK_STATUS(vxReleaseNode(&nodesTrack[i]));
    }
    ERROR_CHECK_STATUS(vxReleaseImage(&opticalflowYuvImage));
    ERROR_CHECK_STATUS(vxReleaseImage(&opticalFlowGrayImage));
    ERROR_CHECK_STATUS(vxVerifyGraph(graphTrack));

    std::string option = argv[1];
    cv::Mat input;

    if (option == "--video")
    {
        cv::VideoCapture cap(argv[2]);
        int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        if (!cap.isOpened())
        {
            vxAddLogEntry((vx_reference)context, VX_FAILURE, "ERROR: Unable to open video\n");
            return VX_FAILURE;
        }
        for (int frameIndex = 0; !abortRequested(); frameIndex++)
        {
            if (frameIndex >= totalFrames)
            {
                vxAddLogEntry((vx_reference)context, VX_SUCCESS, "INFO: End Of Video File\n");
                break;
            }

            cap >> input;
            resize(input, input, cv::Size(width, height));
            cv::imshow("inputWindow", input);
            cv::Mat output = input;
            if (cv::waitKey(30) >= 0)
                break;
            vx_rectangle_t cvRgbImageRegion;
            cvRgbImageRegion.start_x = 0;
            cvRgbImageRegion.start_y = 0;
            cvRgbImageRegion.end_x = width;
            cvRgbImageRegion.end_y = height;
            vx_imagepatch_addressing_t cvRgbImageLayout;
            cvRgbImageLayout.stride_x = 3;
            cvRgbImageLayout.stride_y = input.step;
            vx_uint8 *cvRgbImageBuffer = input.data;
            ERROR_CHECK_STATUS(vxCopyImagePatch(inputRgbImage, &cvRgbImageRegion, 0,
                                                &cvRgbImageLayout, cvRgbImageBuffer,
                                                VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

            // process graph
            ERROR_CHECK_STATUS(vxProcessGraph(frameIndex == 0 ? graphHarris : graphTrack));

            // Mark the keypoints in display
            vx_size numCorners = 0, numTracking = 0;
            previousKeypoints = (vx_array)vxGetReferenceFromDelay(keypointsDelay, -1);
            currentKeypoints = (vx_array)vxGetReferenceFromDelay(keypointsDelay, 0);
            ERROR_CHECK_OBJECT(currentKeypoints);
            ERROR_CHECK_OBJECT(previousKeypoints);
            ERROR_CHECK_STATUS(vxQueryArray(previousKeypoints, VX_ARRAY_NUMITEMS, &numCorners, sizeof(numCorners)));
            if (numCorners > 0)
            {
                vx_size kpOldStride, kpNewStride;
                vx_map_id kpOldMap, kpNewMap;
                vx_uint8 *kpOldBuf, *kpNewBuf;
                ERROR_CHECK_STATUS(vxMapArrayRange(previousKeypoints, 0, numCorners, &kpOldMap,
                                                   &kpOldStride, (void **)&kpOldBuf,
                                                   VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
                ERROR_CHECK_STATUS(vxMapArrayRange(currentKeypoints, 0, numCorners, &kpNewMap,
                                                   &kpNewStride, (void **)&kpNewBuf,
                                                   VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
                for (vx_size i = 0; i < numCorners; i++)
                {
                    vx_keypoint_t *kpOld = (vx_keypoint_t *)(kpOldBuf + i * kpOldStride);
                    vx_keypoint_t *kpNew = (vx_keypoint_t *)(kpNewBuf + i * kpNewStride);
                    if (kpNew->tracking_status)
                    {
                        numTracking++;
                        drawArrow(output, kpOld->x, kpOld->y, kpNew->x, kpNew->y);
                    }
                }
                ERROR_CHECK_STATUS(vxUnmapArrayRange(previousKeypoints, kpOldMap));
                ERROR_CHECK_STATUS(vxUnmapArrayRange(currentKeypoints, kpNewMap));
            }

            // Increase the age of the delay objects to make the current entry become previous entry
            ERROR_CHECK_STATUS(vxAgeDelay(pyramidDelay));
            ERROR_CHECK_STATUS(vxAgeDelay(keypointsDelay));

            // Display the results
            char text[128];
            sprintf(text, "Keyboard: [ESC/Q] -- Quit [SPACE] -- Pause [FRAME %d]", frameIndex);
            drawText(output, 0, 16, text);
            sprintf(text, "Number of Corners: %d [tracking %d]", (int)numCorners, (int)numTracking);
            drawText(output, 0, 36, text);
            cv::imshow("opticalFlow", output);
        }
    }
    else if (option == "--live")
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            vxAddLogEntry((vx_reference)context, VX_FAILURE, "ERROR: Unable to open camera\n");
            return VX_FAILURE;
        }
        for (int frameIndex = 0; !abortRequested(); frameIndex++)
        {
            cap >> input;
            resize(input, input, cv::Size(width, height));
            cv::imshow("inputWindow", input);
            cv::Mat output = input;
            if (cv::waitKey(30) >= 0)
                break;
            vx_rectangle_t cvRgbImageRegion;
            cvRgbImageRegion.start_x = 0;
            cvRgbImageRegion.start_y = 0;
            cvRgbImageRegion.end_x = width;
            cvRgbImageRegion.end_y = height;
            vx_imagepatch_addressing_t cvRgbImageLayout;
            cvRgbImageLayout.stride_x = 3;
            cvRgbImageLayout.stride_y = input.step;
            vx_uint8 *cvRgbImageBuffer = input.data;
            ERROR_CHECK_STATUS(vxCopyImagePatch(inputRgbImage, &cvRgbImageRegion, 0,
                                                &cvRgbImageLayout, cvRgbImageBuffer,
                                                VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

            // process graph
            ERROR_CHECK_STATUS(vxProcessGraph(frameIndex == 0 ? graphHarris : graphTrack));

            // Mark the keypoints in display
            vx_size numCorners = 0, numTracking = 0;
            previousKeypoints = (vx_array)vxGetReferenceFromDelay(keypointsDelay, -1);
            currentKeypoints = (vx_array)vxGetReferenceFromDelay(keypointsDelay, 0);
            ERROR_CHECK_OBJECT(currentKeypoints);
            ERROR_CHECK_OBJECT(previousKeypoints);
            ERROR_CHECK_STATUS(vxQueryArray(previousKeypoints, VX_ARRAY_NUMITEMS, &numCorners, sizeof(numCorners)));
            if (numCorners > 0)
            {
                vx_size kpOldStride, kpNewStride;
                vx_map_id kpOldMap, kpNewMap;
                vx_uint8 *kpOldBuf, *kpNewBuf;
                ERROR_CHECK_STATUS(vxMapArrayRange(previousKeypoints, 0, numCorners, &kpOldMap,
                                                   &kpOldStride, (void **)&kpOldBuf,
                                                   VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
                ERROR_CHECK_STATUS(vxMapArrayRange(currentKeypoints, 0, numCorners, &kpNewMap,
                                                   &kpNewStride, (void **)&kpNewBuf,
                                                   VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
                for (vx_size i = 0; i < numCorners; i++)
                {
                    vx_keypoint_t *kpOld = (vx_keypoint_t *)(kpOldBuf + i * kpOldStride);
                    vx_keypoint_t *kpNew = (vx_keypoint_t *)(kpNewBuf + i * kpNewStride);
                    if (kpNew->tracking_status)
                    {
                        numTracking++;
                        drawArrow(output, kpOld->x, kpOld->y, kpNew->x, kpNew->y);
                    }
                }
                ERROR_CHECK_STATUS(vxUnmapArrayRange(previousKeypoints, kpOldMap));
                ERROR_CHECK_STATUS(vxUnmapArrayRange(currentKeypoints, kpNewMap));
            }

            // Increase the age of the delay objects to make the current entry become previous entry
            ERROR_CHECK_STATUS(vxAgeDelay(pyramidDelay));
            ERROR_CHECK_STATUS(vxAgeDelay(keypointsDelay));

            // Display the results
            char text[128];
            sprintf(text, "Keyboard: [ESC/Q] -- Quit [SPACE] -- Pause [FRAME %d]", frameIndex);
            drawText(output, 0, 16, text);
            sprintf(text, "Number of Corners: %d [tracking %d]", (int)numCorners, (int)numTracking);
            drawText(output, 0, 36, text);
            cv::imshow("opticalFlow", output);
        }
    }
    else
    {
        printf("ERROR: Usage --\n"
               "\t./opticalFlow --video <Video File>\n"
               "\t./opticalFlow --live  <Capture Device ID>\n");
        return 0;
    }

    // Query graph performance using VX_GRAPH_PERFORMANCE and print timing in milliseconds
    vx_perf_t perfHarris = {0}, perfTrack = {0};
    ERROR_CHECK_STATUS(vxQueryGraph(graphHarris, VX_GRAPH_PERFORMANCE, &perfHarris, sizeof(perfHarris)));
    ERROR_CHECK_STATUS(vxQueryGraph(graphTrack, VX_GRAPH_PERFORMANCE, &perfTrack, sizeof(perfTrack)));
    printf("GraphName NumFrames Avg(ms) Min(ms)\n"
           "Harris    %9d %7.3f %7.3f\n"
           "Track     %9d %7.3f %7.3f\n",
           (int)perfHarris.num, (float)perfHarris.avg * 1e-6f, (float)perfHarris.min * 1e-6f,
           (int)perfTrack.num, (float)perfTrack.avg * 1e-6f, (float)perfTrack.min * 1e-6f);

    // Release all the OpenVX objects
    ERROR_CHECK_STATUS(vxReleaseGraph(&graphHarris));
    ERROR_CHECK_STATUS(vxReleaseGraph(&graphTrack));
    ERROR_CHECK_STATUS(vxReleaseImage(&inputRgbImage));
    ERROR_CHECK_STATUS(vxReleaseDelay(&pyramidDelay));
    ERROR_CHECK_STATUS(vxReleaseDelay(&keypointsDelay));
    ERROR_CHECK_STATUS(vxReleaseScalar(&strengthThresh));
    ERROR_CHECK_STATUS(vxReleaseScalar(&minDistance));
    ERROR_CHECK_STATUS(vxReleaseScalar(&sensitivity));
    ERROR_CHECK_STATUS(vxReleaseScalar(&epsilon));
    ERROR_CHECK_STATUS(vxReleaseScalar(&numIterations));
    ERROR_CHECK_STATUS(vxReleaseScalar(&useInitialEstimate));
    ERROR_CHECK_STATUS(vxReleaseContext(&context));

    return 0;
}