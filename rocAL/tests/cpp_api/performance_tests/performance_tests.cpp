/*
MIT License

Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"
#include "rocal_api.h"
using namespace cv;
#if USE_OPENCV_4
#define CV_LOAD_IMAGE_COLOR IMREAD_COLOR
#define CV_BGR2GRAY COLOR_BGR2GRAY
#define CV_GRAY2RGB COLOR_GRAY2RGB
#define CV_RGB2BGR COLOR_RGB2BGR
#define CV_FONT_HERSHEY_SIMPLEX FONT_HERSHEY_SIMPLEX
#define CV_FILLED FILLED
#define CV_WINDOW_AUTOSIZE WINDOW_AUTOSIZE
#define cvDestroyWindow destroyWindow
#endif

#define DISPLAY
using namespace std::chrono;

int test(int test_case, const char* path, int rgb, int processing_device, int width, int height, int batch_size, int shards, int shuffle);
int main(int argc, const char** argv) {
    // check command-line usage
    const int MIN_ARG_COUNT = 4;
    if (argc < MIN_ARG_COUNT) {
        printf("Usage: performance_tests <image-dataset-folder - required> <width - required> <height - required> <test_case> <batch_size> <gpu=1/cpu=0> <rgb=1/grayscale=0> <shard_count>  <shuffle=1>\n");
        return -1;
    }

    int argIdx = 1;
    const char* path = argv[argIdx++];
    int width = atoi(argv[argIdx++]);
    int height = atoi(argv[argIdx++]);

    int rgb = 1;  // process color images
    bool processing_device = 1;
    int test_case = 0;
    int batch_size = 10;
    int shards = 4;
    int shuffle = 0;

    if (argc > argIdx)
        test_case = atoi(argv[argIdx++]);

    if (argc > argIdx)
        batch_size = atoi(argv[argIdx++]);

    if (argc > argIdx)
        processing_device = atoi(argv[argIdx++]);

    if (argc > argIdx)
        rgb = atoi(argv[argIdx++]);

    if (argc > argIdx)
        shards = atoi(argv[argIdx++]);

    if (argc > argIdx)
        shuffle = atoi(argv[argIdx++]);

    return test(test_case, path, rgb, processing_device, width, height, batch_size, shards, shuffle);
}

int test(int test_case, const char* path, int rgb, int processing_device, int width, int height, int batch_size, int shards, int shuffle) {
    size_t num_threads = shards;
    int inputBatchSize = batch_size;
    int decode_max_width = 0;
    int decode_max_height = 0;
    std::cout << "Test case " << test_case << std::endl;
    std::cout << "Running on " << (processing_device ? "GPU" : "CPU") << ", " << (rgb ? "Color " : "Grayscale ") << std::endl;
    std::cout << "Batch size: "<< inputBatchSize << std::endl << "shard count: " << num_threads << std::endl;

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24 : RocalImageColor::ROCAL_COLOR_U8;

    auto handle = rocalCreate(inputBatchSize, processing_device ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0, 1);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the Rocal context\n";
        return -1;
    }

    /*>>>>>>>>>>>>>>>> Creating Rocal parameters  <<<<<<<<<<<<<<<<*/

    rocalSetSeed(0);

    // Creating uniformly distributed random objects to override some of the default augmentation parameters
    RocalFloatParam rand_crop_area = rocalCreateFloatUniformRand(0.3, 0.5);
    RocalIntParam color_temp_adj = rocalCreateIntParameter(-50);

    // Creating a custom random object to set a limited number of values to randomize the rotation angle
    const size_t num_values = 3;
    float values[num_values] = {0, 10, 135};
    double frequencies[num_values] = {1, 5, 5};
    RocalFloatParam rand_angle = rocalCreateFloatRand(values, frequencies,
                                                      sizeof(values) / sizeof(values[0]));

    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    RocalTensor tensor0;
    RocalTensor tensor0_b;

    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    if (decode_max_height <= 0 || decode_max_width <= 0)
        tensor0 = rocalJpegFileSource(handle, path, color_format, num_threads, false, shuffle, true);
    else
        tensor0 = rocalJpegFileSource(handle, path, color_format, num_threads, false, shuffle, true,
                                      ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "JPEG source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    int resize_w = width, resize_h = height;

    switch (test_case) {
        case 0: {
            std::cout << "Running rocalResize" << std::endl;
            rocalResize(handle, tensor0, resize_w, resize_h, true);
        } break;
        case 1: {
            std::cout << "Running rocalCropResize" << std::endl;
            rocalCropResize(handle, tensor0, resize_w, resize_h, true, rand_crop_area);
        } break;
        case 2: {
            std::cout << "Running rocalRotate" << std::endl;
            rocalRotate(handle, tensor0, true, rand_angle);
        } break;
        case 3: {
            std::cout << "Running rocalBrightness" << std::endl;
            rocalBrightness(handle, tensor0, true);
        } break;
        case 4: {
            std::cout << "Running rocalGamma" << std::endl;
            rocalGamma(handle, tensor0, true);
        } break;
        case 5: {
            std::cout << "Running rocalContrast" << std::endl;
            rocalContrast(handle, tensor0, true);
        } break;
        case 6: {
            std::cout << "Running rocalFlip" << std::endl;
            rocalFlip(handle, tensor0, true);
        } break;
        case 7: {
            std::cout << "Running  rocalBlur" << std::endl;
            rocalBlur(handle, tensor0, true);
        } break;
        case 8: {
            std::cout << "Running rocalBlend" << std::endl;
            tensor0_b = rocalRotateFixed(handle, tensor0, 30, false);
            rocalBlend(handle, tensor0, tensor0_b, true);
        } break;
        case 9: {
            std::cout << "Running rocalWarpAffine" << std::endl;
            rocalWarpAffine(handle, tensor0, true);
        } break;
        case 10: {
            std::cout << "Running rocalFishEye" << std::endl;
            rocalFishEye(handle, tensor0, true);
        } break;
        case 11: {
            std::cout << "Running rocalVignette" << std::endl;
            rocalVignette(handle, tensor0, true);
        } break;
        case 12: {
            std::cout << "Running rocalJitter" << std::endl;
            rocalJitter(handle, tensor0, true);
        } break;
        case 13: {
            std::cout << "Running rocalSnPNoise" << std::endl;
            rocalSnPNoise(handle, tensor0, true);
        } break;
        case 14: {
            std::cout << "Running rocalSnow" << std::endl;
            rocalSnow(handle, tensor0, true);
        } break;
        case 15: {
            std::cout << "Running rocalRain" << std::endl;
            rocalRain(handle, tensor0, true);
        } break;
        case 16: {
            std::cout << "Running rocalColorTemp" << std::endl;
            rocalColorTemp(handle, tensor0, true, color_temp_adj);
        } break;
        case 17: {
            std::cout << "Running rocalFog" << std::endl;
            rocalFog(handle, tensor0, true);
        } break;
        case 18: {
            std::cout << "Running rocalLensCorrection" << std::endl;
            rocalLensCorrection(handle, tensor0, true);
        } break;
        case 19: {
            std::cout << "Running rocalPixelate" << std::endl;
            rocalPixelate(handle, tensor0, true);
        } break;
        case 20: {
            std::cout << "Running rocalExposure" << std::endl;
            rocalExposure(handle, tensor0, true);
        } break;
        case 21: {
            std::cout << "Running rocalHue" << std::endl;
            rocalHue(handle, tensor0, true);
        } break;
        case 22: {
            std::cout << "Running rocalSaturation" << std::endl;
            rocalSaturation(handle, tensor0, true);
        } break;
        case 23: {
            std::cout << "Running rocalCopy" << std::endl;
            rocalCopy(handle, tensor0, true);
        } break;
        case 24: {
            std::cout << "Running rocalColorTwist" << std::endl;
            rocalColorTwist(handle, tensor0, true);
        } break;
        case 25: {
            std::cout << "Running rocalCropMirrorNormalize" << std::endl;
            std::vector<float> mean;
            std::vector<float> std_dev;
            rocalCropMirrorNormalize(handle, tensor0, 200, 200, 50, 50, mean, std_dev, true);
        } break;
        case 26: {
            std::cout << "Running rocalCrop " << std::endl;
            rocalCrop(handle, tensor0, true);
        } break;
        case 27: {
            std::cout << "Running rocalResizeCropMirror" << std::endl;
            rocalResizeCropMirror(handle, tensor0, resize_w, resize_h, true);
        } break;
        case 28: {
            std::cout << "Running No-Op" << std::endl;
            rocalNop(handle, tensor0, true);
        } break;
        default:
            std::cout << "Not a valid option! Exiting!\n";
            return -1;
    }

    // Calling the API to verify and build the augmentation graph
    rocalVerify(handle);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    printf("Augmented copies count %lu\n", rocalGetAugmentationBranchCount(handle));

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    int i = 0;
    while (i++ < 100 && !rocalIsEmpty(handle)) {
        if (rocalRun(handle) != 0) {
            std::cout << "rocalRun Failed with runtime error" << std::endl;
            rocalRelease(handle);
            return -1;
        }

        // auto last_colot_temp = rocalGetIntValue(color_temp_adj);
        // rocalUpdateIntParameter(last_colot_temp + 1, color_temp_adj);

        // rocalCopyToOutput(handle, mat_input.data, h * w * p);
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "Load     time " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cout << "Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;

    rocalRelease(handle);

    return 0;
}
