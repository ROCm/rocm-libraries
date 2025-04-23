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
#endif

#define DISPLAY 0
using namespace std::chrono;

int test(int test_case, const char* path, int rgb, int gpu, int width, int height, int batch_size, int graph_depth);
int main(int argc, const char** argv) {
    // check command-line usage
    const int MIN_ARG_COUNT = 4;
    printf("Usage: performance_tests_with_depth <image-dataset-folder - required> <width - required> <height - required> test_case batch_size graph_depth gpu=1/cpu=0 rgb=1/grayscale =0  \n");
    if (argc < MIN_ARG_COUNT)
        return -1;

    int argIdx = 1;
    const char* path = argv[argIdx++];
    int width = atoi(argv[argIdx++]);
    int height = atoi(argv[argIdx++]);

    int rgb = 1;  // process color images
    bool gpu = 1;
    int test_case = 0;
    int batch_size = 10;
    int graph_depth = 1;

    if (argc > argIdx)
        test_case = atoi(argv[argIdx++]);

    if (argc > argIdx)
        batch_size = atoi(argv[argIdx++]);

    if (argc > argIdx)
        graph_depth = atoi(argv[argIdx++]);

    if (argc > argIdx)
        gpu = atoi(argv[argIdx++]);

    if (argc > argIdx)
        rgb = atoi(argv[argIdx++]);

    return test(test_case, path, rgb, gpu, width, height, batch_size, graph_depth);
}

int test(int test_case, const char* path, int rgb, int gpu, int width, int height, int batch_size, int graph_depth) {
    size_t num_threads = 1;
    int inputBatchSize = 1;
    int decode_max_width = width;
    int decode_max_height = height;
    std::cout << "Test case " << test_case << std::endl;
    std::cout << "Running on " << (gpu ? "GPU" : "CPU") << ", " << (rgb ? " Color " : " Grayscale ") << std::endl;

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24
                                              : RocalImageColor::ROCAL_COLOR_U8;

    auto handle = rocalCreate(inputBatchSize,
                              gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0,
                              1);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the Rocal contex\n";
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
    RocalTensor input_image;
    RocalTensor tensor0;
    RocalTensor tensor0_b;

    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    if (decode_max_height <= 0 || decode_max_width <= 0)
        input_image = rocalJpegFileSource(handle, path, color_format, num_threads, false, true, true);
    else
        input_image = rocalJpegFileSource(handle, path, color_format, num_threads, false, true, true,
                                          ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "JPEG source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    int resize_w = width, resize_h = height;

    switch (test_case) {
        case 0: {
            std::cout << "Running rocalResize" << std::endl;

            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalResize(handle, tensor0, resize_w, resize_h, true);
                }
            }
        } break;
        case 1: {
            std::cout << "Running rocalCropResize" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalCropResize(handle, tensor0, resize_w, resize_h, true, rand_crop_area);
                }
            }
        } break;
        case 2: {
            std::cout << "Running rocalCropResizeFixed" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalCropResizeFixed(handle, tensor0, resize_w, resize_h, true, 0.25, 1.2, 0.6, 0.4);
                }
            }
        } break;
        case 3: {
            std::cout << "Running rocalRotate" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalRotate(handle, tensor0, true, rand_angle);
                }
            }
        } break;
        case 4: {
            std::cout << "Running rocalRotateFixed" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalRotateFixed(handle, tensor0, 45, true, resize_w, resize_h);
                }
            }
        } break;
        case 5: {
            std::cout << "Running rocalBrightness" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalBrightness(handle, tensor0, true);
                }
            }
        } break;
        case 6: {
            std::cout << "Running rocalBrightnessFixed" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalBrightnessFixed(handle, tensor0, 4, 50, true);
                }
            }
        } break;
        case 7: {
            std::cout << "Running rocalGamma" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalGamma(handle, tensor0, true);
                }
            }
        } break;
        case 8: {
            std::cout << "Running rocalGammaFixed" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalGammaFixed(handle, tensor0, 0.5, true);
                }
            }
        } break;
        case 9: {
            std::cout << "Running rocalContrast" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalContrast(handle, tensor0, true);
                }
            }
        } break;
        case 10: {
            std::cout << "Running rocalContrastFixed" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalContrastFixed(handle, tensor0, 30, 380, true);
                }
            }
        } break;
        case 11: {
            std::cout << "Running rocalFlip horizontal" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalFlip(handle, tensor0, true);
                }
            }
        } break;
        case 12: {
            std::cout << "Running rocalFlip vertical" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalFlip(handle, tensor0, true);
                }
            }
        } break;
        case 13: {
            std::cout << "Running rocalBlur" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalBlur(handle, tensor0, true);
                }
            }
        } break;
        case 14: {
            std::cout << "Running rocalBlurFixed" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalBlurFixed(handle, tensor0, 17, true);
                }
            }
        } break;
        case 15: {
            std::cout << "Running rocalBlend" << std::endl;

            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                tensor0_b = rocalRotateFixed(handle, tensor0, 30, false);
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalBlend(handle, tensor0, tensor0_b, true);
                }
            }
        } break;
        case 16: {
            std::cout << "Running rocalBlendFixed" << std::endl;
            tensor0 = input_image;
            tensor0_b = rocalRotateFixed(handle, tensor0, 30, false);
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                tensor0_b = rocalRotateFixed(handle, tensor0, 30, false);
                for (int k = 0; k < graph_depth; k++) {
                    rocalBlendFixed(handle, tensor0, tensor0_b, 0.5, true);
                }
            }
        } break;

        case 17: {
            std::cout << "Running rocalWarpAffine" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalWarpAffine(handle, tensor0, true);
                }
            }
        } break;
        case 18: {
            std::cout << "Running rocalWarpAffineFixed" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalWarpAffineFixed(handle, tensor0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, true);
                }
            }
        } break;
        case 19: {
            std::cout << "Running rocalFishEye" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalFishEye(handle, tensor0, true);
                }
            }
        } break;
        case 20: {
            std::cout << "Running rocalVignette" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalVignette(handle, tensor0, true);
                }
            }
        } break;
        case 21: {
            std::cout << "Running rocalVignetteFixed" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalVignetteFixed(handle, tensor0, 40, true);
                }
            }
        } break;
        case 22: {
            std::cout << "Running rocalJitter" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalJitter(handle, tensor0, true);
                }
            }
        } break;
        case 23: {
            std::cout << "Running rocalJitterFixed" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalJitterFixed(handle, tensor0, 3, true);
                }
            }
        } break;
        case 24: {
            std::cout << "Running rocalSnPNoise" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalSnPNoise(handle, tensor0, true);
                }
            }
        } break;
        case 25: {
            std::cout << "Running rocalSnPNoiseFixed" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalSnPNoiseFixed(handle, tensor0, 0.2, 0.2, 0.2, 0.5, true, 0);
                }
            }
        } break;
        case 26: {
            std::cout << "Running rocalSnow" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalSnow(handle, tensor0, true);
                }
            }
        } break;
        case 27: {
            std::cout << "Running rocalSnowFixed" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalSnowFixed(handle, tensor0, 0.5, true);
                }
            }
        } break;
        case 28: {
            std::cout << "Running rocalRain" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalRain(handle, tensor0, true);
                }
            }
        } break;
        case 29: {
            std::cout << "Running rocalRainFixed" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalRainFixed(handle, tensor0, 0.5, 2, 16, 0.25, true);
                }
            }
        } break;
        case 30: {
            std::cout << "Running rocalColorTemp" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalColorTemp(handle, tensor0, true, color_temp_adj);
                }
            }
        } break;
        case 31: {
            std::cout << "Running rocalColorTempFixed" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalColorTempFixed(handle, tensor0, 70, true);
                }
            }
        } break;
        case 32: {
            std::cout << "Running rocalFog" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalFog(handle, tensor0, true);
                }
            }
        } break;
        case 33: {
            std::cout << "Running rocalFogFixed" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalFogFixed(handle, tensor0, 2.5, true);
                }
            }
        } break;
        case 34: {
            std::cout << "Running rocalLensCorrection" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalLensCorrection(handle, tensor0, true);
                }
            }
        } break;
        case 35: {
            std::cout << "Running rocalLensCorrectionFixed" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalLensCorrectionFixed(handle, tensor0, 2.9, 1.5, true);
                }
            }
        } break;
        case 36: {
            std::cout << "Running rocalPixelate" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalPixelate(handle, tensor0, true);
                }
            }
        } break;
        case 37: {
            std::cout << "Running rocalExposure" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalExposure(handle, tensor0, true);
                }
            }
        } break;
        case 38: {
            std::cout << "Running rocalExposureFixed" << std::endl;
            for (int j = 0; j < batch_size; j++) {
                tensor0 = input_image;
                for (int k = 0; k < graph_depth; k++) {
                    tensor0 = rocalExposureFixed(handle, tensor0, 1, true);
                }
            }
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

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    int h = rocalGetAugmentationBranchCount(handle) * rocalGetOutputHeight(handle);
    int w = rocalGetOutputWidth(handle);
    auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_output(h, w, cv_color_format);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color;
    if (DISPLAY)
        cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    printf("Remaining images %lu \n", rocalGetRemainingImages(handle));
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    int i = 0;
    while (i++ < 1000) {
        if (rocalRun(handle) != 0) {
            std::cout << "rocalRun Failed with runtime error" << std::endl;
            rocalRelease(handle);
            return -1;
        }

        auto last_colot_temp = rocalGetIntValue(color_temp_adj);
        rocalUpdateIntParameter(last_colot_temp + 1, color_temp_adj);

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
    mat_input.release();
    mat_output.release();

    return 0;
}
