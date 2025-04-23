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

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;

#if USE_OPENCV_4
#define CV_FONT_HERSHEY_DUPLEX FONT_HERSHEY_DUPLEX
#define CV_WINDOW_AUTOSIZE WINDOW_AUTOSIZE
#define CV_RGB2BGR cv::COLOR_BGR2RGB
#else
#include <opencv/highgui.h>
#endif

#include "rocal_api.h"

#define DISPLAY
using namespace std::chrono;

int main(int argc, const char** argv) {
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if (argc < MIN_ARG_COUNT) {
        printf(
            "Usage: image_augmentation <image_dataset_folder/video_file> <processing_device=1/cpu=0>  \
              decode_width decode_height decoder_mode gray_scale/rgb display_on_off decode_shard_count  <shuffle:0/1> <jpeg_dec_mode<0(tjpeg)/1(opencv)/2(hwdec)>\n");
        return -1;
    }
    int argIdx = 1;
    const char* folderPath1 = argv[argIdx++];
    int decoder_mode = 0;  // 0 means no video decode, 1 means hardware, 2 means software decoding
    bool display = 1;    // Display the images
    int aug_depth = 1;   // how deep is the augmentation tree
    int rgb = 1;         // process color images
    int decode_width = 0;
    int decode_height = 0;
    bool processing_device = 1;
    size_t shard_count = 2;
    int shuffle = 0;
    int decoder_type = 0;
    const char *outName = "image_augmentation_app";

    if (argc > argIdx)
        processing_device = atoi(argv[argIdx++]);

    if (argc > argIdx)
        decode_width = atoi(argv[argIdx++]);

    if (argc > argIdx)
        decode_height = atoi(argv[argIdx++]);

    if (argc > argIdx)
        decoder_mode = atoi(argv[argIdx++]);

    if (argc > argIdx)
        rgb = atoi(argv[argIdx++]);

    if (argc > argIdx)
        display = atoi(argv[argIdx++]);

    if (argc > argIdx)
        shard_count = atoi(argv[argIdx++]);

    if (argc > argIdx)
        shuffle = atoi(argv[argIdx++]);

    if (argc > argIdx)
        decoder_type = atoi(argv[argIdx++]);

    if (argc > argIdx)
        outName = argv[argIdx++];

    int inputBatchSize = 4;

    std::cout << ">>> Running on " << (processing_device ? "GPU" : "CPU") << std::endl;

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24 : RocalImageColor::ROCAL_COLOR_U8;

    auto handle = rocalCreate(inputBatchSize, processing_device ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0, 1);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the rocAL contex\n";
        return -1;
    }

    RocalDecoderType dec_type = (RocalDecoderType)decoder_type;

    /*>>>>>>>>>>>>>>>> Creating rocAL parameters  <<<<<<<<<<<<<<<<*/

    // Creating uniformly distributed random objects to override some of the default augmentation parameters
    RocalFloatParam rand_crop_area = rocalCreateFloatUniformRand(0.3, 0.5);
    RocalIntParam color_temp_adj = rocalCreateIntParameter(0);

    // Creating a custom random object to set a limited number of values to randomize the rotation angle
    const size_t num_values = 3;
    float values[num_values] = {0, 10, 135};
    double frequencies[num_values] = {1, 5, 5};

    RocalFloatParam rand_angle = rocalCreateFloatRand(values, frequencies, num_values);

    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    RocalTensor input1;

    if (decoder_mode >= 2) {
        unsigned sequence_length = 1;
        unsigned frame_step = 1;
        unsigned frame_stride = 1;
        if (decode_height <= 0 || decode_width <= 0) {
            std::cout << "Output width and height is needed for video decode\n";
            return -1;
        }
        input1 = rocalVideoFileSource(handle, folderPath1, color_format, (dec_type == 2) ? ROCAL_HW_DECODE : ROCAL_SW_DECODE, shard_count, sequence_length, shuffle, 
                                      true, false, (dec_type == 2) ? ROCAL_DECODER_VIDEO_FFMPEG_HW : ROCAL_DECODER_VIDEO_FFMPEG_SW, frame_step, frame_stride);
    } else if (decoder_mode == 1) {
            std::vector<float> area = {0.08, 1};
            std::vector<float> aspect_ratio = {3.0f / 4, 4.0f / 3};
            input1 = rocalFusedJpegCrop(handle, folderPath1, color_format, shard_count, false, area, aspect_ratio, 10, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_width, decode_height);

    } else {
        // The jpeg file loader can automatically select the best size to decode all images to that size
        // User can alternatively set the size or change the policy that is used to automatically find the size
        if (dec_type == RocalDecoderType::ROCAL_DECODER_OPENCV) std::cout << "Using OpenCV decoder for Jpeg Source\n";
        if (decode_height <= 0 || decode_width <= 0)
            input1 = rocalJpegFileSource(handle, folderPath1, color_format, shard_count, false, shuffle, false);
        else
            input1 = rocalJpegFileSource(handle, folderPath1, color_format, shard_count, false, shuffle, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_width, decode_height, dec_type);
    }

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "JPEG source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    RocalTensor tensor0;
    int resize_w = 112, resize_h = 112;
    if (decoder_mode >= 2) {
        resize_h = decode_height;
        resize_w = decode_width;
        tensor0 = input1;
    } else {
        tensor0 = rocalResize(handle, input1, resize_w, resize_h, true);
    }
    RocalTensor tensor1 = rocalRain(handle, tensor0, false);

    RocalTensor tensor11 = rocalFishEye(handle, tensor1, false);

    rocalRotate(handle, tensor11, true, rand_angle);

    // Creating successive blur nodes to simulate a deep branch of augmentations
    RocalTensor tensor2 = rocalCropResize(handle, tensor0, resize_w, resize_h, false, rand_crop_area);
    for (int i = 0; i < aug_depth; i++) {
        tensor2 = rocalBlurFixed(handle, tensor2, 17, (i == (aug_depth - 1)) ? true : false);
    }
    // Commenting few augmentations out until tensor support is added in rpp
    // RocalTensor tensor4 = rocalColorTemp(handle, tensor0, true, color_temp_adj);

    // RocalTensor tensor6 = rocalJitter(handle, tensor5, true);

    // rocalVignette(handle, tensor5, true);

    // RocalTensor tensor7 = rocalPixelate(handle, tensor0, true);

    RocalTensor tensor8 = rocalSnow(handle, tensor0, true);

    RocalTensor tensor9 = rocalBlend(handle, tensor0, tensor8, true);

    RocalTensor tensor10 = rocalLensCorrection(handle, tensor9, true);

    rocalExposure(handle, tensor10, true);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Error while adding the augmentation nodes " << std::endl;
        auto err_msg = rocalGetErrorMessage(handle);
        std::cout << err_msg << std::endl;
    }
    // Calling the API to verify and build the augmentation graph
    if (rocalVerify(handle) != ROCAL_OK) {
        std::cout << "Could not verify the augmentation graph" << std::endl;
        return -1;
    }

    std::cout << "Remaining images " << rocalGetRemainingImages(handle) << std::endl;
    std::cout << "Augmented copies count " << rocalGetAugmentationBranchCount(handle) << std::endl;

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    // initializations for logos and heading
    cv::Mat AMD_Epyc_Black_resize, AMD_ROCm_Black_resize;
    AMD_Epyc_Black_resize = cv::imread("../../../../docs/data/amd-epyc-black-resize.png");
    AMD_ROCm_Black_resize = cv::imread("../../../../docs/data/rocm-black-resize.png");
    int fontFace = CV_FONT_HERSHEY_DUPLEX;
    int thickness = 1;
    std::string bufferName = "rocAL Image Augmentation";

    int h = rocalGetAugmentationBranchCount(handle) * rocalGetOutputHeight(handle) * inputBatchSize;
    int w = rocalGetOutputWidth(handle);
    int p = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? 3 : 1);
    std::cout << "output width " << w << " output height " << h << " color planes " << p << std::endl;
    const unsigned number_of_cols = (decoder_mode >= 2) ? 1 : 10;
    auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_output(h + AMD_ROCm_Black_resize.rows, w * number_of_cols, cv_color_format);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color;
    int col_counter = 0;

    // adding heading to output display
    cv::Rect roi = Rect(0, 0, w * number_of_cols, AMD_Epyc_Black_resize.rows);
    mat_output(roi).setTo(cv::Scalar(128, 128, 128));
    putText(mat_output, bufferName, Point(250, 70), fontFace, 1.2, cv::Scalar(66, 13, 9), thickness, 5);

    // adding logos to output display
    cv::Mat mat_output_ROI = mat_output(cv::Rect(w * number_of_cols - AMD_Epyc_Black_resize.cols, 0, AMD_Epyc_Black_resize.cols, AMD_Epyc_Black_resize.rows));
    cv::Mat mat_output_ROI_1 = mat_output(cv::Rect(0, 0, AMD_ROCm_Black_resize.cols, AMD_ROCm_Black_resize.rows));
    AMD_Epyc_Black_resize.copyTo(mat_output_ROI);
    AMD_ROCm_Black_resize.copyTo(mat_output_ROI_1);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int counter = 0;
    int color_temp_increment = 1;
    while (!rocalIsEmpty(handle)) {
        if (rocalRun(handle) != 0) {
            std::cout << "rocalRun Failed with runtime error" << std::endl;
            rocalRelease(handle);
            return -1;
        }

        if (rocalGetIntValue(color_temp_adj) <= -99 || rocalGetIntValue(color_temp_adj) >= 99)
            color_temp_increment *= -1;

        rocalUpdateIntParameter(rocalGetIntValue(color_temp_adj) + color_temp_increment, color_temp_adj);
        auto ouput_tensor_list = rocalGetOutputTensors(handle);
        unsigned char* output = mat_input.data;
        for (uint i = 0; i < ouput_tensor_list->size(); i++) {
            ouput_tensor_list->at(i)->copy_data(output);
            output += ouput_tensor_list->at(i)->data_size();
        }
        counter += inputBatchSize;
        if (!display)
            continue;

        std::string out_filename = std::string(outName) + ".png"; 
        mat_input.copyTo(mat_output(cv::Rect(col_counter * w, AMD_ROCm_Black_resize.rows, w, h)));
        if (color_format == RocalImageColor::ROCAL_COLOR_RGB24) {
            cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
            cv::imwrite(out_filename, mat_color);
        } else {
            cv::imwrite(out_filename, mat_output);
        }
        cv::waitKey(1);
        col_counter = (col_counter + 1) % number_of_cols;
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "Load     time " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cout << ">>>>> " << counter << " images/frames Processed. Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    rocalRelease(handle);
    mat_input.release();
    mat_output.release();
    return 0;
}
