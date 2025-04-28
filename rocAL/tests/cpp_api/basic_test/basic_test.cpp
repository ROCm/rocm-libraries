/*
MIT License

Copyright (c) 2018 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include <vector>

#include "rocal_api.h"
#define TEST_2

#include "opencv2/opencv.hpp"
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
#define DISPLAY 0
int main(int argc, const char **argv) {
    // check command-line usage
    const int MIN_ARG_COUNT = 3;
    if (argc < MIN_ARG_COUNT) {
        std::cout << "Usage: basic_test <image_dataset_folder - required> <label_text_file_path - required> <test_case:0/1> <processing_device=1/cpu=0>  decode_width decode_height <gray_scale:0/rgb:1> decode_shard_counts decoder_type \n";
        return -1;
    }
    int argIdx = 1;
    const char *folderPath1 = argv[argIdx++];
    const char *label_text_file_path = argv[argIdx++];
    int rgb = 1;  // process color images
    int decode_width = 0;
    int decode_height = 0;
    int test_case = 0;
    bool processing_device = 0;
    size_t decode_shard_counts = 1;
    int decoder_type = 0;   // Set to default TurboJpeg decoder

    if (argc > argIdx)
        test_case = atoi(argv[argIdx++]);

    if (argc > argIdx)
        processing_device = atoi(argv[argIdx++]);

    if (argc > argIdx)
        decode_width = atoi(argv[argIdx++]);

    if (argc > argIdx)
        decode_height = atoi(argv[argIdx++]);

    if (argc > argIdx)
        rgb = atoi(argv[argIdx++]);

    if (argc > argIdx)
        decode_shard_counts = atoi(argv[argIdx++]);

    if (argc > argIdx)
        decoder_type = atoi(argv[argIdx++]);

    const int inputBatchSize = 4;

    // Set the rocAL decoder type
    RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_TJPEG;
    if (decoder_type == 1) {
        rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_OPENCV;
    } else if (decoder_type == 2) {
        rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_ROCJPEG;
        processing_device = 1;  // Requires GPU backend for rocJpeg decoder
    }

    std::cout << ">>> Running on " << (processing_device ? "GPU" : "CPU") << std::endl;

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24 : RocalImageColor::ROCAL_COLOR_U8;

    auto handle = rocalCreate(inputBatchSize, processing_device ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0, 1);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }

    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    RocalTensor decoded_output;

    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    if (decode_height <= 0 || decode_width <= 0)
        decoded_output = rocalJpegFileSource(handle, folderPath1, color_format, decode_shard_counts, false, false);
    else
        decoded_output = rocalJpegFileSource(handle, folderPath1, color_format, decode_shard_counts, false, false, false,
                                             ROCAL_USE_USER_GIVEN_SIZE, decode_width, decode_height, rocal_decoder_type);
    if (strcmp(label_text_file_path, "") == 0)
        rocalCreateLabelReader(handle, folderPath1);
    else
        rocalCreateTextFileBasedLabelReader(handle, label_text_file_path);
    rocalCropResizeFixed(handle, decoded_output, 224, 224, true, 0.9, 1.1, 0.1, 0.1);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "JPEG source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

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

    std::cout << "Augmented copies count " << rocalGetAugmentationBranchCount(handle) << std::endl;

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    int h = rocalGetAugmentationBranchCount(handle) * rocalGetOutputHeight(handle) * inputBatchSize;
    int w = rocalGetOutputWidth(handle);
    int p = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? 3 : 1);
    std::cout << "output width " << w << " output height " << h << " color planes " << p << std::endl;
    auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);

    const int total_tests = 2;
    int test_id = -1;
    int ImageNameLen[inputBatchSize];
    int run_len[] = {2 * inputBatchSize, 4 * inputBatchSize, 1 * inputBatchSize, 50 * inputBatchSize};

    std::vector<std::string> names;
    names.resize(inputBatchSize);

    while (++test_id < total_tests) {
        std::cout << "Start test id " << test_id << "\n";
        std::cout << "Available images = " << rocalGetRemainingImages(handle) << std::endl;
        int process_image_count = ((test_case == 0) ? rocalGetRemainingImages(handle) : run_len[test_id]);
        std::cout << "Process " << process_image_count << " images" << std::endl;
        if (DISPLAY)
            cv::waitKey(0);
        const unsigned number_of_cols = process_image_count / inputBatchSize;
        cv::Mat mat_output(h, w * number_of_cols, cv_color_format);
        cv::Mat mat_input(h, w, cv_color_format);
        cv::Mat mat_color;
        auto win_name = "output";
        if (DISPLAY)
            cv::namedWindow(win_name, CV_WINDOW_AUTOSIZE);

        int col_counter = 0;
        int counter = 0;

        while ((test_case == 0) ? !rocalIsEmpty(handle) : (counter < run_len[test_id])) {
            if (rocalRun(handle) != 0) {
                std::cout << "rocalRun Failed with runtime error" << std::endl;
                rocalRelease(handle);
                return -1;
            }

            rocalCopyToOutput(handle, mat_input.data, h * w * p);

            counter += inputBatchSize;
            RocalTensorList labels = rocalGetImageLabels(handle);

            unsigned imagename_size = rocalGetImageNameLen(handle, ImageNameLen);
            std::vector<char> imageNames(imagename_size);
            rocalGetImageName(handle, imageNames.data());
            std::string imageNamesStr(imageNames.data());

            int pos = 0;
            int *labels_buffer = reinterpret_cast<int *>(labels->at(0)->buffer());
            for (int i = 0; i < inputBatchSize; i++) {
                names[i] = imageNamesStr.substr(pos, ImageNameLen[i]);
                pos += ImageNameLen[i];
                std::cout << "name: " << names[i] << " label: " << labels_buffer[i] << std::endl;
            }
            std::cout << std::endl;

            mat_input.copyTo(mat_output(cv::Rect(col_counter * w, 0, w, h)));
            if (color_format == RocalImageColor::ROCAL_COLOR_RGB24) {
                cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
                if (DISPLAY)
                    cv::imshow(win_name, mat_output);
                else
                    cv::imwrite("output.png", mat_output);
            } else {
                if (DISPLAY)
                    cv::imshow(win_name, mat_output);
                else
                    cv::imwrite("output.png", mat_output);
            }
            // The delay here simulates possible latency between runs due to training
            if (DISPLAY)
                cv::waitKey(200);
            col_counter = (col_counter + 1) % number_of_cols;
        }
        std::cout << "Completed test id: " << test_id << " processed " << counter << " images\n";
        if (DISPLAY)
            cv::waitKey(0);
        std::cout << "rocAL reset\n";
        rocalResetLoaders(handle);
        mat_input.release();
        mat_output.release();
        mat_color.release();
        if (DISPLAY)
            cvDestroyWindow(win_name);
    }

    rocalRelease(handle);

    return 0;
}
