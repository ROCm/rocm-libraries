/*
MIT License

Copyright (c) 2020 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include <string.h>
#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
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

using namespace std::chrono;

bool IsPathExist(const char *s) {
    struct stat buffer;
    return (stat(s, &buffer) == 0);
}

int check_extension(std::string file_name) {
    // store the position of last '.' in the file name
    int position = file_name.find_last_of(".");
    // store the characters after the '.' from the file_name string
    std::string result = file_name.substr(position + 1);
    if ((result.compare("txt") == 0) || (result.size() == 0) || (result.compare("mp4") == 0))
        return -1;
    return 0;
}

int main(int argc, const char **argv) {
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if (argc < MIN_ARG_COUNT) {
        printf("Usage: video_tests <video_file/video_dataset_folder/text file> <reader_case> <processing_device=1/cpu=0> <hardware_decode_mode=0/1> <batch_size> <sequence_length> <frame_step> <frame_stride> <gray_scale/rgb> <display_on_off> <shuffle:0/1> <resize_width> <resize_height> <filelist_framenum:0/1> <enable_meta_data:0/1> <enable_framenumber:0/1> <enable_timestamps:0/1> <enable_sequence_rearrange:0/1>\n");
        return -1;
    }

    int argIdx = 0;
    const char *source_path = argv[++argIdx];
    int reader_case = 0;
    bool save_frames = 1;  // Saves the frames
    int rgb = 1;           // process color images
    unsigned resize_width = 0;
    unsigned resize_height = 0;
    bool processing_device = 1;
    size_t shard_count = 1;
    bool shuffle = false;
    unsigned input_batch_size = 1;
    unsigned sequence_length = 1;
    unsigned ouput_frames_per_sequence = 1;
    unsigned frame_step = 1;
    unsigned frame_stride = 1;
    bool file_list_frame_num = true;
    bool enable_metadata = false;
    bool enable_framenumbers = false;
    bool enable_timestamps = true;
    bool enable_sequence_rearrange = false;
    bool is_output = true;
    unsigned decoder_type = 0;
    if (argc >= argIdx + MIN_ARG_COUNT)
        reader_case = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        processing_device = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        decoder_type = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        input_batch_size = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        ouput_frames_per_sequence = sequence_length = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        frame_step = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        frame_stride = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        save_frames = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        shuffle = atoi(argv[++argIdx]) ? true : false;
    if (argc >= argIdx + MIN_ARG_COUNT)
        resize_width = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        resize_height = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        file_list_frame_num = atoi(argv[++argIdx]) ? true : false;
    if (argc >= argIdx + MIN_ARG_COUNT)
        enable_metadata = atoi(argv[++argIdx]) ? true : false;
    if (argc >= argIdx + MIN_ARG_COUNT)
        enable_framenumbers = atoi(argv[++argIdx]) ? true : false;
    if (argc >= argIdx + MIN_ARG_COUNT)
        enable_timestamps = atoi(argv[++argIdx]) ? true : false;
    if (argc >= argIdx + MIN_ARG_COUNT)
        enable_sequence_rearrange = atoi(argv[++argIdx]) ? true : false;

    auto decoder_mode = ((decoder_type > 0) ? RocalDecodeDevice::ROCAL_HW_DECODE : RocalDecodeDevice::ROCAL_SW_DECODE);
    RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_VIDEO_FFMPEG_SW;
    if (decoder_type == 1) {
        rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_VIDEO_FFMPEG_HW;
    } else if (decoder_type == 2) {
        rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_VIDEO_ROCDECODE;
        if (processing_device != 1) {
            std::cerr << "Setting the processing device to GPU for rocDecode decoder\n";
            processing_device = 1;
        }
    }

    if (!IsPathExist(source_path)) {
        std::cout << "\nThe folder/file path does not exist\n";
        return -1;
    }
    if (enable_sequence_rearrange) {
        is_output = false;
    }
    std::cerr << "Batch size : " << input_batch_size << std::endl;
    std::cerr << "Sequence length : " << sequence_length << std::endl;
    std::cerr << "Frame step : " << frame_step << std::endl;
    std::cerr << "Frame stride : " << frame_stride << std::endl;
    if (reader_case == 2) {
        std::cerr << "Resize Width : " << resize_width << std::endl;
        std::cerr << "Resize height : " << resize_height << std::endl;
    }

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24 : RocalImageColor::ROCAL_COLOR_U8;
    RocalContext handle;
    handle = rocalCreate(input_batch_size, processing_device ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0, 1);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }
    if (reader_case == 3) {
        if (check_extension(source_path) < 0) {
            std::cerr << "\n[ERR]   Text file/ Video File passed as input to SEQUENCE READER\n";
            return -1;
        }
        if (enable_metadata) {
            std::cout << "METADATA cannot be enabled for SEQUENCE READER";
            enable_metadata = false;
        }
        if (enable_framenumbers)
            enable_framenumbers = false;
        if (enable_timestamps)
            enable_timestamps = false;
    } else if (enable_metadata) {
        std::cout << "\nMETA DATA READER\n";
        rocalCreateVideoLabelReader(handle, source_path, sequence_length, frame_step, frame_stride, file_list_frame_num);
    }

    RocalTensor input1;
    switch (reader_case) {
        default: {
            std::cout << "\nVIDEO READER\n";
            input1 = rocalVideoFileSource(handle, source_path, color_format, decoder_mode, shard_count, sequence_length, shuffle, is_output, false, rocal_decoder_type, frame_step, frame_stride, file_list_frame_num);
            break;
        }
        case 2: {
            std::cout << "\nVIDEO READER RESIZE\n";
            if (resize_width == 0 || resize_height == 0) {
                std::cerr << "\n[ERR]Resize width and height are passed as NULL values\n";
                return -1;
            }
            input1 = rocalVideoFileResize(handle, source_path, color_format, decoder_mode, shard_count, sequence_length, resize_width, resize_height, shuffle, is_output, false, rocal_decoder_type, frame_step, frame_stride, file_list_frame_num);
            break;
        }
        case 3: {
            std::cout << "\nSEQUENCE READER\n";
            enable_framenumbers = enable_timestamps = 0;
            input1 = rocalSequenceReader(handle, source_path, color_format, shard_count, sequence_length, is_output, shuffle, false, frame_step, frame_stride);
            break;
        }
    }
    if (enable_sequence_rearrange) {
        std::cout << "\nENABLE SEQUENCE REARRANGE\n";
        std::vector<unsigned> new_order = {0, 0, 1, 1, 0};  // The integers in new order should range only from 0 to sequence_length - 1
        ouput_frames_per_sequence = new_order.size();
        input1 = rocalSequenceRearrange(handle, input1, new_order, true);
    }
    RocalIntParam color_temp_adj = rocalCreateIntParameter(0);

    // Calling the API to verify and build the augmentation graph
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cerr << "Error while adding the augmentation nodes " << std::endl;
        auto err_msg = rocalGetErrorMessage(handle);
        std::cout << err_msg << std::endl;
    }

    // Calling the API to verify and build the augmentation graph
    if (rocalVerify(handle) != ROCAL_OK) {
        std::cerr << "[ERR]Could not verify the augmentation graph" << std::endl;
        return -1;
    }
    std::cout << "\nRemaining images " << rocalGetRemainingImages(handle) << std::endl;
    std::cout << "Augmented copies count " << rocalGetAugmentationBranchCount(handle) << std::endl;

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    if (save_frames)
        mkdir("output_frames", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);  // Create directory in which images will be stored
    int h = rocalGetAugmentationBranchCount(handle) * rocalGetOutputHeight(handle) * input_batch_size * ouput_frames_per_sequence;
    int w = rocalGetOutputWidth(handle);
    int p = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? 3 : 1);
    int single_image_height = h / (input_batch_size * ouput_frames_per_sequence);
    std::cout << "output width " << w << " output height " << h << " color planes " << p << std::endl;
    auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color, mat_output;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int counter = 0;
    int color_temp_increment = 1;
    int count = 0;
    while (!rocalIsEmpty(handle)) {
        count++;
        if (rocalRun(handle) != 0) {
            std::cout << "rocalRun Failed with runtime error" << std::endl;
            rocalRelease(handle);
            return -1;
        }

        if (rocalGetIntValue(color_temp_adj) <= -99 || rocalGetIntValue(color_temp_adj) >= 99)
            color_temp_increment *= -1;

        rocalUpdateIntParameter(rocalGetIntValue(color_temp_adj) + color_temp_increment, color_temp_adj);
        rocalCopyToOutput(handle, mat_input.data, h * w * p);
        counter += input_batch_size;
        if (save_frames) {
            std::string batch_path = "output_frames/" + std::to_string(count);
            int status = mkdir(batch_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            if (status) continue;
            for (unsigned b = 0; b < input_batch_size; b++)  // Iterates over each sequence in the batch
            {
                std::string seq_path = batch_path + "/seq_" + std::to_string(b);
                std::string save_video_path = seq_path + "_output_video.avi";

                int frame_width = static_cast<int>(w);                     // get the width of frames of the video
                int frame_height = static_cast<int>(single_image_height);  // get the height of frames of the video
                Size frame_size(frame_width, frame_height);
                int frames_per_second = 10;

                // Create and initialize the VideoWriter object
                VideoWriter video_writer(save_video_path, VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                         frames_per_second, frame_size, true);
                // If the VideoWriter object is not initialized successfully, exit the program
                if (video_writer.isOpened() == false) {
                    std::cout << "Cannot save the video to a file" << std::endl;
                    return -1;
                }

                for (unsigned i = 0; i < ouput_frames_per_sequence; i++)  // Iterates over the frames in each sequence
                {
                    std::string save_image_path = seq_path + "_output_" + std::to_string(i) + ".png";
                    mat_output = mat_input(cv::Rect(0, ((b * single_image_height * ouput_frames_per_sequence) + (i * single_image_height)), w, single_image_height));
                    if (color_format == RocalImageColor::ROCAL_COLOR_RGB24) {
                        cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
                        cv::imwrite(save_image_path, mat_color);
                        video_writer.write(mat_color);
                    } else {
                        cv::imwrite(save_image_path, mat_output);
                        video_writer.write(mat_output);
                    }
                }
                video_writer.release();
            }
        }
        if (enable_metadata) {
            std::vector<int> image_name_length(input_batch_size);
            RocalTensorList labels = rocalGetImageLabels(handle);
            int img_size = rocalGetImageNameLen(handle, image_name_length.data());
            std::vector<char> img_name(img_size);
            rocalGetImageName(handle, img_name.data());

            std::cout << "\nImage names: " << img_name.data() << "\n";
            std::cout << "Label id: ";
            int *label_id = reinterpret_cast<int *>(labels->at(0)->buffer());
            for (unsigned i = 0; i < input_batch_size; i++) {
                std::cout << label_id[i] << "\t";
            }
            std::cout << std::endl;
        }
        if (enable_framenumbers || enable_timestamps) {
            std::vector<unsigned int> start_frame_num(input_batch_size);
            std::vector<float> frame_timestamps(input_batch_size * sequence_length);
            rocalGetSequenceStartFrameNumber(handle, start_frame_num.data());
            if (enable_timestamps) {
                rocalGetSequenceFrameTimestamps(handle, frame_timestamps.data());
            }
            for (unsigned i = 0; i < input_batch_size; i++) {
                if (enable_framenumbers)
                    std::cout << "\nFrame number : " << start_frame_num[i] << std::endl;
                if (enable_timestamps)
                    for (unsigned j = 0; j < sequence_length; j++)
                        std::cout << "T" << j << " : " << frame_timestamps[(i * sequence_length) + j] << "\t";
                std::cout << "\n";
            }
        }
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "Load     time " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cout << "Processed " << counter << " images/frames" << std::endl << "Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    rocalRelease(handle);
    mat_input.release();
    return 0;
}
