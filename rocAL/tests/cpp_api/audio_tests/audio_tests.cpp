/*
MIT License

Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

#include "rocal_api.h"

using namespace std::chrono;

int verify_non_silent_region_output(int *nsr_begin, int *nsr_length, std::string case_name, std::string rocal_data_path) {
    int status = -1;
    // read data from golden outputs
    std::string ref_file_path = rocal_data_path + "rocal_data/GoldenOutputsTensor/reference_outputs_audio/" + case_name + "_output.bin";
    std::ifstream fin(ref_file_path, std::ios::binary);  // Open the binary file for reading

    if (!fin.is_open()) {
        std::cout << "Error: Unable to open the input binary file\n";
        return -1;
    }

    // Get the size of the file
    fin.seekg(0, std::ios::end);
    std::streampos fileSize = fin.tellg();
    fin.seekg(0, std::ios::beg);

    std::size_t numFloats = fileSize / sizeof(int);

    std::vector<int> ref_output(numFloats);

    // Read the floats from the file
    fin.read(reinterpret_cast<char *>(ref_output.data()), fileSize);

    if (fin.fail()) {
        std::cout << "Error: Failed to read from the input binary file\n";
        return -1;
    }

    fin.close();

    if ((nsr_begin[0] == ref_output[0]) && (nsr_length[0] == ref_output[1]))
        status = 0;

    return status;
}

int verify_output(float *dst_ptr, long int frames, long int channels, std::string case_name, int max_samples, int max_channels, int buffer_size, std::string rocal_data_path) {
    int status = -1;
    // read data from golden outputs
    std::string ref_file_path = rocal_data_path + "rocal_data/GoldenOutputsTensor/reference_outputs_audio/" + case_name + "_output.bin";
    std::ifstream fin(ref_file_path, std::ios::binary);  // Open the binary file for reading

    if (!fin.is_open()) {
        std::cout << "Error: Unable to open the input binary file\n";
        return -1;
    }

    // Get the size of the file
    fin.seekg(0, std::ios::end);
    std::streampos fileSize = fin.tellg();
    fin.seekg(0, std::ios::beg);

    std::size_t numFloats = fileSize / sizeof(float);

    std::vector<float> ref_output(numFloats);

    // Read the floats from the file
    fin.read(reinterpret_cast<char *>(ref_output.data()), fileSize);

    if (fin.fail()) {
        std::cout << "Error: Failed to read from the input binary file\n";
        return -1;
    }

    fin.close();

    auto atol = (case_name != "normalize") ? 1e-20 : 1e-5;  // Absolute tolerance
    int matched_indices = 0;
    for (int i = 0; i < frames; i++) {
        for (int j = 0; j < channels; j++) {
            float ref_val, out_val;
            ref_val = ref_output[i * channels + j];
            out_val = dst_ptr[i * max_channels + j];
            bool invalid_comparison = ((out_val == 0.0f) && (ref_val != 0.0f));
            if (!invalid_comparison && std::abs(out_val - ref_val) < atol)
                matched_indices += 1;
        }
    }

    std::cout << std::endl << "Results for Test case: " << std::endl;
    if ((matched_indices == buffer_size) && matched_indices != 0) {
        status = 0;
    }

    return status;
}

int test(int test_case, const char *path, int qa_mode, int downmix, int gpu);
int main(int argc, const char **argv) {
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if (argc < MIN_ARG_COUNT) {
        printf("Usage: ./audio_tests <audio-dataset-folder - required> <test_case> <downmix=0/1> <device-gpu=1/cpu=0> <qa_mode=0/1>\n");
        return -1;
    }

    int argIdx = 1;
    const char *path = argv[argIdx++];
    int qa_mode = 0;
    unsigned test_case = 0;
    bool downmix = false;
    bool gpu = 0;

    if (argc > argIdx)
        test_case = atoi(argv[argIdx++]);

    if (argc > argIdx)
        downmix = atoi(argv[argIdx++]);

    if (argc > argIdx)
        gpu = atoi(argv[argIdx++]);

    if (argc > argIdx)
        qa_mode = atoi(argv[argIdx++]);

    if (gpu) {  // TODO - Will be removed when GPU support is added for Audio pipeline
        std::cout << "WRN : Currently Audio unit test supports only HOST backend\n";
        gpu = false;
    }

    int return_val = test(test_case, path, qa_mode, downmix, gpu);
    return return_val;
}

int test(int test_case, const char *path, int qa_mode, int downmix, int gpu) {
    int input_batch_size = 1;
    bool is_output_audio_decoder = false;
    std::cout << "test case " << test_case << std::endl;
    std::cout << "Running on " << (gpu ? "GPU" : "CPU") << std::endl;

    auto handle = rocalCreate(input_batch_size,
                              gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0,
                              1);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }

    std::string file_list_path;  // User can modify this with the file list path if required
    std::string rocal_data_path = std::getenv("ROCAL_DATA_PATH");
    if (qa_mode && test_case != 3) {  // setting the default file list path from ROCAL_DATA_PATH
        file_list_path = rocal_data_path + "rocal_data/audio/wav_file_list.txt";
    }

    std::cout << "Running LABEL READER" << std::endl;
    rocalCreateLabelReader(handle, path, file_list_path.c_str());

    is_output_audio_decoder = (test_case == 0 || test_case == 3) ? true : false;
    RocalTensor decoded_output = rocalAudioFileSourceSingleShard(handle, path, file_list_path.c_str(), 0, 1, is_output_audio_decoder, false, false, downmix);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Audio source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    std::string case_name;
    switch (test_case) {
        case 0: {
            case_name = "audio_decoder";
            std::cout << "Running AUDIO DECODER" << std::endl;
        } break;
        case 1: {
            std::cout << "Running PREEMPHASIS" << std::endl;
            case_name = "preemphasis_filter";
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            RocalAudioBorderType preemph_border_type = RocalAudioBorderType::ROCAL_CLAMP;
            RocalFloatParam p_preemph_coeff = rocalCreateFloatParameter(0.97);
            rocalPreEmphasisFilter(handle, decoded_output, true, p_preemph_coeff, preemph_border_type, tensorOutputType);

        } break;
        case 2: {
            std::cout << "Running SPECTROGRAM" << std::endl;
            case_name = "spectrogram";
            std::vector<float> window_fn;
            rocalSpectrogram(handle, decoded_output, true, window_fn, true, true, 2, 512, 320, 160, ROCAL_NFT, ROCAL_FP32);

        } break;
        case 3: {
            case_name = "downmix";
            std::cout << "Running AUDIO DECODER + DOWNMIX" << std::endl;
        } break;
        case 4: {
            std::cout << "Running TO DECIBELS" << std::endl;
            case_name = "to_decibels";
            rocalToDecibels(handle, decoded_output, true, std::log(1e-20), std::log(10), 1.0f, ROCAL_FP32);
        } break;
        case 5: {
            std::cout << "Running RESAMPLE" << std::endl;
            case_name = "resample";
            float resample = 16000.00;
            std::vector<float> range = {1.15, 1.15};
            RocalTensor uniform_distribution_resample = rocalUniformDistribution(handle, decoded_output, false, range);
            RocalTensor resampled_rate = rocalTensorMulScalar(handle, uniform_distribution_resample, false, resample, ROCAL_FP32);
            rocalResample(handle, decoded_output, resampled_rate, true, 1.15 * 255840, 50.0, ROCAL_FP32);
        } break;
        case 6: {
            std::cout << "Running TENSOR ADD TENSOR" << std::endl;
            case_name = "tensor_add_tensor";
            std::vector<float> range = {1.15, 1.15};
            RocalTensor uniform_distribution_sample = rocalUniformDistribution(handle, decoded_output, false, range);
            rocalTensorAddTensor(handle, decoded_output, uniform_distribution_sample, true, ROCAL_FP32);
        } break;
        case 7: {
            std::cout << "Running TENSOR MUL SCALAR" << std::endl;
            case_name = "tensor_mul_scalar";
            rocalTensorMulScalar(handle, decoded_output, true, 1.15, ROCAL_FP32);
        } break;
        case 8: {
            std::cout << "Running NON SILENT REGION " << std::endl;
            case_name = "non_silent_region";
            rocalNonSilentRegionDetection(handle, decoded_output, true, -60, 0.0, 8192, 2048);
        } break;
        case 9: {
            std::cout << "Running SLICE " << std::endl;
            case_name = "slice";
            std::vector<float> fill_values = {0.0};
            std::vector<unsigned> axes = {0};
            auto nsr_output = rocalNonSilentRegionDetection(handle, decoded_output, false, -60.0, 0.0, 8192, 2048);
            rocalSlice(handle, decoded_output, true, nsr_output.anchor, nsr_output.shape, fill_values, ROCAL_ERROR, ROCAL_FP32);
        } break;
        case 10: {
            std::cout << "Running MEL FILTER BANK " << std::endl;
            case_name = "mel_filter_bank";
            std::vector<float> window_fn;
            RocalTensor spec_output = rocalSpectrogram(handle, decoded_output, false, window_fn, true, true, 2, 512, 320, 160, ROCAL_NFT, ROCAL_FP32);
            rocalMelFilterBank(handle, spec_output, true, 8000, 0.0, RocalMelScaleFormula::ROCAL_MELSCALE_SLANEY, 80, true, 16000, ROCAL_FP32);
        } break;
        case 11: {
            std::cout << "Running NORMALIZE " << std::endl;
            case_name = "normalize";
            std::vector<unsigned> axes = {1};
            std::vector<float> mean;
            std::vector<float> stddev;
            std::vector<float> window_fn;
            RocalTensor spec_output = rocalSpectrogram(handle, decoded_output, false, window_fn, true, true, 2, 512, 320, 160, ROCAL_NFT, ROCAL_FP32);
            rocalNormalize(handle, spec_output, axes, mean, stddev, true, 1, 0, ROCAL_FP32);
        } break;
        default: {
            std::cout << "Not a valid test case ! Exiting!\n";
            return -1;
        }
    }

    // Calling the API to verify and build the augmentation graph
    rocalVerify(handle);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    int iteration = 0;
    float *buffer = nullptr;
    int *nsr_begin = nullptr;
    int *nsr_length = nullptr;
    int frames = 0, channels = 0;
    int max_channels = 0;
    int max_samples = 0;
    int buffer_size = 0;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    while (rocalGetRemainingImages(handle) >= static_cast<size_t>(input_batch_size)) {
        std::cout << "\n Iteration:: " << iteration << "\n";
        iteration++;
        if (rocalRun(handle) != 0) {
            std::cout << "rocalRun Failed with runtime error" << std::endl;
            rocalRelease(handle);
            return -1;
        }
        RocalTensorList output_tensor_list = rocalGetOutputTensors(handle);
        std::vector<int> file_name_length(input_batch_size);
        int file_name_size = rocalGetImageNameLen(handle, file_name_length.data());
        std::vector<char> audio_file_name(file_name_size);
        std::vector<int> roi(4 * input_batch_size, 0);
        rocalGetImageName(handle, audio_file_name.data());
        RocalTensorList labels = rocalGetImageLabels(handle);
        int *label_id = reinterpret_cast<int *>(labels->at(0)->buffer());  // The labels are present contiguously in memory
        std::cout << "Audio file : " << audio_file_name.data() << "\n";
        std::cout << "Label : " << *label_id << "\n";
        if (test_case == 8) {  // Non silent region detection outputs
            nsr_begin = static_cast<int *>(output_tensor_list->at(0)->buffer());
            nsr_length = static_cast<int *>(output_tensor_list->at(1)->buffer());
        } else {
            for (uint idx = 0; idx < output_tensor_list->size(); idx++) {
                buffer = static_cast<float *>(output_tensor_list->at(idx)->buffer());
                output_tensor_list->at(idx)->copy_roi(roi.data());
                max_channels = output_tensor_list->at(idx)->dims().at(2);
                max_samples = max_channels == 1 ? 1 : output_tensor_list->at(idx)->dims().at(1);
                frames = roi[idx * 4 + 2];
                channels = roi[idx * 4 + 3];
                buffer_size = roi[idx * 4 + 2] * roi[idx * 4 + 3];
            }
        }
    }

    if (qa_mode) {
        std::cout << "\n *****************************Verifying Audio output**********************************\n";
        if (rocal_data_path.empty()) {
            std::cout << "\n ROCAL_DATA_PATH env variable has not been set. ";
            exit(0);
        }
        if (test_case != 8 && (verify_output(buffer, frames, channels, case_name, max_samples, max_channels, buffer_size, rocal_data_path) == 0)) {
            std::cout << "PASSED!\n\n";
        } else if (test_case == 8 && (verify_non_silent_region_output(nsr_begin, nsr_length, case_name, rocal_data_path) == 0)) {
            std::cout << "PASSED!\n\n";
        } else {
            std::cout << "FAILED!\n\n";
        }
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
