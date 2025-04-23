/*
Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "vx_amd_migraphx.h"
#include <cstring>
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <sys/stat.h>
#define MAX_STRING_LENGTH 100

using namespace std;

#define ERROR_CHECK_STATUS(status) { \
    vx_status status_ = (status); \
    if (status_ != VX_SUCCESS) { \
        printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
        exit(1); \
    } \
}

#define ERROR_CHECK_OBJECT(obj) { \
    vx_status status_ = vxGetStatus((vx_reference)(obj)); \
    if(status_ != VX_SUCCESS) { \
        printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
        exit(1); \
    } \
}

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    size_t len = strnlen(string, MAX_STRING_LENGTH);
    if (len > 0) {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}

inline int64_t clockCounter()
{
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline int64_t clockFrequency()
{
    return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
}

void read_input_digit(std::vector<int> n, std::vector<float>& input_digit) {
    std::ifstream file("../digits.txt");
    const int Digits = 10;
    const int Height = 28;
    const int Width  = 28;
    if(!file.is_open()) {
        return;
    }
    for(int d = 0; d < Digits; ++d) {
        for(int i = 0; i < Height * Width; ++i) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            for(int j = 0; j < n.size(); j++) {
                if(d == n[j]) {
                    float data = temp / 255.0;
                    input_digit.push_back(data);
                }
            }
        }
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    
    if(argc < 4) {
        std::cout << "Usage: \n ./migraphx_node_test <path-to-mnist ONNX model> --batch-size n" << std::endl;
        return -1;
    }
    
    std::string modelFileName = argv[1];
    vx_size batch_size = stoul(argv[3]);

    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);
    vxRegisterLogCallback(context, log_callback, vx_false_e);

    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph);

    // initialize variables
    vx_tensor input_tensor, output_tensor;
    vx_size input_num_of_dims = 4;
    vx_size input_dims[4] = {batch_size, 1, 28, 28}; //input dimensions for the mnist model
    vx_size output_num_of_dims = 2;
    vx_size output_dims[2] = {batch_size, 10};  //output dimensions for the mnist model
    vx_size stride[4];
    vx_map_id map_id;
    void *ptr = nullptr;
    vx_status status = 0;

    //create a reults folder
    std::ofstream outputFile;
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream datetime;
    datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%X");
    std::string date = "../results-" + datetime.str();
    if (mkdir(date.c_str(), 0777) == -1)
        cerr << "Error, cannot create results folder:  " << strerror(errno) << endl;

    input_tensor = vxCreateTensor(context, input_num_of_dims, input_dims, VX_TYPE_FLOAT32, 0);
    output_tensor = vxCreateTensor(context, output_num_of_dims, output_dims, VX_TYPE_FLOAT32, 0);

    std::vector<float> input_digit;
    std::random_device rd;
    std::uniform_int_distribution<int> dist(0, 9);
    std::vector<int> rand_digit(batch_size);
    for (int i = 0; i < batch_size; i++) {
        rand_digit[i] = dist(rd);
    } 
    read_input_digit(rand_digit, input_digit);
    
    ERROR_CHECK_STATUS(vxMapTensorPatch(input_tensor, input_num_of_dims, nullptr, nullptr, &map_id, stride,
        (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    if (status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed " << std::endl;
        return status;
    }

    memcpy(ptr, static_cast<void*>(input_digit.data()), input_digit.size() * sizeof(float));

    status = vxUnmapTensorPatch(input_tensor, map_id);
    if (status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for " << status << ")" << std::endl;
        return status;
    }

    ERROR_CHECK_STATUS(vxLoadKernels(context, "vx_amd_migraphx"));

    vx_node node = amdMIGraphXnode(graph, modelFileName.c_str(), input_tensor, output_tensor);
    ERROR_CHECK_OBJECT(node);

    ERROR_CHECK_STATUS(vxVerifyGraph(graph));
    
    // graph process timing
    int64_t freq = clockFrequency(), t0, t1;
    t0 = clockCounter();
    ERROR_CHECK_STATUS(vxProcessGraph(graph));
    t1 = clockCounter();
    printf("OK: vxProcessGraph() took %.3f msec (1st iteration)\n", (float)(t1-t0)*1000.0f/(float)freq);

    t0 = clockCounter();
    int N = 100;
    for(int i = 0; i < N; i++) {
        status = vxProcessGraph(graph);
        if(status != VX_SUCCESS)
            break;
    }
    t1 = clockCounter();
    printf("OK: vxProcessGraph() took %.3f msec (average over %d iterations)\n", (float)(t1-t0)*1000.0f/(float)freq/(float)N, N);

    status = vxMapTensorPatch(output_tensor, output_num_of_dims, nullptr, nullptr, &map_id, stride,
        (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if (status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for output tensor" << std::endl;
        return status;
    }

    //copy results into file
    outputFile.open(date + "/mnist-output-results.csv");
    outputFile << "iteration, Randomly chosen digit, Result from inference, result\n";

    float *output_buf = (float*)ptr;
    auto num_results = 10;
    for(int i = 0; i < batch_size; i++, output_buf += num_results) {
        float* max = std::max_element(output_buf, output_buf + num_results);
        int answer = max - output_buf;
        outputFile << i + 1 << "," << rand_digit[i] << "," << answer << "," << (answer == rand_digit[i] ? "Correct":"Incorrect") << "\n";
    }

    status = vxUnmapTensorPatch(output_tensor, map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for output_tensor" << std::endl;
        return status;
    }

    // release resources
    ERROR_CHECK_STATUS(vxReleaseNode(&node));
    ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
    ERROR_CHECK_STATUS(vxReleaseTensor(&input_tensor));
    ERROR_CHECK_STATUS(vxReleaseTensor(&output_tensor));
    ERROR_CHECK_STATUS(vxReleaseContext(&context));

    printf("OK: successful\n");

    return 0;
}