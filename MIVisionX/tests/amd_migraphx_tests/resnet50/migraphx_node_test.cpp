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
#include <algorithm>
#include <dirent.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#define MAX_STRING_LENGTH 100

using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;

#if USE_OPENCV_4
#define CV_LOAD_IMAGE_COLOR IMREAD_COLOR
#endif

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

int main(int argc, char **argv) {

    if(argc < 6) {
        std::cout << "Usage: \n ./migraphx_node_test <path-to-resnet50 ONNX model> --image/--tensor <path to image directory/tensor file> --batch-size n" << std::endl;
        return -1;
    }
    
    std::string modelFileName = argv[1];
    std::string inputFileName = argv[3];
    vx_size batch_size = stoul(argv[5]);

    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);
    vxRegisterLogCallback(context, log_callback, vx_false_e);

    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph);

    // initialize variables
    vx_tensor input_tensor, output_tensor;
    vx_size input_num_of_dims = 4;
    vx_size input_dims[4] = {batch_size, 3, 224, 224}; //input dimensions for the resnet50 model
    vx_size output_num_of_dims = 2;
    vx_size output_dims[2] = {batch_size, 1000}; //output dimensions for the resnet50 model
    vx_size stride[4];
    vx_map_id map_id;
    void *ptr = nullptr;
    vx_status status = 0;
    float mean_vec[3] = {0.485, 0.456, 0.406};
    float stddev_vec[3] = {0.229, 0.224, 0.225};
    std::vector<float> mulVec = {1 / (255 * stddev_vec[0]), 1 / (255 * stddev_vec[1]), 1 / (255 * stddev_vec[2])}, 
                        addVec = {(mean_vec[0] / stddev_vec[0]), (mean_vec[1] / stddev_vec[1]), (mean_vec[2] / stddev_vec[2])};

    //create a reults folder
    std::ofstream outputFile;
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream datetime;
    datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%X");
    std::string date = "../results-" + datetime.str();
    if (mkdir(date.c_str(), 0777) == -1)
        cerr << "Error, cannot create results folder:  " << strerror(errno) << endl;

    //imagenet label file
    std::string labelText[1000];
    std::string labelFileName = ("../labels.txt");

    std::string line;
    std::ifstream out(labelFileName);
    if(!out) {
      std::cout << "label file failed to open" << std::endl;
      return -1; 
    }
    int lineNum = 0;
    while(getline(out, line)) {
        labelText[lineNum] = line;
        lineNum++;
    }
    out.close();

    input_tensor = vxCreateTensor(context, input_num_of_dims, input_dims, VX_TYPE_FLOAT32, 0);
    output_tensor = vxCreateTensor(context, output_num_of_dims, output_dims, VX_TYPE_FLOAT32, 0);
    int count = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];
    
    ERROR_CHECK_STATUS(vxMapTensorPatch(input_tensor, input_num_of_dims, nullptr, nullptr, &map_id, stride,
    (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    if (status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed " << std::endl;
        return status;
    }
    //read an image and resize to correct dimensions -- opencv imread()
    if(!(strcmp(argv[2], "--image"))) {
        struct dirent *pDirent;
        DIR *pDir;
        pDir = opendir(inputFileName.c_str());
        if(pDir == NULL) {
            std::cerr <<"ERROR: Image file directory invalid" << std::endl;
        }
        int bs = 0;
        while((pDirent = readdir(pDir)) && bs < batch_size) {
            std::string imageFileName;
            bool inference = false;
            if ((std::string(pDirent->d_name)).size() > 2) {
                inference = true;
            }
            else {
                continue;
            }
            imageFileName = inputFileName + std::string(pDirent->d_name);
            cv::Mat input_image, input_image_224x224;

            input_image = cv::imread(imageFileName.c_str(), cv::CV_LOAD_IMAGE_COLOR);
            //resizing
            int input_width = input_image.size().width;
            int input_height = input_image.size().height;
            if(input_height > input_width) {
            int dif = input_height - input_width;
            int bar = floor(dif / 2);
            cv::Range rows((bar + (dif % 2)), (input_height - bar));
            cv::Range cols(0, input_width);
            cv::Mat square = input_image(rows, cols);
            cv::resize(square, input_image_224x224, cv::Size(224, 224));
            } 
            else if(input_width > input_height) {
            int dif = input_width - input_height;
            int bar = floor(dif / 2);
            cv::Range rows(0, input_height);
            cv::Range cols((bar + (dif % 2)), (input_width - bar));
            cv::Mat square = input_image(rows, cols);
            cv::resize(square, input_image_224x224, cv::Size(224, 224));
            } 
            else {
                cv::resize(input_image, input_image_224x224, cv::Size(224, 224));
            }

            //preprocess
            cv::Mat RGB_input_image;
            cv::cvtColor(input_image_224x224, RGB_input_image, cv::COLOR_BGR2RGB);  // cv::imread reads the image in order BGR. SO need to convert
            int rows = RGB_input_image.rows; int cols = RGB_input_image.cols; 
            int total = RGB_input_image.total() * RGB_input_image.channels();
            unsigned char *input_image_vector = (RGB_input_image.data);
            float *R = (float *)ptr + bs * total;
            float *G = R + rows * cols;
            float *B = G + rows * cols;
            
            for(int i = 0; i < rows * cols; i++, input_image_vector += 3) {
                *R++ = ((float)input_image_vector[0] * mulVec[0]) - addVec[0]; 
                *G++ = ((float)input_image_vector[1] * mulVec[1]) - addVec[1]; 
                *B++ = ((float)input_image_vector[2] * mulVec[2]) - addVec[2]; 
            }

            if(inference) { 
                bs++;
            }      
        }
        closedir(pDir);
    }

    else if(!(strcmp(argv[2], "--tensor"))) {
        FILE * fp = fopen(inputFileName.c_str(), "rb");
        if(!fp) {
            std::cerr << "ERROR: unable to open: " << inputFileName << std::endl;
            return -1;
        }
        for(size_t n = 0; n < input_dims[0]; n++) {
            for(size_t c = 0; c < input_dims[1]; c++) {
                for(size_t y = 0; y < input_dims[2]; y++) {
                    float * buf = (float *)ptr + (n * input_dims[3] * input_dims[2] * input_dims[1] + c * input_dims[3] * input_dims[2] + y * input_dims[3]);
                    vx_size w = fread(buf, sizeof(float), input_dims[3], fp);
                    if(w != input_dims[3]) {
                        std::cerr << "ERROR: expected char[" << count*sizeof(float) << "], but got less in " << inputFileName << std::endl;
                        return -1;
                    }
                    for(size_t x = 0; x < input_dims[3]; x++) {
                        *(buf+x) = *(buf+x) * mulVec[c] + addVec[c];
                    }
                }
            }
        }
        fclose(fp);
    }
    
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
    outputFile.open(date + "/resnet50-output-results.csv");
    outputFile << "image, classification, probability, label\n";
    
    //find the argmax
    float *output_buf = (float*)ptr;
    auto num_results = 1000;
    for(int i = 0; i < batch_size; i++, output_buf += num_results) {
        int final_argmax_result = std::distance(output_buf, std::max_element(output_buf, output_buf + num_results));
        std::string output_label = labelText[final_argmax_result];
        outputFile << i + 1 << "," << final_argmax_result << "," << output_buf[final_argmax_result] << "," << output_label.c_str() << "\n";
    }
    outputFile.close();
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

