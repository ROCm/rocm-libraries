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
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#define MAX_STRING_LENGTH 100

using namespace std;

#define ERROR_CHECK_STATUS(status) { \
    vx_status status_ = (status); \
    if (status_ != VX_SUCCESS) { \
        std::printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
        exit(1); \
    } \
}

#define ERROR_CHECK_OBJECT(obj) { \
    vx_status status_ = vxGetStatus((vx_reference)(obj)); \
    if(status_ != VX_SUCCESS) { \
        std::printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
        exit(1); \
    } \
}

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    size_t len = strnlen(string, MAX_STRING_LENGTH);
    if (len > 0) {
        std::printf("%s", string);
        if (string[len - 1] != '\n')
            std::printf("\n");
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

void show_usage() {
    std::printf(
            "\n"
            "Usage: ./runMIGraphXTests \n"
            "--tensor                    <path to tensor directory>\n"
            "--profiler_mode  <range:0-2; default:1> [optional]\n"
            "  Mode 0 - Run all tests\n"
            "  Mode 1 - Run all ONNX tests\n"
            "  Mode 2 - Run all JSON tests\n"
            "--profiler_level <range:0-N; default:1> [N = batch size][optional]\n"
            "--resnet50       <resnet50-model>  \n"
            "--googlenet      <googlenet-model> \n" 
            "--squeezenet     <resnet101-model> \n"
            "--alexnet        <resnet152-model> \n"
            "--vgg19          <vgg19-model>     \n"
            "--densenet       <densenet-model>  \n"
            "\n"
        ); 
}

int main(int argc, char **argv) {

    // check command-line usage
    std::string binaryFilename_squeezenet_str;
    std::string binaryFilename_resnet50_str;
    std::string binaryFilename_vgg19_str;
    std::string binaryFilename_googlenet_str;
    std::string binaryFilename_alexnet_str;
    std::string binaryFilename_densenet_str;
    std::string inputTensor_foldername;

    int parameter = 0;
    int profiler_level = 1;
    int64_t freq = clockFrequency(), t0, t1;
    int N = 1000;
    bool runResnet50 = false, runVgg19 = false, runGooglenet = false, runDensenet = false, runAlexnet = false, runSqueezenet = false, runAnyImagenet = false;

    for(int arg = 1; arg < argc; arg++) {
        if (!strcasecmp(argv[arg], "--help") || !strcasecmp(argv[arg], "--H") || !strcasecmp(argv[arg], "--h")) {
            show_usage();
            exit(-1);
        }
        else if (!strcasecmp(argv[arg], "--tensor")) {
            if ((arg + 1) == argc) {
                std::printf("\n\nERROR: missing tensor file on command-line (see help for details)\n\n\n");
                show_usage();
                exit(-1);
            }
            arg++;
            inputTensor_foldername = (argv[arg]);
            parameter++;
        }
        else if (!strcasecmp(argv[arg], "--profiler_mode")) {
            int profiler_mode = 1;
            arg++;
            profiler_mode = std::stoi(argv[arg]);
            parameter++;
        }
        else if (!strcasecmp(argv[arg], "--profiler_level")){
            arg++;
            profiler_level = std::stoi(argv[arg]);
            if(profiler_level < 1 || profiler_level > 8) {
                std::printf("\n\nERROR: profiler level has to be between 1-7\n\n\n");
                exit(-1);
            }
            parameter++;
        }
        else if (!strcasecmp(argv[arg], "--alexnet")) {
            if ((arg + 1) == argc) {
                std::printf("\n\nERROR: missing alexnet ONNX .model file location on command-line (see help for details)\n\n\n");
                show_usage();
                exit(-1);
            }
            runAlexnet = true;
            runAnyImagenet = true;
            arg++;
            binaryFilename_alexnet_str = (argv[arg]);
            parameter++;
        }
        else if (!strcasecmp(argv[arg], "--squeezenet")) {
            if ((arg + 1) == argc) {
                std::printf("\n\nERROR: missing squeezenet ONNX .model file location on command-line (see help for details)\n\n\n");
                show_usage();
                exit(-1);
            }
            runSqueezenet = true;
            runAnyImagenet = true;
            arg++;
            binaryFilename_squeezenet_str = (argv[arg]);
            parameter++;
        }
        else if (!strcasecmp(argv[arg], "--resnet50")) {
            if ((arg + 1) == argc) {
                std::printf("\n\nERROR: missing resnet50 ONNX .model file location on command-line (see help for details)\n\n\n");
                show_usage();
                exit(-1);
            }
            runResnet50 = true;
            runAnyImagenet = true;
            arg++;
            binaryFilename_resnet50_str = (argv[arg]);
            parameter++;
        }

        else if (!strcasecmp(argv[arg], "--vgg19")) {
            if ((arg + 1) == argc) {
                std::printf("\n\nERROR: missing vgg19 ONNX .model file location on command-line (see help for details)\n\n\n");
                show_usage();
                exit(-1);
            }
            runVgg19 = true;
            runAnyImagenet = true;
            arg++;
            binaryFilename_vgg19_str = (argv[arg]);
            parameter++;
        }
        else if (!strcasecmp(argv[arg], "--googlenet")) {
            if ((arg + 1) == argc) {
                std::printf("\n\nERROR: missing googlenet ONNX .model file location on command-line (see help for details)\n\n\n");
                show_usage();
                exit(-1);
            }
            runGooglenet = true;
            runAnyImagenet = true;
            arg++;
            binaryFilename_googlenet_str = (argv[arg]);
            parameter++;
        }
        else if (!strcasecmp(argv[arg], "--densenet")) {
            if ((arg + 1) == argc) {
                std::printf("\n\nERROR: missing densenet ONNX .model file location on command-line (see help for details)\n\n\n");
                show_usage();
                exit(-1);
            }
            runDensenet = true;
            runAnyImagenet = true;
            arg++;
            binaryFilename_densenet_str = (argv[arg]);
            parameter++;
        }
    }

    if (parameter < 2) {
        std::printf("\nERROR: missing parameters in command-line.\n");
        show_usage();
        exit(-1);
    }

    // create context, input, output, and graph
    vx_status status = 0;
    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);
    vxRegisterLogCallback(context, log_callback, vx_false_e);
    
    // load vx_nn kernels
    ERROR_CHECK_STATUS(vxLoadKernels(context, "vx_amd_migraphx"));

    // initialize variables
    vx_tensor input_tensor_224x224;
    vx_size input_num_of_dims = 4;
    vx_size output_num_of_dims_2 = 2;
    vx_size output_num_of_dims_4 = 4;
    vx_size stride[4];
    vx_map_id map_id;
    void *ptr = nullptr;
    auto num_results_imagenet = 1000;
    
    //create a results folder
    std::ofstream outputFile;
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream datetime;
    datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%X");
    std::string date = "../results-" + datetime.str();
    if (mkdir(date.c_str(), 0777) == -1)
        cerr << "Error, cannot create results folder:  " << strerror(errno) << endl;

    float mean_vec[3] = {0.485, 0.456, 0.406};
    float stddev_vec[3] = {0.229, 0.224, 0.225};
    std::vector<float> mulVec = {1 / (255 * stddev_vec[0]), 1 / (255 * stddev_vec[1]), 1 / (255 * stddev_vec[2])}, 
                        addVec = {(mean_vec[0] / stddev_vec[0]), (mean_vec[1] / stddev_vec[1]), (mean_vec[2] / stddev_vec[2])};
    //imagenet label file
    std::string labelText[1000];
    std::string imagenetLabelFileName = ("../labels.txt");

    std::string line;
    std::ifstream labelFile(imagenetLabelFileName);
    if(!labelFile) {
      std::cout << "failed to open label file" << std::endl;
      return -1; 
    }
    int lineNum = 0;
    while(getline(labelFile, line)) {
        labelText[lineNum] = line;
        lineNum++;
    }
    labelFile.close();

    for(int lev = 0; lev < profiler_level; lev++) {
        vx_size batch_size = std::pow(2, lev);
        printf("batch_size = %lu\n", batch_size);
        vx_size input_dims_data_224x224[4] = {batch_size, 3, 224,224};
        vx_size output_dims_data_1x1000[2] = {batch_size, 1000};
        vx_size output_dims_data_1x1000x1x1[4] = {batch_size, 1000, 1, 1};
        int count = input_dims_data_224x224[0] * input_dims_data_224x224[1] * input_dims_data_224x224[2] * input_dims_data_224x224[3];

        //create input data
        input_tensor_224x224 = vxCreateTensor(context, input_num_of_dims, input_dims_data_224x224, VX_TYPE_FLOAT32, 0);

        if(runAnyImagenet) {
            ERROR_CHECK_STATUS(vxMapTensorPatch(input_tensor_224x224, input_num_of_dims, nullptr, nullptr, &map_id, stride,
            (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            if (status) {
                std::cerr << "ERROR: vxMapTensorPatch() failed for imagenet" << std::endl;
                return status;
            }

            std::string inputTensor_filename = inputTensor_foldername + "/mike-tensor-" + std::to_string(batch_size) + "-224x224.fp";
            FILE * fp = fopen(inputTensor_filename.c_str(), "rb");
            if(!fp) {
                std::cerr << "ERROR: unable to open: " << inputTensor_filename << std::endl;
                return -1;
            }
            for(size_t n = 0; n < input_dims_data_224x224[0]; n++) {
                for(size_t c = 0; c < input_dims_data_224x224[1]; c++) {
                    for(size_t y = 0; y < input_dims_data_224x224[2]; y++) {
                        float * buf = (float *)ptr + 
                                        (n * input_dims_data_224x224[3] * input_dims_data_224x224[2] * input_dims_data_224x224[1] 
                                        + c * input_dims_data_224x224[3] * input_dims_data_224x224[2] 
                                        + y * input_dims_data_224x224[3]);
                        vx_size w = fread(buf, sizeof(float), input_dims_data_224x224[3], fp);
                        if(w != input_dims_data_224x224[3]) {
                            std::cerr << "ERROR: expected char[" << count*sizeof(float) << "], but got less in " << inputTensor_filename << std::endl;
                            return -1;
                        }
                        for(size_t x = 0; x < input_dims_data_224x224[3]; x++) {
                            *(buf+x) = *(buf+x) * mulVec[c] + addVec[c];
                        }
                    }
                }
            }
            fclose(fp);

            status = vxUnmapTensorPatch(input_tensor_224x224, map_id);
            if (status) {
                std::cerr << "ERROR: vxUnmapTensorPatch() failed for imagenet" << status << ")" << std::endl;
                return status;
            }
            if (runResnet50) {
                //output tensor
                vx_tensor output_tensor_resnet50 = vxCreateTensor(context, output_num_of_dims_2, output_dims_data_1x1000, VX_TYPE_FLOAT32, 0);
                
                //graph creation
                vx_graph graph_resnet50 = vxCreateGraph(context);
                status = vxGetStatus((vx_reference)graph_resnet50);
                if(status) {
                    printf("ERROR: vxCreateGraph(...) for renset50 failed (%d)\n", status);
                    return -1;
                }

                vx_node node_resnet50 = amdMIGraphXnode(graph_resnet50, binaryFilename_resnet50_str.c_str(), input_tensor_224x224, output_tensor_resnet50);
                ERROR_CHECK_OBJECT(node_resnet50);
                ERROR_CHECK_STATUS(vxVerifyGraph(graph_resnet50));
                ERROR_CHECK_STATUS(vxProcessGraph(graph_resnet50));

                //resnet50 timing for 1000 iterations
                t0 = clockCounter();
                for(int i = 0; i < N; i++) {
                    ERROR_CHECK_STATUS(vxProcessGraph(graph_resnet50));
                }
                t1 = clockCounter();
                float resnet50Time = (float)(t1 - t0) * 1000.0f / (float)freq / (float)N;
                printf("OK: resnet50 took %.3f msec (average over %d iterations)\n", resnet50Time, N);

                //resnet50 results
                status = vxMapTensorPatch(output_tensor_resnet50, output_num_of_dims_2, nullptr, nullptr, &map_id, stride,
                    (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
                if (status) {
                    std::cerr << "ERROR: vxMapTensorPatch() failed for output tensor" << std::endl;
                    return status;
                }

                if (lev == profiler_level - 1) {
                    outputFile.open(date + "/resnet50-output-results.csv");
                    outputFile << "image, classification, probability, label\n";
                }
                float *output_buf = (float*)ptr;
                for(int i = 0; i < batch_size; i++, output_buf += num_results_imagenet) {
                    int ID_resnet50 = std::distance(output_buf, std::max_element(output_buf, output_buf + num_results_imagenet));
                    std::string output_label_resnet50 = labelText[ID_resnet50];
                    if (lev == profiler_level - 1) {
                        outputFile << i + 1 << "," << ID_resnet50 << "," << output_buf[ID_resnet50] << "," << output_label_resnet50.c_str() << "\n";
                    }
                }
                outputFile.close();
                status = vxUnmapTensorPatch(output_tensor_resnet50, map_id);
                if(status) {
                    std::cerr << "ERROR: vxUnmapTensorPatch() failed for output_tensor" << std::endl;
                    return status;
                }

                ERROR_CHECK_STATUS(vxReleaseNode(&node_resnet50));
                ERROR_CHECK_STATUS(vxReleaseGraph(&graph_resnet50));
                ERROR_CHECK_STATUS(vxReleaseTensor(&output_tensor_resnet50));
            }

            if (runVgg19) {
                //output tensor
                vx_tensor output_tensor_vgg19 = vxCreateTensor(context, output_num_of_dims_2, output_dims_data_1x1000, VX_TYPE_FLOAT32, 0);

                //graph creation
                vx_graph graph_vgg19 = vxCreateGraph(context);
                status = vxGetStatus((vx_reference)graph_vgg19);
                if(status) {
                    printf("ERROR: vxCreateGraph(...) for vgg19 failed (%d)\n", status);
                    return -1;
                }
                vx_node node_vgg19 = amdMIGraphXnode(graph_vgg19, binaryFilename_vgg19_str.c_str(), input_tensor_224x224, output_tensor_vgg19);
                ERROR_CHECK_OBJECT(node_vgg19);
                ERROR_CHECK_STATUS(vxVerifyGraph(graph_vgg19));
                ERROR_CHECK_STATUS(vxProcessGraph(graph_vgg19));

                //vgg19 timing for 1000 iterations
                t0 = clockCounter();
                for(int i = 0; i < N; i++) {
                    ERROR_CHECK_STATUS(vxProcessGraph(graph_vgg19));
                }
                t1 = clockCounter();
                float vgg19Time = (float)(t1 - t0) * 1000.0f / (float)freq / (float)N;
                printf("OK: vgg19 took %.3f msec (average over %d iterations)\n", vgg19Time, N);

                //vgg19 results
                status = vxMapTensorPatch(output_tensor_vgg19, output_num_of_dims_2, nullptr, nullptr, &map_id, stride,
                    (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
                if (status) {
                    std::cerr << "ERROR: vxMapTensorPatch() failed for output tensor" << std::endl;
                    return status;
                }

                if (lev == profiler_level - 1) {
                    outputFile.open(date + "/vgg19-output-results.csv");
                    outputFile << "image, classification, probability, label\n";
                }
                float *output_buf = (float*)ptr;
                for(int i = 0; i < batch_size; i++, output_buf += num_results_imagenet) {
                    int ID_vgg19 = std::distance(output_buf, std::max_element(output_buf, output_buf + num_results_imagenet));
                    std::string output_label_vgg19 = labelText[ID_vgg19];
                    if (lev == profiler_level - 1) {
                        outputFile << i + 1 << "," << ID_vgg19 << "," << output_buf[ID_vgg19] << "," << output_label_vgg19 << "\n";
                    }
                }
                outputFile.close();
                status = vxUnmapTensorPatch(output_tensor_vgg19, map_id);
                if(status) {
                    std::cerr << "ERROR: vxUnmapTensorPatch() failed for output_tensor" << std::endl;
                    return status;
                }

                //release resources
                ERROR_CHECK_STATUS(vxReleaseNode(&node_vgg19));
                ERROR_CHECK_STATUS(vxReleaseGraph(&graph_vgg19));
                ERROR_CHECK_STATUS(vxReleaseTensor(&output_tensor_vgg19));
            }

            if (runGooglenet) {
                //output tensor
                vx_tensor output_tensor_googlenet = vxCreateTensor(context, output_num_of_dims_2, output_dims_data_1x1000, VX_TYPE_FLOAT32, 0);

                //graph creation
                vx_graph graph_googlenet = vxCreateGraph(context);
                status = vxGetStatus((vx_reference)graph_googlenet);
                if(status) {
                    printf("ERROR: vxCreateGraph(...) for googlenet failed (%d)\n", status);
                    return -1;
                }
                vx_node node_googlenet = amdMIGraphXnode(graph_googlenet, binaryFilename_googlenet_str.c_str(), input_tensor_224x224, output_tensor_googlenet);
                ERROR_CHECK_OBJECT(node_googlenet);
                ERROR_CHECK_STATUS(vxVerifyGraph(graph_googlenet));
                ERROR_CHECK_STATUS(vxProcessGraph(graph_googlenet));

                //googlenet timing for 1000 iterations
                t0 = clockCounter();
                for(int i = 0; i < N; i++) {
                    ERROR_CHECK_STATUS(vxProcessGraph(graph_googlenet));
                }
                t1 = clockCounter();
                float googlenetTime = (float)(t1 - t0) * 1000.0f / (float)freq / (float)N;
                printf("OK: googlenet took %.3f msec (average over %d iterations)\n", googlenetTime, N);

                //googlenet results
                status = vxMapTensorPatch(output_tensor_googlenet, output_num_of_dims_2, nullptr, nullptr, &map_id, stride,
                    (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
                if (status) {
                    std::cerr << "ERROR: vxMapTensorPatch() failed for output tensor" << std::endl;
                    return status;
                }

                if (lev == profiler_level - 1) {
                    outputFile.open(date + "/googlenet-output-results.csv");
                    outputFile << "image, classification, probability, label\n";
                }
                float *output_buf = (float*)ptr;
                for(int i = 0; i < batch_size; i++, output_buf += num_results_imagenet) {
                    int ID_googlenet = std::distance(output_buf, std::max_element(output_buf, output_buf + num_results_imagenet));
                    std::string output_label_googlenet = labelText[ID_googlenet];
                    if (lev == profiler_level - 1) {
                        outputFile << i + 1 << "," << ID_googlenet << "," << output_buf[ID_googlenet] << "," << output_label_googlenet << "\n";
                    }
                }
                outputFile.close();
                status = vxUnmapTensorPatch(output_tensor_googlenet, map_id);
                if(status) {
                    std::cerr << "ERROR: vxUnmapTensorPatch() failed for output_tensor" << std::endl;
                    return status;
                }

                //release resources
                ERROR_CHECK_STATUS(vxReleaseNode(&node_googlenet));
                ERROR_CHECK_STATUS(vxReleaseGraph(&graph_googlenet));
                ERROR_CHECK_STATUS(vxReleaseTensor(&output_tensor_googlenet));
            }

            if (runAlexnet) {
                //output tensor
                vx_tensor output_tensor_alexnet = vxCreateTensor(context, output_num_of_dims_2, output_dims_data_1x1000, VX_TYPE_FLOAT32, 0);

                //graph creation
                vx_graph graph_alexnet = vxCreateGraph(context);
                status = vxGetStatus((vx_reference)graph_alexnet);
                if(status) {
                    printf("ERROR: vxCreateGraph(...) for alexnet failed (%d)\n", status);
                    return -1;
                }
                vx_node node_alexnet = amdMIGraphXnode(graph_alexnet, binaryFilename_alexnet_str.c_str(), input_tensor_224x224, output_tensor_alexnet);
                ERROR_CHECK_OBJECT(node_alexnet);
                ERROR_CHECK_STATUS(vxVerifyGraph(graph_alexnet));
                ERROR_CHECK_STATUS(vxProcessGraph(graph_alexnet));

                //alexnet timing for 1000 iterations
                t0 = clockCounter();
                for(int i = 0; i < N; i++) {
                    ERROR_CHECK_STATUS(vxProcessGraph(graph_alexnet));
                }
                t1 = clockCounter();
                float alexnetTime = (float)(t1 - t0) * 1000.0f / (float)freq / (float)N;
                printf("OK: alexnet took %.3f msec (average over %d iterations)\n", alexnetTime, N);

                //alexnet results
                status = vxMapTensorPatch(output_tensor_alexnet, output_num_of_dims_2, nullptr, nullptr, &map_id, stride,
                    (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
                if (status) {
                    std::cerr << "ERROR: vxMapTensorPatch() failed for output tensor" << std::endl;
                    return status;
                }

                if (lev == profiler_level - 1) {
                    outputFile.open(date + "/alexnet-output-results.csv");
                    outputFile << "image, classification, probability, label\n";
                }
                float *output_buf = (float*)ptr;
                for(int i = 0; i < batch_size; i++, output_buf += num_results_imagenet) {
                    int ID_alexnet = std::distance(output_buf, std::max_element(output_buf, output_buf + num_results_imagenet));
                    std::string output_label_alexnet = labelText[ID_alexnet];
                    if (lev == profiler_level - 1) {
                        outputFile << i + 1 << "," << ID_alexnet << "," << output_buf[ID_alexnet] << "," << output_label_alexnet << "\n";
                    }
                }
                outputFile.close();
                status = vxUnmapTensorPatch(output_tensor_alexnet, map_id);
                if(status) {
                    std::cerr << "ERROR: vxUnmapTensorPatch() failed for output_tensor" << std::endl;
                    return status;
                }

                //release resources
                ERROR_CHECK_STATUS(vxReleaseNode(&node_alexnet));
                ERROR_CHECK_STATUS(vxReleaseGraph(&graph_alexnet));
                ERROR_CHECK_STATUS(vxReleaseTensor(&output_tensor_alexnet));

            }

            if (runSqueezenet) {
                //output tensor
                vx_tensor output_tensor_squeezenet = vxCreateTensor(context, output_num_of_dims_4, output_dims_data_1x1000x1x1, VX_TYPE_FLOAT32, 0);

                //graph creation
                vx_graph graph_squeezenet = vxCreateGraph(context);
                status = vxGetStatus((vx_reference)graph_squeezenet);
                if(status) {
                    printf("ERROR: vxCreateGraph(...) for squeezenet failed (%d)\n", status);
                    return -1;
                }

                vx_node node_squeezenet = amdMIGraphXnode(graph_squeezenet, binaryFilename_squeezenet_str.c_str(), input_tensor_224x224, output_tensor_squeezenet);
                ERROR_CHECK_OBJECT(node_squeezenet);
                ERROR_CHECK_STATUS(vxVerifyGraph(graph_squeezenet));
                ERROR_CHECK_STATUS(vxProcessGraph(graph_squeezenet));

                //squeezenet timing for 1000 iterations
                t0 = clockCounter();
                for(int i = 0; i < N; i++) {
                    ERROR_CHECK_STATUS(vxProcessGraph(graph_squeezenet));
                }
                t1 = clockCounter();
                float squeezenetTime = (float)(t1 - t0) * 1000.0f / (float)freq / (float)N;
                printf("OK: squeezenet took %.3f msec (average over %d iterations)\n", squeezenetTime, N);

                //squeezenet results
                status = vxMapTensorPatch(output_tensor_squeezenet, output_num_of_dims_4, nullptr, nullptr, &map_id, stride,
                    (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
                if (status) {
                    std::cerr << "ERROR: vxMapTensorPatch() failed for output tensor" << std::endl;
                    return status;
                }

                if (lev == profiler_level - 1) {
                    outputFile.open(date + "/squeezenet-output-results.csv");
                    outputFile << "image, classification, probability, label\n";
                }
                float *output_buf = (float*)ptr;
                for(int i = 0; i < batch_size; i++, output_buf += num_results_imagenet) {
                    int ID_squeezenet = std::distance(output_buf, std::max_element(output_buf, output_buf + num_results_imagenet));
                    std::string output_label_squeezenet = labelText[ID_squeezenet];
                    if (lev == profiler_level - 1) {
                        outputFile << i + 1 << "," << ID_squeezenet << "," << output_buf[ID_squeezenet] << "," << output_label_squeezenet << "\n";
                    }
                }
                outputFile.close();
                status = vxUnmapTensorPatch(output_tensor_squeezenet, map_id);
                if(status) {
                    std::cerr << "ERROR: vxUnmapTensorPatch() failed for output_tensor" << std::endl;
                    return status;
                }

                // release resources   
                ERROR_CHECK_STATUS(vxReleaseNode(&node_squeezenet));
                ERROR_CHECK_STATUS(vxReleaseGraph(&graph_squeezenet));
                ERROR_CHECK_STATUS(vxReleaseTensor(&output_tensor_squeezenet));
            }

            if (runDensenet) {
                //output tensor
                vx_tensor output_tensor_densenet = vxCreateTensor(context, output_num_of_dims_4, output_dims_data_1x1000x1x1, VX_TYPE_FLOAT32, 0);

                //graph creation
                vx_graph graph_densenet = vxCreateGraph(context);
                status = vxGetStatus((vx_reference)graph_densenet);
                if(status) {
                    printf("ERROR: vxCreateGraph(...) for densenet failed (%d)\n", status);
                    return -1;
                }
                vx_node node_densenet = amdMIGraphXnode(graph_densenet, binaryFilename_densenet_str.c_str(), input_tensor_224x224, output_tensor_densenet);
                ERROR_CHECK_OBJECT(node_densenet);  
                ERROR_CHECK_STATUS(vxVerifyGraph(graph_densenet));
                ERROR_CHECK_STATUS(vxProcessGraph(graph_densenet));

                //densenet timing for 1000 iterations
                t0 = clockCounter();
                for(int i = 0; i < N; i++) {
                    ERROR_CHECK_STATUS(vxProcessGraph(graph_densenet));
                }
                t1 = clockCounter();
                float densenetTime = (float)(t1 - t0) * 1000.0f / (float)freq / (float)N;
                printf("OK: densenet took %.3f msec (average over %d iterations)\n", densenetTime, N);

                //densenet results
                status = vxMapTensorPatch(output_tensor_densenet, output_num_of_dims_4, nullptr, nullptr, &map_id, stride,
                    (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
                if (status) {
                    std::cerr << "ERROR: vxMapTensorPatch() failed for output tensor" << std::endl;
                    return status;
                }

                if (lev == profiler_level - 1) {
                    outputFile.open(date + "/densenet-output-results.csv");
                    outputFile << "image, classification, probability, label\n";
                }
                float *output_buf = (float*)ptr;
                for(int i = 0; i < batch_size; i++, output_buf += num_results_imagenet) {
                    int ID_densenet = std::distance(output_buf, std::max_element(output_buf, output_buf + num_results_imagenet));
                    std::string output_label_densenet = labelText[ID_densenet];
                    if (lev == profiler_level - 1) {
                        outputFile << i + 1 << "," << ID_densenet << "," << output_buf[ID_densenet] << "," << output_label_densenet << "\n";
                    }
                }
                outputFile.close();
                status = vxUnmapTensorPatch(output_tensor_densenet, map_id);
                if(status) {
                    std::cerr << "ERROR: vxUnmapTensorPatch() failed for output_tensor" << std::endl;
                    return status;
                }
                
                //release resources
                ERROR_CHECK_STATUS(vxReleaseNode(&node_densenet));
                ERROR_CHECK_STATUS(vxReleaseGraph(&graph_densenet));
                ERROR_CHECK_STATUS(vxReleaseTensor(&output_tensor_densenet));
            }
        }
        //release common resources
        if (runAnyImagenet) {
            ERROR_CHECK_STATUS(vxReleaseTensor(&input_tensor_224x224));
        }
    }
 
    ERROR_CHECK_STATUS(vxReleaseContext(&context)); 
    return 0;
}

