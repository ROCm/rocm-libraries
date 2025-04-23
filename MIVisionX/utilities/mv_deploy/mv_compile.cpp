/*
MIT License

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

#include "mvdeploy_api.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <iostream>
#include <string>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#if __cplusplus >= 201703L && __has_include(<filesystem>)
    #include <filesystem>
    namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#endif
// helper functions
void info(const char * format, ...)
{
    printf("INFO: ");
    va_list args;
    va_start(args, format);
    int r = vprintf(format, args);
    va_end(args);
    printf("\n");
}

void warning(const char * format, ...)
{
    printf("WARNING: ");
    va_list args;
    va_start(args, format);
    int r = vprintf(format, args);
    va_end(args);
    printf("\n");
}

void error(const char * format, ...)
{
    printf("ERROR: ");
    va_list args;
    va_start(args, format);
    int r = vprintf(format, args);
    va_end(args);
    printf("\n");
}

//! \brief Compiles and validates model for the specific backend from a trained input model with weights.
// models supported: caffe, onnx and nnef
// for OpenVX_WinML, only onnx model is supported
static mv_status MIVID_API_CALL mvLoadUpdateAndCompileModelForBackend(mivid_backend backend, const char *model_name, const char *install_folder, mivid_update_model_params *update_params, size_t input_dims[4])
{
    std::string command;
    int status;
    bool bUpdateModel = false;
    size_t batchSize = (input_dims[3] <= 0)? 1: input_dims[3];

    if (backend == OpenVX_Rocm_GPU) {
        printf("compiling model for backend OpenVX_Rocm_GPU\n");
        std::string compiler_path = "/opt/rocm/libexec/mivisionx/model_compiler";       // default
        char *model_compiler_path = getenv("MIVISIONX_MODEL_COMPILER_PATH");
        std::string install_dir = std::string(install_folder).empty() ? "mvdeploy_lib" : std::string(install_folder); 
        if (model_compiler_path != nullptr) {
            compiler_path = std::string(model_compiler_path);
            //return MV_FAILURE;  
        } else {
            printf("Env MIVISIONX_MODEL_COMPILER_PATH is not specified, using default %s\n", compiler_path.c_str());
        }
        if (!fs::exists(model_name)) {
            error("model %s does not exist", model_name);
            return MV_FAILURE;
        }
        // run model compiler and generate NNIR graph
        // step-1: run python3 caffe_to_nnir.py <.caffemodel> nnir_output --input-dims <args->getBatchSize(),dimOutput[2], dimOutput[1], dimOutput[0]>
        std::string model_extension = std::string(strchr(model_name, '.'));
        if (!model_extension.compare(".caffemodel")) {
            command = "python3 ";
            command += compiler_path + "/python" + "/caffe_to_nnir.py";
            command += " " + std::string(model_name);
            command += " nnir-output --input-dims";
            command += " " + std::to_string(batchSize)
                    +  "," + std::to_string(input_dims[2])
                    +  "," + std::to_string(input_dims[1])
                    +  "," + std::to_string(input_dims[0]);
            info("executing: %% %s", command.c_str());
            status = system(command.c_str());
        }
        else if (!model_extension.compare(".onnx")) {
            command = "python3 ";
            command += compiler_path + "/python" + "/onnx_to_nnir.py";
            command += " " + std::string(model_name);
            command += " nnir-output --input_dims";
            command += " " + std::to_string(batchSize)
                    +  "," + std::to_string(input_dims[2])
                    +  "," + std::to_string(input_dims[1])
                    +  "," + std::to_string(input_dims[0]);
            info("executing: %% %s", command.c_str());
            status = system(command.c_str());

        } else if (!model_extension.compare(".nnef")) {
            command = "python3 ";
            command += compiler_path + "/python" + "/nnef_to_nnir.py";
            command += " " + std::string(model_name);
            command += " nnir-output";
            info("executing: %% %s", command.c_str());
            status = system(command.c_str());
        }
        else{
            return MV_ERROR_NOT_SUPPORTED;  
        }
        if (status) {
            error("model conversion to nnir failed with status %d", status);
            return MV_FAILURE;
        }
        else {
            info("model conversion successful");
        }
        // step-2: run nnir_update.py for fusing kernels and quantizing
        if (update_params != nullptr) {
            std::string sub_command = " ";
            if ((update_params->batch_size > 0) && (update_params->batch_size != batchSize)) {
                batchSize = update_params->batch_size;
                sub_command += " --batch-size " + std::to_string(batchSize);
                bUpdateModel = true;
            }
            if (update_params->fused_convolution_bias_activation) {
                sub_command += " --fuse-ops 1";
                bUpdateModel = true;
            }
            if (update_params->quantize_model) {
                if (update_params->quantization_mode == quant_fp16) {
                    sub_command += " --convert-fp16 1";
                    bUpdateModel = true;
                }
                else {
                    error("quant_mode %d not supported", update_params->quantization_mode);
                    return MV_ERROR_NOT_SUPPORTED;
                }
            }
            if (bUpdateModel) {
                command = "python3 "+ compiler_path + "/python/nnir_update.py";
                command += sub_command;
                command += " nnir-output nnir-output";
                info("executing: %% %s", command.c_str());
                status = system(command.c_str());
                info("python3 nnir-update.py %s nnir-output nnir-output completed (%d)", sub_command.c_str(), status);
                if (status) {
                    return MV_FAILURE;
                }
            }
            // step-3: run nnir_to_clib.py for generating OpenVX code for the inference deployment
            command = "python3 "+ compiler_path + "/python/nnir_to_clib.py nnir-output ";
        }
        else {
            printf("Not Update Model\n");
            // step-3: run nnir_to_clib.py for generating OpenVX code for the inference deployment
            command = "python3 "+ compiler_path + "/python/" + "nnir_to_clib.py nnir-output ";
        }

        command += install_dir + " >> nnir_to_clib.log"; 
        info("executing: %% %s", command.c_str());
        status = system(command.c_str());
        if (status) {
            error("command-failed(%d): %s", status, command.c_str());
            return MV_FAILURE;
        }
        info("nnir_to_clib completed");
        // step-4: do cmake and make to generate clib
        std::string buildFolder = install_dir + "/build";
        if((mkdir(buildFolder.c_str(), 0770) < 0)) {
            error("unable to create folder: %s", buildFolder.c_str());
        }
        status = chdir(buildFolder.c_str());
        command = "cmake ../ >> ../cmake.log";
        info("executing: %% %s", command.c_str());
        status = system(command.c_str());
        command = "make >> ../make.log";
        info("executing: %% %s", command.c_str());
        status = system(command.c_str());
        if (status) {
            error("command-failed(%d): %s", status, command.c_str());
            return MV_FAILURE;
        }
        command = "rm -rf /" + buildFolder;
        status = system(command.c_str());
        command = "rm -rf /" + install_dir + "/nnir-output/";
        status = system(command.c_str());
        return MV_SUCCESS;
    } 
    else if (backend == OpenVX_WinML) {
        if (!strcmp(strchr(model_name, '.'), "onnx")){
            return MV_ERROR_NOT_SUPPORTED;
        }
        // todo:: do the required initialization for WinML
        // compile and generate single node executable
        return MV_ERROR_NOT_IMPLEMENTED;
    } 
    else {
        return MV_ERROR_NOT_SUPPORTED;
    }
}

void printUsage() {
    printf("Usage: mv_compile options..\n"
        "\t--model <model_name> : name of the trained model with full path    \t\t[required]\n"
        "\t--install_folder <install_folder> : the location for compiled model\t\t[required]\n"
        "\t--input_dims <n,c,h,w>: dimension of input for the model given in format NCHW\t[required]\n"
        "\t--backend <backend>: is the name of the backend for compilation\t\t\t[optional-default:OpenVX_Rocm_GPU]\n"
        "\t--fuse_cba <0/1> :enable or disable Convolution_bias_activation fuse mode(0/1)\t[optional-default:0]\n"
        "\t--quant_mode <fp32/fp16>: quant_mode for the model, if enabled the model and weights are converted to FP16\t[optional(default:fp32)]\n"
        "\n"
    );
}

int main(int argc, const char ** argv)
{
    // check command-line usage
    if(argc < 6) {
        printUsage();
        return -1;
    }
    // load and compile model for backend first
    mv_status status;
    const char *model, *install_folder, *input_dim_str;
    mivid_update_model_params model_update_params = {0};
    mivid_backend backend = (mivid_backend)OpenVX_Rocm_GPU;   // default
    install_folder = "";
    int quant_mode = quant_fp32;
    for (int arg = 1; arg < argc; arg++) {
        if (!strcmp(argv[arg], "--model")) {
            arg++;
            model = argv[arg];
        }
        if (!strcmp(argv[arg], "--install_folder")) {
            arg++;
            install_folder = argv[arg];
        }
        if (!strcmp(argv[arg], "--input_dims")) {
            arg++;
            input_dim_str = argv[arg];
        }
        if (!strcmp(argv[arg], "--backend")) {
            arg++;
            backend = (mivid_backend)atoi(argv[arg]);
        }
        if (!strcmp(argv[arg], "--fuse_cba")) {
            arg++;
            model_update_params.fused_convolution_bias_activation = atoi(argv[arg]);
        }
        if (!strcmp(argv[arg], "--quant_mode")) {
            arg++;
            quant_mode = atoi(argv[arg]);
        }
    }
    if (input_dim_str == nullptr)
    {
        printf("Error:: input dims not specified \n");
        return -1;        
    }
    std::stringstream input_dims(input_dim_str);
    size_t inp_dims[4];
    for (int i=0; i < 4; i++) {
        std::string substr;
        getline(input_dims, substr, ',' );
        inp_dims[3-i] = atoi(substr.c_str());
    }
    // set model update params
    model_update_params.batch_size = inp_dims[3];
    if (quant_mode > quant_fp32) {
        model_update_params.quantize_model = 1;
        model_update_params.quantization_mode = (mivid_quantization_mode)quant_mode;
    }

    status = mvLoadUpdateAndCompileModelForBackend(backend, model, install_folder, &model_update_params, inp_dims);
    if (status != MV_SUCCESS) {
        printf("Error in importing model to MIVisionX \n");
        return -1;
    }
    printf("OK: MIVisionX model compilation Successful \n");

    return 0;    
}

