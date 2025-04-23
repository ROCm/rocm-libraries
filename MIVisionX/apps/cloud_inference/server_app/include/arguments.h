/*
Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef ARGUMENTS_H
#define ARGUMENTS_H

#include "common.h"
#include <vector>
#include <string>
#include <tuple>
#include <mutex>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#if ENABLE_OPENCL
#include <CL/cl.h>
#else
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#endif

#define MAX_DEVICE_USE_LIMIT     1   // number of parallel sessions allowed per device

class Arguments {
public:
    Arguments();
    ~Arguments();
    int initializeConfig(int argc, char * argv[]);

    // configuration
    int getPort() {
        return port;
    }
    int getBatchSize() {
        return batchSize;
    }
    int getNumGPUs() {
        return numGPUs;
    }
    const std::string& getConfigurationDir() {
        return configurationDir;
    }
#if ENABLE_OPENCL
    cl_platform_id getPlatformId() {
        return platform_id;
    }
#else
    // todo: for hip
#endif    
    const std::string& getlocalShadowRootDir() {
        return localShadowRootDir;
    }

    const std::string& getModelCompilerPath() {
        return modelCompilerPath;
    }

    // global mutex
    void lock() {
        mutex.lock();
    }
    void unlock() {
        mutex.unlock();
    }

    // model configurations
    int getNumConfigureddModels() {
        std::lock_guard<std::mutex> lock(mutex);
        return configuredModels.size();
    }
    int getNumUploadedModels() {
        std::lock_guard<std::mutex> lock(mutex);
        return uploadedModels.size();
    }
    std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float,std::string> getConfiguredModelInfo(int index) {
        std::lock_guard<std::mutex> lock(mutex);
        return configuredModels[index];
    }
    std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float> getUploadedModelInfo(int index) {
        std::lock_guard<std::mutex> lock(mutex);
        return uploadedModels[index];
    }
    int getNextModelUploadCounter() {
        std::lock_guard<std::mutex> lock(mutex);
        modelFileDownloadCounter++;
        saveConfig();
        return modelFileDownloadCounter;
    }
    void addConfigToUploadedList(std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float>& ann) {
        std::lock_guard<std::mutex> lock(mutex);
        uploadedModels.push_back(ann);
    }
    void addConfigToPreconfiguredList(std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float,std::string>& ann) {
        std::lock_guard<std::mutex> lock(mutex);
        bool replaced = false;
        for(auto it = configuredModels.begin(); it != configuredModels.end(); it++) {
            if(std::get<0>(*it) == std::get<0>(ann)) {
                *it = ann;
                replaced = true;
                break;
            }
        }
        if(!replaced) {
            configuredModels.push_back(ann);
        }
    }
    bool checkPassword(std::string code) {
        return (password == code) ? true : false;
    }
    // set localShadowRootDir (full absolute path)
    void setLocalShadowRootDir(const std::string& localShadowDir)
    {
        localShadowRootDir = localShadowDir;
    }
    // set modelCompiler(nnir) path
    void setModelCompilerPath(const std::string& modelCompDir)
    {
        modelCompilerPath = modelCompDir;
    }
    
    bool fp16Inference()
    {
        return useFp16Inference;
    }
    int decThreads()
    {
        return numDecThreads;
    }
#if ENABLE_OPENCL
    // device resources
    int lockGpuDevices(int GPUs, cl_device_id * device_id_);
    void releaseGpuDevices(int GPUs, const cl_device_id * device_id_);
#else
    // device resources
    int lockGpuDevices(int GPUs, int * device_id_);
    void releaseGpuDevices(int GPUs, const int * device_id_);
#endif

protected:
    void setConfigurationDir();
    void loadConfig();
    void saveConfig();
    void getPreConfiguredModels();

private:
    // loaded configuration
    std::string workFolder;
    int modelFileDownloadCounter;
    int port;
    int batchSize;
    int maxPendingBatches;
    int numGPUs;
    int useFp16Inference;
    int numDecThreads;
    int gpuIdList[MAX_NUM_GPU];
    std::string password;
    // derived configuration
    int maxGpuId;
    int deviceUseCount[MAX_NUM_GPU];
#if ENABLE_OPENCL    
    cl_uint num_devices;
    cl_device_id device_id[MAX_NUM_GPU];
    cl_platform_id platform_id;
#else
    int num_devices;
    void *platform_id;
    int device_id[MAX_NUM_GPU];
#endif
    std::string configurationFile;
    std::string configurationDir;
    std::string localShadowRootDir;
    std::string modelCompilerPath;
    std::vector<std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float,std::string>> configuredModels;
    std::vector<std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float>> uploadedModels;
    // misc
    std::mutex mutex;
};

#endif
