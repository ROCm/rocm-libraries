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

#include "arguments.h"
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#define BUILD_VERSION "alpha3"

Arguments::Arguments()
        : workFolder{ "~" }, modelFileDownloadCounter{ 0 },
          password{ "radeon" },
          modelCompilerPath{ "/opt/rocm/libexec/mivisionx/model_compiler/python" },
          port{ 28282 }, batchSize{ 64 }, maxPendingBatches{ 4 }, numGPUs{ 1 }, gpuIdList{ 0 },
          maxGpuId{ 0 }, platform_id{ NULL }, num_devices{ 0 },  deviceUseCount{ 0 }
{
    ////////
    /// \brief set default configuration file
    ///
    configurationFile = getenv("HOME");
    configurationFile += "/.inference_server_app.txt";

#if ENABLE_OPENCL
    ////////
    /// \brief get AMD OpenCL platform (if available)
    ///
    cl_uint num_platforms;
    cl_int status;
    if ((status = clGetPlatformIDs(0, NULL, &num_platforms)) != CL_SUCCESS) {
        fatal("clGetPlatformIDs(0,0,*) => %d (failed)", status);
    }
    cl_platform_id * platform_list = new cl_platform_id[num_platforms];
    if ((status = clGetPlatformIDs(num_platforms, platform_list, NULL)) != CL_SUCCESS) {
        fatal("clGetPlatformIDs(%d,*,0) => %d (failed)", num_platforms, status);
    }
    cl_uint platform_index = 0;
    const char * opencl_platform_override = getenv("AGO_OPENCL_PLATFORM");
    if(opencl_platform_override) {
        cl_uint index = (cl_uint)atoi(opencl_platform_override);
        if(index < num_platforms) {
            platform_id = platform_list[index];
            platform_index = index;
        }
    }
    if(!platform_id) {
        platform_id = platform_list[0];
        for (cl_uint i = 0; i < num_platforms; i++) {
            char vendor[128] = { 0 };
            if ((status = clGetPlatformInfo(platform_list[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL)) != CL_SUCCESS) {
                fatal("clGetPlatformInfo([%d],...) => %d (failed)", i, status);
            }
            if (!strcmp(vendor, "Advanced Micro Devices, Inc.")) {
                platform_id = platform_list[i];
                platform_index = i;
                break;
            }
        }
    }
    delete [] platform_list;

    ////////
    /// \brief get GPU OpenCL devices
    ///
    size_t size;
    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, MAX_NUM_GPU, device_id, &num_devices);
    if (status != CL_SUCCESS) {
        fatal("clGetDeviceIDs(*,CL_DEVICE_TYPE_GPU,%d,...) => %d", MAX_NUM_GPU, status);
    }
    info("using OpenCL platform#%d with %d GPU devices ...", platform_index, num_devices);
#elif ENABLE_HIP
    hipError_t status;
    status = hipGetDeviceCount(&num_devices);
    if (status != hipSuccess) {
        fatal("ERROR: hipGetDeviceCount => %d (failed)\n", status);
    }
    if (num_devices < 1) {
        fatal("ERROR: didn't find any GPU!\n", status);
    }
    info("using HIP with %d GPU devices ...", num_devices);
#endif
    // set default config to use all GPUs
    numGPUs = num_devices;
    for(int gpuId = 0; gpuId < num_devices; gpuId++) {
        gpuIdList[gpuId] = gpuId;
    }

    ////////
    /// \brief load default configuration
    ///
    loadConfig();
    setConfigurationDir();
    useFp16Inference = 0;
    numDecThreads = 0;
}

Arguments::~Arguments()
{
#if ENABLE_OPENCL
    // release OpenCL resources
    for(int gpuId = 0; gpuId < num_devices; gpuId++) {
        if(device_id[gpuId]) {
            clReleaseDevice(device_id[gpuId]);
        }
    }
#elif ENABLE_HIP
    //todo for hip
#endif
}

void Arguments::setConfigurationDir()
{
    // generate configuration directory
    if(workFolder == "~") {
        configurationDir = getenv("HOME");
        configurationDir += "/.inference_server_app.dir";
    }
    else {
        configurationDir = workFolder;
    }
    // make sure that folders are created
    std::string uploadFolder = configurationDir + "/upload";
    mkdir(configurationDir.c_str(), 0700);
    mkdir(uploadFolder.c_str(), 0700);
}

void Arguments::loadConfig()
{
    ////////
    /// \brief get default configuration
    ///
    FILE * fp = fopen(configurationFile.c_str(), "r");
    if(fp) {
        bool valid = false;
        char version[256], workFolder_[256];
        int modelFileDownloadCounter_;
        int port_, batchSize_, maxPendingBatches_;
        int numGPUs_, gpuIdList_[MAX_NUM_GPU] = { 0 }, maxGpuId_;
        char password_[256] = { 0 };
        int n = fscanf(fp, "%s%s%d%d%d%d%d", version, workFolder_, &modelFileDownloadCounter_, &port_, &batchSize_, &maxPendingBatches_, &numGPUs_);
        if(n == 7 && !strcmp(version, BUILD_VERSION)) {
            if(numGPUs_ > num_devices) {
                warning("reseting GPUs to default as numGPUs(%d) exceeded num_devices(%d) in %s", numGPUs_, num_devices, configurationFile.c_str());
            }
            else {
                valid = true;
                maxGpuId_ = 0;
                for(int i = 0; i < numGPUs_; i++) {
                    n = fscanf(fp, "%d", &gpuIdList_[i]);
                    maxGpuId_ = std::max(maxGpuId_, gpuIdList_[i]);
                    if(n != 1 || maxGpuId_ >= num_devices) {
                        warning("reseting GPUs to default as gpuId entry is missing or not within num_devices(%d) in %s", num_devices, configurationFile.c_str());
                        valid = false;
                        break;
                    }
                }
                if(valid) {
                    if(fscanf(fp, "%s", password_) == 0) {
                        password_[0] = '\0';
                    }
                }
            }
        }
        fclose(fp);
        if(valid) {
            workFolder = workFolder_;
            modelFileDownloadCounter = modelFileDownloadCounter_;
            port = port_;
            batchSize = batchSize_;
            maxPendingBatches = maxPendingBatches_;
            numGPUs = numGPUs_;
            for(int i = 0; i < numGPUs; i++) {
                gpuIdList[i] = gpuIdList_[i];
            }
            maxGpuId = maxGpuId_;
            password = password_;
            // set configuration directory
            setConfigurationDir();
        }
    }
}

void Arguments::saveConfig()
{
    ////////
    /// \brief save default configuration
    ///
    FILE * fp = fopen(configurationFile.c_str(), "w");
    if(fp) {
        fprintf(fp, "%s\n", BUILD_VERSION);
        fprintf(fp, "%s\n", workFolder.c_str());
        fprintf(fp, "%d\n", modelFileDownloadCounter);
        fprintf(fp, "%d\n", port);
        fprintf(fp, "%d\n", batchSize);
        fprintf(fp, "%d\n", maxPendingBatches);
        fprintf(fp, "%d", numGPUs);
        for(int i = 0; i < numGPUs; i++) {
            fprintf(fp, " %d", gpuIdList[i]);
        }
        fprintf(fp, "\n");
        fprintf(fp, "%s\n", password.c_str());
        fclose(fp);
    }
}

void Arguments::getPreConfiguredModels()
{
    ////////
    /// \brief get preconfigured models
    ///
    DIR * dir = opendir(configurationDir.c_str());
    if(!dir) {
        fatal("unable to open folder: %s", configurationDir.c_str());
    }
    for(struct dirent * entry = readdir(dir); entry != nullptr; entry = readdir(dir)) {
        if((entry->d_type & DT_DIR) == DT_DIR && entry->d_name[0] != '.') {
            std::string annModuleConfigFile = configurationDir + "/" + entry->d_name + "/" + MODULE_CONFIG;
            FILE * fp = fopen(annModuleConfigFile.c_str(), "r");
            if(fp) {
                int dimInput[3], dimOutput[3];
                char name[64];
                int reverseInputChannelOrder;
                float preprocessMpy[3] = { 1, 1, 1 };
                float preprocessAdd[3] = { 0, 0, 0 };
                int n = fscanf(fp, "%s%d%d%d%d%d%d%d%g%g%g%g%g%g", name,
                               &dimInput[0], &dimInput[1], &dimInput[2],
                               &dimOutput[0], &dimOutput[1], &dimOutput[2],
                               &reverseInputChannelOrder,
                               &preprocessMpy[0], &preprocessMpy[1], &preprocessMpy[2],
                               &preprocessAdd[0], &preprocessAdd[1], &preprocessAdd[2]);
                if(n == 8 || n == 14) {
                    std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float,std::string>
                            ann(name, dimInput[0], dimInput[1], dimInput[2], dimOutput[0], dimOutput[1], dimOutput[2],
                                reverseInputChannelOrder,
                                preprocessMpy[0], preprocessMpy[1], preprocessMpy[2],
                                preprocessAdd[0], preprocessAdd[1], preprocessAdd[2],
                                entry->d_name);
                    configuredModels.push_back(ann);
                    info("found pre-configured model [name %s] [input %dx%dx%d] [output %dx%dx%d] [reverseInputChannelOrder %d] [mpy %g %g %g] [add %g %g %g] [folder %s]",
                            name, dimInput[2], dimInput[1], dimInput[0], dimOutput[2], dimOutput[1], dimOutput[0],
                            reverseInputChannelOrder,
                            preprocessMpy[0], preprocessMpy[1], preprocessMpy[2],
                            preprocessAdd[0], preprocessAdd[1], preprocessAdd[2],
                            entry->d_name);
                }
                fclose(fp);
            }
        }
    }
    closedir(dir);
}

//usage information
static void show_usage()
{
    printf("\n");
    printf("Usage:\n");
    printf("\tinference_server_app");
    printf("\t[-p \t<port>\t\t\t\t default:26262]\n");
    printf("\t\t\t\t[-b \t<batch size>\t\t\t default:64]\n");
    printf("\t\t\t\t[-n \t<model compiler path>\t\t default:/opt/rocm/libexec/mivisionx/model_compiler/python]\n");
    printf("\t\t\t\t[-fp16 \t<ON:1 or OFF:0>\t\t\t default:0]\n");
    printf("\t\t\t\t[-w \t<server working directory>\t default:~/]\n");
    printf("\t\t\t\t[-t \t<num cpu decoder threads [2-64]> default:1]\n");
    printf("\t\t\t\t[-gpu \t<comma separated list of GPUs>]\n");
    printf("\t\t\t\t[-q \t<max pending batches>]\n");
    printf("\t\t\t\t[-s \t<local shadow folder full path>]\n\n");
}

int Arguments::initializeConfig(int argc, char * argv[])
{
    ////////
    /// \brief process command-lines
    ///
    while(argc > 2) {
        if(!strcmp(argv[1], "-p")) {
            port = atoi(argv[2]);
            argc -= 2;
            argv += 2;
        }
        else if(!strcmp(argv[1], "-b")) {
            batchSize = atoi(argv[2]);
            argc -= 2;
            argv += 2;
        }
        else if(!strcmp(argv[1], "-gpu")) {
            numGPUs = 0;
            maxGpuId = 0;
            int gpuId = 0;
            for(const char * p = argv[2]; *p; p++) {
                if(*p >= '0' && *p <= '9') {
                    gpuId = gpuId * 10 + *p - '0';
                    if(gpuId >= num_devices) {
                        return error("GPU#%d is out-of-range. There are only %d devices in this system.", gpuId, num_devices);
                    }
                }
                else if(*p == ',' && numGPUs < MAX_NUM_GPU-1) {
                    gpuIdList[numGPUs++] = gpuId;
                    maxGpuId = std::max(maxGpuId, gpuId);
                    gpuId = 0;
                }
                else {
                    error("invalid GPU-list: %s", argv[2]);
                    show_usage();
                    return -1;
                }
            }
            gpuIdList[numGPUs++] = gpuId;
            argc -= 2;
            argv += 2;
        }
        else if(!strcmp(argv[1], "-q")) {
            maxPendingBatches = atoi(argv[2]);
            argc -= 2;
            argv += 2;
        }
        else if(!strcmp(argv[1], "-fp16")) {
            useFp16Inference = atoi(argv[2]);
            argc -= 2;
            argv += 2;
        }
        else if(!strcmp(argv[1], "-w")) {
            workFolder = argv[2];
            argc -= 2;
            argv += 2;
            // set configuration directory
            setConfigurationDir();
        }
        else if(!strcmp(argv[1], "-s")) {
            if (!strcmp(argv[2],"")) {
                error("invalid shadow folder name %s", argv[2]);
                return -1;
            }else {
                setLocalShadowRootDir(argv[2]);
                printf("Set shadow folder to %s\n", localShadowRootDir.c_str());
            }
            argc -= 2;
            argv += 2;
        }
        else if(!strcmp(argv[1], "-n")) {
            if (!strcmp(argv[2],"")) {
                error("invalid Model Compiler folder name %s", argv[2]);
                return -1;
            }else {
                setModelCompilerPath(argv[2]);
                printf("Set Model Compiler folder to %s\n", modelCompilerPath.c_str());
            }
            argc -= 2;
            argv += 2;
        }
        else if(!strcmp(argv[1], "-t")) {
            numDecThreads = atoi(argv[2]);
            if (numDecThreads < 2) numDecThreads=0;
            argc -= 2;
            argv += 2;
        }
        else
            break;
    }
    if(argc > 1) {
        if(strcmp(argv[1], "-h") != 0)
            error("invalid option: %s", argv[1]);
        show_usage();
        return -1;
    }

    ////////
    /// \brief get pre-configured models
    ///
    getPreConfiguredModels();

    ////////
    /// \brief save configuration
    ///
    saveConfig();

    return 0;
}

#if ENABLE_OPENCL
int Arguments::lockGpuDevices(int GPUs, cl_device_id * device_id_)
{
    std::lock_guard<std::mutex> lock(mutex);

    // make sure number of devices are available
    if(GPUs > numGPUs)
        return -1;
    int deviceAvail = 0;
    for(int i = 0; i < num_devices; i++) {
        if(deviceUseCount[i] < MAX_DEVICE_USE_LIMIT)
            deviceAvail++;
    }
    if(deviceAvail < GPUs)
            return -1;

    // allocate devices
    for(int i = 0; i < GPUs; i++) {
        int gpuId = 0;
        for(int j = 1; j < num_devices; j++) {
            if(deviceUseCount[gpuId] > deviceUseCount[j])
                gpuId = j;
        }
        deviceUseCount[gpuId]++;
        device_id_[i] = device_id[gpuId];
    }

    return 0;
}
void Arguments::releaseGpuDevices(int GPUs, const cl_device_id * device_id_)
{
    std::lock_guard<std::mutex> lock(mutex);

    for(int i = 0; i < GPUs; i++) {
        for(int gpuId = 0; gpuId < num_devices; gpuId++) {
            if(device_id_[i] == device_id[gpuId]) {
                deviceUseCount[gpuId]--;
                break;
            }
        }
    }
}

#elif ENABLE_HIP
int Arguments::lockGpuDevices(int GPUs, int * device_id_)
{
    std::lock_guard<std::mutex> lock(mutex);

    // make sure number of devices are available
    if(GPUs > numGPUs)
        return -1;
    int deviceAvail = 0;
    for(int i = 0; i < num_devices; i++) {
        if(deviceUseCount[i] < MAX_DEVICE_USE_LIMIT)
            deviceAvail++;
    }
    if(deviceAvail < GPUs)
            return -1;

    // allocate devices
    for(int i = 0; i < GPUs; i++) {
        int gpuId = 0;
        for(int j = 1; j < num_devices; j++) {
            if(deviceUseCount[gpuId] > deviceUseCount[j])
                gpuId = j;
        }
        deviceUseCount[gpuId]++;
        device_id_[i] = device_id[gpuId];
    }

    return 0;
}

void Arguments::releaseGpuDevices(int GPUs, const int * device_id_)
{
    std::lock_guard<std::mutex> lock(mutex);

    for(int i = 0; i < GPUs; i++) {
        for(int gpuId = 0; gpuId < num_devices; gpuId++) {
            if(device_id_[i] == device_id[gpuId]) {
                deviceUseCount[gpuId]--;
                break;
            }
        }
    }
}
#endif
