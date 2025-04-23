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

#include "inference.h"
#include "netutil.h"
#include "common.h"
#include <thread>
#include <chrono>
#include <dlfcn.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <numeric>

#if USE_SSE_OPTIMIZATION
#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <immintrin.h>
#endif
#endif

extern void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[]);

#if ENABLE_HIP
InferenceEngineHip::InferenceEngineHip(int sock_, Arguments * args_, const std::string clientName_, InfComCommand * cmd)
                  :InferenceEngine(sock_, args_, clientName_, cmd),
                  hip_dev_prop{ nullptr }, hip_stream{ nullptr },
                  queueDeviceInputMemIdle{ nullptr }, queueDeviceInputMemBusy{ nullptr },
                  queueDeviceOutputMemIdle{ nullptr }, queueDeviceOutputMemBusy{ nullptr }
{
  device_id[MAX_NUM_GPU-1] = {-1};
  if(!args->lockGpuDevices(GPUs, device_id))
    deviceLockSuccess = true;
}

InferenceEngineHip::~InferenceEngineHip()
{
#if INFERENCE_SCHEDULER_MODE == NO_INFERENCE_SCHEDULER && !DONOT_RUN_INFERENCE
    if(openvx_graph) {
        vxReleaseGraph(&openvx_graph);
    }
    if(openvx_input) {
        vxReleaseTensor(&openvx_input);
    }
    if(openvx_output) {
        vxReleaseTensor(&openvx_output);
    }
    if(openvx_context) {
        vxReleaseContext(&openvx_context);
    }
#elif INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER        
    // wait for all threads to complete and release all resources
    std::tuple<int,char*,int> endOfSequenceInput(-1,nullptr,0);
    inputQ.enqueue(endOfSequenceInput);
    if(threadMasterInputQ && threadMasterInputQ->joinable()) {
        threadMasterInputQ->join();
    }
    std::tuple<char*,int> endOfSequenceImage(nullptr,0);
    int endOfSequenceTag = -1;
    for(int i = 0; i < GPUs; i++) {
        if(queueDeviceTagQ[i]) {
            queueDeviceTagQ[i]->enqueue(endOfSequenceTag);
        }
        if(queueDeviceImageQ[i]) {
            queueDeviceImageQ[i]->enqueue(endOfSequenceImage);
        }
        if(threadDeviceInputCopy[i] && threadDeviceInputCopy[i]->joinable()) {
            threadDeviceInputCopy[i]->join();
        }
        if(threadDeviceProcess[i] && threadDeviceProcess[i]->joinable()) {
            threadDeviceProcess[i]->join();
        }
        if(threadDeviceOutputCopy[i] && threadDeviceOutputCopy[i]->joinable()) {
            threadDeviceOutputCopy[i]->join();
        }
        while(queueDeviceInputMemIdle[i] && queueDeviceInputMemIdle[i]->size() > 0) {
            std::pair<void *, void*> image;
            queueDeviceInputMemIdle[i]->dequeue(image);
            hipHostFree(image.first);
        }
        while(queueDeviceOutputMemIdle[i] && queueDeviceOutputMemIdle[i]->size() > 0) {
            std::pair<void *, void*> image;
            queueDeviceOutputMemIdle[i]->dequeue(image);
            hipFree(image.first);
        }
        if(queueDeviceTagQ[i]) {
            delete queueDeviceTagQ[i];
        }
        if(queueDeviceImageQ[i]) {
            delete queueDeviceImageQ[i];
        }
        if(queueDeviceInputMemIdle[i]) {
            delete queueDeviceInputMemIdle[i];
        }
        if(queueDeviceInputMemBusy[i]) {
            delete queueDeviceInputMemBusy[i];
        }
        if(queueDeviceOutputMemIdle[i]) {
            delete queueDeviceOutputMemIdle[i];
        }
        if(queueDeviceOutputMemBusy[i]) {
            delete queueDeviceOutputMemBusy[i];
        }
        if(openvx_graph[i]) {
            vxReleaseGraph(&openvx_graph[i]);
        }
        if(openvx_input[i]) {
            vxReleaseTensor(&openvx_input[i]);
        }
        if(openvx_output[i]) {
            vxReleaseTensor(&openvx_output[i]);
        }
        if(openvx_context[i]) {
            vxReleaseContext(&openvx_context[i]);
        }
        if(hip_stream[i]) {
          if (hipStreamDestroy(hip_stream[i]) != hipSuccess)
              error("hipStreamDestroy failed");
          device_id[i] = -1;
        }
    }
#endif
    // release all device resources
    if(deviceLockSuccess) {
        args->releaseGpuDevices(GPUs, device_id);
    }
    if(moduleHandle) {
        dlclose(moduleHandle);
    }
    if (region) delete region;
    PROFILER_SHUTDOWN();
}

int InferenceEngineHip::run()
{
    //////
    /// make device lock is successful
    ///
    if(!deviceLockSuccess) {
        return error_close(sock, "could not lock %d GPUs devices for inference request from %s", GPUs, clientName.c_str());
    }

    //////
    /// check if server and client are in the same mode for data
    ///
    if (receiveFileNames && !useShadowFilenames)
    {
        return error_close(sock, "client is sending filenames but server is not configured with shadow folder\n");
    }

    //////
    /// check if client is requesting topK which is not supported
    ///
    if (topK > 5)
    {
        return error_close(sock, "Number of topK confidances: %d not supported\n", topK);
    }

    //////
    /// check for model validity
    ///
    bool found = false;
    for(size_t i = 0; i < args->getNumConfigureddModels(); i++) {
        std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float,std::string> info = args->getConfiguredModelInfo(i);
        if(std::get<0>(info) == modelName &&
           std::get<1>(info) == dimInput[0] &&
           std::get<2>(info) == dimInput[1] &&
           std::get<3>(info) == dimInput[2] &&
           std::get<4>(info) == dimOutput[0] &&
           std::get<5>(info) == dimOutput[1] &&
           std::get<6>(info) == dimOutput[2])
        {
            reverseInputChannelOrder = std::get<7>(info);
            preprocessMpy[0] = std::get<8>(info);
            preprocessMpy[1] = std::get<9>(info);
            preprocessMpy[2] = std::get<10>(info);
            preprocessAdd[0] = std::get<11>(info);
            preprocessAdd[1] = std::get<12>(info);
            preprocessAdd[2] = std::get<13>(info);
            modelPath = args->getConfigurationDir() + "/" + std::get<14>(info);
            found = true;
            break;
        }
    }
    if(!found) {
        for(size_t i = 0; i < args->getNumUploadedModels(); i++) {
            std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float> info = args->getUploadedModelInfo(i);
            if(std::get<0>(info) == modelName &&
               std::get<1>(info) == dimInput[0] &&
               std::get<2>(info) == dimInput[1] &&
               std::get<3>(info) == dimInput[2] &&
               std::get<4>(info) == dimOutput[0] &&
               std::get<5>(info) == dimOutput[1] &&
               std::get<6>(info) == dimOutput[2])
            {
                reverseInputChannelOrder = std::get<7>(info);
                preprocessMpy[0] = std::get<8>(info);
                preprocessMpy[1] = std::get<9>(info);
                preprocessMpy[2] = std::get<10>(info);
                preprocessAdd[0] = std::get<11>(info);
                preprocessAdd[1] = std::get<12>(info);
                preprocessAdd[2] = std::get<13>(info);
                modelPath = args->getConfigurationDir() + "/" + modelName;
                found = true;
                break;
            }
        }
    }
    if(found) {
        modulePath = modelPath + "/build/" + MODULE_LIBNAME;
        moduleHandle = dlopen(modulePath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if(!moduleHandle) {
            found = false;
            error("could not locate module %s for %s", modulePath.c_str(), clientName.c_str());
        }
        if (args->getModelCompilerPath().empty()) {
            if(!(annCreateGraph = (type_annCreateGraph *) dlsym(moduleHandle, "annCreateGraph"))) {
                found = false;
                error("could not find function annCreateGraph() in module %s for %s", modulePath.c_str(), clientName.c_str());
            }
        }
        else if(!(annAddtoGraph = (type_annAddToGraph *) dlsym(moduleHandle, "annAddToGraph"))) {
            found = false;
            error("could not find function annAddToGraph() in module %s for %s", modulePath.c_str(), clientName.c_str());
        }
    }
    else {
        error("unable to find requested model:%s input:%dx%dx%d output:%dx%dx%d from %s", modelName.c_str(),
              dimInput[2], dimInput[1], dimInput[0], dimOutput[2], dimOutput[1], dimOutput[0], clientName.c_str());
    }
    if(!found) {
        // send and wait for INFCOM_CMD_DONE message
        InfComCommand reply = {
            INFCOM_MAGIC, INFCOM_CMD_DONE, { 0 }, { 0 }
        };
        ERRCHK(sendCommand(sock, reply, clientName));
        ERRCHK(recvCommand(sock, reply, clientName, INFCOM_CMD_DONE));
        close(sock);
        return -1;
    }
    info("found requested model:%s input:%dx%dx%d output:%dx%dx%d from %s", modelName.c_str(),
          dimInput[2], dimInput[1], dimInput[0], dimOutput[2], dimOutput[1], dimOutput[0], clientName.c_str());

    // send and wait for INFCOM_CMD_INFERENCE_INITIALIZATION message
    InfComCommand updateCmd = {
        INFCOM_MAGIC, INFCOM_CMD_INFERENCE_INITIALIZATION, { 0 }, "started initialization"
    };
    ERRCHK(sendCommand(sock, updateCmd, clientName));
    ERRCHK(recvCommand(sock, updateCmd, clientName, INFCOM_CMD_INFERENCE_INITIALIZATION));
    info(updateCmd.message);

#if INFERENCE_SCHEDULER_MODE == NO_INFERENCE_SCHEDULER
#if DONOT_RUN_INFERENCE
    info("InferenceEngine: using NO_INFERENCE_SCHEDULER and DONOT_RUN_INFERENCE");
#else
    { // create OpenVX resources
        info("InferenceEngine: using NO_INFERENCE_SCHEDULER");
        vx_status status;
        openvx_context = vxCreateContext();
        if((status = vxGetStatus((vx_reference)openvx_context)) != VX_SUCCESS)
            fatal("InferenceEngine: vxCreateContext() failed (%d)", status);
        vx_size idim[4] = { (vx_size)dimInput[0], (vx_size)dimInput[1], (vx_size)dimInput[2], (vx_size)batchSize };
        vx_size odim[4] = { (vx_size)dimOutput[0], (vx_size)dimOutput[1], (vx_size)dimOutput[2], (vx_size)batchSize };
        openvx_input = vxCreateTensor(openvx_context, 4, idim, VX_TYPE_FLOAT32, 0);
        openvx_output = vxCreateTensor(openvx_context, 4, odim, VX_TYPE_FLOAT32, 0);
        if((status = vxGetStatus((vx_reference)openvx_input)) != VX_SUCCESS)
            fatal("InferenceEngine: vxCreateTensor(input) failed (%d)", status);
        if((status = vxGetStatus((vx_reference)openvx_output)) != VX_SUCCESS)
            fatal("InferenceEngine: vxCreateTensor(output) failed (%d)", status);
        //////
        // load the model
        openvx_graph = annCreateGraph(openvx_context, openvx_input, openvx_output, modelPath.c_str());
        if((status = vxGetStatus((vx_reference)openvx_graph)) != VX_SUCCESS)
            fatal("InferenceEngine: annCreateGraph() failed (%d)", status);

        // send and wait for INFCOM_CMD_INFERENCE_INITIALIZATION message
        updateCmd.data[0] = 80;
        sprintf(updateCmd.message, "completed OpenVX graph");
        ERRCHK(sendCommand(sock, updateCmd, clientName));
        ERRCHK(recvCommand(sock, updateCmd, clientName, INFCOM_CMD_INFERENCE_INITIALIZATION));
        info(updateCmd.message);
    }
#endif
#elif INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
    info("InferenceEngine: using LIBRE_INFERENCE_SCHEDULER");
    //////
    /// allocate OpenVX and HIP resources
    /// 
    for(int gpu = 0; gpu < GPUs; gpu++) {
        //////
        // Initialize hip
        hipError_t err = hipInit(0);
        if (err != hipSuccess) {
            fatal("ERROR: hipInit(0) => (#%d,%d) failed", gpu, err);
        }
        // initialize HIP device for rocAL
        int hip_num_devices = -1;
        err = hipGetDeviceCount(&hip_num_devices);
        if (err != hipSuccess) {
            fatal("ERROR: hipGetDeviceCount() => (#%d,%d) failed",  hip_num_devices, err);
        }
        //////
        // create OpenVX context
        vx_status status;
        openvx_context[gpu] = vxCreateContext();
        if((status = vxGetStatus((vx_reference)openvx_context[gpu])) != VX_SUCCESS)
            fatal("InferenceEngine: vxCreateContext(#%d) failed (%d)", gpu, status);
        //set the device for context if specified.
        if (gpu < hip_num_devices) {
            int hipDevice = gpu;
            if((status = vxSetContextAttribute(openvx_context[gpu],
                    VX_CONTEXT_ATTRIBUTE_AMD_HIP_DEVICE,
                    &hipDevice, sizeof(hipDevice)) != VX_SUCCESS))
                fatal("vxSetContextAttribute for hipDevice(#%d, %d) failed ", hipDevice, status);
        }else
            fatal("ERROR: HIP Device(%d) out of range %d", gpu);

        // create scheduler device queues
#if  USE_ADVANCED_MESSAGE_Q
        queueDeviceTagQ[gpu] = new MessageQueueAdvanced<int>(MAX_DEVICE_QUEUE_DEPTH);
        queueDeviceImageQ[gpu] = new MessageQueueAdvanced<std::tuple<char*,int>>(MAX_INPUT_QUEUE_DEPTH);
#else
        queueDeviceTagQ[gpu] = new MessageQueue<int>();
        queueDeviceTagQ[gpu]->setMaxQueueDepth(MAX_DEVICE_QUEUE_DEPTH);
        queueDeviceImageQ[gpu] = new MessageQueue<std::tuple<char*,int>>();
#endif
        queueDeviceInputMemIdle[gpu] = new MessageQueue<std::pair<void *, void *>>();
        queueDeviceInputMemBusy[gpu] = new MessageQueue<std::pair<void *, void *>>();
        queueDeviceOutputMemIdle[gpu] = new MessageQueue<std::pair<void *, void *>>();
        queueDeviceOutputMemBusy[gpu] = new MessageQueue<std::pair<void *, void *>>();

        // create HIP buffers for input/output and add them to queueDeviceInputMemIdle/queueDeviceOutputMemIdle
        void* memInput, *hostmemI, *memOutput, *hostmemO;
        for(int i = 0; i < INFERENCE_PIPE_QUEUE_DEPTH; i++) {
            memInput = nullptr, hostmemI = nullptr, memOutput = nullptr, hostmemO = nullptr;
            hipError_t err = hipHostMalloc((void **)&hostmemI, inputSizeInBytes, hipHostMallocDefault);
            if(err != hipSuccess || !hostmemI)
            {
                fatal("InferenceEngine:hipHostMalloc of size of size %d failed<%d>", inputSizeInBytes, err);
            }
            if (hipHostGetDevicePointer((void **)&memInput, hostmemI, 0)  != hipSuccess)
            {
                fatal("InferenceEngine:hipHostGetDevicePointer of size %d failed \n", inputSizeInBytes);
            }
            err = hipHostMalloc((void **)&hostmemO, outputSizeInBytes, hipHostMallocDefault);
            if(err != hipSuccess || !hostmemO)
            {
                fatal("InferenceEngine:hipHostMalloc of size of size %d failed<%d>", outputSizeInBytes, err);
            }
            if (hipHostGetDevicePointer((void **)&memOutput, hostmemO, 0)  != hipSuccess)
            {
                fatal("InferenceEngine:hipHostGetDevicePointer of size %d failed \n", outputSizeInBytes);
            }
            queueDeviceInputMemIdle[gpu]->enqueue(std::make_pair(memInput, hostmemI));
            queueDeviceOutputMemIdle[gpu]->enqueue(std::make_pair(memOutput, hostmemO));
        }
        memInput = nullptr, memOutput = nullptr;
        vx_size idim[4] = { (vx_size)dimInput[0], (vx_size)dimInput[1], (vx_size)dimInput[2], (vx_size)batchSize };
        vx_size odim[4] = { (vx_size)dimOutput[0], (vx_size)dimOutput[1], (vx_size)dimOutput[2], (vx_size)batchSize };
        if (useFp16) {
            vx_size istride[4] = { 2, (vx_size)2 * dimInput[0], (vx_size)2 * dimInput[0] * dimInput[1], (vx_size)2 * dimInput[0] * dimInput[1] * dimInput[2] };
            vx_size ostride[4] = { 2, (vx_size)2 * dimOutput[0], (vx_size)2 * dimOutput[0] * dimOutput[1], (vx_size)2 * dimOutput[0] * dimOutput[1] * dimOutput[2] };
            openvx_input[gpu] = vxCreateTensorFromHandle(openvx_context[gpu], 4, idim, VX_TYPE_FLOAT16, 0, istride, memInput, VX_MEMORY_TYPE_HIP);
            openvx_output[gpu] = vxCreateTensorFromHandle(openvx_context[gpu], 4, odim, VX_TYPE_FLOAT16, 0, ostride, memOutput, VX_MEMORY_TYPE_HIP);
            if (openvx_output[gpu] == nullptr)
                printf(" vxCreateTensorFromHandle(output) failed for gpu#%d\n", gpu);
        } else {
            vx_size istride[4] = { 4, (vx_size)4 * dimInput[0], (vx_size)4 * dimInput[0] * dimInput[1], (vx_size)4 * dimInput[0] * dimInput[1] * dimInput[2] };
            vx_size ostride[4] = { 4, (vx_size)4 * dimOutput[0], (vx_size)4 * dimOutput[0] * dimOutput[1], (vx_size)4 * dimOutput[0] * dimOutput[1] * dimOutput[2] };
            openvx_input[gpu] = vxCreateTensorFromHandle(openvx_context[gpu], 4, idim, VX_TYPE_FLOAT32, 0, istride, memInput, VX_MEMORY_TYPE_HIP);
            openvx_output[gpu] = vxCreateTensorFromHandle(openvx_context[gpu], 4, odim, VX_TYPE_FLOAT32, 0, ostride, memOutput, VX_MEMORY_TYPE_HIP);
        }
        if((status = vxGetStatus((vx_reference)openvx_input[gpu])) != VX_SUCCESS)
            fatal("InferenceEngine: vxCreateTensorFromHandle(input#%d) failed (%d)", gpu, status);
        if((status = vxGetStatus((vx_reference)openvx_output[gpu])) != VX_SUCCESS)
            fatal("InferenceEngine: vxCreateTensorFromHandle(output#%d) failed (%d)", gpu, status);

        //////
        // load the model
        if (annCreateGraph != nullptr) {
            openvx_graph[gpu] = annCreateGraph(openvx_context[gpu], openvx_input[gpu], openvx_output[gpu], modelPath.c_str());
            if((status = vxGetStatus((vx_reference)openvx_graph[gpu])) != VX_SUCCESS)
                fatal("InferenceEngine: annCreateGraph(#%d) failed (%d)", gpu, status);
        }
        else if (annAddtoGraph != nullptr) {
            std::string weightsFile = modelPath + "/weights.bin";
            vxRegisterLogCallback(openvx_context[gpu], log_callback, vx_false_e);
            openvx_graph[gpu] = vxCreateGraph(openvx_context[gpu]);
            status = vxGetStatus((vx_reference)openvx_graph[gpu]);
            if(status) {
                fatal("InferenceEngine: vxCreateGraph(#%d) failed (%d)", gpu, status);
                return -1;
            }
            status = annAddtoGraph(openvx_graph[gpu], openvx_input[gpu], openvx_output[gpu], weightsFile.c_str());
            if(status) {
                fatal("InferenceEngine: annAddToGraph(#%d) failed (%d)", gpu, status);
                return -1;
            }
        }

        // send and wait for INFCOM_CMD_INFERENCE_INITIALIZATION message
        updateCmd.data[0] = 80 * (gpu + 1) / GPUs;
        sprintf(updateCmd.message, "completed OpenVX graph for GPU#%d", gpu);
        ERRCHK(sendCommand(sock, updateCmd, clientName));
        ERRCHK(recvCommand(sock, updateCmd, clientName, INFCOM_CMD_INFERENCE_INITIALIZATION));
        info(updateCmd.message);
    }
#endif

    //////
    /// start scheduler threads
    ///
#if INFERENCE_SCHEDULER_MODE == NO_INFERENCE_SCHEDULER
    // nothing to do
#elif INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
    threadMasterInputQ = new std::thread(&InferenceEngineHip::workMasterInputQ, this);
    for(int gpu = 0; gpu < GPUs; gpu++) {
        threadDeviceInputCopy[gpu] = new std::thread(&InferenceEngineHip::workDeviceInputCopy, this, gpu);
        threadDeviceProcess[gpu] = new std::thread(&InferenceEngineHip::workDeviceProcess, this, gpu);
        threadDeviceOutputCopy[gpu] = new std::thread(&InferenceEngineHip::workDeviceOutputCopy, this, gpu);
    }
#endif

    // send and wait for INFCOM_CMD_INFERENCE_INITIALIZATION message
    updateCmd.data[0] = 100;
    sprintf(updateCmd.message, "inference engine is ready");
    ERRCHK(sendCommand(sock, updateCmd, clientName));
    ERRCHK(recvCommand(sock, updateCmd, clientName, INFCOM_CMD_INFERENCE_INITIALIZATION));
    info(updateCmd.message);

    ////////
    /// \brief keep running the inference in loop
    ///
    bool endOfImageRequested = false;
    for(bool endOfSequence = false; !endOfSequence; ) {
        bool didSomething = false;

        // send all the available results to the client
        int resultCountAvailable = outputQ.size();
        if(resultCountAvailable > 0) {
            didSomething = true;
            while(resultCountAvailable > 0) {
                if (!detectBoundingBoxes){
                    if (topK < 1){
                        int resultCount = std::min(resultCountAvailable, (INFCOM_MAX_IMAGES_FOR_TOP1_PER_PACKET/2));
                        InfComCommand cmd = {
                            INFCOM_MAGIC, INFCOM_CMD_INFERENCE_RESULT, { resultCount, 0 }, { 0 }
                        };
                        for(int i = 0; i < resultCount; i++) {
                            std::tuple<int,int> result;
                            outputQ.dequeue(result);
                            int tag = std::get<0>(result);
                            int label = std::get<1>(result);
                            if(tag < 0) {
                                endOfSequence = true;
                                resultCount = i;
                                break;
                            }
                            cmd.data[2 + i * 2 + 0] = tag; // tag
                            cmd.data[2 + i * 2 + 1] = label; // label
                        }
                        if(resultCount > 0) {
                            cmd.data[0] = resultCount;
                            ERRCHK(sendCommand(sock, cmd, clientName));
                            resultCountAvailable -= resultCount;
                            ERRCHK(recvCommand(sock, cmd, clientName, INFCOM_CMD_INFERENCE_RESULT));
                        }
                        if(endOfSequence) {
                            break;
                        }
                    }else {
                        // send topK labels
                        int maxResults = INFCOM_MAX_IMAGES_FOR_TOP1_PER_PACKET/(topK+1);
                        int resultCount = std::min(resultCountAvailable, maxResults);
                        InfComCommand cmd = {
                            INFCOM_MAGIC, INFCOM_CMD_TOPK_INFERENCE_RESULT, { resultCount, topK }, { 0 }
                        };
                        for(int i = 0; i < resultCount; i++) {
                            std::tuple<int,int> result;
                            std::vector<unsigned int> labels;
                            outputQ.dequeue(result);
                            int tag = std::get<0>(result);
                            if(tag < 0) {
                                endOfSequence = true;
                                resultCount = i;
                                break;
                            }
                            outputQTopk.dequeue(labels);
                            cmd.data[2 + i * (topK+1) + 0] = tag; // tag
                            for (int j=0; j<topK; j++){
                                cmd.data[3 + i * (topK+1) + j] = labels[j]; // label[j]
                            }
                            labels.clear();
                        }
                        if(resultCount > 0) {
                            cmd.data[0] = resultCount;
                            ERRCHK(sendCommand(sock, cmd, clientName));
                            resultCountAvailable -= resultCount;
                            ERRCHK(recvCommand(sock, cmd, clientName, INFCOM_CMD_TOPK_INFERENCE_RESULT));
                        }
                        if(endOfSequence) {
                            break;
                        }
                    }
                }else
                {
                    // Dequeue the bounding box
                    std::tuple<int,int> result;
                    std::vector<ObjectBB> bounding_boxes;
                    outputQ.dequeue(result);
                    int tag = std::get<0>(result);
                    int label = std::get<1>(result);        // label of first bounding box
                    if(tag < 0) {
                        endOfSequence = true;
                        resultCountAvailable--;
                        break;
                    }else
                    {
                        int numBB = 0;
                        int numMessages = 0;
                        if (label >= 0) {
                            OutputQBB.dequeue(bounding_boxes);
                            numBB = bounding_boxes.size();
                            if (numBB) numMessages = numBB/3;   // max 3 bb per mesasge
                            if (numBB % 3) numMessages++;
                        }
                        if (!numBB) {
                            InfComCommand cmd = {
                                INFCOM_MAGIC, INFCOM_CMD_BB_INFERENCE_RESULT, { tag, 0 }, { 0 }        // no bb detected
                            };
                            ERRCHK(sendCommand(sock, cmd, clientName));
                            ERRCHK(recvCommand(sock, cmd, clientName, INFCOM_CMD_BB_INFERENCE_RESULT));
                        } else
                        {
                            ObjectBB *pObj= &bounding_boxes[0];
                            for (int i=0, j=0; (i < numMessages && j < numBB); i++) {
                                int numBB_per_message = std::min((numBB-j), 3);
                                int bb_info = (numBB_per_message & 0xFFFF) | (numBB << 16);
                                InfComCommand cmd = {
                                    INFCOM_MAGIC, INFCOM_CMD_BB_INFERENCE_RESULT, { tag, bb_info }, { 0 }        // 3 bounding boxes in one message
                                };
                                cmd.data[2] = (unsigned int)((pObj->y*0x7FFF)+0.5)<<16  | (unsigned int)((pObj->x*0x7FFF)+0.5);
                                cmd.data[3] = (unsigned int)((pObj->h*0x7FFF)+0.5)<<16  | (unsigned int)((pObj->w*0x7FFF)+0.5);
                                cmd.data[4] = (unsigned int) ((pObj->confidence*0x3FFFFFFF)+0.5);    // convert float to Q30.1
                                cmd.data[5] = pObj->label;
                                pObj++;
                                if (numBB_per_message > 1) {
                                    cmd.data[6] = (unsigned int)((pObj->y*0x7FFF)+0.5)<<16  | (unsigned int)((pObj->x*0x7FFF)+0.5);
                                    cmd.data[7] = (unsigned int)((pObj->h*0x7FFF)+0.5)<<16  | (unsigned int)((pObj->w*0x7FFF)+0.5);
                                    cmd.data[8] = (unsigned int) ((pObj->confidence*0x3FFFFFFF)+0.5);    // convert float to Q30.1
                                    cmd.data[9] = pObj->label;
                                    pObj++;
                                }
                                if (numBB_per_message > 2) {
                                    cmd.data[10] = (unsigned int)((pObj->y*0x7FFF)+0.5)<<16  | (unsigned int)((pObj->x*0x7FFF)+0.5);
                                    cmd.data[11] = (unsigned int)((pObj->h*0x7FFF)+0.5)<<16  | (unsigned int)((pObj->w*0x7FFF)+0.5);
                                    cmd.data[12] = (unsigned int) ((pObj->confidence*0x3FFFFFFF)+0.5);    // convert float to Q30.1;
                                    cmd.data[13] = pObj->label;
                                    pObj++;
                                }
                                ERRCHK(sendCommand(sock, cmd, clientName));
                                ERRCHK(recvCommand(sock, cmd, clientName, INFCOM_CMD_BB_INFERENCE_RESULT));
                                j += numBB_per_message;
                            }
                        }
                        resultCountAvailable--;
                    }
                    bounding_boxes.clear();
                }
            }
        }

        // if not endOfImageRequested, request client to send images
        if(!endOfImageRequested) {
            // get number of empty slots in the input queue
            int imageCountRequested = 0;
#if INFERENCE_SCHEDULER_MODE == NO_INFERENCE_SCHEDULER
            imageCountRequested = 1;
#elif INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
            imageCountRequested = MAX_INPUT_QUEUE_DEPTH - inputQ.size();
#endif
            if(imageCountRequested > 0) {
                didSomething = true;
                // send request for upto INFCOM_MAX_IMAGES_PER_PACKET images
                imageCountRequested = std::min(imageCountRequested, (INFCOM_MAX_IMAGES_FOR_TOP1_PER_PACKET/2));
                InfComCommand cmd = {
                    INFCOM_MAGIC, INFCOM_CMD_SEND_IMAGES, { imageCountRequested }, { 0 }
                };
                ERRCHK(sendCommand(sock, cmd, clientName));
                ERRCHK(recvCommand(sock, cmd, clientName, INFCOM_CMD_SEND_IMAGES));

                // check of endOfImageRequested and receive images one at a time
                int imageCountReceived = cmd.data[0];
                if(imageCountReceived < 0) {
                    // submit the endOfSequence indicator to scheduler
#if INFERENCE_SCHEDULER_MODE == NO_INFERENCE_SCHEDULER
                    endOfSequence = true;
#elif INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
                    inputQ.enqueue(std::tuple<int,char*,int>(-1,nullptr,0));
#endif
                    endOfImageRequested = true;
                }
                int i = 0;
                for(; i < imageCountReceived; i++) {
                    // get header with tag and size info
                    int header[2] = { 0, 0 };
                    ERRCHK(recvBuffer(sock, &header, sizeof(header), clientName));
                    int tag = header[0];
                    int size = header[1];
                    // do sanity check with unreasonable parameters
                    if(tag < 0 || size <= 0 || size > 50000000) {
                        return error_close(sock, "invalid (tag:%d,size:%d) from %s", tag, size, clientName.c_str());
                    }
                    char * byteStream = 0;
                    if (receiveFileNames)
                    {
                        std::string fileNameDir = args->getlocalShadowRootDir() + "/";
                        char * buff = new char [size];
                        ERRCHK(recvBuffer(sock, buff, size, clientName));
                        fileNameDir.append(std::string(buff, size));
                        FILE * fp = fopen(fileNameDir.c_str(), "rb");
                        if(!fp) {
                            return error_close(sock, "filename %s (incorrect)", fileNameDir.c_str());
                        }
                        fseek(fp,0,SEEK_END);
                        int fsize = ftell(fp);
                        fseek(fp,0,SEEK_SET);
                        byteStream = new char [fsize];
                        size = (int)fread(byteStream, 1, fsize, fp);
                        fclose(fp);
                        delete[] buff;
                        if (size != fsize) {
                            return error_close(sock, "error reading %d bytes from file:%s", fsize, fileNameDir.c_str());
                        }
                    }
                    else
                    {
                        // allocate and receive the image and EOF marker
                        byteStream = new char [size];
                        ERRCHK(recvBuffer(sock, byteStream, size, clientName));
                    }
                    int eofMarker = 0;
                    ERRCHK(recvBuffer(sock, &eofMarker, sizeof(eofMarker), clientName));
                    if(eofMarker != INFCOM_EOF_MARKER) {
                        return error_close(sock, "eofMarker 0x%08x (incorrect)", eofMarker);
                    }

#if INFERENCE_SCHEDULER_MODE == NO_INFERENCE_SCHEDULER
#if DONOT_RUN_INFERENCE
                    // consume the input immediately since there is no scheduler
                    // simulate the input (tag,byteStream,size) processing using a 4ms sleep
                    int label = tag % dimOutput[2];
                    std::this_thread::sleep_for(std::chrono::milliseconds(4));
                    // release byteStream and keep the results in outputQ
                    delete[] byteStream;
                    outputQ.enqueue(std::tuple<int,int>(tag,label));
#else
                    // process the input immediately since there is no scheduler
                    // decode, scale, and format convert into the OpenVX input buffer
                    vx_map_id map_id;
                    vx_size stride[4];
                    float * ptr = nullptr;
                    vx_status status;
                    status = vxMapTensorPatch(openvx_input, 4, NULL, NULL, &map_id, stride, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
                    if(status != VX_SUCCESS) {
                        fatal("workDeviceProcess: vxMapTensorPatch(input)) failed(%d)", status);
                    }
                    DecodeScaleAndConvertToTensor(dimInput[0], dimInput[1], size, (unsigned char *)byteStream, ptr, useFp16);
                    status = vxUnmapTensorPatch(openvx_input, map_id);
                    if(status != VX_SUCCESS) {
                        fatal("workDeviceProcess: vxUnmapTensorPatch(input)) failed(%d)", status);
                    }
                    // process the graph
                    status = vxProcessGraph(openvx_graph);
                    if(status != VX_SUCCESS) {
                        fatal("workDeviceProcess: vxProcessGraph()) failed(%d)", status);
                    }
                    ptr = nullptr;
                    status = vxMapTensorPatch(openvx_output, 4, NULL, NULL, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
                    if(status != VX_SUCCESS) {
                        fatal("workDeviceProcess: vxMapTensorPatch(output)) failed(%d)", status);
                    }
                    int label = 0;
                    float max_prob = ptr[0];
                    for(int c = 1; c < dimOutput[2]; c++) {
                        float prob = ptr[c];
                        if(prob > max_prob) {
                            label = c;
                            max_prob = prob;
                        }
                    }
                    status = vxUnmapTensorPatch(openvx_output, map_id);
                    if(status != VX_SUCCESS) {
                        fatal("workDeviceProcess: vxUnmapTensorPatch(output)) failed(%d)", status);
                    }
                    // release byteStream and keep the results in outputQ
                    delete[] byteStream;
                    outputQ.enqueue(std::tuple<int,int>(tag,label));
#endif
#elif INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
                    // submit the input (tag,byteStream,size) to scheduler
                    inputQ.enqueue(std::tuple<int,char*,int>(tag,byteStream,size));
#endif
                }
            }
        }

        // if nothing done, wait for sometime
        if(!didSomething && INFERENCE_SERVICE_IDLE_TIME > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(INFERENCE_SERVICE_IDLE_TIME));
        }
    }
    info("runInference: terminated for %s", clientName.c_str());

    // send and wait for INFCOM_CMD_DONE message
    InfComCommand reply = {
        INFCOM_MAGIC, INFCOM_CMD_DONE, { 0 }, { 0 }
    };
    ERRCHK(sendCommand(sock, reply, clientName));
    ERRCHK(recvCommand(sock, reply, clientName, INFCOM_CMD_DONE));

    return 0;
}

#if INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
void InferenceEngineHip::workMasterInputQ()
{
    args->lock();
    info("workMasterInputQ: started for %s", clientName.c_str());
    args->unlock();

    int batchSize = args->getBatchSize();
    int totalInputCount = 0;
    int inputCountInBatch = 0, gpu = 0;
    for(;;) {
        PROFILER_START(inference_server_app, workMasterInputQ);
         // get next item from the input queue
        std::tuple<int,char*,int> input;
        inputQ.dequeue(input);
        int tag = std::get<0>(input);
        char * byteStream = std::get<1>(input);
        int size = std::get<2>(input);

        // check for end of input
        if(tag < 0 || byteStream == nullptr || size == 0)
            break;
        totalInputCount++;

        // add the image to selected deviceQ
        std::tuple<char*,int> image(byteStream,size);
        queueDeviceTagQ[gpu]->enqueue(tag);
        queueDeviceImageQ[gpu]->enqueue(image);
        PROFILER_STOP(inference_server_app, workMasterInputQ);

        // at the end of Batch pick another device
        inputCountInBatch++;
        if(inputCountInBatch == batchSize) {
            inputCountInBatch = 0;
            gpu = (gpu + 1) % GPUs;
            for(int i = 0; i < GPUs; i++) {
                if(i != gpu && queueDeviceTagQ[i]->size() < queueDeviceTagQ[gpu]->size()) {
                    gpu = i;
                }
            }
        }
    }

    // send endOfSequence indicator to all scheduler threads
    for(int i = 0; i < GPUs; i++) {
        int endOfSequenceTag = -1;
        std::tuple<char*,int> endOfSequenceImage(nullptr,0);
        queueDeviceTagQ[i]->enqueue(endOfSequenceTag);
        queueDeviceImageQ[i]->enqueue(endOfSequenceImage);
    }
    args->lock();
    info("workMasterInputQ: terminated for %s [scheduled %d images]", clientName.c_str(), totalInputCount);
    args->unlock();
}

void InferenceEngineHip::workDeviceInputCopy(int gpu)
{
    args->lock();
    info("workDeviceInputCopy: GPU#%d started for %s", gpu, clientName.c_str());
    args->unlock();

    // create HIP stream
    hipStream_t stream;
    hipError_t err;
    if(hipSuccess != hipStreamCreate(&stream))
      fatal("workDeviceInputCopy: hipStreamCreate(device_id[%d]) failed (%d)", gpu, err);

    int totalBatchCounter = 0, totalImageCounter = 0;
    for(bool endOfSequenceReached = false; !endOfSequenceReached; ) {
        PROFILER_START(inference_server_app, workDeviceInputCopyBatch);
        // get an empty HIP buffer and lock the buffer for writing
        std::pair<void *, void*> input;
        queueDeviceInputMemIdle[gpu]->dequeue(input);
        if(input.first == nullptr) {
            fatal("workDeviceInputCopy: unexpected nullptr in queueDeviceInputMemIdle[%d] buffer", gpu);
        }
        void * mapped_ptr = input.second;
        if(mapped_ptr == nullptr) {
            fatal("workDeviceInputCopy: unexpected nullptr in queueDeviceInputMemIdle[%d] mapped_ptr", gpu);
        }

        // get next batch of inputs and convert them into tensor and release input byteStream
        // TODO: replace with an efficient implementation
        int inputCount = 0;
        if (numDecThreads > 0) {
            std::vector<std::tuple<char*, int>> batch_q;
            //int sub_batch_size = batchSize/numDecThreads;
            //std::thread dec_threads[numDecThreads];
            //int numT = numDecThreads;
            // dequeue batch
            for (; inputCount<batchSize; inputCount++)
            {
                std::tuple<char*, int> image;
                queueDeviceImageQ[gpu]->dequeue(image);
                char * byteStream = std::get<0>(image);
                int size = std::get<1>(image);
                if(byteStream == nullptr || size == 0) {
                    printf("workDeviceInputCopy:: Eos reached inputCount: %d\n", inputCount);
                    endOfSequenceReached = true;
                    break;
                }
                batch_q.push_back(image);
            }
            if (inputCount){
                PROFILER_START(inference_server_app, workDeviceInputCopyJpegDecode);    
#if 0            
                if (inputCount < batchSize)
                {
                    sub_batch_size = (inputCount+numT-1)/numT;
                    numT = (inputCount+(sub_batch_size-1))/sub_batch_size;
                }
                int start = 0; int end = sub_batch_size-1;
                for (unsigned int t = 0; t < (numT - 1); t++)
                {
                    dec_threads[t]  = std::thread(&InferenceEngine::DecodeScaleAndConvertToTensorBatch, this, std::ref(batch_q), start, end, dimInput, (float *)mapped_ptr);
                    start += sub_batch_size;
                    end += sub_batch_size;
                }
                start = std::min(start, (inputCount - 1));
                end = std::min(end, (inputCount-1));
                // do some work in this thread
                DecodeScaleAndConvertToTensorBatch(batch_q, start, end, dimInput, (float *)mapped_ptr);
                for (unsigned int t = 0; t < (numT - 1); t++)
                {
                    dec_threads[t].join();
                }
#else
                #pragma omp parallel for num_threads(inputCount)  // default(none) TBD: option disabled in Ubuntu 20.04
                for (size_t i = 0; i < inputCount; i++) {
                  DecodeScaleAndConvertToTensorBatch(batch_q, i, i, dimInput, (float *)mapped_ptr);
                }
#endif 
                PROFILER_STOP(inference_server_app, workDeviceInputCopyJpegDecode);
            }
        } else {
            for(; inputCount < batchSize; inputCount++) {
                // get next item from the input queue and check for end of input
                std::tuple<char*,int> image;
                queueDeviceImageQ[gpu]->dequeue(image);
                char * byteStream = std::get<0>(image);
                int size = std::get<1>(image);
                if(byteStream == nullptr || size == 0) {
                    endOfSequenceReached = true;
                    break;
                }
                // decode, scale, and format convert into the HIP buffer
                void *buf;
                if (useFp16)
                    buf = (unsigned short *)mapped_ptr + dimInput[0] * dimInput[1] * dimInput[2] * inputCount;
                else
                    buf = (float *)mapped_ptr + dimInput[0] * dimInput[1] * dimInput[2] * inputCount;

                PROFILER_START(inference_server_app, workDeviceInputCopyJpegDecode);
                DecodeScaleAndConvertToTensor(dimInput[0], dimInput[1], size, (unsigned char *)byteStream, (float *)buf, useFp16);
                PROFILER_STOP(inference_server_app, workDeviceInputCopyJpegDecode);
                // release byteStream
                delete[] byteStream;
            }
        }
        if(hipStreamSynchronize(stream) != hipSuccess) {
            fatal("workDeviceInputCopy: hipStreamSynchronize(#%d) failed", gpu);
        }

        if(inputCount > 0) {
            // add the input for processing
            queueDeviceInputMemBusy[gpu]->enqueue(input);
            // update counters
            totalBatchCounter++;
            totalImageCounter += inputCount;
        }
        else {
            // add the input back to idle queue
            queueDeviceInputMemIdle[gpu]->enqueue(input);
        }
        PROFILER_STOP(inference_server_app, workDeviceInputCopyBatch);
    }
    // release HIP stream
    hipStreamDestroy(stream);

    // add the endOfSequenceMarker to next stage
    void* endOfSequenceMarker = nullptr;
    queueDeviceInputMemBusy[gpu]->enqueue(std::make_pair(endOfSequenceMarker, endOfSequenceMarker));

    args->lock();
    info("workDeviceInputCopy: GPU#%d terminated for %s [processed %d batches, %d images]", gpu, clientName.c_str(), totalBatchCounter, totalImageCounter);
    args->unlock();
}

void InferenceEngineHip::workDeviceProcess(int gpu)
{
    args->lock();
    info("workDeviceProcess: GPU#%d started for %s", gpu, clientName.c_str());
    args->unlock();

    int processCounter = 0;
    for(;;) {
        // get a busy HIP buffer for input and check for end of sequence marker
        std::pair<void *, void*> input;
        queueDeviceInputMemBusy[gpu]->dequeue(input);
        if(!input.first) {
            break;
        }
        // get an empty HIP buffer for output and a busy HIP buffer for input
        std::pair<void *, void*> output;
        queueDeviceOutputMemIdle[gpu]->dequeue(output);
        if(!output.first) {
            fatal("workDeviceProcess: unexpected nullptr in queueDeviceOutputMemIdle[%d]", gpu);
        }
        // process the graph
        vx_status status;
        status = vxSwapTensorHandle(openvx_input[gpu], input.first, nullptr);
        if(status != VX_SUCCESS) {
            fatal("workDeviceProcess: vxSwapTensorHandle(input#%d) failed(%d)", gpu, status);
        }
        status = vxSwapTensorHandle(openvx_output[gpu], output.first, nullptr);
        if(status != VX_SUCCESS) {
            fatal("workDeviceProcess: vxSwapTensorHandle(output#%d) failed(%d)", gpu, status);
        }
#if !DONOT_RUN_INFERENCE
        PROFILER_START(inference_server_app, workDeviceProcess);
        status = vxProcessGraph(openvx_graph[gpu]);
        PROFILER_STOP(inference_server_app, workDeviceProcess);
        if(status != VX_SUCCESS) {
            fatal("workDeviceProcess: vxProcessGraph(#%d) failed(%d)", gpu, status);
        }
#else
        info("InferenceEngine:workDeviceProcess DONOT_RUN_INFERENCE mode");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));  // simulate some work
#endif
        // add the input for idle queue and output to busy queue
        queueDeviceInputMemIdle[gpu]->enqueue(input);
        queueDeviceOutputMemBusy[gpu]->enqueue(output);
        processCounter++;
    }

    // add the endOfSequenceMarker to next stage
    void * endOfSequenceMarker = nullptr;
    queueDeviceOutputMemBusy[gpu]->enqueue(std::make_pair(endOfSequenceMarker, endOfSequenceMarker));

    args->lock();
    info("workDeviceProcess: GPU#%d terminated for %s [processed %d batches]", gpu, clientName.c_str(), processCounter);
    args->unlock();
}

void InferenceEngineHip::workDeviceOutputCopy(int gpu)
{
    args->lock();
    info("workDeviceOutputCopy: GPU#%d started for %s", gpu, clientName.c_str());
    args->unlock();

    // create HIP stream
    hipStream_t stream;
    if(hipStreamCreate(&stream) != hipSuccess) {
        fatal("workDeviceOutputCopy: hipStreamCreate(device_id[%d]) failed", gpu);
    }

    int totalBatchCounter = 0, totalImageCounter = 0;
    for(bool endOfSequenceReached = false; !endOfSequenceReached; ) {
        // get an output HIP buffer and lock the buffer for reading
        std::pair<void *, void*> output;
        queueDeviceOutputMemBusy[gpu]->dequeue(output);
        void * mem = output.first, * host_ptr = output.second;
        if(mem == nullptr || host_ptr == nullptr) {
            break;
        }
        
        PROFILER_START(inference_server_app, workDeviceOutputCopy);

        // get next batch of inputs
        int outputCount = 0;
        int useFp16 = args->fp16Inference();
        for(; outputCount < batchSize; outputCount++) {
            // get next item from the tag queue and check for end of input
            int tag;
            queueDeviceTagQ[gpu]->dequeue(tag);
            if(tag < 0) {
                endOfSequenceReached = true;
                break;
            }

            // decode, scale, and format convert into the HIP buffer
            void *buf;
            if (!useFp16)
                buf = (float *)host_ptr + dimOutput[0] * dimOutput[1] * dimOutput[2] * outputCount;
            else
                buf = (unsigned short *)host_ptr + dimOutput[0] * dimOutput[1] * dimOutput[2] * outputCount;

            if (!detectBoundingBoxes)
            {
                if (topK < 1){
                    int label = 0;
                    if (!useFp16) {
                        float *out = (float *)buf;
                        float max_prob = out[0];
                        for(int c = 1; c < dimOutput[2]; c++) {
                            float prob = out[c];
                            if(prob > max_prob) {
                                label = c;
                                max_prob = prob;
                            }
                        }
                    } else {
                        unsigned short *out = (unsigned short *)buf;
                        float max_prob = _cvtsh_ss(out[0]);
                        for(int c = 1; c < dimOutput[2]; c++) {
                            float prob = _cvtsh_ss(out[c]);
                            if(prob > max_prob) {
                                label = c;
                                max_prob = prob;
                            }
                        }
                    }
                    outputQ.enqueue(std::tuple<int,int>(tag,label));
                }else {
                    // todo:: add support for fp16
                    std::vector<float>  prob_vec((float*)buf, (float*)buf + dimOutput[2]);
                    std::vector<size_t> idx(prob_vec.size());
                    std::iota(idx.begin(), idx.end(), 0);
                    sort_indexes(prob_vec, idx);            // sort indeces based on prob
                    std::vector<unsigned int>    labels;
                    outputQ.enqueue(std::tuple<int,int>(tag,idx[0]));
                    int j=0;
                    for (auto i: idx) {
                        // make label which is index and prob
                        int packed_label_prob = (i&0xFFFF)|(((unsigned int)((prob_vec[i]*0x7FFF)+0.5))<<16);   // convert prob to 16bit float and store in MSBs
                        labels.push_back(packed_label_prob);
                        if (++j >= topK) break;
                    }
                    outputQTopk.enqueue(labels);
                }
            }else
            {
                std::vector<ObjectBB> detected_objects;
                region->GetObjectDetections((float *)buf, BB_biases, dimOutput[2], dimOutput[1], dimOutput[0], BOUNDING_BOX_NUMBER_OF_CLASSES, dimInput[0], dimInput[1], BOUNDING_BOX_CONFIDENCE_THRESHHOLD, BOUNDING_BOX_NMS_THRESHHOLD, 13, detected_objects);
                if (detected_objects.size() > 0) {
                    // add it to outputQ
                    outputQ.enqueue(std::tuple<int,int>(tag,detected_objects[0].label));
                    // add detected objects with BB into BoundingBox Q
                    OutputQBB.enqueue(detected_objects);
                } else
                {
                    // add it to outputQ
                    outputQ.enqueue(std::tuple<int,int>(tag,-1));
                }
            }
        }

        // add the output back to idle queue
        queueDeviceOutputMemIdle[gpu]->enqueue(output);

        PROFILER_STOP(inference_server_app, workDeviceOutputCopy);

        // update counter
        if(outputCount > 0) {
            totalBatchCounter++;
            totalImageCounter += outputCount;
        }
    }

    // release HIP stream
    hipStreamDestroy(stream);

    // send end of sequence marker to next stage
    outputQ.enqueue(std::tuple<int,int>(-1,-1));
    args->lock();
    info("workDeviceOutputCopy: GPU#%d terminated for %s [processed %d batches, %d images]", gpu, clientName.c_str(), totalBatchCounter, totalImageCounter);
    args->unlock();
}
#endif

void InferenceEngineHip::dumpBuffer(hipStream_t stream, void * mem, size_t size, std::string fileName)
{
    void *host_ptr = nullptr;
    hipError_t err = hipMemcpyDtoH(mem, host_ptr, size);
    if(err) return;
    FILE * fp = fopen(fileName.c_str(), "wb");
    if (fp) {
      fwrite(host_ptr, 1, size, fp);
      fclose(fp);
    }
    printf("OK: dumped %lu bytes into %s\n", size, fileName.c_str());
}

#endif