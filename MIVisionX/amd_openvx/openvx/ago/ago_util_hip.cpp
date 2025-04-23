/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "ago_internal.h"
#include "ago_haf_gpu.h"
#if  ENABLE_HIP

// Create HIP Context
int agoGpuHipCreateContext(AgoContext *context, int deviceID) {
    if (deviceID >= 0) {
      // release context if already set.
        context->hip_context_imported = true;
        agoGpuHipReleaseContext(context);
        // use the given HIP device

    }
    else {
        // select the first device
        deviceID = 0;
    }

    hipError_t err;
    err = hipGetDeviceCount(&context->hip_num_devices);
    if (err != hipSuccess) {
        agoAddLogEntry(NULL, VX_FAILURE, "ERROR: hipGetDeviceCount => %d (failed)\n", err);
        return -1;
    }
    if (context->hip_num_devices < 1) {
        agoAddLogEntry(NULL, VX_FAILURE, "ERROR: didn't find any GPU!\n", err);
        return -1;
    }

    if (deviceID >= context->hip_num_devices) {
        agoAddLogEntry(NULL, VX_FAILURE, "ERROR: the requested deviceID is not found!\n", deviceID);
        return -1;
    }

    err = hipSetDevice(deviceID);
    if (err != hipSuccess) {
        agoAddLogEntry(NULL, VX_FAILURE, "ERROR: hipSetDevice(%d) => %d (failed)\n", deviceID, err);
        return -1;
    }
    context->hip_device_id = deviceID;

    err = hipGetDeviceProperties(&context->hip_dev_prop, deviceID);
    if (err != hipSuccess) {
        agoAddLogEntry(NULL, VX_FAILURE, "ERROR: hipGetDeviceProperties(%d) => %d (failed)\n", deviceID, err);
    }

    bool wgpMode = false;
    if (context->hip_dev_prop.major >= 10) {
        wgpMode = true;
        char temp[2] = {};
        if (agoGetEnvironmentVariable("GPU_ENABLE_WGP_MODE", temp, sizeof(temp))) {
            wgpMode = (temp[0] == '0') ? false : true;
        }
    }
    int totalCUs = wgpMode ? context->hip_dev_prop.multiProcessorCount * 2 : context->hip_dev_prop.multiProcessorCount;

    agoAddLogEntry(&context->ref, VX_SUCCESS, "OK: OpenVX using GPU device - %d: %s [%s] with %d CUs on PCI bus %02x:%02x.%x\n",
                   deviceID, context->hip_dev_prop.name, context->hip_dev_prop.gcnArchName, totalCUs,
                   context->hip_dev_prop.pciBusID, context->hip_dev_prop.pciDomainID, context->hip_dev_prop.pciDeviceID);

    return 0;
}

int agoGpuHipReleaseContext(AgoContext * context) {

    // nothing to do here
    return 0;
}

int agoGpuHipReleaseGraph(AgoGraph * graph) {
    if (graph->hip_stream0) {
        hipError_t err = hipStreamDestroy(graph->hip_stream0);
        if (err != hipSuccess) {
            agoAddLogEntry(NULL, VX_FAILURE, "ERROR: agoGpuHipReleaseGraph: hipStreamDestroy(%p) failed (%d)\n", graph->hip_stream0, err);
            return -1;
        }
        graph->hip_stream0 = NULL;
    }
    return 0;
}

int agoGpuHipReleaseData(AgoData * data) {
    if (data->hip_memory_allocated) {
        hipError_t status = hipFree((void *)data->hip_memory_allocated);
        if (status != hipSuccess) {
            agoAddLogEntry(NULL, VX_FAILURE, "ERROR: agoGpuHipReleaseData: hipFree(%p) failed (%d)\n", data->hip_memory_allocated, status);
        }
        data->hip_memory_allocated = NULL;
        data->ref.context->hip_mem_release_count++;
    }
    data->hip_memory = NULL;
    data->gpu_buffer_offset = 0;
    return 0;
}

int agoGpuHipReleaseSuperNode(AgoSuperNode * supernode) {
    if (supernode->hip_stream0) {
        // supernode->hip_stream0 is used from context, hence no need to destroy here
        supernode->hip_stream0 = NULL;
    }
    return 0;
}

// create HIP device memory
static void agoGpuHipCreateBuffer(AgoContext * context, void ** host_ptr, size_t size, hipError_t &errcode_ret) {
    errcode_ret = hipMalloc( host_ptr, size);
    if (host_ptr && (errcode_ret == hipSuccess) ) {
        context->hip_mem_alloc_count++;
        context->hip_mem_alloc_size += size;
    } else {
        agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: hipMalloc Failed with status: %d\n", errcode_ret);
    }
    return;
}

int agoGpuHipAllocBuffer(AgoData * data) {
    // make sure buffer is valid
    if (agoDataSanityCheckAndUpdate(data)) {
        return -1;
    }
    // allocate buffer
    AgoContext * context = data->ref.context;
    if (data->ref.type == VX_TYPE_IMAGE) {
        // to handle image ROI
        AgoData * dataMaster = data->u.img.roiMasterImage ? data->u.img.roiMasterImage : data;
        if (!dataMaster->hip_memory && !dataMaster->u.img.enableUserBufferGPU && !(dataMaster->import_type == VX_MEMORY_TYPE_HIP)) {
            hipError_t err = hipSuccess;
            {
                // allocate hip_memory
                dataMaster->hip_memory = nullptr;
                agoGpuHipCreateBuffer(context, (void **)&dataMaster->hip_memory, dataMaster->size + dataMaster->gpu_buffer_offset, err);
                dataMaster->hip_memory_allocated = dataMaster->hip_memory;
            }
            if (!dataMaster->hip_memory || err) {
                agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: agoGpuHipAllocBuffer(MEM_READ_WRITE,%d,0,*) FAILED with status <%p, %d>\n",
                        (int)dataMaster->size + dataMaster->gpu_buffer_offset, dataMaster->hip_memory, err);
                return -1;
            }
            if (dataMaster->u.img.isUniform) {
                // make sure that CPU buffer is allocated
                if (!dataMaster->buffer) {
                    if (agoAllocData(dataMaster)) {
                        return -1;
                    }
                }
                // copy the uniform image into HIP memory because there won't be any commits happening to this buffer
                hipError_t err = hipMemcpyHtoD((void *)(dataMaster->hip_memory + dataMaster->gpu_buffer_offset), dataMaster->buffer, dataMaster->size);
                if (err != hipSuccess) {
                    agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: agoGpuHipAllocBuffer: hipMemcpyHtoD() => %d\n", err);
                    return -1;
                }
                dataMaster->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
            }
        }
        if (data != dataMaster) {
            // special handling for image ROI
            data->hip_memory = dataMaster->hip_memory;
            if((dataMaster->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_BY_WRITE)) {
                // copy the image into HIP buffer because commits aren't done to this buffer
                hipError_t err = hipMemcpyHtoD((void *)(dataMaster->hip_memory + dataMaster->gpu_buffer_offset), dataMaster->buffer, dataMaster->size);
                if (err != hipSuccess) {
                    agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: agoGpuHipAllocBuffer: hipMemcpyHtoD() => %d\n", err);
                }
            }
        }
    }
    else if (data->ref.type == VX_TYPE_ARRAY || data->ref.type == AGO_TYPE_CANNY_STACK) {
        hipError_t err = hipSuccess;
        if (!data->hip_memory) {
            // first few bytes reserved for numitems/stacktop
            data->gpu_buffer_offset = DATA_GPU_ARRAY_OFFSET;
            err = hipSuccess;
            {
                // normal hip buffer allocation
                data->hip_memory = 0;
                agoGpuHipCreateBuffer(context, (void **)&data->hip_memory, data->size + data->gpu_buffer_offset, err);
                data->hip_memory_allocated = data->hip_memory;
                if (data->hip_memory) {
                    // initialize array header which containts numitems
                    err = hipMemset(data->hip_memory, 0, data->size + data->gpu_buffer_offset);
                }
            }
            if (err) {
                agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: agoGpuHipAllocBuffer(MEM_READ_WRITE,%d,0,*) FAILED\n", (int)data->size + data->gpu_buffer_offset);
                return -1;
            }
        }
    }
    else if (data->ref.type == VX_TYPE_SCALAR || data->ref.type == VX_TYPE_THRESHOLD) {
        // nothing to do
    }
    else if (data->ref.type == VX_TYPE_LUT) {
        hipError_t err = hipSuccess;
        if (!data->hip_memory) {
            if (data->u.lut.type == VX_TYPE_UINT8) {
                data->gpu_buffer_offset = 0;
                agoGpuHipCreateBuffer(context, (void **)&data->hip_memory, data->size + data->gpu_buffer_offset, err);
                data->hip_memory_allocated = data->hip_memory;
                if (err) {
                    agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: agoGpuHipAllocBuffer(MEM_READ_WRITE,%d,0,*) FAILED\n",
                        (int)data->size + data->gpu_buffer_offset);
                    return -1;
                }
            }
            else {
                // normal Hip memory allocation
                data->hip_memory = 0;
                agoGpuHipCreateBuffer(context, (void **)&data->hip_memory, data->size + data->gpu_buffer_offset, err);
                data->hip_memory_allocated = data->hip_memory;
                if (err) {
                    agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: agoGpuHipAllocBuffer(MEM_READ_WRITE,%d,0,*) FAILED\n",
                        (int)data->size + data->gpu_buffer_offset);
                    return -1;
                }
            }
        }
    }
    else if (data->ref.type == VX_TYPE_CONVOLUTION) {
        hipError_t err = hipSuccess;
        if (!data->hip_memory) {
            data->gpu_buffer_offset = 0;
            agoGpuHipCreateBuffer(context, (void **)&data->hip_memory, (data->size << 1) + data->gpu_buffer_offset, err);
            data->hip_memory_allocated = data->hip_memory;
            if (err) {
                agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: agoGpuHipAllocBuffer(MEM_READ_WRITE,%d,0,*) FAILED\n",
                    (int)data->size + data->gpu_buffer_offset);
                return -1;
            }
        }
    }
    else if (data->ref.type == VX_TYPE_REMAP) {
        hipError_t err = hipSuccess;
        if (!data->hip_memory) {
            data->hip_memory = 0;
            agoGpuHipCreateBuffer(context, (void **)&data->hip_memory, data->size + data->gpu_buffer_offset, err);
            data->hip_memory_allocated = data->hip_memory;
            if (err) {
                agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: agoGpuHipAllocBuffer(MEM_READ_WRITE,%d,0,*) FAILED\n",
                    (int)data->size + data->gpu_buffer_offset);
                return -1;
            }
            data->gpu_buffer_offset = 0;
        }
    }
    else if (data->ref.type == VX_TYPE_MATRIX) {
        if (!data->hip_memory) {
            hipError_t err = hipSuccess;
            data->hip_memory = 0;
            agoGpuHipCreateBuffer(context, (void **)&data->hip_memory, data->size + data->gpu_buffer_offset, err);
            data->hip_memory_allocated = data->hip_memory;
            if (err) {
                agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: agoGpuHipAllocBuffer(MEM_READ_WRITE,%d,0,*) FAILED\n",
                    (int)data->size + data->gpu_buffer_offset);
                return -1;
            }
            data->gpu_buffer_offset = 0;
        }
    }
    else if (data->ref.type == VX_TYPE_TENSOR) {
        // to handle tensor ROI
        AgoData * dataMaster = data->u.tensor.roiMaster ? data->u.tensor.roiMaster : data;
        if (!dataMaster->hip_memory) {
            hipError_t err = hipSuccess;
            dataMaster->hip_memory = 0;
            agoGpuHipCreateBuffer(context, (void **)&dataMaster->hip_memory, dataMaster->size + dataMaster->gpu_buffer_offset, err);
            dataMaster->hip_memory_allocated = dataMaster->hip_memory;
            if (err) {
                agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: agoGpuHipAllocBuffer(MEM_READ_WRITE,%d,0,*) FAILED\n",
                    (int)dataMaster->size + dataMaster->gpu_buffer_offset);
                return -1;
            }
            dataMaster->gpu_buffer_offset = 0;
        }
        if (data != dataMaster) {
            // special handling for tensor ROI
            data->hip_memory = dataMaster->hip_memory;
            data->gpu_buffer_offset = (vx_uint32)data->u.tensor.offset;
        }
    }
    else if (data->numChildren > 0) {
        for (vx_uint32 child = 0; child < data->numChildren; child++) {
            if (agoGpuHipAllocBuffer(data->children[child]) < 0) {
                return -1;
            }
        }
    }
    else {
        agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoGpuHipAllocBuffer: doesn't support object type %s of %s\n",
            agoEnum2Name(data->ref.type), data->name.length() ? "?" : data->name.c_str());
        return -1;
    }
    // allocate CPU buffer
    if (agoAllocData(data)) {
        return -1;
    }
    return 0;
}

static int agoGpuHipDataInputSync(AgoGraph * graph, AgoData * data, vx_uint32 dataFlags, vx_uint32 group, bool need_access, bool need_read_access) {
    if (data->ref.type == VX_TYPE_IMAGE) {
        // only use image objects that need read access
        if (need_access) {
            if (!data->hip_memory && data->isVirtual && data->ownerOfUserBufferGPU &&
                data->ownerOfUserBufferGPU->akernel->gpu_buffer_update_callback_f)
            { // need to update hip_memory from user kernel
                vx_status status = data->ownerOfUserBufferGPU->akernel->gpu_buffer_update_callback_f(data->ownerOfUserBufferGPU,
                    (vx_reference *)data->ownerOfUserBufferGPU->paramList, data->ownerOfUserBufferGPU->paramCount);
                if (status || !data->hip_memory) {
                    agoAddLogEntry(&data->ownerOfUserBufferGPU->ref, status, "ERROR: gpu_buffer_update_callback_f: failed(%d:%p)\n", status, data->hip_memory);
                    return -1;
                }
            }
            if (need_read_access) {
                auto dataToSync = data->u.img.isROI ? data->u.img.roiMasterImage : data;
                if (!(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                    if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE | AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT)) {
                        int64_t stime = agoGetClockCounter();
                        // HIP write HostToDevice
                        if (dataToSync->hip_memory) {
                            hipError_t err = hipMemcpy(dataToSync->hip_memory + dataToSync->gpu_buffer_offset, dataToSync->buffer, dataToSync->size, hipMemcpyHostToDevice);
                            if (err) {
                                agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: hipMemcpyHtoD() => %d\n", err);
                                return -1;
                            }
                        }
                        dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                        int64_t etime = agoGetClockCounter();
                        graph->gpu_perf.buffer_write += etime - stime;
                    }
                }
            }
        }
    }
    else if (data->ref.type == VX_TYPE_ARRAY) {
        if (need_read_access) {
            if (!(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE | AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT)) {
                    int64_t stime = agoGetClockCounter();
                    vx_size size = data->u.arr.numitems * data->u.arr.itemsize;
                    if (size > 0 && data->hip_memory) {
                        hipError_t err = hipMemcpyHtoD(data->hip_memory + data->gpu_buffer_offset, data->buffer, data->size);
                        if (err) {
                            agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: hipMemcpyHtoD() => %d (array)\n", err);
                            return -1;
                        }
                    }
                    data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                    int64_t etime = agoGetClockCounter();
                    graph->gpu_perf.buffer_write += etime - stime;
                }
            }
        }
    }
    else if (data->ref.type == AGO_TYPE_CANNY_STACK) {
        if (need_read_access) {
            agoAddLogEntry(&data->ref, VX_FAILURE,
                "ERROR: agoGpuHipDataSyncInputs: doesn't support object type %s for read-access in group#%d for kernel arg setting\n",
                agoEnum2Name(data->ref.type), group);
            return -1;
        }
    }
    else if (data->ref.type == VX_TYPE_THRESHOLD) {
        // nothing to do.. the node will
    }
    else if (data->ref.type == VX_TYPE_SCALAR) {
        // nothing to do.. the node will
    }
    else if (data->ref.type == VX_TYPE_MATRIX) {
        if (need_read_access) {
            if (data->hip_memory && !(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE | AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT)) {
                    int64_t stime = agoGetClockCounter();
                    if (data->size > 0 && data->hip_memory) {
                        hipError_t err = hipMemcpyHtoD(data->hip_memory + data->gpu_buffer_offset, data->buffer, data->size);
                        if (err) {
                            agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: hipMemcpyHtoD() => %d (array)\n", err);
                            return -1;
                        }
                    }
                    data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                    int64_t etime = agoGetClockCounter();
                    graph->gpu_perf.buffer_write += etime - stime;
                }
            }
        }
    }
    else if (data->ref.type == VX_TYPE_LUT) {
        // only use lut objects that need read access
        if (need_access) {
            if (need_read_access) {
                if (!(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                    if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE | AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT)) {
                        int64_t stime = agoGetClockCounter();
                        if (data->u.lut.type == VX_TYPE_UINT8 && data->hip_memory && data->size >0) {
                            hipError_t err = hipMemcpyHtoD(data->hip_memory + data->gpu_buffer_offset, data->buffer, data->size);
                            if (err) {
                                agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoGpuHipDataInputSync: hipMemcpyHtoD() => %d (for LUT)\n", err);
                                return -1;
                            }
                        }
                        else if (data->u.lut.type == VX_TYPE_INT16 && data->hip_memory && data->size >0) {
                            hipError_t err = hipMemcpyHtoD(data->hip_memory + data->gpu_buffer_offset, data->buffer, data->size);
                            if (err) {
                                agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoGpuHipDataInputSync: hipMemcpyHtoD() => %d (for LUT)\n", err);
                                return -1;
                            }
                        }
                        data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                        int64_t etime = agoGetClockCounter();
                        graph->gpu_perf.buffer_write += etime - stime;
                    }
                }
            }
        }
    }
    else if (data->ref.type == VX_TYPE_CONVOLUTION) {
        // only use conv objects that need read access
        if (need_access) {
            if (need_read_access) {
                if (!(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                    if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE | AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT)) {
                        int64_t stime = agoGetClockCounter();
                        if (data->hip_memory && data->size >0) {
                            hipError_t err = hipMemcpyHtoD(data->hip_memory + data->gpu_buffer_offset, data->reserved, data->size << 1);
                            if (err) {
                                agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoGpuHipDataInputSync: hipMemcpyHtoD() => %d (for LUT)\n", err);
                                return -1;
                            }
                        }
                        data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                        int64_t etime = agoGetClockCounter();
                        graph->gpu_perf.buffer_write += etime - stime;
                    }
                }
            }
        }
    }
    else if (data->ref.type == VX_TYPE_REMAP) {
         // only use image objects that need read access
        if (need_access) {
            if (need_read_access) {
                if (data->hip_memory && !(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                    if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE | AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT)) {
                        int64_t stime = agoGetClockCounter();
                        hipError_t err = hipMemcpyHtoD(data->hip_memory + data->gpu_buffer_offset, data->buffer, data->size);
                        if (err) {
                            agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoGpuHipDataInputSync: hipMemcpyHtoD() => %d (for Remap)\n", err);
                            return -1;
                        }
                        data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                        int64_t etime = agoGetClockCounter();
                        graph->gpu_perf.buffer_write += etime - stime;
                    }
                }
            }
        }
    }
    else if (data->ref.type == VX_TYPE_TENSOR) {
        if (need_read_access) {
            auto dataToSync = data->u.tensor.roiMaster ? data->u.tensor.roiMaster : data;
            if (!(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE | AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT)) {
                    int64_t stime = agoGetClockCounter();
                    if (dataToSync->hip_memory) {
                        hipError_t err = hipMemcpyHtoD(dataToSync->hip_memory + dataToSync->gpu_buffer_offset, dataToSync->buffer, dataToSync->size);
                        if (err) {
                            agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: hipMemcpyHtoD() => %d (tensor)\n", err);
                            return -1;
                        }
                    }
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                    int64_t etime = agoGetClockCounter();
                    graph->gpu_perf.buffer_write += etime - stime;
                }
            }
        }
    }
    else {
        agoAddLogEntry(&data->ref, VX_FAILURE,
            "ERROR: agoGpuHipDataSyncInputs: doesn't support object type %s in group#%d for kernel arg setting\n", agoEnum2Name(data->ref.type), group);
        return -1;
    }
    return 0;
}

int agoGpuHipSuperNodeMerge(AgoGraph * graph, AgoSuperNode * supernode, AgoNode * node) {
    // sanity check
    if (!node->akernel->func && !node->akernel->kernel_f) {
        agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: agoGpuHipSuperNodeMerge: doesn't support kernel %s\n", node->akernel->name);
        return -1;
    }
    // merge node into supernode
    supernode->nodeList.push_back(node);
    for (vx_uint32 i = 0; i < node->paramCount; i++) {
        AgoData * data = node->paramList[i];
        if (data) {
            size_t index = std::find(supernode->dataList.begin(), supernode->dataList.end(), data) - supernode->dataList.begin();
            if (index == supernode->dataList.size()) {
                // add data with zero entries into the lists
                AgoSuperNodeDataInfo info = { 0 };
                info.needed_as_a_kernel_argument = true;
                supernode->dataInfo.push_back(info);
                supernode->dataList.push_back(data);
                supernode->dataListForAgeDelay.push_back(data);
            }
            // update count for data direction
            supernode->dataInfo[index].argument_usage[node->parameters[i].direction]++;
        }
    }
    return 0;
}

int agoGpuHipSuperNodeUpdate(AgoGraph * graph, AgoSuperNode * supernode) {
    // make sure that all output images have same dimensions
    // check to make sure that max input hierarchy level is less than min output hierarchy level
    vx_uint32 width = 0, height = 0;
    vx_uint32 max_input_hierarchical_level = 0, min_output_hierarchical_level = INT_MAX;
    for (size_t index = 0; index < supernode->dataList.size(); index++) {
        AgoData * data = supernode->dataList[index];
        if (data->ref.type == VX_TYPE_IMAGE && supernode->dataInfo[index].argument_usage[VX_INPUT] == 0) {
            if (!width || !height) {
                width = data->u.img.width;
                height = data->u.img.height;
            }
            else if (width != data->u.img.width || height != data->u.img.height) {
                agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoGpuHipSuperNodeUpdate: doesn't support different image dimensions inside same group#%d\n", supernode->group);
                return -1;
            }
        }
        if (data->isVirtual && data->ref.type != VX_TYPE_SCALAR &&
            data->inputUsageCount == supernode->dataInfo[index].argument_usage[VX_INPUT] &&
            data->outputUsageCount == supernode->dataInfo[index].argument_usage[VX_OUTPUT] &&
            data->inoutUsageCount == supernode->dataInfo[index].argument_usage[VX_BIDIRECTIONAL])
        {
            // no need of this parameter as an argument into the kernel
            // mark that this will be an internal variable for the kernel
            supernode->dataInfo[index].needed_as_a_kernel_argument = false;
            // TBD: mark this the buffer doesn't need allocation
        }
        if (data->hierarchical_level > min_output_hierarchical_level) min_output_hierarchical_level = data->hierarchical_level;
        if (data->hierarchical_level < max_input_hierarchical_level) max_input_hierarchical_level = data->hierarchical_level;
    }
    if (max_input_hierarchical_level > min_output_hierarchical_level) {
        agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: agoGpuHipSuperNodeUpdate: doesn't support mix of hierarchical levels inside same group#%d\n", supernode->group);
        return -1;
    }
    supernode->width = width;
    supernode->height = height;

    // mark hierarchical level (start,end) of all supernodes
    for (AgoSuperNode * supernode = graph->supernodeList; supernode; supernode = supernode->next) {
        supernode->hierarchical_level_start = INT_MAX;
        supernode->hierarchical_level_end = 0;
        for (AgoNode * node : supernode->nodeList) {
            supernode->hierarchical_level_start = min(supernode->hierarchical_level_start, node->hierarchical_level);
            supernode->hierarchical_level_end = max(supernode->hierarchical_level_end, node->hierarchical_level);
        }
    }
    return 0;
}

int agoGpuHipSuperNodeWait(AgoGraph * graph, AgoSuperNode * supernode) {
    // wait for completion
    int64_t stime = agoGetClockCounter();
    hipError_t err;
    err = hipStreamSynchronize(supernode->hip_stream0);
    if (err != hipSuccess) {
        agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: hipStreamSynchronize(1,%p) failed(%d) for group#%d\n", supernode->hip_stream0, err, supernode->group);
        return -1;
    }
    int64_t etime = agoGetClockCounter();
    graph->gpu_perf.kernel_wait += etime - stime;
    // TODO::ENABLE_DEBUG_DUMP_HIP_BUFFERS implement to dump hip buffers
    return 0;
}

int agoGpuHipSuperNodeFinalize(AgoGraph * graph, AgoSuperNode * supernode) {
    // Super node is not fully supported in hip yet
    // clear the data flags
    for (size_t index = 0; index < supernode->dataList.size(); index++) {
        supernode->dataInfo[index].data_type_flags = 0;
    }
    return 0;
}

static int agoGpuHipDataOutputMarkDirty(AgoGraph * graph, AgoData * data, bool need_access, bool need_write_access) {
    if (data->ref.type == VX_TYPE_IMAGE) {
        // only use image objects that need write access
        if (need_access) {
            if (need_write_access) {
                auto dataToSync = data->u.img.isROI ? data->u.img.roiMasterImage : data;
                dataToSync->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL;
            }
        }
    }
    else if (data->ref.type == VX_TYPE_ARRAY || data->ref.type == VX_TYPE_MATRIX) {
        // only use image objects that need write access
        if (need_access) {
            if (need_write_access) {
                data->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL;
            }
        }
    }
    else if (data->ref.type == VX_TYPE_TENSOR) {
        // only use tensor objects that need write access
        if (need_access) {
            if (need_write_access) {
                auto dataToSync = data->u.tensor.roiMaster ? data->u.tensor.roiMaster : data;
                dataToSync->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL;
            }
        }
    }
    return 0;
}

static int agoGpuHipDataOutputAtomicSync(AgoGraph * graph, AgoData * data) {
    if (data->ref.type == VX_TYPE_ARRAY) {
        // update number of items
        int64_t stime = agoGetClockCounter();
        vx_uint32 numItems = 0;
        if (data->hip_memory) {
            hipError_t err = hipMemcpyDtoH((void *)&numItems, data->hip_memory, sizeof(vx_uint32));
            if (err) {
                agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: hipMemcpyDtoH() for numitems => %d\n", err);
                return -1;
            }
        }
        int64_t etime = agoGetClockCounter();
        graph->gpu_perf.buffer_read += etime - stime;
        data->u.arr.numitems = numItems;
    }
    else if (data->ref.type == AGO_TYPE_CANNY_STACK) {
        // update number of items and reset it for next use
        int64_t stime = agoGetClockCounter();
        vx_uint32 stack = 0;
        if (data->hip_memory) {
            hipError_t err = hipMemcpyDtoH((void *)&stack, data->hip_memory, sizeof(vx_uint32));
            if (err) {
                agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: hipMemcpyDtoH() for stacktop => %d\n", err);
                return -1;
            }
        }
        int64_t etime = agoGetClockCounter();
        graph->gpu_perf.buffer_read += etime - stime;
        data->u.cannystack.stackTop = stack;
        if (data->u.cannystack.stackTop > 0) {
            if (data->hip_memory) {
                hipError_t err = hipMemcpyDtoH((void *)data->buffer, (data->hip_memory + data->gpu_buffer_offset), data->u.cannystack.stackTop * sizeof(ago_coord2d_ushort_t));
                if (err) {
                    agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: hipMemcpyDtoH() for stacktop => %d\n", err);
                    return -1;
                }
            }
            int64_t etime = agoGetClockCounter();
            graph->gpu_perf.buffer_read += etime - stime;
        }
    }
    return 0;
}

int agoGpuHipSingleNodeLaunch(AgoGraph * graph, AgoNode * node) {
    // make sure that all input buffers are synched and other arguments are updated
    for (size_t index = 0; index < node->paramCount; index++) {
        if (node->paramList[index] ) {
            bool need_read_access = node->parameters[index].direction != VX_OUTPUT ? true : false;
            vx_uint32 dataFlags = NODE_HIP_TYPE_NEED_IMGSIZE;
            if (agoGpuHipDataInputSync(graph, node->paramList[index], dataFlags, 0, true, need_read_access) < 0) {
                return -1;
            }
        }
    }

    // call execute kernel
    AgoKernel * kernel = node->akernel;
    vx_status status = VX_SUCCESS;
    int64_t stime = agoGetClockCounter();
    if (kernel->func) {
        status = kernel->func(node, ago_kernel_cmd_hip_execute);
        if (status == AGO_ERROR_KERNEL_NOT_IMPLEMENTED) {
            status = VX_ERROR_NOT_IMPLEMENTED;
        }
    }
    else if (kernel->kernel_f) {
        status = kernel->kernel_f(node, (vx_reference *)node->paramList, node->paramCount);
    }
    if (status) {
        if (status == VX_ERROR_GRAPH_ABANDONED)
            agoAddLogEntry((vx_reference)graph, VX_FAILURE, "INFO: kernel %s exec returned graph_stopped status: (this could mean EOS for amd_media extension (%d))\n", kernel->name, status);
        else
            agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: kernel %s exec failed (%d:%s)\n", kernel->name, status, agoEnum2Name(status));
        return -1;
    }
    if(graph->enable_node_level_gpu_flush) {
        hipError_t err = hipStreamSynchronize(graph->hip_stream0);
        if (err) {
            agoAddLogEntry(&node->ref, VX_FAILURE, "ERROR: hipStreamSynchronize(singlenode) failed(%d) for %s\n", err, node->akernel->name);
            return -1;
        }
    }
    int64_t etime = agoGetClockCounter();
    graph->gpu_perf.kernel_enqueue += etime - stime;

    // mark that node outputs are dirty
    for (size_t index = 0; index < node->paramCount; index++) {
        if (node->paramList[index]) {
            bool need_write_access = node->parameters[index].direction != VX_INPUT ? true : false;
            if (agoGpuHipDataOutputMarkDirty(graph, node->paramList[index], true, need_write_access) < 0) {
                    return -1;
            }
        }
    }
    return 0;
}

int agoGpuHipSuperNodeLaunch(AgoGraph * graph, AgoSuperNode * supernode)
{
    // Not implemented
    return -1;
}

int agoGpuHipSingleNodeWait(AgoGraph * graph, AgoNode * node)
{
    // wait for completion
    int64_t stime = agoGetClockCounter();
    hipError_t err;
    err = hipStreamSynchronize(node->hip_stream0);
    if (err != hipSuccess) {
        agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: hipStreamSynchronize(1,%p) failed(%d)\n", node->hip_stream0, err);
        return -1;
    }
    int64_t etime = agoGetClockCounter();
    graph->gpu_perf.kernel_wait += etime - stime;

    // sync the outputs
    for (size_t index = 0; index < node->paramCount; index++) {
        if (node->paramList[index]) {
            bool need_write_access = node->parameters[index].direction != VX_INPUT ? true : false;
            if (need_write_access) {
                if (agoGpuHipDataOutputAtomicSync(graph, node->paramList[index]) < 0) {
                    return -1;
                }
            }
        }
    }

    if (node->gpu_scalar_array_output_sync.enable &&
        node->paramList[node->gpu_scalar_array_output_sync.paramIndexScalar] &&
        node->paramList[node->gpu_scalar_array_output_sync.paramIndexArray]) {
        // updated scalar with numitems of array
        node->paramList[node->gpu_scalar_array_output_sync.paramIndexScalar]->u.scalar.u.s =
            node->paramList[node->gpu_scalar_array_output_sync.paramIndexArray]->u.arr.numitems;
    }

    // The num items in an array should not exceed the capacity unless kernels need it for reporting number of items detected (ex. FAST corners)
    for (size_t index = 0; index < node->paramCount; index++) {
        if (node->paramList[index]) {
            bool need_write_access = node->parameters[index].direction != VX_INPUT ? true : false;
            if (need_write_access) {
                if (node->paramList[index]->ref.type == VX_TYPE_ARRAY) {
                    node->paramList[index]->u.arr.numitems = min(node->paramList[index]->u.arr.numitems, node->paramList[index]->u.arr.capacity);
                }
            }
        }
    }
    return 0;
}
#endif
