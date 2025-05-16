// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <atomic>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "rocfft/rocfft.h"

#include "logging.h"
#include "plan.h"
#include "rtc_kernel.h"
#include "transform.h"
#include "tuning_helper.h"

#include "kernel_launch.h"

#include "function_pool.h"

#include "real2complex.h"

#include "../../shared/array_predicate.h"
#include "../../shared/environment.h"
#include "../../shared/fft_hash.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/precision_type.h"
#include "../../shared/printbuffer.h"
#include "../../shared/ptrdiff.h"
#include "../../shared/rocfft_complex.h"
#include "../../shared/rocfft_hip.h"

// This function is called during creation of plan: enqueue the HIP kernels by function
// pointers. Return true if everything goes well. Any internal device memory allocation
// failure returns false right away.
bool PlanPowX(ExecPlan& execPlan)
{
    for(const auto& node : execPlan.execSeq)
    {
        if(node->CreateDeviceResources() == false)
            return false;

        if(node->CreateDevKernelArgs() == false)
            return false;
    }

    for(const auto& node : execPlan.execSeq)
    {
        GridParam gp;

        // if this kernel is runtime compiled, use the grid params
        // from compilation as a default.  the node is free to
        // override this default in its SetupGridParam_internal
        // method.
        auto& rtcKernel = node->compiledKernel.get();
        if(rtcKernel)
        {
            gp.b_x   = rtcKernel->gridDim.x;
            gp.b_y   = rtcKernel->gridDim.y;
            gp.b_z   = rtcKernel->gridDim.z;
            gp.wgs_x = rtcKernel->blockDim.x;
            gp.wgs_y = rtcKernel->blockDim.y;
            gp.wgs_z = rtcKernel->blockDim.z;
        }
        node->SetupGridParam(gp);

        execPlan.gridParam.push_back(gp);
    }

    return true;
}

bool GetTuningKernelInfo(ExecPlan& execPlan)
{
    auto tuningPacket = TuningBenchmarker::GetSingleton().GetPacket();
    if(!tuningPacket)
        return false;

    for(size_t i = 0; i < execPlan.execSeq.size(); ++i)
    {
        TreeNode*    curNode             = execPlan.execSeq[i];
        RTCKernel*   localCompiledKernel = curNode->compiledKernel.get().get();
        GridParam    gp                  = execPlan.gridParam[i];
        FMKey        key                 = curNode->GetKernelKey();
        auto         lengths             = key.lengths;
        auto         scheme              = key.scheme;
        KernelConfig config              = key.kernel_config;

        // get occupancy: 0 means it's compiled (AOT)
        //               -1 means failed on getting occupancy
        // TODO- get occupancy of non-RTCKernel
        int occupancy = 0;
        if(localCompiledKernel)
        {
            // if queried occupancy = 0, which is very likely that this kernel
            // can't be loaded
            if(!localCompiledKernel->get_occupancy(
                   {gp.wgs_x, gp.wgs_y, gp.wgs_z}, gp.lds_bytes, occupancy)
               || occupancy == 0)
                occupancy = -1;
        }

        // factors as string, we will output this to CSV file,
        // and carry this from phase-0 to phase-1
        std::string factors_str = "[";
        std::string COMMA       = "";
        for(auto factor : config.factors)
        {
            factors_str += COMMA + std::to_string(factor);
            COMMA = ", ";
        }
        factors_str += "]";

        // utilization info as string, we will output this to CSV file
        std::stringstream util_ss;
        util_ss << "[";
        util_ss.precision(4);
        float util_rate = 0.0f;
        for(auto width : config.factors)
        {
            float height = static_cast<float>(lengths[0]) / width / config.threads_per_transform[0];
            util_ss << std::fixed << height << ", ";

            util_rate += height;
        }
        util_rate /= config.factors.size();
        util_ss << std::fixed << util_rate << "]";

        // 2D_SINGLE can get this value easily, others are not that naive
        tuningPacket->globalRW_per_thread[i]
            = (scheme == CS_KERNEL_2D_SINGLE) ? (lengths[0] * lengths[1]) / config.workgroup_size
                                              : -1;
        tuningPacket->num_of_blocks[i] = gp.b_x;
        tuningPacket->lds_bytes[i]     = gp.lds_bytes;
        tuningPacket->occupancy[i]     = occupancy;
        tuningPacket->wgs[i]           = config.workgroup_size;
        tuningPacket->tpt0[i]          = config.threads_per_transform[0];
        tuningPacket->tpt1[i]          = config.threads_per_transform[1];
        tuningPacket->tpb[i]           = config.transforms_per_block;
        tuningPacket->util_rate[i]     = util_ss.str();
        tuningPacket->factors_str[i]   = factors_str;

        // save some data back to kernel-config (to the saved references)
        // since we've done buffer-assignment and collapse-dim now.
        if(tuningPacket->is_builtin_kernel[i] == false)
        {
            KernelConfig* sol_kernel_config = execPlan.sol_kernel_configs[i];
            sol_kernel_config->static_dim   = curNode->GetStaticDim();
            sol_kernel_config->iAryType     = curNode->inArrayType;
            sol_kernel_config->oAryType     = curNode->outArrayType;
            sol_kernel_config->placement
                = (curNode->placement == rocfft_placement_inplace) ? PC_IP : PC_OP;
        }
    }

    return true;
}

static size_t data_size_bytes(const std::vector<size_t>& lengths,
                              rocfft_precision           precision,
                              rocfft_array_type          type)
{
    // first compute the raw number of elements
    const size_t elems = product(lengths.begin(), lengths.end());
    // size of each element
    const size_t elemsize = real_type_size(precision);
    switch(type)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_complex_planar:
        // complex needs two numbers per element
        return 2 * elems * elemsize;
    case rocfft_array_type_real:
        // real needs one number per element
        return elems * elemsize;
    case rocfft_array_type_hermitian_interleaved:
    case rocfft_array_type_hermitian_planar:
    {
        // hermitian requires 2 numbers per element, but innermost
        // dimension is cut down to roughly half
        size_t non_innermost = elems / lengths[0];
        return 2 * non_innermost * elemsize * ((lengths[0] / 2) + 1);
    }
    case rocfft_array_type_unset:
        // we should really have an array type at this point
        assert(false);
        return 0;
    }
}

static float execution_bandwidth_GB_per_s(size_t data_size_bytes, float duration_ms)
{
    // divide bytes by (1000000 * milliseconds) to get GB/s
    return static_cast<float>(data_size_bytes) / (1000000.0 * duration_ms);
}

// NOTE: HIP returns the maximum global frequency in kHz, which might
// not be the actual frequency when the transform ran.  This function
// might also return 0.0 if the bandwidth can't be queried.
static float max_memory_bandwidth_GB_per_s()
{
    // Try to get the device bandwidth from an environment variable:
    auto pdevbw = rocfft_getenv("ROCFFT_DEVICE_BW");
    if(!pdevbw.empty())
    {
        return atof(pdevbw.c_str());
    }

    // Try to get the device bandwidth from hip calls:
    int deviceid = 0;
    if(hipGetDevice(&deviceid) != hipSuccess)
        // default to first device
        deviceid = 0;
    int max_memory_clock_kHz = 0;
    int memory_bus_width     = 0;
    if(hipDeviceGetAttribute(&max_memory_clock_kHz, hipDeviceAttributeMemoryClockRate, deviceid)
       != hipSuccess)
        max_memory_clock_kHz = 0;
    if(hipDeviceGetAttribute(&memory_bus_width, hipDeviceAttributeMemoryBusWidth, deviceid)
       != hipSuccess)
        memory_bus_width = 0;
    auto max_memory_clock_MHz = static_cast<float>(max_memory_clock_kHz) / 1000.0;
    // multiply by 2.0 because transfer is bidirectional
    // divide by 8.0 because bus width is in bits and we want bytes
    // divide by 1000 to convert MB to GB
    float result = (max_memory_clock_MHz * 2.0 * memory_bus_width / 8.0) / 1000.0;
    return result;
}

// Copy device buffer to host buffer in column-major format with given strides
void CopyDeviceBufferToHost(const rocfft_array_type    type,
                            const rocfft_precision     precision,
                            void*                      buffer[],
                            const std::vector<size_t>& length_cm,
                            const std::vector<size_t>& stride_cm,
                            const size_t               dist,
                            const size_t               batch,
                            std::vector<hostbuf>&      bufvec)
{
    const size_t size_elems = compute_ptrdiff(length_cm, stride_cm, batch, dist);

    size_t base_type_size = real_type_size(precision);
    if(type != rocfft_array_type_real)
    {
        // complex elements
        base_type_size *= 2;
    }

    size_t size_bytes = size_elems * base_type_size;
    if(array_type_is_planar(type))
    {
        // separate the real/imag data, so printbuffer will print them separately
        bufvec.resize(2);
        bufvec.front().alloc(size_bytes / 2);
        bufvec.back().alloc(size_bytes / 2);
        if(hipMemcpy(bufvec.front().data(), buffer[0], size_bytes / 2, hipMemcpyDeviceToHost)
           != hipSuccess)
            throw std::runtime_error("hipMemcpy failure");
        if(hipMemcpy(bufvec.back().data(), buffer[1], size_bytes / 2, hipMemcpyDeviceToHost)
           != hipSuccess)
            throw std::runtime_error("hipMemcpy failure");
    }
    else
    {
        bufvec.resize(1);
        bufvec.front().alloc(size_bytes);
        if(hipMemcpy(bufvec.front().data(), buffer[0], size_bytes, hipMemcpyDeviceToHost)
           != hipSuccess)
            throw std::runtime_error("hipMemcpy failure");
    }
}

// Print buffer 64-bit hash identifier, given column-major dimensions
void DebugPrintHash(rocfft_ostream&             stream,
                    rocfft_array_type           type,
                    rocfft_precision            precision,
                    const std::vector<hostbuf>& bufvec,
                    const std::vector<size_t>&  length_cm,
                    const std::vector<size_t>&  stride_cm,
                    size_t                      dist,
                    size_t                      batch)
{
    auto hash_in  = hash_input(precision, length_cm, stride_cm, dist, type, batch);
    auto hash_out = hash_output<size_t>();
    compute_hash(bufvec, hash_in, hash_out);

    stream << "(" << hash_out.buffer_real << "," << hash_out.buffer_imag << ")" << std::endl;
}

// Print either an input or output buffer, given column-major dimensions
void DebugPrintBuffer(rocfft_ostream&             stream,
                      rocfft_array_type           type,
                      rocfft_precision            precision,
                      const std::vector<hostbuf>& bufvec,
                      const std::vector<size_t>&  length_cm,
                      const std::vector<size_t>&  stride_cm,
                      size_t                      dist,
                      size_t                      batch)
{
    // convert length, stride to row-major for use with printbuffer
    auto length_rm = length_cm;
    auto stride_rm = stride_cm;
    std::reverse(length_rm.begin(), length_rm.end());
    std::reverse(stride_rm.begin(), stride_rm.end());
    std::vector<size_t> print_offset{0, 0};

    if(array_type_is_planar(type))
    {
        switch(precision)
        {
        case rocfft_precision_half:
        {
            buffer_printer<rocfft_fp16> s;
            s.print_buffer_half(bufvec, length_rm, stride_rm, batch, dist, print_offset, stream);
            break;
        }
        case rocfft_precision_single:
        {
            buffer_printer<float> s;
            s.print_buffer_single(bufvec, length_rm, stride_rm, batch, dist, print_offset, stream);
            break;
        }
        case rocfft_precision_double:
        {
            buffer_printer<double> s;
            s.print_buffer_double(bufvec, length_rm, stride_rm, batch, dist, print_offset, stream);
            break;
        }
        }
    }
    else
    {
        switch(precision)
        {
        case rocfft_precision_half:
        {
            switch(type)
            {
            case rocfft_array_type_complex_interleaved:
            case rocfft_array_type_hermitian_interleaved:
            {
                buffer_printer<rocfft_complex<rocfft_fp16>> s;
                s.print_buffer_half(
                    bufvec, length_rm, stride_rm, batch, dist, print_offset, stream);
                break;
            }
            case rocfft_array_type_real:
            {
                buffer_printer<rocfft_fp16> s;
                s.print_buffer_half(
                    bufvec, length_rm, stride_rm, batch, dist, print_offset, stream);
                break;
            }
            default:
                throw std::runtime_error("invalid array format");
            }
            break;
        }
        case rocfft_precision_single:
        {
            switch(type)
            {
            case rocfft_array_type_complex_interleaved:
            case rocfft_array_type_hermitian_interleaved:
            {
                buffer_printer<rocfft_complex<float>> s;
                s.print_buffer_single(
                    bufvec, length_rm, stride_rm, batch, dist, print_offset, stream);
                break;
            }
            case rocfft_array_type_real:
            {
                buffer_printer<float> s;
                s.print_buffer_single(
                    bufvec, length_rm, stride_rm, batch, dist, print_offset, stream);
                break;
            }
            default:
                throw std::runtime_error("invalid array format");
            }
            break;
        }
        case rocfft_precision_double:
        {
            switch(type)
            {
            case rocfft_array_type_complex_interleaved:
            case rocfft_array_type_hermitian_interleaved:
            {
                buffer_printer<rocfft_complex<double>> s;
                s.print_buffer_double(
                    bufvec, length_rm, stride_rm, batch, dist, print_offset, stream);
                break;
            }
            case rocfft_array_type_real:
            {
                buffer_printer<double> s;
                s.print_buffer_double(
                    bufvec, length_rm, stride_rm, batch, dist, print_offset, stream);
                break;
            }
            default:
                throw std::runtime_error("invalid array format");
            }
            break;
        }
        }
    }
}

enum struct SetCallbackType
{
    LOAD,
    STORE,
};

void SetDefaultCallback(const TreeNode* node, const SetCallbackType& type, void** cb)
{
    auto result = hipSuccess;

    auto array_type = (type == SetCallbackType::LOAD) ? node->inArrayType : node->outArrayType;
    auto node_callback_type = node->GetCallbackType(true);

    bool is_complex = array_type_is_complex(array_type);
    // load r2c kernels and store c2r kernels need real-valued callbacks
    if((type == SetCallbackType::LOAD && node_callback_type == CallbackType::USER_LOAD_STORE_R2C)
       || (type == SetCallbackType::STORE
           && node_callback_type == CallbackType::USER_LOAD_STORE_C2R))
        is_complex = false;

    if(is_complex && type == SetCallbackType::LOAD)
    {
        switch(node->precision)
        {
        case rocfft_precision_half:
            result
                = hipMemcpyFromSymbol(cb, HIP_SYMBOL(load_cb_default_complex_half), sizeof(void*));
            break;
        case rocfft_precision_single:
            result
                = hipMemcpyFromSymbol(cb, HIP_SYMBOL(load_cb_default_complex_float), sizeof(void*));
            break;
        case rocfft_precision_double:
            result = hipMemcpyFromSymbol(
                cb, HIP_SYMBOL(load_cb_default_complex_double), sizeof(void*));
            break;
        }
    }
    else if(is_complex && type == SetCallbackType::STORE)
    {
        switch(node->precision)
        {
        case rocfft_precision_half:
            result
                = hipMemcpyFromSymbol(cb, HIP_SYMBOL(store_cb_default_complex_half), sizeof(void*));
            break;
        case rocfft_precision_single:
            result = hipMemcpyFromSymbol(
                cb, HIP_SYMBOL(store_cb_default_complex_float), sizeof(void*));
            break;
        case rocfft_precision_double:
            result = hipMemcpyFromSymbol(
                cb, HIP_SYMBOL(store_cb_default_complex_double), sizeof(void*));
            break;
        }
    }
    else if(!is_complex && type == SetCallbackType::LOAD)
    {
        switch(node->precision)
        {
        case rocfft_precision_half:
            result = hipMemcpyFromSymbol(cb, HIP_SYMBOL(load_cb_default_half), sizeof(void*));
            break;
        case rocfft_precision_single:
            result = hipMemcpyFromSymbol(cb, HIP_SYMBOL(load_cb_default_float), sizeof(void*));
            break;
        case rocfft_precision_double:
            result = hipMemcpyFromSymbol(cb, HIP_SYMBOL(load_cb_default_double), sizeof(void*));
            break;
        }
    }
    else if(!is_complex && type == SetCallbackType::STORE)
    {
        switch(node->precision)
        {
        case rocfft_precision_half:
            result = hipMemcpyFromSymbol(cb, HIP_SYMBOL(store_cb_default_half), sizeof(void*));
            break;
        case rocfft_precision_single:
            result = hipMemcpyFromSymbol(cb, HIP_SYMBOL(store_cb_default_float), sizeof(void*));
            break;
        case rocfft_precision_double:
            result = hipMemcpyFromSymbol(cb, HIP_SYMBOL(store_cb_default_double), sizeof(void*));
            break;
        }
    }

    if(result != hipSuccess)
        throw std::runtime_error("hipMemcpyFromSymbol failure");
}

// Internal plan executor.
// For in-place transforms, in_buffer == out_buffer.
void TransformPowX(const ExecPlan&       execPlan,
                   void*                 in_buffer[],
                   void*                 out_buffer[],
                   rocfft_execution_info info,
                   size_t                multiPlanIdx)
{
    assert(execPlan.execSeq.size() == execPlan.gridParam.size());

    bool processing_tuning = TuningBenchmarker::GetSingleton().IsProcessingTuning();
    auto tuningPacket      = TuningBenchmarker::GetSingleton().GetPacket();
    // we can log profile information if we're on the null stream,
    // since we will be able to wait for the transform to finish
    bool emit_profile_log  = (processing_tuning || LOG_PROFILE_ENABLED()) && !info->rocfft_stream;
    bool emit_kernelio_log = LOG_KERNELIO_ENABLED();

    rocfft_ostream*    kernelio_stream = nullptr;
    float              max_memory_bw   = 0.0;
    hipEvent_wrapper_t start, stop;
    if(emit_profile_log)
    {
        start.alloc();
        stop.alloc();
        max_memory_bw = max_memory_bandwidth_GB_per_s();
    }

    // assign callbacks to the node that are actually doing the
    // loading and storing to/from global memory
    TreeNode* load_node             = nullptr;
    TreeNode* store_node            = nullptr;
    std::tie(load_node, store_node) = execPlan.get_load_store_nodes();

    load_node->callbacks.load_cb_fn        = info->callbacks.load_cb_fn;
    load_node->callbacks.load_cb_data      = info->callbacks.load_cb_data;
    load_node->callbacks.load_cb_lds_bytes = info->callbacks.load_cb_lds_bytes;

    store_node->callbacks.store_cb_fn        = info->callbacks.store_cb_fn;
    store_node->callbacks.store_cb_data      = info->callbacks.store_cb_data;
    store_node->callbacks.store_cb_lds_bytes = info->callbacks.store_cb_lds_bytes;

    for(size_t i = 0; i < execPlan.execSeq.size(); i++)
    {
        DeviceCallIn data;
        data.node          = execPlan.execSeq[i];
        data.rocfft_stream = (info == nullptr) ? 0 : info->rocfft_stream;
        data.deviceProp    = execPlan.deviceProp;

        // Size of complex type
        const size_t complexTSize = complex_type_size(data.node->precision);

        switch(data.node->obIn)
        {
        case OB_USER_IN:
            data.bufIn[0] = in_buffer[0];
            if(data.node->inArrayType == rocfft_array_type_complex_planar
               || data.node->inArrayType == rocfft_array_type_hermitian_planar)
            {
                data.bufIn[1] = in_buffer[1];
            }
            break;
        case OB_USER_OUT:
            data.bufIn[0] = out_buffer[0];
            if(data.node->inArrayType == rocfft_array_type_complex_planar
               || data.node->inArrayType == rocfft_array_type_hermitian_planar)
            {
                data.bufIn[1] = out_buffer[1];
            }
            break;
        case OB_TEMP:
            data.bufIn[0] = info->workBuffer;
            if(data.node->inArrayType == rocfft_array_type_complex_planar
               || data.node->inArrayType == rocfft_array_type_hermitian_planar)
            {
                // Assume planar using the same extra size of memory as
                // interleaved format, and we just need to split it for
                // planar.
                data.bufIn[1]
                    = (void*)((char*)info->workBuffer + execPlan.tmpWorkBufSize * complexTSize / 2);
            }
            break;
        case OB_TEMP_CMPLX_FOR_REAL:
            data.bufIn[0]
                = (void*)((char*)info->workBuffer + execPlan.tmpWorkBufSize * complexTSize);
            // TODO: Can we use this in planar as well ??
            // if(data.node->inArrayType == rocfft_array_type_complex_planar
            //    || data.node->inArrayType == rocfft_array_type_hermitian_planar)
            // {
            //     data.bufIn[1] = (void*)((char*)info->workBuffer
            //                             + (execPlan.tmpWorkBufSize + execPlan.copyWorkBufSize / 2)
            //                                   * complexTSize);
            // }
            break;
        case OB_TEMP_BLUESTEIN:
            data.bufIn[0]
                = (void*)((char*)info->workBuffer
                          + (execPlan.tmpWorkBufSize + execPlan.copyWorkBufSize) * complexTSize);
            // Bluestein mul-kernels (3 types) work well for CI->CI
            // so we only consider CI->CI now
            break;
        case OB_UNINIT:
            rocfft_cerr << "Error: operating buffer not initialized for kernel!\n";
            assert(data.node->obIn != OB_UNINIT);
            break;
        default:
            rocfft_cerr << "Error: operating buffer not specified for kernel!\n";
            assert(false);
        }

        switch(data.node->obOut)
        {
        case OB_USER_IN:
            data.bufOut[0] = in_buffer[0];
            if(data.node->outArrayType == rocfft_array_type_complex_planar
               || data.node->outArrayType == rocfft_array_type_hermitian_planar)
            {
                data.bufOut[1] = in_buffer[1];
            }
            break;
        case OB_USER_OUT:
            data.bufOut[0] = out_buffer[0];
            if(data.node->outArrayType == rocfft_array_type_complex_planar
               || data.node->outArrayType == rocfft_array_type_hermitian_planar)
            {
                data.bufOut[1] = out_buffer[1];
            }
            break;
        case OB_TEMP:
            data.bufOut[0] = info->workBuffer;
            if(data.node->outArrayType == rocfft_array_type_complex_planar
               || data.node->outArrayType == rocfft_array_type_hermitian_planar)
            {
                // assume planar using the same extra size of memory as
                // interleaved format, and we just need to split it for
                // planar.
                data.bufOut[1]
                    = (void*)((char*)info->workBuffer + execPlan.tmpWorkBufSize * complexTSize / 2);
            }
            break;
        case OB_TEMP_CMPLX_FOR_REAL:
            data.bufOut[0]
                = (void*)((char*)info->workBuffer + execPlan.tmpWorkBufSize * complexTSize);
            // TODO: Can we use this in planar as well ??
            // if(data.node->outArrayType == rocfft_array_type_complex_planar
            //    || data.node->outArrayType == rocfft_array_type_hermitian_planar)
            // {
            //     data.bufOut[1] = (void*)((char*)info->workBuffer
            //                              + (execPlan.tmpWorkBufSize + execPlan.copyWorkBufSize / 2)
            //                                    * complexTSize);
            // }
            break;
        case OB_TEMP_BLUESTEIN:
            data.bufOut[0]
                = (void*)((char*)info->workBuffer
                          + (execPlan.tmpWorkBufSize + execPlan.copyWorkBufSize) * complexTSize);
            // Bluestein mul-kernels (3 types) work well for CI->CI
            // so we only consider CI->CI now
            break;
        default:
            assert(false);
        }

        // apply offsets to pointers
        if(data.node->iOffset)
        {
            if(data.bufIn[0])
                data.bufIn[0] = ptr_offset(data.bufIn[0],
                                           data.node->iOffset,
                                           data.node->precision,
                                           data.node->inArrayType);
            if(data.bufIn[1])
                data.bufIn[1] = ptr_offset(data.bufIn[1],
                                           data.node->iOffset,
                                           data.node->precision,
                                           data.node->inArrayType);
        }
        if(data.node->oOffset)
        {
            if(data.bufOut[0])
                data.bufOut[0] = ptr_offset(data.bufOut[0],
                                            data.node->oOffset,
                                            data.node->precision,
                                            data.node->outArrayType);
            if(data.bufOut[1])
                data.bufOut[1] = ptr_offset(data.bufOut[1],
                                            data.node->oOffset,
                                            data.node->precision,
                                            data.node->outArrayType);
        }

        // single-kernel bluestein requires a bluestein temp buffer separate from input and output
        if(data.node->scheme == CS_KERNEL_BLUESTEIN_SINGLE)
        {
            data.bufTemp = ((char*)info->workBuffer
                            + (execPlan.tmpWorkBufSize + execPlan.copyWorkBufSize) * complexTSize);
        }

        // if callbacks are enabled, make sure load_cb_fn and store_cb_fn are not nullptrs
        if((data.node->callbacks.load_cb_fn == nullptr
            && data.node->callbacks.store_cb_fn != nullptr))
        {
            // set default load callback
            SetDefaultCallback(data.node, SetCallbackType::LOAD, &data.node->callbacks.load_cb_fn);
        }
        else if((data.node->callbacks.load_cb_fn != nullptr
                 && data.node->callbacks.store_cb_fn == nullptr))
        {
            // set default store callback
            SetDefaultCallback(
                data.node, SetCallbackType::STORE, &data.node->callbacks.store_cb_fn);
        }

        data.gridParam = execPlan.gridParam[i];

        // chirp kernel has no input - it constructs the chirp buffer from nothing
        if(emit_kernelio_log && data.node->scheme != CS_KERNEL_CHIRP)
        {
            kernelio_stream = LogSingleton::GetInstance().GetKernelIOOS();
            *kernelio_stream << "--- --- multiPlanIdx " << multiPlanIdx << " kernel " << i << " ("
                             << PrintScheme(data.node->scheme) << ") input: " << std::endl;
            if(hipDeviceSynchronize() != hipSuccess)
                throw std::runtime_error("hipDeviceSynchronize failure");

            std::vector<hostbuf> bufInHost;
            CopyDeviceBufferToHost(data.node->inArrayType,
                                   data.node->precision,
                                   data.bufIn,
                                   data.node->length,
                                   data.node->inStride,
                                   data.node->iDist,
                                   data.node->batch,
                                   bufInHost);

            DebugPrintBuffer(*kernelio_stream,
                             data.node->inArrayType,
                             data.node->precision,
                             bufInHost,
                             data.node->length,
                             data.node->inStride,
                             data.node->iDist,
                             data.node->batch);
            *kernelio_stream << "--- --- multiPlanIdx " << multiPlanIdx << " kernel " << i << " ("
                             << PrintScheme(data.node->scheme) << ") input hash: " << std::endl;
            DebugPrintHash(*kernelio_stream,
                           data.node->inArrayType,
                           data.node->precision,
                           bufInHost,
                           data.node->length,
                           data.node->inStride,
                           data.node->iDist,
                           data.node->batch);
            *kernelio_stream << std::endl;
        }

#ifdef REF_DEBUG
        rocfft_cout << "\n---------------------------------------------\n";
        rocfft_cout << "\n\nkernel: " << i << std::endl;
        rocfft_cout << "\tscheme: " << PrintScheme(execPlan.execSeq[i]->scheme) << std::endl;
        rocfft_cout << "\titype: " << execPlan.execSeq[i]->inArrayType << std::endl;
        rocfft_cout << "\totype: " << execPlan.execSeq[i]->outArrayType << std::endl;
        rocfft_cout << "\tlength: ";
        for(const auto& i : execPlan.execSeq[i]->length)
        {
            rocfft_cout << i << " ";
        }
        rocfft_cout << std::endl;
        rocfft_cout << "\tbatch:   " << execPlan.execSeq[i]->batch << std::endl;
        rocfft_cout << "\tidist:   " << execPlan.execSeq[i]->iDist << std::endl;
        rocfft_cout << "\todist:   " << execPlan.execSeq[i]->oDist << std::endl;
        rocfft_cout << "\tistride:";
        for(const auto& i : execPlan.execSeq[i]->inStride)
        {
            rocfft_cout << " " << i;
        }
        rocfft_cout << std::endl;
        rocfft_cout << "\tostride:";
        for(const auto& i : execPlan.execSeq[i]->outStride)
        {
            rocfft_cout << " " << i;
        }
        rocfft_cout << std::endl;

        RefLibOp refLibOp(&data);
#endif

        // execution kernel:
        if(emit_profile_log)
            if(hipEventRecord(start) != hipSuccess)
                throw std::runtime_error("hipEventRecord failure");

        // give callback parameters to kernel launcher
        data.callbacks = execPlan.execSeq[i]->callbacks;

        // choose which compiled kernel to run
        RTCKernel* localCompiledKernel = data.get_callback_type() == CallbackType::NONE
                                             ? data.node->compiledKernel.get().get()
                                             : data.node->compiledKernelWithCallbacks.get().get();

        if(localCompiledKernel)
            localCompiledKernel->launch(data, data.node->deviceProp);
        else
            throw std::runtime_error("rocFFT null ptr function call error");

        if(emit_profile_log)
            if(hipEventRecord(stop) != hipSuccess)
                throw std::runtime_error("hipEventRecord failure");

        // If we were on the null stream, measure elapsed time
        // and emit profile logging.  If a stream was given, we
        // can't wait for the transform to finish, so we can't
        // emit any information.
        if(emit_profile_log)
        {
            if(hipEventSynchronize(stop) != hipSuccess)
                throw std::runtime_error("hipEventSynchronize failure");
            size_t in_size_bytes
                = data_size_bytes(data.node->length, data.node->precision, data.node->inArrayType);
            size_t out_size_bytes
                = data_size_bytes(data.node->length, data.node->precision, data.node->outArrayType);
            size_t total_size_bytes = (in_size_bytes + out_size_bytes) * data.node->batch;

            float duration_ms = 0.0f;
            if(hipEventElapsedTime(&duration_ms, start, stop) != hipSuccess)
                throw std::runtime_error("hipEventElapsedTime failure");
            auto exec_bw        = execution_bandwidth_GB_per_s(total_size_bytes, duration_ms);
            auto efficiency_pct = 0.0;
            if(max_memory_bw != 0.0)
                efficiency_pct = 100.0 * exec_bw / max_memory_bw;
            if(processing_tuning)
                tuningPacket->bw_effs[i] = efficiency_pct;

            log_profile(__func__,
                        "scheme",
                        PrintScheme(execPlan.execSeq[i]->scheme),
                        "duration_ms",
                        duration_ms,
                        "in_size",
                        std::make_pair(static_cast<const size_t*>(data.node->length.data()),
                                       data.node->length.size()),
                        "total_size_bytes",
                        total_size_bytes,
                        "exec_GB_s",
                        exec_bw,
                        "max_mem_GB_s",
                        max_memory_bw,
                        "bw_efficiency_pct",
                        efficiency_pct,
                        "kernel_index",
                        i);
        }

#ifdef REF_DEBUG
        refLibOp.VerifyResult(&data);
#endif

        if(emit_kernelio_log && data.node->scheme != CS_KERNEL_CHIRP)
        {
            hipError_t err = hipPeekAtLastError();
            if(err != hipSuccess)
            {
                *kernelio_stream << "Error: " << hipGetErrorName(err) << ", "
                                 << hipGetErrorString(err) << std::endl;
                throw std::runtime_error("hipPeekAtLastError failed");
            }
            if(hipDeviceSynchronize() != hipSuccess)
                throw std::runtime_error("hipDeviceSynchronize failure");
            *kernelio_stream << "executed kernel " << i << " (" << PrintScheme(data.node->scheme)
                             << ")\n\n";
        }
    }

    if(emit_kernelio_log)
    {
        // offsets have only been applied to pointers given to kernels,
        // so apply them here for printing too
        void* out_buffer_offset[2] = {out_buffer[0], nullptr};
        if(execPlan.rootPlan->oOffset)
        {
            out_buffer_offset[0] = ptr_offset(out_buffer_offset[0],
                                              execPlan.rootPlan->oOffset,
                                              execPlan.rootPlan->precision,
                                              execPlan.rootPlan->outArrayType);
        }

        if(array_type_is_planar(execPlan.rootPlan->outArrayType))
        {
            out_buffer_offset[1] = out_buffer[1];

            if(execPlan.rootPlan->oOffset)
                out_buffer_offset[1] = ptr_offset(out_buffer_offset[1],
                                                  execPlan.rootPlan->oOffset,
                                                  execPlan.rootPlan->precision,
                                                  execPlan.rootPlan->outArrayType);
        }

        std::vector<hostbuf> bufOutHost;
        CopyDeviceBufferToHost(execPlan.rootPlan->outArrayType,
                               execPlan.rootPlan->precision,
                               out_buffer_offset,
                               execPlan.rootPlan->GetOutputLength(),
                               execPlan.rootPlan->outStride,
                               execPlan.rootPlan->oDist,
                               execPlan.rootPlan->batch,
                               bufOutHost);

        *kernelio_stream << "multiPlanIdx " << multiPlanIdx << " final output: " << std::endl;
        DebugPrintBuffer(*kernelio_stream,
                         execPlan.rootPlan->outArrayType,
                         execPlan.rootPlan->precision,
                         bufOutHost,
                         execPlan.rootPlan->GetOutputLength(),
                         execPlan.rootPlan->outStride,
                         execPlan.rootPlan->oDist,
                         execPlan.rootPlan->batch);
        *kernelio_stream << "multiPlanIdx " << multiPlanIdx << " final output hash: " << std::endl;
        DebugPrintHash(*kernelio_stream,
                       execPlan.rootPlan->outArrayType,
                       execPlan.rootPlan->precision,
                       bufOutHost,
                       execPlan.rootPlan->GetOutputLength(),
                       execPlan.rootPlan->outStride,
                       execPlan.rootPlan->oDist,
                       execPlan.rootPlan->batch);
        *kernelio_stream << std::endl;
    }
}
