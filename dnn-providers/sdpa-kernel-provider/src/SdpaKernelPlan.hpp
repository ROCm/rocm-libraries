// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "SdpaKernelHandle.hpp"
#include "SdpaKernelSettings.hpp"

namespace sdpa_kernel_provider
{

/**
* @brief SDPA kernel plan.
*/
class SdpaKernelPlan : public hipdnn_plugin_sdk::IPlan<SdpaKernelHandle>
{
public:
    /**
     * @brief Construct a plan with kernel module and precomputed metadata.
     */
    SdpaKernelPlan(hipModule_t kernelModule,
                   hipFunction_t function,
                   int64_t qUid,
                   int64_t kUid,
                   int64_t vUid,
                   int64_t oUid,
                   unsigned int batchSize,
                   unsigned int numHeadsQ,
                   unsigned int numHeadsKv,
                   unsigned int seqLenQ,
                   unsigned int seqLenKv,
                   unsigned int headDimQk,
                   unsigned int headDimV,
                   unsigned int qStrideSeq,
                   unsigned int qStrideRow,
                   unsigned int qStrideHead,
                   unsigned int qStrideBatch,
                   unsigned int kStrideSeq,
                   unsigned int kStrideHead,
                   unsigned int kStrideBatch,
                   unsigned int vStrideSeq,
                   unsigned int vStrideHead,
                   unsigned int vStrideBatch,
                   unsigned int oStrideSeq,
                   unsigned int oStrideHead,
                   unsigned int oStrideBatch,
                   float attnScale);

    ~SdpaKernelPlan() override;

    // Delete copy operations (resource ownership)
    SdpaKernelPlan(const SdpaKernelPlan&) = delete;
    SdpaKernelPlan& operator=(const SdpaKernelPlan&) = delete;

    // Move operations
    SdpaKernelPlan(SdpaKernelPlan&& other) noexcept;
    SdpaKernelPlan& operator=(SdpaKernelPlan&& other) noexcept;

    size_t getWorkspaceSize(const SdpaKernelHandle& handle) const override;

    void execute(const SdpaKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    hipModule_t _module;
    hipFunction_t _function;

    // Tensor UIDs
    int64_t _qUid;
    int64_t _kUid;
    int64_t _vUid;
    int64_t _oUid;

    // Tensor dimensions
    unsigned int _batchSize; // B
    unsigned int _numHeadsQ; // H_q
    unsigned int _numHeadsKv; // H_kv
    unsigned int _seqLenQ; // S_q
    unsigned int _seqLenKv; // S_kv
    unsigned int _headDimQk; // D_qk (128 for POC)
    unsigned int _headDimV; // D_v

    // Q tensor strides (in elements)
    unsigned int _qStrideSeq;
    unsigned int _qStrideRow;
    unsigned int _qStrideHead;
    unsigned int _qStrideBatch;

    // K tensor strides (in elements)
    unsigned int _kStrideSeq;
    unsigned int _kStrideHead;
    unsigned int _kStrideBatch;

    // V tensor strides (in elements)
    unsigned int _vStrideSeq;
    unsigned int _vStrideHead;
    unsigned int _vStrideBatch;

    // O tensor strides (in elements)
    unsigned int _oStrideSeq;
    unsigned int _oStrideHead;
    unsigned int _oStrideBatch;

    // Attention scale
    float _attnScale;
};

}
