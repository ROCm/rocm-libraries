// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <hip/hip_runtime.h>

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
    SdpaKernelPlan(
        hipModule_t module,
        hipFunction_t function,
        int64_t qUid, int64_t kUid, int64_t vUid, int64_t oUid,
        size_t batchSize, size_t numHeadsQ, size_t numHeadsKv,
        size_t seqLenQ, size_t seqLenKv, size_t headDimQk, size_t headDimV,
        size_t qStrideSeq, size_t qStrideRow, size_t qStrideHead, size_t qStrideBatch,
        size_t kStrideSeq, size_t kStrideHead, size_t kStrideBatch,
        size_t vStrideSeq, size_t vStrideHead, size_t vStrideBatch,
        size_t oStrideSeq, size_t oStrideHead, size_t oStrideBatch,
        float attnScale);

    ~SdpaKernelPlan() override;

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
    size_t _batchSize;       // B
    size_t _numHeadsQ;       // H_q
    size_t _numHeadsKv;      // H_kv
    size_t _seqLenQ;         // S_q
    size_t _seqLenKv;        // S_kv
    size_t _headDimQk;       // D_qk (128 for POC)
    size_t _headDimV;        // D_v

    // Q tensor strides (in elements)
    size_t _qStrideSeq;
    size_t _qStrideRow;
    size_t _qStrideHead;
    size_t _qStrideBatch;

    // K tensor strides (in elements)
    size_t _kStrideSeq;
    size_t _kStrideHead;
    size_t _kStrideBatch;

    // V tensor strides (in elements)
    size_t _vStrideSeq;
    size_t _vStrideHead;
    size_t _vStrideBatch;

    // O tensor strides (in elements)
    size_t _oStrideSeq;
    size_t _oStrideHead;
    size_t _oStrideBatch;

    // Attention scale
    float _attnScale;
};

}
