// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaBwdPlan.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../../runtime/KernelArtifact.hpp"
#include "../../runtime/LaunchAbi.hpp"

namespace ck_dsl_provider {

namespace {

/// Linear scan over the device-buffer array. The array is typically
/// <16 entries so an O(n) lookup is the right shape. Throws with the
/// missing uid in the message so a graph-vs-buffer mismatch surfaces
/// with concrete context.
const hipdnnPluginDeviceBuffer_t& findDeviceBuffer(std::int64_t uid,
                                                   const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                                   std::uint32_t numDeviceBuffers,
                                                   const char* role) {
    for (std::uint32_t i = 0; i < numDeviceBuffers; ++i) {
        if (deviceBuffers[i].uid == uid) {
            return deviceBuffers[i];
        }
    }
    std::ostringstream oss;
    oss << "SdpaBwdPlan::execute: no device buffer for " << role << " (uid=" << uid
        << "); searched " << numDeviceBuffers << " entries";
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE, oss.str());
}

void checkHip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::ostringstream oss;
        oss << "SdpaBwdPlan::execute: " << what << " failed: " << hipGetErrorString(err);
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       oss.str());
    }
}

}  // namespace

SdpaBwdPlan::SdpaBwdPlan(
    std::shared_ptr<HipModule> bwdModule, std::shared_ptr<HipModule> prepModule, std::int64_t qUid,
    std::int64_t kUid, std::int64_t vUid, std::int64_t doUid, std::int64_t statsUid,
    std::int64_t dqUid, std::int64_t dkUid, std::int64_t dvUid, std::int32_t B, std::int32_t Hq,
    std::int32_t Hkv, std::int32_t Sq, std::int32_t Skv, std::int32_t D, float scaleLog2,
    float scaleInv, std::int32_t strideQToken, std::int32_t strideQHead, std::int32_t strideKToken,
    std::int32_t strideKHead, std::int32_t strideVToken, std::int32_t strideVHead,
    std::int32_t strideDoToken, std::int32_t strideDoHead, std::int32_t strideDqToken,
    std::int32_t strideDkToken, std::int32_t strideDvToken)
    : _bwdModule(std::move(bwdModule)),
      _prepModule(std::move(prepModule)),
      _qUid(qUid),
      _kUid(kUid),
      _vUid(vUid),
      _doUid(doUid),
      _statsUid(statsUid),
      _dqUid(dqUid),
      _dkUid(dkUid),
      _dvUid(dvUid),
      _B(B),
      _Hq(Hq),
      _Hkv(Hkv),
      _Sq(Sq),
      _Skv(Skv),
      _D(D),
      _scaleLog2(scaleLog2),
      _scaleInv(scaleInv),
      _strideQToken(strideQToken),
      _strideQHead(strideQHead),
      _strideKToken(strideKToken),
      _strideKHead(strideKHead),
      _strideVToken(strideVToken),
      _strideVHead(strideVHead),
      _strideDoToken(strideDoToken),
      _strideDoHead(strideDoHead),
      _strideDqToken(strideDqToken),
      _strideDkToken(strideDkToken),
      _strideDvToken(strideDvToken) {
    if (_bwdModule == nullptr) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "SdpaBwdPlan: refusing to construct with null backward HipModule");
    }
    if (_prepModule == nullptr) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "SdpaBwdPlan: refusing to construct with null LSE-prep HipModule");
    }

    // Validate the backward kernel's argument schema is exactly the
    // 24-slot ABI execute() packs against: 9 Pointer, 2 F32, 13 I32. If a
    // future kernel variant grows or rearranges its parameter list, the
    // plan must refuse to launch rather than pack values into wrong-typed
    // slots.
    {
        const auto& schema = _bwdModule->argSchema();
        auto rejectSlot = [&](const std::string& detail) {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "SdpaBwdPlan: backward kernel argSchema does not match the expected "
                "(Pointer x9, F32 x2, I32 x13) layout for the FMHA-backward kernel signature; " +
                    detail);
        };
        if (schema.size() != 24) {
            std::ostringstream oss;
            oss << "got " << schema.size() << " slots, expected 24";
            rejectSlot(oss.str());
        }
        for (std::size_t i = 0; i < 9; ++i) {
            if (schema[i].kind != ArgSchema::Kind::Pointer) {
                std::ostringstream oss;
                oss << "slot " << i << " must be Pointer";
                rejectSlot(oss.str());
            }
        }
        for (std::size_t i = 9; i < 11; ++i) {
            if (schema[i].kind != ArgSchema::Kind::F32) {
                std::ostringstream oss;
                oss << "slot " << i << " must be F32";
                rejectSlot(oss.str());
            }
        }
        for (std::size_t i = 11; i < 24; ++i) {
            if (schema[i].kind != ArgSchema::Kind::I32) {
                std::ostringstream oss;
                oss << "slot " << i << " must be I32";
                rejectSlot(oss.str());
            }
        }
    }

    // Validate the LSE-prep kernel's argument schema is the 6-slot ABI:
    // 3 Pointer (stats, M, L), 3 I32 (B, Hq, Sq).
    {
        const auto& schema = _prepModule->argSchema();
        auto rejectSlot = [&](const std::string& detail) {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "SdpaBwdPlan: LSE-prep kernel argSchema does not match the expected "
                "(Pointer x3, I32 x3) layout for the LSE-prep kernel signature; " +
                    detail);
        };
        if (schema.size() != 6) {
            std::ostringstream oss;
            oss << "got " << schema.size() << " slots, expected 6";
            rejectSlot(oss.str());
        }
        for (std::size_t i = 0; i < 3; ++i) {
            if (schema[i].kind != ArgSchema::Kind::Pointer) {
                std::ostringstream oss;
                oss << "slot " << i << " must be Pointer";
                rejectSlot(oss.str());
            }
        }
        for (std::size_t i = 3; i < 6; ++i) {
            if (schema[i].kind != ArgSchema::Kind::I32) {
                std::ostringstream oss;
                oss << "slot " << i << " must be I32";
                rejectSlot(oss.str());
            }
        }
    }

    // One-shot launch-shape log: useful operator diagnostic at plan
    // construction without polluting the per-call hot path.
    HIPDNN_PLUGIN_LOG_INFO("SdpaBwdPlan: built plan for bwd kernel '"
                           << _bwdModule->kernelName() << "' + prep kernel '"
                           << _prepModule->kernelName() << "' B=" << _B << " Hq=" << _Hq
                           << " Hkv=" << _Hkv << " seqlen_q=" << _Sq << " seqlen_k=" << _Skv
                           << " head_size=" << _D);
}

std::size_t SdpaBwdPlan::getWorkspaceSize(const ::CkDslHandle& /*handle*/) const {
    // The prep kernel writes two per-(B, Hq, Sq) reductions (M_saved and
    // L_saved) the backward kernel then reads. Each is B * Sq * Hq
    // floats; the workspace holds both back-to-back.
    return static_cast<std::size_t>(2) * static_cast<std::size_t>(_B) *
           static_cast<std::size_t>(_Sq) * static_cast<std::size_t>(_Hq) * sizeof(float);
}

void SdpaBwdPlan::execute(const ::CkDslHandle& handle,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          std::uint32_t numDeviceBuffers, void* workspace) const {
    if (deviceBuffers == nullptr && numDeviceBuffers > 0) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "SdpaBwdPlan::execute: deviceBuffers is null but numDeviceBuffers > 0");
    }

    const std::size_t reductionCount = static_cast<std::size_t>(_B) *
                                       static_cast<std::size_t>(_Sq) *
                                       static_cast<std::size_t>(_Hq);
    if (workspace == nullptr && reductionCount > 0) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "SdpaBwdPlan::execute: workspace is null but the LSE-prep scratch "
            "(2 * B * Sq * Hq floats) is required");
    }

    const auto& qBuf = findDeviceBuffer(_qUid, deviceBuffers, numDeviceBuffers, "Q");
    const auto& kBuf = findDeviceBuffer(_kUid, deviceBuffers, numDeviceBuffers, "K");
    const auto& vBuf = findDeviceBuffer(_vUid, deviceBuffers, numDeviceBuffers, "V");
    const auto& doBuf = findDeviceBuffer(_doUid, deviceBuffers, numDeviceBuffers, "dO");
    const auto& statsBuf = findDeviceBuffer(_statsUid, deviceBuffers, numDeviceBuffers, "stats");
    const auto& dqBuf = findDeviceBuffer(_dqUid, deviceBuffers, numDeviceBuffers, "dQ");
    const auto& dkBuf = findDeviceBuffer(_dkUid, deviceBuffers, numDeviceBuffers, "dK");
    const auto& dvBuf = findDeviceBuffer(_dvUid, deviceBuffers, numDeviceBuffers, "dV");

    void* qPtr = qBuf.ptr;
    void* kPtr = kBuf.ptr;
    void* vPtr = vBuf.ptr;
    void* doPtr = doBuf.ptr;
    void* statsPtr = statsBuf.ptr;
    void* dqPtr = dqBuf.ptr;
    void* dkPtr = dkBuf.ptr;
    void* dvPtr = dvBuf.ptr;

    hipStream_t stream = handle.getStream();

    // Carve the M/L scratch out of the workspace. M occupies the first
    // B*Sq*Hq floats, L the next B*Sq*Hq floats.
    float* M = static_cast<float*>(workspace);
    float* L = M + reductionCount;

    // Element counts for the gradient zero-init. dQ mirrors Q's
    // [B, Hq, Sq, D]; dK/dV mirror K/V's [B, Hkv, Skv, D]. The gradients
    // are f32 accumulators.
    const std::size_t dqElems = static_cast<std::size_t>(_B) * static_cast<std::size_t>(_Sq) *
                                static_cast<std::size_t>(_Hq) * static_cast<std::size_t>(_D);
    const std::size_t dkElems = static_cast<std::size_t>(_B) * static_cast<std::size_t>(_Skv) *
                                static_cast<std::size_t>(_Hkv) * static_cast<std::size_t>(_D);
    const std::size_t dvElems = dkElems;

    // Zero-init the gradients on the stream: the backward kernel
    // accumulates into them across kv tiles, so they must start at zero.
    // Enqueued before the per-batch launches, on the same stream, so the
    // ordering is guaranteed.
    checkHip(hipMemsetAsync(dqPtr, 0, dqElems * sizeof(float), stream), "hipMemsetAsync(dQ)");
    checkHip(hipMemsetAsync(dkPtr, 0, dkElems * sizeof(float), stream), "hipMemsetAsync(dK)");
    checkHip(hipMemsetAsync(dvPtr, 0, dvElems * sizeof(float), stream), "hipMemsetAsync(dV)");

    // LSE-prep launch: derive the M/L scratch from the natural-log stats
    // buffer. ABI: (stats, M, L, B, Hq, Sq).
    {
        std::vector<ArgValue> values = {
            ArgValue::pointer(statsPtr), ArgValue::pointer(M), ArgValue::pointer(L),
            ArgValue::i32(_B),           ArgValue::i32(_Hq),   ArgValue::i32(_Sq),
        };
        std::vector<std::byte> packed = LaunchAbi::pack(_prepModule->argSchema(), values);
        _prepModule->launch(packed.data(), packed.size(), _prepModule->grid(), _prepModule->block(),
                            _prepModule->ldsBytes(), stream);
    }

    // Per-batch backward launches. The kernels carry no batch stride, so
    // the host folds the batch offset into each base pointer:
    //   * HALF (Q/K/V/dO) advance by 2 bytes per element;
    //   * FLOAT (dQ/dK/dV) advance by 4 bytes per element;
    //   * M/L advance by float* arithmetic (B*Sq*Hq laid out batch-major
    //     within each reduction).
    // All launches share the stream, so the prep result and the
    // zero-init are visible before each bwd reads M/L / accumulates.
    constexpr std::size_t kHalf = 2;  // bytes per HALF element
    constexpr std::size_t kF32 = 4;   // bytes per FLOAT element
    for (std::int32_t b = 0; b < _B; ++b) {
        const std::size_t bb = static_cast<std::size_t>(b);

        char* q_b = static_cast<char*>(qPtr) + bb * static_cast<std::size_t>(_Sq) *
                                                   static_cast<std::size_t>(_strideQToken) * kHalf;
        char* k_b = static_cast<char*>(kPtr) + bb * static_cast<std::size_t>(_Skv) *
                                                   static_cast<std::size_t>(_strideKToken) * kHalf;
        char* v_b = static_cast<char*>(vPtr) + bb * static_cast<std::size_t>(_Skv) *
                                                   static_cast<std::size_t>(_strideVToken) * kHalf;
        char* do_b = static_cast<char*>(doPtr) + bb * static_cast<std::size_t>(_Sq) *
                                                     static_cast<std::size_t>(_strideDoToken) *
                                                     kHalf;

        float* M_b = M + bb * static_cast<std::size_t>(_Sq) * static_cast<std::size_t>(_Hq);
        float* L_b = L + bb * static_cast<std::size_t>(_Sq) * static_cast<std::size_t>(_Hq);

        char* dq_b = static_cast<char*>(dqPtr) + bb * static_cast<std::size_t>(_Sq) *
                                                     static_cast<std::size_t>(_strideDqToken) *
                                                     kF32;
        char* dk_b = static_cast<char*>(dkPtr) + bb * static_cast<std::size_t>(_Skv) *
                                                     static_cast<std::size_t>(_strideDkToken) *
                                                     kF32;
        char* dv_b = static_cast<char*>(dvPtr) + bb * static_cast<std::size_t>(_Skv) *
                                                     static_cast<std::size_t>(_strideDvToken) *
                                                     kF32;

        // ABI order: Q, K, V, dO, M, L, dQ, dK, dV, scale_log2,
        // scale_inv, seqlen_q, seqlen_k, then the eleven token/head
        // strides for Q/K/V/dO and the three gradient token strides.
        std::vector<ArgValue> values = {
            ArgValue::pointer(q_b),        ArgValue::pointer(k_b),
            ArgValue::pointer(v_b),        ArgValue::pointer(do_b),
            ArgValue::pointer(M_b),        ArgValue::pointer(L_b),
            ArgValue::pointer(dq_b),       ArgValue::pointer(dk_b),
            ArgValue::pointer(dv_b),       ArgValue::f32(_scaleLog2),
            ArgValue::f32(_scaleInv),      ArgValue::i32(_Sq),
            ArgValue::i32(_Skv),           ArgValue::i32(_strideQToken),
            ArgValue::i32(_strideQHead),   ArgValue::i32(_strideKToken),
            ArgValue::i32(_strideKHead),   ArgValue::i32(_strideVToken),
            ArgValue::i32(_strideVHead),   ArgValue::i32(_strideDoToken),
            ArgValue::i32(_strideDoHead),  ArgValue::i32(_strideDqToken),
            ArgValue::i32(_strideDkToken), ArgValue::i32(_strideDvToken),
        };
        std::vector<std::byte> packed = LaunchAbi::pack(_bwdModule->argSchema(), values);
        _bwdModule->launch(packed.data(), packed.size(), _bwdModule->grid(), _bwdModule->block(),
                           _bwdModule->ldsBytes(), stream);
    }
}

}  // namespace ck_dsl_provider
