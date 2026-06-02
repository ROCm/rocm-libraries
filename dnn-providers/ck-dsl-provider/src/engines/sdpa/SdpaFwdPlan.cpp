// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaFwdPlan.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../../adapters/sdpa/SdpaMarshalling.hpp"
#include "../../runtime/KernelArtifact.hpp"
#include "../../runtime/LaunchAbi.hpp"

namespace ck_dsl_provider {

namespace {

/// Linear scan over the device-buffer array. The array is typically
/// <10 entries so an O(n) lookup is the right shape. Throws with the
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
    oss << "SdpaFwdPlan::execute: no device buffer for " << role << " (uid=" << uid
        << "); searched " << numDeviceBuffers << " entries";
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE, oss.str());
}

/// Per-array alignment for the workspace sub-regions. The three i32 arrays
/// are laid out back-to-back, each starting on a 256-byte boundary so the
/// device sees naturally-aligned bases (256 comfortably exceeds the i32
/// natural alignment and matches a typical cache-line/coalescing grain).
constexpr std::size_t kWsAlign = 256u;

/// Round ``n`` up to the next multiple of ``kWsAlign``.
std::size_t roundUpWs(std::size_t n) {
    return ((n + kWsAlign - 1u) / kWsAlign) * kWsAlign;
}

}  // namespace

SdpaFwdPlan::SdpaFwdPlan(std::shared_ptr<HipModule> module, std::int64_t qUid, std::int64_t kUid,
                         std::int64_t vUid, std::int64_t oUid, float scaleLog2,
                         std::int32_t seqlenQ, std::int32_t seqlenK, std::int32_t strideQToken,
                         std::int32_t strideQHead, std::int32_t strideKToken,
                         std::int32_t strideKHead, std::int32_t strideVToken,
                         std::int32_t strideVHead, std::int32_t strideOToken,
                         std::int32_t strideOHead, std::int32_t batch, std::int32_t blockSize,
                         bool isPaged, bool isVarlen, bool useSinks, std::int64_t pageTableUid,
                         std::int64_t seqLenQUid, std::int64_t seqLenKvUid, std::int64_t sinkUid,
                         bool hasStats, std::int64_t statsUid)
    : _module(std::move(module)),
      _qUid(qUid),
      _kUid(kUid),
      _vUid(vUid),
      _oUid(oUid),
      _scaleLog2(scaleLog2),
      _seqlenQ(seqlenQ),
      _seqlenK(seqlenK),
      _strideQToken(strideQToken),
      _strideQHead(strideQHead),
      _strideKToken(strideKToken),
      _strideKHead(strideKHead),
      _strideVToken(strideVToken),
      _strideVHead(strideVHead),
      _strideOToken(strideOToken),
      _strideOHead(strideOHead),
      _batch(batch),
      _blockSize(blockSize),
      _isPaged(isPaged),
      _isVarlen(isVarlen),
      _useSinks(useSinks),
      _pageTableUid(pageTableUid),
      _seqLenQUid(seqLenQUid),
      _seqLenKvUid(seqLenKvUid),
      _sinkUid(sinkUid),
      _hasStats(hasStats),
      _statsUid(statsUid) {
    if (_module == nullptr) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "SdpaFwdPlan: refusing to construct with null HipModule");
    }

    // Validate the kernel's argument schema matches the unified
    // paged/varlen tiled-2D ABI: a fixed 18-slot layout
    // (Pointer x10, F32 x5, I32 x3). If a future kernel variant grows or
    // rearranges its parameter list, the plan must refuse to launch
    // rather than pack values into wrong-typed slots. The schema arrives
    // via KernelArtifact at JIT time, so this is fixed for the lifetime of
    // each cached HipModule. The unified kernel emits no LSE, so
    // ``hasStats`` must be false here.
    const auto& schema = _module->argSchema();
    auto rejectSlot = [&](const std::string& detail) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaFwdPlan: kernel argSchema does not match the expected "
            "(Pointer x10, F32 x5, I32 x3) layout for the unified paged/varlen "
            "FMHA-forward kernel signature; " +
                detail);
    };
    if (_hasStats) {
        rejectSlot("opt-in stats (LSE) are not supported on the unified paged path");
    }
    constexpr std::size_t kUnifiedSlots = 18u;
    if (schema.size() != kUnifiedSlots) {
        std::ostringstream oss;
        oss << "got " << schema.size() << " slots, expected " << kUnifiedSlots;
        rejectSlot(oss.str());
    }
    for (std::size_t i = 0; i < 10; ++i) {
        if (schema[i].kind != ArgSchema::Kind::Pointer) {
            std::ostringstream oss;
            oss << "slot " << i << " must be Pointer";
            rejectSlot(oss.str());
        }
    }
    for (std::size_t i = 10; i < 15; ++i) {
        if (schema[i].kind != ArgSchema::Kind::F32) {
            std::ostringstream oss;
            oss << "slot " << i << " must be F32";
            rejectSlot(oss.str());
        }
    }
    for (std::size_t i = 15; i < 18; ++i) {
        if (schema[i].kind != ArgSchema::Kind::I32) {
            std::ostringstream oss;
            oss << "slot " << i << " must be I32";
            rejectSlot(oss.str());
        }
    }

    // One-shot launch-shape log: useful operator diagnostic at plan
    // construction without polluting the per-call hot path.
    HIPDNN_PLUGIN_LOG_INFO("SdpaFwdPlan: built plan for kernel '"
                           << _module->kernelName() << "' grid=(" << _module->grid().x << ","
                           << _module->grid().y << "," << _module->grid().z << ") block=("
                           << _module->block().x << "," << _module->block().y << ","
                           << _module->block().z << ") seqlen_q=" << _seqlenQ << " seqlen_k="
                           << _seqlenK << " batch=" << _batch << " block_size=" << _blockSize);
}

std::size_t SdpaFwdPlan::getWorkspaceSize(const ::CkDslHandle& /*handle*/) const {
    // The unified kernel takes three host-marshalled i32 arrays the host
    // must stage in device memory: block_tables, cu_seqlens_q
    // (query_start_len), and seqused_k (seq_lens). They live in the
    // caller-provided workspace, each in a 256B-aligned sub-region so the
    // device sees naturally-aligned bases.
    //
    // Element counts (dense-degenerate layout, mirroring
    // marshalDenseDegenerate):
    //   block_tables : num_seqs * blockTableStride(Skv, blockSize)
    //                  = batch * ceil(Skv / blockSize)
    //   cu_seqlens_q : num_seqs + 1  = batch + 1
    //   seqused_k    : num_seqs      = batch
    //
    // Defensive: a non-positive block size cannot form a valid stride, so
    // report zero (execute() will reject such a plan before any upload).
    if (_blockSize <= 0) {
        return 0u;
    }
    const std::int32_t stride = blockTableStride(_seqlenK, _blockSize);
    const std::size_t blockTablesBytes =
        static_cast<std::size_t>(_batch) * static_cast<std::size_t>(stride) * sizeof(std::int32_t);
    const std::size_t cuSeqlensQBytes =
        (static_cast<std::size_t>(_batch) + 1u) * sizeof(std::int32_t);
    const std::size_t sequsedKBytes = static_cast<std::size_t>(_batch) * sizeof(std::int32_t);

    return roundUpWs(blockTablesBytes) + roundUpWs(cuSeqlensQBytes) + roundUpWs(sequsedKBytes);
}

void SdpaFwdPlan::execute(const ::CkDslHandle& handle,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          std::uint32_t numDeviceBuffers, void* workspace) const {
    if (deviceBuffers == nullptr && numDeviceBuffers > 0) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "SdpaFwdPlan::execute: deviceBuffers is null but numDeviceBuffers > 0");
    }

    // Resolve the output/query/key/value buffers (+ optional sink) so a
    // graph-vs-buffer UID mismatch surfaces here with concrete context.
    const auto& oBuf = findDeviceBuffer(_oUid, deviceBuffers, numDeviceBuffers, "O");
    const auto& qBuf = findDeviceBuffer(_qUid, deviceBuffers, numDeviceBuffers, "Q");
    const auto& kBuf = findDeviceBuffer(_kUid, deviceBuffers, numDeviceBuffers, "K");
    const auto& vBuf = findDeviceBuffer(_vUid, deviceBuffers, numDeviceBuffers, "V");
    void* sinkPtr = nullptr;
    if (_useSinks) {
        sinkPtr = findDeviceBuffer(_sinkUid, deviceBuffers, numDeviceBuffers, "sink").ptr;
    }

    if (_blockSize <= 0) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaFwdPlan::execute: block_size must be positive");
    }

    const hipStream_t stream = handle.getStream();

    // Common marshalling inputs (shape-driven). The varlen builders ignore
    // ``seqlen_q`` / ``seqlen_k`` (they take explicit per-sequence
    // lengths) but every builder reads ``num_seqs`` / ``block_size`` /
    // ``max_seqlen_k``.
    SdpaMarshalInputs in;
    in.num_seqs = _batch;
    in.block_size = _blockSize;
    in.max_seqlen_k = _seqlenK;
    in.seqlen_q = _seqlenQ;
    in.seqlen_k = _seqlenK;

    // Varlen paths need the graph's per-sequence Q/KV lengths, which live
    // in device memory (the seq_len_q / seq_len_kv tensors). Pull them down
    // with a small synchronous D2H copy before marshalling. ``_batch`` is
    // num_seqs for the varlen layout. The copies are SYNC (plain
    // hipMemcpy) so the host vectors are valid for the marshalling below.
    std::vector<std::int32_t> qLens;
    std::vector<std::int32_t> kLens;
    if (_isVarlen) {
        qLens.resize(static_cast<std::size_t>(_batch));
        kLens.resize(static_cast<std::size_t>(_batch));
        const auto& seqQBuf =
            findDeviceBuffer(_seqLenQUid, deviceBuffers, numDeviceBuffers, "seq_len_q");
        const auto& seqKvBuf =
            findDeviceBuffer(_seqLenKvUid, deviceBuffers, numDeviceBuffers, "seq_len_kv");
        const std::size_t lenBytes = static_cast<std::size_t>(_batch) * sizeof(std::int32_t);
        auto downloadLens = [&](void* src, std::vector<std::int32_t>& dst, const char* role) {
            const hipError_t err = hipMemcpy(dst.data(), src, lenBytes, hipMemcpyDeviceToHost);
            if (err != hipSuccess) {
                std::ostringstream oss;
                oss << "SdpaFwdPlan::execute: hipMemcpy (D2H) failed for " << role << " ("
                    << hipGetErrorString(err) << ")";
                throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                               oss.str());
            }
        };
        downloadLens(seqQBuf.ptr, qLens, "seq_len_q");
        downloadLens(seqKvBuf.ptr, kLens, "seq_len_kv");
    }

    // Marshal the host integer arrays for the selected path. cu_seqlens_q
    // and seqused_k are always bound from the workspace; block_tables is
    // bound from the workspace only on the non-paged paths (the paged
    // paths bind the graph's physical block table directly). On the paged
    // paths the marshalled ``block_tables`` is intentionally ignored.
    //
    // Real-paged (non-varlen) reuses ``marshalDenseDegenerate`` for
    // ``cu_seqlens_q`` / ``seqused_k``, i.e. UNIFORM full-length sequences
    // (Sq / Skv repeated). This is correct for the perf probe and any
    // full-length paged use; a real-paged path with per-sequence KV
    // lengths would instead source ``seqused_k`` from the seq-len inputs
    // (e.g. via ``marshalRealPaged``).
    const SdpaMarshalledArrays arrays =
        _isVarlen ? marshalVarlen(in, qLens, kLens) : marshalDenseDegenerate(in);

    // The caller owns the device scratch. We never hipMalloc internally:
    // a null workspace is a programming error (call getWorkspaceSize and
    // allocate first).
    if (workspace == nullptr) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "SdpaFwdPlan::execute: workspace is null; allocate getWorkspaceSize() "
            "bytes of device memory and pass it in (the plan does not allocate "
            "device memory internally)");
    }

    // Compute the same 256B-aligned sub-offsets getWorkspaceSize sized.
    // Layout order: block_tables, then cu_seqlens_q, then seqused_k. The
    // block_tables sub-region is left untouched on the paged paths (we
    // bind the graph buffer instead); keeping its offset in the layout is
    // harmless and keeps the offset math identical to getWorkspaceSize.
    const std::size_t blockTablesBytes = arrays.block_tables.size() * sizeof(std::int32_t);
    const std::size_t cuSeqlensQBytes = arrays.cu_seqlens_q.size() * sizeof(std::int32_t);
    const std::size_t sequsedKBytes = arrays.seqused_k.size() * sizeof(std::int32_t);

    auto* wsBase = static_cast<std::byte*>(workspace);
    const std::size_t blockTablesOff = 0u;
    const std::size_t cuSeqlensQOff = blockTablesOff + roundUpWs(blockTablesBytes);
    const std::size_t sequsedKOff = cuSeqlensQOff + roundUpWs(cuSeqlensQBytes);

    void* dCuSeqlensQ = wsBase + cuSeqlensQOff;
    void* dSequsedK = wsBase + sequsedKOff;

    // The block_tables slot binds the graph's Page_table_K device buffer
    // directly on the paged paths; otherwise it binds the workspace
    // sub-region the marshalled table is uploaded into.
    void* dBlockTables = nullptr;
    if (_isPaged) {
        dBlockTables =
            findDeviceBuffer(_pageTableUid, deviceBuffers, numDeviceBuffers, "page_table_k").ptr;
    } else {
        dBlockTables = wsBase + blockTablesOff;
    }

    // The marshalled ``arrays`` host vectors are consumed by the H2D
    // copies below before this call returns: hipMemcpyAsync from PAGEABLE
    // host memory stages synchronously (it cannot retire the copy after
    // the source is freed), so the local vectors outliving only this
    // function body is safe. If a future optimisation pins these buffers,
    // it must keep them alive until a stream sync.
    auto upload = [&](void* dst, const std::vector<std::int32_t>& src, std::size_t bytes,
                      const char* role) {
        // hipMemcpyAsync takes a const src + an explicit kind, so no
        // const_cast / reinterpret_cast of the device pointer is needed.
        const hipError_t err =
            hipMemcpyAsync(dst, src.data(), bytes, hipMemcpyHostToDevice, stream);
        if (err != hipSuccess) {
            std::ostringstream oss;
            oss << "SdpaFwdPlan::execute: hipMemcpyAsync (H2D) failed for " << role << " ("
                << hipGetErrorString(err) << ")";
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                           oss.str());
        }
    };
    // Only upload the workspace-bound arrays. The block_tables upload is
    // skipped on the paged paths (the slot points at the graph buffer).
    if (!_isPaged) {
        upload(dBlockTables, arrays.block_tables, blockTablesBytes, "block_tables");
    }
    upload(dCuSeqlensQ, arrays.cu_seqlens_q, cuSeqlensQBytes, "cu_seqlens_q");
    upload(dSequsedK, arrays.seqused_k, sequsedKBytes, "seqused_k");

    // The kernel computes ``score * scale * rcp_ln2`` then ``exp2`` (see
    // attention_unified.py), so ``scale`` here is the RAW softmax scale,
    // NOT the log2-space value. The plan carries ``_scaleLog2 =
    // attn_scale * log2(e)`` (folded by the adapter); recover the raw
    // scale via ``raw = _scaleLog2 / log2(e) = _scaleLog2 * ln2`` (since
    // ``1 / log2(e) == ln2``). The constant is spelled out locally to
    // avoid the POSIX-only ``M_LN2`` macro (matching SdpaAdapter.cpp's
    // local ``kLog2E``).
    constexpr float kLn2 = 0.69314718055994530942f;
    const float rawScale = _scaleLog2 * kLn2;

    // Non-fp8 path: the dequant scales are identity and out_scale is
    // identity; softcap was declined (0.0). qq_bias_stride_0 is 0 (no
    // qq_bias). alibi / qq_bias pointers are null (features declined).
    constexpr float kIdentityScale = 1.0f;
    constexpr float kNoSoftcap = 0.0f;
    constexpr std::int32_t kQqBiasStride0 = 0;

    // Build the 18 args in schema order:
    //   pointers: output, query, key_cache, value_cache, sink|null,
    //             block_tables@ws, seq_lens@ws (=seqused_k),
    //             alibi=null, qq_bias=null, query_start_len@ws (=cu_seqlens_q)
    //   f32:      scale (raw), k_scale, v_scale, out_scale, softcap
    //   i32:      num_seqs (=batch), block_table_stride, qq_bias_stride_0
    std::vector<ArgValue> values = {
        ArgValue::pointer(oBuf.ptr),
        ArgValue::pointer(qBuf.ptr),
        ArgValue::pointer(kBuf.ptr),
        ArgValue::pointer(vBuf.ptr),
        ArgValue::pointer(sinkPtr),
        ArgValue::pointer(dBlockTables),
        ArgValue::pointer(dSequsedK),
        ArgValue::pointer(nullptr),  // alibi_slopes (declined)
        ArgValue::pointer(nullptr),  // qq_bias (declined)
        ArgValue::pointer(dCuSeqlensQ),
        ArgValue::f32(rawScale),
        ArgValue::f32(kIdentityScale),  // k_scale
        ArgValue::f32(kIdentityScale),  // v_scale
        ArgValue::f32(kIdentityScale),  // out_scale
        ArgValue::f32(kNoSoftcap),
        ArgValue::i32(_batch),  // num_seqs
        ArgValue::i32(arrays.block_table_stride),
        ArgValue::i32(kQqBiasStride0),
    };

    // Defensive: the count must match the schema width validated in the
    // ctor. A drift here would otherwise surface as a confusing pack
    // failure deeper in LaunchAbi.
    const auto& schema = _module->argSchema();
    if (values.size() != schema.size()) {
        std::ostringstream oss;
        oss << "SdpaFwdPlan::execute: built " << values.size() << " arg values but the schema has "
            << schema.size() << " slots";
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       oss.str());
    }

    std::vector<std::byte> packed = LaunchAbi::pack(schema, values);

    _module->launch(packed.data(), packed.size(), _module->grid(), _module->block(),
                    _module->ldsBytes(), stream);
}

}  // namespace ck_dsl_provider
