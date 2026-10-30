// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DeviceContext.hpp"

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Utils.hpp>

#ifdef TENSILELITE_ENABLE_SDMA
#include "SdmaQueue.hpp"
#endif

#include <stdexcept>

namespace TensileLite
{
    namespace Client
    {
        namespace
        {
            // Counter regions at the head of the Synchronizer, each on a 64-byte line:
            //
            //   +0    cursors   MAX_RANKS * 2 * u64   SDMA cursor pair per queue
            //   +128  counter2  MAX_RANKS * u32       counter2[dst_rank]
            //   +192  counter3  1 * u32
            //   +256  counter1  W * tokenTiles * u32  counter1[dst_rank*tokenTiles + j]
            //
            // Twinned with FUSED_A2A_CURSOR_BYTES / FUSED_A2A_COUNTER*_OFFSET in
            // Tensile/Components/Signature.py.
            constexpr size_t kCursorsPerQueue = 2;
            constexpr size_t kCursorBytes     = fusedA2AAlignLine(
                (size_t)FUSED_A2A_MAX_RANKS * kCursorsPerQueue * sizeof(uint64_t));
            constexpr size_t kCounter2Offset = kCursorBytes;
            constexpr size_t kCounter3Offset = fusedA2AAlignLine(
                kCounter2Offset + (size_t)FUSED_A2A_MAX_RANKS * sizeof(uint32_t));
            constexpr size_t kCounter1Offset
                = fusedA2AAlignLine(kCounter3Offset + sizeof(uint32_t));

            // FusedA2A: A2AGemm regions of the same block, each on a 64-byte line:
            //
            //   +0    cursors   as above
            //   +128  flag      tokenBlocks * (W-1) * S * u32  flag[b][queue]
            //   ...   tally     (tokenBlocks+1) * u32          block tallies, grid tally
            //   ...   counter   batches * u32                  counter[iB]
            //
            // Twinned with FUSED_A2A_A2AGEMM_FLAG_OFFSET in Tensile/Components/Signature.py
            // and a2aFlagBlockBytes / a2aTallyBytes / a2aBatchSpan in KernelWriterAssembly.py.
            constexpr size_t kA2AGemmFlagOffset = kCursorBytes;

            size_t a2aGemmCounterBytes(ContractionProblemGemm const& gemm,
                                       ContractionSolution const&    solution,
                                       Hardware const&               hardware,
                                       size_t                        world)
            {
                auto const* amd = dynamic_cast<AMDGPU const*>(&hardware);
                if(amd == nullptr || amd->computeUnitCount == 0)
                    throw std::runtime_error("[fused-a2a] no CU count to size the counter.");
                if(gemm.transposeC01())
                    throw std::runtime_error("[fused-a2a] transposeC01 is not supported.");
                const size_t mt0           = solution.sizeMapping.macroTile.x;
                const size_t mt1           = solution.sizeMapping.macroTile.y;
                const size_t featureBlocks = (gemm.freeSizeA(0) + mt0 - 1) / mt0;
                const size_t tokenBlocks   = (gemm.freeSizeB(0) + mt1 - 1) / mt1;
                const size_t batches
                    = (tokenBlocks - 1) * featureBlocks / amd->computeUnitCount + 1;
                const size_t flagBytes
                    = fusedA2AAlignLine(tokenBlocks * (world - 1)
                                        * solution.sizeMapping.a2aSubRounds * sizeof(uint32_t));
                const size_t tallyBytes = fusedA2AAlignLine((tokenBlocks + 1) * sizeof(uint32_t));
                return kA2AGemmFlagOffset + flagBytes + tallyBytes + batches * sizeof(uint32_t);
            }

            constexpr int kA2AGemmPoisonByte = 0xC7;
        }

        DeviceContext::DeviceContext(int                     deviceId,
                                     bool                    useDefaultStream,
                                     size_t                  commFlagsBytes,
                                     std::vector<int> const& sdmaPeers,
                                     int                     sdmaRotate)
            : deviceId(deviceId)
            , adapter(std::make_shared<hip::SolutionAdapter>())
            , commFlagsBytes(commFlagsBytes)
        {
            try
            {
                if(!useDefaultStream)
                {
                    ScopedDevice guard(deviceId);
                    HIP_CHECK_EXC(hipStreamCreate(&stream));
                }
                if(commFlagsBytes != 0)
                {
                    ScopedDevice guard(deviceId);
                    // Fine-grained: written by remote peers, must bypass stale L2.
                    HIP_CHECK_EXC(hipExtMallocWithFlags(
                        &commFlags, commFlagsBytes, hipDeviceMallocFinegrained));
                    HIP_CHECK_EXC(hipMemset(commFlags, 0, commFlagsBytes));
                }
                if(!sdmaPeers.empty())
                {
#ifdef TENSILELITE_ENABLE_SDMA
                    ScopedDevice guard(deviceId);
                    for(int peer : sdmaPeers)
                    {
                        if(peer == deviceId)
                            continue;
                        hipError_t pe = hipDeviceEnablePeerAccess(peer, 0);
                        if(pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled)
                            HIP_CHECK_EXC(pe);
                    }
                    const uint32_t node = sdmaNodeIdForDevice(deviceId);
                    for(int peer : sdmaPeers)
                    {
                        const uint32_t engine = sdmaSelectEngine(node, sdmaNodeIdForDevice(peer));
                        sdmaQueues.push_back(std::make_shared<SdmaQueue>(
                            node,
                            peer == deviceId ? engine : sdmaRotateEngine(engine, sdmaRotate)));
                    }
#else
                    throw std::runtime_error(
                        "DeviceContext: SDMA queues requested, but this client was built without "
                        "TENSILELITE_ENABLE_SDMA.");
#endif
                }
            }
            catch(...)
            {
                release();
                throw;
            }
        }

        DeviceContext::~DeviceContext()
        {
            release();
        }

        void DeviceContext::release()
        {
            if(commFlags != nullptr)
                static_cast<void>(hipFree(commFlags));
            if(stream != nullptr)
                static_cast<void>(hipStreamDestroy(stream));
        }

        void resetCommBuffers(ContractionProblem const&                          problem,
                              std::vector<std::shared_ptr<DeviceContext>> const& devices)
        {
            auto const* gemm = dynamic_cast<ContractionProblemGemm const*>(&problem);
            if(gemm == nullptr || gemm->fusedA2A() == FusedA2A::Disabled)
                return;
            const bool   gemmA2A = gemm->fusedA2A() == FusedA2A::GemmA2A;
            auto const&  tensors = gemm->tensors();
            const size_t bufBytes
                = tensors[ContractionProblemGemm::TENSOR::A2A_BUF].totalAllocatedBytes();
            const size_t synchronizerBytes
                = tensors[ContractionProblemGemm::TENSOR::Synchronizer].totalAllocatedBytes();
            forEachDevice(devices, [&](DeviceContext& ctx) {
                HIP_CHECK_EXC(hipMemset(ctx.commFlags, 0, ctx.commFlagsBytes));
                for(auto const& slot : ctx.inputArr)
                {
                    auto const& inputs = dynamic_cast<ContractionInputs const&>(*slot);
                    HIP_CHECK_EXC(
                        hipMemset(inputs.a2aBuf, gemmA2A ? 0 : kA2AGemmPoisonByte, bufBytes));
                    HIP_CHECK_EXC(hipMemset(inputs.Synchronizer, 0, synchronizerBytes));
                }
            });
        }

        ProblemInputs const* rankInputs(ContractionInputs&                                 scratch,
                                        ContractionProblem const&                          problem,
                                        ContractionSolution const&                         solution,
                                        Hardware const&                                    hardware,
                                        std::vector<std::shared_ptr<DeviceContext>> const& devices,
                                        size_t                                             rank,
                                        size_t                                             launch)
        {
            const size_t slot = launch % devices[rank]->inputArr.size();
            auto const*  gemm = dynamic_cast<ContractionProblemGemm const*>(&problem);
            if(gemm == nullptr || gemm->fusedA2A() == FusedA2A::Disabled)
                return devices[rank]->inputArr[slot].get();

            const FusedA2A mode       = gemm->fusedA2A();
            const size_t   world      = devices.size();
            const size_t   channel    = launch % kCommChannels;
            const size_t   macroTileN = solution.sizeMapping.macroTile.y;
            const size_t   tokenTiles = (gemm->freeSizeB(0) + macroTileN - 1) / macroTileN;
            const size_t   counterBytes
                = mode == FusedA2A::GemmA2A
                      ? kCounter1Offset + world * tokenTiles * sizeof(uint32_t)
                      : a2aGemmCounterBytes(*gemm, solution, hardware, world);
            const size_t synchronizerBytes
                = gemm->tensors()[ContractionProblemGemm::TENSOR::Synchronizer]
                      .totalAllocatedBytes();
            if(counterBytes > synchronizerBytes)
                throw std::runtime_error(concatenate("[fused-a2a] the counter needs ",
                                                     counterBytes,
                                                     " bytes; the Synchronizer holds ",
                                                     synchronizerBytes,
                                                     "."));
            if(solution.requiredWorkspaceSize(*gemm, hardware) != 0)
                throw std::runtime_error(
                    "[fused-a2a] solutions that need a workspace are not supported.");

            scratch = dynamic_cast<ContractionInputs const&>(*devices[rank]->inputArr[slot]);
            scratch.fusedA2APeers.clear();
#ifdef TENSILELITE_ENABLE_SDMA
            for(size_t j = 0; j < world; j++)
            {
                if(slot >= devices[j]->inputArr.size())
                    throw std::runtime_error(concatenate("[fused-a2a] device ",
                                                         j,
                                                         " has ",
                                                         devices[j]->inputArr.size(),
                                                         " rotating buffers; slot ",
                                                         slot,
                                                         " was requested."));
                auto const& peer
                    = dynamic_cast<ContractionInputs const&>(*devices[j]->inputArr[slot]);
                auto const&             queue = *devices[rank]->sdmaQueues[j];
                const HsaQueueResource& q     = queue.queueResource();
                scratch.fusedA2APeers.push_back(
                    {(char*)devices[j]->commFlags + channel * FUSED_A2A_FLAG_BLOCK_BYTES,
                     mode == FusedA2A::GemmA2A || j == rank ? peer.a2aBuf
                                                            : const_cast<void*>(peer.b),
                     queue.ringBase(),
                     (void*)q.Queue_read_ptr_aql,
                     (void*)q.Queue_write_ptr_aql,
                     (void*)q.Queue_DoorBell_aql});
            }
#endif
            scratch.fusedA2ACounter = scratch.Synchronizer;
            scratch.fusedA2AMyRank  = (uint32_t)rank;
            scratch.fusedA2ADrain   = FUSED_A2A_DRAIN_RECV | FUSED_A2A_DRAIN_SEND;
            return &scratch;
        }
    } // namespace Client
} // namespace TensileLite
