// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "FusedA2ARunner.hpp"

#include "DataInitialization.hpp"

#include <Tensile/FusedA2AKernArg.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipUtils.hpp>

// SdmaQueue.hpp is header-only and pulls in hsakmt, which is only on the
// include path (and linked) under this option.
#ifdef TENSILELITE_ENABLE_SDMA
#include "SdmaQueue.hpp"
#endif

#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace TensileLite
{
    namespace Client
    {
        namespace
        {
            // log2(16-byte SDMA packet element / 2-byte bf16).
            constexpr size_t kElemShift    = 3;
            constexpr size_t kElemMultiple = size_t(1) << kElemShift;

            // Counter regions at the head of the Synchronizer, each on a 64-byte line:
            //
            //   +0    cursors   MAX_RANKS * 2 * u64   SDMA cursor pair per queue
            //   +128  counter2  MAX_RANKS * u32       counter2[dst_rank]
            //   +192  counter3  1 * u32
            //   +256  counter1  W * tokenTiles * u32  counter1[dst_rank*tokenTiles + j]
            //
            // Twinned with FUSED_A2A_COUNTER*_OFFSET in Tensile/Components/Signature.py.
            constexpr size_t kCursorsPerQueue = 2;
            constexpr size_t kCounter2Offset  = fusedA2AAlignLine(
                (size_t)FUSED_A2A_MAX_RANKS * kCursorsPerQueue * sizeof(uint64_t));
            constexpr size_t kCounter3Offset = fusedA2AAlignLine(
                kCounter2Offset + (size_t)FUSED_A2A_MAX_RANKS * sizeof(uint32_t));
            constexpr size_t kCounter1Offset
                = fusedA2AAlignLine(kCounter3Offset + sizeof(uint32_t));
        }

        int FusedA2ARunner::worldSize(po::variables_map const& args)
        {
            const int toValidate = args["num-elements-to-validate"].as<int>();
            if(toValidate > 0)
                throw std::runtime_error(concatenate(
                    "[fused-a2a] --num-elements-to-validate=",
                    toValidate,
                    " is not supported; this path compares every element or none. Use -1 for "
                    "all, 0 for none."));
            if(toValidate != 0 && args["print-max"].as<int>() <= 0)
                throw std::runtime_error(
                    "[fused-a2a] validation needs --print-max > 0; with print-max <= 0 a "
                    "mismatch in D is reported as PASSED.");
            for(auto name : {"init-a", "init-b"})
                if(args[name].as<InitMode>() != InitMode::Random)
                    throw std::runtime_error(concatenate("[fused-a2a] --",
                                                         name,
                                                         " must be ",
                                                         InitMode::Random,
                                                         "; got ",
                                                         args[name].as<InitMode>()));
            if(args["device-idx"].as<int>() != 0)
                throw std::runtime_error(
                    "[fused-a2a] only --device-idx=0 is supported; rank r runs on device r. Use "
                    "HIP_VISIBLE_DEVICES to pick the set.");
            auto const& flush = args["icache-flush-args"].as<std::vector<bool>>();
            if(std::find(flush.begin(), flush.end(), true) != flush.end())
                throw std::runtime_error(
                    "[fused-a2a] --icache-flush-args is not supported; leave it at false.");
            if(args["fused-a2a-drain-recv"].as<int>() == 0)
                throw std::runtime_error(
                    "[fused-a2a] --fused-a2a-drain-recv=0 is not supported: the kernel clears its "
                    "flags only in the receive drain. Leave it at 1.");
            if(args["fused-a2a-drain-send"].as<int>() == 0)
                throw std::runtime_error(
                    "[fused-a2a] --fused-a2a-drain-send=0 is not supported: the kernel clears its "
                    "outbound counter only in the send drain. Leave it at 1.");
            if(args["use-user-args"].as<bool>())
                throw std::runtime_error("[fused-a2a] --use-user-args is not supported.");
#ifndef TENSILELITE_ENABLE_SDMA
            throw std::runtime_error(
                "[fused-a2a] this client was built without TENSILELITE_ENABLE_SDMA, so no SDMA "
                "rings exist, but the fused epilogue unconditionally submits SDMA packets and "
                "would dereference a null queue handle. Reconfigure with "
                "-DTENSILELITE_ENABLE_SDMA=ON.");
#endif

            int deviceCount = 0;
            HIP_CHECK_EXC(hipGetDeviceCount(&deviceCount));
            const int worldArg = args["fused-a2a-world"].as<int>();
            const int world    = worldArg > 0 ? worldArg : deviceCount;
            if(!fusedA2AWorldSizeValid(world))
                throw std::runtime_error(
                    concatenate("[fused-a2a] world size W=",
                                world,
                                " is out of range; the kernarg segment reserves exactly ",
                                FUSED_A2A_MAX_RANKS,
                                " peer groups. Require 1 <= W <= ",
                                FUSED_A2A_MAX_RANKS,
                                "."));
            if(deviceCount < world)
                throw std::runtime_error(
                    concatenate("[fused-a2a] need ", world, " devices, found ", deviceCount));
            for(int s = 0; s < world; s++)
                for(int t = 0; t < world; t++)
                {
                    if(t == s)
                        continue;
                    int canAccess = 0;
                    HIP_CHECK_EXC(hipDeviceCanAccessPeer(&canAccess, s, t));
                    if(!canAccess)
                        throw std::runtime_error(
                            concatenate("[fused-a2a] device ", s, " cannot P2P device ", t));
                }
            return world;
        }

        void FusedA2ARunner::configureProblem(po::variables_map const& args,
                                              ContractionProblemGemm&  problem,
                                              int                      world,
                                              int                      runIdx)
        {
            // The first `AM` FEATURE columns go all-to-all; [AM, M) stay local.
            auto const& amArgs = args["fused-a2a-am"].as<std::vector<int>>();
            if(amArgs.size() > 1 && (size_t)runIdx >= amArgs.size())
                throw std::runtime_error(concatenate(
                    "[fused-a2a] --fused-a2a-am has ",
                    amArgs.size(),
                    " values, fewer than the problems selected by "
                    "--problem-start-idx/--num-problems; pass one value per selected problem, or "
                    "one value for all."));
            const uint32_t M = (uint32_t)problem.freeSizeA(0);
            const uint32_t am
                = amArgs.empty() ? M : (uint32_t)amArgs[amArgs.size() == 1 ? 0 : runIdx];

            // freeIndices()[j].d is the D dim for free index j: 0 = A's M(feature),
            // 1 = B's N(token).
            const auto&  dDesc    = problem.d();
            const size_t dMStride = dDesc.strides()[problem.freeIndices()[0].d];
            const size_t dNStride = dDesc.strides()[problem.freeIndices()[1].d];
            if(dMStride != 1)
                throw std::runtime_error(concatenate(
                    "[fused-a2a] D's feature axis is not contiguous: D mStride=",
                    dMStride,
                    " must be 1. The packet copies n_shard 16-byte elements as one contiguous "
                    "run, so a strided feature axis would ship unrelated data to every peer."));
            // dNStride is the packet's src_pitch (StrideD1J), a 19-bit field.
            if(dNStride % kElemMultiple != 0)
                throw std::runtime_error(
                    concatenate("[fused-a2a] ldd(D nStride)=",
                                dNStride,
                                " must be a multiple of ",
                                kElemMultiple,
                                "; the emitter's >>",
                                kElemShift,
                                " would truncate the pitch and skew every token row."));
            if((dNStride >> kElemShift) >= (1u << 19))
                throw std::runtime_error(concatenate(
                    "[fused-a2a] ldd(D nStride)=",
                    dNStride,
                    " overflows the SDMA packet's 19-bit src_pitch field (ldd must be < ",
                    (size_t(1u << 19) << kElemShift),
                    "). Reduce M or the D padding."));

            problem.setFusedA2AExtent(am);
            problem.setFusedA2AWorld((uint32_t)world);
        }

        FusedA2ARunner::FusedA2ARunner(Hardware const&                                    hardware,
                                       ContractionProblemGemm const&                      problem,
                                       ContractionSolution const&                         solution,
                                       std::vector<std::shared_ptr<DeviceContext>> const& devices)
            : m_hardware(hardware)
            , m_problem(problem)
            , m_solution(solution)
            , m_devices(devices)
        {
            const int    W = (int)devices.size();
            const size_t R = devices[0]->inputArr.size();

            if(!solution.problemType.fusedGemmA2A)
                throw std::runtime_error(
                    "[fused-a2a] solution was not built with FusedGemmA2A; it has no fused "
                    "epilogue to drive.");

            {
                auto const&                        pt            = solution.problemType;
                const std::pair<bool, char const*> unsupported[] = {
                    {pt.useBias != 0, "bias"},
                    {pt.useE, "E"},
                    {pt.useGateResidual, "gate residual"},
                    {!pt.useScaleAB.empty(), "scaleA/scaleB"},
                    {pt.useScaleCD, "scaleC/scaleD"},
                    {pt.useScaleAlphaVec != 0, "scaleAlphaVec"},
                    {pt.outputAmaxD, "amaxD"},
                    {pt.sparse != 0, "sparse metadata"},
                    {pt.groupedGemm, "grouped GEMM"},
                    {pt.activationType != ActivationType::None, "activation"},
                    {solution.requiredWorkspaceSize(problem, hardware) != 0, "workspace"},
                };
                std::string missing;
                for(auto const& [needed, name] : unsupported)
                    if(needed)
                        missing += (missing.empty() ? "" : ", ") + std::string(name);
                if(!missing.empty())
                    throw std::runtime_error(
                        "[fused-a2a] solution needs inputs this client does not set: " + missing);
            }

            // The kernel epilogue derives dst_rank and the counter index from MT0/MT1.
            const uint32_t macroTileM = (uint32_t)solution.sizeMapping.macroTile.x;
            const uint32_t macroTileN = (uint32_t)solution.sizeMapping.macroTile.y;
            if(macroTileM == 0 || macroTileN == 0)
                throw std::runtime_error(
                    concatenate("[fused-a2a] solution macro-tile is zero (MT0=",
                                macroTileM,
                                " MT1=",
                                macroTileN,
                                ")"));

            // M/N-swap (col-major first-class): A=w[feature,K], B=x[token,K].
            const uint32_t M          = (uint32_t)problem.freeSizeA(0);
            const uint32_t N          = (uint32_t)problem.freeSizeB(0);
            const uint32_t K          = (uint32_t)problem.boundSize(0);
            const uint32_t AM         = (uint32_t)problem.fusedA2AExtent();
            const uint32_t nShard     = AM / (uint32_t)W;
            const uint32_t tokenTiles = (N + macroTileN - 1) / macroTileN;

            if(AM % (uint32_t)W != 0 || (nShard % macroTileM) != 0 || (M % macroTileM) != 0
               || (AM % macroTileM) != 0 || AM > M)
                throw std::runtime_error(concatenate(
                    "[fused-a2a] problem shape violates fused-A2A constraints: M(feature)=",
                    M,
                    " N(token)=",
                    N,
                    " AM=",
                    AM,
                    " W=",
                    W,
                    " MacroTile0(feature)=",
                    macroTileM,
                    ". Require AM % W == 0, (AM/W) % ",
                    macroTileM,
                    " == 0, M % ",
                    macroTileM,
                    " == 0, AM % ",
                    macroTileM,
                    " == 0, AM <= M; otherwise the DRAIN barrier deadlocks."));

            if(!(*solution.problemPredicate)(problem))
            {
                std::ostringstream msg;
                msg << "[fused-a2a] solution predicate does not match the problem:\n";
                solution.problemPredicate->debugEval(problem, msg);
                throw std::runtime_error(msg.str());
            }

            // SDMA COPY_SUBWIN rect_x/rect_y are 14-bit; rect_x = n_shard scaled into
            // 16-byte packet elements, rect_y <= MT1. `>=` is one tighter than the
            // hardware: the extents are minus-one encoded.
            if(nShard % kElemMultiple != 0)
                throw std::runtime_error(concatenate("[fused-a2a] n_shard=AM/W=",
                                                     nShard,
                                                     " must be a multiple of ",
                                                     kElemMultiple,
                                                     "; the emitter's >>",
                                                     kElemShift,
                                                     " would truncate it and copy a short band."));
            const size_t maxRectX = (size_t)nShard >> kElemShift;
            const size_t maxRectY = (size_t)macroTileN;
            if(maxRectX >= (1u << 14) || maxRectY >= (1u << 14))
                throw std::runtime_error(concatenate(
                    "[fused-a2a] geometry overflows the SDMA packet's 14-bit rect fields: rect_x="
                    "n_shard>>",
                    kElemShift,
                    "=",
                    maxRectX,
                    " rect_y=MT1=",
                    maxRectY,
                    "; each must be < ",
                    (1u << 14),
                    ". Reduce AM or raise W (the bound is AM < ",
                    (size_t(1u << 14) << kElemShift),
                    "*W)."));

            // recv is feature-contiguous [W, token, feature_shard].
            m_recvBytes = (size_t)W * N * nShard * sizeof(uint16_t);

            const size_t counterBytes = kCounter1Offset + (size_t)W * tokenTiles * sizeof(uint32_t);
            const size_t synchronizerBytes
                = problem.tensors()[ContractionProblemGemm::TENSOR::Synchronizer]
                      .totalAllocatedBytes();
            if(counterBytes > synchronizerBytes)
                throw std::runtime_error(concatenate("[fused-a2a] the counter needs ",
                                                     counterBytes,
                                                     " bytes; the Synchronizer holds ",
                                                     synchronizerBytes,
                                                     "."));

            std::cout << "[fused-a2a] W=" << W << " nFeature(M)=" << M << " nToken(N)=" << N
                      << " K=" << K << " AM=" << AM << " nShard=" << nShard
                      << " tokenTiles=" << tokenTiles << "\n";

            m_recvBase.assign(W, nullptr);
            m_recv.assign(R, std::vector<void*>(W, nullptr));
            try
            {
                for(int d = 0; d < W; d++)
                {
                    if(devices[d]->inputArr.size() != R)
                        throw std::runtime_error(concatenate("[fused-a2a] device ",
                                                             d,
                                                             " has ",
                                                             devices[d]->inputArr.size(),
                                                             " rotating buffers, device 0 has ",
                                                             R));
                    ScopedDevice device(devices[d]->deviceId);
                    HIP_CHECK_EXC(hipMemset(devices[d]->commFlags, 0, devices[d]->commFlagsBytes));
                    for(auto const& slot : devices[d]->inputArr)
                        HIP_CHECK_EXC(
                            hipMemset(dynamic_cast<ContractionInputs const&>(*slot).Synchronizer,
                                      0,
                                      synchronizerBytes));
                    HIP_CHECK_EXC(hipExtMallocWithFlags(
                        &m_recvBase[d], R * m_recvBytes, hipDeviceMallocFinegrained));
                    HIP_CHECK_EXC(hipMemset(m_recvBase[d], 0, R * m_recvBytes));
                    for(size_t r = 0; r < R; r++)
                        m_recv[r][d] = (char*)m_recvBase[d] + r * m_recvBytes;
                }
            }
            catch(...)
            {
                release();
                throw;
            }
        }

        FusedA2ARunner::~FusedA2ARunner()
        {
            release();
        }

        void FusedA2ARunner::release()
        {
            for(size_t d = 0; d < m_devices.size(); d++)
            {
                ScopedDevice device(m_devices[d]->deviceId);
                static_cast<void>(hipDeviceSynchronize());
            }
            for(size_t d = 0; d < m_recvBase.size(); d++)
            {
                ScopedDevice device(m_devices[d]->deviceId);
                static_cast<void>(hipFree(m_recvBase[d]));
            }
        }

        std::vector<std::vector<std::vector<KernelInvocation>>> FusedA2ARunner::solve()
        {
            const size_t W = m_devices.size();
            const size_t R = m_recv.size();
            for(size_t r = 0; r < R; r++)
            {
                for(size_t d = 0; d < W; d++)
                {
                    auto& inputs = dynamic_cast<ContractionInputs&>(*m_devices[d]->inputArr[r]);
                    inputs.fusedA2APeers.clear();
#ifdef TENSILELITE_ENABLE_SDMA
                    // One group per peer, flattened here so the packer needs no hsakmt type.
                    for(size_t j = 0; j < W; j++)
                    {
                        const HsaQueueResource& q = m_devices[d]->sdmaQueues[j]->queueResource();
                        inputs.fusedA2APeers.push_back({m_devices[j]->commFlags,
                                                        m_recv[r][j],
                                                        m_devices[d]->sdmaQueues[j]->ringBase(),
                                                        (void*)q.Queue_read_ptr_aql,
                                                        (void*)q.Queue_write_ptr_aql,
                                                        (void*)q.Queue_DoorBell_aql});
                    }
#endif
                    inputs.fusedA2ACounter = inputs.Synchronizer;
                    inputs.fusedA2AMyRank  = (uint32_t)d;
                    inputs.fusedA2ADrain   = FUSED_A2A_DRAIN_RECV | FUSED_A2A_DRAIN_SEND;
                }
            }

            std::vector<std::vector<std::vector<KernelInvocation>>> kernels(
                std::lcm(R, kChannels), std::vector<std::vector<KernelInvocation>>(W));
            for(size_t k = 0; k < kernels.size(); k++)
            {
                for(size_t d = 0; d < W; d++)
                {
                    ScopedDevice      device(m_devices[d]->deviceId);
                    ContractionInputs inputs
                        = dynamic_cast<ContractionInputs const&>(*m_devices[d]->inputArr[k % R]);
                    for(auto& peer : inputs.fusedA2APeers)
                        peer[FUSED_A2A_SLOT_FLAG_PTR]
                            = (char*)peer[FUSED_A2A_SLOT_FLAG_PTR]
                              + (k % kChannels) * FUSED_A2A_FLAG_BLOCK_BYTES;
                    kernels[k][d] = m_solution.solve(
                        m_problem, inputs, m_hardware, nullptr, 0, m_devices[d]->stream);
                    if(kernels[k][d].size() != 1)
                        throw std::runtime_error(concatenate("[fused-a2a] expected exactly one "
                                                             "kernel on device ",
                                                             d,
                                                             ", got ",
                                                             kernels[k][d].size()));
                }
            }
            return kernels;
        }
    } // namespace Client
} // namespace TensileLite
