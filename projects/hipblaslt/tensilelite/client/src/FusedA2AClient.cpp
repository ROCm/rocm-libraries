// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Single-process multi-GPU orchestration for the fused GEMM.A2A kernel,
// dispatched from main.cpp when the fused-gemm-a2a option is set.

// TODO(AIHPBLAS-4534): fold this into the tensilelite-client path instead of
// duplicating its reference and data-init machinery.

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/FusedA2AKernArg.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <array>

#include "FusedA2ACounterSentinel.hpp"
#include "SolutionIterator.hpp"

// SdmaQueue.hpp is header-only and pulls in hsakmt, which is only on the
// include path (and linked) under this option.
#ifdef TENSILELITE_ENABLE_SDMA
#include "SdmaQueue.hpp"
#endif

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        // Returns a process exit code; 0 == all iterations passed.
        static int runFusedA2AForSolution(po::variables_map const&             args,
                                          std::shared_ptr<Hardware>            hardware,
                                          ContractionProblem*                  problemIn,
                                          std::shared_ptr<ContractionSolution> solution,
                                          int                                  runIdx)
        {
#ifdef TENSILELITE_ENABLE_SDMA
            const int      drainRecv  = args["fused-a2a-drain-recv"].as<int>() ? 1 : 0;
            const int      drainSend  = args["fused-a2a-drain-send"].as<int>() ? 1 : 0;
            const int      toValidate = args["num-elements-to-validate"].as<int>();
            const bool     validate   = toValidate != 0;
            const uint32_t drain
                = (drainRecv ? FUSED_A2A_DRAIN_RECV : 0u) | (drainSend ? FUSED_A2A_DRAIN_SEND : 0u);

            if(toValidate > 0)
            {
                std::cerr << "[fused-a2a] ERROR: --num-elements-to-validate=" << toValidate
                          << " is not supported; this path compares every element or none. "
                             "Use -1 for all, 0 for none. Refusing to launch."
                          << std::endl;
                return -1;
            }

            int deviceCount = 0;
            HIP_CHECK_EXC(hipGetDeviceCount(&deviceCount));

            const int worldArg = args["fused-a2a-world"].as<int>();
            const int W        = worldArg > 0 ? worldArg : deviceCount;

            // Bound W before it is used as a divisor further down.
            if(!fusedA2AWorldSizeValid(W))
            {
                std::cerr << "[fused-a2a] ERROR: world size W=" << W
                          << " is out of range; the kernarg segment reserves exactly "
                          << FUSED_A2A_MAX_RANKS << " peer groups.\n"
                          << "  require: 1 <= W <= " << FUSED_A2A_MAX_RANKS
                          << ". Refusing to launch." << std::endl;
                return -1;
            }

            const int deviceIdx = args["device-idx"].as<int>();
            if(deviceIdx != 0)
            {
                std::cerr << "[fused-a2a] ERROR: only --device-idx=0 is supported; rank r runs "
                             "on device r. Use HIP_VISIBLE_DEVICES to pick the set. Refusing to "
                             "launch."
                          << std::endl;
                return -1;
            }

            std::cout << "[fused-a2a] single-process " << W << "-GPU setup + launch smoke\n";

            if(deviceCount < W)
            {
                std::cerr << "[fused-a2a] need " << W << " devices, found " << deviceCount
                          << std::endl;
                return 1;
            }

            auto* problem = dynamic_cast<ContractionProblemGemm*>(problemIn);
            if(!problem)
            {
                std::cerr << "[fused-a2a] problem is not a plain GEMM" << std::endl;
                return 1;
            }

            std::cout << "[fused-a2a] solution: " << solution->name() << std::endl;

            if(!solution->problemType.fusedGemmA2A)
            {
                std::cerr << "[fused-a2a] ERROR: solution was not built with FusedGemmA2A; it "
                             "has no fused epilogue to drive. Refusing to launch."
                          << std::endl;
                return -1;
            }

            {
                auto const&                        pt            = solution->problemType;
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
                    {solution->requiredWorkspaceSize(*problem, *hardware) != 0, "workspace"},
                };
                std::string missing;
                for(auto const& [needed, name] : unsupported)
                    if(needed)
                        missing += (missing.empty() ? "" : ", ") + std::string(name);
                if(!missing.empty())
                {
                    std::cerr << "[fused-a2a] ERROR: solution needs inputs this client does not "
                                 "set: "
                              << missing << ". Refusing to launch." << std::endl;
                    return -1;
                }
            }

            // Tile sizes must come from THIS solution's macro-tile: the kernel
            // epilogue derives dst_rank and the counter index from MT0/MT1.
            const uint32_t macroTileM = (uint32_t)solution->sizeMapping.macroTile.x;
            const uint32_t macroTileN = (uint32_t)solution->sizeMapping.macroTile.y;
            if(macroTileM == 0 || macroTileN == 0)
            {
                std::cerr << "[fused-a2a] solution macro-tile is zero (MT0=" << macroTileM
                          << " MT1=" << macroTileN << "); cannot derive fused-A2A tile sizes"
                          << std::endl;
                return 1;
            }
            std::cout << "[fused-a2a] macro-tile from solution: MT0(M)=" << macroTileM
                      << " MT1(N)=" << macroTileN << "\n";

            // M/N-swap (col-major first-class): A=w[feature,K], B=x[token,K].
            const uint32_t M = (uint32_t)problem->freeSizeA(0); // = nFeature (A2A dim)
            const uint32_t N = (uint32_t)problem->freeSizeB(0); // = nToken (all output cols)
            const uint32_t K = (uint32_t)problem->boundSize(0); // GEMM contraction dim K

            const size_t numBatch = problem->batchIndices().size();
            if(numBatch > 1 || (numBatch == 1 && problem->batchSize(0) != 1))
            {
                std::cerr << "[fused-a2a] ERROR: batched GEMM is not supported" << std::endl;
                return 1;
            }

            // The first `AM` FEATURE columns go all-to-all; [AM, M) stay local in
            // `out`.
            auto const& amArgs = args["fused-a2a-am"].as<std::vector<int>>();
            if(!amArgs.empty() && amArgs.size() != 1 && (size_t)runIdx >= amArgs.size())
            {
                std::cerr << "[fused-a2a] ERROR: --fused-a2a-am has " << amArgs.size()
                          << " values, fewer than the problems selected by "
                             "--problem-start-idx/--num-problems; pass one value per selected "
                             "problem, or one value for all. Refusing to launch."
                          << std::endl;
                return -1;
            }
            const uint32_t AM
                = amArgs.empty() ? M : (uint32_t)amArgs[amArgs.size() == 1 ? 0 : runIdx];
            const uint32_t nShard       = AM / (uint32_t)W;
            const uint32_t tilesPerRank = (uint32_t)(nShard / macroTileM);
            // tokenTiles sizes the counter array and the padded recv buffer.
            const uint32_t tokenTiles = (N + macroTileN - 1) / macroTileN;
            // mTiles: diagnostic only.
            const uint32_t mTiles = M / macroTileM;

            if(AM % (uint32_t)W != 0 || (nShard % macroTileM) != 0 || (M % macroTileM) != 0
               || (AM % macroTileM) != 0 || AM > M)
            {
                std::cerr << "[fused-a2a] ERROR: problem shape violates fused-A2A "
                             "constraints (spec section 0).\n"
                          << "  M(feature)=" << M << " N(token)=" << N << " AM=" << AM << " W=" << W
                          << " n_shard=AM/W=" << nShard << " MacroTile0(feature)=" << macroTileM
                          << "\n"
                          << "  require: AM % W == 0, (AM/W) % " << macroTileM
                          << " == 0 (so n_shard >= " << macroTileM
                          << " and every rank is covered), M % " << macroTileM << " == 0, AM % "
                          << macroTileM << " == 0, AM <= M.\n"
                          << "  e.g. W=4 needs AM >= " << ((size_t)W * macroTileM)
                          << " (n_shard >= " << macroTileM
                          << "). Refusing to launch (would deadlock in the DRAIN barrier)."
                          << std::endl;
                return -1;
            }

            problem->setFusedA2AExtent(AM);
            problem->setFusedA2AWorld((uint32_t)W);

            if(!(*solution->problemPredicate)(*problem))
            {
                std::cerr << "[fused-a2a] solution predicate does not match the problem:"
                          << std::endl;
                solution->problemPredicate->debugEval(*problem, std::cerr);
                return 1;
            }

            // SDMA COPY_SUBWIN rect_x/rect_y are 14-bit; rect_x = n_shard scaled into
            // 16-byte packet elements, rect_y <= MT1. These are the only guard
            // for the fields SdmaPacketEmitter.py packs unmasked. `>=` is one
            // tighter than the hardware: the extents are minus-one encoded.
            const size_t elemShift    = 3; // log2(16B packet elem / 2B bf16)
            const size_t elemMultiple = (size_t)1 << elemShift;
            if(nShard % elemMultiple != 0)
            {
                std::cerr << "[fused-a2a] ERROR: n_shard is not addressable at the SDMA "
                             "packet's 16-byte element.\n"
                          << "  n_shard=AM/W=" << nShard << " must be a multiple of "
                          << elemMultiple << ".\n"
                          << "  Refusing to launch (the emitter's >>" << elemShift
                          << " would truncate it and copy a short band)." << std::endl;
                return -1;
            }
            const size_t maxRectX = (size_t)nShard >> elemShift;
            const size_t maxRectY = (size_t)macroTileN;
            if(maxRectX >= (1u << 14) || maxRectY >= (1u << 14))
            {
                std::cerr << "[fused-a2a] ERROR: geometry overflows the SDMA packet's "
                             "14-bit rect fields.\n"
                          << "  W=" << W << " AM=" << AM << " n_shard=AM/W=" << nShard
                          << " N(token)=" << N << " MacroTile1(token)=" << macroTileN
                          << " tokenTiles=" << tokenTiles << "\n"
                          << "  rect_x=n_shard>>" << elemShift << "=" << maxRectX
                          << " max rect_y=MT1=" << maxRectY << "; each must be < " << (1u << 14)
                          << ".\n"
                          << "  Refusing to launch (the copy would silently move the "
                             "wrong band into the wrong recv slot). rect_x is the copy "
                             "width itself and cannot be folded into the base address "
                             "the way the coordinates were: reduce AM or raise W "
                             "(the bound is AM < "
                          << ((size_t)(1u << 14) << elemShift) << "*W)." << std::endl;
                return -1;
            }

            if(problem->a().dataType() != rocisa::DataType::BFloat16
               || problem->b().dataType() != rocisa::DataType::BFloat16)
            {
                std::cerr << "[fused-a2a] ERROR: only bf16 A and B are supported, but A="
                          << problem->a().dataType() << " B=" << problem->b().dataType()
                          << ". Refusing to launch." << std::endl;
                return -1;
            }

            // recv is feature-contiguous [W, token, feature_shard]. Token is padded to
            // a whole MacroTile1 tile: the PUSH store writes the full macro-tile edge
            // with no edge clamp.
            const size_t nTokenPad = (size_t)tokenTiles * macroTileN;
            const size_t recvBytes = (size_t)W * nTokenPad * nShard * sizeof(uint16_t); // bf16
            // Per-launch reset range. Must stay in step with emitComputeFlagAddr's
            // *4 stride and the DRAIN poll's j*4.
            const size_t flagBytes = FUSED_A2A_FLAG_BLOCK_BYTES;
            // SDMA cursor pairs, counter2[dst_rank], counter3, then
            // counter[dst_rank][token-tile], then a guard tail
            // (FusedA2ACounterSentinel.hpp). Only counterBytes is memset per launch.
            const size_t counterBytes      = fusedA2ACounterPayloadBytes((uint32_t)W, tokenTiles);
            const size_t counterAllocBytes = fusedA2ACounterAllocBytes((uint32_t)W, tokenTiles);
            const size_t aBytes            = problem->a().totalAllocatedBytes();
            const size_t bBytes            = problem->b().totalAllocatedBytes();
            const size_t cBytes            = problem->c().totalAllocatedBytes();
            const size_t dBytes            = problem->d().totalAllocatedBytes();

            std::cout << "[fused-a2a] nFeature(M)=" << M << " nToken(N)=" << N << " K=" << K
                      << " AM=" << AM << " nShard=" << nShard << " tilesPerRank=" << tilesPerRank
                      << " tokenTiles=" << tokenTiles << " mTiles=" << mTiles
                      << " drainRecv=" << drainRecv << " drainSend=" << drainSend
                      << " drain=" << drain << "\n";

            // Physical layouts come from the tensor descriptors, never hardcoded:
            // A(m,k) sits at m*aFreeStride + k*aBoundStride, likewise B(k,n), D(m,n).
            const auto&  aDesc        = problem->a();
            const auto&  bDesc        = problem->b();
            const auto&  dDesc        = problem->d();
            const size_t aFreeStride  = aDesc.strides()[problem->freeIndicesA()[0].i];
            const size_t aBoundStride = aDesc.strides()[problem->boundIndices()[0].a];
            const size_t bFreeStride  = bDesc.strides()[problem->freeIndicesB()[0].i];
            const size_t bBoundStride = bDesc.strides()[problem->boundIndices()[0].b];
            // freeIndices()[j].d is the D dim for free index j: 0 = A's M(feature),
            // 1 = B's N(token).
            const size_t dMStride = dDesc.strides()[problem->freeIndices()[0].d];
            const size_t dNStride = dDesc.strides()[problem->freeIndices()[1].d];
            std::cout << "[fused-a2a] layout A(freeStride=" << aFreeStride
                      << " boundStride=" << aBoundStride << ") B(freeStride=" << bFreeStride
                      << " boundStride=" << bBoundStride << ") D(mStride=" << dMStride
                      << " nStride=" << dNStride << ")\n";

            // rect_x above is n_shard CONTIGUOUS bf16 features per token row, so the
            // feature axis has to be unit-stride for the copy to mean anything.
            if(dMStride != 1)
            {
                std::cerr << "[fused-a2a] ERROR: D's feature axis is not contiguous.\n"
                          << "  D mStride=" << dMStride << " must be 1.\n"
                          << "  Refusing to launch (the packet copies rect_x=" << maxRectX
                          << " 16-byte elements as one contiguous run, so a strided "
                             "feature axis would ship unrelated data to every peer)."
                          << std::endl;
                return -1;
            }

            // dNStride is the packet's src_pitch (StrideD1J), a 19-bit field, and is
            // only knowable once the descriptors are read. dst_pitch is n_shard,
            // already bounded by the rect_x check above.
            if(dNStride % elemMultiple != 0)
            {
                std::cerr << "[fused-a2a] ERROR: D's token-axis stride is not "
                             "addressable at the SDMA packet's 16-byte element.\n"
                          << "  ldd(D nStride)=" << dNStride << " must be a multiple of "
                          << elemMultiple << ".\n"
                          << "  Refusing to launch (the emitter's >>" << elemShift
                          << " would truncate the pitch and skew every token row)." << std::endl;
                return -1;
            }
            if((dNStride >> elemShift) >= (1u << 19))
            {
                std::cerr << "[fused-a2a] ERROR: D's token-axis stride overflows the "
                             "SDMA packet's 19-bit src_pitch field.\n"
                          << "  ldd(D nStride)=" << dNStride << " -> ldd>>" << elemShift << "="
                          << (dNStride >> elemShift) << " must be < " << (1u << 19)
                          << " (i.e. ldd < " << ((size_t)(1u << 19) << elemShift) << ").\n"
                          << "  Refusing to launch (the pitch would OR into the "
                             "neighbouring packet field). Reduce M or the D padding."
                          << std::endl;
                return -1;
            }

            // Same (x%7)-3 alphabet as DataInitialization's InitMode::Random. Fixed
            // seed, and B is drawn PER RANK so the W goldens differ.
            constexpr uint32_t kInitSeed = 42;
            auto               draw      = [](std::mt19937& g, float scale) {
                return BFloat16((float)((int)(g() % 7) - 3) * scale);
            };
            const size_t                       aElems = aDesc.totalAllocatedElements();
            const size_t                       bElems = bDesc.totalAllocatedElements();
            std::vector<BFloat16>              hA(aElems, BFloat16(0.0f));
            std::vector<std::vector<BFloat16>> hB(W, std::vector<BFloat16>(bElems, BFloat16(0.0f)));
            {
                std::seed_seq aSeq{kInitSeed, 0u};
                std::mt19937  aRng(aSeq);
                for(size_t m = 0; m < M; m++)
                    for(size_t k = 0; k < K; k++)
                        hA[m * aFreeStride + k * aBoundStride] = draw(aRng, 0.5f);
                for(int s = 0; s < W; s++)
                {
                    std::seed_seq bSeq{kInitSeed, 1u + (uint32_t)s};
                    std::mt19937  bRng(bSeq);
                    for(size_t k = 0; k < K; k++)
                        for(size_t n = 0; n < N; n++)
                            hB[s][k * bBoundStride + n * bFreeStride] = draw(bRng, 0.25f);
                }
            }

            // Dgold[s][m,n] = bf16( sum_k f32(A[m,k]) * f32(B_s[k,n]) ), read from the
            // arrays actually uploaded. Left empty when validate=0.
            const size_t          goldStride = (size_t)M * N; // elements per rank within Dgold
            std::vector<BFloat16> Dgold;
            if(validate)
            {
                Dgold.assign((size_t)W * goldStride, BFloat16(0.0f));
                for(int s = 0; s < W; s++)
                {
                    const BFloat16* bSrc = hB[s].data();
                    BFloat16*       dOut = Dgold.data() + (size_t)s * goldStride;
#pragma omp parallel for collapse(2)
                    for(size_t m = 0; m < M; m++)
                    {
                        for(size_t n = 0; n < N; n++)
                        {
                            float acc = 0.0f;
                            for(size_t k = 0; k < K; k++)
                                acc += (float)hA[m * aFreeStride + k * aBoundStride]
                                       * (float)bSrc[k * bBoundStride + n * bFreeStride];
                            dOut[m * N + n] = BFloat16(acc); // row-major [M,N] per rank
                        }
                    }
                }
            }
            else
            {
                std::cout << "[fused-a2a] num-elements-to-validate=0: SKIPPING host golden GEMM "
                             "+ numeric compares (race = clean-exit only, not byte-verified)\n";
            }

            // --- Per-device fresh allocation. ---
            // Bytes reserved ahead of recv for the flag array. The ABI passes flag
            // and recv as independent pointers; carving both from one allocation is
            // this client's choice, not a layout the kernel knows about.
            constexpr size_t kFlagBytes = 4096;
            static_assert(FUSED_A2A_FLAG_BLOCK_BYTES <= kFlagBytes,
                          "flag block must fit ahead of recv in the shared allocation");
            std::vector<void*> peer(W, nullptr), counter(W, nullptr);
            // Views into peer[d]: flag first, recv past kFlagBytes.
            std::vector<void*> recv(W, nullptr), flag(W, nullptr);
            std::vector<void*> wA(W, nullptr), xB(W, nullptr), cC(W, nullptr), outD(W, nullptr);

            // Staging buffer for arming each device's guard tail.
            std::vector<uint32_t> hCounterGuard(FUSED_A2A_COUNTER_SENTINEL_WORDS);
            fusedA2ACounterSentinelFill(hCounterGuard.data());

            for(int d = 0; d < W; d++)
            {
                HIP_CHECK_EXC(hipSetDevice(d));
                // Fine-grained: written by remote peers, must bypass stale L2.
                HIP_CHECK_EXC(hipExtMallocWithFlags(
                    &peer[d], kFlagBytes + recvBytes, hipDeviceMallocFinegrained));
                flag[d] = peer[d];
                recv[d] = (char*)peer[d] + kFlagBytes;
                // Zero the unused flag-array padding once, up to the recv offset.
                HIP_CHECK_EXC(hipMemset(peer[d], 0, kFlagBytes));
                // Local (not remotely written): plain device memory.
                HIP_CHECK_EXC(hipMalloc(&counter[d], counterAllocBytes));
                // Arm the guard tail. Sits past counterBytes, so the per-launch
                // memset below leaves it untouched.
                HIP_CHECK_EXC(hipMemcpy((char*)counter[d] + counterBytes,
                                        hCounterGuard.data(),
                                        FUSED_A2A_COUNTER_SENTINEL_BYTES,
                                        hipMemcpyHostToDevice));
                HIP_CHECK_EXC(hipMalloc(&wA[d], aBytes));
                HIP_CHECK_EXC(hipMalloc(&xB[d], bBytes));
                HIP_CHECK_EXC(hipMalloc(&cC[d], cBytes));
                HIP_CHECK_EXC(hipMalloc(&outD[d], dBytes));
                // A is shared by every card; B is this card's own draw.
                HIP_CHECK_EXC(hipMemcpy(wA[d], hA.data(), aBytes, hipMemcpyHostToDevice));
                HIP_CHECK_EXC(hipMemcpy(xB[d], hB[d].data(), bBytes, hipMemcpyHostToDevice));
                HIP_CHECK_EXC(hipMemset(cC[d], 0, cBytes));
                HIP_CHECK_EXC(hipMemset(outD[d], 0, dBytes));
                HIP_CHECK_EXC(hipMemset(recv[d], 0, recvBytes));
            }

            // --- P2P pairwise enable. AlreadyEnabled is benign. ---
            for(int s = 0; s < W; s++)
            {
                HIP_CHECK_EXC(hipSetDevice(s));
                for(int t = 0; t < W; t++)
                {
                    if(t == s)
                        continue;
                    int canAccess = 0;
                    HIP_CHECK_EXC(hipDeviceCanAccessPeer(&canAccess, s, t));
                    if(!canAccess)
                    {
                        std::cerr << "[fused-a2a] ERROR: device " << s << " cannot P2P device " << t
                                  << std::endl;
                        return 1;
                    }
                    hipError_t pe = hipDeviceEnablePeerAccess(t, 0);
                    if(pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled)
                        HIP_CHECK_EXC(pe);
                }
            }

            // One ring per (device, peer), created AFTER P2P enable so peer pages are
            // already mapped. The self entry (j == d) is a loopback queue, which gives
            // this card's own flag slot a real producer.
            std::vector<std::vector<std::unique_ptr<SdmaQueue>>> sdmaQueues(W);
            {
                std::vector<uint32_t> nodes(W);
                for(int j = 0; j < W; j++)
                    nodes[j] = sdmaNodeIdForDevice(j);
                for(int d = 0; d < W; d++)
                {
                    HIP_CHECK_EXC(hipSetDevice(d));
                    for(int j = 0; j < W; j++)
                        sdmaQueues[d].push_back(std::make_unique<SdmaQueue>(
                            nodes[d], sdmaSelectEngine(nodes[d], nodes[j])));
                }
            }
            std::cout << "[fused-a2a] created " << W << " SDMA queues per device (one per peer)\n";

            // Each device needs its own adapter: the main one binds its modules to
            // device 0, so launches on 1..W-1 would not resolve the kernel.
            auto        filename = args["library-file"].as<std::string>();
            size_t      dirPos   = filename.rfind('/');
            std::string libraryDirectory
                = (dirPos != std::string::npos) ? filename.substr(0, dirPos + 1) : std::string(".");

            std::vector<std::shared_ptr<hip::SolutionAdapter>> adapters(W);
            std::vector<hipStream_t>                           streams(W, nullptr);
            auto const& codeObjectFiles = args["code-object"].as<std::vector<std::string>>();

            for(int d = 0; d < W; d++)
            {
                HIP_CHECK_EXC(hipSetDevice(d));
                HIP_CHECK_EXC(hipStreamCreate(&streams[d]));
                adapters[d] = std::make_shared<hip::SolutionAdapter>();
                for(auto const& co : codeObjectFiles)
                {
                    (void)adapters[d]->loadCodeObjectFile(co);
                }
                // Lazy loading discovers the fused .co by kernel name from the
                // TensileLibrary directory (same mechanism as main()).
                (void)adapters[d]->initializeLazyLoading(hardware->archName(), libraryDirectory);
            }

            // Repeat loop: race detection + p50/p90 latency. Each iteration re-zeroes
            // counter/flag/recv (else the DRAIN barrier releases trivially and a
            // stale-correct recv masks a broken scatter).
            const int iters  = std::max(1,
                                       args["num-enqueues-per-sync"].as<int>()
                                           * args["num-syncs-per-benchmark"].as<int>());
            int       warmup = args["num-warmups"].as<int>();
            if(warmup < 0)
                warmup = 0;
            if(warmup >= iters)
                warmup = iters - 1; // keep at least one measured iteration

            std::cout << "[fused-a2a] repeat: iters=" << iters << " warmup=" << warmup
                      << " (post-warmup measured=" << (iters - warmup)
                      << ") validate=" << (validate ? "1 (numeric)" : "0 (clean-exit only)")
                      << "\n";

            // bf16 tolerance: ~3 decimal digits, compared in fp32.
            auto closeBf16 = [](float got, float want) {
                float diff = std::fabs(got - want);
                float tol  = 1e-2f * std::max(1.0f, std::fabs(want));
                return diff <= tol;
            };
            // slotStride uses the UNPADDED N, to match the kernel's SizeJ slot
            // multiply.
            const size_t slotStride = (size_t)N * nShard; // elems per src slot
            const size_t rowStride  = (size_t)nShard; // per-token stride (feature-shard contiguous)

            // Only sized when validating; empty otherwise, and no D2H copy-back.
            std::vector<uint16_t> hRecv, hOut;
            if(validate)
            {
                hRecv.resize((size_t)W * nTokenPad * nShard);
                hOut.resize(dBytes / sizeof(uint16_t));
            }

            // DRAIN=ON gates each kernel's exit on receiving its data, so the
            // iteration's latency is the MAX across the W cards.
            std::vector<hipEvent_t> startEv(W, nullptr), stopEv(W, nullptr);
            for(int d = 0; d < W; d++)
            {
                HIP_CHECK_EXC(hipSetDevice(d));
                HIP_CHECK_EXC(hipEventCreate(&startEv[d]));
                HIP_CHECK_EXC(hipEventCreate(&stopEv[d]));
            }

            // Both fed only when all W cards reported. A partial row would misalign
            // the per-card percentiles against each other and against the spread, and
            // would make maxCardUs a max over the survivors -- 0.0 if none reported.
            std::vector<double>              latMeasUs; // post-warmup only (percentiles)
            std::vector<std::vector<double>> perCardUs(W);
            std::vector<int>                 slowestCount(W, 0);
            int                              perCardSkipped = 0;
            int                              passIters      = 0;
            bool                             raceFail       = false;
            int                              firstFailIt    = -1;
            bool                             anyHipError    = false;
            bool guardFail = false; // counter guard tail corrupted (see below)
            bool wptrFail  = false; // an engine write pointer went backwards (see below)
            bool sendFail  = false; // outbound completion counter never reached W (see below)
            constexpr int kOutboundPolls = 1000;

            // Per-queue engine write pointer, carried across iterations. Must
            // never go backwards; the engine would read the difference as ~4 GB
            // of packets that were never written.
            std::vector<std::vector<uint64_t>> prevWptr(W);
            if(validate)
            {
                for(int d = 0; d < W; d++)
                {
                    prevWptr[d].resize(sdmaQueues[d].size());
                    for(size_t q = 0; q < prevWptr[d].size(); q++)
                        prevWptr[d][q] = *sdmaQueues[d][q]->queueResource().Queue_write_ptr_aql;
                }
            }

            std::vector<std::vector<KernelInvocation>> perDeviceKernels(W);
            for(int d = 0; d < W; d++)
            {
                HIP_CHECK_EXC(hipSetDevice(d));

                ContractionInputs inputs;
                inputs.a     = wA[d];
                inputs.b     = xB[d];
                inputs.c     = cC[d];
                inputs.d     = outD[d];
                inputs.alpha = static_cast<float>(1);
                inputs.beta  = static_cast<float>(0);
                inputs.gpu   = true;
                // One group per peer, flattened here so the packer needs no hsakmt type.
                for(size_t j = 0; j < sdmaQueues[d].size(); j++)
                {
                    const HsaQueueResource& r = sdmaQueues[d][j]->queueResource();
                    inputs.fusedA2APeers.push_back({j < flag.size() ? flag[j] : nullptr,
                                                    j < recv.size() ? recv[j] : nullptr,
                                                    sdmaQueues[d][j]->ringBase(),
                                                    (void*)r.Queue_read_ptr_aql,
                                                    (void*)r.Queue_write_ptr_aql,
                                                    (void*)r.Queue_DoorBell_aql});
                }
                inputs.fusedA2ACounter = counter[d];
                inputs.fusedA2AMyRank  = (uint32_t)d;
                inputs.fusedA2ADrain   = drain;

                auto kernels = solution->solve(*problem, inputs, *hardware, nullptr, 0, streams[d]);
                // Not back(): solve() appends conversion/reduction kernels after the
                // GEMM, and the fused segment belongs to the GEMM.
                if(kernels.size() != 1)
                {
                    std::cerr << "[fused-a2a] ERROR: expected exactly one kernel on device " << d
                              << ", got " << kernels.size() << std::endl;
                    return 1;
                }

                std::cout << "[fused-a2a] dev " << d
                          << " kernarg size=" << kernels.front().args.size() << "\n";

                perDeviceKernels[d] = std::move(kernels);
            }

            for(int it = 0; it < iters; it++)
            {
                const bool verbose = (it == 0); // full per-card breakdown only on iter 0

                // -- Re-zero counter/flag/recv on every device BEFORE launch. --
                for(int d = 0; d < W; d++)
                {
                    HIP_CHECK_EXC(hipSetDevice(d));
                    HIP_CHECK_EXC(hipMemset(counter[d], 0, counterBytes)); // inc from 0
                    HIP_CHECK_EXC(hipMemset(flag[d], 0, flagBytes)); // NOT_READY
                    HIP_CHECK_EXC(hipMemset(recv[d], 0, recvBytes)); // clear prior recv
                }
                for(int d = 0; d < W; d++)
                {
                    HIP_CHECK_EXC(hipSetDevice(d));
                    HIP_CHECK_EXC(hipDeviceSynchronize());
                }

                // Under DRAIN=ON the last WG polls this device's own flag, set by
                // peers, so all W must be enqueued before any can be synchronized.
                for(int d = 0; d < W; d++)
                {
                    HIP_CHECK_EXC(hipSetDevice(d));
                    HIP_CHECK_EXC(hipEventRecord(startEv[d], streams[d]));
                    HIP_CHECK_EXC(adapters[d]->launchKernels(
                        perDeviceKernels[d], streams[d], nullptr, nullptr));
                    HIP_CHECK_EXC(hipEventRecord(stopEv[d], streams[d]));
                }

                // -- Wait for every device; collect per-card elapsed time. --
                bool   ok        = true;
                double maxCardUs = 0.0;
                // -1 marks "this card did not report" (HIP error); see perCardUs decl.
                std::vector<double> cardUs(W, -1.0);
                for(int d = 0; d < W; d++)
                {
                    HIP_CHECK_EXC(hipSetDevice(d));
                    hipError_t se = hipStreamSynchronize(streams[d]);
                    if(se != hipSuccess)
                    {
                        std::cerr << "[fused-a2a] device " << d << " kernel FAILED (iter " << it
                                  << "): " << hipGetErrorString(se) << std::endl;
                        ok          = false;
                        anyHipError = true;
                    }
                    else
                    {
                        float ms = 0.0f;
                        HIP_CHECK_EXC(hipEventElapsedTime(&ms, startEv[d], stopEv[d]));
                        double us = (double)ms * 1000.0;
                        cardUs[d] = us;
                        if(us > maxCardUs)
                            maxCardUs = us;
                        if(verbose)
                            std::cout << "[fused-a2a] device " << d << " kernel exited cleanly ("
                                      << std::fixed << std::setprecision(1) << us << " us)\n";
                    }
                }

                // Counter guard tail (FusedA2ACounterSentinel.hpp), checked every
                // iteration independent of `validate`. Non-throwing hipMemcpy so a
                // device already wedged by a failed launch degrades to a warning
                // instead of masking the kernel error just reported.
                for(int d = 0; d < W; d++)
                {
                    HIP_CHECK_EXC(hipSetDevice(d));
                    std::vector<uint32_t> devGuard(FUSED_A2A_COUNTER_SENTINEL_WORDS);
                    hipError_t            ge = hipMemcpy(devGuard.data(),
                                              (const char*)counter[d] + counterBytes,
                                              FUSED_A2A_COUNTER_SENTINEL_BYTES,
                                              hipMemcpyDeviceToHost);
                    if(ge != hipSuccess)
                    {
                        std::cerr << "[fused-a2a] WARNING: could not read counter guard on device "
                                  << d << " (iter " << it << "): " << hipGetErrorString(ge)
                                  << std::endl;
                        continue;
                    }
                    int bad = fusedA2ACounterSentinelFirstBad(devGuard.data());
                    if(bad >= 0)
                    {
                        std::cerr << "[fused-a2a] COUNTER OVERRUN iter=" << it << " device=" << d
                                  << ": guard word " << bad << " (byte "
                                  << counterBytes + (size_t)bad * sizeof(uint32_t) << " of a "
                                  << counterAllocBytes << "-byte allocation) holds 0x" << std::hex
                                  << devGuard[bad] << ", expected 0x"
                                  << fusedA2ACounterSentinelWord((size_t)bad) << std::dec
                                  << " -- a counter index ran past the " << counterBytes
                                  << "-byte payload" << std::endl;
                        guardFail = true;
                        ok        = false;
                    }
                }

                for(int d = 0; AM != 0 && d < W; d++)
                {
                    HIP_CHECK_EXC(hipSetDevice(d));
                    uint32_t   outbound = 0;
                    hipError_t oe       = hipSuccess;
                    for(int spin = 0; spin < kOutboundPolls && outbound != (uint32_t)W; spin++)
                    {
                        oe = hipMemcpy(&outbound,
                                       (const char*)flag[d] + FUSED_A2A_OUTBOUND_OFFSET,
                                       sizeof(outbound),
                                       hipMemcpyDeviceToHost);
                        if(oe != hipSuccess)
                            break;
                    }
                    if(oe != hipSuccess)
                    {
                        std::cerr << "[fused-a2a] WARNING: could not read the outbound counter "
                                     "on device "
                                  << d << " (iter " << it << "): " << hipGetErrorString(oe)
                                  << std::endl;
                    }
                    else if(outbound != (uint32_t)W)
                    {
                        std::cerr << "[fused-a2a] OUTBOUND SIGNAL SHORT iter=" << it
                                  << " device=" << d << ": counter at byte "
                                  << FUSED_A2A_OUTBOUND_OFFSET << " of the flag block reads "
                                  << outbound << " after " << kOutboundPolls
                                  << " polls, expected " << W
                                  << " -- some queue never carried its completion ATOMIC, so "
                                     "drainSend would hang on it"
                                  << std::endl;
                        sendFail = true;
                        ok       = false;
                    }
                }

                if(validate)
                {
                    for(int d = 0; d < W; d++)
                    {
                        for(size_t q = 0; q < prevWptr[d].size(); q++)
                        {
                            const uint64_t now
                                = *sdmaQueues[d][q]->queueResource().Queue_write_ptr_aql;
                            const uint64_t was = prevWptr[d][q];
                            if(now < was)
                            {
                                std::cerr
                                    << "[fused-a2a] WPTR WENT BACKWARDS iter=" << it
                                    << " device=" << d << " queue=" << q << ": " << was << " -> "
                                    << now << " -- a producer reserved from a cursor behind the "
                                       "hardware write pointer; the engine now sees ~"
                                    << ((was - now) >> 20) << " MB of packets that were never "
                                       "written"
                                    << std::endl;
                                wptrFail = true;
                                ok       = false;
                            }
                            else if((now - was) % 4 != 0)
                            {
                                std::cerr << "[fused-a2a] WPTR NOT DWORD-ALIGNED iter=" << it
                                          << " device=" << d << " queue=" << q << ": " << was
                                          << " -> " << now << " (delta " << (now - was)
                                          << ") -- packets and wrap padding are whole dwords, so "
                                             "this is a torn or garbage publish"
                                          << std::endl;
                                wptrFail = true;
                                ok       = false;
                            }
                            prevWptr[d][q] = now;
                        }
                    }
                }

                // -- Dual-segment numeric validation, EVERY iteration. --
                // Skipped entirely when validate=0 (recvPass/localPass default to `ok`,
                // so the per-iteration verdict reduces to "kernel exited cleanly").
                bool recvPass  = ok;
                bool localPass = ok;
                if(ok && validate)
                {
                    // Recv: on destination card dst, slot src must hold features
                    // [dst*nShard, dst*nShard+nShard) of card SRC's golden, across all
                    // N tokens. Depending on src is what makes a slot filled by the
                    // wrong sender visible.
                    for(int dst = 0; dst < W && recvPass; dst++)
                    {
                        HIP_CHECK_EXC(hipSetDevice(dst));
                        HIP_CHECK_EXC(
                            hipMemcpy(hRecv.data(), recv[dst], recvBytes, hipMemcpyDeviceToHost));
                        size_t mism = 0;
                        for(int src = 0; src < W; src++)
                        {
                            for(size_t t = 0; t < N; t++)
                            {
                                for(uint32_t f = 0; f < nShard; f++)
                                {
                                    size_t   off = (size_t)src * slotStride + t * rowStride + f;
                                    BFloat16 g;
                                    g.data     = hRecv[off];
                                    float got  = (float)g;
                                    float want = (float)Dgold[(size_t)src * goldStride
                                                              + ((size_t)dst * nShard + f) * N + t];
                                    if(!closeBf16(got, want))
                                    {
                                        if(mism < 5)
                                            std::cerr << "[fused-a2a] RECV MISMATCH iter=" << it
                                                      << " card=" << dst << " src=" << src
                                                      << " t=" << t << " f=" << f << " got=" << got
                                                      << " want=" << want << "\n";
                                        mism++;
                                    }
                                }
                            }
                        }
                        if(verbose || mism)
                            std::cout << "[fused-a2a] RECV card " << dst << ": "
                                      << (mism == 0 ? "PASS" : "FAIL") << " (mismatches=" << mism
                                      << ")\n";
                        if(mism)
                            recvPass = false;
                    }

                    // Local: card d's local tail out[m in [AM,M)] against its OWN
                    // golden, read through the descriptor strides the kernel was told
                    // to write with. The layout itself is pinned by the dMStride and
                    // dNStride guards at launch, not re-derived here.
                    for(int d = 0; d < W && localPass; d++)
                    {
                        HIP_CHECK_EXC(hipSetDevice(d));
                        HIP_CHECK_EXC(
                            hipMemcpy(hOut.data(), outD[d], dBytes, hipMemcpyDeviceToHost));
                        size_t mism = 0;
                        for(size_t m = AM; m < M; m++) // feature-local tail beyond A2A slice
                        {
                            for(size_t n = 0; n < N; n++) // all tokens
                            {
                                float    want = (float)Dgold[(size_t)d * goldStride + m * N + n];
                                BFloat16 g;
                                g.data    = hOut[m * dMStride + n * dNStride];
                                float got = (float)g;
                                if(!closeBf16(got, want))
                                {
                                    if(mism < 5)
                                        std::cerr << "[fused-a2a] LOCAL MISMATCH iter=" << it
                                                  << " card=" << d << " m=" << m << " n=" << n
                                                  << " got=" << got << " want=" << want << "\n";
                                    mism++;
                                }
                            }
                        }
                        if(verbose || mism)
                            std::cout << "[fused-a2a] LOCAL card " << d << ": "
                                      << (mism == 0 ? "PASS" : "FAIL") << " (mismatches=" << mism
                                      << ")\n";
                        if(mism)
                            localPass = false;
                    }
                }

                // -- Per-iteration verdict + latency bookkeeping. --
                const bool iterPass = ok && recvPass && localPass;
                if(iterPass)
                    passIters++;
                else
                {
                    if(!raceFail)
                        firstFailIt = it;
                    raceFail = true;
                    std::cerr << "[fused-a2a] RACE FAIL at iter " << it << " (hipOk=" << ok
                              << " recv=" << recvPass << " local=" << localPass << ")\n";
                }

                if(it >= warmup)
                {
                    bool rowComplete = true;
                    for(int d = 0; d < W; d++)
                        if(cardUs[d] < 0.0)
                            rowComplete = false;
                    if(rowComplete)
                    {
                        latMeasUs.push_back(maxCardUs);

                        int slowest = 0;
                        for(int d = 1; d < W; d++)
                            if(cardUs[d] > cardUs[slowest])
                                slowest = d;
                        for(int d = 0; d < W; d++)
                            perCardUs[d].push_back(cardUs[d]);
                        slowestCount[slowest]++;
                    }
                    else
                        perCardSkipped++;
                }

                // Compact progress line (skip iter 0, which printed full breakdown).
                if(it != 0)
                    std::cout << "[fused-a2a] iter " << it << "/" << (iters - 1) << " "
                              << (iterPass ? "PASS" : "FAIL") << " maxCard=" << std::fixed
                              << std::setprecision(1) << maxCardUs << " us"
                              << (it < warmup ? " (warmup)" : "") << "\n";
            }

            for(int d = 0; d < W; d++)
            {
                (void)hipEventDestroy(startEv[d]);
                (void)hipEventDestroy(stopEv[d]);
            }

            std::cout << "[fused-a2a] race: " << passIters << "/" << iters
                      << (validate ? " iterations passed (numeric)"
                                   : " iterations exited cleanly (clean-exit, not byte-verified)")
                      << (raceFail ? "  (FAIL)" : "  (PASS)") << "\n";
            if(raceFail)
                std::cout << "[fused-a2a] race: first failing iteration = " << firstFailIt << "\n";

            // p50 = sorted[floor(0.5*n)], p90 = sorted[floor(0.9*n)].
            if(!latMeasUs.empty())
            {
                std::vector<double> s = latMeasUs;
                std::sort(s.begin(), s.end());
                const size_t n    = s.size();
                auto         pct  = [&](double p) { return s[std::min(n - 1, (size_t)(p * n))]; };
                const double p50  = pct(0.5);
                const double p90  = pct(0.9);
                const double lmin = s.front();
                const double lmax = s.back();
                std::cout << std::fixed << std::setprecision(1)
                          << "[fused-a2a] latency (post-warmup, " << n << " iters, MAX across " << W
                          << " cards/iter): p50=" << p50 << " us p90=" << p90 << " us min=" << lmin
                          << " us max=" << lmax << " us\n";
                std::cout << "[fused-a2a] latency NOTE: single-process " << W
                          << "-GPU P2P on ONE node (not multi-node xGMI); relative on-node "
                             "datum only.\n";
            }
            else
            {
                std::cout << "[fused-a2a] latency: no post-warmup samples collected\n";
            }

            // Outside both blocks below: when EVERY post-warmup row was incomplete,
            // both are empty and this is the only line that says why.
            if(perCardSkipped)
                std::cout << "[fused-a2a] timing: " << perCardSkipped
                          << " post-warmup iteration(s) excluded from both the latency and "
                             "per-card stats (not all "
                          << W << " cards reported)\n";

            // Per-card breakdown. The slowest-card histogram separates the three
            // sources of the max-vs-mean gap: concentrated on the first-enqueued id
            // means enqueue-order skew, concentrated elsewhere means one slow card,
            // spread evenly means order statistics over W roughly-iid cards.
            if(!perCardUs[0].empty())
            {
                const size_t n = perCardUs[0].size();
                // By value: the spread computation below needs the rows to stay
                // index-aligned.
                auto pctOf = [](std::vector<double> v, double p) {
                    std::sort(v.begin(), v.end());
                    return v[std::min(v.size() - 1, (size_t)(p * (double)v.size()))];
                };

                std::cout << std::fixed << std::setprecision(1);
                for(int d = 0; d < W; d++)
                {
                    double sum = 0.0;
                    for(double v : perCardUs[d])
                        sum += v;
                    std::cout << "[fused-a2a] per-card dev " << d << ": mean=" << (sum / (double)n)
                              << " us p50=" << pctOf(perCardUs[d], 0.5)
                              << " us p90=" << pctOf(perCardUs[d], 0.9) << " us  slowest in "
                              << slowestCount[d] << "/" << n << " iters\n";
                }

                // Spread = max-min across the W cards: what the MAX metric pays over
                // the mean on a given iteration.
                std::vector<double> spread(n, 0.0);
                for(size_t i = 0; i < n; i++)
                {
                    double lo = perCardUs[0][i], hi = perCardUs[0][i];
                    for(int d = 1; d < W; d++)
                    {
                        lo = std::min(lo, perCardUs[d][i]);
                        hi = std::max(hi, perCardUs[d][i]);
                    }
                    spread[i] = hi - lo;
                }
                std::cout << "[fused-a2a] per-card spread (max-min across " << W
                          << " cards): p50=" << pctOf(spread, 0.5)
                          << " us p90=" << pctOf(spread, 0.9) << " us max=" << pctOf(spread, 1.0)
                          << " us\n";
            }

            // Cleanup.
            for(int d = 0; d < W; d++)
            {
                HIP_CHECK_EXC(hipSetDevice(d));
                if(peer[d])
                    (void)hipFree(peer[d]);
                if(counter[d])
                    (void)hipFree(counter[d]);
                if(wA[d])
                    (void)hipFree(wA[d]);
                if(xB[d])
                    (void)hipFree(xB[d]);
                if(cC[d])
                    (void)hipFree(cC[d]);
                if(outD[d])
                    (void)hipFree(outD[d]);
                if(streams[d])
                    (void)hipStreamDestroy(streams[d]);
            }

            std::cout << "[fused-a2a] overall " << (raceFail ? "FAILED" : "PASSED") << std::endl;
            // Exit codes: 2 = a kernel returned a HIP error in some iteration, a
            //     counter guard tail came back corrupted, an engine write
            //     pointer went backwards, or the outbound completion counter
            //     never reached W -- all hard runtime faults rather than
            //     numeric disagreement;
            // 3 = all kernels ran but some iteration failed numeric validation
            //     (only reachable when validate=1);
            // 0 = every iteration passed (validate=1: dual-segment numeric check
            //     plus the write-pointer invariant; validate=0: clean exit on
            //     all iterations).
            if(anyHipError || guardFail || wptrFail || sendFail)
                return 2;
            return raceFail ? 3 : 0;
#else
            std::cerr << "[fused-a2a] ERROR: this client was built without "
                         "TENSILELITE_ENABLE_SDMA, so no SDMA rings exist, but the "
                         "fused epilogue unconditionally submits SDMA packets and would "
                         "dereference a null queue handle. Reconfigure with "
                         "-DTENSILELITE_ENABLE_SDMA=ON."
                      << std::endl;
            return 1;
#endif
        }

        int runFusedA2A(po::variables_map const&                                       args,
                        std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> library,
                        std::shared_ptr<Hardware>                                      hardware,
                        ContractionProblem*                                            problemIn,
                        int                                                            runIdx)
        {
            auto solutionIterator = SolutionIterator::Default(library, hardware, args);
            solutionIterator->preProblem(problemIn);

            if(!solutionIterator->moreSolutionsInProblem())
            {
                std::cerr << "[fused-a2a] no solution for problem" << std::endl;
                return 1;
            }

            while(solutionIterator->moreSolutionsInProblem())
            {
                std::shared_ptr<ContractionSolution> solution = solutionIterator->getSolution();
                if(!solution)
                {
                    std::cerr << "[fused-a2a] getSolution returned null" << std::endl;
                    return 1;
                }

                int rc = runFusedA2AForSolution(args, hardware, problemIn, solution, runIdx);
                if(rc != 0)
                    return rc;

                solutionIterator->postSolution();
            }
            return 0;
        }

    } // namespace Client
} // namespace TensileLite
