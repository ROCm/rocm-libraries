// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Single-process 4-GPU orchestration entry point for the fused GEMM.A2A kernel
// (Task 10). This is deliberately independent of the single-GPU benchmark loop
// in main.cpp: main() dispatches here (before the benchmark loop) when
// --fused-a2a is set and returns immediately afterwards. Nothing in the
// single-GPU path is touched.
//
// What this does (spec §3.1 / §3.2):
//   1. For each of W devices: allocate fresh per-device GEMM operands
//      (x=A, w=B, c=C, out=D) plus the fused-A2A buffers recv[] and flag[]
//      (fine-grained, since they are written by remote peers) and a device-scope
//      counter[].
//   2. Enable pairwise P2P access between all device pairs.
//   3. Per launch: zero counter[]/flag[] on every device, then for each device
//      build the host GEMM kernarg via solution->solve(), APPEND the fixed
//      156-byte fused-A2A segment (8 recv_ptr + 8 flag_ptr + counter_ptr +
//      5 u32 scalars) to that same KernelArguments object, and launch on the
//      device's stream. Because the launch reads kernel.args.size(), appending
//      to the host-generated args auto-sizes the launch to include the fused
//      tail — which is what fills the previously-garbage fused kernarg and makes
//      the hipErrorIllegalAddress(700) crash disappear.
//
// Scope: setup once, then repeat launch + dual-segment numeric validation for
// N iterations (Task 13a). Each iteration RE-ZEROES counter/flag/recv before
// the launch so the DRAIN handshake is actually exercised (race detection), and
// times the launch with per-device hipEvents (the iteration's latency is the MAX
// across the W cards, since DRAIN gates each kernel's exit on data receipt).
// After the loop it reports "race: N/N iterations passed" and p50/p90 latency.
// Success == every iteration passes the L2(recv)+L1(out) check with no HIP error
// (the benign hipErrorPeerAccessAlreadyEnabled aside).
//
// L1 validates the local out segment two independent ways (ROCM-27524 scheme D):
// (a) through the D descriptor strides, and (b) through a HARDCODED row-major
// stride (off=m*N+n) read straight from the copied-back raw bytes. (b) proves
// out's physical layout really is [M,N] with N contiguous -- it cannot be
// satisfied by a column-major out that merely agrees with a column-major
// descriptor (the original bug's false-green disguise).

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include "ClientProblemFactory.hpp"
#include "FusedA2ACounterSentinel.hpp"
#include "FusedA2AKernArg.hpp"
#include "SolutionIterator.hpp"

// The GPU-initiated SDMA route needs the host to create one ring per (device,
// peer) and hand the kernel the device-visible handle array. SdmaQueue.cpp is
// only compiled (and hsakmt only linked) when the option is on, so the include
// and every use of it are gated on the same macro.
#ifdef TENSILELITE_ENABLE_SDMA_A2A
#include "SdmaQueue.hpp"
#endif

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        // FUSED_A2A_MAX_RANKS, FUSED_A2A_SEGMENT_BYTES, fusedA2AWorldSizeValid
        // and appendFusedSegment come from FusedA2AKernArg.hpp so that the
        // gtest can exercise the real definitions rather than a copy.

        // Entry point invoked from main() when --fused-a2a is passed. Returns a
        // process exit code (0 == all iterations passed: numeric validation when
        // --fused-a2a-validate=1, else clean exit on all iterations).
        int runFusedA2A(po::variables_map const&                                       args,
                        std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> library,
                        std::shared_ptr<Hardware>                                      hardware,
                        ClientProblemFactory&                                          problemFactory)
        {
            const int  W        = args["fused-a2a-world"].as<int>();
            const int  drain    = args["fused-a2a-drain"].as<int>() ? 1 : 0;
            // validate=1 (default): compute host golden + numerically check every
            // iteration (correctness bridge, current behavior). validate=0: SKIP
            // the golden triple-loop and both compares — used on the full
            // production shape whose CPU golden (~309 GMAC) is prohibitively slow;
            // race detection then degrades to "kernel exited cleanly" (no HIP
            // error / no DRAIN hang), i.e. clean-exit, not byte-verified.
            const bool validate = args["fused-a2a-validate"].as<int>() != 0;

            // Bound W FIRST -- before it is printed, compared against deviceCount, or
            // used as a divisor. fusedA2AWorldSizeValid carries why the range is what
            // it is; what matters *here* is the position. The deviceCount check below
            // only bounds W by the machine, not by the ABI, and it cannot stand in for
            // the lower bound either: for W <= 0 `deviceCount < W` is false, so it
            // falls through. The check also has to precede the coordinate guard
            // further down, because W is already a divisor by then (`AM % W`,
            // `AM / W`) -- a W of 0 would divide by zero, and a negative W would
            // produce a misleading divisibility error, both before the range check
            // could ever run.
            if(!fusedA2AWorldSizeValid(W))
            {
                std::cerr << "[fused-a2a] ERROR: world size W=" << W
                          << " is out of range; the kernarg segment reserves exactly "
                          << FUSED_A2A_MAX_RANKS
                          << " recv_ptr/flag_ptr slots.\n"
                          << "  require: 1 <= W <= " << FUSED_A2A_MAX_RANKS
                          << ". Refusing to launch." << std::endl;
                return -1;
            }

            std::cout << "[fused-a2a] single-process " << W << "-GPU setup + launch smoke\n";

            int deviceCount = 0;
            HIP_CHECK_EXC(hipGetDeviceCount(&deviceCount));
            if(deviceCount < W)
            {
                std::cerr << "[fused-a2a] need " << W << " devices, found " << deviceCount
                          << std::endl;
                return 1;
            }

            // Pick the first problem / first solution. Task 10 only needs one
            // fused kernel launched on all W devices to prove the kernarg fill.
            auto problems = problemFactory.problems();
            if(problems.empty())
            {
                std::cerr << "[fused-a2a] no problems in config" << std::endl;
                return 1;
            }
            auto* problem = dynamic_cast<ContractionProblemGemm*>(problems.front().get());
            if(!problem)
            {
                std::cerr << "[fused-a2a] first problem is not a plain GEMM" << std::endl;
                return 1;
            }

            // Resolve the solution via the normal iterator (needs preProblem to
            // set its internal m_problem before getSolution()).
            auto solutionIterator = SolutionIterator::Default(library, hardware, args);
            solutionIterator->preProblem(problem);
            if(!solutionIterator->moreSolutionsInProblem())
            {
                std::cerr << "[fused-a2a] no solution for problem" << std::endl;
                return 1;
            }
            std::shared_ptr<ContractionSolution> solution = solutionIterator->getSolution();
            if(!solution)
            {
                std::cerr << "[fused-a2a] getSolution returned null" << std::endl;
                return 1;
            }
            std::cout << "[fused-a2a] solution: " << solution->name() << std::endl;

            // Tile sizes MUST come from THIS solution's macro-tile, not a hardcoded
            // 256: the kernel epilogue gates PUSH/local and computes dst_rank +
            // the counter index/target from the compile-time MacroTile0/MacroTile1
            // (see GlobalWriteBatch.py _emitFusedA2AHandshake, which uses
            // self.kernel["MacroTile0"]). sizeMapping.macroTile.{x,y} are those
            // same MT0/MT1 (the runtime WG grid is CeilDivide(M,macroTile.x) x
            // CeilDivide(N,macroTile.y), ContractionProblem.cpp:795-796).
            // `tilesPerRank` (= n_shard/MT0) is an EXACT count of the PUSH workgroups
            // sharing one counter slot (dst_rank, token-tile), compared for equality
            // kernel-side, and `tokenTiles` (= CeilDivide(N,MT1)) is the counter array's
            // token dimension; a hardcoded 256 against a 128 macro-tile makes the tile
            // factors wrong and over-restricts admissible shapes via the M%256/AM%256
            // guards. macroTile.x = MT0 (M dim), macroTile.y = MT1 (N dim).
            const uint32_t FUSED_A2A_M_TILE = (uint32_t)solution->sizeMapping.macroTile.x;
            const uint32_t FUSED_A2A_N_TILE = (uint32_t)solution->sizeMapping.macroTile.y;
            if(FUSED_A2A_M_TILE == 0 || FUSED_A2A_N_TILE == 0)
            {
                std::cerr << "[fused-a2a] solution macro-tile is zero (MT0="
                          << FUSED_A2A_M_TILE << " MT1=" << FUSED_A2A_N_TILE
                          << "); cannot derive fused-A2A tile sizes" << std::endl;
                return 1;
            }
            std::cout << "[fused-a2a] macro-tile from solution: MT0(M)=" << FUSED_A2A_M_TILE
                      << " MT1(N)=" << FUSED_A2A_N_TILE << "\n";

            // --- Derive fused shape from the problem (spec §0 relations, but
            //     using THIS problem's real M/N/K, not the big §0 defaults). ---
            // M/N-swap (col-major first-class, design §3.1): A=w[feature,K],
            // B=x[token,K]. freeSizeA now carries FEATURE (index-0=M, the
            // A2A-scattered dim), freeSizeB carries TOKEN. Keep M/N as the working
            // names for the arithmetic below to minimise churn; nFeature/nToken are
            // semantic aliases used in comments and log lines.
            const size_t M = problem->freeSizeA(0); // = nFeature (A2A-scattered dim)
            const size_t N = problem->freeSizeB(0); // = nToken (all output cols)
            const size_t K = problem->boundSize(0); // GEMM contraction dim K
            const size_t nFeature = M; // semantic alias: feature = M = index-0
            const size_t nToken   = N; // semantic alias: token   = N
            // A2A column count along FEATURE (M, index-0). Was AN (feature=N) before
            // the col-major swap. The FIRST `AM` FEATURE columns go all-to-all (PUSH
            // to remote recv); the remaining [AM, M) FEATURE columns stay local in
            // `out`. Chosen so AM < M (a local segment exists) and (AM/W)%MT0==0.
            // AM is supplied via --fused-a2a-am so it can match the shape being run
            // (medium: AM=2048, full: AM=10240) without editing this source. The
            // pre-swap flag --fused-a2a-an is renamed outright to --fused-a2a-am
            // (design §0): no in-tree caller passed the old flag, and this client's
            // program_options has no "defaulted" query, so a value-preserving alias
            // cannot be implemented reliably. A stale --fused-a2a-an now errors loudly
            // as an unknown option rather than being silently ignored.
            const size_t AM = (size_t)args["fused-a2a-am"].as<int>();
            if(AM % (size_t)W != 0)
            {
                std::cerr << "[fused-a2a] AM(" << AM << ") not divisible by W(" << W << ")"
                          << std::endl;
                return 1;
            }
            // nShard = AM/W is a FEATURE sub-segment (one rank's slice of feature M).
            const uint32_t nShard       = (uint32_t)(AM / (size_t)W);
            // tilesPerRank: whole feature-tiles per rank shard (nShard is feature).
            const uint32_t tilesPerRank = (uint32_t)(nShard / FUSED_A2A_M_TILE);
            // tokenTiles: token-tiles across the full token dim N. Post-swap the
            // A2A-scattered dim is FEATURE (WG0), so TOKEN (WG1) is the replicated
            // dim -- every token-tile workgroup in a rank's feature shard contributes
            // one PUSH to that rank.
            //
            // CEIL, not floor: tokenTiles is a DIMENSION of the counter array (the
            // kernel indexes counter[dst_rank*tokenTiles + WorkGroup1]) and the grid
            // has CeilDivide(N, MT1) token-tiles. There is no N % MT1 == 0 guard below
            // (token = batch*seqlen, the user gives what they give), so a floor here
            // would let WG1 == tokenTiles index one past the row -> counter overrun or
            // a slot no WG ever completes -> DRAIN deadlock.
            const uint32_t tokenTiles   = (uint32_t)((N + FUSED_A2A_N_TILE - 1) / FUSED_A2A_N_TILE);
            // mTiles: feature-tiles across the full feature dim M (diagnostic only).
            const uint32_t mTiles       = (uint32_t)(M / FUSED_A2A_M_TILE);
            // DEPRECATED: the kernel's election target is now FusedTilesPerRank (the
            // counter is per (dst_rank, token-tile), so only tilesPerRank WGs share a
            // slot). Still passed so the kernarg layout / offsets stay untouched.
            const uint32_t target       = tilesPerRank * tokenTiles;

            // Fail-fast on shapes that violate the fused-A2A design constraints
            // (spec section 0). The kernel maps a whole PUSH workgroup to a
            // SINGLE dst_rank, which is only correct when each rank's shard is an
            // integer number of macro-tiles along the A2A-scattered dim. Post-swap
            // the scattered dim is FEATURE = M, so the shard (n_shard = AM/W) must be
            // a multiple of the MacroTile0-wide (feature) tile. If n_shard < MT0 (or
            // not a multiple), one workgroup spans several ranks: its lanes are all
            // attributed to one rank, so data is scattered to the wrong recv buffer
            // AND, under DRAIN, ranks with no supplying workgroup poll a flag slot no
            // one ever sets -> the GPU hangs forever. Reject such shapes on the host
            // instead of launching into a deadlock. Constraints apply to AM (the A2A
            // width along FEATURE), not the whole M: AM % W == 0, (AM/W) % MT0 == 0
            // (=> n_shard >= MT0, all W ranks covered), M % MT0 == 0, AM % MT0 == 0
            // (whole feature-tiles), and AM <= M (local segment fits inside output).
            if(AM % (size_t)W != 0 || (nShard % FUSED_A2A_M_TILE) != 0
               || (M % (size_t)FUSED_A2A_M_TILE) != 0
               || (AM % (size_t)FUSED_A2A_M_TILE) != 0 || AM > M)
            {
                std::cerr
                    << "[fused-a2a] ERROR: problem shape violates fused-A2A "
                       "constraints (spec section 0).\n"
                    << "  M(feature)=" << M << " N(token)=" << N << " AM=" << AM
                    << " W=" << W << " n_shard=AM/W=" << nShard
                    << " MacroTile0(feature)=" << FUSED_A2A_M_TILE << "\n"
                    << "  require: AM % W == 0, (AM/W) % " << FUSED_A2A_M_TILE
                    << " == 0 (so n_shard >= " << FUSED_A2A_M_TILE
                    << " and every rank is covered), M % " << FUSED_A2A_M_TILE
                    << " == 0, AM % " << FUSED_A2A_M_TILE << " == 0, AM <= M.\n"
                    << "  e.g. W=4 needs AM >= " << ((size_t)W * FUSED_A2A_M_TILE)
                    << " (n_shard >= " << FUSED_A2A_M_TILE
                    << "). Refusing to launch (would deadlock in the DRAIN barrier)."
                    << std::endl;
                return -1;
            }

            // src_x, rect_x AND dst_y of the SDMA COPY_SUBWIN packet are 14-BIT
            // fields, and the kernel packs all three with bare s_lshl_b32/s_or_b32
            // -- no mask (SdmaPacketEmitter._packXY, the DW8 inline shift,
            // _packRectMinus1). The pure-Python reference encoder DOES mask, so past
            // 2^14 the two disagree and NEITHER complains: the packet would silently
            // address a wrapped-around token row / feature column and scatter data
            // into the wrong recv slot. Reject the shape here instead -- this mirrors
            // Tensile/Components/SdmaPacketEmitter.py:checkA2AFieldsFit.
            //   src_x  max = (W-1)*n_shard        (top peer's feature offset into D)
            //   rect_x     = n_shard              (the X extent itself; binding only
            //                                      at W == 1, else <= max src_x)
            //   dst_y  max = (W-1)*N + (tokenTiles-1)*MT1  (top rank's last tile)
            // src_y = j*MT1 is <= dst_y, so it needs no separate term. The `>=`
            // comparison is one value tighter than the hardware for rect_x (which is
            // minus-one encoded); deliberate, so all three terms read the same.
            const size_t maxSrcX  = (size_t)(W - 1) * (size_t)nShard;
            const size_t maxRectX = (size_t)nShard;
            const size_t maxDstY
                = (size_t)(W - 1) * N + (size_t)(tokenTiles - 1) * FUSED_A2A_N_TILE;
            if(maxSrcX >= (1u << 14) || maxRectX >= (1u << 14) || maxDstY >= (1u << 14))
            {
                std::cerr << "[fused-a2a] ERROR: geometry overflows the SDMA packet's "
                             "14-bit coordinate fields.\n"
                          << "  W=" << W << " AM=" << AM << " n_shard=AM/W=" << nShard
                          << " N(token)=" << N
                          << " MacroTile1(token)=" << FUSED_A2A_N_TILE
                          << " tokenTiles=" << tokenTiles << "\n"
                          << "  max src_x=(W-1)*n_shard=" << maxSrcX
                          << " rect_x=n_shard=" << maxRectX
                          << " max dst_y=" << maxDstY
                          << "; each must be < " << (1u << 14) << ".\n"
                          << "  Refusing to launch (the copy would silently move the "
                             "wrong band into the wrong recv slot). Reduce W, AM or N."
                          << std::endl;
                return -1;
            }

            // recv is feature-contiguous [W, token, feature_shard]: token is the outer
            // (strided-by-n_shard) axis, feature-shard is the inner stride-1 axis. The
            // fused PUSH store writes the FULL macro-tile edge (not just the logical
            // token count) and the recv SRD uses no edge clamp (num_records=BufferOOB),
            // so a PUSH WG's lanes address token rows up to the padded MT1 tile. Size
            // token to the MacroTile1-wide tile so those padding-row writes stay inside
            // the allocation. n_shard is already a multiple of MacroTile0 (host
            // constraint (AM/W)%MT0==0), so the contiguous feature extent is n_shard.
            const size_t nTokenPad = ((N + FUSED_A2A_N_TILE - 1) / FUSED_A2A_N_TILE) * FUSED_A2A_N_TILE;
            const size_t recvBytes    = (size_t)W * nTokenPad * nShard * sizeof(uint16_t); // bf16
            // flag slots are u64, NOT u32: the release signal is an SDMA ATOMIC
            // ADD64 (MORI's SDMA packet set has ADD64 and no ADD32), so each slot
            // is written 8 bytes wide. With a 4-byte stride the top rank's atomic
            // would run past the end of this allocation.
            const size_t flagBytes    = (size_t)W * sizeof(uint64_t);
            // counter is indexed [dst_rank][token-tile] -> W*tokenTiles u32 slots,
            // followed by a W-entry second-level counter2[dst_rank] (target
            // tokenTiles) at word index W*tokenTiles, then a single third-level
            // u32 counter3 at word index W*tokenTiles + W. counter2 converges the
            // DRAIN spinners to one per peer; counter3 is reserved for the
            // grid-wide workgroup tally that will elect the single DRAIN owner
            // (Task 4). All three ride this same allocation (and this same
            // per-iteration memset below) so the kernarg layout stays untouched.
            //
            // Past those live slots the allocation carries a guard tail (see
            // FusedA2ACounterSentinel.hpp). The tail catches only an overrun past
            // the TOP level, and it catches it by absorbing the write: the tail is
            // inside the allocation, so the store reddens the pattern rather than
            // reaching memory that is not ours. Absent the tail that store lands in
            // whatever hipMalloc handed back next and stays silent -- the counters
            // themselves still reach their expected values and every numeric check
            // passes. An off-by-one in a lower level stays inside the payload and
            // lands on a live slot instead: at the top of counter2's range that
            // slot is counter3, so the write would mis-elect the DRAIN owner rather
            // than raise anything. Only counterBytes is memset per launch; the tail
            // keeps its pattern and is re-checked after each launch.
            const size_t counterBytes      = fusedA2ACounterPayloadBytes((uint32_t)W, tokenTiles);
            const size_t counterAllocBytes = fusedA2ACounterAllocBytes((uint32_t)W, tokenTiles);
            const size_t aBytes       = problem->a().totalAllocatedBytes();
            const size_t bBytes       = problem->b().totalAllocatedBytes();
            const size_t cBytes       = problem->c().totalAllocatedBytes();
            const size_t dBytes       = problem->d().totalAllocatedBytes();

            std::cout << "[fused-a2a] nFeature(M)=" << nFeature << " nToken(N)=" << nToken
                      << " K=" << K << " AM=" << AM << " nShard=" << nShard
                      << " tilesPerRank=" << tilesPerRank << " tokenTiles=" << tokenTiles
                      << " mTiles=" << mTiles
                      << " target=" << target << " drain=" << drain << "\n";

            // --- Host golden setup (Task 11 numeric validation) ---------------
            // The GEMM is a TN GEMM (op(A)=A^T, op(B)=B), bf16 in, fp32 accumulate,
            // alpha=1, beta=0, C=0. Under the col-major swap, A carries FEATURE (m)
            // and B carries TOKEN (n): logically A=w[feature,K], B=x[token,K], and the
            // golden D'=[feature,token]. The golden math Dgold[m,n]=sum_k A[m,k]*B[k,n]
            // is INVARIANT under the swap -- only the semantic roles of m/n flip and
            // (for L1) the physical D layout becomes col-major. Physical layouts come
            // straight from the tensor descriptors (no hardcoded assumption): A element
            // (m,k) sits at m*aFreeStride + k*aBoundStride, similarly for B(k,n) and
            // D(m,n); the descriptor-derived strides carry the swapped shapes through
            // automatically. Every card runs the SAME A,B, so there is ONE golden
            // Dgold, stored row-major [M,N] (m=feature slow, n=token fast) as before.
            const auto&  aDesc = problem->a();
            const auto&  bDesc = problem->b();
            const auto&  dDesc = problem->d();
            const size_t aFreeAx  = problem->freeIndicesA()[0].i;   // A axis carrying M (feature)
            const size_t aBoundAx = problem->boundIndices()[0].a;   // A axis carrying K
            const size_t bFreeAx  = problem->freeIndicesB()[0].i;   // B axis carrying N (token)
            const size_t bBoundAx = problem->boundIndices()[0].b;   // B axis carrying K
            const size_t aFreeStride  = aDesc.strides()[aFreeAx];
            const size_t aBoundStride = aDesc.strides()[aBoundAx];
            const size_t bFreeStride  = bDesc.strides()[bFreeAx];
            const size_t bBoundStride = bDesc.strides()[bBoundAx];
            // D free-index axes: freeIndices()[j].d is the D dim for free index j.
            // Free index 0 is the A(M=feature) index, free index 1 is the B(N=token)
            // index. Post-swap D' is col-major, so dMStride==1 (feature contiguous).
            const size_t dMAx = problem->freeIndices()[0].d;
            const size_t dNAx = problem->freeIndices()[1].d;
            const size_t dMStride = dDesc.strides()[dMAx];
            const size_t dNStride = dDesc.strides()[dNAx];
            std::cout << "[fused-a2a] layout A(freeStride=" << aFreeStride << " boundStride="
                      << aBoundStride << ") B(freeStride=" << bFreeStride << " boundStride="
                      << bBoundStride << ") D(mStride=" << dMStride << " nStride=" << dNStride
                      << ")\n";

            // Deterministic small-magnitude bf16 inputs (indexed by logical coords,
            // written into the physical slot via the descriptor strides). Small
            // integers scaled by 0.5/0.25 keep the fp32 partial sums representable
            // and the final bf16 round predictable.
            const size_t aElems = aDesc.totalAllocatedElements();
            const size_t bElems = bDesc.totalAllocatedElements();
            std::vector<BFloat16> hA(aElems, BFloat16(0.0f));
            std::vector<BFloat16> hB(bElems, BFloat16(0.0f));
            auto aVal = [](size_t m, size_t k) {
                return BFloat16((float)(((int)((m * 3 + k) % 7)) - 3) * 0.5f);
            };
            auto bVal = [](size_t k, size_t n) {
                return BFloat16((float)(((int)((k + n * 2) % 5)) - 2) * 0.25f);
            };
            for(size_t m = 0; m < M; m++)
                for(size_t k = 0; k < K; k++)
                    hA[m * aFreeStride + k * aBoundStride] = aVal(m, k);
            for(size_t k = 0; k < K; k++)
                for(size_t n = 0; n < N; n++)
                    hB[k * bBoundStride + n * bFreeStride] = bVal(k, n);

            // Host golden GEMM: Dgold[m,n] = bf16( sum_k f32(A[m,k]) * f32(B[k,n]) ).
            // Only computed when validate=1; the triple loop is O(M*N*K) MACs and is
            // the expensive part we SKIP on the full shape (~309 GMAC). When
            // validate=0 Dgold stays empty and the numeric compares below are
            // bypassed entirely (not computed-then-ignored).
            std::vector<BFloat16> Dgold;
            if(validate)
            {
                Dgold.assign((size_t)M * N, BFloat16(0.0f));
                for(size_t m = 0; m < M; m++)
                {
                    for(size_t n = 0; n < N; n++)
                    {
                        float acc = 0.0f;
                        for(size_t k = 0; k < K; k++)
                            acc += (float)aVal(m, k) * (float)bVal(k, n);
                        Dgold[m * N + n] = BFloat16(acc); // row-major [M,N] golden store
                    }
                }
            }
            else
            {
                std::cout << "[fused-a2a] validate=0: SKIPPING host golden GEMM + numeric "
                             "compares (race = clean-exit only, not byte-verified)\n";
            }

            // --- Phase 1: per-device fresh allocation (spec §3.1). ---
            std::vector<void*> recv(W, nullptr), flag(W, nullptr), counter(W, nullptr);
            std::vector<void*> xA(W, nullptr), wB(W, nullptr), cC(W, nullptr), outD(W, nullptr);

            // Reference image of the counter guard tail: written once per device at
            // allocation, compared against the device copy after every launch.
            std::vector<uint32_t> hCounterGuard(FUSED_A2A_COUNTER_SENTINEL_WORDS);
            fusedA2ACounterSentinelFill(hCounterGuard.data());

            for(int d = 0; d < W; d++)
            {
                HIP_CHECK_EXC(hipSetDevice(d));
                // Fine-grained: written by remote peers, must bypass stale L2.
                HIP_CHECK_EXC(hipExtMallocWithFlags(&recv[d], recvBytes, hipDeviceMallocFinegrained));
                HIP_CHECK_EXC(hipExtMallocWithFlags(&flag[d], flagBytes, hipDeviceMallocFinegrained));
                // Local (not remotely written): plain device memory.
                HIP_CHECK_EXC(hipMalloc(&counter[d], counterAllocBytes));
                // Arm the guard tail. Sits past counterBytes, so the per-launch
                // memset below leaves it untouched.
                HIP_CHECK_EXC(hipMemcpy((char*)counter[d] + counterBytes,
                                        hCounterGuard.data(),
                                        FUSED_A2A_COUNTER_SENTINEL_BYTES,
                                        hipMemcpyHostToDevice));
                HIP_CHECK_EXC(hipMalloc(&xA[d], aBytes));
                HIP_CHECK_EXC(hipMalloc(&wB[d], bBytes));
                HIP_CHECK_EXC(hipMalloc(&cC[d], cBytes));
                HIP_CHECK_EXC(hipMalloc(&outD[d], dBytes));
                // Give GEMM operands deterministic real contents (same on every
                // card); zero C, out, recv. A/B are host-filled bf16 patterns so
                // the kernel computes a non-trivial GEMM we can check numerically.
                HIP_CHECK_EXC(hipMemcpy(xA[d], hA.data(), aBytes, hipMemcpyHostToDevice));
                HIP_CHECK_EXC(hipMemcpy(wB[d], hB.data(), bBytes, hipMemcpyHostToDevice));
                HIP_CHECK_EXC(hipMemset(cC[d], 0, cBytes));
                HIP_CHECK_EXC(hipMemset(outD[d], 0, dBytes));
                HIP_CHECK_EXC(hipMemset(recv[d], 0, recvBytes));
            }

            // --- P2P pairwise enable (spec §3.1). AlreadyEnabled is benign. ---
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
                        std::cerr << "[fused-a2a] WARNING: device " << s << " cannot P2P device "
                                  << t << std::endl;
                        continue;
                    }
                    hipError_t pe = hipDeviceEnablePeerAccess(t, 0);
                    if(pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled)
                        HIP_CHECK_EXC(pe);
                }
            }

            // --- Per-device SDMA queue sets: one ring per (device, peer), created
            //     AFTER P2P is enabled so a peer's recv/flag pages are already
            //     mapped into this device's VA space when the engine dereferences
            //     them. The self entry (j == d) is a loopback queue: §1.5 routes the
            //     p == my_rank packet through SDMA too, which is what gives this
            //     card's own flag slot a real producer (no DRAIN special case).
            //
            //     sdmaHandles is declared OUTSIDE the #ifdef on purpose: the kernarg
            //     append below is ordinary code that the preprocessor still has to
            //     parse in an SDMA-off build (the `return 1` in the #else is a
            //     RUNTIME return, it does not remove later statements from the token
            //     stream). Referring to the SdmaQueueSet vector directly down there
            //     made the default build fail to compile. ---
            std::vector<void*> sdmaHandles(W, nullptr);
#ifdef TENSILELITE_ENABLE_SDMA_A2A
            std::vector<std::unique_ptr<SdmaQueueSet>> sdmaSets(W);
            {
                std::vector<uint32_t> nodes(W);
                for(int j = 0; j < W; j++)
                    nodes[j] = sdmaNodeIdForDevice(j);
                for(int d = 0; d < W; d++)
                {
                    HIP_CHECK_EXC(hipSetDevice(d));
                    sdmaSets[d]   = std::make_unique<SdmaQueueSet>(nodes[d], nodes);
                    sdmaHandles[d] = sdmaSets[d]->deviceHandles();
                }
            }
            std::cout << "[fused-a2a] created " << W << " SDMA queues per device (one per peer)\n";
#else
            std::cerr << "[fused-a2a] ERROR: this client was built without "
                         "TENSILELITE_ENABLE_SDMA_A2A, so no SDMA rings exist, but the "
                         "fused epilogue unconditionally submits SDMA packets and would "
                         "dereference a null queue handle. Reconfigure with "
                         "-DTENSILELITE_ENABLE_SDMA_A2A=ON."
                      << std::endl;
            return 1;
#endif

            // --- Per-device streams + code-object adapters. The main adapter's
            //     modules are bound to device 0; give each device its own adapter
            //     with the fused .co loaded in that device's context so launches
            //     on devices 1..W-1 resolve the kernel correctly. ---
            auto filename = args["library-file"].as<std::string>();
            size_t dirPos = filename.rfind('/');
            std::string libraryDirectory = (dirPos != std::string::npos)
                                               ? filename.substr(0, dirPos + 1)
                                               : std::string(".");

            std::vector<std::shared_ptr<hip::SolutionAdapter>> adapters(W);
            std::vector<hipStream_t>                           streams(W, nullptr);
            auto const& codeObjectFiles = args["code-object"].as<std::vector<std::string>>();

            for(int d = 0; d < W; d++)
            {
                HIP_CHECK_EXC(hipSetDevice(d));
                HIP_CHECK_EXC(hipStreamCreate(&streams[d]));
                adapters[d] = std::make_shared<hip::SolutionAdapter>();
                bool loadedAny = false;
                for(auto const& co : codeObjectFiles)
                {
                    if(adapters[d]->loadCodeObjectFile(co) == hipSuccess)
                        loadedAny = true;
                }
                // Lazy loading discovers the fused .co by kernel name from the
                // TensileLibrary directory (same mechanism as main()).
                (void)adapters[d]->initializeLazyLoading(hardware->archName(), libraryDirectory);
                (void)loadedAny;
            }

            // --- Repeat loop (Task 13a): race detection + p50/p90 latency. ---
            // The launch → sync → validate sequence is repeated `iters` times.
            // Each iteration RE-ZEROES counter/flag/recv on all W devices before
            // the launch (otherwise a run leaves counters at target and flags at
            // READY, so the DRAIN barrier releases trivially and the race test is
            // vacuous). recv is re-zeroed too so a stale-correct recv from the
            // previous iteration cannot mask a broken scatter this iteration. The
            // GEMM operands (A/B), P2P access, streams, and code-object adapters
            // are set up ONCE above and reused. per iteration we rebuild the
            // KernelInvocation via solution->solve() + appendFusedSegment() so the
            // kernarg carries exactly one fused segment (reusing the same
            // invocation would append the tail repeatedly).
            const int iters  = std::max(1, args["fused-a2a-iters"].as<int>());
            int       warmup = args["fused-a2a-warmup"].as<int>();
            if(warmup < 0)
                warmup = 0;
            if(warmup >= iters)
                warmup = iters - 1; // keep at least one measured iteration

            std::cout << "[fused-a2a] repeat: iters=" << iters << " warmup=" << warmup
                      << " (post-warmup measured=" << (iters - warmup) << ") validate="
                      << (validate ? "1 (numeric)" : "0 (clean-exit only)") << "\n";

            // bf16 tolerance: ~3 decimal digits. Compare in fp32. (shared by
            // both validation segments, all iterations)
            auto closeBf16 = [](float got, float want) {
                float diff = std::fabs(got - want);
                float tol  = 1e-2f * std::max(1.0f, std::fabs(want));
                return diff <= tol;
            };
            // recv is feature-contiguous [W, token, feature_shard]: token stride =
            // n_shard (FusedNShard) and slot stride = N_token * n_shard (N = logical
            // SizeJ = nToken), with feature-shard as the stride-1 inner axis, i.e.
            // element offset = slotElem + t*n_shard + f_local.
            // recv is a bf16 buffer. slotStride uses the UNPADDED N to match the
            // kernel's SizeJ slot multiply. See task3-index-derivation.md.
            // NOTE (Task 6): nothing writes recv any more -- the CU-side remote PUSH
            // store was removed and the SDMA copy that replaces it lands in Task 7, so
            // this L2 check is EXPECTED to fail until then.
            const size_t slotStride = (size_t)N * (size_t)nShard; // elems per src slot (nToken*nShard)
            const size_t rowStride  = (size_t)nShard;             // per-token stride (feature-shard contiguous)

            // Persistent host scratch (reused each iteration, no per-iter alloc).
            // Only sized when validating; empty otherwise (no D2H copy-back either).
            std::vector<uint16_t> hRecv, hOut;
            if(validate)
            {
                hRecv.resize((size_t)W * nTokenPad * nShard);
                hOut.resize(dBytes / sizeof(uint16_t));
            }

            // Per-iteration events: start/stop on each device's stream to time the
            // fused launch. Because DRAIN=ON gates each kernel's exit on receiving
            // its data, the iteration's latency is the MAX across the W cards (the
            // slowest card gates all-to-all completion).
            std::vector<hipEvent_t> startEv(W, nullptr), stopEv(W, nullptr);
            for(int d = 0; d < W; d++)
            {
                HIP_CHECK_EXC(hipSetDevice(d));
                HIP_CHECK_EXC(hipEventCreate(&startEv[d]));
                HIP_CHECK_EXC(hipEventCreate(&stopEv[d]));
            }

            std::vector<double> latMeasUs;  // post-warmup only (for percentiles)
            int  passIters   = 0;
            bool raceFail     = false;
            int  firstFailIt  = -1;
            bool anyHipError  = false;
            bool guardFail    = false; // counter guard tail corrupted (see below)

            for(int it = 0; it < iters; it++)
            {
                const bool verbose = (it == 0); // full per-card breakdown only on iter 0

                // -- Re-zero counter/flag/recv on every device BEFORE launch. --
                for(int d = 0; d < W; d++)
                {
                    HIP_CHECK_EXC(hipSetDevice(d));
                    HIP_CHECK_EXC(hipMemset(counter[d], 0, counterBytes)); // inc from 0
                    HIP_CHECK_EXC(hipMemset(flag[d], 0, flagBytes));       // NOT_READY
                    HIP_CHECK_EXC(hipMemset(recv[d], 0, recvBytes));       // clear prior recv
                }
                for(int d = 0; d < W; d++)
                {
                    HIP_CHECK_EXC(hipSetDevice(d));
                    HIP_CHECK_EXC(hipDeviceSynchronize());
                }

                // -- Solve + append fused segment + record start + launch. DRAIN=ON:
                //    the last WG polls this device's own flag (set by peers), so all
                //    W must be launched for the barrier to release. Enqueue all,
                //    then synchronize. --
                std::vector<std::vector<KernelInvocation>> perDeviceKernels(W);
                for(int d = 0; d < W; d++)
                {
                    HIP_CHECK_EXC(hipSetDevice(d));

                    ContractionInputs inputs;
                    inputs.a     = xA[d];
                    inputs.b     = wB[d];
                    inputs.c     = cC[d];
                    inputs.d     = outD[d];
                    inputs.alpha = static_cast<float>(1);
                    inputs.beta  = static_cast<float>(0);
                    inputs.gpu   = true;

                    auto kernels
                        = solution->solve(*problem, inputs, *hardware, nullptr, 0, streams[d]);
                    if(kernels.empty())
                    {
                        std::cerr << "[fused-a2a] solve() produced no kernels on device " << d
                                  << " (iter " << it << ")" << std::endl;
                        return 1;
                    }

                    // recv/flag pointer views for device d: slot j = peer j's buffer.
                    std::vector<void*> recvView(W), flagView(W);
                    for(int j = 0; j < W; j++)
                    {
                        recvView[j] = recv[j];
                        flagView[j] = flag[j];
                    }

                    KernelInvocation& last       = kernels.back();
                    size_t            beforeSize = last.args.size();
                    appendFusedSegment(last.args,
                                       recvView,
                                       flagView,
                                       counter[d],
                                       (uint32_t)d, // my_rank
                                       target,
                                       (uint32_t)W,
                                       nShard,
                                       (uint32_t)drain,
                                       // kernarg "FusedAM" (Signature.py); pass AM as
                                       // the value to keep the client/kernel ABI matched.
                                       (uint32_t)AM,
                                       // SDMA offload args: this device's W-element
                                       // SdmaQueueDeviceHandle array (one queue per peer).
                                       sdmaHandles[d],
                                       tilesPerRank,
                                       tokenTiles);
                    // Print kernarg size only on iter 0 to avoid log spam; a constant
                    // size across iterations confirms exactly one fused segment.
                    if(it == 0)
                        std::cout << "[fused-a2a] dev " << d
                                  << " kernarg: host base(before append)=" << beforeSize
                                  << " size(after)=" << last.args.size() << "\n";

                    perDeviceKernels[d] = std::move(kernels);
                    HIP_CHECK_EXC(hipEventRecord(startEv[d], streams[d]));
                    HIP_CHECK_EXC(adapters[d]->launchKernels(perDeviceKernels[d], streams[d],
                                                             nullptr, nullptr));
                    HIP_CHECK_EXC(hipEventRecord(stopEv[d], streams[d]));
                }

                // -- Wait for every device; collect per-card elapsed time. --
                bool   ok       = true;
                double maxCardUs = 0.0;
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
                        if(us > maxCardUs)
                            maxCardUs = us;
                        if(verbose)
                            std::cout << "[fused-a2a] device " << d
                                      << " kernel exited cleanly (" << std::fixed
                                      << std::setprecision(1) << us << " us)\n";
                    }
                }

                // -- Counter guard tail (see FusedA2ACounterSentinel.hpp). --
                // Checked EVERY iteration and independently of `validate`: an
                // overrun past the counter payload corrupts unrelated device
                // memory, which no numeric check can see -- the counters
                // themselves still hold their expected values. Read with a
                // non-throwing hipMemcpy so that a device already wedged by a
                // failed launch degrades to a warning instead of masking the
                // kernel error that was just reported.
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
                                  << counterBytes + (size_t)bad * sizeof(uint32_t)
                                  << " of a " << counterAllocBytes << "-byte allocation) holds 0x"
                                  << std::hex << devGuard[bad] << ", expected 0x"
                                  << fusedA2ACounterSentinelWord((size_t)bad) << std::dec
                                  << " -- a counter index ran past the " << counterBytes
                                  << "-byte payload" << std::endl;
                        guardFail = true;
                        ok        = false;
                    }
                }

                // -- Dual-segment numeric validation (Task 11), EVERY iteration. --
                // Skipped entirely when validate=0 (l2Pass/l1Pass default to `ok`,
                // so the per-iteration verdict reduces to "kernel exited cleanly").
                bool l2Pass = ok;
                bool l1Pass = ok;
                if(ok && validate)
                {
                    // ---- L2: recv (PUSH segment). recv is feature-contiguous
                    // [W, token, feature_shard]. For destination card dst, slot src must
                    // hold the feature sub-segment [dst*nShard, dst*nShard+nShard) of
                    // Dgold across all N tokens, laid out as [src, t(token, outer),
                    // f(feature-local, inner/contiguous)]. Token is NOT sharded -- every
                    // rank holds all N tokens. All W src slots carry the identical shard
                    // in single-card emulation. See task3-index-derivation.md.
                    for(int dst = 0; dst < W && l2Pass; dst++)
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
                                    // global feature = dst*nShard + f, token = t;
                                    // Dgold row-major [M,N] -> Dgold[feature*N + token].
                                    float want = (float)Dgold[((size_t)dst * nShard + f) * N + t];
                                    if(!closeBf16(got, want))
                                    {
                                        if(mism < 5)
                                            std::cerr << "[fused-a2a] L2 MISMATCH iter=" << it
                                                      << " card=" << dst << " src=" << src
                                                      << " t=" << t << " f=" << f << " got=" << got
                                                      << " want=" << want << "\n";
                                        mism++;
                                    }
                                }
                            }
                        }
                        if(verbose || mism)
                            std::cout << "[fused-a2a] L2 recv card " << dst << ": "
                                      << (mism == 0 ? "PASS" : "FAIL")
                                      << " (mismatches=" << mism << ")\n";
                        if(mism)
                            l2Pass = false;
                    }

                    // ---- L1: out (local segment). Under the col-major swap the A2A
                    // slice runs along FEATURE (M): the first AM feature columns PUSH
                    // to recv (not written to out), the remaining feature columns
                    // [AM, M) stay local in out. Every token has such a tail value, so
                    // this checks out[m in [AM,M), n in [0,N)] against Dgold[m,n].
                    //
                    // TWO checks per card, both must pass (l1Pass &= both):
                    //   (a) descriptor-driven: off = m*dMStride + n*dNStride. This
                    //       reads out through the SAME strides the kernel was told
                    //       to write with. It confirms out matches golden UNDER the
                    //       descriptor's own layout -- but it CANNOT distinguish the
                    //       intended col-major out from some other layout the
                    //       descriptor also happens to describe (a false green).
                    //   (b) raw-bytes col-major: off = n*M + m, HARDCODED col-major
                    //       physical stride (M/feature contiguous), independent of the
                    //       descriptor. This proves the physical byte layout of out
                    //       really is [M,N] col-major with FEATURE contiguous, which is
                    //       what the A2A downstream consumes post-swap. If out were
                    //       physically row-major, (a) could still pass while (b) fails
                    //       -- so (b) is the anti-false-green proof (ROCM-27524
                    //       gemm-a2a plan validation point 2). When the descriptor is
                    //       col-major (dMStride==1, dNStride==M) the two offset formulas
                    //       coincide; (b) still stands as an explicit, descriptor-
                    //       independent statement of the physical layout.
                    const bool descColMajor = (dMStride == 1 && dNStride == M);
                    if(verbose)
                        std::cout << "[fused-a2a] D descriptor layout: dMStride=" << dMStride
                                  << " dNStride=" << dNStride << " -> "
                                  << (descColMajor ? "COL-MAJOR [M,N] (M/feature contiguous)"
                                                   : "NOT col-major (row-major or padded)")
                                  << "  (raw-bytes L1 check uses hardcoded off=n*M+m"
                                     " regardless)\n";
                    for(int d = 0; d < W && l1Pass; d++)
                    {
                        HIP_CHECK_EXC(hipSetDevice(d));
                        HIP_CHECK_EXC(
                            hipMemcpy(hOut.data(), outD[d], dBytes, hipMemcpyDeviceToHost));
                        size_t mismDesc = 0; // (a) descriptor-driven
                        size_t mismRaw  = 0; // (b) raw-bytes col-major (off=n*M+m)
                        for(size_t m = AM; m < M; m++) // feature-local tail beyond A2A slice
                        {
                            for(size_t n = 0; n < N; n++) // all tokens
                            {
                                float want = (float)Dgold[m * N + n];

                                // (a) descriptor-driven read.
                                {
                                    size_t   off = m * dMStride + n * dNStride;
                                    BFloat16 g;
                                    g.data    = hOut[off];
                                    float got = (float)g;
                                    if(!closeBf16(got, want))
                                    {
                                        if(mismDesc < 5)
                                            std::cerr << "[fused-a2a] L1(desc) MISMATCH iter=" << it
                                                      << " card=" << d << " m=" << m << " n=" << n
                                                      << " got=" << got << " want=" << want << "\n";
                                        mismDesc++;
                                    }
                                }

                                // (b) raw-bytes col-major read: hardcoded off=n*M+m,
                                //     NOT via the descriptor strides. Proves M/feature
                                //     is physically contiguous in out.
                                {
                                    size_t   off = n * M + m;
                                    BFloat16 g;
                                    g.data    = hOut[off];
                                    float got = (float)g;
                                    if(!closeBf16(got, want))
                                    {
                                        if(mismRaw < 5)
                                            std::cerr << "[fused-a2a] L1(raw col-major n*M+m) "
                                                         "MISMATCH iter="
                                                      << it << " card=" << d << " m=" << m
                                                      << " n=" << n << " got=" << got
                                                      << " want=" << want << "\n";
                                        mismRaw++;
                                    }
                                }
                            }
                        }
                        if(verbose || mismDesc || mismRaw)
                            std::cout << "[fused-a2a] L1 out card " << d
                                      << ": desc=" << (mismDesc == 0 ? "PASS" : "FAIL") << "("
                                      << mismDesc << ") rawColMajor="
                                      << (mismRaw == 0 ? "PASS" : "FAIL") << "(" << mismRaw << ")\n";
                        if(mismDesc || mismRaw)
                            l1Pass = false;
                    }
                }

                // -- Per-iteration verdict + latency bookkeeping. --
                const bool iterPass = ok && l2Pass && l1Pass;
                if(iterPass)
                    passIters++;
                else
                {
                    if(!raceFail)
                        firstFailIt = it;
                    raceFail = true;
                    std::cerr << "[fused-a2a] RACE FAIL at iter " << it
                              << " (hipOk=" << ok << " L2=" << l2Pass << " L1=" << l1Pass << ")\n";
                }

                if(it >= warmup)
                    latMeasUs.push_back(maxCardUs);

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

            // --- Race verdict + latency percentiles. ---
            // Wording reflects the mode: validate=1 -> numerically verified;
            // validate=0 -> exited cleanly (no HIP error / no DRAIN hang), not
            // byte-verified. Under DRAIN=ON a clean exit still means the barrier
            // released (data received), just not content-checked.
            std::cout << "[fused-a2a] race: " << passIters << "/" << iters
                      << (validate ? " iterations passed (numeric)"
                                   : " iterations exited cleanly (clean-exit, not byte-verified)")
                      << (raceFail ? "  (FAIL)" : "  (PASS)") << "\n";
            if(raceFail)
                std::cout << "[fused-a2a] race: first failing iteration = " << firstFailIt << "\n";

            // Percentiles over post-warmup samples. p50 = sorted[floor(0.5*n)],
            // p90 = sorted[floor(0.9*n)]. Single-process 4-GPU P2P on ONE node
            // (not real multi-node xGMI) — a relative on-node datum, not a
            // production figure. Deliberately NOT compared to any baseline.
            if(!latMeasUs.empty())
            {
                std::vector<double> s = latMeasUs;
                std::sort(s.begin(), s.end());
                const size_t n     = s.size();
                auto         pct   = [&](double p) { return s[std::min(n - 1, (size_t)(p * n))]; };
                const double p50   = pct(0.5);
                const double p90   = pct(0.9);
                const double lmin  = s.front();
                const double lmax  = s.back();
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

            // Cleanup.
            for(int d = 0; d < W; d++)
            {
                HIP_CHECK_EXC(hipSetDevice(d));
                if(recv[d])
                    (void)hipFree(recv[d]);
                if(flag[d])
                    (void)hipFree(flag[d]);
                if(counter[d])
                    (void)hipFree(counter[d]);
                if(xA[d])
                    (void)hipFree(xA[d]);
                if(wB[d])
                    (void)hipFree(wB[d]);
                if(cC[d])
                    (void)hipFree(cC[d]);
                if(outD[d])
                    (void)hipFree(outD[d]);
                if(streams[d])
                    (void)hipStreamDestroy(streams[d]);
            }

            std::cout << "[fused-a2a] overall " << (raceFail ? "FAILED" : "PASSED") << std::endl;
            // Exit codes: 2 = a kernel returned a HIP error in some iteration, or a
            //     counter guard tail came back corrupted -- both are hard runtime
            //     faults rather than numeric disagreement;
            // 3 = all kernels ran but some iteration failed numeric validation
            //     (only reachable when validate=1);
            // 0 = every iteration passed (validate=1: dual-segment numeric check;
            //     validate=0: clean exit on all iterations).
            if(anyHipError || guardFail)
                return 2;
            return raceFail ? 3 : 0;
        }

    } // namespace Client
} // namespace TensileLite
