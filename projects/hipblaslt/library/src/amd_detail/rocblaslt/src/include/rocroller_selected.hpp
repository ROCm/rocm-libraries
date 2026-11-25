// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocblaslt.h"

// RocRollerSelectedTile, the kernel match and the walk, in headers that depend
// on nothing GPU-side so they can be unit-tested GPU-free.
#include "rocroller_kernel_select.hpp"
#include "rocroller_buffer_limit.hpp"

#include <array>
#include <cstddef>
#include <cstring>
#include <vector>


// Copy of the problem the object API reads again at heuristic time. Host
// scalars are copied for scalarBytes. Buffer pointers become sentinels.
// Inline, and the caller passes the scalar width, so this header does not
// include auxiliary.hpp and the test does not need an exported symbol.
inline RocblasltContractionProblem
    problemKeptForSelection(const RocblasltContractionProblem& problem, size_t scalarBytes)
{
    RocblasltContractionProblem kept = problem;
    auto own = [&](const void* src, std::array<int8_t, 16>& storage) -> const void* {
        if(src == nullptr || scalarBytes > storage.size())
            return src;
        std::memcpy(storage.data(), src, scalarBytes);
        return storage.data();
    };
    kept.alpha = own(problem.alpha, kept.alpha_owned);
    kept.beta  = own(problem.beta, kept.beta_owned);

    static const char biasPresent = 1;
    static const char sentinelA   = 2;
    static const char sentinelB   = 3;
    kept.bias = problem.bias ? static_cast<const void*>(&biasPresent) : nullptr;
    auto mark = [](const void* p, const char* token) -> const void* {
        return p ? static_cast<const void*>(token) : nullptr;
    };
    kept.A = mark(problem.A, &sentinelA);
    kept.B = mark(problem.B, &sentinelB);
    if(problem.C && problem.D && problem.C == problem.D)
    {
        kept.C = const_cast<char*>(&sentinelA);
        kept.D = const_cast<char*>(&sentinelA);
    }
    else
    {
        kept.C = const_cast<char*>(problem.C ? &sentinelA : nullptr);
        kept.D = const_cast<char*>(problem.D ? &sentinelB : nullptr);
    }
    kept.E             = const_cast<char*>(problem.E ? &sentinelA : nullptr);
    kept.scaleA        = mark(problem.scaleA, &sentinelA);
    kept.scaleB        = mark(problem.scaleB, &sentinelB);
    kept.scaleC        = mark(problem.scaleC, &sentinelA);
    kept.scaleD        = mark(problem.scaleD, &sentinelB);
    kept.scaleE        = mark(problem.scaleE, &sentinelA);
    kept.scaleAlphaVec = mark(problem.scaleAlphaVec, &sentinelA);
    kept.amaxD         = const_cast<char*>(problem.amaxD ? &sentinelA : nullptr);
    kept.dPeer         = const_cast<char*>(problem.dPeer ? &sentinelB : nullptr);
    kept.batch_A       = nullptr;
    kept.batch_B       = nullptr;
    kept.batch_C       = nullptr;
    kept.batch_D       = nullptr;
    kept.batch_E       = nullptr;
    kept.workspace     = nullptr;
    kept.Synchronizer  = nullptr;
    kept.streamKFlags  = nullptr;
    kept.adaptive_launch = nullptr;
    kept.stream          = nullptr;
#if HIPBLASLT_HAS_GEMM_A2A_FUSION
    kept.fused_epilogue      = nullptr;
    kept.fused_prologue      = nullptr;
    kept.fused_a2a_peer_flag = nullptr;
#endif
    return kept;
}

// Same rejects and the same rank order as getRocRollerBestSolutions.
// Implemented by calling chooseSolutionIndexParameters. A problem with an
// operand of 4 GiB or more gets no tiles: rocRoller refuses it at every tile.
// A tile is marked unbuildable when unbuildable_configs.def records it.
rocblaslt_status rankRocRollerSelectedTiles(const RocblasltContractionProblem& prob,
                                            std::vector<RocRollerSelectedTile>& tiles);
