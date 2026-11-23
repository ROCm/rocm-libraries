// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// How to cut one dimension of a GEMM into equal slices so that every slice of a matrix starts on a
// multiple of 256 bytes (a misaligned start makes the stores slow). Header-only and free of any
// other dependency, so the TensileLite client and hipBLASLt share one rule.

#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace TensileLite::SliceAlignment
{
    // Auto: round the share up when that moves at most kMaxImbalance of the work to the last
    // slice. Off: always total / parts. Force: round whenever the last slice keeps work.
    enum class Mode
    {
        Auto,
        Off,
        Force
    };

    // What the split cuts; only used for the wording of the hints.
    enum class Dim
    {
        M,
        N,
        Batch
    };

    // A matrix whose slice offset moves by `step` elements per unit of the split dimension.
    // `bits` is the element size in bits (4 for fp4, 6 for fp6, ...).
    struct Matrix
    {
        const char* name;
        int64_t     step;
        int64_t     bits;
    };

    // The first partition (>= 1) of a matrix whose slice does not start on 256 bytes.
    struct Misaligned
    {
        const char* name;
        size_t      partition;
        int64_t     byteOffset;
        int64_t     step; // |step| of the matrix: 1 = contiguous along the split dimension
    };

    struct Result
    {
        int64_t                 share; // elements of every slice but the last; 0 = more parts than total
        int64_t                 granularity; // the share is rounded up to a multiple of this
        std::vector<Misaligned> misaligned;
    };

    constexpr int64_t kAlignBytes   = 256;
    constexpr double  kMaxImbalance = 0.01;

    inline bool parseMode(std::string const& s, Mode& mode)
    {
        if(s == "auto")
            mode = Mode::Auto;
        else if(s == "off")
            mode = Mode::Off;
        else if(s == "force")
            mode = Mode::Force;
        else
            return false;
        return true;
    }

    inline Result split(int64_t total, size_t parts, std::vector<Matrix> const& cut, Mode mode)
    {
        constexpr int64_t alignBits = kAlignBytes * 8;
        auto              absv      = [](int64_t v) { return v < 0 ? -v : v; };

        Result r{0, 1, {}};
        if(parts == 0)
            return r;
        for(auto const& m : cut)
        {
            int64_t stepBits = absv(m.step) * m.bits;
            if(m.step != 0 && stepBits != 0)
                r.granularity = std::lcm(r.granularity, alignBits / std::gcd(alignBits, stepBits));
        }

        int64_t n     = static_cast<int64_t>(parts);
        int64_t plain = total / n;
        if(plain == 0)
            return r;
        int64_t g       = r.granularity;
        int64_t aligned = (plain + g - 1) / g * g;
        int64_t last    = total - aligned * (n - 1);

        r.share = plain;
        if(mode != Mode::Off && last > 0)
        {
            bool nearlyFree = static_cast<double>(aligned - last) / static_cast<double>(aligned)
                              <= kMaxImbalance;
            if(mode == Mode::Force || nearlyFree)
                r.share = aligned;
        }

        for(auto const& m : cut)
        {
            int64_t stepBits = absv(m.step) * m.bits;
            if(stepBits == 0)
                continue;
            for(size_t i = 1; i < parts; i++)
            {
                int64_t startBits = static_cast<int64_t>(i) * r.share * stepBits;
                if(startBits % alignBits != 0)
                {
                    r.misaligned.push_back({m.name, i, startBits / 8, absv(m.step)});
                    break;
                }
            }
        }
        return r;
    }

    inline std::vector<std::string> hints(Result const& r, Dim dim, size_t parts)
    {
        const char* dimName = dim == Dim::M ? "M" : dim == Dim::N ? "N" : "the batch count";
        std::vector<std::string> out;
        for(auto const& m : r.misaligned)
        {
            std::string s = "the slice of " + std::string(m.name) + " for partition "
                            + std::to_string(m.partition) + " starts "
                            + std::to_string(m.byteOffset) + " bytes in, not a multiple of "
                            + std::to_string(kAlignBytes) + ", which is slow: ";
            // A slice starts partition * share * step elements in. With step 1 only the share
            // moves the start; otherwise padding the step (a leading dimension or the batch
            // stride) to 256 bytes aligns every slice whatever the share.
            if(m.step == 1)
                s += std::string("pad ") + dimName + " to a multiple of "
                     + std::to_string(parts * r.granularity);
            else
                s += std::string("pad the ")
                     + (dim == Dim::Batch ? "batch stride" : "leading dimension") + " of "
                     + m.name + " to a multiple of " + std::to_string(kAlignBytes) + " bytes";
            if(dim == Dim::M)
                s += ", or split N";
            out.push_back(std::move(s));
        }
        return out;
    }
}
