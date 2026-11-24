#pragma once

// Tile order (Grid): runs per XCD, a workgroup's static tiles in its run, and tile decoding.
// Parenthesized section names refer to docs/algorithms/direct/direct-flex-cdna4.md.

#include "kernel_args.h"
#include "traits.h"
#include "persistent_grid.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>

namespace hipconv::cdna4::direct_flex
{

// A tile's first output position and first output channel.
struct Tile
{
    int x0, k0;
};

// Part x of q * n + r items in n parts, the first r of them one longer (Grid).
struct Share
{
    int first, len;
};
__host__ __device__ inline Share share(int q, int r, int x)
{
    return {x * q + std::min(x, r), q + (x < r ? 1 : 0)};
}
// Unsigned division: total is never negative, and a power-of-two n is a shift and a mask.
__host__ __device__ inline Share share_of(int total, int n, int x)
{
    const unsigned t = total, m = n;
    return share(static_cast<int>(t / m), static_cast<int>(t % m), x);
}

template <class T>
class TileOrder
{
    using Args = KernelArgs<T::PHASED>;

    static constexpr Config cfg = T::cfg;
    static constexpr int XCDS   = persistent::NUM_XCD;

public:
    struct Run
    {
        int first, len, wgs;
    };

    __device__ explicit TileOrder(const Args& args) : args_(args) {}

    // Run x and its workgroups: XCD x's, or a phased segment's run x (Partitioned grid).
    __device__ __forceinline__ Run run_of(int x) const
    {
        if constexpr(T::PHASED)
        {
            typename Args::Seg& sg = args_.seg();
            const Share t          = share(sg.run_q, sg.run_r, x);
            return Run{t.first, t.len, share(sg.wg_q, sg.wg_r, x).len};
        }
        typename Args::Pars& kp = args_.cold();
        const Share t           = share_of(kp.tiles, XCDS, x);
        return Run{t.first, t.len, share_of(kp.blocks, XCDS, x).len};
    }

    // The workgroup's run, and its index among the run's workgroups.
    __device__ __forceinline__ int run_x() const
    {
        if constexpr(T::PHASED)
        {
            const unsigned i = args_.block() - args_.seg().wg0;
            return args_.seg().runs == XCDS ? static_cast<int>(i % XCDS) : static_cast<int>(i);
        }
        return static_cast<int>(static_cast<unsigned>(args_.block()) % XCDS);
    }
    __device__ __forceinline__ int run_idx() const
    {
        if constexpr(T::PHASED)
        {
            const unsigned i = args_.block() - args_.seg().wg0;
            return args_.seg().runs == XCDS ? static_cast<int>(i / XCDS) : 0;
        }
        return static_cast<int>(static_cast<unsigned>(args_.block()) / XCDS);
    }

    // The workgroup's tile i in the static order, round-robin within its run, or -1.
    __device__ __forceinline__ int static_tile(int i) const
    {
        const Run r = run_of(run_x());
        const int j = run_idx() + i * r.wgs;
        return __builtin_amdgcn_readfirstlane(j < r.len ? r.first + j : -1);
    }
    // Static tiles 0 to n - 1, reading the run once.
    // cold() and block() keep CSE from merging the reads of separate static_tile() calls.
    // hot, if given: the tile and workgroup counts come from SGPRs (unphased).
    template <int n>
    __device__ __forceinline__ void static_tiles(int (&t)[n], const Hot* hot = nullptr) const
    {
        const Run r  = own_run(hot);
        const int j0 = run_idx();
#pragma unroll
        for(int i = 0; i < n; ++i)
        {
            const int j = j0 + i * r.wgs;
            t[i]        = __builtin_amdgcn_readfirstlane(j < r.len ? r.first + j : -1);
        }
    }
    // The workgroup's run; hot, if given: the tile and workgroup counts come from SGPRs (unphased).
    __device__ __forceinline__ Run own_run(const Hot* hot = nullptr) const
    {
        if constexpr(!T::PHASED)
        {
            const int x   = run_x();
            const Share s = share_of(hot ? hot->tiles : args_.cold().tiles, XCDS, x);
            return Run{
                s.first, s.len, share_of(hot ? hot->blocks : args_.cold().blocks, XCDS, x).len};
        }
        return run_of(run_x());
    }
    // The static order's tile after t, or -1; end is one past the run's last tile.
    __device__ __forceinline__ static int static_next(int t, int end, int wgs)
    {
        return t >= 0 && t + wgs < end ? t + wgs : -1;
    }
    __device__ __forceinline__ int static_next(int t) const
    {
        const Run r = own_run();
        return static_next(t, r.first + r.len, r.wgs);
    }

    // Decodes tile t; fprop skips the phase struct, which moved its allocation.
    // hot, if given: in_full, k_group, x_tiles and two divisors come from SGPRs (unphased).
    __device__ __forceinline__ Tile decode(int t, const Hot* hot = nullptr) const
    {
        typename Args::Pars& kp = args_.cold();
        int x_idx, k_idx;
        if constexpr(T::PHASED)
        {
            const auto& ph = kp.phases[args_.seg().phase];
            if(t < ph.in_full)
            {
                const int group =
                    static_cast<int>(detail::magic_div(ph.group_div, static_cast<uint32_t>(t)));
                const int rem = t - group * kp.k_group * ph.x_tiles;
                x_idx = static_cast<int>(detail::magic_div(kp.kg_div, static_cast<uint32_t>(rem)));
                k_idx = group * kp.k_group + (rem - x_idx * kp.k_group);
            }
            else
            {
                const int rest = t - ph.in_full;
                const int kr   = kp.tiles_k - kp.k_full;
                x_idx = static_cast<int>(detail::magic_div(kp.kr_div, static_cast<uint32_t>(rest)));
                k_idx = kp.k_full + (rest - x_idx * kr);
            }
            return Tile{__builtin_amdgcn_readfirstlane(x_idx) * cfg.x_tile(),
                        __builtin_amdgcn_readfirstlane(k_idx) * cfg.k_tile()};
        }
        if(t < (hot ? hot->in_full : kp.in_full))
        {
            const auto u      = static_cast<uint32_t>(t);
            const int k_group = hot ? hot->k_group : kp.k_group;
            const int group =
                static_cast<int>(hot ? hot->group_div.div(u) : detail::magic_div(kp.group_div, u));
            const int rem = t - group * k_group * (hot ? hot->x_tiles : kp.x_tiles);
            const auto r  = static_cast<uint32_t>(rem);
            x_idx = static_cast<int>(hot ? hot->kg_div.div(r) : detail::magic_div(kp.kg_div, r));
            k_idx = group * k_group + (rem - x_idx * k_group);
        }
        else
        {
            const int rest = t - (hot ? hot->in_full : kp.in_full);
            const int kr   = kp.tiles_k - kp.k_full;
            x_idx = static_cast<int>(detail::magic_div(kp.kr_div, static_cast<uint32_t>(rest)));
            k_idx = kp.k_full + (rest - x_idx * kr);
        }
        return Tile{__builtin_amdgcn_readfirstlane(x_idx) * cfg.x_tile(),
                    __builtin_amdgcn_readfirstlane(k_idx) * cfg.k_tile()};
    }

    // Tile count; Split-K item i is tile i % split_tiles() over chunk range i / split_tiles().
    __device__ __forceinline__ int split_tiles() const
    {
        if constexpr(T::PHASED)
            return args_.cold().phases[args_.seg().phase].x_tiles * args_.cold().tiles_k;
        return args_.cold().split_tiles;
    }

private:
    const Args& args_;
};

} // namespace hipconv::cdna4::direct_flex
