#pragma once

// Dynamic tile order, where workgroups claim tiles from per-XCD counters: layout and limits.
// Parenthesized section names refer to docs/algorithms/direct/direct-flex-cdna4.md.

#include "kernel_args.h"
#include "layer_pars.h"
#include "tile_order.h"
#include "bunnies.hpp"
#include "persistent_grid.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>

namespace hipconv::cdna4::direct_flex
{

// A workgroup takes its first STATIC_TILES tiles in the static order (Dynamic tile order).
constexpr int STATIC_TILES = 3;
// Fewer tiles per workgroup keep the static order: the counters' memset costs about 1.65 us.
constexpr int MIN_ROUNDS = 6;
// Each XCD's counter fills a 128-byte line: front claims in the low half, back in the high.
// A run whose claims could reach 2^16 keeps the static order.
constexpr int CTR_WORDS     = 32;
constexpr int CTR_BYTES     = persistent::NUM_XCD * CTR_WORDS * 4;
constexpr int MAX_CLAIMS    = 1 << 15;
constexpr int MAILBOX_SLOTS = 2;
// A mailbox entry's flag for a tile taken from another XCD's run; -1 is no tile.
constexpr int STOLEN = 1 << 30;

// Whether tiles and wgs workgroups, split into runs, allow the dynamic order (Dynamic tile order).
inline bool runs_claim(int tiles, int wgs, int runs)
{
    if(tiles < MIN_ROUNDS * wgs)
        return false;
    for(int x = 0; x < runs; ++x)
    {
        const int len  = share_of(tiles, runs, x).len;
        const int left = len - STATIC_TILES * share_of(wgs, runs, x).len;
        if(left < 0 || left > MAX_CLAIMS)
            return false;
    }
    return true;
}

// Whether a launch claims tiles dynamically (Dynamic tile order).
inline bool dynamic_order(const Config& cfg, const ConvParams& par, int grid)
{
    // A csplit tile runs the static order alone (csplit_kernel.h).
    if(cfg.csplit != 1)
        return false;
    return runs_claim(tile_count(cfg, par), grid, persistent::NUM_XCD);
}

// A workgroup's claims of its run's tiles, and its steals from other runs (Dynamic tile order).
template <class T>
class DynamicOrder
{
    using Args = KernelArgs<T::PHASED>;
    using Pars = typename Args::Pars;
    using Run  = typename TileOrder<T>::Run;

    static constexpr int XCDS = persistent::NUM_XCD;

public:
    __device__ DynamicOrder(const Args& args, const TileOrder<T>& order)
        : args_(args)
        , order_(order)
    {
    }

    // Whether the workgroup claims its tiles; a phased segment claims only with enough rounds.
    __device__ __forceinline__ bool enabled() const
    {
        if constexpr(T::PHASED)
            return args_.cold().ws != nullptr && args_.seg().dyn != 0;
        return args_.cold().ws != nullptr;
    }

    // Run x's claim counter; agent scope, as other XCDs take from a run's back.
    __device__ __forceinline__ uint32_t* counter(int x) const
    {
        if constexpr(T::PHASED)
            return args_.cold().ws + (args_.seg_idx() * XCDS + x) * CTR_WORDS;
        return args_.cold().ws + x * CTR_WORDS;
    }

    // Lane 0 adds v through a VGPR address: on a uniform one the optimizer adds a vmcnt(0).
    __device__ __forceinline__ uint32_t add_lane0(uint32_t* p, uint32_t v) const
    {
        uint32_t old = 0;
        if(lane() == 0)
            old = __hip_atomic_fetch_add(p + bunnies::pin_vgpr(0),
                                         bunnies::pin_vgpr(v),
                                         __ATOMIC_RELAXED,
                                         __HIP_MEMORY_SCOPE_AGENT);
        return old;
    }

    // Mailbox entry for a claim on the workgroup's run, or for a steal.
    __device__ __forceinline__ int claim(uint32_t front_old, bool stealing) const
    {
        if(!stealing)
        {
            front_old   = __builtin_amdgcn_readfirstlane(front_old);
            Pars& kp    = args_.cold();
            const int x = order_.run_x();
            const int t = dyn_first_of(kp, x) + static_cast<int>(front_old & 0xffff);
            if(t + static_cast<int>(front_old >> 16) < run_end_of(kp, x))
                return t;
        }
        return steal();
    }

    __device__ __forceinline__ void post(int slot, int entry) const
    {
        if(lane() == 0)
            __hip_atomic_store(mailbox(slot), entry, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    __device__ __forceinline__ int* mailbox(int slot) const
    {
        int lines = XCDS;
        if constexpr(T::PHASED)
            lines = args_.cold().n_segs * XCDS;
        return reinterpret_cast<int*>(args_.cold().ws + lines * CTR_WORDS) +
               args_.block() * MAILBOX_SLOTS + slot;
    }

private:
    // Derived at each use: held through direct_flex_impl's main loop, it cost a VGPR there.
    __device__ __forceinline__ static int lane() { return bunnies::volatile_lane_id(); }

    // Run x's end and the first tile its workgroups claim; the run count and the run x wrapped.
    __device__ __forceinline__ int run_end_of(Pars& kp, int x) const
    {
        if constexpr(T::PHASED)
        {
            const Run r = order_.run_of(x);
            return r.first + r.len;
        }
        return kp.run_end[x];
    }
    __device__ __forceinline__ int dyn_first_of(Pars& kp, int x) const
    {
        if constexpr(T::PHASED)
        {
            const Run r = order_.run_of(x);
            return r.first + STATIC_TILES * r.wgs;
        }
        return kp.dyn_first[x];
    }
    __device__ __forceinline__ int runs() const
    {
        if constexpr(T::PHASED)
            return args_.seg().runs;
        return XCDS;
    }
    __device__ __forceinline__ int wrap(int x) const
    {
        if constexpr(T::PHASED)
            return x >= args_.seg().runs ? x - args_.seg().runs : x;
        return x % XCDS;
    }

    // Steals from the back of the run with the most tiles left; -1 once all are taken.
    __device__ __forceinline__ int steal() const
    {
        Pars& kp       = args_.cold();
        const int self = order_.run_x();
        for(;;)
        {
            const uint32_t c = __hip_atomic_load(
                counter(lane() % XCDS), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            int best = 0;
#pragma nounroll
            for(int d = 1; d <= runs(); ++d)
            {
                const int v       = wrap(self + d);
                const uint32_t cv = __builtin_amdgcn_readlane(c, v);
                const int left    = run_end_of(kp, v) - dyn_first_of(kp, v) -
                                 static_cast<int>(cv & 0xffff) - static_cast<int>(cv >> 16);
                best = std::max(best, left > 0 ? left * XCDS + XCDS - d : 0);
            }
            if(best == 0)
                return -1;
            const int v      = wrap(self + XCDS - best % XCDS);
            const uint32_t n = __builtin_amdgcn_readfirstlane(add_lane0(counter(v), 1u << 16));
            const int back   = static_cast<int>(n >> 16);
            if(dyn_first_of(kp, v) + static_cast<int>(n & 0xffff) + back < run_end_of(kp, v))
                return (run_end_of(kp, v) - 1 - back) | STOLEN;
        }
    }

    const Args& args_;
    const TileOrder<T>& order_;
};

} // namespace hipconv::cdna4::direct_flex
