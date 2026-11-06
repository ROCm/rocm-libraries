#pragma once

// Reads of direct_flex's kernel arguments, and a phased dgrad workgroup's segment (Strided dgrad).
// Parenthesized section names refer to docs/algorithms/direct/direct-flex-cdna4.md.

#include "layer_pars.h"
#include "magic_division.h"
#include "bunnies.hpp"

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace hipconv::cdna4::direct_flex
{

namespace detail
{

// Constant address space (4), so loads of these kernel-argument fields can be scalar.
using KernargDiv = const __attribute__((address_space(4))) MagicDiv;

// Divides by a magic divisor held in the kernel arguments.
__device__ inline uint32_t magic_div(KernargDiv& m, uint32_t x)
{
    MagicDiv d;
    d.multiplier = m.multiplier;
    d.shift      = m.shift;
    return d.div(x);
}

} // namespace detail

// The 14 leading dwords of the kernel arguments, preloaded into SGPRs at wave launch.
//
// A by-value struct argument is not preloaded, so direct_flex_cdna4 takes these fields as scalar
// parameters ahead of LayerPars and rebuilds Hot. The prologue decodes the first tile from Hot
// while the other kernel-argument lines load.
struct Hot
{
    int tiles, blocks, in_full, k_group, x_tiles, rebase_in;
    MagicDiv group_div, kg_div, q_div, p_div;
};
// LayerPars' offset in the kernel arguments.
constexpr int PARS_OFFSET = sizeof(Hot);
static_assert(PARS_OFFSET == 14 * 4 && PARS_OFFSET % alignof(LayerPars) == 0);

inline Hot hot_args(const LayerPars& pars)
{
    return {pars.tiles,
            pars.blocks,
            pars.in_full,
            pars.k_group,
            pars.x_tiles,
            pars.rebase_in,
            pars.group_div,
            pars.kg_div,
            pars.q_div,
            pars.p_div};
}

__device__ inline MagicDiv magic_of(uint32_t multiplier, uint32_t shift)
{
    MagicDiv d;
    d.multiplier = multiplier;
    d.shift      = shift;
    return d;
}

// The output geometry of the layer or of a phased workgroup's phase.
struct Geom
{
    int p, q, ph, pw, rows;
    detail::KernargDiv* q_div;
    detail::KernargDiv* p_div;
    int out0;
};

template <bool PHASED>
class KernelArgs
{
public:
    using Pars = const __attribute__((address_space(4))) LayerPars;
    using Seg  = const __attribute__((address_space(4))) LayerPars::Seg;

    // A phased workgroup's segment, kept in a VGPR: in SGPRs it put the kernel on the stack.
    __device__ KernelArgs()
    {
        if constexpr(PHASED)
        {
            Pars& kp    = cold();
            const int b = block();
            int s       = 0;
#pragma nounroll
            while(s + 1 < kp.n_segs && b >= kp.segs[s + 1].wg0)
                ++s;
            seg_reg_ = bunnies::pin_vgpr(s);
        }
    }

    __device__ static int block() { return bunnies::pin_sgpr(static_cast<int>(blockIdx.x)); }

    // Read through an opaque pointer: hoisted, the per-X-tile reads held SGPRs and spilled.
    __device__ static Pars& cold()
    {
        using kchar = const __attribute__((address_space(4))) char;
        auto* kp    = reinterpret_cast<Pars*>(
            __builtin_bit_cast(kchar*, __builtin_amdgcn_kernarg_segment_ptr()) + PARS_OFFSET);
        kp = bunnies::pin_sgpr(kp);
        return *kp;
    }

    // The kernel-argument values read before the main loop; wait_front waits for them at once.
    //
    // Read at their uses, each line costs its own wait.
    struct Front
    {
        int v[9], p0, p2, th, tw;
    };
    __device__ static Front fetch_front()
    {
        using word_t       = const volatile __attribute__((address_space(4))) int;
        constexpr int LINE = 64;
        constexpr int FRONT =
            PARS_OFFSET + offsetof(LayerPars, seq) + TAB_ROWS * sizeof(LayerPars::Seq);
        // in, wei, out follow pars
        constexpr int POINTERS = PARS_OFFSET + (sizeof(LayerPars) + 7) / 8 * 8;
        static_assert((FRONT + LINE - 1) / LINE == 9, "one operand per line in wait_front");
        auto* kp = __builtin_bit_cast(word_t*, __builtin_amdgcn_kernarg_segment_ptr());
        Front f;
        for(int i = 0; i < 9; ++i)
            f.v[i] = kp[i * LINE / 4];
        f.p0 = kp[POINTERS / 4];
        f.p2 = kp[(POINTERS + 2 * sizeof(void*)) / 4];
        f.th = kp[(PARS_OFFSET + offsetof(LayerPars, tap_h)) / 4];
        f.tw = kp[(PARS_OFFSET + offsetof(LayerPars, tap_w)) / 4];
        return f;
    }
    // One use of every value, so the loads share one s_waitcnt.
    //
    // Called after the work on preloaded arguments, which then overlaps the loads. The asm takes
    // SGPRs, as pin_sgpr's does, so it is emitted under the same condition.
    __device__ static void wait_front(const Front& f)
    {
        if constexpr(bunnies::enable_pin_sgpr)
            asm volatile("" ::"s"(f.v[0]),
                         "s"(f.v[1]),
                         "s"(f.v[2]),
                         "s"(f.v[3]),
                         "s"(f.v[4]),
                         "s"(f.v[5]),
                         "s"(f.v[6]),
                         "s"(f.v[7]),
                         "s"(f.v[8]),
                         "s"(f.p0),
                         "s"(f.p2),
                         "s"(f.th),
                         "s"(f.tw));
    }

    __device__ int seg_idx() const { return __builtin_amdgcn_readfirstlane(seg_reg_); }
    __device__ Seg& seg() const { return cold().segs[seg_idx()]; }

    __device__ Geom geom() const
    {
        Pars& kp = cold();
        if constexpr(PHASED)
        {
            const auto& g = kp.phases[seg().phase];
            return Geom{g.p, g.q, g.ph, g.pw, g.rows, &g.q_div, &g.p_div, g.out0};
        }
        return Geom{kp.p, kp.q, kp.ph, kp.pw, kp.rows, &kp.q_div, &kp.p_div, 0};
    }

private:
    int seg_reg_ = 0;
};

} // namespace hipconv::cdna4::direct_flex
