#pragma once

// Depth taps, reduction step ranges and the equal-work split-K plan of a conv3d launch.
//
// Host and device evaluate the same functions, so the item decode and its inverse can be
// tested on the host. See docs/algorithms/direct/direct-3d.md.

#include "mathutil.h"

#include <hip/hip_runtime.h>

namespace hipconv::cdna4::direct_l1
{

// Half-open range of depth taps whose input plane lies inside the image.
struct TapRange
{
    int lo;
    int hi;
};

// The in-bounds tap range for an output plane whose tap 0 reads input depth d0.
//
// Tap t reads d0 + t*dilation_d, monotonic in t, so the in-bounds taps are one
// interval. Non-empty for every plane given is_valid_config's pad_d gate.
//
// Host-callable so depth_test.cpp can check the arithmetic against a scalar
// reference over padding, stride and dilation cases without a GPU.
__host__ __device__ inline constexpr TapRange
tap_range_for(int d0, int depth, int dilation_d, int kd)
{
    const int lo = d0 >= 0 ? 0 : divup(-d0, dilation_d);
    const int hi = minimum(kd, divup(depth - d0, dilation_d));
    return {lo, hi};
}

// Half-open range of a tile's reduction steps.
struct StepRange
{
    int lo;
    int hi;
};

// The steps split s of S runs: ceil(steps / S) of them, the last range clipped.
//
// Steps are (depth tap, C(64)) pairs, C innermost.
__host__ __device__ inline constexpr StepRange split_step_range(int s, int S, int steps)
{
    const int len = divup(steps, S);
    return {minimum(s * len, steps), minimum((s + 1) * len, steps)};
}

// The steps split s of S runs on a plane whose in-bounds taps are `taps`.
//
// The plane's live steps [taps.lo * C64, taps.hi * C64) are split as a whole, so an
// edge plane's items share its fewer taps instead of reading taps outside the image.
__host__ __device__ inline constexpr StepRange
split_live_range(int s, int S, TapRange taps, int C64)
{
    const StepRange r = split_step_range(s, S, (taps.hi - taps.lo) * C64);
    return {taps.lo * C64 + r.lo, taps.lo * C64 + r.hi};
}

// Whether every range is non-empty and every tap segment of it at least two C(64) blocks.
//
// A one-block segment would need a one-iteration path in the tap loop, which spills K128.
__host__ __device__ inline constexpr bool split_ranges_valid(int S, int steps, int C64)
{
    for(int s = 0; s < S; ++s)
    {
        const StepRange r = split_step_range(s, S, steps);
        if(r.lo >= r.hi)
            return false;
        for(int lo = r.lo; lo < r.hi;)
        {
            const int hi = minimum(r.hi, (lo / C64 + 1) * C64);
            if(hi - lo < 2)
                return false;
            lo = hi;
        }
    }
    return true;
}

// Depth geometry as the host passes it; DepthPars carries the same members on the device.
struct DepthGeometry
{
    int D;
    int E;
    int stride_d;
    int pad_d;
    int dilation_d;
};

// Live taps of output plane e.
template <class Geom>
__host__ __device__ inline constexpr int live_taps(const Geom& g, int kd, int e)
{
    const TapRange t = tap_range_for(e * g.stride_d - g.pad_d, g.D, g.dilation_d, kd);
    return t.hi - t.lo;
}

// Largest filter depth a split plan covers.
constexpr int SPLIT_PLAN_MAX_KD = 3;

// Edge planes, those with a tap outside the image, the device decode scans one by one.
constexpr int SPLIT_PLAN_MAX_EDGE_PLANES = 8;

// The split-K work items of a launch, equal in work rather than in count per tile.
//
// A plane with nt live taps has nt * C64 reduction steps and S_nt[nt] items, each a
// split_step_range share of them, so every item runs at most the plan's item length
// whatever its plane. Items are numbered plane-major: item j of a (group, batch, Q, P,
// K) column is split s of plane e. Planes [e_lo, e_hi) have every tap live and are
// contiguous, so only the edge planes outside them are decoded one at a time.
struct SplitPlan
{
    int S_nt[SPLIT_PLAN_MAX_KD + 1] = {};
    int T                           = 1; // items per column, summed over the planes
    int e_lo                        = 0;
    int e_hi                        = 0;
    int T_lo                        = 0; // items in planes [0, e_lo)
};

// The plan whose planes with nt live taps take ceil(nt * C64 / len) items.
template <class Geom>
inline SplitPlan make_split_plan(const Geom& g, int kd, int C64, int len)
{
    SplitPlan p;
    for(int nt = 1; nt <= kd; ++nt)
        p.S_nt[nt] = divup(nt * C64, len);

    // Every plane fully live lies in one interval, since d0 rises with e.
    p.e_lo = g.E;
    p.e_hi = g.E;
    for(int e = 0; e < g.E; ++e)
    {
        if(live_taps(g, kd, e) != kd)
            continue;
        if(p.e_lo == g.E)
            p.e_lo = e;
        p.e_hi = e + 1;
    }

    p.T = 0;
    for(int e = 0; e < g.E; ++e)
    {
        if(e == p.e_lo)
            p.T_lo = p.T;
        p.T += p.S_nt[live_taps(g, kd, e)];
    }
    if(p.e_lo == g.E)
        p.T_lo = p.T;
    return p;
}

// One split-K item: split s of the S splits of plane e.
struct PlaneItem
{
    int e;
    int s;
    int S;
};

// Item j of a column. div_full(x) returns x / S_nt[kd], by MagicDiv on the device.
template <class Geom, class DivFull>
__host__ __device__ inline PlaneItem
plane_item(const SplitPlan& p, const Geom& g, int kd, int j, DivFull div_full)
{
    for(int e = 0; e < p.e_lo; ++e)
    {
        const int S = p.S_nt[live_taps(g, kd, e)];
        if(j < S)
            return {e, j, S};
        j -= S;
    }

    const int S_full   = p.S_nt[kd];
    const int interior = (p.e_hi - p.e_lo) * S_full;
    if(j < interior)
    {
        const int q = div_full(j);
        return {p.e_lo + q, j - q * S_full, S_full};
    }
    j -= interior;

    int e = p.e_hi;
    for(; e < g.E - 1; ++e)
    {
        const int S = p.S_nt[live_taps(g, kd, e)];
        if(j < S)
            break;
        j -= S;
    }
    return {e, j, p.S_nt[live_taps(g, kd, e)]};
}

// The first item of plane e, inverting plane_item.
template <class Geom>
__host__ __device__ inline int plane_first_item(const SplitPlan& p, const Geom& g, int kd, int e)
{
    int j = 0;
    if(e <= p.e_lo)
    {
        for(int f = 0; f < e; ++f)
            j += p.S_nt[live_taps(g, kd, f)];
        return j;
    }
    j = p.T_lo + (minimum(e, p.e_hi) - p.e_lo) * p.S_nt[kd];
    for(int f = p.e_hi; f < e; ++f)
        j += p.S_nt[live_taps(g, kd, f)];
    return j;
}

} // namespace hipconv::cdna4::direct_l1
