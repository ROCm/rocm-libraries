#pragma once

#include "magic_division.h"
#include "mathutil.h"
#include "persistent_grid.h"

// BalancedL2Grid: balances each local L2 between resident weights and streamed inputs.
//
// A static schedule for the persistent direct-convolution grid that trades weight
// residency for input residency via output-channel (K) partitioning.
//
// Specific to an architecture of 8 local L2 caches (XCDs) with 32 CUs each; the
// NUM_XCD / NUM_CU_PER_XCD topology constants (persistent_grid.h) are baked into
// the partition and decode below.
//
// The 8 XCDs form a kparts x xgroups grid (xgroups = NUM_XCD / kparts). The
// blocks_k output-channel blocks split into kparts contiguous K-slices of per_xcd
// blocks each; the flattened G x N x P x Q work axis (X) splits into xgroups
// contiguous input segments. Each XCD owns one (kslice, xgroup) cell: it runs the
// per_xcd K-blocks of its slice over its X-segment. So an XCD holds only 1/kparts
// of the weights, freeing L2 for inputs.
//
// kparts == 1 is the unpartitioned grid: xgroups == NUM_XCD, per_xcd == blocks_k,
// every XCD holds all K and owns a 1/NUM_XCD X-segment. All formulas below collapse
// to that case exactly.
//
// Within an XCD the per_xcd K-blocks are innermost, requiring per_xcd <= 32; wider
// slices are rejected upstream. The x_xcd workgroups co-active in one round form an
// x_xcd x per_xcd tile with K and P x Q locality (L2 blocking), and a CU advances its
// X-local index by 1 per round so consecutive rounds touch adjacent X tiles (halo reuse).
//
// Decode for (wg, round) on the XCD owning K-slice kslice and X-segment
// [x_begin, x_begin + x_count), with x_xcd = max(1, 32 / per_xcd) and
// round_count = ceil(x_count / x_xcd):
//   t, local_k = divmod(wg, per_xcd)   // t = X-slot, local_k = K-block within slice
//   k          = kslice * per_xcd + local_k   // global K-block index
//   xl         = t * round_count + round      // strided
//   (p, q, n, g) = decode of x_begin + xl, P fastest then Q, N, G outermost
// This is a bijection over (k in [0, kparts*per_xcd), xl in [0, x_count)); padded
// slots are never run.
//
// The G outermost axis needs a 4th divmod (magic_n splits group from batch); it
// and n_blocks are parked in LDS (RoundInvariantContext) to stay off K128's
// register budget. The X-segment statics are pure index math (host+device testable).

namespace hipconv::cdna4::direct_l1
{

// A K-block partition of the persistent grid across XCDs.
//
// blocks_k output-channel blocks split into kparts contiguous K-slices; the
// num_xcd XCDs form a kparts x xgroups grid, so each XCD owns per_xcd_blocks
// blocks of one slice and one of xgroups input segments. kparts == 1 is the
// unpartitioned grid (every XCD holds all K).
struct KPartition
{
    int kparts;
    int per_xcd_blocks;
    int xgroups;
};

// Choose the K-block partition for blocks_k.
//
// 1. Factor blocks_k = 2^k * m (m odd).
// 2. If m > pad_threshold, pad m up so blocks_k gains a power-of-two factor: among
//    power-of-two multipliers d >= 2 with 2^k * d <= num_xcd, take the largest d
//    that ties the smallest padding remainder. Otherwise leave blocks_k as is.
// 3. Pick kparts (a power of two dividing the padded count, <= num_xcd) minimizing
//    |per_xcd_blocks - target_per_xcd|; ties go to fewer kparts (more K per XCD,
//    higher arithmetic intensity).
//
// num_xcd must be a power of two. Defaults match MI355X (8 XCDs) and the ~512-channel
// target (4 blocks of a 128-channel config).
inline constexpr KPartition
plan_k_partition(int blocks_k, int num_xcd = 8, int target_per_xcd = 4, int pad_threshold = 7)
{
    const SplitPow2 s = split_pow2(static_cast<uint32_t>(blocks_k));
    const int pow2    = static_cast<int>(s.pow2);
    const int m       = static_cast<int>(s.odd);
    const int max_d   = num_xcd / pow2; // largest extra power-of-two factor we can use

    int padded = blocks_k;
    if(m > pad_threshold && max_d >= 2)
    {
        int best_d   = 1;
        int best_rem = m; // any real candidate beats leaving m unpadded
        for(int d = 2; d <= max_d; d *= 2)
        {
            const int rem = make_divisible(m, d) - m;
            if(rem < best_rem || (rem == best_rem && d > best_d))
            {
                best_rem = rem;
                best_d   = d;
            }
        }
        padded = pow2 * make_divisible(m, best_d);
    }

    int best_kparts = 1;
    int best_dist   = padded - target_per_xcd; // kparts == 1 baseline
    if(best_dist < 0)
        best_dist = -best_dist;
    for(int kp = 2; kp <= num_xcd; kp *= 2)
    {
        if(padded % kp != 0)
            continue;
        const int per = padded / kp;
        int dist      = per - target_per_xcd;
        if(dist < 0)
            dist = -dist;
        if(dist < best_dist) // strict: ascending kp keeps ties at the smaller kparts
        {
            best_dist   = dist;
            best_kparts = kp;
        }
    }

    return {best_kparts, padded / best_kparts, num_xcd / best_kparts};
}

class BalancedL2Grid
{
public:
    // Flattened G x N x P x Q work-axis size (groups outermost).
    __host__ __device__ static int x_total(int blocks_p, int blocks_q, int n_blocks, int groups)
    {
        return groups * blocks_p * blocks_q * n_blocks;
    }

    // Per-xgroup X-segment length (ceil; trailing xgroups absorb the remainder).
    __host__ __device__ static int
    x_seg(int blocks_p, int blocks_q, int n_blocks, int groups, int xgroups)
    {
        const int total = x_total(blocks_p, blocks_q, n_blocks, groups);
        return divup(total, xgroups);
    }

    // First X index input-segment `xgroup` owns (start of its contiguous segment).
    __host__ __device__ static int
    x_begin(int blocks_p, int blocks_q, int n_blocks, int groups, int xgroups, int xgroup)
    {
        return xgroup * x_seg(blocks_p, blocks_q, n_blocks, groups, xgroups);
    }

    // Number of X items input-segment `xgroup` owns (its segment clamped to [0, X_seg]).
    __host__ __device__ static int
    x_count(int blocks_p, int blocks_q, int n_blocks, int groups, int xgroups, int xgroup)
    {
        const int total = x_total(blocks_p, blocks_q, n_blocks, groups);
        const int seg   = x_seg(blocks_p, blocks_q, n_blocks, groups, xgroups);
        const int begin = xgroup * seg;
        const int rem   = total - begin;
        if(rem <= 0)
            return 0;
        return rem < seg ? rem : seg;
    }

    // X-slots processed in parallel per round: 32 CUs / per_xcd, clamped to >= 1.
    __host__ __device__ static int x_xcd(int per_xcd)
    {
        const int x = persistent::NUM_CU_PER_XCD / per_xcd;
        return x > 0 ? x : 1;
    }

    // Rounds an XCD runs: its X-segment spread across x_xcd slots (ceil).
    __host__ __device__ static int round_count(int per_xcd,
                                               int blocks_p,
                                               int blocks_q,
                                               int n_blocks,
                                               int groups,
                                               int xgroups,
                                               int xgroup)
    {
        const int xcnt = x_count(blocks_p, blocks_q, n_blocks, groups, xgroups, xgroup);
        const int xx   = x_xcd(per_xcd);
        return divup(xcnt, xx);
    }

    // Uniform upper bound on rounds (xgroup 0 owns the largest segment).
    //
    // For callers needing a wg/xcc-independent bound, e.g. test buffer sizing.
    __host__ __device__ static int
    num_rounds(int per_xcd, int blocks_p, int blocks_q, int n_blocks, int groups, int xgroups)
    {
        return round_count(per_xcd, blocks_p, blocks_q, n_blocks, groups, xgroups, 0);
    }

    // Exact number of rounds THIS workgroup iterates (every one valid).
    //
    // wg's slot t = wg / per_xcd; slots t >= x_xcd are idle. An active slot walks
    // xl in [t*round_count, x_count), so its trip count is min(round_count,
    // x_count - t*round_count) clamped to >= 0.
    __host__ __device__ static int num_rounds_for_workgroup(int per_xcd,
                                                            int blocks_p,
                                                            int blocks_q,
                                                            int n_blocks,
                                                            int groups,
                                                            int xgroups,
                                                            int xgroup,
                                                            int workgroup_id)
    {
        const int xx = x_xcd(per_xcd);
        const int t  = workgroup_id / per_xcd;
        if(t >= xx)
            return 0;
        const int xcnt = x_count(blocks_p, blocks_q, n_blocks, groups, xgroups, xgroup);
        const int rc  = round_count(per_xcd, blocks_p, blocks_q, n_blocks, groups, xgroups, xgroup);
        const int rem = xcnt - t * rc;
        if(rem <= 0)
            return 0;
        return rem < rc ? rem : rc;
    }

    // Decode the work item for (workgroup_id, round_idx) on one XCD.
    //
    // x_begin, x_count, round_count, per_xcd, and kslice are round-invariant, so the
    // kernel computes them once and passes them in every round, keeping the decode
    // spill-free on K128.
    __device__ BalancedL2Grid(int per_xcd,
                              int kslice,
                              int blocks_p,
                              int blocks_q,
                              int n_blocks,
                              int x_begin,
                              int x_count,
                              int round_count,
                              MagicDiv magic_per_xcd,
                              MagicDiv magic_p,
                              MagicDiv magic_q,
                              MagicDiv magic_n,
                              int workgroup_id,
                              int round_idx)
    {
        const int xcnt  = x_count;
        const int begin = x_begin;
        const int rc    = round_count;

        // Decode wg -> (t = wg / per_xcd, local_k = wg % per_xcd).
        //
        // readfirstlane the seed so the whole decode chain stays in SGPRs; the uniform
        // params arrive un-propagated through this inlined body, so without it the
        // chain defaults to VGPRs and spills K128 across the round loop.
        const int wg = __builtin_amdgcn_readfirstlane(workgroup_id);
        uint32_t t_u = 0, k_u = 0;
        magic_per_xcd.divmod(static_cast<uint32_t>(wg), static_cast<uint32_t>(per_xcd), t_u, k_u);
        const int t = static_cast<int>(t_u);
        // Global K-block index: this XCD's slice base plus the local block.
        const int k_block = kslice * per_xcd + static_cast<int>(k_u);

        // Strided X-local index: slot t advances by 1 per round.
        const int xl = __builtin_amdgcn_readfirstlane(t * rc + round_idx);
        // valid_ drops the ragged tail and idle slots; only the test probe reads it.
        valid_ = round_idx < rc && xl < xcnt;

        // x -> (p, q, n, g), P fastest then Q, N, G outermost.
        //
        // After P and Q are peeled, r2 = g*n_blocks + n; magic_n splits them.
        const int x = begin + xl;
        uint32_t r1 = 0, p = 0, r2 = 0, qq = 0, gg = 0, nn = 0;
        magic_p.divmod(static_cast<uint32_t>(x), static_cast<uint32_t>(blocks_p), r1, p);
        magic_q.divmod(r1, static_cast<uint32_t>(blocks_q), r2, qq);
        magic_n.divmod(r2, static_cast<uint32_t>(n_blocks), gg, nn);

        k_ = k_block;
        p_ = static_cast<int>(p);
        q_ = static_cast<int>(qq);
        n_ = static_cast<int>(nn);
        g_ = static_cast<int>(gg);
    }

    __device__ bool valid() const { return valid_; }
    __device__ int k_idx() const { return k_; }
    __device__ int p_idx() const { return p_; }
    __device__ int q_idx() const { return q_; }
    __device__ int n_idx() const { return n_; }
    __device__ int g_idx() const { return g_; }

private:
    bool valid_;
    int k_;
    int p_;
    int q_;
    int n_;
    int g_;
};

} // namespace hipconv::cdna4::direct_l1
