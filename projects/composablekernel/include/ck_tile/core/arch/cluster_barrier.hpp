// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"

// Cluster barrier (gfx1250): barrier id -3 spans every workgroup of the cluster, as id -1 spans
// the waves of one workgroup. One wave per workgroup joins it, so a full cluster sync is a
// workgroup barrier and then the cluster barrier on one wave.
//
// These are inline asm, in the sequence llvm.amdgcn.s.cluster.barrier lowers to. The asm keeps
// the code where the caller put it: no LDS drain is added and nothing is scheduled into the
// sequence. A caller that needs a wait (s_wait_dscnt, s_wait_tensorcnt) issues it first.
//
// On other targets these are no-ops.

namespace ck_tile {

/// Every wave of every workgroup in the cluster: the workgroup barrier, then the workgroup's
/// first wave signals and waits on the cluster barrier.
CK_TILE_DEVICE void cluster_barrier()
{
#if defined(__gfx125__)
    asm volatile("s_barrier_signal_isfirst -1\n"
                 "s_barrier_wait -1\n"
                 "s_cbranch_scc0 1\n"
                 "s_barrier_signal -3\n"
                 "s_barrier_wait -3" ::
                     : "scc", "memory");
#endif
}

/// Arrive on the cluster barrier for this workgroup from its wave 0 only (wave: this wave's
/// index, wave-uniform), without a workgroup barrier first. cluster_barrier_wait() completes it.
CK_TILE_DEVICE void cluster_barrier_signal_wave0([[maybe_unused]] index_t wave)
{
#if defined(__gfx125__)
    asm volatile("s_cmp_eq_u32 %0, 0\n"
                 "s_cbranch_scc0 1\n"
                 "s_barrier_signal -3" ::"s"(wave)
                 : "scc", "memory");
#endif
}

/// Wait until every workgroup of the cluster has arrived on the cluster barrier.
CK_TILE_DEVICE void cluster_barrier_wait()
{
#if defined(__gfx125__)
    asm volatile("s_barrier_wait -3" ::: "memory");
#endif
}

} // namespace ck_tile
