#pragma once

// Prime the two LDS rings and the delta register ring.
//
// The prologue and epilogue section of docs/algorithms/direct/direct-wgrad.md gives the rows
// the main loop needs resident before its first iteration.
//
// The register fill does not stall the ring loads, because the rows it reads land in scratch
// buffers that are their own LDS objects. Sharing one object with the ring would cost the
// whole prefetch: the waitcnt pass strengthens any wait before a transpose read to vmcnt(0)
// when the object being read has DMA outstanding.

#include "bunnies.hpp"
#include "row_schedule.h"
#include "workgroup_tiles.h"

#include <hip/hip_runtime.h>

namespace hipconv::cdna4::direct_wgrad
{

// Issue every load the prologue owns.
//
// Order matters, because buffer loads retire in order per wave and the wait below counts on
// it: the scratch rows go first, then the ring rows the loop will consume. Nothing here is
// conditional on the wave, so every wave leaves the same count outstanding and one immediate
// serves them all.
//
// Ring rows are placed by iteration index: iteration i reads slot i % row_buffers, so the rows
// the first prefetch_rows iterations consume go to slots 0 .. prefetch_rows - 1.
template <Config cfg, hipconv::DataType DT>
__device__ void issue_prologue_loads(int load_wave,
                                     const RowSchedule& sched,
                                     const SRowLoader<cfg, DT>& s_loader,
                                     const DeltaRowLoader<cfg, DT>& delta_loader,
                                     const SRing<cfg, DT>& s_ring,
                                     const DeltaRing<cfg, DT>& delta_ring,
                                     const DeltaScratch<cfg, DT>& scratch)
{
    bunnies::static_unroll<cfg.scratch_rows()>([&](auto j) {
        const int row = sched.prologue_delta_row(j);
        if(row >= 0)
            delta_loader.load(load_wave, row, scratch.template get<decltype(j)::value>());
    });

    bunnies::static_unroll<cfg.prefetch_rows>([&](auto i) {
        constexpr int slot = decltype(i)::value;
        delta_loader.load(load_wave, sched.delta_row(slot), delta_ring.template get<slot>());
        s_loader.load(load_wave, sched.s_row(slot), s_ring.template get<slot>());
    });
}

// Fill register-ring slots 0 .. kh-2 from the scratch buffers.
// pad is wave-uniform, so the zero-row branch is scalar.
template <Config cfg, hipconv::DataType DT>
__device__ void prime_delta_ring(const RowSchedule& sched,
                                 const DeltaScratch<cfg, DT>& scratch,
                                 int item,
                                 int k_base,
                                 DeltaRegRing<cfg, DT>& ring)
{
    const int item_elems = item * DeltaRowLayout<cfg>::size_elems;

    bunnies::static_unroll<cfg.kh - 1>([&](auto j) {
        constexpr int slot = decltype(j)::value;
        if(sched.prologue_delta_row(slot) < 0)
            zero_tile(ring.rows[slot]);
        else
            load_delta<cfg, DT, DeltaRowLayout<cfg>>(
                ring.rows[slot], scratch.template get<slot>() + item_elems, k_base);
    });
}

// Issue the loads and prime the register ring, leaving the back buffers filling.
//
// The register fill needs both the wait and the barrier before it reads LDS: vmcnt for this
// wave's own scratch loads, the barrier because every wave running the item wrote part of the
// row.
//
// The wait leaves loads_in_flight, which also retires S and delta row 0. Ping reads row 0 in
// the loop's first memory phase, before pong reaches a drain of its own, so the prologue is the
// only place pong can confirm row 0 has landed.
//
// The barrier is the raw intrinsic because __syncthreads fences all address spaces, which
// lowers to vmcnt(0) ahead of the barrier and drains the ring loads the schedule keeps in
// flight.
template <Config cfg, hipconv::DataType DT>
__device__ void run_prologue(int load_wave,
                             int item,
                             const RowSchedule& sched,
                             const SRowLoader<cfg, DT>& s_loader,
                             const DeltaRowLoader<cfg, DT>& delta_loader,
                             const SRing<cfg, DT>& s_ring,
                             const DeltaRing<cfg, DT>& delta_ring,
                             const DeltaScratch<cfg, DT>& scratch,
                             int k_base,
                             DeltaRegRing<cfg, DT>& ring)
{
    constexpr int keep = loads_in_flight<cfg, DT>;
    static_assert(keep <= arch::max_vmcnt);

    issue_prologue_loads<cfg, DT>(
        load_wave, sched, s_loader, delta_loader, s_ring, delta_ring, scratch);

    arch::s_wait_vmcnt<keep>();
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    prime_delta_ring<cfg, DT>(sched, scratch, item, k_base, ring);
}

} // namespace hipconv::cdna4::direct_wgrad
