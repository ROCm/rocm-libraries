// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_utils.hpp"
#include "ck_tile/ops/megakernel/megakernel_sync.hpp"

namespace ck_tile {

/// @file
/// A megakernel runs several ops ("phases") in one persistent launch. Each workgroup walks the
/// phases in order and, within a phase, takes work items blockIdx.x, blockIdx.x + gridDim.x, ...
/// Phases share one LDS arena sized for the largest. A phase that consumes another phase's output
/// waits on that phase's completion counters instead of a kernel boundary.
///
/// A phase type provides:
/// @code
///   using Args;                                    // trivially copyable device arguments
///   static constexpr index_t kBlockSize;           // waves * get_warp_size() of the compile pass
///   static constexpr index_t kSlotRows;            // output rows covered by one counter slot
///   static constexpr index_t GetSmemSize();
///   CK_TILE_HOST_DEVICE static index_t NumWork(const Args&);
///   CK_TILE_HOST_DEVICE static index_t NumSlots(const Args&);
///   CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args&, index_t slot);
///   CK_TILE_DEVICE static index_t SlotOf(const Args&, index_t work);
///   CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args&, index_t work); // [r0, r1)
///   CK_TILE_HOST static bool IsSupportedArgument(const Args&);
///   using barrier_pipeline;                        // void, or the named barriers it arms
///   CK_TILE_DEVICE static auto InitBarriers();     // collective, once per launch
///   CK_TILE_DEVICE static void Run(const Args&, index_t work, void* smem, Token barriers);
/// @endcode
/// RowRange is read only when a later stage depends on this phase's rows (after_rows).
///
/// Optional prefetch hook, for inputs that do not depend on earlier phases (weights):
/// @code
///   static constexpr bool kHasPrefetch = true;
///   CK_TILE_DEVICE static void Prefetch(const Args&, index_t work, void* smem);
/// @endcode
/// If kHasPrefetch is true, every Run(args, work, smem, ...) is immediately preceded by
/// Prefetch(args, work, smem), called before the item waits on its dependency (if any) and after
/// the previous item's LDS reads are ordered. Prefetch may start loads into smem that Run then
/// consumes; between the two, the megakernel itself touches no LDS. Phases without the member are
/// unaffected.
///
/// Deadlock: a workgroup waits only on items of earlier phases, so the launch makes progress only
/// if every workgroup is resident at once. Launch at most MaxResidentGridSize() workgroups.
///
/// Counters: one zeroed buffer of NumCounters() words, kept across launches. Its first two words
/// track the launch epoch on the device, so a launch needs no reset and can be replayed by a graph.
/// Split-K phases that accumulate with atomics still need their output zeroed by the caller.

/// Stage dependency: none.
struct no_dependency
{
    static constexpr index_t kProducer = -1;
    static constexpr bool kRows        = false;
};

/// Stage dependency: wait until every item of stage @p Producer is done.
template <index_t Producer>
struct after_all
{
    static constexpr index_t kProducer = Producer;
    static constexpr bool kRows        = false;
};

/// Stage dependency: wait until stage @p Producer has finished the output rows this item reads,
/// i.e. this phase's RowRange() of the item, in units of the producer's kSlotRows.
template <index_t Producer>
struct after_rows
{
    static constexpr index_t kProducer = Producer;
    static constexpr bool kRows        = true;
};

/// True when phase @p T declares kHasPrefetch = true (see the file comment).
template <typename T, typename = void>
struct megakernel_phase_has_prefetch : std::false_type
{
};
template <typename T>
struct megakernel_phase_has_prefetch<T, std::void_t<decltype(T::kHasPrefetch)>>
    : std::bool_constant<T::kHasPrefetch>
{
};

/// One phase of a megakernel and what it waits for.
template <typename Phase_, typename Dependency_ = no_dependency>
struct megakernel_stage
{
    using Phase      = remove_cvref_t<Phase_>;
    using Dependency = Dependency_;
};

template <typename... Stages>
struct Megakernel
{
    static constexpr index_t kNumStages = sizeof...(Stages);
    static_assert(kNumStages > 0, "a megakernel needs at least one stage");

    template <index_t I>
    using stage_t = std::tuple_element_t<I, std::tuple<Stages...>>;
    template <index_t I>
    using phase_t = typename stage_t<I>::Phase;

    using ArgsTuple = tuple<typename Stages::Phase::Args...>;

    static constexpr index_t kBlockSize = phase_t<0>::kBlockSize;
    static_assert(((Stages::Phase::kBlockSize == kBlockSize) && ...),
                  "every phase must be built for the same workgroup size");

    // kBlockSize follows the compile pass: the host pass sees 64-lane waves, gfx1250 runs 32, so the
    // launch size comes from the wave count, which both passes agree on.
    static constexpr index_t kNumWaves = kBlockSize / get_warp_size();

    // At least one byte: a zero-size __shared__ array does not compile.
    static constexpr index_t kSmemSize = max(index_t{1}, Stages::Phase::GetSmemSize()...);

    /// Words at the start of the counter buffer: [0] last finished epoch, [1] workgroups exited.
    static constexpr index_t kStateWords = 2;

    struct Kargs
    {
        ArgsTuple args;
        uint32_t* counters;
        index_t slot_offset[kNumStages];
    };

    private:
    template <index_t I>
    static constexpr bool kHasDependency = stage_t<I>::Dependency::kProducer >= 0;

    // Stage I signals its counters only if a later stage waits on it.
    template <index_t I>
    static constexpr bool kIsProducer = []() {
        bool r = false;
        static_for<I + 1, kNumStages, 1>{}(
            [&](auto j) { r = r || stage_t<j.value>::Dependency::kProducer == I; });
        return r;
    }();

    template <index_t I>
    static constexpr void check_dependency()
    {
        constexpr index_t P = stage_t<I>::Dependency::kProducer;
        static_assert(P < I, "a stage may only wait on an earlier stage");
    }

    // A kernel arms at most one named barrier pipeline, so every phase that uses one must agree.
    template <typename T>
    static constexpr bool has_barriers = !std::is_void_v<typename T::barrier_pipeline>;

    static constexpr index_t kFirstBarrierStage = []() {
        index_t r = -1;
        static_for<0, kNumStages, 1>{}([&](auto i) {
            if(r < 0 && has_barriers<phase_t<i.value>>)
                r = i.value;
        });
        return r;
    }();

    struct no_barriers
    {
    };

    CK_TILE_DEVICE static auto InitBarriers()
    {
        if constexpr(kFirstBarrierStage < 0)
            return no_barriers{};
        else
            return phase_t<kFirstBarrierStage>::InitBarriers();
    }

    template <index_t I>
    CK_TILE_DEVICE static void WaitDependency(const Kargs& kargs, index_t work, uint32_t epoch)
    {
        using Dep            = typename stage_t<I>::Dependency;
        constexpr index_t P  = Dep::kProducer;
        using Producer       = phase_t<P>;
        const auto& p_args   = kargs.args.at(number<P>{});
        const uint32_t* base = kargs.counters + kargs.slot_offset[P];

        index_t s0 = 0;
        index_t s1 = Producer::NumSlots(p_args);
        if constexpr(Dep::kRows)
        {
            const auto [r0, r1] = phase_t<I>::RowRange(kargs.args.at(number<I>{}), work);
            s0                  = r0 / Producer::kSlotRows;
            s1 = min(s1, integer_divide_ceil(static_cast<index_t>(r1), Producer::kSlotRows));
        }
        if(megakernel_counter::is_poll_wave())
        {
            for(index_t s = s0; s < s1; ++s)
            {
                const uint32_t items = static_cast<uint32_t>(Producer::ItemsPerSlot(p_args, s));
                megakernel_counter::poll(base + s, epoch * items);
            }
        }
        megakernel_counter::acquire();
    }

    template <index_t I, typename Token>
    CK_TILE_DEVICE static void
    RunStage(const Kargs& kargs, void* smem, uint32_t epoch, const Token& barriers)
    {
        using Phase        = phase_t<I>;
        const auto& args   = kargs.args.at(number<I>{});
        const index_t num  = amd_wave_read_first_lane(Phase::NumWork(args));
        const index_t grid = amd_wave_read_first_lane(static_cast<index_t>(gridDim.x));

        for(index_t w = amd_wave_read_first_lane(static_cast<index_t>(blockIdx.x)); w < num;
            w += grid)
        {
            // Orders the previous item's (or phase's) LDS reads before this item's writes.
            s_waitcnt_barrier();
            // Loads that need nothing from earlier phases start before the wait.
            if constexpr(megakernel_phase_has_prefetch<Phase>::value)
            {
                Phase::Prefetch(args, w, smem);
            }
            if constexpr(kHasDependency<I>)
            {
                check_dependency<I>();
                WaitDependency<I>(kargs, w, epoch);
            }

            if constexpr(has_barriers<Phase>)
            {
                static_assert(std::is_same_v<Token, decltype(Phase::InitBarriers())>,
                              "phases arm different named barrier pipelines");
                Phase::Run(args, w, smem, barriers);
            }
            else
            {
                Phase::Run(args, w, smem, no_barriers{});
            }

            if constexpr(kIsProducer<I>)
            {
                megakernel_counter::signal(kargs.counters + kargs.slot_offset[I] +
                                           Phase::SlotOf(args, w));
            }
        }
    }

    public:
    /// Threads per workgroup on the device we launch on (wave32 on gfx1250).
    CK_TILE_HOST static dim3 BlockSize()
    {
        return dim3(kNumWaves * (is_wave32() ? 32 : 64));
    }

    CK_TILE_HOST static index_t NumCounters(const ArgsTuple& args)
    {
        index_t n = kStateWords;
        static_for<0, kNumStages, 1>{}(
            [&](auto i) { n += phase_t<i.value>::NumSlots(args.at(number<i.value>{})); });
        return n;
    }

    /// @param counters NumCounters(args) words, zeroed once and reused by every launch.
    CK_TILE_HOST static Kargs MakeKargs(const ArgsTuple& args, uint32_t* counters)
    {
        Kargs k{args, counters, {}};
        index_t off = kStateWords;
        static_for<0, kNumStages, 1>{}([&](auto i) {
            k.slot_offset[i.value] = off;
            off += phase_t<i.value>::NumSlots(args.at(number<i.value>{}));
        });
        return k;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Kargs& kargs)
    {
        bool ok = kargs.counters != nullptr;
        static_for<0, kNumStages, 1>{}([&](auto i) {
            ok = ok && phase_t<i.value>::IsSupportedArgument(kargs.args.at(number<i.value>{}));
        });
        return ok;
    }

    /// The largest grid whose workgroups are all resident at once; launch at most this many.
    CK_TILE_HOST static index_t MaxResidentGridSize(const stream_config& s)
    {
        const auto kernel = kentry<1, Megakernel, Kargs>;
        int occupancy     = 0;
        hip_check_error(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, BlockSize().x, 0));
        return get_available_compute_units(s) * occupancy;
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // 16-byte aligned: phases use 64-bit LDS atomics and vector LDS accesses on the arena.
        __shared__ alignas(16) char smem[kSmemSize];

        // The previous launch on this stream has finished, so its epoch store is visible.
        uint32_t* state      = kargs.counters;
        const uint32_t epoch = megakernel_counter::load(state) + 1;

        const auto barriers = InitBarriers();
        static_for<0, kNumStages, 1>{}(
            [&](auto i) { RunStage<i.value>(kargs, smem, epoch, barriers); });

        // The last workgroup out publishes the epoch for the next launch.
        __syncthreads();
        if(threadIdx.x == 0)
        {
            const uint32_t grid = gridDim.x * gridDim.y * gridDim.z;
            const uint32_t prev =
                __scoped_atomic_fetch_add(state + 1, 1u, __ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE);
            if(prev == grid - 1)
            {
                __scoped_atomic_store_n(state + 1, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
                __scoped_atomic_store_n(state, epoch, __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
            }
        }
    }
};

} // namespace ck_tile
