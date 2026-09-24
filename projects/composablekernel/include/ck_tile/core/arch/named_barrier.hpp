// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/ignore.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

// Hardware named (split) barriers, gfx1250 and later.
//
// A named barrier lets a subset of a workgroup's waves rendezvous without dragging in the
// other waves, which is what separates a producer/consumer pipeline from a ping-pong one:
// the loader waves can run ahead instead of meeting the compute waves at a workgroup
// barrier every K step.
//
// ABI, as emitted by the gfx1250 backend (established by compile probe, not by published
// documentation -- see the UNVERIFIED note on named_barrier_signal before relying on it):
//   - The compiler assigns ids, not the programmer: within a kernel, the i'th element of a
//     named barrier array is id i+1. Numbering restarts per kernel, so two kernels may each
//     size their array independently.
//   - S_BARRIER_INIT takes m0 = (member_count << 16) | id; S_BARRIER_JOIN takes the bare id in
//     m0's low bits. Both are materialised by the backend from the barrier object.
//   - S_BARRIER_SIGNAL and S_BARRIER_WAIT encode the id as a literal, so ids must be
//     compile-time constants and cannot be recovered from the object.
//   - Named barriers are a separate hardware pool and cost no group-segment bytes. A kernel
//     reserves ceil(extent / 4) groups, driven by the array's EXTENT, not by how many
//     barriers it actually uses -- so the array is sized to the pipeline, not to the
//     hardware maximum.
//
// Why a kernel may hold only one array. Ids are handed out per array in allocation order,
// which the front end cannot observe, so a second array leaves S_BARRIER_SIGNAL/WAIT pointing
// at the first array's ids while INIT/JOIN follow the second's -- a silent hang plus
// corruption of the other array's barriers. No static_assert can see the allocation, so the
// rule is enforced through the type system instead: every ring operation requires a token that
// only named_barrier_pipeline::init() can mint, and init() only accepts a kernel that names
// this pipeline via
//
//     using barrier_pipeline = ck_tile::named_barrier_pipeline<...>;
//
// So a kernel cannot signal barriers it never reserved, and cannot drive a second pipeline it
// never armed. The unit is the kernel because that is the hardware's unit -- two kernels may
// each size their array independently.
//
// That defends ck_tile's own uses. A __shared__ __amdgpu_named_workgroup_barrier_t[] declared
// anywhere else in the same kernel reintroduces the defect exactly, and nothing can catch it.
//
// Deliberately NOT ck_tile::static_counter: a counter allocates bases in instantiation order,
// so the same pipeline type would get different ids in different TUs -- an ODR violation on its
// own members, and strictly worse than a collision that at least reproduces.
//
// Usage. Consumer waves first, then one producer wave, over a ring of kNumSlots LDS buffers:
//
//     template <typename Pipe>
//     struct my_kernel
//     {
//         using barrier_pipeline = Pipe;                 // required: names this kernel's pipeline
//         using ring             = typename Pipe::template ring<0>;
//
//         CK_TILE_DEVICE void operator()(index_t num_steps, ...) const
//         {
//             if constexpr(ring::kIsSupported)           // must be template context to discard
//             {
//                 __shared__ T slots[kNumSlots][kElems]; // barriers cost no LDS; slots do
//                 const auto bar = Pipe::template init<my_kernel>();   // collective, once
//
//                 if(get_warp_id() < kNumConsumerWaves)  // wave-uniform role selection
//                 {
//                     ring::consumer::run(bar, num_steps, [&](auto slot, index_t step) {
//                         consume(slots[slot], step);
//                         __threadfence_block();         // retire the reads before release
//                     });
//                 }
//                 else
//                 {
//                     ring::template producer<0>::run(
//                         bar,
//                         num_steps,
//                         [&](auto slot, index_t step) { fill(slots[slot], step); },
//                         [](auto) { __threadfence_block(); });  // retire the stores
//                 }
//             }
//         }
//     };
//
// run() walks every wave through one shared schedule, so the rules a hand-written loop must
// keep -- prime every slot before the steady state, never leave the loop early, run the same
// number of steps on both sides -- hold by construction. Two stay with the caller, each a hang
// if broken: every wave passes the same wave-uniform num_steps, and exactly one wave plays each
// producer<P>, which is a role, not an identity. The lower-level prime/wait/publish/release
// remain for code that must interleave the handshake with other work; that code owns all of
// the rules above.

namespace ck_tile {

/** Conservative policy cap on named barrier ids. NOT a verified hardware limit: this toolchain
 * emits ids past 15 (the m0 id field is 6 bits) and reserves up to 7 groups.
 */
inline constexpr index_t kNumNamedBarriers  = 16;
inline constexpr index_t kMaxNamedBarrierId = kNumNamedBarriers - 1;

/// .amdhsa_named_barrier_count counts groups of 4 and saturates at 7. At 8 groups it wraps to
/// 0 with no diagnostic, and the kernel then signals barriers it never reserved.
inline constexpr index_t kMaxNamedBarrierGroups = 7;

/** True only in a gfx125x DEVICE pass.
 *
 * It is false in the host pass even when building for gfx1250, because __gfx125__ is defined
 * only in the matching device pass. Use it to guard device code in template context; host-side
 * dispatch must use ck_tile::is_gfx125_supported() instead.
 */
inline constexpr bool kNamedBarriersSupported =
#if defined(__gfx125__)
    true;
#else
    false;
#endif

#if defined(__gfx125__)
/// A hardware named barrier. Opaque: the program never reads or writes it.
using named_barrier = __amdgpu_named_workgroup_barrier_t;
#else
/** Complete so it can appear in signatures, uninstantiable so it can never reach LDS. A
 * hand-rolled barrier object compiles and runs the same instructions but leaves
 * .amdhsa_named_barrier_count at 0, so the workgroup signals barriers it never reserved.
 */
struct named_barrier
{
    named_barrier() = delete;
};
#endif

namespace impl {

// Reads Kernel::barrier_pipeline, yielding void when the kernel does not declare one so that
// init() can say so plainly instead of failing substitution.
template <typename Kernel, typename = void>
struct kernel_barrier_pipeline
{
    using type = void;
};

template <typename Kernel>
struct kernel_barrier_pipeline<Kernel, std::void_t<typename Kernel::barrier_pipeline>>
{
    using type = typename Kernel::barrier_pipeline;
};

} // namespace impl

namespace impl {
/** The kernel's named barrier array, sized to the pipeline that owns it.
 *
 * Keyed on the PIPELINE, not on the extent: two pipelines that happen to need the same number
 * of barriers would otherwise share one array and both number from id 1, which aliases their
 * handshakes with nothing left to detect. Pipeline is also what ties a pool to the token that
 * proves init() ran. T is a parameter only so the __shared__ declaration is dependent and
 * therefore instantiated on use -- off gfx1250 named_barrier is uninstantiable by design.
 */
template <typename Pipeline, typename T = named_barrier>
CK_TILE_DEVICE T (&named_barrier_pool())[Pipeline::kNumBarriers]
{
    static_assert(std::is_same_v<T, named_barrier>, "the pool element type is not a knob");
    __shared__ T pool[Pipeline::kNumBarriers];
    return pool;
}

// Every primitive takes the pool by reference so Id is checked against its extent.
//
// Provisioning comes from any op that consumes the barrier OBJECT -- init or join -- not from
// signal/wait, which carry only a literal. A kernel that solely signals reserves nothing. Every
// handshake here joins, so this is sound; it is recorded because it is not obvious.

template <index_t Id, index_t N>
CK_TILE_DEVICE void named_barrier_init(named_barrier (&pool)[N], uint32_t member_count)
{
    static_assert(1 <= Id && Id <= N, "named barrier id outside the pool backing it");
    ignore = pool;
#if defined(__gfx125__)
    __builtin_amdgcn_s_barrier_init(&pool[Id - 1], member_count);
#else
    ignore = member_count;
    static_assert(always_false_v<number<Id>>, "named barriers require gfx1250 or later");
#endif
}

// Register this wave as a member of the current generation.
template <index_t Id, index_t N>
CK_TILE_DEVICE void named_barrier_join(named_barrier (&pool)[N])
{
    static_assert(1 <= Id && Id <= N, "named barrier id outside the pool backing it");
    ignore = pool;
#if defined(__gfx125__)
    __builtin_amdgcn_s_barrier_join(&pool[Id - 1]);
#else
    static_assert(always_false_v<number<Id>>, "named barriers require gfx1250 or later");
#endif
}

/* Arrive without blocking. This is the run-ahead side of the handshake.
 *
 * UNVERIFIED, and load-bearing for every handshake in this header: that a signal from a wave
 * which never joined still counts toward the generation's member_count. Both the DATA and the
 * FREE protocols rely on it. It matches observed codegen but is not confirmed by published
 * gfx1250 documentation; if it is false the rings deadlock universally, not marginally.
 */
template <index_t Id, index_t N>
CK_TILE_DEVICE void named_barrier_signal(named_barrier (&pool)[N])
{
    static_assert(1 <= Id && Id <= N, "named barrier id outside the pool backing it");
    ignore = pool;
#if defined(__gfx125__)
    __builtin_amdgcn_s_barrier_signal(Id);
#else
    static_assert(always_false_v<number<Id>>, "named barriers require gfx1250 or later");
#endif
}

// Block until the current generation completes.
template <index_t Id, index_t N>
CK_TILE_DEVICE void named_barrier_wait(named_barrier (&pool)[N])
{
    static_assert(1 <= Id && Id <= N, "named barrier id outside the pool backing it");
    ignore = pool;
#if defined(__gfx125__)
    __builtin_amdgcn_s_barrier_wait(Id);
#else
    static_assert(always_false_v<number<Id>>, "named barriers require gfx1250 or later");
#endif
}

/* Join, arrive, then block. Joining first is what makes the wait safe: a wave must join a
 * barrier before that generation completes, or it misses the completion broadcast and blocks
 * forever.
 */
template <index_t Id, index_t N>
CK_TILE_DEVICE void named_barrier_arrive_and_wait(named_barrier (&pool)[N])
{
    named_barrier_join<Id>(pool);
    named_barrier_signal<Id>(pool);
    named_barrier_wait<Id>(pool);
}

} // namespace impl

/// Shape of one producer/consumer ring. Carries no storage and no ids.
template <index_t NumSlots, index_t NumProducerWaves, index_t NumConsumerWaves>
struct ring_spec
{
    static_assert(NumSlots >= 2, "a ring of one slot cannot decouple anything");
    static_assert(NumProducerWaves >= 1, "the ring needs at least one producer wave");
    static_assert(NumConsumerWaves >= 1, "the ring needs at least one consumer wave");

    static constexpr index_t kNumSlots         = NumSlots;
    static constexpr index_t kNumProducerWaves = NumProducerWaves;
    static constexpr index_t kNumConsumerWaves = NumConsumerWaves;
    static constexpr index_t kNumBarriers      = NumSlots * (1 + NumProducerWaves);
};

template <typename... Specs>
struct named_barrier_pipeline;

namespace impl {

/** How a ring walks its K steps: the one definition both of its sides share.
 *
 * A ring hangs when its sides disagree on how many steps there are or on which slot a step
 * uses, so neither side spells its loop out; both walk this schedule with the same count. The
 * slot reaches the body as a compile-time constant because S_BARRIER_SIGNAL/WAIT encode the id
 * as a literal. The count may be anything, zero and ragged tails included, but must be
 * wave-uniform: whole trips round the ring run unguarded, and only the tail branches.
 */
template <index_t NumSlots>
struct ring_schedule
{
    /// Calls body(number<Slot>{}, step) for every step in [0, num_steps), in order.
    template <typename Body>
    CK_TILE_HOST_DEVICE static void for_each_step(index_t num_steps, Body&& body)
    {
        index_t first_step = 0;
        for(; first_step + NumSlots <= num_steps; first_step += NumSlots)
        {
            static_for<0, NumSlots, 1>{}([&](auto slot) { body(slot, first_step + slot.value); });
        }
        static_for<0, NumSlots, 1>{}([&](auto slot) {
            if(first_step + slot.value < num_steps)
            {
                body(slot, first_step + slot.value);
            }
        });
    }
};

// Publishes a slot known only at run time. The slot must be wave-uniform.
template <typename Ops, index_t NumSlots, typename Token>
CK_TILE_HOST_DEVICE void publish_slot(Token token, index_t slot)
{
    static_for<0, NumSlots, 1>{}([&](auto s) {
        if(s.value == slot)
        {
            Ops::template publish<s.value>(token);
        }
    });
}

// The producer side of the protocol. Generic over the role operations only so that a host test
// can replay it against a model of the barrier semantics; kernels reach it through run().
//
// fill(slot, step) starts step's transfer into slot; retire(number<N>{}) returns once at most N
// transfers are still in flight, oldest retiring first. Step k publishes step k - Lag, which
// keeps Lag + 1 asynchronous transfers overlapping the consumers' math. Lag must stay below
// NumSlots: otherwise a producer waits for the drain of a step it has not published.
template <typename Ops,
          index_t NumSlots,
          index_t Lag,
          typename Token,
          typename Fill,
          typename Retire>
CK_TILE_HOST_DEVICE void
drive_producer(Token token, index_t num_steps, Fill&& fill, Retire&& retire)
{
    static_assert(0 <= Lag && Lag < NumSlots,
                  "a producer can leave at most NumSlots - 1 filled slots unpublished");

    ring_schedule<NumSlots>::for_each_step(num_steps, [&](auto slot, index_t step) {
        constexpr index_t kSlot = decltype(slot)::value;
        // A slot's first use has no drain to wait for.
        if(step >= NumSlots)
        {
            Ops::template wait<kSlot>(token);
        }
        fill(slot, step);
        if(Lag == 0 || step >= Lag)
        {
            retire(number<Lag>{});
            Ops::template publish<(kSlot + NumSlots - Lag) % NumSlots>(token);
        }
    });

    // The last Lag fills are still unpublished; publish them oldest first.
    if constexpr(Lag > 0)
    {
        retire(number<0>{});
        static_for<0, Lag, 1>{}([&](auto i) {
            const index_t step = num_steps - Lag + i.value;
            if(step >= 0)
            {
                publish_slot<Ops, NumSlots>(token, step % NumSlots);
            }
        });
    }

    // Answer each used slot's final release, so every FREE generation completes and the ring
    // ends at rest: reusable without re-arming, never holding signals nobody will wait on.
    static_for<0, NumSlots, 1>{}([&](auto slot) {
        if(slot.value < num_steps)
        {
            Ops::template wait<decltype(slot)::value>(token);
        }
    });
}

// The consumer side, generic for the same reason. drain(slot, step) consumes step from slot and
// must retire its reads of the slot before returning: the slot is released straight after.
template <typename Ops, index_t NumSlots, typename Token, typename Drain>
CK_TILE_HOST_DEVICE void drive_consumer(Token token, index_t num_steps, Drain&& drain)
{
    ring_schedule<NumSlots>::for_each_step(num_steps, [&](auto slot, index_t step) {
        constexpr index_t kSlot = decltype(slot)::value;
        Ops::template wait<kSlot>(token);
        drain(slot, step);
        Ops::template release<kSlot>(token);
    });
}

/** Per-slot producer/consumer handshake over a ring of LDS buffers.
 *
 * Decouples loader waves from compute waves: a producer blocks only when the slot it is about
 * to overwrite has not been drained, and a consumer blocks only when the slot it is about to
 * read has not been filled. With NumSlots buffers the producers run up to NumSlots steps
 * ahead, so their DMA overlaps the consumers' math instead of meeting it at a workgroup
 * barrier every step.
 *
 * Reachable only as named_barrier_pipeline::ring<I>, which is what guarantees the BaseId
 * values tile the pool without overlapping.
 *
 * Barrier layout, ids ascending from BaseId:
 *   DATA[s]      producers signal, consumers wait      -- slot s is filled
 *   FREE[p][s]   consumers signal, producer p waits    -- slot s is drained
 *
 * FREE is per producer rather than shared. With one shared FREE per slot, a consumer signal
 * can complete the generation on behalf of a producer that is running behind, releasing
 * whichever producer happened to be joined and hanging the others.
 *
 * Member counts are chosen so the waiting side is always a required signaller, which is what
 * rules out a wave waiting on a generation it never joined.
 *
 * Required call sequence. A producer must fill each slot's first generation with prime(), not
 * wait(): on the first pass nothing has been drained yet, so a wait() would block on a FREE
 * generation the consumers cannot complete until that same producer has filled the slot.
 *
 *   producer:  [fill slot s; prime\<s\>() for every s]  then  { wait; fill; publish } per step
 *   consumer:  { wait; drain; release } per step
 *
 * producer::run() and consumer::run() follow this sequence for any step count.
 *
 * Neither side may leave the loop early. A wave that returns without signalling the
 * generations its peers are waiting on hangs them, and a wave cannot retroactively signal a
 * generation it has left. Ragged K must be handled by driving every wave through the same
 * number of steps. The converse -- consumer signals left unmatched on each FREE barrier after
 * the last step -- does not hang, but the ring cannot be reused without re-arming. run()
 * answers those final releases, so it leaves the ring at rest.
 *
 * Producer<P> is a role, not an identity: nothing binds P to the calling wave. Two waves that
 * both instantiate producer<0> leave FREE[1][*] with no waiter and hang. The caller owns that
 * mapping.
 *
 * Every call must be wave-uniform. S_BARRIER_* are scalar and execute regardless of EXEC, so a
 * call under a partially divergent branch signals from a wave that should not have.
 */
template <typename Pipeline, index_t BaseId, typename Spec>
struct named_barrier_slot_ring
{
    /// Proof that Pipeline::init() armed these barriers. Only the pipeline can mint one.
    using token = typename Pipeline::token;

    static constexpr bool kIsSupported = kNamedBarriersSupported;

    static constexpr index_t kNumSlots         = Spec::kNumSlots;
    static constexpr index_t kNumProducerWaves = Spec::kNumProducerWaves;
    static constexpr index_t kNumConsumerWaves = Spec::kNumConsumerWaves;
    static constexpr index_t kNumBarriers      = Spec::kNumBarriers;

    static_assert(BaseId >= 1, "the backend numbers barriers from 1");
    static_assert(BaseId + kNumBarriers - 1 <= Pipeline::kNumBarriers,
                  "the ring's id range runs past the end of its pool");

    static constexpr uint32_t kDataMemberCount =
        static_cast<uint32_t>(kNumProducerWaves + kNumConsumerWaves);
    static constexpr uint32_t kFreeMemberCount = static_cast<uint32_t>(1 + kNumConsumerWaves);

    /// Named barriers are a separate hardware pool, so the ring occupies no LDS.
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    // Exposed for verification, not for call sites; the role views below take the slot instead.
    template <index_t Slot>
    CK_TILE_HOST_DEVICE static constexpr index_t data_id()
    {
        static_assert(0 <= Slot && Slot < kNumSlots, "slot out of range");
        return BaseId + Slot;
    }

    template <index_t Producer, index_t Slot>
    CK_TILE_HOST_DEVICE static constexpr index_t free_id()
    {
        static_assert(0 <= Producer && Producer < kNumProducerWaves, "producer out of range");
        static_assert(0 <= Slot && Slot < kNumSlots, "slot out of range");
        return BaseId + kNumSlots + Producer * kNumSlots + Slot;
    }

    /// One producer wave's view. Binding Producer once is what keeps it off every call site.
    template <index_t Producer>
    struct producer
    {
        static_assert(0 <= Producer && Producer < kNumProducerWaves, "producer out of range");

        /** Fill a slot's first generation without waiting for a drain that cannot have
         * happened yet. Every slot must be primed once before the steady-state loop.
         */
        template <index_t Slot>
        CK_TILE_DEVICE static void prime(token)
        {
            named_barrier_signal<data_id<Slot>()>(named_barrier_pool<Pipeline>());
        }

        /// Block until the consumers have drained this slot.
        template <index_t Slot>
        CK_TILE_DEVICE static void wait(token)
        {
            named_barrier_arrive_and_wait<free_id<Producer, Slot>()>(
                named_barrier_pool<Pipeline>());
        }

        /** Announce that this slot is filled. Never blocks, which is what lets the producer run
         * ahead. The caller must have retired the slot's stores first; the ring issues no drain
         * of its own, and __threadfence_block() does not wait for async global-to-LDS DMA, so
         * an async loader must drain its own counter before publishing.
         */
        template <index_t Slot>
        CK_TILE_DEVICE static void publish(token)
        {
            named_barrier_signal<data_id<Slot>()>(named_barrier_pool<Pipeline>());
        }

        /** Drive this producer through a whole K loop: first fills, steady state, ragged tail,
         * the publishes the lag defers, and the waits that leave the ring at rest.
         *
         * @tparam Lag   Fills left in flight when one is published; below kNumSlots.
         * @param fill   fill(number<Slot>{}, step) starts step's transfer into Slot.
         * @param retire retire(number<N>{}) returns once at most N transfers are in flight.
         */
        template <index_t Lag = 0, typename Fill, typename Retire>
        CK_TILE_DEVICE static void
        run(token barriers, index_t num_steps, Fill&& fill, Retire&& retire)
        {
            drive_producer<producer, kNumSlots, Lag>(barriers, num_steps, fill, retire);
        }
    };

    /// The consumer waves' view.
    struct consumer
    {
        /// Block until every producer has filled this slot.
        template <index_t Slot>
        CK_TILE_DEVICE static void wait(token)
        {
            named_barrier_arrive_and_wait<data_id<Slot>()>(named_barrier_pool<Pipeline>());
        }

        /** Hand this slot back to every producer. Never blocks. The caller must have retired
         * the slot's loads first, or the producers race the reads.
         */
        template <index_t Slot>
        CK_TILE_DEVICE static void release(token)
        {
            static_for<0, kNumProducerWaves, 1>{}([](auto p) {
                named_barrier_signal<free_id<p.value, Slot>()>(named_barrier_pool<Pipeline>());
            });
        }

        /** Drive the consumers through a whole K loop, in lockstep with every producer's run().
         *
         * @param drain drain(number<Slot>{}, step) consumes step from Slot and retires its reads
         *              before returning; the slot goes back to the producers straight after.
         */
        template <typename Drain>
        CK_TILE_DEVICE static void run(token barriers, index_t num_steps, Drain&& drain)
        {
            drive_consumer<consumer, kNumSlots>(barriers, num_steps, drain);
        }
    };

    private:
    template <typename...>
    friend struct ck_tile::named_barrier_pipeline;

    // Arming a live barrier resets its generation under joined waves, so only the pipeline's
    // collective init() may reach this.
    CK_TILE_DEVICE static void arm()
    {
        static_for<0, kNumSlots, 1>{}([](auto s) {
            named_barrier_init<data_id<s.value>()>(named_barrier_pool<Pipeline>(),
                                                   kDataMemberCount);
            static_for<0, kNumProducerWaves, 1>{}([s](auto p) {
                named_barrier_init<free_id<p.value, s.value>()>(named_barrier_pool<Pipeline>(),
                                                                kFreeMemberCount);
            });
        });
    }
};

} // namespace impl

/** One or more rings tiling the kernel's named barrier pool.
 *
 * Bases are computed, never written, so adding or resizing a ring cannot silently collide with
 * another. Rings may differ in slot depth, which is the usual case when the operands they carry
 * differ in size. The pool is sized to exactly this pipeline, so a kernel reserves only the
 * barrier groups it uses.
 *
 * Put every ring in one pipeline: init() static_asserts that this is the pipeline the program
 * names via `using barrier_pipeline`, so a second one is a compile error rather than a hang.
 */
template <typename... Specs>
struct named_barrier_pipeline
{
    static_assert(sizeof...(Specs) >= 1, "a pipeline needs at least one ring");

    static constexpr bool kIsSupported    = kNamedBarriersSupported;
    static constexpr index_t kNumBarriers = (... + Specs::kNumBarriers);
    static constexpr index_t kNumRings    = sizeof...(Specs);

    static_assert(kNumBarriers <= kMaxNamedBarrierId,
                  "the pipeline needs more named barriers than the hardware has: reduce slots "
                  "or producer waves");
    static_assert((kNumBarriers + 3) / 4 <= kMaxNamedBarrierGroups,
                  "the pool would overflow amdhsa_named_barrier_count, which wraps silently");

    /// Named barriers cost no LDS; this exists so a consumer can aggregate it uniformly.
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    /** Proof that init() armed this pipeline's barriers in this kernel.
     *
     * Every ring operation takes one, and only init() can construct one, so a kernel cannot
     * signal barriers it never reserved and cannot drive a second pipeline it never armed --
     * the two ways the one-array-per-kernel rule used to be bypassable.
     */
    class token
    {
        friend struct named_barrier_pipeline;
        // User-provided, not `= default`: a defaulted ctor leaves this an aggregate in C++17,
        // and `token{}` would then bypass access control and forge one.
        CK_TILE_DEVICE token() {}
    };

    private:
    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr index_t base_id()
    {
        static_assert(0 <= I && I < kNumRings, "ring index out of range");
        constexpr index_t counts[] = {Specs::kNumBarriers...};
        index_t base               = 1;
        for(index_t i = 0; i < I; ++i)
        {
            base += counts[i];
        }
        return base;
    }

    public:
    template <index_t I>
    using ring = impl::named_barrier_slot_ring<named_barrier_pipeline,
                                               base_id<I>(),
                                               __type_pack_element<I, Specs...>>;

    /** Arm every ring and publish the result. Collective: call it from all waves, once, before
     * any of them touches a ring.
     *
     * Kernel is the calling kernel's own type, which must declare
     * `using barrier_pipeline = <this pipeline>;`. That is what pins one barrier array per
     * kernel: a second pipeline cannot also be the kernel's declared one. Passing it as a
     * template parameter also defers the lookup to the call site, where the kernel type is
     * complete.
     */
    template <typename Kernel>
    CK_TILE_DEVICE static token init()
    {
        static_assert(
            std::is_same_v<named_barrier_pipeline,
                           typename impl::kernel_barrier_pipeline<Kernel>::type>,
            "a kernel may hold only one named barrier array, so it may use only one pipeline: "
            "declare it in the kernel as `using barrier_pipeline = <this pipeline>;` and pass "
            "the kernel type to init()");

        // Workitems linearise x-fastest into waves, so the linear id selects exactly wave 0 for
        // any block shape. threadIdx.x alone does not: for a 2-D block it is true in every
        // wave, and for blockDim.x = 48 it is divergent *within* a wave -- and S_BARRIER_INIT
        // is scalar, so it arms regardless of EXEC.
        //
        // Elsewhere there is nothing to arm, and the token stays inert because no ring
        // operation compiles there; this is what lets kernels call init() on every target.
        if constexpr(kIsSupported)
        {
            const uint32_t linear_tid =
                threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
            if(linear_tid < static_cast<uint32_t>(get_warp_size()))
            {
                static_for<0, kNumRings, 1>{}([](auto i) { ring<i.value>::arm(); });
            }
        }
        __syncthreads();
        return token{};
    }
};

} // namespace ck_tile
