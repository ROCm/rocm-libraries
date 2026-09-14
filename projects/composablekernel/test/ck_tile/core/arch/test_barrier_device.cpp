// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core/arch/named_barrier.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace {

// gfx1250 is RDNA, so a wave is 32 lanes. The kernel maps one slot element per lane.
constexpr int kWaveSize = 32;
constexpr int kNumSlots = 2;
constexpr int kNumSteps = 8;

static_assert(kNumSteps % kNumSlots == 0,
              "the ring is driven kNumSlots steps at a time; a ragged tail leaves a generation "
              "unsignalled and hangs");

using ring_pipe = ck_tile::named_barrier_pipeline<ck_tile::ring_spec<kNumSlots, 1, 1>>;

// Never zero, so an unwritten output element and an undrained slot both stay distinguishable
// from a legitimately delivered value.
constexpr int32_t slot_value(int step, int lane) { return (step + 1) * 1000 + lane + 1; }

// One producer wave feeds one consumer wave through the ring. A broken DATA handshake lets a
// consumer read a slot before it is filled; a broken FREE handshake lets the producer
// overwrite a slot the consumer has not drained. The consumer records what it saw per step
// rather than accumulating, because a sum cannot distinguish a reordered ring from an ordered
// one and duplicate/skip pairs cancel out of it exactly.
//
// Templated on the pipeline so `if constexpr` genuinely discards off gfx1250 -- in a
// non-template the discarded branch is still fully checked. This is how a real consumer
// (a pipeline) would be written.
template <typename Pipe>
struct slot_ring_kernel
{
    // Names this kernel's one barrier pipeline. init() rejects any other, which is what keeps a
    // kernel to a single barrier array.
    using barrier_pipeline = Pipe;

    using ring = typename Pipe::template ring<0>;

    static constexpr ck_tile::index_t kBlockSize = 2 * kWaveSize;

    // Used only by the supported branch below.
    [[maybe_unused]] static constexpr int kNumIters = kNumSteps / kNumSlots;

    CK_TILE_DEVICE void operator()(int32_t* __restrict__ out) const
    {
        if constexpr(ring::kIsSupported)
        {
            // Dependent on Pipe, so it cannot fire from the discarded branch.
            static_assert(!ring::kIsSupported || ck_tile::get_warp_size() == kWaveSize,
                          "threadIdx.x / kWaveSize must be wave-uniform; under wave64 one wave "
                          "would run both sides of the handshake and deadlock");

            // The ring's barriers are a separate hardware pool, so this is all the LDS needed.
            __shared__ int32_t p_slots[kNumSlots * kWaveSize];

            const int lane    = static_cast<int>(threadIdx.x) % kWaveSize;
            const int wave_id = static_cast<int>(threadIdx.x) / kWaveSize;

            // The token proves init() ran; every ring call requires one.
            const auto bar = Pipe::template init<slot_ring_kernel>();

            if(wave_id == 0)
            {
                using prod = typename ring::template producer<0>;

                // Every slot must be primed rather than waited on. On the first pass nothing
                // has been drained, so a wait would block on a FREE generation the consumer
                // cannot complete until this producer has filled the slot.
                p_slots[0 * kWaveSize + lane] = slot_value(0, lane);
                __threadfence_block();
                prod::template prime<0>(bar);

                p_slots[1 * kWaveSize + lane] = slot_value(1, lane);
                __threadfence_block();
                prod::template prime<1>(bar);

                for(int iter = 1; iter < kNumIters; ++iter)
                {
                    prod::template wait<0>(bar);
                    p_slots[0 * kWaveSize + lane] = slot_value(iter * kNumSlots + 0, lane);
                    __threadfence_block();
                    prod::template publish<0>(bar);

                    prod::template wait<1>(bar);
                    p_slots[1 * kWaveSize + lane] = slot_value(iter * kNumSlots + 1, lane);
                    __threadfence_block();
                    prod::template publish<1>(bar);
                }
            }
            else
            {
                using cons = typename ring::consumer;

                for(int iter = 0; iter < kNumIters; ++iter)
                {
                    cons::template wait<0>(bar);
                    const int32_t v0 = p_slots[0 * kWaveSize + lane];
                    __threadfence_block();
                    cons::template release<0>(bar);
                    out[(iter * kNumSlots + 0) * kWaveSize + lane] = v0;

                    cons::template wait<1>(bar);
                    const int32_t v1 = p_slots[1 * kWaveSize + lane];
                    __threadfence_block();
                    cons::template release<1>(bar);
                    out[(iter * kNumSlots + 1) * kWaveSize + lane] = v1;
                }
            }
        }
        else
        {
            ck_tile::ignore = out;
        }
    }
};

} // namespace

TEST(NamedBarrierSlotRingDevice, ProducerConsumerHandshakeOrdersEveryStep)
{
    if(!ck_tile::is_gfx125_supported())
    {
        // CTest reports a skip as PASS, so on a runner that is meant to be gfx1250 this would
        // be indistinguishable from having run. Set CK_TILE_REQUIRE_GFX125=1 there to make it
        // a failure instead.
        const char* required = std::getenv("CK_TILE_REQUIRE_GFX125");
        if(required != nullptr && required[0] == '1')
        {
            FAIL() << "CK_TILE_REQUIRE_GFX125=1 but the device reports '"
                   << ck_tile::get_device_name() << "'";
        }
        GTEST_SKIP() << "hardware named barriers require gfx1250; device reports '"
                     << ck_tile::get_device_name() << "'";
    }

    constexpr int kOutElems = kNumSteps * kWaveSize;

    ck_tile::DeviceMem out_buf(kOutElems * sizeof(int32_t));
    out_buf.SetBytePattern(0xFF);

    ck_tile::launch_and_check(
        ck_tile::stream_config{},
        ck_tile::make_kernel(slot_ring_kernel<ring_pipe>{},
                             dim3(1),
                             dim3(2 * kWaveSize),
                             0,
                             static_cast<int32_t*>(out_buf.GetDeviceBuffer())));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess) << "kernel did not complete";

    std::vector<int32_t> out(kOutElems);
    out_buf.FromDevice(out.data());

    int mismatches = 0;
    int first      = -1;
    for(int step = 0; step < kNumSteps; ++step)
    {
        for(int lane = 0; lane < kWaveSize; ++lane)
        {
            const int i = step * kWaveSize + lane;
            if(out[i] != slot_value(step, lane))
            {
                ++mismatches;
                first = (first < 0) ? i : first;
            }
        }
    }

    ASSERT_EQ(mismatches, 0) << "first at step " << (first / kWaveSize) << " lane "
                             << (first % kWaveSize) << ": expected "
                             << slot_value(first / kWaveSize, first % kWaveSize) << ", got "
                             << out[first];
}
