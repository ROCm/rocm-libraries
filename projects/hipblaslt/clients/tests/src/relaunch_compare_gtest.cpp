// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for the device-side comparison in clients/common/include/relaunch_compare.hpp, which
// hipblaslt-bench --post_bench_verify_iters uses to compare each relaunch's D with the first
// launch's. They need a GPU, unlike the HostUnit tests.

#include <gtest/gtest.h>

#include "hipBuffer.hpp"
#include "relaunch_compare.hpp"

#include <hip/hip_runtime.h>

#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace
{
    constexpr unsigned long long kNone = std::numeric_limits<unsigned long long>::max();

    // A device copy of a host vector.
    template <typename T>
    struct DeviceCopy
    {
        HipDeviceBuffer buffer;

        explicit DeviceCopy(const std::vector<T>& host)
            : buffer(HIP_R_8I, host.size() * sizeof(T))
        {
            EXPECT_EQ(
                hipMemcpy(
                    buffer.buf(), host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
                hipSuccess);
        }

        void* data()
        {
            return buffer.buf();
        }
    };

    // Compares each (d, ref) pair into one slot, in order, the way the bench does for the gemms
    // or batches of one relaunch, and returns the slot.
    template <typename T>
    RelaunchSlot compare(std::vector<std::pair<std::vector<T>, std::vector<T>>> buffers)
    {
        HipDeviceBuffer slot(HIP_R_8I, sizeof(RelaunchSlot));
        auto*           dSlot = slot.as<RelaunchSlot>();
        EXPECT_EQ(relaunch_slots_reset_device(dSlot, 1, nullptr), hipSuccess);
        std::vector<DeviceCopy<T>> keep;
        keep.reserve(2 * buffers.size());
        uint64_t offset = 0;
        for(auto& [d, ref] : buffers)
        {
            keep.emplace_back(d);
            keep.emplace_back(ref);
            void* dd = keep[keep.size() - 2].data();
            void* dr = keep.back().data();
            EXPECT_EQ(relaunch_compare_device(dd, dr, d.size(), sizeof(T), offset, dSlot, nullptr),
                      hipSuccess);
            EXPECT_EQ(
                relaunch_capture_first_device(dd, d.size(), sizeof(T), offset, dSlot, nullptr),
                hipSuccess);
            offset += d.size();
        }
        RelaunchSlot host{};
        EXPECT_EQ(hipMemcpy(&host, dSlot, sizeof(host), hipMemcpyDeviceToHost), hipSuccess);
        return host;
    }

    template <typename T>
    T bits_as(const RelaunchSlot& slot)
    {
        T value;
        std::memcpy(&value, slot.bits, sizeof(T));
        return value;
    }

    TEST(RelaunchCompare, smoke_MatchingBuffersReportNoDifference)
    {
        std::vector<float> ref(1000, 1.5f);
        const auto         slot = compare<float>({{ref, ref}});
        EXPECT_EQ(slot.count, 0u);
        EXPECT_EQ(slot.first, kNone);
    }

    TEST(RelaunchCompare, smoke_CountsDifferencesAndCapturesFirst)
    {
        std::vector<float> ref(1000, 1.5f);
        std::vector<float> d = ref;
        d[700]               = 2;
        d[13]                = 3;
        d[999]               = 4;
        const auto slot      = compare<float>({{d, ref}});
        EXPECT_EQ(slot.count, 3u);
        EXPECT_EQ(slot.first, 13u);
        EXPECT_EQ(bits_as<float>(slot), 3.0f);
    }

    // The first difference is in the second buffer, so its index counts the 100 elements before.
    TEST(RelaunchCompare, smoke_IndexSpansSeveralBuffers)
    {
        std::vector<float> ref0(100, 1.0f), ref1(50, 2.0f);
        std::vector<float> d1 = ref1;
        d1[7]                 = 9;
        d1[30]                = 8;
        const auto slot       = compare<float>({{ref0, ref0}, {d1, ref1}});
        EXPECT_EQ(slot.count, 2u);
        EXPECT_EQ(slot.first, 107u);
        EXPECT_EQ(bits_as<float>(slot), 9.0f);
    }

    TEST(RelaunchCompare, smoke_EarlierBufferHoldsTheFirstDifference)
    {
        std::vector<float> ref0(100, 1.0f), ref1(50, 2.0f);
        std::vector<float> d0 = ref0, d1 = ref1;
        d0[60]          = 5;
        d1[3]           = 9;
        const auto slot = compare<float>({{d0, ref0}, {d1, ref1}});
        EXPECT_EQ(slot.count, 2u);
        EXPECT_EQ(slot.first, 60u);
        EXPECT_EQ(bits_as<float>(slot), 5.0f);
    }

    // A 16-byte element (complex double) differing only in its second half, the imaginary part.
    TEST(RelaunchCompare, smoke_SixteenByteElementsCompareBothHalves)
    {
        std::vector<std::complex<double>> ref(64, {1.0, 2.0});
        std::vector<std::complex<double>> d = ref;
        d[40]                               = {1.0, 2.5};
        const auto slot                     = compare<std::complex<double>>({{d, ref}});
        EXPECT_EQ(slot.count, 1u);
        EXPECT_EQ(slot.first, 40u);
        EXPECT_EQ(bits_as<std::complex<double>>(slot), std::complex<double>(1.0, 2.5));
    }

    TEST(RelaunchCompare, smoke_OneByteElements)
    {
        std::vector<int8_t> ref(33, 7);
        std::vector<int8_t> d = ref;
        d[32]                 = -7;
        const auto slot       = compare<int8_t>({{d, ref}});
        EXPECT_EQ(slot.count, 1u);
        EXPECT_EQ(slot.first, 32u);
        EXPECT_EQ(bits_as<int8_t>(slot), -7);
    }

    // More elements than one pass of the grid covers (4,096 blocks of 256 threads).
    TEST(RelaunchCompare, smoke_LastElementOfLargeBufferIsChecked)
    {
        std::vector<uint16_t> ref(3 * 4096 * 256 + 5, 0x3c00);
        std::vector<uint16_t> d = ref;
        d.back()                = 0x4000;
        const auto slot         = compare<uint16_t>({{d, ref}});
        EXPECT_EQ(slot.count, 1u);
        EXPECT_EQ(slot.first, ref.size() - 1);
        EXPECT_EQ(bits_as<uint16_t>(slot), 0x4000);
    }

    TEST(RelaunchCompare, smoke_UnsupportedElementSizeIsRefused)
    {
        EXPECT_EQ(relaunch_compare_device(nullptr, nullptr, 4, 3, 0, nullptr, nullptr),
                  hipErrorInvalidValue);
        EXPECT_EQ(relaunch_capture_first_device(nullptr, 4, 32, 0, nullptr, nullptr),
                  hipErrorInvalidValue);
    }
}
