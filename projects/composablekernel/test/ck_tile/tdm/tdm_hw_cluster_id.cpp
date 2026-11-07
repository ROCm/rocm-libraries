// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Reads get_hw_cluster_id() in every wave of a plain launch and of a launch of 1x1x1 clusters,
// where it must be 0, and of a launch of 2x1x1 clusters, where it must be non-zero and the same in
// both workgroups of each cluster. TDMWorkgroupMask::checked allows a non-zero workgroup mask only
// where this value is non-zero. The kernel issues no TDM transfer and builds no mask.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_config.hpp"

namespace ck_tile {
namespace test {
namespace {

constexpr int kWaveSize      = 32; // gfx125 runs wave32
constexpr int kWavesPerBlock = 2;
// The cluster launches group the 1D grid into 2x1x1 clusters: blocks 2c and 2c + 1 form cluster c.
constexpr unsigned kClusterSize = 2;
// Marks an entry that no wave wrote: get_hw_cluster_id() returns a 4-bit value.
constexpr uint32_t kUnwritten = 0xFFFFFFFFu;

// Lane 0 of every wave writes the wave's get_hw_cluster_id() to out[block * kWavesPerBlock + wave].
struct ReadHwClusterIdKernel
{
    static constexpr index_t kBlockSize = kWavesPerBlock * kWaveSize;

    // Only the device pass odr-uses this; see the note in tdm_descriptor_stride.cpp.
    [[maybe_unused]] CK_TILE_DEVICE void operator()(uint32_t* out) const
    {
        const uint32_t cluster_id = get_hw_cluster_id();
        if(get_lane_id() == 0)
            out[get_block_id() * kWavesPerBlock + get_warp_id()] = cluster_id;
    }
};

class TDMHwClusterId : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        if(!is_gfx125_supported())
            GTEST_SKIP() << "get_hw_cluster_id() reads a gfx125 register";
    }

    static bool supports_cluster_launch() { return get_device_revision() != 0; }

    // Two workgroups per compute unit, at least 32: the launches spread over the device, and a
    // 2x1x1 launch has more clusters than the 15 non-zero values of the 4-bit field. The count is
    // even, so the 2x1x1 launch has whole clusters.
    static int num_blocks() { return 2 * std::max(16, static_cast<int>(get_num_cus())); }

    // One get_hw_cluster_id() value per wave, in block order. launch(out) throws if the launch
    // fails.
    template <typename Launch>
    static std::vector<uint32_t> read_cluster_ids(Launch&& launch)
    {
        const int num_waves = num_blocks() * kWavesPerBlock;
        DeviceMem out(num_waves * sizeof(uint32_t));
        const std::vector<uint32_t> unwritten(num_waves, kUnwritten);
        out.ToDevice(unwritten.data());

        launch(static_cast<uint32_t*>(out.GetDeviceBuffer()));
        HIP_CHECK_ERROR(hipDeviceSynchronize());

        std::vector<uint32_t> ids(num_waves);
        out.FromDevice(ids.data());
        return ids;
    }

    // read_cluster_ids() of a launch with clusters of cluster_size workgroups along x.
    static std::vector<uint32_t> read_cluster_launch_ids(unsigned cluster_size)
    {
        return read_cluster_ids([cluster_size](uint32_t* out) {
            const auto launch       = make_kernel(ReadHwClusterIdKernel{},
                                            dim3(cluster_size, 1, 1),
                                            dim3(num_blocks()),
                                            dim3(ReadHwClusterIdKernel::kBlockSize),
                                            std::size_t{0},
                                            out);
            const hipError_t status = launch(stream_config{});
            // A failed launch also stays as HIP's last error; clear it so that a later
            // launch_and_check in this process does not report it.
            static_cast<void>(hipGetLastError());
            HIP_CHECK_ERROR(status);
        });
    }

    // A zero read must hold also after cluster waves have run, so the zero tests run a 2x1x1
    // launch first where cluster launch is supported.
    static void run_cluster_launch_first()
    {
        if(supports_cluster_launch())
            static_cast<void>(read_cluster_launch_ids(kClusterSize));
    }

    // Names readback slot w in a failure message.
    static std::string wave_name(int w)
    {
        return "block " + std::to_string(w / kWavesPerBlock) + ", wave " +
               std::to_string(w % kWavesPerBlock);
    }
};

TEST_F(TDMHwClusterId, PlainLaunchReadsZero)
{
    run_cluster_launch_first();
    const std::vector<uint32_t> ids = read_cluster_ids([](uint32_t* out) {
        launch_and_check(stream_config{},
                         make_kernel(ReadHwClusterIdKernel{},
                                     dim3(num_blocks()),
                                     dim3(ReadHwClusterIdKernel::kBlockSize),
                                     std::size_t{0},
                                     out));
    });
    for(int w = 0; w < static_cast<int>(ids.size()); ++w)
        EXPECT_EQ(ids[w], 0u) << wave_name(w);
}

// A 1x1x1 cluster is dispatched without clusters.
TEST_F(TDMHwClusterId, SingleWorkgroupClusterLaunchReadsZero)
{
    if(!supports_cluster_launch())
        GTEST_SKIP() << "Cluster launch is not supported on gfx1250 asicRevision=0";

    run_cluster_launch_first();
    const std::vector<uint32_t> ids = read_cluster_launch_ids(1);
    for(int w = 0; w < static_cast<int>(ids.size()); ++w)
        EXPECT_EQ(ids[w], 0u) << wave_name(w);
}

TEST_F(TDMHwClusterId, ClusterLaunchReadsNonZero)
{
    if(!supports_cluster_launch())
        GTEST_SKIP() << "Cluster launch is not supported on gfx1250 asicRevision=0";

    const std::vector<uint32_t> ids = read_cluster_launch_ids(kClusterSize);
    for(int w = 0; w < static_cast<int>(ids.size()); ++w)
    {
        EXPECT_GE(ids[w], 1u) << wave_name(w);
        EXPECT_LE(ids[w], 15u) << wave_name(w);
        // Every wave of a cluster reads the value of the cluster's first wave.
        const int block                 = w / kWavesPerBlock;
        const int first_wave_of_cluster = block / kClusterSize * kClusterSize * kWavesPerBlock;
        EXPECT_EQ(ids[w], ids[first_wave_of_cluster]) << wave_name(w);
    }
}

} // namespace
} // namespace test
} // namespace ck_tile
