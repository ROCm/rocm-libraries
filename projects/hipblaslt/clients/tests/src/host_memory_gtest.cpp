// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for how hipblaslt-test holds pinned host memory (AIHPBLAS-5352). The client pool keeps
// buffers that tests return; a large one left there stays pinned, and the kernel cannot
// reclaim it for the other test processes on the host. The _pre_checkin suffix puts the suite in
// the standard test tier. It pins at most about 3.3 GiB at a time and launches no kernels.

#include <gtest/gtest.h>

#include "fast_check.hpp"
#include "hipBuffer.hpp"
#include "host_memory.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace
{
    // What the HostPoolReleaseListener in hipblaslt-test does when a test ends.
    void end_test()
    {
        HostPoolReleaseListener().OnTestEnd(
            *::testing::UnitTest::GetInstance()->current_test_info());
    }

    const size_t kSmall = size_t(1) << 20;

    TEST(HostMemory_pre_checkin, large_buffer_released_after_a_test_that_does_not_reuse_it)
    {
        memory_pool<h_memory>::ReleaseCached();
        {
            HipHostBuffer large(HIP_R_8I, kHostPoolRetainLimit);
            HipHostBuffer small(HIP_R_8I, kSmall);
            ASSERT_EQ(large.memcheck(), hipSuccess);
            ASSERT_EQ(small.memcheck(), hipSuccess);
        }

        // The next test may have the same shape, so the large buffer stays for it.
        end_test();
        EXPECT_GE(memory_pool<h_memory>::CachedBytes(), kHostPoolRetainLimit + kSmall);

        // That test did not take it, so it is released; the small one stays.
        end_test();
        EXPECT_LT(memory_pool<h_memory>::CachedBytes(), kHostPoolRetainLimit);
        EXPECT_GE(memory_pool<h_memory>::CachedBytes(), kSmall);
    }

    TEST(HostMemory_pre_checkin, next_test_of_the_same_shape_reuses_the_large_buffer)
    {
        memory_pool<h_memory>::ReleaseCached();
        auto first = memory_pool<h_memory>::Get(kHostPoolRetainLimit);
        ASSERT_NE(first.get(), nullptr);
        first.get()[0]                        = 42;
        first.get()[kHostPoolRetainLimit - 1] = 43;
        memory_pool<h_memory>::Restore(first);
        end_test();
        EXPECT_EQ(memory_pool<h_memory>::CachedBytes(), kHostPoolRetainLimit);

        // Taken from the pool, not pinned anew: it still holds what the last test wrote.
        auto again = memory_pool<h_memory>::Get(kHostPoolRetainLimit);
        ASSERT_NE(again.get(), nullptr);
        EXPECT_EQ(memory_pool<h_memory>::CachedBytes(), 0u);
        EXPECT_EQ(again.get()[0], 42);
        EXPECT_EQ(again.get()[kHostPoolRetainLimit - 1], 43);
        memory_pool<h_memory>::Restore(again);
        memory_pool<h_memory>::ReleaseCached();
    }

    TEST(HostMemory_pre_checkin, large_miss_releases_what_earlier_tests_left)
    {
        // A test of another shape needs a new large buffer; the previous test's ones are
        // released before more memory is pinned, not after this test ends. On any miss the
        // pool already drops the largest buffer that is too small, so two are left here.
        memory_pool<h_memory>::ReleaseCached();
        const size_t step    = kHostPoolRetainLimit / 10;
        auto         earlier = memory_pool<h_memory>::Get(kHostPoolRetainLimit);
        auto         bigger  = memory_pool<h_memory>::Get(kHostPoolRetainLimit + step);
        ASSERT_NE(earlier.get(), nullptr);
        ASSERT_NE(bigger.get(), nullptr);
        memory_pool<h_memory>::Restore(earlier);
        memory_pool<h_memory>::Restore(bigger);
        end_test();
        ASSERT_GE(memory_pool<h_memory>::CachedBytes(), 2 * kHostPoolRetainLimit);

        auto larger = memory_pool<h_memory>::Get(kHostPoolRetainLimit + 2 * step);
        ASSERT_NE(larger.get(), nullptr);
        EXPECT_EQ(memory_pool<h_memory>::CachedBytes(), 0u);
        memory_pool<h_memory>::Restore(larger);
        memory_pool<h_memory>::ReleaseCached();
    }

    TEST(HostMemory_pre_checkin, buffers_at_the_retain_limit_get_no_slack)
    {
        // The pool adds 20% to small requests so that a slightly larger one can reuse them. A
        // large buffer is reused only by the same shape, so the extra would only add pinned
        // memory.
        memory_pool<h_memory>::ReleaseCached();
        auto large = memory_pool<h_memory>::Get(kHostPoolRetainLimit);
        ASSERT_NE(large.get(), nullptr);
        EXPECT_EQ(large.capacity(), kHostPoolRetainLimit);
        memory_pool<h_memory>::Restore(large);

        auto small = memory_pool<h_memory>::Get(kSmall);
        ASSERT_NE(small.get(), nullptr);
        EXPECT_GT(small.capacity(), small.bytes());
        memory_pool<h_memory>::Restore(small);
        memory_pool<h_memory>::ReleaseCached();
    }

    TEST(HostMemory_pre_checkin, failed_host_allocation_is_reported_not_used)
    {
        // Far beyond any host: the allocation fails and leaves a null buffer.
        std::vector<HipHostBuffer> ok, failed;
        ok.emplace_back(HIP_R_8I, 4096);
        failed.emplace_back(HIP_R_8I, size_t(1) << 60);
        ASSERT_EQ(ok[0].memcheck(), hipSuccess);
        EXPECT_EQ(failed[0].buf(), nullptr);
        EXPECT_EQ(failed[0].memcheck(), hipErrorOutOfMemory);

        EXPECT_EQ(host_allocation_failure({&ok}), "");
        std::string why = host_allocation_failure({&ok, &failed});
        EXPECT_NE(why.find("could not allocate 1.07e+09 GiB of pinned host memory"),
                  std::string::npos)
            << why;
    }

#ifndef _WIN32
    constexpr uint64_t kGiB = uint64_t(1) << 30;

    // A temporary stand-in for /proc/meminfo, /proc/self/cgroup and /sys/fs/cgroup.
    class FakeHost
    {
    public:
        FakeHost()
        {
            static int count = 0;
            dir              = std::filesystem::temp_directory_path()
                  / ("hipblaslt_host_memory_" + std::to_string(getpid()) + "_"
                     + std::to_string(count++));
            std::filesystem::create_directories(dir / "cgroup");
            sources.meminfo     = (dir / "meminfo").string();
            sources.self_cgroup = (dir / "self_cgroup").string();
            sources.cgroup_root = (dir / "cgroup").string();
        }
        ~FakeHost()
        {
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
        }

        void write(const std::filesystem::path& file, const std::string& text) const
        {
            std::filesystem::create_directories(file.parent_path());
            std::ofstream(file) << text;
        }
        void mem_available(uint64_t bytes) const
        {
            write(sources.meminfo,
                  "MemTotal:       999999999 kB\nMemFree:        1 kB\nMemAvailable:   "
                      + std::to_string(bytes / 1024) + " kB\n");
        }
        // One cgroup directory under the fake root, relative like "job/step".
        void cgroup(const std::string& rel,
                    const std::string& limit,
                    uint64_t           used,
                    uint64_t           inactive_file = 0,
                    bool               v1            = false) const
        {
            std::filesystem::path d = std::filesystem::path(sources.cgroup_root);
            if(v1)
                d /= "memory";
            d /= rel;
            write(d / (v1 ? "memory.limit_in_bytes" : "memory.max"), limit + "\n");
            write(d / (v1 ? "memory.usage_in_bytes" : "memory.current"),
                  std::to_string(used) + "\n");
            write(d / "memory.stat",
                  std::string(v1 ? "total_inactive_file " : "inactive_file ")
                      + std::to_string(inactive_file) + "\n");
        }

        std::filesystem::path dir;
        HostMemorySources     sources;
    };

    TEST(HostMemory_pre_checkin, query_reads_mem_available_without_a_cgroup)
    {
        FakeHost host;
        host.mem_available(8 * kGiB);
        EXPECT_EQ(available_host_memory_bytes(host.sources), 8 * kGiB);
    }

    TEST(HostMemory_pre_checkin, query_uses_a_cgroup_v2_limit_below_mem_available)
    {
        // 2 GiB limit, 1.5 GiB charged of which 0.25 GiB is page cache the kernel can drop.
        FakeHost host;
        host.mem_available(8 * kGiB);
        host.write(host.sources.self_cgroup, "0::/job\n");
        host.cgroup("job", std::to_string(2 * kGiB), 3 * kGiB / 2, kGiB / 4);
        EXPECT_EQ(available_host_memory_bytes(host.sources), 3 * kGiB / 4);
    }

    TEST(HostMemory_pre_checkin, query_ignores_a_cgroup_v2_without_a_limit)
    {
        FakeHost host;
        host.mem_available(8 * kGiB);
        host.write(host.sources.self_cgroup, "0::/job\n");
        host.cgroup("job", "max", 6 * kGiB);
        EXPECT_EQ(available_host_memory_bytes(host.sources), 8 * kGiB);
    }

    TEST(HostMemory_pre_checkin, query_uses_the_tightest_limit_up_to_the_root)
    {
        // The job's own limit leaves 3 GiB, its parent's 0.5 GiB, and the root's 1 GiB.
        FakeHost host;
        host.mem_available(8 * kGiB);
        host.write(host.sources.self_cgroup, "0::/runner/job\n");
        host.cgroup("runner/job", std::to_string(4 * kGiB), kGiB);
        host.cgroup("runner", std::to_string(2 * kGiB), 3 * kGiB / 2);
        host.cgroup("", std::to_string(6 * kGiB), 5 * kGiB);
        EXPECT_EQ(available_host_memory_bytes(host.sources), kGiB / 2);
    }

    TEST(HostMemory_pre_checkin, query_finds_a_container_limit_at_its_root)
    {
        // A container lists its cgroup by the host's path but sees only its own limit at the root.
        FakeHost host;
        host.mem_available(8 * kGiB);
        host.write(host.sources.self_cgroup, "0::/kubepods/pod1/ctr\n");
        host.cgroup("", std::to_string(3 * kGiB), kGiB);
        EXPECT_EQ(available_host_memory_bytes(host.sources), 2 * kGiB);
    }

    TEST(HostMemory_pre_checkin, query_reads_cgroup_v1)
    {
        FakeHost host;
        host.mem_available(8 * kGiB);
        host.write(host.sources.self_cgroup, "7:cpu,cpuacct:/job\n5:memory:/job\n");
        host.cgroup("job", std::to_string(3 * kGiB), kGiB, 0, true);
        EXPECT_EQ(available_host_memory_bytes(host.sources), 2 * kGiB);

        // v1 writes a value near 2^63 for no limit.
        host.cgroup("job", "9223372036854771712", kGiB, 0, true);
        EXPECT_EQ(available_host_memory_bytes(host.sources), 8 * kGiB);
    }

    TEST(HostMemory_pre_checkin, query_handles_missing_and_unexpected_files)
    {
        FakeHost host;
        EXPECT_EQ(available_host_memory_bytes(host.sources), 0u) << "no meminfo";

        host.mem_available(8 * kGiB);
        host.write(host.sources.self_cgroup, "0::/job\n");
        host.cgroup("job", "not-a-number", kGiB);
        EXPECT_EQ(available_host_memory_bytes(host.sources), 8 * kGiB) << "garbled limit";

        host.cgroup("job", std::to_string(kGiB), 2 * kGiB);
        EXPECT_EQ(available_host_memory_bytes(host.sources), 0u) << "usage above the limit";
    }

    TEST(HostMemory_pre_checkin, query_of_this_host_is_at_most_mem_available)
    {
        // /proc files report a size of 0, so copy the text rather than the file.
        FakeHost          meminfo_only;
        std::stringstream text;
        text << std::ifstream(HostMemorySources().meminfo).rdbuf();
        meminfo_only.write(meminfo_only.sources.meminfo, text.str());
        const size_t available = available_host_memory_bytes();
        EXPECT_GT(available, 0u);
        // MemAvailable moves between the two reads, so allow 1 GiB of drift.
        EXPECT_LE(available, available_host_memory_bytes(meminfo_only.sources) + kGiB);
    }

    // Sets HIPBLASLT_CLIENT_HOST_MEMORY_LIMIT for one scope.
    class ScopedLimit
    {
    public:
        explicit ScopedLimit(const char* value)
        {
            setenv("HIPBLASLT_CLIENT_HOST_MEMORY_LIMIT", value, 1);
        }
        ~ScopedLimit()
        {
            unsetenv("HIPBLASLT_CLIENT_HOST_MEMORY_LIMIT");
        }
    };

    TEST(HostMemory_pre_checkin, limit_is_a_byte_count_with_an_optional_suffix)
    {
        EXPECT_EQ(parse_host_memory_size("1024"), size_t(1024));
        EXPECT_EQ(parse_host_memory_size("0"), size_t(0));
        EXPECT_EQ(parse_host_memory_size("4K"), size_t(4) << 10);
        EXPECT_EQ(parse_host_memory_size("512m"), size_t(512) << 20);
        EXPECT_EQ(parse_host_memory_size("64G"), size_t(64) << 30);
        EXPECT_EQ(parse_host_memory_size("2T"), size_t(2) << 40);
        for(const char* bad :
            {"", "G", "64GB", "-1", "1.5G", " 64G", "64 G", "16777216T", "99999999999999999999"})
            EXPECT_EQ(parse_host_memory_size(bad), std::nullopt) << '"' << bad << '"';
    }

    TEST(HostMemory_pre_checkin, limit_caps_the_query_and_never_raises_it)
    {
        FakeHost host;
        host.mem_available(8 * kGiB);
        {
            ScopedLimit limit("2G");
            EXPECT_EQ(available_host_memory_bytes(host.sources), 2 * kGiB);
        }
        {
            ScopedLimit limit("16G");
            EXPECT_EQ(available_host_memory_bytes(host.sources), 8 * kGiB);
        }
        {
            // A full container stays full.
            host.write(host.sources.self_cgroup, "0::/job\n");
            host.cgroup("job", std::to_string(kGiB), kGiB);
            ScopedLimit limit("2G");
            EXPECT_EQ(available_host_memory_bytes(host.sources), 0u);
        }
        {
            ScopedLimit limit("not-a-size");
            EXPECT_EQ(available_host_memory_bytes(host.sources), 0u);
            EXPECT_EQ(host_memory_limit_override(), std::nullopt);
        }
        EXPECT_EQ(available_host_memory_bytes(host.sources), 0u);
    }

    TEST(HostMemory_pre_checkin, limit_is_the_budget_when_the_system_cannot_be_read)
    {
        FakeHost    host;
        ScopedLimit limit("3G");
        EXPECT_EQ(available_host_memory_bytes(host.sources), 3 * kGiB);
    }

    TEST(HostMemory_pre_checkin, limit_makes_the_preflight_skip)
    {
        EXPECT_EQ(fast_check_memory_shortfall(0, 2 * kGiB), "");
        ScopedLimit limit("1G");
        std::string why = fast_check_memory_shortfall(0, 2 * kGiB);
        EXPECT_NE(why.find("needs 2 GiB of host memory and 1 GiB is available"), std::string::npos)
            << why;
    }
#endif
} // namespace
