// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for how hipblaslt-test holds pinned host memory (AIHPBLAS-5352). The client pool keeps
// buffers that tests return; a large one left there stays pinned, and the kernel cannot
// reclaim it for the other test processes on the host. The _pre_checkin suffix puts the suite in
// the standard test tier. It pins at most about 3.3 GiB at a time and launches no kernels.

#include <gtest/gtest.h>

#include "fast_check.hpp"
#include "hipBuffer.hpp"
#include "hipblaslt_test.hpp"
#include "host_memory.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
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

    // A host without that much pinned memory to spare skips the test rather than fails it.
#define SKIP_UNLESS_PINNED(PTR)                                    \
    do                                                             \
    {                                                              \
        if(!(PTR))                                                 \
            GTEST_SKIP() << "could not pin " #PTR " on this host"; \
    } while(0)

    // Sets HIPBLASLT_CLIENT_HOST_MEMORY_LIMIT, or unsets it for nullptr, for one scope, and then
    // restores the value the run started with.
    class ScopedLimit
    {
    public:
        explicit ScopedLimit(const char* value)
        {
            if(const char* old = std::getenv(kName))
                saved = old;
            set(value);
        }
        ~ScopedLimit()
        {
            set(saved ? saved->c_str() : nullptr);
        }
        ScopedLimit(const ScopedLimit&)            = delete;
        ScopedLimit& operator=(const ScopedLimit&) = delete;

    private:
        static constexpr const char* kName = "HIPBLASLT_CLIENT_HOST_MEMORY_LIMIT";

        static void set(const char* value)
        {
#ifdef _WIN32
            _putenv_s(kName, value ? value : "");
#else
            if(value)
                setenv(kName, value, 1);
            else
                unsetenv(kName);
#endif
        }

        std::optional<std::string> saved;
    };

    TEST(HostMemory_pre_checkin, large_buffer_released_after_a_test_that_does_not_reuse_it)
    {
        // The pool refuses to pin beyond a host memory limit, and these sizes are the test.
        ScopedLimit no_limit(nullptr);
        memory_pool<h_memory>::ReleaseCached();
        {
            HipHostBuffer large(HIP_R_8I, kHostPoolRetainLimit);
            HipHostBuffer small(HIP_R_8I, kSmall);
            SKIP_UNLESS_PINNED(large.buf());
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

    TEST(HostMemory_pre_checkin, small_buffer_released_after_tests_that_do_not_reuse_it)
    {
        memory_pool<h_memory>::ReleaseCached();
        auto idle = memory_pool<h_memory>::Get(kSmall);
        ASSERT_NE(idle.get(), nullptr);
        memory_pool<h_memory>::Restore(idle);
        // This test returned the buffer; it stays through the next kHostPoolSmallIdleTests - 1.
        end_test();
        for(uint64_t t = 1; t < kHostPoolSmallIdleTests; t++)
            end_test();
        EXPECT_GE(memory_pool<h_memory>::CachedBytes(), kSmall);
        end_test();
        EXPECT_EQ(memory_pool<h_memory>::CachedBytes(), 0u);

        // A test that reuses the buffer starts its count again.
        auto reused = memory_pool<h_memory>::Get(kSmall);
        ASSERT_NE(reused.get(), nullptr);
        memory_pool<h_memory>::Restore(reused);
        for(uint64_t t = 0; t < kHostPoolSmallIdleTests / 2; t++)
            end_test();
        reused = memory_pool<h_memory>::Get(kSmall);
        memory_pool<h_memory>::Restore(reused);
        end_test();
        for(uint64_t t = 1; t < kHostPoolSmallIdleTests; t++)
            end_test();
        EXPECT_GE(memory_pool<h_memory>::CachedBytes(), kSmall);
        memory_pool<h_memory>::ReleaseCached();
    }

    TEST(HostMemory_pre_checkin, next_test_of_the_same_shape_reuses_the_large_buffer)
    {
        if(memory_pool<h_memory>::FreesBeforeEachGet())
            GTEST_SKIP() << "the pool on this host frees its idle buffers before every request";
        ScopedLimit no_limit(nullptr);
        memory_pool<h_memory>::ReleaseCached();
        auto first = memory_pool<h_memory>::Get(kHostPoolRetainLimit);
        SKIP_UNLESS_PINNED(first.get());
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
        ScopedLimit no_limit(nullptr);
        // A test of another shape needs a new large buffer; the previous test's ones are
        // released before more memory is pinned, not after this test ends. On any miss the
        // pool already drops the largest buffer that is too small, so two are left here.
        memory_pool<h_memory>::ReleaseCached();
        const size_t step    = kHostPoolRetainLimit / 10;
        auto         earlier = memory_pool<h_memory>::Get(kHostPoolRetainLimit);
        auto         bigger  = memory_pool<h_memory>::Get(kHostPoolRetainLimit + step);
        SKIP_UNLESS_PINNED(earlier.get());
        SKIP_UNLESS_PINNED(bigger.get());
        memory_pool<h_memory>::Restore(earlier);
        memory_pool<h_memory>::Restore(bigger);
        end_test();
        ASSERT_GE(memory_pool<h_memory>::CachedBytes(), 2 * kHostPoolRetainLimit);

        auto larger = memory_pool<h_memory>::Get(kHostPoolRetainLimit + 2 * step);
        SKIP_UNLESS_PINNED(larger.get());
        EXPECT_EQ(memory_pool<h_memory>::CachedBytes(), 0u);
        memory_pool<h_memory>::Restore(larger);
        memory_pool<h_memory>::ReleaseCached();
    }

    TEST(HostMemory_pre_checkin, buffers_at_the_retain_limit_get_no_slack)
    {
        ScopedLimit no_limit(nullptr);
        // The pool adds 20% to small requests so that a slightly larger one can reuse them. A
        // large buffer is reused only by the same shape, so the extra would only add pinned
        // memory.
        memory_pool<h_memory>::ReleaseCached();
        auto large = memory_pool<h_memory>::Get(kHostPoolRetainLimit);
        SKIP_UNLESS_PINNED(large.get());
        EXPECT_EQ(large.capacity(), kHostPoolRetainLimit);
        memory_pool<h_memory>::Restore(large);

        auto small = memory_pool<h_memory>::Get(kSmall);
        ASSERT_NE(small.get(), nullptr);
        EXPECT_GT(small.capacity(), small.bytes());
        memory_pool<h_memory>::Restore(small);
        memory_pool<h_memory>::ReleaseCached();

        // The extra would take this one to the limit, where it would count as large.
        const size_t request = kHostPoolRetainLimit - kHostPoolRetainLimit / 10;
        auto         below   = memory_pool<h_memory>::Get(request);
        SKIP_UNLESS_PINNED(below.get());
        EXPECT_EQ(below.capacity(), request);
        memory_pool<h_memory>::Restore(below);
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

        EXPECT_EQ(host_allocation_failure(ok[0]), "");
        EXPECT_EQ(host_allocation_failure(failed[0]),
                  "could not allocate 1.07e+09 GiB of pinned host memory");
        EXPECT_EQ(host_allocation_failure({&ok}), "");
        std::string why = host_allocation_failure({&ok, &failed});
        EXPECT_NE(why.find("could not allocate 1.07e+09 GiB of pinned host memory"),
                  std::string::npos)
            << why;
    }

    TEST(HostMemory_pre_checkin, memory_shortages_are_listed_with_their_reasons)
    {
        MemoryShortageLog log;
        EXPECT_EQ(log.summary(), "");
        log.record("Suite.passes", "Succeeded");
        log.record("Suite.other_skip", "streamk is not available");
        log.record("Suite.no_message", nullptr);
        EXPECT_EQ(log.count(), 0u);

        log.record("Suite.big",
                   "Skipped\n" MEMORY_SHORTAGE_STRING
                   "this case needs 48 GiB of host memory and 30 GiB is available");
        log.record("Suite.huge",
                   MEMORY_SHORTAGE_STRING "a device allocation failed: out of memory");
        EXPECT_EQ(log.count(), 2u);
        EXPECT_EQ(log.summary(),
                  "[ MEMORY   ] 2 tests stopped because memory was short:\n"
                  "[ MEMORY   ] Suite.big: this case needs 48 GiB of host memory and 30 GiB is "
                  "available\n"
                  "[ MEMORY   ] Suite.huge: a device allocation failed: out of memory\n");
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
            sources.mountinfo   = (dir / "mountinfo").string();
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

    private:
        // A limit set for the whole run would cap every query here.
        ScopedLimit no_limit{nullptr};
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

        // mountinfo names another cgroup at the mount, so the listed path is not below it. The
        // walk stays inside the mount: a full limit beside it is not read.
        const auto beside = std::filesystem::path(host.sources.cgroup_root).parent_path();
        host.write(beside / "memory.max", std::to_string(kGiB) + "\n");
        host.write(beside / "memory.current", std::to_string(kGiB) + "\n");
        host.write(host.sources.mountinfo,
                   "30 24 0:26 /kubepods/pod2 " + host.sources.cgroup_root
                       + " rw - cgroup2 cgroup2 rw\n");
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
        EXPECT_EQ(available_host_memory_bytes(host.sources), std::nullopt) << "no meminfo";

        host.mem_available(8 * kGiB);
        host.write(host.sources.self_cgroup, "0::/job\n");
        host.cgroup("job", "not-a-number", kGiB);
        EXPECT_EQ(available_host_memory_bytes(host.sources), 8 * kGiB) << "garbled limit";

        host.cgroup("job", std::to_string(kGiB), 2 * kGiB);
        EXPECT_EQ(available_host_memory_bytes(host.sources), 0u) << "usage above the limit";
    }

    TEST(HostMemory_pre_checkin, query_of_this_host_reads_the_real_files)
    {
        // The fake-tree tests cover the cgroup logic. This one only checks that the default
        // sources are readable on a real host and that its cgroups can only lower MemAvailable;
        // 0 is a valid answer in a full container. /proc files report a size of 0, so copy
        // the text rather than the file.
        FakeHost          meminfo_only;
        std::stringstream text;
        text << std::ifstream(HostMemorySources().meminfo).rdbuf();
        meminfo_only.write(meminfo_only.sources.meminfo, text.str());
        const std::optional<size_t> available = available_host_memory_bytes();
        ASSERT_TRUE(available.has_value());
        // MemAvailable moves between the two reads, so allow 1 GiB of drift.
        EXPECT_LE(*available, available_host_memory_bytes(meminfo_only.sources).value_or(0) + kGiB);
    }

    TEST(HostMemory_pre_checkin, query_counts_active_page_cache_as_reclaimable)
    {
        // A 2 GiB container that is full, mostly with page cache the kernel reclaims (active or
        // inactive) before it kills anything. Pages still being written back are not counted.
        FakeHost host;
        host.mem_available(8 * kGiB);
        host.write(host.sources.self_cgroup, "0::/job\n");
        host.cgroup("job", std::to_string(2 * kGiB), 2 * kGiB);
        host.write(std::filesystem::path(host.sources.cgroup_root) / "job/memory.stat",
                   "anon 524288\nfile 1610612746\nactive_file " + std::to_string(3 * kGiB / 2)
                       + "\ninactive_file 10\nfile_dirty 0\nfile_writeback 0\n");
        EXPECT_EQ(available_host_memory_bytes(host.sources), 3 * kGiB / 2 + 10);

        host.write(std::filesystem::path(host.sources.cgroup_root) / "job/memory.stat",
                   "active_file " + std::to_string(3 * kGiB / 2) + "\ninactive_file 10\nfile_dirty "
                       + std::to_string(kGiB / 4) + "\nfile_writeback " + std::to_string(kGiB / 4)
                       + "\n");
        EXPECT_EQ(available_host_memory_bytes(host.sources), kGiB + 10);
    }

    TEST(HostMemory_pre_checkin, query_maps_the_cgroup_path_through_its_mount)
    {
        // The hierarchy is mounted from /runner, so /proc/self/cgroup's /runner/job is the
        // mount's job directory. Its 2 GiB limit, not the mount top's 6 GiB, applies.
        FakeHost host;
        host.mem_available(8 * kGiB);
        host.write(host.sources.self_cgroup, "0::/runner/job\n");
        host.write(host.sources.mountinfo,
                   "24 1 0:22 / / rw - ext4 /dev/root rw\n30 24 0:26 /runner "
                       + host.sources.cgroup_root + " rw,nosuid - cgroup2 cgroup2 rw\n");
        host.cgroup("", std::to_string(6 * kGiB), kGiB);
        host.cgroup("job", std::to_string(2 * kGiB), kGiB);
        EXPECT_EQ(available_host_memory_bytes(host.sources), kGiB);
    }

    TEST(HostMemory_pre_checkin, query_finds_the_v1_memory_controller_where_it_is_mounted)
    {
        // The v1 memory controller shares a mount with cpu, at "cpu,memory", not "memory".
        FakeHost host;
        host.mem_available(8 * kGiB);
        host.write(host.sources.self_cgroup, "4:cpu,memory:/job\n");
        const std::string point = host.sources.cgroup_root + "/cpu,memory";
        host.write(host.sources.mountinfo,
                   "31 24 0:27 / " + point + " rw - cgroup cgroup rw,cpu,memory\n");
        host.write(point + "/job/memory.limit_in_bytes", std::to_string(3 * kGiB) + "\n");
        host.write(point + "/job/memory.usage_in_bytes", std::to_string(kGiB) + "\n");
        EXPECT_EQ(available_host_memory_bytes(host.sources), 2 * kGiB);
    }

    TEST(HostMemory_pre_checkin, query_applies_the_cgroup_limit_without_meminfo)
    {
        // With /proc/meminfo unreadable, a full container is still full, with or without a limit.
        FakeHost host;
        host.write(host.sources.self_cgroup, "0::/job\n");
        host.cgroup("job", std::to_string(kGiB), kGiB);
        EXPECT_EQ(available_host_memory_bytes(host.sources), size_t(0));
        ScopedLimit limit("2G");
        EXPECT_EQ(available_host_memory_bytes(host.sources), size_t(0));
    }

    TEST(HostMemory_pre_checkin, query_follows_a_removed_cgroup)
    {
        // /proc/self/cgroup lists a cgroup that has been removed with a " (deleted)" suffix.
        FakeHost host;
        host.mem_available(8 * kGiB);
        host.write(host.sources.self_cgroup, "0::/job (deleted)\n");
        host.cgroup("job", std::to_string(2 * kGiB), kGiB);
        EXPECT_EQ(available_host_memory_bytes(host.sources), kGiB);
    }

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
            // An unreadable value is ignored, not read as 0.
            ScopedLimit limit("not-a-size");
            EXPECT_EQ(available_host_memory_bytes(host.sources), 8 * kGiB);
            EXPECT_EQ(host_memory_limit_override(), std::nullopt);
        }
        // Each limit above ended with its scope.
        EXPECT_EQ(available_host_memory_bytes(host.sources), 8 * kGiB);
        {
            // A full container stays full.
            host.write(host.sources.self_cgroup, "0::/job\n");
            host.cgroup("job", std::to_string(kGiB), kGiB);
            ScopedLimit limit("2G");
            EXPECT_EQ(available_host_memory_bytes(host.sources), 0u);
        }
    }

    TEST(HostMemory_pre_checkin, limit_is_the_budget_when_the_system_cannot_be_read)
    {
        FakeHost    host;
        ScopedLimit limit("3G");
        EXPECT_EQ(available_host_memory_bytes(host.sources), 3 * kGiB);
    }

    TEST(HostMemory_pre_checkin, limit_makes_the_preflight_skip)
    {
        // Only a runner with 2 GiB free shows that the limit, not the runner, made it skip.
        {
            ScopedLimit no_limit(nullptr);
            if(available_host_memory_bytes().value_or(0) < 2 * kGiB)
                GTEST_SKIP() << "this runner has less than 2 GiB of host memory free";
        }
        ScopedLimit limit("1G");
        std::string why = fast_check_memory_shortfall(0, 2 * kGiB);
        EXPECT_NE(why.find("needs 2 GiB of host memory and 1 GiB is available"), std::string::npos)
            << why;
    }

    TEST(HostMemory_pre_checkin, no_host_memory_left_makes_the_preflight_skip)
    {
        // A budget of 0, from a full container or this limit, means nothing is left. It differs
        // from a budget the system could not report, under which the preflight lets the case run.
        ScopedLimit limit("0");
        std::string why = fast_check_memory_shortfall(0, kGiB);
        EXPECT_NE(why.find("needs 1 GiB of host memory and 0 GiB is available"), std::string::npos)
            << why;
    }
#endif
} // namespace
