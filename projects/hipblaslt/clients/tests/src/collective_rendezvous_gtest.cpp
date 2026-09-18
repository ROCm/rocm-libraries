// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for collective_rendezvous.hpp. Multi-rank cases run the
// real TCP path with one thread per rank against loopback; no GPU is involved.

#include "collective_rendezvous.hpp"

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using hipblaslt_bench::LauncherEnv;
using hipblaslt_bench::read_launcher_env;
using hipblaslt_bench::TcpRendezvous;

namespace
{
    void clear_launcher_env()
    {
        for(const char* name :
            {"RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"})
            ::unsetenv(name);
    }

    // Binds port 0, reads back what the OS assigned, then releases it.
    bool free_port(uint16_t& port);
} // namespace

TEST(collective_rendezvous_smoke, absent_env_degrades_to_single_rank)
{
    clear_launcher_env();

    const LauncherEnv env = read_launcher_env();

    EXPECT_EQ(env.rank, 0u);
    EXPECT_EQ(env.world, 1u);
    EXPECT_EQ(env.local_rank, 0);
}

TEST(collective_rendezvous_smoke, env_is_read_verbatim)
{
    clear_launcher_env();
    ::setenv("RANK", "3", 1);
    ::setenv("WORLD_SIZE", "4", 1);
    ::setenv("LOCAL_RANK", "3", 1);
    ::setenv("MASTER_ADDR", "127.0.0.1", 1);
    ::setenv("MASTER_PORT", "29500", 1);

    const LauncherEnv env = read_launcher_env();

    EXPECT_EQ(env.rank, 3u);
    EXPECT_EQ(env.world, 4u);
    EXPECT_EQ(env.local_rank, 3);
    EXPECT_EQ(env.master_addr, "127.0.0.1");
    EXPECT_EQ(env.master_port, 29500);

    clear_launcher_env();
}

TEST(collective_rendezvous_smoke, invalid_world_size_degrades_to_single_rank)
{
    clear_launcher_env();

    ::setenv("WORLD_SIZE", "0", 1);
    EXPECT_EQ(read_launcher_env().world, 1u);

    ::setenv("WORLD_SIZE", "not-a-number", 1);
    EXPECT_EQ(read_launcher_env().world, 1u);

    clear_launcher_env();
}

TEST(collective_rendezvous_smoke, single_rank_allgather_is_a_copy)
{
    LauncherEnv env;
    env.rank  = 0;
    env.world = 1;

    TcpRendezvous r(env, 10);

    const uint64_t mine = 0xAABBCCDDu;
    uint64_t       got  = 0;

    EXPECT_EQ(r.allgather(&mine, &got, sizeof(mine)), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(got, mine);
}

TEST(collective_rendezvous_smoke, four_ranks_gather_in_rank_order)
{
    uint16_t port = 0;
    ASSERT_TRUE(free_port(port));

    constexpr uint32_t   kWorld = 4;
    std::vector<uint64_t> got(kWorld* kWorld, 0);
    std::vector<std::thread> ranks;

    for(uint32_t i = 0; i < kWorld; ++i)
        ranks.emplace_back([i, port, &got] {
            LauncherEnv env;
            env.rank        = i;
            env.world       = kWorld;
            env.master_addr = "127.0.0.1";
            env.master_port = port;

            TcpRendezvous r(env, 30);

            const uint64_t mine = 100 + i;
            ASSERT_EQ(r.allgather(&mine, &got[i * kWorld], sizeof(mine)),
                      HIPBLAS_STATUS_SUCCESS);
        });

    for(std::thread& t : ranks)
        t.join();

    // Every rank must see the same vector, ordered by rank.
    for(uint32_t i = 0; i < kWorld; ++i)
        for(uint32_t j = 0; j < kWorld; ++j)
            EXPECT_EQ(got[i * kWorld + j], 100 + j) << "rank " << i << " slot " << j;
}

TEST(collective_rendezvous_smoke, missing_rank_times_out_rather_than_blocking)
{
    uint16_t port = 0;
    ASSERT_TRUE(free_port(port));

    constexpr uint32_t kWorld = 2;

    LauncherEnv env;
    env.rank        = 0;
    env.world       = kWorld;
    env.master_addr = "127.0.0.1";
    env.master_port = port;

    TcpRendezvous r(env, 1);

    const uint64_t mine = 7;
    uint64_t       got[kWorld] = {};

    // Rank 1 never arrives.
    EXPECT_NE(r.allgather(&mine, got, sizeof(mine)), HIPBLAS_STATUS_SUCCESS);
}

TEST(collective_rendezvous_smoke, single_rank_is_trivially_one_host)
{
    LauncherEnv env;
    env.rank  = 0;
    env.world = 1;

    TcpRendezvous r(env, 10);

    EXPECT_TRUE(r.same_host_group());
}

TEST(collective_rendezvous_smoke, two_ranks_agree_on_same_host_group)
{
    uint16_t port = 0;
    ASSERT_TRUE(free_port(port));

    constexpr uint32_t kWorld = 2;

    std::vector<std::thread> ranks;
    bool                     result[kWorld] = {};

    for(uint32_t i = 0; i < kWorld; ++i)
        ranks.emplace_back([i, port, &result] {
            LauncherEnv env;
            env.rank        = i;
            env.world       = kWorld;
            env.master_addr = "127.0.0.1";
            env.master_port = port;

            TcpRendezvous r(env, 30);

            result[i] = r.same_host_group();
        });

    for(std::thread& t : ranks)
        t.join();

    EXPECT_TRUE(result[0]);
    EXPECT_TRUE(result[1]);
}

TEST(collective_rendezvous_smoke, multi_segment_payload_arrives_intact)
{
    uint16_t port = 0;
    ASSERT_TRUE(free_port(port));

    constexpr uint32_t kWorld = 2;
    constexpr size_t   kBytes = 1u << 20;

    std::vector<std::vector<uint8_t>> got(kWorld, std::vector<uint8_t>(kBytes * kWorld, 0));
    std::vector<std::thread>          ranks;

    for(uint32_t i = 0; i < kWorld; ++i)
        ranks.emplace_back([i, port, &got] {
            LauncherEnv env;
            env.rank        = i;
            env.world       = kWorld;
            env.master_addr = "127.0.0.1";
            env.master_port = port;

            TcpRendezvous r(env, 30);

            std::vector<uint8_t> mine(kBytes);
            for(size_t b = 0; b < kBytes; ++b)
                mine[b] = uint8_t((b + i) & 0xff);

            ASSERT_EQ(r.allgather(mine.data(), got[i].data(), kBytes), HIPBLAS_STATUS_SUCCESS);
        });

    for(std::thread& t : ranks)
        t.join();

    for(uint32_t j = 0; j < kWorld; ++j)
    {
        std::vector<uint8_t> want(kBytes);
        for(size_t b = 0; b < kBytes; ++b)
            want[b] = uint8_t((b + j) & 0xff);

        for(uint32_t i = 0; i < kWorld; ++i)
            EXPECT_EQ(std::memcmp(got[i].data() + size_t(j) * kBytes, want.data(), kBytes), 0)
                << "rank " << i << " slot " << j;
    }
}

namespace
{
    bool free_port(uint16_t& port)
    {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if(fd < 0)
            return false;

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
        addr.sin_port        = 0;

        socklen_t  len = sizeof(addr);
        const bool ok  = ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0
                         && ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0
                         && addr.sin_port != 0;
        ::close(fd);
        if(!ok)
            return false;

        port = ::ntohs(addr.sin_port);
        return true;
    }
} // namespace
