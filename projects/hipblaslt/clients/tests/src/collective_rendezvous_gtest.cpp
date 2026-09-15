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
    uint16_t free_port();
} // namespace

TEST(collective_rendezvous, absent_env_degrades_to_single_rank)
{
    clear_launcher_env();

    const LauncherEnv env = read_launcher_env();

    EXPECT_EQ(env.rank, 0u);
    EXPECT_EQ(env.world, 1u);
    EXPECT_EQ(env.local_rank, 0);
}

TEST(collective_rendezvous, env_is_read_verbatim)
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

TEST(collective_rendezvous, single_rank_allgather_is_a_copy)
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

TEST(collective_rendezvous, four_ranks_gather_in_rank_order)
{
    const uint16_t       port = free_port();
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

TEST(collective_rendezvous, missing_rank_times_out_rather_than_blocking)
{
    const uint16_t     port   = free_port();
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

TEST(collective_rendezvous, single_rank_is_trivially_one_host)
{
    LauncherEnv env;
    env.rank  = 0;
    env.world = 1;

    TcpRendezvous r(env, 10);

    EXPECT_TRUE(r.same_host_group());
}

TEST(collective_rendezvous, two_ranks_agree_on_same_host_group)
{
    const uint16_t     port   = free_port();
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

namespace
{
    uint16_t free_port()
    {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
        addr.sin_port        = 0;
        ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        socklen_t len = sizeof(addr);
        ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
        const uint16_t port = ::ntohs(addr.sin_port);
        ::close(fd);
        return port;
    }
} // namespace
