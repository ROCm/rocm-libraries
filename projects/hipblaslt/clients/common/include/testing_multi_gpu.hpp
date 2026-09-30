// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// A multi-GPU test runs each rank as a copy of the test binary, started with
// the RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT variables that
// read_launcher_env() reads. Each rank exits with one of the codes below.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern char** environ;

namespace hipblaslt_bench
{
    constexpr int kRankChildPassed  = 0;
    constexpr int kRankChildFailed  = 1;
    constexpr int kRankChildSkipped = 2;

    constexpr int kReapTimeoutSec = 10;

    // Binds port 0, reads back what the OS assigned, then releases it.
    inline bool free_port(uint16_t& port)
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

    // Starts this binary with `role` as its only argument, as rank `rank` of a
    // single-host group on loopback. Each variable in `extraEnv` replaces the
    // inherited one of the same name.
    inline int spawn_rank(const char*                                             role,
                          uint32_t                                                rank,
                          uint32_t                                                world,
                          uint16_t                                                port,
                          const std::vector<std::pair<std::string, std::string>>& extraEnv,
                          pid_t&                                                  pid)
    {
        std::vector<std::pair<std::string, std::string>> variables = {
            {"RANK", std::to_string(rank)},
            {"LOCAL_RANK", std::to_string(rank)},
            {"WORLD_SIZE", std::to_string(world)},
            {"MASTER_ADDR", "127.0.0.1"},
            {"MASTER_PORT", std::to_string(port)},
        };
        variables.insert(variables.end(), extraEnv.begin(), extraEnv.end());

        std::vector<std::string> assignments;
        for(const auto& kv : variables)
            assignments.push_back(kv.first + "=" + kv.second);

        std::vector<char*> envp;
        for(const std::string& assignment : assignments)
            envp.push_back(const_cast<char*>(assignment.c_str()));
        for(char** entry = environ; *entry != nullptr; ++entry)
        {
            bool assigned = false;
            for(const auto& kv : variables)
                assigned = assigned
                           || (strncmp(*entry, kv.first.c_str(), kv.first.size()) == 0
                               && (*entry)[kv.first.size()] == '=');
            if(!assigned)
                envp.push_back(*entry);
        }
        envp.push_back(nullptr);

        char* const argv[]
            = {const_cast<char*>("hipblaslt-test"), const_cast<char*>(role), nullptr};

        return posix_spawn(&pid, "/proc/self/exe", nullptr, nullptr, argv, envp.data());
    }

    // Gives up on a rank that SIGKILL cannot reach.
    inline void kill_ranks(const std::vector<pid_t>& pids)
    {
        for(pid_t pid : pids)
            if(pid > 0)
                kill(pid, SIGKILL);

        const auto deadline
            = std::chrono::steady_clock::now() + std::chrono::seconds(kReapTimeoutSec);
        for(pid_t pid : pids)
        {
            if(pid <= 0)
                continue;

            int discarded = 0;
            while(waitpid(pid, &discarded, WNOHANG) == 0
                  && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    // False when the deadline passes with a rank still running.
    inline bool wait_for_ranks(std::vector<pid_t>& pids, std::vector<int>& codes, int timeoutSec)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
        codes.assign(pids.size(), -1);

        size_t exited = 0;
        for(size_t i = 0; i < pids.size(); ++i)
            if(pids[i] <= 0)
            {
                codes[i] = kRankChildFailed;
                ++exited;
            }

        while(exited < pids.size())
        {
            for(size_t i = 0; i < pids.size(); ++i)
            {
                if(pids[i] <= 0)
                    continue;

                int         status = 0;
                const pid_t seen   = waitpid(pids[i], &status, WNOHANG);
                if(seen != pids[i])
                    continue;

                codes[i] = WIFEXITED(status) ? WEXITSTATUS(status) : kRankChildFailed;
                pids[i]  = -1;
                ++exited;
            }

            if(exited == pids.size())
                break;
            if(std::chrono::steady_clock::now() > deadline)
                return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return true;
    }
} // namespace hipblaslt_bench
