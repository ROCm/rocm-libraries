// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "benchmark_collective.hpp"
#include "hipblaslt_ostream.hpp"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace hipblaslt_bench
{
    constexpr int kRendezvousTimeoutSec = 60;

    struct LauncherEnv
    {
        uint32_t    rank        = 0;
        uint32_t    world       = 1;
        int         local_rank  = 0;
        std::string master_addr = "127.0.0.1";
        uint16_t    master_port = 0;
    };

    inline LauncherEnv read_launcher_env()
    {
        LauncherEnv env;
        if(const char* v = std::getenv("RANK"))
            env.rank = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
        if(const char* v = std::getenv("WORLD_SIZE"))
            env.world = std::max(1u, static_cast<uint32_t>(std::strtoul(v, nullptr, 10)));
        if(const char* v = std::getenv("LOCAL_RANK"))
            env.local_rank = static_cast<int>(std::strtol(v, nullptr, 10));
        if(const char* v = std::getenv("MASTER_ADDR"))
            env.master_addr = v;
        if(const char* v = std::getenv("MASTER_PORT"))
            env.master_port = static_cast<uint16_t>(std::strtoul(v, nullptr, 10));
        return env;
    }

    // Rank 0 serves; the rest connect. Each allgather sends one contribution and
    // receives the group's, ordered by rank.
    class TcpRendezvous
    {
    public:
        TcpRendezvous(const LauncherEnv& env, int timeout_sec)
            : m_env(env)
            , m_timeout_sec(timeout_sec)
        {
        }

        ~TcpRendezvous()
        {
            for(int fd : m_peers)
                if(fd >= 0)
                    ::close(fd);
            if(m_listen >= 0)
                ::close(m_listen);
            if(m_to_server >= 0)
                ::close(m_to_server);
        }

        TcpRendezvous(const TcpRendezvous&)            = delete;
        TcpRendezvous& operator=(const TcpRendezvous&) = delete;

        hipblasStatus_t allgather(const void* sendbuf, void* recvbuf, size_t bytesPerRank)
        {
            if(m_env.world == 1)
            {
                std::memcpy(recvbuf, sendbuf, bytesPerRank);
                return HIPBLAS_STATUS_SUCCESS;
            }
            if(!connected() && !connect_group())
                return HIPBLAS_STATUS_INTERNAL_ERROR;

            return (m_env.rank == 0) ? serve(sendbuf, recvbuf, bytesPerRank)
                                     : participate(sendbuf, recvbuf, bytesPerRank);
        }

        bool same_host_group()
        {
            if(m_env.world == 1)
                return true;

            char mine[65] = {};
            if(::gethostname(mine, sizeof(mine) - 1) != 0)
                mine[0] = '\0';

            std::vector<char> all(sizeof(mine) * m_env.world);
            if(allgather(mine, all.data(), sizeof(mine)) != HIPBLAS_STATUS_SUCCESS)
                return false;

            for(uint32_t j = 0; j < m_env.world; ++j)
                if(all[j * sizeof(mine)] == '\0')
                    return false;

            for(uint32_t j = 1; j < m_env.world; ++j)
                if(std::memcmp(all.data(), all.data() + j * sizeof(mine), sizeof(mine)) != 0)
                    return false;
            return true;
        }

    private:
        bool connected() const
        {
            return (m_env.rank == 0) ? m_listen >= 0 : m_to_server >= 0;
        }

        void apply_timeout(int fd) const
        {
            timeval tv{};
            tv.tv_sec = m_timeout_sec;
            ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
            const int one = 1;
            ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        }

        sockaddr_in server_addr() const
        {
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port   = ::htons(m_env.master_port);
            ::inet_pton(AF_INET, m_env.master_addr.c_str(), &addr.sin_addr);
            return addr;
        }

        bool connect_group()
        {
            return (m_env.rank == 0) ? accept_peers() : dial_server();
        }

        bool accept_peers()
        {
            m_listen = ::socket(AF_INET, SOCK_STREAM, 0);
            if(m_listen < 0)
                return false;
            const int one = 1;
            ::setsockopt(m_listen, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
            apply_timeout(m_listen);

            sockaddr_in addr = server_addr();
            addr.sin_addr.s_addr = ::htonl(INADDR_ANY);
            if(::bind(m_listen, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
                return false;
            if(::listen(m_listen, int(m_env.world)) != 0)
                return false;

            m_peers.assign(m_env.world, -1);
            for(uint32_t accepted = 1; accepted < m_env.world; ++accepted)
            {
                const int fd = ::accept(m_listen, nullptr, nullptr);
                if(fd < 0)
                    return false;
                apply_timeout(fd);

                uint32_t who = 0;
                if(!read_exact(fd, &who, sizeof(who)) || who == 0 || who >= m_env.world)
                {
                    ::close(fd);
                    return false;
                }
                m_peers[who] = fd;
            }
            return true;
        }

        bool dial_server()
        {
            const sockaddr_in addr = server_addr();
            for(int attempt = 0; attempt < m_timeout_sec * 10; ++attempt)
            {
                m_to_server = ::socket(AF_INET, SOCK_STREAM, 0);
                if(m_to_server < 0)
                    return false;
                apply_timeout(m_to_server);

                if(::connect(m_to_server,
                             reinterpret_cast<const sockaddr*>(&addr),
                             sizeof(addr))
                   == 0)
                    return write_exact(m_to_server, &m_env.rank, sizeof(m_env.rank));

                ::close(m_to_server);
                m_to_server = -1;
                ::usleep(100000);
            }
            return false;
        }

        hipblasStatus_t serve(const void* sendbuf, void* recvbuf, size_t bytesPerRank)
        {
            char* const out = static_cast<char*>(recvbuf);
            std::memcpy(out, sendbuf, bytesPerRank);

            for(uint32_t j = 1; j < m_env.world; ++j)
                if(!read_exact(m_peers[j], out + j * bytesPerRank, bytesPerRank))
                    return HIPBLAS_STATUS_INTERNAL_ERROR;

            for(uint32_t j = 1; j < m_env.world; ++j)
                if(!write_exact(m_peers[j], out, bytesPerRank * m_env.world))
                    return HIPBLAS_STATUS_INTERNAL_ERROR;

            return HIPBLAS_STATUS_SUCCESS;
        }

        hipblasStatus_t participate(const void* sendbuf, void* recvbuf, size_t bytesPerRank)
        {
            if(!write_exact(m_to_server, sendbuf, bytesPerRank))
                return HIPBLAS_STATUS_INTERNAL_ERROR;
            if(!read_exact(m_to_server, recvbuf, bytesPerRank * m_env.world))
                return HIPBLAS_STATUS_INTERNAL_ERROR;
            return HIPBLAS_STATUS_SUCCESS;
        }

        static bool read_exact(int fd, void* buf, size_t bytes)
        {
            char* p = static_cast<char*>(buf);
            while(bytes > 0)
            {
                const ssize_t n = ::recv(fd, p, bytes, 0);
                if(n < 0 && errno == EINTR)
                    continue;
                if(n <= 0)
                    return false;
                p += n;
                bytes -= size_t(n);
            }
            return true;
        }

        static bool write_exact(int fd, const void* buf, size_t bytes)
        {
            const char* p = static_cast<const char*>(buf);
            while(bytes > 0)
            {
                const ssize_t n = ::send(fd, p, bytes, 0);
                if(n < 0 && errno == EINTR)
                    continue;
                if(n <= 0)
                    return false;
                p += n;
                bytes -= size_t(n);
            }
            return true;
        }

        LauncherEnv      m_env;
        int              m_timeout_sec = 0;
        int              m_listen      = -1;
        int              m_to_server   = -1;
        std::vector<int> m_peers;
    };

    inline hipblasStatus_t rendezvous_allgather_trampoline(void*       userData,
                                                           const void* sendbuf,
                                                           void*       recvbuf,
                                                           size_t      bytesPerRank)
    {
        return static_cast<TcpRendezvous*>(userData)->allgather(sendbuf, recvbuf, bytesPerRank);
    }

    inline CollectiveAgreement make_agreement(TcpRendezvous& rendezvous, uint32_t world)
    {
        CollectiveAgreement agreement;
        agreement.world     = world;
        agreement.allgather = [&rendezvous](const void* send, void* recv, size_t bytes) {
            return rendezvous.allgather(send, recv, bytes) == HIPBLAS_STATUS_SUCCESS;
        };
        return agreement;
    }

    inline bool peers_reachable(const LauncherEnv& env)
    {
        if(env.world == 1)
            return true;

        if(env.local_rank != int(env.rank))
        {
            hipblaslt_cerr << "error: LOCAL_RANK " << env.local_rank << " must equal RANK "
                           << env.rank << " when every rank shares a host\n";
            return false;
        }

        int visible = 0;
        if(hipGetDeviceCount(&visible) != hipSuccess || visible < int(env.world))
        {
            hipblaslt_cerr << "error: " << visible << " device(s) visible, need " << env.world
                           << "\n";
            return false;
        }

        if(hipSetDevice(env.local_rank) != hipSuccess)
        {
            hipblaslt_cerr << "error: hipSetDevice(" << env.local_rank << ") failed\n";
            return false;
        }

        for(uint32_t j = 0; j < env.world; ++j)
        {
            if(int(j) == env.local_rank)
                continue;

            int canAccess = 0;
            if(hipDeviceCanAccessPeer(&canAccess, env.local_rank, int(j)) != hipSuccess
               || canAccess == 0)
            {
                hipblaslt_cerr << "error: device " << env.local_rank << " cannot peer with " << j
                               << "\n";
                return false;
            }

            const hipError_t e = hipDeviceEnablePeerAccess(int(j), 0);
            if(e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
            {
                hipblaslt_cerr << "error: hipDeviceEnablePeerAccess(" << env.local_rank << " -> "
                               << j << ") -> " << hipGetErrorString(e) << "\n";
                return false;
            }
        }
        return true;
    }

    inline bool join_group(const LauncherEnv& env, TcpRendezvous& rendezvous, uint32_t maxWorld)
    {
        if(env.world > maxWorld)
        {
            hipblaslt_cout << "skipped: WORLD_SIZE " << env.world << " exceeds " << maxWorld
                           << "\n";
            return false;
        }

        if(!rendezvous.same_host_group())
        {
            hipblaslt_cout << "skipped: ranks span hosts\n";
            return false;
        }

        if(!make_agreement(rendezvous, env.world).agree(peers_reachable(env), std::logical_and<>{}))
        {
            hipblaslt_cout << "skipped: peer access unavailable on at least one rank\n";
            return false;
        }
        return true;
    }

    // peers[j] is rank j's `local` mapped into this process; peers[env.rank] is
    // `local` itself.
    inline bool exchange_ipc_pointers(const LauncherEnv& env,
                                      TcpRendezvous&     rendezvous,
                                      void*              local,
                                      void**             peers)
    {
        if(env.world == 1)
        {
            peers[0] = local;
            return true;
        }

        struct HandleContribution
        {
            uint8_t           ok;
            hipIpcMemHandle_t handle;
        };
        HandleContribution mine{};
        mine.ok = hipIpcGetMemHandle(&mine.handle, local) == hipSuccess ? 1 : 0;

        std::vector<HandleContribution> all(env.world);
        if(rendezvous.allgather(&mine, all.data(), sizeof(mine)) != HIPBLAS_STATUS_SUCCESS)
        {
            hipblaslt_cerr << "error: IPC handle allgather failed\n";
            return false;
        }
        bool gotAllHandles = true;
        for(uint32_t j = 0; j < env.world; ++j)
            gotAllHandles = gotAllHandles && all[j].ok != 0;
        if(!gotAllHandles)
        {
            hipblaslt_cerr << "error: hipIpcGetMemHandle failed on at least one rank\n";
            return false;
        }

        bool openedAll = true;
        for(uint32_t j = 0; j < env.world; ++j)
        {
            if(j == env.rank)
                peers[j] = local;
            else if(hipIpcOpenMemHandle(&peers[j], all[j].handle, hipIpcMemLazyEnablePeerAccess)
                    != hipSuccess)
                openedAll = false;
        }

        const bool groupOpened
            = make_agreement(rendezvous, env.world).agree(openedAll, std::logical_and<>{});
        if(!groupOpened)
            hipblaslt_cerr << "error: hipIpcOpenMemHandle failed on at least one rank\n";
        return groupOpened;
    }
} // namespace hipblaslt_bench
