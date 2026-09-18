// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

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
#include <string>
#include <vector>

namespace hipblaslt_bench
{
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
} // namespace hipblaslt_bench
