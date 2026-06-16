/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "communicator.hpp"
#include "def.hpp"
#include "log_mpi.hpp"

#include <complex>

namespace rocalution
{
    template <>
    void communication_sync_exscan(int64_t* send, int64_t* recv, int count, const void* comm)
    {
        int status = MPI_Exscan(send, recv, count, MPI_INT64_T, MPI_SUM, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    // Allreduce single SUM - SYNC
    template <>
    void communication_sync_allreduce_single_sum(double* local, double* global, const void* comm)
    {
        int status = MPI_Allreduce(local, global, 1, MPI_DOUBLE, MPI_SUM, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_sync_allreduce_single_sum(float* local, float* global, const void* comm)
    {
        int status = MPI_Allreduce(local, global, 1, MPI_FLOAT, MPI_SUM, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

#ifdef SUPPORT_COMPLEX
    template <>
    void communication_sync_allreduce_single_sum(std::complex<double>* local,
                                                 std::complex<double>* global,
                                                 const void*           comm)
    {
        int status = MPI_Allreduce(local, global, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_sync_allreduce_single_sum(std::complex<float>* local,
                                                 std::complex<float>* global,
                                                 const void*          comm)
    {
        int status = MPI_Allreduce(local, global, 1, MPI_COMPLEX, MPI_SUM, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }
#endif

    template <>
    void communication_sync_allreduce_single_sum(int* local, int* global, const void* comm)
    {
        int status = MPI_Allreduce(local, global, 1, MPI_INT, MPI_SUM, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_sync_allreduce_single_sum(unsigned int* local,
                                                 unsigned int* global,
                                                 const void*   comm)
    {
        int status = MPI_Allreduce(local, global, 1, MPI_UNSIGNED, MPI_SUM, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_sync_allreduce_single_sum(int64_t* local, int64_t* global, const void* comm)
    {
        int status = MPI_Allreduce(local, global, 1, MPI_INT64_T, MPI_SUM, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    // Allreduce single SUM - ASYNC
    template <>
    void communication_async_allreduce_single_sum(double*     local,
                                                  double*     global,
                                                  const void* comm,
                                                  MRequest*   request)
    {
        int status = MPI_Iallreduce(
            local, global, 1, MPI_DOUBLE, MPI_SUM, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_allreduce_single_sum(float*      local,
                                                  float*      global,
                                                  const void* comm,
                                                  MRequest*   request)
    {
        int status
            = MPI_Iallreduce(local, global, 1, MPI_FLOAT, MPI_SUM, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

#ifdef SUPPORT_COMPLEX
    template <>
    void communication_async_allreduce_single_sum(std::complex<double>* local,
                                                  std::complex<double>* global,
                                                  const void*           comm,
                                                  MRequest*             request)
    {
        int status = MPI_Iallreduce(
            local, global, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_allreduce_single_sum(std::complex<float>* local,
                                                  std::complex<float>* global,
                                                  const void*          comm,
                                                  MRequest*            request)
    {
        int status = MPI_Iallreduce(
            local, global, 1, MPI_COMPLEX, MPI_SUM, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }
#endif

    template <>
    void communication_async_allreduce_single_sum(int*        local,
                                                  int*        global,
                                                  const void* comm,
                                                  MRequest*   request)
    {
        int status
            = MPI_Iallreduce(local, global, 1, MPI_INT, MPI_SUM, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_allreduce_single_sum(unsigned int* local,
                                                  unsigned int* global,
                                                  const void*   comm,
                                                  MRequest*     request)
    {
        int status = MPI_Iallreduce(
            local, global, 1, MPI_UNSIGNED, MPI_SUM, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_allreduce_single_sum(int64_t*    local,
                                                  int64_t*    global,
                                                  const void* comm,
                                                  MRequest*   request)
    {
        int status = MPI_Iallreduce(
            local, global, 1, MPI_INT64_T, MPI_SUM, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    // Allreduce single MAX - SYNC
    template <>
    void communication_sync_allreduce_single_max(double* local, double* global, const void* comm)
    {
        int status = MPI_Allreduce(local, global, 1, MPI_DOUBLE, MPI_MAX, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_sync_allreduce_single_max(float* local, float* global, const void* comm)
    {
        int status = MPI_Allreduce(local, global, 1, MPI_FLOAT, MPI_MAX, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

#ifdef SUPPORT_COMPLEX
    template <>
    void communication_sync_allreduce_single_max(std::complex<double>* local,
                                                 std::complex<double>* global,
                                                 const void*           comm)
    {
        int status = MPI_Allreduce(local, global, 1, MPI_DOUBLE_COMPLEX, MPI_MAX, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_sync_allreduce_single_max(std::complex<float>* local,
                                                 std::complex<float>* global,
                                                 const void*          comm)
    {
        int status = MPI_Allreduce(local, global, 1, MPI_COMPLEX, MPI_MAX, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }
#endif

    template <>
    void communication_sync_allreduce_single_max(int* local, int* global, const void* comm)
    {
        int status = MPI_Allreduce(local, global, 1, MPI_INT, MPI_MAX, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_sync_allreduce_single_max(unsigned int* local,
                                                 unsigned int* global,
                                                 const void*   comm)
    {
        int status = MPI_Allreduce(local, global, 1, MPI_UNSIGNED, MPI_MAX, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_sync_allreduce_single_max(int64_t* local, int64_t* global, const void* comm)
    {
        int status = MPI_Allreduce(local, global, 1, MPI_INT64_T, MPI_MAX, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    // Allreduce single MAX - ASYNC
    template <>
    void communication_async_allreduce_single_max(double*     local,
                                                  double*     global,
                                                  const void* comm,
                                                  MRequest*   request)
    {
        int status = MPI_Iallreduce(
            local, global, 1, MPI_DOUBLE, MPI_MAX, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_allreduce_single_max(float*      local,
                                                  float*      global,
                                                  const void* comm,
                                                  MRequest*   request)
    {
        int status
            = MPI_Iallreduce(local, global, 1, MPI_FLOAT, MPI_MAX, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

#ifdef SUPPORT_COMPLEX
    template <>
    void communication_async_allreduce_single_max(std::complex<double>* local,
                                                  std::complex<double>* global,
                                                  const void*           comm,
                                                  MRequest*             request)
    {
        int status = MPI_Iallreduce(
            local, global, 1, MPI_DOUBLE_COMPLEX, MPI_MAX, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_allreduce_single_max(std::complex<float>* local,
                                                  std::complex<float>* global,
                                                  const void*          comm,
                                                  MRequest*            request)
    {
        int status = MPI_Iallreduce(
            local, global, 1, MPI_COMPLEX, MPI_MAX, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }
#endif

    template <>
    void communication_async_allreduce_single_max(int*        local,
                                                  int*        global,
                                                  const void* comm,
                                                  MRequest*   request)
    {
        int status
            = MPI_Iallreduce(local, global, 1, MPI_INT, MPI_MAX, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_allreduce_single_max(unsigned int* local,
                                                  unsigned int* global,
                                                  const void*   comm,
                                                  MRequest*     request)
    {
        int status = MPI_Iallreduce(
            local, global, 1, MPI_UNSIGNED, MPI_MAX, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_allreduce_single_max(int64_t*    local,
                                                  int64_t*    global,
                                                  const void* comm,
                                                  MRequest*   request)
    {
        int status = MPI_Iallreduce(
            local, global, 1, MPI_INT64_T, MPI_MAX, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    // AlltoAll single - SYNC
    template <>
    void communication_sync_alltoall_single(double* send, double* recv, const void* comm)
    {
        int status = MPI_Alltoall(send, 1, MPI_DOUBLE, recv, 1, MPI_DOUBLE, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_sync_alltoall_single(float* send, float* recv, const void* comm)
    {
        int status = MPI_Alltoall(send, 1, MPI_FLOAT, recv, 1, MPI_FLOAT, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

#ifdef SUPPORT_COMPLEX
    template <>
    void communication_sync_alltoall_single(std::complex<double>* send,
                                            std::complex<double>* recv,
                                            const void*           comm)
    {
        int status = MPI_Alltoall(
            send, 1, MPI_DOUBLE_COMPLEX, recv, 1, MPI_DOUBLE_COMPLEX, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_sync_alltoall_single(std::complex<float>* send,
                                            std::complex<float>* recv,
                                            const void*          comm)
    {
        int status = MPI_Alltoall(send, 1, MPI_COMPLEX, recv, 1, MPI_COMPLEX, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }
#endif

    template <>
    void communication_sync_alltoall_single(int* send, int* recv, const void* comm)
    {
        int status = MPI_Alltoall(send, 1, MPI_INT, recv, 1, MPI_INT, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void
        communication_sync_alltoall_single(unsigned int* send, unsigned int* recv, const void* comm)
    {
        int status = MPI_Alltoall(send, 1, MPI_UNSIGNED, recv, 1, MPI_UNSIGNED, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_sync_alltoall_single(int64_t* send, int64_t* recv, const void* comm)
    {
        int status = MPI_Alltoall(send, 1, MPI_INT64_T, recv, 1, MPI_INT64_T, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    // AlltoAll single - ASYNC
    template <>
    void communication_async_alltoall_single(double*     send,
                                             double*     recv,
                                             const void* comm,
                                             MRequest*   request)
    {
        int status = MPI_Ialltoall(
            send, 1, MPI_DOUBLE, recv, 1, MPI_DOUBLE, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_alltoall_single(float*      send,
                                             float*      recv,
                                             const void* comm,
                                             MRequest*   request)
    {
        int status = MPI_Ialltoall(
            send, 1, MPI_FLOAT, recv, 1, MPI_FLOAT, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

#ifdef SUPPORT_COMPLEX
    template <>
    void communication_async_alltoall_single(std::complex<double>* send,
                                             std::complex<double>* recv,
                                             const void*           comm,
                                             MRequest*             request)
    {
        int status = MPI_Ialltoall(send,
                                   1,
                                   MPI_DOUBLE_COMPLEX,
                                   recv,
                                   1,
                                   MPI_DOUBLE_COMPLEX,
                                   *(MPI_Comm*)comm,
                                   &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_alltoall_single(std::complex<float>* send,
                                             std::complex<float>* recv,
                                             const void*          comm,
                                             MRequest*            request)
    {
        int status = MPI_Ialltoall(
            send, 1, MPI_COMPLEX, recv, 1, MPI_COMPLEX, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }
#endif

    template <>
    void communication_async_alltoall_single(int*        send,
                                             int*        recv,
                                             const void* comm,
                                             MRequest*   request)
    {
        int status
            = MPI_Ialltoall(send, 1, MPI_INT, recv, 1, MPI_INT, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_alltoall_single(unsigned int* send,
                                             unsigned int* recv,
                                             const void*   comm,
                                             MRequest*     request)
    {
        int status = MPI_Ialltoall(
            send, 1, MPI_UNSIGNED, recv, 1, MPI_UNSIGNED, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_alltoall_single(int64_t*    send,
                                             int64_t*    recv,
                                             const void* comm,
                                             MRequest*   request)
    {
        int status = MPI_Ialltoall(
            send, 1, MPI_INT64_T, recv, 1, MPI_INT64_T, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    // Allgather single - SYNC
    template <>
    void communication_sync_allgather_single(double* send, double* recv, const void* comm)
    {
        int status = MPI_Allgather(send, 1, MPI_DOUBLE, recv, 1, MPI_DOUBLE, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_sync_allgather_single(float* send, float* recv, const void* comm)
    {
        int status = MPI_Allgather(send, 1, MPI_FLOAT, recv, 1, MPI_FLOAT, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

#ifdef SUPPORT_COMPLEX
    template <>
    void communication_sync_allgather_single(std::complex<double>* send,
                                             std::complex<double>* recv,
                                             const void*           comm)
    {
        int status = MPI_Allgather(
            send, 1, MPI_DOUBLE_COMPLEX, recv, 1, MPI_DOUBLE_COMPLEX, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_sync_allgather_single(std::complex<float>* send,
                                             std::complex<float>* recv,
                                             const void*          comm)
    {
        int status = MPI_Allgather(send, 1, MPI_COMPLEX, recv, 1, MPI_COMPLEX, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }
#endif

    template <>
    void communication_sync_allgather_single(int* send, int* recv, const void* comm)
    {
        int status = MPI_Allgather(send, 1, MPI_INT, recv, 1, MPI_INT, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_sync_allgather_single(unsigned int* send,
                                             unsigned int* recv,
                                             const void*   comm)
    {
        int status = MPI_Allgather(send, 1, MPI_UNSIGNED, recv, 1, MPI_UNSIGNED, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_sync_allgather_single(int64_t* send, int64_t* recv, const void* comm)
    {
        int status = MPI_Allgather(send, 1, MPI_INT64_T, recv, 1, MPI_INT64_T, *(MPI_Comm*)comm);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    // Allgather single - ASYNC
    template <>
    void communication_async_allgather_single(double*     send,
                                              double*     recv,
                                              MRequest*   request,
                                              const void* comm)
    {
        int status = MPI_Iallgather(
            send, 1, MPI_DOUBLE, recv, 1, MPI_DOUBLE, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_allgather_single(float*      send,
                                              float*      recv,
                                              MRequest*   request,
                                              const void* comm)
    {
        int status = MPI_Iallgather(
            send, 1, MPI_FLOAT, recv, 1, MPI_FLOAT, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

#ifdef SUPPORT_COMPLEX
    template <>
    void communication_async_allgather_single(std::complex<double>* send,
                                              std::complex<double>* recv,
                                              MRequest*             request,
                                              const void*           comm)
    {
        int status = MPI_Iallgather(send,
                                    1,
                                    MPI_DOUBLE_COMPLEX,
                                    recv,
                                    1,
                                    MPI_DOUBLE_COMPLEX,
                                    *(MPI_Comm*)comm,
                                    &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_allgather_single(std::complex<float>* send,
                                              std::complex<float>* recv,
                                              MRequest*            request,
                                              const void*          comm)
    {
        int status = MPI_Iallgather(
            send, 1, MPI_COMPLEX, recv, 1, MPI_COMPLEX, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }
#endif

    template <>
    void communication_async_allgather_single(int*        send,
                                              int*        recv,
                                              MRequest*   request,
                                              const void* comm)
    {
        int status
            = MPI_Iallgather(send, 1, MPI_INT, recv, 1, MPI_INT, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_allgather_single(unsigned int* send,
                                              unsigned int* recv,
                                              MRequest*     request,
                                              const void*   comm)
    {
        int status = MPI_Iallgather(
            send, 1, MPI_UNSIGNED, recv, 1, MPI_UNSIGNED, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_allgather_single(int64_t*    send,
                                              int64_t*    recv,
                                              MRequest*   request,
                                              const void* comm)
    {
        int status = MPI_Iallgather(
            send, 1, MPI_INT64_T, recv, 1, MPI_INT64_T, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    // Receive - ASYNC
    template <>
    void communication_async_recv(
        double* buf, int count, int source, int tag, MRequest* request, const void* comm)
    {
        int status
            = MPI_Irecv(buf, count, MPI_DOUBLE, source, tag, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_recv(
        float* buf, int count, int source, int tag, MRequest* request, const void* comm)
    {
        int status = MPI_Irecv(buf, count, MPI_FLOAT, source, tag, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

#ifdef SUPPORT_COMPLEX
    template <>
    void communication_async_recv(std::complex<double>* buf,
                                  int                   count,
                                  int                   source,
                                  int                   tag,
                                  MRequest*             request,
                                  const void*           comm)
    {
        int status = MPI_Irecv(
            buf, count, MPI_DOUBLE_COMPLEX, source, tag, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_recv(std::complex<float>* buf,
                                  int                  count,
                                  int                  source,
                                  int                  tag,
                                  MRequest*            request,
                                  const void*          comm)
    {
        int status
            = MPI_Irecv(buf, count, MPI_COMPLEX, source, tag, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }
#endif

    template <>
    void communication_async_recv(
        bool* buf, int count, int source, int tag, MRequest* request, const void* comm)
    {
        int status
            = MPI_Irecv(buf, count, MPI_CXX_BOOL, source, tag, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_recv(
        int* buf, int count, int source, int tag, MRequest* request, const void* comm)
    {
        int status = MPI_Irecv(buf, count, MPI_INT, source, tag, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_recv(
        int64_t* buf, int count, int source, int tag, MRequest* request, const void* comm)
    {
        int status
            = MPI_Irecv(buf, count, MPI_INT64_T, source, tag, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    // Send - ASYNC
    template <>
    void communication_async_send(
        double* buf, int count, int dest, int tag, MRequest* request, const void* comm)
    {
        int status = MPI_Isend(buf, count, MPI_DOUBLE, dest, tag, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_send(
        float* buf, int count, int dest, int tag, MRequest* request, const void* comm)
    {
        int status = MPI_Isend(buf, count, MPI_FLOAT, dest, tag, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

#ifdef SUPPORT_COMPLEX
    template <>
    void communication_async_send(std::complex<double>* buf,
                                  int                   count,
                                  int                   dest,
                                  int                   tag,
                                  MRequest*             request,
                                  const void*           comm)
    {
        int status
            = MPI_Isend(buf, count, MPI_DOUBLE_COMPLEX, dest, tag, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_send(
        std::complex<float>* buf, int count, int dest, int tag, MRequest* request, const void* comm)
    {
        int status = MPI_Isend(buf, count, MPI_COMPLEX, dest, tag, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }
#endif

    template <>
    void communication_async_send(
        bool* buf, int count, int dest, int tag, MRequest* request, const void* comm)
    {
        int status
            = MPI_Isend(buf, count, MPI_CXX_BOOL, dest, tag, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_send(
        int* buf, int count, int dest, int tag, MRequest* request, const void* comm)
    {
        int status = MPI_Isend(buf, count, MPI_INT, dest, tag, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    template <>
    void communication_async_send(
        int64_t* buf, int count, int dest, int tag, MRequest* request, const void* comm)
    {
        int status = MPI_Isend(buf, count, MPI_INT64_T, dest, tag, *(MPI_Comm*)comm, &request->req);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    // Synchronization
    void communication_sync(MRequest* request)
    {
        int status = MPI_Wait(&request->req, MPI_STATUSES_IGNORE);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

    void communication_syncall(int count, MRequest* requests)
    {
        int status = MPI_Waitall(count, &requests->req, MPI_STATUSES_IGNORE);
        CHECK_MPI_ERROR(status, __FILE__, __LINE__);
    }

} // namespace rocalution
