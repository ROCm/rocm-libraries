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

#ifndef ROCALUTION_UTILS_COMMUNICATOR_HPP_
#define ROCALUTION_UTILS_COMMUNICATOR_HPP_

#include <mpi.h>

namespace rocalution
{
    struct MRequest
    {
        MPI_Request req;
    };

    template <typename ValueType>
    void communication_sync_exscan(ValueType* send, ValueType* recv, int count, const void* comm);

    template <typename ValueType>
    void communication_sync_allreduce_single_sum(ValueType*  local,
                                                 ValueType*  global,
                                                 const void* comm);

    template <typename ValueType>
    void communication_async_allreduce_single_sum(ValueType*  local,
                                                  ValueType*  global,
                                                  const void* comm,
                                                  MRequest*   request);

    template <typename ValueType>
    void communication_sync_allreduce_single_max(ValueType*  local,
                                                 ValueType*  global,
                                                 const void* comm);

    template <typename ValueType>
    void communication_async_allreduce_single_max(ValueType*  local,
                                                  ValueType*  global,
                                                  const void* comm,
                                                  MRequest*   request);

    template <typename ValueType>
    void communication_sync_alltoall_single(ValueType* send, ValueType* recv, const void* comm);

    template <typename ValueType>
    void communication_async_alltoall_single(ValueType*  send,
                                             ValueType*  recv,
                                             const void* comm,
                                             MRequest*   request);

    template <typename ValueType>
    void communication_sync_allgather_single(ValueType* send, ValueType* recv, const void* comm);

    template <typename ValueType>
    void communication_async_allgather_single(ValueType*  send,
                                              ValueType*  recv,
                                              MRequest*   request,
                                              const void* comm);

    template <typename ValueType>
    void communication_async_recv(
        ValueType* buf, int count, int source, int tag, MRequest* request, const void* comm);

    template <typename ValueType>
    void communication_async_send(
        ValueType* buf, int count, int dest, int tag, MRequest* request, const void* comm);

    void communication_sync(MRequest* request);
    void communication_syncall(int count, MRequest* requests);

} // namespace rocalution

#endif // ROCALUTION_UTILS_COMMUNICATOR_HPP_
