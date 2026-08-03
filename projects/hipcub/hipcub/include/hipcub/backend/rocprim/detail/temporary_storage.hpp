/*
 *  Copyright 2021 NVIDIA Corporation
 *  Modifications Copyright (c) 2026, Advanced Micro Devices, Inc.  All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include "../../../config.hpp"
#include "../../../util_temporary_storage.hpp"

#include _HIPCUB_LIBCXX_INCLUDE(stream)

BEGIN_HIPCUB_NAMESPACE

namespace detail::temporary_storage
{
template<typename MRT>
HIPCUB_RUNTIME_FUNCTION
hipError_t allocate(_HIPCUB_LIBCXX::stream_ref stream,
                    void*&                     d_temp_storage,
                    size_t                     temp_storage_bytes,
                    MRT&                       mr)
{
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (
                          try {
                              d_temp_storage = mr.allocate(stream, temp_storage_bytes);
                          } catch(...) { return hipErrorMemoryAllocation; }),
                      (d_temp_storage = mr.allocate(stream, temp_storage_bytes);));
    return hipSuccess;
}

template<typename MRT>
HIPCUB_RUNTIME_FUNCTION
hipError_t deallocate(_HIPCUB_LIBCXX::stream_ref stream,
                      void*                      d_temp_storage,
                      size_t                     temp_storage_bytes,
                      MRT&                       mr)
{
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (
                          try {
                              mr.deallocate(stream, d_temp_storage, temp_storage_bytes);
                          } catch(...) { return hipErrorMemoryAllocation; }),
                      (mr.deallocate(stream, d_temp_storage, temp_storage_bytes);));
    return hipSuccess;
}
} // namespace detail::temporary_storage

END_HIPCUB_NAMESPACE
