/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/detail/static_assert.h>

#include <string>
#undef THRUST_STATIC_ASSERT
#undef THRUST_STATIC_ASSERT_MSG

#define THRUST_STATIC_ASSERT(B)          unittest::assert_static((B), __FILE__, __LINE__);
#define THRUST_STATIC_ASSERT_MSG(B, msg) unittest::assert_static((B), __FILE__, __LINE__);

namespace unittest
{
THRUST_HOST_DEVICE void assert_static(bool condition, const char* filename, int lineno);
}

#include <thrust/detail/nv_target.h>
#include <thrust/device_delete.h>
#include <thrust/device_new.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP

#  define ASSERT_STATIC_ASSERT(X)                                                                        \
    {                                                                                                    \
      bool triggered                      = false;                                                       \
      using ex_t                          = unittest::static_assert_exception;                           \
      thrust::device_ptr<ex_t> device_ptr = thrust::device_new<ex_t>();                                  \
      ex_t* raw_ptr                       = thrust::raw_pointer_cast(device_ptr);                        \
      hipError_t err = ::hipMemcpyToSymbol(unittest::detail::device_exception, &raw_ptr, sizeof(ex_t*)); \
      if (err != hipSuccess)                                                                             \
      {                                                                                                  \
        thrust::device_free(device_ptr);                                                                 \
        raw_ptr = nullptr;                                                                               \
        unittest::UnitTestFailure f;                                                                     \
        f << "[" << __FILE__ << ":" << __LINE__ << "] hipMemcpyToSymbol failed";                         \
        throw f;                                                                                         \
      }                                                                                                  \
      try                                                                                                \
      {                                                                                                  \
        X;                                                                                               \
      }                                                                                                  \
      catch (ex_t)                                                                                       \
      {                                                                                                  \
        triggered = true;                                                                                \
      }                                                                                                  \
      if (!triggered)                                                                                    \
      {                                                                                                  \
        triggered = static_cast<ex_t>(*device_ptr).triggered;                                            \
      }                                                                                                  \
      thrust::device_free(device_ptr);                                                                   \
      raw_ptr = nullptr;                                                                                 \
      err     = ::hipMemcpyToSymbol(unittest::detail::device_exception, &raw_ptr, sizeof(ex_t*));        \
      if (err != hipSuccess)                                                                             \
      {                                                                                                  \
        unittest::UnitTestFailure f;                                                                     \
        f << "[" << __FILE__ << ":" << __LINE__ << "] hipMemcpyToSymbol failed";                         \
        throw f;                                                                                         \
      }                                                                                                  \
      if (!triggered)                                                                                    \
      {                                                                                                  \
        unittest::UnitTestFailure f;                                                                     \
        f << "[" << __FILE__ << ":" << __LINE__ << "] did not trigger a THRUST_STATIC_ASSERT";           \
        throw f;                                                                                         \
      }                                                                                                  \
    }

#else

#  define ASSERT_STATIC_ASSERT(X)                                                              \
    {                                                                                          \
      bool triggered = false;                                                                  \
      using ex_t     = unittest::static_assert_exception;                                      \
      try                                                                                      \
      {                                                                                        \
        X;                                                                                     \
      }                                                                                        \
      catch (ex_t)                                                                             \
      {                                                                                        \
        triggered = true;                                                                      \
      }                                                                                        \
      if (!triggered)                                                                          \
      {                                                                                        \
        unittest::UnitTestFailure f;                                                           \
        f << "[" << __FILE__ << ":" << __LINE__ << "] did not trigger a THRUST_STATIC_ASSERT"; \
        throw f;                                                                               \
      }                                                                                        \
    }

#endif

namespace unittest
{
class static_assert_exception
{
public:
  THRUST_HOST_DEVICE static_assert_exception()
      : triggered(false)
  {}

  THRUST_HOST_DEVICE static_assert_exception(const char* filename, int lineno)
      : triggered(true)
      , filename(filename)
      , lineno(lineno)
  {}

  bool triggered;
  const char* filename;
  int lineno;
};

namespace detail
{
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC || THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
__attribute__((used))
#endif
THRUST_DEVICE static static_assert_exception* device_exception = nullptr;
} // namespace detail

THRUST_HOST_DEVICE void assert_static(bool condition, const char* filename, int lineno)
{
  if (!condition)
  {
    static_assert_exception ex(filename, lineno);

    NV_IF_TARGET(NV_IS_DEVICE, (*detail::device_exception = ex;), (throw ex;));
  }
}
} // namespace unittest
