// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*
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

/*! \file thrust/system/hip/interpose_allocations.hpp
 *  \brief Interposed allocations/deallocations implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)
#  if defined(__HIPSTDPAR_INTERPOSE_ALLOC__)
#    include <hip/hip_runtime.h>

#    include <algorithm>
#    include <cerrno>
#    include <cstddef>
#    include <cstdint>
#    include <cstring>
#    include <limits>
#    include <memory>
#    include <memory_resource>
#    include <new>

#    if __has_include(<malloc.h>)
#      include <malloc.h>
#      define __HIPSTDPAR_HAS_MALLOC_USABLE_SIZE__
#    endif

namespace hipstd
{
struct Header
{
  void* alloc_ptr;
  std::size_t size;
  std::size_t align;
};

inline std::pmr::synchronized_pool_resource heap{
  std::pmr::pool_options{0u, 15u * 1024u}, []() {
    static class final : public std::pmr::memory_resource
    {
      void* do_allocate(std::size_t n, std::size_t a) override
      {
        void* r{};
        if (hipMallocManaged(&r, n) != hipSuccess || !r)
        {
          throw std::bad_alloc{};
        }

        return r;
      }

      void do_deallocate(void* p, std::size_t, std::size_t) override
      {
        static_cast<void>(hipFree(p));
      }

      bool do_is_equal(const std::pmr::memory_resource& x) const noexcept override
      {
        return dynamic_cast<const decltype(this)>(&x);
      }
    } r;

    return &r;
  }()};
} // Namespace hipstd.

extern "C" inline __attribute__((used)) void* __hipstdpar_aligned_alloc(std::size_t a, std::size_t n)
{
  constexpr auto max_size             = (std::numeric_limits<std::size_t>::max)();
  constexpr auto header_size          = sizeof(hipstd::Header);
  constexpr auto allocation_alignment = alignof(hipstd::Header);

  if (a == 0 || (a & (a - 1)) != 0)
  {
    errno = EINVAL;
    return nullptr;
  }

  const auto padding = a - 1;
  if (padding > max_size - header_size || n > max_size - header_size - padding)
  {
    errno = ENOMEM;
    return nullptr;
  }

  const auto allocation_size = header_size + n + padding;

  void* allocation{};
  try
  {
    allocation = hipstd::heap.allocate(allocation_size, allocation_alignment);
  }
  catch (...)
  {
    // Rewritten C allocation calls must not let exceptions escape.
    errno = ENOMEM;
    return nullptr;
  }

  void* aligned = static_cast<std::byte*>(allocation) + header_size;
  auto space    = allocation_size - header_size;
  if (!std::align(a, n, aligned, space))
  {
    hipstd::heap.deallocate(allocation, allocation_size, allocation_alignment);
    errno = ENOMEM;
    return nullptr;
  }

  static_cast<hipstd::Header*>(aligned)[-1] = {allocation, allocation_size, allocation_alignment};

  return aligned;
}

extern "C" inline __attribute__((used)) void* __hipstdpar_malloc(std::size_t n)
{
  constexpr auto a = alignof(std::max_align_t);

  return __hipstdpar_aligned_alloc(a, n);
}

extern "C" inline __attribute__((used)) void* __hipstdpar_calloc(std::size_t n, std::size_t sz)
{
  constexpr auto max_size = (std::numeric_limits<std::size_t>::max)();
  if (sz != 0 && n > max_size / sz)
  {
    errno = ENOMEM;
    return nullptr;
  }

  const auto bytes = n * sz;
  auto p           = __hipstdpar_malloc(bytes);
  if (!p)
  {
    // A zero-sized request may return nullptr; nullptr alone does not imply ENOMEM.
    if (bytes != 0)
    {
      errno = ENOMEM;
    }
    return nullptr;
  }

  return std::memset(p, 0, bytes);
}

extern "C" inline __attribute__((used)) int __hipstdpar_posix_aligned_alloc(void** p, std::size_t a, std::size_t n)
{
  if (!p || a < sizeof(void*) || (a & (a - 1)) != 0)
  {
    return EINVAL;
  }

  if (n == 0)
  {
    *p = nullptr;
    return 0;
  }

  const auto saved_errno = errno;
  auto allocation        = __hipstdpar_aligned_alloc(a, n);
  errno                  = saved_errno;
  if (!allocation)
  {
    return ENOMEM;
  }

  *p = allocation;
  return 0;
}

extern "C" __attribute__((weak)) void __hipstdpar_hidden_free(void*);

// Declared ahead of __hipstdpar_realloc, which frees through it below.
extern "C" inline __attribute__((used)) void __hipstdpar_free(void*);

extern "C" inline __attribute__((used)) void* __hipstdpar_realloc(void* p, std::size_t n)
{
  if (!p)
  {
    return __hipstdpar_malloc(n);
  }

  if (n == 0)
  {
    __hipstdpar_free(p);
    return nullptr;
  }

  auto q = __hipstdpar_malloc(n);
  if (!q)
  {
    return nullptr;
  }

  auto h = static_cast<hipstd::Header*>(p) - 1;

  hipPointerAttribute_t tmp{};
  auto r = hipPointerGetAttributes(&tmp, h);

  if (!tmp.isManaged)
  {
    std::size_t old = n;
#    if defined(__HIPSTDPAR_HAS_MALLOC_USABLE_SIZE__)
    old = malloc_usable_size(p);
#    endif
    std::memcpy(q, p, std::min(old, n));
    __hipstdpar_hidden_free(p);
  }
  else
  {
    const auto old = reinterpret_cast<std::uintptr_t>(h->alloc_ptr) + h->size
                     - reinterpret_cast<std::uintptr_t>(p);
    std::memcpy(q, p, std::min<std::size_t>(old, n));
    hipstd::heap.deallocate(h->alloc_ptr, h->size, h->align);
  }

  return q;
}

extern "C" inline __attribute__((used)) void* __hipstdpar_realloc_array(void* p, std::size_t n, std::size_t sz)
{
  // Checked before reallocating: a wrapped product of zero would be taken as a
  // request to free p, leaving the caller holding a dangling pointer.
  constexpr auto max_size = (std::numeric_limits<std::size_t>::max)();
  if (sz != 0 && n > max_size / sz)
  {
    errno = ENOMEM;
    return nullptr;
  }

  return __hipstdpar_realloc(p, n * sz);
}

extern "C" inline __attribute__((used)) void __hipstdpar_free(void* p)
{
  if (!p)
  {
    return;
  }

  auto h = static_cast<hipstd::Header*>(p) - 1;

  hipPointerAttribute_t tmp{};
  auto r = hipPointerGetAttributes(&tmp, h);

  if (!tmp.isManaged)
  {
    return __hipstdpar_hidden_free(p);
  }

  return hipstd::heap.deallocate(h->alloc_ptr, h->size, h->align);
}

extern "C" inline __attribute__((used)) void* __hipstdpar_operator_new_aligned(std::size_t n, std::size_t a)
{
  const auto allocation_size = n == 0 ? 1 : n;
  while (true)
  {
    if (auto p = __hipstdpar_aligned_alloc(a, allocation_size))
    {
      return p;
    }

    if (auto handler = std::get_new_handler())
    {
      handler();
    }
    else
    {
      throw std::bad_alloc{};
    }
  }
}

extern "C" inline __attribute__((used)) void* __hipstdpar_operator_new(std::size_t n)
{
  return __hipstdpar_operator_new_aligned(n, alignof(std::max_align_t));
}

extern "C" inline __attribute__((used)) void* __hipstdpar_operator_new_nothrow(std::size_t n, std::nothrow_t) noexcept
{
  try
  {
    return __hipstdpar_operator_new(n);
  }
  catch (...)
  {
    return nullptr;
  }
}

extern "C" inline __attribute__((used)) void*
__hipstdpar_operator_new_aligned_nothrow(std::size_t n, std::size_t a, std::nothrow_t) noexcept
{
  try
  {
    return __hipstdpar_operator_new_aligned(n, a);
  }
  catch (...)
  {
    return nullptr;
  }
}

extern "C" inline __attribute__((used)) void
__hipstdpar_operator_delete_aligned_sized(void* p, std::size_t, std::size_t) noexcept
{
  return __hipstdpar_free(p);
}

extern "C" inline __attribute__((used)) void __hipstdpar_operator_delete(void* p) noexcept
{
  return __hipstdpar_free(p);
}

extern "C" inline __attribute__((used)) void __hipstdpar_operator_delete_aligned(void* p, std::size_t) noexcept
{
  return __hipstdpar_free(p);
}

extern "C" inline __attribute__((used)) void __hipstdpar_operator_delete_sized(void* p, std::size_t n) noexcept
{
  return __hipstdpar_operator_delete_aligned_sized(p, n, alignof(std::max_align_t));
}
#  else // __HIPSTDPAR_INTERPOSE_ALLOC__
#    error \
      "__HIPSTDPAR_INTERPOSE_ALLOC__ should be defined. Please use the '--hipstdpar-interpose-alloc' compile option."
#  endif // __HIPSTDPAR_INTERPOSE_ALLOC__

#else // __HIPSTDPAR__
#  error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
