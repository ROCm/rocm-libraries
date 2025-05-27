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

#    include <cstddef>
#    include <cstdint>
#    include <cstring>
#    include <memory_resource>
#    include <new>
#    include <stdexcept>

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
      // TODO: add exception handling
      void* do_allocate(std::size_t n, std::size_t a) override
      {
        void* r{};
        hipMallocManaged(&r, n);

        return r;
      }

      void do_deallocate(void* p, std::size_t, std::size_t) override
      {
        hipFree(p);
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
{ // TODO: tidy up, revert to using std.
  auto m = n + sizeof(hipstd::Header) + a - 1;

  auto r = hipstd::heap.allocate(m, a);

  if (!r)
  {
    return r;
  }

  const auto h                             = static_cast<hipstd::Header*>(r) + 1;
  const auto p                             = (reinterpret_cast<std::uintptr_t>(h) + a - 1) & -a;
  reinterpret_cast<hipstd::Header*>(p)[-1] = {r, m, a};

  return reinterpret_cast<void*>(p);
}

extern "C" inline __attribute__((used)) void* __hipstdpar_malloc(std::size_t n)
{
  constexpr auto a = alignof(std::max_align_t);

  return __hipstdpar_aligned_alloc(a, n);
}

extern "C" inline __attribute__((used)) void* __hipstdpar_calloc(std::size_t n, std::size_t sz)
{
  return std::memset(__hipstdpar_malloc(n * sz), 0, n * sz);
}

extern "C" inline __attribute__((used)) int __hipstdpar_posix_aligned_alloc(void** p, std::size_t a, std::size_t n)
{ // TODO: check invariants on alignment
  if (!p || n == 0)
  {
    return 0;
  }

  *p = __hipstdpar_aligned_alloc(a, n);

  return 1;
}

extern "C" __attribute__((weak)) void __hipstdpar_hidden_free(void*);

extern "C" inline __attribute__((used)) void* __hipstdpar_realloc(void* p, std::size_t n)
{
  auto q = std::memcpy(__hipstdpar_malloc(n), p, n);

  auto h = static_cast<hipstd::Header*>(p) - 1;

  hipPointerAttribute_t tmp{};
  auto r = hipPointerGetAttributes(&tmp, h);

  if (!tmp.isManaged)
  {
    __hipstdpar_hidden_free(p);
  }
  else
  {
    hipstd::heap.deallocate(h->alloc_ptr, h->size, h->align);
  }

  return q;
}

extern "C" inline __attribute__((used)) void* __hipstdpar_realloc_array(void* p, std::size_t n, std::size_t sz)
{ // TODO: handle overflow in n * sz gracefully, as per spec.
  return __hipstdpar_realloc(p, n * sz);
}

extern "C" inline __attribute__((used)) void __hipstdpar_free(void* p)
{
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
  if (auto p = __hipstdpar_aligned_alloc(a, n))
  {
    return p;
  }

  throw std::runtime_error{"Failed __hipstdpar_operator_new_aligned"};
}

extern "C" inline __attribute__((used)) void* __hipstdpar_operator_new(std::size_t n)
{ // TODO: consider adding the special handling for operator new
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
    // TODO: handle the potential exception
  }
}

extern "C" inline __attribute__((used)) void*
__hipstdpar_operator_new_aligned_nothrow(std::size_t n, std::size_t a, std::nothrow_t) noexcept
{ // TODO: consider adding the special handling for operator new
  try
  {
    return __hipstdpar_operator_new_aligned(n, a);
  }
  catch (...)
  {
    // TODO: handle the potential exception.
  }
}

extern "C" inline __attribute__((used)) void
__hipstdpar_operator_delete_aligned_sized(void* p, std::size_t n, std::size_t a) noexcept
{
  hipPointerAttribute_t tmp{};
  auto r = hipPointerGetAttributes(&tmp, p);

  if (!tmp.isManaged)
  {
    return __hipstdpar_hidden_free(p);
  }

  return hipstd::heap.deallocate(p, n, a);
}

extern "C" inline __attribute__((used)) void __hipstdpar_operator_delete(void* p) noexcept
{
  return __hipstdpar_free(p);
}

extern "C" inline __attribute__((used)) void __hipstdpar_operator_delete_aligned(void* p, std::size_t) noexcept
{ // TODO: use alignment
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
