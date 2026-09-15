// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstdint>
#include <cstdlib>
#include <new>

#include <malloc.h>

extern "C" void* __libc_calloc(std::size_t, std::size_t);
extern "C" void __libc_cfree(void*);
extern "C" void __libc_free(void*);
extern "C" void* __libc_malloc(std::size_t);
extern "C" void* __libc_memalign(std::size_t, std::size_t);
extern "C" void* __libc_realloc(void*, std::size_t);
extern "C" int __posix_memalign(void**, std::size_t, std::size_t);

int main()
{
  try
  {
    if (auto p = std::aligned_alloc(8u, 42))
    {
      std::free(p);
    }
    if (auto p = std::calloc(1, 42))
    {
      std::free(p);
    }
    if (auto p = std::malloc(42))
    {
      std::free(p);
    }
    if (auto p = memalign(8, 42))
    {
      std::free(p);
    }
    if (void* p; posix_memalign(&p, 8, 42) == 0)
    {
      std::free(p);
    }
    if (auto p = std::realloc(std::malloc(42), 42))
    {
      std::free(p);
    }
    if (auto p = reallocarray(std::calloc(1, 42), 1, 42))
    {
      std::free(p);
    }
    if (auto p = new std::uint8_t)
    {
      delete p;
    }
    if (auto p = new (std::align_val_t{8}) std::uint8_t)
    {
      ::operator delete(p, std::align_val_t{8});
    }
    if (auto p = new (std::nothrow) std::uint8_t)
    {
      delete p;
    }
    if (auto p = new (std::align_val_t{8}, std::nothrow) std::uint8_t)
    {
      ::operator delete(p, std::align_val_t{8});
    }
    if (auto p = new std::uint8_t[42])
    {
      delete[] p;
    }
    if (auto p = new (std::align_val_t{8}) std::uint8_t[42])
    {
      ::operator delete[](p, std::align_val_t{8});
    }
    if (auto p = new (std::nothrow) std::uint8_t[42])
    {
      delete[] p;
    }
    if (auto p = new (std::align_val_t{8}, std::nothrow) std::uint8_t[42])
    {
      ::operator delete[](p, std::align_val_t{8});
    }
    if (auto p = __builtin_calloc(1, 42))
    {
      __builtin_free(p);
    }
    if (auto p = __builtin_malloc(42))
    {
      __builtin_free(p);
    }
    if (auto p = __builtin_operator_new(42))
    {
      __builtin_operator_delete(p);
    }
    if (auto p = __builtin_operator_new(42, std::align_val_t{8}))
    {
      __builtin_operator_delete(p, std::align_val_t{8});
    }
    if (auto p = __builtin_operator_new(42, std::nothrow))
    {
      __builtin_operator_delete(p);
    }
    if (auto p = __builtin_operator_new(42, std::align_val_t{8}, std::nothrow))
    {
      __builtin_operator_delete(p, std::align_val_t{8});
    }
    if (auto p = __builtin_realloc(__builtin_malloc(42), 41))
    {
      __builtin_free(p);
    }
    if (auto p = __libc_calloc(1, 42))
    {
      __libc_free(p);
    }
    if (auto p = __libc_malloc(42))
    {
      __libc_free(p);
    }
    if (auto p = __libc_memalign(8, 42))
    {
      __libc_free(p);
    }
  }
  catch (...)
  {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
