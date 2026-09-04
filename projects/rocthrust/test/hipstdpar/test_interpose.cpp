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

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>

#include <malloc.h>
#if defined(__HIPSTDPAR_INTERPOSE_ALLOC_CAN_MMAP__)
#  include <sys/mman.h>
#endif

extern "C" void* __libc_calloc(std::size_t, std::size_t);
extern "C" void __libc_cfree(void*);
extern "C" void __libc_free(void*);
extern "C" void* __libc_malloc(std::size_t);
extern "C" void* __libc_memalign(std::size_t, std::size_t);
extern "C" void* __libc_realloc(void*, std::size_t);
extern "C" int __posix_memalign(void**, std::size_t, std::size_t);

namespace
{
int new_handler_calls{};

void test_new_handler()
{
  ++new_handler_calls;
  std::set_new_handler(nullptr);
}

__attribute__((noinline)) void* runtime_memalign(std::size_t alignment, std::size_t size)
{
  return memalign(alignment, size);
}

__attribute__((noinline)) void runtime_free(void* p)
{
  std::free(p);
}

__attribute__((noinline)) void* runtime_operator_new(std::size_t size)
{
  return ::operator new(size);
}

__attribute__((noinline)) void* runtime_operator_new_aligned(std::size_t size, std::size_t alignment)
{
  return ::operator new(size, std::align_val_t{alignment});
}

__attribute__((noinline)) void runtime_operator_delete(void* p) noexcept
{
  ::operator delete(p);
}

__attribute__((noinline)) void runtime_operator_delete_sized(void* p, std::size_t size) noexcept
{
  ::operator delete(p, size);
}

__attribute__((noinline)) void
runtime_operator_delete_aligned_sized(void* p, std::size_t size, std::size_t alignment) noexcept
{
  ::operator delete(p, size, std::align_val_t{alignment});
}
} // namespace

int main()
{
  try
  {
    if (auto p = std::aligned_alloc(8u, 64))
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
    volatile std::size_t invalid_alignment = 3;
    errno = 0;
    if (auto p = runtime_memalign(invalid_alignment, 42))
    {
      std::free(p);
      return EXIT_FAILURE;
    }
    if (errno != EINVAL)
    {
      return EXIT_FAILURE;
    }
    {
      void* p = nullptr;
      if (posix_memalign(&p, 64, 42) != 0 || !p
          || reinterpret_cast<std::uintptr_t>(p) % 64 != 0)
      {
        return EXIT_FAILURE;
      }
      std::free(p);
    }
    {
      int sentinel{};
      void* p           = &sentinel;
      const auto result = posix_memalign(&p, 3, 42);
      if (result != EINVAL || p != &sentinel)
      {
        return EXIT_FAILURE;
      }
    }
    {
      int sentinel{};
      void* p = &sentinel;
      if (posix_memalign(&p, alignof(std::max_align_t), 0) != 0 || p != nullptr)
      {
        return EXIT_FAILURE;
      }
    }
    {
      int sentinel{};
      void* p = &sentinel;
      errno = EDOM;
      if (posix_memalign(&p, 64, (std::numeric_limits<std::size_t>::max)()) != ENOMEM
          || p != &sentinel || errno != EDOM)
      {
        return EXIT_FAILURE;
      }
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

    // Throwing allocation functions must report failure with std::bad_alloc.
    volatile std::size_t impossible_size = (std::numeric_limits<std::size_t>::max)();
    const auto previous_new_handler      = std::set_new_handler(test_new_handler);
    try
    {
      auto p = ::operator new(impossible_size);
      ::operator delete(p);
      return EXIT_FAILURE;
    }
    catch (const std::bad_alloc&)
    {}
    if (new_handler_calls != 1)
    {
      std::set_new_handler(previous_new_handler);
      return EXIT_FAILURE;
    }

    try
    {
      auto p = ::operator new(impossible_size, std::align_val_t{64});
      ::operator delete(p, std::align_val_t{64});
      return EXIT_FAILURE;
    }
    catch (const std::bad_alloc&)
    {}

    // Nothrow allocation functions must return nullptr rather than falling
    // through the exception handler.
    if (::operator new(impossible_size, std::nothrow) != nullptr
        || ::operator new(impossible_size, std::align_val_t{64}, std::nothrow) != nullptr)
    {
      std::set_new_handler(previous_new_handler);
      return EXIT_FAILURE;
    }
    std::set_new_handler(previous_new_handler);

    // operator new(0) must still return a distinct, deletable allocation.
    auto zero_sized_new = runtime_operator_new(0);
    if (!zero_sized_new)
    {
      return EXIT_FAILURE;
    }
    runtime_operator_delete(zero_sized_new);

    // Exercise sized and aligned-sized deallocation explicitly so optimizer
    // selection of a delete-expression overload cannot hide these paths.
    auto sized_new = runtime_operator_new(42);
    if (!sized_new)
    {
      return EXIT_FAILURE;
    }
    runtime_operator_delete_sized(sized_new, 42);

    auto aligned_sized_new = runtime_operator_new_aligned(42, 64);
    if (!aligned_sized_new || reinterpret_cast<std::uintptr_t>(aligned_sized_new) % 64 != 0)
    {
      return EXIT_FAILURE;
    }
    runtime_operator_delete_aligned_sized(aligned_sized_new, 42, 64);

    // free(nullptr) is required to be a no-op.
    void* volatile null_pointer = nullptr;
    runtime_free(null_pointer);

#if defined(__HIPSTDPAR_INTERPOSE_ALLOC_CAN_MMAP__)
    // mmap reports failure with MAP_FAILED, not nullptr.
    errno = 0;
    if (mmap(nullptr, 4096, PROT_READ, MAP_SHARED, -1, 0) != MAP_FAILED)
    {
      return EXIT_FAILURE;
    }

    auto mapping = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED)
    {
      return EXIT_FAILURE;
    }
    static_cast<unsigned char*>(mapping)[0] = 0x5a;
    if (munmap(mapping, 4096) != 0)
    {
      return EXIT_FAILURE;
    }
#endif

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
