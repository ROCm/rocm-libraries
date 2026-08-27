// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Regression test for __hipstdpar_realloc, which is what std::realloc resolves
// to when translation units are compiled with --hipstdpar-interpose-alloc.
//
// The interposer used to copy n (the requested new size) bytes out of the old
// block unconditionally. Growing an allocation therefore over-read past the end
// of the old block, and realloc(nullptr, n) fed a NULL pointer straight to
// memcpy. This test exercises the fixed contract:
//   * realloc(nullptr, n) behaves like malloc(n),
//   * realloc(p, 0)       frees p and returns nullptr,
//   * growing preserves every byte of the old block (no over-read),
//   * shrinking preserves the retained prefix,
//   * reallocarray rejects an overflowing product and leaves the block intact.
//
// Interposed allocations are backed by hipMallocManaged, so the storage is
// host-accessible and the payload can be validated directly on the host.

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace
{
__attribute__((noinline)) void* runtime_malloc(std::size_t size)
{
  return std::malloc(size);
}

__attribute__((noinline)) void* runtime_realloc(void* p, std::size_t size)
{
  return std::realloc(p, size);
}

__attribute__((noinline)) void* runtime_reallocarray(void* p, std::size_t count, std::size_t size)
{
  return reallocarray(p, count, size);
}

// Deterministic, size-dependent fill so a stale/short copy is detectable.
std::uint8_t pattern(std::size_t i)
{
  return static_cast<std::uint8_t>((i * 31u + 7u) & 0xffu);
}

void fill(std::uint8_t* p, std::size_t n)
{
  for (std::size_t i = 0; i < n; ++i)
  {
    p[i] = pattern(i);
  }
}

bool verify(const std::uint8_t* p, std::size_t n)
{
  for (std::size_t i = 0; i < n; ++i)
  {
    if (p[i] != pattern(i))
    {
      return false;
    }
  }
  return true;
}
} // namespace

int main()
{
  try
  {
    // realloc(nullptr, n) must behave like malloc(n).
    {
      auto p = static_cast<std::uint8_t*>(runtime_realloc(nullptr, 64));
      if (!p)
      {
        return EXIT_FAILURE;
      }
      fill(p, 64);
      if (!verify(p, 64))
      {
        return EXIT_FAILURE;
      }
      std::free(p);
    }

    // realloc(p, 0) must free p and return nullptr.
    {
      auto p = runtime_malloc(64);
      if (!p)
      {
        return EXIT_FAILURE;
      }
      if (runtime_realloc(p, 0) != nullptr)
      {
        return EXIT_FAILURE;
      }
    }

    // Growing must preserve the whole old block without over-reading past it.
    {
      constexpr std::size_t old_size = 32;
      constexpr std::size_t new_size = 8192;

      auto p = static_cast<std::uint8_t*>(runtime_malloc(old_size));
      if (!p)
      {
        return EXIT_FAILURE;
      }
      fill(p, old_size);

      auto q = static_cast<std::uint8_t*>(runtime_realloc(p, new_size));
      if (!q)
      {
        return EXIT_FAILURE;
      }
      if (!verify(q, old_size))
      {
        std::free(q);
        return EXIT_FAILURE;
      }
      std::free(q);
    }

    // Shrinking must preserve the retained prefix.
    {
      constexpr std::size_t old_size = 8192;
      constexpr std::size_t new_size = 32;

      auto p = static_cast<std::uint8_t*>(runtime_malloc(old_size));
      if (!p)
      {
        return EXIT_FAILURE;
      }
      fill(p, old_size);

      auto q = static_cast<std::uint8_t*>(runtime_realloc(p, new_size));
      if (!q)
      {
        return EXIT_FAILURE;
      }
      if (!verify(q, new_size))
      {
        std::free(q);
        return EXIT_FAILURE;
      }
      std::free(q);
    }

    // An overflowing reallocarray must be rejected. Left unchecked, the wrapped
    // product is passed to realloc, which either shrinks the block to a few
    // bytes and reports success or, when it wraps to zero, frees the block and
    // hands the caller back a dangling pointer.
    {
      constexpr std::size_t old_size = 64;
      constexpr auto max_size        = (std::numeric_limits<std::size_t>::max)();
      const auto wrap_to_two         = max_size / 2 + 2;

      auto p = static_cast<std::uint8_t*>(runtime_malloc(old_size));
      if (!p)
      {
        return EXIT_FAILURE;
      }
      fill(p, old_size);

      errno = 0;
      if (auto q = runtime_reallocarray(p, wrap_to_two, 2))
      {
        std::free(q);
        return EXIT_FAILURE;
      }
      if (errno != ENOMEM)
      {
        std::free(p);
        return EXIT_FAILURE;
      }

      // A rejected request must leave the original block owned and intact.
      if (!verify(p, old_size))
      {
        std::free(p);
        return EXIT_FAILURE;
      }
      std::free(p);
    }
  }
  catch (...)
  {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
