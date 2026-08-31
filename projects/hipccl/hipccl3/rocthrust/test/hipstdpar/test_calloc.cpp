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

// Regression tests for the calloc implementation selected by
// --hipstdpar-interpose-alloc.

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <utility>

namespace
{
__attribute__((noinline)) void* runtime_calloc(std::size_t count, std::size_t size)
{
  return std::calloc(count, size);
}
} // namespace

int main()
{
  constexpr auto max_size = (std::numeric_limits<std::size_t>::max)();

  // An overflowing product must fail instead of allocating the wrapped size.
  errno = 0;
  const auto wrap_count = max_size / 2 + 1;
  if (auto p = runtime_calloc(wrap_count, 2))
  {
    std::free(p);
    return EXIT_FAILURE;
  }
  if (errno != ENOMEM)
  {
    return EXIT_FAILURE;
  }

  // A product that wraps to a small non-zero value must fail too. This is the
  // dangerous shape: a huge request served by a tiny allocation, which the
  // caller then writes to as if it were huge.
  errno = 0;
  const auto wrap_to_two = max_size / 2 + 2;
  if (auto p = runtime_calloc(wrap_to_two, 2))
  {
    std::free(p);
    return EXIT_FAILURE;
  }
  if (errno != ENOMEM)
  {
    return EXIT_FAILURE;
  }

  // A zero-sized request may return either nullptr or a unique pointer; the
  // interposer must not manufacture ENOMEM solely because the size is zero.
  const std::pair<std::size_t, std::size_t> zero_sized[]{{0, 16}, {16, 0}};
  for (const auto& args : zero_sized)
  {
    errno = 0;
    if (auto p = runtime_calloc(args.first, args.second))
    {
      std::free(p);
    }
    else if (errno == ENOMEM)
    {
      return EXIT_FAILURE;
    }
  }

  // Allocation failure must be returned instead of passing nullptr to memset.
  errno = 0;
  if (auto p = runtime_calloc(1, max_size))
  {
    std::free(p);
    return EXIT_FAILURE;
  }
  if (errno != ENOMEM)
  {
    return EXIT_FAILURE;
  }

  // A successful allocation must still be fully zero-initialized.
  constexpr std::size_t count = 128;
  constexpr std::size_t size  = sizeof(unsigned int);
  auto p                       = static_cast<unsigned int*>(runtime_calloc(count, size));
  if (!p)
  {
    return EXIT_FAILURE;
  }
  for (std::size_t i = 0; i < count; ++i)
  {
    if (p[i] != 0)
    {
      std::free(p);
      return EXIT_FAILURE;
    }
  }
  std::free(p);

  return EXIT_SUCCESS;
}
