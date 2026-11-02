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

// Regression test for zero-size allocations under --hipstdpar-interpose-alloc.
//
// __hipstdpar_aligned_alloc used to call hipMemAdvise(r, n, ...) unconditionally
// after a successful underlying allocation. hipMemAdvise rejects zero-length
// ranges regardless of pointer validity, so the wrapper discarded an already
// valid pointer and returned nullptr for any n == 0 request. That, in turn, made
// __hipstdpar_operator_new_aligned throw std::runtime_error for operator new(0),
// even though the standard requires zero-size allocation to succeed and return a
// valid, distinct, non-null pointer ([expr.new], [new.delete.single]).
//
// This test exercises zero-size requests through the main entry points that
// reach __hipstdpar_aligned_alloc: operator new (plain, aligned, nothrow and
// aligned nothrow), new T[0] (plain and over-aligned),
// std::allocator<T>::allocate(0), malloc(0), calloc(0, 0) and
// aligned_alloc(a, 0). None of them may throw or abort, and all of them except
// std::allocator<T>::allocate(0) must return a non-null pointer.
//
// Each result is stored to a volatile sink: the interposition pass runs after
// the optimizer, which would otherwise remove unused allocation/deallocation
// pairs before they reach the interposed allocator.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>

namespace
{
void* volatile sink = nullptr;

constexpr std::size_t over_alignment = 64;

struct alignas(over_alignment) over_aligned
{
  char c;
};

void* keep(void* p)
{
  sink = p;
  return p;
}

bool is_aligned(const void* p, std::size_t a)
{
  return reinterpret_cast<std::uintptr_t>(p) % a == 0;
}

int fail(const char* step)
{
  std::fprintf(stderr, "FAILED: %s\n", step);
  return EXIT_FAILURE;
}
} // namespace

int main()
{
  const char* step = "";
  try
  {
    // operator new(0) must succeed and return a valid, non-null pointer.
    step    = "operator new(0)";
    void* p = keep(::operator new(0));
    if (!p)
    {
      return fail(step);
    }
    ::operator delete(p);

    // new T[0] must succeed and return a valid, non-null pointer.
    step     = "new int[0]";
    int* arr = static_cast<int*>(keep(new int[0]));
    if (!arr)
    {
      return fail(step);
    }
    delete[] arr;

    // Two zero-size allocations must return distinct pointers.
    step     = "operator new(0) returns distinct pointers";
    void* p1 = keep(::operator new(0));
    void* p2 = keep(::operator new(0));
    if (!p1 || !p2 || p1 == p2)
    {
      return fail(step);
    }
    ::operator delete(p1);
    ::operator delete(p2);

    // Aligned operator new(0) and new T[0] for an over-aligned T must succeed
    // and return a suitably aligned, non-null pointer.
    step     = "aligned operator new(0)";
    void* ap = keep(::operator new(0, std::align_val_t{over_alignment}));
    if (!ap || !is_aligned(ap, over_alignment))
    {
      return fail(step);
    }
    ::operator delete(ap, std::align_val_t{over_alignment});

    step              = "new over_aligned[0]";
    over_aligned* oar = static_cast<over_aligned*>(keep(new over_aligned[0]));
    if (!oar || !is_aligned(oar, over_alignment))
    {
      return fail(step);
    }
    delete[] oar;

    // Empty-container allocator path: libstdc++ forwards allocate(0) to
    // operator new(0).
    step = "std::allocator<int>::allocate(0)";
    std::allocator<int> alloc;
    int* ip = static_cast<int*>(keep(alloc.allocate(0)));
    alloc.deallocate(ip, 0);

    // malloc(0)/calloc(0, 0) are legally allowed to return nullptr, but the
    // interposed allocator must not discard the non-null pointer returned by
    // the underlying glibc allocator.
    step    = "malloc(0)";
    void* m = keep(std::malloc(0));
    if (!m)
    {
      return fail(step);
    }
    std::free(m);

    step    = "calloc(0, 0)";
    void* c = keep(std::calloc(0, 0));
    if (!c)
    {
      return fail(step);
    }
    std::free(c);

    step     = "aligned_alloc(64, 0)";
    void* aa = keep(std::aligned_alloc(over_alignment, 0));
    if (!aa)
    {
      return fail(step);
    }
    std::free(aa);

    // The nothrow overloads must return a non-null pointer instead of
    // throwing or crashing.
    step     = "nothrow operator new(0)";
    void* np = keep(::operator new(0, std::nothrow));
    if (!np)
    {
      return fail(step);
    }
    ::operator delete(np);

    step      = "nothrow aligned operator new(0)";
    void* nap = keep(::operator new(0, std::align_val_t{over_alignment}, std::nothrow));
    if (!nap || !is_aligned(nap, over_alignment))
    {
      return fail(step);
    }
    ::operator delete(nap, std::align_val_t{over_alignment});
  }
  catch (...)
  {
    std::fprintf(stderr, "FAILED: %s (threw)\n", step);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
