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
// This test exercises operator new(0), new T[0], malloc(0), and calloc(0, 0):
// none of them may throw or abort, and operator new/new[] must return a
// non-null pointer.

#include <cstdlib>
#include <new>

int main()
{
  try
  {
    // operator new(0) must succeed and return a valid, non-null pointer.
    void* p = ::operator new(0);
    if (!p)
    {
      return EXIT_FAILURE;
    }
    ::operator delete(p);

    // new T[0] must succeed and return a valid, non-null pointer.
    int* arr = new int[0];
    if (!arr)
    {
      return EXIT_FAILURE;
    }
    delete[] arr;

    // malloc(0)/calloc(0, 0) are legally allowed to return nullptr, but must
    // never crash or abort the process.
    void* m = std::malloc(0);
    std::free(m);

    void* c = std::calloc(0, 0);
    std::free(c);
  }
  catch (...)
  {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
