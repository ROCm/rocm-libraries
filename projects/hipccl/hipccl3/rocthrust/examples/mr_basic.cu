// Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/disjoint_pool.h>
#include <thrust/mr/new.h>
#include <thrust/mr/pool.h>

#include <cassert>

template <typename Vec>
void do_stuff_with_vector(typename Vec::allocator_type alloc)
{
  Vec v1(alloc);
  v1.push_back(1);
  assert(v1.back() == 1);

  Vec v2(alloc);
  v2 = v1;

  v1.swap(v2);

  v1.clear();
  v1.resize(2);
  assert(v1.size() == 2);
}

int main()
{
  thrust::mr::new_delete_resource memres;

  {
    // no virtual calls will be issued
    using Alloc = thrust::mr::allocator<int, thrust::mr::new_delete_resource>;
    Alloc alloc(&memres);

    do_stuff_with_vector<thrust::host_vector<int, Alloc>>(alloc);
  }

  {
    // virtual calls will be issued - wrapping in a polymorphic wrapper
    thrust::mr::polymorphic_adaptor_resource<void*> adaptor(&memres);
    using Alloc = thrust::mr::polymorphic_allocator<int, void*>;
    Alloc alloc(&adaptor);

    do_stuff_with_vector<thrust::host_vector<int, Alloc>>(alloc);
  }

  {
    // use the global device_ptr-flavored device memory resource
    using Resource = thrust::device_ptr_memory_resource<thrust::device_memory_resource>;
    thrust::mr::polymorphic_adaptor_resource<thrust::device_ptr<void>> adaptor(
      thrust::mr::get_global_resource<Resource>());
    using Alloc = thrust::mr::polymorphic_allocator<int, thrust::device_ptr<void>>;
    Alloc alloc(&adaptor);

    do_stuff_with_vector<thrust::device_vector<int, Alloc>>(alloc);
  }

  using Pool = thrust::mr::unsynchronized_pool_resource<thrust::mr::new_delete_resource>;
  Pool pool(&memres);
  {
    using Alloc = thrust::mr::allocator<int, Pool>;
    Alloc alloc(&pool);

    do_stuff_with_vector<thrust::host_vector<int, Alloc>>(alloc);
  }

  using DisjointPool =
    thrust::mr::disjoint_unsynchronized_pool_resource<thrust::mr::new_delete_resource, thrust::mr::new_delete_resource>;
  DisjointPool disjoint_pool(&memres, &memres);
  {
    using Alloc = thrust::mr::allocator<int, DisjointPool>;
    Alloc alloc(&disjoint_pool);

    do_stuff_with_vector<thrust::host_vector<int, Alloc>>(alloc);
  }
}
