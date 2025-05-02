..
    Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.

.. _libcudacxx-extended-api-functional-proclaim-return-type:

cuda::proclaim_return_type
==============================

Defined in the header ``<cuda/functional>``:

.. code:: cuda

   template <class Ret, class Fn>
   __host__ __device__
   unspecified<Ret, Fn> proclaim_return_type(Fn&& fn) {
     return unspecified<Ret, Fn>{fn};
   }

``cuda::proclaim_return_type`` creates a forwarding call wrapper that uses ``Ret`` as a return type.
The wrapper is useful in the case of extended device lambdas since an attempt to determine the return type of
their ``operator()`` function may work incorrectly in host code.

Template Parameters
-------------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``Ret``
     - Return type that's being proclaimed
   * - ``Fn``
     - Callable object type that's being wrapped

Parameters
----------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``fn``
     - Callable object that's being wrapped

Example
-------

.. code:: cuda

   #include <cuda/functional>

   template <class T, class Fn>
   __global__ void example_kernel(T *out, Fn fn) {
     *out = fn();
   }

   __host__ void example() {
     auto fn = cuda::proclaim_return_type<char>([] __device__ () { return 'd'; });
     using rt = cuda::std::invoke_result_t<decltype(fn)>;

     rt* out {};
     hipMalloc(&out, sizeof(rt));

     example_kernel<<<1, 1>>>(out, fn);

     // ...
   }
