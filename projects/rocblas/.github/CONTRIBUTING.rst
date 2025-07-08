=======================
Contributing to rocBLAS
=======================

We welcome contributions to rocBLAS. Please follow these details to help ensure your contributions will be successfully accepted.

Issue Discussion
================

Please use the GitHub Issues tab to notify us of issues.

- Use your best judgment for issue creation. If your issue is already listed, upvote the issue and
  comment or post to provide additional details, such as how you reproduced this issue.
- If you're not sure if your issue is the same, err on the side of caution and file your issue.
  You can add a comment to include the issue number (and link) for the similar issue. If we evaluate
  your issue as being the same as the existing issue, we'll close the duplicate.
- If your issue doesn't exist, use the issue template to file a new issue.

  - When filing an issue, be sure to provide as much information as possible, including script output so
    we can collect information about your configuration. This helps reduce the time required to
    reproduce your issue.
  - Check your issue regularly, as we may require additional information to successfully reproduce the
    issue.
- You may also open an issue to ask questions to the maintainers about whether a proposed change
  meets the acceptance criteria, or to discuss an idea pertaining to the library.

Acceptance Criteria
===================
The aim of rocBLAS is to provide an optimized implementation of BLAS for ROCm.
The library includes extensions like mixed precision and batched versions of
functions.

Contributors wanting to submit new implementations, improvements, or bug fixes
should follow the below mentioned guidelines.

Pull requests will be reviewed by members of
`CODEOWNERS.md <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocblas/.github/CODEOWNERS>`__
Continuous Integration tests will be run on the pull request. Once the pull request
is approved and tests pass it will be merged by a member of
`CODEOWNERS.md <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocblas/.github/CODEOWNERS>`__.
Attribution for your commit will be preserved when it is merged.



Pull-request guidelines
=======================

By creating a pull request, you agree to the statements made in the
`Code License`_
section. Your pull request should target the default branch. Our current
default branch is the develop branch, which serves as our integration branch.

Pull requests should:

-  ensure code builds successfully.
-  do not break existing test cases.
-  new functionality will only be merged with new unit tests.
-  new unit tests should integrate within the existing googletest framework.
-  tests must have good code coverage.
-  code must also have benchmark tests, and performance must approach.
   the compute bound limit or memory bound limit.

Code License
============
All code contributed to this project will be licensed under the license identified in the
`LICENSE.md <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocblas/LICENSE.md>`__.
Your contribution will be accepted under the same license.

For each new file in repository, please include the licensing header

.. code:: cpp

    /*******************************************************************************
     * Copyright (c) 20xx Advanced Micro Devices, Inc.
     *
     * Permission is hereby granted, free of charge, to any person obtaining a copy
     * of this software and associated documentation files (the "Software"), to deal
     * in the Software without restriction, including without limitation the rights
     * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
     * copies of the Software, and to permit persons to whom the Software is
     * furnished to do so, subject to the following conditions:
     *
     * The above copyright notice and this permission notice shall be included in all
     * copies or substantial portions of the Software.
     *
     * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
     * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
     * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
     * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
     * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
     * SOFTWARE.
     *
     *******************************************************************************/


Coding Style
============

#.  rocBLAS code should avoid calling ``hipMalloc()`` or ``hipFree()`` within their code, as these are synchronizing APIs.
    Instead, users must rely on the rocBLAS device memory manager, which handles pre-allocated memory allocated during the creation of the rocBLAS handle.

    rocBLAS device memory allocation system offers:

    - A ``device_malloc()`` method in ``rocblas_handle`` for temporarily using device memory that has been either pre-allocated or allocated on demand.
    - A method in ``rocblas_handle`` to reuse device memory across rocBLAS calls without allocating and deallocating for each call.
    - A method in ``rocblas_handle`` for users to query the device memory required for optimal performance of a specific kernel call.
    - A method in ``rocblas_handle`` for users to control the amount of device memory allocated or allow rocBLAS to handle on-demand allocation.

    A rocBLAS function must allocate all of its device memory upfront for the entire duration of the function call and must not allocate and deallocate device memory at different kernel call levels.
    Lower-level kernels needing device memory must have it allocated by higher-level routines and passed down. When device memory can be shared between operations, the maximum size needed by all operations should be reported or allocated.

    When allocating memory, use a variable name that indicates it is a workspace memory, such as ``workspace`` or with a ``w_`` prefix, for example:

    .. code:: cpp

            auto w_mem = handle->device_malloc(dev_bytes);
            if(!w_mem)
            {
                return rocblas_status_memory_error;
            }

    rocBLAS device memory manager also provides support for stream order allocation ( using ``hipMallocAsync()`` and ``hipFreeAsync()`` ).

    For more information refer to `rocBLAS Device Memory Allocation <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/how-to/Programmers_Guide.html#device-memory-allocation>`__ and `Stream Order Allocation <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/how-to/Programmers_Guide.html#handle-stream-and-device-management>`__.

#.  Logging, argument error checking and device memory allocation should
    only occur at the top-level API functions. Therefore, if one
    rocBLAS routine calls another, the lower-level internally called routine(s)
    should not perform logging, argument checking, or device memory
    allocation. This can be accomplished in one of two ways:

    A. (Preferred.) Abstract out the computational part of the function
    that launches device kernels into a separate template (usually named
    ``rocblas_<function_name>_launcher``), and call it from a higher-level
    template (usually named ``rocblas_<function_name>_impl``) which
    does error-checking, device memory allocation, and logging, and
    which gets called by the C API level functions:

    .. code:: cpp

        template <...>
        rocblas_status rocblas_<function_name>_launcher(..., T* device_memory)
        {
            // Performs fast computation
            // No argument error checking
            // No logging
            // No device memory allocation -- any temporary device memory must be passed in through pointers
            // Can be called by other computational functions
            // Called by rocblas_<function_name>_impl
            // Private internal API
        }

        template <...>
        rocblas_status rocblas_<function_name>_impl()
        {
            // Argument error checking
            // Logging
            // Responding to device memory size queries
            // Device memory allocation (through handle->device_malloc())
            // Temporarily switching to host pointer mode if scalar constants are used
            // Calls rocblas_<function_name>_launcher()
            // Private internal API
        }

        extern "C" rocblas_status rocblas_[hsdcz]<function_name>[_64]()
        {
            // C wrapper
            // Calls rocblas_<function_name>_impl()
            // Public API
        }

    B. Use a ``bool`` template argument to specify if the kernel
    template should perform full functionality or not. Pass device
    memory pointer(s) which will be used if full functionality is turned
    off:

    .. code:: cpp

        template <bool full_function, ...>
        rocblas_status rocblas_<function_name>_launcher(..., T* device_memory = nullptr)
        {
            if(full_function)
            {
                // Argument error checking
                // Logging
                // Responding to device memory size queries
                // Device memory allocation (memory pointer assumed already allocated otherwise)
                // Temporarily switching to host pointer mode if scalar constants are used
                return ROCBLAS_API(rocblas_<function_name>_launcher)<false, ...>(...);
            }
            // Perform fast computation
            // Private internal API
        }

    Device memory allocation, and temporarily switching pointer mode,
    might be difficult to enclose in an ``if`` statement with the RAII
    design, so the code might have to use recursion to call the
    non-fully-functional version of itself after setting these things
    up. That's why method A above is preferred, but for some huge
    functions like GEMM, method B might be more practical to implement,
    since it disrupts existing code less.

    When an internal API is exported for reuse, an additional template layer may be present between the ``_impl`` and ``_launcher``
    templates (i.e. ``rocblas_<function_name>_template`` ).   It may exist in a non batched and batched form.
    Additionally, when an ILP64 API is provided for a function, the above launcher template may end with an ``_64`` suffix.
    The ``_impl`` template starts with an additional template typename API_INT which will be instantiated as either rocblas_int
    or int64_t.   The macro ``ROCBLAS_API`` may be used to call the ``_64`` or original form of a template instantiation.

    For more information refer to the `rocBLAS Programmers Guide <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/how-to/Programmers_Guide.html>`__

#.  The pointer mode should be temporarily switched to host mode during
    kernels which pass constants to other kernels, so that host-side
    constants of ``-1.0``, ``0.0`` and ``1.0`` can be passed to kernels
    like ``GEMM``, without causing synchronizing host<->device memory
    copies. For example:

    .. code:: cpp

        // Temporarily switch to host pointer mode, saving current pointer mode, restored on return
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        // Get alpha
        T alpha_h;
        if(saved_pointer_mode == rocblas_pointer_mode_host)
            alpha_h = *alpha;
        else
            RETURN_IF_HIP_ERROR(hipMemcpy(&alpha_h, alpha, sizeof(T), hipMemcpyDeviceToHost));

    ``saved_pointer_mode`` can be read to get the old pointer mode. If
    the old pointer mode was host pointer mode, then the host pointer is
    dereferenced to get the value of alpha. If the old pointer mode was
    device pointer mode, then the value of ``alpha`` is copied from the
    device to the host.

    After the above code switches to host pointer mode, constant values
    can be passed to ``GEMM`` or other kernels by always assuming host
    mode:

    .. code:: cpp

        static constexpr T negative_one = -1;
        static constexpr T zero = 0;
        static constexpr T one = 1;

        rocblas_internal_gemm_template( handle, transA, transB, jb, n, jb, alpha, invA, BLOCK, B, ldb, &zero, X, m);

    When ``saved_pointer_mode`` is destroyed, the handle's pointer mode
    returns to the previous pointer mode.

#.  When tests are added to ``rocblas-test`` and ``rocblas-bench``,
    refer to `this
    guide <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocblas/clients/gtest/README.md>`__.

    The test framework is templated, and uses
    SFINAE (substitution failure is not an error) pattern and ``std::enable_if<...>`` to enable and disable certain types for
    certain tests.

    YAML files are used to describe tests as combinations of arguments.
    `rocblas_gentest.py <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocblas/clients/common/rocblas_gentest.py>`__
    is used to parse the YAML files and generate tests in the form of a
    binary file of
    `Arguments <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocblas/clients/include/rocblas_arguments.hpp>`__
    records.

    The ``rocblas-test`` and ``rocblas-bench`` `type dispatch
    file <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocblas/clients/include/type_dispatch.hpp>`__
    is central to all tests. Basically, rather than duplicate:

    .. code:: cpp

        if(type == rocblas_datatype_f16_r)
            func1<rocblas_half>(args);
        else if(type == rocblas_datatype_f32_r)
            func<float>(args);
        else if(type == rocblas_datatype_f64_r)
            func<double>(args);

    etc. everywhere, it's done only in one place, and a ``template``
    template argument is passed to specify which action is actually
    taken. It's fairly abstract, but it is powerful. There are examples
    of using the type dispatch in
    `clients/gtest/*_gtest.cpp <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocblas/clients/gtest>`__
    and
    `clients/benchmarks/client.cpp <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocblas/clients/benchmarks/client.cpp>`__.

#.  Code should not be copied-and pasted, but rather, templates, macros,
    SFINAE (substitution failure is not an error) pattern and CRTP (curiously recurring template pattern),
    etc. should be used to factor out differences in similar code.

    A code should be made more generalized, rather than copied and
    modified, unless it is a completely different kernel function, and
    the old code is just being used as a start.

    If a new function is similar to an existing function, then the
    existing function should be generalized, or the new function and
    existing function should be refactored and based on a third
    templated function or class, rather than duplicating code.

#.  To differentiate between scalars located on either the host or
    device memory, a special function has been created, called
    ``load_scalar()``. If its argument is a pointer, it is dereferenced
    on the device. If the argument is a scalar, it is simply copied.
    This allows single HIP kernels to be written for both device and
    host memory:

    .. code:: cpp

        template <typename T, typename U>
        ROCBLAS_KERNEL void axpy_kernel(rocblas_int n, U alpha_device_host, const T* x, rocblas_int incx, T* y, rocblas_int incy)
        {
            auto alpha = load_scalar(alpha_device_host);
            ptrdiff_t tid = blockIdx.x * blockDim.x + threadIdx.x;

           // bound
           if(tid < n)
               y[tid * incy] += alpha * x[tid * incx];
        }

    Here, ``alpha_device_host`` can either be a pointer to device
    memory, or a numeric value passed directly to the kernel from the
    host. The ``load_scalar()`` function dereferences it if it's a
    pointer to device memory, and simply returns its argument if it's
    numerical. The kernel is called from the host in one of two ways
    depending on the pointer mode:

    .. code:: cpp

        if(handle->pointer_mode == rocblas_pointer_mode_device)
            ROCBLAS_LAUNCH_KERNEL(axpy_kernel, blocks, threads, 0, handle->get_stream(), n, alpha, x, incx, y, incy);
        else if(*alpha) // alpha is on host
            ROCBLAS_LAUNCH_KERNEL(axpy_kernel, blocks, threads, 0, handle->get_stream(), n, *alpha, x, incx, y, incy);

    When the pointer mode indicates ``alpha`` is on the host, the
    ``alpha`` pointer is dereferenced on the host and the numeric value
    it points to is passed to the kernel. When the pointer mode
    indicates ``alpha`` is on the device, the ``alpha`` pointer is
    passed to the kernel and dereferenced by the kernel on the device.
    This allows a single kernel to handle both cases, eliminating
    duplicate code.

#.  If new arithmetic datatypes (like ``rocblas_bfloat16``) are created,
    then unless they correspond *exactly* to a predefined system type,
    they should be wrapped into a ``struct``, and not simply be a
    ``typedef`` to another type of the same size, so that their type is
    unique and can be differentiated from other types.

    Right now ``rocblas_half`` is ``typedef``\ ed to ``uint16_t``, which
    unfortunately prevents ``rocblas_half`` and ``uint16_t`` from being
    differentiable. If ``rocblas_half`` were simply a ``struct`` with a
    ``uint16_t`` member, then it would be a distinct type.

    It is legal to convert a pointer to a standard-layout
    ``class``/``struct``
    to a pointer to its first element, and vice-versa, so the C API is
    unaffected by whether the type is enclosed in a ``struct`` or not.

#.  RAII (resource acquisition is initialization) patterned
    classes should be used instead of explicit ``new``/``delete``,
    ``hipMalloc``/``hipFree``, ``malloc``/``free``, etc. RAII classes
    are automatically exception-safe because their destructor gets
    called during unwinding. They only have to be declared once to
    construct them, and they are automatically destroyed when they go
    out of scope. This is better than having to match ``new``/``delete``
    ``malloc``/``free`` calls in the code, especially when exceptions or
    early returns are possible.

    Even if an operation does not allocate and free memory, if it
    represents a change in state which must be undone when a function
    returns, then it belongs in an RAII class. For example,
    ``handle->push_pointer_mode()`` creates an RAII object which saves
    the pointer mode on construction, and restores it on destruction.

#.  When writing function templates, place any non-type parameters
    before type parameters, i.e., leave the type parameters at the end.
    For example:

    .. code:: cpp

        template <rocblas_int NB, typename T> // T is at end
        static rocblas_status rocblas_trtri_batched_template(rocblas_handle handle,
                                                             rocblas_fill uplo,
                                                             rocblas_diagonal diag,
                                                             rocblas_int n,
                                                             const T* A,
                                                             rocblas_int lda,
                                                             rocblas_int bsa,
                                                             T* invA,
                                                             rocblas_int ldinvA,
                                                             rocblas_int bsinvA,
                                                             rocblas_int batch_count,
                                                             T* C_tmp)
        {
            if(!n || !batch_count)
                return rocblas_status_success;

             if(n <= NB)
                 return rocblas_trtri_small_batched<NB>(  // T is automatically deduced
                     handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
             else
                 return rocblas_trtri_large_batched<NB>(  // T is automatically deduced
                     handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count, C_tmp);
        }

    The reason for this, is that the type template arguments can be
    automatically deduced from the actual function arguments, so that
    you don't have to pass the types in calls to the function, as shown
    in the example above when calling ``rocblas_trtri_small_batched``
    and ``rocblas_trtri_large_batched``. They have a ``typename T``
    parameter too, but it can be automatically deduced, so it doesn't
    need to be explicitly passed.

#.  When writing functions like the above which are heavily dependent on
    block sizes, especially if they are in header files included by
    other files, template parameters for block sizes are strongly
    preferred to ``#define`` macros or ``constexpr`` variables. For
    ``.cpp`` files which are not included in other files, a
    ``static constexpr`` variable can be used. **Macros should never be
    used for constants.**

    Note: For constants inside of functions, ``static constexpr`` is
    preferred to just ``constexpr``, so that the variables do not need
    to be initialized at runtime.

    Note: C++14 variable templates can sometimes be used to provide
    constants. For example:

    .. code:: cpp

        template <typename T>
        static constexpr T negative_one = -1;

        template <typename T>
        static constexpr T zero = 0;

        template <typename T>
        static constexpr T one = 1;

#.  static duration variables which aren't constants should usually be
    made function-local ``static`` variables, rather than namespace or
    class static variables. This is to avoid the static initialization
    order fiasco. For example:

    .. code:: cpp

        static auto& get_table()
        {
            // Placed inside function to avoid dependency on initialization order
            static std::unordered_map<std::string, size_t>* table = test_cleanup::allocate(&table);
            return *table;
        }

    This is sometimes called the *singleton* pattern. A ``static``
    variable is made local to a function rather than a namespace or
    class, and it gets initialized the first time the function is
    called. A reference to the ``static`` variable is returned from the
    function, and the function is used everywhere access to the variable
    is needed. In the case of multithreaded programs, the C++11 and
    later standards guarantee that there won't be any race conditions.
    It is preferred to initialize function-local ``static`` variables than
    it is to explicitly call ``std::call_once``. For example:

    .. code:: cpp

        void my_func()
        {
            static int dummy = (func_to_call_once(), 0);
        }

    This is much simpler and faster than explicitly calling
    ``std::call_once``, since the compiler has special ways of
    optimizing ``static`` initialization. The first time ``my_func()``
    is called, it will call ``func_to_call_once()`` once in a
    thread-safe way. After that, there is almost no overhead in later
    calls to ``my_func()``.

#.  Functions are preferred to macros. Functions or functors inside of
    ``class`` / ``struct`` templates can be used when partial template
    specializations are needed.

    When C preprocessor macros are needed (such as if they contain a
    ``return`` statement to return from the calling function), if the
    macro's definition contains more than one simple expression, then
    it should be wrapped in a
    ``do { } while(0)``,
    without a terminating semicolon. This is to allow them to be used
    inside ``if`` statements. For example:

    .. code:: cpp

        #define RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(h) \
            do                                               \
            {                                                \
                if((h)->is_device_memory_size_query())       \
                    return rocblas_status_size_unchanged;    \
            } while(0)

    The ``do { } while(0)`` allows the macro expansion to be a single
    statement which can be terminated with a semicolon, and which can be
    used anywhere a regular function call can be used.

#.  For most template functions which are used in other compilation
    units, it is preferred that they be put in header files, rather than
    ``.cpp`` files, because putting them in ``.cpp`` files requires
    explicit instantiation of them for all possible arguments, and there
    are less opportunities for inlining and interprocedural
    optimization.

    The C++ standard explicitly says that unused templates can be
    omitted from the output, so including unused templates in a header
    file does not increase the size of the program, since only the used
    ones are in the final output.

    For template functions which are only used in one ``.cpp`` file,
    they can be placed in the ``.cpp`` file.

    Templates, like inline functions, are granted an exception to the
    one definition rule (ODR) as long as the sequence of tokens in each
    compilation unit is identical.

#.  Functions and namespace-scope variables which are not a part of the
    public interface of rocBLAS, should either be marked static, be
    placed in an unnamed namespace, or be placed in
    ``namespace rocblas``. For example:

    .. code:: cpp

        namespace
        {
            // Private internal implementation
        } // namespace

        extern "C"
        {
            // Public C interfaces
        } // extern "C"

    However, unnamed namespaces should not be used in header files. If
    it is absolutely necessary to mark a function or variable as private
    to a compilation unit but defined in a header file, it should be
    declared ``static``, ``constexpr`` and/or ``inline`` (``constexpr``
    implies ``static`` for non-template variables and ``inline`` for
    functions).

    Even though rocBLAS goes into a shared library which exports a
    limited number of symbols, this is still a good idea, to decrease
    the chances of name collisions *inside* of rocBLAS.

#.  ``std::string`` should only be used for strings which can grow, or
    which must be dynamically allocated as read-write strings. For
    simple static strings, strings returned from functions like
    ``getenv()``, or strings which are initialized once and then used
    read-only, ``const char*`` should be used to refer to the string or
    pass it as an argument.

    ``std::string`` involves dynamic memory allocation and copying of
    temporaries, which can be slow. ``std::string_view`` is supposed to
    help alleviate that, which became available in C++17. ``const char*``
    can be used for read-only views of strings, in the interest of efficiency.

#.  For code brevity and readability, when converting to *numeric*
    types, uniform initialization or function-style casts are preferred
    to ``static_cast<>()`` or C-style casts. For example, ``T{x}`` or ``T(x)``
    is preferred to ``static_cast<T>(x)`` or ``(T)x``. ``T{x}`` differs from
    ``T(x)`` in that narrowing conversions, which reduce the precision of an
    integer or floating-point, are not allowed.

    When writing general containers or templates which can accept
    *arbitrary* types as parameters, not just *numeric* types, then the
    specific cast (``static_cast``, ``const_cast``,
    ``reinterpret_cast``) should be used, to avoid surprises.

    But when converting to *numeric* types, which have very
    well-understood behavior and are *side-effect free*, ``type{x}`` or
    ``type(x)`` are  more compact and clearer than ``static_cast<type>(x)``.
    For pointers, C-style casts are okay, such as ``(T*)A``.

#.  For BLAS2 functions and BLAS1 functions with two vectors, the
    ``incx`` and/or ``incy`` arguments can be negative, which means the
    vector is treated backwards from the end. A simple trick to handle
    this, is to adjust the pointer to the end of the vector if the
    increment is negative, as in:

    .. code:: cpp

        if(incx < 0)
            x -= ptrdiff_t(incx) * (n - 1);
        if(incy < 0)
            y -= ptrdiff_t(incy) * (n - 1);

    After that adjustment, the code does not need to treat negative
    increments any differently than positive ones.

    Note: Some blocked matrix-vector algorithms which call other BLAS
    kernels may not work if this simple transformation is used; see
    `TRSV <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocblas/library/src/blas2/rocblas_trsv.cpp>`__
    for an example, and how it's handled there.

#.  For reduction operations, the file
    `reduction.hpp <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocblas/library/src/blas1/reduction.hpp>`_
    has been created to systematize reductions and perform their device
    kernels in one place. This works for ``amax``, ``amin``, ``asum``,
    ``nrm2``, and (partially) ``dot`` and ``gemv``.
    ``rocblas_reduction_kernel`` is a generalized kernel which takes 3
    *functors* as template arguments:

    -  One to *fetch* values (such as fetching a complex value and
       taking the sum of the squares of its real and imaginary parts
       before reducing it)
    -  One to *reduce* values (such as to compute a sum or maximum)
    -  One to *finalize* the reduction (such as taking the square root
       of a sum of squares)

    There is a ``default_value()`` function which returns the default
    value for a reduction. The default value is the value of the
    reduction when the size is 0, and reducing a value with the
    ``default_value()`` does not change the value of the reduction.

#.  When type punning
    is needed, ``union`` should be used instead of pointer-casting,
    which violates *strict aliasing*. For example:

    .. code:: cpp

        // zero extend lower 16 bits of bfloat16 to convert to IEEE float
        explicit __host__ __device__ operator float() const
        {
            union
            {
                uint32_t int32;
                float    fp32;
            } u = {uint32_t(data) << 16};
            return u.fp32; // Legal in C, nonstandard extension in C++
        }

    This violates the strict aliasing rule of C and C++:

    .. code:: cpp

        // zero extend lower 16 bits of bfloat16 to convert to IEEE float
        explicit __host__ __device__ operator float() const
        {
            uint32_t int32 = uint32_t(data) << 16;
            return *(float *) &int32; // Violates strict aliasing rule in both C and C++
        }

    The only 100% standard C++ way to do it, is to use ``memcpy()``, but
    this should not be required as long as GCC or Clang are used:

    .. code:: cpp

        // zero extend lower 16 bits of bfloat16 to convert to IEEE float
        explicit __host__ __device__ operator float() const
        {
            uint32_t int32 = uint32_t(data) << 16;
            float fp32;
            static_assert(sizeof(int32) == sizeof(fp32), "Different sizes");
            memcpy(&fp32, &int32, sizeof(fp32));
            return fp32;
        }

#.  ``<type_traits>`` classes which return Boolean values can be
    converted to ``bool`` in Boolean contexts. Hence many traits can be
    tested by simply creating an instance of them with ``{}``. However,
    for type_traits accessors such as ::value or ::type, these can be replaced by suffixes
    added in C++17 such as is_same_v and enable_if_t:

    .. code:: cpp

        template<typename T, typename = typename std::enable_if_t<std::is_same_v<T, float> ||
                                                                  std::is_same_v<T, double>>>
        void function(T x)
        {
        }

    For other traits created with the ``{}`` syntax the resulting temporary objects can
    be explicitly converted to ``bool``, which is what occurs when an
    object appears in a conditional expression (``if``, ``while``,
    ``for``, ``&&``, ``||``, ``!``, ``? :``, etc.).

#.  ``rocblas_cout`` and ``rocblas_cerr`` should be used instead of ``std::cout``, ``std::cerr``, ``stdout`` or ``stderr``, and ``rocblas_internal_ostream`` should be used instead of ``std::ostream``, ``std::ofstream`` or ``std::ostringstream``.

    In ``rocblas-bench`` and ``rocblas-test``, ``std::cout``, ``std::cerr``, ``printf``, ``fprintf``, ``stdout``, ``stderr``, ``puts()``, ``fputs()``, and other symbols are "poisoned", to remind you to use ``rocblas_cout``, ``rocblas_cerr``, and ``rocblas_internal_ostream`` instead.

    ``rocblas_cout`` and ``rocblas_cerr`` are instances of ``rocblas_internal_ostream`` which output to standard output and standard error, but in a way that prevents interlacing of different threads' output.

    ``rocblas_internal_ostream`` provides standardized thread-safe formatted output for rocBLAS datatypes. It can be constructed in 3 ways:
    - By default, in which case it behaves like a ``std::ostringstream``
    - With a file descriptor number, in which case the file descriptor is ``dup()``ed and the same file it points to is outputted to
    - With a string, in which case a new file is opened for writing, with file creation, truncation and appending enabled (``O_WRONLY | O_CREAT | O_TRUNC | O_APPEND | O_CLOEXEC``)

    ``std::endl`` or ``std::flush`` should be used at the end of an output sequence when an atomic flush of the output is needed (atomic meaning that multiple threads can be writing to the same file, but that their flushes will be atomic). Until then, the output will accumulate in the ``rocblas_internal_ostream`` and will not be flushed until either ``rocblas_internal_ostream::flush()`` is called, ``std::endl`` or ``std::flush`` is outputted, or the ``rocblas_internal_ostream`` is destroyed.

    The ``rocblas_internal_ostream::yaml_on`` and ``rocblas_internal_ostream::yaml_off`` IO modifiers enable or disable YAML formatting, for when outputting abitrary types as YAML source code. For example, to output a ``key: value`` pair as YAML source code, you would use:

    .. code:: cpp

        os << key << ": " << rocblas_internal_ostream::yaml_on << value << rocblas_internal_ostream::yaml_off;

    The ``key`` is outputted normally as a bare string, but the ``value`` uses YAML metacharacters and lexical syntax to output the value, so that when it's read in as YAML, it has the type and value of ``value``.


#.  C++ templates, including variadic templates, are preferred to macros or runtime interpreting of values, although it is understood that sometimes macros are necessary.

    For example, when creating a class which models zero or more rocBLAS kernel arguments, it is preferable to use:

    .. code:: cpp

        template<rocblas_argument... Args>
        class ArgumentModel
        {
            public:
                void func()
                {
                    for(auto arg: { Args... })
                    {
                        //do something with argument arg
                    }
                }
        };
        ArgumentModel<e_A, e_B>{}.func();

    instead of:

    .. code:: cpp

        class ArgumentModel
        {
            std::vector<rocblas_argument> args;
            public:
                ArgumentModel(const std::vector<rocblas_argument>& args): args(args)
                {
                }

                void func()
                {
                    for(auto arg: args)
                    {
                        //do something with argument arg
                    }
                }
        };
        ArgumentModel model({e_A, e_B});
        model.func();

    The former denotes the rocBLAS arguments as a list which is passed as a variadic template argument, and whose properties are known and can be optimized at compile-time, and which can be passed on as arguments to other templates, while the latter requires creating a dynamically-allocated runtime object which must be interpreted at runtime, such as by using ``switch`` statements on the arguments. The ``switch`` statement will need to list out and handle every possible argument, while the template solution simply passes the argument as another template argument, and hence can be resolved at compile-time.


#.  Automatically-generated files should always go into ``build/`` directories, and should not go into source directories (even if marked ``.gitignore``). The CMake philosophy is such that you can create any ``build/`` directory, run ``cmake`` from there, and then have a self-contained build environment which will not touch any files outside of it.


#.  The ``library/include`` subdirectory of rocBLAS, to be distinguished from the ``library/src/include`` subdirectory, shall consist only of C-compatible header files for public rocBLAS APIs. It should not include internal APIs, even if they are used in other projects, e.g., rocSOLVER, and the headers must be compilable with a C compiler, and must use ``.h`` extensions.


#.  Macro parameters should only be evaluated once when practical, and should be parenthesized if there is a chance of ambiguous precedence. They should be stored in a local temporary variable if needed more than once.

    Macros which expand to code with local variables, should use double-underscore suffixes in the local variable names, to prevent their conflict with variables passed in macro parameters. However, if they are in a completely separate block scope than the macro parameter is expanded in, or if they are only passed to another macro/function, then they do not need to use trailing underscores.

    .. code:: cpp

        #define CHECK_DEVICE_ALLOCATION(ERROR)                   \
            do                                                   \
            {                                                    \
                /* Use error__ in case ERROR contains "error" */ \
                hipError_t error__ = (ERROR);                    \
                if(error__ != hipSuccess)                        \
                {                                                \
                    if(error__ == hipErrorOutOfMemory)           \
                        GTEST_SKIP() << LIMITED_VRAM_STRING;     \
                    else                                         \
                        FAIL() << hipGetErrorString(error__);    \
                    return;                                      \
                }                                                \
            } while(0)


    The ``ERROR`` macro parameter is evaluated only once, and is stored in the temporary variable ``error__``, for use multiple times later.

    The ``ERROR`` macro parameter is parenthesized when initializing ``error__``, to avoid ambiguous precedence, such as if ``ERROR`` contains a comma expression.

    The ``error__`` variable name is used, to prevent it from conflicting with variables passed in the ``ERROR`` macro parameter, such as ``error``.


#.  Do not use variable-length arrays (VLA), which allocate on the stack, for arrays of unknown size.

    .. code:: cpp

        Ti* hostA[batch_count];
        Ti* hostB[batch_count];
        To* hostC[batch_count];
        To* hostD[batch_count];

        func(hostA, hostB, hostC, hostD);

    Instead, allocate on the heap, using smart pointers to avoid memory leaks:

    .. code:: cpp

        auto hostA = std::make_unique<Ti*[]>(batch_count);
        auto hostB = std::make_unique<Ti*[]>(batch_count);
        auto hostC = std::make_unique<To*[]>(batch_count);
        auto hostD = std::make_unique<To*[]>(batch_count);

        func(&hostA[0], &hostB[0], &hostC[0], &hostD[0]);


#.  Do not define unnamed (anonymous) namespaces in header files (for explanation see DCL59-CPP)

    If the reason for using an unnamed namespace in a header file is to prevent multiple definitions, keep in mind that the following are allowed to be defined in multiple compilation units, such as if they all come from the same header file, as long as they are defined with identical token sequences in each compilation unit:

    -  ``classes``
    -  ``typedefs`` or type aliases
    -  ``enums``
    -  ``template`` functions
    -  ``inline`` functions
    -  ``constexpr`` functions (implies ``inline``)
    -  ``inline`` or ``constexpr`` variables or variable ``template``s (only for C++17 or later, although some C++14 compilers treat ``constexpr`` variables as ``inline``)

    If functions defined in header files are declared ``template``, then multiple instantiations with the same ``template`` arguments are automatically merged, something which cannot happen if the ``template`` functions are declared ``static``, or appear in unnamed namespaces, in which case the instantiations are local to each compilation unit, and are not combined.

    If a function defined in a header file at ``namespace`` scope (outside of a ``class``) contains ``static`` _local variables which are expected to be singletons holding state throughout the entire library, then the function cannot be marked ``static`` or be part of an unnamed ``namespace``, because then each compilation unit will have its own separate copy of that function and its local ``static`` variables. (``static`` member functions of classes always have external linkage, and it is okay to define ``static`` ``class`` member functions in-place inside of header files, because all in-place ``static`` member function definitions, including their ``static`` local variables, will be automatically merged.)

Guidelines:

-  Do not use unnamed ``namespaces`` inside of header files.

-  Use either ``template`` or ``inline`` (or both) for functions defined outside of classes in header files.

-  Do not declare namespace-scope (not ``class``-scope) functions ``static`` inside of header files unless there is a very good reason, that the function does not have any non-``const`` ``static`` local variables, and that it is acceptable that each compilation unit will have its own independent definition of the function and its ``static`` local variables. (``static`` ``class`` member functions defined in header files are okay.)

-  Use ``static`` for ``constexpr`` ``template`` variables until C++17, after which ``constexpr`` variables become ``inline`` variables, and thus can be defined in multiple compilation units. It is okay if the ``constexpr`` variables remain ``static`` in C++17; it just means there might be a little bit of redundancy between compilation units.


Process
=======

rocBLAS uses the ``clang-format`` tool for formatting C and C++ code. To format a file, use:

::

    clang-format -style=file -i <path-to-source-file>

To format all files, run the following script in rocBLAS directory:

::

    #!/bin/bash
    git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0 clang-format -style=file -i

Also, githooks can be installed to format the code per-commit:

::

    ./.githooks/install


``cppcheck`` is an open-source static analysis tool. This project uses this tool for performing static code analysis.

Users can use the following command to run cppcheck locally to generate the report for all files.

.. code:: bash

   $ cd rocBLAS-internal
   $ cppcheck --enable=all --inconclusive --library=googletest --inline-suppr -i./build --suppressions-list=./CppCheckSuppressions.txt --template="{file}:{line}: {severity}: {id} :{message}" . 2> cppcheck_report.txt


Also, githooks can be installed to perform static analysis on new/modified files using pre-commit:

::

    ./.githooks/install

For more information on the command line options, refer to the cppcheck manual on the web.

References
==========

`rocBLAS documentation <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html>`__
