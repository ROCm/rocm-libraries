.. meta::
  :description: rocThrust migration to CCCL 3.0.3
  :keywords: rocThrust, ROCm, migration, CCCL 3.0.3

.. _synchronization-and-blocking:

**********************************************
rocThrust migration to CCCL 3.0.3
**********************************************

Changes have been made to rocThrust to achieve parity with CCCL 3.0.3, establish a hard dependency on libhipcxx, and align with the changes made in Thrust.

Changes to platform support
==============================

rocThrust now requires:

* C++17 or later
* Clang version 14 or later
* GCC version 7 or later
* Visual Studio 2019 or later on Windows

CUDA Dynamic Parallelism V1 (CDPv1) and Intel ICC (icpx) are no longer supported.

Changes to iterator traits
===========================

A new standard type traits system is now provided by libhipcxx and libcudacxx.

``hip::std::iterator_traits`` uses the new system and should be preferred over ``thrust::iterator_traits``.

``thrust::iterator_traits`` can no longer be specialized.

Removed macros
===============

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Macro
     - Migration guidance
   * - ``CUDA_CUB_RET_IF_FAIL``
     - No replacement.
   * - ``THRUST_CLANG_VERSION``
     - No replacement.
   * - ``THRUST_DECLTYPE_RETURNS_WITH_SFINAE_CONDITION``
     - No replacement.
   * - ``THRUST_DEVICE_CODE``
     - No replacement.
   * - ``THRUST_DEVICE_COMPILER*``
     - No replacement.
   * - ``THRUST_GCC_VERSION``
     - No replacement.
   * - ``THRUST_HOST_BACKEND``
     - Use ``THRUST_HOST_SYSTEM`` instead.
   * - ``THRUST_HOST_COMPILER*``
     - No replacement.
   * - ``THRUST_INCLUDE_DEVICE_CODE``
     - No replacement.
   * - ``THRUST_INCLUDE_HOST_CODE``
     - No replacement.
   * - ``THRUST_INLINE_CONSTANT``
     - Use ``inline constexpr`` instead.
   * - ``THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT``
     - Use ``static constexpr`` instead.
   * - ``THRUST_IS_DEVICE_CODE``
     - No replacement.
   * - ``THRUST_IS_HOST_CODE``
     - No replacement.
   * - ``THRUST_LEGACY_GCC``
     - No replacement.
   * - ``THRUST_MODERN_GCC``
     - No replacement.
   * - ``THRUST_MODERN_GCC_REQUIRED_NO_ERROR``
     - No replacement.
   * - ``THRUST_MSVC_VERSION``
     - No replacement.
   * - ``THRUST_MSVC_VERSION_FULL``
     - No replacement.
   * - ``THRUST_MVCAP``
     - No replacement.
   * - ``THRUST_NODISCARD``
     - Use ``[[nodiscard]]`` instead.
   * - ``THRUST_RETOF``
     - No replacement.
   * - ``THRUST_RETOF1``
     - No replacement.
   * - ``THRUST_RETOF2``
     - No replacement.
   * - ``THRUST_STATIC_ASSERT(expr)``
     - Use ``static_assert(expr)`` instead.
   * - ``THRUST_TUNING_ARCH``
     - | No direct replacement.
       | Use compiler-specific ``__CUDA_ARCH__`` (nvcc) or ``__NVCOMPILER_CUDA_ARCH__`` (nvc++) instead.
   * - ``THRUST_CDP_DISPATCH``
     - No replacement. Support for CUDA Dynamic Parallelism V1 (CDPv1) has been removed.

Removed functions and classes
==============================

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - API
     - Migration guidance
   * - ``_ReadWriteBarrier`` and ``__thrust_compiler_fence``
     - Use ``hip::atomic`` instead.
   * - ``thrust::async::*``
     - | No replacement.
       | To make a thrust algorithm skip syncing, use ``thrust::hip::par_nosync`` as the execution policy.
   * - ``thrust::*::[first_argument_type|second_argument_type|result_type]``
     - | No replacement.
       | The nested aliases have been removed for all function object types.
       | Affected types: ``thrust::plus``, ``thrust::minus``, ``thrust::multiplies``,
         ``thrust::divides``, ``thrust::modulus``, ``thrust::negate``, ``thrust::square``,
         ``thrust::equal_to``, ``thrust::not_equal_to``, ``thrust::greater``,
         ``thrust::less``, ``thrust::greater_equal``, ``thrust::less_equal``,
         ``thrust::logical_and``, ``thrust::logical_or``, ``thrust::logical_not``,
         ``thrust::bit_and``, ``thrust::bit_or``, ``thrust::bit_xor``,
         ``thrust::identity``, ``thrust::maximum``, ``thrust::minimum``,
         ``thrust::project1st``, ``thrust::project2nd``
   * - ``thrust::[binary|unary]_function``
     - | No replacement.
       | Remove any base classes that inherit from these types.
   * - ``thrust::[binary|unary]_traits``
     - No replacement.
   * - ``thrust::bidirectional_universal_iterator_tag``
     - No replacement.
   * - ``thrust::conjunction_value<Ts...>``
     - Use ``hip::std::bool_constant<(Ts && ...)>`` instead.
   * - ``thrust::conjunction_value_v<Ts...>``
     - Use a fold expression: ``Ts && ...`` instead.
   * - ``thrust::cuda_cub::core::*``
     - Implementation detail. No public exposure is provided.
   * - ``thrust::[cuda_cub|hip_rocprim]::counting_iterator_t``
     - Use ``thrust::counting_iterator`` instead.
   * - ``thrust::[cuda_cub|hip_rocprim]::identity``
     - Use ``hip::std::identity`` instead.
   * - ``thrust::cuda_cub::launcher::triple_chevron``
     - No replacement.
   * - ``thrust::[cuda_cub|hip_rocprim]::terminate``
     - Use ``hip::std::terminate()`` instead.
   * - ``thrust::[cuda_cub|hip_rocprim]::transform_input_iterator_t``
     - Use ``thrust::transform_iterator`` instead.
   * - ``thrust::[cuda_cub|hip_rocprim]::transform_pair_of_input_iterators_t``
     - Use ``thrust::transform_iterator`` of a ``thrust::zip_iterator`` instead.
   * - ``thrust::disjunction_value<Ts...>``
     - Use ``hip::std::bool_constant<(Ts || ...)>`` instead.
   * - ``thrust::disjunction_value_v<Ts...>``
     - Use a fold expression: ``Ts || ...`` instead.
   * - ``thrust::forward_universal_iterator_tag``
     - No replacement.
   * - ``thrust::identity<T>``
     - | Use ``hip::std::identity`` instead.
       | If ``thrust::identity`` was used to perform a cast to ``T``, define a custom function object.
   * - ``thrust::input_universal_iterator_tag``
     - No replacement.
   * - ``thrust::negation_value<T>``
     - Use ``hip::std::bool_constant<!T>`` instead.
   * - ``thrust::negation_value_v<T>``
     - Use a plain negation ``!T``.
   * - ``thrust::not[1|2]``
     - Use ``hip::std::not_fn`` instead.
   * - ``thrust::null_type``
     - No replacement.
   * - ``thrust::numeric_limits<T>``
     - Use ``hip::std::numeric_limits<T>`` instead.
   * - ``thrust::optional<T>``
     - Use ``hip::std::optional<T>`` instead.
   * - ``thrust::output_universal_iterator_tag``
     - No replacement.
   * - ``thrust::random_access_universal_iterator_tag``
     - No replacement.
   * - ``thrust::remove_cvref{_t}``
     - Use ``hip::std::remove_cvref{_t}`` instead.
   * - ``thrust::void_t``
     - Use ``hip::std::void_t`` instead.

Deprecations with planned removal
===================================

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - API
     - Migration guidance
   * - ``THRUST_UNUSED_VAR``
     - No replacement.
   * - ``thrust::iterator_difference{_t}<T>``
     - Use ``hip::std::iterator_traits<T>::difference_type`` or ``hip::std::iter_difference_t<T>`` instead.
   * - ``thrust::iterator_pointer{_t}<T>``
     - Use ``hip::std::iterator_traits<T>::pointer`` instead.
   * - ``thrust::iterator_reference{_t}<T>``
     - Use ``hip::std::iterator_traits<T>::reference`` or ``hip::std::iter_reference_t<T>`` instead.
   * - ``thrust::iterator_traits<T>``
     - Use ``hip::std::iterator_traits<T>`` instead.
   * - ``thrust::iterator_value{_t}<T>``
     - Use ``hip::std::iterator_traits<T>::value_type`` or ``hip::std::iter_value_t<T>`` instead.

Aliased functions
==================

The following Thrust function object types have been made aliases to the equally-named types in ``hip::std``. No change is needed if you are using these in your code.

- ``thrust::plus``
- ``thrust::minus``
- ``thrust::multiplies``
- ``thrust::divides``
- ``thrust::modulus``
- ``thrust::negate``
- ``thrust::equal_to``
- ``thrust::not_equal_to``
- ``thrust::greater``
- ``thrust::less``
- ``thrust::greater_equal``
- ``thrust::less_equal``
- ``thrust::logical_and``
- ``thrust::logical_or``
- ``thrust::logical_not``
- ``thrust::bit_and``
- ``thrust::bit_or``
- ``thrust::bit_xor``
- ``thrust::identity``
- ``thrust::maximum``
- ``thrust::minimum``

Other API changes
===================

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Change
   * - ``thrust::tabulate_output_iterator``
     - The ``value_type`` has been changed to ``void``.
   * - ``thrust::pair``
     - Converted to an alias for ``hip::std::pair`` and no longer a distinct type.
   * - ``thrust::tuple``
     - Converted to an alias for ``hip::std::tuple`` and no longer a distinct type.
   * - ``thrust::transform_iterator``
     - Copying this iterator will now always copy its contained function. If the contained function is neither copy constructible nor copy assignable, compilation will fail when an attempt is made to copy the iterator.
   * - ``thrust::universal_host_pinned_memory_resource``
     - The alias has changed to a different memory resource, potentially changing pointer types derived from an allocator or container using this memory resource.
