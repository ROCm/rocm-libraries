.. meta::
   :description: hipCUB updates for CCCL 3.0.3
   :keywords: hipCUB, ROCm, migration, CCCL 3.0.3

*****************************
hipCUB CCCL 3.0.3 updates
*****************************

Changes have been made to hipCUB to achieve parity with CCCL 3.0.3, establish a hard dependency on libhipcxx, and align with the changes made in CUB.


hipCUB traits
==============

A new standard type traits system is now provided by libhipcxx and libcudacxx. Because of this the functionality and internal use of ``hipcub::Traits`` has been minimized. ``hipcub::Traits`` is now only used in hipCUB's radix sort implementation for bit-twiddling.

``hipcub::BaseTraits`` and ``hipcub::Traits`` can no longer be specialized for custom types. ``hipcub::FpLimits`` has been removed.

Classification of types should be done using the functionality in the ``hip/std/type_traits`` and ``hip/type_traits`` header files. 

Floating-point limits should be obtained using ``hip::std::numeric_limits<T>`` instead of ``hipcub::FpLimits<T>``.

``hip::std::is_floating_point{_v}`` only recognizes C++ standard floating point types. ``hip::is_floating_point{_v}`` must be used to correctly classify extended types such as ``__half`` and ``hip_bfloat16``.

Users can still specialize ``hipcub::NumericTraits`` for custom floating point types, inheriting from ``hipcub::BaseTraits`` and providing the necessary type information. The traits from libcu++ must also be specialized. For example, a custom floating point type ``my_half`` can be registered with hipCUB and libcu++ as follows:

.. code-block:: cpp

   template <>
   inline constexpr bool ::hip::is_floating_point<my_half>::value = true;

   template <>
   class ::hip::std::numeric_limits<my_half> {
   public:
     static constexpr bool is_specialized = true;
     static __host__ __device__ my_half max()    { return /* TODO */; }
     static __host__ __device__ my_half min()    { return /* TODO */; }
     static __host__ __device__ my_half lowest() { return /* TODO */; }
   };

   template <>
   struct ::hipcub::NumericTraits<my_half> : ::hipcub::BaseTraits<FLOATING_POINT, true, uint16_t, my_half> {};


Removed macros
===============

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Macro
     - Migration guidance
   * - ``HIPCUB_IF_CONSTEXPR``
     - Use ``constexpr`` instead.
   * - ``HIPCUB_IS_INT128_ENABLED``
     - No replacement.
   * - ``HIPCUB_MAX(a, b)``
     - Use ``hip::std::max(a, b)`` instead.
   * - ``HIPCUB_MIN(a, b)``
     - Use ``hip::std::min(a, b)`` instead.
   * - ``HIPCUB_QUOTIENT_CEILING(a, b)``
     - Use ``hip::ceil_div(a, b)`` instead.
   * - ``HIPCUB_QUOTIENT_FLOOR(a, b)``
     - Use plain integer division ``a / b`` instead.
   * - ``HIPCUB_ROUND_UP_NEAREST(a, b)``
     - Use ``hip::round_up(a, b)`` instead.
   * - ``LEGACY_PTX_ARCH``
     - No replacement.

Removed functions and classes
==============================

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - API
     - Migration guidance
   * - ``hipcub::AliasTemporaries``
     - No replacement.
   * - ``hipcub::BAR``
     - No replacement.
   * - ``hipcub::BFI``
     - No replacement.
   * - ``hipcub::BaseTraits::CATEGORY``
     - Use the functionality available in ``hip/std/type_traits`` instead.
   * - ``hipcub::BaseTraits::nullptr_TYPE``
     - No replacement.
   * - ``hipcub::BaseTraits::PRIMITIVE``
     - Use the functionality available in ``hip/std/type_traits`` instead.
   * - ``hipcub::ConstantInputIterator``
     - Use ``thrust::constant_iterator`` instead.
   * - ``hipcub::CountingInputIterator``
     - Use ``thrust::counting_iterator`` instead.
   * - ``hipcub::CTA_SYNC``
     - Use ``__syncthreads()`` instead.
   * - ``hipcub::DeviceSpmv``
     - Use `hipSPARSE <https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/>`_ instead.
   * - ``hipcub::Difference``
     - Use ``hip::std::minus`` instead.
   * - ``hipcub::DiscardOutputIterator``
     - Use ``thrust::discard_iterator`` instead.
   * - ``hipcub::DivideAndRoundUp``
     - Use ``hip::round_up`` instead.
   * - ``hipcub::Division``
     - Use ``hip::std::divides`` instead.
   * - ``hipcub::Equality``
     - Use ``hip::std::equal_to`` instead.
   * - ``hipcub::FpLimits<T>``
     - Use ``hip::std::numeric_limits<T>`` instead.
   * - ``hipcub::GridBarrier``
     - Use the APIs from cooperative groups instead.
   * - ``hipcub::GridBarrierLifetime``
     - Use the APIs from cooperative groups instead.
   * - ``hipcub::IADD3``
     - No replacement.
   * - ``hipcub::Inequality``
     - Use ``hip::std::not_equal_to`` instead.
   * - ``hipcub::Int2Type``
     - Use ``hip::std::integral_constant`` instead.
   * - ``hipcub::IterateThreadStore``
     - No replacement.
   * - ``hipcub::LaneId()``
     - Use ``hip::ptx::get_sreg_laneid()`` instead.
   * - ``hipcub::LaneMaskGe()``
     - Use ``hip::ptx::get_sreg_lanemask_ge()`` instead.
   * - ``hipcub::LaneMaskGt()``
     - Use ``hip::ptx::get_sreg_lanemask_gt()`` instead.
   * - ``hipcub::LaneMaskLe()``
     - Use ``hip::ptx::get_sreg_lanemask_le()`` instead.
   * - ``hipcub::LaneMaskLt()``
     - Use ``hip::ptx::get_sreg_lanemask_lt()`` instead.
   * - ``hipcub::max``
     - Use ``hip::std::max`` instead.
   * - ``hipcub::min``
     - Use ``hip::std::min`` instead.
   * - ``hipcub::PRMT``
     - Use ``hip::ptx::prmt()`` instead.
   * - ``hipcub::SHL_ADD``
     - No replacement.
   * - ``hipcub::SHR_ADD``
     - No replacement.
   * - ``hipcub::Sum``
     - Use ``hip::std::plus`` instead.
   * - ``hipcub::Swap(a, b)``
     - Use ``hip::std::swap(a, b)`` instead.
   * - ``hipcub::TransformInputIterator``
     - Use ``thrust::transform_iterator`` instead.
   * - ``hipcub::WARP_ALL(predicate, member_mask)``
     - Use ``__all_sync()`` instead.
   * - ``hipcub::WARP_ANY(predicate, member_mask)``
     - Use ``__any_sync()`` instead.
   * - ``hipcub::WARP_BALLOT(predicate, member_mask)``
     - Use ``__ballot_sync()`` instead.
   * - ``hipcub::WARP_SYNC``
     - Use ``rocprim::wave_barrier()`` instead.
   * - ``hipcub::WarpId()``
     - Use ``hip::ptx::get_sreg_warpid()`` instead.

Deprecations with planned removal
===================================

``hipcub::Traits<T>::Max()`` has been deprecated and will be removed in a later version. Users are encouraged to use ``hip::std::numeric_limits<T>::max()`` instead.


Other changes
===================

``hipcub::DeviceReduce::{Arg}[Max|Min]`` now uses ``hip::std::numeric_limits<T>::[max|min]()`` instead of ``hipcub::Traits`` to determine the initial value.

