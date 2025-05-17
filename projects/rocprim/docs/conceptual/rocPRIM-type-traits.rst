.. meta::
  :description: Implementing traits for custom types in rocPRIM
  :keywords: rocPRIM, ROCm, custom types, type traits

.. _type_traits:

********************************************************************
Implementing traits for custom types in rocPRIM
********************************************************************

This interface is designed to enable users to provide additional type trait information to rocPRIM, facilitating better compatibility with custom types.

Accurately describing custom types is important for performance optimization and computational correctness.

Custom types that implement arithmetic operators can behave like built-in arithmetic types but might still be interpreted by rocPRIM algorithms as generic ``struct`` or ``class`` types.

The rocPRIM type traits interface lets users add custom trait information for their types, improving compatibility between these types and rocPRIM algorithms.

This interface is similar to operator overloading.

Traits should be implemented as required by specific algorithms. Some traits can't be defined if they can be inferred from others.

Interface
=========

.. doxygenstruct:: rocprim::traits::define

.. doxygenstruct:: rocprim::traits::get
  :members:

Available traits
================

.. doxygenstruct:: rocprim::traits::is_arithmetic
  :members:

.. doxygenstruct:: rocprim::traits::is_scalar
  :members:

.. doxygenstruct:: rocprim::traits::number_format
  :members:

.. doxygenstruct:: rocprim::traits::integral_sign
  :members:

.. doxygenstruct:: rocprim::traits::float_bit_mask
  :members:

.. doxygenstruct:: rocprim::traits::is_fundamental
  :members:

Type traits wrappers
====================
.. doxygengroup::  rocprim_type_traits_wrapper
  :content-only:

.. doxygenstruct:: rocprim::is_floating_point
  :no-link:

.. doxygenstruct:: rocprim::is_integral
  :no-link:

.. doxygenstruct:: rocprim::is_arithmetic
  :no-link:

.. doxygenstruct:: rocprim::is_fundamental
  :no-link:

.. doxygenstruct:: rocprim::is_unsigned
  :no-link:

.. doxygenstruct:: rocprim::is_signed
  :no-link:

.. doxygenstruct:: rocprim::is_scalar
  :no-link:

.. doxygenstruct:: rocprim::is_compound
  :no-link:

Types with predefined traits
============================

.. doxygengroup:: rocprim_pre_defined_traits
  :content-only:
  :members:
  :outline:
  :no-link:
