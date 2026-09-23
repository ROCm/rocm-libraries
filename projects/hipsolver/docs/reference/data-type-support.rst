.. meta::
  :description: hipSOLVER library precision support
  :keywords: hipSOLVER, ROCm, CUDA, API, Linear Algebra, documentation, precision support, data types

.. _hipsolver-data-type-support:

********************************************************************
hipSOLVER precision support
********************************************************************

This section provides an overview of the numerical precision types supported by the hipSOLVER library.
hipSOLVER provides a consistent interface to linear algebra solvers that can run on AMD hardware.

This page lists the data types supported by the library itself and does not
indicate hardware support. A type listed here is only usable if the GPU
architecture also supports it; otherwise it is unsupported. For data type support
across the other ROCm libraries and by GPU architecture, see the
:doc:`Data types and precision support page <rocm:reference/precision-support>`.

.. _hipsolver-input-output-type-support:

Supported data types overview
=============================

The following table summarizes the input and output data types supported by
hipSOLVER. For the precision-prefix naming convention and how precision appears
in function signatures, see the sections that follow.

.. list-table::
    :header-rows: 1

    *
      - Icon
      - Definition
    *
      - ✅
      - Fully supported as both an input and output type.
    *
      - ⚠️
      - Partially supported as an input or output type.

Data types not listed in the table below are not supported.

.. datatemplate:yaml:: /data/reference/precision-support.yaml

    .. list-table::
        :header-rows: 1
        :widths: 70, 30

        *
            - Data type
            - Support
    {% for data_type in data.data_types %}
        *
            - {{ data_type.type }}
            - {{ data_type.support }}
    {% endfor %}

Supported precision types
=========================

hipSOLVER supports four primary precision types across its functions:

.. list-table::
    :header-rows: 1

    *
      - Type prefix
      - C++ type
      - Description

    *
      - ``s``
      - ``float``
      - Single-precision real (32-bit)

    *
      - ``d``
      - ``double``
      - Double-precision real (64-bit)

    *
      - ``c``
      - ``hipFloatComplex``
      - Single-precision complex (32-bit real, 32-bit imaginary)

    *
      - ``z``
      - ``hipDoubleComplex``
      - Double-precision complex (64-bit real, 64-bit imaginary)

Function naming convention
--------------------------

hipSOLVER follows the LAPACK naming convention where the first letter of the function name indicates the
precision type:

* Functions beginning with ``hipsolverS`` operate on single-precision real data.
* Functions beginning with ``hipsolverD`` operate on double-precision real data.
* Functions beginning with ``hipsolverC`` operate on single-precision complex data.
* Functions beginning with ``hipsolverZ`` operate on double-precision complex data.

For example, the Cholesky factorization function ``potrf`` is implemented as:

* ``hipsolverSpotrf`` - For single-precision real matrices
* ``hipsolverDpotrf`` - For double-precision real matrices
* ``hipsolverCpotrf`` - For single-precision complex matrices
* ``hipsolverZpotrf`` - For double-precision complex matrices

In the documentation, these are sometimes represented generically as ``hipsolver<T>potrf()``, where ``<T>``
is a placeholder for the precision type prefix.

Understanding precision in function signatures
----------------------------------------------

In the function signatures throughout the documentation, precision information is indicated directly in
the parameter types. For example:

.. code-block:: c

    hipsolver_status hipsolverSpotrf(hipsolverHandle_t handle,
                                     hipsolverFillMode_t uplo,
                                     int n,
                                     float *A,
                                     int lda,
                                     float *work,
                                     int lwork,
                                     int *devInfo)

The parameter types (``float``, ``double``, ``hipFloatComplex``, or ``hipDoubleComplex``) correspond to the
function prefix and indicate the precision used by that specific function variant.

Real versus complex precision
-----------------------------

Some LAPACK-based functions in hipSOLVER have different behaviors or names when operating on real versus
complex data:

* Functions for symmetric matrices (prefix ``sy``) use the same name for both real precision types.
* Functions for Hermitian matrices (prefix ``he``) are used for complex precision types.
* Some auxiliary routines might be specific to real or complex precision types.

For example, ``hipsolverSsyevj`` and ``hipsolverDsyevj`` handle real symmetric matrices, while ``hipsolverCheevj``
and ``hipsolverZheevj`` handle complex Hermitian matrices.
