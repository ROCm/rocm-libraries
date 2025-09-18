.. meta::
  :description: rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation

.. _extension:

********************************************************************
rocBLAS extension
********************************************************************

Level-1 Extension functions support the ILP64 API. For more information on these ``_64`` functions, see the :ref:`ILP64 API` section.

rocblas_axpy_ex + batched, strided_batched
============================================

.. doxygenfunction:: rocblas_axpy_ex
   :outline:
.. doxygenfunction:: rocblas_axpy_batched_ex
   :outline:
.. doxygenfunction:: rocblas_axpy_strided_batched_ex

``axpy_ex``, ``axpy_batched_ex``, and ``axpy_strided_batched_ex functions`` support the ``_64`` interface. See the :ref:`ILP64 API` section.

rocblas_dot_ex + batched, strided_batched
============================================

.. doxygenfunction:: rocblas_dot_ex
   :outline:
.. doxygenfunction:: rocblas_dot_batched_ex
   :outline:
.. doxygenfunction:: rocblas_dot_strided_batched_ex

``dot_ex``, ``dot_batched_ex``, and ``dot_strided_batched_ex`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

rocblas_dotc_ex + batched, strided_batched
============================================

.. doxygenfunction:: rocblas_dotc_ex
   :outline:
.. doxygenfunction:: rocblas_dotc_batched_ex
   :outline:
.. doxygenfunction:: rocblas_dotc_strided_batched_ex

``dotc_ex``, ``dotc_batched_ex``, and ``dotc_strided_batched_ex`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

rocblas_nrm2_ex + batched, strided_batched
============================================

.. doxygenfunction:: rocblas_nrm2_ex
   :outline:
.. doxygenfunction:: rocblas_nrm2_batched_ex
   :outline:
.. doxygenfunction:: rocblas_nrm2_strided_batched_ex

``nrm2_ex``, ``nrm2_batched_ex``, and ``nrm2_strided_batched_ex`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

rocblas_rot_ex + batched, strided_batched
============================================

.. doxygenfunction:: rocblas_rot_ex
   :outline:
.. doxygenfunction:: rocblas_rot_batched_ex
   :outline:
.. doxygenfunction:: rocblas_rot_strided_batched_ex

``rot_ex``, ``rot_batched_ex``, and ``rot_strided_batched_ex`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

rocblas_scal_ex + batched, strided_batched
============================================

.. doxygenfunction:: rocblas_scal_ex
   :outline:
.. doxygenfunction:: rocblas_scal_batched_ex
   :outline:
.. doxygenfunction:: rocblas_scal_strided_batched_ex

``scal_ex``, ``scal_batched_ex``, and ``scal_strided_batched_ex`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

rocblas_gemm_ex + batched, strided_batched
============================================

.. doxygenfunction:: rocblas_gemm_ex

``gemm_ex`` functions support the ``_64`` interface.  However, no arguments larger than ``(int32_t max value * 16)`` are currently supported.
See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_gemm_batched_ex

``gemm_batched_ex`` functions support the ``_64`` interface.  Only the parameter ``batch_count`` larger than ``(int32_t max value * 16)`` is currently supported.
See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_gemm_strided_batched_ex

``gemm_strided_batched_ex`` functions support the ``_64`` interface.  Only the parameter ``batch_count`` larger than ``(int32_t max value * 16)`` is currently supported.
See the :ref:`ILP64 API` section.

rocblas_trsm_ex + batched, strided_batched
============================================

.. doxygenfunction:: rocblas_trsm_ex
   :outline:
.. doxygenfunction:: rocblas_trsm_batched_ex
   :outline:
.. doxygenfunction:: rocblas_trsm_strided_batched_ex

rocblas_Xgeam + batched, strided_batched
============================================

.. doxygenfunction:: rocblas_sgeam
   :outline:
.. doxygenfunction:: rocblas_dgeam
   :outline:
.. doxygenfunction:: rocblas_cgeam
   :outline:
.. doxygenfunction:: rocblas_zgeam

The ``geam`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_sgeam_batched
   :outline:
.. doxygenfunction:: rocblas_dgeam_batched
   :outline:
.. doxygenfunction:: rocblas_cgeam_batched
   :outline:
.. doxygenfunction:: rocblas_zgeam_batched

The ``geam_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_sgeam_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dgeam_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cgeam_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zgeam_strided_batched

The ``geam_strided_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

rocblas_geam_ex
============================================

.. doxygenfunction:: rocblas_geam_ex

rocblas_Xdgmm + batched, strided_batched
============================================

.. doxygenfunction:: rocblas_sdgmm
   :outline:
.. doxygenfunction:: rocblas_ddgmm
   :outline:
.. doxygenfunction:: rocblas_cdgmm
   :outline:
.. doxygenfunction:: rocblas_zdgmm

The ``dgmm`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_sdgmm_batched
   :outline:
.. doxygenfunction:: rocblas_ddgmm_batched
   :outline:
.. doxygenfunction:: rocblas_cdgmm_batched
   :outline:
.. doxygenfunction:: rocblas_zdgmm_batched

The ``dgmm_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_sdgmm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ddgmm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cdgmm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zdgmm_strided_batched

The ``dgmm_strided_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

rocblas_Xgemmt + batched, strided_batched
============================================

.. doxygenfunction:: rocblas_sgemmt
   :outline:
.. doxygenfunction:: rocblas_dgemmt
   :outline:
.. doxygenfunction:: rocblas_cgemmt
   :outline:
.. doxygenfunction:: rocblas_zgemmt

The ``gemmt`` functions support the ``_64`` interface. Parameter ``n`` larger than ``int32_t`` max value is not currently supported.
See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_sgemmt_batched
   :outline:
.. doxygenfunction:: rocblas_dgemmt_batched
   :outline:
.. doxygenfunction:: rocblas_cgemmt_batched
   :outline:
.. doxygenfunction:: rocblas_zgemmt_batched

The ``gemmt_batched`` functions support the ``_64`` interface. Parameter ``n`` larger than ``int32_t`` max value is not currently supported.
See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_sgemmt_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dgemmt_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cgemmt_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zgemmt_strided_batched

The ``gemmt_strided_batched`` functions support the ``_64`` interface. Parameter ``n`` larger than ``int32_t`` max value is not currently supported.
See the :ref:`ILP64 API` section.
