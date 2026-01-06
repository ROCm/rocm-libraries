.. meta::
  :description: rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation

.. _level-1:

********************************************************************
rocBLAS Level-1 functions
********************************************************************

rocBLAS Level-1 functions perform scalar, vector, and vector-vector operations. [Level1]_

Level-1 functions support the ILP64 API.  For more information on these ``_64``
functions, see the :ref:`ILP64 API` section.

.. _rocblas_amax:

rocblas_iXamax + batched, strided_batched
=========================================

.. doxygenfunction:: rocblas_isamax
   :outline:
.. doxygenfunction:: rocblas_idamax
   :outline:
.. doxygenfunction:: rocblas_icamax
   :outline:
.. doxygenfunction:: rocblas_izamax

The ``amax`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_isamax_batched
   :outline:
.. doxygenfunction:: rocblas_idamax_batched
   :outline:
.. doxygenfunction:: rocblas_icamax_batched
   :outline:
.. doxygenfunction:: rocblas_izamax_batched

The ``amax_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_isamax_strided_batched
   :outline:
.. doxygenfunction:: rocblas_idamax_strided_batched
   :outline:
.. doxygenfunction:: rocblas_icamax_strided_batched
   :outline:
.. doxygenfunction:: rocblas_izamax_strided_batched

The ``amax_strided_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. _rocblas_amin:

rocblas_iXamin + batched, strided_batched
==========================================

.. doxygenfunction:: rocblas_isamin
   :outline:
.. doxygenfunction:: rocblas_idamin
   :outline:
.. doxygenfunction:: rocblas_icamin
   :outline:
.. doxygenfunction:: rocblas_izamin

The ``amin`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_isamin_batched
   :outline:
.. doxygenfunction:: rocblas_idamin_batched
   :outline:
.. doxygenfunction:: rocblas_icamin_batched
   :outline:
.. doxygenfunction:: rocblas_izamin_batched

The ``amin_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_isamin_strided_batched
   :outline:
.. doxygenfunction:: rocblas_idamin_strided_batched
   :outline:
.. doxygenfunction:: rocblas_icamin_strided_batched
   :outline:
.. doxygenfunction:: rocblas_izamin_strided_batched

The ``amin_strided_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. _rocblas_asum:

rocblas_Xasum + batched, strided_batched
========================================

.. doxygenfunction:: rocblas_sasum
   :outline:
.. doxygenfunction:: rocblas_dasum
   :outline:
.. doxygenfunction:: rocblas_scasum
   :outline:
.. doxygenfunction:: rocblas_dzasum

The ``asum`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_sasum_batched
   :outline:
.. doxygenfunction:: rocblas_dasum_batched
   :outline:
.. doxygenfunction:: rocblas_scasum_batched
   :outline:
.. doxygenfunction:: rocblas_dzasum_batched

The ``asum_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_sasum_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dasum_strided_batched
   :outline:
.. doxygenfunction:: rocblas_scasum_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dzasum_strided_batched

The ``asum_strided_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. _rocblas_axpy:

rocblas_Xaxpy + batched, strided_batched
========================================

.. doxygenfunction:: rocblas_saxpy
   :outline:
.. doxygenfunction:: rocblas_daxpy
   :outline:
.. doxygenfunction:: rocblas_haxpy
   :outline:
.. doxygenfunction:: rocblas_caxpy
   :outline:
.. doxygenfunction:: rocblas_zaxpy

The ``axpy`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_saxpy_batched
   :outline:
.. doxygenfunction:: rocblas_daxpy_batched
   :outline:
.. doxygenfunction:: rocblas_haxpy_batched
   :outline:
.. doxygenfunction:: rocblas_caxpy_batched
   :outline:
.. doxygenfunction:: rocblas_zaxpy_batched

The ``axpy_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_saxpy_strided_batched
   :outline:
.. doxygenfunction:: rocblas_daxpy_strided_batched
   :outline:
.. doxygenfunction:: rocblas_haxpy_strided_batched
   :outline:
.. doxygenfunction:: rocblas_caxpy_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zaxpy_strided_batched

The ``axpy_strided_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. _rocblas_copy:

rocblas_Xcopy + batched, strided_batched
========================================

.. doxygenfunction:: rocblas_scopy
   :outline:
.. doxygenfunction:: rocblas_dcopy
   :outline:
.. doxygenfunction:: rocblas_ccopy
   :outline:
.. doxygenfunction:: rocblas_zcopy

The ``copy`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_scopy_batched
   :outline:
.. doxygenfunction:: rocblas_dcopy_batched
   :outline:
.. doxygenfunction:: rocblas_ccopy_batched
   :outline:
.. doxygenfunction:: rocblas_zcopy_batched

The ``copy_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_scopy_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dcopy_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ccopy_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zcopy_strided_batched

The ``copy_strided_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. _rocblas_dot:

rocblas_Xdot + batched, strided_batched
=======================================

.. doxygenfunction:: rocblas_sdot
   :outline:
.. doxygenfunction:: rocblas_ddot
   :outline:
.. doxygenfunction:: rocblas_hdot
   :outline:
.. doxygenfunction:: rocblas_bfdot
   :outline:
.. doxygenfunction:: rocblas_cdotu
   :outline:
.. doxygenfunction:: rocblas_cdotc
   :outline:
.. doxygenfunction:: rocblas_zdotu
   :outline:
.. doxygenfunction:: rocblas_zdotc

The ``dot/c/u`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_sdot_batched
   :outline:
.. doxygenfunction:: rocblas_ddot_batched
   :outline:
.. doxygenfunction:: rocblas_hdot_batched
   :outline:
.. doxygenfunction:: rocblas_bfdot_batched
   :outline:
.. doxygenfunction:: rocblas_cdotu_batched
   :outline:
.. doxygenfunction:: rocblas_cdotc_batched
   :outline:
.. doxygenfunction:: rocblas_zdotu_batched
   :outline:
.. doxygenfunction:: rocblas_zdotc_batched

The ``dot/c/u_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_sdot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ddot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_hdot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_bfdot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cdotu_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cdotc_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zdotu_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zdotc_strided_batched

The ``dot/c/u_strided_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. _rocblas_nrm2:

rocblas_Xnrm2 + batched, strided_batched
========================================

.. doxygenfunction:: rocblas_snrm2
   :outline:
.. doxygenfunction:: rocblas_dnrm2
   :outline:
.. doxygenfunction:: rocblas_scnrm2
   :outline:
.. doxygenfunction:: rocblas_dznrm2

The ``nrm2`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_snrm2_batched
   :outline:
.. doxygenfunction:: rocblas_dnrm2_batched
   :outline:
.. doxygenfunction:: rocblas_scnrm2_batched
   :outline:
.. doxygenfunction:: rocblas_dznrm2_batched

The ``nrm2_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_snrm2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dnrm2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_scnrm2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dznrm2_strided_batched

The ``nrm2_strided_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. _rocblas_rot:

rocblas_Xrot + batched, strided_batched
=======================================

.. doxygenfunction:: rocblas_srot
   :outline:
.. doxygenfunction:: rocblas_drot
   :outline:
.. doxygenfunction:: rocblas_crot
   :outline:
.. doxygenfunction:: rocblas_csrot
   :outline:
.. doxygenfunction:: rocblas_zrot
   :outline:
.. doxygenfunction:: rocblas_zdrot

The ``rot`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_srot_batched
   :outline:
.. doxygenfunction:: rocblas_drot_batched
   :outline:
.. doxygenfunction:: rocblas_crot_batched
   :outline:
.. doxygenfunction:: rocblas_csrot_batched
   :outline:
.. doxygenfunction:: rocblas_zrot_batched
   :outline:
.. doxygenfunction:: rocblas_zdrot_batched

The ``rot_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_srot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_drot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_crot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csrot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zrot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zdrot_strided_batched

The ``rot_strided_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. _rocblas_rotg:

rocblas_Xrotg + batched, strided_batched
========================================

.. doxygenfunction:: rocblas_srotg
   :outline:
.. doxygenfunction:: rocblas_drotg
   :outline:
.. doxygenfunction:: rocblas_crotg
   :outline:
.. doxygenfunction:: rocblas_zrotg

The ``rotg`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_srotg_batched
   :outline:
.. doxygenfunction:: rocblas_drotg_batched
   :outline:
.. doxygenfunction:: rocblas_crotg_batched
   :outline:
.. doxygenfunction:: rocblas_zrotg_batched

The ``rotg_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_srotg_strided_batched
   :outline:
.. doxygenfunction:: rocblas_drotg_strided_batched
   :outline:
.. doxygenfunction:: rocblas_crotg_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zrotg_strided_batched

The ``rotg_strided_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. _rocblas_rotm:

rocblas_Xrotm + batched, strided_batched
========================================

.. doxygenfunction:: rocblas_srotm
   :outline:
.. doxygenfunction:: rocblas_drotm

The ``rotm`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_srotm_batched
   :outline:
.. doxygenfunction:: rocblas_drotm_batched

The ``rotm_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_srotm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_drotm_strided_batched

The ``rotm_strided_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. _rocblas_rotmg:

rocblas_Xrotmg + batched, strided_batched
=========================================

.. doxygenfunction:: rocblas_srotmg
   :outline:
.. doxygenfunction:: rocblas_drotmg

The ``rotmg`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_srotmg_batched
   :outline:
.. doxygenfunction:: rocblas_drotmg_batched

The ``rotmg_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_srotmg_strided_batched
   :outline:
.. doxygenfunction:: rocblas_drotmg_strided_batched

The ``rotmg_strided_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. _rocblas_scal:

rocblas_Xscal + batched, strided_batched
=========================================

.. doxygenfunction:: rocblas_sscal
   :outline:
.. doxygenfunction:: rocblas_dscal
   :outline:
.. doxygenfunction:: rocblas_cscal
   :outline:
.. doxygenfunction:: rocblas_zscal
   :outline:
.. doxygenfunction:: rocblas_csscal
   :outline:
.. doxygenfunction:: rocblas_zdscal

The ``scal`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_sscal_batched
   :outline:
.. doxygenfunction:: rocblas_dscal_batched
   :outline:
.. doxygenfunction:: rocblas_cscal_batched
   :outline:
.. doxygenfunction:: rocblas_zscal_batched
   :outline:
.. doxygenfunction:: rocblas_csscal_batched
   :outline:
.. doxygenfunction:: rocblas_zdscal_batched

The ``scal_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_sscal_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dscal_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cscal_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zscal_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csscal_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zdscal_strided_batched

The ``scal_strided_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. _rocblas_swap:

rocblas_Xswap + batched, strided_batched
=========================================

.. doxygenfunction:: rocblas_sswap
   :outline:
.. doxygenfunction:: rocblas_dswap
   :outline:
.. doxygenfunction:: rocblas_cswap
   :outline:
.. doxygenfunction:: rocblas_zswap

The ``swap`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_sswap_batched
   :outline:
.. doxygenfunction:: rocblas_dswap_batched
   :outline:
.. doxygenfunction:: rocblas_cswap_batched
   :outline:
.. doxygenfunction:: rocblas_zswap_batched

The ``swap_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: rocblas_sswap_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dswap_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cswap_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zswap_strided_batched

The ``swap_strided_batched`` functions support the ``_64`` interface. See the :ref:`ILP64 API` section.
