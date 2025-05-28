.. meta::
  :description: hipSOLVER refactorization helper functions API documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation, refactorization, helper functions

.. _refactor_helpers:

**********************************
Refactorization helper functions
**********************************

These helper functions control aspects of the hipSOLVER library. They are divided 
into the following categories:

* :ref:`refactor_initialize`: Functions to initialize and cleanup the library handle.
* :ref:`refactor_input`: Functions to manipulate function input.
* :ref:`refactor_output`: Functions to access function output.
* :ref:`refactor_parameters`: Functions to manipulate parameters.


.. _refactor_initialize:

Handle setup and teardown
===============================

.. contents:: List of handle initialization functions
   :local:
   :backlinks: top

hipsolverRfCreate()
-----------------------------------------
.. doxygenfunction:: hipsolverRfCreate

hipsolverRfDestroy()
-----------------------------------------
.. doxygenfunction:: hipsolverRfDestroy



.. _refactor_input:

Input manipulation
===============================

.. contents:: List of input functions
   :local:
   :backlinks: top

hipsolverRfSetupDevice()
-----------------------------------------
.. doxygenfunction:: hipsolverRfSetupDevice

hipsolverRfSetupHost()
-----------------------------------------
.. doxygenfunction:: hipsolverRfSetupHost

.. _refactor_batch_setup_host:

hipsolverRfBatchSetupHost()
-----------------------------------------
.. doxygenfunction:: hipsolverRfBatchSetupHost

hipsolverRfAnalyze()
-----------------------------------------
.. doxygenfunction:: hipsolverRfAnalyze

.. _refactor_batch_analyze:

hipsolverRfBatchAnalyze()
-----------------------------------------
.. doxygenfunction:: hipsolverRfBatchAnalyze

hipsolverRfResetValues()
-----------------------------------------
.. doxygenfunction:: hipsolverRfResetValues

.. _refactor_batch_reset_values:

hipsolverRfBatchResetValues()
-----------------------------------------
.. doxygenfunction:: hipsolverRfBatchResetValues



.. _refactor_output:

Output manipulation
===============================

.. contents:: List of output functions
   :local:
   :backlinks: top

hipsolverRfAccessBundledFactorsDevice()
-----------------------------------------
.. doxygenfunction:: hipsolverRfAccessBundledFactorsDevice

hipsolverRfExtractBundledFactorsHost()
-----------------------------------------
.. doxygenfunction:: hipsolverRfExtractBundledFactorsHost

hipsolverRfExtractSplitFactorsHost()
-----------------------------------------
.. doxygenfunction:: hipsolverRfExtractSplitFactorsHost

.. _refactor_batch_zero_pivot:

hipsolverRfBatchZeroPivot()
-----------------------------------------
.. doxygenfunction:: hipsolverRfBatchZeroPivot



.. _refactor_parameters:

Parameter manipulation
===============================

.. contents:: List of parameter functions
   :local:
   :backlinks: top

hipsolverRfGet_Algs()
-----------------------------------------
.. doxygenfunction:: hipsolverRfGet_Algs

hipsolverRfGetMatrixFormat()
-----------------------------------------
.. doxygenfunction:: hipsolverRfGetMatrixFormat

hipsolverRfGetNumericBoostReport()
-----------------------------------------
.. doxygenfunction:: hipsolverRfGetNumericBoostReport

hipsolverRfGetNumericProperties()
-----------------------------------------
.. doxygenfunction:: hipsolverRfGetNumericProperties

hipsolverRfGetResetValuesFastMode()
-----------------------------------------
.. doxygenfunction:: hipsolverRfGetResetValuesFastMode

.. _refactor_set_algs:

hipsolverRfSetAlgs()
-----------------------------------------
.. doxygenfunction:: hipsolverRfSetAlgs

.. _refactor_set_matrix_format:

hipsolverRfSetMatrixFormat()
-----------------------------------------
.. doxygenfunction:: hipsolverRfSetMatrixFormat

.. _refactor_set_numeric_properties:

hipsolverRfSetNumericProperties()
-----------------------------------------
.. doxygenfunction:: hipsolverRfSetNumericProperties

.. _refactor_set_reset_values_fast_mode:

hipsolverRfSetResetValuesFastMode()
-----------------------------------------
.. doxygenfunction:: hipsolverRfSetResetValuesFastMode

