.. meta::
  :description: runtime compilation in rocFFT
  :keywords: rocFFT, ROCm, API, documentation, runtime, compilation

.. _runtime-compilation:

********************************************************************
Runtime compilation
********************************************************************

rocFFT includes many kernels for common FFT problems. Many plans
require additional kernels aside from the ones built into the
library. In these cases, rocFFT compiles optimized kernels for
the plan when the plan is created.

Compiled kernels are stored in memory by default. They will be reused
if they are required again for plans in the same process.

If the ``ROCFFT_RTC_CACHE_PATH`` environment variable is set to a
writable file location, rocFFT writes the compiled kernels to this
location. rocFFT reads the kernels from this location for plans in
other processes that need runtime-compiled kernels. rocFFT will
create the specified file if it does not already exist.
