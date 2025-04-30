.. meta::
  :description: rocFFT computations
  :keywords: rocFFT, ROCm, API, documentation, install, computation, fft

.. _fft-computation:

********************************************************************
FFT computation
********************************************************************

rocFFT is an implementation of the Discrete Fourier Transform (DFT) that makes use of symmetries in the DFT definition to
reduce the mathematical complexity from :math:`O(N^2)` to :math:`O(N \log N)`.

How does the library compute DFTs?
==================================

Here are the formulas for 1D, 2D, and 3D complex DFTs:

For a 1D complex DFT:

:math:`{\tilde{x}}_j = \sum_{k=0}^{n-1}x_k\exp\left({\pm i}{{2\pi jk}\over{n}}\right)\hbox{ for } j=0,1,\ldots,n-1`

Where :math:`x_k` is the complex data to be transformed, :math:`\tilde{x}_j` is the transformed data,
and the sign :math:`\pm`
determines the direction of the transform: :math:`-` for forward and :math:`+` for backward.

For a 2D complex DFT:

:math:`{\tilde{x}}_{jk} = \sum_{q=0}^{m-1}\sum_{r=0}^{n-1}x_{rq}\exp\left({\pm i} {{2\pi jr}\over{n}}\right)\exp\left({\pm i}{{2\pi kq}\over{m}}\right)`

For :math:`j=0,1,\ldots,n-1\hbox{ and } k=0,1,\ldots,m-1`, where :math:`x_{rq}` is the complex data to be transformed,
:math:`\tilde{x}_{jk}` is the transformed data, and the sign :math:`\pm` determines the direction of the transform.

For a 3D complex DFT:

:math:`\tilde{x}_{jkl} = \sum_{s=0}^{p-1}\sum_{q=0}^{m-1}\sum_{r=0}^{n-1}x_{rqs}\exp\left({\pm i} {{2\pi jr}\over{n}}\right)\exp\left({\pm i}{{2\pi kq}\over{m}}\right)\exp\left({\pm i}{{2\pi ls}\over{p}}\right)`

For :math:`j=0,1,\ldots,n-1\hbox{ and } k=0,1,\ldots,m-1\hbox{ and } l=0,1,\ldots,p-1`, where :math:`x_{rqs}` is the complex data to
be transformed, :math:`\tilde{x}_{jkl}` is the transformed data, and the sign :math:`\pm` determines the direction of the transform.
