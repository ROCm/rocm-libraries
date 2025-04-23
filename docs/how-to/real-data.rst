.. meta::
  :description: How rocFFT handles real data
  :keywords: rocFFT, ROCm, API, real data, FFT, DFT, documentation

Real data
=========

When real data serves as input to a DFT, the resulting complex output data follows a special property, which is that about half of the
output is redundant because it consists of complex conjugates of the other half. This is called
Hermitian redundancy. So it's only necessary to store the non-redundant part of the data.
Most FFT libraries use this property to
offer specific storage layouts for FFTs involving real data. rocFFT
provides three enumeration values for :cpp:enum:`rocfft_array_type` to deal with real data FFTs:

*  ``REAL``: :cpp:enumerator:`rocfft_array_type_real`
*  ``HERMITIAN_INTERLEAVED``: :cpp:enumerator:`rocfft_array_type_hermitian_interleaved`
*  ``HERMITIAN_PLANAR``: :cpp:enumerator:`rocfft_array_type_hermitian_planar`

The ``REAL`` (:cpp:enumerator:`rocfft_array_type_real`) enumeration specifies that the data is purely real.
This can be used to feed real input or get back real output. The ``HERMITIAN_INTERLEAVED``
(:cpp:enumerator:`rocfft_array_type_hermitian_interleaved`) and ``HERMITIAN_PLANAR``
(:cpp:enumerator:`rocfft_array_type_hermitian_planar`) enumerations are similar to the corresponding
full complex enumerations in the way
they store real and imaginary components but store only about half of the complex output. Client applications can perform a
forward transform and analyze the output or they can process the output and do a backward transform to get real data back.
This is illustrated in the following figure.

.. figure:: ../data/images/realfft_fwdinv.jpg

   **Forward and backward real FFTs**

.. note::

   Real backward FFTs require that the input data be
   Hermitian-symmetric, which would naturally happen in the output of a
   real forward FFT. rocFFT will produce undefined results if
   this requirement is not met.

Consider the full output of a 1D real FFT of length :math:`N`, as shown in following figure:

.. figure:: ../data/images/realfft_1dlen.jpg

   **1D real FFT of length N**

Here, ``C*`` denotes the complex conjugate. Because the values at indices greater than :math:`N/2` can be deduced from the first half
of the array, rocFFT only stores the data up to the index :math:`N/2`. This means that the output contains only :math:`1 + N/2` complex
elements, where the division :math:`N/2` is rounded down. Examples for even and odd lengths are given below.

An example for :math:`N = 8` is shown in following figure.

.. figure:: ../data/images/realfft_ex_n8.jpg

   **Example for N = 8**

An example for :math:`N = 7` is shown in following figure.

.. figure:: ../data/images/realfft_ex_n7.jpg

   **Example for N = 7**

For a length of 8, only :math:`(1 + 8/2) = 5` of the output complex numbers are stored, with the index ranging from 0 through 4.
Similarly, for a length of 7, only :math:`(1 + 7/2) = 4` of the output complex numbers are stored, with the index ranging from 0 through 3.
For 2D and 3D FFTs, the FFT length along the innermost dimension is used to compute the :math:`(1 + N/2)` value. This is because
the FFT along the innermost dimension is computed first and is logically a real-to-Hermitian transform. The FFTs that are along
other dimensions are computed next and are complex-to-complex transforms. For example, assuming :math:`Lengths[2]`
is used to set up a 2D real FFT, let :math:`N1 = Lengths[1]` and :math:`N0 = Lengths[0]`. The output FFT has :math:`N1*(1 + N0/2)` complex elements.
Similarly, for a 3D FFT with :math:`Lengths[3]`, :math:`N2 = Lengths[2]`, :math:`N1 = Lengths[1]`, and :math:`N0 = Lengths[0]`, the output has :math:`N2*N1*(1 + N0/2)`
complex elements.

Supported array type combinations
---------------------------------

Not in-place transforms:

* Forward: ``REAL`` to ``HERMITIAN_INTERLEAVED``
* Forward: ``REAL`` to ``HERMITIAN_PLANAR``
* Backward: ``HERMITIAN_INTERLEAVED`` to ``REAL``
* Backward: ``HERMITIAN_PLANAR`` to ``REAL``

In-place transforms:

* Forward: ``REAL`` to ``HERMITIAN_INTERLEAVED``
* Backward: ``HERMITIAN_INTERLEAVED`` to ``REAL``

Setting strides
---------------

The library requires the user to explicitly set input and output strides for real transforms for non-simple cases.
See the following examples to understand which values to use for input and output strides under different scenarios. These examples show
typical use cases, but you can allocate the buffers and choose a data layout according to your needs.

The following figures and examples explain the real FFT features of this library in detail.

This schematic illustrates the forward 1D FFT (real to Hermitian).

.. figure:: ../data/images/realfft_expl_01.jpg

   **1D FFT - Real to Hermitian**

This schematic shows an example of a not in-place transform with an even :math:`N` and how strides and distances are set.

.. figure:: ../data/images/realfft_expl_02.jpg

   **1D FFT - Real to Hermitian: example 1**

This schematic shows an example of an in-place transform with an even :math:`N` and how strides and distances are set.
Even though this example only deals with one buffer (in-place), the output strides/distance can take different
values compared to the input strides/distance.

.. figure:: ../data/images/realfft_expl_03.jpg

   **1D FFT - Real to Hermitian: example 2**

Here is an example of an in-place transform with an odd :math:`N` and how strides and distances are set.
Even though this example only deals with one buffer (in-place), the output strides/distance can take different
values than the input strides/distance.

.. figure:: ../data/images/realfft_expl_04.jpg

   **1D FFT - Real to Hermitian: example 3**

This schematic illustrates the backward 1D FFT (Hermitian to real).

.. figure:: ../data/images/realfft_expl_05.jpg

   **1D FFT - Hermitian to real**

Here is an example of an in-place transform with an even :math:`N` and how strides and distances are set.
Even though this example only deals with one buffer (in-place), the output strides/distance can take different
values compared to the input strides/distance.

.. figure:: ../data/images/realfft_expl_06.jpg

   **1D FFT - Hermitian to real example**

This schematic illustrates the in-place forward 2D FFT for real to Hermitian.

.. figure:: ../data/images/realfft_expl_07.jpg

   **2D FFT - Real to Hermitian in-place**

Here is an example of an in-place 2D transform and how strides and distances are set.
Even though this example only deals with one buffer (in-place), the output strides/distance can take different
values compared to the input strides/distance.

.. figure:: ../data/images/realfft_expl_08.jpg

   **2D FFT - Real to Hermitian example**
