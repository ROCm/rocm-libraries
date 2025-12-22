.. meta::
  :description: hipFFT documentation, hipFFTW documentation, and API reference library
  :keywords: FFT, hipFFT, hipFFTW, rocFFT, ROCm, API, documentation

.. _hipfftw-api-usage:

********************************************************************
hipFFTW API usage
********************************************************************

This section describes how to use the hipFFTW library API. The hipFFTW
API is a partial reproduction of the FFTW_ API.

.. _FFTW: https://www.fftw.org/

Data types
==========

Similarly to FFTW_, hipFFTW uses the following specific types.

.. doxygentypedef:: fftw_complex

.. doxygentypedef:: fftwf_complex

.. doxygentypedef:: fftw_plan

.. doxygentypedef:: fftwf_plan

.. doxygentypedef:: fftw_iodim

.. doxygentypedef:: fftwf_iodim

.. doxygentypedef:: fftw_iodim64

.. doxygentypedef:: fftwf_iodim64

Constant values
===============

.. _hipfftw-flag-values:

Flag values
-----------

All the hipFFTW :ref:`plan creation functions <hipfftw-plan-creation>` request an ``unsigned flags`` argument.
The value of that argument conditions what hipFFTW will consider when creating the requested plan. Valid
values are bitwise OR of zero of more of the following constant values.

.. doxygendefine:: FFTW_MEASURE

.. doxygendefine:: FFTW_DESTROY_INPUT

.. doxygendefine:: FFTW_UNALIGNED

.. doxygendefine:: FFTW_CONSERVE_MEMORY

.. doxygendefine:: FFTW_EXHAUSTIVE

.. doxygendefine:: FFTW_PRESERVE_INPUT

.. doxygendefine:: FFTW_PATIENT

.. doxygendefine:: FFTW_ESTIMATE

.. doxygendefine:: FFTW_WISDOM_ONLY

.. note::
  Even if seemingly accepted, flag values are currently ignored by hipFFTW. In particular, note that:

  - no measurement is ever done at plan creation (plan configurations are chosen based on heuristics);
  - preservation of input data is never guaranteed;
  - requests for a minimized memory footprint are ignored;

  with the current status of hipFFTW.

.. _hipfftw-sign-values:

Sign values
-----------

The hipFFTW :ref:`plan creation functions <hipfftw-plan-creation>` for complex transforms request an ``int sign``
argument. This argument determines whether the plan being created is meant to compute forward or backward (inverse)
discrete Fourier transforms. Valid values of that argument are either of the following constant values.

.. doxygendefine:: FFTW_FORWARD

.. doxygendefine:: FFTW_BACKWARD

.. _hipfftw-buffer-management:

Buffer management
=================

hipFFTW supports the following buffer-management functions.

.. doxygenfunction:: fftw_malloc

.. doxygenfunction:: fftwf_malloc

.. doxygenfunction:: fftw_alloc_real

.. doxygenfunction:: fftwf_alloc_real

.. doxygenfunction:: fftw_alloc_complex

.. doxygenfunction:: fftwf_alloc_complex

.. doxygenfunction:: fftw_free

.. doxygenfunction:: fftwf_free

The memory blocks allocated by any of the above ``{fftw,fftwf}_{malloc,alloc_real,alloc_complex}``
are always directly accessible to the host, but not necessarily to the GPU. hipFFTW decides on the
type of memory being allocated following a ranked-choice strategy. It attempts to allocate:

1. pinned host memory (first);
2. pageable host memory (if the latter failed or if the request exceeds a :ref:`user-defined threshold <hipfftw-env-byte-size-host-alloc>`).

.. important::
  Every buffer allocated via ``fftw_malloc``, ``fftwf_malloc``, ``fftw_alloc_real``, ``fftwf_alloc_real``,
  ``fftw_alloc_complex``, or ``fftwf_alloc_complex`` **must** be freed using ``fftw_free`` or ``fftwf_free``.

.. _hipfftw-plan-creation:

Plan creation
=============

hipFFTW supports the creation of :ref:`basic <hipfftw-basic-plan-creation>`, :ref:`advanced <hipfftw-advanced-plan-creation>`,
and :ref:`general <hipfftw-general-plan-creation>` plans, using interleaved formats for complex floating-point data.
Plans capture:

- the type of transform, namely, complex or real, forward or backward (inverse) discrete Fourier transform;
- the length(s) and batch size(s) of the transform;
- layout information for input and output data, for example, stride(s), distance(s);
- pointers to buffers that it should consider as input and output data buffers when computing a transform via a :ref:`generic execution function <hipfftw-execute-with-creation-io>`;
- the flag value(s) to use at creation.

A plan of interest can be created using more than one plan creation function. For instance, any plan created
by ``fftw_plan_dft`` (or ``fftwf_plan_dft``) could be created by ``fftw_plan_many_dft``
(or ``fftwf_plan_many_dft``) without any difference.

For all plan creation functions, the requested plan is said to be configured for in-place operations if identical
input and output buffers are used when the plan is created. The plan is configured for out-of-place operations otherwise.

.. _hipfftw-unsupported-configurations:

.. note::
  - hipFFTW does not support split complex formats, real-to-real transforms, nor distributed transforms;
  - hipFFTW does not support transforms of more than 3 dimensions, that is, ``rank > 3`` is not supported;
  - hipFFTW does not support transforms of more than 1 batch dimension, that is, ``howmany_rank > 1`` is not supported.

.. _hipfftw-basic-plan-creation:

Basic plans
-----------

Basic plans are implicitly configured for unbatched transforms with compact, default data layouts. For :math:`d`-dimensional (:math:`d > 0`)
transforms of lengths :math:`n_0 \times n_1 \times \ldots \times n_{d-1}` (:math:`n_{i} > 0\, \forall i \in \lbrace 0, 1, \ldots, d-1\rbrace`),
default data layouts use strides :math:`s_i` along dimension :math:`i` where

- :math:`s_{d-1} = 1`;
- if :math:`d > 1`,
    - :math:`s_{d-2} = 2 \lfloor n_{d-1} / 2 + 1 \rfloor` on input (resp. output) and :math:`s_{d-2} = \lfloor n_{d-1} / 2 + 1 \rfloor` on output (resp. input) for forward (resp. backward) real in-place transforms;
    - :math:`s_{d-2} = n_{d-1}` on input (resp. output) and :math:`s_{d-2} = \lfloor n_{d-1} / 2 + 1 \rfloor` on output (resp. input) for forward (resp. backward) real out-of-place transforms;
    - :math:`n_{d-1}` otherwise.
- if :math:`d > 2`, :math:`s_{i} = n_{i+1}s_{i+1}` for :math:`0 \leq i < d-2`.

The following functions can be used for creating basic hipFFTW plans.

.. doxygenfunction:: fftw_plan_dft_1d
.. doxygenfunction:: fftwf_plan_dft_1d
.. doxygenfunction:: fftw_plan_dft_2d
.. doxygenfunction:: fftwf_plan_dft_2d
.. doxygenfunction:: fftw_plan_dft_3d
.. doxygenfunction:: fftwf_plan_dft_3d
.. doxygenfunction:: fftw_plan_dft
.. doxygenfunction:: fftwf_plan_dft
.. doxygenfunction:: fftw_plan_dft_r2c_1d
.. doxygenfunction:: fftwf_plan_dft_r2c_1d
.. doxygenfunction:: fftw_plan_dft_r2c_2d
.. doxygenfunction:: fftwf_plan_dft_r2c_2d
.. doxygenfunction:: fftw_plan_dft_r2c_3d
.. doxygenfunction:: fftwf_plan_dft_r2c_3d
.. doxygenfunction:: fftw_plan_dft_r2c
.. doxygenfunction:: fftwf_plan_dft_r2c
.. doxygenfunction:: fftw_plan_dft_c2r_1d
.. doxygenfunction:: fftwf_plan_dft_c2r_1d
.. doxygenfunction:: fftw_plan_dft_c2r_2d
.. doxygenfunction:: fftwf_plan_dft_c2r_2d
.. doxygenfunction:: fftw_plan_dft_c2r_3d
.. doxygenfunction:: fftwf_plan_dft_c2r_3d
.. doxygenfunction:: fftw_plan_dft_c2r
.. doxygenfunction:: fftwf_plan_dft_c2r

.. _hipfftw-advanced-plan-creation:

Advanced plans
--------------

Advanced plans support batched transforms and some non-default data layouts. The following
additional arguments parameterize the input data layout for a :math:`d`-dimensional transform (:math:`d > 0`)
of lengths ``n[0] x n[1] x ... x n[d-1]``:

- ``istride`` is the elementary stride, that is, the stride along the last dimension;
- ``idist`` is the distance between consecutive data sequences in the batch;
- ``inembed`` is an array of :math:`d` integer values representing the dimensions of a :math:`d`-dimensional array embedding the input data.

``ostride``, ``odist``, and ``onembed`` parameterize the output data layout of the transform
similarly.

Setting ``inembed`` to ``NULL`` is equivalent to a using ``inembed[i] = n[i]`` for all :math:`0 \leq i < d-1`
and

- ``inembed[d-1] = n[d-1]`` for complex transforms or out-of-place real forward transforms;
- ``inembed[d-1] = n[d-1]/2 + 1`` for backward real transforms;
- ``inembed[d-1] = 2 * (n[d-1]/2 + 1)`` for in-place forward real transforms.

The same behavior holds for setting ``onembed`` to ``NULL``, subsituting "forward" for "backward"
in the description above (and vice versa).

If set explicitly, all the ``inembed`` and ``onembed`` values must be larger than or equal to their
default values.

.. note::
  Using ``NULL`` for ``inembed`` and for ``onembed`` as well as ``istride = 1`` and ``ostride = 1``
  results in the same data layout as for :ref:`basic plan<hipfftw-basic-plan-creation>` in case of
  unbatched transforms (``howmany = 1``).

Concretely, considering a batched, three-dimensional transform, the data element of logical
index :math:`\left(j_0, j_1, j_2\right)` in the :math:`k`-th batch is
``in[idist * k + istride * (j_2 + inembed[2] * (j_1 + inembed[1] * j_0))]``
on input and ``out[odist * k + ostride * (j_2 + onembed[2] * (j_1 + onembed[1] * j_0))]`` on output.

The following functions can be used for creating advanced hipFFTW plans.

.. doxygenfunction:: fftw_plan_many_dft
.. doxygenfunction:: fftwf_plan_many_dft
.. doxygenfunction:: fftw_plan_many_dft_r2c
.. doxygenfunction:: fftwf_plan_many_dft_r2c
.. doxygenfunction:: fftw_plan_many_dft_c2r
.. doxygenfunction:: fftwf_plan_many_dft_c2r


.. _hipfftw-general-plan-creation:

Arbitrary plans
---------------

Arbitrary plans support batched transforms with arbitrary data layouts.

Considering a :math:`d`-dimensional transform (:math:`d > 0`) of lengths ``n[0] x n[1] x ... x n[d-1]``
batched ``m[0] x m[1] x ... x m[q-1]`` times (:math:`q > 0`), arbitrary input and output data layouts
can be set via the ``dims`` and ``howmany_dims`` arguments of the plan creation functions below. Their
``rank`` and ``howmany_rank`` arguments capture the (strictly positive) values of :math:`d` and
:math:`q`, respectively.

Specifically, ``dims`` must be an array of :math:`d` ``fftw_iodim`` (or ``fftw_iodim64``) values such
that, for all :math:`0 \leq i < d`,

- ``dims[i].n`` is equal to ``n[i]`` (must be strictly positive);
- ``dims[i].is`` is the input stride along the ``i``-th data dimension;
- ``dims[i].os`` is the output stride along the ``i``-th data dimension.

Similarly, ``howmany_dims`` must be an array of :math:`q` ``fftw_iodim`` (or ``fftw_iodim64``) values
such that, for all :math:`0 \leq j < q`,

- ``howmany_dims[j].n`` is equal to ``m[j]`` (must be strictly positive);
- ``howmany_dims[j].is`` is the input distance along the ``j``-th batch dimension;
- ``howmany_dims[j].os`` is the output distance along the ``j``-th batch dimension.

The following functions can be used for creating arbitrary hipFFTW plans.

.. doxygenfunction:: fftw_plan_guru_dft
.. doxygenfunction:: fftwf_plan_guru_dft
.. doxygenfunction:: fftw_plan_guru_dft_r2c
.. doxygenfunction:: fftwf_plan_guru_dft_r2c
.. doxygenfunction:: fftw_plan_guru_dft_c2r
.. doxygenfunction:: fftwf_plan_guru_dft_c2r
.. doxygenfunction:: fftw_plan_guru64_dft
.. doxygenfunction:: fftwf_plan_guru64_dft
.. doxygenfunction:: fftw_plan_guru64_dft_r2c
.. doxygenfunction:: fftwf_plan_guru64_dft_r2c
.. doxygenfunction:: fftw_plan_guru64_dft_c2r
.. doxygenfunction:: fftwf_plan_guru64_dft_c2r

.. _hipfftw-data-layout-requirements:

Data layout requirements for hipFFTW plans
------------------------------------------
For complex in-place transforms, hipFFTW requires all strides and distances to be equal in input and output data layouts.
For real in-place transforms, hipFFTW requires:

- unit elementary strides along the :math:`\left(d-1\right)`-th dimension for :math:`d`-dimensional transforms;
- all other input (resp. output) strides and distances to be twice the corresponding output (resp. input) strides and distances for forward (resp. backward) transforms.

Negative strides and distances are *not* supported by hipFFTW.

.. _hipfftw-execution:

Plan execution
==============

After they are successfully created, hipFFTW plans can be executed, that is, used for computing the discrete Fourier transform that they capture.
The :ref:`generic execution functions <hipfftw-execute-with-creation-io>` implicitly reuse the input and output buffers that were set
at :ref:`plan creation <hipfftw-plan-creation>`. If that is not possible or impractical, new input and output buffers can be communicated
instead by using the :ref:`new-arrays execution functions <hipfftw-execute-with-new-io>`.

.. _hipfftw-execute-with-creation-io:

Using buffers set at plan creation for computation
--------------------------------------------------

.. doxygenfunction:: fftw_execute
.. doxygenfunction:: fftwf_execute

.. _hipfftw-execute-with-new-io:

Using new buffers for computation
---------------------------------

.. doxygenfunction:: fftw_execute_dft
.. doxygenfunction:: fftwf_execute_dft
.. doxygenfunction:: fftw_execute_dft_r2c
.. doxygenfunction:: fftwf_execute_dft_r2c
.. doxygenfunction:: fftw_execute_dft_c2r
.. doxygenfunction:: fftwf_execute_dft_c2r

.. note::
  When new input and output buffers are used for execution, they must honor the placement that was set at plan creation.
  In other words, the result of ``(void*)in == (void*)out`` must be unchanged between plan creation and plan execution.

.. note::
  If the type of memory for the output buffer is directly accessible by the host, hipFFTW enforces synchronization before
  returning from any of the above :ref:`execution functions <hipfftw-execution>` to guarantee that the results are readily
  available to the host upon completion of the execution function.

Plan destruction
================

When no longer needed or going out of scope, hipFFTW plans must be destructed using either of the following functions
matching the plan's precision.

.. doxygenfunction:: fftw_destroy_plan
.. doxygenfunction:: fftwf_destroy_plan

Other utility functions (existing yet non-functional)
=====================================================

The following functions exist in hipFFTW but are **not** functional in any way. They effectively ignore all
arguments and systematically return ``0.0`` when a ``double`` value needs to be returned.

- ``fftw_print_plan`` and ``fftwf_print_plan``;
- ``fftw_set_timelimit`` and ``fftwf_set_timelimit``;
- ``fftw_cost`` and ``fftwf_cost``;
- ``fftw_flops`` and ``fftwf_flops``;
- ``fftw_cleanup`` and ``fftwf_cleanup``.

.. _hipfftw-env-vars:

Environment variables specific to hipFFTW
=========================================

.. _hipfftw-env-byte-size-host-alloc:

Enforcing size limits on types of memory for user-managed buffers
-----------------------------------------------------------------

By setting any of the environment variables

- ``HIPFFTW_BYTE_SIZE_LIMIT_PINNED_HOST_ALLOC``;
- ``HIPFFTW_BYTE_SIZE_LIMIT_PAGEABLE_HOST_ALLOC``;

to a non-negative integer value, users can instruct hipFFTW to observe the value set for the maximal
byte size that can be considered for the corresponding kind of host-accessible memory for any individual
allocation requested via the :ref:`buffer allocation functions <hipfftw-buffer-management>`. Setting
any of the above variables to ``0`` effectively prevents the corresponding kind of memory to be
considered in user-requested buffer allocations altogether.

.. _hipfftw-env-verbose-exceptions:

Making hipFFTW verbose
----------------------

Debugging failures or errors suspected to be triggered by hipFFTW can be challenging, particularly
if the root cause lies in any of the :ref:`execution functions <hipfftw-execution>`, given
that these functions' signatures make such errors silent by design. Setting the environment variable
``HIPFFTW_LOG_EXCEPTIONS`` to a strictly positive integer value effectively instructs hipFFTW to become
verbose about internal exceptions it might encounter by reporting them to the standard error stream.

.. note::
  The hipFFTW interface is C-compatible, even when ``HIPFFTW_LOG_EXCEPTIONS`` is set as
  described above.

