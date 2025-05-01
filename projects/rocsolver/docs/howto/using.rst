
.. meta::
  :description: rocSOLVER documentation and API reference library
  :keywords: rocSOLVER, ROCm, API, documentation

.. _using:

*************************
Using rocSOLVER Library
*************************

Once installed, rocSOLVER can be used just like any other library with a C API.
The header file will need to be included in the user code, and both the rocBLAS and rocSOLVER shared libraries
will become link-time and run-time dependencies for the user application.

Next, some examples are used to illustrate the basic use of rocSOLVER API and rocSOLVER batched API.

QR factorization of a single matrix
================================================

The following code snippet uses rocSOLVER to compute the QR factorization of a general m-by-n real matrix in double precision.
For a full description of the used rocSOLVER routine, see the API documentation here: :ref:`rocsolver_dgeqrf() <geqrf>`.

.. literalinclude:: ../../clients/samples/example_basic.c
    :language: c

The exact command used to compile the example above may vary depending on the
system environment, but here is a typical example:

.. code-block:: bash

    /opt/rocm/bin/hipcc -I/opt/rocm/include -c example.c
    /opt/rocm/bin/hipcc -o example -L/opt/rocm/lib -lrocsolver -lrocblas example.o


QR factorization of a batch of matrices
================================================

One of the advantages of using GPUs is the ability to execute in parallel many operations of the same type but on different data sets.
Based on this idea, rocSOLVER and rocBLAS provide a `batch` version of most of their routines. These batch versions allow the user to execute
the same operation on a set of different matrices and/or vectors with a single library call.

Strided_batched version
---------------------------

The following code snippet uses rocSOLVER to compute the QR factorization of a series of general m-by-n real matrices in double precision.
The matrices must be stored in contiguous memory locations on the GPU and, in this example, are accessed by a pointer to the first matrix and a
stride value that gives the separation between one matrix and the next.
For a full description of the used rocSOLVER routine, see the API documentation of :ref:`rocsolver_dgeqrf_strided_batched() <geqrf_strided_batched>`.

.. literalinclude:: ../../clients/samples/example_strided_batched.c
    :language: c

Batched version
---------------------------

The following code snippet uses rocSOLVER to compute the QR factorization of a series of general m-by-n real matrices in double precision.
In this case, the matrices do not need to be in contiguous memory locations on the GPU, and will be accessed by an array of pointers.
For a full description of the used rocSOLVER routine, see the API documentation of :ref:`rocsolver_dgeqrf_batched <geqrf_batched>`.

.. literalinclude:: ../../clients/samples/example_batched.c
    :language: c

