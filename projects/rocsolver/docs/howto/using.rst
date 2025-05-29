
.. meta::
  :description: How to use the rocSOLVER library and API
  :keywords: rocSOLVER, ROCm, API, documentation, usage guide

.. _using:

***************************
Using the rocSOLVER library
***************************

After rocSOLVER is installed, it can be used like any other library with a C API.
To use rocSOLVER, include the header file in the user code. This means that the rocBLAS and rocSOLVER shared libraries
become link-time and runtime dependencies for the user application.

Here are some examples that show how to use the rocSOLVER and rocSOLVER batched APIs.

QR factorization of a single matrix
================================================

The following code snippet uses rocSOLVER to compute the QR factorization of a general m-by-n real matrix in double precision.
For a full description of the rocSOLVER routine shown here, see the API documentation for :ref:`rocsolver_dgeqrf() <geqrf>`.

.. literalinclude:: ../../clients/samples/example_basic.c
    :language: c

The exact command required to compile the example above might vary depending on the
system environment, but here is a typical example:

.. code-block:: bash

   /opt/rocm/bin/hipcc -I/opt/rocm/include -c example.c
   /opt/rocm/bin/hipcc -o example -L/opt/rocm/lib -lrocsolver -lrocblas example.o

QR factorization of a batch of matrices
================================================

One advantage of using GPUs is the ability to execute many operations of the same type in parallel on different data sets.
Based on this idea, rocSOLVER and rocBLAS provide a ``batch`` version for most routines. These batch versions let you execute
the same operation on a set of different matrices, vectors, or both with a single library call.

Strided_batched version
---------------------------

The following code snippet uses rocSOLVER to compute the QR factorization of a series of general m-by-n real matrices in double precision.
The matrices must be stored in contiguous memory locations on the GPU and accessed by a pointer to the first matrix and a
stride value that gives the separation between one matrix and the next, as shown in this example.
For a full description of the rocSOLVER routine that is used here, see the API documentation for :ref:`rocsolver_dgeqrf_strided_batched() <geqrf_strided_batched>`.

.. literalinclude:: ../../clients/samples/example_strided_batched.c
    :language: c

Batched version
---------------------------

The following code snippet uses rocSOLVER to compute the QR factorization of a series of general m-by-n real matrices in double precision.
In this case, the matrices do not need to be in contiguous memory locations on the GPU and can be accessed by an array of pointers.
For a full description of this rocSOLVER routine, see the API documentation for :ref:`rocsolver_dgeqrf_batched <geqrf_batched>`.

.. literalinclude:: ../../clients/samples/example_batched.c
    :language: c

