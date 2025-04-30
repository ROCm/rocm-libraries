.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Tensile supported problems, GEMM, Tensor

.. _supported-problems:

===================
Supported problems
===================

This topic discusses the types of problems supported by Tensile.


Problem types
===============

Here are some of the common problem types supported by Tensile for creating benchmark-driven library.

- Four variants of standard GEMM with two free indices (i,j) and one summation index l.

  - NN where N = Non-transpose:

    - C[i,j] = Sum[l] A[i,l] * B[l,j]

  - NT where T = Transpose:

    - C[i,j] = Sum[l] A[i,l] * B[j, l]

  - TN:

    - C[i,j] = Sum[l] A[l, i] * B[l,j]

  - TT:

    - C[i,j] = Sum[l] A[l, i] * B[j, l]

- Batched-GEMM with two free indices, one batched index k, and one summation index l:

  - C[i,j,k] = Sum[l] A[i,l,k] * B[l,j,k]

- 2D summation:

  - C[i,j] = Sum[k,l] A[i,k,l] * B[j,l,k]

- GEMM with three batched indices:

  - C[i,j,k,l,m] = Sum[n] A[i,k,m,l,n] * B[j,k,l,n,m]

- Four free indices, two summation indices, and one batched index:

  - C[i,j,k,l,m] = Sum[n,o] A[i,k,m,o,n] * B[j,m,l,n,o]

- Batched image convolution mapped to 7D tensor contraction:

  - C[i,j,k,l] = Sum[m,n] A[i,j,m,n,l] * B[m,n,k,j,l]

The nomenclature of these problems is explained in the following section.

Nomenclature
==============

The indices describe the dimensionality of the problem to be solved. A GEMM operation takes two 2-dimensional matrices as input, adding up to four input dimensions and contracts them along one dimension, which cancels out two dimensions, leading to a 2-dimensional result.
When an index shows up in multiple tensors, those tensors must be the same size along with the dimension however, they can have different strides.

There are three categories of indices or dimensions used in the problems supported by Tensile: free, batch and bound. These are discussed in the following sections with reference to the examples given in the `problem types <problem_types>`.

Free indices
-------------

Free indices are the paired indices of tensor C with one pair in tensor A and another pair in tensor B. i,j,k, and l are the four free indices of tensor C where indices i and k are present in tensor A while indices j and l are present in tensor B.

Batch indices
---------------

Batch indices are the indices of tensor C that are present in both tensor A and tensor B.
The difference between the GEMM example and the batched-GEMM example is the additional index.
In the batched-GEMM example, the index k is the batch index, which batches together multiple independent GEMMs.

Bound indices
---------------

The bound indices are also known as summation indices. These indices are not present in tensor C but in the summation symbol (Sum[k]) and in tensors A and B. The inner products (pairwise multiply then sum) are performed along these indices.

Limitation
-------------

Tensile supports only problems with at least one pair of free indices.
