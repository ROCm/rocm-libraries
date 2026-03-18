.. meta::
  :description: hipSPARSELt storage formats
  :keywords: hipSPARSELt, ROCm, API library, API reference, storage formats

.. _storage-format:

********************************
hipSPARSELt storage format
********************************

hipSPARSELt uses a structured sparsity storage format, which is represented by an
:math:`m \times n` matrix, where:

* **m** = number of rows (integer)
* **n** = number of columns (integer)
* **sparsity** = 50%, the ratio of ``nnz`` elements in every 2:1 (int) or 4:2 (others) element along the row
  (4:2 means every 4 continuous elements only has 2 ``nnz`` elements)
* **compressed matrix** = matrix of ``nnz`` elements containing data
* **metadata** = matrix of ``nnz`` elements containing the element indices in every 4:2 or 2:1 array
  along the row (the contents or structure of the metadata is dependent on the chosen solution by the backend
  implementation)

Consider the following :math:`4 \times 4` matrix and the structured sparsity structures using
:math:`m = 4, n = 4`:

.. math::
  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 0.0 \\
        0.0 & 0.0 & 3.0 & 4.0 \\
        0.0 & 6.0 & 7.0 & 0.0 \\
        0.0 & 6.0 & 0.0 & 8.0 \\
      \end{pmatrix}

where

.. math::
  Compressed A = \begin{pmatrix}
                  1.0 & 2.0 \\
                  3.0 & 4.0 \\
                  6.0 & 7.0 \\
                  6.0 & 8.0 \\
                \end{pmatrix}
  metadata =    \begin{pmatrix}
                  0 & 1 \\
                  2 & 3 \\
                  1 & 2 \\
                  1 & 3 \\
                \end{pmatrix}
