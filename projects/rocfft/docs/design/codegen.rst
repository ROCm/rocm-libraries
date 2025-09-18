.. meta::
  :description: rocFFT documentation and API reference library
  :keywords: rocFFT, ROCm, API, documentation

.. _codegen:

********************************************************************
Code Generator Design Document for rocFFT
********************************************************************

Proposal
========

Create a new code generator for rocFFT.

Rationale
---------

The current code generator:

* dates from clFFT
* is based on string concatenation

Ideally, a new code generator:

* based on an abstract-syntax-tree (AST)
* generates faster, more robust kernels

ASTs allow generated code to be transformed and manipulated before
being emitted.  A concrete example of this for FFT kernels would be:
automatically translating a kernel from interleaved to planar format.

How the generator is designed and implemented is crucial for both
conciseness and ease-of-use.


Required kernels (scope)
========================

For rocFFT, we need/want to generate:

* Host functions to launch the FFT kernels
* Tiled (row/column) + strided + batched Stockham kernels for
  arbitrary factorization
* May want to extend to Cooley-Tukey kernels as well

Kernels need to handle all combinations of:

* single/double precision (and be extendable to half-float and bfloat)
* in-place/out-of-place
* planar/interleaved
* real/complex
* small/large twiddle tables
* unit/non-unit stride
* transposed output, including with twiddle multiplies for large 1D
* fusing with pre and post-processing kernels (e.g. real even-length)

Ideally any configuration/runtime parameters required by the kernels
would be defined in a single place to avoid repetition between rocFFT
and the generator.

We have flexibility in handling these combinations at compile-time or
run-time.  For example, multiple kernels could be generated for
single/double precision, but unit/non-unit stride could be handled at
runtime.

Fundamentally, all multidimensional and batched FFTs can be written in
terms of 1D transforms (with affine indexing).  As such, an FFT is
broken down into:

* A *host* function that is aware of dimensions, strides, batches, and
  tiling.  This function would be responsible for determining how the
  problem will be broken down into GPU thread blocks.
* A *global* function that is aware of GPU thread blocks, dimensions,
  strides, batches, and tiling.  This function would be responsible
  for determining offsets and strides for the device function, and
  declaring LDS memory buffers.
* A *device* function that is passed offsets and strides, and is aware
  of GPU threads.  The device function would perform a (short) 1D
  transform.

A device function may be called so that a thread block is actually
transforming multiple batches.  As such, indexes (the spatial index in
the FFT) should be computed as:

.. code-block:: c

   int fft_index = threadIdx.x % width;


Tiling
======

Launching device kernels in a way that traverses memory in tiles will
be handled at the host/global level.

Kernels need to support reading/writing in columns/rows.  These are
the block CC/RC/CR flavors (where C and R refer to column and row) of
the existing kernels.


Strides and batches
===================

Host
----

Host/global functions should support arbitrary dimensions, lengths,
strides, offsets, and batches.

Users should be allowed to store their arrays arbitrarily.  For an
:math:`N` dimensional dataset, the flat index :math:`a` corresponding
to indices :math:`(i_1,\ldots,i_N,i_b)`, where :math:`i_b` is the
batch index, is given by

.. math::

   a(i_1,\ldots,i_N,i_b) = s_b i_b + \sum_{d=1}^N s_d i_d

Where :math:`s_d` is the stride along dimension :math:`d`.  To support
these strides, the device function to compute the FFT along dimension
:math:`D` would be passed:

.. code-block:: c

   int offset = 0;
   offset += batch_index * batch_stride;
   for (int d=0; d < N; ++d)
     if (d != D)
       offset += spatial_index[d] * strides[d];

   int stride = strides[D];

For example, in three dimensions, to compute the FFT along the
y-dimension given x and z indices ``i`` and ``k`` for batch ``b``,
the device function would be passed:

.. code-block:: c

   int offset = 0;
   offset += b * batch_stride;
   offset += i * strides[0];
   offset += k * strides[2];

   int stride = strides[1];

Device
------

Device functions should support arbitrary offsets and strides.  Array
indexes in device functions should be computed as, eg:

.. code-block:: c

   int fft_index = threadIdx.x % width;
   int array_index = offset + fft_index * stride;


Large twiddle tables
====================

Large 1D transforms are decomposed into multiple transforms.  To
reduce the size of twiddle tables, rotations can be decomposed into
multiple stages as well.  For example, the rotation through
:math:`2\pi \cdot 280 / 256^2` can be decomposed into :math:`2\pi
\cdot 1 / 256 + 2\pi 24 / 256^2`.  The resulting twiddle table
contains 512 entries instead of 65536 entries.

Generated kernels should support these "large twiddle tables".


Launching
=========

For a specific transform length, the generator is free to choose among
several algorithms and related tuning parameters.  These choices may
influence how the kernel is launched.  The generator will create both
the kernel and the accompanying struct, which gives indications of how
the kernel may be used in both rocFFT and other applications.

The generator will populate a function pool with structs of the form

.. code-block:: c++

    struct ROCFFTKernel
    {
        void *device_function = nullptr;
        std::vector<int> factors;
        int              transforms_per_block = 0;
        int              workgroup_size = 0;
        // ...
    };

This moves the responsibility of figuring how a kernel should be
launched to the generator.

Currently, kernels are launched with:

* dimension
* number of blocks (batches)
* number of threads (threads per batch; kernel parameter)
* stream
* twiddle table
* length(s)
* strides
* batch count
* in/out buffers


Implementation
==============

The code generator will be implemented in Python using only standard
modules.

The AST will be represented as a tree structure, with nodes in the
tree representing operations, such as assignment, addition, or a block
containing multiple operations.  Nodes will be represented as objects
(e.g., ``Add``) extending the base class ``BaseNode``.  Operands will be
stored in a simple list called ``args``:

.. code-block:: python

    class BaseNode:
        args: List[Any]


To facilitate building ASTs, the base node will have a constructor
that simply stores its arguments as operands:

.. code-block:: python

    class BaseNode:
        args: List[Any]
        def __init__(self, *args, **kwargs):
            self.args = list(args)


To facilitate rewriting ASTs, node object's constructors should accept
a simple list of argument/operands.

This, for example, allows a depth-first tree re-write to be
implemented trivially as:

.. code-block:: python

    def depth_first(x, f):
        '''Depth first traversal of the AST in 'x'.  Each node is transformed by 'f(x)'.'''
        if isinstance(x, BaseNode):
            y = type(x)(*[ depth_first(a, f) for a in x.args ])
            return f(y)
        return f(x)

To emit code, each node must implement ``__str__``.  For example:

.. code-block:: python

    class Add(BaseNode):
        def __str__(self):
            return ' + '.join([ str(x) for x in self.args ])


Stockham tiling implementation
------------------------------

To support tiling, the *global* function is responsible for loading
data from global memory into LDS memory in a tiled manner.  Once in
LDS memory, a singly strided *device* function performs an
interleaved, in-place FFT entirely within LDS.

Polymorphism will be used to abstract tiling strategies.  Different
tiling strategies should extend the ``StockhamTiling`` object and
overload the ``load_from_global`` and ``store_to_global`` methods.

For example:

.. code-block:: python

    tiling = StockhamTilingRR()
    scheme = StockhamDeviceKernelUWide()

    body = StatementList()
    body += tiling.compute_offsets(...)
    body += tiling.load_from_global(out=lds, in=global_buffer)
    body += scheme.fft(lds)
    body += tiling.store_to_global(out=global_buffer, in=lds)

Different tiling strategies may require new template parameters and/or
function arguments.  Tiling strategies can manipulate the following methods:

* ``add_templates``
* ``add_global_arguments``
* ``add_device_arguments``
* ``add_device_call_arguments``

Each of these methods is passed a ``TemplateList`` or
``ArgumentList`` argument, and should return a new template/argument
list with any extra parameters added.


Large twiddle tables
--------------------

Device kernels may need to apply additional twiddles during their
execution.  These extra twiddle tables are implemented similarly to
tiling.  Different twiddle table strategies should extend the
``StockhamLargeTwiddles`` object and overload the ``load`` and
``multiply`` methods.

Twiddle tables may also require additional templates and arguments.
See :ref:`Stockham tiling implementation`.

Copyright and disclaimer
========================

The information contained herein is for informational purposes only,
and is subject to change without notice. While every precaution has
been taken in the preparation of this document, it may contain
technical inaccuracies, omissions and typographical errors, and AMD is
under no obligation to update or otherwise correct this information.
Advanced Micro Devices, Inc. makes no representations or warranties
with respect to the accuracy or completeness of the contents of this
document, and assumes no liability of any kind, including the implied
warranties of non-infringement, merchantability or fitness for
particular purposes, with respect to the operation or use of AMD
hardware, software or other products described herein.  No license,
including implied or arising by estoppel, to any intellectual property
rights is granted by this document.  Terms and limitations applicable
to the purchase or use of AMD’s products are as set forth in a signed
agreement between the parties or in AMD's Standard Terms and
Conditions of Sale.

AMD is a trademark of Advanced Micro Devices, Inc. Other product
names used in this publication are for identification purposes only
and may be trademarks of their respective companies.

Copyright (C) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.


