.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _iterators:

********************************************************************
 Iterators
********************************************************************

Constant
==========

.. doxygenclass:: rocprim::constant_iterator
   :members:


.. note::

   For example, ``constant_iterator(20)`` generates the infinite sequence::

     20
     20
     20
     ...

Counting
==========

.. doxygenclass:: rocprim::counting_iterator
   :members:

.. note::
   For example, ``counting_iterator(20)`` generates the infinite sequence::

     20
     21
     22
     23
     ...

.. _transform:

Transform
============

.. doxygenclass:: rocprim::transform_iterator
   :members:

.. note::

   ``transform_iterator(sequence, transform)`` should generate the sequence::

     transform(sequence(0))
     transform(sequence(1))
     ...

Predicate
---------

.. doxygenclass:: rocprim::predicate_iterator
   :members:

.. note::
   ``predicate_iterator(sequence, test, predicate)`` generates the sequence::

     predicate(test[0]) ? sequence[0] : default
     predicate(test[1]) ? sequence[1] : default
     predicate(test[2]) ? sequence[2] : default
     ...

Pairing Values with Indices
=============================

.. doxygenclass:: rocprim::arg_index_iterator
   :members:

.. note::
   ``arg_index_iterator(sequence)`` generates the sequence of tuples::

     (0, sequence[0])
     (1, sequence[1])
     ...

Zip
==============

.. doxygenclass:: rocprim::zip_iterator
   :members:

.. note::
   ``zip_iterator(sequence_X, sequence_Y)`` generates the sequence of tuples::

     (sequence_X[0], sequence_Y[0])
     (sequence_X[1], sequence_Y[1])
     ...

Discard
==============

.. doxygenclass:: rocprim::discard_iterator
   :members:

Texture Cache
================

.. doxygenclass:: rocprim::texture_cache_iterator
   :members:
