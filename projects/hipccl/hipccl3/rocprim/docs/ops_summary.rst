Summary of the Operations
=========================

Basics
------

* ``transform`` applies a function to each element of the sequence, equivalent to the functional operation ``map``
* ``select`` takes the first N elements of the sequence satisfying a condition  (via a selection mask or a predicate function)
* ``unique``
* ``histogram`` generates a summary of the statistical distribution of the sequence.

Aggregation
-----------

* ``reduce`` traverses the sequence while accumulating some data, equivalent to the functional operation ``fold_left``.
* ``scan`` is the cumulative version of ``reduce`` which returns the sequence of the intermediate values taken by the accumulator.

Differentiation
---------------

* ``adjacent_difference`` computes the difference between the current element and the previous or next one in the sequence.
* ``discontinuity`` detects value change between the current element and the previous or next one in the sequence.

Rearrangement
-------------

* ``sort`` rearranges the sequence by sorting it. It could be according to a comparison operator or a value using a radix approach.
* ``exchange`` rearranges the elements according to a different stride configuration which is equivalent to a tensor axis transposition
* ``shuffle`` rotates the elements.

Partition/Merge
---------------

* ``partition`` divides the sequence into two or more sequences according to a predicate while preserving some ordering properties.
* ``merge`` merges two ordered sequences into one while preserving the order.

Data Movement
-------------

* ``store`` stores the sequence to a continuous memory zone. There are variations to use an optimized path or to specify how to store the sequence to better fit the access patterns of the CUs.
* ``load`` the complementary operations of the above ones.

Other operations
----------------

* ``run_length_encode`` generates a compact representation of a sequence
* ``binary_search`` finds for each element the index of an element with the same value in another sequence (which has to be sorted).
