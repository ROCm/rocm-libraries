.. meta::
  :description: rocPRIM operations 
  :keywords: rocPRIM, ROCm, API, documentation, operations

********************************************************************
Types of rocPRIM operations
********************************************************************

A rocPRIM operation is a computation over a sequence of elements. 

The elements of the sequence can be of any type and belong to any class. Template specialization optimizes the computations over data types. 

The :doc:`scope <./rocPRIM-scope>` of an operation determines whether an operation is running at the thread, warp, block, or grid level. 

There are different types of operations. The following tables provide a summary of the different types of operations and what they do.



+------------------------+---------------------------+-----------------------------------------------+ 
| Type                   | Operation                 | Description                                   |
+========================+===========================+===============================================+
| Basic                  |Transform                  |Transform operations are device-level          |
|                        |                           |operations that apply a function to each       |
|                        |                           |element in the sequence. It's equivalent to    |
|                        |                           |the ``map`` functional operation.              |
|                        +---------------------------+-----------------------------------------------+
|                        |Select                     |Select operations are device-level operations  |
|                        |                           |that select the first ``N`` elements in the    |
|                        |                           |sequence that satisfy a condition set by a     |
|                        |                           |selection mask or a predicate function.        |
|                        +---------------------------+-----------------------------------------------+
|                        |Unique                     |Unique operations are device-level operations  | 
|                        |                           |that return only the first element from a      |
|                        |                           |group of similar consecutive elements.         |
|                        +---------------------------+-----------------------------------------------+
|                        |Histogram                  |Histogram operations are device-level          |
|                        |                           |operations that return a statistical           |
|                        |                           |distribution of the sequence.                  |
+------------------------+---------------------------+-----------------------------------------------+
|Aggregation             |Reduce                     |Reduce operations can run at the thread, warp, |
|                        |                           |block, grid, or device level. They're          |
|                        |                           |equivalent to the functional ``fold_left``.    |
|                        |                           |Reduce runs in a pairwise manner over the      |
|                        |                           |sequence in multiple iterations, reducing the  |
|                        |                           |sequence until only one result remains.        |
|                        +---------------------------+-----------------------------------------------+
|                        |Scan                       |Scan is a cumulative version of reduce that    |
|                        |                           |returns the sequence of intermediate values.   |
+------------------------+---------------------------+-----------------------------------------------+
|Difference              |Adjacent difference        |Adjacent difference operations are             |
|                        |                           |device-level and block-level operations that   |
|                        |                           |compute the difference between either the      |
|                        |                           |current element and the next one in the        |
|                        |                           |or between the current element and the previous|
|                        |                           |one in the sequence.                           |
|                        +---------------------------+-----------------------------------------------+
|                        |Discontinuity              |Discontinuity operations are block-level       |
|                        |                           |that find items in an ordered sequence that    |
|                        |                           |are discontinuous and don't follow the order   |
+------------------------+---------------------------+-----------------------------------------------+
|Reordering              |Sort                       |Sort operations are block-level and warp-level |
|                        |                           |operations that reorder a sequence based on    |
|                        |                           |item comparisons.                              |
|                        +---------------------------+-----------------------------------------------+
|                        |Partial sort               |Partial sort is a device-level operation that  |
|                        |                           |reorders a sequence by sorting it up to and    |
|                        |                           |including a given index.                       |
|                        +---------------------------+-----------------------------------------------+
|                        |Nth element                |Nth element is a device-level operation that   |
|                        |                           |inserts in its sorted position in the sequence |
|                        |                           |after items that are less than it and before   |
|                        |                           |items that are greater than it.                |
|                        +---------------------------+-----------------------------------------------+
|                        |Exchange                   |The exchange operations are block-level and    |
|                        |                           |warp-level operations that transpose stripe    |
|                        |                           |arrangements to block arrangements and vice    |
|                        |                           |versa.                                         |
|                        +---------------------------+-----------------------------------------------+
|                        |Shuffle                    |Shuffle operations are block-level and         |
|                        |                           |warp-level operations that move items between  |
|                        |                           |warps or threads in order to share memory.     |
+------------------------+---------------------------+-----------------------------------------------+
|Partitioning            |Partition                  |Partition operations separate the sequence     |
|                        |                           |based on a selection predicate. The order of   |
|                        |                           |the items in the sequence that return ``true`` |
|                        |                           |is retained.                                   |
|------------------------+---------------------------+-----------------------------------------------+
|Merging                 |Merge                      |Merges two ordered sequence into one while     |
|                        |                           |preserving the order.                          |
+------------------------+---------------------------+-----------------------------------------------+
|Data movement           |Store                      |Store provides methods for storing either      |
|                        |                           |block or stripe arrangement of items in        |
|                        |                           |contiguous memory.                             |
|                        +---------------------------+-----------------------------------------------+
|                        |Load                       |Load provides methods for loading data stored  |
|                        |                           |in contiguous memory into a block arrangement. |
|                        +---------------------------+-----------------------------------------------+
|                        |memcpy                     |Runs multiple                                  |
|                        |                           |device-to-device memory copy operations as a   |
|                        |                           |single batched operation.                      |
+------------------------+---------------------------+-----------------------------------------------+
|Search                  |Find first of              |Searches the sequence for the first occurrence |
|                        |                           |of any of the items in a range of items.       |
|                        +---------------------------+-----------------------------------------------+
|                        |Adjacent find              |Searches the sequence for the first occurrence |
|                        |                           |of two consecutive equal items   .             |
|                        +---------------------------+-----------------------------------------------+
|                        |Search                     |Searches the sequence for the first occurrence |
|                        |                           |of a series of items.                          |
|                        +---------------------------+-----------------------------------------------+
|                        |Search N                   |Searches the sequence for the first occurrence |
|                        |                           |of a series of N items that are all equal to   |
|                        |                           |the a given value.                             |
|                        +---------------------------+-----------------------------------------------+
|                        |Find end                   |Finds the last occurrence of a series of items |
|                        |                           |in the sequence.                               |
|                        +---------------------------+-----------------------------------------------+
|                        |Binary search              |Does a binary search on a sorted range of      |
|                        |                           |inputs.                                        |
+------------------------+---------------------------+-----------------------------------------------+
|Miscellaneous           |Run length encode          |Does a device-wide encoding of runs of         |
|                        |                           |consecutive equal values.                      |
|                        +---------------------------+-----------------------------------------------+
|                        |Configuring the kernel     |Sets the grid and block dimensions, and the    |
|                        |                           |algorithms to use for store and load.          |
+------------------------+---------------------------+-----------------------------------------------+