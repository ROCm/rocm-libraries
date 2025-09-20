.. meta::
  :description: rocPRIM operations 
  :keywords: rocPRIM, ROCm, API, documentation, operations

********************************************************************
Types of rocPRIM operations
********************************************************************

A rocPRIM operation is a computation over a sequence of elements. 

The elements of the sequence can be of any type and belong to any class. Template specialization optimizes the computations over data types. 

The :doc:`scope <./rocPRIM-scope>` of an operation determines whether an operation is running at the thread, warp, block, or grid level. 

The following tables provide a summary of the different types of operations and what they do.



+------------------------+---------------------------+-----------------------------------------------+ 
| Type                   | Operation                 | Description                                   |
+========================+===========================+===============================================+
| Basic                  |Transform                  |Transform operations are device-level          |
|                        |                           |operations that apply a function to each       |
|                        |                           |element in the sequence. They're equivalent to |
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
|                        |                           |sequence, or between the current element and   |
|                        |                           |the previous one in the sequence.              |
|                        +---------------------------+-----------------------------------------------+
|                        |Discontinuity              |Discontinuity operations are block-level       |
|                        |                           |operations that find items in a                |
|                        |                           |sequence that don't follow the sequence order. |
+------------------------+---------------------------+-----------------------------------------------+
|Reordering              |Sort                       |Sort operations are device-level, block-level, |
|                        |                           |and warp-level operations that reorder a       |
|                        |                           |sequence based on item comparisons.            |
|                        +---------------------------+-----------------------------------------------+
|                        |Partial sort               |Partial sort is a device-level operation that  |
|                        |                           |reorders a sequence by sorting it up to and    |
|                        |                           |including a given index.                       |
|                        +---------------------------+-----------------------------------------------+
|                        |Nth element                |Nth element is a device-level operation that   |
|                        |                           |inserts a value in its sorted position in the  |
|                        |                           |sequence.                                      |
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
|Partitioning            |Partition                  |Partition operations are device-level          |
|                        |                           |operations that separate the sequence          |
|                        |                           |based on a selection predicate. The order of   |
|                        |                           |the items in the sequence that return ``true`` |
|                        |                           |is retained.                                   |
+------------------------+---------------------------+-----------------------------------------------+
|Merging                 |Merge                      |Merge operations are device-level operations   |
|                        |                           |that merge two ordered sequence into one while |
|                        |                           |preserving the order.                          |
+------------------------+---------------------------+-----------------------------------------------+
|Data movement           |Store                      |Store operations are block, warp, and thread   |
|                        |                           |level operations that store items in block or  |
|                        |                           |stripe arrangements in contiguous memory.      |
|                        +---------------------------+-----------------------------------------------+
|                        |Load                       |Load operations are block, warp, and thread    |
|                        |                           |level operations that load data stored         |
|                        |                           |in contiguous memory into a block arrangement. |
|                        +---------------------------+-----------------------------------------------+
|                        |Memcpy                     |Memcopy operations are device-level operations |
|                        |                           |that run multiple device-to-device memory copy |
|                        |                           |operations as single batch operations.         |
+------------------------+---------------------------+-----------------------------------------------+
|Search                  |Find first of              |Find first of is a device-level operation that |
|                        |                           |searches the sequence for the first occurrence |
|                        |                           |of any of the items in a given range of items. |
|                        +---------------------------+-----------------------------------------------+
|                        |Find end                   |Find end is a device-level operation that      |
|                        |                           |finds the last occurrence of a series of items |
|                        |                           |in the sequence.                               |
|                        +---------------------------+-----------------------------------------------+
|                        |Adjacent find              |Adjacent find is a device-level operation that |
|                        |                           |searches for the first occurrence of two       |
|                        |                           |consecutive equal items.                       |
|                        +---------------------------+-----------------------------------------------+
|                        |Search                     |Search is a device-level operation that        |
|                        |                           |searches the sequence for the first occurrence |
|                        |                           |of a series of items.                          |
|                        +---------------------------+-----------------------------------------------+
|                        |Search N                   |Search N is a device-level operation that      |
|                        |                           |searches the sequence for the first occurrence |
|                        |                           |of a series of N items that are all equal to   |
|                        |                           |a given value.                                 |
|                        +---------------------------+-----------------------------------------------+
|                        |Binary search              |Binary search is a device-level operation that |
|                        |                           |does a binary search on a sorted range of      |
|                        |                           |inputs.                                        |
+------------------------+---------------------------+-----------------------------------------------+
|Miscellaneous           |Run length encode          |Run length encode operations are device-level  |
|                        |                           |operations that do device-wide encodings of    |
|                        |                           |runs of consecutive equal values.              |
|                        +---------------------------+-----------------------------------------------+
|                        |Run length decode          |Run length decode operations are block-level   |
|                        |                           |operations that decode run length encoded      |
|                        |                           |arrays of items.                               | 
|                        +---------------------------+-----------------------------------------------+
|                        |Kernel config              |Kernel config constructs are device-level      |
|                        |                           |constructs that set the grid and block         |
|                        |                           |dimensions as well as the algorithms to use for|
|                        |                           |the store and load operations.                 |
+------------------------+---------------------------+-----------------------------------------------+