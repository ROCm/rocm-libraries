.. meta::
  :description: rocPRIM block versus stripe arrangement
  :keywords: rocPRIM, ROCm, API, documentation, block, stripe, stride

********************************************************************
Block and stripe configurations
********************************************************************

There are two arrangements that can be used to assign items to threads for processing.

In the block arrangement, each thread is assigned as many contiguous items in the sequence as it can accommodate. When the thread can't accommodate any more items, items will be assigned to the next thread. 

In a stripe arrangement, each thread in the stride is assigned one item until the end of the stride is reached. When the end of the stride is reached, items are assigned to each thread starting at the first thread in the stride.

The stride length is generally the number of threads in the block, but it can be changed by the user.
