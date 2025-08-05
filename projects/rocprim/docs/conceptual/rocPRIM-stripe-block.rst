.. meta::
  :description: rocPRIM block versus stripe arrangement
  :keywords: rocPRIM, ROCm, API, documentation, block, stripe, stride

********************************************************************
Block and stripe configurations
********************************************************************

There are two arrangements that can be used to assign items to threads for processing.

In the block arrangement, each thread is assigned as many contiguous items as it can accommodate. When a thread can't accommodate any more items, items are assigned to the next thread. Threads in this arrangement contain sequential items.

In a striped arrangement, items are assigned to threads sequentially. The first item is assigned to the first thread, the second item is assigned to the second thread, and so on. If there are still items remaining once the last thread is reached, the assignment process will continue starting again at the first thread. Threads in this arrangement contain items separated by the block size. 

The stride length is generally the block size, but it can be changed by the user.
