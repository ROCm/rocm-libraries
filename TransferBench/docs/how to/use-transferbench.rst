.. meta::
  :description: TransferBench is a utility to benchmark simultaneous transfers between user-specified devices (CPUs or GPUs)
  :keywords: TransferBench usage, TransferBench how to, TransferBench user guide, TransferBench user manual

.. _using-transferbench:

---------------------
Using TransferBench
---------------------

You can control the SRC and DST memory locations by indicating the memory type followed by the device index. TransferBench supports the following memory types:

* Coarse-grained pinned host
* Unpinned host
* Fine-grained host
* Coarse-grained global device
* Fine-grained global device
* Null (for an empty transfer)

In addition, you can determine the size of the transfer (number of bytes to copy) for the tests.

You can also specify transfer executors. The options are CPU, kernel-based GPU, and SDMA-based GPU (DMA) executors. TransferBench also provides the option to choose the number of Sub-Executors (SE). The number of SEs specifies the number of CPU threads in the case of a CPU executor and the number of compute units (CU) for a GPU executor.
For a DMA executor, the SE argument determines the number of streams to be used.

You can specify the transfers in a configuration file or use preset configurations for transfers.

Specifying transfers in a configuration file
----------------------------------------------

A transfer is defined as a single operation where an executor reads and adds together values from SRC memory locations, followed by writing the sum to the DST memory locations.
This simplifies to a copy operation when using a single SRC or DST.
Here's a copy operation from a single SRC to DST:

.. code-block:: bash

   SRC 0                DST 0
   SRC 1 -> Executor -> DST 1
   SRC X                DST Y

Three executors are supported by TransferBench:

.. code-block:: bash

  Executor:        SubExecutor:
  1. CPU           CPU thread
  2. GPU           GPU threadblock/Compute Unit (CU)
  3. DMA           N/A (Can only be used for a single SRC to DST copy)

Each line in the configuration file defines a set of transfers, also known as a test, to run in parallel.

There are two ways to specify a test:

- **Basic**

  The basic specification assumes the same number of SEs used per transfer.
  A positive number of transfers is specified, followed by the number of SEs and triplets describing each transfer:

  .. code-block:: bash

    Transfers SEs (srcMem1->Executor1->dstMem1) ... (srcMemL->ExecutorL->dstMemL)

  The arguments used to specify transfers in the config file are described in the :ref:`arguments table <config_file_arguments_table>`.

  **Example**:

  .. code-block:: bash

   1 4 (G0->G0->G1)                   Uses 4 CUs on GPU0 to copy from GPU0 to GPU1
   1 4 (G2->C1->G0)                   Uses 4 CUs on GPU2 to copy from CPU1 to GPU0
   2 4 G0->G0->G1 G1->G1->G0          Copies from GPU0 to GPU1, and GPU1 to GPU0, each with 4 SEs

- **Advanced**

  In the advanced specification, a negative number of transfers is specified, followed by quintuplets describing each transfer.
  Specifying a non-zero number of bytes overrides any provided value.

  .. code-block:: bash

    Transfers (srcMem1->Executor1->dstMem1 SEs1 Bytes1) ... (srcMemL->ExecutorL->dstMemL SEsL BytesL)

  The arguments used to specify transfers in the config file are described in the :ref:`arguments table <config_file_arguments_table>`.

  **Example**:

  .. code-block:: bash

   -2 (G0 G0 G1 4 1M) (G1 G1 G0 2 2M) Copies 1Mb from GPU0 to GPU1 with 4 SEs and 2Mb from GPU1 to GPU0 with 2 SEs

Here is the list of arguments used to specify transfers in the config file:

.. _config_file_arguments_table:

.. list-table::
   :header-rows: 1

   * - Argument
     - Description

   * - Transfers
     - Number of transfers to be run in parallel

   * - SE
     - Number of SEs to use (CPU threads or GPU threadblocks)

   * - srcMemL
     - Source memory locations (where the data is read)

   * - Executor
     - | Executor is specified by a character indicating type, followed by the device index (0-indexed):
       | - C: CPU-executed  (indexed from 0 to NUMA nodes - 1)
       | - G: GPU-executed  (indexed from 0 to GPUs - 1)
       | - D: DMA-executor  (indexed from 0 to GPUs - 1)

   * - dstMemL
     - Destination memory locations (where the data is written)

   * - bytesL
     - | Number of bytes to copy (use command-line specified size when 0).
       | Must be a multiple of four and can be suffixed with ('K','M', or 'G').
       | Memory locations are specified by one or more device characters or device index pairs.
       | Characters indicate memory type and are followed by device index (0-indexed).
       | Here are the characters and their respective memory locations:
       | - C:    Pinned host memory       (on NUMA node, indexed from 0 to [NUMA nodes-1])
       | - U:    Unpinned host memory     (on NUMA node, indexed from 0 to [NUMA nodes-1])
       | - B:    Fine-grain host memory   (on NUMA node, indexed from 0 to [NUMA nodes-1])
       | - G:    Global device memory     (on GPU device, indexed from 0 to [GPUs - 1])
       | - F:    Fine-grain device memory (on GPU device, indexed from 0 to [GPUs - 1])
       | - N:    Null memory              (index ignored)

Round brackets and arrows "->" can be included for human clarity, but will be ignored.
Lines starting with # are ignored while lines starting with ## are echoed to the output.

**Transfer examples:**

Single GPU-executed transfer between GPU 0 and 1 using 4 CUs::

   1 4 (G0->G0->G1)

Single DMA-executed transfer between GPU 0 and 1::

   1 1 (G0->D0->G1)

Copying 1Mb from GPU 0 to GPU 1 with 4 CUs, and 2Mb from GPU 1 to GPU 0 with 8 CUs::

   -2 (G0->G0->G1 4 1M) (G1->G1->G0 8 2M)

"Memset" by GPU 0 to GPU 0 memory::

   1 32 (N0->G0->G0)

"Read-only" by CPU 0::

   1 4 (C0->C0->N0)

Broadcast from GPU 0 to GPU 0 and GPU 1::

   1 16 (G0->G0->G0G1)

.. note::

   Running TransferBench with no arguments displays usage instructions and detected topology information.

Using preset configurations
------------------------------

Here is the list of preset configurations that can be used instead of configuration files:

.. list-table::
   :header-rows: 1

   * - Configuration
     - Description

   * - ``a2a``
     - All-to-all benchmark test

   * - ``cmdline``
     - Allows transfers to run from the command line instead of a configuration file

   * - ``healthcheck``
     - Simple health check (supported on AMD Instinct MI300 series only)

   * - ``p2p``
     - Peer-to-peer benchmark test

   * - ``pcopy``
     - Benchmark parallel copies from a single GPU to other GPUs

   * - ``rsweep``
     - Random sweep across possible sets of transfers

   * - ``rwrite``
     - Benchmark parallel remote writes from a single GPU to other GPUs

   * - ``scaling``
     - GPU subexecutor scaling tests

   * - ``schmoo``
     - Read, write, or copy operation on local or remote between two GPUs

   * - ``sweep``
     - Sweep across possible sets of transfers

Performance tuning
---------------------

When you use the same GPU executor in multiple simultaneous transfers on separate streams by setting ``USE_SINGLE_STREAM=0``, the performance might be serialized due to the maximum number of hardware queues available.
To improve the performance, adjust the number of maximum hardware queues using ``GPU_MAX_HW_QUEUES``.
