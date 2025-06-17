.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Tensile environment variables, GEMM, Tensor, tensor

.. _environment-variables:

********************************************************************
Environment variables
********************************************************************

This topic lists the environment variables that enable testing, debugging, and experimental features for Tensile clients and applications.

.. list-table::
    :header-rows: 1
    :widths: 50,50

    * - **Environment variable**
      - **Value**

    * - | ``TENSILE_DB``
        | Enables debugging features based on the supplied value. Bit field options can be set individually or combined.
      - | Hexadecimal bit field values:
        | ``0x2`` or ``0x4``: Solution selection process information
        | ``0x8``: Hardware selection process information
        | ``0x10``: Predicate evaluation debug information
        | ``0x20``: Code object library loading status
        | ``0x40``: Kernel launch arguments and parameters
        | ``0x80``: Allocated tensor sizes
        | ``0x100``: Convolution reference calculation debug info
        | ``0x200``: Detailed convolution reference calculations
        | ``0x1000``: Library loading information (YAML/MessagePack)
        | ``0x4000``: Solution lookup efficiency
        | ``0x8000``: Selected kernel names
        | ``0x80000``: Detailed kernel parameters (Matrix Instruction, MacroTile, etc.)
        | ``0xFFFF``: Enable all debug output

    * - | ``TENSILE_DB2``
        | Enables extended debugging features. Skips kernel launches but continues kernel selection and data allocation.
      - | ``1``: Enable extended debugging
        | ``2``: Disable extended debugging

    * - | ``TENSILE_NAIVE_SEARCH``
        | Performs naive search for matching kernels instead of optimized search.
      - | ``1``: Enable naive search
        | ``2``: Disable naive search

    * - | ``TENSILE_TAM_SELECTION_ENABLE``
        | Enables tile aware solution selection.
      - | ``1``: Enable tile aware selection
        | ``2``: Disable tile aware selection

    * - | ``TENSILE_SOLUTION_INDEX``
        | Prints the index of the selected solution.
      - | ``1``: Enable solution index printing
        | ``2``: Disable solution index printing

    * - | ``TENSILE_METRIC``
        | Overrides the default distance matrix for solution selection.
      - | "Euclidean": Euclidean distance metric
        | "JSD": Jensen-Shannon divergence metric
        | "Manhattan": Manhattan distance metric
        | "Ratio": Ratio-based metric
        | "Random": Random selection metric

    * - | ``TENSILE_PROFILE``
        | Enables profiling of functions decorated with @profile decorator. Results generated as .prof files.
      - | ``1``, "ON", "TRUE": Enable profiling
        | Any other value: Disable profiling
