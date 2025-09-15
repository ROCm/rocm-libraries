.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Tensile concepts, solution parameter, Tensor contractions, tensor contractions

.. _kernel-parameters:

******************
Kernel parameters
******************

Kernel parameters are configuration parameters used by Tensile to make decisions about what assembly code to generate. The kernel parameters affect many aspects of performance. Changing a parameter might help address one performance bottleneck but worsen another. Hence, searching through the parameter space is vital to discovering the fastest kernel for a given problem.

The following table lists the kernel parameters:

.. list-table:: Kernel parameters
  :header-rows: 1

  * - Kernel parameter
    - Description

  * - ``LoopDoWhile``
    - Setting to True = Do-While loop and Setting to False = While or For loop.

  * - ``LoopTail``
    - Additional loop with ``LoopUnroll``=1.

  * - ``EdgeType``
    - Branch, ``ShiftPtr`` or None.

  * - ``WorkGroup``
    - [dim0, dim1, ``LocalSplitU``]

  * - ``ThreadTile``
    - [dim0, dim1]

  * - ``MatrixInstruction``
    - Type of matrix instruction used for the calculation and wave tiling parameters [``InstructionM``, ``InstructionN``, ``InstructionK``, ``InstructionB``, ``BlocksInMDir``, ``WaveTileM``, ``WaveTileN``, ``WaveGroupM``, ``WaveGroupN``].

  * - ``GlobalSplitU``
    - Split up summation among work-groups to create more concurrency. This option launches a kernel to handle the beta scaling and then a second kernel with atomic writes to the global memory.

  * - ``PrefetchGlobalRead``
    - Setting to True ensures that the outer loop prefetches global data one iteration ahead.

  * - ``PrefetchLocalRead``
    - Setting to True ensures that the inner loop prefetches local data one iteration ahead.

  * - ``WorkGroupMapping``
    - The order in which the work-groups compute C. This affects cacheing.

  * - ``LoopUnroll``
    - The number of iterations to unroll the inner loop. This helps in loading coalesced memory.

  * - ``MacroTile``
    - Derived using ``WorkGroup``\*``ThreadTile``.

  * - ``DepthU``
    - Derived using ``LoopUnroll``\*``SplitU``.

  * - ``NumLoadsCoalescedA,B``
    - The number of loads from A in coalesced dimension.

  * - ``GlobalReadCoalesceGroupA,B``
    - Setting to True ensures that the adjacent threads map to adjacent global read elements. However, if transposing data then write to LDS is scattered.

  * - ``GlobalReadCoalesceVectorA,B``
    - Setting to True ensures that the vector components map to adjacent global read elements. However, if transposing data then write to LDS is scattered.

  * - ``VectorWidth``
    - As the thread tile elements are contiguous for faster memory accesses, a ``VectorWidth``= 4 implies that a thread will read a float4 from memory instead of 4 non-contiguous floats.

  * - ``KernelLanguage``
    - Decides if the kernels should be written in the source code (HIP) or assembly (gfx803, gfx900, ...).

For the exhaustive list of solution parameters and their defaults, see `Common.py <https://github.com/ROCm/Tensile/blob/develop/Tensile/Common.py>`_.

GPU kernel dimensions
======================

Tensile allows for 3-dimensional grid of work-groups. Each work-group can be a 3-dimensional grid of work-items. Tensile assigns D0 to the dimension-0 and D1 to the dimension-1 of the work-group and work-item grid. All other free or batch dimensions are flattened into the final dimension-2 of the work-group and work-item grids. Within the GPU kernel, dimension-2 is reconstituted back into whatever dimensions it represents.

Mapping between N-dimensional tensor contractions and finite-dimensional GPU kernels
--------------------------------------------------------------------------------------

For a traditional GEMM, the 2-dimensional output, C[i,j], is mapped to launching a 2-dimensional grid of work-groups. Each work-group has a 2-dimensional grid of work-items with one dimension belonging to i and another to j. The 1-dimensional summation is represented by a single loop within the kernel body.

Special dimensions: D0, D1, and DU
-----------------------------------

To handle arbitrary dimensionality, Tensile begins by determining three special dimensions: D0, D1, and DU.

D0 and D1 are the free indices of A and B respectively having the shortest strides. This allows the fastest reads for innermost loops from A and B via coalescing. In a traditional GEMM, every matrix has a dimension with a shortest stride of one, but Tensile doesn't rely on this assumption. Of these two dimensions, D0 is the dimension with the shortest tensor C stride, which allows for fast writing.

DU represents the summation index with the shortest combined stride (stride in A + stride in B). DU is the innermost loop that gets "U"nrolled. This assignment is also meant to assure fast reading in the innermost summation loop. In case of multiple summation indices (embedded loops), DU iterates over the innermost loop.

Kernel names
=============

Kernel names contain abbreviations of relevant parameters along with their value. Here is what a typical kernel name looks like:

.. code-block:: shell

    Cijk_Ailk_Bjlk_SB_MT64x256x16_<PARAMETERS>

The given kernel name example is a GEMM. The different parts of the kernel name are described here:

- The first part (C***_A***_B***) indicates the type of operation the kernel performs.

- The second part indicates the data type supported by the kernel. In the preceding example, "S" indicates single-precision floating-point numbers and "B" indicates that the kernel can use beta values.

  For a list of supported data types and their corresponding code names, please refer to :ref:`precision-support`.

- The third part "MT" stands for macro tile, which is 64x256 here. The third number listed with macro tile (16 in the example) is the unroll depth, specified by the ``DepthU`` parameter.

- The last part "<PARAMETERS>" is an alphabetized list of abbreviations of relevant kernel parameters. The table below lists parameters, their kernel name abbreviations, and their default values to help interpret the meaning of a kernel name:

  .. list-table:: kernel name parameters
    :header-rows: 1
    :widths: 30 30 30

    * - Code
      - Parameter
      - Default

    * - ``1LDSB``
      - ``1LDSBuffer``
      - 0

    * - ``APM``
      - ``AggressivePerfMode``
      - 1

    * - ``AAV``
      - ``AssertAlphaValue``
      - False

    * - ``ABV``
      - ``AssertBetaValue``
      - False

    * - ``ACED``
      - ``AssertCEqualsD``
      - False

    * - ``AF0EM``
      - ``AssertFree0ElementMultiple``
      - 1

    * - ``AF1EM``
      - ``AssertFree1ElementMultiple``
      - 1

    * - ``AMAS``
      - ``AssertMinApproxSize``
      - -1

    * - ``ASE``
      - ``AssertSizeEqual``
      - {}

    * - ``ASGT``
      - ``AssertSizeGreaterThan``
      - {}

    * - ``ASLT``
      - ``AssertSizeLessThan``
      - {}

    * - ``ASM``
      - ``AssertSizeMultiple``
      - {}

    * - ``ASAE``
      - ``AssertStrideAEqual``
      - {}

    * - ``ASBE``
      - ``AssertStrideBEqual``
      - {}

    * - ``ASCE``
      - ``AssertStrideCEqual``
      - {}

    * - ``ASDE``
      - ``AssertStrideDEqual``
      - {}

    * - ``ASEM``
      - ``AssertSummationElementMultiple``
      - 1

    * - ``AAC``
      - ``AtomicAddC``
      - False

    * - ``BL``
      - ``BufferLoad``
      - True

    * - ``BS``
      - ``BufferStore``
      - True

    * - ``CDO``
      - ``CheckDimOverflow``
      - 0

    * - ``CTDA``
      - ``CheckTensorDimAsserts``
      - False

    * -
      - ``CustomKernelName``
      - ""

    * - ``DU``
      - ``DepthU``
      - -1

    * - ``DULD``
      - ``DepthULdsDivisor``
      - 1

    * - ``DTL``
      - ``DirectToLds``
      - False

    * - ``DTVA``
      - ``DirectToVgprA``
      - False

    * - ``DTVB``
      - ``DirectToVgprB``
      - False

    * - ``DAF``
      - ``DisableAtomicFail``
      - 0

    * - ``DKP``
      - ``DisableKernelPieces``
      - 0

    * - ``DVO``
      - ``DisableVgprOverlapping``
      - False

    * - ``ET``
      - ``EdgeType``
      - Branch

    * - ``EPS``
      - ``ExpandPointerSwap``
      - True

    * - ``R``
      - ``Fp16AltImpl``
      - False

    * - ``FL``
      - ``FractionalLoad``
      - 0

    * - ``GR2A``
      - ``GlobalRead2A``
      - True

    * - ``GR2B``
      - ``GlobalRead2B``
      - True

    * - ``GRCGA``
      - ``GlobalReadCoalesceGroupA``
      - True

    * - ``GRCGB``
      - ``GlobalReadCoalesceGroupB``
      - True

    * - ``GRCVA``
      - ``GlobalReadCoalesceVectorA``
      - True

    * - ``GRCVB``
      - ``GlobalReadCoalesceVectorB``
      - True

    * - ``GRPM``
      - ``GlobalReadPerMfma``
      - 1

    * - ``GRVW``
      - ``GlobalReadVectorWidth``
      - -1

    * - ``GSU``
      - ``GlobalSplitU``
      - 1

    * - ``GSUA``
      - ``GlobalSplitUAlgorithm``
      - ``SingleBuffer``

    * - ``GSUSARR``
      - ``GlobalSplitUSummationAssignmentRoundRobin``
      - True

    * - ``GSUWGMRR``
      - ``GlobalSplitUWorkGroupMappingRoundRobin``
      - False

    * - ``GLS``
      - ``GroupLoadStore``
      - False

    * - ``ISA``
      - ``ISA``
      -

    * - ``IU``
      - ``InnerUnroll``
      - 1

    * - ``IA``
      - ``InterleaveAlpha``
      - 0

    * - ``KL``
      - ``KernelLanguage``
      - Source

    * - ``LEL``
      - ``LdcEqualsLdd``
      - True

    * - ``LBSPP``
      - ``LdsBlockSizePerPad``
      - -1

    * - ``LPA``
      - ``LdsPadA``
      - 0

    * - ``LPB``
      - ``LdsPadB``
      - 0

    * - ``LDL``
      - ``LocalDotLayout``
      - 1

    * - ``LRVW``
      - ``LocalReadVectorWidth``
      - -1

    * - ``LWPM``
      - ``LocalWritePerMfma``
      - -1

    * - ``LR2A``
      - ``LocalRead2A``
      - True

    * - ``LR2B``
      - ``LocalRead2B``
      - True

    * - ``LW2A``
      - ``LocalWrite2A``
      - True

    * - ``LW2B``
      - ``LocalWrite2B``
      - True

    * - ``LDW``
      - ``LoopDoWhile``
      - False

    * - ``LT``
      - ``LoopTail``
      - True

    * - ``MAD`` or ``FMA``
      - ``MACInstruction``
      - ``FMA``

    * - ``MT``
      - ``MacroTile``
      -

    * - ``MTSM``
      - ``MacroTileShapeMax``
      - 64

    * - ``MTSM``
      - ``MacroTileShapeMin``
      - 1

    * - ``MDA``
      - ``MagicDivAlg``
      - 2

    * - ``MI``
      - ``MatrixInstruction``
      - []

    * - ``MO``
      - ``MaxOccupancy``
      - 40

    * - ``MVN``
      - ``MaxVgprNumber``
      - 256

    * - ``MIAV``
      - ``MIArchVgpr``
      - False

    * - ``MVN``
      - ``MinVgprNumber``
      - 0

    * - ``NTA``
      - ``NonTemporalA``
      - 0

    * - ``NTB``
      - ``NonTemporalB``
      - 0

    * - ``NTC``
      - ``NonTemporalC``
      - 0

    * - ``NTD``
      - ``NonTemporalD``
      - 0

    * - ``NR``
      - ``NoReject``
      - False

    * - ``NEPBS``
      - ``NumElementsPerBatchStore``
      - 0

    * - ``NLCA``
      - ``NumLoadsCoalescedA``
      - 1

    * - ``NLCB``
      - ``NumLoadsCoalescedB``
      - 1

    * - ``ONLL``
      - ``OptNoLoadLoop``
      - 1

    * - ``OPLV``
      - ``OptPreLoopVmcnt``
      - True

    * - ``PBD``
      - ``PackBatchDims``
      - 0

    * - ``PFD``
      - ``PackFreeDims``
      - 1

    * - ``PG``
      - ``PackGranularity``
      - 2

    * - ``PSD``
      - ``PackSummationDims``
      - 0

    * - ``PSL``
      - ``PerformanceSyncLocation``
      - -1

    * - ``PWC``
      - ``PerformanceWaitCount``
      - -1

    * - ``PWL``
      - ``PerformanceWaitLocation``
      - -1

    * - ``PK``
      - ``PersistentKernel``
      - 0

    * - ``PKAB``
      - ``PersistentKernelAlongBatch``
      - False

    * - ``PAP``
      - ``PrefetchAcrossPersistent``
      - 0

    * - ``PAPM``
      - ``PrefetchAcrossPersistentMode``
      - 0

    * - ``PGR``
      - ``PrefetchGlobalRead``
      - True

    * - ``PLR``
      - ``PrefetchLocalRead``
      - 1

    * - ``RK``
      - ``ReplacementKernel``
      - False

    * - ``SGR``
      - ``ScheduleGlobalRead``
      - 1

    * - ``SIA``
      - ``ScheduleIterAlg``
      - 1

    * - ``SLW``
      - ``ScheduleLocalWrite``
      - 1

    * - ``SS``
      - ``SourceSwap``
      - False

    * - ``SU``
      - ``StaggerU``
      - 32

    * - ``SUM``
      - ``StaggerUMapping``
      - 0

    * - ``SUS``
      - ``StaggerUStride``
      - 256

    * - ``SCIU``
      - ``StoreCInUnroll``
      - False

    * - ``SCIUE``
      - ``StoreCInUnrollExact``
      - False

    * - ``SCIUI``
      - ``StoreCInUnrollInterval``
      - 1

    * - ``SCIUP``
      - ``StoreCInUnrollPostLoop``
      - False

    * - ``SPO``
      - ``StorePriorityOpt``
      - False

    * - ``SRVW``
      - ``StoreRemapVectorWidth``
      - 0

    * - ``SSO``
      - ``StoreSyncOpt``
      - 0

    * - ``SVW``
      - ``StoreVectorWidth``
      - -1

    * - ``SNLL``
      - ``SuppressNoLoadLoop``
      - False

    * - ``TSGRA``
      - ``ThreadSeparateGlobalReadA``
      - 0

    * - ``TSGRB``
      - ``ThreadSeparateGlobalReadB``
      - 0

    * - ``TT``
      - ``ThreadTile``
      - [4, 4]

    * - ``TLDS``
      - ``TransposeLDS``
      - 0

    * - ``UIIDU``
      - ``UnrollIncIsDepthU``
      - 0

    * - ``UMF``
      - ``UnrollMemFence``
      - False

    * - ``U64SL``
      - ``Use64bShadowLimit``
      - 1

    * - ``UIOFGRO``
      - ``UseInstOffsetForGRO``
      - 0

    * - ``USFGRO``
      - ``UseSgprForGRO``
      - -1

    * - ``VAW``
      - ``VectorAtomicWidth``
      - -1

    * - ``VS``
      - ``VectorStore``
      - True

    * - ``VW``
      - ``VectorWidth``
      - -1

    * - ``WSGRA``
      - ``WaveSeparateGlobalReadA``
      - 0

    * - ``WSGRB``
      - ``WaveSeparateGlobalReadB``
      - 0

    * - ``WS``
      - ``WavefrontSize``
      - 64

    * - ``WG``
      - ``WorkGroup``
      - [16, 16, 1]

    * - ``WGM``
      - ``WorkGroupMapping``
      - 8

    * - ``WGMT``
      - ``WorkGroupMappingType``
      - B
