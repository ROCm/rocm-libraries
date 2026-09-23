.. meta::
   :description: Predicate selection walkthrough for the hipBLASLt library
   :keywords: hipBLASLt, ROCm, library, runtime, predicate, selection

.. _predicate-selection-walkthrough:

*********************************************
PCI chip ID predicates walkthrough
*********************************************

Predicates are one of the essential integration points between the TensileLite build system and the hipBLASLt library runtime.
Predicates are built into the master solution libraries. They are files that contain information about which kernels are available 
and their location. These files are loaded at runtime and used in the kernel selection process.

.. note::

   This guide covers advanced technical concepts and is intended for highly experienced users and developers only.

Chip ID registry
-----------------------

The chip ID registry is a namespace in
`Tensile/AMDGPUPredicates.hpp <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblaslt/tensilelite/include/Tensile/AMDGPUPredicates.hpp>`_ that
contains the official mapping of supported PCI chip IDs and their fallback relationships.

.. code-block:: yaml

    - {PCIChipId: 0x75a3, DeviceName: AMD Instinct MI355X}
    - {PCIChipId: 0x75a2, DeviceName: AMD Instinct MI355X}

Hardware predicates
-----------------------

Hardware predicates are the most coarse-grained predicates. They are used to qualify, or reject, a kernel based on
attributes that can be determined at runtime through system inspection. These include the processor type (for example, gfx1201),
the compute-unit counter (for example, 128), and, as of ROCm 7.13, the PCI chip ID (for example, 0x7890).

Building hardware predicates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source of truth for hardware predicates is in the hipBLASLt library logic (LL) files (``find . -name Logic``). In particular, as of 
`TensileLite version 5.0.0 <https://github.com/ROCm/rocm-libraries/blob/273fcdc27e2f37d81420929d2105649494c9bb9d/projects/hipblaslt/tensilelite/Tensile/__init__.py>`_,
the standard four-statement metadata at the top of each LL is authoritative (some LLs could have fields omitted for backwards compatibility):

.. code-block:: yaml

    - {MinimumRequiredVersion: 5.0.0}
    - gfx950
    - {Architecture: gfx950, CUCount: 64}
    - [Device 75a3, Device 75a2]

In the TensileLite build system, the PCI chip ID is added as an ``Or``-style predicate, to support multiple device IDs
for the same device.

At build time, when the predicates are written to the master solution libraries, chip ID predicates are added as ``Or``-style conditions
to the top-level lazy lookup (``TensileLibrary_lazy_....dat/yaml``). For example, a snippet from ``TensileLibrary_lazy_gfx1201.yaml``
could appear as:

.. code-block:: yaml

    predicate:
      type: AMDGPU                              # <-- AMDGPU hardware predicate
      value:
        type: And
        value:
        - {type: Processor, value: gfx1201}
        - {type: CUCount, value: 64}
        # if only one chip ID is specified
        - {type: PciChipId, value: 30032}       # <-- placed directly alongside Process
        
        # if multiple chip IDs are specified
        - type: Or                              # <-- expanded into an `Or` conditioned list
          value:
          - {type: PciChipId, value: 30032}
          - {type: PciChipId, value: 30583}

That is, groups of solutions are co-resident in a LL file and marked by a set of ``Device xxxx`` strings. This allows solution filtering
(predicate matching) to be performed at the lazy library level. Consequently, any candidate libraries that don't at least match the
processor and the current PCI chip ID (plus optional CU counts) will be skipped before the runtime code tries to load the library.

Using hardware predicates
^^^^^^^^^^^^^^^^^^^^^^^^^

How does this play out during runtime? When the lazy solution library is loaded, it is deserialized and the predicates are matched
against the known program constraints to determine which **loadable** libraries are relevant (match), and which are not (don't match).
Set ``TENSILE_DB`` to see how the chip IDs are matched at runtime. For example, take the output from a test library logic
that has the gfx1201 chip ID (7550) and another, random one:

.. code-block:: text

    --------------------------------------------------------------------------------
    PREDICATE: ExactLogic: Hardware
    --------------------------------------------------------------------------------
    [!!]  And                           (3 predicates)
    [OK]    Processor                   gpu=gfx950:sramecc+:xnack- == sol=gfx950
    [!!]    CUCount                     gpu=256 == sol=64
    [OK]    Or                          (2 predicates)
    [OK]      PciChipId                 [AMD Instinct MI355X] gpu=0x75a3 == sol=0x75a3
    [!!]      PciChipId                 [AMD Instinct MI355X] gpu=0x75a3 == sol=0x75a2
    --------------------------------------------------------------------------------
    Result: NO MATCH
    --------------------------------------------------------------------------------

Here you can see how different chip IDs are being selected against the known device properties.


Build-time row ordering (fallback-aware)
----------------------------------------

When a logic file declares multiple device IDs, ``tensilelite`` treats them as a **set**
for ordering purposes (not "first ID wins"). This matters because runtime selection is
``first-match-wins``, and ``PciChipId`` supports one-way fallback.

Fallback graph (example for ``gfx950`` variants):

.. code-block:: text

    75a3  ---> 75a0
    75a2  ---> 75a0

Interpretation:

- Arrow direction means "may fallback to".
- Source-like IDs (left) are more specific than fallback targets (right).
- Rows must be ordered so source-like exact rows are considered first.

The build-time comparator for hardware rows applies chip-ID precedence as follows:

#. **Chip-set present vs absent**: rows with chip IDs sort before rows without chip IDs.
#. **Exactness first**: smaller chip-ID sets sort first (for example,
   ``{75a3}`` before ``{75a3,75a0}``).
#. **Fallback-aware rank**: for equal-size sets, compare IDs by fallback-topology rank
   (farther from fallback roots first), then by chip ID value for deterministic tie-breaks.
#. **The CU count**: if chip precedence does not decide the order, the higher ``CUCount`` sorts first.

Here is a minimal ordering example (for the same processor):

.. code-block:: text

    1) {75a3}, CU=256
    2) {75a3}, CU=64
    3) {75a0}, CU=256
    4) {75a0}, CU=64
    5) {75a3,75a0}, CU=*
    6) {no chip id}, CU=*

Why this matters:

- Device ``75a3`` can still use ``75a0`` fallback kernels at runtime.
- However, exact ``75a3`` rows are evaluated first, preventing broad/mixed rows from
  accidentally shadowing more specific rows.


Runtime selection
-----------------

``ExactLogicLibrary::findBestSolution`` iterates hardware predicate rows in priority order. 
For each row that matches the GPU, it calls ``HardwarePredicate::isFallbackMatch(hardware)``
to classify the match:

- Exact match (``isFallbackMatch`` returns false): The row's ``PciChipIdEqual`` target
  matches the GPU's chip ID exactly (for example, an Instinct™ MI350 GPU matches a ``PciChipIdEqual(0x75a0)`` row),
  or the row has no chip ID predicate at all. The solution is returned immediately.
- Fallback match (``isFallbackMatch`` returns true): The GPU matched via chip ID fallback 
  (for example, an Instinct MI355 GPU ``0x75a3`` matches an Instinct MI350 GPU ``0x75a0`` row through ``ChipIdRegistry``). The solution is
  saved but the loop continues to search for an exact match.

After all rows are checked, if no exact match was found, the saved fallback solution is returned.
This ensures that device-specific solutions (Instinct MI355-targeted kernels) are always preferred over 
fallback solutions (Instinct MI350 kernels running on an Instinct MI355 GPU), regardless of row ordering.
An exact match anywhere in the list beats a fallback match that appeared earlier.

.. important::

    This means that the first fallback match is the one that is returned, and all subsequent ones are ignored.