.. meta::
   :description: Predicate selection walkthrough for the hipBLASLt library
   :keywords: hipBLASLt, ROCm, library, runtime, predicate, selection

.. _predicate-selection-walkthrough:

*********************************************
PCI Chip ID Predicates - A Walkthrough
*********************************************

Predicates are one of the essential integration points between the tensilelite build system, and the hipBLASLt library runtime.
Predicates are built into master solution libraries, files that contain information about which kernels are available, 
and their location. These files are loaded at runtime, and used in the kernel selection process.

Hardware predicates
-----------------------

Hardware predicates are the most coarse grained predicates. They are used to qualify, or reject, a kernel based on
attributes that can be determined at runtime through system inspection. These include the processor type (e.g. gfx1201),
the compute-unit counter (e.g., 128), and as of https://github.com/ROCm/rocm-libraries/pull/3924, the PCI chip ID (e.g., 0x7890).

**Building of Hardware Predicates**:

The source of truth for hardware predicates is in the hipblaslt library logic (LL) files (``find . -name Logic``). In particular, as of 
`tensilelite version 5.0.0 <https://github.com/ROCm/rocm-libraries/blob/273fcdc27e2f37d81420929d2105649494c9bb9d/projects/hipblaslt/tensilelite/Tensile/__init__.py>`_,
the standard 4-statement metadata at the top of each LL is authoritative (some LLs may have fields omitted for backwards compatibility):

.. code-block:: yaml

    - {MinimumRequiredVersion: 5.0.0}
    - gfx950
    - {Architecture: gfx950, CUCount: 64}
    - [Device 75a3, Device 75a2]

In the tensilelite build system, the PCI chip ID is add as an `Or`-style predicate, to allow multiple device IDs to be supported
for the same device.

At build, when the predicates are written to the master solution libraries, chip ID predicates are added as ``Or`` style conditions
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
        - {type: PciChipID, value: 30032}       # <-- placed directly alongside Process
        
        # if multiple chip IDs are specified
        - type: Or                              # <-- expanded into an `Or` conditioned list
          value:
          - {type: PciChipID, value: 30032}
          - {type: PciChipID, value: 30583}

That is, groups of solutions are co-resident in a LL file and marked by a set of `Device xxxx` strings. This allows solution filtering
(predicate matching) to be performed at the lazy library level. Consequentially, any candidate libraries that don't at least match the
processor and the current PCI chip ID (plus optional CU counts) will be skipped before the runtime code tries to load the library.

**Using Hardware Predicates**:

How does this play out during runtime? When the lazy solution library is loaded, it is deserialized and the predicates are matched
against the known program constraints to determine which *loadable* libraries are relevant (match), and which are not (don't match).
By setting TENSILE_DB, we can see how the chip IDs are being matched at runtime, for example, take the output from a test lib logic
that has the gfx1201 chip ID (7550) and another, random one:

.. code-block:: text

    --------------------------------------------------------------------------------
    PREDICATE: ExactLogic: Hardware
    --------------------------------------------------------------------------------
    [!!]  And                           (3 predicates)
    [OK]    Processor                   gpu=gfx950:sramecc+:xnack- == sol=gfx950
    [!!]    CUCount                     gpu=256 == sol=64
    [OK]    Or                          (2 predicates)
    [OK]      PciChipID                 [AMD Instinct MI355X] gpu=0x75a3 == sol=0x75a3
    [!!]      PciChipID                 [AMD Instinct MI355X] gpu=0x75a3 == sol=0x75a2
    --------------------------------------------------------------------------------
    Result: NO MATCH
    --------------------------------------------------------------------------------

