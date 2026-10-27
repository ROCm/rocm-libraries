# ADR 0036: Name register allocation by its mode

Status:  Accepted
Defect:  none — behavior is intended
Related: [ADR 0034](0034-distinguish-code-changing-kernel-modes.md)

## Context

`StinkyTofuRegisterAllocation` is a mode from 0 to 3 or a StinkyTofu
configuration string such as `"2,classes=vs"`. The kernel name records the
mode: no tag for 0, and `STRA1` to `STRA3` otherwise. The other settings, such
as `classes`, `allocator` and `pinReg`, also change the generated assembly.
ADR 0034 keeps code-changing controls in the kernel name so that different
kernels cannot share a compiled artifact. Applied here, it would put every
setting in the name.

Tensile compiles and deduplicates by kernel name, so two configurations with
one mode become one kernel. The alternative was to refuse such a pair as an
error.

## Decision

The settings after the mode are an exception to ADR 0034: the kernel name
records only the mode. Two configurations with one mode are one kernel, built
with the first one enumerated. That is fork order in a benchmark and sorted
logic-file order in a library build. Tensile prints one warning for each pair
of configurations it merges (`RegisterAllocationMerges`). The warning names the
configuration not built, the one built, and the first kernel affected.

## Consequences

A benchmark fork over two configurations with one mode measures only the
first, so such configurations are compared in separate runs. The benchmark
cache key includes the fork parameters, so those runs do not reuse each other's
code objects. In a library build, a solution whose configuration was not built
runs the first one's code object, while its logic file still records its own
configuration.

`test_mut_Naming_char.py` pins the shared name. `test_kernel_compile_group.py`
and `TestGenerateForkedSolutions` in `test_r7_benchproblems2_char.py` pin the
first-wins order and the warning. To give each configuration its own kernel,
put the settings in the name, update the naming goldens, and supersede this
ADR.
