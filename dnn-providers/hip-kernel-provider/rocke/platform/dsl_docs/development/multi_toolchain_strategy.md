# Supporting multiple ROCm, LLVM, and target combinations

This document describes rocKE's current LLVM-flavor model and proposed
changes to target availability and declaration maintenance. The intrinsic
availability generator is implemented; lowerer consumption and the other
refactorings below remain proposals.

The [intrinsic availability design](arch_axis_proposal.md) defines the
generator, artifact schema, status meanings, and verification commands.

## 1. Current LLVM-flavor model

### 1.1 ROCm release mapping

[`core/lower_llvm.py`](../../python/rocke/core/lower_llvm.py) defines
`LLVM_FLAVORS` and `_ROCM_FLAVOR_LADDER`. The ladder maps a ROCm version
to the declaration and datalayout forms rocKE emits:

```python
_ROCM_FLAVOR_LADDER = (
    ((7, 13), LLVM_FLAVOR_LLVM23),
    ((7, 2), LLVM_FLAVOR_LLVM22),
)
```

`_flavor_for_rocm()` selects the first matching row. Versions older than
all rows map to `LLVM_FLAVORS[0]`; versions newer than the latest row map
to the latest supported flavor. This is a version-mapping policy, not a
query of the compiler's intrinsic support.

### 1.2 Flavor detection and caching

`_detect_llvm_flavor()` uses these sources in order:

1. A recognized `ROCKE_LLVM_FLAVOR` override.
2. The ROCm version associated with the resolved COMGR library.
3. `torch.version.hip`, if torch is already imported.
4. The system ROCm version file.
5. The default `llvm22` flavor.

Unknown environment values fall through to detection. Explicit
`llvm_flavor=` API arguments are validated against the supported flavor set.

`_resolve_llvm_flavor()` caches the result using the resolved COMGR library
path as its basis. If library discovery changes after torch is imported,
the changed path causes flavor detection to run again. The override bypasses
that cache.

The C++ implementation in
[`core.cpp`](../../cpp/core/lower_llvm/core.cpp) uses
`ROCKE_LL_FLAVOR_LADDER` for flavor names, validation, and version mapping.
`ll_resolve_flavor()` reads `ROCKE_LLVM_FLAVOR` directly with `getenv`.

### 1.3 Datalayout compatibility

`LlvmDatalayoutKind` groups flavors by the buffer-resource address-space
index layout:

| Kind | Marker | Flavors |
|---|---|---|
| `P8_PLAIN` | `p8:128:128` | `llvm20` |
| `P8_INDEXED` | `p8:128:128:128:48` | `llvm22`, `llvm23` |

LLVM 22 and LLVM 23 share the indexed marker but differ in other datalayout
fields. The guard in
[`runtime/comgr.py`](../../python/rocke/runtime/comgr.py),
`_assert_ir_flavor_matches_lib()`, therefore compares datalayout kinds.
That check cannot identify an exact LLVM flavor from the `p8` marker alone.

### 1.4 Declaration text

`_Lowerer.__init__` copies `_INTRINSIC_DECLS` and applies the selected
flavor's override dictionary. Keys remain stable across those dictionaries,
so callers of `_need()` do not select declaration strings themselves.

The LLVM 22 dictionary overrides four FP8/BF8 MFMA declarations and
`make.buffer.rsrc.p1`. The LLVM 23 dictionary inherits those entries and
also overrides `mfma.scale.f32.16x16x128.f8f6f4`.

The base forms describe LLVM 20. For example, the FP8/BF8 MFMA declarations
use packed `<2 x i32>` operands there and `i64` in the later overrides.
The [LLVM 20 artifact](../../python/rocke/core/arch/data/intrinsic_arch_domain.llvm20.json)
contains successful probes for those base declarations on supported targets.
An older diagnosis of signature drift should not be applied to the current
table without reproducing it with the selected compiler.

### 1.5 Existing checks

Flavor and datalayout tests in
[`tests/test_rocke.py`](../../tests/test_rocke.py) cover supported flavor
membership, layout partitions, and Python/C++ API agreement.
[`test_rocke_multiarch.py`](../../tests/instances/test_rocke_multiarch.py)
includes comparison with IR emitted by an available HIP compiler.

The validation tools cover different contracts:

| Check | Coverage | Evidence |
|---|---|---|
| Representative IR golden | Flavor is an explicit input to `lower_case()`; the file has separate flavor records. | Emitted text remains stable. |
| Byte identity | One selected flavor per invocation, passed through the environment to both engines. | Python and C++ emit matching IR for the tested cases. |
| Module validity | Selected corpus cases compiled and linked with a selected toolchain. | That compiler accepts those modules. |
| Arch-domain regeneration | Declaration probes for the selected compiler/flavor and wired targets. | Committed probe results and configuration reproduce. |
| GPU numerical validation | Executed kernels and tested inputs on a compatible device. | Outputs agree with an independent reference within the declared tolerance. |

The developer runner
[`tests/run_all.py`](../../tests/run_all.py) invokes module validity by
default. The arch-domain test imports the generator and is excluded from
installed pytest. Installed CTest coverage is defined by
[`tests/CMakeLists.txt`](../../tests/CMakeLists.txt) and
[installation rules](../../CMakeLists.txt); source-tree gates must not be
assumed to run in an installed CI lane.

## 2. Remaining limitations

### 2.1 Target availability is not checked at declaration resolution

`_Lowerer` selects an ISA backend from the target, but constructs its
declaration table from the flavor. `_need()` records requests without
consulting measured availability. Some operation handlers enforce their
own target requirements; there is no common availability check for every
declaration key.

The internal `arch or "gfx950"` default also means an omitted target
selects gfx950. An availability check on that substituted target would not
detect the caller's omission.

### 2.2 Compiler acceptance depends on both target and flavor

A recognized intrinsic name does not imply that a particular compiler can
lower it for every target. The committed `s.wait.dscnt/gfx1201` result is
`arch_absent` in LLVM 20 and `ok` in LLVM 22 and LLVM 23. Both stages of
availability therefore remain indexed by flavor:

```text
availability(key, target, flavor)
declaration_text(key, flavor)
```

The artifact measures compiler behavior for selected operands. A compiler
may implement an operation with different instructions or support it only
in later releases. Hardware capability and compiler acceptance require
separate evidence.

### 2.3 Declaration data is maintained in both engines

Python stores declaration strings in dictionaries; C++ stores them in
[`data.cpp`](../../cpp/core/lower_llvm/data.cpp). Changes must keep both
representations consistent. Byte-identity coverage checks emitted
declarations in its corpus, but does not by itself establish agreement for
every unused table entry.

Python also distributes flavor metadata across `LLVM_FLAVORS`,
`_DATALAYOUT_KIND_FLAVORS`, `_P8_MARKERS`, `_ROCM_FLAVOR_LADDER`, and
`_datalayout_for_flavor()`. Adding a flavor requires reviewing each of
those structures.

## 3. Toolchain evidence and support policy

Each probe invocation selects one toolchain and flavor. A machine may have
multiple toolchain installations or containers; separate invocations can
validate separate columns. The generator needs both `opt` for declaration
recognition and `clang` with a linker for target probes.

`ROCKE_LLVM_BIN` selects an authoritative tool directory. Without it,
tool discovery searches the resolved ROCm installation, ROCm environment
prefixes, and PATH. The generator compares a recognized clang version
banner with the resolved flavor and refuses a mismatch. The module
validity tool reports a mismatch as unvalidated unless `--force` is used
for a diagnostic run. Unrecognized banners do not establish identity and
do not trigger the version comparison.

The committed files retain compiler identity and probe configuration.
Regeneration compares measured cells and configuration while allowing a
different compiler build banner if the other fields agree. No matching
column means that flavor has not been recorded; it does not imply
unavailability.

A future support policy should explicitly list supported compiler/flavor
and target combinations and the evidence required for each. Compile/link
results, Python/C++ parity, and GPU numerical validation should remain
separate fields. Missing evidence should be reported as unvalidated rather
than converted into a positive or negative capability claim.

## 4. Proposed changes

### R1: generated availability artifacts — implemented

[`gen_arch_domain.py`](../../tools/gen_arch_domain.py) probes the merged
declaration table and writes one file per flavor. It checks declaration
recognition with `opt`, then compiles and links target probes at
`-O0 -nogpulib`. Failed recognition processes and persistent probe errors
prevent output.

The [artifact tests](../../tests/core/test_arch_domain_artifact.py) validate
structure and regenerate the selected flavor. See the
[artifact design](arch_axis_proposal.md) for statuses, provenance, exclusions
from installed tests, and commands. The lowerers do not consume this data yet.

### R2: common availability validation

Add a lookup on `(key, target, flavor)` at the declaration-request boundary.
Route direct writes to the requested-intrinsic set through that boundary.
Specify behavior for measured negatives, missing rows, unsupported
toolchains, crashes, and timeouts before enabling the consumer.

Introduce equivalent warnings in Python and C++ first. A later enforcement
change can reject measured unavailable requests once affected callers
have been reviewed. Audit target propagation and preserve public defaults
where compatibility requires them.

### R3: explicit support and validation records

Define the combinations the project supports and the evidence needed to
claim that support. Run the required toolchain and GPU lanes for those
combinations. A skipped lane must retain its reason and must not be counted
as a successful validation.

### R4: structured declaration records

Consider replacing duplicated declaration text with records containing
names, return types, operand types, and overload information. Before
adopting a renderer, require it to reproduce every existing declaration
byte for byte across supported flavors. Architecture availability should
remain a separate compiler-dependent measurement.

This proposal can use repository-owned records. It does not require LLVM
source `.td` files or headers to be available in an installed toolchain.

### R5: generate both engines' tables

If structured records are introduced, generate Python and C++ tables from
the same source and check regeneration. Keep independent parity checks:
shared generation reduces manual duplication but does not prove the
renderer or each consumer is correct.

### R6: consolidate Python flavor metadata

Consider one Python flavor table for names, version thresholds, and layout
metadata, corresponding to the C++ ladder. Preserve detection precedence,
cache invalidation, fallback behavior, and explicit-argument validation.
This is a separate refactoring from adding availability checks.

## 5. Related coverage work

Whole-module validation and declaration probes cover different inputs.
Extend representative cases when an operation gains a new target or
operand form, and validate the resulting kernels numerically where runtime
support is claimed.

Target routing is also distinct from intrinsic availability. A kernel
built for the wrong target may contain declarations that are all valid for
that wrong target. Builder and dispatch tests must therefore check that
the requested target reaches the selected kernel and lowerer.
