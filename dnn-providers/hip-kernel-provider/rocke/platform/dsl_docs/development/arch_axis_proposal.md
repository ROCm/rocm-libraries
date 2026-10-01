# Intrinsic availability by target and LLVM flavor

**Status:** S1 is implemented: the generator, committed artifacts for
`llvm20`, `llvm22`, and `llvm23`, and source-tree validation tests.
S2–S4 describe proposed changes to declaration resolution. The lowering
engines do not yet read the artifacts.

**Scope:** intrinsic declaration resolution in
[`core/lower_llvm.py`](../../python/rocke/core/lower_llvm.py) and
[`cpp/core/lower_llvm/`](../../cpp/core/lower_llvm/).
Related: [multi-toolchain strategy](multi_toolchain_strategy.md).

## 1. Current declaration resolution

Python's `_Lowerer.__init__` selects an ISA backend with
`backend_for(arch or "gfx950")`. It constructs `self._decls` from
`_INTRINSIC_DECLS` and the selected LLVM flavor's overrides. The backend
uses the target for instruction lowering, but the declaration table is
selected by flavor alone.

Most intrinsic requests pass through `_need()`, which records a key in
`self._needs_intrin` without checking its availability for the target.
The `global.atomic.fadd.v2f16` path writes that dictionary directly.
Any future validation in `_need()` must also cover that path.

The C++ lowerer uses the declaration arrays in
[`data.cpp`](../../cpp/core/lower_llvm/data.cpp) and target selection in
[`core.cpp`](../../cpp/core/lower_llvm/core.cpp). It also retains a default
target of `gfx950`. Changing either engine's validation requires equivalent
behavior in the other engine.

## 2. What the artifact measures

[`gen_arch_domain.py`](../../tools/gen_arch_domain.py) measures whether a
selected LLVM toolchain recognizes each declaration and can compile and link
a probe for each target returned by `wired_arches()`. It writes one file
per flavor under
[`core/arch/data/`](../../python/rocke/core/arch/data/):

```text
intrinsic_arch_domain.<flavor>.json
```

The lookup key is `(declaration key, target, LLVM flavor)`. A result describes
the tested compiler and probe operands. It does not establish hardware
instruction availability, correctness for every operand combination, or
GPU numerical correctness.

Compiler support can differ between flavors even when both recognize the
name. For example, the committed `s.wait.dscnt/gfx1201` cells report
`arch_absent` in
[`llvm20`](../../python/rocke/core/arch/data/intrinsic_arch_domain.llvm20.json)
and `ok` in
[`llvm22`](../../python/rocke/core/arch/data/intrinsic_arch_domain.llvm22.json).
The name is recognized in both columns. A flavor-independent set of targets
would lose this distinction.

Each invocation measures one compiler/flavor pair. Multiple toolchains on
one machine, including toolchains in containers, can produce separate
columns. `ROCKE_LLVM_BIN` selects a tool directory and is authoritative:
missing tools are not taken from a different installation. The generator
requires both `opt` and `clang`, with a linker available to clang. It rejects
a recognized clang version that differs from rocKE's resolved flavor;
an unrecognized version banner does not establish a match and is retained
as provenance without enforcing that comparison.

## 3. Probe algorithm and result states

### Stage A: declaration recognition

For each declaration, `_name_exists()` runs `opt -S` on a declaration-only
module. The generator treats attached intrinsic attributes, a remangled
name, or removal by LLVM auto-upgrade as recognition. A declaration that
round-trips unchanged without intrinsic attributes is `name_absent`.

An unsuccessful `opt` invocation is a failed measurement. It stops the
generator without writing an artifact and reports the affected key and
diagnostic. A process failure cannot establish that an intrinsic is absent.

### Stage B: compilation and linking

For each recognized key and target, the generator builds a module containing
a call to the intrinsic. Non-void results are stored with `volatile`.
Clang compiles and links it with `-O0 -nogpulib`.

Linking checks symbol resolution: an unknown `llvm.*` name can survive IR
verification and assembly generation as an external call. A compile-only
check can therefore miss an unresolved intrinsic name. `-O0` limits the
optimizations that could remove the call before instruction selection.
`-nogpulib` excludes device bitcode discovered in another ROCm installation.

Each compile runs in a subprocess with the timeout specified by
`PROBE_TIMEOUT_S`. This isolates compiler failures from the generator.
Failed probes with operand-related diagnostics may be retried with literal
integer operands or alternative immediate values. A successful alternative
establishes `ok` for that probe configuration. If every alternative remains
a probe error, the generator retains the error and refuses to write;
it does not restore the initial target negative.

The generator retries probe errors, compiler crashes, and timeouts serially
after the parallel pass. Persistent probe errors prevent output. Crashes and
timeouts remain recorded outcomes, with diagnostic evidence.

| Status | Interpretation |
|---|---|
| `ok` | The probe compiled and linked for this compiler and target. |
| `name_absent` | The declaration was not recognized in stage A, or linking reported an undefined symbol. |
| `arch_absent` | The compiler rejected lowering for the target with a recognized diagnostic after applicable operand retries. |
| `target_unsupported` | Clang rejected the target ID before compiling the module. |
| `toolchain_crash` | The compiler emitted a recognized crash diagnostic. |
| `toolchain_timeout` | The compiler did not finish within the probe timeout. |
| `probe_error` | Probe construction failed or the diagnostic could not be classified. A persistent error prevents an artifact write. |

`name_absent` is not itself a build error in the current implementation.
The generator records it, and the structural tests accept it as a measured
outcome. Enforcement in a lowering engine is proposed in S2 below.

## 4. S1: artifact format and validation

The artifact contains:

- `schema`: the schema identifier.
- `toolchain`: flavor, clang version banner, target list, generator revision,
  compile flags, and timeout.
- `keys`: one result per declaration key and target. Each cell includes
  `status` and `verified_on`; non-`ok` cells include `evidence`.
- `canonical`: recognized or auto-upgraded names from stage A.

An abbreviated example follows. The complete committed files contain all
selected keys and targets.

```json
{
  "schema": "rocke.intrinsic_arch_domain/v1",
  "toolchain": {
    "flavor": "llvm22",
    "clang": "AMD clang version 22.0.0git ...",
    "arches": ["gfx950"],
    "generator": 4,
    "probe_cflags": ["-O0", "-nogpulib"],
    "probe_timeout_s": 60
  },
  "keys": {
    "ds.read.tr16.b64": {
      "gfx950": {"status": "ok", "verified_on": "llvm22"}
    }
  },
  "canonical": {
    "ds.read.tr16.b64": "llvm.amdgcn.ds.read.tr16.b64.v4i16"
  }
}
```

A cell rescued by the immediate-value sweep also contains `probe_imm`.
That field identifies the successful immediate value; it is not a complete
description of all permitted operands.

[`test_arch_domain_artifact.py`](../../tests/core/test_arch_domain_artifact.py)
provides two checks:

1. Structural checks read the committed files and compare keys, targets,
   statuses, and provenance against the current source. Stale keys fail for
   every flavor. Missing keys fail for the resolved flavor and are reported
   as skipped subtests for other flavors; later columns are still checked.
2. Regeneration runs `gen_arch_domain.py --check` for the resolved flavor.
   It compares measurements and probe configuration. A different clang build
   banner alone is allowed when the rest of the artifact agrees.

The test imports source-tree tools and is excluded from installed pytest
by [CMake](../../CMakeLists.txt). The JSON files are included in both the
Python package and CMake installation. Installed CI does not acquire the
source-tree artifact gate merely by installing those files.

The separate
[`check_ir_validity.py`](../../tools/check_ir_validity.py) checks whole
modules from the representative corpus. It covers emitted combinations of
operations; the artifact generator covers declaration keys even when no
corpus case uses them. Neither check replaces GPU numerical validation.

## 5. Proposed consumers: S2–S4

### S2: validate requests in both lowering engines

A future lookup must retain the flavor:

```text
arch_domain_status(key, arch, flavor)
```

The initial rollout should warn for measured unavailable results and define
how missing columns and unvalidated outcomes are reported. Only a later,
explicit enforcement stage should reject unavailable requests. A compiler
crash, timeout, or unsupported target must not be treated as proof of
intrinsic unavailability.

Route direct writes to `_needs_intrin` through the same validation path before
relying on `_need()` for coverage. Keep diagnostic behavior equivalent in
Python and C++.

### S3: require an explicit target at the internal boundary

Audit callers before making `arch` required at the internal lowerer
boundary. Public APIs may need to preserve `gfx950` defaults for
compatibility. Those APIs should pass the selected default explicitly.
This is a proposed API change; the current default remains in effect.

The audit must include non-kernel lowering entrypoints. Target validation
cannot detect a caller that already substituted the wrong target.

### S4: derive the C++ availability table from the same artifacts

Generate C++ lookup data from the committed records instead of transcribing
it manually. Check regeneration and Python/C++ result parity before
enabling warnings or errors in either engine.

For supported requests, availability checks should preserve emitted IR.
Warnings and rejection behavior are observable API changes and require
separate tests. Byte-identity checks cover successful emission; tests must
also compare rejection behavior.

### Rollout and acceptance

1. Audit explicit target propagation and compatibility requirements (S3).
2. Define handling of every status and missing measurement (S2).
3. Implement both consumers from the same records (S2 and S4).
4. Validate warning behavior and successful Python/C++ emission at each flavor.
5. Promote warnings to errors only after resolving the affected call sites.

S1 is already implemented and is consumed by developer tools and source-tree
tests. It does not change production declaration resolution. Future consumer
changes need their own acceptance evidence; a successful generator run
does not establish that the consumers are correct.

## 6. Verification commands

Run these commands from `rocke/platform/`. The tools bootstrap their
source-tree imports.

```bash
# Generate a partial diagnostic artifact at an explicit destination.
python3 tools/gen_arch_domain.py --only ds.read.tr --arch gfx950 \
    --out ds-read-tr-probes.json --verbose

# Reproduce the full committed column for the selected toolchain.
python3 tools/gen_arch_domain.py --check

# Validate structure and regenerate the selected flavor.
python3 -m pytest tests/core/test_arch_domain_artifact.py

# Validate selected representative modules.
python3 tools/check_ir_validity.py --only gemm --arch gfx942
```

Filtered runs require `--out`, including filtered `--check` runs. Unfiltered
generation writes the resolved flavor's committed column unless `--out`
overrides it. To select another installed toolchain, set
`ROCKE_LLVM_BIN` to its tool directory and `ROCKE_LLVM_FLAVOR` to the matching
supported flavor.

`--prune` removes stale declaration rows from every committed column
without probing. It cannot add missing measurements.

The module validity tool reports unsupported targets, cases requiring a
newer flavor, and missing or mismatched toolchains as unvalidated.
`--strict` treats those outcomes as failures. Its `KNOWN_BAD` allowlist
currently contains two gfx942 attention-3D cases; a selected entry that starts
compiling is reported as stale. Passing this gate requires no new failures
and no stale allowlist entries.
