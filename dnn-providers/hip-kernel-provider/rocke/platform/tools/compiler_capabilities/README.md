# Understanding AMD LLVM variation

AMD LLVM builds can differ in the LLVM IR they emit and accept. rocKE must
observe the differences that affect it, emit matching IR through both its Python
and C++ engines, and then qualify that response.

A compiler version or source revision identifies what was tested. It does not by
itself prove how that compiler behaves. This standalone demonstration measures
two focused behaviors directly:

1. the AMDGPU module data layout emitted by `clang`;
2. the buffer-resource intrinsic call forms accepted by `opt`.

The generated JSON preserves machine-readable evidence. This README explains
what that evidence means for rocKE.

## The compatibility loop

```text
Identify the exact compiler
            ↓
Probe focused behavior
            ↓
Choose matching rocKE emission
            ↓
Verify Python/C++ byte identity
            ↓
Compile complete modules
            ↓
Run correctness qualification
```

This demonstration implements the first two steps. It does not configure rocKE
lowering or qualify a compiler build.

## Why the probe uses `clang` and `opt`

The demonstration uses two LLVM tools because they answer different questions.

- `clang` emits a minimal AMDGPU module. Its output supplies the target
  data-layout observation.
- `opt` parses a minimal LLVM-IR module containing a candidate intrinsic
  declaration and call. Its result supplies intrinsic acceptance and, when the
  candidate is accepted, the canonical declaration LLVM writes back.

`opt` is used here as an LLVM-IR validator and canonicalizer, not to evaluate
optimization quality. Its name means optimizer, but the probe runs no
optimization pipeline.

## What the report contains

The report deliberately separates compiler identity from compiler behavior:

| Report section | Purpose | A compiler capability? |
| --- | --- | --- |
| `schema_version` | Identifies the report format | No |
| `probe` | Records the measurement context and fingerprint | No |
| `compiler` | Identifies the exact `clang` that was measured | No |
| `tools` | Identifies supporting tools such as `opt` | No |
| `capabilities` | Records observed compiler behavior | Yes |

This separation prevents compiler banners, revisions, and executable hashes
from becoming indirect version checks. They establish provenance; the focused
probes establish behavior.

## Behaviors measured by this demonstration

### AMDGPU module layout contract

**Machine ID:** `amdgpu.ir.data_layout`

#### What can vary

AMD LLVM builds may emit different target data-layout strings. Relevant
differences include the representation of address space 8, the presence of ELF
mangling information, and pointer address spaces added beyond the baseline
set. For example, a compiler may emit:

```llvm
target datalayout = "e-m:e-p:64:64-p8:128:128:128:48"
```

#### Why rocKE cares

rocKE emits complete LLVM modules, including their target data layout. Its
module preamble must match the compiler contract used to consume the module.

#### How it is measured

The probe asks `clang` to emit a minimal AMDGPU module and records:

- the complete data-layout string;
- the ELF mangling field, when present;
- the address-space-8 layout, when present;
- pointer address spaces declared beyond address space 9.

The complete string is the primary observation. The parsed fields make
important differences easier to compare. In particular, two builds can agree
on ELF mangling and address space 8 but still differ because one declares
additional pointer address spaces.

### Buffer-resource intrinsic call contract

**Machine ID:** `amdgpu.intrinsic.make_buffer_resource.abis`

AMD LLVM builds may accept different declarations and operand types for the
buffer-resource intrinsic.

| Probe candidate | Submitted intrinsic spelling | Count type |
| --- | --- | --- |
| `legacy_p1_i32` | `llvm.amdgcn.make.buffer.rsrc.p1` | `i32` |
| `p8_p1_i64` | `llvm.amdgcn.make.buffer.rsrc.p8.p1` | `i64` |

#### Why rocKE cares

rocKE must emit a matching declaration, call shape, and count conversion. A
form rejected by the selected compiler makes the generated module invalid.

#### How it is measured

The probe submits both candidates to:

```bash
opt -S - -o -
```

Here `-S` requests textual LLVM IR, the first `-` reads the minimal test module
from standard input, and `-o -` writes the normalized module to standard
output. A nonzero result rejects the candidate. On success, the probe reads the
output declaration and records:

- `accepted`: the submitted candidate and the canonical ABI/declaration `opt`
  resolved it to;
- `rejected`: candidates the compiler examined and rejected;
- `error`: the probe did not obtain a reliable answer.

This distinction matters. A newer compiler can accept the legacy spelling but
canonicalize it to a different ABI, such as `p8_p1_i32`. The candidate is probe
input; `canonical_abi` is the compiler's observed output contract.

A rejected candidate is a successful observation of unsupported behavior. It is
not a probe failure.

### Observation is not rocKE selection policy

The compiler can accept more than one canonical ABI. Acceptance alone does not
select the form rocKE should emit. rocKE must apply a separate, explicit policy
that is qualified for the relevant compiler set. For this family, the source
inventory identifies a natural future policy:

```text
Prefer p8_p1_i64 when that canonical ABI is accepted.
Use legacy p1_i32 only when the preferred ABI is unavailable.
```

That policy is not a compiler-reported fact and is not implemented by this
demonstration. It reflects rocKE's need to preserve `i64` byte counts and its
existing modern emission path. The report intentionally records facts first;
any future resolver must make its preference and fallback rules separately
visible and testable.

## How to interpret a report

A `known` result means that the focused probe completed and produced an
observation. It does **not** mean that the compiler is qualified for rocKE.
Qualification still requires:

1. selecting the matching emission behavior;
2. producing byte-identical Python and C++ IR;
3. compiling complete generated modules;
4. running the relevant correctness tests.

Likewise, two compiler builds with the same version text may produce different
capability results, and two differently labelled builds may produce the same
results. The local compiler collection demonstrates this directly: compilers
reporting `24.0.0git` produced two distinct complete module layouts, while the
same label retained both canonical buffer-resource overloads. That is why
behavior and provenance are recorded separately.

## Run the demonstration

From the rocKE root:

```bash
python platform/tools/compiler_capabilities/probe.py \
  --llvm-bin <directory-containing-clang-and-opt> \
  --output capabilities.json
```

`--llvm-bin` takes precedence over `ROCKE_LLVM_BIN`; `PATH` is the final
fallback. Use `--processor` to change the target-specific probe processor.

The CLI prints the compiler source revision, cache status, observed capability
values, output path, and elapsed time. Use `--refresh` to rerun the probes
instead of reading a matching cache entry:

```bash
python platform/tools/compiler_capabilities/probe.py \
  --llvm-bin <directory-containing-clang-and-opt> \
  --refresh
```

The default cache is under the platform temporary directory. Override it with
`--cache-dir` when demonstrating cache contents or isolation.

## Scope and limitations

The source inventory found more compiler-dependent decisions than the two in
this demonstration, including other intrinsic ABIs, operation availability,
processor-name support, and intrinsic availability by processor. The v1 report
is deliberately not a complete compatibility matrix.

It selects one module-level contract and one intrinsic-call contract because
each has visible flavor-dependent IR, a small direct probe, and focused existing
rocKE tests. Expanding the report should follow qualification evidence: add the
smallest new probe only when a real compiler-variable behavior needs to be made
explicit.

For the source-code research and full taxonomy, see
[`flavor_inventory.md`](flavor_inventory.md). For the machine contract and an
illustrative report, see
[`capabilities-v1.schema.json`](capabilities-v1.schema.json) and
[`example.capabilities.json`](example.capabilities.json).
