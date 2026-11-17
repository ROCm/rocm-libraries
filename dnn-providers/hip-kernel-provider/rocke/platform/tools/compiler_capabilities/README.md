# Understanding AMD LLVM variation

AMD LLVM builds can differ in the LLVM IR they emit and accept. rocKE must
observe the differences that affect it, emit matching IR through both its Python
and C++ engines, and then qualify that response.

A compiler version or source revision identifies what was tested. It does not by
itself prove how that compiler behaves. The broader compatibility model follows
one practical rule:

> **Probe what the compiler accepts, inspect what it produces, and qualify what
> the GPU executes.**

The source inventory explains the kernel journey and eight concrete ways
compatibility can vary, with developer-visible failures and mitigation
strategies. The standalone structural probe measures:

1. the AMDGPU module data layout;
2. the canonical module preamble and explicit-layout handling policy;
3. every relevant buffer-resource spelling and count width;
4. all eight fp8/bf8 MFMA packed-operand declaration families;
5. all four named-barrier pointer declarations;
6. the current and historical scaled-MFMA call forms at both known shapes;
7. Clang's canonical FP16 MFMA builtin prototype;
8. requested and generic processor-name acceptance;
9. focused codegen at `-O0` through `-O3`;
10. known backend-option acceptance; and
11. known function-attribute codegen acceptance.

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

These demonstrations implement the first two steps. They do not configure rocKE
lowering or qualify a compiler build.

## Why the probe uses `clang` and `opt`

The demonstration uses two LLVM tools because they answer different questions.

- `clang` emits a minimal AMDGPU kernel module. Its output supplies the target
  data layout, triple, kernel calling convention, and module flags. Re-reading
  modules carrying each historical layout reveals whether the compiler accepts,
  overwrites, or rejects explicit layout evidence.
- `opt` parses minimal LLVM-IR modules containing candidate intrinsic
  declarations and calls. Its results supply intrinsic acceptance and, when a
  candidate is accepted, the canonical declaration LLVM writes back.

`opt` is used here as an LLVM-IR validator and canonicalizer, not to evaluate
optimization quality. Its name means optimizer, but the probe runs no
optimization pipeline.

## What the report contains

The report deliberately separates compiler identity from compiler behavior:

| Report section | Purpose | A compiler capability? |
| --- | --- | --- |
| `probe` | Records source digests, measurement context, and fingerprint | No |
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
| `p1_i32` | `llvm.amdgcn.make.buffer.rsrc.p1` | `i32` |
| `p1_i64` | `llvm.amdgcn.make.buffer.rsrc.p1` | `i64` |
| `p8_p1_i32` | `llvm.amdgcn.make.buffer.rsrc.p8.p1` | `i32` |
| `p8_p1_i64` | `llvm.amdgcn.make.buffer.rsrc.p8.p1` | `i64` |

#### Why rocKE cares

rocKE must emit a matching declaration, call shape, and count conversion. A
form rejected by the selected compiler makes the generated module invalid.

#### How it is measured

The probe submits all four spelling/count-width candidates to:

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

## Additional structural contracts

### Module preamble and explicit-layout policy

**Machine IDs:** `amdgpu.ir.module_preamble` and
`amdgpu.ir.explicit_data_layout_policy`

The preamble probe records the compiler-emitted target triple, kernel calling
convention, and module flags. The policy probe then submits L0 through L3
layouts and records `accepted`, `overwritten`, or `rejected` for each. A
canonical layout alone cannot reveal whether a compatibility superset survives
code generation.

### Complete buffer-resource candidate matrix

The probe crosses both known intrinsic spellings (`p1`, `p8.p1`) with both
count widths (`i32`, `i64`). Accepted candidates retain the submitted form and
the canonical intrinsic, declaration, and ABI selected by LLVM.

### FP8/BF8 MFMA packed operands

**Machine ID:** `amdgpu.intrinsic.mfma_fp8_bf8.packed_operand_type`

All homogeneous and mixed fp8/bf8 families at both affected shapes are tested
with `<2 x i32>` and `i64` operands. Keeping all eight families prevents one
source table from standing in for another.

### Named-barrier pointer address space

**Machine ID:** `amdgpu.intrinsic.named_barrier.pointer_address_space`

The four pointer-taking barrier intrinsics are independently tested with address
spaces 3 and 15. Acceptance by one operation is not assumed to prove the other
three.

### Scaled-MFMA call ABI

**Machine ID:** `amdgpu.intrinsic.mfma_f8f6f4.call_abi`

Both known shapes are tested with the current nine-argument form and the
historical eleven-argument form containing the earlier source-control operands.
The report preserves the submitted candidate separately from LLVM's canonical
declaration.

### Frontend, target, pipeline, option, and attribute probes

The remaining focused probes use a common observation shape:

- `candidate` names the exact source type, processor form, optimization level,
  option, or attribute submitted;
- `family` names the independently varying contract;
- `stage` records whether the answer came from the frontend, option parser, or
  target codegen;
- `outcome` distinguishes accepted, rejected, crash, and timeout; and
- `evidence` retains compact canonical output such as a builtin prototype or an
  output digest.

The frontend probe asks Clang for the canonical type of a representative FP16
MFMA builtin, detecting source-ABI migrations that LLVM-IR probes cannot see.
The processor probe independently checks the requested processor and a
compiler-facing LLVM generic-name candidate. That candidate can be accepted by
LLVM without being a generic fallback catalogued by rocKE. The optimization-level
probe compiles a reduced high-half
vector extraction reproducer at every level rocKE uses. The option and
attribute probes cover controls already emitted or relied on by rocKE.

Option or attribute acceptance is only the first fact. Equal or different
assembly from a minimal kernel does not prove that a control is honored for a
resource-sensitive workload. Effect checks remain separate generated-code
probes and qualification sentinels.

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

## Run the probe

`probe.py` is the single front door and writes `capabilities.json` matching the
schema in the same checkout. From the rocKE root, point it at a normal ROCm Python
environment containing `rocm-sdk-core`:

```bash
python platform/tools/compiler_capabilities/probe.py \
  --python-env <environment> \
  --processor gfx942 \
  --validate
```

The probe finds the packaged `clang`/`opt` pair, prints a compact summary, and
reuses an exact matching cache. `--validate` checks the generated report against
the shipped schema and requires the optional `jsonschema` package. Omit it when
that package is unavailable; report construction still enforces the probe's
internal contracts.

Advanced users can select a tool directory directly:

```bash
python platform/tools/compiler_capabilities/probe.py \
  --llvm-bin <directory-containing-clang-and-opt>
```

The equivalent environment variables are `ROCKE_PYTHON_ENV` and
`ROCKE_LLVM_BIN`; explicit command-line arguments take precedence. `PATH` is the
final fallback. Use `--output`, `--cache-dir`, and `--quiet` to control the
artifact and presentation.

Use `--refresh` to bypass matching behavior evidence while collecting a study:

```bash
python platform/tools/compiler_capabilities/probe.py \
  --python-env <environment> \
  --processor gfx942 \
  --refresh \
  --validate
```

The compiler and tool identities are still read before a cache hit, so changing
the compiler cannot silently reuse another compiler's evidence. The default
cache is under the platform temporary directory.

### Weekly study workflow

1. Select one public wheel snapshot for each tracked weekly window.
2. Reuse a snapshot when no distinct wheel exists; ignore duplicate
   installations of the same snapshot.
3. Run the canonical command with the same processor and `--refresh` for every
   selected snapshot.
4. Keep source and patch revisions as provenance, not capability labels.
5. Preserve unavailable windows as missing evidence; do not infer them from
   adjacent snapshots.
6. Render features as rows and weekly windows as columns, using the compact cell
   vocabulary documented in `historical_study.md`.

## Scope and limitations

The source inventory still contains more compiler-dependent behavior than the
structural slice. The remaining backlog includes the complete declaration
universe, processor and intrinsic target support, independently gated operation
families, backend options, function attributes, instruction selection,
code-object metadata, failure modes, and semantic qualification. Neither report
is a complete compatibility matrix or the shape of the eventual resolver. This
change intentionally adds no CI integration and no production consumer.

Further probes can land incrementally, but they should be implemented from the
source-derived backlog and its mechanical completeness rules rather than waiting
for qualification failures to rediscover known scope. A qualification failure
can still expose a genuinely new behavior axis.

For the source-code research, exhaustive backlog, and full taxonomy, see
[`flavor_inventory.md`](flavor_inventory.md). For the machine contract and an
illustrative report, see
[`capabilities.schema.json`](capabilities.schema.json) and
[`example.capabilities.json`](example.capabilities.json). The status, cache,
evolution, and retirement decisions are normative in
[`SCHEMA_DECISION.md`](SCHEMA_DECISION.md). The measured change frequency,
interpretation toil, and steady-state assessment are in
[`churn_report.md`](churn_report.md). Sanitized observations live in
[`historical_observations.json`](historical_observations.json); run
`python platform/tools/compiler_capabilities/check_historical_study.py` to check
the published matrix and churn boundaries.
