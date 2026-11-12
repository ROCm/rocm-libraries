# Historic study: available AMD LLVM compiler observations

This report records a focused, reproducible study of the AMD LLVM compilers
available on one development machine when this document was written. It compares
observed compiler behavior, not rocKE LLVM flavor labels. The local collection
is incomplete and is not a supported-compiler matrix, a release history, or a
qualification result.

The purpose is narrower: show why a reported upstream-development LLVM major is
insufficient as a rocKE compatibility contract, and record the initial evidence
for the capability model.

## Questions and method

For each locally available compiler pair, the study ran the standalone probe
with the AMDGPU target triple and processor `gfx942`:

```bash
python platform/tools/compiler_capabilities/probe.py \
  --llvm-bin <directory-containing-clang-and-opt> \
  --processor gfx942 \
  --refresh \
  --output <report-path>
```

The probe records two facts:

1. the target data layout emitted by `clang`;
2. the canonical buffer-resource ABIs accepted by `opt`.

`--refresh` ensured every row came from a fresh compiler invocation. `clang`
supplies the module-layout observation; `opt` validates and canonicalizes each
intrinsic candidate. The study uses `opt`'s emitted declaration, not its
optimization behavior. The compiler banner and source revision identify the
measured build; they do not supply the behavior result.

## Vocabulary used in the table

The complete target data-layout string remains the evidence in each report. The
study uses the following short names only to make the comparison readable.

| Layout form | ELF mangling | Address-space-8 fields | Additional pointer address spaces |
| --- | --- | --- | --- |
| L0 | absent | `128:128` | none |
| L1 | absent | `128:128:128:48` | none |
| L2 | `e` | `128:128:128:48` | none |
| L3 | `e` | `128:128:128:48` | `10, 11, 12, 13, 14, 15` |

The buffer-resource column lists **canonical ABIs**, not the text submitted to
`opt`:

| Canonical ABI | Meaning |
| --- | --- |
| `p1_i32` | basic-pointer intrinsic with a 32-bit count |
| `p8_p1_i32` | indexed-pointer intrinsic with a 32-bit count |
| `p8_p1_i64` | indexed-pointer intrinsic with a 64-bit count |

## Observations

The installation names are local labels only. They are included so another
engineer with the same collection can reproduce a row; they are not official
compiler release names. Revisions are the compiler-reported source revisions.

| Local installation | Reported compiler label | Source revision | Layout | Accepted canonical ABIs | Rejected probe candidates |
| --- | --- | --- | --- | --- | --- |
| `rocm-7.0.2` | `20.0.0git` | `0dda3adf56766e0aac0d03173ced3759e1ffecbc` | L0 | `p1_i32` | `p8_p1_i64` |
| `rocm-7.1.1` | `20.0.0git` | `27682a16360e33e37c4f3cc6adf9a620733f8fe1` | L0 | `p1_i32` | `p8_p1_i64` |
| `rocm-7.2.4` | `22.0.0git` | `f58b06dce1f9c15707c5f808fd002e18c2accf7e` | L1 | `p8_p1_i64` | `legacy_p1_i32` |
| `rocm-10.1.0a20260822` | `23.0.0git` | `0bace1908348b840e6aa1b4b6e12151dae208158` | L2 | `p8_p1_i64` | `legacy_p1_i32` |
| `rocm100` | `23.0.0git` | `8f497e0992fb7513f7f78a6f6b6f1056c375e961` | L2 | `p8_p1_i64` | `legacy_p1_i32` |
| `rocm713` | `23.0.0git` | `43215c73116c407735c85a180d174f718798c328` + patch | L2 | `p8_p1_i64` | `legacy_p1_i32` |
| `rocm714` | `23.0.0git` | `46fcb339fb61119b337f973c7ca9e710a319fdd0` + patch | L2 | `p8_p1_i64` | `legacy_p1_i32` |
| `rocm101` | `24.0.0git` | `d6f6cb691863e9258034ab48c1dfce21689d5606` | L2 | `p8_p1_i32`, `p8_p1_i64` | none |
| `rocm-10.2.0a20261015` | `24.0.0git` | `bdc35841c97ff51400273b91d7bc05c96e19c2d9` | L3 | `p8_p1_i32`, `p8_p1_i64` | none |
| `rocm-10.2.0a20261016` | `24.0.0git` | `bdc35841c97ff51400273b91d7bc05c96e19c2d9` | L3 | `p8_p1_i32`, `p8_p1_i64` | none |
| `rocm-10.3.0a20261026` | `24.0.0git` | `2792f8ceeb5aa6e188c1ce2b735923b2406a8e59` | L3 | `p8_p1_i32`, `p8_p1_i64` | none |
| `rocm-10.3.0a20261029` | `24.0.0git` | `4208a88993d31a4311127ddae671ec6a6b1000c6` | L3 | `p8_p1_i32`, `p8_p1_i64` | none |
| `rocm-10.3.0a20261110` | `24.0.0git` | `3eea51fc599698882db4dc927d80ec0e47bec0d9` | L3 | `p8_p1_i32`, `p8_p1_i64` | none |

All rows had a `known` result for both measured capabilities. The two
installations sharing revision `bdc358…` also had identical observed values,
which is a useful local repeatability check but not a broad reproducibility
claim.

## What the study shows

### One reported LLVM label, two module contracts

The most direct result is inside the `24.0.0git` group:

```text
revision d6f6cb…  → L2: no additional pointer address spaces
revision bdc358…  → L3: pointer address spaces 10 through 15
later sampled revisions → L3
```

All of these compilers report `24.0.0git`, but their complete module layouts
are not the same. A rule based only on the reported major cannot select the
complete module preamble.

### The observed facts remain independent

Within the sample, the L2 and L3 LLVM-24 rows accept the same two canonical
buffer-resource ABIs. Therefore the layout difference does not imply a
buffer-resource difference.

The L2 LLVM-23 rows accept only `p8_p1_i64`, whereas the L2 LLVM-24 row accepts
both `p8_p1_i32` and `p8_p1_i64`. Thus the observed buffer-resource set can
differ while the short layout form remains L2. The study does not establish the
precise historical change point; it demonstrates that the two facts should not
be collapsed into one version profile.

### Acceptance is not a selection rule

For every sampled LLVM-24 compiler, the legacy 32-bit input candidate was
accepted but canonicalized by `opt` to `p8_p1_i32`. The direct indexed 64-bit
candidate canonicalized to `p8_p1_i64`. The compiler therefore reveals the
indexed intrinsic family as the current convention while retaining two distinct
count-width overloads.

That result does not tell rocKE which overload to emit. A future rocKE resolver
must own and qualify an explicit policy, such as preferring `p8_p1_i64` when it
is accepted and using legacy `p1_i32` only when necessary for compatibility.
This is a rocKE decision informed by observations, not a capability implied by
a reported LLVM label.

## Limits and next study

This is an opportunistic local sample. It omits many compiler revisions,
possible patch sets, and processor contexts. It also covers only two of the
behaviors identified by the source inventory. In particular, it does not
qualify generated rocKE modules, Python/C++ byte identity, or numeric kernel
correctness.

The next study should obtain a deliberately selected set of compiler builds
across the intended support window, preserve the generated reports in an
approved evidence location, and run the same probes for every relevant target
context. It should add further focused probes only where the existing
qualification work exposes a compiler-variable behavior not described by the
current observations.
