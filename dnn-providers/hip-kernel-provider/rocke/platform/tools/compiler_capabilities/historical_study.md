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

The probe records twelve focused capability families:

1. the target data layout emitted by `clang`;
2. the module preamble and explicit-layout policy;
3. the complete buffer-resource candidate matrix;
4. FP8/BF8 MFMA packed-operand declarations;
5. named-barrier pointer address spaces;
6. current and historical scaled-MFMA call forms;
7. Clang's canonical FP16 MFMA builtin source prototype;
8. requested and generic processor-name acceptance;
9. focused codegen at every optimization level rocKE uses;
10. known compiler-option acceptance; and
11. known function-attribute codegen acceptance.

The preamble and layout policy are separate capability IDs, which accounts for
twelve report entries across the eleven groups above. `--refresh` ensured every row
came from a fresh compiler invocation. `clang` supplies module observations;
`opt` validates and canonicalizes each intrinsic candidate. The study uses
`opt`'s emitted declaration, not its optimization behavior. The compiler banner
and source revision identify the measured build; they do not supply the behavior
result.

## Weekly-candidate alignment

Each W-numbered column measures one selected public wheel snapshot associated
with that tracked compiler week. An exact match to the final candidate branch
head is preferred but is not required. This treats the nightly wheel stream as
the observable path toward production while retaining the compiler-reported
source and patch revisions needed to reproduce and interpret each column.

A column therefore means “compiler packaged for this weekly-candidate window,”
not “final branch head for this candidate.” Candidate-only changes made after a
wheel cut may be absent. When two weeks use the same available wheel, both
columns remain: the repeated values show that the packaged compiler did not
move during that interval. Only one snapshot per week was probed, and duplicate
installations of that snapshot were ignored.

The selected snapshots come from both historical nightly layouts documented by
TheRock: the legacy per-family indexes for the earlier window and the legacy
multi-arch aggregate index for W25 through W34. A few candidate cut dates lack a
same-day Linux wheel; those columns use the first following Linux snapshot and
say so in the key rather than pretending it is an exact cut-date artifact.

### Public artifact provenance

The historical locations and installation model are documented publicly in
TheRock's version-controlled packaging guides:

- [legacy per-family releases](https://github.com/ROCm/TheRock/blob/main/docs/packaging/legacy_per_family_releases.md);
- [legacy multi-arch releases](https://github.com/ROCm/TheRock/blob/main/docs/packaging/legacy_multi_arch_releases.md); and
- [current release streams](https://github.com/ROCm/TheRock/blob/main/RELEASES.md).

Each selected weekly package was publicly indexed at its recorded `artifact_url`
when the study was collected. Package versions in the key identify the wheel
snapshot; source and patch revisions are read from that wheel's public
`amdclang --version` output.
No internal branch name, link, or tracker content is used as study evidence.
The sanitized machine-readable observations are committed in
`historical_observations.json`; `check_historical_study.py` verifies every matrix
cell and the churn boundaries against that data.

## Observed feature matrix

Features are rows and weekly compiler windows are columns so transitions and
stable runs are visible horizontally. The table is split into chronological
bands only for readability.

### Weekly wheel snapshots: W06–W24

| Feature probe | W06 | W07 | W08 | W09 | W10 | W11 | W12 | W13 | W14 | W15 | W16 | W17 | W18 | W19 | W20 | W21 | W22 | W23 | W24 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reported LLVM major | 22 | 22 | 22 | 22 | 22 | 22 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 |
| Complete data layout | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 |
| ELF mangling field | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Address-space-8 field count | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| Pointer address spaces 10–15 | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N |
| Legacy `p1` / i32 candidate | R | R | R | R | R | R | R | R | R | R | R | R | R | R | R | R | R | R | R |
| Indexed `p8.p1` / i64 candidate | I | I | I | I | I | I | I | I | I | I | I | I | I | I | I | I | I | I | I |
| Canonical target triple | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 |
| Kernel calling convention | K | K | K | K | K | K | K | K | K | K | K | K | K | K | K | K | K | K | K |
| Module flags | F0 | F0 | F0 | F0 | F0 | F0 | F0 | F0 | F1 | F1 | F1 | F1 | F1 | F1 | F1 | F1 | F1 | F1 | F1 |
| Layout policy L0/L1/L2/L3 | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO |
| Buffer-resource accepted ABIs | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 |
| FP8/BF8 packed operand | I | I | I | I | I | I | I | I | I | I | I | I | I | I | I | I | I | I | I |
| Named-barrier pointer AS | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| Scaled-MFMA call ABI | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 |
| FP16 builtin source ABI | H | H | H | H | H | H | F | F | F | F | F | F | F | F | F | F | F | F | F |
| Generic processor name | G | G | G | G | G | G | G | G | G | G | G | G | G | G | G | G | G | G | G |
| Optimization-level codegen | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ |
| Compiler options S/V/I | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI |
| Function attributes 1/2/A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A |

### Weekly wheel snapshots: W25–W34

| Feature probe | W25 | W26 | W26-1 | W27 | W28 | W29a | W29b | W30 | W31 | W32 | W33 | W34 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reported LLVM major | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 |
| Complete data layout | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 | L2 |
| ELF mangling field | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Address-space-8 field count | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| Pointer address spaces 10–15 | N | N | N | N | N | N | N | N | N | N | N | N |
| Legacy `p1` / i32 candidate | R | R | R | R | R | R | R | R | R | R | R | R |
| Indexed `p8.p1` / i64 candidate | I | I | I | I | I | I | I | I | I | I | I | I |
| Canonical target triple | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 | T0 |
| Kernel calling convention | K | K | K | K | K | K | K | K | K | K | K | K |
| Module flags | F1 | F1 | F1 | F1 | F1 | F2 | F2 | F2 | F2 | F2 | F2 | F2 |
| Layout policy L0/L1/L2/L3 | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO |
| Buffer-resource accepted ABIs | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 | I64 |
| FP8/BF8 packed operand | I | I | I | I | I | I | I | I | I | I | I | I |
| Named-barrier pointer AS | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| Scaled-MFMA call ABI | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 |
| FP16 builtin source ABI | F | F | F | F | F | F | F | F | F | F | F | F |
| Generic processor name | G | G | G | G | G | G | G | G | G | G | G | G |
| Optimization-level codegen | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ |
| Compiler options S/V/I | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI |
| Function attributes 1/2/A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A |

`W29a` is the July 14 candidate represented by the July 15 Linux wheel;
`W29b` is the separate July 16 candidate represented by its same-day wheel.

### Weekly wheel snapshots: W35–W46

| Feature probe | W35 | W36 | W37 | W38 | W39 | W40 | W41 | W42 | W43 | W44 | W45 | W46 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reported LLVM major | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| Complete data layout | L2 | L2 | L2 | L2 | L2 | L2 | L3 | L3 | L3 | L3 | L3 | L3 |
| ELF mangling field | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Address-space-8 field count | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| Pointer address spaces 10–15 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| Legacy `p1` / i32 candidate | I | I | I | I | I | I | I | I | I | I | I | I |
| Indexed `p8.p1` / i64 candidate | I | I | I | I | I | I | I | I | I | I | I | I |
| Canonical target triple | T0 | T0 | T0 | T1 | T1 | T1 | T1 | T1 | T1 | T1 | T1 | T1 |
| Kernel calling convention | K | K | K | K | K | K | K | K | K | K | K | K |
| Module flags | F2 | F2 | F2 | F2 | F2 | F2 | F2 | F2 | F2 | F2 | F2 | F2 |
| Layout policy L0/L1/L2/L3 | OOAO | OOAO | OOAO | OOAO | OOAO | OOAO | OOOA | OOOA | OOOA | OOOA | OOOA | OOOA |
| Buffer-resource accepted ABIs | I32+I64 | I32+I64 | I32+I64 | I32+I64 | I32+I64 | I32+I64 | I32+I64 | I32+I64 | I32+I64 | I32+I64 | I32+I64 | I32+I64 |
| FP8/BF8 packed operand | I | I | I | I | I | I | I | I | I | I | I | I |
| Named-barrier pointer AS | 3 | 3 | 3 | 3 | 3 | 3 | 15 | 15 | 15 | 15 | 15 | 15 |
| Scaled-MFMA call ABI | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 |
| FP16 builtin source ABI | F | F | F | F | F | F | F | F | F | F | F | F |
| Generic processor name | G | G | G | G | G | G | G | G | G | G | G | G |
| Optimization-level codegen | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ |
| Compiler options S/V/I | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI | SVI |
| Function attributes 1/2/A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A | 12A |

### Release baselines

The original release measurements remain useful endpoints, but are separate
from the weekly wheel timeline.

| Feature probe | 7.0.2 | 7.1.1 | 7.2.4 | 7.14.0 | 10.0.0 | 10.1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reported LLVM major | 20 | 20 | 22 | 23 | 23 | 24 |
| Complete data layout | L0 | L0 | L1 | L2 | L2 | L2 |
| ELF mangling field | N | N | N | Y | Y | Y |
| Address-space-8 field count | 2 | 2 | 4 | 4 | 4 | 4 |
| Pointer address spaces 10–15 | N | N | N | N | N | N |
| Legacy `p1` / i32 candidate | B | B | R | R | R | I |
| Indexed `p8.p1` / i64 candidate | R | R | I | I | I | I |
| Canonical target triple | T0 | T0 | T0 | T0 | T0 | T0 |
| Kernel calling convention | K | K | K | K | K | K |
| Module flags | F0 | F0 | F0 | F1 | F2 | F2 |
| Layout policy L0/L1/L2/L3 | AOOO | AOOO | OAOO | OOAO | OOAO | OOAO |
| Buffer-resource accepted ABIs | B32 | B32 | I64 | I64 | I64 | I32+I64 |
| FP8/BF8 packed operand | I | I | I | I | I | I |
| Named-barrier pointer AS | 3 | 3 | 3 | 3 | 3 | 3 |
| Scaled-MFMA call ABI | 9 | 9 | 9 | 9 | 9 | 9 |
| FP16 builtin source ABI | H | H | H | F | F | F |
| Generic processor name | G | G | G | G | G | G |
| Optimization-level codegen | 0+1+2+3+ | 0+1+2+3+ | 0+1+2+3+ | 0-1+2+3+ | 0-1+2+3+ | 0-1+2+3+ |
| Compiler options S/V/I | S-I | S-I | SVI | SVI | SVI | SVI |
| Function attributes 1/2/A | 12A | 12A | 12A | 12A | 12A | 12A |

### Weekly snapshot key

| Week | Selected public wheel | Source revision | Patch revision |
| ---: | --- | --- | --- |
| W06 | `7.12.0a20260214` | `3123662d…` | `9a6ac45c…` |
| W07 | `7.12.0a20260218` | `2de9eb60…` | `9a6ac45c…` |
| W08 | `7.12.0a20260218` | `2de9eb60…` | `9a6ac45c…` |
| W09 | `7.12.0a20260228` | `c849bc16…` | `9a6ac45c…` |
| W10 | `7.12.0a20260304` | `c849bc16…` | `9a6ac45c…` |
| W11 | `7.13.0a20260312` | `c849bc16…` | `5b9824ae…` |
| W12 | `7.13.0a20260319` | `91093752…` | `7c736456…` |
| W13 | `7.13.0a20260326` | `926d9066…` | `c8517a28…` |
| W14 | `7.13.0a20260402` | `bc4a1dd0…` | `2506c552…` |
| W15 | `7.13.0a20260409` | `c90f6c11…` | `2506c552…` |
| W16 | `7.13.0a20260416` | `43215c73…` | `2506c552…` |
| W17 | `7.13.0a20260423` | `43215c73…` | `2506c552…` |
| W18 | `7.13.0a20260430` | `43215c73…` | `2506c552…` |
| W19 | `7.13.0a20260507` | `43215c73…` | `2506c552…` |
| W20 | `7.13.0a20260514` | `43215c73…` | `2506c552…` |
| W21 | `7.14.0a20260521` | `5c9bfa94…` | `440716f8…` |
| W22 | `7.14.0a20260528` | `5c9bfa94…` | `440716f8…` |
| W23 | `7.14.0a20260604` | `5c9bfa94…` | `440716f8…` |
| W24 | `7.14.0a20260610` | `5c9bfa94…` | `440716f8…` |
| W25 | `7.14.0a20260619` | `2abe93d5…` | `440716f8…` |
| W26 | `7.15.0a20260626` (first Linux wheel after June 25) | `46fcb339…` | `440716f8…` |
| W26-1 | `7.15.0a20260628` | `46fcb339…` | `440716f8…` |
| W27 | `7.15.0a20260702` | `aa451e1f…` | `440716f8…` |
| W28 | `7.15.0a20260709` | `aa451e1f…` | `440716f8…` |
| W29a | `7.15.0a20260715` (first Linux wheel after July 14) | `97bdf8be…` | — |
| W29b | `7.15.0a20260716` | `0039c223…` | — |
| W30 | `7.15.0a20260723` (first Linux wheel after July 22) | `9f1cce5a…` | — |
| W31 | `10.0.0a20260730` | `4a4173e7…` | — |
| W32 | `10.1.0a20260805` | `998a696f…` | — |
| W33 | `10.1.0a20260813` | `a01cdbd9…` | — |
| W34 | `10.1.0a20260819` | `0bace190…` | — |
| W35 | `10.1.0a20260829` | `2aec646c…` | — |
| W36 | `10.1.0a20260902` | `32fb4582…` | — |
| W37 | `10.1.0a20260909` | `d6f6cb69…` | — |
| W38 | `10.2.0a20260915` | `bc1e171b…` | — |
| W39 | `10.2.0a20260922` | `528402e0…` | — |
| W40 | `10.2.0a20261001` | `4f43f474…` | — |
| W41 | `10.2.0a20261008` | `7667f52d…` | — |
| W42 | `10.2.0a20261014` | `bdc35841…` | — |
| W43 | `10.2.0a20261022` | `2792f8ce…` | — |
| W44 | `10.3.0a20261029` | `4208a889…` | — |
| W45 | `10.3.0a20261105` | `36c1564f…` | — |
| W46 | `10.3.0a20261112` | `caa9e462…` | — |

### Cell key

- `L0`: no ELF mangling field; address space 8 is `128:128`.
- `L1`: no ELF mangling field; address space 8 is `128:128:128:48`.
- `L2`: ELF mangling `e`; address space 8 is `128:128:128:48`; no additional
  pointer address spaces.
- `L3`: L2 plus pointer address spaces 10 through 15.
- `B`: accepted and canonicalized to basic-pointer `p1_i32`.
- `I`: accepted and canonicalized to the indexed ABI at the submitted count
  width (`p8_p1_i32` or `p8_p1_i64`).
- `B32`, `I32`, and `I64`: canonical buffer-resource ABIs accepted by the
  complete four-candidate matrix; `+` joins multiple accepted ABIs.
- `T0`: `amdgcn-amd-amdhsa`; `T1`: processor-specific
  `amdgpu9.42-amd-amdhsa`.
- `K`: the `amdgpu_kernel` calling convention.
- `F0`: code-object, `wchar_size`, PIC-level, and frame-pointer module flags.
- `F1`: F0 without `wchar_size`; `F2`: F1 plus `uwtable`.
- Layout-policy letters are ordered L0/L1/L2/L3: `A` means accepted as
  supplied, `O` means overwritten with the compiler's canonical layout, and
  `R` means rejected.
- FP8/BF8 `I`: all eight families accept `i64` packed operands and reject
  `<2 x i32>`; `M` would mean the families disagree.
- Named-barrier `3` or `15`: all four pointer-taking intrinsics accept that
  address space; `R` means neither candidate was accepted.
- Scaled-MFMA `9`: both shapes accept the current nine-argument form and reject
  the historical eleven-argument form.
- FP16 builtin `H` uses `__fp16` in Clang's canonical builtin prototype; `F`
  uses `_Float16`.
- Generic processor `G`: both the requested processor and its LLVM generic name
  are accepted by the frontend.
- Optimization-level codegen lists each level and `+`/`-` for the focused
  uniform-half high-element extraction reproducer. `0-` is a codegen rejection,
  not a claim that all `-O0` modules fail.
- Compiler options `S`, `V`, and `I` mean the compiler accepts, respectively,
  SLP-off, MFMA VGPR-form, and iterative-ILP scheduler controls; `-` means the
  option is rejected. Acceptance does not prove the requested effect.
- Function attributes `1`, `2`, and `A` mean minimal codegen accepts waves-per-EU
  1, waves-per-EU 2, and zero-AGPR requests. Acceptance or equal assembly does
  not prove the request was honored.
- `R`: rejected by the probe.
- `NA`: the selected weekly snapshot could not be measured; no result is
  inferred.

All measured capabilities had a `known` result.

## What the study shows

### One reported LLVM label, two module contracts

The weekly wheel sequence locates the sampled layout transition inside the
`24.0.0git` group:

```text
W35–W40 → L2: no additional pointer address spaces
W41–W46 → L3: pointer address spaces 10 through 15
```

All of these compilers report `24.0.0git`, but their complete module layouts are
not the same. The first observed weekly-wheel transition is W40 to W41. This
bounds the change in the packaged stream; it does not claim that the W41 wheel's
source revision is the exact upstream commit that introduced it. A rule based
only on the reported major cannot select the complete module preamble.

### The observed facts remain independent

Within the sample, the L2 and L3 LLVM-24 rows accept the same two canonical
buffer-resource ABIs. Therefore the layout difference does not imply a
buffer-resource difference.

The L2 LLVM-23 rows accept only `p8_p1_i64`, whereas the L2 LLVM-24 row accepts
both `p8_p1_i32` and `p8_p1_i64`. Thus the observed buffer-resource set can
differ while the short layout form remains L2. The study does not establish the
precise historical change point; it demonstrates that the two facts should not
be collapsed into one version profile.

### Additional transitions and stable contracts

The denser rows expose five more independent changes:

- Clang's canonical FP16 MFMA builtin prototype moves from `__fp16` to
  `_Float16` between W11 and W12;
- the focused `-O0` high-half extraction reproducer starts failing between W20
  and W21, then recovers between W40 and W41 while `-O1` through `-O3` remain
  accepted throughout;
- the canonical target triple changes between W37 and W38;
- module flags change between W13 and W14, then add `uwtable` between W28 and
  W29a; and
- the named-barrier pointer ABI moves from address space 3 to 15 at the same
  sampled W40-to-W41 boundary as the L2-to-L3 layout and explicit-layout-policy
  transition.

Every W25-through-W34 wheel still reports LLVM 23, including snapshots aligned
with candidate branches that had moved their source label to LLVM 24. This is a
concrete example of a weekly branch and its nearby packaged stream propagating
on different schedules.

Coincidence in the sampled wheel stream is not dependency: the report keeps the
layout and barrier observations separate. In contrast, the FP8/BF8 packed
operand and scaled-MFMA call ABIs remain stable across every sampled weekly
wheel. Those unchanged rows are useful evidence that the matrix captures
stability as well as churn.

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

This is a wheel-snapshot study, not a final-candidate-branch study. It covers a
selected public snapshot for every tracked window from W06 through W46, but not
every post-cut candidate fix or other processor context. Three columns use the
first following Linux wheel because no same-day Linux artifact is indexed. The
study covers twelve focused capability families, not the complete declaration
universe or generated-code behavior. In particular, it does not qualify
generated rocKE modules, Python/C++ byte identity, or numeric kernel correctness.

The next study should preserve the generated reports in an approved evidence
location and run the same probes for every relevant target context. It should
then work systematically through the source-derived backlog in
`flavor_inventory.md`: declaration ABIs and the complete declaration table,
operation families, processor support, backend options, function attributes,
ISA/code-object results, failure modes, and semantic qualification. Existing
qualification can still reveal behavior not in that inventory, but it should
not be the mechanism used to rediscover scope already visible in source.
