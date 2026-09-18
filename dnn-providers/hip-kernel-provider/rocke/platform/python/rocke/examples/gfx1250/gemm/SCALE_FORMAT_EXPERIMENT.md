# gfx1250 scale-format discrimination experiment

Status: planned; this experiment has not been implemented or run. Complete the
FP6 / scale-format PR split before executing it. Existing mixed-format fixtures
are separate evidence and do not count as execution of this plan.

## Contents

- [Question and evidence boundary](#question-and-evidence-boundary)
- [Minimal experiment](#minimal-experiment)
- [Literal predictions](#literal-predictions)
- [Controls and follow-up](#controls-and-follow-up)
- [Evidence and interpretation](#evidence-and-interpretation)

## Question and evidence boundary

Does scale selector 1 numerically decode unsigned E5M3 on gfx1250, distinctly
from selector 2 (E4M3), selector 0 (E8M0), or ignored scales?

The public [machine-readable ISA archive](https://gpuopen.com/download/machine-readable-isa/latest/)
snapshot `AMD_GPU_MR_ISA_XML_2026_08_06.zip`, specifically
`amdgpu_isa_cdna5.xml`, describes SCALE and SCALE16, their block sizes of 32 and
16, and B32/B64 scale operands. It does not define E5M3 scale decoding or the
complete scale-format legality table. LLVM's `WMMA::MatrixScaleFmt` names E8,
E5M3, and E4M3 selectors 0, 1, and 2. These are distinct evidence sources:
compiler acceptance alone does not establish hardware numerical behavior.

The experiment tests the encoding hypotheses below. It does not establish
undocumented combinations as an architectural guarantee.

## Minimal experiment

1. Use one native 16x16x128 atom with FP4 A, FP8 E4M3 B, and a zero accumulator.
   Store BF16 output using the existing builder. Every predicted finite value
   below is exactly representable in BF16.
2. Set `A[m, k0] = 1` and `B[k0, n] = 1` for every output row/column. Set all
   other matrix entries to zero. For the builder's RCR storage, B is stored as
   `[n, k0]`. Use raw FP4 code `0x2` and FP8 E4M3 code `0x38` for +1.
3. Set B scale selector to 0 and every B scale byte to E8M0 unity, `0x7f`.
4. Fill A's scale bytes with one raw code from the table. Compile variants with
   A selectors 0, 1, and 2, changing only this selector. Keep the raw input bytes,
   dimensions, and B selector identical across these variants.
5. Run first with `k0 = 0` and SCALE. Each of the 256 output elements should
   equal the decoded A scale: there is exactly one nonzero product.
6. Compare with literal expected values, without calling the production or
   verifier scale decoder. Record all outputs, including unexpected values.

Initialize output memory to a NaN sentinel and fail missing stores. Distinguish
compilation, module load, launch, and numeric failures. Use a subprocess timeout
for each probe. A failure in the mixed-format case alone cannot distinguish
scale-format support from matrix-pair restrictions; use the FP4 control below.

## Literal predictions

| Raw A scale | Unsigned E5M3 (bias 15) | E4M3 (bias 7) | E8M0 | Scale ignored |
| --- | ---: | ---: | ---: | ---: |
| `0x3c` | `3/512` | `1.5` | `2**-67` | `1` |
| `0x7c` | `1.5` | `384` | `0.125` | `1` |
| `0x84` | `3` | `-1/128` | `32` | `1` |

The first two rows distinguish the principal hypotheses using positive normal
values. The last row probes use of bit 7 as an exponent bit under E5M3; its
E4M3 value assumes ordinary signed decoding. Negative-scale handling may impose
additional restrictions, so an E4M3 mismatch on that row is not by itself a
failure of positive E4M3 support. Record any clamping or other behavior.

For an E5M2 confusion control, ordinary E5M2 decodes `0x3c` as 1 and `0x7c` as
positive infinity, also distinct from the E5M3 hypothesis. Do not use infinity
as an expected pass value in the main finite-value test.

## Controls and follow-up

- Verify unity separately for selectors 0/1/2 with raw bytes `0x7f`/`0x78`/
  `0x38`. These controls do not replace the fixed-byte discrimination cases.
- Inspect the emitted LLVM/HIP arguments and decoded instruction fields/raw
  instruction bytes. Confirm selector changes survive compilation to SCALE or
  SCALE16. A compiler remapping must not be attributed to hardware semantics.
- Swap A and B: put FP4 and the tested scale on B, leaving FP8 A with E8M0 unity.
- Repeat for SCALE16. Move `k0` through 0, 15, 16, 31, 32, 63, 64, 95, 96, and
  127 to cover scale-group and lane-half boundaries. Uniform scales isolate
  decoding; then vary only the active group's scale to check routing.
- Run FP4 x FP4 with matching selectors on A and B. Keep the untested operand's
  scale at that selector's unity encoding. Since unity bytes differ between
  selectors, report this as a separate control from the fixed-buffer experiment.
- Validate with COMGR and Python-generated HIP, using the same literal inputs.
  Require Python/C++ LLVM identity where the C++ backend is available. Do not
  infer C++ HIP compilation from Python HIP success.
- After these discriminating cases pass, sweep candidate finite E5M3 codes
  0..254 with an independent host reference, including zero, subnormals, and
  high-exponent values. Treat code 255 and other special-value semantics as a
  separate experiment. Retain raw output bits for any deviation.

## Evidence and interpretation

Record source SHA and patch hash, resolved device target, compiler/COMGR/runtime
versions, compilation route, generated source/IR, HSACO hash, disassembly, raw
input bytes, selectors, active K position, raw output bits, and expected values.
Use a machine-readable row for each combination and separate stage statuses.

If selector 1 returns `3/512` and `1.5`, while selector 2 returns `1.5` and `384`
for the same raw bytes, the tested routes implement distinct numerical decoding.
Selector 1 returning 3 for `0x84` additionally supports unsigned E5M3 decoding.
Identical output vectors across selectors require checking instruction encoding
before concluding that the hardware ignores or aliases a selector.

Passing establishes numerical behavior only for the tested target, instruction,
matrix pairs, selectors, and code points. It does not prove arbitrary-input
rounding, all format combinations, NaN behavior, or performance. Keep these
results separate from the FP6 packing validation and architectural documentation.
