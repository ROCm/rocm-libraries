# StreamK modes 4 and 5 cannot be assembled on any RDNA target

**Files:** `projects/hipblaslt/tensilelite/rocisa/rocisa/include/instruction/mem.hpp`
**Patch:** `sk_modes/artifacts/rocisa_glc_fix.patch`
**Verified on:** gfx1100 (Radeon RX 7900 XTX), ROCm 7.2

---

## Problem

`StreamK: 4` (Dynamic) and `StreamK: 5` (Hybrid) pass solution validation but **fail at the
assembler** on gfx1100:

```
error: not a valid operand.
global_atomic_inc_u32 v3, v1, v2, s[46:47] th:TH_ATOMIC_RETURN
                                           ^
# Actual Solutions: 12 / 12 after SolutionStructs
# Actual Solutions:  3 / 12 after KernelWriter      <- only the SK3 kernels survive
```

They are not rejected — nothing in `Solution.py` refuses them on this architecture. They
simply cannot be built, so no RDNA target has ever been able to ship an SK4 or SK5 kernel.

## Root cause

`GlobalAtomicIncU32Saddr::toString()` appends the returning-atomic modifier
**unconditionally**, in gfx12-only spelling:

```cpp
// mem.hpp, GlobalAtomicIncU32Saddr::toString()
std::string kStr = instStr + " " + getArgStr();
if(modifier)
    kStr += modifier->toString();
kStr += " th:TH_ATOMIC_RETURN";        // every architecture, always
```

That instruction is reached from `StreamK._fetchNextWorkItem`
(`Tensile/Components/StreamK.py:336`), which branches on `asmCaps["HasSAtomic"]`:

```python
if writer.states.asmCaps["HasSAtomic"]:
    module.add(SAtomicInc(...)); return module     # scalar path
# else: returning VECTOR atomic — written for gfx12/gfx1250
```

`HasSAtomic` probes `s_atomic_dec`. RDNA has no scalar atomics, so gfx11 takes the else
branch — which was written assuming "no scalar atomics ⇒ gfx12". gfx11 falsifies that:

| | `s_atomic_dec` | `th:TH_ATOMIC_RETURN` | result |
|---|---|---|---|
| gfx90a, gfx942 (CDNA) | yes | no | scalar path — fine |
| **gfx1100 (RDNA3)** | **no** | **no** | **falls through both — cannot assemble** |
| gfx1201 (RDNA4) | no | yes | vector path — fine |

gfx1100 is the only probed architecture failing **both** tests.

**This is a spelling problem, not a hardware limitation.** gfx1100 supports the exact
operation under the older name:

```asm
global_atomic_inc_u32 v3, v[0:1], v2, off      glc     ; assembles on gfx1100
global_atomic_inc_u32 v3, v1,     v2, s[46:47] glc     ; SADDR form also fine
```

On an atomic, `glc` means "return the pre-operation value" — precisely what
`th:TH_ATOMIC_RETURN` names.

## Fix

Emit the modifier the target actually accepts. Both the capability check and the helper
already exist in the codebase:

```cpp
if(rocIsa::getInstance().getAsmCaps()["HasTHModifier"])
    kStr += " th:TH_ATOMIC_RETURN";      // gfx12+
else
    kStr += " " + getGlcBitName();       // gfx11 -> "glc", CDNA -> "sc0"
```

`getGlcBitName()` (`base.hpp:314`) already resolves `glc` vs `sc0` per architecture, and
`HasTHModifier` (`hardware_caps.hpp:450`) is the established gate for this split — it is
what `Components/NonTemporal.py` uses for the same purpose.

## Validation

Built and benchmarked `StreamK: [3, 4, 5]` on gfx1100, TN HHS (fp16 in/out, fp32 compute),
24 shapes × 2 geometries, `NumElementsToValidate: -1`:

| | before | after |
|---|---|---|
| assembler errors | **6** | **0** |
| solutions surviving KernelWriter | 3 / 12 | 6 / 6 |
| SK3 validation | 48/48 PASSED | 48/48 PASSED |
| **SK4 validation** | *(could not build)* | **48/48 PASSED** |
| **SK5 validation** | *(could not build)* | **48/48 PASSED** |

Across the full campaign: **2 064 validated SK4 measurements, 0 failures.**

## Blast radius

Minimal, and it cannot affect anything currently shipping:

- `StreamK: 4` and `StreamK: 5` appear in **no** shipped logic file on any architecture
  (anchored count across all of `Logic/`: only `StreamK: 3`, plus `StreamK: 5` on gfx950).
- gfx1100 has no scalar atomics, so no existing gfx11 kernel reaches this code path.
- On gfx12 the behaviour is byte-identical: `HasTHModifier` is true, so the `th:` branch is
  taken exactly as before.
- GSU's analogous no-scalar-atomic fallback uses a different instruction
  (`FlatAtomicDecU32`) and is untouched.

**Not verified:** gfx1250 hardware. The logic is unchanged there by construction, but it has
not been run.

## The same defect breaks GSU too — second site, now fixed and validated

Checked against `origin/develop` @ `a9b7332a925` on 2026-08-22. The patch still applies
cleanly and targets the right class — `GlobalAtomicIncU32Saddr`, which is exactly what
`Components/StreamK.py:363` emits. That part is confirmed.

But `mem.hpp` contains **a second unconditional emission of the same gfx12-only modifier**,
in a different instruction:

```cpp
// struct FlatAtomicDecU32 : public FLATStoreInstruction   (develop line 2221)
std::string kStr = instStr + " " + getArgStr();
kStr += " th:TH_ATOMIC_RETURN";        // unconditional, same defect as the original
```

The body of this document states that GSU's analogous fallback "uses a different instruction
(`FlatAtomicDecU32`) and is untouched". That is true of *this patch*, but it should not be
read as "and is therefore safe". `Components/GSU.py:1151` and `:1551` emit
`FlatAtomicDecU32` **without any scalar-atomic capability gate** — the modifiers there key on
`DefaultScopeIsCULocal` and `RequiresXCntForVolatileVMEM`, neither of which is the
gfx11-vs-gfx12 spelling split that caused this bug.

**This is assembler-verified, not inferred.** Three checks on gfx1100 with ROCm 7.2's
`clang`:

| check | result | consequence |
|---|---|---|
| `s_store_b32` / `s_store_dword` | **rejected** | `HasScalarStore` is **false**, so `GSU.py:1121` takes the `else` branch and *must* emit `FlatAtomicDecU32` |
| `flat_atomic_dec_u32 ... th:TH_ATOMIC_RETURN` | **rejected** | that emission cannot assemble on gfx11 |
| `flat_atomic_dec_u32 ... glc` | **accepted** | the same `getGlcBitName()` fix applies verbatim |

**It is dormant on navi31 today, not safe.** `lastGsuWgBusyWaiting` is reached only under
`GlobalSplitUAlgorithm: MultipleBufferSingleKernel`, and shipped navi31 logic is **2240
solutions, all `MultipleBuffer`** (GSU 0/1/2/4). That is why GSU works on gfx1100 despite
the defect. Enable MBSK on any gfx11 part and it fires immediately.

*(Naming trap: built kernels carry a `_GSUAMB_` token, which abbreviates GSU Algorithm
**MultipleBuffer** — not MBSK. Do not read it as evidence that MBSK ships.)*

**Reproduced end-to-end, then fixed and validated.** Config
`sk_modes/configs/P20_mbsk_probe.yaml` — TN HHS, `GlobalSplitU: 4`,
`GlobalSplitUAlgorithm: MultipleBufferSingleKernel`, `StreamK: 0` (the two are mutually
exclusive: `Solution.py` rejects "Either GSU or StreamK must be enabled"):

```
error: not a valid operand.
flat_atomic_dec_u32 v97, v[94:95], v96 th:TH_ATOMIC_RETURN
                                       ^
```

Same signature as the StreamK failure, in an unrelated subsystem — and note this was on a
tree **with the `GlobalAtomicIncU32Saddr` fix already applied**, which is what proves the
two sites are independent.

Applying the identical guard to `FlatAtomicDecU32::toString()`:

| | before | after |
|---|---|---|
| exit code | 1 | **0** |
| assembler errors | 1 | **0** |
| solutions surviving KernelWriter | 0 / 1 | **1 / 1** |
| validation | *could not build* | **PASSED** |

**So this patch now covers both sites**, and `git apply --check` confirms it still applies to
`origin/develop` @ `a9b7332a925` unchanged. The consequence for the PR framing: this is not a
StreamK bug — it is a `mem.hpp` bug that independently broke **StreamK modes 4/5 and GSU
MultipleBufferSingleKernel** on every RDNA target.

Reproduce:

```bash
cd projects/hipblaslt/tensilelite
./Tensile/bin/Tensile ~/sk_modes/configs/P20_mbsk_probe.yaml /tmp/mbsk_out
grep "not a valid operand" /tmp/mbsk_out/**/*.log     # before the fix
```

### Regression check — the second guard does not disturb the first

Re-ran the original StreamK probe (`configs/probe_sk345.yaml`) with **both** guards in place:

| | result |
|---|---|
| exit code | **0** |
| assembler errors | **0** |
| solutions | 12/12 after SolutionStructs, 9/12 after KernelWriter |
| validation | **18 PASSED, 0 FAILED** |
| SK modes built | **SK3 ×7, SK4 ×7, SK5 ×7** |

The 3 solutions dropped at KernelWriter are the documented VGPR-budget rejections
(MT128x128 needs 266 against a 256 budget in this minimal ProblemType), not assembler
failures.

### Completeness check for reviewers

Every emission in `mem.hpp` is now guarded — there is no third site:

```
line 2230  guarded    (GlobalAtomicIncU32Saddr — StreamK)
line 2360  guarded    (FlatAtomicDecU32        — GSU MBSK)
```

The check is `kStr += " th:TH_ATOMIC_RETURN"` not preceded within a few lines by
`HasTHModifier`. Worth re-running on any branch this is ported to, since the second site was
added after the first was written.

Worth surfacing because it makes the defect a *pattern* rather than a one-off: any
`toString()` in this file that appends `th:` unconditionally is wrong on gfx11 and CDNA. A
grep for `th:TH_ATOMIC_RETURN` outside a `HasTHModifier` guard is the check.

## Reproduction

```bash
cd projects/hipblaslt/tensilelite
# a config forking StreamK: [3, 4, 5] on any gfx1100 HHS solution
./Tensile/bin/Tensile <config>.yaml <out>
grep "not a valid operand" <out>/../*.log
```

Any `StreamK: [4]` or `[5]` config targeting gfx1100 reproduces it; the geometry is
irrelevant.

---

## Re-verified 2026-08-23 against `origin/develop` @ `dab5e862a64`

Develop has moved ~150 commits since this was written. Both the defect and the fix still hold:

| check | result |
|---|---|
| unconditional `kStr += " th:TH_ATOMIC_RETURN"` emissions | **2** (`GlobalAtomicIncU32Saddr`, `FlatAtomicDecU32`) |
| occurrences of `HasTHModifier` in `mem.hpp` | **0** — neither emission is guarded |
| `rocisa_glc_fix.patch` applies to develop | **yes, cleanly** |

So the two-site fix is still exactly what is needed, and has not bit-rotted.
