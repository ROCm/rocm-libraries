# D3ZJ NGL/NLL LCC Investigation — CMS-Side Verification

Investigation only. No production code changes. Evidence drawn from
`tensilelite/Tensile/Tests/unit/kernel_cms.s` and `kernel_default.s` (the real
emitted assembly produced by `_dump_carveout_assembly.py` for the canonical
TF32 4x4 TN kernel) plus `Tensile/KernelWriter.py`.

The d3zj count under investigation:

| body | SCmpEQI32 | SSubU32 (LoopCounterL) |
|------|-----------|------------------------|
| ML   | 1         | 1                      |
| ML-1 | 1         | 1                      |
| NGL  | 0         | 0                      |
| NLL  | 0         | 0                      |

## Q1 — Does kernel_cms.s contain SCmpEQI32 / SSubU32 sgprLoopCounterL anywhere?

**Yes, exactly once each, both inside the MAINLOOP macro definition.**

`kernel_cms.s:2184-2186`:

```
.if \useLoop == 1
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
.endif                                             // EndIf \useLoop == 1
```

`kernel_cms.s:2191-2193`:

```
.if \useLoop == 1
s_cmp_eq_i32 s[sgprLoopCounterL], 0x2              // counterL==2
.endif                                             // EndIf \useLoop == 1
```

These are the only two LCC ops in the file (count verified: 2 occurrences of
`s_sub_u32 s[sgprLoopCounterL]` plus `s_cmp_eq_i32 s[sgprLoopCounterL]`
combined). Both are textually inside the body of `.macro MAINLOOP ID, useGR=1,
usePLR=1, useGRInc=1, useLoop=1` (declared at `kernel_cms.s:1846`) and both are
guarded by `.if \useLoop == 1`.

## Q2 — Which body label do they land between?

Macro **definition** sits at `kernel_cms.s:1846-2194`. The macro is invoked
four times from labeled body regions:

| Site | Line | Invocation                       | useLoop | Region                                              |
|------|------|----------------------------------|---------|-----------------------------------------------------|
| 1    | 2196 | `MAINLOOP 0`                     | 1 (def) | ML — between `label_LoopBeginL_0:` (2195) and `label_LoopEndL:` (2198) |
| 2    | 2210 | `MAINLOOP 0, 0, 1, 1, 0`         | 0       | NGL — under `/* Ord. NoGlobalLoadLoop_1 - Begin */` (2207) |
| 3    | 2242 | `MAINLOOP 0, 0, 0, 0, 0`         | 0       | OptNLL — under `/* Opt. NoLoadLoop - Begin */` (2217) |
| 4    | 2441 | `MAINLOOP 0, 0, 0, 0, 0`         | 0       | OrdNLL — under `/* Ord. NoLoadLoop - Begin */` (2438) |

Only site 1 (ML) expands with `useLoop=1`, so the LCC pair is materialized
into a single instance — physically between `label_LoopBeginL_0:` and
`label_LoopEndL:`. Sites 2-4 expand with `useLoop=0` and the `.if/.endif`
guard suppresses both LCC ops at assembly time. Net: **NGL, OptNLL, and
OrdNLL bodies in `kernel_cms.s` contain zero LCC ops**.

For comparison, `kernel_default.s` carries the same accounting but emits the
LCC pair *directly* (not through a macro) at lines 2170-2171, between
`label_LoopEndL:` (2173) and `/* Ord. NoGlobalLoadLoop_1 - Begin */` (2178).
Same total count (2), same effective placement (post-ML close-loop), no LCC
inside NGL/NLL on either side.

## Q3 — Are NGL and NLL completely free of `sgprLoopCounterL` references in kernel_cms.s?

**No** — they read the counter, they just don't emit cmp/sub against it.

NGL/OrdNLL/OptNLL bodies still reference `sgprLoopCounterL` for boundary checks
related to the tail loop and final iteration cleanup. Examples in the
NLL/tail-loop region of `kernel_cms.s`:

- `2463: s_and_b32 s[sgprLoopCounterL], 31, s[sgprSizesSum+0]` — recompute
  remaining iterations for the tail loop.
- `2467: s_cmov_b32 s[sgprLoopCounterL], 0` — GSU adjustment.
- `2494: s_cmov_b32 s[sgprLoopCounterL], 0` — GSU adjustment.
- `2496: s_cmp_eq_u32 s[sgprLoopCounterL], 0` — branch to skip tail loop.
- `2650-2916` (range): many `v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]
  // check K index >= Size L` — per-thread tail-edge masking.

Distinguish: **"no LCC ops" (true)** vs **"no LoopCounterL touch" (false)**.
The bodies still consume the counter; they just don't iterate against it.

## Q4 — CMS macro construction trace

### What does `useLoop=0` at `KernelWriter.py:3134, 3137` control?

It controls **only LCC emission** — specifically the
`s_sub_u32 s[sgprLoopCounterL], …, 1` decrement and the
`s_cmp_eq_i32 s[sgprLoopCounterL], 0x2` end-of-iteration test.

Evidence: `\useLoop` appears in `kernel_cms.s` at exactly four sites
(grep verified): the macro signature (1846), the comment-line headers in
three NGL/NLL invocations (2209, 2241, 2440), and the two `.if/.endif`
guards around the sub/cmp pair (2184-2186, 2191-2193). Nothing else in
the macro body is gated by `\useLoop`. So `useLoop=0` is *exactly* "skip
the LCC pair," not "stop iterating" or "skip global reads" (those have
their own flags `useGR`, `usePLR`, `useGRInc`).

The CMS-side `_emitNoLoadLoopBodyCMSMacro` at
`KernelWriter.py:3102-3139` codifies this:

- NGLL path (line 3134-3135): emits comment `useLoop = 0` then
  `MAINLOOP 0, 0, 1, 1, 0` (useGR=0, usePLR=1, useGRInc=1, useLoop=0).
- non-NGLL (NLL) path (line 3137-3138): emits `MAINLOOP 0, 0, 0, 0, 0`
  (everything off including useLoop).

There is no `simdSpecDispatch` wrapper around either tail body (see the
docstring at 3102-3131); they are unrolled tail bodies that don't iterate.

### Where would LCC for NGL/NLL be added if they were treated as iterating bodies?

The mechanism already exists. Flipping the last positional arg from `0` to
`1` on either MAINLOOP invocation in `_emitNoLoadLoopBodyCMSMacro` would
cause the assembler's `.if \useLoop == 1` to admit both LCC instructions
into that body's expansion. No additional codegen changes required. The
fact that both call sites are coded with the same macro wrapper but
different `useLoop` argument is the explicit declarative statement: NGL/NLL
do not iterate, so they don't decrement/test the loop counter.

## Q5 — Verdict on the user's stated invariant

**(A)** — every iterating body has 1 LCC pair; NGL/NLL legitimately have
0 because they don't iterate.

Evidence:

1. The CMS macro emits LCC iff `useLoop == 1` (Q1, Q4).
2. NGL and both NLL invocations explicitly pass `useLoop = 0`
   (`KernelWriter.py:3135` and `:3138`); ML is the only invocation that
   uses the default `useLoop = 1` (`kernel_cms.s:2196`, no override).
3. NGL and NLL bodies are unrolled tails, reached via fall-through after the
   true unrolled loop closes at `label_LoopEndL:`. They execute exactly
   once each; there is no `s_cbranch_scc0` back-edge into them. By contrast,
   the ML body is bracketed by `label_LoopBeginL_0:` / `s_cbranch_scc0
   label_LoopBeginL_0` (`kernel_cms.s:2195, 2197`) — a real iterating
   loop that needs LCC to terminate.
4. The default-side `kernel_default.s` agrees: its NGL/NLL bodies contain
   zero LCC ops (verified by absence in lines 2178-3239), matching CMS.
   So both code paths model NGL/NLL the same way.

The d3zj 0+0 capture for NGL/NLL is therefore **not a capture defect** —
it correctly reflects the absence of LCC emission in both code paths.

The "ML-1 = 1, ML = 1" rows in the d3zj table reflect a single physical
LCC pair (one sub at mfmaIndex 46, one cmp at mfmaIndex 47) split across
two reported sub-iterations. There is exactly one logical LCC pair per
iterating body, materialized once per ML expansion.

## Q6 — Narrowed invariant

> **Every body whose `useLoop=1` flag is set in the CMS `MAINLOOP` macro
> invocation has exactly one LCC pair (1 SSubU32 + 1 SCmpEQI32 on
> `sgprLoopCounterL`); bodies invoked with `useLoop=0` have zero LCC ops.**

Predicate to consume in validation code:

- Source-of-truth predicate (declarative): the `useLoop` positional arg
  to the `MAINLOOP` MacroInstruction emitted from the CMS noLoadLoopBody
  dispatcher (`KernelWriter.py:3134-3138`) and the unrolled loop dispatcher.
  CMS bodies with `useLoop=1` must have one LCC pair; bodies with
  `useLoop=0` must have zero.
- Equivalent assembly-side predicate: a body label region whose contained
  `MAINLOOP` invocation has fewer than 5 args (defaults apply, useLoop=1)
  OR whose 5th arg is `1` should contain 1 LCC pair; bodies with the 5th
  arg explicitly `0` should contain 0.
- Cross-side equivalence on the default path: because `_noLoadLoopBodyDefault`
  emits NGL/NLL without the close-loop tail (LCC is emitted by the
  unrolled-loop close, not the body), default-side NGL/NLL likewise have 0
  LCC ops. Validation should expect parity.

User-stated invariant ("every iterating loop body has 1 LCC pair") was
correct in spirit but ambiguous about what counts as "iterating." NGL/NLL
are body-labelled but do not iterate (no back-edge); they are unrolled
tails. The narrowed form keys off the explicit `useLoop` flag rather than
the body label, which is the actual codegen predicate.
