# Architectural Principle: The Proxy Predicate

## 1. The Pattern Name

**Proxy Predicate** — a cheap, observable piece of state (a counter, flag, or
non-null pointer) is read at a decision site as if it meant a *richer*
precondition that it does not actually encode. The proxy and the real predicate
happen to agree today; nothing in the type system or the writer guarantees they
will continue to.

## 2. The Structural Flaw

The decision site asks one question ("is it safe to take this shortcut?") but
tests a different expression that answers a *narrower* question ("has something
been filled / initialized / allocated?"). The real precondition is a conjunction
of several conditions, and those conditions are established — or relied upon to
be no-ops — in *other* functions that never name the precondition they uphold.

The shape is:

- A **reader** branches on `proxy` and, on the strength of it, *skips* work
  (dispatch, re-upload, side effects).
- The **writers** that set `proxy` do so under conditions that *currently*
  imply the full precondition, but enforce only a strict subset of it.
- The link between the two is maintained by call-site discipline and the
  coincidental no-op-ness of the skipped work, not by construction.

The fragility is that the implication `proxy → safe` is *load-bearing* but lives
nowhere: it is the emergent product of several functions, none of which is
responsible for it.

## 3. Why It Is Dangerous

In a benchmarking / validation client the failure mode is the worst kind:
**silently wrong numbers that still look plausible**. The kernel runs, produces
output, validation may even pass against a reference that was prepared down the
*same* skipped path — so the bug self-conceals. There is no crash, no NaN, no
segfault to anchor a bisect.

It is dangerous specifically because the flaw is *non-local*. Auditing the
reader tells you nothing; the invariant it depends on is asserted by the absence
of a write somewhere else. A future, locally-reasonable edit to a writer (e.g.
"let `beginAsyncReset` fill slots during bounds-check mode for speed", or
"initialize batched inputs only on the slow path") breaks the implication while
every line at the reader still reads as correct. The flaw also compounds with
**async/overlapped execution**: when the proxy gates a GPU-side shortcut, the
window between proxy-set and proxy-consumed spans stream boundaries, so a stale
proxy yields a data race rather than a clean error. The honest fix is
structural — move the full precondition to the single writer — but the *tempting*
fix is a comment at the reader, which documents the invariant without making the
compiler or a test enforce it.

## 4. The Diagnostic Signal

- **A counter or bool read for a meaning broader than its name.** `if(count >
  0)` / `if(inited)` guarding a branch that *skips type dispatch or side
  effects*, where correctness needs more than "count > 0" / "inited". The
  `m_availableSlots > 0` fast path is the canonical tell.
- **The same multi-term condition spelled out in several places.** When
  `a && b && !c` appears verbatim in three functions that must stay in sync by
  hand, the real predicate has no name and no home — each copy is a chance for
  drift.
- **A comment that asserts a claim about *other* code.** "// safe because the
  ring is only active when bounds-check is Disable" at a site that does not
  itself check bounds-check mode. The comment is the precondition; the compiler
  never sees it.

## 5. The Structural Fix Pattern

Name the full precondition as a single predicate, and **move it to the writer
that sets the proxy**, so the proxy becomes invariant-carrying: the only code
that sets it has already proven the whole predicate. Then every reader inherits
the proof for free, and the reader's cheap test (`count > 0`) is backed by a
provable implication rather than a coincidence. Add a debug `assert(predicate)`
at the reader to convert any future drift into a loud test failure instead of a
silent wrong answer. Collapse the duplicated copies of the condition onto the
one named predicate so they cannot diverge. This turns *correct by convention*
into *correct by construction* without restructuring dispatch or paying hot-path
cost.

## 6. Instances in This Codebase to Investigate

- **`m_batchInit` (high confidence).** This *is* the same bug, already once
  bitten (the `BatchPointerReset_test` regression): a bool that meant "batch
  pointers uploaded for *some* problem" was read as "current for *this*
  problem". The current fix resets it in `preProblem`; verify every path that
  *consumes* batch pointers is gated by it and that no writer sets it without
  having actually uploaded for the active problem.
- **`m_altSlotsReady` warm-path in `beginAsyncReset` (high confidence).** When
  set, the DMA is skipped on the premise "A/B/C read-only, D fully
  overwritten." That premise is a precondition the flag does not encode; a
  kernel/op that reads stale D (beta-only, or future fused epilogue) would
  break it silently. Investigate whether the flag should carry the "output
  fully overwritten next kernel" assumption explicitly.
- **`m_cpuInit && Disable && !problemDependent` in the `prepareCPUInputs`
  overloads (medium).** The exact multi-term condition appears twice (grouped +
  plain) and mirrors the GPU side. Prime candidate for folding onto a named
  predicate so CPU and GPU fast-path eligibility cannot drift apart.
- **`preSolution` MX re-init gated on `!m_gpuPtrs.empty()` (medium).** Non-empty
  is used as a proxy for "GPU inputs are initialized and consistent for the
  current problem". Confirm emptiness genuinely implies that, especially across
  the ring's `advanceBuffer` pointer swaps.

## 7. Where This Does Not Apply

- **`m_activeIdx` / `(m_activeIdx + ...) % m_numActiveBuffers`.** The modular
  ring index correctly implies "valid slot" *by construction* — it is computed,
  not asserted, and cannot name a slot outside the allocated range. There is no
  hidden second meaning; the value *is* the precondition.
- **`SlotGuard`'s RAII swap.** The "active pointers are redirected" state is
  guaranteed by scope, not by a flag a later reader trusts. Construction
  establishes it and destruction restores it on every path including
  exceptions, so there is no proxy that can outlive its truth.
