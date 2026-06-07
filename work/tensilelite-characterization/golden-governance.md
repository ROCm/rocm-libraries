# Golden governance — TensileLite codegen characterization

How the characterization goldens are keyed, when a change is a signal vs noise, and the
retention policy. Companion to `PLAN-CODEGEN-WORKFLOW.md` / `BASELINE-AND-PROGRESS.md`.

## What a golden is (and why)

Each codegen-emit test stores an **order-invariant `{basename, err}` digest** (syrupy
`__snapshots__/*.ambr`), **not** a hash of the full assembly text. Rationale (carried from
PLAN-80 P2): the emitter's assembly text is **order-coupled through process-global rocisa
scheduler state**, so a full-text hash is flaky across runs/process layouts. The
`{basename, err}` digest (sorted by basename) is stable across runs while still pinning:

- **which kernels** the input derives + emits (the `Cijk_…` basenames), and
- **the emit outcome** per kernel (`err==0` success, or a pinned non-zero reject).

`canonicalize_asm` (strip register/addr/temp-label numbering) is available for any test that
does assert on assembly text, but the digest goldens are the default.

## Keying: (architecture, compiler version)

Goldens are produced **in-container** (root-owned) against the `tl-char` toolchain. A golden is
conceptually keyed by **(architecture, compiler version)**:

- **Architecture** — explicit in the config/test (`gfx908/90a/942/950/1100/1201/1250`) and in
  the emitted `.amdgcn_target`.
- **Compiler version** — the `amdclang++` baked into `tl-char` (the capability probe
  `makeIsaInfoMap(isa, cxx)` is the single toolchain↔snapshot coupling point, per
  `toolchain-inventory.md`). The current goldens were recorded against that one toolchain.

## Stable vs evolving architectures

- **Stable archs** (gfx908, gfx90a, gfx942): one golden per (arch, config). A **golden change
  is a signal** — treat a digest diff as a **suspected compiler/codegen regression** and
  investigate before re-recording. Do **not** blanket `--snapshot-update`.
- **Evolving archs** (gfx950, gfx1250 — newest, codegen still churning): expect digest churn
  across compiler generations. Keep up to **N (default 2) compiler generations** of the golden
  side-by-side; retire the oldest as releases settle. When the arch stabilizes, collapse to the
  stable-arch policy (one golden).

## Recording / updating procedure (controlled)

1. Record only in the dedicated step: `pytest … --snapshot-update <node>` **in-container**.
2. Immediately re-run the node **without** `--snapshot-update` **twice** — must be byte-identical
   (two-run stability). Churn ⇒ fix via `canonicalize_asm` / the `{basename,err}` digest, **not**
   by re-recording.
3. A digest change on a **stable arch** requires a written justification (compiler bump? real
   codegen change?) in `DECISIONS.md` before the new golden is committed.

## Caveat — coverage measurement is not part of the golden

Codegen runs through multiprocessing workers, so a test's **coverage line attribution** can jitter
run-to-run under coverage.py `concurrency=multiprocessing` (see `BASELINE-AND-PROGRESS.md`). This
affects **coverage measurement only**, never the `{basename, err}` golden (which is deterministic).
The whole-project gate is therefore taken from the **combined full-suite run**, not per-test line
counts. Per-test two-run *golden* stability is still required; per-test two-run *coverage* identity
is **not** (it would wrongly reject good tests — this was the R7 method fix).
