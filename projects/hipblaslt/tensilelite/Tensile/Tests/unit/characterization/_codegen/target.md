# `_codegen` — CPU-only assembly-emit harness (Phase G0)

**What:** the shared mechanism that lets the characterization suites cover the
codegen surface (`KernelWriterAssembly`, `KernelWriter`, `Components/*`, `Asm*`)
**without a GPU**. See `../../../../../work/tensilelite-characterization/PLAN-80.md`.

**How:** `codegen_harness.py` drives the real emit pipeline —
`parseLibraryLogicFile` → `Solution` → `generateKernelObjectsFromSolutions` →
`KernelWriterAssembly.getSourceFileString` → assembly text. Only the *emit* runs
(deterministic, CPU-only); assembling to a code object and running on hardware
are out of scope.

**Determinism:** the emitter tags labels with a random 16-char suffix — the only
run-to-run variation. `canonicalize_asm()` maps each distinct suffix to a stable
`_LBL{n}` id (preserving label↔reference pairing), making the text byte-stable.
Goldens snapshot a compact **digest** (deterministic kernel name + emit return
code + line count + sha256 of canonicalized text) rather than ~200 KB of raw asm.

**Proven (Phase G0):** one gfx942 HSS kernel alone covers
`KernelWriterAssembly` 35.97% / `KernelWriter` 42.22% / `GlobalWriteBatch`
34.90%. Phase 1+ widen the config matrix (dtypes × schedules × ISAs) to lift
these toward the 80% goal. Diverse, valid multi-ISA logic inputs are available
in-checkout under
`projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/`.

**Add-only:** `conftest.py` at the `characterization/` level only inserts the
`_codegen` dir on `sys.path` and exposes read-only session fixtures; it changes
no existing behavior.
