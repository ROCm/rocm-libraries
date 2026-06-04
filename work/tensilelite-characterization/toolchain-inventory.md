# Toolchain-Dependency Inventory (codegen) — First Pass

Findings §9. Purpose: classify notable codegen functions/boundaries in the TensileLite
assembly emission path as either

- **our-logic-shapes-it → snapshot**: output is determined by Tensile's own Python logic
  (given fixed inputs incl. the capability tables). Safe to characterize with snapshot tests.
- **compiler-shapes-it → mock/skip**: behavior or output depends on the *external toolchain*
  (the C++/HIP compiler, the GPU assembler, the ROCm version, or `rocisa`'s compiler probe).
  Must be mocked, pinned, or skipped to keep tests hermetic.

Scope inspected: `KernelWriterAssembly.py`, `KernelWriter.py`, `Components/*`, `Asm*.py`,
plus the caps/ISA plumbing in `Common/Capabilities.py`, `Common/Types.py`, `SolutionStructs/Solution.py`.

Paths are relative to `projects/hipblaslt/tensilelite/Tensile/`.

## Key architectural finding

The codegen emission path itself does **not** shell out to a compiler or assembler. The
single real toolchain dependency is the **capability probe**: `rocisa.rocIsa.getInstance().init(isa, cxxCompiler, ...)`
in `Common/Capabilities.py::makeIsaInfoMap`, which inspects `cxxCompiler` to populate the
`asmCaps` / `archCaps` / `regCaps` / `asmBugs` dicts. Everything downstream in
`KernelWriterAssembly.py` / `KernelWriter.py` reads those dicts (`self.states.asmCaps[...]`,
`self.states.archCaps[...]`) as plain data. **If the caps dicts are pinned (snapshot/fixture),
the entire emission path becomes our-logic and is snapshot-able.** This is the lever that
makes B1 (union of per-ISA snapshot suites) viable — the existing char suites already pin
per-ISA caps and the union climbs KWA 52%→59% / KW 57%→66% / TOTAL 54%→62%.

Actual subprocess/compiler invocations live **outside** the codegen module, in build/run
helpers: `ClientWriter.py`, `ClientExecutable.py`, `ParallelExecution.py`,
`GenerateSummations.py`, `TensileBenchmarkCluster.py`. These are out of scope for codegen
coverage and should be skipped, not snapshotted.

## Classification table

| Location | Function / boundary | Toolchain coupling | Classification | Confidence | Notes |
|---|---|---|---|---|---|
| `Common/Capabilities.py:30` | `makeIsaInfoMap(targetIsas, cxxCompiler)` | Calls `rocIsa.getInstance().init(v, cxxCompiler, False)` — probes the actual compiler | **mock/skip** | High | THE chokepoint. Replace with a pinned fixture of `IsaInfo` per ISA so downstream is deterministic. |
| `KernelWriterAssembly.py` (~3000 sites) | reads `self.states.asmCaps[...]` / `self.states.archCaps[...]` (e.g. `HasWMMA_V3`, `HasMFMA_f8f6f4`, `HasVgprMSB`, `s_sub_u64`, `HasSWMMAC_gfx1250`, `CrosslaneWait`, `SgprPreloadPad`, `WorkGroupIdFromTTM`) | Indirect: only via the caps dicts above | **snapshot** | High | Pure branching on pinned data. This is where the per-ISA snapshot suites earn their coverage. |
| `SolutionStructs/Solution.py` (many) | `isaInfoMap[isa].asmCaps[...]` / `.archCaps[...]` in `assignDerivedParameters`, `_deriveAndValidateMXScaleLayoutAndTransport`, `isLDSTrEnabled`, etc. | Indirect via caps dicts | **snapshot** | High | Solution derivation is our logic given pinned caps; already partly covered by SolutionDerivationSweep. |
| `KernelWriterAssembly.py:140` | `_getCustomKernelSource` | Reads `self.assembler.rocm_version` and gates on `major>=6 and patch>=32650`; also opens a `.s` file from disk | **mock/skip** (version gate) | Medium | The ROCm-version branch is toolchain-shaped; pin `assembler.rocm_version` to exercise both arms. The file read is fixturable. |
| `KernelWriterAssembly.py:16373` / `Components/GlobalWriteBatch.py` | `globalWriteBatch(...)` passes `self.assembler.version` into the write component | Carries assembler version into emission | **snapshot w/ pinned version** | Medium | Output depends on `assembler.version`; pin it as part of the fixture, then snapshot. |
| `KernelWriterAssembly.py:157` | `self.states.version = kernel["ISA"]` | None directly (ISA comes from kernel dict) | **snapshot** | High | ISA is an input parameter, not probed here. |
| `KernelWriterActivationFunction.py:90` | `getInlineAsm` → `tf.init(isa, self.cxxCompiler)` / `tf.setKernel(...)` | Re-invokes the rocisa compiler probe per arch | **mock/skip** | Medium | Same probe as `makeIsaInfoMap`. If a global pinned `rocIsa` instance is already initialized (via fixture), `tf.isInit()` short-circuits; otherwise it probes. Ensure the fixture inits first. |
| `KernelWriter.py:6045` | `getKernelSource` cmpasm path writing `cmpasm/orig/*.s` and `cmpasm/st/*.s` | Writes assembly source to disk for external diffing | **skip** (or snapshot the string only) | Medium | The emitted *string* is our logic (snapshot-able); the disk side-effect / external compare is not codegen-under-test. |
| `KernelWriter.py:9640` | `getSourceFileString` (abstract; KWA override) | Doc says side-effects: writes `.s`, object file, code object, byte-array script — i.e. assembles | **mock/skip** | Medium | The base returns the source string (our logic → snapshot); the *assembled* object/code-object artifacts require the GPU assembler → skip those branches. Need to confirm exactly where KWA assembles vs. just emits text. UNCERTAIN: locate the assemble call (not found inline in KWA at first pass; likely delegated to a build helper). |
| `KernelWriter.py:9660` | `setRocIsa(data, outOptions)` → `rocIsa.getInstance().setData/.setOutputOptions` | Configures the rocisa singleton | **mock/skip** (setup) | Medium | Fixture setup, not an assertion target. Drive it from the test fixture with pinned data. |
| `Components/MAC_*.py` (MAC_F16, MAC_F32, MAC_BF16_HPA, MAC_I8_HPA, ...) | MFMA/MAC instruction emission | Branch on `asmCaps`/dtype only | **snapshot** | Medium | Instruction text is our logic given pinned caps; uncertain whether any path emits arch-specific encodings that vary by assembler version — mark uncertain. |
| `Components/*` (GlobalWriteBatch, LocalRead, LSU, GSU, StreamK, Signature, etc.) | kernel-section emitters | Indirect via caps/version dicts | **snapshot** | Medium | First-pass assumption: pure given pinned fixture. Some (Signature) embed version/ROCm strings → pin those. |
| `Asm*.py` (AsmAddressCalculation, AsmMemoryHelpers, AsmMemoryInstruction, AsmStoreState) | address/memory instruction helpers | None obvious; pure arithmetic + container builders | **snapshot** | Medium | No subprocess/compiler refs found; appear to be pure helpers. |
| `ClientWriter.py`, `ClientExecutable.py`, `ParallelExecution.py`, `GenerateSummations.py`, `TensileBenchmarkCluster.py` | `subprocess.run/Popen/check_call` to build & launch clients | Direct: invokes external build/assembler/GPU | **skip** (out of codegen scope) | High | Not part of the KWA/KW emission path; do not target for codegen coverage. |

## Uncertainties to resolve in the next pass

1. **Exact assemble call site.** `getSourceFileString`'s docstring promises object/code-object
   side-effects, but no inline subprocess/assembler call was found in `KernelWriterAssembly.py`
   on first pass. Need to find whether KWA assembles directly or returns the string and a
   build helper assembles. Determines whether any KWA branch is genuinely compiler-shaped vs.
   all snapshot-able.
2. **Version-string leakage in Components** (e.g. `Signature.py`, `globalWriteBatch` via
   `assembler.version`). Confirm every place a version/ROCm string is embedded so the fixture
   pins them; otherwise snapshots become non-deterministic across toolchains.
3. **`rocisa` init ordering** for `KernelWriterActivationFunction.getInlineAsm` — confirm the
   char fixtures init the singleton before this runs so `tf.isInit()` short-circuits the probe.
4. **MAC/MFMA arch-encoding stability** across assembler versions — whether snapshot text is
   stable or needs caps-version pinning beyond the existing dicts.

## B1/B2 recommendation (for PLAN-CODEGEN-WORKFLOW.md — not edited here)

**B1.** The combine unioned cleanly and the result is well above the best single shard
(KWA 52.00%→59.18%, KW 57.27%→65.93%, TOTAL 53.88%→61.59%). The per-shard ceiling is an
artifact of single-ISA execution, not a structural cap — union bypasses it. Because the only
real toolchain coupling is the `rocisa`/compiler caps probe (a single chokepoint that the
existing char fixtures already pin per ISA), the emission path is overwhelmingly
our-logic-shaped and snapshot-able. Recommend the B1 path: keep growing the per-ISA snapshot
suite set and combine, pinning the caps fixture + ROCm/assembler version, and mock/skip only
the handful of compiler-probe and subprocess boundaries listed above.
