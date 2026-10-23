# Lane: runtime integration (`IMPACT`)

Find changes that break the kernel ingestor at run time while the build stays green: a kernel's
argument list (its ABI) that no longer matches a copy, a dispatch predicate that drifts from the
kernel it selects, a descriptor or generic-target edit that changes what ships. The ingestor C++
compiles only with `HIPDNN_ENABLE_KERNEL_INGESTOR` ON, so judge the change as built with the
ingestor, rocKE and SDPA flags ON, whatever their defaults.

You judge; two scripts collect facts. `impact.py` reads git objects only. `impact_emit.py`
imports and runs rocKE code from the reviewed trees (a `git archive` extract, in a child process
whose whole process tree is killed at its timeout, with an environment reduced to system, temp,
locale and Python user-site variables, so credentials such as `GH_TOKEN` are not passed): the
same code execution as `RUN` building the head.

## Contracts

| Contract | Producer | Copies the change must keep in step |
|---|---|---|
| Kernel arguments (ABI) | rocKE builder `b.param(...)` sequence under `dnn-providers/hip-kernel-provider/rocke/library/kernels/` | static `KernelArgument` lists in `src/engines/kernel_ingestor_engine/packs/*.cpp`, literal pins in tests, descriptor `signature` arrays |
| Launch surface | Python grid, block, LDS and tile tables | the `cpp_mirror` files a `launch_surface:` entry names in `projects/hipdnn/tools/IngestorGenerator/configs/*.engine.yaml` |
| Applicability | Python spec validity and constants (scale range, tile legality) | native pack matchers and scores |
| Shipped variants | `*.kdp.json` (DVC-held for gfx950; the emitter regenerates its variant list with the generator's registry enumerator, so a changed DVC pointer stays `changed`: the file's other fields are not read) | loader, native packs |
| Generic targets | `projects/hipdnn/plugin_sdk/data/gpu_generic_targets.json` and `projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/GpuGenericTargets*` | packer and loader |
| Python symbols | rocKE and `hkp_pack` modules | other modules and tests that import them |

Test code, CMake, CI test-category lists and `requirements*.txt` are not yours: they are `outside`
for status (the rollup names how many files it left to `BUILD` and `DEPS`), but tests are still
searched as copies and readers, so a stale pin or a stale import is yours.

## Method

1. Collect the facts. The report covers base to head and, when the target branch moved past the
   merge base, the head merged into `$TARGET_TIP`, where contract breaks between two PRs that each
   pass alone show up. `DRIFT` builds and tests that merge; you compare its contracts statically:

   ```bash
   . ./env.sh; mkdir -p "$SCRATCH/IMPACT" lanes/IMPACT-evidence
   python3 "$SKILL/scripts/impact.py" report --round-dir . > lanes/IMPACT-evidence/report.md
   ```

   Sourcing `env.sh` moves to the round directory. Exit 1 means the script could not run: return
   `STOP:` with its message.
2. Run the emitter when any rocKE or `hkp_pack` Python file changed in code, including rows the
   report calls `inert`: the static reachability it uses (a module counts as shipped when a shipped
   builder imports it) misses a call through a dynamic lookup. The emitter builds every shipped
   variant at both sides (a few minutes, CPU only) and records each variant's argument list, IR
   digest and COMGR inputs (ISA name, codegen options, fresh-process compile):

   ```bash
   rm -f "$SCRATCH/IMPACT/emit.json"
   python3 "$SKILL/scripts/impact_emit.py" run --repo "$CLONE" --base "$BASE" --head "$HEAD" --out "$SCRATCH/IMPACT/emit.json"
   python3 "$SKILL/scripts/impact.py" report --round-dir . --emit "$SCRATCH/IMPACT/emit.json" > lanes/IMPACT-evidence/report-emit.md
   ```

   The emitter exiting 1 (it could not run, or a side timed out) leaves no proof: return `STOP:`
   with its message, and never fold an `emit.json` from an earlier run. `impact.py` refuses an
   emit file made for another base or head, or one it cannot parse. Record the flavor it used. A
   family listed under `failures` or a KDP under `not_inspected` leaves its rows `changed`: never
   read a failure as equal. An all-equal emit folds only the rows of modules a built family
   imports, and never a row on code that runs after lowering (`rocke/helpers/compile.py`,
   `rocke/runtime/`): the IR digest does not cover it.
3. Judge every row whose status is not `inert` or `outside`. Read both sides at `$HEAD` (and the
   merged tree when the row says `merged-only`) before filing.
4. End **Checked, no finding** with the rollup line the script printed, corrected by your
   judgment and saying why when it differs:
   `Runtime impact: BREAKS | NEEDS-DEVICE <arch: suite, ...> | NONE (<n> files inspected) | N/A (no surface file)`,
   keeping any `not judged here:` note the script added.

## From row to finding

| Status | Action |
|---|---|
| `mismatch` | A `source-claim` at `BLOCKING` ([../policy.md](../policy.md): an ABI break without a migration path). **Where**: the producer line and every disagreeing copy. **Proof**: the command and both quoted lists. A `merged-only` mismatch breaks the target once merged: **Fix** is to rebase and update the named copy. If the same mismatch exists at `$BASE`, it is pre-existing: `FUTURE WORK` at most |
| `changed`, dispatch | Read the C++ function and its Python counterpart. A divergence (a constant, a range, a tile set, a mask or layout rule that the other side does not share) is a `source-claim` at `IMPORTANT` or higher, quoting both sides. Consistent: no finding |
| `changed`, emitted code or arguments | Consistent with every copy, names and types: no finding. A row that says `types not compared` or `cannot compare` is not consistency: compare the types and sizes of both lists by hand. The variants changed on an architecture need a device run there |
| `changed`, descriptor or generic target | Check the loader's rules for the changed fields (`projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp`); a value the loader rejects or ignores is a finding. A membership change names the architectures whose archives change |
| `inert`, `outside` | Count them under **Checked, no finding**; list each `inert` reason once |

For every `changed` row without a finding, name the device run that would settle it: the ctest
entry (find it from the pack or kernel family name in the test sources and `ctest -N -R <regex>`
in `RUN`'s build when one exists), the architecture, and the flags. Put each one under **Approval
requests** and add `AUTH: requires <arch> <suite> with the ingestor flags ON` under **Handoffs**;
`AUTH` decides whether a run executed.

## Handoffs

- `API`: a changed declaration in a plugin_sdk header, or a `.fbs` change.
- `BUILD`: flag matrix, option defaults, test registration, install rules.
- `ROCK`, `AUTH`: whether CI builds and tests the change; you supply the required pairs only.
- `DRIFT`: a `merged-only` row, so the merged build and tests cover the copy you name; and the
  conflicted files of a merged view the report marks `conflicted`, which it does not analyse.
- `TESTB`: a contract with no test that would fail on the break you describe.

Known gaps: HIP-kind kernels (`ConvNative`, `PointwiseNative`) have their C++ argument lists
checked but not their HIP producers; a dynamic C++ signature (`Gfx1151WmmaAttentionNative`)
cannot be compared statically; one LLVM flavor is emitted per run, so an edit that changes only
another flavor's lowering reads `inert`. The symbol check sees a reader only through an import
of the defining module, `m.name` on it, or an attribute read in a file that names the class; it
cannot see `from m import *`, `getattr(m, "f")`, a dotted import read in full (`import a.b.m`,
then `a.b.m.f()`), or a module reached only through
`importlib.import_module(f"...{arch}...")` (the emitter covers that last one for shipped
variants, a bare `report` does not). A method removed from a base class and read only through a
subclass or an untyped receiver, in a file whose imports reach the class's module, gives a
`changed` row, not a `mismatch`: read the receivers. The emitter compares what goes into COMGR,
not the COMGR library that compiles it: a change to which comgr the packer loads is the
`BUILD` and `DEPS` lanes'. A rocKE C++ engine source (`rocke/platform/cpp/`) reads `inert` only
while no provider build file outside rocKE links the engine and the packer pins the Python
backend; the row says so, so check that both still hold for the change you review.
