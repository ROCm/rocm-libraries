# Ingestor impact reference

What the two scripts check, what their rows mean, and how to judge them. [SKILL.md](SKILL.md) runs
them on their own; the `IMPACT` lane of
[hipdnn-deep-review](../hipdnn-deep-review/lanes/runtime-impact.md) runs them inside a review and
maps the judged rows to findings.

The kernel ingestor C++ compiles only with `HIPDNN_ENABLE_KERNEL_INGESTOR` ON, so judge a change as
built with the ingestor, rocKE and SDPA flags ON, whatever their defaults. A default build compiles
the ingestor out, and several contracts below are restated by hand with no compiler check, so a
change that breaks them leaves the default build green.

## Contracts

| Contract | Producer | Copies the change must keep in step |
|---|---|---|
| Kernel arguments (ABI) | rocKE builder `b.param(...)` sequence under `dnn-providers/hip-kernel-provider/rocke/library/kernels/` | static `KernelArgument` lists in `src/engines/kernel_ingestor_engine/packs/*.cpp`, literal pins in tests, descriptor `signature` arrays |
| Launch surface | Python grid, block, LDS and tile tables | the `cpp_mirror` files a `launch_surface:` entry names in `projects/hipdnn/tools/IngestorGenerator/configs/*.engine.yaml` |
| Applicability | Python spec validity and constants (scale range, tile legality) | native pack matchers and scores |
| Shipped variants | `*.kdp.json` (DVC-held for gfx950; the emitter regenerates its variant list with the generator's registry enumerator, so a changed DVC pointer stays `changed`: the file's other fields are not read) | loader, native packs |
| Generic targets | `projects/hipdnn/plugin_sdk/data/gpu_generic_targets.json` and `projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/GpuGenericTargets*` | packer and loader |
| Python symbols | rocKE and `hkp_pack` modules | other modules and tests that import them |

Test code, CMake, CI test-category lists and `requirements*.txt` are not judged here: they are
`outside` for status, but tests are still searched as copies and readers, so a stale pin or a stale
import is reported.

How the copies are paired:

- A `launch_surface` entry's `python_source` symbol resolves as `launch_surface.py --check`
  resolves it: any function of that name anywhere in the module (`ast.walk`), a method, a nested
  function or a def under `if`, `try` or `with` included. Every function of that name is compared,
  and the symbol is reported missing only when none exists.
- A declared `kernargs` pair compares each static C++ list in its `cpp_mirror` with the builder
  whose `b.param` list it matches, searched in the surface's module and the manifest's own
  `source` module, the declared function first. A multi-launch pack holds one list per launch (a
  segment and a reduce kernel), and a declared function may be a `SignatureBuilder` factory with no
  `b.param` list; the row then names the builder it was paired with. A list that matches no
  builder but is related to one (it shares at least two names, and at least half of its own) is
  compared with the related builder it shares most names with, and is a `mismatch`. An unrelated
  list is skipped; when no list of the mirror was compared, one `changed` row says `cannot
  compare`.
- A copy that holds the old list is `changed`, not `mismatch`, when it still matches the new
  list with the new conditional parameters left out: variants that do not enable them pass that
  list. The variants that do enable them are what the emitter or a device run settles.

## The scripts

- `scripts/impact.py` reads git objects only (`cat-file`, `diff`, `ls-tree`, `grep`,
  `merge-tree`); it writes no ref, index or worktree. Its subcommands are `files`, `kernargs`,
  `symbols`, `surfaces`, `data` and `report`; `report` runs all of them and ends in the rollup line.
- `scripts/impact_emit.py` imports and runs rocKE code from the two trees: a `git archive` extract
  per side, in a child process whose whole process tree is killed at its timeout, with an
  environment reduced to system, temp, locale and Python user-site variables, so credentials such
  as `GH_TOKEN` are not passed. That is the same code execution as building the head. It builds
  every shipped variant at both sides on the CPU (no GPU, comgr or network): a few minutes, with
  the two sides on two concurrent workers.

Both exit 0 when the analysis ran, whatever it found, and 1 with one line (`impact: <reason>` or
`impact_emit: <reason>`) on stderr when they could not run. An exit 1 is never a verdict.

## Views

The report covers base to head and, when `--target-tip` differs from the base, the head merged into
that tip (`git merge-tree --write-tree`). A row that holds only in the merged view is marked
`merged-only`: two PRs that each pass alone, where the other one landed a copy this diff cannot
see. A conflicted merge is still analysed except for its conflicted files, which one `changed` row
stands in for when one of them is on the surface or a test.

## Row statuses and the rollup

| Status | Meaning |
|---|---|
| `mismatch` | A producer and a copy disagree, or a removed symbol still has a reader |
| `changed` | A contract side changed, or static analysis cannot settle it; a surface file the script does not understand is always `changed` |
| `inert` | Touched, and provably without runtime effect; the row gives the reason |
| `outside` | Not on the surface; counted, never dropped |

The last line of `report` is the rollup:
`Runtime impact: BREAKS | NEEDS-DEVICE <arch: suite, ...> | NONE (<n> files inspected) | N/A (no surface file)`.
`BREAKS` is any `mismatch`, `NEEDS-DEVICE` any `changed`, `NONE` all surface rows `inert`, `N/A` no
surface file. An empty diff is an error, never `NONE`. A `not judged here:` note counts the files
left to other checks: `BUILD` is CMake and CI test-category lists, `DEPS` is `requirements*.txt`
(named after the hipdnn-deep-review lanes that judge them).

## The emitter

Run it when any rocKE or `hkp_pack` Python file changed in code, including rows the report calls
`inert`: the static reachability it uses (a module counts as shipped when a shipped builder imports
it) misses a call through a dynamic lookup. It records each variant's argument list, IR digest and
COMGR inputs (ISA name, codegen options, fresh-process compile); a builder that returns a
`KernelModule` is lowered kernel by kernel, as the packer does, and each companion launch's
argument list is recorded and compared too. `impact.py report --emit` folds the result into the
rows.

- The emitter exiting 1 (it could not run, or a side timed out) leaves no proof. Never fold an emit
  file from an earlier run. `impact.py` refuses an emit file made for another base or head, or one
  it cannot parse.
- Record the flavor it used. One LLVM flavor is emitted per run (`--flavor` picks another).
- A family listed under `failures` or a KDP under `not_inspected` leaves its rows `changed`: never
  read a failure as equal.
- An all-equal emit folds only the rows of modules a built family imports, and never a row on code
  that runs after lowering (`rocke/helpers/compile.py`, `rocke/runtime/`): the IR digest does not
  cover it. A static kernarg `mismatch` is never downgraded.

## Judging the rows

Read both sides at the head revision, and at the merged tree when the row says `merged-only`, before
deciding.

| Status | Judgment |
|---|---|
| `mismatch` | A break: quote the producer line and every disagreeing copy. A `merged-only` mismatch breaks the target once merged; the fix is to rebase and update the named copy. If the same mismatch exists at the base, it predates the change |
| `changed`, dispatch | Read the C++ function and its Python counterpart. A divergence (a constant, a range, a tile set, a mask or layout rule that the other side does not share) is a break; quote both sides. Consistent: no break |
| `changed`, emitted code or arguments | Consistent with every copy, names and types: no break. A row that says `types not compared` or `cannot compare` is not consistency: compare the types and sizes of both lists by hand. The variants changed on an architecture need a device run there |
| `changed`, descriptor or generic target | Check the loader's rules for the changed fields (`projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp`); a value the loader rejects or ignores is a break. A membership change names the architectures whose archives change |
| `inert`, `outside` | Count them; list each `inert` reason once |

For every `changed` row that is not a break, name the device run that would settle it: the ctest
entry (find it from the pack or kernel family name in the test sources, and `ctest -N -R <regex>` in
a flags-ON build when one exists), the architecture, and the flags.

## Known gaps

- HIP-kind kernels (`ConvNative`, `PointwiseNative`) have their C++ argument lists checked but not
  their HIP producers.
- A dynamic C++ signature (`Gfx1151WmmaAttentionNative`) cannot be compared statically.
- One LLVM flavor is emitted per run, so an edit that changes only another flavor's lowering reads
  `inert`.
- The symbol check sees a reader only through an import of the defining module, `m.name` on it, or
  an attribute read in a file that names the class. It cannot see `from m import *`,
  `getattr(m, "f")`, a dotted import read in full (`import a.b.m`, then `a.b.m.f()`), or a module
  reached only through `importlib.import_module(f"...{arch}...")` (the emitter covers that last one
  for shipped variants; a bare `report` does not).
- A method removed from a base class and read only through a subclass or an untyped receiver, in a
  file whose imports reach the class's module, gives a `changed` row, not a `mismatch`: read the
  receivers.
- The emitter compares what goes into COMGR, not the COMGR library that compiles it: a change to
  which comgr the packer loads is a build and dependency question.
- A rocKE C++ engine source (`rocke/platform/cpp/`) reads `inert` only while no provider build file
  outside rocKE links the engine and the packer pins the Python backend; the row says so, so check
  that both still hold for the change.
- A static list is paired with the builder its names match. A C++ list that by mistake matches a
  different builder of the same modules exactly is compared with that one, not the launch it is
  used for; the row names the pairing, so check it.
- The emitter compares emitted arguments with the static C++ lists only for variants whose
  arguments changed between base and head. The variants of a new engine have no base side, so
  their lists are checked statically only.
- Only `std::vector<KernelArgument>` functions holding `KernelArgument{...}` literals are read as
  static C++ argument lists. A list built by helper calls instead (`gfx942Pointer("query_ptr",
  ...)`) is not parsed. When such a function exists, its row reads "`<f>() is a dynamic kernarg
  signature: it cannot be compared with <function> statically`", and only when the mirror file
  changed; when there is none, the pair's row reads "`the C++ mirror holds no
  std::vector<KernelArgument> function (its list is built another way): cannot compare`". With
  only the builder edited, a dynamic list gives no pair row. Compare such a list with the builder
  by hand.
