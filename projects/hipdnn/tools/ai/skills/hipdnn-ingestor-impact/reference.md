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
| Generic targets | `projects/hipdnn/plugin_sdk/data/gpu_generic_targets.json`, `projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/GpuGenericTargets*` and its CMake template `projects/hipdnn/plugin_sdk/cmake/GpuGenericTargetsTable.hpp.in` | packer and loader |
| Loader code | the ingestor SDK headers under `projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/ingestor/`, and every plugin SDK header they include, by `<hipdnn_plugin_sdk/...>` or by a quoted path beside the including header (`ArchMatch.hpp`, `heuristics/uhd/`; tagged `sdk-loader`) | the native packs and the loader that compile them |
| Python symbols | rocKE and `hkp_pack` modules | other modules and tests that import them |

Test code, CMake, CI test-category lists and `requirements*.txt` are not judged here: they are
`outside` for status, but tests are still searched as copies and readers, so a stale pin or a stale
import is reported.

How the copies are paired:

- A `launch_surface` entry's `python_source` symbol resolves as `launch_surface.py --check`
  resolves it: any function of that name anywhere in the module (`ast.walk`), a method, a nested
  function or a def under `if`, `try` or `with` included. Every function of that name is compared,
  and the symbol is reported missing only when none exists. A locator that is a path alone names
  no symbol, and the whole module is its Python side.
- A host-side surface function (a grid or block helper: anything but the builder the packer
  compiles, or a `kernargs` entry) is compared together with every top-level def, class and
  assignment it reaches by name, followed through `from` imports and module aliases (`import a.b
  as m`, then `m.f`). A grid that hands off to `NONPERSIST_DECODES[...].grid` changes when that
  class does, and no emitted IR covers it. A call through an argument (`spec.grid()` on an
  un-annotated `spec`) is not followed: see *Known gaps*.
- A declared `kernargs` pair compares each static C++ list in its `cpp_mirror` with the builder
  whose `b.param` list it matches, searched in the surface's module and the manifest's own
  `source` module, the declared function first. A multi-launch pack holds one list per launch (a
  segment and a reduce kernel), and a declared function may be a `SignatureBuilder` factory with no
  `b.param` list; the row then names the builder it was paired with. A list related to the
  declared builder (it shares at least two names, and at least half of its own) is compared with
  it unless another builder shares more names, so a list cut short of the declared builder's
  params is a `mismatch`, not a match with a smaller builder. A list that matches no builder but
  is related to one is compared with the related builder it shares most names with, and is a
  `mismatch`. An unrelated list is skipped; when no list of the mirror was compared, one
  `changed` row says `cannot compare`.
- A `global_buffer` entry is a pointer unless its declared size says otherwise; a 4-byte one
  disagrees with a pointer param, as `requireSignatureMatch` refuses it. The C++ side has no type
  field, so the scripts compare names, order and sizes.
- A copy that holds the old list is `changed`, not `mismatch`, when it still matches the new
  list with the new conditional parameters left out: variants that do not enable them pass that
  list. The variants that do enable them are what the emitter or a device run settles.
- A `NEEDS-DEVICE` arch list comes from the shipped descriptors: the UKD's `arch` array, else its
  KDP's, with generic targets expanded through `gpu_generic_targets.json`. A file outside the
  shipped closure (a pack, a descriptor, a DVC pointer) names its arch by path, widened to every
  arch a KDP named for that arch ships on.
- A flatc `*_generated.h` reads as regenerated only beside a change to its own `.fbs` or
  `flatc_flags.txt` (for the flatbuffers SDK, `schemas/<name>.fbs` and `cmake/flatc_flags.txt`).
  Any other generated artifact needs a change under its source tree; a test file never counts.

## The scripts

- `scripts/impact.py` reads git objects only (`cat-file`, `diff`, `ls-tree`, `grep`,
  `merge-tree`); it writes no ref, index or worktree. Its subcommands are `files`, `kernargs`,
  `symbols`, `surfaces`, `data` and `report`; `report` runs all of them and ends in the rollup line.
  The merged view needs git 2.38 or later (`merge-tree --write-tree`); an older git skips it as
  `changed`.
- `scripts/impact_emit.py` imports and runs rocKE code from the two trees: a `git archive` extract
  per side, in a child process whose whole process tree is killed at its timeout, with an
  environment reduced to system, temp, locale and Python user-site variables, so credentials such
  as `GH_TOKEN` are not passed. That is the same code execution as building the head, and it is
  not a sandbox: the child keeps `HOME` (`USERPROFILE` on Windows) and network access, and only
  tokens held in environment variables are dropped. It builds every shipped variant at both sides
  on the CPU (no GPU or comgr): a few minutes, with the two sides on two concurrent workers. Every
  `IMPACT_EMIT_*` variable passes through to the child; it reads `IMPACT_EMIT_TABLE_SECONDS`
  (see *Predicate tables*) and `IMPACT_EMIT_ENUM_TIMEOUT`, the seconds the registry enumerator
  gets for a DVC-held catalog (default 600, inside the per-side `--timeout` of 900).

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
`Runtime impact: BREAKS | NEEDS-DEVICE <arch, ...> | NONE (<n> files inspected) | N/A (no surface file)`.
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
argument list is recorded and compared too. A multi-launch UKD (`sources`) gives one variant per
distinct source, builder, spec and target of its launches; one the emitter cannot read is listed
under `not_inspected`. `impact.py report --emit` folds the result into the rows.

- The emitter exiting 1 (it could not run, or a side timed out) leaves no proof. Never fold an emit
  file from an earlier run. `impact.py` refuses an emit file made for another base or head, or one
  it cannot parse.
- Record the flavor it used. One LLVM flavor is emitted per run (`--flavor` picks another).
- A family listed under `failures` or a KDP under `not_inspected` leaves its rows `changed`: never
  read a failure as equal.
- An all-equal emit folds only the rows of modules a built family imports, and never a row on code
  that runs after lowering (`rocke/helpers/compile.py`, `rocke/runtime/`): the IR digest does not
  cover it. A static kernarg `mismatch` is never downgraded. In the merged view it folds only the
  rows of modules whose merged content is the head's, and only when the target changed no
  descriptor and no module the shipped code imports since the base: the emitter compared the base
  with the change, and a target that moved what the change builds on makes another program.

## Predicate tables

The emitter also puts each builder's support predicate to the test, on both sides, after the
builds. It works on the family's shipped builder, and on every other `build_*` function of its
module that the builds called (a `KernelModule`'s kernel builders, such as a `reduce` companion's).
It takes the shipped specs (an inner builder's are the specs the builds passed it) and varies one
int or bool field at a time from up to four seeds. The values are the shipped ones, 0, -1, and the
negation, double, half and 1.5 times of each shipped positive value. A field that ships more than
six distinct ints is cut to its four smallest and two largest positive values, which stand in for
the shipped ones too. It asks the predicate the packer would resolve (`_resolve_support_predicate`).

| Check | Finding |
|---|---|
| `builder` | The predicate accepts the spec and the builder raises on it: the packer would pass a spec its builder rejects. The detail names the exception and the `file:line` that raised it. |
| `sign` | Between two builders of one family, on a field both specs have: a value outside the sign range of every shipped value (<= 0 where all are >= 1, < 0 where all are >= 0) that this predicate accepts and the other refuses. |

The `sign` check compares only predicates built into the same module. A lone predicate that
accepts 0 is not a finding: many fields mean "off" or "default" at 0 or -1.

`at_base` marks two things:

- **On a table**: the base ran the same builder's table.
- **On a finding**: the base's table had the same check on the same field, so the finding
  predates the change.

`report` gives each finding a state and lists the findings in their own section, in this order:

| State | When |
|---|---|
| `new` | The base's table of the same builder lacked it; or the base did not table the builder and the change edits its predicate or that builder (a function the diff's hunks fall in, in the family's source module) |
| `first shipped` | The base did not table the builder and the change touches neither: an older predicate defect that the change ships for the first time |
| `at base` | The base's table had it: it predates the change |

These are pack-time facts. `report` never turns them into rows, so they never change the rollup.
Instead it prints a `Pack-time:` line just above the `Runtime impact:` line:

| Line | Meaning |
|---|---|
| `NEW <n>` | The `new` findings |
| `FIRST-SHIPPED <n>` | The `first shipped` findings |
| `CLEAN (<n> tables)` | Every table ran in full; any finding is `at base` |
| `NOT RUN (<why>)` | No emit file, or no shipped rocKE builder to table |
| `NOT RUN for <k> of <n> tables (<why>); no finding in the rest` | No finding, but a table was skipped or truncated, so the line cannot read clean |

`NEW` and `FIRST-SHIPPED` can appear together, followed by `; not run for ...` when some table
did not run.

A table that cannot run says why (`skipped`): the packer has no resolver at that revision, no
predicate takes the spec, or the specs are not dataclasses. A family's tables stop after
`IMPACT_EMIT_TABLE_SECONDS` (default 60) and say `truncated`. A builder call that never returns
(a probed spec it loops on) is abandoned 5 s past that budget: that family's tables and those of
the families after it read `skipped: probe timed out`, and the argument and IR comparison is
still written. A builder stuck in native code that holds the interpreter lock still runs the side
to its `--timeout`.

Judge each finding whose state is `new` or `first shipped`. Read the predicate and the
builder at `HEAD`.

- A `builder` finding is a defect unless something refuses that spec before the packer.
- A `sign` finding is a defect when the builder or a launch uses the field as a count, size or
  index.

A finding that holds is `IMPORTANT` (*Severities*). A finding that is `at base` predates the
change and gets no severity of its own; mention it only when the change makes it reachable.

## C++ differential tables (judge-written)

The scripts compare the names, order and sizes of a C++ copy, never its values or what it refuses.
For a `changed` `launch_surface` row whose C++ side is header-only and includes no HIP header
(geometry, split rule, workspace, arguments), settle the values with a throwaway table. A
`kernargs` or `workspace` row qualifies like any other, whether its C++ list is built from
`KernelArgument` literals or helper calls. One table compares one C++ function with its Python
counterpart; a header that restates several functions gives several tables, and the
`C++ tables:` count is per function:

1. Extract the side with `git archive <HEAD> <paths>` into a scratch directory, as the emitter
   does. Never use the worktree.
2. Write a small `main()` that includes the header, reads one input row per line from stdin and
   prints the C++ result or `throw <message>`. Compile it with a host compiler, adding `-I` for the
   header's directory and `projects/hipdnn/plugin_sdk/include`. On Windows use the ROCm wheel's
   `clang++` with the MSVC `INCLUDE` and `LIB` set. No hipDNN build is needed.
3. Write a Python driver that imports the `python_source` module (or the class the `guard:` text
   names) from the extract. It builds the rows, runs both sides and prints each row where they
   disagree. The rows are each value the shipped catalog uses for the fields the C++ side reads
   (each value in at least one row, not the cross product), plus each Python refusal's boundary
   just inside and just outside it: every `raise` and `return False, ...` in the function and in
   the `__post_init__` of the classes it builds. When the Python function compiles or launches
   kernels, stub that layer (no GPU, comgr or kernel cache) and say in *C++ table detail* what
   was stubbed. Keep a table to a few thousand rows: sample each field's values, do not
   enumerate their combinations.
4. Run it at `HEAD`, and on the merged tree when the row is `merged-only` or the target moved
   rocKE.

Judge every disagreement, naming the function on each side:

| Disagreement | Severity |
|---|---|
| A valid row whose values differ | `BLOCKING` |
| Python refuses, C++ accepts, and no C++ refusal on the engine's call path refuses it first | `BLOCKING` |
| Python refuses, C++ accepts; a C++ refusal on the engine's call path refuses it first (name it), but the function itself does not enforce the precondition, and some call site outside that guarded path reaches the function (a test included, whatever inputs that call site passes) | `IMPORTANT` |
| The same, with no call site outside the guarded path | No finding; list it in *C++ table detail* |
| C++ refuses, Python accepts, on an input the matcher lets through | `IMPORTANT`: the engine declines what rocKE serves |
| C++ refuses, Python accepts, on an input the matcher refuses first | No finding; list it in *C++ table detail* |

The `IMPORTANT` row applies on purpose to any function a unit test calls directly: a test is a
caller outside the guarded path, and the finding marks a precondition the function relies on but
does not check, as `gfx942StridedKvArguments` on #14259 did.

Matchers in `*Native.cpp` need the graph data types, so they get no table: read them by hand and
say so in *C++ table detail*. Delete the scratch directory when done.

Summarise the outcome on the `C++ tables:` line at the top of the report, beside the two verdict
lines. `<k>` counts every table with at least one disagreeing row, whatever the judgment, so a
disagreement judged harmless still shows in the count. For each disagreeing table, the line gives
the function, what disagrees and the severity. When a row qualified but no table was written, the
line says `none written` and why.

## Severities

Every judged finding in the report carries one:

| Severity | Use for |
|---|---|
| `BLOCKING` | A break on the engine's path: a holding `mismatch` row, a dispatch divergence, values that differ on a valid row, a refusal no C++ check on the engine path restates, a descriptor or generic-target value the loader rejects or ignores |
| `IMPORTANT` | Real risk off the engine's path or not yet reachable: a predicate-table finding that holds, a precondition the function does not enforce while another caller reaches it, the engine declining what rocKE serves |
| `SUGGESTION` | Anything else worth fixing in code the change touches |

A `changed` row that is consistent is not a finding; it names the device run that settles it.

## Judging the rows

Read both sides at the head revision, and at the merged tree when the row says `merged-only`, before
deciding.

| Status | Judgment |
|---|---|
| `mismatch` | A break: quote the producer line and every disagreeing copy. A `merged-only` mismatch breaks the target once merged; the fix is to rebase and update the named copy. If the same mismatch exists at the base, it predates the change |
| `changed`, dispatch | Read the C++ function and its Python counterpart. A divergence (a constant, a range, a tile set, a mask or layout rule that the other side does not share) is a break; quote both sides. Consistent: no break |
| `changed`, emitted code or arguments | Consistent with every copy, names and sizes: no break. A row that says `sizes not compared` or `cannot compare` is not consistency: compare the sizes of both lists by hand. The variants changed on an architecture need a device run there |
| `changed`, descriptor or generic target | Check the loader's rules for the changed fields (`projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp`); a value the loader rejects or ignores is a break. A membership change names the architectures whose archives change |
| `inert`, `outside` | Count them; list each `inert` reason once |

For every `changed` row that is not a break, name the device run that would settle it: the ctest
entry (find it from the pack or kernel family name in the test sources, and `ctest -N -R <regex>` in
a flags-ON build when one exists), the architecture, and the flags.

## Known gaps

- A host-side surface function is compared with the code it reaches by name and through module
  aliases, not with a method called on an argument whose class the code does not name
  (`spec.grid()` on an un-annotated `spec`). An edit to such a method can still fold to `NONE`
  on an all-equal emit: read the surface function's callees when its row folds.
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
- A static list is paired with the builder its names match, the declared builder first. A C++
  list that by mistake matches a different builder of the same modules exactly, sharing more names
  with it than with the declared one, is compared with that one, not the launch it is used for;
  the row names the pairing, so check it.
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
- A predicate table varies one field at a time, so a defect that needs two fields to move
  together is out of reach, unless one seed already has the other value. It builds the accepted
  specs but does not lower them, so a builder that only fails in lowering reads clean. It finds
  inner builders only through the module's own `build_*` globals: a builder imported under
  another name, or called through a dict, is not tabled.
- `new` versus `first shipped` reads only the family's source module. A predicate the module
  imports from elsewhere, edited there by the change, reads `first shipped`. Check where the
  predicate is defined before calling a `first shipped` finding older.
- No table, scripted or judge-written, sees a change made in step on both sides: Python and C++
  still agree. That is a test-quality question.
