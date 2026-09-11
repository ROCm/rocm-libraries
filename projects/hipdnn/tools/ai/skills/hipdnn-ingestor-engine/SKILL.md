---
name: hipdnn-ingestor-engine
description: "Drive a COMPLETE hipDNN generic-kernel-ingestor integration for a new kernel: descriptors, the native pack's matcher/score/dispatch bodies, the CMake splice, a build, and an on-device test that proves the engine actually dispatches. For a rocKE kernel, mines the Python sources for the applicability rules the matcher must enforce. Not a descriptor generator -- descriptors are step four of nine. Use when the user wants to add a new ingestor engine, add a pack or kernel to an existing one, or integrate a rocKE builder into hipDNN."
argument-hint: "[create|extend] [<kernel-source-path-or-dir> | <existing-descriptor-dir>]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
---

# hipDNN Ingestor Engine Skill

Turns "I have a new kernel" into an ingestor engine that **actually runs**: descriptors,
native hook bodies, CMake wiring, a build, and a device test that dispatches it. One
skill, two flows — not one skill per descriptor type — because descriptors cross-reference
each other by UUID and the UMD-vs-`graph_match` decision is a property of the whole
engine, not of any single file.

Every file below has exactly one job. **If you are here to do an
integration, read this page's completion contract and then drive from `RUNBOOK.md`** —
everything else is reference material it sends you to at a named step, and each of those
files tells you what you owe before you return.

| File | When | Shape |
|---|---|---|
| `SKILL.md` (this) | First, and only this far. The completion contract and the dialect decision. | Why |
| `RUNBOOK.md` | **The thing you execute.** Every step, its commands, and the gate that ends it. | How |
| `graph-contract.md` | Sent from runbook step 2a. Which hipDNN operation — or composition — you are implementing, and everything the graph can ask for. | How |
| `rocke-mining.md` | Sent from runbook step 2b. Extracting applicability rules that exist only in the kernel's Python. | How |
| `native-pack.md` | Sent from runbook step 6. The five hooks and their silent-failure traps. | How |
| `workloads.md` | Sent from runbook 2a, 3 and 8e. Real workload shapes: deciding what to compile, and proving it runs. | How |
| `extend.md` | The **extend** flow: adding a pack or kernel to an engine that already exists. | How |

**This skill is op-agnostic.** Every step works for any hipDNN operation: the operation
catalog is enumerated, never assumed, and every discovery step is a command you run
against *your* kernel. A worked example naming a specific operation is an illustration,
not a description of yours — run the command, never reuse the example's answer. Spec
fields, layout, ABI and baked constants all differ per kernel, and assuming otherwise
produces silently wrong numbers.

**This skill is source-agnostic too, and that rule has teeth.** Trees get refactored;
files move, split and get renamed. So this page and its siblings name **what to look
for and the pattern that finds it** — never a fixed path or a line number. A path
written down here is a coordinate, and a coordinate is stale the moment someone
refactors. Two things are safe to state outright, and they are the only two:

- **Contract names** — a CMake option, a schema table, a registration function, a
  CLI flag. Renaming one of those is a reviewed, breaking change, so a name that
  breaks *should* break your run loudly.
- **A search that finds the thing** — an anchor directory plus a glob, or a `git
  ls-files` / `git log` query. That survives every move that preserves meaning.

**When discovery finds nothing, STOP. Do not guess, and do not proceed on a
near-match.** A path that used to exist is not evidence of where it went, and an
integration built against the wrong source fails much later, in a way that reads as
a kernel bug. Zero hits is a legitimate outcome and it is *information*: it means the
tree moved out from under this skill.

Escalate to the user. Say, explicitly:

1. **What you were looking for** — in terms of its job, not its filename ("the rocKE
   builder that defines the spec for this kernel").
2. **The exact command you ran** and that it returned nothing.
3. **Where you looked** and what you ruled out.
4. **That the skill's own expectation appears to be out of date**, and recommend the
   skill text be updated once they point you at the real location — otherwise the
   next run hits the identical wall.

Then ask where it lives, and wait. Blocking on a question costs minutes; guessing
costs a debugging session at stage 8, or a wrong integration nobody catches.

**A symptom is not a cause. Confirm the mechanism in source before you name one, and
especially before you escalate.** The same discipline as the rule above, pointed at
diagnosis instead of discovery. Two different faults routinely print the same message in
this stack — "no engines applicable" is *either* a wrong plugin path *or* a build with
the ingestor OFF; a FAILED validation row is *either* a numeric mismatch *or* a row where
no comparison ran at all. Find the line that emits the message and check that its
preconditions actually hold.

Two tells that a "failure" is not the failure you think:

- **Numbers that do not fit the dtype.** An fp32-shaped tolerance on a bf16 or fp16 graph
  means the dtype-aware path was never taken — so nothing was compared.
- **A failure with no magnitude.** If the tool prints a difference whenever it has one and
  none appears, none was computed. An empty field is not a zero.

The same trap applies to the tools you diagnose WITH. This shell's `grep` has no BRE `\|`
alternation, so `grep "a\|b"` matches nothing and reports success. **An empty result is
not evidence until the pattern has matched something you know is there** — use `grep -E`
and give it a positive control. Two "defects" reported against this skill were that
mistake, not defects.

## Completion contract — read this before starting

**A generated, validated descriptor bundle is NOT a finished integration.** It is roughly
a third of one. An engine whose descriptors validate perfectly and whose matcher returns
`nullopt` serves exactly zero graphs, and every mechanical check stays green while it does
so. That is the failure mode this skill exists to prevent, and reporting a validated
bundle as "done" reproduces it.

**If you were handed a PARTIAL run — "take this from stage 7 to stage 10", "just do the
device run and the report" — the stage numbers in that prompt are a pointer to this
table, never a redefinition of it.** A kickoff that says "stage 8: bundles and an
on-device run" has silently dropped 8e, which is part of the stage-8 gate, and a run that
skips it and reports "stage 8 complete" overstates what was proven. This has happened:
8e was skipped on a stage-scoped kickoff and the miss was only caught when a human asked
why the variant set was so small. **Re-read this table and the runbook's own step list
before accepting any prompt's account of what a stage contains.**

The ten stages, in order. A run is **incomplete** until stage 9:

| # | Stage | Done when |
|---|---|---|
| 1 | Dialect settled | `direct_load` vs `packaged` decided from the kernel's nature |
| 2 | Sources mined | **The graph contract first** (which hipDNN operation or composition you serve, and every field it can carry), **then** the kernel: spec introspected and its applicability rules extracted. Runbook steps 2a and 2b. |
| 3 | Batch confirmed with the human | Name, arch, **which knobs are exposed and which knob values ship AOT**, the variant set **and how it lands against real workload shapes**, workspace, UMD-vs-graph_match answered |
| 4 | Descriptors generated | `generate.py` exit 0, emitting the **full variant set** — not one kernel (see `RUNBOOK.md` § Sizing the variant set) |
| 5 | **Hook bodies implemented** | `graph_match`, `kernel_match`, `score`, dispatch all written — no `// TODO` left in a code path the engine reaches |
| 6 | Spliced | Every applicable CMake/registration point applied, edits made not just described |
| 7 | Built | Provider compiles with `HIPDNN_ENABLE_KERNEL_INGESTOR=ON`; engine appears in `hipdnn_list_engines` |
| 8 | **Tested on device, and wired into CI** | Quick-tier cases covering every supported feature of THIS op (many tiny graphs, budget-bounded) plus standard-tier cases at realistic sizes, added to `dnn-providers/integration-tests/`; **an `add_external_integration_test_target` pinned to this engine via `ENGINE_NAME`**, so the suite exercises yours rather than whichever engine wins; and a real graph dispatching on the target arch, verified against a reference; **plus an exploratory pass through `dnn-benchmarking`** (RUNBOOK 8e) validating against PyTorch and triaging that repo's shipped workloads -- required, but never wired into CI. **A green suite is NOT this gate**: green means only that whatever ran was correct. Reconcile passed vs skipped, confirm your engine id actually served the cases, and report a fraction rather than "all passed" |
| 9 | **Post-integration verified** | The SHIPPED descriptor set swept over a real corpus (the kernel team's published shapes plus `dnn-benchmarking`), the integration-test project run from the INSTALL tree against the engine-pinned target, **every graph either one flags triaged to a named cause with zero unexplained**, and **every decline reconciled against the reference library** — `tools/reconcile_applicability.py` at zero `ONLY THE REFERENCE` rows, or a written justification per row that remains. Stage 8 tests the engine on graphs you wrote; this tests the INTEGRATION on graphs you did not. A decline the reference does not share is missing coverage or a matcher bug, never a scope decision — "we chose not to serve it" does not discharge a row. Three times in this project's history an integration has been complete, green, and serving zero real workloads — each caught only by leaving the integration and counting against an external corpus. |
| 10 | Handed back | Residual judgment calls surfaced to the human with a recommendation |

**Each stage leaves an artifact on disk, and you commit it before starting the next.**
`RUNBOOK.md`'s sequence table carries the per-step `Produces` / `Gate` / `Typical time`
contract. Three stages have a mechanical done-check and they are the ones runs die on:
stage 2 is done when **both** `graph_contract.md` and `mining.md` exist, and stage 5 when
`generate.py --check-placeholders` exits 0. Research that produced no file is not a
completed stage — it is a stall, and the cure is to write down what you have, mark the
uncertain rows, and move.

**Scan for site-specific content immediately before every push, over the CURRENT file
set.** These branches are public. Scheduler payloads, run scripts and logs accumulate
hostnames, cluster and partition names, account names, shared image paths and other
people's home directories — and they are written at stage 8, long after the early stages
you may have scanned. A scan scoped to an earlier diff reports clean about files it never
saw, which is exactly how a leak has shipped: the scan ran, passed, and the offending
files were created afterwards. Re-scan the full set each time, and give the pattern a
positive control before you believe a zero (see the `grep -E` note above).

**Check at stage 1 that stage 8 is reachable.** The shared reference executors are dense
and stride-based and decline paged KV, varlen, ragged tensors and block-sparse — see
`dnn-providers/integration-tests/README.md` § *What the reference executors cannot
verify*, which owns that list. Attention sinks are a narrower case: the CPU reference
computes them, but the GPU reference still declines them, so a kernel needing sinks is
reachable only if your integration is verified against the CPU executor. A kernel
requiring one of the fully-declined features has nothing to be verified against, and you
will not find out until stage 8, after the descriptors, hooks and build are already done.
For a first integration, pick a kernel the shared references cover.

**Stage 8 has a second prerequisite, and it is hardware.** The engine's `arch` list is
the arch the test must run on — packs arch-prune before the matcher, so a clean run on
any other GPU proves nothing. Check at stage 1 that a node of that arch is actually
*schedulable to you*, not merely listed:

```
ssh <slurm-login> "sinfo -N -h -o '%P|%N|%t|%G' | grep <arch>"      # live? drained?
ssh <slurm-login> "squeue -h -o '%b' | grep -c <arch>"              # how deep is the queue
ssh <slurm-login> "sbatch --test-only -p <part> -A <acct> --gres=gpu:<type>:1 \
    --time=00:20:00 --wrap=hostname"                                # may I, and when
```

`--test-only` is the one that answers it: it reports the estimated start time, and reports
an access failure without consuming a submission. A partition can be visible in `sinfo`
and still reject you (`invalid partition specified`) when your account has no association
with it. Do this before mining, not after the build — a scarce arch behind a deep queue
turns a 9-stage run into a 7b run for reasons unrelated to the integration, and knowing on
day one lets you stage artifacts elsewhere while the other work proceeds.

If the arch is unreachable, say so at stage 1 and get a decision, exactly as with an
unverifiable feature. A run that reaches stage 7b with the device test queued is a
legitimate outcome — but it must be *reported* as stage 7b, never as stage 8.

**Stage 5 is where agents quit, and it is the stage that matters most.** The generator
emits a native stub whose every body is `// TODO - FILL THIS OUT` because those bodies
need kernel knowledge no tool can infer. *You* are the one expected to supply it — from
the kernel's own sources, from the reference implementations, and from the human when
neither answers. Filling them is the work, not a follow-up.

**Placeholders are allowed; silence is not.** A first-pass `score` that returns a single
metadata field, or a `workspaceBytes` that returns 0 because the kernel needs none, is a
legitimate starting point — *if* you say so explicitly and explain what would change it.
What is never acceptable is leaving a body empty, or reporting a stage you skipped as
though it were out of scope.

One carve-out, because disclosure does not rescue it: **a `score` returning a
*constant* is not a placeholder.** It makes ranking arbitrary and hides
mis-specialization, and saying so in the report does not change that. Rank on one real
knob or explain why the pack has nothing to rank on (`native-pack.md` § score).

**If you cannot reach stage 8, say which stage you stopped at and why**, in those terms.
"Blocked at stage 7: no gfx950 device reachable" is a good report. "Descriptors generated
and validated" — with no mention that stages 5-8 exist — is the failure this contract
forbids.

## When to invoke this skill

- The user wants to add a **new** ingestor engine for a kernel (or set of kernels) —
  **create flow**.
- The user wants to add **one more pack or kernel** to an engine that already has a
  descriptor directory — **extend flow**.
- The user asks to validate an existing descriptor tree, or asks what the CMake splice
  for a generated bundle looks like.

Do not invoke this skill for anything that edits the ingestor runtime itself
(`DescriptorLoader.hpp`, `KernelIngestorStateManager.hpp`, `NativeRegistry.hpp`) — this
skill only calls that code through the generator and the validator, never modifies it.

## The two dialects — settle this before anything else

A bundle is authored in one of two dialects, and they are not interchangeable:

| | `direct_load` | `packaged` |
|---|---|---|
| Kernel | a `.cpp`/`.hip` the provider embeds | a rocKE builder, or a `.cpp` compiled at build time |
| `kernel_source.kind` | `embedded_source` | `rocke` or `hip` |
| Consumed by | `DescriptorLoader.hpp` at runtime | `hkp_pack` at build time |
| Validated with | `hipdnn_validate_descriptors` | `hkp_pack`'s `load_flat_input`, then pack, then `hipdnn_validate_descriptors` on the PACKED tree |
| `HIPDNN_DESCRIPTOR_FILES` | spliced in | **never** |

**A rocKE kernel is always `packaged`.** The runtime has no rocKE adapter: `hkp_pack`
lowers the builder through comgr at build time and rewrites the shipped descriptor to
`kind: kpack` before the loader sees it. Authoring `rocke_builder` for the runtime is
rejected, and correctly so.

## The two flows, at a glance

**Create** (new engine): settle the dialect, ask for the kernel sources *first*, establish
the graph contract (which hipDNN operation you are implementing and everything it can
carry), mine the kernel, then confirm the remainder in **one batch** — engine
name/namespace, arch list, which fields are knobs, dispatch/workspace policy, and whether
a distinction becomes a UMD or lives in the UED's `graph_match`. Then generate, validate,
implement the pack, splice, build and test on device. **Drive it from `RUNBOOK.md`.**

**Extend** (existing engine): point at the existing descriptor directory, add one pack
or one kernel, mint only the *new* UUIDs, append to the existing CMake lists rather than
rewriting them, then re-run the validator over the **whole** directory — whole-directory
revalidation is what demonstrates the pieces that were already there stayed valid. Full
steps in `extend.md`.

## Validator availability — check this before promising validation

`hipdnn_validate_descriptors` is built and installed **only** when the consuming build
was configured with `HIPDNN_ENABLE_KERNEL_INGESTOR=ON` (find where the option is
declared with `git grep -n 'option(HIPDNN_ENABLE_KERNEL_INGESTOR'`), and its default is **OFF**. A build
directory configured the ordinary way will not contain the binary at all. Absence is
the **common** case, not an edge case — both flows must detect it explicitly (search the
active build directory's `bin/`, and any install prefix, for `hipdnn_validate_descriptors`)
and say so plainly in the completion report, rather than silently skipping the
validation step.

`hkp_pack`'s authored-form validation needs no build at all — it is Python, importable
straight from `descriptor-packaging/python`. Packing additionally needs `msgpack` and
`zstandard` (the kpack reader's own dependencies) and a comgr for the rocKE path.

## Output contract (every run, both flows)

Every completion report this skill produces states, explicitly:

1. **The stage reached**, by number, from the completion contract above. If it is not 9,
   name the stage that stopped you and what would unblock it.
2. **What was proven, and what was not.** Be precise about the ladder: a green validator
   proves parse + cross-reference + symbol resolution + construction. `hipdnn_list_engines`
   adds "the pack registered". **Neither says anything about matching** — that needs a
   real graph on the target arch. PR #10839's SDPA defect enumerated cleanly on gfx90a
   (where the packs arch-pruned before the matcher ran) and failed all 27 cases on
   gfx942.
3. **Every hook body's state.** Per hook: implemented, or a stated placeholder with what
   would replace it. A `// TODO` left in a reachable path is an unfinished integration,
   and the report must say so rather than omit it.
4. **The splice points that applied, and the ones that did not.** For a packaged bundle,
   say explicitly that `HIPDNN_DESCRIPTOR_FILES` and `HIPDNN_INGESTOR_PACK_KERNELS` do
   **not** apply and why, so a reviewer does not helpfully add them back. Where you made
   the edits, say so; where you only described them, say that instead.
5. **The integration tests you added**, by tier and path — quick-tier for smoke, standard
   for the shape matrix. If you added neither, that is a stage-8 miss, not a footnote.
   **And the 8e exploratory result**: which of `dnn-benchmarking`'s shipped workloads your
   engine served, which it correctly declined and against which named checklist row, and
   which it *could* build but ships no variant for. That last bucket is the one your own
   bundles structurally cannot reveal.
   **And the CI target that runs them against YOUR engine**, by name: an
   `add_external_integration_test_target(... ENGINE_NAME <yours> ...)` block, with the
   build flags it is gated on. Bundles run against whichever engine *wins* a graph, so
   without that target a new engine competing with an incumbent can sit unexercised while
   the suite reports green. Tests with no engine-pinned target are tests that do not
   defend your engine. `--test-engine` matches the engine's `engineName`, which for a
   descriptor-registered engine comes from the UED's own `name` field — so pin **exactly
   what the UED spells**, and the target runs your engine rather than the graph's winner.
   **Then prove the cases ran.** Pinning fixes attribution, not coverage: if your matcher
   declines everything, every case SKIPs and the suite still exits 0. Report
   *"N of M dispatched this engine, K skipped for <reason>, 0 unexplained"* — a bare
   "green" hides a matcher that serves nothing.
6. **The judgment calls you are handing back**: rocKE restrictions you could not check
   from a graph, knobs you fixed that could be searched, and coverage the tests do not
   have. Each with a recommendation.
7. **Whether the validator ran at all.** If `hipdnn_validate_descriptors` could not be
   located because the build has `HIPDNN_ENABLE_KERNEL_INGESTOR=OFF` (or no build
   exists), say so by name, name the flag, and state that structural validation did
   **not** happen — never report a bundle as validated when the binary was never invoked.

## What this tooling does NOT do for you

Most of the pipeline is op-agnostic and driven by a per-kernel **profile** — a kernel
states its dispatcher, request class, predicate, matcher vocabulary, policy-owned knobs,
sweep candidates and launch surfaces, and the tools do the rest without being edited.
A profile lives at `configs/<slug>.profile.yaml`; the integration branch stacked on this
one carries a worked example, and a second arch was driven end to end by copying it and
changing about six values.

**Four things are genuinely per-op, and pretending otherwise is worse than saying so.**
Each one below is a place where you must supply knowledge the tools cannot infer, and
each carries a REQUIRED test so the knowledge lands as something that fails when it
drifts, rather than as a comment.

### 1. The shape corpus is op-shaped

`tools/mine_shapes.py` mines attention: it reads attention request fields, attention CSV
columns, and graph tensors named `query`/`key`. **Another op family needs its own miner**
— that is a per-op file, not a defect, because a corpus format IS op-specific.

What you owe: a miner for your op that emits the same request-mapping JSON, carries
PROVENANCE on every shape, and refuses an unrecognised categorical value rather than
defaulting it. **Test:** an unknown value in your op's equivalent of a mask column must
be REFUSED, with a test that asserts it. Defaulting one is how a windowed graph got served
as plain causal — a wrong answer, not a decline.

### 2. The dangerous-graph class is op-shaped

`tools/sweep.sh` requires `EXCLUDE_TENSORS` and has NO default, because the graphs that
are unservable-and-dangerous differ per op. For attention it is backward graphs, marked
by gradient tensors. For your op it is something else, or nothing.

What you owe: name the class, or write `EXCLUDE_TENSORS=none`. There is no third option,
and the harness refuses to run without one — a defaulted marker set would run, find
nothing, and report protection it was not providing. One such graph faulted the DEVICE
mid-sweep with every later phase then measuring dead hardware.

### 3. The launch contract is restated by hand, per engine

Your C++ recomputes what the kernel's Python computes — grid, block, kernargs, baked
constants, spec resolution, applicability. Nothing in the build, the packer or the
validator compares the two halves, and a mismatch does not fail: the kernel runs and
computes something else. Two defects shipped this way.

What you owe: a `launch_surface:` block in your profile enumerating every surface with
its Python source, its C++ mirror, the KMD fields it branches on, its guard and its test.
`$GEN/tools/launch_surface.py --check` verifies the declaration is honest and **names any
surface with no guard or no test**. An unguarded surface is a legitimate thing to ship; an
unguarded surface nobody wrote down is not. **Test:** extract the geometry into a pure
function of descriptor metadata, in a `<Pack>Geometry.hpp` beside the pack, and test it
per shape family. Inside `prepare()` that arithmetic is unreachable without a
device, which is exactly why it went unchecked while most shipped shapes never ran.

### 4. The scorer is native, so reachability needs a declaration

`$GEN/tools/variant_reachability.py` can tell you a variant matches no shape. It cannot call
your C++ `score`, so ranking comes from a profile declaration; without one it treats every
applicable variant as reachable and says so rather than guessing.

What you owe: declare the ranking. **Test:** a variant that is applicable but always
outranked is dead weight that reports green — an integration once shipped 48 variants of
which 24 no graph could select, because every shape made both tiles applicable and the
scorer always preferred the wider one. The fix is a shape where the rival is ILLEGAL, not
more variants.

### And two known gaps in the surrounding tooling, not in yours

- **Packing cost scales with the shape you compile, not just the variant count**, so a
  variant-count budget that looks like a design limit is partly a tooling artifact. Time
  one pack and check whether it saturates the machine before you shrink a set to fit it.
  (`hkp_pack` parallelises its prewarm across `HKP_PACK_JOBS` workers; `HKP_PACK_JOBS=1`
  forces serial, which is the knob to reach for when a pack's failure output interleaves.)
- **Descriptors land in the packager's tree, not the engine tree**, because `hkp_pack`
  rejects the engine tree's dialect. RUNBOOK step 4 has the detail. This is a known
  deviation, not your layout mistake — but say so in your report rather than leaving the
  next reader to rediscover it.

## Reference materials this skill relies on

- Descriptor-format authority, in precedence order: the loader itself
  (`git ls-files '*ingestor/DescriptorLoader.hpp'` — `FILE_TYPES` and the `parse*`
  functions are what actually accepts or rejects a
  file), the struct definitions beside it (`Descriptors.hpp`), and the native hook
  signatures (`NativeRegistry.hpp`). The design intent is in
  `projects/hipdnn/docs/rfcs/0017_UniversalKernelDescriptor.md` and
  `0020_UniversalEngineDescriptor.md`, but where an RFC and the loader disagree, **the
  loader wins** — the RFCs are design documents, not specifications of the code.
- Worked examples to copy: the shipped descriptor sets in the engine's `descriptors/`
  directory — `git ls-files '*kernel_ingestor_engine/descriptors/*'` (at time of writing
  `conv_fwd/` is the smallest complete engine and `pointwise/` the multi-pack shape),
  with their native halves in the sibling `packs/` directory. **Read them as shape
  references, not as your destination:** they are `direct_load` (`kind: embedded_source`),
  and `hkp_pack` rejects that kind, so a `packaged` integration cannot currently be
  authored in that tree. RUNBOOK step 4 has the detail; your descriptors go to the
  packager's `PRODUCTION_SOURCE_ROOT` until it is fixed.
- The generator's CLI: `generate.py --config <yaml> --output-dir <dir> [--dry-run] [--force]`,
  in the `IngestorGenerator` tool directory (`git ls-files '*IngestorGenerator/generate.py'`).
- The validator's CLI and `--json` shape:
  `hipdnn_validate_descriptors <root>... [--native-source <cpp>]... [--expect-engine <name>]... [--json]`,
  built from the `ValidateDescriptors.cpp` beside that generator.
