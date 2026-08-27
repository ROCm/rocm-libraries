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

This skill is six files, each with exactly one job. **If you are here to do an
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
| `extend.md` | The **extend** flow: adding a pack or kernel to an engine that already exists. | How |

**This skill is op-agnostic.** hipDNN spans convolution, matmul, normalization, attention,
quantization and more, and every step here is written to work for any of them: the
operation catalog is enumerated, never assumed, and every discovery step is a command you
run against *your* kernel. Where a worked example names a specific operation it is there to
make a shape concrete — read it as an illustration, never as a description of your kernel.
Run the command; do not reuse the example's answer. A different kernel's spec fields,
layout, ABI and baked constants will all differ, and the failure mode for assuming
otherwise is silently wrong numbers.

## Completion contract — read this before starting

**A generated, validated descriptor bundle is NOT a finished integration.** It is roughly
a third of one. An engine whose descriptors validate perfectly and whose matcher returns
`nullopt` serves exactly zero graphs, and every mechanical check stays green while it does
so. That is the failure mode this skill exists to prevent, and reporting a validated
bundle as "done" reproduces it.

The nine stages, in order. A run is **incomplete** until stage 8:

| # | Stage | Done when |
|---|---|---|
| 1 | Dialect settled | `direct_load` vs `packaged` decided from the kernel's nature |
| 2 | Sources mined | **The graph contract first** (which hipDNN operation or composition you serve, and every field it can carry), **then** the kernel: spec introspected and its applicability rules extracted. Runbook steps 2a and 2b. |
| 3 | Batch confirmed with the human | Name, arch, **which knobs are exposed and which knob values ship AOT**, the variant set, workspace, UMD-vs-graph_match answered |
| 4 | Descriptors generated | `generate.py` exit 0, emitting the **full variant set** — not one kernel (see `RUNBOOK.md` § Sizing the variant set) |
| 5 | **Hook bodies implemented** | `graph_match`, `kernel_match`, `score`, dispatch all written — no `// TODO` left in a code path the engine reaches |
| 6 | Spliced | Every applicable CMake/registration point applied, edits made not just described |
| 7 | Built | Provider compiles with `HIPDNN_ENABLE_KERNEL_INGESTOR=ON`; engine appears in `hipdnn_list_engines` — **as an FNV-1a hex id, not as your engine name.** Grepping the literal name finds nothing *even on success* (AICK-1901); compute the hash and grep for that, per `RUNBOOK.md` step 7c |
| 8 | **Tested on device, and wired into CI** | Quick-tier cases covering every supported feature of THIS op (many tiny graphs, budget-bounded) plus standard-tier cases at realistic sizes, added to `dnn-providers/integration-tests/`; **an `add_external_integration_test_target` pinned to this engine via `ENGINE_NAME`**, so the suite exercises yours rather than whichever engine wins; and a real graph dispatching on the target arch, verified against a reference |
| 9 | Handed back | Residual judgment calls surfaced to the human with a recommendation |

**Each stage leaves an artifact on disk, and you commit it before starting the next.**
`RUNBOOK.md`'s sequence table carries the per-step `Produces` / `Gate` / `Typical time`
contract. Three stages have a mechanical done-check and they are the ones runs die on:
stage 2 is done when **both** `graph_contract.md` and `mining.md` exist, and stage 5 when
`grep -c "FILL THIS OUT"` on the pack returns 0. Research that produced no file is not a
completed stage — it is a stall, and the cure is to write down what you have, mark the
uncertain rows, and move.

**Check at stage 1 that stage 8 is reachable.** The shared reference executors are dense
and stride-based and decline paged KV, varlen, ragged tensors and block-sparse/sinks —
see `dnn-providers/integration-tests/README.md` § *What the reference executors cannot
verify*, which owns that list. A kernel requiring one of those has nothing to be verified
against, and you will not find out until stage 8, after the descriptors, hooks and build
are already done. For a first integration, pick a kernel the shared references cover.

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

`--test-only` is the one that answers it: it reports the estimated start time, and it
reports an access failure without consuming a submission. A partition can be visible in
`sinfo` and still reject you (`invalid partition specified`) when your account has no
association with it. Do this before mining, not after the build: a scarce single-GPU arch
class behind a deep queue can turn a 9-stage run into a 7b run for reasons that have
nothing to do with the integration, and knowing on day one lets you stage artifacts to a
less contended site while the other work proceeds.

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
was configured with `HIPDNN_ENABLE_KERNEL_INGESTOR=ON`
(`projects/hipdnn/CMakeLists.txt:65`), and that option's default is **OFF**. A build
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
   **And the CI target that runs them against YOUR engine**, by name: an
   `add_external_integration_test_target(... ENGINE_NAME <yours> ...)` block, with the
   build flags it is gated on. Bundles run against whichever engine *wins* a graph, so
   without that target a new engine competing with an incumbent can sit unexercised while
   the suite reports green. Tests with no engine-pinned target are tests that do not
   defend your engine. **Write that target even though it cannot pass yet**: engine-name
   exposure is in progress, so today `--test-engine` sees a bare hex id and the target
   fails with `Error: Engine '<name>' is not loaded`. Report that as *pending engine-name
   exposure (AICK-1901)* — an expected state, not a test failure and not your provider's
   bug. Nothing ships before the name lands, so registering ahead of it is correct.
6. **The judgment calls you are handing back**: rocKE restrictions you could not check
   from a graph, knobs you fixed that could be searched, and coverage the tests do not
   have. Each with a recommendation.
7. **Whether the validator ran at all.** If `hipdnn_validate_descriptors` could not be
   located because the build has `HIPDNN_ENABLE_KERNEL_INGESTOR=OFF` (or no build
   exists), say so by name, name the flag, and state that structural validation did
   **not** happen — never report a bundle as validated when the binary was never invoked.

## Reference materials this skill relies on

- Descriptor-format authority, in precedence order: the loader itself
  (`projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp`
  — `FILE_TYPES` and the `parse*` functions are what actually accepts or rejects a
  file), the struct definitions beside it (`Descriptors.hpp`), and the native hook
  signatures (`NativeRegistry.hpp`). The design intent is in
  `projects/hipdnn/docs/rfcs/0017_UniversalKernelDescriptor.md` and
  `0020_UniversalEngineDescriptor.md`, but where an RFC and the loader disagree, **the
  loader wins** — the RFCs are design documents, not specifications of the code.
- Worked examples to copy: the shipped descriptor sets under
  `dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/`
  (`conv_fwd/` is the smallest complete engine; `pointwise/` is the multi-pack shape),
  with their native halves in the sibling `packs/` directory.
- A's frozen CLI: `generate.py --config <yaml> --output-dir <dir> [--dry-run] [--force]`,
  under `projects/hipdnn/tools/IngestorGenerator/`.
- B's frozen CLI and `--json` shape:
  `hipdnn_validate_descriptors <root>... [--native-source <cpp>]... [--expect-engine <name>]... [--json]`,
  built from `projects/hipdnn/tools/ValidateDescriptors.cpp`.
