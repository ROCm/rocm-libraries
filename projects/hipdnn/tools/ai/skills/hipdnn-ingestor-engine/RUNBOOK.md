# Runbook: production mining and lowering for an ingestor engine

This is the **only ordered workflow for production mining and lowering** — corpus and
scope, packaging, tuning and final corpus proof. It is **not** the create
path: a plain HIP integration belongs to
[hipdnn-kernel-integration](../hipdnn-kernel-integration/SKILL.md), whose RUNBOOK owns
the six-step sequence from a kernel to descriptors, native hooks, registration and
graphs, and which links back here for the production half. Use [SKILL.md](SKILL.md) for
entry/completion requirements and the linked domain pages for contracts. Each gate
requires current evidence; an old install or copied command is not an observation.

**A genuinely new native symbol is a rebuild.** That is true of every dialect, not only
of a drop-in: it is ordinary engine work through stages 3-5, and the ordered form of it
is [hipdnn-kernel-integration](../hipdnn-kernel-integration/SKILL.md)'s RUNBOOK rather
than this page. Decide that before you decide a source dialect.

## Paths and interpreters

Resolve absolute paths before executing commands. These are explicit arguments,
not implicit tool inputs:

```bash
REPO=/absolute/path/to/rocm-libraries
PROVIDER="$REPO/dnn-providers/hip-kernel-provider"
GEN="$REPO/projects/hipdnn/tools/IngestorGenerator"
PY="$GEN/.venv/bin/python"
BUILD=/absolute/path/to/build
INSTALL=/absolute/path/to/future-install
SWEEP_ROOT=/existing/device-visible/writable/experiment-root
GENERATED=/absolute/path/to/empty-generation-destination
CONFIG=/absolute/path/to/generator-config.yaml
PROFILE=/absolute/path/to/authoring.profile.yaml
SHAPES=/absolute/path/to/request-shapes.json
CORPUS_DIR=/absolute/path/to/graph-corpora
ARCH=gfx942
ENGINE=hipkernel:Gfx942AttentionDense
```

Use [generator setup](../../../IngestorGenerator/README.md#setup). Authoring/mining
imports need the profile's rocKE library environment; production packaging uses
its selected compiler/wheel interpreter instead. Full artifact checking also needs
`rocm_kpack` and its dependencies in `$PY` (including `zstandard` and `msgpack`).
`--kpack-python-dir` supplies an import root, not missing Python dependencies.

**Resolve the interpreter by probing for the layout that exists.** `$PY` above is the
POSIX layout. A Windows virtual environment puts the same interpreter at
`.venv\Scripts\python.exe`, with or without the suffix, and a `.venv` may not be there
at all — it is not checked in and no step here creates one. Any interpreter that can
import the generator's dependencies serves; probe for the one present rather than
assuming a path, and treat every path on this page as a POSIX example to translate,
not a literal to copy. The same rule covers `$REPO`, `$BUILD`, `$INSTALL` and the
corpus roots.

**A tool that cannot run has not reported a negative.** A probe or inspection utility
may be absent on a given host or packaging even when the condition it reports is
satisfied, and an absent utility is silence, not a finding. Distinguish *tool
unavailable* from *condition false* before treating either as a gate failure. When a
prescribed utility is missing, observe the same facts by another means and record the
substitution — what you ran instead, and what it did and did not show — rather than
recording a failure or a pass. A gate discharged by substitution is discharged; a
check you could not run and did not substitute for is **unobserved**, which blocks its
gate without being evidence against the thing it was to observe. An unattended run
that conflates the two strands on a healthy host.

Verify source, build, install, corpus and output paths on the execution host.

**Decide whether the target device is local or scheduled before running anything in
this section.** When the device is in the host you are already on and directly
reachable, the staging paragraph below does not apply at all — there is no login host,
no staging hop and no allocation to acquire — and working through it anyway invents
both the work and a failure mode that do not exist here. Skip it entirely.

*Scheduled targets only.* A workspace symlink backed by login-local **`/var/tmp` is
invisible from compute nodes**, even when its link name is under shared storage. Use
shared source or stage the exact checkout/artifacts to compute-local scratch through
the active scheduler skill. Do not run GPU work on a login host.

Retain command logs and source/artifact, machine, device and allocation identities
under the workspace's evidence policy.

## Tools this skill assumes, and what to do without them

Several rules on this page need a fact that **no command in this tree computes**. Each
entry below names what is missing, what exists that is close to it, what to observe by
hand instead, and what the outcome of that observation is.

The outcome is never a pass. A fact derived by hand is a **recorded escalation**:
record the substitution, the inputs you read, the conclusion you drew and the revision
of the source you read it from, and carry it into the handoff as a finding the
requester owes a decision on — not as a discharged gate. A rule whose inputs cannot be
computed and were not substituted for is itself an escalation. Writing prose, evidence
or a gate verdict as though one of these commands had run and agreed is the failure
this section exists to prevent, and it is the same failure as reading a missing
utility as a negative result.

**1. Envelope declaration — absent.** Nothing declares, as data, the envelope the
engine's matcher enforces. The generator config's engine-level graph-match block
carries layout and discrimination documentation
(`IngestorGenerator/codegen/models.py:324-336`), not admitted ranks, dtypes per
operand role, shape relations, stride admissibility or aliasing rules, and nothing
diffs any declaration against the matcher body. *Instead:* reconstruct the envelope by
reading the `graph_match` and kernel-matcher bodies and write it into the run's
evidence as data — node type and count, admitted ranks, admitted dtypes per operand
role, required operand shape relations, stride admissibility, virtual and
pass-by-value disposition, aliasing rules, and every node field the matcher gates on —
with the source lines each item came from. *Outcome:* a recorded envelope, escalated
as unvalidated. Nothing checks that the matcher still agrees with it, so it is correct
only as of the revision you read.

**2. Coverage computation — partial.** `mine_shapes.py` supplies per-source totals,
the distinct count and the provenance split; it deliberately does not filter by what
the engine can serve. For rocKE only, `dispatch_parity.py --report-gaps` prints every
shape the dispatcher would not serve with its reason and the layer that refused —
a genuine per-input exclusion reason, but from the reference library's dispatcher
rather than from this engine's matcher, and it requires a profile. Nothing computes
per-source servable/excluded counts against a declared envelope, and for a direct-load
engine nothing computes them at all. *Instead:* take the totals from `mine_shapes.py`,
then classify each input by hand against entry 1's reconstructed envelope, recording
for every exclusion the specific axis that blocked it rather than a bare count.
*Outcome:* a hand-built coverage table naming the envelope revision it was computed
against, escalated as hand-derived. No command reproduces it and nothing recomputes it
when the matcher changes.

**3. Reference-capability lookup — partial.** The fact exists in the tree but not as a
query: the GPU reference's capability is one `Gpu<Op>Plan.hpp` per op family under
`dnn-providers/integration-tests/src/harness/gpu-graph-executor/detail/`, and the
`admits(request) -> (bool, str)` contract is real. No command answers "which executors
implement this operation". *Instead:* read the current family list and the
gpu/cpu/never-`auto` decision table in
[hipdnn-kernel-integration](../hipdnn-kernel-integration/RUNBOOK.md), then confirm it
against the headers actually present before choosing a mode. *Outcome:* the mode and
the evidence you chose it from, recorded. Where no executor implements the operation
this is not a mode to work around — it is [graph-contract.md](graph-contract.md)'s
missing-capable-numerics block, escalated.

**4. Workload fetch by digest — absent.** Content-addressed corpora are pulled with a
data-versioning client. No script here fetches an object by digest without that
client; `verify_golden_bundles.py` reports an unpulled pointer rather than resolving
it. *Instead:* where the client is unavailable, read the digest and size out of the
pointer file and fetch the object directly from the content-addressed remote, then
verify the bytes against the digest. The layout is documented with the remote path
shape in
[the bundle README](../../../../../../dnn-providers/integration-tests/integration-test-bundles/README.md).
A missing client is not evidence that the corpus is unavailable. *Outcome:* either the
corpus, fetched and digest-verified with the method recorded, or the source recorded
as unavailable naming what you attempted — escalated, never folded silently into a
smaller denominator.

**5. Corpus normalization — absent in the direction needed.** The migration scripts
run the other way: `place_bundles.py` converts captured standalone cases *into*
template+sweep form, and the template-instantiation primitive it round-trips through
(`expand` in
`dnn-providers/integration-tests/migration-scripts/bundle_utils.py:252`)
is internal to those scripts. Nothing renders a parameterized in-tree case tree into
the standalone per-graph form the measurement harness consumes. *Instead:* measure
only corpora already in standalone per-graph form, or expand the specific cases in
scope by hand and record which cases the expansion covered. *Outcome:* a corpus with
its construction method recorded. A case corpus you could not render is a source
excluded for tooling reasons and is reported as exactly that — never as unservable,
which is a claim about the engine.

**6. Runtime join — partial.** `sweep.py` builds and gates on an outcome ledger from
its own measurement output, and `reconcile_applicability.py --declines` consumes a
graph-name-to-reason JSON. Nothing assembles [workloads.md](workloads.md)'s complete
per-input ledger from measurement output *plus* engine logs, which is the input stage 7
requires. *Instead:* perform the join as the explicit evidence review stage 7 already
describes, preserving its distinctions — reasons absent from runtime evidence stay
absent, and offline policy is never substituted for them. *Outcome:* a hand-built
ledger. Stage 7's gate therefore cannot close unattended today; record that as the
stage-7 blocker in the handoff rather than as a limitation of the run.

**7. Candidate enumeration — absent.** `knob_sweep.py` is the nearest tool and
deliberately does not do this: its documented order is isolate, pair the survivors,
ship what survived, and it records the cross-product as the arm that bought nothing
(`IngestorGenerator/tools/knob_sweep.py:1-20`). It has no exhaustive mode for any
dialect. *Instead:* enumerate the legal candidate space by hand from the authored
source's specialization axes and entry 1's envelope, marking each cell with what
adding it costs — a descriptor entry, a rebuild, or new authored source. *Outcome:* a
recorded enumeration; where it exceeds the measurement window, record exactly which
cells went unmeasured and never present a partial sweep as a complete one. Making
exhaustive enumeration the default would contradict `knob_sweep.py`'s own documented
stance, which is a question for that tool's owner and not a decision to take inside a
run.

**8. Per-candidate selection report — absent.** `sweep.py`'s ledger is per-graph,
per-arm outcome and timing, and the
[sweep reference](../../../IngestorGenerator/tools/README-sweeps.md) documents
arm-versus-baseline aggregates. Neither answers, per candidate, which corpus subsets
it wins, by what margin, and whether it is selected anywhere at all. *Instead:* derive
that by hand from the ledger for the candidates actually under consideration.
*Outcome:* recorded per-candidate findings. A candidate may be dropped only against a
measurement that shows it selected nowhere; **not measured is not never selected**, and
dropping on that basis is an escalation rather than a decision.

## 1. Entry and early feasibility

Record the entry contract; for extend, inventory the installed baseline according
to [extend.md](extend.md).

**Inventory the packs before you choose a dialect.** Ask, in this order:

1. **Does an ingestor pack already exist for this operation, and is it yours?** Three
   packs exist — `ConvNative.cpp`, `PointwiseNative.cpp` and
   `BatchnormInferenceNative.cpp` under
   `dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/packs/`,
   registered at `IngestorPacks.cpp:16-26`. The answer turns on **ownership**, with one
   independent disqualifier on top of it:
   - **No pack for the operation** — layernorm, RMSnorm, resample and most requests.
     Those three are `hip_mlops_engine` plan builders with no ingestor pack. Create
     path.
   - **The only candidate is `PointwiseAdd` or `ConvFwd`** — still no, on their own
     merits: `PointwiseAdd` computes one element
     (`kernels/PointwiseAdd.cpp:11-12`), `ConvFwd` serves 6 of 1218 `ConvolutionFwd`
     bundle cases. Both are reference scaffolds, and a real kernel attached to either
     inherits a matcher, a geometry and an ABI chosen for a toy.
   - **The candidate is `BatchnormInference`** — the first legitimate yes, and it is
     conditional on the pack being yours. It is no scaffold: a full-tensor kernel with
     its own bounds guard (`kernels/BatchnormInference.cpp:64-94`), a computed grid
     (`BatchnormInferenceNative.cpp:642-647`), three io dtypes
     (`BatchnormInferenceNative.cpp:97-101`), nine shipped variants
     (`TestBatchnormInferencePacks.cpp:44-63`) and a matcher refusing 15 parameterized
     cases against 9 acceptances (`TestBatchnormInferenceMatchers.cpp:165-314`,
     `TestBatchnormInferenceMatchers.cpp:65-129`). Extending it means another block
     size, io dtype, architecture or variant *of batchnorm inference*, by whoever
     shipped it — never an unrelated kernel hung off it. Its open axes are exactly its
     limits: one proved architecture
     (`IngestorGenerator/configs/batchnorm_inference.yaml:49`) and no device-level
     integration test wiring it, with its 10 of 82 `BatchnormInference/Default` bundle
     cases mirroring its own unit-test shapes rather than an independent corpus. Note
     that batchnorm now has *both* an ingestor pack and the `hip_mlops_engine` builder
     still claiming the same single-node graph (`BatchnormPlanBuilder.cpp:371-374`,
     `BatchnormPlanBuilder.cpp:550-554`).
2. **If no, this is the create path.** Go to
   [hipdnn-kernel-integration](../hipdnn-kernel-integration/SKILL.md) and return here for
   corpus, sweeps, tuning and packaging once the pack exists.
3. **Only if yes** — a pack that is genuinely yours — select `direct_load` or
   `packaged`, and for `direct_load` select `embedded_source` or `hiprtc_file`: the first
   is embedded into the provider at configure time, the second ships as a source bundle
   beside the descriptors and needs no rebuild. See [hiprtc-mining.md](hiprtc-mining.md).

Establish that the graph is representable and has a capable independent numerical
reference using [graph-contract.md](graph-contract.md). A reference's skip is not
verification; unavailable semantics require an explicit scope/reference decision.

On the actual allocated execution host:

```bash
"$PY" "$GEN/tools/device_probe.py" \
  --mode early --arch "$ARCH" --sweep-root "$SWEEP_ROOT"
```

Early mode requires the requested device and an existing writable root. It ignores
inherited `INSTALL` and rejects `--install`; `$INSTALL` may name a future directory.
Exit 0 proves feasibility only, exit 2 an invalid invocation, and exit 1 **any** of the
checks failing — device, path or write — without saying which. For rocKE, confirm the
actual builder/spec and `(spec, *, arch)` interface; an unknown architecture inventory
needs source investigation.

**Exit 1 is not a statement that the device is absent.** The probe reads the device
through `rocminfo` and catches that utility being missing and the architecture being
missing in the same handler, appending both to one undifferentiated failure list
(`IngestorGenerator/tools/device_probe.py:64-68`), which the exit path turns into a
single `1` (`IngestorGenerator/tools/device_probe.py:83-86`). On a healthy host where
`rocminfo` simply is not on `PATH` — a packaging difference, not a device one — this
gate reports a device failure and an unattended run halts at the first thing it does.
Apply the tool-unavailable rule above: read the printed `FAIL` line, establish whether
`rocminfo` is on `PATH` at all, and where it is not, observe the device by another
means — the driver's own device nodes, or the device enumeration any installed runtime
exposes — then record what you ran, what it reported, and that the device half of this
gate was discharged by substitution. The writable-root half still owes its own
observation: one exit status covers both, so satisfying one does not answer the other.

For `hiprtc_file`, feasibility is a symbol question before it is a device one. The
target installation must already register every `match_symbol`, `graph_match`,
`dispatch_symbol` and score symbol the new set names, and the entry point must take the
same arguments in the same order as that pack's registered `IKernelDispatchHandler`
launches. `loadValidatedDescriptorSets` pre-flights those symbols and drops the whole
engine on any miss with one `LOG_ERROR`, which is indistinguishable from a healthy
decline at the API. Confirm both against the installed provider's native pack source
per [hiprtc-mining.md](hiprtc-mining.md). The pack's dispatch handler must additionally
call `buildIngestorKernelCode`; a handler that compiles `kernel.source.sourceFile`
directly serves `embedded_source` only and throws at `prepare()`.

**Gate:** feasible target/workspace, representable scope and capable reference, plus the
pack-inventory answer from the questions above.

For a **drop-in**: an installed pack whose symbols and launch ABI the new variants reuse,
and whose handler routes.

For the **create case** — any work adding a native symbol, which arrives here only for
its production half — each of the four hooks must name both the page that specifies it
and the artifact you owe for it, and the gate does not pass until every one has both:

| Hook | Specified by | Artifact you must produce |
|---|---|---|
| Engine `graph_match` | [native-pack.md](native-pack.md) §Roles, §Matching | The matcher body, plus the admitted-shape envelope it enforces |
| Kernel- or graph-scoped matcher | [native-pack.md](native-pack.md) §Roles, §Matching | The matcher body, plus the KMD fields it compares |
| `workspaceBytes` | [native-pack.md](native-pack.md) §Workspace, preparation and launch | The formula and the metadata field it reads, or a stated zero |
| `IKernelDispatchHandler` | [native-pack.md](native-pack.md) §Workspace, preparation and launch | `prepare()` with its geometry and defines, and `launch()` with its exact argument order |
| Native score, if declared | [native-pack.md](native-pack.md) §Roles | The ranking axis and its justification, or `heuristic: none` with no score anywhere |
| Registration | [native-pack.md](native-pack.md) §Registration and inventory proof | The `SymbolScope<Handle>` additions **and** the `IngestorPacks.cpp` row |

A missing dependency blocks its gate; host-only research may continue while a device
allocation is pending, but cannot discharge device proof.

## 2. Contracts, corpus and scope

Record [graph-contract.md](graph-contract.md)'s topology/UID edges and field
dispositions, then [rocke-mining.md](rocke-mining.md)'s applicability, specialization,
layout, geometry/workspace and ABI evidence. Direct-load engines obtain these facts
from HIP source without inventing a profile. Reuse unchanged extension contracts;
reopen any topology or feature the addition changes.

Inventory owner-published results, owner benchmark shapes and external graphs per
[workloads.md](workloads.md). `$SHAPES` is a JSON list of semantic requests; the
benchmark consumes actual graph JSON under `$CORPUS_DIR/<corpus>`. For the attention
miner, using the sources in scope:

```bash
"$PY" "$GEN/tools/mine_shapes.py" \
  --published /absolute/path/to/owner-results.csv \
  --graphs "$CORPUS_DIR/servable" --arch "$ARCH" \
  --include-windowed --out "$SHAPES"
```

Add `--rocke-bench <actual-benchmark-tree>` when applicable. `mine_shapes.py` mines and
deduplicates: it prints the rows found per source, the distinct count with duplicates
merged, and the by-source breakdown of what it wrote, and every emitted shape carries
its provenance. It deliberately **does not filter by what the engine can serve** —
that is the dispatcher's answer, not the corpus's — so it supplies the *total*
denominator and the provenance split, and no servable, covered or excluded figure.
Those you reconcile yourself. For rocKE, `dispatch_parity.py --report-gaps` below
prints every shape the dispatcher would not serve with its reason and the layer that
refused, which is the servable/excluded split for that path; for every other dialect
no tool computes it and the fallback in
[Tools this skill assumes, and what to do without them](#tools-this-skill-assumes-and-what-to-do-without-them)
applies. Do not discard window/sink or independent operand dimensions to fit the
request schema.

For rocKE, resolve the provisional baseline through its actual dispatcher:

```bash
"$PY" "$GEN/tools/dispatch_parity.py" --profile "$PROFILE" \
  --shapes "$SHAPES" --out "$CONFIG" --report-knobs --report-gaps
"$PY" "$GEN/tools/reconcile_applicability.py" \
  --profile "$PROFILE" --shapes "$SHAPES"
```

The second command is **offline applicability**, not runtime coverage or numerics.
Scope reference candidates to the kernel family/algorithm and required opt-in
selector. API failures are operational errors, never unsupported-shape evidence.

### Deciding scope

Record the feature/shape boundary, per-source coverage and exclusions, architecture,
engine identity, knobs and provisional baseline. Scope is then *derived* from the rules
below and from the record of which rule fired on which inputs. There is no approval
step: the record is the decision.

For every axis that excludes in-scope inputs, classify it:

| Class | Test | Disposition |
|---|---|---|
| A | Descriptor/matcher only — no kernel change, the authored source already compiles the value, and a capable independent reference covers it | **Widen** |
| B | Requires a kernel or source change | Out of scope here; emit an upstream handoff naming the axis and the coverage it would unlock, then continue with what is admitted |
| C | No capable independent numerical reference covers it | Out of scope, recorded as reference-blocked |

**Prefer coverage — and pay for it in the same breath.** Widen every class-A axis that
admits additional in-scope inputs; a widened axis is a **candidate** axis, not a served
one, and every value it admits owes reference-verified numerics at §5 and measurement
at §6 before it may ship. **Widening adds candidates, never claims.** These two are one
rule and are not separable: widening without the debt it incurs ships unproven numerics
under the engine's identity, which is the failure the second half exists to prevent.

A legal cross-product is still not a measured shipping set, and pruning an axis that
was never measured is the same error inverted. An axis with two or more candidates and
no timing evidence sends **every** candidate into §6; prune only against measurements.
An axis with exactly one candidate is genuinely single-candidate and owes no
comparison.

Include every corpus source with at least one servable input. Exclude the rest,
recording for each exclusion both the reason and the specific axis that would admit it
— that axis is itself a class A/B/C input above, so exclusions feed the widening
decision rather than merely documenting a gap.

**Coverage computation does not exist, so this is how the classification is made.** No
command in this tree computes per-source servable/excluded counts, and nothing declares
the envelope they would be computed against; entries 1 and 2 of
[Tools this skill assumes, and what to do without them](#tools-this-skill-assumes-and-what-to-do-without-them)
state the substitution and its standing, and are not restated here. Classify by reading
the engine's `graph_match` and kernel-matcher bodies against `mine_shapes.py`'s
per-source totals, distinct count and provenance split, recording for every judgement
the source lines it rests on. What you produce is a hand-derived table carrying the
engine revision it was computed against, and it carries that section's outcome: a
recorded escalation, not a discharged fact. An axis you cannot classify at all is an
escalation — never a silent widening, and never a silent exclusion.

### Matcher-versus-reference parity

Diff the declared envelope — or, until a declaration exists, the reconstruction
[graph-contract.md](graph-contract.md)'s §Field dispositions and kernel mapping
requires — against the predicates the chosen numerical reference enforces for the same
operation. **Any predicate the reference enforces that the engine's matcher does not is
a finding**: the engine admits inputs its own oracle refuses, so those inputs are
validated by nothing, and the absence of an in-tree case exercising them is what hides
the hole rather than evidence there is none. This check needs no tool — both sides are
source you are already reading — and it is required, not advisory. Resolve every
finding before the gate: either enforce the condition in the matcher, or record why it
cannot arise.

**Gate:** recorded scope derived from the rules above, complete per-source denominators
with provenance, every excluded axis classified, every parity finding resolved, and no
escalation condition met.

### When to escalate

Escalate to the requester **only** when:

- no corpus source has a servable input and no class-A widening exists;
- the included corpus leaves a declared-envelope axis with **no measurable input** — a
  corpus that cannot exercise an axis cannot distinguish candidates along it. This
  condition is envelope-relative on purpose and is deliberately not a key count:
  sufficiency depends on the operation, the declared envelope and what the kernel
  admits, so there is no global floor to compare against;
- no capable independent numerical reference covers the admitted features;
- two rules select conflicting scopes, or a rule's inputs cannot be computed and the
  hand substitution above does not supply them either.

Anything else: decide, record which rule fired and the inputs it consumed, and continue.
A recorded rule application is the audit trail approval used to provide.

## 3. Generate, implement and splice

Generate into an empty scratch directory, never over the live engine:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" \
  --output-dir "$GENERATED" --dry-run
"$PY" "$GEN/generate.py" --config "$CONFIG" --output-dir "$GENERATED"
"$PY" "$GEN/tools/verify_variant_sets.py" --mode structural \
  baseline "$GENERATED/descriptors"
```

Review the finalized inventory after deduplication. Resolve KDP → UED → KMD by UUID;
tuple identity includes schema types/defaults and effective architecture overlap.
Structural mode reports compiled agreement as `NOT CHECKED` and can exit 0 with
unrun checks. `--profile` optionally selects the bundle and supplies vocabulary;
it cannot supply compiler evidence. Full artifact checking belongs after packing.

Implement [native-pack.md](native-pack.md)'s referenced hooks and behavioral tests.
Extensions remap scratch references and copy only additions per [extend.md](extend.md),
including consumer IDs in shared KDP or per-UKD specialization declarations. Never
edit packed evidence to match a changed descriptor; rebuild from authored inputs.

Apply fragments to their actual consumers, preserving unrelated entries:

| Splice | Direct-load | Packaged |
|---|---|---|
| `HIPDNN_DESCRIPTOR_FILES` | Authored runtime descriptors | Never unlowered authored descriptors |
| `HIPDNN_INGESTOR_PACK_KERNELS` | Actual source-file stems | Not applicable |
| Engine `target_sources` | Native implementation | Native implementation |
| `IngestorPacks.hpp` declaration and `.cpp` table | Both | Both |
| Engine test `target_sources` | Applicable tests | Applicable tests/census |
| `HKP_CENSUS_TEST_SUITES` in `hkp_register_census_tests()` | Not the direct-load inventory gate | Literal generated packaged census suite name |

The census list lives in
`dnn-providers/hip-kernel-provider/descriptor-packaging/cmake/HkpPackaging.cmake`.
Appending the literal `Test<Name>Packs` suite is what creates the census obligation;
an empty list registers none.

Stage packaged authored descriptors under the configured
`HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT`, preserving their generated relative
subpath. Update that staging after every regeneration and compare content/identities,
not only counts. A fragment file is not evidence that its splice was applied.

Set `SCHEMA` to the actual operation `.fbs` and `NATIVE_SOURCE` to its implementation;
pass all relevant source files and repeat for each schema in a fusion:

```bash
"$PY" "$GEN/tools/field_audit.py" "$SCHEMA" "$NATIVE_SOURCE"
```

Exit 0 covers lexical accessor references only. It is an inventory of where a schema
field's accessor is *named* in the native source: a reference it counts may sit in
dead code or fail to implement the field's semantics, so exit 0 proves neither
consumption nor correctness. Review semantic dispositions separately. For an unspliced
tree, check placeholders with:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" \
  --output-dir "$GENERATED" --check-placeholders
```

For a packaged splice, set `DESCRIPTOR_STAGING_ROOT` to the production tree holding
the generated descriptor subpath, then check all destinations below. For direct-load,
omit the third root: its descriptors already live under the engine root.

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" --output-dir "$GENERATED" \
  --check-placeholders \
  --emitted-root "$PROVIDER/src/engines/kernel_ingestor_engine" \
  --emitted-root "$PROVIDER/src/tests/engines/kernel_ingestor_engine" \
  --emitted-root "$DESCRIPTOR_STAGING_ROOT"
```

With `--emitted-root`, `--output-dir` is required but not searched. The check covers
all emitted shippable files at engine-specific paths: nonexistent roots, missing
files and ambiguous matches fail. An unrelated same-basename file cannot satisfy it.
If the profile declares a launch-surface audit, also run:

```bash
"$PY" "$GEN/tools/launch_surface.py" "$PROFILE" --check
```

**Gate:** final authored inventory, completed hooks and source/test splices, no
selected-path placeholders, and reviewed structural/field/ABI results. None proves
native loading or numerical dispatch.

## 4. Build, pack, install and prove the host boundary

Configure from `$REPO` using `hipdnn-superbuild`, with
`CMAKE_INSTALL_PREFIX="$INSTALL"`, `HIPDNN_ENABLE_KERNEL_INGESTOR=ON` and
`HIPKERNELPROVIDER_ENABLE_TESTS=ON`. SDPA needs `HIPDNN_ENABLE_SDPA=ON` consistently
in SDK and provider. Packaged producers additionally need
`HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE` and/or `..._HIP`, the authored source
root, and the [packaging dependencies](../../../../../../dnn-providers/hip-kernel-provider/descriptor-packaging/README.md).
`HIPKERNELPROVIDER_ENABLE_ROCKE` does not replace the production producer switch.

**Test inputs are copied into the build tree at configure time.** A bundle case
imported after your last configure is absent from the build tree and from the
installation, and the test that would consume it never registers — a silent absence,
not an error. Import corpus inputs first, then configure and build; if you have already
configured, reconfigure. The CMake mechanism and the import command it governs are
stated once, in [hipdnn-kernel-integration](../hipdnn-kernel-integration/RUNBOOK.md)
§6 — read it there rather than a second copy here. No gate in this stage detects an
input that never arrived: a corpus that quietly lost a case narrows every denominator
downstream while every check on this page still passes.

Build the provider, validator and required test targets through the configured
superbuild. For packaged engines, run `hkp_packaging_product` after the full build
and after any reconfigure. Require the final staged descriptors, not merely an
up-to-date packaging stamp or an intermediate archive. Then install:

```bash
cmake --install "$BUILD" --prefix "$INSTALL"
```

**Install before provider-wide integration runs.** ASM SDPA loads loose `.co`
kernels from the configured install prefix by default; building its executable or
copying kernels into a build tree does not satisfy that path. Its explicit runtime
override is a separate test setup, not evidence of the final installation. Keep the
configured prefix aligned with `$INSTALL`, rather than relocating only this command.

**The installed test binary takes a runtime override for its input-data root.** It is
`--gd` / `--golden-data-dir`, or the `HIPDNN_TEST_GOLDEN_DATA_DIR` environment
variable, defaulting to `<exe>/../lib/integration-test-bundles/`
(`dnn-providers/integration-tests/src/main.cpp:132-135`). That lets the **installed**
binary consume an arbitrary input tree with no reconfigure and no rebuild, which is the
cheap way to exercise a candidate corpus against the real installation rather than
waiting on the configure-time copy above. Resolve the flag from the binary's own
argument parser rather than trusting this name to have survived — an unrecognised
override is a silent default-root run, not an error you will notice. It does not
replace shipping the inputs: **an override proves the corpus runs, not that it is
installed.** Label every run that used one as override-derived, and do not let it
discharge this stage's installed-boundary gate.

Set `FINAL_DESCRIPTOR_ROOT` to the actual installed per-arch shard (or the
arch-independent direct-load tree). Resolve `VALIDATOR` to the built
`hipdnn_validate_descriptors` executable and validate the runtime dialect:

```bash
"$VALIDATOR" "$FINAL_DESCRIPTOR_ROOT" --expect-engine "$ENGINE" --json
```

For packaged output, check each selected architecture using the producing-build
record, not today's imported producer:

```bash
"$PY" "$GEN/tools/verify_variant_sets.py" --mode full --arch "$ARCH" \
  --profile "$PROFILE" final "$FINAL_DESCRIPTOR_ROOT"
```

Omit `--profile` when neither bundle selection nor extra vocabulary needs it. Add
`--kpack-python-dir <dir>` if required by the reader environment. Interpret outcomes
separately:

- Use full checking only for **packed `kind: kpack` descriptors**; confirm the input
  dialect instead of inferring it from exit 0. Missing declarations, mismatches
  and required checks `NOT RUN` block acceptance, including missing vocabulary.
- A packed kernel declaring no specialized `metadata_fields` and carrying no
  `effective_spec` reports **`NOT VERIFIED HERE`**. It neither fails the gate nor
  gains compiled-specialization proof. Full-mode exit 0 does not certify those
  binaries; AOT HIP specialization remains outside this check.
- For declared specialization, the per-kernel record binds the current descriptor,
  schema, metadata, architecture and named payload bytes. Generation supplies the
  declaration, not that compiler-owned evidence; see [rocke-mining.md](rocke-mining.md).

### Packaged census: direct native CTest entries

Run real provider registration/loading and inventory checks in fresh processes.
The census **covers packaged engines only**, and it is a direct native obligation
with no Python launcher and no XML guard. For each literal suite in
`HKP_CENSUS_TEST_SUITES` and each configured packaging architecture, CMake registers
an independent test `hip-kernel-provider-hkp-census-<arch>-<suite>` that invokes

```text
hip_kernel_provider_tests --gtest_filter=<suite>.*
```

with `HIPDNN_TEST_CENSUS_SUITE=<suite>`, `HIPDNN_TEST_EXPECTED_ARCH=<arch>` and
`HIPDNN_DESCRIPTOR_DIR=<descriptor-build-dir>/<arch>`. The architecture comes from
the configured packaging list, never from a detected device or from the descriptors
themselves. Set `CENSUS_SUITE` to the generated suite name and `PROVIDER_BUILD` to
the provider's own binary directory (`$BUILD/dnn-providers/hip-kernel-provider` in
the superbuild layout), then run every requested arch's entry:

```bash
ctest --test-dir "$PROVIDER_BUILD" --no-tests=error -V \
  -R "^hip-kernel-provider-hkp-census-${ARCH}-${CENSUS_SUITE}$"
```

Those entries bind the build-tree shard. For **final installed packaged evidence**,
run the same suite against the installed shard by supplying the same explicit
environment to the installed binary:

```bash
HIPDNN_TEST_CENSUS_SUITE="$CENSUS_SUITE" \
HIPDNN_TEST_EXPECTED_ARCH="$ARCH" \
HIPDNN_DESCRIPTOR_DIR="$FINAL_DESCRIPTOR_ROOT" \
"$INSTALL/bin/hip_kernel_provider_tests" --gtest_filter="${CENSUS_SUITE}.*"
```

Adjust the binary path for a nondefault install bindir. A nonempty
`HIPDNN_TEST_CENSUS_SUITE` activates the native strict guard: before default-root
setup it rejects an empty expected arch and a missing, empty or nonexistent explicit
descriptor root, and it rejects an absent or empty named suite. Every registered
case in that suite must execute and pass **without skipping in every iteration**,
with at least one completed iteration. Disabled, filtered-out, sharded-out, failed
or skipped cases, list-only invocations and zero iterations cannot satisfy it, and
repeated partial runs do not accumulate coverage. Normal invocations without the
variable keep ordinary GoogleTest filtering and skip behavior, and the production
runtime's descriptor-root fallback is unchanged.

Declared suites with a missing test target or an empty configured architecture list
are configuration errors. Tests built OFF, or no declared suite, yields **no census
evidence**: that is absence, not a pass. Direct-load engines use their ordinary
unit/inventory suites against their arch-independent descriptor tree and are not
covered by a packaged-census pass; a generated inventory suite invoked directly
still requires an explicit expected arch and descriptor root. Check retained
extension inventory and heuristic-disabled score absence per
[native-pack.md](native-pack.md).

### Drop-in install: `hiprtc_file` variants, no build and no packaging

A `hiprtc_file` variant set of an **already-installed** pack skips this stage's build,
packaging and `cmake --install` entirely: nothing native changes, so nothing is
compiled here. Generate into an empty destination as in stage 3, then copy the
generated `descriptors/<pack>/` directory **whole** — its descriptor JSONs plus the
staged bundle directory — into a drop-in root, and point the installed process at it:

```bash
DROPIN_ROOT=/absolute/path/to/drop-in-descriptor-root
cp -r "$GENERATED/descriptors/<pack>" "$DROPIN_ROOT/"
export HIPDNN_DESCRIPTOR_RUNTIME_DIR="$DROPIN_ROOT"
```

That root is additive to the shipped tree, and a descriptor redefining an installed
id is refused, not honoured. To add variants to an already-installed engine rather than
ship a new one, copy only the `.kdp.json` and its bundle, with `engine`, `dispatch` and
`matchers` rewritten to the installed set's uuids and fresh uuids for the KDP and its
kernels — but that shape is served under the **installed** engine's id, so stage 5 cannot
attribute a dispatch to it; a full set with its own UED over the installed pack's symbols
can. Choose deliberately: [hiprtc-mining.md](hiprtc-mining.md) §Scope. `packs/`, `tests/`
and `fragments/` are the rebuild-requiring half and are
not part of a drop-in. Nothing is picked up until the consuming process restarts —
discovery is memoized per process — and a restart is also what clears the
process-lifetime compile cache after a bundle source is edited in place.

Validate with a `hipdnn_validate_descriptors` built from a provider that knows
`hiprtc_file`; an older one rejects `bundle` and `defines` as unknown `kernel_source`
keys, which is a validator-vintage failure and not a defect in the tree. Pass both
roots in runtime order, since a KDP-only drop-in resolves its cross-references against
the installed tree and validates as nothing on its own:

```bash
"$VALIDATOR" "$FINAL_DESCRIPTOR_ROOT" "$DROPIN_ROOT" --expect-engine "$ENGINE" --json
```

Packaging checks and the packaged census do not apply: there is no `kind: kpack`
descriptor and no per-arch shard. Proof that a drop-in works is stage 5's device
dispatch from the unchanged installation, not this copy.

**Gate:** current installation — or, for a drop-in, an unchanged installation plus a
validated drop-in root — artifact checks at their stated strength, and real
registration/loading plus applicable inventory/census checks. Report `NOT VERIFIED
HERE` separately. Neither the structural validator's stubs nor host loading proves
dispatch.

## 5. Baseline device proof from the installation

**Derive the reference mode from capability; never assume one.** Before running
anything in this stage, resolve which reference executors actually implement the
operation under test, and select the mode from that capability. Which op families the
GPU reference implements, the executor headers that are its source of truth, and the
gpu/cpu/never-`auto` decision table are maintained as one copy in
[hipdnn-kernel-integration](../hipdnn-kernel-integration/RUNBOOK.md): read them there
rather than transcribing a list that changes whenever a family is added. A reference
that does not implement the operation **declines every case and can still exit zero**,
reporting a passing run in which nothing was verified. Zero selected, or every case
declined, is a gate failure regardless of exit status — which is why the counts below
are recorded separately rather than collapsed into one "passed" number.

On the allocated target host:

```bash
"$PY" "$GEN/tools/device_probe.py" --mode installed --arch "$ARCH" \
  --sweep-root "$SWEEP_ROOT" --install "$INSTALL"
```

A missing/invisible installation fails even if early feasibility passed, and this
invocation carries stage 1's caveat unchanged: exit 1 still cannot distinguish a
missing `rocminfo` from a missing device, so read the printed `FAIL` line before
concluding anything about the host. Use
`hipdnn-superbuild-test` discovery with component **`hip-kernel`**:

```bash
"$PY" "$REPO/projects/hipdnn/tools/ai/skills/hipdnn-superbuild-test/scripts/discover_test_targets.py" \
  --build-dir "$BUILD" --component hip-kernel --scope external-integration
```

Do not accept the helper's first provider-prefixed command as exact-engine proof.
The provider's default installed CTest root is **`$INSTALL/bin/hip_kernel_provider`**,
not `$INSTALL`; substitute the configured bindir if customized. For gfx942 dense:

```bash
CTEST_ROOT="$INSTALL/bin/hip_kernel_provider"
DEVICE_TEST=hip_kernel_provider_gfx942_attention_dense_gpu_ref_integration_tests
ctest --test-dir "$CTEST_ROOT" -N -V -R "^${DEVICE_TEST}$"
```

Require exactly that registration. Inspect its command/config for
`hipkernel:Gfx942AttentionDense`, installed executable/plugin/config paths and the
intended quick/standard selection. Then execute:

```bash
ctest --test-dir "$CTEST_ROOT" --no-tests=error -V -R "^${DEVICE_TEST}$"
```

Keep verbose output in the retained log. **`--output-on-failure` hides passing
suites' case counts; an all-skip suite can still report CTest PASS.** Record selected,
served, skipped/declined and failed counts and observed reasons. Missing registration,
zero selected, all-skipped support, wrong engine/path or numerical failure blocks
this gate. Resolve the exact UED name → engine ID from the installation; a prefix
or registry listing is not dispatch attribution.

Use nontrivial inputs for quick feature breadth and bounded standard numerical
depth. Exercise required declines separately; another winning engine must not hide
them. NaN/unwritten output is a failure, not a tolerance adjustment.

Where the corpus you must exercise is not the one installed beside the binary, point
the installed binary at it with `--gd` / `--golden-data-dir` (env
`HIPDNN_TEST_GOLDEN_DATA_DIR`) instead of reconfiguring for it. Record that the run
used an override and which tree it read: that proves those inputs run against this
installation, and nothing about whether the installation ships them.

Extensions must select the addition explicitly. The disposable pointwise example
adds HALF/block_size=256 to ADD, preserves MUL/SUB and changes ADD's expected census
from three to four. Select HALF/256 on logical dims `{1,1,1,1}`, check the actual
`hipkernel:Pointwise` plan and arithmetic, and retain old ADD/MUL/SUB and required
multi-element/two-node declines. Its source computes one element; a default-FLOAT
pass or this one-element smoke proves no arbitrary-size coverage.

**Gate:** intended-engine dispatch, capable-reference numerics and complete case
accounting on `$ARCH`.

## 6. Tune the runnable baseline and rebuild the final selection

For rocKE, propose bounded candidates with the actual profile. `knob_sweep.py`
implements one staged order — isolate each knob against the dispatcher's own value,
pair only the knobs that moved, ship what survived — and it never measures. `--plan`
resolves the corpus through the dispatcher and prints the partition: the knobs the
dispatcher varies per shape, the surviving candidates with any declared hazard
attached, and every excluded knob with its reason. Three exclusions are automatic
(`IngestorGenerator/tools/knob_sweep.py:277-288`): a knob whose verdict is already
settled in the kernel's own history, unless `--include-settled`; a knob the dispatcher
varies per shape, which is a production axis rather than a sweep candidate; and a knob
declaring fewer than two values. Read that output before generating arms — and note
that a knob the profile never declares is not excluded, it is invisible.

```bash
"$PY" "$GEN/tools/knob_sweep.py" --profile "$PROFILE" --shapes "$SHAPES" --plan
"$PY" "$GEN/tools/knob_sweep.py" --profile "$PROFILE" --shapes "$SHAPES" \
  --isolate --out-dir "$ARM_CONFIG_ROOT"
```

Set `ARM_CONFIG_ROOT` to an experiment-owned directory. Generate, implement/splice,
build, pack, install and check every supported arm before measurement. Keep separate
install/output trees; never mutate the baseline and call it a comparison.

Populate `configs/sweep-isolation.sweep.yaml.example` using actual corpus counts,
installed KDP-entry counts and proven engine identity. YAML paths resolve relative
to the YAML file; there is no shell/environment interpolation. Set `SWEEP_CONFIG`
to its absolute path, then run:

```bash
"$PY" "$GEN/tools/sweep.py" --config "$SWEEP_CONFIG"
```

Follow [workloads.md](workloads.md)'s one-session, baseline-first order, gated
warmup, rounds, isolated caches and separate correctness requirements. The sweep
selects the discovered engine ID through the benchmark's `--engine` option; do not
override its phase-owned arguments.

For rocKE, investigate measured survivors. Set `PAIRWISE_KNOBS` to comma-separated
surviving knob names and `PAIRWISE_CONFIG_ROOT` to their config output directory,
then generate, build and install the supported pairwise arms:

```bash
"$PY" "$GEN/tools/knob_sweep.py" --profile "$PROFILE" --shapes "$SHAPES" \
  --pairwise "$PAIRWISE_KNOBS" --out-dir "$PAIRWISE_CONFIG_ROOT"
```

Obtain selection approval from coverage, correctness and per-corpus measurements.
For the rocKE shipping cross, `APPROVED_KNOBS_JSON` contains the approved JSON object
of knob value lists, **not a filename**:

```bash
"$PY" "$GEN/tools/dispatch_parity.py" --profile "$PROFILE" --shapes "$SHAPES" \
  --knobs "$APPROVED_KNOBS_JSON" --out "$FINAL_CONFIG"
```

Repeat **stages 3–5** with the final config and a new empty generation destination.
Neither an isolation arm nor an old install certifies the regenerated shipping set.
An explicitly untuned extension may retain its approved baseline selection, but
still requires final installed artifact and corpus proof.

**Gate:** justified selection and revalidated final installation.

## 7. Final corpus proof and runtime reconciliation

Run the final installed artifact through a fresh-output YAML sweep with
`correctness.enabled: true`. Require the exact phase key set and validated
`SWEEP_DONE`. Resume may reuse content-valid evidence diagnostically; it does not
make separate sessions a single comparative cohort. `SWEEP_TIMING_ONLY` is not
final success.

Harvest final phase results and available engine logs into [workloads.md](workloads.md)'s
complete per-input outcome ledger. Preserve semantic identity, all original
corpus/source/graph occurrences, current phase/input fingerprints, exact engines
and result/log locations. Distinguish served, explicitly declined, execution-error,
missing and ambiguous outcomes. A missing timing row is not a decline reason.

Join within each corpus/phase before making its graph-name-to-reason JSON. Reject
missing outcomes, duplicate/ambiguous names and mismatched fingerprints. Runtime
reasons unavailable in the evidence stay unavailable; do not reconstruct them from
offline policy. This is an explicit evidence review, not a promised automatic
matcher-reason extractor.

For rocKE, set `CORPUS_SHAPES` and `RUNTIME_DECLINES` to that corpus's requests and
complete runtime-derived mapping, then reconcile:

```bash
"$PY" "$GEN/tools/reconcile_applicability.py" --profile "$PROFILE" \
  --shapes "$CORPUS_SHAPES" --declines "$RUNTIME_DECLINES"
```

Without the complete join, label reconciliation **offline only**. Investigate
reference-only supported rows and obtain explicit scope decisions for exclusions;
escape flags cannot excuse broken reference APIs or unexplained gaps. Direct-load
pointwise instead uses its explicit one-element corpus, installed engine results
and independent arithmetic/reference correctness, with `exclude_tensors: none`.

Report full per-source denominators, served/reference-validated populations and
all remaining outcomes separately from timing. Include geomean-of-ratios,
time-weighted sum-baseline/sum-arm, round drift and byte-identical controls chosen
from artifact hashes.

**Gate:** zero wrong answers, complete final-runtime accounting, and no missing,
ambiguous, erroneous or unexplained in-scope outcomes. Changed installed artifacts
invalidate old evidence and return to stages 3–5.

## 8. Handoff

Report [SKILL.md](SKILL.md)'s completion evidence and exact limitations. Keep
experiment copies/probes disposable and retain their inputs/results under the
workspace evidence policy. For blocked work, name the last completed stage and
missing prerequisite; do not substitute a proposed command or queued job for proof.
