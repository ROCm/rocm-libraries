# Runbook — a kernel into hipDNN as an ingestor engine, end to end

**Read this file first and drive from it.** The other files are reference material you are
sent to at specific steps, and each tells you what you owe before you return. This one is
the sequence, and it is written to be executed rather than interpreted.

Every command is copy-paste with the variables below substituted. Every step opens with a
`Produces` / `Gate` / `Typical time` contract and ends in a **GATE** — a command whose
output you check before continuing. A step whose gate you have not run is a step you have
not finished.

**Scope: this runbook is written for the `packaged` dialect** — a rocKE builder, lowered
at build time by `hkp_pack`. That is the common case and the one with the most traps. For
a **`direct_load`** bundle (a `.cpp`/`.hip` the provider embeds, `kind: embedded_source`,
compiled at plan-build time on device) the sequence is the same but four steps differ:

- **Step 2b** does not apply — there is no Python to mine. The graph contract (2a) still
  does, in full: it is about hipDNN's side, not the kernel's.
- **Step 4**: `kernel_source` is `{kind: embedded_source, source_file, entry_point}`, one
  pair per kernel entry point. Candidate KMD fields come from what the kernel is templated
  or `#define`d on — one field per axis.
- **Step 5** has one rung, not three: there is nothing to pack, so run
  `hipdnn_validate_descriptors` directly against the authored tree, adding
  `--native-source <pack.cpp>` to cross-check the stub's symbol constants against the
  descriptors.
- **Step 7a**: all five splice points apply, including points 1
  (`HIPDNN_DESCRIPTOR_FILES`) and 2 (`HIPDNN_INGESTOR_PACK_KERNELS`), which a packaged
  bundle must never touch. A kernel missing from point 2 fails at plan-build time with a
  missing embedded source.

Everywhere else, read `packaged` instructions as applying to you.

## Set these once, at the top of your run

```bash
REPO=<path to rocm-libraries checkout>          # the worktree you are working in
PROVIDER=$REPO/dnn-providers/hip-kernel-provider
GEN=$REPO/projects/hipdnn/tools/IngestorGenerator
BUILD=$REPO/<build-dir>                          # existing build, or one you configure at step 7
INSTALL=$REPO/<install-dir>                      # `cmake --install $BUILD --prefix $INSTALL`;
                                                 # steps 8c and 9c run the registered target
                                                 # from HERE, not from the build tree
PROJECT=hip-kernel-provider                      # the provider's own project() name, which
                                                 # prefixes its ctest targets. Confirm with
                                                 # `grep -m1 '^project(' $PROVIDER/CMakeLists.txt`

MODULE=kernels/<arch>/<module>.py                # e.g. kernels/gfx950/attention_dense.py
BUILDERS=<builder fn>[ <builder fn>...]          # ENUMERATE, never guess: there is no
                                                 # naming convention. Real modules define
                                                 # build_unified_attention_3d_tiled,
                                                 # build_gfx942_4warp_gqa, ... :
                                                 #   grep -nE '^def build_' $PROVIDER/rocke/library/$MODULE
                                                 # A module exposing SEVERAL builders may be
                                                 # one engine or several -- a split-KV design
                                                 # needs its segment AND reduce kernels packed
                                                 # together. That is ONE PACK PER BUILDER in
                                                 # one engine: each pack carries its own
                                                 # `builder:`, so no schema change is needed.
                                                 # Decide at 1a, confirm the split at step 3.
ARCH=<gfxNNN>                                    # the ONE arch this engine ships for
ENGINE=hipkernel:<CamelName>                     # scoped name; unscoped is rejected
SLUG=<arch>_<op>                                 # e.g. gfx950_attention_dense
OPTABLE=<Op>Attributes                           # the .fbs table, e.g. SdpaAttributes
SHAPES=<path to shapes.json>                     # the request corpus step 4a resolves;
                                                 # a JSON LIST of request-field mappings
CORPUS_DIR=<path to staged graph corpora>        # DIFFERENT: one dir per corpus name,
                                                 # each holding real graph BUNDLES, which
                                                 # is what sweep.sh runs at step 9a
SWEEP_ROOT=<path the DEVICE machine can write>   # install trees, corpora, sweep output.
                                                 # Must be visible from the machine that
                                                 # runs, not just from your shell
LOGIN=<submit host>                              # ONLY if you reach the device through a
                                                 # scheduler; unused on a local box
```

### Two Python environments, and they are not interchangeable

**The generator** runs from the tool's own venv, per `IngestorGenerator/README.md` — the
same convention as its sibling `DescriptorGenerator`. Its deps are declared in
`requirements.txt`; create it once:

```bash
cd $GEN && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Then invoke the generator as `$GEN/.venv/bin/python generate.py …` — never bare
`python3`, which will not see Jinja2.

**The packer and the introspector** run on **system `python3`**, and take their imports
from `PYTHONPATH` rather than a venv:

```bash
export PYTHONPATH=$GEN:$PROVIDER/descriptor-packaging/python:$PROVIDER/rocke/library:$PROVIDER/rocke/platform/python
```

`hkp_pack` additionally needs `msgpack` and `zstandard` (the kpack reader's own deps) and
`rocm_kpack`. On a provisioned dev box these are already present — `msgpack`/`zstandard`
system-wide, and `rocm_kpack` at **`/opt/rocm-kpack/python`**, which is what you pass as
`rocm_kpack_dir`. Check before assuming, since this is environment-dependent:

```bash
python3 -c "import msgpack, zstandard; print('kpack reader deps OK')"
ls -d /opt/rocm-kpack/python/rocm_kpack || find / -maxdepth 6 -name rocm_kpack -type d 2>/dev/null
```

The tool venv deliberately does **not** inherit system site-packages, so it has Jinja2 but
not `msgpack`/`zstandard`; system `python3` is the reverse. That split is fine — each
environment serves the step that needs it. Do not try to make one venv do both, and do
not `pip install --target` into the worktree: PEP 668 blocks a plain `pip install` on
these boxes and a stray dep directory is not the project's convention.

---

## The sequence at a glance

Every step below opens with the same three-line contract, so you can always tell whether
you are finished:

```
Produces:      the artifact that must exist on disk
Gate:          the command whose output you check
Typical time:  how long this usually takes
```

`Typical time` is not a target — it is a stall detector. **At 4× the estimate, stop and
write down what you have**, mark the uncertain parts, and move to the next step. A step
that produced no file is not a step in progress; it is a stall, and the cure is always the
same: write the artifact, however incomplete, and continue.

`GPU` marks the steps that need a device. Everything else is host-only and can run while
you wait for an allocation — a host-only check scheduled behind a device step is a
sequencing bug, not a dependency.

| Step | GPU | Produces | Gate |
|---|---|---|---|
| 0 | — | Environment ready, dialect stated | `.venv` present; rocKE ⇒ `packaged` |
| 1 | 1c | Feasibility verdict | Reference **and** hardware both reachable |
| 2a | — | `graph_contract.md` | File exists; op matched; disagreement table has rows |
| 2b | — | `mining.md` | File exists, every constraint row has a verdict |
| 3 | — | Batch message | Sent **and you proceeded on the stated defaults**; do not wait |
| 4 | — | `config.yaml`, descriptors | `generate.py` exit 0; kernel count = agreed set; **staged copy count == generated count** |
| 5 | — | Packed + validated tree | `success: true`, 0 ERROR, desk check clean |
| 6 | — | Native pack | `generate.py --check-placeholders` exit 0 |
| 7 | — | Built, packed, staged | Engine id in `hipdnn_list_engines` |
| 8 | **yes** | Tests + an engine-pinned CI target | A real graph dispatched and matched a reference on `$ARCH`, and the shipped `dnn-benchmarking` workloads triaged (8e) |
| 9 | 9a, 9c | Post-integration verification | The SHIPPED set swept over a real corpus, the integration suite run, every flagged graph triaged to a named cause, zero unexplained |
| 10 | — | Report | All ten stages named, by number |

**9b-0 is host-only and is the gate for step 9** — run it as soon as you have declines to
reconcile, which is before the 9a sweep, not after it. It is written after 9a only
because its input is easiest to describe there.

**This runbook is meant to be executed without supervision.** Every step above has a
command whose output decides pass or fail, and every judgement call has a documented
default (step 3). Two things stop a run and only two: an operation the graph schema cannot
express, and target hardware you cannot reach. Everything else you decide, record as an
assumption, and carry to step 10. If you find yourself waiting, you have misread step 3.

**An unverifiable feature is a stop, and the decision is the human's — but state the
options rather than only the stop.** When a reference executor declines a feature your
kernel serves (paged KV is the recurring one), you cannot prove correctness for it, so
step 8's gate cannot pass for those graphs. Three responses are legitimate, in this
order of preference:

1. **Ship the rest, decline that feature, and record it** — the default. The engine
   serves what it can prove; the gap is named in the step-10 report.
2. **Repair the shared reference executor.** This is **in scope** if the repair is
   bounded and you can state it in a sentence. Shared harness code is not off limits
   merely because it is shared — an ingestor engine that cannot be verified is not
   finished, and the reference is the thing making it unverifiable.
3. **Stop and escalate** when the repair is open-ended, or the feature is the point of
   the integration rather than an edge of it.

Two runs have hit this and both invented their own answer from outside the skill, one
recording the repair as "outside the skill's scope as written". It is not: pick from the
three above, say which and why in the report, and take the choice to the human at step 3
rather than deciding it silently at step 8.

The ten **stages** of the completion contract in `SKILL.md` map onto these steps; 2a and 2b are
two halves of stage 2. Stage 2a is new because the matcher you write at step 6 is a
translation between hipDNN's description of the operation and the kernel's, and reading
only the kernel half is how silent-wrong-answer defects get written.

**Commit after every step.** What you committed is the deliverable if you stop.

---

## Step 0 — Environment and dialect

```
Produces:      a working generator venv, and the dialect in one line
Gate:          test -x $GEN/.venv/bin/python
Typical time:  5 minutes
```

Set the variables at the top of this file, then make the generator's venv real before
anything needs it:

```bash
cd $GEN && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
test -x $GEN/.venv/bin/python && echo "GENERATOR VENV OK" || echo "SETUP INCOMPLETE"
```

**GATE:** prints `GENERATOR VENV OK`. Step 4 invokes `$GEN/.venv/bin/python` directly and
fails with "no such file" without this. Do not improvise a dependency directory or
`pip install --target` into the worktree — see § *Two Python environments* above for why
the split exists and which tool uses which.

Then the dialect. Is the kernel a rocKE builder? Then it is **`packaged`**, `kind: rocke`,
and there is no alternative — the runtime has no rocKE adapter. Say so in one line and
move.

The one fatal mistake here: splicing a packaged bundle into `HIPDNN_DESCRIPTOR_FILES`.
That installs an unlowered `kind: rocke` copy the loader rejects, silently dropping your
engine. Points 1 and 2 of the CMake splice **never** apply to you.

---

## Step 1 — Feasibility, before you mine anything

```
Produces:      a feasibility verdict on three axes, stated in the run log
Gate:          the GATE blocks in 1a, 1b and 1c below — all three, as written there
Typical time:  15 minutes
```

Three independent things must be true. Check all three now; each one costs a minute and
each one, discovered at step 8 instead, wastes the whole run.

**Re-run this step after ANY change to your base** — a rebase, a merge, a submodule bump.
The kernel moves under you: in one run a rebase silently reintroduced a keyword-only
parameter on the builder, taking `signature_error` from empty to a hard refusal and the
spec class from the arch subclass back to the shared base. The gate below catches it in
five seconds; nothing else will, and the failure surfaces much later as a packing error.
More generally: **verify locally before spending a remote job.**

### 1a. The builder is packable

```bash
# PYTHONPATH inline, not inherited. The export in the preamble is a separate fenced
# block; a fresh shell -- or a fresh agent turn -- does not have it, and the failure is
# `ModuleNotFoundError: No module named 'kernels'` raised from inside the introspector,
# which names the missing dependency but not the reason you are missing it.
cd $GEN && PYTHONPATH=$GEN:$PROVIDER/descriptor-packaging/python:$PROVIDER/rocke/library:$PROVIDER/rocke/platform/python \
python3 -c "
from codegen.sources import introspect
for b in '''$BUILDERS'''.split():
    i = introspect('$MODULE', b)
    print('==', b)
    print('signature_error:', repr(i.signature_error))
    print('spec_class:', i.spec_class)
    print('required:', [f.name for f in i.required_fields])
    print('arches:', i.supported_arches)
    for f in i.fields: print(' ', f.name, '|', f.type_name, '| default=', repr(getattr(f,'default',None)))
"
```

Run it for **every** builder in `$BUILDERS`. Two builders of one split design must agree
on the spec fields they share; where they diverge, that divergence is an applicability
rule, not a detail — record it in the mining table.

**GATE:** `signature_error` is empty. Non-empty ⇒ **STOP** — the builder does not take
`(spec, *, arch)` and `hkp_pack` will refuse it. Report the message and ask whether to
target a different arch or wait for the refactor.

Two readings that trip people:
- `supported_arches: []` means **unknown, not unsupported.** Get the real answer from the
  source: `grep -n 'arch != ' <module>` usually shows a hard gate in `supports_*` and
  `build_*`.
- `required` fields are **mandatory** in the descriptor's `spec` block — `hkp_pack`
  hydrates with `Spec(**fields)`, so a missing one is a `TypeError` at pack time, after
  the descriptor already looks complete.

### 1b. A reference executor can verify it

The shared executors are dense and stride-based; they decline paged KV, varlen, ragged
tensors and block-sparse/sinks. `dnn-providers/integration-tests/README.md`
§ *What the reference executors cannot verify* owns the current list — read it.

Then check whether **your** kernel has a path the reference can express:

```bash
grep -nE "if spec\.(paged|varlen|ragged|sliding_window|use_sinks)" \
     $PROVIDER/rocke/library/$MODULE | head
```

`-E`, always: the `\|` BRE spelling matches **zero** lines on several `grep` builds here
and exits 1, which reads exactly like "this kernel has no such branch" — the wrong
conclusion, reached silently.

An `if` around the feature means a dense alternative exists and you can ship a dense-only
variant set. No branch — the feature is unconditional — means there is no dense subset,
and stage 8 has nothing to compare against. Say so **now**.

### 1c. A machine with an `$ARCH` device is reachable by you

Packs arch-prune before the matcher runs, so stages 8 and 9 must run on `$ARCH` and no
other GPU will do. Establish on day one that you can get to one, because "the code is
finished but was never run on the target" is a 7b result, not a 9.

#### The requirement

You need, before stage 8:

1. **A machine with an `$ARCH` device you can execute on.** A workstation, a
   pre-provisioned node, a container with the device passed through, or a scheduled
   cluster allocation — the runbook does not care which.
2. **Enough time on it for the reference executor**, not just the kernel. The shared
   references are untiled and cost far more than the kernel they verify; step 8a sizes
   this, and it is the usual reason a "quick" device run is not quick.
3. **A writable path the DEVICE MACHINE can see**, for the install tree, the graph
   corpora and the logs. Not merely a path *you* can see.

Verify all three **on the machine that will run stage 8**, which is the whole point —
the same commands on a login node prove nothing and will happily report success:

```bash
# Directly, if you already have the device:
$GEN/tools/device_probe.sh $ARCH $SWEEP_ROOT $INSTALL

# On a scheduled cluster, SUBMIT it — do not run it where you type:
srun -p <partition> -A <account> --gpus=1 \
    $GEN/tools/device_probe.sh $ARCH $SWEEP_ROOT $INSTALL
```

It checks the arch, the install tree's visibility and the write path from wherever it
runs, and exits non-zero if any fails.

**GATE:** `device_probe.sh` exits 0 **on the machine that will run the tests**. If the
arch is unreachable, get a decision now and report the run as the stage it reached —
never as stage 8.

#### Worked example: a scheduled cluster

Only one way to satisfy the requirement, shown because it is the fiddliest and the
lessons generalise. On a workstation with a local device, skip it — run the commands
directly and move on.

A queue you can see is not a queue you can use, so ask before you build:

```bash
# Substitute your site's submit host, partition, account and resource names.
ssh $LOGIN "sinfo -N -h -o '%P|%N|%t|%G' | grep $ARCH"   # present, and live or drained?
ssh $LOGIN "squeue -h -o '%b' | grep -c $ARCH"           # how contended?
ssh $LOGIN "sbatch --test-only -p <partition> -A <account> \
    --gres=gpu:<type>:1 --time=00:20:00 --wrap=hostname"  # may I, and when?
```

Three failure modes worth knowing, because each cost real time to diagnose:

- **A visible partition can still reject you.** `--test-only` reports an access failure
  without consuming a submission; `invalid partition specified` means your account has no
  association with it, however healthy it looks in `sinfo`. Treat the estimate itself
  with suspicion, though — it is a worst-case backfill figure and can be wrong by orders
  of magnitude in *either* direction. Verify by actually submitting something trivial.
- **A shared `$HOME` is not a given.** Where sites mount different filesystems, a job
  whose `--output` points at a path only the submit host mounts dies *before* your
  payload runs: the scheduler cannot open the log, so nothing explains the failure.
  Create the directory on the target site's filesystem first.
- **A launch failure can masquerade as a pending job.** `squeue` shows the requeued row
  and hides the failure. `sacct -j <jobid> --duplicates -o JobID,State,ExitCode,Start,End`
  shows the truth: `State=FAILED, Start=None` means it never ran at all.
- **The device payload is workspace tooling, not product.** A job script carries scheduler
  flags, image paths and site names — none of which belong in the repository, and some of
  which must never be pushed to a public branch. If the compute node needs the script,
  **stage it** to a filesystem that node can read and point the job at it; do not commit
  it so the node can clone it. This has gone wrong: payloads were committed to a product
  branch purely so a node could fetch them, and they carried site identifiers with them.

The general lesson under all of them: **confirm the device machine can see every path you
hand it, from that machine**, before you queue anything expensive.

---

## Step 2a — The graph contract: what hipDNN can ask for

```
Produces:      graph_contract.md
Gate:          ls graph_contract.md  (op matched; disagreement table has rows)
Typical time:  45-90 minutes, mostly reading
```

**Read `graph-contract.md` now** and produce its five sections. Do this *before* touching
the kernel's Python.

Why this order: the matcher you write at step 6 translates between hipDNN's description of
the operation and the kernel's. Agents who start on the kernel side arrive at step 6 having
never read the graph side, then reverse-engineer it while writing C++. Every
silent-wrong-answer defect this skill knows about came from that order.

What 2a settles, in one line each:

- **Which operation you are implementing** — a node, or a *composition* of nodes.
  hipDNN is a graph API; fused kernels are subgraphs and have no table of their own.
- **Every field the graph can carry**, consumed or explicitly rejected.
- **How the frontend spells it** — intent, defaults and deprecations the schema omits.
- **What a real graph of this op looks like** — from the in-tree bundles AND from
  `dnn-benchmarking`'s shipped model workloads, which use the same JSON schema and
  frequently disagree with the in-tree ones on layout, mask spelling and shape. Plus the
  framework operator's own semantics, which is the reference 8e validates against.
- **The disagreement table** — every kernel field against its hipDNN spelling. This is
  what step 6 consumes, and it is the highest-value artifact in the run.

If no operation matches, `graph-contract.md` carries a six-check disconfirmation list.
Work all six and record the results before escalating; most first-pass "hipDNN cannot
express this" calls are actually "I have not understood this API yet."

**GATE:** `ls graph_contract.md` succeeds, section 1 names the node(s) and any UID edges,
and section 5 has a row per pinned kernel field.

---

## Step 2b — Mine the kernel: what it can actually answer

```
Produces:      mining.md
Gate:          ls mining.md  (every constraint row has a verdict)
Typical time:  60-90 minutes
```

**Read `rocke-mining.md` now.** It tells you what to extract and how to classify it. You
are classifying rules as graph-derivable or not — which is only answerable because 2a told
you what the graph carries.

The budget, because this is where runs die: **draft `mining.md` after the kernel module
and its spec, before ANY third source.** Then **five sources *beyond the kernel module*,
maximum** — count them and name them in the file. Rows you are unsure of go in marked
`UNSURE` and become step-3 questions; that is what step 3 is for. Hitting the cap is not
failure, it is the step working.

**The self-check, when you feel the "let me verify this properly" pull:** ask *is this
file open because a specific `UNSURE` row needs it, or because I am not ready to write
yet?* The second is the stall, and from the inside it is indistinguishable from
diligence. If you cannot name the row the source resolves, you are past the cap — write
the file.

**If you are supervising this agent, do not rely on either rule above.** They were
written after an agent stalled here for 110 minutes; the agent that then read them, and
said unprompted that they were clear and binding, drifted past them twice more in the
same run, for 25 and 50 minutes. Its own diagnosis was right: *"there's no forcing
function that fires until I choose to invoke it myself."* Self-monitoring does not close
this one, and an agent affirming the rule is not evidence that it will hold.

The check that works costs one command:

```bash
ls $REPO/mining.md
```

Run it around 20 minutes in. If the file is missing, interrupt with **do not read another
file until `mining.md` exists** — that exact instruction produced a 331-line file within
minutes, both times it was used. And when asking a quiet agent for status, ask what it
has **written**, not what it has **found**: a stalled agent answers the second question
fluently and at length, and that fluency is the tell.

Discovery commands for the five deliverables. **None of these names is guaranteed** — a
command that prints nothing means you looked for the wrong name, not that the kernel has
no rules. `rocke-mining.md` carries the fallbacks:

```bash
M=$PROVIDER/rocke/library/$MODULE

# Constraint table: the spec's validation, wherever it lives. Both shapes -- a raise,
# and the `return False, "why"` verdict pair rocke's supports_<op> uses instead.
grep -nE "raise ValueError|return False," $M

# Layout: the address arithmetic. Look for the stride and base terms.
grep -nE "stride_[a-z]+_tok|_base = |b\.global_(load|store)|buffer_rsrc" $M | head -30

# Grid / block / ABI — named-function convention first
grep -nA12 -E "^def [a-zA-Z_0-9]+_(grid|block|signature)\(" $M

# ABI slots, and whether each is conditional, IN ORDER. rocKE declares the signature
# through a SignatureBuilder chain, not `b.param(...)` -- `b.param(` matches zero lines
# in both the gfx950 and gfx942 dense modules.
grep -nE "\.ptr\(|\.scalar\(" $M

# Launch-time guards that are NOT in the spec: everything the run_<op> wrapper rejects.
# awk, not `grep -A3 | sed`: grep's context blocks are disjoint, so a sed range anchored
# on `def run_` only ever fires if a raise happens to sit within 3 lines of it. That
# pipeline printed NOTHING on both dense modules -- it has never returned data.
awk '/^def run_/{f=1} f && /raise |return False,/{print FILENAME":"NR": "$0}' $M
```

**GATE:** `ls mining.md` succeeds and every row of the constraint table has a verdict.

---

## Step 3 — One batch message to the human

```
Produces:      one message, containing proposals rather than questions
Gate:          sent, after the source budget is spent
Typical time:  30 minutes to prepare; proceed on the defaults if no reply
```

**Do not block on this.** Every item below has a default you can derive and defend from
the work you have already done; the message exists so a human can *correct* you cheaply,
not so you can wait for permission. Send it, state the default you will use for each
item, and **proceed**. Mark each as an assumption and list them at step 9. A run that
stalls here has converted a 30-minute review into an open-ended pause.

The defaults, so you are never blocked:

| Decision | Default if unanswered |
|---|---|
| Engine name / namespace | `<provider-prefix>:<CamelKernelName>`, scoped; unscoped is rejected |
| Arch list | exactly the arch the builder gates on — never wider |
| Variant set | every real workload shape you can serve (`workloads.md`), plus a tuning twin where a knob would otherwise be unreachable |
| Exposed knobs | every **int-typed** KMD field; non-int knobs are silently dropped by the loader |
| Workspace policy | `none`, if the kernel allocates no scratch and the ABI has no workspace slot |
| Declined features | everything your step-2b checklist marks SCOPE; each gets a negative test at 8b |
| UMD vs `graph_match` | `graph_match` unless two packs genuinely need to differ on one graph fact |

Escalate and *stop* for exactly two things: an operation the schema cannot express
(`graph-contract.md`'s six-check disconfirmation, all six worked), and target hardware you
cannot reach (step 1c). Everything else is a default plus a note.

**Exhaust the sources first.** This batch is for decisions only a human can make: scope,
naming, what they intend to run. Anything the tree answers, answer yourself. Two things
look like judgment calls and are not:

- **A numeric mapping between the kernel and the graph** (an off-by-one, a units
  difference). The stage-8 reference executor *defines* it — read its predicate under
  `integration-tests/gpu-ref/kernels/<op>/`.
- **What an unfamiliar graph attribute means.** The frontend header and the cuDNN
  compatibility shim both spell out intent and defaults in comments.

If you must ask something source-derivable anyway, ask with your answer and its evidence
attached, as a confirmation. If the human replies "go and check" — the correct reply to a
question you should not have asked — go and check; do not re-ask, and do not stall.

Bring a proposal, not a question. Present: engine name, arch, the variant set with counts,
**how it lands against real workload shapes** (§ *Sizing the variant set*, item 3 — which
real configurations a variant serves and which you are scoping out), which knobs are
exposed and which values ship AOT, workspace policy, the layout you derived with its
arithmetic, and the rejection checklist. If they do not answer, ship the
proposal and mark it an assumption.

**Two separate knob decisions, and both are the human's.** *Exposed knobs* — which KMD
fields become `knobs` on the UED, i.e. what a caller or the autotuner may steer. *AOT
variant values* — which concrete values of each are compiled into the shipped set. An
exposed knob with one compiled value is a knob in name only.

**Only `int`-typed fields can be a real knob.** The loader's `getCustomKnobs` silently
drops a non-int knob at plan-build time — no error, no warning, discovered only against a
real device. Screen for this *before* you ask: if a knob is wanted on a non-int field, say
so now and offer to retype the field or drop it.

Ask both as one question, e.g. "expose `<knob_a>` and `<knob_b>`; ship
`<knob_a> ∈ {…} × <knob_b> ∈ {…}` = N tuning variants per capability — confirm or adjust?"

### Sizing the variant set

**A one-kernel engine is not an integration, it is a demo.** With a single variant `score`
never ranks anything, the UED's knobs select nothing, autotuning has no candidate set, and
the first graph whose dtype or shape differs finds nothing to serve it. The heuristic path
becomes dead code that still reports green.

**Never carry another arch's knob set across.** Run `knob_sweep.py --plan` before you
propose one: it names the candidates, excludes the settled knobs *with reasons*, and flags
predicates that differ between arches. A set transcribed from a sibling engine imports
that engine's exclusions in both directions — you inherit a knob it ruled out for reasons
that do not hold on your arch, and you miss one it never had. A proposal made by reading a
sibling's shipped metadata was wrong three ways when the tool was finally asked.

The set must deliver three different things:

1. **Feature coverage** — one variant per combination of *supported capability* a graph
   can ask for. A capability with no variant behind it is one the engine advertises and
   cannot serve. That is the applicability defect, arriving as wrong numbers.
2. **Performance headroom** — several variants along the *tuning* knobs for the same
   capability, so the heuristic and the autotuner have something to choose between.
3. **Workload realism** — variants at the shapes callers actually run. **This is the one
   that gets skipped**, because the other two can be satisfied entirely from kernel-side
   sources and produce a set that passes every gate in this runbook while serving nobody.

**On (3), because it is the failure this section exists to prevent.** People have real
workloads that need these kernels; an engine exists to serve them. Feature coverage and
tuning headroom are properties of the *matcher* and the *heuristic* — you can satisfy both
with a tidy little cross-product of small shapes, watch every check go green, and ship an
engine that declines every graph a real model sends. For a kernel that bakes its extents
the gap is total: "capable but no variant" and "unsupported" are the same thing to a
caller.

So make the shapes an input to this decision, not an afterthought:

- **Read modern model and workload shapes before you choose.** What sequence lengths,
  batch sizes and head counts do current models actually run at? Those are the
  configurations "that matter" below — and they are frequently *orders of magnitude* away
  from what a hand-written test matrix reaches for.
- **Read `workloads.md` now** — it owns the corpus, the three-bucket triage, the
  coverage count this step is gated on, and the setup traps.
- **The concrete source for this project is `dnn-benchmarking`'s `Workloads/` tree**
  (github.com/ROCm/dnn-benchmarking): per-library microbenchmarks and real model traces,
  shipped as the same graph JSON your matcher already walks, one `dvc pull` away. Step 2a
  (`graph-contract.md` §4) already sends you there to read a real graph; this is the other
  thing to take away from that reading. Extract the distinct shape tuples for your op and
  keep them in front of you while sizing.
- **Then check your proposed set against them, before generating.** For each real shape,
  would a variant match? A set that answers "no" to most of them is a test matrix, not a
  shipping set. Widen it, or state the scope limit deliberately — but do not discover it
  at stage 8.

If the real shapes are too large or too many to ship whole, that is a legitimate scope
decision — say which you cover, which you decline, and why, in the step-3 batch. What is
not legitimate is never having looked.

**Where to find the candidate axes**, in order of authority:

- **The rocKE dispatcher for this kernel family**, under `rocke/library/dispatch/`. It
  already encodes which configurations are worth generating. **Read the module your
  builder actually reaches, not a shared one** — a family directory can hold several
  unrelated kernel families side by side whose shared constants do not apply to yours.
  Grep for the builder you are integrating and read only what that call path touches.
  Not every op-family has a dispatcher directory; where there is none, the per-op module
  is both dispatcher and sizing source.
- **The spec dataclass's own knob comments**, which frequently record measured results.
  Prefer a knob the kernel author says matters; skip one they say is neutral.
- **The `supports_*` predicate's allowed sets**, for the hard capability bounds.

Those three are **kernel-side**: they tell you what the kernel *can* be built for. None of
them tells you what anyone will *ask* it for — for that, use the workload shapes above.
Cross the two: the kernel's legal configurations, intersected with the shapes real models
run, is the set worth shipping.

The dispatcher is also where *policy* lives. If it does not auto-select your kernel — some
candidates are opt-in only, matching solely when a request names them — say so in the
step-9 report: you are exposing something rocKE itself does not route to by default.

**Budget: keep the total under ~100 kernels for a first integration.** Every variant is a
separately compiled code object, costing build time, archive size and install footprint.
Cover each capability once, then spend what remains on tuning variants for the
configurations that matter. Say what you pruned and why.

**Once the set is driven by tuning axes rather than a handful of shapes, do not enumerate
it by hand.** Two forms collapse the enumeration at load time into ordinary kernels —
nothing downstream can tell the difference:

- `axes:` plus a `kernel_template:` crosses ONE template with a few value lists.
  `configs/axes_example.yaml` is the worked example.
- `variants:` crosses a SHAPE LIST with per-shape named knob sets. That is the form a
  dispatcher-derived set takes, because `dispatch_parity.py` asks the library for a spec
  per shape and every shape carries its own resolved values — there is no one template to
  cross. `configs/variants_example.yaml` is the worked example, and
  `dispatch_parity.py --out` emits this form directly.

Enumerating instead is what turns a config into a six-figure-line diff that no build step
reads and nobody can review: the largest shipped gfx942 set was 89,265 lines for 2,710
kernels, and about 1,150 in `variants` form. Under `axes` the expanded name encodes every
axis by construction. Under `variants` the template is yours to write: it must encode
everything that varies, and the loader rejects a pack whose expansion produces two kernels
with the same name.

**Commit the config as PLAIN TEXT.** The loader still reads `.gz` transparently, but
compressing a generated set is not the fix for its size — it is how an unreviewable file
got committed in the first place. The config is the one file in a descriptor PR worth
reviewing, because the descriptors are its deterministic output. To convert a config that
is already enumerated, run `tools/factorise_config.py`, which refuses to write anything
that does not re-expand to the input kernel-for-kernel.

**When the axes are not orthogonal.** The guidance above assumes axes that cross freely.
Many kernels are not like that: features can be mutually exclusive in non-obvious ways, so
a full cross-product is not merely large but mostly *illegal* — most cells fail the spec's
own validation. For a kernel like that the honest first integration is **narrow and
explicit**: ship the core, and have `graph_match` explicitly decline the rest. That is a
*stated scope limit*, not a gap — the engine serves what it ships and loudly declines
everything else, which is the correct, debuggable failure.

What makes that legitimate rather than lazy is saying it: name the declined features here
in step 3, give each a negative test at step 8b, and list them at step 9 as follow-on work.

**GATE:** the message is sent, and it contains a concrete variant list with counts — not a
question about what the variant set should be — **and it states how that set lands against
real workload shapes**: which of them a variant serves, and which are deliberately out of
scope. "N kernels covering every capability" without that second half is the set that
passes every check and serves nobody (§ *Sizing the variant set*, item 3).

---

## Step 4 — Generate

```
Produces:      configs/$SLUG.yaml and a generated descriptor tree
Gate:          generate.py exit 0, and kernel count == the set agreed at step 3
Typical time:  45 minutes
```

### 4a-0. Build the shape corpus — from the sources that decide, not the ones nearby

`$SHAPES` is an input, not something to invent. Three sources answer three different
questions and no one of them is sufficient:

| source | answers |
|---|---|
| the kernel team's **published results CSV** | what they measure, tune, and escalate a regression on |
| **dnn-benchmarking** graphs | what real callers ask for |
| the `supports_*` predicate | what is legal to build |

```bash
cd $REPO
$GEN/.venv/bin/python $GEN/tools/mine_shapes.py \
    --published <the team's perf_*.csv> \
    --graphs    <a dnn-benchmarking graph tree> \
    --arch      $ARCH \
    --out       $SHAPES
```

**Ask for the results CSV before mining anything.** It is the shape list already
resolved — no inferring a sweep from nested loops — it names which kernel a published
number refers to, and it carries `priority` and `ticket_group`, a shipping-priority
signal available from nowhere else. It also enumerates shapes the benchmark source
does not.

**GATE:** a non-zero distinct-shape count, and a `by source` line naming every source
you intended to mine. A source you meant to include that contributes zero shapes is a
path typo, not an empty corpus.

Two traps this step exists for:

- **A `microbench/` path is a PROVENANCE label, not a synthetic-data warning.** One
  suite was dismissed on the strength of its directory name; its own manifest said
  every shape was rendered from a real source and none invented. That cost 72 shapes.
- **Provenance is carried onto every shape, and every reported result must be split by
  it.** A geomean over a mixed corpus reports the synthetic population's win as though
  it were everyone's — on the run behind this guidance the same data read as a large
  win on one microbenchmark suite and close to parity on real model traces. It costs
  nothing here and cannot be recovered afterwards.

Backward graphs are excluded structurally, by their gradient tensors rather than by
filename: a prefill kernel has no backward path, so such a graph falls through to
another engine, and one of them faults the DEVICE mid-run.

### 4a. Start from parity — the dispatcher's own set. REQUIRED FIRST.

Do not hand-write the first config. Generate the set rocKE's dispatcher would itself
resolve, and make everything after it a deviation you can name:

```bash
cd $REPO
$GEN/.venv/bin/python $GEN/tools/dispatch_parity.py \
    --profile $GEN/configs/$SLUG.profile.yaml \
    --shapes  $SHAPES \
    --out     $GEN/configs/${SLUG}_A.yaml \
    --report-knobs --report-gaps
```

**GATE:** exit 0, `servable` matches the shape count you expect, and every shape under
`declined` or `rejected` has a reason you accept. An unexplained gap is a defect until
proven otherwise — the proof is the reason string the tool already printed.

This is the only configuration whose correctness argues from **rocKE's own behaviour**
rather than from measurement. Everything after it is a deviation that needs a
justification, and this is the thing those deviations get measured against.

It also removes the most expensive mistake available at this step. The dispatcher sets
some fields as CONSTANTS you can read and copy, and others as RULES computed from the
request — `persistent = work >= num_persistent` is a line of control flow that reads
like an ordinary local variable. Transcribing by hand copies the constants and misses
the rules, the descriptor takes the dataclass default instead, and the default was the
opposite of the dispatcher's answer on most of a shipped set. Nothing failed:
descriptors validated, the desk check was clean, correctness passed on device. The only
symptom was a performance number, misattributed three times before anyone found the
field. Calling the factory cannot make that mistake, because a rule is applied rather
than read.

`--report-knobs` prints which spec fields VARY across the dispatcher's decisions and
which are CONSTANT. Read it before step 3's sizing conversation and before any sweep: a
knob the dispatcher fixes for every shape it serves is not a tuning axis, it is a value
the library ships, and sweeping it measures a configuration rocKE would never resolve
to. A commit message saying the author swept a knob means it was *explored*, not that it
*ships*.

If your kernel has no profile yet, write one at `configs/<slug>.profile.yaml` — the
integration branch stacked on this one carries a worked example. It declares the
dispatcher entry point, the request class, the predicate, the matcher's vocabulary, and
any knob the kernel resolves BY POLICY. The same file drives the variant-set gate at
step 5, so the two cannot disagree about which kernel they are discussing.

### 4a-2. Sweeping a knob — isolate, then pair the survivors

Only after parity. A sweep is not "try the knobs"; the last one moved 2 of 22, chose
them by hand, shipped a cross-product, and the uplift landed almost entirely on one
synthetic shape family while the wide arm bought nothing measurable over the condensed
one.

```bash
cd $REPO
# 1. What is even a candidate, and why is the rest not?
$GEN/.venv/bin/python $GEN/tools/knob_sweep.py \
    --profile $GEN/configs/$SLUG.profile.yaml --shapes $SHAPES --plan

# 2. Two arms per candidate, everything else at parity.
$GEN/.venv/bin/python $GEN/tools/knob_sweep.py \
    --profile $GEN/configs/$SLUG.profile.yaml --shapes $SHAPES \
    --isolate --out-dir /tmp/$SLUG-arms

# 3. Only the knobs that moved individually.
$GEN/.venv/bin/python $GEN/tools/knob_sweep.py \
    --profile $GEN/configs/$SLUG.profile.yaml --shapes $SHAPES \
    --pairwise <survivor>,<survivor> --out-dir /tmp/$SLUG-pairs
```

#### 4a-3. Then build the SHIPPING package from what actually mattered — REQUIRED

Isolation and pairwise are how you find candidates. They are not the deliverable. The
deliverable is one package containing the knobs that measurably earned a slot, over the
shapes you intend to serve:

```bash
# The base set stays dispatcher-resolved. --knobs crosses it with the values that
# SURVIVED isolation and pairwise -- nothing here is transcribed by hand.
$GEN/.venv/bin/python $GEN/tools/dispatch_parity.py \
    --profile $GEN/configs/$SLUG.profile.yaml --shapes $SHAPES \
    --knobs '{"<survivor>": [<v1>, <v2>], "<survivor2>": [<v1>, <v2>]}' \
    --out $GEN/configs/${SLUG}_shipping.yaml
```

Pass ONLY the survivors. A knob absent from `--knobs` keeps the value the kernel's
own policy resolved, which is the point: pinning a knob the kernel decides by policy
DISCARDS the policy, and that is how a generated set ends up strictly worse than a
smaller one.

This is not a pack `axes:` block and cannot be one. `axes:` crosses a single
`kernel_template`; here every shape carries its own dispatcher-resolved spec, so the
cross-product has to happen where the specs are. The tool prints the arithmetic
(`N servable shapes x K combinations`) so the descriptor count is a number you read
rather than one you discover at pack time.

**A pinned knob is written into the SPEC as well as the metadata, and it has to be.**
The spec is what the builder compiles; the metadata is only what the matcher compares.
Pinning one without the other emits two catalog entries over ONE binary — the arms are
the same kernel, the sweep measures 1.000x, and the knob gets recorded as "no effect"
when its other side was never compiled. This is the same three-layer trap as any
tri-state, in the one place the tool rather than the author controls it, and it is why
`--knobs` now refuses a name the builder's spec class does not accept: such a field can
only ever reach the catalog, never the binary. If you get that refusal, either the name
is wrong or the field is not a build-time knob of this kernel — it is never something to
work around. Declare the builder's own spec class in the profile's `arch_spec:` block,
or the tool has nothing to check the name against.

**GATE, three parts, all of them hard:**

1. **Descriptor count is capped in the low thousands.** Past that the pack time, the
   archive and the catalog all stop being reasonable, and the marginal variant is
   almost never the one that wins. If you are over, cut axes — not shapes.
2. **Every surviving knob traces to a measurement.** Name the isolation arm that moved
   it. A knob in the shipping set with no arm behind it is a guess wearing evidence's
   clothes.
3. **The package passes 5e, 5f and 5g** — set properties, the three rungs, and
   reachability. A shipping set with an `APPLICABLE-BUT-NEVER-WINS` variant is paying
   compile time and catalog space for a choice that cannot happen.

Then take THIS package through step 9 in full. The isolation arms answered "does this
knob do anything"; only the shipping package answers "does the thing we are actually
going to ship serve real traffic, correctly, at a reasonable speed" — and those are
different questions with different answers.

**GATE on the plan:** every excluded knob has a reason you accept, and every candidate
has two values that are genuinely reachable. **GATE on the arms:** no arm is marked
`== parity, measures nothing` unless you meant it — that arm is the baseline under
another name, measures exactly 1.000x, and would have the sweep report "no effect" for
a knob whose other side was never tried.

Three things the plan step decides before any GPU time is spent:

- **A knob CONSTANT across every dispatch decision is not an axis.** The dispatcher
  fixing a value is the library shipping it.
- **A knob with a measured verdict is settled.** `iglp` is marked "RESOLVED — do not
  re-attempt"; `block_m` FAULTS at other values. Re-measuring those spends GPU time to
  rediscover what the source already says.
- **A knob whose alternative the predicate rejects is gated here**, not discovered on a
  device — `block_n=128` needs 69,632 B of LDS at D=128 and is refused.

Every arm still has to pass step 5's gates. Note that **arms do not nest** and should
not: an arm is a deviation from parity, not a superset of it, so `verify_variant_sets.py`
will correctly report the binaries as non-nesting when you point it at two arms. Nesting
is a property of a SHIPPING set against a smaller one, which is step 4b's question.

A useful pre-GPU signal: two arms whose binaries are largely identical cannot produce
much of a difference, and the gate already prints `distinct-binaries` per set. Pinning
a policy-resolved tri-state to one side changes the majority of binaries on a corpus
that spans the policy's threshold — visible statically, before a single kernel runs.
An arm that changes almost nothing is one you can decline to measure.

#### Measuring the arms

`$GEN/tools/sweep.sh` runs them. It takes a config —
`$GEN/configs/sweep-isolation.env.example` is the worked one — rather than being
forked per comparison: the entire difference
between the two one-off sweeps this was generalised from was an output directory, a
corpus list, an arm list, an install tree per arm, and an expected descriptor count.

```bash
SWEEP_CONFIG=my-sweep.env $GEN/tools/sweep.sh
```

Read the header before changing anything in it. Every structural choice — one node,
one session, a discarded warmup, three rounds, **fixed arm order** — exists because
clocks usually cannot be pinned on a shared or containerised machine (check whether
yours can — if so, pin them), and the fixed order in particular is what
makes the sign check work: drift penalises whichever arm runs later, so an effect
moving the OPPOSITE way to the confound cannot be explained by position. Put the
baseline first.

Three gates run per phase and they are not redundant. The descriptor count is a
property of files on disk. The plugin-provenance grep matches anywhere in the log, not
`head -1`. The **ingestor-served count** is the one that catches a dropped engine —
the phase otherwise runs to completion, exits 0, passes the other two, and serves every
graph from a different engine.

When you report: **geomean-of-ratios and time-weighted side by side**, and split by
provenance. On the run this harness comes from, those two statistics differed by more
than an order of magnitude in how much of a win they described, because the shapes
driving the geomean held a tiny fraction of total GPU time. Both are true; only one of
them is what a reader hears.

### 4b. Deviate, if you have a reason

Edit the generated config, or write your own using it as the shape reference:

```bash
ls $GEN/configs/
```

Then:

```bash
cd $GEN
$GEN/.venv/bin/python generate.py --config configs/$SLUG.yaml --output-dir /tmp/$SLUG --dry-run
$GEN/.venv/bin/python generate.py --config configs/$SLUG.yaml --output-dir /tmp/$SLUG
echo "EXIT=$?"
python3 -c "
import glob, json
for p in sorted(glob.glob('/tmp/$SLUG/descriptors/rocKE/$SLUG/*.kdp.json')):
    print(p.rsplit('/', 1)[1], '->', len(json.load(open(p))['kernelDescriptors']), 'kernels')"
```

**Every deviation from 4a's set is a claim you owe evidence for.** Say which knob you
moved, why the dispatcher's value was not right for the shapes you serve, and what you
expect to gain. "More variants might help" is not a reason; it is what a cross-product
is, and a cross-product bought nothing measurable over a condensed set on either corpus.

If you hand-wrote the config instead of starting from 4a, the pre-generate diff is
manual: `mining.md`'s dispatcher table lists each field the library sets per request,
with its kind. Diff the fields in your config's `spec:` block against the fields that
table names. Anything missing is a silent default — which is the failure 4a exists to
make impossible.

**GATE:** exit 0, at least one `.kdp.json` printed, and the kernel count equals the
variant set you agreed in step 3. Exit 1 is a `ConfigError` — read it; usually a mistyped
field, a knob on a non-int field, or a kernel `arch` outside its pack's `arch`.

The glob is deliberate: **do not construct the KDP filename by hand.** The generator
derives it (`git grep -n 'def kdp_stem' -- '*.py'`) — the bare engine slug for a
single-pack engine (`$SLUG.kdp.json`), the slug plus the pack name for a multi-pack one.
Printing nothing means the generator wrote nothing, which is a step-4 failure, not a
naming detail to work around.

**Step 5 does not need this copy.** All three validation rungs and the desk check run
against `/tmp/$SLUG/descriptors` and `/tmp/${SLUG}_pack` directly — verified end to end on
a kernel this runbook was not written from. The copy below is what makes a **CMake build**
pick your descriptors up, so it is a step-7 prerequisite. Doing it now is fine; if
anything below blocks you, go do step 5 first rather than stalling.

Then place the authored descriptors under **the packager's actual source root**. Do not
guess it — the build tells you, and it is a CMake cache variable a build may point
anywhere:

```bash
grep PRODUCTION_SOURCE_ROOT $BUILD/CMakeCache.txt
```

An empty `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` means production packaging is dormant
and your descriptors are never packed at all — fix that before copying anything. When it
is set, copy into it, preserving your `authored_subpath`:

```bash
SRC_ROOT=$(grep '^HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT' $BUILD/CMakeCache.txt | cut -d= -f2-)
test -n "$SRC_ROOT" || echo "PACKAGING DORMANT -- nothing will be packed"
mkdir -p "$SRC_ROOT/rocKE"
cp -r /tmp/$SLUG/descriptors/rocKE/$SLUG "$SRC_ROOT/rocKE/"
```

**Where this SHOULD go, and why it does not yet — read before you copy.** Real integrations
belong in the engine tree, `$PROVIDER/src/engines/kernel_ingestor_engine/descriptors/`,
beside the `packs/*Native.cpp` you write at step 6 and alongside the shipped `conv_fwd/`
and `pointwise/` engines. That is the release-and-bundling location.

**A `packaged` integration cannot land there today.** `hkp_pack` validates every descriptor
under its source root and hard-fails on any `kind` it does not produce; the engine tree's
existing descriptors are `kind: embedded_source`, which is not in that set. Point
`PRODUCTION_SOURCE_ROOT` at the engine tree and the pack dies on `conv_fwd` before it ever
reaches yours. When `hkp_pack` learns the engine tree's dialect, this step becomes "author
in the engine tree" and the copy disappears.

Until then `$SRC_ROOT` resolves — on a CI-shaped build — to
`$PROVIDER/descriptor-packaging/examples/descriptors`. **That tree is a pinned test
fixture as well as the packager's source root**, and `tests/test_hkp_pack_layout.py`
asserts its directory set *exactly*. Adding your integration means editing that assertion.

Do it, but understand what you are doing: that test exists to stop the example tree
growing real integrations, and you are widening a guardrail rather than passing it. Note
it at step 9 as a known deviation. If you find yourself editing a test
to make room for your work and it feels like an obstacle, that is the signal to stop and
ask whether you are in the right tree — the answer here is no, and it is a packager gap
rather than your mistake.

**This copy is part of the REGENERATE loop, not just the first run.** The config is what
you author; `$SRC_ROOT` is what the build compiles. Change the config, rebuild, and the
build faithfully packs the **stale** descriptors — config says N variants, the packed tree
says the old count, and every check stays green. Re-run the copy after every
`generate.py --force`, and confirm the two agree:

```bash
python3 -c "
import json
a=json.load(open('/tmp/$SLUG/descriptors/rocKE/$SLUG/$SLUG.kdp.json'))['kernelDescriptors']
b=json.load(open('$SRC_ROOT/rocKE/$SLUG/$SLUG.kdp.json'))['kernelDescriptors']
print('generated', len(a), '| staged', len(b), '| MATCH' if len(a)==len(b) else '| STALE')"
```

On a typical configured build `$SRC_ROOT` resolves to
`$PROVIDER/descriptor-packaging/examples/descriptors` — that tree is **both** the
packager's production source root **and** a pinned test fixture, and nothing warns you.
`tests/test_hkp_pack_layout.py` asserts its directory set **exactly**:

```python
assert rel_dirs == {"hip/pointwise_add", "rocKE/gfx942_tiled_attention"}
```

So the copy above turns a green suite red, and the failure names neither your integration
nor the copy that caused it. This is expected and the fix is one line: **add your
`rocKE/$SLUG` to that set, in the same commit as the descriptors.** Do not work around it
by copying somewhere else — somewhere else is not packed.

```bash
PYTHONPATH=$PROVIDER/descriptor-packaging/python:$PROVIDER/rocke/library:$PROVIDER/rocke/platform/python:/opt/rocm-kpack/python \
  python3 -m pytest $PROVIDER/descriptor-packaging/tests/test_hkp_pack_layout.py -q
```

**GATE:** still green. Placing descriptors must not break the packager's own tests.

---

## Step 5 — Validate, three rungs

```
Produces:      a packed tree that loads, plus a clean desk check
Gate:          5a count printed; 5b skipped=False; 5c success:true, 0 ERROR; 5d clean
Typical time:  30 minutes, plus pack time (comgr can take minutes per arch)
```

Each proves something different. Run all three.

**Set the comgr cache before 5b, not after it times out.** Every kernel this step packs
is a comgr compile, and the cache defaults under `$HOME` — on a network home that turns
a minutes-long pack into an afternoon, with no error, just slowness you attribute to the
kernel. `$PROVIDER/descriptor-packaging/README.md` owns the variables and the
measurements (`AMD_COMGR_CACHE_DIR` on local disk or a RAM disk; `AMD_COMGR_CACHE=0` to
disable). Read it there rather than trusting a copy here — check `df -T "$HOME"` for a
network filesystem type if you are unsure which case you are in.

```bash
# 5a. Authored form -- exactly what the build enforces
python3 -c "
from hkp_pack.descriptors import load_flat_input
flat = load_flat_input('/tmp/$SLUG/descriptors'); print('OK,', len(flat.descriptors), 'descriptors')"

# 5b. Pack -- the rung that proves the builder actually lowers through comgr
python3 -c "
from hkp_pack.pipeline import run_pipeline
res = run_pipeline(source_root='/tmp/$SLUG/descriptors', arches=['$ARCH'],
                   out_root='/tmp/${SLUG}_pack', hipcc='/opt/rocm/bin/hipcc',
                   rocm_kpack_dir='/opt/rocm-kpack/python')
print({a: (r.skipped, str(r.kpack_path)) for a, r in res.items()})"

# 5c. Shipped form -- through the real runtime loader. PACKED tree, never the authored one.
$BUILD/bin/hipdnn_validate_descriptors /tmp/${SLUG}_pack/$ARCH \
    --expect-engine $ENGINE --json | python3 -c "
import json,sys; d=json.load(sys.stdin)
print('success:', d['success'], '| engines:', d['engines'], '| missing:', d['expected_engines_missing'])
print('errors:', [g for g in d['diagnostics'] if g['severity'] in ('ERROR','FATAL')])"
```

**GATES:** 5a prints a count; 5b reports `skipped=False` and a `.kpack` path; 5c prints
`success: True` with your engine listed and an empty error list.

Expect `WARN`s about the `provenance` extension key — the packager writes it, the loader
ignores it. Correct, not a problem.

**If `hipdnn_validate_descriptors` does not exist**, there are three different causes and
they need different actions. Do not guess — this command tells them apart:

```bash
if   [ ! -d "$BUILD" ];                                  then echo "NO BUILD -- $BUILD does not exist"
elif [ ! -f "$BUILD/CMakeCache.txt" ];                   then echo "NOT CONFIGURED -- no CMakeCache.txt"
elif ! grep -q "HIPDNN_ENABLE_KERNEL_INGESTOR:BOOL=ON" "$BUILD/CMakeCache.txt"
then echo "INGESTOR OFF -- configured without -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON"
else echo "CONFIGURED AND ON -- binary missing means the build did not run or failed"
fi
```

- **NO BUILD / NOT CONFIGURED.** This is the *normal* state at step 5 if you are working
  in a fresh worktree: `$BUILD` is defined at the top of this file as "an existing build,
  or one you configure at step 7", and step 5 runs before step 7. Nothing is wrong. Either
  configure one now with **`skill://hipdnn-superbuild`** (do not hand-roll a configure),
  or point `$BUILD` at another worktree that already has one — the validator only reads
  your packed tree, so a sibling's binary is a valid substitution as long as its own cache
  says `HIPDNN_ENABLE_KERNEL_INGESTOR:BOOL=ON`. Say which you did.
- **INGESTOR OFF.** Reconfigure with `-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON` (default OFF).
- **CONFIGURED AND ON.** The build never ran, or it failed. Build, and read the error.

In every case, if you do not run the validator, state by name that structural validation
did not run, and which of the three causes applied. Never report a bundle as validated
when the binary was never invoked.

### 5d. Desk-check the shipped set — no GPU required

The validator proves the tree parses and cross-references. It does **not** check that your
variant set is internally coherent or that anything will match. Four invariants do, they
cost seconds, and each catches a defect that otherwise surfaces at stage 8 or never:

```bash
PYTHONPATH=$PROVIDER/descriptor-packaging/python \
  python3 $PROVIDER/descriptor-packaging/tools/hkp_desk_check.py \
          <the shipped .kdp.json under the packed tree>
echo "EXIT=$?"
```

**GATE:** exit 0. Exit 1 means an invariant is violated *or* could not be evaluated — the
tool refuses to report a clean result it did not actually check.

What the four are, and why each matters:

1. **Metadata must agree with the spec it claims to describe.** The matcher reads
   `metadata`; the compiler read `spec`. Drift between them is invisible and fatal. The
   spec moves during packing (`kernel_source.spec` when authored, `provenance.spec` once
   packed), so the tool checks both and reports `COULD NOT CHECK` when neither is present
   rather than a false clean.
2. **No two kernels may share a matcher tuple on the same arch** — one of them is
   unreachable forever.
3. **Every variant must be individually addressable**: distinct `toc_key`. A collision
   means two variants resolve to one blob.
4. **Symbol names are NOT guaranteed unique**, and that is legal. A rocKE `kernel_name()`
   may omit fields the kernel still bakes in — read yours and compare what it interpolates
   against what the spec pins. Uniqueness is `(toc_key, symbol)`; never key anything on the
   symbol string alone.

Your spec and your metadata may spell the same value two different ways on purpose — a
rocKE spec's `"bf16"` against a KMD's `"BFLOAT16"`. That is not drift and the tool knows
it: dtype spellings are normalised before comparison, so `bf16`/`BFLOAT16` agrees while
`bf16`/`HALF` still fails. For any *other* field your engine translates, narrow the drift
comparison with `--drift-field` — **not** `--field`. `--field` is the matcher-tuple
identity: dropping a field from it removes that field from what makes a variant distinct,
and invariant 2 then reports collisions between variants that are genuinely different.
The two lists are deliberately separate for exactly this reason.

This logic ships as a tested module rather than a snippet in this file, because the
snippet it replaces was **dead code for its entire life**: it read only
`kernel_source.spec`, which does not exist on the packed tree this step tells you to point
it at, so invariant 1 printed "none" no matter what. Code that lives in prose cannot be
tested, and untested checks fail silently in the direction that looks like success.

Then the check that decides whether stage 8 can pass at all: **for each test graph you
plan to run, does a shipped variant match it?** Derive the tuple the matcher will compute
from the graph — remembering any field that is a *derivation* rather than a copy, which
`graph_contract.md` §5 already listed — and look it up in the set. A bundle with no
matching variant is declined, and a stage-8 run of declined graphs proves nothing while
looking like a green suite.

### 5e. Gate the SET — properties no per-bundle check can see

5d checks one bundle against itself. These are properties OF THE SET, and each one
failed at least once while every other check stayed green:

```bash
cd $REPO
$GEN/.venv/bin/python $GEN/tools/verify_variant_sets.py \
    A /tmp/$SLUG/descriptors \
    --profile $GEN/configs/$SLUG.profile.yaml
echo "EXIT=$?"
```

**GATE:** exit 0 **and no `NOT CHECKED` line in the output.** Without a profile the tool
still runs its structural checks and reports the rest as NOT CHECKED, by name — that is a
narrowed run, not a pass, and treating it as one is how a gate stops looking while still
printing a reassuring last line.

Comparing several sets — a parity arm against a wider one — passes them in order, and
each must be a binary subset of the next:

```bash
$GEN/.venv/bin/python $GEN/tools/verify_variant_sets.py \
    A /tmp/${SLUG}_A/descriptors  C /tmp/${SLUG}_C/descriptors \
    --profile $GEN/configs/$SLUG.profile.yaml
```

What it checks, and why none of it is a count:

1. **Nesting is about COMPILED BINARIES, not labels.** "Do more variants help?" is only
   answerable if the larger set can still choose everything the smaller one could.
   Normalised labels once made two sets look nested while the binaries diverged on 43
   shapes.
2. **Catalog tuples must be unique once KMD `default_value` is substituted.** A duplicate
   does not drop an entry — it drops the WHOLE ENGINE at load, and the arm then serves
   every graph from a different engine while exiting 0 and passing a descriptor count.
3. **No descriptor ships the unset sentinel**, which describes no compiled artifact.
4. **No descriptor's metadata mislabels its binary**, including for a knob the kernel
   resolves by POLICY: `absent` and `explicitly false` are different kernels whenever the
   policy would have answered true.
5. **dtype is in the matcher's vocabulary, not the builder's.** The wrong spelling loads
   cleanly, reconciles on every count, and matches nothing.

**A descriptor-count check answers none of these.** Counts are about disk. A set that
passed a count gate served zero graphs.

A superset must ADD, never REPLACE. Where a wider set overrides a knob the narrower one
leaves to policy, it has to carry BOTH variants — overriding alone silently removes the
narrower set's kernel from the candidate list, and the nesting check is what catches it.

### 5f. The three rungs, in one command — and what each one CANNOT tell you

```bash
cd $REPO
$GEN/.venv/bin/python $GEN/tools/coverage_gate.py \
    --tree      $BUILD/lib/hipdnn_plugins/engines/arch_content \
    --profile   $GEN/configs/$SLUG.profile.yaml \
    --validator $BUILD/bin/hipdnn_validate_descriptors \
    --expect-engine "$ENGINE" \
    --min-served <N>
```

| rung | question | needs | catches |
|---|---|---|---|
| 1 STATIC | do the descriptors describe what they claim? | nothing | mislabelled metadata, sentinels, duplicate tuples, broken nesting |
| 2 LOADS | does the ENGINE survive the loader's own rules? | a build | **the dropped engine** — nothing else can |
| 3 SERVES | does it serve graphs, and how many? | a GPU | everything the first two structurally cannot see |

**GATE:** rungs 1 and 2 PASS with no `NOT CHECKED` and no `NOT RUN`. Rung 3 is reported
as owed and you still owe it.

**Point rung 2 at the PACKED tree, not the authored one.** `kind: rocke` is an authoring
form that `hkp_pack` lowers to `kind: kpack` at build time; the runtime loader has never
heard of `builder` and rejects it. Aiming rung 2 at `/tmp/$SLUG/descriptors` therefore
fails with a real-looking error about an unknown key, which is the loader being correct.

Why three and not one. A descriptor-count gate once passed on an arm that served
**zero** graphs. The count was right — every descriptor was on disk, correctly named.
They never reached a GPU: a duplicate catalog tuple made the loader reject the whole
engine, every graph fell through to a different one, and the phase ran to completion
and exited 0. Rung 2 is the cheap check that would have caught it, and it is the one
that was missing.

Rungs 1 and 2 green means *the descriptors are well-formed and the engine loads*. It is
not evidence that anything was served, and the tool refuses to imply otherwise. When you
do run rung 3, **filter on `engine_name`**: a graph another engine served is not your
coverage, and an aggregate that does not filter reports someone else's work as yours.

### 5g. The CONVERSE question: can any graph SELECT this variant?

5d asks, for each graph, *"does a shipped variant match it?"*. Nobody asked the other
direction, and that is where dead weight hides:

```bash
cd $REPO
$GEN/.venv/bin/python $GEN/tools/variant_reachability.py \
    --kdp    <the shipped .kdp.json> \
    --shapes $SHAPES \
    --field-map nhead_q=num_query_heads --field-map nhead_k=num_kv_heads \
    --field-map seqlen_k=seqlen_kv      --field-map hdim_q=head_size \
    --divides block_m=seqlen_q --divides block_n=seqlen_kv \
    --score-field block_n --score-prefer max
```

**GATE:** zero `UNREACHABLE` and zero `APPLICABLE-BUT-NEVER-WINS`, or a stated reason for
each. The `--field-map` and `--divides` flags are per-op: a corpus names its fields the
way the REQUEST does and metadata names them the way the MATCHER does, and a tile knob
is applicable when it DIVIDES a sequence length rather than equalling it.

Three buckets, and the middle one is the whole point:

| bucket | meaning |
|---|---|
| SELECTED | wins for at least one shape. Fine. |
| **APPLICABLE-BUT-NEVER-WINS** | legal everywhere it applies, and something always outranks it **on the cold path**. Read the caveat below before calling it dead weight. |
| UNREACHABLE | applicable to no shape at all. Either the corpus is missing a family or the variant should not exist. |

**`variant_reachability.py` models the PRE-AUTOTUNE path only, and cannot model
measurement.** It reproduces `score`-based ranking; it cannot call the native scorer or
run a benchmark. So `APPLICABLE-BUT-NEVER-WINS` means **"never wins before autotuning"**,
not "can never be selected". Where the plan builder benchmarks, every applicable candidate
is measured and the winner persisted — those variants are the autotuner's candidate set,
which is what the sizing discussion asks you to ship. A reader who missed this concluded a
shipped set was three-quarters dead weight; it was not, and the claim was withdrawn.

What the bucket *does* tell you is real: on the cold path the pick among tuned siblings is
decided by `score`, and if `score` is constant across them the tie breaks on an arbitrary
identifier. That is worth fixing in `score`. It is not a reason to delete variants.

An integration once shipped 48 variants of which **24 no graph could select**. Not
laziness in authoring: every shipped shape had a sequence length divisible by the wider
tile, so both tiles were always applicable and `score` — which ranks the wider one higher
— chose it every single time. Half the tuning axis was unreachable, and the suite was
green throughout, because every other gate in this runbook is structurally blind to it:
5d, 5e and 5f all ask about variants that DO match, never about one that never wins.

**The fix for a never-wins variant is a shape where its rival is ILLEGAL, not another
variant.** Above, that meant adding a sequence length the wider tile cannot divide.

If you declare no ranking the tool treats every applicable variant as reachable **and
says so** — it cannot call your native `score`, so it refuses to guess rather than
reporting a pass it did not earn.

---

## Step 6 — Implement the native pack. THIS IS THE WORK.

```
Produces:      packs/<Name>Native.cpp with all five hooks implemented, and a
               launch_surface: block in the profile enumerating what it restates
Gate:          `generate.py --check-placeholders` exit 0, and launch_surface.py --check clean
Typical time:  half a day. The largest step, and the point of the whole run.
```

### 6a. Declare what your C++ RESTATES, before you finish writing it

Everything the Python launch path computes, this file recomputes by hand — grid, block,
kernarg order, baked constants, the dispatcher's spec resolution, applicability. **No
build step, packer, validator or test compares the two halves**, and a mismatch does not
fail: the kernel runs and computes something else. Two defects shipped this way.

So write it down as you go, in your profile's `launch_surface:` block — one entry per
surface with its Python source, its C++ mirror, the KMD fields it branches on, its guard
and its test. The worked six-surface block lives at the bottom of the integration
branch's profile.

```bash
cd $REPO
$GEN/.venv/bin/python $GEN/tools/launch_surface.py --check \
    $GEN/configs/$SLUG.profile.yaml
```

**GATE:** zero structural failures. Every surface with `guard: none` or `test: none` is
named and fails the command — either fix it, or accept it deliberately with
`--allow-unguarded` and say so at step 10.

The structural check is not bureaucracy. It verifies that **every KMD field your C++
branches on actually exists in the descriptor**, which is a real defect this project hit:
the engine read `block_m` from metadata on two paths and the profile never declared it,
so descriptors stated no tile at all. The C++ compiles fine either way — adding a
mirrored branch is TWO changes and only the first fails to build. `--report` emits the
table for your PR.


The generator wrote `/tmp/$SLUG/packs/*Native.cpp` with every body
`// TODO - FILL THIS OUT`. An engine in that state parses, validates, enumerates — and
serves zero graphs, with every mechanical check green. This is the state agents mistake
for done.

```bash
cp /tmp/$SLUG/packs/*Native.cpp \
   $PROVIDER/src/engines/kernel_ingestor_engine/packs/
```

**Read `native-pack.md` now** and implement all five hooks. Your step-2 rejection
checklist is the `graph_match` body, in its severity order — silent-wrong-answer checks
first.

**GATE — two commands, and the second is the one that catches real holes:**

```bash
# 1. No unfinished hook, in ANY file this engine ships. Derived from the config
#    and located by BASENAME, so it covers the generated tests/ stubs a
#    packs/*Native.cpp glob silently skipped.
#
#    Point it at a root spanning BOTH halves of the splice: the provider puts
#    packs in the engine dir and the test stubs under src/tests/engines/... .
#    $PROVIDER/src covers both. A file it cannot locate is an unfinished splice
#    and fails -- "found nothing" is never a pass.
$GEN/.venv/bin/python $GEN/generate.py --config <your-config> \
    --output-dir $PROVIDER/src --check-placeholders

# 2. No graph attribute silently ignored. ONE schema, and every pack source of
#    this engine -- pass them all, a multi-pack engine splits its handling.
$GEN/tools/field_audit.sh \
    $REPO/projects/hipdnn/flatbuffers_sdk/schemas/<op>_attributes.fbs \
    $PROVIDER/src/engines/kernel_ingestor_engine/packs/<Name>Native.cpp
```

The first must be `0`. A `// TODO` in a path the engine reaches is an unfinished
integration, not a placeholder. Placeholders are allowed — a `score` that ranks one knob,
a `workspaceBytes` that returns 0 because the kernel needs no scratch — *if you say so and
say what would replace it*. Silence is not.

The second must print nothing you have not deliberately accounted for. An empty
`grep -c` and a wall of `UNCHECKED:` lines is the exact state that ships a matcher which
ignores an attribute the graph can set — on a branch that ships a rocKE SDPA pack, that
pack still prints `UNCHECKED: implementation`. `native-pack.md` § 6 has the reasoning and what
counts as accounted for.

---

## Step 7 — Splice, build, pack, confirm

```
Produces:      an engine that builds, packs, stages and enumerates
Gate:          your FNV-1a id appears in hipdnn_list_engines output
Typical time:  1-2 hours, dominated by the build
```

### 7a. Splice

Read the emitted fragments first — for a packaged bundle they tell you what does *not*
apply:

```bash
cat /tmp/$SLUG/fragments/*.txt
```

| Point | File | Packaged? |
|---|---|---|
| 1 | `HIPDNN_DESCRIPTOR_FILES` | **NEVER** |
| 2 | `HIPDNN_INGESTOR_PACK_KERNELS` | **NEVER** |
| 3 | the engine's own `CMakeLists.txt` `target_sources` — `git ls-files '*engines/kernel_ingestor_engine/CMakeLists.txt' ':!*tests*'` (the `:!` excludes the test tree; without it the pattern returns row 5's file too, and these are different splice points) | yes |
| 4a | `IngestorPacks.hpp` — the declaration | yes |
| 4b | `IngestorPacks.cpp` — the `s_packs` row | yes |
| 5 | the engine's unit-test `CMakeLists.txt` — the same glob under the provider's test tree (`git ls-files '*tests*kernel_ingestor_engine/CMakeLists.txt'`) | yes |

4a and 4b are **two edits for one pack**. Declaration without the row: the static-archive
linker drops the translation unit, so the pack vanishes from unit tests while the plugin
`.so` still works — no build error either way.

**Check the struct before writing the row**; the fragment may not match it. Do not
eyeball this — a test already does the comparison mechanically, against the real header:

```bash
sed -n '/struct IngestorPack/,/};/p' \
    $PROVIDER/src/engines/kernel_ingestor_engine/IngestorPacks.hpp
$GEN/.venv/bin/python -m pytest $GEN/tests/test_fragment_struct_arity.py -q
```

**GATE:** that suite green. It parses the real `IngestorPack` struct out of the real
header and compares its field count against the arity the generator emits, so a struct
that gained a field while the template did not is caught here rather than as a confusing
link error later. A `sed` print alone is a look-and-compare instruction, which is exactly
what got hand-splicing wrong before.

### 7b. Build and pack

**The build flags, which are easy to get wrong.** Two of these read like each other and
do entirely different jobs:

| Flag | Default | Job |
|---|---|---|
| `HIPDNN_ENABLE_KERNEL_INGESTOR` | **OFF** | The ingestor itself: descriptor loading, the kpack adapter, and `hipdnn_validate_descriptors`. **ON for any descriptor-backed integration** — nothing here works without it, and it is why the validator is usually missing. |
| `HIPDNN_ENABLE_<OP>` | **OFF** | The **frontend** for an op that has one. With it off the graph API for that op is `#ifdef`-compiled out, so the graph cannot be expressed at all and every plan silently DECLINEs. Must be ON in **both** the hipDNN SDK at `$REPO/projects/hipdnn` and the provider. Check whether your op has such a flag: `grep -rhoE "HIPDNN_ENABLE_[A-Z_0-9]+" --include=CMakeLists.txt --include=*.cmake $REPO/projects/hipdnn` — most ops have none and are always compiled in. |
| `ENABLE_<X>_ENGINE` | often **ON** | A **competing** hand-written engine for the same op. Nothing to do with the frontend, despite the similar name. Yours must beat it, or be pinned past it (8c). |
| `HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE` | **OFF** | Required for a rocKE `packaged` engine. With a production source root set and every language backend off, `HkpPackaging.cmake` **aborts the configure** rather than packing nothing — so this reads as a CMake error, not a missing engine. Its `_HIP` sibling does the same job for HIP kernels. |

A missing frontend flag wastes a whole build: the provider compiles, your engine
enumerates, and every graph declines with nothing pointing at the flag.

**This table is for recognising the flags, not for enumerating them.** CMake owns the
list; a hand-maintained mirror falls behind it. Get the current set and its defaults
from the source before configuring:

```bash
# `option(...)` AND `set(... CACHE BOOL)` -- the packaging flags use the latter,
# so an option()-only grep silently omits every one of them.
git grep -nE '(option|set)\((HIPDNN_ENABLE|HIPKERNELPROVIDER_PRODUCTION)[A-Z_]*' \
    -- '*CMakeLists.txt' '*.cmake'
```

**Use `skill://hipdnn-superbuild` rather than hand-rolling a configure.** It carries the
repo-root rule, the preset table, the toolchain and the stale-cache retry; hand-rolling
those loses ~30 minutes to a system-compiler fallback that presents as dozens of
`-Werror` failures in files you never touched. If a build already exists, rebuild
incrementally rather than reconfiguring:

```bash
cmake --build $BUILD -j48 --target hip_kernel_provider hipdnn_list_engines \
    hipdnn_validate_descriptors hip_kernel_provider_tests

# PACKAGED DIALECT: none of the above packs your descriptors. This does.
cmake --build $BUILD -j48 --target hkp_packaging_product
```

Without that second command your engine is absent at runtime with every other check
green. Confirm it staged where the loader looks:

```bash
find $BUILD/lib/hipdnn_plugins/engines/arch_content -name '*.kdp.json'
```

(`hkp_packaging_testfixture` is the *other* source root and is not yours.)

### 7c. Confirm the engine loads

```bash
python3 -c "
h=0xcbf29ce484222325
for b in '$ENGINE'.encode(): h=((h^b)*0x100000001b3)&0xFFFFFFFFFFFFFFFF
print(f'expect 0x{h:016X}')"
$BUILD/bin/hipdnn_list_engines
```

**GATE:** your engine appears. `grep $ENGINE` should now find it by name — the provider
answers `getEngineName` for descriptor-registered engines from the UED's `name` field.
The FNV-1a above is the id that name hashes to: compute it as a cross-check, and use it
if a tool renders ids rather than names.

Missing engine, in the order worth checking: (1) the pack step never ran, (2) splice point
4, (3) a symbol mismatch — isolate with
`hipdnn_validate_descriptors --native-source <pack.cpp>`.

---

## Step 8 — Test, on the target arch

```
Produces:      bundles in both tiers, C++ negatives, a pinned CI target, a device run,
               and an exploratory pass over dnn-benchmarking's workloads (8e)
Gate:          a real graph dispatched and matched a reference on $ARCH, and 8e's
               three-bucket triage recorded
Typical time:  half a day, plus whatever the GPU queue costs you
```

Enumeration proves construction, not matching. Only a real graph on the target arch proves
the engine serves anything — and adding to the shared suite is a **required deliverable**,
not an optional extra. An integration whose only evidence is a one-off script leaves
nothing behind: the next change to the matcher, the kernel or the packager has no way to
notice it broke.

### The two tiers, and you owe both

The split is **functional breadth vs numeric depth**, not "one case vs many".

| Tier | Question it answers | Content |
|---|---|---|
| **quick** | Is every supported feature wired up and matching? | Many tiny graphs, one per meaningful support combination, smallest legal shape. Fast enough to run on every change. |
| **standard** (and `full`) | Is it numerically right at sizes people use? | Realistic shapes, deeper verification, combinations too expensive per-commit. |

Quick's job is functional signal, not numeric confidence. If a supported option is never
exercised there, the commit that silently unwires it ships green.

**Deciding your op's matrix is a real design decision — make it deliberately.** There is
no universal axis list: a normalization's interesting axes are nothing like an attention
op's, and a matmul's are different again. Derive yours from the step-2b constraint table —
the axes a *graph* can vary and your matcher claims to support. Typical families: dtype, a
shape parameter the kernel specializes on, an optional mode flag, memory layout, a fused
epilogue, a degenerate or boundary shape. Which exist, and which are independent, is yours
to work out.

Then prune against a time budget, deliberately:

- Cover each supported feature **at least once**. Zero coverage means nobody notices it
  breaking.
- Prefer *independent* combinations over a full cross-product. If two axes do not interact
  in the kernel's code paths, you do not need every pairing.
- Weight toward what step 2b flagged as fragile — layout handling, boundary shapes,
  anything whose failure mode is silent.
- **When the budget binds, drop coverage rather than slow the tier down.** A quick tier
  that takes minutes gets disabled, and then it protects nothing. Move what you dropped to
  standard and say so.

For scale, count the neighbours rather than guessing — and note the two families count
differently, so do not compare them directly:

```bash
cd $REPO/dnn-providers/integration-tests/integration-test-bundles/quick
# per-graph ops: one directory per case
find <Op> -name '*.json' ! -name '*.meta.json' | sed 's|/[^/]*$||' | sort -u | wc -l
# sweep ops: cases live inside sweep.json
python3 -c "import json,glob; print(sum(len(json.load(open(f))['cases']) for f in glob.glob('<Op>/*/sweep.json')))"
```

**The rejection checklist is a coverage list too.** Each "must decline" row deserves a
negative case — cheap, and exactly the assertion that catches an over-broad matcher before
it returns wrong numbers. Those belong in C++ (8b), not in bundles: a bundle for a graph
you decline is simply served by another engine.

### 8a. Bundles — graph coverage

**Assume you owe NEW bundles, and budget for it.** The shipped bundles were authored for
whatever engine came before you. They are not a coverage plan for your engine, and reusing
them is the single easiest way to produce a green run that tests nothing: a bundle your
matcher declines SKIPs, and a bundle your matcher accepts *by accident* — right op, wrong
layout, wrong dtype — passes without exercising the path you wrote.

This is the step that was done worst on the first run of this skill, and it was invisible
for exactly that reason: the suite went green early and stayed green while coverage was
thin.

**Derive the bundle list from your matcher, not from what exists.** Work down the step-2b
constraint table and the `graph_match` body you wrote at step 6, and for each branch ask
what graph reaches it:

| Your code | Bundle you owe |
|---|---|
| every feature you ACCEPT | at least one bundle that exercises it and passes |
| every feature you DECLINE | a C++ applicability negative (8b) — cheaper than a bundle, and it must assert the *reason* |
| every shape axis you specialize on | a bundle at each specialization, not just the middle |
| every boundary in the spec (`% block_n`, min/max, unit extents) | a bundle either side of it |

Then count: **an accepted feature with no bundle is untested, and nothing will tell you.**
Write that count in the step-9 report next to the feature list.

**Bundles cost real time — check before authoring, not after.** The reference executor
runs on every case, and `gpu-ref` is one thread per output element, so cost scales with
the full attention/output extent. A production-sized shape can exceed a whole tier's
budget on its own; see 8e for how that was found the expensive way. Cost the shape first,
then pick the tier, then author.

Check what the shipped bundles for your op already do, because they may not match your
kernel's layout:

```bash
ls $REPO/dnn-providers/integration-tests/integration-test-bundles/quick/
python3 -c "
import json; d=json.load(open('<a shipped bundle>.json'))
[print(t['name'], t['dims'], t['strides']) for t in d['tensors']]"
```

If your kernel bakes a layout the shipped bundles do not use, you need your own — a graph
your matcher correctly *declines* proves nothing about it.

**Check what verifying a bundle COSTS before authoring it.** The shared reference is
correctness-first, not fast: `gpu-ref` runs one thread per output element, each looping
over the full contraction, untiled. Cost grows with the problem, and production shapes can
exceed a tier's entire budget by orders of magnitude — in one run, milliseconds at a toy
size, ~20 minutes at a mid size, and ~8 hours at the largest real shape, against tier caps
of 600 s (smoke) and 1800 s (standard). Those are not slow tests; they never finish.

So: verify correctness at shapes the reference can evaluate, and push the production
shapes to step 8e, where `dnn-benchmarking` validates against PyTorch instead. A bundle
you cannot afford to verify does not belong in a tier.

Author bundles at `integration-test-bundles/{quick,standard}/<Op>/<layout>/<dtype>/<name>/`.
Quick = many tiny graphs, one per supported feature, smallest legal shape. Standard =
realistic sizes. They are auto-discovered:

```bash
cmake --build $BUILD -j48 --target hipdnn_integration_tests
$BUILD/bin/hipdnn_integration_tests --gtest_list_tests | grep <your-bundles>
```

`file(COPY)` runs at configure time, so a *new* bundle needs either a reconfigure or a
copy into `$BUILD/lib/integration-test-bundles/`.

### 8b. Applicability negatives — C++

Every "must decline" row of your rejection checklist deserves a case. These belong in C++,
not a bundle: a bundle for a graph you decline is just served by another engine. Model on
a sibling pack test in `src/tests/engines/kernel_ingestor_engine/packs/`.

### 8c. Wire the engine into CI — otherwise the tests you wrote never run against it

**This is a required deliverable, not a follow-up.** Bundles are discovered and run
against whatever engine *wins* the graph. A new engine competing with an established
incumbent for the same op may never execute while the suite passes — green, and blind to
you.
Adding tests without this step leaves no CI evidence that your engine works, and the next
change to the matcher, the kernel or the packager breaks it silently.

Two local aids while you develop, neither a substitute for the pinned target: build with
the incumbent engine's `ENABLE_<X>_ENGINE=OFF` so it cannot win (proves nothing about CI,
where it is present), and **assert on engine identity, not only on numbers** — a test that
checks the output is correct proves *something* computed it; one that also checks which
engine was selected proves yours did.

The mechanism is one target per engine, pinned with `--test-engine`:

```cmake
# dnn-providers/hip-kernel-provider/src/integration_tests/CMakeLists.txt
if(NOT COMMAND add_external_integration_test_target)
    find_package(hipdnn_integration_tests CONFIG QUIET)
endif()

# Gate on the engine actually being BUILT. The descriptors exist only when the ingestor
# is on AND the rocKE producer ran over a source root; without that the engine never
# appears and every case FAILS rather than skips.
if(HIPDNN_ENABLE_KERNEL_INGESTOR AND HIPDNN_ENABLE_SDPA
   AND HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE
   AND TARGET hipdnn_integration_tests AND COMMAND add_external_integration_test_target)
    add_external_integration_test_target(
        TARGET_NAME   <provider>_<engine>_gpu_ref_integration_tests
        PLUGIN_TARGET hip_kernel_provider
        ENGINE_NAME   <namespace:Engine>      # exactly as the UED spells it
        INSTALL_SUBDIR hip_kernel_provider
        REFERENCE_EXECUTOR gpu
        GTEST_FILTER  Smoke/<YourSuite>.Correctness/*
    )
endif()
```

Find the helper that defines the registration function and read it for the full option
list (`TEST_CONFIG` for per-case tolerance, `ENVIRONMENT`, `TEST_CATEGORIES_YAML`):

```bash
git grep -ln 'function(add_external_integration_test_target' -- '*.cmake'
```

Copy the shape of an existing block — grep that helper's callers for an `ENGINE_NAME`
already in use and use the closest match as the worked example:

```bash
git grep -n 'ENGINE_NAME' -- '*CMakeLists.txt'
```

**There are TWO registration sites and they are not interchangeable.** The block above
goes in the provider's own `src/integration_tests/CMakeLists.txt`, beside the other
engine-pinned targets. But the provider ALSO registers engine-scoped checks one level up,
in the provider's top-level `src/CMakeLists.txt`, and **that is the site whose
targets CI drives**: those blocks carry `TEST_CONFIG` (per-engine tolerances),
`TEST_CATEGORIES_YAML` and `INSTALL_TEST_FILE`, which is the plumbing that makes a target
appear as a named, labelled `<project>-<engine>-external-integration-check` in the
INSTALLED `CTestTestfile.cmake` — the file TheRock's `test_runner.py` drives `ctest -L
quick` against.

```bash
# What the two sites look like on disk. Read BOTH before writing yours.
grep -n "add_external_integration_test_target" \
    $PROVIDER/src/CMakeLists.txt \
    $PROVIDER/src/integration_tests/CMakeLists.txt
```

A target written from the inner site alone works when you run it by hand and is **not
wired the way the shipped engines are** — it will not appear in the installed CTest file
and CI will never run it. Follow the `-external-integration-check` naming convention:
that suffix is what the category labelling and the CI driver both key on.

**Then run it from the INSTALL tree, not the build tree.** The registered target stages an
install-tree `add_test()` specifically for CI flows that invoke ctest from an installed
artifact, and its real invocation is `--test-article <plugin.so>` + `--reference-executor`
— not the `--plugin` + `--bundle-dir` form an ad-hoc command line reaches for:

```bash
cmake --install $BUILD --prefix $INSTALL
cd $INSTALL && ctest -R "external-integration-check" -V
```

**Anchor the selector when you run ctest yourself.** `-R`, `-L` and `-E` are unanchored
regexes, not literal names. `ctest -L quick` also selects every `ffm-quick` suite — on a
`hipdnn-dev-all` build that is 74 tests rather than 62, and the 12 extra are the on-device
ones, so the run roughly doubles and reads as a hang. Use `-L '^quick$'`, and dry-run
`ctest -N -L '^quick$'` first: `Total Tests: 0` means the selector matched nothing (a
green exit 0 that proves nothing), and a count well above what you expected means the
regex widened. The same applies to `-R "external-integration-check"` above, which
deliberately matches every provider's check — narrow it when you want just one.

Testing only out of a build tree with hand-rolled arguments for a whole run, and never
executing the registered target, is a real cost already paid. The install-tree run is the
one that proves the wiring, and step 9c makes it a gate.

Four things that decide whether this works:

1. **`--test-engine` pins exactly ONE engine.** A case belonging to a different pack
   reports not-applicable and skips. So it is **one target per engine**, not per provider
   — and each target's TOML/comment should say which cases it expects to skip.
2. **`REFERENCE_EXECUTOR gpu`** computes the reference live, which is what makes
   golden-free bundles checkable. Confirm the GPU reference *accepts* your graphs first
   (see step 1b) — it declines dropout, paging, varlen, stats and fp8 descale.
3. **Gate on the build flags that produce your engine.** An ungated block turns "engine
   not built" from a skip into a wall of failures.
4. **The engine must resolve to a NAME, not a bare id, for `--test-engine` to select it.**
   The harness matches against the loaded engines' `engineName` — the integration-test
   harness `main.cpp`, `git grep -ln 'engineName' -- '*integration*/src/main.cpp'`; no
   match means it exits with `Error: Engine '<name>' is not loaded` before running a
   single case.

**Pin exactly what the UED spells.** For a descriptor-registered engine the provider's
`Container::getEngineName` answers from the UED's own `name` field, so
`ENGINE_NAME` and `"name"` in the `.ued.json` must be the same string. A mismatch is the
one way this still fails, and it fails before any case runs.

- **Do not** substitute a workaround — do not drop `ENGINE_NAME`, do not pin a different
  engine, do not delete the target. An inert correct target beats a green misleading one.
- **Do not** try the hex id from `hipdnn_list_engines` instead. `--test-engine` matches
  names; the id is the obvious thing to reach for and every case reports
  not-applicable and SKIPs, which reads like a matcher bug rather than a wrong flag.
- **Your other stage-8 evidence does NOT stand on its own without this.** An unpinned run
  reports whichever engine won the graph, and green tells you nothing about yours: this
  integration once reported 14/14 on gfx942 while hipDNN's own backend path served every
  case. The pinned target is what makes the on-device run evidence about your engine.

### 8c-2. When a correctness case fails: instrument FIRST, bisect second

Turn the log on before you form a hypothesis:

```bash
HIPDNN_LOG_LEVEL=info HIPDNN_LOG_FILE=/tmp/$SLUG-fail.log <the failing command>
grep -nE "NaN|Inf|not written|declin|cannot build a plan" /tmp/$SLUG-fail.log | head
```

Three cluster jobs were once spent bisecting a failure whose cause the log stated in one
line, unprompted, the first time it was enabled. The log is cheaper than every experiment
you are about to run.

**Read `allClose=false` with ZERO finite mismatches as "an output element was never
written".** Never as a tolerance problem. The harness sentinel-fills outputs with NaN
precisely so an unwritten element cannot pass — and the diff report counts only FINITE
mismatches, so it prints `Mismatched: 0 (0.00%)` and `Max abs diff: 0.000000e+00` **while
failing**. That reads as a contradiction and sends people toward tolerances for rounds.
It is the documented design; the report simply does not say so at the point of failure.

**When your result disagrees with an upstream harness, compare the OUTPUT FILL before
concluding either is wrong.** A benchmark that zero-fills its output buffer and a test
harness that NaN-fills it will disagree honestly about the same kernel: with zeros, an
element the kernel never writes contributes an error indistinguishable from rounding
noise at low density, so a structural bug hides inside dtype tolerance. The sentinel-filled
one is the harness telling you the truth. This is a one-line difference between two
honest harnesses, and it has explained a "green upstream, failing here" standoff before.

### 8d. Run it, on `$ARCH`

> ## GREEN DOES NOT MEAN GOOD. READ THIS BEFORE YOU REPORT A PASS.
>
> **Green means: everything that passed applicability and actually ran was correct.** It
> says *nothing* about what ran. These all produce a green suite:
>
> - a `graph_match` bug that declines every graph — every case SKIPs, ctest exits 0
> - packaging that shipped no descriptors — the engine never registers, cases SKIP
> - a variant set that covers none of the bundles — matched nothing, SKIP
> - too few bundles to exercise the features you implemented — nothing to fail
> - a filter narrow enough that the interesting cases were never selected
>
> A skip is not a pass. `ctest` counts it as not-a-failure, which is not the same thing,
> and the summary line will not tell you the difference. **This integration reported
> `14/14 passed` on real hardware while a different engine served every single case.**
>
> ### The check, every run, before you claim anything
>
> ```bash
> # The two locals this block needs. LOG is wherever you captured the run's stdout;
> # the engine id is the FNV-1a hash the plan log prints, which step 7c derives.
> LOG=<the test run's captured stdout>          # e.g. /tmp/$SLUG-ctest.log
> YOUR_ENGINE_ID=<0x... from step 7c>           # the id, not the name
>
> # 1. What actually ran, and what did not.
> grep -cE "^\[       OK \]" $LOG          # passed
> grep -cE "^\[  SKIPPED \]" $LOG          # did NOT run -- these prove nothing
> grep -E "SKIPPED|does not support this graph|is not loaded" $LOG | sort | uniq -c
>
> # 2. Did YOUR engine serve them? Engine id in the plan log, not just a green line.
> grep -c "engineId.*$YOUR_ENGINE_ID" $LOG
> ```
>
> An unset `$LOG` greps an empty path and prints `0` — which reads exactly like "zero
> skips" and is the most flattering possible misreading. Set it, and sanity-check that
> the passed count is non-zero before you trust the skip count.
>
> **Reconcile the numbers out loud.** `passed + skipped == total`, and you must be able to
> say why each skip is expected — a declined feature, a different pack, an arch guard. A
> skip you cannot explain is a finding. Zero dispatches of your engine id is a **failure**
> reported as success.
>
> ### State it in the step-9 report as a fraction, never as "green"
>
> > "N of M cases dispatched this engine (id `0x…`); K skipped, each because
> > `<reason>`; 0 unexplained."
>
> "All tests passed" is not a stage-8 result. If you cannot produce that sentence with
> real numbers, you have not finished stage 8 — you have finished running the tests.
>
> ### The blunt tool, and its limits
>
> `hipdnn_integration_tests --fail-on-unsupported` turns "no engine supports this graph"
> into a `FAIL` instead of a skip. Useful, and **all-or-nothing**: it fails on *any*
> unsupported graph, including the ones your engine correctly declines, so it only works
> when your `GTEST_FILTER` already excludes everything you reject. There is no way to
> declare *which* cases must run — and `add_external_integration_test_target` has no
> passthrough to set the flag, so it is a manual run only. Per-target expected-coverage
> claims are coming; until they land, **the count reconciliation above is the gate**, and
> it is manual.

**Detach long jobs properly.** `nohup … &` dies with your session and leaves an empty log
that reads exactly like a job still running. Use `setsid nohup … < /dev/null &`.

**A stage-8 failure is not automatically yours.** `No space left on device` or
`pyxis: failed to create container filesystem` means the payload never started — re-run
with that node excluded before debugging the engine.

Run on the machine step 1c established. **Do not substitute another arch** — packs
arch-prune before the matcher, so a clean run on the wrong GPU reads as success while
proving nothing. Confirm with `rocminfo | grep -m1 -E "Name:\s+gfx"` on the machine that
actually executes, not the one you typed the command on.

**GATE:** a real graph dispatched and matched a reference, on `$ARCH`. Zero-filled inputs
are not verification: `softmax(0)·0 = 0`, and so is the output of a kernel that never
wrote a byte.

When something fails, **suspect applicability before the kernel**: which variant was
selected, what does its `spec` pin, and does the failing graph differ on any axis? A
kernel computing the wrong answer for a problem it was never compiled for is behaving
correctly — the defect is upstream.

### 8e. Exploratory validation against `dnn-benchmarking` — REQUIRED, not a CI gate

**Read `workloads.md`** — it carries the setup (including the submodule route that points
the tool at YOUR engine), the commands, and the perf comparison recipe.

```
Produces:      an exploratory report: PyTorch cross-validation + a benchmark sweep
Gate:          your bundles run there, AND the repo's workloads for YOUR op are triaged
Typical time:  1-2 hours, most of it environment setup
```

8a-8d prove the engine against **hipDNN's own reference** on graphs **you wrote**. Both
halves of that are a blind spot: a shared misunderstanding between your matcher and the
in-tree reference cancels out silently, and a bundle set you authored only covers shapes
you already thought of. This step closes both with an independent implementation and
someone else's workloads.

**It is required, and it is NOT CI.** `ROCm/dnn-benchmarking`'s own README says *"in
early development … Do not use it in build workflows or CI pipelines."* Respect that: do
not wire a target, do not add it to `ctest`. Run it by hand, record what you find, and
carry the findings into step 9. Nothing here gates a merge; everything here changes what
you know.

**It is a separate repository**, not part of this tree:

```bash
git clone https://github.com/ROCm/dnn-benchmarking && cd dnn-benchmarking
python3 setup_env.py --workspace .workspace --torch-mode rocm   # or --reuse-artifacts
source .workspace/.venv/bin/activate
```

It needs `hipdnn_frontend` (the **Python bindings**) and a torch build. The bindings are
a separate component: a provider-only preset does not build them, so configure with
`ROCM_LIBS_ENABLE_COMPONENTS="hipdnn;hipdnn-python;hipdnn-integration-tests;<your-provider>"`
or let `setup_env.py` build its own. Its graph JSON is the **same schema** as an
integration-test bundle — the same node/tensor structure your matcher already walks — so
your bundles feed straight in with no conversion, and its graphs load with the same
reader you used at step 2a.

**Three things it gives you that step 8 cannot:**

| | Why step 8 cannot |
|---|---|
| `--validate pytorch` | An **independent** reference. The in-tree GPU reference and your matcher can share a misunderstanding; PyTorch cannot participate in it. |
| `--engine <id>` | Selects by **numeric id**, a second way to attribute a run to your engine — useful as a cross-check that the id and the name resolve to the same engine. |
| `--pmc` / `--perf` / `--roofline` | rocprofv3 passes. Step 8 has **no** performance evidence at all, while `score` is required to rank on a real knob — this is where that ranking gets checked instead of asserted. |

#### 8e.1 Run your own bundles through it

```bash
dnn-benchmark --graph '<your bundle>/*.json' --validate pytorch -v
dnn-benchmark --graph '<your bundle>/*.json' --engine <your-id> -o mine.json
```

#### 8e.2 Triage the repo's shipped workloads — the high-value half

`Workloads/` carries DVC-tracked suites: per-library microbenchmarks
(`microbench/rocke.tar.gz`, `aiter`, `hipblaslt`, …) and **real model traces**
(`models/llama3.1.tar.gz`, `dsv3`, `gpt_oss`, `qwen3*`, …).

```bash
dvc pull Workloads/microbench/<yours>.tar.gz.dvc Workloads/models/<model>.tar.gz.dvc
dnn-benchmark --graph Workloads/models/<model>.tar.gz --validate pytorch -o model.json
```

**Enumerate the suites BEFORE you pull any — the denominator is the gate.** List every
suite in the checkout (`ls Workloads/*/*.dvc`), write the total down, then justify each
**exclusion** in one line. Report `served / declined / could-build` **out of that
enumerated total**, never out of what you happened to fetch. A suite you did not run is a
suite you cannot report on, and "I ran the ones I pulled" is not a scope decision.

This is the same shape as the step-3 gate, which is `covered / servable` rather than a
bare count of covered shapes. Without a denominator a numerator can be reported alone, and
it has been: a run triaged a single-digit percentage of the available graphs and carried
the numerator forward as a coverage result, on its way to a variant-set decision.

**A hardcoded workload list in a script is a scope decision in disguise.** If your harness
carries a `WORKLOADS` default it will be inherited by every later run without being
re-examined — that is how the sample above became four consecutive runs. Derive the list
from the enumeration at runtime, or print it *and its justification* at the top of every
run so there is a claim to check.

**Classify every graph in the suites relevant to your op into exactly three buckets**, and
put the table in your step-9 report:

1. **Served** — your engine matches and computes correctly. Real evidence, on graphs you
   did not author.
2. **Correctly declined** — outside your stated scope. Each one should map to a row of
   your step-2b rejection checklist **by name**. A decline you cannot name is a bug.
3. **Declined but shippable** — the kernel *could* build it and you simply have no variant.
   **This is the bucket that changes your variant set**, and it is invisible from inside
   your own bundles.

**Worked example — an illustration of the shape, not a description of your op.** Your
counts, reasons and shapes will all differ; what transfers is that the third bucket
existed and was invisible from inside the integration's own bundles. **That run's op was
relevant to two suites and said so — enumerate yours rather than reusing this pair; it is
the only concrete number on this page and reading it as a scope is exactly how the
under-triage above happened.** From the run this
step was written during: of 42 graphs for that op across
`microbench/rocke` and `models/llama3.1`, **26 were buildable by the kernel and 16
correctly declined** — every decline traceable to a named row of that engine's step-2b
checklist (a tile-divisibility rule, an unsupported head size, and three features it
scopes out). But **none of the 26 matched a shipped variant**: the engine baked its
sequence lengths and shipped them two orders of magnitude smaller than every real
workload. An engine that passes its own suite 22/22 and serves **zero** real model graphs
is exactly what 8a-8d cannot show you.

**Two traps this exercise surfaces, both worth checking deliberately:**

- **Producers spell optional features differently.** Where a schema carries both a modern
  field set and the deprecated one it replaced, authored test bundles and real traces
  routinely pick opposite spellings. A matcher handling only the convention its own
  bundles use passes step 8 and mis-serves every production graph. `graph_contract.md`
  §3-§4 tell you to check this at 2a; this is where you find out whether you did.
- **Audit with the SAME predicate your matcher uses.** A triage script that checks two or
  three attributes will call a graph "servable" that your `graph_match` declines on the
  fourth. Walk your whole Tier-3 decline list, or the triage lies in the optimistic
  direction — and optimistic is the direction that wastes a variant-set decision.

#### 8e.3 Performance sanity

Compare against a neighbouring engine on the same graph (`--engine a,b`), and against the
kernel's own documented claims. rocKE modules frequently record measured results in their
docstrings — a tuning knob that is proven-negative for one configuration, a bound the
kernel is known to hit. Your `score` encodes some of that as a ranking. **Check the ranking
against a measurement rather than shipping the assertion**, and report any disagreement:
a `score` that ranks backwards is a real defect that every correctness test passes.

**GATE:** your bundles have run under `--validate pytorch`; the relevant shipped workloads
are triaged into the three buckets above; and any *declined-but-shippable* rows are either
added to the variant set or listed at step 9 as scoped-out with a reason.

---
## Step 9 — Post-integration verification. THE ENGINE IS LANDED; PROVE IT LANDED RIGHT.

```
Produces:      a benchmark sweep over the SHIPPED package, an integration-suite run,
               a decline reconciliation against rocKE, and a triage of everything flagged
Gate:          every arm serves its expected count; ZERO declines rocKE does not share;
               zero correctness failures; zero causes recorded as "unexplained"
Typical time:  a GPU session, plus an hour of triage
```

Step 8 tested the engine. **This step tests the INTEGRATION** — the package that actually
ships, on corpora real callers send, after everything landed. They are not the same
question, and the gap between them is where every expensive defect in this skill's
history has lived:

- 8a-8d run bundles YOU wrote against hipDNN's own reference.
- 8e is exploratory and runs BEFORE the set is final.
- **Step 9 runs the SHIPPED package over someone else's corpus, and asks what the
  benchmark flags that you did not.**

An integration has been "complete, green, and serving zero real workloads" three times
in this project's history. Every one was caught by leaving the integration and counting
against an external corpus — never from inside it.

**The sweep answers three separate questions, and you owe all three.** They are one run
but not one result, and conflating them is how a set ships that is fast on the shapes it
covers and covers almost nothing:

| | question | evidence | fails when |
|---|---|---|---|
| **coverage** | which graphs do we serve, and is every decline defensible? | served/declined counts, reconciled against rocKE (9b-0) | rocKE serves something we decline |
| **correctness** | are the ones we serve *right*? | `--validate` against an independent reference | any mismatch, or an unwritten output |
| **performance** | how does the shipped package actually land? | timings, split by corpus provenance | you cannot say which population a number describes |

**Two corpora, both required, measured separately.** `dnn-benchmarking`'s graphs are what
real callers send; the kernel team's own `library/benchmarks/**` sweep and its published
results are what they judge themselves against. Neither is sufficient: the first tells you
whether you serve production traffic, the second whether you are competitive on the shapes
the owners care about. **Do not merge them into one number** — a geomean over a mixed
corpus reports one population's result as everyone's.

**`$GEN/tools/README-sweeps.md` is the how.** This step says when and what the gates
are; that page has the runnable commands, the corpus staging, the sizing costs, and the
drift-control reasoning behind the harness — written to be followable on any machine
with the target device, not a particular cluster.

### 9a. Sweep the shipped set

**Two different corpora, and they are not the same file.** This trips people because
both are "the shapes":

| | what it is | who reads it |
|---|---|---|
| `$SHAPES` | a JSON LIST of request-field mappings, from `mine_shapes.py` | the generator side — `dispatch_parity.py`, `knob_sweep.py`, `variant_reachability.py` |
| `$CORPUS_DIR/<name>/*.json` | real hipDNN GRAPH BUNDLES, with `tensors` and `nodes` | the benchmark side — `sweep.sh`, and `dnn-benchmark` itself |

`sweep.sh` never reads `$SHAPES`. It stages `$CORPUS_DIR/<name>/*.json` and inspects
their tensors, so it needs the graph files themselves — the ones you dvc-pulled at 8e,
staged one directory per corpus name:

```bash
cd $REPO
# The graph bundles the benchmark runs. One directory per name in the config's CORPORA.
mkdir -p $CORPUS_DIR/published $CORPUS_DIR/servable
cp <dvc-pulled published graphs>/*.json $CORPUS_DIR/published/
cp <your servable-graph corpus>/*.json  $CORPUS_DIR/servable/

SWEEP_CONFIG=<your sweep config> $GEN/tools/sweep.sh
```

You produced `$SHAPES` back at 4a-0 and it is still the right input for the generator
tools — it is simply not what the benchmark consumes. If your `EXPECT_GRAPHS_<name>`
gate fails immediately with a count of zero, this is why: the directory is empty because
nothing staged the graphs into it.

One arm is enough if you are not comparing variant sets: the point here is not a ratio,
it is **what the benchmark says about graphs you did not write**. Read the harness header
before you touch the config — the fixed arm order, the discarded warmup and the round
count all exist because clocks usually cannot be pinned on a shared machine.

**GATE:** the run reaches `SWEEP_DONE`, every phase reports its expected
`ingestor-served` count, and the phase inventory lists no `MISSING`.

### 9b-0. Reconcile EVERY decline against the reference. This is the gate.

**The rule: if rocKE serves an equivalent request and its result validates, this
integration must serve it too.** A decline rocKE does not share is a defect — missing
coverage, or applicability logic that is wrong. It is never a scope decision.

```bash
cd $REPO
$GEN/.venv/bin/python $GEN/tools/reconcile_applicability.py \
    --profile  $GEN/configs/$SLUG.profile.yaml \
    --shapes   $SHAPES
```

**That offline form is the default and it is the one you run.** It compares the
dispatcher-resolved parity set against the reference, needs no GPU, and answers the
applicability question completely. `--declines` exists for the case where you have
already harvested per-graph runtime reasons by hand and want the ACTUAL runtime
behaviour to override the offline answer — nothing converts a 9a sweep log into that
format, because the sweep records served/not-served per graph and not the reason.
If you pass it, key it on graph names where your corpus carries them: index keys shift
the moment the corpus is re-mined with different flags, and the same file then marks a
different shape. A key that matches nothing is a hard failure rather than a silent
no-op, but only because that trap was hit first.

**GATE:** zero `ONLY THE REFERENCE` rows. `--allow-unreconciled` is not a fix — it
records that you shipped anyway and owes a written justification per row at step 10.

| outcome | meaning | what you owe |
|---|---|---|
| both serve | fine | nothing |
| both decline | your decline is defensible | record rocKE's reason — independent evidence, better than your own matcher's |
| **only rocKE serves** | **FAIL** | a variant, a matcher fix, or proof rocKE is wrong |

**Scope to the kernel you are porting — not the whole library.** rocKE registers several
candidates per operation, and they are different kernels with different capabilities. A
shape only a SIBLING candidate serves is not your coverage gap: it is a different
kernel's job, and integrating it is separate work with its own variant set.

Getting this wrong invents work that looks real. Comparing library-wide against the
published corpus for this op reported 51 decode and large-head-size shapes as gaps in a
dense integration — when the dense kernel declines every one of them for exactly the
reasons hipDNN does, and the shapes were being served by tiled siblings. A false alarm
with a plausible story attached is the expensive kind.

The profile's `reference_candidates` block does the scoping, and two details of it are
easy to get wrong:

- **Match on the attribute that actually discriminates.** Every rocKE attention
  candidate shares `family: attention_unified`, so matching on `family` selects
  everything or nothing. `algorithm` is what separates them. Print the registry for your
  own op and look, rather than assuming.
- **An opt-in kernel needs its selector set.** Where a candidate only matches when the
  request names it, the oracle must pass that selector — otherwise it declines every
  shape and every decline reconciles trivially, a gate that passes by asking nothing.

**The fourth case, and it is real.** rocKE may ACCEPT a request it then computes
incorrectly. That is a rocKE defect and a finding to report — with the failing shape and
the evidence — **not** licence to decline quietly. The reconciler cannot detect it: it
asks about applicability, never numerics, so correctness comes from 9a's `--validate`
pass. The third legitimate response to an unreconciled decline is therefore *"the
reference is wrong, and here is the incorrect result it produces"*. "We chose not to" is
not on the list.

### 9b. Triage EVERY graph the benchmark flags. This is the actual work.

A sweep that completes is not a sweep that passed. Go through the output and put every
non-served, non-passing graph into exactly one bucket:

| bucket | meaning | what you owe |
|---|---|---|
| **served, correct** | the engine ran it and matched | nothing |
| **declined, and rocKE declines too** | your decline is defensible | rocKE's reason, from 9b-0 — not just your matcher's |
| **declined, but rocKE serves it** | **FAIL** — missing coverage or a matcher bug | a variant, a matcher fix, or proof rocKE computes it wrongly |
| **served, WRONG** | it ran and the numbers disagree | **stop. This is a wrong answer.** |
| **unexplained** | you cannot say which of the above it is | **not done. Keep going.** |

**The fourth row is why this stage exists.** A wrong answer does not announce itself: the
kernel runs, returns, and computes something else. Two shipped that way — a persistent
grid launched on the default grid shape leaving output rows unwritten, and a windowed
causal graph served as plain causal. Neither failed a test; both were wrong.

**The fifth row is the one people skip.** An unexplained skip is a finding, not noise.
Every gate in this runbook can be green while the engine serves nothing, so "the suite
was green" is not a bucket.

Two specific things the benchmark can flag that YOUR tests structurally cannot:

- **A graph shape you never imagined.** Your bundles cover shapes you thought of. The
  corpus does not care what you thought of.
- **A disagreement with an INDEPENDENT reference.** Your matcher and hipDNN's in-tree
  reference can share a misunderstanding and cancel it out silently. PyTorch cannot.

When a correctness row fails, **turn the log on before bisecting**:

```bash
HIPDNN_LOG_LEVEL=info HIPDNN_LOG_FILE=/tmp/$SLUG-fail.log <the failing command>
```

Chasing one such failure through three cluster jobs before enabling the log, when the
log named the conclusion in one line, is a real cost already paid. And read
`allClose=false` with **zero finite mismatches** as *"an output element was never
written"*, never as a tolerance problem — outputs are NaN-sentinel-filled precisely so an
unwritten element cannot pass, and the diff report prints `Mismatched: 0` while failing.

### 9c. Run the integration-test project

Less likely to surface something 9b missed, but it is the suite CI drives and it
exercises the registered target rather than a hand-rolled command line.

```bash
cmake --install $BUILD --prefix $INSTALL
cd $INSTALL && ctest -R "$PROJECT-.*-external-integration-check" -V
```

or directly, which is the invocation the registered target actually makes:

```bash
$INSTALL/bin/hipdnn_integration_tests \
    --test-article  $INSTALL/lib/hipdnn_plugins/engines/lib<your_provider>.so \
    --test-engine   "$ENGINE" \
    --reference-executor gpu \
    --fail-on-unsupported
```

`--fail-on-unsupported` turns "no engine supports this graph" from SKIP into FAIL. It is
all-or-nothing, so it only works once your filter already excludes the graphs you
legitimately decline — which is exactly the triage 9b just made you do.

Two flags worth knowing, both underused:

- `--generate-support-matrix <file>` writes what the engine claims to support. Diff it
  against your step-3 capability table; a disagreement is a matcher bug or a stale table.
- `--enforce-support-claims` checks the engine against `.support.json` sidecars, so a
  claim and the behaviour cannot drift apart silently.

**GATE:** the registered target runs from the INSTALL tree (not just the build tree) and
reports a fraction, per `SKILL.md`'s output contract: *"N of M cases dispatched this
engine; K skipped, each because <reason>; 0 unexplained."* If you cannot produce that
sentence with real numbers, this step is not finished — the tests have merely been run.

### 9d. What to do with what you found

- A **wrong answer** blocks the integration. Do not report it as a follow-up.
- A **declined-but-servable** shape is a defect until proven otherwise, and the proof is
  one call to the kernel's own `supports_*` predicate: either it returns False and you
  record the reason, or it returns True and you add the variant.
- A **surface with no test** found here goes into the launch-surface table (see
  `$GEN/tools/launch_surface.py --report`) with a test attached, not into a comment.

---

## Step 10 — Report

```
Produces:      a completion report against all ten stages
Gate:          ten numbered stage sections in the report (command below)
Typical time:  30 minutes
```

Against all ten stages, by number. Name the stage you reached; if it is not 9, say which
stage stopped you and what would unblock it. Then, per `SKILL.md` § Output contract: what
was proven and what was not, every hook's state, which splice points applied, the tests
you added by tier and path, whether the validator actually ran, and the judgment calls you
are handing back — each with a recommendation.

Step 9 adds three things this report owes that nothing earlier can supply. The **triage
buckets** from 9b, with a named cause for every flagged graph and an explicit zero for
"unexplained". The **launch-surface table** with each surface's guard and test, or the
word UNGUARDED where there is none — an unguarded surface is a legitimate thing to ship;
an unguarded surface nobody wrote down is not. And the **9b-0 reconciliation result**:
zero `ONLY THE REFERENCE` rows, or a written justification for each one that remains.
A decline the reference does not share is missing coverage or a matcher bug, never a
scope decision, so "we chose not to serve it" does not discharge a row.

**GATE:**

```bash
test "$(grep -c '^### [0-9]' <report>)" -eq 10 && echo "PASS" || echo "FAIL"
```

`grep -c` alone will not do it: its exit status means "matched at least once", not
"matched exactly this many times", so a report with three sections exits 0 just like a
complete one. The `test -eq` wrapper is the part that can actually fail.

Be precise about the ladder. A green validator proves parse, cross-reference, symbol
resolution and construction. `hipdnn_list_engines` adds "the pack registered." **Neither
says anything about matching.** Only a real graph on the target arch does.
