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

MODULE=kernels/<arch>/<module>.py                # e.g. kernels/gfx950/attention_dense.py
BUILDER=build_<op>                               # e.g. build_attention_dense
ARCH=<gfxNNN>                                    # the ONE arch this engine ships for
ENGINE=hipkernel:<CamelName>                     # scoped name; unscoped is rejected
SLUG=<arch>_<op>                                 # e.g. gfx950_attention_dense
OPTABLE=<Op>Attributes                           # the .fbs table, e.g. SdpaAttributes
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

| Step | Produces | Gate |
|---|---|---|
| 0 | Environment ready, dialect stated | `.venv` present; rocKE ⇒ `packaged` |
| 1 | Feasibility verdict | Reference **and** hardware both reachable |
| 2a | `graph_contract.md` | File exists; op matched; disagreement table has rows |
| 2b | `mining.md` | File exists, every constraint row has a verdict |
| 3 | Batch message | Sent, after the source budget is spent |
| 4 | `config.yaml`, descriptors | `generate.py` exit 0, kernel count = agreed set |
| 5 | Packed + validated tree | `success: true`, 0 ERROR, desk check clean |
| 6 | Native pack | `grep -c "FILL THIS OUT"` = 0 |
| 7 | Built, packed, staged | Engine id in `hipdnn_list_engines` |
| 8 | Tests + an engine-pinned CI target | A real graph dispatched and matched a reference on `$ARCH` |
| 9 | Report | All nine stages named, by number |

The nine **stages** of the completion contract in `SKILL.md` are unchanged; 2a and 2b are
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
Gate:          1a signature_error empty; 1b a dense path exists; 1c --test-only accepted
Typical time:  15 minutes
```

Three independent things must be true. Check all three now; each one costs a minute and
each one, discovered at step 8 instead, wastes the whole run.

### 1a. The builder is packable

```bash
# PYTHONPATH inline, not inherited. The export in the preamble is a separate fenced
# block; a fresh shell -- or a fresh agent turn -- does not have it, and the failure is
# `ModuleNotFoundError: No module named 'kernels'` raised from inside the introspector,
# which names the missing dependency but not the reason you are missing it.
cd $GEN && PYTHONPATH=$GEN:$PROVIDER/descriptor-packaging/python:$PROVIDER/rocke/library:$PROVIDER/rocke/platform/python \
python3 -c "
from codegen.sources import introspect
i = introspect('$MODULE', '$BUILDER')
print('signature_error:', repr(i.signature_error))
print('spec_class:', i.spec_class)
print('required:', [f.name for f in i.required_fields])
print('arches:', i.supported_arches)
for f in i.fields: print(' ', f.name, '|', f.type_name, '| default=', repr(getattr(f,'default',None)))
"
```

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

### 1c. The target hardware is reachable by you

Packs arch-prune before the matcher runs, so stage 8 must run on `$ARCH` and no other GPU
will do. A partition visible in `sinfo` can still reject you.

**`skill://alola-gpu-test` owns the submit host, partition, account and gres names** —
read it rather than guessing, and rather than inventing a hostname. Step 8d sends you
there to *run*; you need it here, at step 1, to find out whether running is possible at
all.

```bash
LOGIN=<submit host from skill://alola-gpu-test>
ssh $LOGIN "sinfo -N -h -o '%P|%N|%t|%G' | grep $ARCH"        # live, or drained?
ssh $LOGIN "squeue -h -o '%b' | grep -c $ARCH"                 # queue depth
ssh $LOGIN "sbatch --test-only -p <part> -A <acct> \
    --gres=gpu:<gres-type>:1 --time=00:20:00 --wrap=hostname"  # may I, and when?
```

**GATE:** `--test-only` returns an estimated start time you can live with. It reports
access failures without consuming a submission; `invalid partition specified` means your
account has no association with that partition, however it looks in `sinfo`.

A scarce single-GPU arch behind a deep queue turns a 9-stage run into a 7b run for
reasons unrelated to your integration. Knowing on day one lets you stage artifacts to a
less contended site while everything else proceeds. If the arch is unreachable, get a
decision now — and report the run as the stage it reached, never as stage 8.

**If you go off-site, create the log directory on THAT site's filesystem first.** `/home`
is a different filesystem per site, so a job whose `--output` points at your usual
`$HOME/...` path fails the instant it is scheduled on a node elsewhere: SLURM cannot open
the file, and the job dies before your payload runs. The tell is subtle, because
`squeue` keeps showing `PENDING` (it is requeued) while `sacct --duplicates` carries a
`FAILED` row with `Start=None` and `End == Submit`:

```bash
sacct -j <jobid> --duplicates -o JobID,State,ExitCode,Start,End
```

`State=FAILED, ExitCode=1:0, Start=None` means it never ran — a launch failure, not a
payload failure, and no log exists to explain it. Fix by creating the directory on the
target site's root, which the submit host mounts:

```bash
ssh $LOGIN "mkdir -p /home_aus/<user>/.alola-gpu-tests/<job-dir>"   # or /home_sc
```

Check `sacct --duplicates` — not `squeue` — whenever an off-site job seems to queue
forever. Plain `sacct` shows only the live requeued row and hides the failure entirely.

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
- **What a real graph of this op looks like**, read from an actual bundle.
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
Typical time:  30 minutes to prepare, then you wait
```

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
which knobs are exposed and which values ship AOT, workspace policy, the layout you
derived with its arithmetic, and the rejection checklist. If they do not answer, ship the
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

The set must deliver two different things:

1. **Feature coverage** — one variant per combination of *supported capability* a graph
   can ask for. A capability with no variant behind it is one the engine advertises and
   cannot serve. That is the applicability defect, arriving as wrong numbers.
2. **Performance headroom** — several variants along the *tuning* knobs for the same
   capability, so the heuristic and the autotuner have something to choose between.

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

The dispatcher is also where *policy* lives. If it does not auto-select your kernel — some
candidates are opt-in only, matching solely when a request names them — say so in the
step-9 report: you are exposing something rocKE itself does not route to by default.

**Budget: keep the total under ~100 kernels for a first integration.** Every variant is a
separately compiled code object, costing build time, archive size and install footprint.
Cover each capability once, then spend what remains on tuning variants for the
configurations that matter. Say what you pruned and why.

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
question about what the variant set should be.

---

## Step 4 — Generate

```
Produces:      configs/$SLUG.yaml and a generated descriptor tree
Gate:          generate.py exit 0, and kernel count == the set agreed at step 3
Typical time:  45 minutes
```

Write `$GEN/configs/$SLUG.yaml`. Start from an existing packaged config as a shape
reference:

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

**GATE:** exit 0, at least one `.kdp.json` printed, and the kernel count equals the
variant set you agreed in step 3. Exit 1 is a `ConfigError` — read it; usually a mistyped
field, a knob on a non-int field, or a kernel `arch` outside its pack's `arch`.

The glob is deliberate: **do not construct the KDP filename by hand.** The generator names
it from `IngestorGenerator/codegen/models.py:374` (`kdp_stem`) — the bare engine slug for a
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

On a typical configured build `$SRC_ROOT` resolves to
`$PROVIDER/descriptor-packaging/examples/descriptors` — that tree is **both** the
packager's production source root **and** a pinned test fixture, and nothing warns you.
`tests/test_hkp_pack_layout.py:562` asserts its directory set **exactly**:

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

---

## Step 6 — Implement the native pack. THIS IS THE WORK.

```
Produces:      packs/<Name>Native.cpp with all five hooks implemented
Gate:          grep -c "FILL THIS OUT" == 0
Typical time:  half a day. The largest step, and the point of the whole run.
```

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
# 1. No unfinished hook.
grep -c "FILL THIS OUT" $PROVIDER/src/engines/kernel_ingestor_engine/packs/*Native.cpp

# 2. No graph attribute silently ignored. -h, and ONE schema -- see native-pack.md.
FBS=$REPO/projects/hipdnn/flatbuffers_sdk/schemas/<op>_attributes.fbs
grep -hoP '^\s+\K[a-z_]+(?=:)' "$FBS" | while read f; do
    grep -q "$f(" $PROVIDER/src/engines/kernel_ingestor_engine/packs/<Name>Native.cpp \
        || echo "UNCHECKED: $f"
done
```

The first must be `0`. A `// TODO` in a path the engine reaches is an unfinished
integration, not a placeholder. Placeholders are allowed — a `score` that ranks one knob,
a `workspaceBytes` that returns 0 because the kernel needs no scratch — *if you say so and
say what would replace it*. Silence is not.

The second must print nothing you have not deliberately accounted for. An empty
`grep -c` and a wall of `UNCHECKED:` lines is the exact state that ships a matcher which
ignores an attribute the graph can set — the shipped `AttentionDenseNative.cpp` still
prints `UNCHECKED: implementation` today. `native-pack.md` § 6 has the reasoning and what
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
| 3 | `.../kernel_ingestor_engine/CMakeLists.txt` `target_sources` | yes |
| 4a | `IngestorPacks.hpp` — the declaration | yes |
| 4b | `IngestorPacks.cpp` — the `s_packs` row | yes |
| 5 | `.../src/tests/engines/kernel_ingestor_engine/CMakeLists.txt` | yes |

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
| `HIPDNN_ENABLE_<OP>` | **OFF** | The **frontend** for an op that has one. With it off the graph API for that op is `#ifdef`-compiled out, so the graph cannot be expressed at all and every plan silently DECLINEs. Must be ON in **both** the hipDNN SDK at `HIPDNN_ROOT` and the provider. Check whether your op has such a flag: `grep -rhoE "HIPDNN_ENABLE_[A-Z_0-9]+" --include=CMakeLists.txt --include=*.cmake $REPO/projects/hipdnn` — most ops have none and are always compiled in. |
| `ENABLE_<X>_ENGINE` | often **ON** | A **competing** hand-written engine for the same op. Nothing to do with the frontend, despite the similar name. Yours must beat it, or be pinned past it (8c). |

A missing frontend flag wastes a whole build: the provider compiles, your engine
enumerates, and every graph declines with nothing pointing at the flag.

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

**GATE:** that id appears. `hipdnn_list_engines` prints a **hash**, not your name — it
only spells out engines in its own interning table, so `grep $ENGINE` finds nothing even
on success. Compute the FNV-1a and grep for that.

Missing engine, in the order worth checking: (1) the pack step never ran, (2) splice point
4, (3) a symbol mismatch — isolate with
`hipdnn_validate_descriptors --native-source <pack.cpp>`.

---

## Step 8 — Test, on the target arch

```
Produces:      bundles in both tiers, C++ negatives, a pinned CI target, a device run
Gate:          a real graph dispatched and matched a reference on $ARCH
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

Read `integration-tests/cmake/HipdnnIntegrationTestHelpers.cmake` for the full option
list (`TEST_CONFIG` for per-case tolerance, `ENVIRONMENT`, `TEST_CATEGORIES_YAML`), and
copy the shape of an existing block — `ENGINE_NAME ASM_SDPA_ENGINE` in that same file is
the worked example.

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
   The harness matches against the loaded engines' `engineName`
   (`integration-tests/src/main.cpp`); no match means it exits with
   `Error: Engine '<name>' is not loaded` before running a single case.

**Point 4 will fail today, and that is expected. Write the block anyway.**

Engine-name exposure is **in progress**. Until it lands, a descriptor-backed engine
registers and dispatches perfectly while still presenting to the harness as a bare hex id,
so a pinned target built today fails at runtime with exactly that error. **This is not a
defect in your integration and not a reason to omit the target.**

Why it happens, so you recognise it rather than debug it: `getEngineInfos`
(`projects/hipdnn/backend/src/plugin/EnginePluginResourceManager.cpp`) fills `engineName`
from `getEngineNameFromId(id)`, falling back to a hex rendering when the id is not in the
process-wide interning map. The descriptor loader *does* register the name, but that
registry is process-local with hidden visibility — tracked as **AICK-1901**, and being
fixed. The tell from step 7c: your engine shows as a bare hex id rather than its scoped
name.

So:

- **Write the `add_external_integration_test_target` block now**, gated as above. Nothing
  ships before the name API lands, so registering ahead of it is correct, not premature —
  and the target goes live the moment it does, with no further work.
- **Expect `Error: Engine '<engine>' is not loaded`** when you run that target today.
  Record it in the step-9 report as *pending engine-name exposure (AICK-1901)*, not as a
  test failure and not as a blocking dependency on your provider.
- **Do not** substitute a workaround — do not drop `ENGINE_NAME`, do not pin a different
  engine, do not delete the target. An inert correct target beats a green misleading one.
- Your other stage-8 evidence stands on its own meanwhile: the bundles, the C++
  applicability negatives, and the on-device run in 8d do not depend on `--test-engine`.

### 8d. Run it, on `$ARCH`

Use `skill://alola-gpu-test`. **Do not substitute another arch** — packs arch-prune before
the matcher, so a clean run on the wrong GPU reads as success while proving nothing.

**GATE:** a real graph dispatched and matched a reference, on `$ARCH`. Zero-filled inputs
are not verification: `softmax(0)·0 = 0`, and so is the output of a kernel that never
wrote a byte.

When something fails, **suspect applicability before the kernel**: which variant was
selected, what does its `spec` pin, and does the failing graph differ on any axis? A
kernel computing the wrong answer for a problem it was never compiled for is behaving
correctly — the defect is upstream.

---

## Step 9 — Report

```
Produces:      a completion report against all nine stages
Gate:          nine numbered stage sections in the report (command below)
Typical time:  30 minutes
```

Against all nine stages, by number. Name the stage you reached; if it is not 8, say which
stage stopped you and what would unblock it. Then, per `SKILL.md` § Output contract: what
was proven and what was not, every hook's state, which splice points applied, the tests
you added by tier and path, whether the validator actually ran, and the judgment calls you
are handing back — each with a recommendation.

**GATE:**

```bash
test "$(grep -c '^### [0-9]' <report>)" -eq 9 && echo "PASS" || echo "FAIL"
```

`grep -c` alone will not do it: its exit status means "matched at least once", not
"matched exactly nine times", so a report with three sections exits 0 just like a
complete one. The `test -eq` wrapper is the part that can actually fail.

Be precise about the ladder. A green validator proves parse, cross-reference, symbol
resolution and construction. `hipdnn_list_engines` adds "the pack registered." **Neither
says anything about matching.** Only a real graph on the target arch does.
