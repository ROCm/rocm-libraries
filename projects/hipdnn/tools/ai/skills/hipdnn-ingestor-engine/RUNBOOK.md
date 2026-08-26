# Runbook — a rocKE kernel into hipDNN, end to end

**Read this file first and drive from it.** The other three files are reference material
you are sent to at specific steps. This one is the sequence, and it is written to be
executed rather than interpreted.

Every command is copy-paste with the variables below substituted. Every step ends in a
**GATE** — a command whose output you check before continuing. A step whose gate you have
not run is a step you have not finished.

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

| Step | Produces | Gate |
|---|---|---|
| 0 | Dialect, one line | rocKE ⇒ `packaged`, no alternative |
| 1 | Feasibility verdict | Reference **and** hardware both reachable |
| 2 | `mining.md` | File exists, every row has a verdict |
| 3 | Batch message | Sent, after exhausting sources |
| 4 | `config.yaml`, descriptors | `generate.py` exit 0 |
| 5 | Packed + validated tree | `success: true`, 0 ERROR |
| 6 | Native pack | `grep -c "FILL THIS OUT"` = 0 |
| 7 | Built, packed, staged | Engine id in `hipdnn_list_engines` |
| 8 | Tests, on device | A real graph dispatched and matched a reference |
| 9 | Report | All nine stages named |

**Commit after every step.** What you committed is the deliverable if you stop.

---

## Step 0 — Dialect

Is the kernel a rocKE builder? Then it is **`packaged`**, `kind: rocke`, and there is no
alternative — the runtime has no rocKE adapter. Say so in one line and move.

The one fatal mistake here: splicing a packaged bundle into `HIPDNN_DESCRIPTOR_FILES`.
That installs an unlowered `kind: rocke` copy the loader rejects, silently dropping your
engine. Points 1 and 2 of the CMake splice **never** apply to you.

---

## Step 1 — Feasibility, before you mine anything

Three independent things must be true. Check all three now; each one costs a minute and
each one, discovered at step 8 instead, wastes the whole run.

### 1a. The builder is packable

```bash
cd $GEN && python3 -c "
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
grep -n "if spec\.\(paged\|varlen\|ragged\)" $PROVIDER/rocke/library/$MODULE | head
```

An `if` around the feature means a dense alternative exists and you can ship a dense-only
variant set. No branch — the feature is unconditional — means there is no dense subset,
and stage 8 has nothing to compare against. Say so **now**.

### 1c. The target hardware is reachable by you

Packs arch-prune before the matcher runs, so stage 8 must run on `$ARCH` and no other GPU
will do. A partition visible in `sinfo` can still reject you.

```bash
LOGIN=<slurm-submit-host>
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

---

## Step 2 — Mine the kernel

**Read `rocke-mining.md` now.** It tells you what to extract and how to classify it. This
step's output is `mining.md`.

The budget, because this is where runs die: **draft `mining.md` after the kernel module
and its spec, before ANY third source.** Then five sources maximum. Rows you are unsure of
go in marked `UNSURE` and become step-3 questions — that is what step 3 is for.

Discovery commands for the five deliverables:

```bash
M=$PROVIDER/rocke/library/$MODULE

# Constraint table: __post_init__ is the densest source
grep -n "raise ValueError" $M

# Layout: the address arithmetic. Look for stride_*_tok and the *_base terms.
grep -nE "stride_[a-z]+_tok|_base = |b\.global_(load|store)|buffer_rsrc" $M | head -30

# Grid / block / ABI
grep -nA12 "^def .*_grid\|^def .*_block\|^def .*_signature" $M

# ABI slots, and whether each is conditional: read the b.param declarations IN ORDER
grep -nB2 "b\.param(" $M

# Launch-time guards that are NOT in the spec
grep -nA3 "raise ValueError" $M | sed -n '/def run_/,$p' | head -40
```

**GATE:** `ls mining.md` succeeds and every row of the constraint table has a verdict.

Two things the Python cannot tell you, and both are required:

**The graph-side audit.** Open your op's table and account for *every* optional field —
implemented, or explicitly rejected. An unchecked mode is accepted and then silently not
performed.

```bash
sed -n "/^table $OPTABLE/,/^}/p" $REPO/projects/hipdnn/flatbuffers_sdk/schemas/*.fbs
```

**The spelling check.** For each field your spec pins, confirm hipDNN spells it the same
way. Where it does not, the rule is a *derivation*, not a comparison:

```bash
grep -rn "<spec_field_name>" $REPO/projects/hipdnn/flatbuffers_sdk/schemas/ || \
  echo "NOT A GRAPH FIELD -- this is a derivation, see rocke-mining.md"
```

---

## Step 3 — One batch message to the human

**Exhaust the sources first.** This batch is for decisions only a human can make: scope,
naming, what they intend to run. Anything the tree answers, answer yourself — see
`prompt.md` § Step 3 for the two categories that look like judgment calls and are not.

Bring a proposal, not a question. Present: engine name, arch, the variant set with counts,
which knobs are exposed and which values ship AOT, workspace policy, the layout you
derived with its arithmetic, and the rejection checklist. If they do not answer, ship the
proposal and mark it an assumption.

Sizing the variant set is in `prompt.md` § *Sizing the variant set*. The two questions it
answers: does every capability you advertise have a variant behind it, and does any knob
have more than one value (a one-value knob makes `score` a no-op — legitimate, but say so).

---

## Step 4 — Generate

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
import json; d=json.load(open('/tmp/$SLUG/descriptors/rocKE/$SLUG/'+'$SLUG'.split('_',1)[1]+'.kdp.json'))
print('kernels:', len(d['kernelDescriptors']))"
```

**GATE:** exit 0, and the kernel count equals the variant set you agreed in step 3.
Exit 1 is a `ConfigError` — read it; usually a mistyped field, a knob on a non-int field,
or a kernel `arch` outside its pack's `arch`.

Then place the authored descriptors under the packager's source root at your
`authored_subpath`, and confirm the build points at that root:

```bash
cp -r /tmp/$SLUG/descriptors/rocKE/$SLUG \
      $PROVIDER/descriptor-packaging/examples/descriptors/rocKE/
grep PRODUCTION_SOURCE_ROOT $BUILD/CMakeCache.txt
```

An empty `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` means production packaging is dormant
and your descriptors are never packed at all.

---

## Step 5 — Validate, three rungs

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

**If `hipdnn_validate_descriptors` does not exist**, it is because the build was not
configured with `-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON` (default OFF). Say so by flag name
and state that structural validation did not run. Never report a bundle as validated when
the binary was never invoked.

```bash
grep HIPDNN_ENABLE_KERNEL_INGESTOR $BUILD/CMakeCache.txt
```

---

## Step 6 — Implement the native pack. THIS IS THE WORK.

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

**GATE:**

```bash
grep -c "FILL THIS OUT" $PROVIDER/src/engines/kernel_ingestor_engine/packs/*Native.cpp
```

Must be `0`. A `// TODO` in a path the engine reaches is an unfinished integration, not a
placeholder. Placeholders are allowed — a `score` that ranks one knob, a `workspaceBytes`
that returns 0 because the kernel needs no scratch — *if you say so and say what would
replace it*. Silence is not.

---

## Step 7 — Splice, build, pack, confirm

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

**Check the struct before writing the row**; the fragment may not match it:

```bash
sed -n '/struct IngestorPack/,/};/p' \
    $PROVIDER/src/engines/kernel_ingestor_engine/IngestorPacks.hpp
```

### 7b. Build and pack

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

Two deliverables, both required. `prompt.md` § Step 8 has the matrix reasoning; this is
the mechanics.

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

### 8c. Run it, on `$ARCH`

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

Against all nine stages, by number. Name the stage you reached; if it is not 8, say which
stage stopped you and what would unblock it. Then, per `SKILL.md` § Output contract: what
was proven and what was not, every hook's state, which splice points applied, the tests
you added by tier and path, whether the validator actually ran, and the judgment calls you
are handing back — each with a recommendation.

Be precise about the ladder. A green validator proves parse, cross-reference, symbol
resolution and construction. `hipdnn_list_engines` adds "the pack registered." **Neither
says anything about matching.** Only a real graph on the target arch does.
