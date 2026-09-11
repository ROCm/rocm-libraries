# The extend flow — one more pack or kernel, on an engine that already ships

**You were sent here from `SKILL.md`.** Extend is one of the two top-level flows, not a
runbook step — you don't arrive here from `RUNBOOK.md`, and nothing in it sends you back.
You owe: the addition spliced into a live descriptor directory, only *new* UUIDs minted,
and a **whole-directory** validator pass proving the pieces already there stayed valid.

**Variables.** `$REPO`, `$PROVIDER`, `$GEN`, `$BUILD` are set in `RUNBOOK.md`'s "Set
these once" block. You do not otherwise use RUNBOOK in this flow, but you still need that
block — set them before running anything below. `$EXISTING_DIR` is the descriptor
directory the first command below locates.

Everything about turning a kernel into descriptors and hook bodies is unchanged from the
create flow — mine the kernel per `rocke-mining.md`, implement the five hooks per
`native-pack.md`, read `graph-contract.md` when the addition changes what the engine can
match (step 4). This file covers only what differs because something already exists.

---

## Step 1 — Locate the directory, name and confirm the addition

```bash
find $PROVIDER -maxdepth 4 -type d -iname descriptors
```

Decide, from the request and what that directory holds: **one new pack** (a new
operation variant — same KMD/UED/UHD/UDD, a new `*.kdp.json`, plus an operation-scoped
UMD if the engine is or becomes multi-pack), or **one new kernel** (a new entry inside an
existing pack's `kernelDescriptors[]`, no new KDP).

Read every file already there first — its `metadata`/`knobs`/`graph_match`/`heuristic`
choices are the contract the addition must fit into, not something to re-derive:

```bash
cat $EXISTING_DIR/*.kmd.json $EXISTING_DIR/*.ued.json $EXISTING_DIR/*.kdp.json
```

Then mine the new kernel the same way the create flow does. Most of the batch question
is already answered by what you just read; confirm only:

- **New pack**: its name, its `arch` (each kernel's `arch` a subset of the pack's own —
  packs are independent of each other), and whether it needs a new UMD or fits the
  existing `graph_match`.
- **New kernel**: its `metadata` (type-checks against the existing KMD, every mandatory
  field present), its `arch` (covered by the pack's), and whether its metadata tuple
  collides with a sibling's on any arch they both reach — `archOverlaps`-based
  uniqueness: arch-disjoint kernels may share a tuple, same-pack same-arch ones may not
  (`DeviceProperties.hpp`'s `archOverlaps`, exercised by
  `TestKernelIngestorStateManager.cpp`'s
  `RejectsATupleSharedByAnArchIndependentAndAPerArchPack`).

**A single-pack engine becomes multi-pack the instant a second pack is added**, which
flips `IngestorGenerator`'s UMD policy: zero graph-scoped UMDs for one pack, but a
`discriminator` required on *every* pack once there are two — including the original.
Verified: generating a two-pack version of a single-pack config gave the original pack a
`discriminator` and a matching `operation_is_*.umd.json` where none existed before
(`_check_pack_discriminators` in `$GEN/codegen/config_loader.py` is the check that
fires). Plan on retrofitting the original pack in this case.

---

## Step 2 — Mint only the new UUIDs

**The generator has no incremental mode.** Every run mints a fresh UUID for every
engine-level object — KMD, UED, UDD, the shared kernel-scoped matcher, every pack's own
KDP/UMD/kernel ids — whether or not that object's shape changed. Verified:
generating the same config into two directories gave the UED two different `id`s, and
force-regenerating a config with an unrelated third pack added changed the shared UED's
`id` while the untouched first two packs' on-disk KDPs kept pointing at the now-
nonexistent old one — the exact orphaned cross-reference the validator exists to catch,
on files nobody meant to touch. Pointing `--output-dir` at the live directory with
`--force` does not "add" anything; it silently re-mints ids meant to keep their identity.

The correct sequence: (1) generate **only the addition** into an **empty** scratch
directory; (2) in the fresh output, replace the scratch-minted `engine` / `dispatch` /
shared-matcher-id fields with the **real, existing** ids from the live directory's own
`*.ued.json` / `*.udd.json` / `kernel_dtype_matches_graph.umd.json` — every other id the
scratch run minted (the new KDP's `id`, its kernel ids, a new operation UMD's `id`) is
correct as generated, leave those alone; (3) copy only the new/changed files into the
live directory, never overwriting anything the scratch run re-emitted for an unrelated
pack — that's a byproduct of the engine-wide config, not this addition's output.

**Never touch the `id` of anything that already exists** — cross-references are by id,
and changing one breaks every file naming it, silently, until step 4's validator.

```bash
python3 $GEN/generate.py --config <addition.yaml> --output-dir /tmp/<slug>-addition --dry-run
python3 $GEN/generate.py --config <addition.yaml> --output-dir /tmp/<slug>-addition
```

No `--force` here — the scratch directory is empty. `--force` guards a *non-empty*
directory only, and its `--help` names why: "only safe when you intend to regenerate
over hand-filled content you have already reviewed" — precisely what the live directory
is. `--dry-run` first catches a config mistake before there's anything to clean up.

---

## Step 3 — Append to the existing CMake lists — never rewrite them

The scratch run's `fragments/*.txt` are the same six splice points the create flow uses
(`RUNBOOK.md`'s CMake splice table; `native-pack.md` points at it), scoped to the
addition. **Append** to each existing list — `HIPDNN_DESCRIPTOR_FILES`,
`HIPDNN_INGESTOR_PACK_KERNELS`, the `target_sources` blocks, the `IngestorPacks.cpp`
`s_packs` table — never regenerate or replace the whole list; a rewrite risks silently
dropping an entry this run's fragment never covered. Packaged bundles still never touch
`HIPDNN_DESCRIPTOR_FILES`/`HIPDNN_INGESTOR_PACK_KERNELS`. Retrofitting a single-pack
engine (step 1): the engine-level `IngestorPacks.hpp`/`.cpp` declaration and `s_packs`
row already exist — one `register<Name>Symbols` still covers every pack, so only the
new pack's own registration inside it is new.

---

## Step 4 — Graph contract (conditional), then re-validate the WHOLE directory

A new **kernel** never needs `graph-contract.md` revisited — it adds a metadata tuple the
existing `graph_match` already admits. A new **pack** needs it only when it serves a node
type, composition, or field combination `graph_match` doesn't already accept — the
addition changes what a graph can match, not merely which kernel answers a match that
already happens. Another tuning point for an operation the engine already serves needs no
revisit; a pack that makes the engine accept something it previously declined does,
exactly as if this were a new engine's step 2a. See `graph-contract.md` for the five
sections to produce in that case.

Then the step that makes an extend trustworthy: point every root and `--native-source` at
the **whole** tree — never just the new pack or kernel.

**Which tree: the SHIPPED one.** The validator runs the real runtime loader, which reads
`kernel_source.kind: kpack`. For a `packaged` (rocKE) engine the authored tree still
carries `kind: rocke` with its `builder`/`build` keys, and pointing the validator at it
produces a real, misleading ERROR —
`unknown key 'build' in kernel '...' kernel_source; extension keys must start with 'x-'
or '_'` — which reads like a corrupt descriptor and is actually the wrong directory. Pack
first (`RUNBOOK.md` step 5b), then validate the packed output. A `direct_load` engine has
no packed form and validates its authored tree directly.

```bash
# packaged: the PACKED tree, after step 5b. direct_load: the authored descriptors dir.
<build-dir>/bin/hipdnn_validate_descriptors <packed-tree>/<arch> \
    --native-source <existing-pack-native.cpp> \
    --expect-engine <engine-name> --json
```

(`--native-source`, `--expect-engine`, `--json` — the validator's real flags, from
`ValidateDescriptors.cpp`'s own usage string; no per-file or incremental mode exists.) No
`hipdnn_validate_descriptors` binary has three possible causes, not one — `RUNBOOK.md`
step 5c has the check that tells them apart.

A partial validation (new files only) can't catch a cross-reference the addition broke in
an existing file — exactly step 2's failure mode. Whole-directory revalidation is what
demonstrates the pieces already there are still valid; it is not optional. For a
`packaged` bundle, also re-run the desk check (`RUNBOOK.md` step 5d) over the **whole**
shipped `.kdp.json` set after packing — a metadata-tuple collision between the new kernel
and an existing sibling is a whole-directory property.

**Then re-run the SET-level gates over the widened directory.** The validator above is the
LOADS rung; it answers "does the loader accept this?" and nothing about whether the set is
still internally coherent now that you have added to it:

```bash
cd $REPO
$GEN/.venv/bin/python $GEN/tools/verify_variant_sets.py \
    extended <the widened descriptor tree> --profile $GEN/configs/$SLUG.profile.yaml
$GEN/.venv/bin/python $GEN/tools/variant_reachability.py \
    --kdp <the widened .kdp.json> --shapes $SHAPES \
    <your --field-map/--divides flags> --score-field <field> --score-prefer max
```

Every property those check is a property OF THE SET, so adding one kernel can break any of
them without touching a byte of the existing files:

- a new kernel whose metadata tuple collides with a sibling **drops the whole engine** at
  load, not just the newcomer;
- a new kernel that pins a knob its sibling leaves to policy REPLACES rather than adds,
  silently removing the sibling's kernel from the candidate list (the policy-twin check);
- a new kernel written in the builder's dtype vocabulary rather than the matcher's loads
  cleanly, reconciles on every count, and matches nothing;
- and the newcomer may be **unreachable** — applicable to nothing, or always outranked by
  the sibling it was meant to complement, which is dead weight that reports green.

If the engine already ships a `launch_surface:` block and your addition touches a surface
in it — a new branch in `prepare()`, a new KMD field the C++ reads — update that block and
re-run `$GEN/tools/launch_surface.py --check`. A restated helper needs its inputs in the
descriptor, and adding the branch without adding the field compiles fine.

Report, same shape as the create flow against `SKILL.md`'s stages, plus: which existing
files you read before writing anything (step 1), whether `graph-contract.md` needed
revisiting and why, and that the validator ran against the **whole** directory — name
which existing files were included, not only the new ones.

---

## GATE

```bash
<build-dir>/bin/hipdnn_validate_descriptors <packed-tree>/<arch> \
    --native-source <existing-pack-native.cpp> --expect-engine <engine-name> --json \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('success:', d['success'], '| errors:', [g for g in d['diagnostics'] if g['severity'] in ('ERROR','FATAL')])"
```

Done when: this prints `success: True` with an empty error list, run against the whole
directory, not a subset; every id in the new files is genuinely new (none matches
anything `grep`-able in the pre-addition directory); and every id the addition
*references* (`engine`, `dispatch`, the shared kernel-scoped matcher) equals what the
pre-existing `*.ued.json`/`*.udd.json`/`kernel_dtype_matches_graph.umd.json` already
held — untouched, not regenerated.
