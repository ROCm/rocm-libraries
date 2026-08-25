# hipDNN Ingestor Engine — Detailed Workflow

You are turning either a brand-new kernel (create flow) or an addition to an existing
engine (extend flow) into a hipDNN generic-kernel-ingestor descriptor bundle: a set of
KMD/UED/UMD/UHD/UDD/KDP/UKD JSON files plus the native-symbol stub and CMake fragments
that let a provider build and load it. Generation is mechanical (`IngestorGenerator`);
structural correctness is checked mechanically (`hipdnn_validate_descriptors`); the
judgment calls — what the kernel actually needs, which fields are knobs, whether a
distinction is a UMD or lives in the engine's `graph_match` — are yours.

## Operating principles

- **Infer before you ask.** The kernel source is ground truth for entry points,
  signatures, and template/`#define` parameters. Read it before asking the user
  anything; only the genuinely engine-level decisions (name, arch list, which fields
  are knobs, dispatch/workspace policy, UMD-vs-`graph_match`) get a question, and those
  are asked together, once, not one at a time.
- **The generator emits; it does not decide.** `IngestorGenerator`'s config is a
  declared YAML file — every value a human might override is a field, every value
  mechanically derivable is not. You build that YAML from what you inferred plus what
  the user confirmed, then run `generate.py` unmodified.
- **The generator emits fragments; it does not splice them.** `fragments/*.txt` are
  text to hand-apply at five fixed CMake/registration touch points (below). Never treat
  a successful `generate.py` run as "wired in" — it is not, until the splice happens.
- **Both flows terminate at the validator**, not at the generator. A bundle that
  generates cleanly but was never validated is not done.
- **Never report a green validator run that didn't happen.** If
  `hipdnn_validate_descriptors` cannot be found, say so by name and explain why
  (`HIPDNN_ENABLE_KERNEL_INGESTOR` defaults OFF) rather than silently treating
  generation as sufficient.
- **A green validator proves less than it looks like.** Parse, cross-reference, symbol
  resolution, construction — yes. Matching — no. Say this every time, and name the arch
  a follow-up GPU run must target: the engine's own `arch` list, not whatever GPU
  happens to be at hand. PR #10839's SDPA defect enumerated perfectly on gfx90a and
  failed all 27 cases on gfx942 — the packs arch-pruned before the matcher ever ran on
  gfx90a, so a "clean enumeration" on the wrong arch actively hid the defect.

---

## Create flow

### Step 0 — Settle the dialect first

**Two authored dialects exist and they are not interchangeable.** Everything after this
step differs between them, so decide before reading a single source file. Ask only if
the sources do not already answer it — they usually do.

| | `direct_load` | `packaged` |
|---|---|---|
| Kernel is | a `.cpp`/`.hip` the provider embeds | a rocKE builder, or a `.cpp` compiled at build time |
| `kernel_source.kind` | `embedded_source` | `rocke` or `hip` |
| Compiled | at plan-build time, on device | at build time, by `hkp_pack` |
| Ships as | the descriptor, verbatim | a per-arch `.kpack` archive + a descriptor rewritten to `kind: kpack` |
| Lands in | `descriptors/<slug>/` in the provider | the packager's source root, at your `authored_subpath` |
| Spliced into `HIPDNN_DESCRIPTOR_FILES` | **yes** | **no — never** |

Decide by asking one question: **is the kernel a rocKE builder?** If yes it is
`packaged` with `kind: rocke`, and there is no alternative — the runtime has no rocKE
adapter and is not getting one. A rocKE kernel reaches the loader already lowered to
`kpack`.

The most damaging mistake here is splicing a packaged bundle into
`HIPDNN_DESCRIPTOR_FILES`. That installs a second, *unlowered* copy carrying
`kind: rocke`, which the runtime loader rejects with "no implementation yet" — dropping
the matcher, then the pack, then the engine, at a log level that is off by default. The
generator's fragment says so explicitly for a packaged bundle; read it rather than
assuming the five splice points always apply.

### Step 1 — Ask for the kernel sources first

Before asking anything else, ask the user to point at the kernel source(s): one or more
`.cpp`/`.hip`/`.h` files, or a directory containing them. Do not ask about engine name,
arch, knobs, or anything else yet — those come later, in one batch, after you've read
the source and inferred what you can.

For a **rocKE** kernel there is no file to point at: ask instead for the **builder
module and function** (e.g. `kernels/gfx950/attention_dense.py` and
`build_attention_dense`). The module path is dotted through the importable `kernels`
package, *not* a path under the descriptor root — the rocKE descriptor folders ship no
sources at all, which is the format's sharpest trap.

If the user has already pasted or referenced the source in their request, skip the
prompt and go straight to Step 2.

### Step 2 — Infer aggressively from the source

**For a rocKE builder, do not read the source at all — introspect it.** The builder
carries its own answer in type annotations, so the extraction is exact rather than a
text-scan guess:

```
python3 -c "
from codegen.sources import introspect
i = introspect('kernels/gfx950/attention_dense.py', 'build_attention_dense')
print('signature:', i.signature_error or 'OK (spec, *, arch)')
print('spec class:', i.spec_class)
print('required:', [f.name for f in i.required_fields])
print('arches:', i.supported_arches)
"
```

Run it from the `IngestorGenerator` directory with the rocKE library on `PYTHONPATH`
(`<provider>/rocke/library` and `<provider>/rocke/platform/python`). What it gives you:

- **`signature_error` non-empty → STOP.** The builder does not take `(spec, *, arch)`,
  and `hkp_pack` will refuse it rather than pack it. This is not a config problem you
  can work around: the extra keyword-only parameter has nowhere to live in a descriptor
  and would be silently frozen at its default. The gfx942 `attention_dense` builder is
  exactly this case (an extra `tuning` object); PR #11237 is the fix. Report the message
  and ask the user whether to target a different arch or wait for the refactor.
- **`required_fields` are MANDATORY in the descriptor's `spec` block.** `hkp_pack`
  hydrates with `Spec(**fields)`, so a missing one is a `TypeError` at pack time —
  after the descriptor already looks complete.
- **`supported_arches`** is the predicate's own answer for your spec. Empty means
  *unknown*, never *unsupported*: rocKE declares arch support nowhere, and some
  predicate shapes cannot be called generically at all. Ask the user rather than
  inferring absence.

For a **non-rocKE** source, read every file the user pointed at. Derive, in order:

1. **Entry points and signatures.** Each kernel function (`__global__`, or whatever the
   provider's kernel-embedding convention names as an entry point) becomes one
   `entry_point` / `source_file` pair for `kernel_source: { kind: embedded_source,
   source_file: <file>, entry_point: <symbol> }`. Multiple entry points in one file are
   multiple kernels.
2. **Candidate KMD fields from what the kernel is templated or `#define`d on.** A
   C++ template parameter (`template <int BlockSize, typename T>`), a compile-time
   `#define`, or a set of `#ifdef`-gated code paths are exactly the axes a KMD
   `fields[]` entry exists to name — one field per axis, typed `int`/`float`/`string`/
   `bool`/`int_list` to match. A field only becomes a usable **knob** later if it is
   `int`-typed (see Step 3); note which candidates qualify so the batch question in
   Step 3 doesn't ask about a field that can't be one.
3. **Pack count from operation fan-out.** If the source implements one logical
   operation with several instantiations that differ only in the KMD-field values
   (e.g. one templated kernel compiled for several block sizes, or several dtype
   variants of the same op), that is **one pack** with multiple kernels. If the source
   implements genuinely distinct operations (e.g. add vs. mul vs. sub), that is
   multiple packs, one per operation — mirroring the shipped `pointwise` engine, which
   has one pack per operation and shares a KMD across all of them.

Write down what you inferred (entry points, candidate fields with types, proposed pack
grouping) so you can show it back to the user in Step 3 rather than asking them to
restate it.

### Step 3 — Confirm the remainder in one batch

Ask everything still open in a single message, not one question per turn. Present your
Step 2 inferences alongside each question so the user is confirming/correcting, not
authoring from scratch:

- **Engine name and namespace.** Must be scoped `namespace:local` (e.g.
  `hipkernel:MyEngine`) — an unscoped name is exactly the collision two authors would
  both pick, and the loader rejects it. Propose one from the operation name; ask the
  user to confirm or override.
- **Arch list.** Which `gfx<base-id>` targets this engine (and each pack) ships for.
  Lowercase, no feature suffix (`gfx942`, not `GFX942` or `gfx942:sramecc+`). Warn on
  an unrecognized-but-well-formed id (e.g. `gfx94`) rather than silently accepting it —
  it parses but matches no real device.
- **Which fields are knobs.** From the candidate KMD fields in Step 2, ask which
  should be exposed as `knobs` on the UED. Only `int`-typed fields can become a real
  knob — the loader's `getCustomKnobs` silently drops a non-int knob at plan-build time
  with no error and no warning, so screen this **before** confirming, not after: if the
  user wants a knob on a non-int field, tell them now and ask them to either retype the
  field or drop it from the knob list.
- **Dispatch and workspace policy.** Whether this engine's `IKernelDispatchHandler`
  needs a workspace at all (`none`), a fixed size, or a size derived from the bound
  tokens/kernel metadata (`derived`) — this drives the stub's `workspaceBytes` shape,
  even though the body itself is left `// TODO`.
- **UMD-or-`graph_match`.** Whether a distinction between packs is genuine per-pack
  narrowing (→ a UMD) or a property of the graph's topology/shape/dtype (→ belongs in
  the UED's `graph_match`, evaluated once for the whole engine). Get this right before
  generating: a single-pack engine should end up with **zero** graph-scoped UMDs; if
  the user's answer implies one, point out the shipped convention
  (`TestConvFwdPack.cpp` asserts a single-pack engine has none) and ask them to
  reconsider, rather than silently emitting a redundant UMD.

Wait for the user's answers before proceeding — this is the one blocking prompt in the
create flow.

### Step 4 — Build the config and run the generator

Assemble the YAML config from Steps 0–3: `dialect`, engine name, KMD fields, knobs,
packs with their kernels and `arch`, `heuristic` choice, `graph_match`/UMD decisions,
workspace policy, and `kernel_source_kind` (`embedded_source` for `direct_load`;
`rocke` or `hip` for `packaged`). A packaged config also names `authored_subpath` —
where the descriptors sit under the packager's ONE source root, preserved verbatim into
the staged and installed trees. `configs/gfx950_attention_dense.yaml` is a working
rocKE example. Pick an output directory (a scratch
location unless the user names one) and run:

```
python3 <path-to>/IngestorGenerator/generate.py \
    --config <config.yaml> --output-dir <output-dir>
```

- Exit 0 → generation succeeded; proceed to Step 5.
- Exit 1 (`ConfigError` or a render failure) → read the message, fix the config (most
  likely a field you mis-typed, a knob naming a non-int field, or a kernel `arch` not
  covered by its pack's `arch`), and retry.
- Exit 2 → an argparse usage error (a flag typo); fix the invocation.
- If the output directory already exists and is non-empty, `generate.py` exits 1
  **without** `--force`. For a create flow this should not happen (fresh output dir);
  if it does, confirm with the user before adding `--force` — don't silently overwrite
  something that might be someone else's work.

Consider a `--dry-run` pass first (prints the file list, does not create the output
directory) if you want to sanity-check the shape before writing anything.

### Step 5 — Validate

**Which validator depends on the dialect**, because the two check different artifacts.
Run the one that matches; a packaged bundle has nothing for the runtime validator to
read until it has been packed.

#### 5a. Packaged dialect — validate the authored form, then pack, then validate the shipped form

First, the authored form, through the packager's own validator:

```
PYTHONPATH=<provider>/descriptor-packaging/python:<provider>/rocke/library:<provider>/rocke/platform/python \
python3 -c "
from hkp_pack.descriptors import load_flat_input
flat = load_flat_input('<output-dir>/descriptors')
print('OK,', len(flat.descriptors), 'descriptors')
"
```

This is the same `load_flat_input` the build calls, so it enforces exactly what the
build enforces — required keys per `kind`, the `namespace:local` UED name, closed enum
vocabularies, bare-arch ids, dangling cross-references, the reserved `kpack/` folder
name. Do **not** reimplement these checks; call this.

Then actually pack, which is what proves the builder lowers:

```
python3 -c "
from hkp_pack.pipeline import run_pipeline
res = run_pipeline(source_root='<output-dir>/descriptors', arches=['gfx950'],
                   out_root='<pack-out>', hipcc='/opt/rocm/bin/hipcc',
                   rocm_kpack_dir='<rocm_kpack>/python')
print({a: (r.skipped, str(r.kpack_path)) for a, r in res.items()})
"
```

Needs `msgpack` and `zstandard` importable (the kpack reader's own deps) — a bare venv
will not have them. A skipped arch means no KDP targeted it; an exception names the
failing stage.

Finally the shipped form, through the real runtime loader:

```
<build-dir>/bin/hipdnn_validate_descriptors <pack-out>/<arch> \
    --expect-engine <engine-name> --json
```

Point it at the PACKED tree, never the authored one: the authored descriptor still says
`kind: rocke`, which the loader rejects by design. A `WARN` about the `provenance`
extension key is expected and correct — the packager writes it, the loader ignores it.

#### 5b. Direct-load dialect — the runtime validator alone

See **Detecting an absent validator** below before this step — do it first if you
haven't already located the binary this session.

```
<build-dir>/bin/hipdnn_validate_descriptors <output-dir>/descriptors \
    --native-source <output-dir>/packs/<Name>Native.cpp \
    --expect-engine <engine-name> \
    --json
```

Use `--json` and parse the structured output (shape below) rather than scraping
stdout text. Exit 0 iff zero ERROR/FATAL diagnostics, the expected engine is present,
and the native-source cross-check is clean.

- `success: true` and the engine present in `engines[]` → structural validation passed;
  proceed to Step 6.
- `success: false` → read `diagnostics[]` for every `ERROR`/`FATAL` entry and
  `expected_engines_missing[]`; fix the descriptor(s) named and re-run. Do not report a
  bundle as validated while any diagnostic is `ERROR`/`FATAL`.
- If `--native-source` was supplied, also check `native_source_checks[]`: `clean: true`
  means the stub's `constexpr std::string_view` symbol constants match every symbol
  name the descriptors reference; a non-empty `in_source_not_in_descriptors` or
  `descriptor_symbols_no_source_declares` means the two sources of truth (pack `.cpp`
  constants vs. descriptor JSON) have diverged — fix whichever side is wrong.

### Step 6 — Report

See **Output contract** in `SKILL.md` — restated here as the concrete report shape for
create flow:

1. What was generated (descriptor files, the native stub, the fragments) and where.
2. The validator invocation used, and its verdict (from the parsed `--json`, not raw
   text) — or, if the validator could not be located, say so explicitly (see below)
   and state that structural validation did not happen this run.
3. The five CMake splice points still pending, with the exact fragment file that
   covers each (see **The CMake splice** below) — never imply the bundle is wired into
   the provider just because it generated and validated.
4. The matching-correctness gap: name the arch(es) this engine targets and the GPU
   test/suite that would need to run there to prove matching, separate from what the
   validator already proved.

---

## Extend flow

### Step 1 — Locate the existing descriptor directory and the addition

Ask the user (or infer from their request) which existing engine's descriptor
directory to extend, and whether the addition is **one new pack** (new operation
variant, same engine) or **one new kernel** (new variant of an existing pack — e.g. a
new block size or arch shard).

Read the existing KMD/UED/UMD/UHD/UDD/KDP files in that directory first — the
existing `metadata`/`knobs`/`graph_match`/`heuristic` choices are the contract the
addition must fit into, not something to re-derive from scratch.

### Step 2 — Infer and confirm, same as create flow's Steps 2–3, narrowed

Read the new kernel source the same way (Step 2 of the create flow). Because this is
an addition to something that already exists, most of the create flow's batch
question is already answered by the existing files — confirm only what's new:

- For a **new pack**: pack name, its `arch` list (must be a subset check against
  nothing — packs are independent — but each of its kernels' `arch` must be a subset of
  the pack's own), and whether it needs a new UMD or fits the engine's existing
  `graph_match` decision.
- For a **new kernel** on an existing pack: the kernel's own `metadata` values (must
  type-check against the existing KMD and supply every mandatory field), its `arch`
  (must be covered by the pack's `arch`), and whether its metadata tuple collides with
  a sibling kernel's on any arch they both reach (`archOverlaps`-based uniqueness —
  arch-disjoint kernels may legally share a tuple, same-pack same-arch ones may not).

### Step 3 — Mint only the new UUIDs

Every new descriptor object (a new pack's KDP/UMD/UDD, or a new kernel's UKD if
standalone) gets a fresh UUID. **Do not regenerate or touch the `id` of anything that
already exists** — cross-references are by id, and changing an existing id breaks
every file that references it.

### Step 4 — Run the generator against the existing directory

Run `generate.py` with `--output-dir` pointed at the existing descriptor tree's parent
(or wherever the generator's own conventions expect an addition — check
`IngestorGenerator`'s own docs for the extend-mode invocation). Because this points at
a **live, hand-filled directory**, `generate.py` requires `--force` to overwrite
anything non-empty:

```
python3 <path-to>/IngestorGenerator/generate.py \
    --config <addition-config.yaml> --output-dir <existing-output-dir> --force
```

Before passing `--force`, review what the generator's `--dry-run` file list would
touch — this is the deliberate deviation from `DescriptorGenerator`'s unconditional
overwrite, and it exists specifically so this step doesn't silently clobber a
hand-filled `graph_match` body or matcher implementation that already lives in this
directory.

### Step 5 — Append to the existing CMake lists — never rewrite them

The generator emits fragments for the new pack/kernel only. When splicing (see **The
CMake splice** below for the five points), **append** the new lines to each existing
list — `HIPDNN_DESCRIPTOR_FILES`, `HIPDNN_INGESTOR_PACK_KERNELS`, the `target_sources`
blocks, and the `IngestorPacks.cpp` table — rather than regenerating or replacing the
whole list. Rewriting risks silently dropping an existing entry that isn't in the new
fragment (it was never meant to be — the fragment only covers what's new).

### Step 6 — Re-run the validator over the WHOLE directory

This is the step that makes an extend flow trustworthy, not just the new files:

```
<build-dir>/bin/hipdnn_validate_descriptors <existing-output-dir>/descriptors \
    --native-source <path-to-existing-pack-native.cpp> \
    --native-source <path-to-new-pack-native.cpp-if-any> \
    --expect-engine <engine-name> \
    --json
```

Point every root/native-source at the **whole** descriptor tree for this engine, not
just the new pack or kernel — a partial validation (new files only) cannot detect a
cross-reference the addition broke in an existing file (e.g. an arch-overlap
uniqueness collision the new kernel introduces against an existing sibling). This
whole-directory revalidation is the actual mechanism that demonstrates the pieces that
were already there are still valid; it is not optional or a "nice to also do."

### Step 7 — Report

Same shape as the create flow's Step 6, plus: state explicitly that the validator ran
against the whole directory (not just the addition), and name which existing files
were included in that pass.

---

## Detecting an absent validator

Do this once per session, before the first validator invocation in either flow:

1. Look for an active build directory (the one the user is working from, or a
   `build*`/`out*` directory containing `CMakeCache.txt`). Check whether its
   `CMakeCache.txt` sets `HIPDNN_ENABLE_KERNEL_INGESTOR:BOOL=ON`.
2. If a build directory with the flag ON exists, look for
   `<build-dir>/bin/hipdnn_validate_descriptors` (or wherever that build's binary
   output directory is). If present, use it.
3. If no such binary is found — because no build exists, or the active build has
   `HIPDNN_ENABLE_KERNEL_INGESTOR` unset/OFF (the default) — **do not** skip validation
   silently. Tell the user plainly: "`hipdnn_validate_descriptors` was not found. It is
   only built when the consuming build configures `-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON`
   (default is OFF), so a default hipDNN build will never contain it." Offer the exact
   configure command needed if the user wants to build it, and proceed with the rest
   of the flow (generation, fragment emission, splice guidance) while stating clearly
   in the final report that **structural validation did not run this session**.

This check belongs in both flows' Step 5/6 — the generator's output is never reported
as "validated" on the strength of generation alone.

---

## The CMake splice

**Read the emitted `fragments/cmake_descriptor_files.txt` before applying anything.**
It states which of the points below apply to the bundle you just generated, and for a
packaged bundle it deliberately contains no CMake payload at all.

### Packaged dialect — points 1 and 2 DO NOT APPLY

A packaged bundle is consumed by `hkp_pack`, not installed for the runtime loader, so:

- **Point 1 (`HIPDNN_DESCRIPTOR_FILES`) — never.** Adding an authored `kind: rocke`
  descriptor there installs a second, unlowered copy beside the packed one. The loader
  rejects it ("no implementation yet"), which drops the matcher, then the pack, then
  the engine — silently, at the default log level. This is the single most damaging
  mistake available in this flow.
- **Point 2 (`HIPDNN_INGESTOR_PACK_KERNELS`) — never.** That list names embedded kernel
  source stems compiled at plan-build time. A packaged kernel is already a compiled code
  object inside the archive; a rocKE kernel's source is not even in this repository.

Instead: place the authored descriptors under the packager's source root at the
`authored_subpath` the config declares, and confirm the build points at that root.

Points 3, 4 and 5 still apply **if and only if** this engine needs its own native pack
(new match/score/dispatch symbols). A packaged bundle that reuses an existing pack's
symbols — which is what a first integration usually does — needs none of them, and then
there is nothing to splice at all. Say so plainly rather than listing five points that
do not apply.

### Direct-load dialect — all five apply

The generator emits `fragments/*.txt`; it never edits provider CMake files itself.
After generation (or after Step 5 of the extend flow), walk the user through all five
touch points, in this order, using the matching fragment file from the generator's
`fragments/` output:

1. **`dnn-providers/hip-kernel-provider/CMakeLists.txt`** — the `HIPDNN_DESCRIPTOR_FILES`
   list (inside the `if(HIPDNN_ENABLE_KERNEL_INGESTOR)` block). One line per new
   descriptor file, path relative to `.../kernel_ingestor_engine/descriptors/`. This
   single list drives staging, install, and the dependency edge — get every new file
   into it or the descriptor never ships.
2. **`.../kernel_ingestor_engine/IngestorKernels.cmake`** — the
   `HIPDNN_INGESTOR_PACK_KERNELS` list. Append the new kernel source stem(s) (no
   extension, e.g. `MyAdd` for `kernels/MyAdd.cpp`) — this list feeds both the provider
   target and the test binary's embedded-kernel copy, so a kernel missing here fails at
   plan-build time with a missing embedded source, on both sides.
3. **`.../kernel_ingestor_engine/CMakeLists.txt`** — the `target_sources(...)` block
   for `hip_kernel_provider_impl`. Add the new pack's `packs/<Name>Native.cpp` (and, for
   a brand-new engine, its `IngestorPacks.cpp`/`KernelIngestorEngine.cpp` are only added
   once, at the very first engine — a create flow for the *first* engine in a fresh
   provider needs this; extending an existing provider does not touch these two lines).
4. **`IngestorPacks.hpp` and `IngestorPacks.cpp` — BOTH, always.** Add the
   `register<Name>Symbols` declaration to `IngestorPacks.hpp`, **and** add the
   corresponding `{"<engine:name>", &register<Name>Symbols}` row to the
   `s_packs` table in `IngestorPacks.cpp`. These are two edits to two files for one
   pack — doing only the header declaration (or only the table row) is a broken build
   or a silent drop:
   - Declaration without table row: the symbol exists but nothing calls it — the pack
     is invisible to `ingestorPacks()`, and because `ingestorPacks()`'s table is the
     *only* reference to the pack's registration function, a static-archive linker
     (the unit-test binary) drops the whole translation unit as unreferenced. The
     plugin `.so` still works (a shared library keeps every exported symbol), so this
     failure mode is invisible in the `.so` and only shows up as a missing engine in
     unit tests — with no build error either way.
   - Table row without declaration: does not compile (undeclared identifier).
5. **`.../src/tests/engines/kernel_ingestor_engine/CMakeLists.txt`** — the
   `target_sources(...)` block for `hip_kernel_provider_tests`. Add the generated
   `Test<Name>Packs.cpp` (complete) and `Test<Name>Matchers.cpp` (stub) so the new
   pack's shape census and matcher tests actually build and run.

State every point that applies explicitly in the completion report even if you also
perform the edits yourself — the report is what lets a reviewer confirm nothing was
missed, especially point 4's two-edit requirement, which is the one silent-failure mode
in this list. For a packaged bundle, state just as explicitly that points 1 and 2 do
**not** apply and why, so a reviewer does not "helpfully" add them back.
