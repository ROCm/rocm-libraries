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

### Step 1 — Ask for the kernel sources first

Before asking anything else, ask the user to point at the kernel source(s): one or more
`.cpp`/`.hip`/`.h` files, or a directory containing them. Do not ask about engine name,
arch, knobs, or anything else yet — those come later, in one batch, after you've read
the source and inferred what you can.

If the user has already pasted or referenced the source in their request, skip the
prompt and go straight to Step 2.

### Step 2 — Infer aggressively from the source

Read every source file the user pointed at. Derive, in order:

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

Assemble the YAML config from Steps 2–3 (engine name, KMD fields, knobs, packs with
their kernels and `arch`, `heuristic` choice, `graph_match`/UMD decisions, workspace
policy, `kernel_source_kind: embedded_source`). Pick an output directory (a scratch
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

### Step 5 — Run the validator

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

State all five explicitly in the completion report even if you also perform the edits
yourself — the report is what lets a reviewer confirm nothing was missed, especially
point 4's two-edit requirement, which is the one silent-failure mode in this list.
