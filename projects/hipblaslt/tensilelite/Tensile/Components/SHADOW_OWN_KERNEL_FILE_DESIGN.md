# SHADOW own-kernel-file design (Approach B, proper refactor)

**Status:** design investigation (read-only), 2026-06-10
**Author scope:** clean refactor design for emitting the actual on-CMS-build
SHADOW (default-schedule) run as its own full assembly file, separate from the
CMS `kernel.s`, so the SHADOW *capture* can be text-diffed against an emitted
artifact — the same faithfulness proof the CMS side already has.

Companion to `SHADOW_FULL_EMISSION_FEASIBILITY.md` (the feasibility triage that
chose Approach B / "SHADOW-tee" over Approach A). This doc is the *how-to-do-it-
properly* follow-up: it specifies the refactor that earns a faithful SHADOW
artifact without wedging output-teeing into the load-bearing SHADOW branches.

---

## §1 Goal + the three user requirements

**Goal.** Produce `shadow_kernel.s` — a full assembly text of the *actual SHADOW
run that happens inside the CMS build* — so we can text-diff
`shadow_capture_listing.txt` against `shadow_kernel.s` and prove the SHADOW
capture is byte-faithful to the instruction stream it walked. Today only the
CMS subject has this proof (`kernel.s` vs `cms_capture_listing.txt`, e.g. NGL
`ds_read kernel.s:2175 -> pack-mfma :2208 -> pack-cvt :2225` matching
`cms_capture_listing.txt:584/604/612` per `H7LO_UVRL_NORMATIVE_ORDERING.md:355-
358`).

**Requirement B — the actual on-CMS-build SHADOW run.** The artifact MUST be the
default scheduler running over the *CMS-mutated kernel dict* on the *same CMS
build* (`source="default-sia3"`, `writer._last_default_capture`). NOT a separate
`UseCustomMainLoopSchedule=0` build (Approach A). Approach A flips
`doFullPackCodePrefetch` and other gates (`KernelWriter.py:1005`,
`_makeSubIterSchedule`'s `doFullPackCodePrefetch` branch), producing a *different*
kernel/codegen than the validator's reference — so it would prove the wrong
thing. This requirement is the entire reason Approach A was rejected.

**Requirement 2 — proper refactor, not a bolt-on.** The current SHADOW path
*entangles* three concerns: (a) run the default scheduler, (b) capture it into a
`FourPartCapture`, (c) discard the emitted `Module`(s). The clean design must
*separate* "produce the default-scheduled instruction stream" from "what we do
with it" — so the stream can be routed to BOTH the capturer AND a text emitter.
The naive bolt-on (un-discard the `iterCode`/`Module` in-place and `str()` it)
re-entangles and risks the label-sharing invariant; we reject it (§5).

**Requirement 3 — separate file, CMS `kernel.s` byte-for-byte untouched.** SHADOW
text goes to its own `shadow_kernel.s`. The CMS `_getKernelSource` flow must not
change at all — SHADOW emission is purely additive (a new method / new code
path), never a modification of the CMS string-assembly flow.

---

## §2 Current SHADOW emission/capture path (full flow with citations)

### §2.1 The flag and where it gates

`self.states._captureDefaultSchedule` is the master switch for the SHADOW path.

- **Auto-set** in `kernelBody` head: `KernelWriter.py:5617-5618` — for any kernel
  with `UseCustomMainLoopSchedule`, `self.states._captureDefaultSchedule = True`.
- **Test/converter activation** via `enable_capture_default_schedule()`
  (`KernelWriter.py:478-515`), which monkey-patches `setupNewTile` to flip the
  flag after `self.states` is initialized but before `_loopBody` reads it.
- **Read sites** (each a SHADOW branch): `KernelWriter.py:4185` (noLoadLoop
  NGL/NLL), `:4980` (macIterCode deepcopy site), `:5103` (`_loopBody` main-loop),
  `:5265`/`:5327` (OptNLL guard + main-loop finalize), `:5947`/`:5961`/`:6020`
  (prologue snapshot/build), `:6177` (FourPartCapture assembly).

### §2.2 The main-loop SHADOW branch (`_loopBody`, ~`:5103-5250`)

Inside the per-subiter `for uIdx` loop, when `_captureDefaultSchedule` is set
(the `elif` at `:5103`), the path:

1. Builds a `{id(item) -> category}` map from the SAME source modules CMS will
   consume (`build_idmap`, `:5141-5156`; inversions `:5157-5163`).
2. Tags prefetch/MFMA leaves (`:5169-5225`).
3. Calls `self._makeSubIterSchedule(... capture=self._capture_context.builder,
   ..., capture_body_label="main_loop", capture_fail_loud_on_missing_category=
   True)` at `:5226-5250`. **Every mutable input is `structural_clone`d**
   (`:5228-5233`) because SIA3 mutates inputs via `popFirstItem`/`popFirstNItems`.
4. **The return value of `_makeSubIterSchedule` is NOT captured** — there is no
   `module.add(...)`. The emitted `iterCode` is discarded; only the builder
   survives. This is the load-bearing discard (§4).

`_makeSubIterSchedule` (`:983`) builds `iterCode` (`:991`), and at `:2694-2704`,
when `capture is not None`, calls `_captureSubIterToBuilder(iterCode=iterCode,
capture=capture, ...)` (`:2708`), which walks `iterCode.flatitems()` into
`TaggedInstruction`s appended to the builder. Then it `return iterCode`
(`:2706`) — and the SHADOW caller drops it on the floor.

Compare the CMS path at the same site: CMS does NOT call `_makeSubIterSchedule`
per-iter for emission. Instead it accumulates per-side `*AllIters` lists and at
`:5478` calls `customMainLoopSchedule(...)` ONCE, which returns a single
`optSchedule` Module that is `module.add(optSchedule)`-ed at `:5483`. That
single Module is what renders into `kernel.s`. **The CMS main-loop is assembled
as one Module; the SHADOW main-loop is never assembled into a Module at all —
it is captured per-iter and discarded.** This asymmetry is the entanglement (§3).

After the per-iter loop, `:5327-5476` finalizes the SHADOW main-loop builder
(`leftover` pack walk `:5371-5466`, LCC harvest via `_appendCloseLoopLCCToBuilder`
`:5472-5473`) into `self._capture_context.default_main = builder.finalize()`
(`:5475`). Note this is a `LoopBodyCapture` (the capture data structure), NOT an
emittable `Module`.

### §2.3 The NGL/NLL SHADOW branch (`noLoadLoop`, ~`:4185-4207`)

When `_captureDefaultSchedule` and not `isOptNLL` (`:4185`):

1. `structural_clone`s `pack`/`packPre` (`:4191-4194`).
2. Builds `shadow_capture = LoopBodyCaptureBuilder()` (`:4195`).
3. Calls `self._noLoadLoopBodyDefault(..., capture=shadow_capture)` (`:4196-
   4202`). **The returned `Module` is NOT added to the outer `module`** — it is
   discarded; only `shadow_capture` is kept.
4. `finalized = shadow_capture.finalize()` (`:4203`), stashed onto
   `ctx.default_n_gl` (NGL) or `ctx.default_n_ll` (NLL) (`:4204-4207`).

Crucially, `_noLoadLoopBodyDefault` (`:3454`) *internally* DOES build a real
`Module` named `"noLoadLoopBody"` (`:3468`) and `module.add(subIterCode)` at
`:4059`, and `return module` at `:4070`. **So the SHADOW NGL/NLL body DOES exist
as a complete emittable `Module` at the moment `_noLoadLoopBodyDefault` returns
— and the SHADOW caller at `:4196` throws it away.** This is the natural seam for
NGL/NLL (§5).

Contrast the CMS NGL/NLL emission: the `else` at `:4238-4241` calls
`self.noLoadLoopBody(...)` (the CMS dispatcher) and `module.add(...)`s it; that
is what lands in `kernel.s`. Also contrast the Approach-A non-CMS path at `:4223-
4237`, which calls `module.add(self._noLoadLoopBodyDefault(..., capture=...))` —
it BOTH captures AND emits, because on a non-CMS build the default body IS the
runnable body.

### §2.4 The prologue SHADOW capture (`kernelBody`, ~`:5947-6025`)

The prologue is captured differently: it is a *snapshot of the real CMS
prologue's pack leaves* (`:5947-5966`), finalized via `build_prologue_capture`
at `:6020-6025`. There is no separate SHADOW prologue emission — the CMS
prologue (which IS in `kernel.s`) is the prologue for both captures. (See §7 for
the consequence on a "full" SHADOW file.)

### §2.5 FourPartCapture assembly (`kernelBody`, ~`:6177-6258`)

`ctx.default = FourPartCapture(main_loop={0: main}, main_loop_prev=..., n_gl=...,
n_ll=..., source="default-sia3", prologue=ctx.prologue)` at `:6247-6258`. The
four parts are `LoopBodyCapture`s (capture structures), assembled from
`ctx.default_main`/`ctx.default_n_gl`/`ctx.default_n_ll`/`ctx.prologue`. This is
the validator's reference (`KernelWriter.py:6413` `ctx.default =
self._last_default_capture`). **No `str()` of a default-schedule Module is ever
produced anywhere in this flow.**

### §2.6 How the CMS path turns a tree into text (`_getKernelSource`)

`_getKernelSource(kernel)` (`:10481-10499`): `_initKernel`, then `(error, kb) =
self.kernelBody(kernel, tPA, tPB)` (`:10491`), then `fileString += str(kb)`
(`:10492`), `return fileString`. `kernelBody` (`:5610`) builds one
`moduleKernelBody = KernelBody("kernelBody")` (`:5636`), adds signature
(`:5642`), the body `module` (`:7143` `moduleKernelBody.addBody(module)`), runs
`rocIsaPass` (`:7158`) and optionally StinkyTofu (`:7168-7227`), and returns
`(error, str(moduleKernelBody))` (`:7273`) or `(error, st_asm)` (`:7270`). **The
single string-emission step is `str(moduleKernelBody)` / `emitAssembly()` — it
is generic over what's in the tree, not CMS-specific.** The dumper
`_dump_hxcx_assembly.py:110-113` writes that return value to `kernel.s`.

### §2.7 `structural_clone` and the label-sharing invariant

`structural_clone(item)` (`ScheduleCapture.py:1350-1386`): recursively clones
Module wrappers but **shares leaf instruction references** (returns the leaf
unchanged for non-Module items, `:1377-1378`). Effect: leaf `id()` identity
survives the clone, so the `{id(item) -> category}` idMap built from the original
source modules still resolves against the cloned tree (`:1360-1362`). This is why
SHADOW uses `structural_clone` not `deepcopy` — `deepcopy` would give every leaf
a new id and break categorization.

The **label-sharing invariant** is stated verbatim at `KernelWriter.py:5115-
5117`: *"Label sharing is safe because iterCode is discarded by the capture
branch — never appended to module — so there's no duplicate-label-position hazard
at the assembler."* Because `structural_clone` shares leaves, the cloned SHADOW
body contains the SAME `Label` objects (and the same instruction objects) as the
CMS source modules. If both the CMS body AND the SHADOW body were `.add()`-ed
into the same runnable kernel tree (`kb`), the assembler would see each label
twice → duplicate-label / double-emission. The current code is correct *because*
it never adds the SHADOW body to `kb`. §4 analyzes whether this hazard survives
the separate-file design (it does not, for the runnable kernel).

---

## §3 The entanglement problem (why the current path can't cleanly emit text)

The SHADOW path fuses three concerns:

1. **Produce** the default-scheduled instruction stream (run SIA3 /
   `_noLoadLoopBodyDefault` over the CMS-mutated dict).
2. **Capture** it into a `FourPartCapture` (the `LoopBodyCaptureBuilder` walk).
3. **Discard** the emitted `Module`(s).

These are not separable today, in two different ways for the two body kinds:

- **NGL/NLL:** The produced `Module` *does* exist as a whole object (the return
  of `_noLoadLoopBodyDefault`, `:4070`) but is discarded at the call site
  (`:4196`). Concerns (1)+(3) are coupled at the caller: the caller's only way to
  "produce" is to call a function whose return it then drops. To emit text you
  must un-drop — but that's the bolt-on (§5) unless you introduce a seam.

- **Main loop:** The produced stream NEVER exists as a whole `Module` on the
  SHADOW side. CMS assembles via `customMainLoopSchedule` → one `optSchedule`
  Module. SHADOW assembles via per-iter `_makeSubIterSchedule` calls whose
  `iterCode` is immediately walked-and-discarded. **There is no single SHADOW
  main-loop Module to `str()`.** The capture (`LoopBodyCapture`) is the only
  whole-body artifact, and it is a capture data structure, not assembly text.

So "emit the SHADOW text" is not "stop discarding a Module" for the main loop —
that Module is never built. The proper design must *build* a SHADOW main-loop
Module (concern 1 made first-class), then route it to BOTH the capturer (concern
2) and a text sink — and never to `kb` (concern 3 becomes "route to a separate
sink", not "discard").

The capture path already half-acknowledges this: it walks `iterCode.flatitems()`
to build the capture (`_captureSubIterToBuilder`). That walk presupposes a built
`iterCode` per iter. The whole-body Module is *almost there* per-iter; it is just
never accumulated. The seam is to accumulate it.

---

## §4 The label-sharing / structural_clone hazard — and why a separate never-assembled file neutralizes most of it

### §4.1 Why the emitted SHADOW body is discarded today

Two reasons, both rooted in `structural_clone`'s leaf-sharing:

1. **Label/instruction object aliasing into `kb`.** The SHADOW body's leaves are
   the SAME Python objects as the CMS source modules' leaves (shared by
   `structural_clone`). If the SHADOW body entered `kb`, the runnable kernel
   would contain each shared `Label` twice (once via the CMS main-loop/NGL Module,
   once via the SHADOW Module) → duplicate-label assembler error / double body
   emission. The `:5115-5117` comment is exactly this guarantee.

2. **Input-consumption.** SIA3 mutates its inputs (`popFirstItem`). The SHADOW
   run operates on `structural_clone`d copies precisely so it does NOT consume
   the items the CMS bucket-accumulation needs immediately afterward (`:5106-
   5110`). The clone isolates *consumption*; the discard isolates *emission*.

### §4.2 Why a separate never-assembled `shadow_kernel.s` neutralizes hazard (1)

The duplicate-label hazard is a property of *one assembler input containing the
same label twice*. `shadow_kernel.s` is a **separate file that is never fed to
the assembler** — it is text for human/diff inspection only. The CMS `kernel.s`
and the SHADOW `shadow_kernel.s` are two independent assembler inputs (and in the
intended use, only `kernel.s` is ever assembled; `shadow_kernel.s` is never
assembled at all). Labels shared *across the two files* never collide, because
they are never in the same translation unit.

Concretely: rendering a SHADOW Module to text via `str(shadow_module)` is a pure
read of the tree — it does not add the Module to `kb`, does not mutate the shared
leaves, and produces an independent string. As long as the SHADOW Module is
`str()`-ed into its own file and **never `.add()`-ed into `kb`**, hazard (1)
cannot fire. **The separate-file requirement is therefore not incidental — it is
what makes the whole thing safe.** It converts the dangerous operation ("add the
SHADOW body to the runnable kernel") into a safe one ("render the SHADOW body to
an isolated string"). This is the single most important property of the design.

### §4.3 Residual hazards that the separate file does NOT neutralize

1. **Input-consumption (hazard 2) is independent of emission.** It is already
   handled by `structural_clone` today and the design must continue to feed the
   SHADOW run cloned inputs. Emitting text from the (already-cloned) SHADOW
   stream adds no new consumption risk — `str()` does not pop. But the design
   must NOT "share" the SHADOW Module's leaves back into any CMS-consumed list.
   Since we only read (str) the SHADOW Module, this holds.

2. **Shared-leaf mutation during render.** `str()`/`rocIsaPass`/StinkyTofu may
   mutate instructions (e.g. wait-count insertion, delay-alu). If the SHADOW
   render ran the SAME passes that the CMS render runs, and those passes mutated
   the *shared* leaf objects in place, the SHADOW render could perturb the CMS
   leaves. **Mitigation:** the SHADOW text render must operate on a `deepcopy` of
   the assembled SHADOW Module before running any mutating pass, OR render raw
   (pre-pass) `str()` only. The capture itself walks raw `iterCode` (pre-
   StinkyTofu), so a raw `str()` of the same assembled SHADOW Module is the
   apples-to-apples artifact for diffing the capture — and it sidesteps shared-
   leaf mutation entirely. (See §5 "render policy" and §9 risk 1.)

3. **Writer scratch state.** The SHADOW run already mutates writer state
   (`localReadsVacancy`, `scheduledGRInstCounts`, etc., per `:5118-5121`); the
   design adds no new state mutation beyond accumulating a side Module, so this
   residual is unchanged from today.

**Net:** the separate never-assembled file neutralizes the PRIMARY risk the prior
doc flagged (duplicate-label / double-emission into the runnable kernel). The
remaining residual (shared-leaf mutation by passes) is neutralized by rendering
the SHADOW Module *raw* (pre-pass) or on a `deepcopy`, which is also the correct
choice for a capture-faithfulness diff (the capture is pre-pass).

---

## §5 Proposed clean design (the refactor)

### §5.1 The seam: make "produce the SHADOW stream as a Module" first-class

The core refactor introduces ONE new concept — a **SHADOW emission accumulator**
on `CaptureContext` (a side `Module` tree, holding the four bodies) — and routes
the *already-produced* SHADOW streams into it at the two production sites, in
addition to (not instead of) the existing capture.

The key insight from §2/§3: at each SHADOW production site the instruction stream
is *already built as a Module-or-iterCode before it is walked into the capture*.
The clean seam is to **tee that already-built tree into the side accumulator at
the exact point it is currently discarded**, gated by an opt-in emission flag.

This is NOT the same as the bolt-on (un-discard in place). The difference is that
the tee goes to a *dedicated, separate, never-assembled accumulator* whose ONLY
consumer is the new `_getShadowKernelSource`, and the routing is expressed as a
single explicit "emit this body" call — not by removing the discard and hoping
nothing else picks the Module up. The discard semantics for `kb` are preserved
exactly; we add a parallel sink.

#### §5.1.1 Main loop

The SHADOW main loop has no whole Module today (§3). Two clean options:

- **Option M-tee-iter (preferred):** at `:5226`, the per-iter
  `_makeSubIterSchedule` call already returns the assembled `iterCode` (it is
  `return iterCode` at `:2706`, currently dropped). Capture that return and
  `shadow_emit.main.add(iterCode)` into the side accumulator. Because the
  capture walk already happened inside `_makeSubIterSchedule` on the same
  `iterCode`, the accumulated Module is *exactly the stream the capture walked* —
  the strongest possible faithfulness guarantee. No new scheduling work; we stop
  dropping a value we already compute.

- **Option M-rebuild (rejected):** re-run `customMainLoopSchedule` on the SHADOW
  dict. Rejected: it re-does scheduling, risks diverging from what the capture
  walked, and duplicates the CMS assembly logic.

Preferred: M-tee-iter. The accumulator's main body is the ordered concatenation
of the per-iter `iterCode`s — i.e. the SHADOW main loop as one Module.

#### §5.1.2 NGL/NLL

The whole Module already exists (`_noLoadLoopBodyDefault` returns it, `:4070`).
At `:4196`, instead of dropping the return, bind it: `shadow_nl_module =
self._noLoadLoopBodyDefault(..., capture=shadow_capture)`, then `shadow_emit.n_gl
= shadow_nl_module` (or `.n_ll`). Still NOT `module.add(...)` — the outer
runnable `module` is untouched. The capture finalize is unchanged.

#### §5.1.3 Prologue

The SHADOW prologue == the CMS prologue (§2.4). For a *full* SHADOW file we reuse
the same prologue text the CMS file has (the prologue is shared and IS in `kb`).
The clean way: `_getShadowKernelSource` stitches the CMS prologue/epilogue
scaffolding around the SHADOW bodies (see §5.3). The prologue is not re-emitted
on a SHADOW-specific path; it is the real one.

### §5.2 The new emission path: `_getShadowKernelSource`

Mirror `_getKernelSource` (§2.6) with a SHADOW-specific assembler that does NOT
touch the CMS flow. Shape:

```
def _getShadowKernelSource(self, kernel) -> str:
    # Pre-req: a CMS build with _captureDefaultSchedule has run, populating
    # self._capture_context.shadow_emit (the side accumulator).
    # Build a KernelBody mirroring kernelBody's scaffolding, but substitute
    # the SHADOW bodies for the CMS main-loop / NGL / NLL modules.
    # Render raw (pre-StinkyTofu) str() so the text matches what the capture
    # walked. Return the string.
```

There are two viable structural shapes for this method; the design recommends
**Shape B (stitch-in-place)** for fidelity, with **Shape A (separate KernelBody)**
as the simpler fallback:

- **Shape A — assemble a standalone SHADOW KernelBody.** Build a fresh
  `KernelBody`, add the (shared) signature, then add: the CMS prologue module,
  the SHADOW main-loop accumulator Module, the SHADOW NGL/NLL Modules, and the
  CMS epilogue module. `str()` it. This requires `kernelBody` to expose its
  prologue/epilogue sub-modules (see §6 refactor). Cleanest separation; the
  SHADOW file is a real, structurally-complete kernel that differs from
  `kernel.s` only in the scheduled bodies.

- **Shape B — stitch the SHADOW bodies into a clone of the CMS body tree.**
  `deepcopy` the CMS `moduleKernelBody`, locate the main-loop / NGL / NLL sub-
  modules by name, replace them with the SHADOW bodies, `str()` the result. More
  invasive to do robustly (relies on findNamedItem of the loop modules) and a
  deepcopy of the whole kernel; offers no fidelity advantage over A for the
  bodies. **Not recommended.**

Recommended: **Shape A**, driven by a single refactor that extracts the
prologue/main-loop/epilogue stitching in `kernelBody` into a small helper both
`kernelBody` (CMS) and `_getShadowKernelSource` (SHADOW) call with different
body modules.

### §5.3 Reuse the CMS body-stitching logic (do not duplicate it)

`kernelBody` (`:5610-7143`) builds the kernel by adding, in order: signature →
prologue (setup/prefetch, `:5640-6040`) → `module.add(loop)` (main loop,
`:6075`) → NGL/NLL (`:6103-6170`) → epilogue (global write, `:7130-7141`). The
clean refactor extracts the **body assembly** into a method parameterized by the
"loop bodies provider":

```
def _assembleKernelBody(self, kernel, tPA, tPB, *, loop_bodies):
    # loop_bodies: an object that supplies the main-loop Module, the NGL/NLL
    # Modules, and (for SHADOW) leaves the prologue/epilogue as the CMS ones.
    # CMS path: loop_bodies = the live CMS emission (today's inline behavior).
    # SHADOW path: loop_bodies = the side accumulator from CaptureContext.
```

In practice the minimal clean extraction is: **a helper that takes the already-
built prologue Module, a main-loop Module, the NGL/NLL Modules, and the epilogue
Module, and returns the assembled `KernelBody`.** `kernelBody` builds its four
modules inline as today and passes them; `_getShadowKernelSource` passes the CMS
prologue/epilogue (captured once during the CMS build) plus the SHADOW bodies.
This shares the stitching order/scaffolding without `kernelBody` knowing about
SHADOW. CMS behavior is identical because it passes the same modules it builds
today.

### §5.4 Contrast: clean refactor vs. the hacky bolt-on

| Aspect | Hacky bolt-on (REJECTED) | Clean refactor (THIS DESIGN) |
|---|---|---|
| Main-loop emission | Un-discard `iterCode` at `:5226` and `str()` fragments inline, manually concatenating per-iter text | Tee the already-returned `iterCode` into a first-class side accumulator Module; render once via `_getShadowKernelSource` |
| NGL/NLL emission | `str()` the dropped return inline at `:4196`, write file mid-build | Bind the return into the accumulator; render via the shared stitch helper |
| Stitching | Re-implement prologue/epilogue concatenation in the SHADOW branch | Reuse the extracted `_assembleKernelBody` stitch helper |
| File write | Side-effect file I/O buried in `_loopBody`/`noLoadLoop` | No I/O in the writer; `_getShadowKernelSource` returns a string, dumper/test writes the file |
| Label hazard | High — easy to accidentally let a teed Module reach `kb` | Structurally impossible — accumulator's only consumer is `_getShadowKernelSource`, never `kb` |
| CMS `kernel.s` | At risk (inline edits in shared branches) | Provably untouched (no edit to the CMS module-build/str path) |
| Separation of concerns | Re-entangles produce/capture/emit | Produce → {capture, accumulate}; emit is a separate read-only pass |

**Why the refactor is worth it.** The bolt-on scatters text-emission and file-I/O
through the two load-bearing SHADOW branches, exactly where the standing rule
says not to perturb, and keeps the produce/capture/emit concerns fused — so every
future change to the SHADOW path has to reason about three things at once and re-
verify the label invariant. The refactor makes "produce the stream" yield a
reusable Module that *any* consumer (capture, emit, future dataflow probes) can
read, isolates emission as a separate read-only pass with a single sink, and
makes the duplicate-label hazard structurally unreachable rather than avoided-by-
discipline. It also makes the SHADOW main loop a real Module for the first time,
which is independently useful (e.g. future per-body assembler-pass experiments).

### §5.5 Render policy (raw vs. post-pass)

`_getShadowKernelSource` should render **raw** `str()` of the assembled SHADOW
`KernelBody` — i.e. BEFORE `rocIsaPass`/StinkyTofu — for two reasons: (1) the
SHADOW *capture* is built by walking raw `iterCode` (`_captureSubIterToBuilder`
runs on pre-pass leaves), so raw text is the apples-to-apples diff target; (2) it
avoids any shared-leaf mutation by passes (§4.3 residual 2). If a post-pass
SHADOW artifact is ever wanted, run the passes on a `deepcopy` of the SHADOW
KernelBody so the shared CMS leaves are never touched. **Default: raw.**

---

## §6 Refactor scope (file/function level — what changes / what must NOT change)

### §6.1 New code

- **`CaptureContext.shadow_emit`** (`ScheduleCapture.py`, the `CaptureContext`
  dataclass at `:582`): a new scratch field holding the SHADOW emission
  accumulator — a small struct with `main` (Module), `n_gl` (Module|None),
  `n_ll` (Module|None). Cleared in `reset()` (`:639-650`). Gated: only populated
  when `_captureDefaultSchedule` AND a new opt-in `_emitShadowKernel` flag are
  set (so production CMS builds that only need the capture pay nothing).
- **`KernelWriterAssembly._getShadowKernelSource(kernel) -> str`** (new method,
  `KernelWriter.py`): mirrors `_getKernelSource` (`:10481`); assembles a SHADOW
  `KernelBody` via the shared stitch helper using the accumulated SHADOW bodies +
  CMS prologue/epilogue; raw-`str()` renders; returns the string. No file I/O.
- **An opt-in flag** `self._emitShadowKernel` (set by a new
  `enable_shadow_kernel_emission()` method mirroring
  `enable_capture_default_schedule`, `:478`), so the accumulator is only built
  when a caller wants the artifact. Auto-activation (`:5617`) does NOT set it —
  production validator builds don't pay the accumulation cost.

### §6.2 Refactors of existing code (introduce the seam)

- **Extract body-stitching from `kernelBody`** into `_assembleKernelBody`
  (or a narrower `_stitchKernelBody(signature, prologue, mainLoop, nglnll,
  epilogue) -> KernelBody`). `kernelBody` calls it with the modules it builds
  today (behavior identical). `_getShadowKernelSource` calls it with SHADOW
  bodies. This is the only structural change to `kernelBody`, and it is a pure
  extraction (no behavior change on the CMS path).
- **`kernelBody` must retain the CMS prologue and epilogue modules** (e.g. stash
  on `self._capture_context.cms_prologue_module` / `cms_epilogue_module`) during
  the CMS build so `_getShadowKernelSource` can reuse them. Additive; does not
  change what CMS emits.
- **`_loopBody` SHADOW branch (`:5226`):** bind the `_makeSubIterSchedule`
  return and, when `_emitShadowKernel`, `ctx.shadow_emit.main.add(iterCode)`.
  Guard: NOT `module.add` — `module` (which feeds `kb`) is untouched. This is a
  *non-destructive* change: when `_emitShadowKernel` is off, the value is dropped
  exactly as today.
- **`noLoadLoop` SHADOW branch (`:4196`):** bind the `_noLoadLoopBodyDefault`
  return and, when `_emitShadowKernel`, stash it on `ctx.shadow_emit.n_gl/n_ll`.
  Again NOT `module.add`.

### §6.3 What must explicitly NOT change

- **`_getKernelSource` (`:10481-10499`)** — the CMS full-kernel emitter. Zero
  edits. SHADOW emission is a sibling method.
- **The CMS string-assembly path** — `str(moduleKernelBody)` /
  `emitAssembly()` (`:7270`/`:7273`), `rocIsaPass` (`:7158`), StinkyTofu
  (`:7168-7227`). Unchanged; SHADOW renders raw via its own path.
- **What SHADOW captures** — `_makeSubIterSchedule`'s capture call (`:2694`),
  `_captureSubIterToBuilder` (`:2708`), the idMap construction (`:5141`), the
  finalize (`:5475`), the `FourPartCapture` assembly (`:6247`). The tee READS the
  same `iterCode`/Module the capture already walked; it does not alter capture
  inputs, order, or content. (Verify post-change: capture listings byte-identical
  before/after.)
- **`customMainLoopSchedule` (`dispatch.py:81`)** and the CMS main-loop assembly
  at `:5478-5483`. Untouched.
- **The validator consumer** (`:6375-6413`, `ctx.default = _last_default_capture`).
  Untouched; the artifact is for offline diffing, not the inline validator.

### §6.4 Test / dumper changes

- **`_dump_hxcx_assembly.py`:** after the existing CMS `_getKernelSource` call
  (`:110`), call `writer.enable_shadow_kernel_emission()` *before* the build (it
  must be armed before `_loopBody` runs — same timing as
  `enable_capture_default_schedule`), then write `writer._getShadowKernelSource(
  solution)` to `shadow_kernel.s`. (Because the emission flag must be set before
  the build, the dumper arms it up front and builds once; the SHADOW accumulator
  is filled during the same CMS build that fills the capture.)
- **A new assertion test** (sibling to the dumper, in `Tensile/Tests/unit/`):
  build the canonical kernel with capture + SHADOW emission, render
  `shadow_kernel.s` and the SHADOW capture listing, and assert the capture's
  instruction sequence matches the emitted SHADOW text (the §8 proof). This is
  the regression pin that keeps the SHADOW capture honest.

---

## §7 How `shadow_kernel.s` gets produced and diffed (dumper/test integration)

### §7.1 Production

One CMS build, armed for both capture and SHADOW emission:

1. `writer.enable_capture_default_schedule()` (already implicit for CMS) —
   builds the `FourPartCapture` reference.
2. `writer.enable_shadow_kernel_emission()` — arms the side accumulator (NEW).
3. `src = writer._getKernelSource(solution)` — the CMS build; writes `kernel.s`
   (unchanged). During this build, the SHADOW branches ALSO fill
   `ctx.shadow_emit`.
4. `shadow_src = writer._getShadowKernelSource(solution)` — NEW; stitches the
   SHADOW bodies into a KernelBody, raw-`str()`s it, returns the text. The dumper
   writes it to `shadow_kernel.s`.

`kernel.s` (CMS) and `shadow_kernel.s` (SHADOW) are now both on disk, from the
SAME CMS build over the SAME CMS-mutated dict. The dumper also already writes
`cms_capture_listing.txt` and `shadow_capture_listing.txt` (`_dump_hxcx_assembly
.py:192-197`).

### §7.2 Diff / assertion

- **CMS proof (exists today):** `cms_capture_listing.txt` ↔ `kernel.s` (the §2-
  style direct text proof).
- **SHADOW proof (NEW):** `shadow_capture_listing.txt` ↔ `shadow_kernel.s` —
  diff the capture's per-body instruction sequence against the emitted SHADOW
  bodies. Because the accumulator holds the EXACT `iterCode`/Module the capture
  walked (M-tee-iter / NGL/NLL return-bind), and the render is raw, the two
  should match leaf-for-leaf modulo capture-listing formatting (the listing
  carries category/slot columns; the `.s` is bare instructions — the assertion
  compares the canonical-instruction sequence, not column-for-column text). The
  new assertion test (§6.4) pins this.

---

## §8 Faithfulness payoff + does it close the SHADOW-capture question

**End state.** The SHADOW side gets the same faithfulness proof the CMS side has:
a full assembly text (`shadow_kernel.s`) emitted from the *actual on-CMS-build
SHADOW run*, text-diffable against `shadow_capture_listing.txt`. This converts the
`"default-sia3"` capture's "faithful by construction / producer-side argument"
status (`H7LO_UVRL_NORMATIVE_ORDERING.md:359-367`;
`NGL_CAPTURE_INVESTIGATION_MEMO.md:238-245`) into a *direct text diff* — closing
the no-emitted-`.s` residual that bead `rocm-libraries-4ydd` tracks.

**uvrl / NGL ds_read ordering.** The uvrl question has two halves:

- **Capture-artifact half (does the SHADOW capture lie about NGL order?):**
  `shadow_kernel.s` answers this **decisively**. If the emitted SHADOW NGL body
  shows `ds_read` after the pack chain in the same order
  `shadow_capture_listing.txt:477/490/610` records (pack-mfma → pack-cvt →
  ds_read), the capture is proven faithful and the CMS-vs-SHADOW NGL divergence is
  REAL, not a capture artifact. This is exactly the proof the prior doc said
  "would fully close it" (`SHADOW_FULL_EMISSION_FEASIBILITY.md:266-272`).
- **Normative half (which order — default's `pack → ds_read` or CMS's `ds_read →
  pack` — is authoritative):** NOT decided by this artifact. That is a scheduler-
  intent question for the `_get_schedule_128x128x32_TF32` NGL-ordering owner
  (uvrl bead). The artifact removes the capture-fidelity uncertainty so the uvrl
  decision rests purely on scheduler intent.

**Net:** this design closes the SHADOW-capture-*faithfulness* question (the
capture is provably what the emitter produced), which is the half that has been
blocking confidence in SHADOW-as-canonical-reference since `dm4p` Phase 2 made it
so (`KernelWriter.py:6413`). It does not — and cannot — decide the normative
ordering question; that is correctly out of scope.

**CMS `kernel.s` untouched — provable.** Under this design `_getKernelSource` and
the entire CMS module-build + `str()` path are not edited (§6.3); the SHADOW
changes are (a) a gated, non-destructive bind-instead-of-drop in two branches
that never touch `module`/`kb`, (b) a pure extraction of stitching that
`kernelBody` calls with the same arguments as today, and (c) a new sibling
method. The regression guard is trivial: byte-compare `kernel.s` before and after
the refactor for the canonical kernel — it must be identical. Combined with the
new SHADOW assertion test, both sides are text-pinned.

---

## §9 Risks / open questions

1. **Shared-leaf mutation by passes (residual from §4.3).** Mitigated by raw
   rendering (§5.5). Open question: do we ever want a *post-pass* SHADOW artifact
   (to compare wait-count insertion between schedulers)? If so, it MUST run on a
   `deepcopy` of the SHADOW KernelBody. Recommend deferring until a concrete need;
   default raw.
2. **Prologue/epilogue identity.** `shadow_kernel.s` uses the CMS prologue/
   epilogue (shared, the real ones). That makes the SHADOW file a real kernel
   that differs from `kernel.s` ONLY in the scheduled bodies — which is exactly
   right for diffing the bodies. It is NOT a standalone "default kernel as if
   built with CMS=0" (that's Approach A). Document this clearly so nobody expects
   `shadow_kernel.s` to match a CMS=0 build.
3. **Timing of the emission flag.** `enable_shadow_kernel_emission` must arm the
   accumulator before `_loopBody` runs (same monkey-patch-`setupNewTile` timing
   as `enable_capture_default_schedule`, `:508-515`). If armed too late the
   accumulator is empty and `_getShadowKernelSource` raises. The new method
   should fail loud if `ctx.shadow_emit` is empty/unset at call time.
4. **Main-loop accumulator order vs. CMS macro order.** M-tee-iter accumulates
   per-iter `iterCode` in subiter order. The SHADOW capture is also per-iter in
   the same order, so they match by construction. But the *CMS* main loop is
   interleaved across codepaths by `customMainLoopSchedule` — so `shadow_kernel.s`
   main-loop will NOT line up with `kernel.s` main-loop line-for-line (different
   scheduler). That is expected and fine: `shadow_kernel.s` is diffed against the
   SHADOW *capture*, not against `kernel.s`.
5. **`loopCopies==1` assumption.** The capture assembly already asserts
   `loopCopies==1` under CMS (`:6186-6190`, `:6383-6388`). The SHADOW accumulator
   inherits the same restriction; `_getShadowKernelSource` should assert it too
   (DTV/ULSGRO multi-body kernels need per-lc accumulator support — out of scope,
   same as the existing capture limitation).
6. **OptNLL guard.** SHADOW capture already rejects CMS+OptNLL (`:5276-5284`).
   The SHADOW emission inherits this — no extra handling, but the accumulator
   will simply lack an NLL body for those (already-rejected) kernels.
7. **StinkyTofu coupling.** `_getKernelSource` may route through StinkyTofu
   (`:7168`). `_getShadowKernelSource` rendering raw skips StinkyTofu entirely,
   which is correct for the capture diff but means `shadow_kernel.s` is pre-
   StinkyTofu while `kernel.s` may be post-StinkyTofu. Since the two files are
   never compared to each other (only each to its own capture), this is fine —
   but note the CMS capture is ALSO pre-StinkyTofu (built from raw `iterCode`),
   so the CMS `kernel.s`-vs-capture proof has the same pre/post caveat today.
   Document it.

---

## §10 Bead

Existing bead **`rocm-libraries-4ydd`** ("Emit full SHADOW/default-schedule
assembly text to ground-truth the SHADOW capture") already scopes Approach B /
SHADOW-tee and is `related` to `rocm-libraries-uvrl`. This design doc IS the
proper-refactor specification that bead asked for. A comment was added to
`rocm-libraries-4ydd` pointing at this doc (the clean seam: tee the already-built
SHADOW `iterCode`/Module into a `CaptureContext.shadow_emit` accumulator + a new
`_getShadowKernelSource`, with `kernelBody`'s stitching extracted into a shared
helper). No new bead created — this is the design for the existing one.
