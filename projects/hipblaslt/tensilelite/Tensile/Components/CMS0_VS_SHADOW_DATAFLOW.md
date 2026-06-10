# cms=0 vs SHADOW: Rotating Pack-Buffer Dataflow, and Reference Validity

Investigation date: 2026-06-10. Kernel: BPG#11 TF32 4x4 TN (`CANONICAL_KERNEL_CONFIG`).
All claims cite `file:line`. cms=0 = `hxcx_artifacts/cms0_kernel.s` (a REAL default
`_getKernelSource` emit, `UseCustomMainLoopSchedule=0`). SHADOW =
`Tensile/Components/h7lo_uvrl_artifacts/shadow_capture_listing.txt`. CMS (cms=1) =
`Tensile/Components/h7lo_uvrl_artifacts/cms_capture_listing.txt`.

---

## SUMMARY (verdict)

- **Does the REAL default (cms=0) kernel repeat math across bodies? NO.** Each
  ML/NLL body ds_reads a DISTINCT k-slice into the `T0_I0` staging buffer (offsets
  advance 0/64/128/.../448; physical LDS buffer alternates via `v_xor ...,0x10000`),
  packs `T0->X0`, then consumes `X0`. The `X0_I0+12..15` registers are NEVER
  ds_read in the cms=0 main loop — only in the TailLoop (`cms0_kernel.s:3302`).
- **Is the SHADOW capture faithful to a real default kernel? NO — it is a synthetic
  hybrid.** SHADOW is the default `_makeSubIterSchedule` ORDERING applied to the
  **cms=1 INSTRUCTION SET** (the CMS-mutated kernel dict, with
  `doFullPackCodePrefetch=False`). The code itself names it the "**synthetic SHADOW
  default capture**" (`KernelWriter.py:493`, `:557-561`). It uses the in-place X0
  scheme that the real default path (cms=0) **never emits in its main loop**.
- **Does the NGL pack repeat ML's data? It consumes ML's ds_read output, but that is
  correct DRAIN semantics for the in-place rotating buffer — not a repeat.** However,
  that question is largely **moot**, because the in-place scheme it lives in only
  exists in the synthetic SHADOW/cms=1 stream, not in a real default kernel.
- **Does this reframe uvrl? YES.** The 16 `EdgeRoutedDifferently` failures compare
  the CMS subject against a reference that does not correspond to any runnable
  default kernel for the affected byte_keys. uvrl is NOT "teach the validator
  rotating-buffer equivalence"; it is "the SHADOW reference construction is
  structurally unfaithful — replace it with the real-vs-real Approach-A reference."
- Confidence: **High.** Bead filed: **rocm-libraries-svds (P0)**, blocking r62g.

---

## §1 The user's repeated-math concern (restated)

Prior analysis concluded that, on the SHADOW schedule, the NGL body's pack reads the
SAME logical k-fragment that the ML body already read/packed. The user objects: if
NGL re-reads ML's data, the math is REPEATED; NGL should operate on the CURRENT
iteration's freshly-read data. Either the "same fragment" conclusion is wrong, or
something is structurally off.

This investigation finds: **something is structurally off — but not in the math.**
The real default kernel does not repeat math at all. The SHADOW capture's in-place
X0 dataflow (which is what generated the "same fragment" observation) is a synthetic
artifact: it is the default scheduler applied to the cms=1 instruction set, not a
representation of how the default path actually emits these reads.

---

## §2 The real cms=0 kernel: per-body pack-buffer dataflow

Body markers (`hxcx_artifacts/cms0_bodies.txt`):
main loop `label_LoopBeginL` 1872 – `label_LoopEndL` 2200; Ord. NoGlobalLoadLoop
begins at 2200 (`/* Ord. NoGlobalLoadLoop_1 - Begin */` at `cms0_kernel.s:2205`);
`label_toPGR1` 2511 (OptNLL); `label_TailLoopBeginL` 3296 – `label_TailLoopEndL` 3766.

### The cms=0 scheme is pure `T0 -> X0` (doFullPackCodePrefetch ON)

- **Load target is T0, separate from pack-destination X0.** Every main-loop ds_read
  lands in `ValuA_T0_I0` (`cms0_kernel.s:1886` offset:256, `:1894` offset:320,
  `:1900` offset:384, `:1906` offset:448; next subiter `:2081/2084/2093/2096`
  offsets 0/64/128/192). The pack-cvt then writes `X0_I0+12..15` FROM `T0_I0`
  inputs (`cms0_kernel.s:1850-1853`, `:2158-2163`).
- **`X0_I0+12..15` is NEVER ds_read in the main loop.** Empirically: ds_reads into
  `ValuA_X0_I0` in region 1872–2200 = **0**; across the whole kernel = **8**, all in
  the TailLoop (`cms0_kernel.s:3299-3306`). The single `ds_read ... X0_I0+12` is at
  `cms0_kernel.s:3302` (TailLoop, offset:192).

### ML (main loop, 1872–2200)
- ds_reads distinct slices into T0: subiter-0 group `:1886/1894/1900/1906`
  (offsets 256/320/384/448, `sync LDS0`); subiter-1 group `:2081/2084/2093/2096`
  (offsets 0/64/128/192, `sync LDS1`). Offsets advance and the LDS buffer toggles
  (`v_xor v[vgprLocalReadAddrA],0x10000` at `:1957`), so **each ds_read targets a
  distinct k-fragment in a distinct physical LDS buffer.**
- Pack: `X0_I0+12 = pack(T0_I0+8,T0_I0+9)` etc. (`:1853`, `:2163`). Consumes T0
  (this iter's freshly-read data), produces X0 for the mfma's (`:1884`, `:1924`...).
- **No repeat:** load(distinct slice)->pack->consume, advancing every iteration.

### NGL / Ord. NoGlobalLoadLoop (2205–2511)
- Still ds_reads into T0 (`:2214/2222/2228/2234` offsets 256/320/384/448 `LDS1`;
  `:2395/2398/2407/2410...` offsets 0/64/128/192 `LDS0`) and packs T0->X0
  (`:2475-2480`). This is the standard drain of the last prefetched LDS buffer plus
  the final buffer's reads. Distinct offsets / distinct LDS buffer → distinct
  k-slices. **No repeat.**

### NLL / OptNLL (2511–2924)
- Same T0->X0 scheme; ds_reads into T0 at `:2548/2551/2554/2557` and
  `:2937/2940/2943/2946`. **No repeat.**

### TailLoop (3296–3766) — the ONLY in-place X0 body
- Here the K-remainder is handled directly: ds_read lands straight into
  `X0_I0+0..28` (`:3299-3306`, including `X0_I0+12` at `:3302` offset:192), then
  packed in-place (`:3640-3641`, `:3653-3656`) and consumed (`:3643`). This is the
  in-place scheme — but it is a SEPARATE, post-main-loop body for the K tail, with
  its own LDS contents. It does not re-derive main-loop data.

**Verdict §2: the real cms=0 kernel does NOT repeat math in any body. Decisive
citation: zero `ds_read ... ValuA_X0_I0` in `cms0_kernel.s:1872-2200`; the lone
`X0_I0+12` ds_read is `cms0_kernel.s:3302` (TailLoop only).**

---

## §3 cms=0 (T0->X0) vs SHADOW (in-place X0): the structural difference

| Aspect | cms=0 (real default) | SHADOW (synthetic) |
|---|---|---|
| `doFullPackCodePrefetch` | True (`KernelWriter.py:9080`) | False (cms=1 build) |
| Main-loop load target | `T0_I0` only | mixed: some `T0_I0`, some `X0_I0` in-place |
| `X0_I0+12` ds_read in ML | **none** (0 in 1872-2200) | **every body** (shadow:194, 388, 610) |
| Pack inputs (final) | `pack(T0_I0+8,+9)` (cms0:1853) | `pack(T0_I0+4,+5)` + in-place `pack(X0+12,+13)` (shadow:474,493) |

Concretely, the SHADOW ML body has BOTH kinds of ds_read interleaved:
`shadow:247` (`T0_I0+8` offset:256), `shadow:257` (`T0_I0+12` offset:384) AND
`shadow:252` (`X0_I0+20` offset:320), `shadow:262` (`X0_I0+28` offset:448),
plus the in-place `X0_I0+12` at `shadow:388` (offset:192). The cms=0 ML body has
ONLY the `T0_I0` form (`cms0:1886/1894/1900/1906`). These are **different
instruction sets**, not merely different orderings.

The flag that produces this divergence is at `KernelWriter.py:9080`:
`self.states.doFullPackCodePrefetch = kernel["UsePLRPack"] and not kernel["UseCustomMainLoopSchedule"]`.
And `KernelWriter.py:5835` keeps `usePLRPack` effectively on under CMS
(`usePLRPack = self.states.doFullPackCodePrefetch or (kernel["UseCustomMainLoopSchedule"] and kernel["UsePLRPack"])`)
but WITHOUT the full prefetch staging — i.e., the in-place X0 form.

---

## §4 Is the SHADOW capture faithful to a real kernel, or a synthetic hybrid? (CRUX)

**It is a synthetic hybrid.** Three independent lines of evidence:

1. **The code says so explicitly.** `KernelWriter.py:478-568` documents two
   reference paths:
   - `_captureDefaultSchedule` (SHADOW): "runs SHADOW captures on a CMS build: a
     **synthetic re-assembly** via `_noLoadLoopBodyDefault` and `_makeSubIterSchedule`
     **against the CMS-mutated `kernel` dict**" (`KernelWriter.py:557-561`); it
     "compares ctx.cms vs a **synthetic SHADOW default capture**"
     (`KernelWriter.py:493`); "finalizes BEFORE `closeLoop` runs ... missing the
     loop-counter code (LCC)" (`KernelWriter.py:560-561`).
   - `_captureNonCmsBuild` (Approach-A): "runs on a **true non-CMS build**, where the
     non-CMS branches ... emit **real runnable instructions**"
     (`KernelWriter.py:563-567`).

2. **SHADOW shares the cms=1 instruction set, not the cms=0 one.** The cms=1 capture
   and SHADOW both contain the in-place `X0_I0+12` ds_read at offset:192
   (`cms_capture:206/402/584` vs `shadow:194/388/610`) AND both pack
   `X0_I0+12 = pack(T0_I0+4,T0_I0+5)` (`cms_capture:243/439` vs `shadow:220/414`).
   They differ only in ordering. SHADOW = default scheduler over the cms=1 op set.
   The real cms=0 has neither the in-place ds_read nor the `pack(T0+4,T0+5)` final
   (it uses `pack(T0+8,T0+9)` at `cms0:1853`).

3. **The single-build provenance confirms it.** `cms_from_default.py:119-121`
   builds ONE writer/solution, enables capture, calls `_getKernelSource(solution)`
   once. SHADOW (`_last_default_capture`) and CMS (`_last_cms_capture`) come from the
   SAME cms=1 kernel dict. The default reference is therefore not a second, genuine
   cms=0 emit — it is the default scheduler re-running over the cms=1 instructions.

**Conclusion: SHADOW default-SCHEDULES the cms=1 INSTRUCTION SET. The
validator's reference does NOT correspond to a kernel the default path would emit
(the default path would use the T0->X0 form, with no in-place X0+12 ds_read in the
main loop).**

---

## §5 The repeated-math question resolved for the SHADOW capture: (a)/(b)/(c)

In SHADOW, the NGL pack at `shadow:474` (`X0+10 = pack(X0+12,X0+13)`) reads X0_I0+12.
That X0_I0+12 content was written by ML's in-place ds_read at `shadow:388`
(offset:192). NGL's own in-place X0+12 ds_read is `shadow:610` (offset:192), which
lands LATE in the NGL body (seq 171), AFTER NGL has consumed (`:474/:491`) and
repacked (`:493`) the prior content. So NGL's first consumer DOES read ML-produced
data — the rotating-buffer drain.

Classification: **primarily (c), with (a) as the local mechanism.**
- (a) is locally true: within the in-place scheme, NGL consuming ML's last-loaded
  X0+12 is the correct DRAIN of the final prefetched fragment — the last real
  iteration's compute, not a repeat. The ds_read at `shadow:610` advances the buffer
  for the next consumer. There is no double-counted math.
- (b) is false: it is not a bug in the captured stream's own logic.
- (c) dominates: the in-place X0 scheme this all lives in is itself a synthetic
  artifact of the cms=1 instruction set re-scheduled by the default scheduler. A real
  default kernel (cms=0) has NO in-place X0+12 ds_read in the main loop (§2), so the
  "ML loads X0+12 / NGL drains X0+12" structure does not exist in any runnable
  default kernel. The repeated-data observation is an artifact of comparing against a
  non-runnable reference.

---

## §6 Does this reframe uvrl?

**Yes, decisively.** The 16 `compare_graphs` failures
(`hxcx_artifacts/compare_graphs_failures.txt`) all have the shape: "Subject's
consumer Pack* reads from subject's producer Pack*, but **reference routes through
LR* (of next iteration)**." Tabulated: 4× `LRA3[1]`, 4× `LRA3[3]`, 4× `LRB3[1]`,
4× `LRB3[3]`; byte_keys = X0_I0 rotating regs (`('v',4/5/6)`, `('v',11/12/13/14)`,
`('v',35/36/37/38)`, `('v',43/44/45/46)`).

The reference's "LR3" producer IS the in-place `X0_I0+12` ds_read at offset:192 —
which exists ONLY because SHADOW is built over the cms=1 instruction set. A real
cms=0 main loop routes `Pack(T0)->X0` exactly like the CMS subject. **Against a
faithful (real cms=0 / Approach-A) reference, these 16 edges would route Pack->Pack
on both sides and match.**

Therefore uvrl is NOT "teach the validator that rotating-buffer Pack/LR routing is
equivalent." It is: **the SHADOW reference is structurally unfaithful; replace it
with the real-vs-real Approach-A reference** (`_captureNonCmsBuild` /
`build_non_cms_reference`). That work is already in flight:
- `rocm-libraries-czby` (closed): repoint FourPartCapture at real Build #2, remove
  shadow feed sites.
- `rocm-libraries-xj16` (in_progress): wire real-vs-real `compare_graphs` inline
  assertion in `_captureNonCmsBuild`; auto-activate per CMS kernel.
- `rocm-libraries-r62g` (open): Phase 3 hard go/no-go gate validating the reference
  across the CMS test surface.

This finding supplies the decisive dataflow proof that the SHADOW reference must go.

---

## §7 Verdict + confidence

1. Real cms=0 default kernel: **no repeated math.** load(distinct slice)->pack->
   consume per body; in-place X0 only in the TailLoop. (`cms0_kernel.s:3302` is the
   sole `X0+12` ds_read; 0 in the main loop.)
2. SHADOW: **synthetic hybrid** — default scheduler over the cms=1 instruction set;
   uses the in-place X0 scheme a real default never emits in its main loop.
   (`KernelWriter.py:493,557-561`; instruction-set match to cms=1 at
   `cms_capture:206/402/584` vs `shadow:194/388/610`.)
3. SHADOW NGL "repeats" ML's X0+12 data: correct drain semantics locally (a), but
   the whole structure is a reference artifact (c). Not a math bug.
4. uvrl reframed: the reference is questionable, not the model. Fix = adopt
   Approach-A real reference (czby/xj16/r62g).

**Confidence: High.** Every claim is grounded in emitted assembly / capture
listings / source line numbers. The one residual unknown is whether EVERY one of the
16 failures flips to a match under the Approach-A reference; that is exactly what
r62g's go/no-go gate exists to confirm.

---

## §8 Bead filed

- **rocm-libraries-svds (P0, bug)** — "SHADOW reference is a synthetic hybrid: 16
  BPG#11 EdgeRoutedDifferently failures route through an in-place X0+12 ds_read that
  a real default (cms=0) kernel never emits." Linked as a blocker of
  `rocm-libraries-r62g` (the Phase 3 SHADOW go/no-go gate). Records the decisive
  dataflow proof supporting SHADOW retirement (czby/xj16/r62g).
