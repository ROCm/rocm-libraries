# CustomSchedule.py Refactor — Phase 1 Investigation

**Bead:** rocm-libraries-bkub
**Branch base tip:** `7bf7158524`
**File under study:** `Tensile/Components/CustomSchedule.py` — 5725 lines, 261 KB
(bead description estimate of "~205KB" was conservative; actual is 261 KB)

## TL;DR

- The file is 84% schedule bodies (4817 lines across 38 schedules) and 16%
  shared infrastructure (~908 lines of helpers, dispatch, decorator, dtype
  predicates, registry).
- Dispatch is a flat list (`_SCHEDULE_REGISTRY`) iterated in registration
  order. Each registered function is a `RegisterSchedule`-wrapped predicate
  that gates on `(TileConfig, dtype_predicate, vector_widths, MatrixInstruction,
  MIWaveGroup)` and returns `(ScheduleMatchStatus, ScheduleInfo|None)`.
- All 38 currently-registered schedules target gfx950 (default
  `isa=(9,5,0)` in `TileConfig`) — but `hasCustomSchedule` also gates
  `IsaVersion(11,5,1)`, anticipating future RDNA registrations.
- Boilerplate is modest per body (5-15 lines around an `optSchedule`
  literal); it is **structural** (the literal-dict shape itself), not
  textual. Splitting into per-schedule files is justified; further
  helper-extraction is *not* the dominant lever.
- Cross-schedule references exist (3 alias-shaped sites where one schedule
  delegates to another via `switch_A_B_schedule`). The refactor must hoist
  these dependencies — see §5.
- Test coupling to internals is small: only `test_CustomSchedule.py`
  (public `hasCustomSchedule`, `ScheduleInfo`) and
  `test_CustomSchedule_LayoutAutoDetection.py` (8 internal symbols) reach
  past the public façade. A schedule-name snapshot in the latter (lines
  187-225) means schedule-function names must remain stable across the
  refactor.

---

## 1. Inventory

### 1a. Public entry points (KernelWriter / Solution / tests reach these)

| Symbol | Lines | Where called | Phase 2 home |
|--------|-------|--------------|--------------|
| `customMainLoopSchedule` | 284-553 | `KernelWriter.py:47, 4581` | `dispatch.py` |
| `hasCustomSchedule` | 556-589 | `Solution.py:47`, `KernelWriter.py` | `dispatch.py` |
| `ScheduleInfo` (class) | 222-274 | tests + every schedule body | `shared.py` |
| `query_cms_kernels` | 592-613 | downstream tuning code | `dispatch.py` |
| `get_cms_kernel_info_objects` | 616-629 | downstream | `dispatch.py` |
| `get_available_dtypes` | 632-634 | downstream | `dispatch.py` |
| `get_available_layouts` | 637-653 | downstream | `dispatch.py` |
| `CMSKernelInfo` (frozen dataclass) | 69-115 | tests + query API | `shared.py` |
| `RegisterSchedule` (decorator class) | 712-899 | every schedule + tests | `dispatch.py` |
| `ScheduleMatchStatus` (Enum) | 51-54 | dispatch + tests | `dispatch.py` |
| `TileConfig` (frozen dataclass) | 690-702 | every `@RegisterSchedule(tile_config=...)` | `shared.py` |

### 1b. Shared-infrastructure helpers

| Symbol | Lines | Use count in file | Phase 2 home |
|--------|-------|-------------------|--------------|
| `SyncSchedule` (dataclass) | 119-148 | 11 | `shared.py` |
| `create_range` | 150-169 | 142 | `shared.py` |
| `inflight` | 171-175 | 12 | `shared.py` |
| `duplicate_list_items` | 177-185 | 25 | `shared.py` |
| `count_items` | 187-202 | 8 | `shared.py` |
| `switch_A_B_schedule` | 204-220 | 3 call sites + 1 definition | `shared.py` |
| `removeComments` | 276-281 | called only inside `customMainLoopSchedule` | `dispatch.py` (private) |
| `isNN`/`isNT`/`isTT`/`isTN` (dtype: layout) | 656-669 | 37/21/27/4 | `shared.py` |
| `is16bit`/`is8bit`/`isTF32`/`isMixed` | 672-687 | 26/3/16/2 | `shared.py` |
| `_register_dtype_name` | 64-67 | 3 calls | `shared.py` |
| `_DTYPE_PREDICATE_NAMES` (module dict) | 62 | populated at import time | `shared.py` |
| `_SCHEDULE_REGISTRY` (module list) | 57 | mutated at decoration time | `dispatch.py` |
| `_SCHEDULE_METADATA` (module list) | 59 | mutated at decoration time | `dispatch.py` |
| `_ProbeDataType` (probe-time stub) | 704-710 | called inside `RegisterSchedule._make_probe_kernel` | `dispatch.py` (private) |

### 1c. Per-schedule bodies (38 functions, all gfx950 today)

Sorted by lines (smallest → largest); see `git grep '^def _get_schedule_'` for current
line numbers.

| Function | Body lines | dtype | Tile (M0×M1×DU) | LDSTr-flag |
|----------|-----------:|-------|-----------------|------------|
| `_get_schedule_128x64x64_TF32` | 15 | TF32 | 128×64×64 | F |
| `_get_schedule_224x320x64_16bit` | 44 | 16bit | 224×320×64 | F |
| `_get_schedule_256x256x128_8bit` | 53 | 8bit | 256×256×128 | F |
| `_get_schedule_256x96x64_16bit_DPLB` | 62 | 16bit | 256×96×64 | T |
| `_get_schedule_352x192x64_16bit` | 64 | 16bit | 352×192×64 | F |
| `_get_schedule_128x192x32_TF32` | 66 | TF32 | 128×192×32 | F |
| `_get_schedule_128x192x64_16bit` | 69 | 16bit | 128×192×64 | F |
| `_get_schedule_192x128x64_16bit` | 73 | 16bit | 192×128×64 | F |
| `_get_schedule_128x256x64_16bit` | 79 | 16bit | 128×256×64 | F |
| `_get_schedule_192x128x32_TF32` | 88 | TF32 | 192×128×32 | F |
| `_get_schedule_128x128x32_TF32` | 89 | TF32 | 128×128×32 | F |
| `_get_schedule_64x128x64_TF32` | 92 | TF32 | 64×128×64 | F |
| `_get_schedule_128x160x64_TF32` | 93 | TF32 | 128×160×64 | F |
| `_get_schedule_256x128x32_TF32` | 100 | TF32 | 256×128×32 | F |
| `_get_schedule_256x240x64_16bit` | 104 | 16bit | 256×240×64 | F |
| `_get_schedule_128x224x64_16bit` | 110 | 16bit | 128×224×64 | F |
| `_get_schedule_256x96x64_16bit` | 111 | 16bit | 256×96×64 | F |
| `_get_schedule_240x256x64_16bit` | 118 | 16bit | 240×256×64 | F |
| `_get_schedule_256x208x64_16bit` | 119 | 16bit | 256×208×64 | F |
| `_get_schedule_160x128x64_TF32` | 123 | TF32 | 160×128×64 | F |
| `_get_schedule_208x256x64_16bit` | 123 | 16bit | 208×256×64 | F |
| `_get_schedule_224x256x64_16bit` | 125 | 16bit | 224×256×64 | F |
| `_get_schedule_192x320x64_16bit` | 128 | 16bit | 192×320×64 | F |
| `_get_schedule_160x256x64_16bit` | 129 | 16bit | 160×256×64 | F |
| `_get_schedule_128x128x64_TF32` | 131 | TF32 | 128×128×64 | F |
| `_get_schedule_256x160x64_16bit` | 131 | 16bit | 256×160×64 | F |
| `_get_schedule_320x192x64_16bit` | 137 | 16bit | 320×192×64 | F |
| `_get_schedule_192x256x64_16bit` | 138 | 16bit | 192×256×64 | F |
| `_get_schedule_256x192x64_16bit` | 138 | 16bit | 256×192×64 | F |
| `_get_schedule_256x224x64_16bit` | 139 | 16bit | 256×224×64 | F |
| `_get_schedule_256x256x64_16bit` | 146 | 16bit | 256×256×64 | F |
| `_get_schedule_224x128x64_16bit` | 151 | 16bit | 224×128×64 | F |
| `_get_schedule_128x128x32_TF32_plr1` | 185 | TF32 | 128×128×32 | F |
| `_get_schedule_256x256x32_TF32` | 207 | TF32 | 256×256×32 | F |
| `_get_schedule_96x256x64_16bit` | 210 | 16bit | 96×256×64 | F |
| `_get_schedule_256x192x32_TF32` | 280 | TF32 | 256×192×32 | F |
| `_get_schedule_128x256x32_TF32` | 317 | TF32 | 128×256×32 | F |
| `_get_schedule_192x256x32_TF32` | 330 | TF32 | 192×256×32 | F |

(LDSTr-flag = `dtl_plus_lds_buf` field of `TileConfig`; `True` only for the
single `..._DPLB` variant of 256×96×64.)

Total schedule-body lines: **4817 of 5725** (84%).
Shared infrastructure: **~908 lines** (16%).

---

## 2. Dispatch surface

### How `hasCustomSchedule` chooses a schedule

```
hasCustomSchedule(kernel)
  ├── early-out: not UseCustomMainLoopSchedule → (False, None)
  ├── early-out: not EnableMatrixInstruction   → (False, None)
  ├── early-out: ISA not in {gfx950, gfx1151}  → (False, None)
  ├── early-out: isMixed(kernel)               → (False, None)
  └── for fn in _SCHEDULE_REGISTRY:                     # linear scan, 38 entries
        status, schedule = fn(kernel, useLDSTr, TLDS)
        if status == FOUND:               return (True,  schedule)
        if status == UNSUPPORTED_VARIANT: return (False, None)   # short-circuit
        if status == NO_MATCH:            continue
      return (False, None)
```

The `UNSUPPORTED_VARIANT` short-circuit means **registration order matters**
when two schedules share `(TileConfig, dtype, vector_widths, MatrixInstruction,
MIWaveGroup)` keys. Today only 256×96×64×16bit has two registrations
(`_DPLB` variant differs by `dtl_plus_lds_buf=True`), so the keys are
distinct and order is irrelevant. But the dispatcher's contract is
"first-FOUND-wins, first-UNSUPPORTED-aborts," not "best-match." Phase 2
should preserve this exactly; converting to a hash-table dispatch is a
behavior change.

### What feeds the registry

`@RegisterSchedule(...)` runs at module-import time and appends a
*wrapped* predicate to `_SCHEDULE_REGISTRY`. The wrapper performs five
gates (in order):

1. `kernel["UnrollLoopSwapGlobalReadOrder"]` is False (ULSGRO TODO).
2. `dtype_predicate(kernel)` returns True.
3. `TileConfig` extracted from the kernel matches the registered one
   exactly (frozen-dataclass equality on 11 fields).
4. `[GRVWA, GRVWB, LRVWA, LRVWB]` matches `vector_widths + [vector_widths[2]]`.
5. `MatrixInstruction` and `MIWaveGroup` lists match.

If all gates pass, the inner function is called; its `(bool, ScheduleInfo)`
return is mapped to `FOUND` or `UNSUPPORTED_VARIANT`. Any earlier mismatch
is `NO_MATCH`.

The decorator also calls `_detect_supported_layouts(func)` at registration
time, brute-force probing all (transA, transB, useLDSTr, TLDS, GRVW_A,
GRVW_B) tuples (192 combinations) with a synthetic `_ProbeDataType`
kernel, and emits a `CMSKernelInfo` row into `_SCHEDULE_METADATA` for each
layout the inner function accepts. This is the data behind
`query_cms_kernels` / `get_cms_kernel_info_objects`.

### Natural granularity of a 'schedule file'

The key the registry uses is `(TileConfig, dtype_predicate, vector_widths,
MatrixInstruction, MIWaveGroup)`. The most discriminating axes in today's
data are **dtype** (16bit / 8bit / TF32; isa is uniform) and **MacroTile
shape** (M0×M1×DU). Layouts (NN/NT/TN/TT) and `useLDSTr`/`TLDS` are
*intra-function* branches; splitting on them would proliferate files
unnecessarily and would not match how the existing code or
`_detect_supported_layouts` think about a schedule.

**Recommended granularity:** one file per registered `def
_get_schedule_*`. The function name already encodes the natural unit:
`MT0xMT1xDUx<dtype>[_suffix]`. See §6 for naming.

---

## 3. Boilerplate-vs-body ratio

Sampling `_get_schedule_256x96x64_16bit` (111 lines, function lines 908-1010):

- 1 line `optSchedule = dict()` initializer (boilerplate)
- 1 line `syncCode = []` initializer (boilerplate)
- 1 line `nglshift = nllshift = 0` initializer (boilerplate)
- 4 lines per layout branch header (`if isTN(kernel) and TLDS == 1:`) (boilerplate)
- 3 lines tail (`numMfma = 48`, `opt1 = ScheduleInfo(...)`, `return True, opt1`) (boilerplate)
- 1 line `else: return False, None` (boilerplate)
- **Remaining ≈100 lines:** the actual `optSchedule` dict literal +
  `syncTable` literal — entirely unique data, not copy-pasteable.

**Ratio:** ≤10% structural boilerplate, ≥90% unique optimization data per
schedule. The "boilerplate" is the *shape* of the literal (a dict with
keys `'SYNC' 'GRA' 'GRB' 'LRA0' 'LRB0' 'LRA1' 'LRB1' 'LWSA' 'LWSB' 'LRSA'
'LRSB' 'LCC'`) — but the *values* are bespoke per-schedule timing tables
that are the entire reason the file exists.

**Conclusion:** further helper extraction is *not* the dominant lever.
The refactor's value is in **review locality** (per-schedule diffs stay
small) and **discoverability** (one file per shape), not in code
deduplication. The file split alone already pays for itself.

There is one cluster of repeated *code-shape* worth a Phase 2/3 note:
the alias-with-A/B-swap pattern (3 call sites + 1 definition — see §5) could be a
`@register_swap_alias(target_func, ...)` helper, but extracting it would
trim only ~12 lines total and isn't worth the indirection cost at
current scale.

---

## 4. Cross-schedule shared state

### Symbols every schedule reads

- **rocisa instructions:** `SWaitCnt`, `SBarrier`, `SNop`, `Module` (re-exported via shared.py imports).
- **Dataclasses / utilities defined in this file:**
  - `ScheduleInfo` — every schedule's return type
  - `SyncSchedule` — used by 11 schedules (TF32-heavy)
  - `create_range` — 142 call sites (the most-used helper)
  - `duplicate_list_items` — 25 call sites
  - `count_items` — 8 call sites
  - `inflight` — 12 call sites
  - `switch_A_B_schedule` — 3 call sites + 1 definition
  - `isTN`/`isNT`/`isNN`/`isTT` — 89 call sites total (every schedule branches on layout)
  - `is16bit`/`is8bit`/`isTF32` — used in `RegisterSchedule(dtype_predicate=...)` decorators

Every one of these is a pure utility / dataclass — no mutable globals are
read by schedule bodies. Schedule bodies *do* mutate the kernel dict
(see `kernel["MfmaInitCVgprs"] = True`, `kernel["UsePLRPack"] = True`,
`kernel["SwapGlobalReadOrder"] = True`), but the dict is the caller's
parameter, not file-level state.

### Module-level mutable state (touched at import time only)

- `_SCHEDULE_REGISTRY: list` — appended-to by every `@RegisterSchedule`
- `_SCHEDULE_METADATA: list[CMSKernelInfo]` — appended-to by every `@RegisterSchedule`
- `_DTYPE_PREDICATE_NAMES: dict` — populated by 3 `_register_dtype_name(...)` calls

These are populated transitively when each per-schedule file is imported.
**`__init__.py` must import every per-schedule file** (or use a deliberate
import-side-effect pattern like `pkgutil.walk_packages`) to ensure
registration runs.

### Schedule-to-schedule references (must be hoisted or resolved lazily)

Three sites where one schedule body calls another:

| Caller (line) | Callee | Pattern |
|---------------|--------|---------|
| `_get_schedule_128x192x64_16bit` (3400) | `_get_schedule_224x128x64_16bit` | call + `switch_A_B_schedule` (alias) |
| `_get_schedule_128x64x64_TF32` (5469) | `_get_schedule_64x128x64_TF32` | call + `switch_A_B_schedule` (alias) |
| `_get_schedule_160x128x64_TF32` (5585) | `_get_schedule_128x160x64_TF32` | call + `switch_A_B_schedule` (alias) |

In each case, the caller is the M↔N transpose of the callee with the same
data type. The callee's output dict is fed to `switch_A_B_schedule`, which
re-keys A↔B entries.

#### Load-bearing decorator behavior (callout)

`RegisterSchedule.__call__` (at `CustomSchedule.py:899`) returns the
**raw `func`**, NOT a wrapped closure. The wrapped predicate-checking
closure is only appended to `_SCHEDULE_REGISTRY`; the bare-name binding
in the module namespace stays pointing at the unwrapped body. The 3
alias call sites exploit this — they call the bare schedule function name
directly, bypassing the wrapper's `tile_config` / `vector_widths` /
`MatrixInstruction` gates entirely.

This behavior is load-bearing for the alias-handling decision and
**functionally rules out** the "registry-lookup-by-name" option below.
A registry lookup would invoke the *wrapped* predicate, which re-checks
the caller's `tile_config` against the alias's tile_config — and those
are different tile shapes by definition (that's what makes them aliases).
The wrapped predicate would return `NO_MATCH` and the alias would simply
fail. **Registry lookup by name is FUNCTIONALLY BROKEN and should not
be considered as an option in Phase 2.**

#### Phase 2 options for resolving the 3 aliases

Ranked by complexity / file count / import-edge cost (no winner picked
here — Phase 2 user decision):

1. **Co-locate alias with callee.** Put the 15-line alias function in
   the same file as its 92-line callee (the function the alias delegates
   to). Cost: +15-30 lines per shared file (3 affected files). Benefit:
   no new files, no cross-file import edge, smallest possible change.
   Trade-off: one file then hosts two registered schedules, weakening
   the "one file per registered `_get_schedule_*`" convention from §6
   for those 3 files.

2. **Extract `_<name>_body.py` siblings.** Keep the callee's schedule
   file as the canonical home, *but* hoist its inner schedule body (the
   if/elif tree without the `RegisterSchedule` wrapping) into a sibling
   `_<name>_body.py`. Both the callee's registered file and the alias's
   registered file then import that body builder. Cost: 3 new
   `_body.py` files, 3 unidirectional acyclic import edges. Benefit:
   preserves "one registered schedule per file"; the body-builder has no
   dispatch role and can never be confused with a registered schedule.
   Sketch:

   ```python
   # CustomSchedule/gfx950/_64x128x64_TF32_body.py
   def _build_schedule_64x128x64_TF32(kernel, useLDSTr, TLDS): ...

   # CustomSchedule/gfx950/_64x128x64_TF32.py
   from ._64x128x64_TF32_body import _build_schedule_64x128x64_TF32
   @RegisterSchedule(...)
   def _get_schedule_64x128x64_TF32(kernel, useLDSTr, TLDS):
       return _build_schedule_64x128x64_TF32(kernel, useLDSTr, TLDS)

   # CustomSchedule/gfx950/_128x64x64_TF32.py
   from ._64x128x64_TF32_body import _build_schedule_64x128x64_TF32
   @RegisterSchedule(...)
   def _get_schedule_128x64x64_TF32(kernel, useLDSTr, TLDS):
       valid, opt = _build_schedule_64x128x64_TF32(kernel, useLDSTr, TLDS)
       ...
   ```

3. **Direct cross-file imports** (per-schedule file imports another
   per-schedule file's bare function name). The bead text says "no
   schedule-to-schedule imports" — but if that rule turns out to be
   implementer-imposed rather than bead-mandated, this becomes the
   simplest text-only change (one `from ._other_schedule import
   _get_schedule_other_name` per alias). Cost: 3 new import edges, no
   new files. Trade-off: import edges between *registered* schedules
   (option 2 imports a non-registered body builder).

4. **Registry lookup by name.** ~~Resolve aliases at call time via
   `_lookup_by_name("_get_schedule_64x128x64_TF32")` against
   `_SCHEDULE_REGISTRY`.~~ **FUNCTIONALLY BROKEN** — see callout above.
   The registry holds the wrapper, not the raw body; the wrapper's
   `tile_config` gate fails for aliases by definition. Do not consider.

**Phase 2 staged decision** — see §8.

---

## 5. Import graph plan

Target shape (per-schedule files import only from `shared.py`, `rocisa`,
`Tensile.Common`, and the few aliased body-builders documented in §4):

```
KernelWriter.py
   └─→ Components.CustomSchedule (package)
         └─→ Components.CustomSchedule.dispatch
               ├─→ Components.CustomSchedule.shared
               ├─→ Components.CMSValidator (existing)
               ├─→ Components.ScheduleCapture (existing)
               └─→ Components.CustomSchedule.gfx950.<schedule_file_x38>
                     ├─→ Components.CustomSchedule.shared
                     └─→ rocisa.{code,container,instruction}
```

Imports inside `shared.py`:

```
shared.py
   ├─→ rocisa.code      (Module, KernelBody, ...)
   ├─→ rocisa.container (sgpr, vgpr, ...)
   ├─→ rocisa.instruction (SWaitCnt, SBarrier, ...)
   ├─→ Tensile.Common.IsaVersion
   └─→ Tensile.Utilities.Decorators.Shared.CallableGuard
```

`shared.py` must NOT import `dispatch.py` (`dispatch` depends on
`shared`, not the reverse), and must NOT import any per-schedule file.

`__init__.py` strategy options for ensuring registration runs:

- **Explicit import list** (38 lines, chatty but unambiguous, easy to grep).
- **`pkgutil.iter_modules` + `importlib.import_module`** at package-init.

Recommendation: **explicit import list**. Schedule registration is
load-bearing; an implicit `iter_modules` walk hides registration ordering
and breaks under tools that prune unused imports (lint, AOT compilers).

### Existing external import surface to preserve

`__init__.py` must re-export *at minimum*:

- `customMainLoopSchedule` (KernelWriter)
- `hasCustomSchedule` (Solution + KernelWriter)
- `ScheduleInfo` (test_CustomSchedule, test_mfma_reorder_e2e)
- `RegisterSchedule`, `TileConfig`, `CMSKernelInfo`,
  `_SCHEDULE_METADATA`, `_SCHEDULE_REGISTRY`, `isTN`, `isNT`, `isNN`,
  `isTT`, `is16bit`, `query_cms_kernels`, `get_available_layouts`,
  `get_available_dtypes`, `get_cms_kernel_info_objects`
  (test_CustomSchedule_LayoutAutoDetection)

The two `_SCHEDULE_*` registries are leaked through the public façade
because the layout-autodetection test mutates them (save/restore around
each test). Phase 2 must keep these as module-level lists in
`dispatch.py` and re-export, not deepcopy them, or the test fixture
breaks.

---

## 6. File-naming convention proposal

### Recommendation: **per-(arch, schedule-function-name)**, mirroring the existing function names

```
Tensile/Components/CustomSchedule/
    __init__.py                         # re-exports public API; imports every per-schedule module for side effects
    shared.py                           # ScheduleInfo, SyncSchedule, TileConfig, CMSKernelInfo,
                                        # create_range, inflight, duplicate_list_items, count_items,
                                        # switch_A_B_schedule, isNN/NT/TN/TT, is16bit/8bit/TF32/Mixed,
                                        # _DTYPE_PREDICATE_NAMES, _register_dtype_name
    dispatch.py                         # _SCHEDULE_REGISTRY, _SCHEDULE_METADATA, ScheduleMatchStatus,
                                        # RegisterSchedule, _ProbeDataType, hasCustomSchedule,
                                        # customMainLoopSchedule, query_cms_kernels, get_*_*
    gfx950/
        __init__.py                     # imports every schedule module below
        _256x96x64_16bit.py             # _get_schedule_256x96x64_16bit
        _256x96x64_16bit_DPLB.py        # _get_schedule_256x96x64_16bit_DPLB
        _192x256x64_16bit.py
        ...                             # 38 files total at present
        _128x128x32_TF32.py
        _128x128x32_TF32_plr1.py        # plr1 variant gets its own file (matches existing function name)
        _224x320x64_16bit.py
```

### Why this granularity (not finer, not coarser)

- **Coarser (per-arch + dtype):** would put 30 16bit schedules in one
  file (~3500 lines) and 14 TF32 schedules in another (~2000 lines).
  Doesn't solve the review-locality problem (the bead's primary
  motivation).
- **Per-(arch, dtype, MT):** identical to per-function with current data
  (every registered function has a unique `(MT0, MT1, DU, dtype)` key
  except the two TF32 128×128×32 variants — and their `_plr1` suffix
  already disambiguates).
- **Finer (per-layout):** would split each schedule's `if isTN()...elif
  isNN()...` into separate files, which is wrong — those branches
  share registration metadata, and `_detect_supported_layouts` discovers
  the layout set from a single function.

### Subdirectory by arch

`gfx950/` even though it's the only arch today. Reasons:

1. The dispatcher already gates on `IsaVersion(11,5,1)` (gfx1151), so
   future RDNA schedules will land. A `gfx1151/` sibling slots in
   without restructuring.
2. The `TileConfig.isa` field defaults to `(9,5,0)` and is keyword-only
   in registrations. Per-arch directories make the per-file `isa=`
   override (or its absence) less surprising.

### Preserving function names

All 38 `_get_schedule_*` names appear as snapshot literals in
`test_CustomSchedule_LayoutAutoDetection.py:187-225`. **Function names
must NOT change** in the move. The file-name convention above
deliberately uses a leading underscore (`_256x96x64_16bit.py`) so the
file basename matches the function suffix without colliding with Python
identifier rules; module-level `__name__ == "_256x96x64_16bit"` is
acceptable inside the package.

---

## 7. Test impact estimate

### Files that import from `Tensile.Components.CustomSchedule`

| File | Symbols imported | Impact |
|------|------------------|--------|
| `Tensile/Tests/unit/test_CustomSchedule.py` | `hasCustomSchedule`, `ScheduleInfo` | none if `__init__.py` re-exports both |
| `Tensile/Tests/unit/test_CustomSchedule_LayoutAutoDetection.py` | `hasCustomSchedule`, `ScheduleInfo`, `RegisterSchedule`, `TileConfig`, `CMSKernelInfo`, `_SCHEDULE_METADATA`, `_SCHEDULE_REGISTRY`, `isTN`, `isNT`, `isNN`, `isTT`, `is16bit`, `query_cms_kernels`, `get_available_layouts`, `get_available_dtypes`, `get_cms_kernel_info_objects` | none if all 16 are re-exported via `__init__.py` |
| `Tensile/Tests/unit/test_mfma_reorder_e2e.py` | `ScheduleInfo` | none if re-exported |
| `Tensile/Tests/unit/cms_validation_base.py` | indirect (`from test_CustomSchedule import ...`) | none |
| `Tensile/SolutionStructs/Solution.py` | `hasCustomSchedule` | none if re-exported |
| `Tensile/KernelWriter.py` | `customMainLoopSchedule` | none if re-exported |

### Tests that reference schedule-function names (literal strings)

- `test_CustomSchedule_LayoutAutoDetection.py:187-225` — 38-entry dict
  mapping `_get_schedule_*` names to expected layout sets. **Function
  names must remain stable.** No test changes needed if names are
  preserved.
- `test_ScheduleCapture.py:1224`, `test_mfma_reorder_e2e.py:56,61,84,120`
  — references in docstrings/comments only, not in executed code.

### YAML test references

- `Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling_tf32.yaml:181,257`
  references `_get_schedule_128x128x32_TF32` in comments only.

### Net test-file change estimate

**Zero functional test changes** if the `__init__.py` re-export surface
matches the current public+leaked-internals API. Optionally, the
internal-symbol imports in `test_CustomSchedule_LayoutAutoDetection.py`
could be redirected to `Tensile.Components.CustomSchedule.dispatch`
during Phase 3 cleanup, but that is a follow-up, not a blocker.

### Tests to run as the structural-equivalence gate during Phase 3

(in worktree, with the standard slow-test exclusion):

```bash
pytest -v projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_CustomSchedule.py \
          projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_CustomSchedule_LayoutAutoDetection.py \
          projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_mfma_reorder_e2e.py \
          projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_cms_flag_reconciliation.py \
          projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_dataflow_graph_comparison.py \
          projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_validate_lr_before_mfma_graph.py \
          --ignore=projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_MatrixInstructionConversion.py
```

These six files exercise (a) the public dispatch path, (b) the layout
auto-detection registration probe, (c) the mfmaReorder hot path, (d)
flag-reconciliation behavior tied to CMS gating, and (e) the dataflow
graph comparison consumed by the validator. Together they cover every
externally-visible `CustomSchedule.py` behavior surfaced today.

---

## 8. Phase 2 decisions — RESOLVED

User decisions recorded; Phase 2 is unblocked.

1. **Cross-schedule alias handling (§4) → Option 3: Direct cross-file imports.**
   Each of the 3 alias files imports the callee's bare `_get_schedule_*`
   function name directly from the callee's per-schedule file. No
   `_body.py` siblings; no co-location. The "no schedule-to-schedule
   imports" constraint earlier framed in the memo is dropped; the 3
   import edges stay between *registered* schedule files.

   Implementation note for Phase 2: the alias caller invokes the bare
   (un-wrapped) function name per the load-bearing decorator behavior at
   §4. That contract is unchanged — option 3 simply means the import
   crosses a file boundary instead of relying on module-namespace
   adjacency. Cycle-check: the 3 alias edges are unidirectional
   (smaller-tile alias → larger-tile callee in two cases, transpose in
   the third) and do not form cycles; verify with a grep after the move.

2. **Backward-compat shim or clean break → Clean break.**
   `CustomSchedule.py` is deleted in the same PR as the structural move.
   No `from .CustomSchedule.dispatch import *` shim. All call sites move
   to `from Tensile.Components.CustomSchedule import ...` (the package
   re-exports — see §5). Test/import-site updates land in the same PR.

3. **Per-arch subdirectories now or later? → Per-arch subdir now.**
   Create `gfx950/` immediately. `gfx1151/` slots in later without
   restructuring. The directory shape is final from day one.

4. **Function-name preservation → Confirmed: do not rename.**
   No `_get_schedule_*` function is renamed during the structural move.
   `test_CustomSchedule_LayoutAutoDetection.py:187-225` snapshot, doc
   references, and YAML comments all stay valid.

5. **Schedule-data-format normalization → IGNORE.**
   Do not flag an epic. Do not file a follow-up bead. The
   `SyncSchedule` vs inline `syncTable` divergence is left as-is and is
   not tracked anywhere downstream of this memo.

### Phase 2 implementation directives (derived from above)

- **Per-schedule files** live at
  `Tensile/Components/CustomSchedule/gfx950/_<MT0>x<MT1>x<DU>_<dtype>[_suffix].py`,
  one per registered `_get_schedule_*` (38 files at present).
- **Shared utilities** live at `Tensile/Components/CustomSchedule/shared.py`.
- **Dispatch + decorator + registries** live at
  `Tensile/Components/CustomSchedule/dispatch.py`.
- **`__init__.py`** uses an explicit import list (no `pkgutil` walk).
- **3 alias files** each carry one `from .._<callee> import _get_schedule_<callee>`
  line at the top of the alias file body; the alias function calls the
  imported bare name directly.
- **`CustomSchedule.py` is deleted in the same PR** as the move; all
  external import sites are updated in that PR.
- **No function renames.**

---

## 10. Pre-merge sweep: stale comment-only line citations

`KernelWriter.py` carries comment-only references that point into
`CustomSchedule.py` either by explicit `:line` numeral or by named
function. After the file is split, these become stale (the line
numbers will dangle to the wrong file; the named references will
remain valid but their "see CustomSchedule.py" framing will need a
package-relative path).

Approximate hit set in `KernelWriter.py` (verify with `grep -nE
"CustomSchedule" KernelWriter.py` immediately before the split):

- **Line 3060** — explicit `CustomSchedule.py:489-497` numeral
  reference (will dangle after split). Update to point at
  `CustomSchedule/dispatch.py:<new-line>` once the move lands.
- **Lines ~4425, ~4448, ~5256, ~6109** — name-only references to
  `customMainLoopSchedule` (no line numerals). These remain valid
  text-wise but should be re-checked for accuracy after the move.

This is a **Phase 3 pre-merge sweep**, not a blocker for the structural
move itself. A simple `grep -nRE "CustomSchedule\.py:[0-9]"` across
the codebase before merging will surface every stale `:line` citation
that needs updating.
