# Supporting many ROCm / LLVM / gfx combinations

Working notes. Two halves: **§1–§2 survey what rocke does today** for the
LLVM-version axis (read this first — the existing design is better than its
reputation and the recommendations build on it), **§3–§4 are the proposed
strategy** for scaling to more toolchains and more targets.

> **Lifecycle.** This is a design/working note, not reference documentation.
> Fold §1–§2 into `dsl_docs/architecture/` once they stop changing, and delete
> the rest as §4's items land. Related: the decl-table half of this problem is
> tracked in [`f8f6f4_and_decl_validation_plan.md`](f8f6f4_and_decl_validation_plan.md).

> **Toolchain this was surveyed on.** AMD clang 23.0.0git at
> `/opt/rocm/lib/llvm/bin/clang`; comgr reports ROCm 7.13, so rocke detects
> flavor **`llvm23`**. Note `/opt/rocm/.info/version` does **not** exist here —
> `_system_rocm_version()` returns `None` and detection falls through to the
> comgr-vintage step, exactly as designed (§1.3). Probe results in §2.4 are
> LLVM-23 facts; re-run before trusting them on another flavor.
>
> **Superseded in part.** §4's R1 has since shipped as
> `tools/gen_arch_domain.py`, and the §2.4 hand-probes are no longer the data —
> the generated columns under
> `python/rocke/core/arch/data/intrinsic_arch_domain.<flavor>.json` are. All
> three flavors are committed (`llvm20`, `llvm22`, `llvm23`), so the survey
> above now has a machine-checked column behind it. Where §2.4
> and the artifacts disagree, the artifacts win — see
> [`arch_axis_proposal.md`](arch_axis_proposal.md) §3.2 for the rows that changed.

---

## 1. What exists today: the flavor architecture

### 1.1 Three axes collapse to one

rocke does **not** carry a ROCm axis and an LLVM axis separately. `_ROCM_FLAVOR_LADDER`
(`core/lower_llvm.py:195`) maps a ROCm release to the LLVM vintage its comgr
bundles, and everything downstream keys off the resulting *flavor*:

```python
_ROCM_FLAVOR_LADDER = (
    ((7, 13), LLVM_FLAVOR_LLVM23),
    ((7, 2),  LLVM_FLAVOR_LLVM22),
)
```

`_flavor_for_rocm` clamps at both ends and never raises: newer-than-newest
resolves to the newest flavor, older-than-oldest to `LLVM_FLAVORS[0]` (which is
what pre-7.2 actually shipped, not a fallback). Both callers are best-effort
guesses about a host, so raising would be wrong; callers wanting strictness pass
`llvm_flavor=` explicitly, which *is* validated.

The ladder's own comment states the design goal: *"Add a row when a ROCm release
bumps its bundled LLVM; nothing else needs editing."* That is the right contract
and it mostly holds.

### 1.2 The flavor set is an SSOT with an anti-drift lint

```python
LLVM_FLAVORS: Tuple[str, ...] = (LLVM_FLAVOR_LLVM20, LLVM_FLAVOR_LLVM22, LLVM_FLAVOR_LLVM23)
```

Every membership test must go through this tuple. This is enforced, not just
documented — `test_no_hand_rolled_flavor_membership_lists` pins that no private
copy reappears. The comment records why: a hand-rolled tuple is how `llvm23` was
once silently rejected by the C++ path while the Python path accepted it.

### 1.3 Detection is resolved against the *compiling* comgr, not the environment

`_detect_llvm_flavor()` (`:324`) resolves in this order:

1. `$ROCKE_LLVM_FLAVOR` (explicit override; test/dev knob)
2. **the ROCm vintage of the comgr lib that will actually compile the IR**
3. `torch.version.hip` (proxy)
4. `/opt/rocm/.info/version` (proxy)
5. `llvm22` (modern default)

Step 2 is the load-bearing one and the design is subtle in a good way: a torch
wheel bundles its *own* `libamd_comgr.so` whose LLVM follows the wheel's ROCm,
not the system `/opt/rocm`. Emitting IR for the wrong one means comgr either
rejects the declares or silently auto-upgrades them.

The cache (`_resolve_llvm_flavor`, `:367`) is keyed on the **resolved comgr lib
path** rather than resolved-once-forever, so an early torch-less call cannot
lock in the `/opt/rocm` flavor — once torch enters the process the basis changes
and the flavor re-resolves. Every step is exception-wrapped so a misconfigured
environment never breaks import.

### 1.4 Two distinct kinds of cross-flavor divergence, modelled differently

**(a) Datalayout — modelled as a coarser *generation*, not per-flavor.**

`LlvmDatalayoutKind` (`:145`) partitions flavors by the `p8` field only:

| kind | shape | flavors |
|---|---|---|
| `P8_PLAIN` | `p8:128:128` | llvm20 |
| `P8_INDEXED` | `p8:128:128:128:48` | llvm22, llvm23 |

The reason it is a *generation* and not a flavor: llvm22 and llvm23 share the p8
shape but differ in `m:e`, so a datalayout string **cannot narrow to a single
flavor**. `runtime/comgr.py:_assert_ir_flavor_matches_lib` therefore compares
*generations*, which is exactly the strongest claim the evidence supports — and
that is asserted by its own test
(`test_comgr_guard_compares_datalayout_generation_not_flavor`).

The enum member names (`P8_INDEXED`, not `MODERN`) are chosen so they stay true
when a newer generation lands. The partition is exhaustiveness-checked by
`test_datalayout_kinds_partition_every_flavor` — a hand-rolled substitute for
the check a language-level `match` would give for free.

**(b) Intrinsic declare text — modelled as base + per-flavor override dicts.**

```python
self._decls = dict(_INTRINSIC_DECLS)                    # LLVM 20 shapes
if flavor == LLVM_FLAVOR_LLVM22: self._decls.update(_INTRINSIC_DECLS_LLVM22_OVERRIDES)
elif flavor == LLVM_FLAVOR_LLVM23: self._decls.update(_INTRINSIC_DECLS_LLVM23_OVERRIDES)
```

The **key is stable across flavors**, so all ~151 `_need(...)` call sites are
flavor-agnostic. Only the text differs. Five overrides per modern flavor today
(4 fp8/bf8 MFMA operand-width changes + `make.buffer.rsrc.p1`'s `i32`→`i64`).

Why it has to be emitted correctly up front rather than left to auto-upgrade:
**comgr verifies the toplevel `declare` lines *before* running the auto-upgrade
pass.**

### 1.5 The C++ mirror is better factored than the Python original

`cpp/core/lower_llvm/core.cpp:62` holds **one** table with every flavor fact as a
column:

```c
} ROCKE_LL_FLAVOR_LADDER[] = {
    {ROCKE_LLVM_FLAVOR_LLVM20, "llvm20", 0, 0, false},
    {ROCKE_LLVM_FLAVOR_LLVM22, "llvm22", 7, 2, true},
    {ROCKE_LLVM_FLAVOR_LLVM23, "llvm23", 7, 13, true},
};
```

Its comment: *"Everything on this side that names, parses, validates,
enumerates, or version-maps a flavor reads this table, so adding a rung is one
row here plus the enumerator in `lower_llvm.h` — previously it was five
coordinated edits across three files."*

Python spreads the same facts across **five** structures: `LLVM_FLAVORS`,
`_DATALAYOUT_KIND_FLAVORS`, `_P8_MARKERS`, `_ROCM_FLAVOR_LADDER`, and the
if-chain in `_datalayout_for_flavor`. Adding a flavor to Python is still the
five-coordinated-edits shape that the C++ side already escaped.

### 1.6 Enforcement surface

Twelve tests. They fall into three groups, and the sizes are the interesting part:

| group | count | what it proves |
|---|---|---|
| internal consistency | 9 | the partition is exhaustive, markers match constants, no hand-rolled lists, unknown flavors raise, the ladder clamps |
| cross-engine agreement | 2 | `test_cpp_engine_accepts_exactly_the_python_flavor_set`, `test_cpp_backend_path_accepts_every_known_flavor` |
| **toolchain conformance** | **1** | `test_datalayout_matches_hipcc_emitted_ir` — the only one that compares against a real toolchain, and only for the flavor of the `hipcc` on PATH |

### 1.7 How the three gates treat the flavor axis

| gate | flavor coverage | why |
|---|---|---|
| golden (`rocke_representative_ir_sha256.json`) | **all three, from any host** | flavor is an explicit *input* to `lower_case(case, flavor)`; pure Python emission, no toolchain needed. The JSON stores one sub-document per flavor under `flavors:`. |
| emitted-IR validity (`tools/check_ir_validity.py`) | **host flavor only** | it compiles and links; only the installed toolchain can judge |
| byte-identity (`tools/check_byte_identity.py`) | one flavor per run, selected **ambiently** | no `--flavor` argument; both engines read `ROCKE_LLVM_FLAVOR`, so CLAUDE.md's recipe is three separate invocations with the env var set |

---

## 2. Assessment

### 2.1 What is genuinely good

- **The ROCm axis is correctly collapsed.** Three axes are really 1.5, and rocke
  already figured that out.
- **The flavor set is an SSOT with a lint that bans copies of it.** Rare, and
  motivated by a real past bug.
- **Detection tracks the *compiling* comgr, cached on the comgr path.** This is
  the hard part of the problem and it is solved well.
- **The comgr guard claims a generation, not a flavor.** It asserts exactly what
  the evidence supports, no more.
- **Golden is flavor-parameterized**, so all three flavors are regression-locked
  from any host without needing three machines.

### 2.2 Gap G1 — the arch dimension is absent from declare resolution

`_Lowerer.__init__` (`:1385`) consumes `arch` **only** to pick `self._backend`;
it never reaches `self._decls`. And it defaults:

```python
self._backend = backend_for(arch or "gfx950")
```

So intrinsic availability — which is an arch-dependent fact — is resolved on the
flavor axis alone. `_need()` (`:1536`) sets a flag and validates nothing.

This is the structural cause of two already-diagnosed bugs: B9 (`ds.read.tr16.b128`,
fictional on the gfx950 path, real-but-different on the gfx1250 path, one table
row for both) and the tiled-3D arch-routing bug (`f8f6f4_..._plan.md` §4.10).

The `arch or "gfx950"` default makes it worse: a caller that forgets `arch=`
silently gets gfx950 rather than an error. Several such call sites were found and
fixed in commit `8a901dd99b`.

### 2.3 Gap G2 — per-flavor overrides are hand-written strings

Each new flavor costs one hand-written string per drifted declare. It is five
today. Plan A's f8f6f4 format parameterization alone would add 5 formats × 2
shapes = 10 more strings **per flavor**. That is linear manual growth, and manual
growth is precisely what produced B1–B8.

Compounding: the **base** table is the LLVM-20 shape and is known-stale for
llvm20 in at least two places (B6 fp8 MFMA `<2 x i32>`, B7 `make.buffer.rsrc`
`i32`), yet it is the *default* — a flavor with no override dict inherits shapes
nobody has validated.

### 2.4 Gap G3 — only the host's flavor is toolchain-verified, and that is not recorded as data

§1.6 shows 9 of 12 flavor tests prove internal consistency. That is worth having,
but internal consistency between two hand-written tables says nothing about
whether either matches LLVM.

The one conformance test only covers the host's flavor. The other two flavors are
*self-consistent and unvalidated*, and nothing in the repo distinguishes those two
states — a green run reads as "all three flavors fine."

Measured on this host (LLVM 23), by declaring each intrinsic, calling it, and
**linking** (15 probes, 21 ms each). Kept as the argument that motivated R1, not
as data — the generated columns superseded it, and on `llvm20` rows 3 and 4 read
`name_absent` rather than `Cannot select`, because the name does not exist in
LLVM 20 at all:

| intrinsic | gfx942 | gfx950 | gfx1250 |
|---|---|---|---|
| `ds.read.tr16.b64.v4i16` | Cannot select | LINK-OK | Cannot select |
| `ds.read.tr16.b128.v8i16` | **undefined symbol** | **undefined symbol** | **undefined symbol** |
| `ds.load.tr16.b128.v8f16` | Cannot select | Cannot select | LINK-OK |
| `ds.load.tr16.b128.v8bf16` | Cannot select | Cannot select | LINK-OK |
| `ds.read.tr8.b64.v2i32` | Cannot select | LINK-OK | Cannot select |

Full `ds.*.tr*` enumeration on this host: `ds.read.{tr16.b64, tr4.b64, tr6.b96,
tr8.b64}` and `ds.load.{tr16.b128, tr4.b64, tr6.b96, tr8.b64}`. There is **no
`ds.read.tr16.b128` at any width**, and the mnemonic `ds_read_b128_tr_b16` does
not assemble on gfx950 or gfx1250 (`ds_read_b64_tr_b16` / `ds_read_b64_tr_b8`
assemble on gfx950 only) — so B9's "likely a rename to `ds.load.tr16.b128`" is
wrong: that intrinsic is gfx1250-only, already correctly declared under two other
keys, and renaming onto it would trade a link-time undefined symbol for a backend
`report_fatal_error`.

Two methodology notes for anyone building on this:

- `<rocm>/llvm/include/llvm/IR/IntrinsicsAMDGPU.h` — the enumeration source §4.1
  of the f8f6f4 plan proposed — **does not exist on this host**. Any design
  depending on install layout is unreliable.
- The intrinsic name table is **not** in the `opt` binary (it links `libLLVM.so`
  dynamically, so `strings opt` finds one match). It is in
  `<rocm>/lib/llvm/lib/libLLVM.so`, which yields **1484** `llvm.amdgcn.*` names
  on this host. That is a cheap, install-layout-independent enumeration oracle.

### 2.5 Gap G4 — Python carries the five-edit shape the C++ side already escaped

See §1.5. The asymmetry is backwards from what one would expect, and Python is the
side that gets edited first when a flavor is added.

### 2.6 Gap G5 — the declare table is hand-mirrored across engines

`data.cpp`'s arrays are a hand transcription of the Python dicts. The Tier 0
parity test (`tests/core/test_intrinsic_decl_table_parity.py`) exists **because**
of that transcription; it is a symptom, not a cure.

### 2.7 Observation — byte-identity's flavor selection is ambient

`check_byte_identity.py` takes no flavor argument; selection flows through
`ROCKE_LLVM_FLAVOR` into both engines' own resolution. It works, but "which
flavor did that run actually cover" is not in the run's own output or arguments.
(Not verified whether the C++ engine reads the env var directly or inherits it
through the binding — worth confirming before acting.)

---

## 3. The strategy

### 3.1 The cube is not dense — it factorizes

`available(key, arch, flavor)` decomposes into two nearly independent one-dimensional
facts:

```
available(key, arch, flavor)  ≈  arch_domain(key, arch)  ∧  exists(key, flavor)
text(key, flavor)                                            ← already modelled (§1.4b)
```

- **`arch_domain(key)` is a hardware fact**, near-independent of LLVM version. An
  instruction absent from an ISA does not appear because the compiler was
  upgraded. `ds_read_b128_tr_b16` is absent from gfx950 in every LLVM.
- **`text(key, flavor)` is a spelling fact**, near-independent of arch. The fp8
  MFMA operand widening, `make.buffer.rsrc`'s `i64`, the `ds.read.tr16.b64`
  → `.v4i16` rename: all uniform across arches.

Every known data point fits:

| item | arch axis | flavor axis |
|---|---|---|
| `ds.load.tr16.b128.*` | gfx1250 only | — |
| `mfma.f32.16x16x32.bf16` | gfx950 only | — |
| `ds.read.tr16.b128` (B9) | **empty set** → delete | — |
| fp8 MFMA operand width (B6) | — | LLVM20 vs 21+ |
| `make.buffer.rsrc.p1` (B7) | — | LLVM20 vs 21+ |
| `global.atomic.fadd.v2f16` (B10) | — | removed in LLVM23 |
| `ds.read.tr16.b64` rename (B8) | — | cross-version rename |

**Consequence: no 151 × 7 × 3 table is needed.** One 151-row `arch_domain`
(a hardware fact, written once, rarely changing), plus the existing flavor
override mechanism, plus a short list of cases where the factorization fails.

### 3.2 One host can only ever validate one column

This is physics, not a limitation to be engineered away (the f8f6f4 plan §4.6
states it as a caveat; it deserves to be a first-class design constraint). The
strategy must therefore make evidence **accumulate across hosts**:

- every cell carries `verified` / `unvalidated` plus the toolchain identity that
  verified it;
- `unvalidated` may report **skip**, never green;
- CI becomes a host matrix — one host per supported flavor. Rows with no host stay
  explicitly "declared supported, unvalidated".

Golden already has the right shape (`flavors:` sub-documents); copy it.

### 3.3 The support matrix is a declared list, not a cross product

Nobody supports N × M × K. Make the actual list data:

```python
SUPPORTED = [
    ("rocm>=7.0,<7.2",  "llvm20", {"gfx942", "gfx950", ...}),
    ("rocm>=7.2,<7.13", "llvm22", {...}),
    ("rocm>=7.13",      "llvm23", {...}),
]
```

**N and M are choices, not givens.** Every retained old flavor multiplies every
cost above. A stated support window (e.g. newest two flavors plus one LTS) is the
only lever here that reduces cost rather than merely organizing it.

---

## 4. Recommendations, in landing order

R1 and R2 change **zero emitted bytes**, need no golden re-bless, and are fully
verifiable on a host without cmake.

**R1 — generate an `arch_domain` artifact from the toolchain; do not hand-write
it. — SHIPPED** as `tools/gen_arch_domain.py`, gated by
`tests/core/test_arch_domain_artifact.py`.

Uses the link probe (declare + call + `store volatile`, compile **and link**;
prototype validated, §2.4), at **`-O0`** — at `-O3` the optimiser can delete the
call being asked about and the probe reports `ok` for a target that cannot run
it. 148 decls × 7 wired arches, ~15 s wall across cores on `llvm22`. Managed like
golden — a generator tool, a committed artifact, a test asserting regeneration is
a no-op. Depends only on `clang`, never on install layout (§2.4).

Per-cell provenance landed as **one file per flavor, named after it**, rather
than a `verified`/`unvalidated` marker inside a shared file as §3.2 suggested. A
host can only measure its own LLVM, so a shared file means whichever flavor ran
last silently overwrites the others. Naming the flavor deletes the `unvalidated`
state entirely: a flavor with no column is absent, and absence skips rather than
reds.

Catches B3, B4 and B9 immediately, and unlike the module gate does **not** depend
on a corpus case happening to emit them (`f8f6f4_..._plan.md` §4.8 admits that
limitation explicitly).

**R2 — make `_need()` arch-aware; land it warn-only.**

`_need()` is the single chokepoint all ~151 declares pass through:

```python
def _need(self, key: str) -> None:
    if not _available(key, self._arch, self._flavor):
        raise NotImplementedError(f"{key} is not available on {self._arch}")
    self._needs_intrin[key] = True
```

Converts an entire class of link-time undefined symbols and backend
`report_fatal_error` aborts into a build-time error naming the key and the arch.

Two prerequisites: (a) close the `arch or "gfx950"` default (§2.2) — otherwise
the check is decorative; (b) **warn-only first**. Live paths survive today on
luck (B8 rides LLVM auto-upgrade), so promote to `error` only once the warning
stream is quiet. Same discipline as Tier 3.

**R3 — make the support matrix data, and forbid `unvalidated` from reporting green** (§3.2, §3.3).

**R4 — store structure, not strings.**

Replace hand-written declare text with `(base_name, operand_types, ret_type,
arch_domain)` and *render* the declare, computing the mangling suffix from which
parameters are `anyvector`. This is the f8f6f4 plan §4.7 idea, promoted from
"nice-to-have at Plan A step 3" to "the answer to the many-flavors question":

- a new flavor adds a **rule**, not 151 strings;
- `arch_domain` attaches to the same record, so R1's table needs no separate home;
- f8f6f4's 5 formats × 2 shapes becomes two width parameters instead of 10
  hand-written declares per flavor.

Land it safely: write the renderer, assert it reproduces all 151 current strings
**byte-for-byte**, and only then delete the hand-written table. That assertion is
the safety net.

**R5 — generate `data.cpp`'s tables instead of hand-mirroring them** (§2.6).

Once R4 exists, emit both engines' tables from the one record set; commit the
generated file; CI checks regeneration is a no-op. Byte-identity at the declare
layer becomes structural rather than test-enforced, and the Tier 0 parity test
stops being necessary.

Note the f8f6f4 plan §4 rejected a generator — but that objection was specific to
a **`.td`-driven** generator (`.td` is absent from the install package, creating a
source-tree dependency). Generating two tables from one in-repo record set has
neither problem: the artifact is committed and no tool is needed at runtime.

**R6 — collapse Python's five flavor structures into one ladder table**, matching
`ROCKE_LL_FLAVOR_LADDER` (§1.5, §2.5). Mechanical, and it removes the
five-coordinated-edits hazard from the side that gets edited first.

Then promote R2 from warn to error.

### 4.1 Out of scope here but adjacent

- Route each kernel family on arch through exactly **one** seam; make direct
  cross-arch module imports illegal via the existing AST walk in
  `library/tests/test_library_layering.py`. (The `kernels/__init__.py` bug
  happened because there were two ways to obtain a builder.)
- Extend corpus coverage to **op × arch**, and assert every op has a case on every
  arch it claims to support. Complementary to R1: R1 validates declares, the
  corpus validates whole modules — the tiled-3D bug had perfectly valid declares.
