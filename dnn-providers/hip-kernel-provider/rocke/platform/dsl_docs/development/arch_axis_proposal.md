# Proposal: give intrinsic resolution an arch axis

**Status:** S1 implemented (`tools/gen_arch_domain.py`, two committed columns,
gated by `tests/core/test_arch_domain_artifact.py`); S2–S4 still proposed.
Delivery breakdown: [`arch_axis_epic.md`](arch_axis_epic.md).
**Non-specialist summary:** [`arch_axis_brief.md`](arch_axis_brief.md) (one page).
**Scope:** `core/lower_llvm.py` + `cpp/core/lower_llvm/` declare resolution.
**Related:** [`multi_toolchain_strategy.md`](multi_toolchain_strategy.md) (the
wider ROCm/LLVM/gfx problem; this doc is its §4 R1+R2 written out),
[`f8f6f4_and_decl_validation_plan.md`](f8f6f4_and_decl_validation_plan.md)
(bug record B1–B11).

> **Provenance.** The hand-run probes quoted in §3 were taken on AMD clang
> 23.0.0git at `/opt/rocm/lib/llvm/bin/clang` (comgr reports ROCm 7.13, so rocke
> detects flavor `llvm23`). They are a **single-flavor snapshot, kept only as the
> argument that motivated S1** — they are not the data. The data is the generated
> artifact, one column per flavor, under
> `python/rocke/core/arch/data/intrinsic_arch_domain.<flavor>.json`; each column
> records the clang identity that produced it. All three flavors are now
> committed: `llvm20` (ROCm 7.1), `llvm22` (ROCm 7.2) and `llvm23` (ROCm 10.0),
> the last measured in a container rather than on a host. So §3 is reproducible
> end-to-end, and where §3 and the artifacts disagree, the artifacts win. See
> §3.2 for the rows that did not survive.

---

## 1. Summary

rocke resolves an intrinsic `declare` on **one** axis — the LLVM flavor. The
target architecture is never consulted, even though "does this intrinsic exist
on this GPU" is an arch question. The result is that using an intrinsic on an
arch that lacks it is not an error at build time; it is a linker `undefined
symbol`, a backend `Cannot select` abort, or — worst case — a silently wrong
kernel.

This proposes adding the missing axis in the cheapest form that actually works:

- **S1** *(done)* — generate an `arch_domain` table by probing the installed
  toolchain; commit it, gate regeneration as a no-op, exactly like golden. One
  column per flavor, `(key, arch) → status + evidence`; a plain `key → set of
  arches` proved too lossy (§3.2, §4/S1).
- **S2** — make `_need()` consult it. One chokepoint, every decl key, all call sites
  unchanged.
- **S3** — remove the implicit `arch or "gfx950"` fallback that makes a forgotten
  `arch=` silently succeed.
- **S4** — mirror S1/S2 into the C++ engine from the same generated source, so
  byte-identity holds.

S1–S4 change **zero emitted IR bytes** and need no golden re-bless.

---

## 2. Motivation

### 2.1 The defect, in code

`core/lower_llvm.py:1398` — `arch` is consumed to pick a backend and then
dropped:

```python
self._backend = backend_for(arch or "gfx950")   # the only use of `arch`
...
self._decls: Dict[str, str] = dict(_INTRINSIC_DECLS)
if flavor == LLVM_FLAVOR_LLVM22:
    self._decls.update(_INTRINSIC_DECLS_LLVM22_OVERRIDES)
elif flavor == LLVM_FLAVOR_LLVM23:
    self._decls.update(_INTRINSIC_DECLS_LLVM23_OVERRIDES)
```

`self._decls` is a function of `flavor` alone. And `_need()` — the single
chokepoint every one of the ~151 declares passes through (`:1536`) — validates
nothing at all:

```python
def _need(self, key: str) -> None:
    self._needs_intrin[key] = True
```

The C++ engine mirrors the defect faithfully, including the fallback
(`cpp/core/lower_llvm/core.cpp:602`):

```c
if(arch == NULL || strcmp(arch, "gfx950") == 0)
```

So on both sides: a caller who omits `arch=` gets gfx950, and a caller who passes
the right `arch` gets no extra checking for having done so.

### 2.2 Why nothing catches it

The natural place to catch this is a verifier. It does not work:

| oracle | verdict on a declare for a nonexistent intrinsic |
|---|---|
| `opt -passes=verify` | **accepts** |
| `clang -S` (to assembly) | **accepts** — emits a GOT-relative external call (`...@gotpcrel32@lo` + `s_swappc_b64`) |
| **link** | `ld.lld: error: undefined symbol` |

An unknown `llvm.amdgcn.*` name is, to LLVM, just an external function. Only
linking forces the symbol to resolve. **The IR-validity oracle must be a link**,
which is why this cannot be a cheap in-process check.

Second complication: for an intrinsic that *exists* but is unsupported on the
target, the backend calls `report_fatal_error`, which **kills the process**.
Every probe must therefore run in a subprocess. Both constraints shape S1.

### 2.3 The bug record

This is not hypothetical. Every one of these traces to the missing axis:

| bug | what happened |
|---|---|
| **B9** | `ds.read.tr16.b128` declared and reachable from gfx950 paths. The intrinsic **does not exist in any LLVM** (§3). Survived only because its call sites sit behind `tr128: bool = False`. |
| **B3 / B4** | fp4 / fp6 MFMA atoms pointed at intrinsics unavailable on the targeted arch. |
| tiled-3D routing | `kernels/__init__.py` re-exported the gfx950 3D builder directly, so gfx942 requests silently built gfx950 kernels. Fixed in `8a901dd99b`. |
| the `gfx950` default | several builder call sites were dropping `arch=` entirely and getting gfx950 by fallback. Found and fixed in the same commit — *by inspection*, not by any gate. |

The pattern: these are found by a human reading code, or by a kernel failing at
run time. Nothing in the test suite is looking for them.

### 2.4 It gets worse with scale

`BACKEND_REGISTRY` has 8 rows (`gfx908`, `gfx90a`, `gfx942`, `gfx950`, `gfx1151`,
`gfx1201`, `gfx1250`, `gfx11-generic`), 7 of them wired with arch metadata. They
span three programming models (GFX9 / GFX11 / GFX12). The number of intrinsics
that exist on some of these and not others is large and growing — new MFMA/WMMA
shapes, new `ds` transpose forms, new buffer ops. Each addition is a fresh chance
to make the B9 mistake, with nothing to catch it.

---

## 3. The key empirical result

The two axes are not tangled — **the compiler reports them as two distinct
errors**. Declaring an intrinsic, calling it, and linking gives exactly three
outcomes, and they map one-to-one onto the two axes:

| outcome | meaning | axis |
|---|---|---|
| `ld.lld: undefined symbol` | this **name** does not exist in this LLVM | flavor |
| `Cannot select` / fatal | name exists, **this target** can't lower it | **arch** |
| link OK | available here | — |

Three outcomes were enough to *motivate* the split, but not to *record* it: the
shipped generator needs six. Two more say "we did not get an answer" —
`target_unsupported` (this clang cannot target this arch at all, so it has no
opinion) and `toolchain_crash` (asking the question killed the compiler) — plus
`probe_error` for a malformed probe module of our own. Collapsing any of those
into `arch_absent` would condemn an intrinsic on the strength of a missing
answer; on the `llvm20` column that alone would have mis-reported 116 cells.

The flavor axis also turned out not to need the link. The generator answers it
first, and separately, with an `opt -S` round-trip: LLVM resolves a recognised
`llvm.*` declare on parse, while an unrecognised name round-trips verbatim. That
ordering is load-bearing, not tidiness — it is the only stage that can answer a
key whose *codegen* crashes.

Measured by hand on `llvm23` (15 probes, 21 ms each) — superseded, see §3.2:

| intrinsic | gfx942 | gfx950 | gfx1250 |
|---|---|---|---|
| `ds.read.tr16.b64.v4i16` | Cannot select | **LINK-OK** | Cannot select |
| `ds.read.tr16.b128.v8i16` | undefined symbol | undefined symbol | undefined symbol |
| `ds.load.tr16.b128.v8f16` | Cannot select | Cannot select | **LINK-OK** |
| `ds.load.tr16.b128.v8bf16` | Cannot select | Cannot select | **LINK-OK** |
| `ds.read.tr8.b64.v2i32` | Cannot select | **LINK-OK** | Cannot select |

Row 2 is B9: `undefined symbol` on *every* arch — a name that exists nowhere. The
full `ds.*.tr*` set in this LLVM is `ds.read.{tr16.b64, tr4.b64, tr6.b96, tr8.b64}`
and `ds.load.{tr16.b128, tr4.b64, tr6.b96, tr8.b64}`; there is no
`ds.read.tr16.b128` at any width, and `ds_read_b128_tr_b16` does not assemble on
gfx950 or gfx1250.

### 3.1 Consequence: the cube factorizes

Because the two failure modes are independent signals:

```
available(key, arch, flavor)  ≈  arch_domain(key, arch)  ∧  exists(key, flavor)
text(key, flavor)                                          ← already modelled
```

- `arch_domain` is a **hardware** fact: an instruction absent from an ISA does not
  appear because the compiler was upgraded.
- `exists` / `text` are **spelling** facts: the fp8 MFMA operand widening,
  `make.buffer.rsrc`'s `i32`→`i64`, the `ds.read.tr16.b64` → `.v4i16` rename —
  all uniform across arches.

Checked against every known data point:

| item | arch axis | flavor axis |
|---|---|---|
| `ds.load.tr16.b128.*` | gfx1250 only | — |
| `ds.read.tr16.b64.*` | gfx950 only | rename (B8) |
| `mfma.f32.16x16x32.bf16` | gfx950 only | — |
| `ds.read.tr16.b128` (B9) | **empty set** | — |
| fp8 MFMA operand width (B6) | — | LLVM20 vs 21+ |
| `make.buffer.rsrc.p1` (B7) | — | LLVM20 vs 21+ |
| `global.atomic.fadd.v2f16` (B10) | — | removed in LLVM23 |

**No `151 × 7 × 3` table is required.** One 151-row `arch_domain` plus the
existing per-flavor override mechanism covers every case on record. Where the
factorization does fail, the artifact in S1 records the exception per cell — it
is measured, so it cannot be wrong about its own host.

### 3.2 What the generated columns changed

S1 replaced the table above with ~1000 measured cells per flavor, and two of its
five rows did not survive contact with a second flavor:

| row | this doc said | `llvm22` | `llvm20` |
|---|---|---|---|
| `ds.read.tr16.b64` | gfx950 only | gfx950 only ✓ | gfx950 only, but gfx1250 is `target_unsupported` — LLVM 20 cannot target it at all |
| `ds.load.tr16.b128.*` | gfx1250 only | gfx1250 only ✓ | **`name_absent` everywhere** — the name does not exist in LLVM 20 |
| `ds.read.tr16.b128` | empty set (B9) | `name_absent` everywhere ✓ | `name_absent` everywhere ✓ |

This matters for §3.1 rather than for B9. The factorization is argued from
"`arch_domain` is a hardware fact, near-independent of LLVM version" — and
`ds.load.tr16.b128.*` is precisely a key whose *arch* answer is unobtainable on
one flavor because its *name* answer is `name_absent` there. The factorization
still holds, but only because the flavor axis is evaluated first and short-
circuits: `arch_domain` is not independent of flavor, it is **undefined** wherever
the name is absent. A table that stored one arch set per key, with no flavor
column, would have had to invent an answer for those cells.

Two further findings from the sweep, neither visible to a 15-probe sample:

- **Probes must compile at `-O0`.** At `-O3` the IR pipeline can delete the very
  call being asked about, and a module with nothing left to select links happily.
  This produced a false `ok` for a cross-lane intrinsic on a target that does not
  have it — the object contained no such instruction at all. Nine of the 148 keys
  are foldable this way. A false `ok` is the dangerous direction: it silently
  admits a kernel that cannot run.
- **Transient failures must not be recorded as facts.** Under `-j` the probes
  fork enough linkers to occasionally hit the process limit, and one abort was
  otherwise about to be committed as a permanent compiler crash. A serial retry
  pass separates real results (stable under retry) from resource failures.

---

## 4. Solution

### S1 — Generate the `arch_domain` artifact; never hand-write it — **DONE**

`tools/gen_arch_domain.py`, for each decl key × each wired arch, emits a minimal
module, compiles **and links** it in a **subprocess**, and classifies the
outcome. Commit the result; `tests/core/test_arch_domain_artifact.py` asserts
regeneration is a no-op. This is exactly how golden is managed.

Shipped schema — per-cell provenance is the point, per
[`multi_toolchain_strategy.md`](multi_toolchain_strategy.md) §3.2, since one host
can only ever validate one flavor column. It differs from what this doc first
proposed in three ways, each of which was a mistake worth recording:

```json
{
  "schema": "rocke.intrinsic_arch_domain/v1",
  "toolchain": {"flavor": "llvm22", "clang": "AMD clang version 22.0.0git ...",
                "arches": ["gfx11-generic", "..."]},
  "keys": {
    "ds.read.tr16.b64": {
      "gfx950":  {"status": "ok",          "verified_on": "llvm22"},
      "gfx942":  {"status": "arch_absent", "verified_on": "llvm22",
                  "evidence": "Cannot select: intrinsic %llvm.amdgcn.ds.read.tr16.b64"}
    }
  },
  "canonical": {"ds.read.tr16.b64": "llvm.amdgcn.ds.read.tr16.b64.v4i16"}
}
```

- **One file per flavor, named after it**, rather than an `unvalidated` status
  inside a shared file. A host can only measure its own LLVM, so a shared file
  means whichever flavor ran last silently overwrites the others and `--check`
  reds on every machine whose toolchain differs. Naming the flavor makes each
  column independently ownable and independently checkable, and deletes the
  `unvalidated` status entirely: a flavor with no column is absent, not unknown.
- **No `"*"` wildcard row.** Every cell is written out, even when all seven agree.
  The wildcard saves bytes and costs the ability to diff a single cell, which is
  the entire failure mode `--check` exists to localise.
- **Every non-`ok` cell carries its `evidence`** — the compiler's own first
  diagnostic. A status with no diagnostic behind it cannot be acted on by whoever
  hits it, and cannot be distinguished from a bug in the probe.

`canonical` records what LLVM resolved each surviving declare to. Mostly
identical to the declared name; the interesting rows are the overloads that
remangle (`ds.read.tr16.b64` → `...b64.v4i16`) and the legacy names AutoUpgrade
rewrites, because both are places where what rocke *emits* and what the toolchain
*executes* differ.

Rules:
- a flavor with no committed column reports **skip**; it may never report green.
- a key whose domain is empty on every arch (`name_absent`) is a **hard error** —
  that is B9, and it should fail the build the day it is introduced.

The flavor axis is enumerated by an `opt -S` round-trip per key (§3), not by
reading names out of the shipped `libLLVM.so`. The name table *is* in that
library (and notably **not** in the `opt` binary, which links it dynamically), but
keying on it means keying on install layout — the same reason
`<rocm>/llvm/include/llvm/IR/IntrinsicsAMDGPU.h` is unusable here: it does not
exist on every install. `opt -S` asks the toolchain the question directly and
costs one process per key, not per cell.

Cost, measured: 148 keys × 7 arches on `llvm22`, ~15 s wall across cores. Cheap
enough to run in CI, not just on demand — which is why the gate is a test rather
than a nightly.

**Why a generated table and not the module-level validity gate.** The existing
`tools/check_ir_validity.py` only sees intrinsics that some corpus case happens to
emit — a limitation the f8f6f4 plan §4.8 states explicitly. B9 evaded it for
exactly that reason (`tr128=False`). S1 probes the table, not the corpus, so
coverage does not depend on corpus luck.

### S2 — Make `_need()` arch-aware, warn-only first

```python
def _need(self, key: str) -> None:
    status = arch_domain_status(key, self._arch, self._flavor)
    if status is UNAVAILABLE:
        raise NotImplementedError(
            f"intrinsic {key!r} is not available on {self._arch} "
            f"({self._flavor}); see dsl_docs/development/arch_axis_proposal.md"
        )
    self._needs_intrin[key] = True
```

One edit covers all ~151 call sites, because they all already funnel through
here. It converts link-time `undefined symbol` and backend `report_fatal_error`
aborts into a build-time error that names the key and the arch.

**Land it warn-only.** Some live paths survive today on luck (B8 rides LLVM
auto-upgrade), so a hard error on day one would break working builds. Promote to
`raise` only once the warning stream is quiet — the same discipline the plan doc
prescribes for Tier 3.

One cleanup is a prerequisite: `lower_llvm.py:2806` assigns
`self._needs_intrin["global.atomic.fadd.v2f16"] = True` directly, bypassing
`_need()`. A chokepoint with a bypass is not a chokepoint; route it through
`_need()` first.

### S3 — Close the implicit `gfx950` default

`arch or "gfx950"` (Python `:1398`, C++ `:602`) turns "caller forgot `arch=`"
into "caller silently targeted gfx950". Without S3, S2 checks the wrong arch and
the whole exercise is decorative.

Make `arch` required at the `_Lowerer` boundary. Where an outer API must keep a
default for compatibility, make the default explicit and logged rather than an
`or` fallback buried in a constructor. Commit `8a901dd99b` already fixed the
known leaking call sites; this prevents the next ones.

### S4 — Mirror into the C++ engine from the same source

Byte-identity is the #1 invariant, and it covers behaviour, not just bytes: if
Python raises on an unavailable key and C++ emits a declare, the engines have
diverged. So S1's artifact must feed **both**.

Generate the C++ table into `cpp/core/lower_llvm/` from the same record set and
commit the generated file, rather than hand-transcribing it — hand-transcription
is what made the Tier 0 declare-parity test necessary
(`tests/core/test_intrinsic_decl_table_parity.py`). One source, two emitted
tables, CI checks regeneration is a no-op.

### 4.1 Rollout order

1. **S3** — close the default. Standalone, mechanical, no new machinery.
2. **S1** *(done)* — land the generator + artifact + CI no-op check. Nothing
   consumes it yet, so it cannot break anything. B9 shows up here immediately.
3. **S2 (warn)** — wire `_need()` to the artifact, warnings only. Route the one
   bypass through `_need()` first.
4. **S4** — mirror to C++, re-run byte-identity at every flavor.
5. **S2 (error)** — promote once the warning stream is quiet.

### 4.2 What does and does not change

| | effect |
|---|---|
| emitted IR bytes | **none** — S1–S4 add checks and a data file; no emission path changes |
| golden | **no re-bless** |
| byte-identity | must be re-run after S4 (behaviour parity), expected GREEN |
| `KNOWN_BAD` | stays empty; arch-unavailable keys become build errors, not allowlist entries |
| B9 | fixed as a consequence: S1 flags it `name_absent` on every arch |

---

## 5. Alternatives considered

**Do nothing; rely on the module validity gate.** Rejected: it only covers what
the corpus emits, which is how B9 survived. Complementary, not sufficient.

**A dense `(key, arch, flavor)` table.** Rejected: 151 × 7 × 3 hand-maintained
cells, when §3.1 shows the cube factorizes and every known case fits the
factorization.

**Generate declares from LLVM's `.td` files.** Rejected, and the f8f6f4 plan §4
already rejected it for the right reason: `.td` is not in the install package, so
it creates a source-tree dependency. S1 differs — it probes the *installed*
compiler and commits the result, so nothing is needed at build or run time.

**Push the check down to the ISA backend.** Partially right and worth doing
eventually (the `ds_tr16_b128_spec` seam is the model), but it only covers
intrinsics reached through a backend method. `_need()` covers all of them, today.

---

## 6. Open questions

- **Where does `arch` come from for a non-kernel lowering path?** S3 needs an
  answer for every entry point, not just the kernel builders.
- **Forward-declared arches.** `gfx908` has a backend row but no `arch_specs.json`
  metadata. Probe it anyway and record the domain, or exclude it from the
  artifact? Probing is nearly free and the data is useful when it gets wired.
- **`gfx11-generic`.** A generic target's domain is presumably the intersection
  over its family. Worth confirming by probe rather than assuming.
- **Cells where the factorization fails** — a key whose arch domain genuinely
  differs between flavors. None are known; the per-cell artifact represents them
  correctly if they appear, but the S2 lookup signature must allow it (hence
  `arch_domain_status(key, arch, flavor)` above, not `(key, arch)`).

---

## 7. Verification

The §3 hand-probes are superseded; ask the generator instead, which is the only
form of these measurements that is gated:

```bash
cd platform && export ROCKE=$(pwd) PYTHONPATH=$ROCKE/python

# detected flavor and its basis
python3 -c "from rocke.core import lower_llvm as L; \
  print(L._resolve_llvm_flavor(), L._comgr_lib_rocm_version())"

# one cell, one family, or the whole column -- on this host's flavor only
python3 tools/gen_arch_domain.py --only ds.read.tr --arch gfx950 --verbose
python3 tools/gen_arch_domain.py --check
```

The generator refuses to run when `clang --version` and the flavor rocke resolved
disagree: a real measurement filed under the wrong LLVM vintage is worse than no
measurement, because nothing downstream can tell.

Per-cell probe, for reference: emit a module declaring the intrinsic, calling it,
and storing the result `volatile`; run `clang -x ir <f>.ll -target
amdgcn-amd-amdhsa -mcpu=<arch> -nogpulib -O0 -o <out>` — note this **links**,
which `-S` would not, and note the `-O0`, without which the optimiser can delete
the call and the probe reports `ok` for a target that cannot run it (§3.2) — in a
subprocess, and classify stderr by the §3 taxonomy.

Acceptance for the change itself:

- `python -m pytest tests/core/test_arch_domain_artifact.py` green (the structure
  half runs without a toolchain; the regeneration half skips where the flavor has
  no committed column)
- `python tools/check_byte_identity.py` GREEN at each flavor (S4)
- `python tools/check_ir_validity.py` GREEN, `KNOWN_BAD` still empty
- golden unchanged (no re-bless in the diff)

---

## 8. Compliance

No measured kernel performance figures appear in this document; the timings
quoted are build-tool wall-clock, which is in scope per
[`platform/AGENTS.md`](../../AGENTS.md) §Compliance.
