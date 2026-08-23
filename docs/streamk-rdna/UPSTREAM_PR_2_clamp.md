# `TENSILE_STREAMK_TILES` larger than a problem's tile count crashes the GPU

**File:** `projects/hipblaslt/tensilelite/src/ContractionSolution.cpp`
**Patch:** `sk_modes/artifacts/sk4_clamp_fix.patch`
**Verified on:** gfx1100 (Radeon RX 7900 XTX), ROCm 7.2

---

## Problem

Setting `TENSILE_STREAMK_TILES` to a value larger than a problem's tile count produces:

```
error: unspecified launch failure
```

Reproduced with `TENSILE_STREAMK_TILES=2048` over a 16-shape TN HHS sweep in which 8 shapes
have fewer than 2048 tiles: **25 launch failures**, run aborted.

## Root cause

The `StreamK == 4` host path applies the debug override without bounding it:

```cpp
// ContractionSolution.cpp, the StreamK==4 branch
uint32_t skTiles = 0;
uint32_t skSplit = 2;
if (overrideTiles > -1)
    skTiles = overrideTiles;          // TENSILE_STREAMK_TILES — no clamp
if (overrideSplit > -1)
    skSplit = overrideSplit;
...
uint32_t totalItems = (tiles - skTiles) + skTiles * skSplit;
```

When `skTiles > tiles`, `(tiles - skTiles)` wraps in `uint32_t`. The subsequent addition
brings it back into a plausible-looking range, so nothing traps — but the result exceeds the
number of real tiles. For a 425-tile problem with `skTiles = 2048`:

```
(425 - 2048) + 2048*2   ->  2473 work items for 425 real tiles
```

Workgroups then pull items off the queue that decode to tiles which do not exist, and index
out of bounds.

**The `StreamK == 3` path already guards against exactly this**, ~170 lines below:

```cpp
// ContractionSolution.cpp:966
skTiles = std::min(skTiles, static_cast<uint32_t>(tiles));
```

The SK4 path is simply missing it.

## Fix

One line, mirroring the SK3 path:

```cpp
skTiles = std::min(skTiles, static_cast<uint32_t>(tiles));
```

placed immediately after the two override assignments.

## Validation

Same command that crashed, TN HHS, 16 shapes (8 of them below the 2048 threshold),
`NumElementsToValidate: -1`:

| `TENSILE_STREAMK_TILES=2048` | before | after |
|---|---|---|
| exit code | 1 | **0** |
| PASSED | 2 | **96** |
| FAILED | 0 | **0** |
| `unspecified launch failure` | **25** | **0** |

**Regression-checked that the clamp does not fire when it should not**, using
`TENSILE_DB=0x40` to read the packed kernel arguments directly:

| | `SKTiles` | `TotalItems` |
|---|---|---|
| unset | 0 | 4224 |
| `=144` | 144 | 4368 |

Both byte-identical to the pre-fix values (`tiles = 4224` for this shape, so the clamp is
correctly inert). A clamp that silently altered working configurations would be worse than
the crash it fixes.

## Before filing: `origin/develop` has TWO instances, this patch covers one

Checked 2026-08-22 against `origin/develop` @ `a9b7332a925` (the fix was developed on
`~/exp/stock`, which is older).

**Good news:** the patch still applies cleanly, landing at line 897 (offset +105) in exactly
the right place — immediately after the `skTiles` / `skSplit` debug overrides in the
`StreamK == 4` branch. No rebasing needed for that site.

**But develop has since added a second unclamped override with the same defect**, inside the
SK5 `effectiveDynamic` block (~lines 930–945):

```cpp
uint32_t sk4_skTiles = 0;
uint32_t sk4_skSplit = 2;
if(overrideTiles > -1)
    sk4_skTiles = overrideTiles;          // no clamp, same as the original bug
...
uint32_t sk4_totalItems = (sk3_tiles - sk4_skTiles) + sk4_skTiles * sk4_skSplit;
```

`(sk3_tiles - sk4_skTiles)` underflows in `uint32_t` for exactly the same reason. **A
complete fix clamps both sites**:

```cpp
sk4_skTiles = std::min(sk4_skTiles, static_cast<uint32_t>(sk3_tiles));
```

Not validated on hardware — `~/exp/stock` predates this code path, so the reproduction in
this document exercises the first site only. The second is identified by inspection and
should be confirmed before or during review.

> **Method note.** `git apply --check` passing is **not** evidence that a patch fixes a bug
> on a target branch — it only proves the context lines still match somewhere. Here it
> matched at a +105 line offset and happened to be correct, but the same check said nothing
> about the second, restructured copy of the defect. Verify by applying to a scratch copy of
> the target branch and reading the surrounding code.

## Blast radius

Essentially nil:

- The clamp only changes behaviour when `TENSILE_STREAMK_TILES` exceeds a problem's tile
  count — a configuration that currently faults.
- `TENSILE_STREAMK_TILES` is a debug override; unset (`-1`) is the default and the code path
  is unchanged in that case.
- `StreamK: 4` ships in no logic file on any architecture, so no shipped kernel is affected.

## Reproduction

```bash
cd projects/hipblaslt/tensilelite
TENSILE_STREAMK_TILES=2048 ./Tensile/bin/Tensile <sk4-config>.yaml <out>
```

Any `StreamK: 4` config where at least one problem has fewer tiles than the environment
value reproduces it. Note this requires the `glc` fix (PR 1) first, since SK4 cannot
otherwise be built on RDNA.

---

## Note for reviewers: related finding, no fix proposed

While investigating, `TENSILE_DB=0x40` showed that **`StreamK: 4` defaults to `SKTiles = 0`
and `TotalItems == tiles`** — i.e. it hands out whole tiles and never splits K. In its
default configuration SK4 is a persistent work-stealing *data-parallel* kernel rather than a
StreamK mode. Enabling splitting via `TENSILE_STREAMK_TILES` changed measured performance by
under 1% even with 12% of tiles split.

That may be intended. Flagging it because the mode's name implies otherwise, and because it
means the only way to exercise SK4's splitting path today is through a debug environment
variable that — before this patch — could crash the GPU.

---

## Re-verified 2026-08-23 against `origin/develop` @ `dab5e862a64`

| check | result |
|---|---|
| `skTiles = std::min(skTiles, ...)` clamps in `ContractionSolution.cpp` | **1** — the SK3 path only |
| `sk4_clamp_fix.patch` applies to develop | **yes, cleanly** |

The SK4 path still lacks the clamp that the SK3 path has, so the `uint32_t` underflow and the
resulting launch failure are still reachable. Patch has not bit-rotted.

Note the second site flagged earlier in this document (`sk4_skTiles` in the SK5
`effectiveDynamic` block) remains uncovered by the patch.
