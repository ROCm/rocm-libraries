# Exposed ds_load latency

How the search prices the cycles a WMMA waits for its operand's `ds_load`. Tier 0 and tier 1
price it differently; tier 0 is what ranks candidates. See [score.md](score.md) for how the
exposure enters the per-trip score.

Code: `perf.NestWalk.uses`, `perf.NestWalk.distances`, `perf.NestWalk.exposure` and `perf.sideCost` (tier 0);
`perf.estimate` (tier 1). Constants: `hw.Gfx1250Costs`.

## Tier 0, step by step

Tier 0 works from the loop nest alone, before θ is built. Per operand side (A or B) and per
register choice (PLR, CLR, VG, VA):

### 1. Walk one trip and record each unit's first and last use (`NestWalk.uses`)

One trip of the steady loop is every point of the ordered axes, outermost first; each point is
one WMMA, numbered 0 to `wmmas − 1`. An operand's **unit** is one value of its axes outside the
unit (for A: the `M` and `K` axes not in the unit), plus which of the `VG` sub-units of the unit
axes the point falls in. For each unit the walk records `(first, last)`: the first and last WMMA
that reads it.

A unit the walk comes back to (an A unit read again for a second `No`) spans from its first to
its last read. Revisiting a unit costs no load.

### 2. Each unit's load-to-use distance (`NestWalk.distances`)

Trips are laid end to end: unit `g` of the stream is unit `g mod n` of trip `g div n`, so its
positions are shifted by `wmmas × (g div n)`. The distances are measured over the third trip
(`g = 2n … 3n−1`), where the ring is in steady state.

```text
steps    = PLR                    if CLR (resident operand)
         = min(PLR, VA − 1)       otherwise (VA buffers hold the unit in use + VA−1 read-ahead units)

issued   = first(g − steps)       the load issues `steps` first-uses early (PLR read-ahead)
free     = last(g − VA) + 1       the slot's previous owner is done (WAR); not applied under CLR
if free > first(g): refuse        the slot is not free by the unit's own first use: WRONG SOURCE
issued   = max(issued, free)
distance = first(g) − issued      in WMMAs
```

### 3. The latency each load has to hide

```text
fragRegs     = MIInputPerThread × bytesPerElement / 4
bytesPerLane = clamp(LocalReadVectorWidth × bytesPerElement, 4, 16)
perTile      = ceil(fragRegs × 4 / bytesPerLane)           ds_loads per tile
perBuffer    = unit tiles / VG                             tiles per load burst
waves        = NumThreads / WavefrontSize

latency = max(dsEffectiveLatency, drainLatency(dsLoadLatency(bytesPerLane), perBuffer × perTile, waves))
```

`drainLatency` mirrors StinkyTofu's `computeDynamicDrainLatency` for one burst of `loads`
ds_loads, with `depth = dsQueueDepth` (16):

```text
loads <= 1      latency
loads <= depth  latency + (loads − 1) × waves
otherwise       latency + (depth − 1) × waves + (loads − depth) × waves / 2
```

`dsEffectiveLatency` (300) is calibrated on measured kernels (`hw.calibration`). It is far above
the per-instruction 56 because it prices a load under the steady loop's whole traffic.

### 4. The exposed cycles per trip

`NestWalk.exposure` runs one in-order clock over three trips laid end to end and keeps the
third. WMMA `p` starts at `p × wmmaCycles` plus every wait before it; a unit's load issues at
its issue position on that clock, and the unit's first WMMA waits for it:

```text
wait(unit) = max(0, issueTime + latency − firstUseTime)
exposed    = Σ over the side's units in one trip of wait(unit)
```

`wmmaCycles` is the cycles one WMMA holds the matrix queue (8 for the MXF8 16×16×128 WMMA).
With no waits this is `Σ max(0, latency − distance × wmmaCycles)`. A load issued before an
earlier unit's wait keeps draining through it, so a burst of refills that are all short pays
about one latency. `KNNMM` with A at PLR 3 / VA 4 is that case: its four A refills per K step
land 6 WMMAs before use, 2 WMMAs apart, and the trip exposes `2 × (300 − 48) = 504` cycles, not
`8 × 252`. ATT measured about 590 on that kernel.

The MX scale operands have no exposure in tier 0: they add one VGPR per tile to the side's
register count and count in `C_lds` ([score.md](score.md)), nothing else.

## Worked example: the tuned MXF8 kernel

The tuned kernel of `config/maf/mxf8_loop_model.yaml` (the `test_loopmodel_auto.py` fixture).
The numbers come from running the code, not from hand estimates.

`LoopOrder` is `KKNMNM`; one trip walks `[Ki 2, No 2, Mo 2, Ni 5, Mi 4]`, outermost first, so
2·2·2·5·4 = 160 WMMAs, numbered 0–159.

- A's unit is `Mi 4`: 4 tiles, and 4 A units per trip, one per (Ki, Mo).
- B's unit is `Ni 5`: 5 tiles, and 4 B units per trip, one per (Ki, No).

### Step 1: first and last use

| Unit | A (first, last) | B (first, last) |
|---|---|---|
| u0 | (0, 59) | (0, 39) |
| u1 | (20, 79) | (40, 79) |
| u2 | (80, 139) | (80, 119) |
| u3 | (100, 159) | (120, 159) |

An A unit spans 60 WMMAs because it is read again for the second `No`; B units are read once.

### Step 2: distances

A with PLR 2, VA 3 (steps = min(2, 3 − 1) = 2, 3 buffers), trip 2:

| Unit | First | Read-ahead: first use of g−2 | Free: last use of g−3, +1 | Issued | Distance |
|---|---|---|---|---|---|
| u0 | 320 | u2 of trip 1: 240 | u1 of trip 1: 239 + 1 = 240 | 240 | **80** |
| u1 | 340 | u3 of trip 1: 260 | u2 of trip 1: 299 + 1 = 300 | 300 | **40** |
| u2 | 400 | u0 of trip 2: 320 | u3 of trip 1: 319 + 1 = 320 | 320 | **80** |
| u3 | 420 | u1 of trip 2: 340 | u0 of trip 2: 379 + 1 = 380 | 380 | **40** |

On u1 and u3 the slot frees later than PLR would issue the load, so the write-after-read
dependence, not PLR, limits the distance. B with PLR 1, VA 2 gives [40, 40, 40, 40] the same way.

### Step 3: latency

- One A tile is 16 VGPRs and a `ds_load_b128` moves 16 bytes per lane: 4 loads per tile, so
  4 × 4 = 16 loads per A unit.
- 128 threads / 32 = 4 waves.
- `drainLatency` = 56 + (16 − 1) × 4 = 116, so latency = max(300, 116) = **300**. The calibrated
  constant dominates; the burst's own drain does not matter here.

### Step 4: exposure

| Choice | Distances (WMMAs) | Per unit | Exposed per trip |
|---|---|---|---|
| A: PLR 2, VA 3 (tuned) | 80, 40, 80, 40 | max(0, 300 − 640) = 0; max(0, 300 − 320) = 0 | **0** |
| B: PLR 1, VA 2 (tuned) | 40, 40, 40, 40 | max(0, 300 − 320) = 0 | **0** |
| A: PLR 1, VA 2 | 20, 20, 20, 20 | 300 − 160 = 140 | **560** |
| A: PLR 1, VG 2, VA 3 | refused | | |

The last row is refused: a slot would be refilled before its previous unit's last read (WRONG
SOURCE).

## Tier 1 (`perf.estimate`)

Tier 1 prices θ analytically and reports the result; it does not rank. Per read operand:

```text
units     = presence tiles / (prefetch-unit tiles / VG)
burst     = ds_loads / units
steps     = min(PLR, VA − 1) if VA > 1 else 0
distance  = steps × (wmmas / units) × wmmaCycles             an average: no walk, no WAR limit
rho       = min(0.95, C_lds / max(C_wmma, C_lds))
queueing  = rho / (2 (1 − rho)) × burst × dsCyclesPerLoad
effective = drainLatency(dsLoadLatency, burst, waves) + queueing
exposed   = max(0, effective − distance) × units              0 for the MX scales
```

Unlike tier 0 it uses the per-instruction ds latency (56) plus an M/D/1-style queueing term
instead of the calibrated 300, and an average distance instead of the walked per-unit ones.

## What the tier-0 calculation leaves out

- **The latency is one constant.** Every unit is charged 300, whatever the LDS traffic.
  `sideCost` takes `rho` but does not use it; only tier 1 adds queueing.
- **Distance is whole WMMAs × `wmmaCycles`**, as if the matrix queue never stalls and nothing
  else delays the loads.
- **Each operand is priced alone.** One operand's units share one clock, so a burst of its
  refills pays about one latency, but A's and B's waits are summed as if neither delayed the
  other, and their loads do not compete for the LDS return queue.
- **The MX scale loads have no exposure.** They count only in `C_lds`.
- **TDM is not modelled.** Nothing prices a `ds_load` waiting for its `tensor_load_to_lds` and
  barrier, which the measurements point to as the real signal.
- **Exposure is all or nothing.** With distances of 40 and 80 WMMAs, any latency up to 320 gives
  the same 0, which is why many candidates tie at the floor.
