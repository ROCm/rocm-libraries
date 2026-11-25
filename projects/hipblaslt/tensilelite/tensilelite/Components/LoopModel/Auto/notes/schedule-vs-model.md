# When perf disagrees with the score, suspect the schedule too

A benchmark that disagrees with the score model has three suspects, not one:

1. **The score model** is missing or mis-pricing a cost.
2. **The emitted schedule** is worse than the parameters allow: an instruction placed too late or a
   wait placed too early, with no data dependency forcing it.
3. **The measurement**: noise, a capture artifact, or the wrong kernel.

Do not tune the model to fit a slow kernel until you have ruled out (2). A score that predicts the
kernel the parameters *permit* is doing its job even when the emitter misses it, and bending it
to match the emitter teaches it the emitter's bugs. Equally, do not trust a fast benchmark number
as proof that the model is right: check that the speed comes from where the model says it does.

## What to do before blaming either one

- **Trace it.** One ATT capture says which wait the time goes to (`s_wait_dscnt`,
  `s_wait_tensorcnt`, barriers). Compare that with the model's term for the same cost, per trip.
- **Find the dependency window.** For the stalling wait, find the producer it retires and
  the earliest legal issue point (write-after-read: after the last read of what the producer
  overwrites). Then find the latest legal wait point (read-after-write: before the first read of
  what it produces). The `<GIR: ...>` tags on emitted instructions name the operand and generation,
  for example `B.shared[gen1[v+2]]`, so both ends can be read off the `.s`.
- **Compare used against allowed.** If the emitted issue-to-wait distance is much smaller than
  the window, the schedule is the suspect, not the model.
- **Prove it by re-scheduling.** Move only the instructions the window says can move, keep every
  data dependency, re-derive the counter waits, reassemble with the toolchain's exact flags (check
  that the unmodified `.s` reassembles byte-identical first), validate, and benchmark the edited
  kernel interleaved with the original on one card. Re-trace it and confirm the stall you targeted
  is the one that went away.

## Worked example: TDM starvation in the MXF8 8×10 top 50 (2026-11-24)

The 50 best tier-0 kernels of `config/maf/mxf8_loop_model.yaml` that emit (tier 0 1377–1789) were
benchmarked and ATT-traced on one gfx1250 card (CU 5). ATT cycles per trip rank them like the
benchmark (Spearman −0.99), so the traces measure what decides speed. Tier 0 ranked them poorly
(Spearman +0.26 against measured cycles per trip).

They split into two groups:

| | healthy (36) | starved (14) |
|---|---|---|
| benchmark | 7.8–8.85 PF | 5.7–7.4 PF |
| measured cycles per trip | 1370–1778 | 2056–3345 |
| `s_wait_tensorcnt` stall per trip | 0 | 400–1100 |
| tier 0 | 1377–1789 | 1465–1789 |

Tier 0 charges `tdmExposed` = 0 to all 14. The tempting conclusion is that `tdmExposed` is wrong.
The dependency windows say otherwise. Distances are in WMMAs (8 cycles each), for the TDM load each
kernel's binding tensor wait retires:

| | healthy | starved |
|---|---|---|
| dependency window (write-after-read bound to read-after-write bound) | 174–215, median 191 | 66–302, median 106 |
| issue-to-wait distance actually emitted | 147–203, median 177 | 12–70, median 48 |
| issue later than the write-after-read bound allows | median 6 | median 29, up to 160 |
| wait earlier than the read-after-write bound requires | median 5 | median 32, up to 98 |

The A and B data alone leave every kernel, starved ones included, at least about 160 WMMAs
(about 1280 cycles) of window. The TDM load needs about 140. The starved kernels lose the window
to three scheduling effects:

- **An early read of the fused scale (10 of 14).** With `TDMFuse: 1` a scale rides in the same
  `tensor_load` as its operand (`MXSB.global+B.global`), so one tensor counter covers both.
  StinkyTofu hoists the scale's next-trip `ds_load_b32` as soon as its 2-deep register ring frees.
  In kernel 46 the LoopIR places it at WMMA 144 and the emitted code at WMMA 80. That drags the
  whole fused load's wait 50–100 WMMAs earlier, although B itself is not read until 130–160 WMMAs
  after the issue.
- **The load is not hoisted to its write-after-read bound** (4 kernels): it issues 37–160 WMMAs
  after its buffer is free.
- **The wait and its fence sit earlier than the first reader** (9 kernels, by 26–98 WMMAs). In
  kernel 46 the fence is at WMMA 48 and the first reader of its slots at WMMA 80.

### The experiment

Kernel 46 (WFM4 WFN5, PLR 3/3, VA 4/4, the slowest at 5.72 PF) was edited by hand in two ways:

- **v1:** the RAW fence (`s_wait_tensorcnt 1` and its barrier) moved from WMMA 48 to just before the
  first reader of its slots, in both copies of the two-copy body.
- **v2:** also sinks the two next-trip MXSB reads to WMMA 130. `v16` already holds their address
  there. They now sit just after the fence, where the B next-trip reads start.

Nothing else changed, and the edit kept every data dependency:

- No instruction was added except one `s_wait_alu`.
- A three-iteration simulation of the DS counter shows that every `s_wait_dscnt` still retires
  what it retired before. Each sunk read is retired before its first consumer, so no count had to
  change.
- The first wait after the loop is `s_wait_dscnt 0`, so the loop exit is covered.
- The unmodified `.s` reassembles byte-identical with `-mcode-object-version=4`.

| | PF, mean of 6 interleaved runs | 4096-element validation | `tensorcnt` stall per trip | measured cycles per trip |
|---|---|---|---|---|
| original | 5.81 | PASSED | 549 | 3345 |
| v1: fence at first reader | 6.35 (+9.3%) | PASSED | | |
| v2: fence and scale reads sunk | 6.40 (+10.2%) | PASSED | 0 | 2773 |

The runs do not overlap (original 5.80–5.82, v2 6.39–6.41). The trip shrank by 572 cycles, the
tensor stall that disappeared. The other stall buckets did not move.

So the stall was the schedule's, not the score's. Had `tdmExposed` been changed to charge these
kernels, the model would have learned to avoid parameter sets that are fine, and missed that the
fix belongs in StinkyTofu:

- Do not hoist a fused scale read past the point where it tightens a TDM wait, or split the
  scales out of `TDMFuse` when it would.
- Anchor the TDM wait at the latest legal barrier.

Kernel 46 is still slower than the healthy kernels. About 1040 cycles per trip of
`s_wait_dscnt` remain, a different problem that this edit did not touch.
