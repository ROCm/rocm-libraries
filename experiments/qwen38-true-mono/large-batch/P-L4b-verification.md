# P-L4b (WP-L2d MoE rebuild): Verification

This is the Verification section of `P-L4b.md`. It is kept in its own file because `P-L4b.md` is still being edited by
the MoE agent (round 2, toward 0.85), which other agents must not touch. Ratios only. Raw data and job logs are in
`hackathon-prep/private-perf/qwen38-true-mono/large-batch/w10-raw/`.

Independent verifier: W10 reverify-LB, 2026-11-02. What was checked: the G-MoE gate (DESIGN-LB 15.5 condition 1), which
requires `moe_lb2` to be at most 1.00 of FlyTape's experts + combine in every cell, on both nodes, with the per-cell
dispatcher pick at TP1 r16.

Inputs:
- `P-L4b.md` sections 0-5;
- the raw `p-l4b-raw/lb2-f8`, `lb2-f9` and `lb2-c1` (`pl4b.json`, `job.log`, `ft-*`);
- the harness (`p-l4b/pl4b_bench.py`, sha `dbed249c...`; `pl4b_report.py`; `../p-l4/pl4_report.py`);
- FlyTape's phase wiring (`flytape/models/qwen4/model.py`, `bodies_moe.py`).

Hardware reruns: MI355X, on frozen copies of the builder's stages (`w10-lb2-rocke` = `lb2-rocke`, `w10-lb2-h` =
`lb2-h`). The verifier's harness is `hackathon-prep/qwen38/truemono-lb/w10-reverify/` (`w10_pl4b.py`, `job.sh`).
`moe_lb2.py` and the builder's harness were not edited.

**Verdict: CONFIRMED, with one process finding.** The final timed binaries were compiled by the q38 container's COMGR
(V2.1). The verifier re-timed the decisive cells on venv-COMGR code objects, against same-job FlyTape brackets, on both
nodes. The gate holds there with a larger margin at TP1 r16: B5 alone is under 1.00 on venv binaries.

## V1. What holds (re-derived from the raw data)

1. **Section 3 tables.** `pl4b_report.py` on `lb2-f8` and `lb2-f9` gives every row: the best arm, best / FT e+c with its
   pre..post range, B5 / FT, line, P-L4, the whole MoE and the floor. The summary's per-cell ranges match. The 0.85
   counts are right: 8 of 18 cells on n01-09, 3 of 18 on n07-17. The largest upper bracket is 0.987 (TP1 fp8 r16,
   n07-17).
2. **Numerics.** In both final jobs, all 20 cells say `numerics PASS`, and no required mutant was missed. Only `bar` (20
   cells) and `flow` (6 / 3) are missed, and both are report-only. The timing error words are 0. The A-vs-A (`lb2b`)
   and recompiled (`lb2rc`) controls stay within 1.5 %.
3. **The comparison is like for like.**
   - FlyTape's tiled `moe_router` phase runs the router GEMM, the top-k into `route`, and (W4A4) the x quantization to
     `x4` / `x4s`.
   - FlyTape's expert phases (`moe_up` / `moe_down` at TP1, the fused `moe_expert` at TP2) rebuild per-expert token lists
     in LDS from `route`, as `moe_lb2` does. Its `moe_combine` matches ours.
   - `moe_lb2`'s timed path starts from route pairs and pre-quantized x (`x`, `xs`), so experts + combine is matched
     phase for phase.
   - The per-layer FlyTape figure (sum over the `gdn` / `qsa` / `gdn_ple` MoE phases, divided by 48) is consistent:
     about 37 per layer in each family at TP1 B8.
4. **Cold data.** 8 layers x 513 experts (10.0 / 5.0 GiB). The touched expert bytes are about 200 MB per layer at TP1
   r8, so one graph replay reads about 6x MALL.
5. **FlyTape inputs.** `bench_qwen4_full.py` draws a random prompt per row (seed 7) with random weights, so its routing
   should be close to the uniform routing timed here. FlyTape's unique-expert count per layer is not observable from
   the trace, so this stays an assumption, not a measurement.

## V2. Findings

1. **The timed binaries of `lb2-f8` / `lb2-f9` / `lb2-c1` were built by the container COMGR.**
   - The jobs ran `--mode container` and compiled in-job: "compiled 369 kernels". The logged COMGR is
     `/opt/venv/.../_rocm_sdk_core/lib/libamd_comgr.so.3`, the ROCm 10 / torch 2.13 image.
   - These are 512-thread kernels, the class alola rules 217 / 219 say must never be timed from a container build.
   - The in-job numerics ran on those same binaries and passed, so the reported times are of correct code.
   - The ISA statements in section 5 do not describe these binaries: "no VGPR scratch on the cluster compiler" and the
     `v_writelane` / `v_readlane` counts come from `isa350.sh`, i.e. the venv COMGR.
   - `pl4b_bench.py` has no prebuilt or strict mode. The verifier's `w10_pl4b.py` adds one without editing it
     (`W10_SAVE` in a venv job, then `W10_LOAD` + `W10_STRICT=1` in the container job). The MoE agent should adopt it
     (or an equivalent) for every further timed MoE job.
2. **The ledger's `w` row is not a WIN in every cell-job.** `lb2-f9` (n07-17), TP1 fp4 r16: `w` / B5 is 0.975 (launch)
   and 0.979 (gs), both TIE. The other three container cell-jobs are W (0.961-0.973). On venv binaries `w` is W in all 6
   TP1 r16 cell-jobs (0.971-0.977; V3).
3. **The source drifted after the gate runs.**
   - `moe_lb2.py` changed after f8 / f9 (sha `ec2a214d...` then, `7a610aea...` now; `lb2-rocke` was re-staged at 02:34 UTC
     on 11-02). No copy of the `ec2a` source remains locally, so the gate binaries cannot be rebuilt.
   - V3 re-times the current source with the same B5 knob string, so the gate now stands for the kernel WP-L3 will
     compose.
4. **Section 6 ("What remains") contradicts section 7.** Section 6 says `combine=fused_next`, `entry=router` and the
   tuner lookback are not built or not done; section 7 built and ran them (W7-STATUS noted this too). For the MoE agent
   to fix.
5. **Neighbours (DESIGN 8.1).** `lb2-f8` lists 4+ jobs of ours on n01-09 but does not flag it against 8.1's
   one-neighbour rule. The verifier's jobs also exceeded it (V3). Same-job FlyTape brackets and paired controls keep the
   ratios usable, but the job-to-job spread is larger than the in-job controls: at TP2 fp8 r32, best / FT over four jobs
   is 0.937-0.984.

## V3. Hardware recheck of the decisive cells (venv-COMGR code objects, strict)

Setup:
- Cells: TP1 r16 (the dispatcher cell, the only one where B5 alone was over 1.00) and TP2 r32 (the closest 2-GPU cell,
  0.93-0.98), fp8 and fp4.
- Arms: B5, `b4`, `w`, `p1`, plus `lb2b` / `lb2rc`.
- 24 shuffled rounds (the builder used 8), numerics and every must-fail mutant first.
- FlyTape TP1 B16 and TP2 B32 traced before and after in the same container job.

Jobs:
1. `w10-v4b-sv` (68245312, venv, n01-09): built 88 code objects with the torch 2.12 / ROCm 7.2 COMGR and saved them by
   kernel name (stage `w10-l4hsaco`).
2. `w10-v4b-n07` (68245343, n07-17, buses 05 + 85) and `w10-v4b-n01` (68245397, n01-09, buses 85 + 95): container jobs
   that loaded 88 / 88 objects with `W10_STRICT=1` and compiled 0. The bench GPUs held 2400 MHz in 59 of 66 samples of
   the timed window; the rest were idle samples.

| cell | `w` / B5, venv job (launch / gs) | n07-17 best / FT e+c, launch (pre..post) | n07-17 B5 / FT | n01-09 best / FT e+c, launch (pre..post) | n01-09 B5 / FT |
|---|---|---|---|---|---|
| tp1 fp8 r16 | 0.976 / 0.972 W | `w` 0.974 (0.973..0.975) | 0.998 | `w` 0.964 (0.964..0.965) | 0.990 |
| tp1 fp4 r16 | 0.977 / 0.972 W | `w` 0.970 (0.969..0.970) | 0.999 | `w` 0.963 (0.962..0.964) | 0.989 |
| tp2 fp8 r32 | `p1` 0.992 / 0.993 T | `p1` 0.984 (0.983..0.984) | 0.993 | `p1` 0.937 (0.933..0.942) | 0.945 |
| tp2 fp4 r32 | `p1` 0.995 / 0.986 T | `p1` 0.965 (0.965..0.966) | 0.975 | `p1` 0.920 (0.916..0.924) | 0.929 |

What V3 shows:
- **Numerics:** all 4 cells PASS in all three jobs, with no required mutant missed. `flowrel` and `flow0` were caught in
  every cell.
- **Controls:** `lb2b` / `lb2rc` 0.995-1.004.
- **`w` / B5 in the container jobs:** 0.971-0.975 W (n07-17) and 0.972-0.974 W (n01-09).
- **The gs metric** agrees with launch within 0.016.
- **The gate is met on both nodes.** On venv binaries B5 alone is 0.989-0.999 of FlyTape at TP1 r16, so the gate no
  longer depends on the per-cell pick there; the pick still wins about 2.5 %.
- **The thinnest cell is TP2 fp8 r32 on n07-17:** best 0.984 (0.988 gs), B5 0.993.
