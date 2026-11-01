# In-kernel all-reduce over IPC peer mailboxes: case study (gfx950, TP2 / TP3 / TP4)

Measured scope: the rocKE in-kernel all-reduce of this directory (`bench_ccl.py bench`) against
RCCL `all_reduce` through `torch.distributed` (backend `nccl`, every `NCCL_PROTO` setting) and
against the aiter custom all-reduce (the TP all-reduce of the ATOM serving stack), on the same
GPUs in the same processes, for rows 1-128 x hidden 2560 / 10240 x bf16 / f32 at TP2, TP3 and
TP4 on one 8-GPU gfx950 server (every GPU pair joined by one xGMI link). Relative results only;
the absolute device times live in an access-controlled internal record.

## Summary

At these measured configurations (one gfx950 node, uncached arena), with RCCL's best protocol
per size as the baseline:

* rocKE's in-kernel all-reduce chain, in which each wave waits only for its own unit (the best
  case), costs 0.28-0.95 (TP2), 0.20-0.71 (TP3) and 0.17-0.85 (TP4) of the fastest RCCL form
  (a captured graph with the best of the default, LL, LL128 and Simple protocols).
* With a grid-wide barrier after every collective (`sync="grid"`, the shape of a model layer
  that needs the whole vector) it costs 0.27-1.04 (TP2), 0.20-0.79 (TP3) and 0.21-1.00 (TP4) of
  RCCL best; restricted to 32 CTAs, 0.27-1.09, 0.21-0.85 and 0.21-1.12. It is not faster than
  RCCL at 2.5-5 MiB per rank at TP2 (1.00-1.04) and at 5 MiB at TP4 (1.00).
* Launched as one kernel per collective (no graph), rocKE costs 0.75-1.04 (TP2), 0.53-0.79 (TP3)
  and 0.51-1.01 (TP4) of RCCL's best captured graph; up to 20 KiB per rank at TP2 it only ties
  RCCL's LL graph (0.95-1.02), because both are bound by per-kernel launch cost there.
* Against the aiter custom all-reduce (a captured graph, ATOM's path; container job), the
  best-case chain costs 0.41-0.98 (TP2) and 0.47-0.96 (TP4); the grid-synced chain costs
  0.51-0.99 up to 80 KiB per rank and **1.13-1.21 (TP2) / 1.13-1.35 (TP4) from 160 KiB**: aiter's
  custom all-reduce is faster than rocKE's layer-shaped exchange at the large-batch sizes. One
  rocKE launch per collective costs 1.13-3.20 of aiter's graph. aiter's graph itself costs
  0.25-0.87 of RCCL's best.
* Every arm passed its numeric gate in the measured runs (rocKE bitwise against the rank-order
  f32 reference on every rank; RCCL and aiter against a float64 sum).

The earlier version of this study compared against RCCL's default protocol only, which switches
from LL to Simple at 40 KiB per rank at TP2 (and is also slower than LL at some TP3 / TP4 sizes),
and reported 0.09-0.32 for rows 1-16 x 2560; against the best protocol those ratios are
0.28-0.33. The "chain / RCCL default protocol" column below keeps that comparison visible.

## Method

* One process per GPU (`torch.distributed.run`), one rocKE `PeerGroup` per rank (uncached
  arena, IPC-mapped peers), RCCL (and aiter) on the same devices and streams.
* Arms, interleaved in a new shuffled order every round (12 rounds), with a gloo barrier and a
  device synchronize before every arm:
  * rocKE persistent chain: one launch runs 96 dependent all-reduces (the input of collective
    i + 1 is the scaled output of collective i). Per-collective latency = (latest CTA end -
    latest CTA end of the first collective) / 95 from `s_memrealtime` stamps, so launch
    overhead and cross-rank launch skew are excluded. Forms: `line` / `pair` x one-shot /
    two-shot, plus the lever arms. Each wave waits only for its own unit from every rank: a
    best case, with up to 256 CTAs.
  * rocKE grid-synced chain (`sync="grid"`): the same chain with a grid-wide barrier (arena
    counter) after every collective, so collective i + 1 starts only when every CTA of the rank
    has collective i. Also run on 32 CTAs.
  * rocKE one collective per launch: 64 back-to-back launches, torch events / 64 (launch cost
    included, no HIP graph), with the launch header off; plus the default model build
    (`auto_allreduce_spec`, launch header on).
  * RCCL: one communicator per `NCCL_PROTO` value (default, LL, LL128, Simple; the variable is
    set while the communicator is created, and RCCL logs "NCCL_PROTO set by environment" for
    each), each with a HIP graph of 96 `all_reduce` calls (per call = graph time / 96) and 64
    eager calls; the default communicator also with a scaling kernel between graph calls.
    "RCCL best" = the fastest of all of these per size (a captured graph at every size).
  * aiter (container job only): `CustomAllreduce` on the gloo group, a HIP graph of 96
    `custom_all_reduce` calls (registered-input capture, as under ATOM's graphs) and eager calls.
  * A/A control: a second instance of the `line` one-shot chain arm (same binary).
* Numerics before timing, every rank, every size: each rocKE arm's 96-collective output is
  compared bit for bit with the numpy rank-order f32 reference; RCCL's and aiter's results are
  checked against a float64 host sum (relative error < 2e-2 bf16, 1e-5 f32). All arms passed.
* Ratio = rank-max of the per-rank medians, rocKE / baseline (< 1: rocKE faster at that
  configuration). Chain arms use device stamps, every other arm wall time per call.

Caveat on the comparison: the chain metrics are device-side per-collective latencies inside one
persistent kernel (what an in-kernel exchange costs a model's dependent layer chain); RCCL and
aiter have no in-kernel form, so their best host-launched form (a captured graph) is the
baseline. The one-collective-per-launch arm is the like-for-like host-launched number. None of
these is the in-situ cost inside a GEMM epilogue (WP-C5, not measured yet).

## Results against RCCL (job 68219499, venv: torch 2.12 + ROCm 7.2 RCCL)

Form names in parentheses: the fastest form of that arm group at that size.

#### TP2

| rows | cols | dtype | KiB / rank | RCCL best (graph, protocol) | chain / RCCL best | grid-sync chain / RCCL best | grid-sync, 32 CTAs / RCCL best | 1 launch / RCCL graph best | default model build, 1 launch / RCCL graph best | chain / RCCL default protocol | A/A |
|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2560 | bf16 | 5 | LL | 0.29 (line_os_hd) | 0.29 (pair_os) | 0.29 | 1.02 (line_ts) | 1.03 | 0.28 | 1.00 |
| 2 | 2560 | bf16 | 10 | LL | 0.28 (pair_os) | 0.27 (pair_os) | 0.27 | 0.97 (line_ts) | 1.02 | 0.27 | 1.00 |
| 4 | 2560 | bf16 | 20 | LL | 0.31 (pair_os) | 0.31 (pair_os) | 0.31 | 0.95 (pair_ts) | 1.05 | 0.30 | 1.01 |
| 8 | 2560 | bf16 | 40 | LL | 0.31 (pair_os) | 0.39 (pair_os) | 0.39 | 0.92 (line_os) | 0.97 | 0.10 | 1.00 |
| 16 | 2560 | bf16 | 80 | LL | 0.32 (pair_os) | 0.54 (pair_os) | 0.54 | 0.87 (pair_os) | 0.97 | 0.11 | 1.00 |
| 32 | 2560 | bf16 | 160 | LL | 0.32 (line_os_hd) | 0.62 (line_os) | 0.62 | 0.77 (line_os) | 0.85 | 0.15 | 1.00 |
| 64 | 2560 | bf16 | 320 | Simple | 0.37 (line_os_cmem) | 0.69 (line_os) | 0.70 | 0.75 (line_os) | 0.81 | 0.36 | 1.00 |
| 128 | 2560 | bf16 | 640 | default | 0.56 (line_os_pcsc1) | 0.83 (line_ts) | 0.84 | 0.86 (line_ts) | 0.96 | 0.56 | 1.00 |
| 1 | 10240 | bf16 | 20 | LL | 0.31 (pair_os) | 0.31 (pair_os) | 0.31 | 0.96 (line_ts) | 0.96 | 0.30 | 1.00 |
| 2 | 10240 | bf16 | 40 | LL | 0.31 (pair_os) | 0.39 (pair_os) | 0.39 | 0.93 (pair_os) | 0.97 | 0.10 | 0.99 |
| 4 | 10240 | bf16 | 80 | LL | 0.32 (pair_os) | 0.54 (pair_os) | 0.53 | 0.91 (line_os) | 0.92 | 0.11 | 1.01 |
| 8 | 10240 | bf16 | 160 | LL | 0.32 (line_os_hd) | 0.63 (line_os) | 0.63 | 0.77 (line_os) | 0.84 | 0.15 | 0.99 |
| 16 | 10240 | bf16 | 320 | Simple | 0.37 (line_os_cmem) | 0.69 (line_os) | 0.69 | 0.75 (line_os) | 0.81 | 0.36 | 1.00 |
| 32 | 10240 | bf16 | 640 | LL128 | 0.56 (line_os_pcsc1) | 0.83 (line_ts) | 0.84 | 0.86 (line_ts) | 0.96 | 0.56 | 1.00 |
| 64 | 10240 | bf16 | 1280 | LL128 | 0.73 (line_os) | 0.94 (line_ts) | 0.95 | 0.94 (line_ts) | 1.03 | 0.73 | 1.00 |
| 128 | 10240 | bf16 | 2560 | LL128 | 0.85 (line_os_cmem) | 1.02 (line_ts) | 1.05 | 1.03 (line_ts) | 1.06 | 0.85 | 1.00 |
| 1 | 2560 | f32 | 10 | LL | 0.28 (pair_os) | 0.29 (pair_os) | 0.29 | 1.01 (pair_ts) | 1.09 | 0.26 | 1.00 |
| 2 | 2560 | f32 | 20 | LL | 0.30 (pair_os) | 0.31 (pair_os) | 0.31 | 0.99 (pair_ts) | 0.97 | 0.29 | 1.01 |
| 4 | 2560 | f32 | 40 | LL | 0.31 (pair_os) | 0.39 (pair_os) | 0.39 | 0.98 (pair_ts) | 1.01 | 0.10 | 1.00 |
| 8 | 2560 | f32 | 80 | LL | 0.32 (pair_os) | 0.54 (pair_os) | 0.54 | 0.93 (line_os) | 0.93 | 0.11 | 1.03 |
| 16 | 2560 | f32 | 160 | LL | 0.33 (line_os_hdcm) | 0.62 (line_os) | 0.62 | 0.77 (line_os) | 0.86 | 0.15 | 1.00 |
| 32 | 2560 | f32 | 320 | Simple | 0.36 (line_os_cmem) | 0.69 (line_os) | 0.69 | 0.75 (line_os) | 0.82 | 0.36 | 1.00 |
| 64 | 2560 | f32 | 640 | LL128 | 0.55 (line_os) | 0.81 (line_ts) | 0.82 | 0.85 (line_ts) | 0.94 | 0.55 | 1.00 |
| 128 | 2560 | f32 | 1280 | LL128 | 0.73 (line_os) | 0.93 (line_ts) | 0.93 | 0.93 (line_ts) | 1.02 | 0.73 | 1.00 |
| 1 | 10240 | f32 | 40 | LL | 0.31 (pair_os) | 0.39 (pair_os) | 0.39 | 0.96 (line_os) | 0.97 | 0.10 | 1.00 |
| 2 | 10240 | f32 | 80 | LL | 0.32 (pair_os) | 0.54 (pair_os) | 0.54 | 0.90 (pair_os) | 0.95 | 0.11 | 0.99 |
| 4 | 10240 | f32 | 160 | LL | 0.33 (line_os_hdcm) | 0.62 (line_os) | 0.62 | 0.77 (line_os) | 0.88 | 0.15 | 0.99 |
| 8 | 10240 | f32 | 320 | Simple | 0.36 (line_os_cmem) | 0.69 (line_os) | 0.69 | 0.75 (line_ts) | 0.81 | 0.36 | 1.00 |
| 16 | 10240 | f32 | 640 | default | 0.55 (line_os) | 0.81 (line_ts) | 0.82 | 0.85 (line_ts) | 0.94 | 0.55 | 1.00 |
| 32 | 10240 | f32 | 1280 | LL128 | 0.73 (line_os_pcsc1) | 0.92 (line_ts) | 0.93 | 0.93 (line_ts) | 1.02 | 0.73 | 1.00 |
| 64 | 10240 | f32 | 2560 | LL128 | 0.85 (line_os_cmem) | 1.00 (line_ts) | 1.02 | 1.01 (line_ts) | 1.05 | 0.85 | 1.00 |
| 128 | 10240 | f32 | 5120 | Simple | 0.95 (line_os) | 1.04 (line_ts) | 1.09 | 1.04 (line_ts) | 1.06 | 0.95 | 1.00 |

#### TP3

| rows | cols | dtype | KiB / rank | RCCL best (graph, protocol) | chain / RCCL best | grid-sync chain / RCCL best | grid-sync, 32 CTAs / RCCL best | 1 launch / RCCL graph best | default model build, 1 launch / RCCL graph best | chain / RCCL default protocol | A/A |
|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2560 | bf16 | 5 | default | 0.22 (line_os_hd) | 0.25 (pair_os) | 0.25 | 0.74 (pair_os) | 0.79 | 0.22 | 1.01 |
| 4 | 2560 | bf16 | 20 | default | 0.20 (pair_os) | 0.22 (pair_os) | 0.22 | 0.61 (line_ts) | 0.62 | 0.20 | 1.02 |
| 16 | 2560 | bf16 | 80 | LL | 0.21 (pair_os) | 0.37 (pair_ts) | 0.37 | 0.59 (line_os) | 0.63 | 0.08 | 1.03 |
| 64 | 2560 | bf16 | 320 | LL | 0.26 (line_ts_hd) | 0.49 (line_ts) | 0.51 | 0.54 (line_ts) | 0.60 | 0.24 | 1.00 |
| 128 | 2560 | bf16 | 640 | default | 0.34 (line_ts) | 0.58 (line_ts) | 0.60 | 0.60 (line_ts) | 0.66 | 0.34 | 1.00 |
| 1 | 10240 | bf16 | 20 | default | 0.21 (pair_os) | 0.22 (pair_os) | 0.22 | 0.62 (line_ts) | 0.63 | 0.21 | 1.03 |
| 4 | 10240 | bf16 | 80 | LL | 0.21 (pair_os) | 0.37 (pair_os) | 0.37 | 0.61 (pair_os) | 0.59 | 0.08 | 1.01 |
| 16 | 10240 | bf16 | 320 | LL | 0.26 (line_ts_hd) | 0.49 (line_ts) | 0.51 | 0.53 (line_ts) | 0.60 | 0.24 | 1.00 |
| 64 | 10240 | bf16 | 1280 | default | 0.47 (line_ts) | 0.66 (line_ts) | 0.69 | 0.68 (line_ts) | 0.70 | 0.47 | 1.00 |
| 128 | 10240 | bf16 | 2560 | default | 0.59 (line_ts) | 0.73 (line_ts) | 0.77 | 0.73 (line_ts) | 0.76 | 0.59 | 1.00 |
| 1 | 2560 | f32 | 10 | LL | 0.20 (pair_os) | 0.20 (pair_os) | 0.21 | 0.60 (pair_ts) | 0.62 | 0.19 | 0.98 |
| 4 | 2560 | f32 | 40 | LL | 0.20 (pair_os) | 0.27 (pair_os) | 0.27 | 0.61 (pair_os) | 0.62 | 0.20 | 1.03 |
| 16 | 2560 | f32 | 160 | LL | 0.22 (line_os_hd) | 0.45 (pair_ts) | 0.45 | 0.56 (pair_ts) | 0.67 | 0.11 | 1.00 |
| 64 | 2560 | f32 | 640 | default | 0.33 (line_ts) | 0.59 (line_ts) | 0.60 | 0.61 (line_ts) | 0.67 | 0.33 | 1.00 |
| 128 | 2560 | f32 | 1280 | default | 0.48 (line_ts) | 0.67 (line_ts) | 0.70 | 0.68 (line_ts) | 0.72 | 0.48 | 1.00 |
| 1 | 10240 | f32 | 40 | LL | 0.20 (pair_os) | 0.27 (pair_os) | 0.27 | 0.62 (line_ts) | 0.65 | 0.20 | 1.01 |
| 4 | 10240 | f32 | 160 | LL | 0.22 (line_os_hd) | 0.45 (pair_ts) | 0.45 | 0.54 (pair_ts) | 0.68 | 0.11 | 0.99 |
| 16 | 10240 | f32 | 640 | default | 0.33 (line_ts) | 0.59 (line_ts) | 0.60 | 0.62 (line_ts) | 0.66 | 0.33 | 1.00 |
| 64 | 10240 | f32 | 2560 | LL128 | 0.61 (line_ts) | 0.75 (line_ts) | 0.79 | 0.74 (line_ts) | 0.78 | 0.61 | 1.00 |
| 128 | 10240 | f32 | 5120 | default | 0.71 (line_ts) | 0.79 (line_ts) | 0.85 | 0.79 (line_ts) | 0.81 | 0.71 | 1.00 |

#### TP4

| rows | cols | dtype | KiB / rank | RCCL best (graph, protocol) | chain / RCCL best | grid-sync chain / RCCL best | grid-sync, 32 CTAs / RCCL best | 1 launch / RCCL graph best | default model build, 1 launch / RCCL graph best | chain / RCCL default protocol | A/A |
|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2560 | bf16 | 5 | LL | 0.23 (pair_ts) | 0.28 (pair_os) | 0.28 | 0.70 (pair_os) | 0.72 | 0.23 | 1.01 |
| 2 | 2560 | bf16 | 10 | Simple | 0.18 (pair_ts) | 0.22 (pair_os) | 0.22 | 0.54 (line_os) | 0.57 | 0.17 | 1.00 |
| 4 | 2560 | bf16 | 20 | LL | 0.17 (pair_ts) | 0.21 (pair_os) | 0.21 | 0.51 (line_os) | 0.51 | 0.16 | 1.00 |
| 8 | 2560 | bf16 | 40 | LL | 0.18 (pair_ts) | 0.25 (pair_ts) | 0.25 | 0.54 (line_ts) | 0.54 | 0.18 | 1.00 |
| 16 | 2560 | bf16 | 80 | LL | 0.18 (pair_ts) | 0.29 (pair_ts) | 0.29 | 0.53 (line_ts) | 0.56 | 0.18 | 0.98 |
| 32 | 2560 | bf16 | 160 | LL | 0.19 (pair_ts) | 0.37 (pair_ts) | 0.42 | 0.52 (pair_ts) | 0.52 | 0.09 | 1.00 |
| 64 | 2560 | bf16 | 320 | LL | 0.36 (line_ts_hd) | 0.65 (line_ts) | 0.73 | 0.77 (line_ts) | 0.82 | 0.20 | 1.00 |
| 128 | 2560 | bf16 | 640 | LL | 0.36 (line_ts_hdcm) | 0.66 (line_ts) | 0.72 | 0.69 (line_ts) | 0.76 | 0.27 | 1.00 |
| 1 | 10240 | bf16 | 20 | LL | 0.17 (pair_ts) | 0.21 (pair_os) | 0.21 | 0.51 (line_ts) | 0.52 | 0.17 | 1.00 |
| 2 | 10240 | bf16 | 40 | LL | 0.18 (pair_ts) | 0.25 (pair_ts) | 0.25 | 0.54 (pair_ts) | 0.54 | 0.18 | 1.00 |
| 4 | 10240 | bf16 | 80 | LL | 0.18 (pair_ts) | 0.29 (pair_ts) | 0.29 | 0.53 (pair_ts) | 0.53 | 0.18 | 1.01 |
| 8 | 10240 | bf16 | 160 | LL | 0.19 (pair_ts) | 0.37 (pair_ts) | 0.42 | 0.53 (pair_ts) | 0.55 | 0.09 | 1.00 |
| 16 | 10240 | bf16 | 320 | LL | 0.36 (line_ts_hd) | 0.65 (line_ts) | 0.73 | 0.77 (line_ts) | 0.84 | 0.21 | 1.00 |
| 32 | 10240 | bf16 | 640 | LL | 0.36 (line_ts_hdcm) | 0.66 (line_ts) | 0.72 | 0.69 (line_ts) | 0.75 | 0.27 | 1.00 |
| 64 | 10240 | bf16 | 1280 | LL128 | 0.45 (line_ts) | 0.72 (line_ts) | 0.75 | 0.75 (line_ts) | 0.79 | 0.45 | 1.00 |
| 128 | 10240 | bf16 | 2560 | LL128 | 0.65 (line_ts) | 0.91 (line_ts) | 0.91 | 0.94 (line_ts) | 0.97 | 0.64 | 1.00 |
| 1 | 2560 | f32 | 10 | Simple | 0.18 (pair_ts) | 0.21 (pair_os) | 0.21 | 0.56 (line_ts) | 0.57 | 0.17 | 1.00 |
| 2 | 2560 | f32 | 20 | default | 0.17 (pair_ts) | 0.21 (pair_os) | 0.21 | 0.52 (pair_ts) | 0.54 | 0.17 | 1.00 |
| 4 | 2560 | f32 | 40 | default | 0.18 (pair_ts) | 0.25 (pair_ts) | 0.25 | 0.56 (line_ts) | 0.57 | 0.18 | 0.99 |
| 8 | 2560 | f32 | 80 | default | 0.18 (pair_ts) | 0.29 (pair_ts) | 0.29 | 0.55 (pair_ts) | 0.55 | 0.18 | 1.01 |
| 16 | 2560 | f32 | 160 | LL | 0.19 (pair_ts) | 0.37 (pair_ts) | 0.41 | 0.55 (line_ts) | 0.55 | 0.09 | 1.02 |
| 32 | 2560 | f32 | 320 | LL | 0.37 (line_ts_hd) | 0.67 (line_ts) | 0.73 | 0.77 (line_ts) | 0.87 | 0.21 | 1.00 |
| 64 | 2560 | f32 | 640 | LL | 0.34 (line_ts) | 0.66 (line_ts) | 0.73 | 0.71 (line_ts) | 0.78 | 0.26 | 1.00 |
| 128 | 2560 | f32 | 1280 | LL128 | 0.45 (line_ts) | 0.71 (line_ts) | 0.79 | 0.75 (line_ts) | 0.79 | 0.45 | 1.00 |
| 1 | 10240 | f32 | 40 | default | 0.18 (pair_ts) | 0.25 (pair_ts) | 0.25 | 0.56 (line_ts) | 0.56 | 0.18 | 1.00 |
| 2 | 10240 | f32 | 80 | LL | 0.18 (pair_ts) | 0.29 (pair_ts) | 0.29 | 0.54 (pair_ts) | 0.55 | 0.18 | 0.96 |
| 4 | 10240 | f32 | 160 | LL | 0.19 (pair_ts) | 0.37 (pair_ts) | 0.41 | 0.54 (line_ts) | 0.55 | 0.09 | 1.01 |
| 8 | 10240 | f32 | 320 | LL | 0.37 (line_ts_hd) | 0.67 (line_ts) | 0.73 | 0.77 (line_ts) | 0.87 | 0.21 | 1.00 |
| 16 | 10240 | f32 | 640 | LL | 0.34 (line_ts) | 0.66 (line_ts) | 0.73 | 0.71 (line_ts) | 0.77 | 0.26 | 0.99 |
| 32 | 10240 | f32 | 1280 | LL128 | 0.45 (line_ts) | 0.71 (line_ts) | 0.79 | 0.75 (line_ts) | 0.79 | 0.45 | 1.01 |
| 64 | 10240 | f32 | 2560 | Simple | 0.65 (line_ts) | 0.90 (line_ts) | 0.95 | 0.92 (line_ts) | 0.96 | 0.64 | 1.00 |
| 128 | 10240 | f32 | 5120 | LL128 | 0.85 (line_ts) | 1.00 (line_ts) | 1.12 | 1.01 (line_ts) | 1.04 | 0.85 | 1.00 |

## Results against the aiter custom all-reduce (job 68219502, container: torch 2.13 + ROCm 10)

Same node, same rocKE code (code digest `dc5fedfab364`), bf16 at rows 1-128 x 2560 / 10240 and
the FlyTape toy shapes (f32 rows 1 / 4 x 2560).

#### TP2 (container)

| rows | cols | dtype | KiB / rank | chain / aiter graph | grid-sync chain / aiter graph | grid-sync, 32 CTAs / aiter graph | 1 launch / aiter graph | aiter graph / RCCL graph best | grid-sync chain / RCCL best | A/A |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2560 | bf16 | 5 | 0.73 | 0.75 | 0.75 | 3.20 | 0.36 | 0.27 | 1.01 |
| 2 | 2560 | bf16 | 10 | 0.81 | 0.75 | 0.74 | 2.95 | 0.38 | 0.29 | 0.99 |
| 4 | 2560 | bf16 | 20 | 0.77 | 0.80 | 0.80 | 2.68 | 0.46 | 0.37 | 1.02 |
| 8 | 2560 | bf16 | 40 | 0.70 | 0.86 | 0.86 | 2.27 | 0.52 | 0.45 | 1.01 |
| 16 | 2560 | bf16 | 80 | 0.60 | 0.98 | 0.98 | 1.74 | 0.59 | 0.57 | 1.00 |
| 32 | 2560 | bf16 | 160 | 0.53 | 1.17 | 1.16 | 1.46 | 0.55 | 0.65 | 1.00 |
| 64 | 2560 | bf16 | 320 | 0.62 | 1.21 | 1.21 | 1.30 | 0.57 | 0.69 | 1.00 |
| 128 | 2560 | bf16 | 640 | 0.76 | 1.15 | 1.13 | 1.19 | 0.69 | 0.79 | 1.00 |
| 1 | 10240 | bf16 | 20 | 0.76 | 0.78 | 0.78 | 2.74 | 0.47 | 0.37 | 0.99 |
| 2 | 10240 | bf16 | 40 | 0.71 | 0.87 | 0.86 | 2.33 | 0.52 | 0.45 | 0.98 |
| 4 | 10240 | bf16 | 80 | 0.60 | 0.98 | 0.98 | 1.77 | 0.59 | 0.58 | 0.99 |
| 8 | 10240 | bf16 | 160 | 0.54 | 1.17 | 1.17 | 1.47 | 0.55 | 0.64 | 1.00 |
| 16 | 10240 | bf16 | 320 | 0.62 | 1.21 | 1.20 | 1.29 | 0.57 | 0.69 | 1.00 |
| 32 | 10240 | bf16 | 640 | 0.76 | 1.15 | 1.13 | 1.18 | 0.69 | 0.79 | 0.99 |
| 64 | 10240 | bf16 | 1280 | 0.87 | 1.13 | 1.14 | 1.13 | 0.81 | 0.91 | 1.00 |
| 128 | 10240 | bf16 | 2560 | 0.98 | 1.15 | 1.19 | 1.16 | 0.87 | 1.00 | 1.00 |
| 1 | 2560 | f32 | 10 | 0.44 | 0.51 | 0.51 | 2.25 | 0.51 | 0.26 | 1.00 |
| 4 | 2560 | f32 | 40 | 0.41 | 0.68 | 0.68 | 1.85 | 0.66 | 0.45 | 1.00 |

#### TP4 (container)

| rows | cols | dtype | KiB / rank | chain / aiter graph | grid-sync chain / aiter graph | grid-sync, 32 CTAs / aiter graph | 1 launch / aiter graph | aiter graph / RCCL graph best | grid-sync chain / RCCL best | A/A |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2560 | bf16 | 5 | 0.78 | 0.99 | 0.99 | 2.54 | 0.33 | 0.32 | 1.00 |
| 2 | 2560 | bf16 | 10 | 0.77 | 0.94 | 0.94 | 2.45 | 0.25 | 0.23 | 0.99 |
| 4 | 2560 | bf16 | 20 | 0.71 | 0.88 | 0.88 | 2.21 | 0.27 | 0.24 | 1.00 |
| 8 | 2560 | bf16 | 40 | 0.59 | 0.84 | 0.84 | 1.86 | 0.43 | 0.36 | 1.00 |
| 16 | 2560 | bf16 | 80 | 0.47 | 0.77 | 0.77 | 1.47 | 0.55 | 0.42 | 1.00 |
| 32 | 2560 | bf16 | 160 | 0.55 | 1.13 | 1.19 | 1.64 | 0.47 | 0.53 | 1.00 |
| 64 | 2560 | bf16 | 320 | 0.70 | 1.29 | 1.46 | 1.45 | 0.51 | 0.66 | 1.00 |
| 128 | 2560 | bf16 | 640 | 0.71 | 1.34 | 1.42 | 1.41 | 0.51 | 0.68 | 0.99 |
| 1 | 10240 | bf16 | 20 | 0.71 | 0.89 | 0.89 | 2.33 | 0.27 | 0.24 | 1.01 |
| 2 | 10240 | bf16 | 40 | 0.59 | 0.83 | 0.83 | 1.91 | 0.43 | 0.36 | 1.01 |
| 4 | 10240 | bf16 | 80 | 0.47 | 0.77 | 0.77 | 1.53 | 0.54 | 0.42 | 1.02 |
| 8 | 10240 | bf16 | 160 | 0.56 | 1.14 | 1.19 | 1.69 | 0.47 | 0.54 | 0.99 |
| 16 | 10240 | bf16 | 320 | 0.70 | 1.27 | 1.44 | 1.49 | 0.52 | 0.66 | 1.00 |
| 32 | 10240 | bf16 | 640 | 0.71 | 1.35 | 1.43 | 1.40 | 0.50 | 0.68 | 1.00 |
| 64 | 10240 | bf16 | 1280 | 0.82 | 1.31 | 1.37 | 1.36 | 0.52 | 0.68 | 1.00 |
| 128 | 10240 | bf16 | 2560 | 0.96 | 1.35 | 1.35 | 1.40 | 0.66 | 0.89 | 1.00 |
| 1 | 2560 | f32 | 10 | 0.57 | 0.65 | 0.65 | 1.87 | 0.35 | 0.23 | 0.99 |
| 4 | 2560 | f32 | 40 | 0.54 | 0.75 | 0.75 | 1.73 | 0.49 | 0.37 | 0.87 |

* Up to 80 KiB per rank the rocKE grid-synced chain (all CTAs or 32) is faster than aiter's
  graph (0.51-0.99); from 160 KiB it is slower (1.13-1.46). The large-batch decode all-reduces of the target model
  (rows 64-128 x 2560 bf16 = 320-640 KiB per rank) fall in the losing range: that is the open
  optimization target for the large-batch work (DESIGN-CCL forms not built yet: `lamport`
  sentinel payloads, wider polls, comm-wave publish).
* aiter's graph is 0.25-0.87 of RCCL's best: comparing against RCCL alone overstates rocKE's
  standing against the serving stack it would replace.

## What decides the form (crossover)

Measured in the job above (and replicated in the container job at TP2 / TP4):

| | grid-synced chain (`select_knobs(shape="layer")`, the default) | best-case chain (`shape="chain"`) |
|---|---|---|
| TP2 | pair one-shot up to 80 KiB, line one-shot up to 320 KiB, line two-shot above | pair one-shot up to 80 KiB, line one-shot above |
| TP3 | pair one-shot up to 80 KiB, pair two-shot up to 160 KiB, line two-shot above | pair one-shot up to 80 KiB, line one-shot up to 160 KiB, line two-shot above |
| TP4 | pair one-shot up to 20 KiB, pair two-shot up to 160 KiB, line two-shot above | pair two-shot up to 160 KiB, line two-shot above |

* Under the grid sync, line two-shot wins at TP2 from 640 KiB (line one-shot 1-8 % slower), and
  pair one-shot wins at TP4 up to 20 KiB (pair two-shot 6-9 % slower); the margins at the TP3
  boundaries are within 2 %.
* At the largest sizes both rocKE and RCCL are bound by the per-pair xGMI bandwidth.

## Levers (default-off knobs, all built and measured)

Ratios are lever arm / its base arm, per-collective chain latency, range over the sizes of each
band (job 68219499; TP2 / TP3 / TP4; the A/A control spans 0.96-1.03).

| lever (knob) | base | up to 160 KiB per rank | 320 KiB and above | verdict |
|---|---|---:|---:|---|
| `homing="dual"` (calibrated IOD-local flag copy) | line one-shot | 0.74-0.97 | 1.04-1.14 | wins up to 160 KiB: `select_knobs(..., calibrated=True)` turns it on there |
| `homing="dual"` | line two-shot | 0.76-1.02 | 0.97-1.08 | small win up to 160 KiB; off by default |
| `carry="mem"` (chain input through memory) | line one-shot (register carry) | 0.89-1.15 | 0.99-1.02 | mixed; a model epilogue produces the data in registers, so off |
| `carry="mem"` | pair one-shot / two-shot | 0.99-1.32 | 0.99-1.09 | loses |
| `homing="dual"` + `carry="mem"` | line dual (one- / two-shot) | 0.97-1.32 | 0.98-1.10 | loses |
| `poll_cpol="sc1"` (device-scope poll bits) | line one-shot | 0.97-1.03 | 0.99-1.01 | neutral; the lint rejects it unless waived, default stays system scope |
| `poll_cpol="none"` (plain loads) | any | - | - | negative control: 0 of 5 launches correct on uncached, fine-grained and coarse arenas |
| `sync="grid"` | chain | see tables | see tables | the layer-shaped cost, not a speed lever |
| default model build: auto form + launch header (default on) | best header-off form, one launch | 0.97-1.26 (all sizes) | - | mixes the form choice with the header's one extra exchange per launch; the price of `CCL_MISMATCH` detection |
| arena memory `fine` / `coarse` (with system-scope bits) | uncached | - | - | correct (full suite passes); not timed |
| `push_cpol="sc0sc1nt"`, `pause`, `slots=3`, `waves=4` | defaults | - | - | built and bitwise-tested; not timed |

## CCL-LM: the large-message levers (layer shape, TP2; container jobs 68224788, 68225304, 68225555, 68225793)

Measured scope: the grid-synced (layer-shaped) all-reduce chain of this directory at TP2 against the
aiter custom all-reduce's captured graph (ATOM's TP path) on the same two GPUs, rows 16-128 x hidden
2560 / 10240, bf16 and f32 (80 KiB - 5 MiB per rank), one 8-GPU gfx950 server, three GPU pairs, q38
container (torch 2.13 + ROCm 10, aiter as installed there). `bench_ccl_lm.py bench`: numerics first
(every rocKE arm bitwise against the rank-order f32 reference on every rank; aiter / RCCL against a
float64 sum), 12-20 rounds in a new shuffled order per round, per-round ratios, the DESIGN 9.1 win rule
(median below both controls' per-round minima by 0.002 and at least 3/4 of the rounds below 1). Controls:
two more separately captured aiter graphs, and each rocKE arm's same-binary and recompiled twins.

The Fix-round loss above (1.13-1.21 at TP2 from 160 KiB) had two causes, found with the `diag` phase
stamps and the `linkbw` probe: the line form waits a full remote-write round trip before it publishes a
line's flag (aiter's LL / LL128 carry the flag in the data), and from about 640 KiB the grid sync waits
for the slowest CTA while the units complete in an XCD-dependent order. The levers that fixed them:
`form="lamport"` (no drain, no flag), `form="pull"` with wider units on a capped grid (`unit_kib`,
`ctas`; aiter's own large-message scheme is a pull), and the striped grid sync. The selection
`layout.lm_select` was declared from the lever sweep and the refinement, then confirmed in a separate
job (68225555, and 68225793 for 5 MiB):

| per-rank payload | selected build | CCL-LM / aiter graph | W3 best form / aiter graph | verdict |
|---|---|---:|---:|---|
| 80 KiB | lamport | 0.73 | 1.00 | WIN |
| 160 KiB | lamport (f32: 2 KiB units) | 0.69-0.81 | 1.01-1.16 | WIN |
| 320 KiB | lamport (f32: 2 KiB units) | 0.78-0.89 | 1.12-1.22 | WIN |
| 640 KiB | bf16: pull, 2 KiB units, 32 CTAs; f32: lamport, 2 KiB units | 0.88-0.90 | 1.13-1.17 | WIN |
| 1.25 MiB | bf16: pull, 2 KiB, 64 CTAs; f32: pull, 4 KiB, 32 CTAs | 0.94-0.95 | 1.18 | WIN |
| 2.5 MiB | pull, 4 KiB units, 48 CTAs | 0.98 | 1.17-1.18 | WIN |
| 5 MiB (f32) | pull, 4 KiB units, 48 CTAs | 0.99 | 1.14 | WIN |

* The margin shrinks with the payload: from about 1 MiB both implementations are bound by the one xGMI
  link between the two GPUs. In the link probe a pull moves the same bytes in 0.955-0.962 of a push's
  time with both GPUs sending (0.89-0.93 one way), and one XCD (or one XCD pair) alone already reaches 0.91-0.99 of the
  all-XCD rate, so per-XCD weighting (`place="table"`) adds nothing (it lost).
* Levers that lost or tied in the layer shape (kept as knobs): `nt` pushes, poll pauses, 3 slots,
  `self_local`, XCD-weighted placement, 4 waves per CTA; dual flag homing wins only up to 320 KiB on the
  line form; the striped sync wins up to 320 KiB and loses a few percent at 2.5-5 MiB.
* The not-built publish variants of DESIGN-CCL now exist: `comm_wave` (0.78-0.90 of the line form's
  drain-then-flag publish) and `counted` (0.84-0.99 once its dummy flag stores were spread over many
  words; a single shared dummy word made it 1.5-2.5x slower); both stay behind the lamport form.

Fix round (`experiments/qwen38-true-mono/large-batch/CCL-LM.md` in copy-rocm-lib, "Fix round").
* **Numerics gate.** The benchmark's gate now uses a non-contracting chain (`ref.LM_GROWTH`). W3's scales reach a
  bitwise fixed point, so a stale-slot read passed the old gate. Must-fail stale-read twins also run at every size.
* **Selection changes.** Three bands changed, each confirmed on both rocke nodes:
  * 80 KiB: lamport with 4 waves per CTA;
  * 1.25 MiB f32: the counted publish;
  * above 1.25 MiB: pull with 4 KiB units on 64 CTAs with the one-word grid sync.
  `lm_select(..., layer=True)` returns each band's grid sync.
* **Final result** (strict: against the per-round best aiter capture, launch included): WIN at every size from 80 KiB
  to 2.5 MiB on both nodes. At 5 MiB f32 it is a link-bound tie, a strict WIN in 3 of 5 runs.

| per-rank payload | selected build (fix round) | CCL-LM / aiter graph, n01-09 and n07-17 | verdict |
|---|---|---:|---|
| 80 KiB | lamport, 4 waves | 0.71 | WIN |
| 160 KiB | lamport (f32: 2 KiB units) | 0.66-0.82 | WIN |
| 320 KiB | lamport (f32: 2 KiB units) | 0.76-0.88 | WIN |
| 640 KiB | bf16: pull, 2 KiB units, 32 CTAs; f32: lamport, 2 KiB units | 0.88-0.93 | WIN |
| 1.25 MiB | bf16: pull, 2 KiB, 64 CTAs; f32: pull, 4 KiB, 32 CTAs, counted publish | 0.93-0.96 | WIN |
| 2.5 MiB | pull, 4 KiB units, 64 CTAs, word grid sync | 0.97-0.99 | WIN (5 of 6 runs for bf16) |
| 5 MiB (f32) | pull, 4 KiB units, 64 CTAs, word grid sync | 0.98-1.00 | tie (link-bound) |
* `ar_dtype="bf16"` (bf16 partials of f32 data) costs 0.53-0.75 of the f32-wire build and 0.60-0.64 of
  aiter's f32 graph at 2.5-5 MiB: a tolerance-class result, not a matched-precision comparison.
* Every arm passed its numeric gate; the suite (17 tests + `lm_bitwise`, `lm_canon`, `lm_mutants`) passes
  at TP2 and TP4.
* Caveats: one server (the second server's bad GPU blocked the container runs there), TP2 only for the
  timing, the microbenchmark's layer shape rather than the in-situ GEMM + all-reduce.

## Against FlyTape's in-kernel exchange (job 68219502, same GPUs)

FlyTape (`scripts/bench_tp.py`, its fp32 toy at hidden 2560, 200 steps, FlyDSL 0.3.4.1) ran in
the same job and on the same GPUs as the rocKE arms at 1 x 2560 and 4 x 2560 f32. FlyTape's
`dense.reduce` phase covers the wait for the ranks' partials, the rank-order sum and the next
layer's grid sync; its pushes overlap the preceding GEMV phase. The rocKE grid-synced chain
covers push, flight, poll, sum and the grid sync; the plain chain has no grid sync. Related,
not identical, quantities.

| | TP2 batch 1 | TP2 batch 4 | TP4 batch 1 | TP4 batch 4 |
|---|---:|---:|---:|---:|
| rocKE best-case chain per all-reduce / FlyTape reduce phase | 0.61 | 0.67 | 0.97 | 0.93 |
| rocKE grid-synced chain per all-reduce / FlyTape reduce phase | 0.71 | 1.11 | 1.11 | 1.29 |

## Replication and variance

The RCCL tables come from one job (four GPUs of one node; TP2 on GPUs 0-1, TP3 on 0-2, TP4 on
0-3); the container job (same node, other processes) replicated the TP2 / TP4 bf16 ordering of
forms and the grid-sync crossover. Absolute small-message latency of a form moved by up to about
1.3x between jobs on different GPU pairs, so only same-job ratios are compared. The A/A control
spans 0.96-1.03 per size, with one outlier of 0.87 (TP4, 4 x 2560 f32, container job).

## Reproduce

```bash
python -m torch.distributed.run --standalone --nproc_per_node 2 -m rocke.examples.gfx950.ccl.bench_ccl bench --out DIR --rounds 12
python -m torch.distributed.run --standalone --nproc_per_node 4 -m rocke.examples.gfx950.ccl.bench_ccl bench --out DIR --rounds 12
# external arms: --extra-arms module:function (the aiter arm is a plugin outside this tree)
```

Software: torch 2.12 + ROCm 7.2 user space (RCCL from that torch build) for the RCCL tables;
torch 2.13 + ROCm 10 with aiter for the aiter tables; rocKE Python engine with torch's COMGR.
Hardware: gfx950, 8 GPUs per node, the job's GPUs idle (no other process on them).
