#!/usr/bin/env python3
"""
Standalone register-reuse ("physical WAR") headroom check for the gfx1250 MXF4 subtile
mainloop's scale (MXSA/MXSB) ds_load -> v_wmma_scale_f32_32x16x128_f4 schedule.

Vendors ~/gemm_wt/sched_solver/schedsolver/model.py's queue_wave() verbatim (pure forward
simulation, no CP-SAT/ortools needed) and the machine constants gemm_wt measured on
hardware for the SAME WMMA instruction on the SAME gfx1250 generation
(sched_solver/experiments/wmma_queue/f4_5x10_mx_qfull.yaml):
    wmma_queue_lead_clk=23, wmma_issue_slots=2, wmma_min_issue_gap_clk=4,
    wmma_pair_issue_gap_clk=8 (no pairing), read_fixed_clk=53

Schedule extracted by hand from outdir5's generated assembly (du512_cl24.yaml build):
main loop period S=128 WMMAs (4 subIterK x 32 WMMAs, MIWT 4x8), P = S * 8 clk = 1024 clk.
Each K-step's 12 scale ds_load_b32 (4 MXSA groups + 8 MXSB groups) are issued as one burst
in the hole right after WMMA #4 (0-indexed index 3) of the PREVIOUS K-step (q=4), i.e. 28-124
WMMA slots before anything reads them:
  - group 0 (first read): first_use = WMMA #0 of the consuming K-step -> lead  28 WMMA = 224 clk
  - group 3 / 7 (last read): first_use = WMMA #24 of the consuming K-step -> lead 52 WMMA = 416 clk
"""
import sys

sys.path.insert(0, "/home/mchirila/gemm_wt/sched_solver/schedsolver")


def queue_wave(S, P, reads, lead, pair_gap=4, iters=6, issue_slots=1, min_gap=1):
    """Verbatim from gemm_wt/sched_solver/schedsolver/model.py (pure Python, no deps)."""
    MMA_ = P // S
    hole = {}
    for x, (q, fu, u, lt) in enumerate(reads):
        hole.setdefault((q - 1) % S, []).append(x)
    need = {}
    for x, (q, fu, u, lt) in enumerate(reads):
        need.setdefault(fu, []).append(x)
    W = {}
    iss = {}
    for it in range(iters):
        for s in range(S):
            g = it * S + s
            v = MMA_ * g - lead
            if g - 1 in W:
                v = max(v, W[g - 1] + issue_slots + sum(reads[x][2] for x in hole.get((g - 1) % S, [])),
                        W[g - 1] + min_gap)
            if g - 2 in W:
                v = max(v, W[g - 2] + pair_gap)
            for x in need.get(s, []):
                q, fu, u, lt = reads[x]
                gq = g - fu + q - 1
                if gq in W:
                    v = max(v, W[gq] + 1 + lt)
            W[g] = v
            for x in hole.get(s, []):
                iss[(x, it)] = v + 1
    last = iters - 1
    return ([W[last * S + s] - last * P for s in range(S)],
            [iss[(x, last)] - last * P for x in range(len(reads))])


S = 128                 # WMMAs per main-loop period (4 subIterK x 32, MIWT 4x8)
MMA = 8                 # clk per WMMA (mma_issue_interval_clk)
P = S * MMA             # 1024 clk period

# gemm_wt's hardware-measured constants for v_wmma_scale_f32_32x16x128_f4 on gfx1250
LEAD = 23
PAIR_GAP = 8            # not binding (no pairing on the scaled WMMA)
ISSUE_SLOTS = 2         # LD_SCALE + WMMA
MIN_GAP = 4
READ_FIXED = 53         # ds_load issue -> s_wait_dscnt release

# Our two representative scale reads (units=1: a single ds_load_b32, 1/4 of a b128)
reads = [
    (4, 32, 1, READ_FIXED),    # scaleMXSA/B group 0: loaded in the hole before WMMA#4, first read at WMMA#32
    (4, 56, 1, READ_FIXED),    # scaleMXSA/B group 3/7 (last group): same load burst, first read at WMMA#56
]

w, iss = queue_wave(S, P, reads, LEAD, pair_gap=PAIR_GAP, issue_slots=ISSUE_SLOTS, min_gap=MIN_GAP)

print(f"period P={P} clk, S={S} WMMAs, lead={LEAD}, issue_slots={ISSUE_SLOTS}, min_gap={MIN_GAP}\n")
for (q, fu, u, lt), i in zip(reads, iss):
    issue_wmma = i // MMA
    lead_wmma = fu - issue_wmma
    rel = i + lt
    slack = fu * MMA - rel          # clk between the read's release and its first-use WMMA's Pre-Read
    print(f"read q={q:3d} first_use(WMMA)={fu:3d}  implied issue clk={i:5d} (~WMMA #{issue_wmma})"
          f"  lead={lead_wmma:3d} WMMA = {lead_wmma*MMA:4d} clk   release=issue+{lt}={rel:5d}"
          f"   slack before first use = {slack:5d} clk")

print("\nFor comparison, the model's own minimum requirement is just lead + read latency "
      f"~= {LEAD + READ_FIXED} clk (~{(LEAD + READ_FIXED + MMA - 1)//MMA} WMMA slots) of lead "
      "for zero stall.")
