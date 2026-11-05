# FINAL STACK (frozen 2026-11-03, user decision; agent "c2", consolidated push 2)

The served Qwen3.8-Flash-Next decode stack at CONC 1 on gfx950 (MI355X, `rocke` partition, TP1), frozen. Ratios only in
this tree; absolute tok/s in `hackathon-prep/private-perf/qwen38-true-mono/quick/FINAL.md` (and CONSOLIDATED2.md).
Every component is a default-off knob of ATOM-truemono / copy-rocm-lib: with none of the variables below set, the server
is the pinned hybrid. Measurements, gates and rejected levers: `quick/README.md` sections "Consolidated stack" and
"Consolidated stack 2". No git state was changed for any of it.

## 1. The frozen arms

| K | arm (`hackathon-prep/qwen38/truemono-quick/arms.d/c2.sh`) | recipe | what |
|---|---|---|---|
| K=3 (MTP, 3 draft tokens) | **`c2s3b`** | `gen_c2Ab` | consolidated full stack (picksH + `qsa:qdense=3` + N8 96K draft vocab + subset-head pick + P9 verify tail) + lean draft round + round-boundary device metadata (TRIM + ADV=1) + WP2c QSA Q3-Q7 functions + WP2a HC functions + WP2b GEMV functions |
| K=0 | **`c2s0b`** | `gen_c2Ab` | the same layer stack + the K=0 fused-argmax tail (no draft, no round-boundary knobs: they need MTP) |

K=0 with W11's N1 "na" (`w11na0`) is NOT frozen in: N1 is a tolerance-class lever and the only T4 run of it
(`w11-t4k3`) is at K=3, which serves the N1 cells qsa r4s / qsa r4 and the drafter's qsa r1 but never delta r1 (35 of the
48 K=0 layers, `delta.r1=Ana1.4+Mna4.2`); the K=0 numerics of N1 are covered by W11's T2 job only (`w11-t2na`). `w11-t4k3` finished 2026-11-03 02:21 UTC and
PASSES T4 at K=3 against the hybrid pool (GSM8K flexible / strict -0.0046 / -0.0044, HumanEval -0.0004, RULER v1
8K / 32K -0.0004 / -0.0017; private `quick/c2-raw/t4_gate_w11na3.txt`), but that does not cover delta r1 at K=0.
Status: **N1 pending** for K=0 (needs a K=0 T4 run and the second node), rejected (NEUTRAL) at K=3 (section 6).

## 2. How to serve it (one GPU, q38 container, `truemono-quick`)

```bash
cd hackathon-prep/qwen38/truemono-quick
# stages (already on the cluster; sha256 in section 4): h-qk-c2-f, rocke-qk-c2-f, atom-truemono-c2-f, hb-atom-c2-f, rocke-tmbase
T=/job/stages/h-qk-c2-f/truemono
Q38_NODE=smci355-ccs-aus-n01-09 bash submit.sh qk-<tag> c2-f --time 03:00:00 \
  --env "ARM_FILES=$T/RB/arms_rb.sh $T/WP2a-fix/arms_wp2a.sh $T/WP2b/arms_wp2b.sh" --env CS_GEN_PP=1 \
  --env WP2B_KNOBS_JSON=picks_all.json --env WP2B_JSON_ONLY=1 \
  --env "ARMS_K3=hybrid hybrid_b c2s3b c2s3b_b" --env "ARMS_K0=hybrid hybrid_b c2s0b c2s0b_b" --env PASSES=4
```

`WP2B_KNOBS_JSON=picks_all.json WP2B_JSON_ONLY=1` are REQUIRED: without them `gen_wp2bHqd` writes the WP2b parity
defaults, not the picks (alola rule 236). On n07-17 add `--env XDG_RUNTIME_DIR=/tmp/vanantha-xdg --env
ENROOT_RUNTIME_PATH=/tmp/vanantha-enroot/runtime` (alola rule 198).

### 2.1 Functions recipe `gen_c2Ab <rocke stage> <FN>` (in order)

1. `gen_wp2bHqd` = `gen_tunedHqd` (lookback/gen_tuned.py over the e2ecomp2 functions with `lookback/picksH.json` and
   `--kind-extra qsa:qdense=3`: per-kind MoE / B3 functions, `sched.txt`, `expect_k*.txt`), then `WP2b/wp2b_gen.py --stk
   $FN --out $FN --variants r1,r2,r3,r4,r2s,r3s,r4s,r4g2,r6g2,r8g2,r6g3,r9g3,r12g3,r8g4,r12g4,r16g4 --knobs-json
   WP2b/picks_all.json --json-only` (GEMV functions `$FN/k/<kind>/gemv/`).
2. `cp $FN/sched.txt $FN/sched.ref.txt`.
3. `WP2a-fix/wp2a_gen.py --stk $FN --picks WP2a-fix/picks_final.json` (HC functions `$FN/hc/<kind>/`; the 12 pick cells'
   schedule groups replace the stack's).
4. `gen_w8c57` with its own base call stubbed: `python -m kernels.gfx950.qwen38_mono.qsa2.gen --out $FN/qsa2 --knobs
   numa=1,qdense=3,prepoll=q5+q6+q7,q5_kpre=True` (all narrow variants), the schedule override `qsa.r1:hold.q7=64`
   (its build lines in `expect_k*.txt` rewritten), then `$FN/qsa2_34` = qsa2's Q3 / Q4 files without r4s.
5. `manifest.sha256` over `k hc qsa2 qsa2_34`.

The 655-file manifest of `$FN` is `hackathon-prep/qwen38/truemono/C2/results/final_functions_c2Ab.manifest.sha256`
(job qk-c2-e4-n01). Two generations from the same stage trees in different jobs gave the same bytes in all files except
the 47 `qsa2/*.bc` bitcode files, whose module ids embed the output path (`/job/fns/c2Ab-<rocke stage>`).

### 2.2 Server environment (K=3 `c2s3b`; `$FN` = the functions dir, `/job/fns/c2Ab-rocke-qk-c2-f`)

```
ATOM_RUNNER_INLINE_SEND=1 ATOM_MONO_ENABLE=1 ATOM_ENABLE_REPLAYSSM=0
ATOM_Q38_SCHED=numa=1;delta.r1:b3pf=1,hold.b1=16,hold.m3=32,spread.a1=10;delta.r4s:b3pf=1,hold.a2=16,hold.m2=64,hold.m3=64,prio.a1=3,prio.a2=2,spread.a1=2;qsa.r4s:dyn.q6=1,dyn.q7=1,head_sleep=128,hold.m1=32,hold.m3=64,hold.q2=192,hold.q5=96,prio.m1=2,prio.q2=2,qdense=3;qsa.r1:hold.m3=128,hold.q1=32,hold.q7=64,qdense=3,spread.a1=10;delta_ple.r4s:hold.a2=32,hold.b1=16,hold.m1=16,hold.m2=64,hold.m3=32,pause=4,prio.a1=1,prio.a2=1,prio.b1=2,prio.b3=2,prio.m5=2,spread.a1=2;delta_ple.r1:b3pf=1,head_sleep=192,hold.a2=96,hold.b1=32,hold.m2=96,prio.m1=1,prio.m2=2,prio.m3=2,prio.p1=2,spread.a1=3;delta.r2s:hold.a2=64,hold.b1=16,hold.m1=128,hold.m2=16,hold.m3=128,pipe=1,prio.b3=2,spread.a1=2;qsa.r2s:dyn.q6=1,head_sleep=128,hold.m3=96,qdense=3,spread.a1=10;delta_ple.r2s:hold.b1=16,spread.a1=1;delta.r3s:hold.a2=96,hold.m1=192,hold.m3=16,prio.m1=3,spread.a1=2,spread.m1=2;qsa.r3s:dyn.q7=1,hold.m2=16,hold.q1=16,hold.q6=128,prio.m5=2,qdense=3,spread.a1=2,spread.m1=2;qsa:qdense=3;delta_ple.r3s:head_sleep=0,hold.a2=16,hold.m1=192,prio.m1=3,spread.a1=2
ATOM_Q38_ROCKE_HEAD=1
ATOM_Q38_EXTERN_MISSING=mono
ATOM_Q38_EXTERN_STAGES=m3=$FN/k/{kind}/moe/m3_{variant}.ll,m4=$FN/k/{kind}/moe/m4_{variant}.ll,m5=$FN/k/{kind}/moe/m5_{variant}.ll,b3=$FN/k/{kind}/b3,b3pre=$FN/k/{kind}/b3,b1=$FN/k/{kind}/gemv/b1_{kind}_{variant}.ll,c1=$FN/k/{kind}/gemv/c1_{kind}_{variant}.ll,q1=$FN/k/{kind}/gemv/q1_{kind}_{variant}.ll,p1=$FN/k/{kind}/gemv/p1_{kind}_{variant}.ll,a1=$FN/hc/{kind},a2=$FN/hc/{kind},m1=$FN/hc/{kind},m2=$FN/hc/{kind},q3=$FN/qsa2_34,q4=$FN/qsa2_34,q5=$FN/qsa2,q6=$FN/qsa2,q7=$FN/qsa2
ATOM_Q38_DRAFT_VOCAB=/job/stages/h-qk-c2-f/truemono-quick/n8/ranking.pt
ATOM_Q38_DRAFT_VOCAB_N=98304
ATOM_Q38_DRAFT_VOCAB_SPEC={"korder":"lane","ctas":256,"xpad":64,"waves_per_eu":1,"fsync":"counter"}
ATOM_Q38_TRUE_MONO=tail
ATOM_Q38_TRUE_MONO_LAYER=kernels.gfx950.qwen38_mono.truemono_tail:layer_passthrough
ATOM_Q38_TRUE_MONO_STATS=100000
ATOM_Q38_TM_TAIL_DRAFT=lean
ATOM_Q38_RB_DEVMETA=1
ATOM_Q38_RB_DEVMETA_TRIM=1
ATOM_Q38_RB_DEVMETA_ADV=1
PYTHONPATH=/atom:/job/stages/rocke-qk-c2-f/platform/python:/job/stages/rocke-qk-c2-f/library
python -m atom.entrypoints.openai_server --model /model --trust-remote-code -tp 1 --server-port <port> --max-num-seqs 4 \
  --max-model-len 10240 --gpu-memory-utilization 0.9 --hf-overrides '{"_multimodal_config": null}' --method mtp --num-speculative-tokens 3
```

K=0 `c2s0b`: the same `ATOM_Q38_SCHED`, `ATOM_Q38_ROCKE_HEAD`, `ATOM_Q38_EXTERN_MISSING`, `ATOM_Q38_EXTERN_STAGES`,
`ATOM_Q38_TRUE_MONO`, `ATOM_Q38_TRUE_MONO_LAYER`, `ATOM_Q38_TRUE_MONO_STATS` and `PYTHONPATH`; no `ATOM_Q38_DRAFT_VOCAB*`,
`ATOM_Q38_TM_TAIL_DRAFT`, `ATOM_Q38_RB_DEVMETA*`, `ATOM_ENABLE_REPLAYSSM`; no `--method mtp`. Exact env lines as the jobs
printed them: `hackathon-prep/qwen38/truemono/C2/results/final_env_k3.txt`, `final_env_k0.txt`. The ATOM tree is
`/atom` = stage `atom-truemono-c2-f`; the cwd and first PYTHONPATH entry is `/atom`.

## 3. Picks and schedule files (hackathon-prep/qwen38, as staged in `h-qk-c2-f`; sha256 of the LF bytes, first 16 hex)

| file | sha256 |
|---|---|
| `truemono-quick/lookback/picksH.json` | 3862b35159a5c710 |
| `truemono-quick/n8/ranking.pt` | 2ae5f3318217bfe2 |
| `truemono/WP2a-fix/picks_final.json` | d15b817ed51743bc |
| `truemono/WP2b/picks_all.json` (the 2026-11-02 04:27 UTC version; the later local file differs only in r8g4 / r16g4 / r2s / r9g3 cells, none served at CONC 1) | 2add3883d72075cd |
| `truemono-quick/arms.d/c2.sh` | 43af0208f9836a28 |
| `truemono-quick/arms.d/cs.sh` | 02ebd0d1100a8160 |
| `truemono-quick/arms.d/tuned.sh` | 3087852b6373124e |
| `truemono-quick/arms.d/w8c.sh` / `w8c2.sh` | 145bc60509f7875f / 629d3a4aa0553c60 |
| `truemono-quick/arms.sh` / `quick.sh` | eaac2c3e4c313056 / 563ca1e5c949f9eb |
| `truemono-quick/lookback/gen_tuned.py` | 42618a1d3a4a1ce9 |
| `truemono/WP2a-fix/arms_wp2a.sh` / `wp2a_gen.py` | 8bf16816341894aa / 74b30564bd5dd71e |
| `truemono/WP2b/arms_wp2b.sh` / `wp2b_gen.py` | 3c6384bbebd0c4b9 / dffac55643661340 |
| `truemono/RB/arms_rb.sh` | 0a06d9ec07db6fde |
| `e2ecomp2/gen.sh` / `common/server.sh` | 6cd4f911da1b31ed / 3f7ac68e50a53f00 |

Full per-file manifest of the harness stage: `hackathon-prep/qwen38/truemono/C2/results/stage-h-qk-c2-f.manifest`.

## 4. Stage trees (cluster `/home_aus/AMD/vanantha/rocke-sdpa-claude/q38/stages/<name>.tar`, sha256 of the tar)

| stage | sha256 | content |
|---|---|---|
| `rocke-qk-c2-f` | 85db9513011efd235e1181555b8c7845f0a16709bda1c779407965448c556b43 | copy-rocm-lib rocKE, byte copy of `rocke-qk-w8wp2afx` (staged 2026-11-02 04:29 UTC); manifest `C2/results/stage-rocke-qk-w8wp2afx.manifest` |
| `atom-truemono-c2-f` | 869f0610025316aaa6c950b0a1d926f40dc8def265caef88072084cb95a9c442 | dev/ATOM-truemono, byte copy of `atom-truemono-w8wp2afx` (04:28 UTC); manifest `C2/results/stage-atom-truemono-w8wp2afx.manifest` |
| `hb-atom-c2-f` | a48b477fc924b70af359a081d91cfecbdb4c2383bdd0e7fc721652ea738b1410 | the pinned hybrid ATOM (= `atom-rkrepro`) |
| `rocke-tmbase` | f6f644d4f0a1c8805c4876791556898d9b941c78d18e4f7c4fe5533c8a3fb6d5 | the pinned hybrid rocKE (copy-rocm-lib 89209ac) |
| `h-qk-c2-f` | ce005780d12f7bad (tar sha, first 16) | the harness (section 3) |

rocKE files the stack exercises (copy-rocm-lib `dnn-providers/hip-kernel-provider/rocke/`, sha256 of the staged LF bytes):

| file | sha256 |
|---|---|
| `library/kernels/gfx950/qwen38_mono/stages2/__init__.py` | 4680ec877a6c2b48 |
| `library/kernels/gfx950/qwen38_mono/stages2/hc.py` (WP2a) | f6f352861ca8d8ff |
| `library/kernels/gfx950/qwen38_mono/stages2/gemv.py` (WP2b; the local file has a later appended Path-B factory, alola rule 228) | ef9da99e03469c79 |
| `library/kernels/gfx950/qwen38_mono/qsa2/{__init__,build,ctx,gen,knobs,layout,lower,q34,q567}.py` (WP2c) | 5bd24dbdf9f2647a, cc4639a0f93f0a9f, 44f76c648bc3a860, a049b17db222bc6d, 31f036b9fca8624a, f80f5de250199f1a, 87801e8fcd05fc7b, 8badf5e20a254dc6, 2093d4df27721fcd |
| `library/kernels/gfx950/qwen38_mono/truemono_tail.py` (tail provider) | 6eb93a6b1848f78f |
| `library/kernels/gfx950/qwen38_mono/probes/p9_tail.py` (P9 kernels) | 10f3dbffe08820a2 |
| `library/kernels/gfx950/qwen38_mono/stage_moe_fused.py` / `stage_b3.py` / `mono_layout.py` | 5fea527156ef0c4a / 594b1a803ece041c / b2f8b251ac56d742 |

ATOM-truemono files (the served hooks; all additive and default-off):

| file | sha256 |
|---|---|
| `atom/model_engine/model_runner.py` (RB devmeta / host timeline hooks, tail hook call sites) | d4167c2619b6f976 |
| `atom/model_engine/engine_core.py` / `async_proc.py` | aaebf1e0a38bccc7 / dd8fded19eef7d94 |
| `atom/models/qwen4_exp_mono/truemono/{tail,hook,slots,rb_devmeta,hosttl,__init__}.py` | 144e2dbb336eb660, bfdd6a1bab6f624d, eb985d45e4165043, 81a441f19e97d68a, fb9fd152274fd1d6, 54e7c03432b66378 |
| `atom/models/qwen4_exp_mono/kernels/{extern_stages,kernel,sched,numa,layout}.py`, `kernels/stages/fold.py` | 8c4cdf3caba10612, 32be269f05a73434, 1db5d77a2069f02a, a77530ab3644cae6, 0a1f31ff6104fa39, d9d7bd35f439808e |

Note: the local `dev/ATOM-truemono/atom/models/qwen4_exp_mono/truemono/rb_devmeta.py` was edited after the freeze copy
(ADV=2 / ADV=3 / TRIM2 work of the round-boundary lane, all default-off); the frozen stack uses the staged copy above.

## 5. Gates the frozen arms passed

- Stacked layer gate (`truemono/C2/c2_layer_gate.py`, jobs qk-c2-gate3 / gate4): every served narrow cell (delta,
  delta_ple, qsa at r1 / r4s; qsa r4 draft; qsa at L 1024 and 32,768) bitwise against the previous stack's layer and the
  plain mono, deterministic, every extern linked; HC `mix_avg`, WP2b C1 `put_bit`, QSA `q7_gate` mutants caught.
- CS tail gate and RB gate (unchanged drivers, bases re-pointed): ATOM check mode 64 / 64, every deep check equal, RB
  check mode 64 / 64 with ADV, d3la deep 1,728 / 1,728, K=1 and K=0 clean, must-fail mutants caught.
- T4 (P11 rev 3, vs the hybrid pool) on `c2s3b`: GSM8K flexible / strict, HumanEval, RULER mix v1 8K / 32K PASS.
- T2 (P11 rev 3, K=0, composed share 1.0, k 4 / n 5) on the K=0 layers: PASS; planted must-fail FAILS.

## 6. Measured (ratios; 2 jobs x 4 rotated passes per node, same-job hybrid with A-vs-A twin, medians over requests)

| | n01-09 | n07-17 |
|---|---|---|
| K=3 `c2s3b` / hybrid, ms per verify round | 0.888 | 0.887 |
| K=3 `c2s3b` / hybrid, tok/s per request | 1.135 | 1.121 |
| K=0 `c2s0b` / hybrid, tok/s per request | 1.135 | 1.139 |

(Pooled 8 pairs per node, DESIGN 9.1 WIN on both metrics on both nodes; jobs qk-c2-e1-n01 + f1-n01, qk-c2-e2-n07 +
f2-n07.) Against the targets of PLAN decision 7: K=3 1000 tok/s per request NOT met (absolute in the private file), K=0
0.78 of the hybrid's per-token time NOT met (0.88).

### Rejected on base c2s3b / c2s0b (default-off knobs, kept; PLAN decision 15: "rejected on base c2s3b")

| lever | e2e on base c2s3b (pooled pairs, both nodes unless noted) | verdict |
|---|---|---|
| device-advanced round, ADV=3 + TRIM2 (`c2s3b_a3`, rb lane; jobs rb-e2e6-n01, rb-e2e7-n07) | round 1.000 (4/8), tok/s 0.999 | NEUTRAL: rejected |
| TRIM2 alone (`c2s3b_t2`; rb-e2e5-n01 / n07) | round 0.997 (7/8), tok/s 1.011 | NEUTRAL: rejected |
| N8 64K draft vocab (`c2s3b_n8_64k`; six jobs: qk-c2-e4-n01 / n07, rb-e2e5/6/7) | round 0.996 (21 of 23 pairs below 1, median inside the twins' band), tok/s 1.005 | NEUTRAL: rejected |
| TRIM2 + N8 64K (rb-e2e5, 2 jobs) | round 0.994 (8/8, at the band edge), tok/s 1.009 | not adopted (frozen before a confirmation) |
| N1 at K=3 (`w11na3` / `w11ks3`; jobs w11-e3k3-n07, w11-e4k3-n01) | round 0.999 / 0.997, tok/s 1.012 / 1.004 | NEUTRAL: rejected at K=3 |
| N1 at K=0 (`w11na0`, job w11-e3k0-n01, n01-09 only) | tok/s 1.035 (4/4, WIN on one node) | **pending**: K=0 T4 not run (section 1), second node not run |
| N2 out-proj K-split into B3 (`w17n3`; jobs qk-w17-e1-n01 / n07) | round 1.012 (0/8), in situ 1.02-1.07 per layer | LOSS: rejected |
| schedule re-climb with the rocKE stages in place (`c2u3` / `c2u0`, C2/c2t_sched_s2.txt) | K=3 round 1.002; K=0 per token 0.995 (inside the band) | NEUTRAL: rejected |
| P16 fold.b1 (K=0) | 0.997 on the no-WP2b base (inside the band); not built with WP2b's B1 | NEUTRAL: rejected |
| P9 hdelay 8 / 16, head_dg 3, P9 draft chain, WP2c all-five at r4s, WP2a port defaults | 0.998-1.003 | NEUTRAL: rejected |
| verify tail only (no lean round), TRIM without ADV | 1.014 / 1.005 | LOSS: rejected |
