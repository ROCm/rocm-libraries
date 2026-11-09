# Qwen3.8-Flash-Next true rocKE monokernel stack: reproduction (TP1, MI355X)

## Reproduce on your own MI355X

You need:
- one MI355X (gfx950) on Linux, with docker (or podman: replace `docker` below);
- about 200 GB of free disk for the image, the venv and the runs, plus the checkpoint;
- the checkpoint `Qwen3.8-Flash-Next-MXFP4`. ATOM's `scripts/qwen38_flash_mono/repro.sh convert` builds it from
  `Qwen/Qwen3.8-Flash-Next` on Hugging Face (335 GiB download, about 500 GiB of disk while it converts).

Everything else is in this directory:
- `patches/` holds three plain `git apply` patches against pinned public commits;
- `knobs/` holds the server environment of every arm;
- `kit/` holds the few inputs that live in no repository: the picks, the schedule recipe, the N8 ranking, the
  generator scripts and the benchmark driver. `kit/FILES.txt` says what each file is, and `kit/SHA256SUMS` lists their
  hashes.

**The patches apply only to the pinned commits.** They do NOT apply to ATOM `main`, which has no
`atom/models/qwen4_exp_mono` (patch 01 fails on `df86c482` with the qwen4_exp_mono files missing). Use the
`qwen38-megakernels` commit below.

### 1. Trees (ATOM 352229d6 + 01 + 02, rocm-libraries 4478c174a13 + 03, the hybrid's 01 and 89209ac)

```bash
REPRO=/path/to/copy-rocm-lib/experiments/qwen38-true-mono/repro     # this directory
T=$HOME/q38-c2; mkdir -p $T && cd $T
# ATOM: the stack's tree (01 + 02) and, as a second worktree, the hybrid's tree (01 only)
git clone --branch qwen38-megakernels https://github.com/proaditya/ATOM.git atom-truemono
git -C atom-truemono checkout --detach 352229d6d4a756cf58146bc5cb37519dd5d5aefd
git -C atom-truemono worktree add --detach ../atom-hybrid 352229d6d4a756cf58146bc5cb37519dd5d5aefd
git -C atom-truemono apply $REPRO/patches/01-atom-hybrid-rocke-on-352229d6.patch
git -C atom-truemono apply $REPRO/patches/02-atom-truemono-c2-on-01.patch
git -C atom-hybrid   apply $REPRO/patches/01-atom-hybrid-rocke-on-352229d6.patch
# rocKE: only its directory of rocm-libraries; the stack's commit + 03, and the hybrid's commit unpatched
git clone --filter=blob:none --no-checkout --branch users/vanantha/rocke-monokernels \
  https://github.com/ROCm/rocm-libraries.git rocm-libraries
git -C rocm-libraries sparse-checkout set dnn-providers/hip-kernel-provider/rocke
git -C rocm-libraries checkout --detach 4478c174a136aa5d3ff76e70b0a53da2dc64c91e
git -C rocm-libraries apply $REPRO/patches/03-rocke-c2f-on-4478c174a13.patch
git -C rocm-libraries worktree add --no-checkout --detach ../rocm-libraries-hyb 89209ac76656aede13abd8bb4e5e6ec371ed6a1f
git -C rocm-libraries-hyb sparse-checkout set dnn-providers/hip-kernel-provider/rocke
git -C rocm-libraries-hyb checkout --detach 89209ac76656aede13abd8bb4e5e6ec371ed6a1f
(cd $REPRO/kit && sha256sum -c --quiet SHA256SUMS) && echo kit OK
```

The trees are file for file the trees the measured jobs ran. `python3 $REPRO/tools/tree_vs_manifest.py <tree>
$REPRO/manifests/<stage>.manifest --ignore .git` checks it. The one expected difference is
`atom/entrypoints/openai/serving_anthropic.py`, a non-qwen change that the patches leave out.

### 2. Container, venv and checkpoint (once)

```bash
export WORK=$T/work GPU=0                                    # GPU = the HIP device index of the MI355X to use
bash $T/atom-hybrid/scripts/qwen38_flash_mono/repro.sh setup
# setup does the following:
#   - pulls rocm/pytorch@sha256:bbdaba66029f905321be3bbc95206a2d9a56a2bc5300877d146ad24d40db9a12 (ROCm 10.0, torch 2.13);
#   - builds $WORK/venv: aiter c53e9b3f, FlyDSL 0.3.4.1 (repro-constraints.txt), ATOM editable;
#   - checks that the GPU is gfx950.
MODEL=/path/to/Qwen3.8-Flash-Next-MXFP4
(cd $MODEL && sha256sum -c --quiet $T/atom-hybrid/scripts/qwen38_flash_mxfp4/Qwen3.8-Flash-Next-MXFP4.sha256) && echo checkpoint OK
```

### 3. Generate the stage functions and run the benchmark

One `docker run` lays the trees out the way the measured jobs saw them, with the stack's ATOM at `/atom`, the other
trees under `/job/stages`, and the kit. It then runs `kit/in_container.sh`:
- `gen` regenerates the stage functions with the recipe `gen_c2Ab` (and the hybrid's `gen_e2ecomp2`), then compares
  them with the measured functions (`kit/expected`). The `.bc` files are left out of the compare, because they embed
  the output path.
- `bench` runs `gen`, then the measured benchmark driver unchanged.
- `k0` does the same as `bench`, at K=0.

```bash
RUN=$T/runs/$(date -u +%Y%m%d-%H%M%S); mkdir -p $RUN
RK=dnn-providers/hip-kernel-provider/rocke
docker run --rm --init --ulimit nofile=65535:65535 --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=$GPU -e PASSES=4 \
  -v $T/atom-truemono:/atom -v $WORK:/work -v $MODEL:/model:ro -v $RUN:/job -v $REPRO/kit:/kit \
  -v $T/atom-hybrid:/job/stages/hb-atom-sa -v $REPRO/kit/harness:/job/stages/h-qk-sa \
  -v $T/rocm-libraries/$RK:/job/stages/rocke-qk-sa -v $T/rocm-libraries-hyb/$RK:/job/stages/rocke-tmbase \
  rocm/pytorch@sha256:bbdaba66029f905321be3bbc95206a2d9a56a2bc5300877d146ad24d40db9a12 \
  bash /kit/in_container.sh bench          # or: gen (functions only, ~1 min), k0 (the K=0 arms)
```

`bench` serves `hybrid`, `hybrid_b`, `c2s3b` and `c2s3b_b` at MTP K=3, CONC 1, TP1:
- one untimed prime server, then `PASSES` (default 4) seeded passes with the arm order rotated;
- one fresh server per arm-run, and 2 untimed warm-up requests per run;
- 32 GSM8K prompts (the DFlash seed-42 selection), greedy, reasoning `xhigh`, a 2048-token cap;
- the server flags of `FINAL-STACK.md` section 2.

`hybrid_b` and `c2s3b_b` are A-vs-A twins: the same configuration under a second name. They show the session's
noise. Every server log is checked: mono on, rocKE head on, every expected build line present, no tracebacks.

The results land in `$RUN/out/q/`: `ratios.json` holds the ratios, `abs.json` the absolutes. `kit/verdict.py`
prints PASS or FAIL against the bands below.

Run time on one MI355X:
- `setup`: about 20-40 min, once (image pull and aiter build);
- `gen`: about 1 min;
- `bench` at 4 passes: about 75 min. One arm-run of the stack takes about 4-4.5 min, because its layer builds
  JIT-compile at every server start. One hybrid arm-run takes about 2.5-3.5 min.

### 4. Expected result (same-session ratios; `kit/verdict.py` checks them)

| check | band |
|---|---|
| K=3 `c2s3b` / `hybrid`, decode tok/s per request (median over requests, mean over passes) | 1.11-1.15 |
| K=3 `c2s3b` / `hybrid`, ms per verify round (median) | 0.88-0.895 |
| K=0 `c2s0b` / `hybrid`, tok/s per request (`k0` mode) | 1.12-1.15 |
| twins `hybrid_b` / `hybrid`, `c2s3b_b` / `c2s3b` | tok/s within about ±1.5 %, round within about ±0.5 % |

Decide on the medians and on 4 passes. Per-request tok/s also carries 2-4 % of run-to-run acceptance noise, because
the served path is not token-deterministic. The twins show how much of a ratio is noise in your session. This tree
states ratios only.

### Serving one arm by hand (hybrid vs c2s3b)

Use this inside the same container, after `gen` (`$FN` = `/job/fns/c2Ab-rocke-qk-sa`, `$FN_HYB` =
`/job/fns/e2ecomp2-rocke-tmbase`, both written by `gen`). `knobs/<arm>.env` holds each arm's full environment, with
`$FN`, `$FN_HYB`, `$H` (= `/job/stages/h-qk-sa`) and `$RK` (= `/job/stages/rocke-qk-sa`) as placeholders. Every server
uses the same command line:

```bash
COMMON="ATOM_RUNNER_INLINE_SEND=1 ATOM_MONO_ENABLE=1 ATOM_ENABLE_REPLAYSSM=0"
ARGS="--model /model --trust-remote-code -tp 1 --server-port 8000 --max-num-seqs 4 --max-model-len 10240 \
  --gpu-memory-utilization 0.9 --method mtp --num-speculative-tokens 3"
# hybrid (the bar): ATOM 352229d6 + 01, rocKE 89209ac
cd /job/stages/hb-atom-sa && env $COMMON ATOM_Q38_SCHED=best ATOM_Q38_ROCKE_HEAD=1 ATOM_Q38_EXTERN_MISSING=mono \
  "ATOM_Q38_EXTERN_STAGES=m3=$FN_HYB/moe/m3_{variant}.ll,m4=$FN_HYB/moe/m4_{variant}.ll,m5=$FN_HYB/moe/m5_{variant}.ll,b3=$FN_HYB/b3,b3pre=$FN_HYB/b3" \
  PYTHONPATH=/job/stages/hb-atom-sa:/job/stages/rocke-tmbase/platform/python:/job/stages/rocke-tmbase/library \
  python -m atom.entrypoints.openai_server $ARGS --hf-overrides '{"_multimodal_config": null}'
# c2s3b (the stack): ATOM + 01 + 02 at /atom, rocKE 4478c174a13 + 03; every ATOM_Q38_* line of knobs/c2s3b.env
cd /atom && env $COMMON "ATOM_Q38_SCHED=$(cat $FN/sched.c2.txt)" ATOM_Q38_ROCKE_HEAD=1 ATOM_Q38_EXTERN_MISSING=mono \
  "ATOM_Q38_EXTERN_STAGES=$(grep '^ATOM_Q38_EXTERN_STAGES=' $REPRO_KNOBS/c2s3b.env | cut -d= -f2- | sed "s#\$FN#$FN#g")" \
  ATOM_Q38_DRAFT_VOCAB=/job/stages/h-qk-sa/truemono-quick/n8/ranking.pt ATOM_Q38_DRAFT_VOCAB_N=98304 \
  'ATOM_Q38_DRAFT_VOCAB_SPEC={"korder":"lane","ctas":256,"xpad":64,"waves_per_eu":1,"fsync":"counter"}' \
  ATOM_Q38_TRUE_MONO=tail ATOM_Q38_TRUE_MONO_LAYER=kernels.gfx950.qwen38_mono.truemono_tail:layer_passthrough \
  ATOM_Q38_TRUE_MONO_STATS=100000 ATOM_Q38_TM_TAIL_DRAFT=lean \
  ATOM_Q38_RB_DEVMETA=1 ATOM_Q38_RB_DEVMETA_TRIM=1 ATOM_Q38_RB_DEVMETA_ADV=1 \
  PYTHONPATH=/atom:/job/stages/rocke-qk-sa/platform/python:/job/stages/rocke-qk-sa/library \
  python -m atom.entrypoints.openai_server $ARGS --hf-overrides '{"_multimodal_config": null}'
```

Notes on the hand-served commands:
- `REPRO_KNOBS` is wherever `knobs/` is mounted; add `-v $REPRO/knobs:/knobs` to the `docker run` and set
  `REPRO_KNOBS=/knobs`.
- `$FN/sched.c2.txt` equals `ATOM_Q38_SCHED` of `knobs/c2s3b.env`.
- Measure with ATOM's own client:
  `python -m atom.benchmarks.qwen38_flash_mono.single_request_bench --base-url http://127.0.0.1:8000 --model /model --dataset gsm8k --num-prompts 32 --max-tokens 2048 --chat-template-kwargs '{"reasoning_effort": "xhigh"}'`.
- `c2s0b` (K=0) uses the same env without `ATOM_Q38_DRAFT_VOCAB*`, `ATOM_Q38_TM_TAIL_DRAFT`, `ATOM_Q38_RB_DEVMETA*`
  and `ATOM_ENABLE_REPLAYSSM`, and without `--method mtp --num-speculative-tokens 3`.

### Troubleshooting

- **No GPU in the container / wrong arch.** `in_container.sh` stops when torch sees no device, or sees one that is not
  gfx950. Check `--device=/dev/kfd --device=/dev/dri`, the `video` / `render` groups and `HIP_VISIBLE_DEVICES`.
- **`GEN FAIL`.** Every generated `.ll`, `.json` and sched file must equal the measured bytes. A failure means that a
  tree, the kit or the container differs: re-check the commits, `git apply`, `kit/SHA256SUMS` and the image digest.
- **Container COMGR, `waves=8`.** The image's COMGR (ROCm 10) miscompiles some 512-thread large-batch QSA kernels.
  None of them is in this stack, and `gen` plus the server-log checks cover the served set. Any new lever that adds
  such kernels must ship code objects built outside the container.
- **The `hz` `s_nop` knob.** Inline-asm VMEM after a VALU write of its SGPR needs `s_nop 4`, which is the rocKE knob
  `hz` (default on). Never turn it off. The served CONC-1 functions of this stack do not contain that pattern.
- **Noise and the compile lottery.** One recompile can shift a layer by about 1 %, and acceptance moves per-request
  tok/s by 2-4 %. Use 4 passes, decide on the medians, and read the twins: if a twin is outside its band, the session
  was noisy, so re-run before you judge the stack. A busy neighbour on the same GPU node also inflates means; the
  per-request medians resist it.
- **Means vs best pass.** A single pass can land above the 4-pass mean. Quote the 4-pass mean of the same-session
  ratio, never the best pass.

## The package in detail (and the alola cluster modes)

This package reproduces the current verified best stack of the true-mono project, "consolidated stack 2". It has
two arms: **`c2s3b`** at MTP K=3 (with **`c2s3b_n8_64k`**, the K=3 integration base) and **`c2s0b`** at K=0. Both
are measured against the pinned hybrid ALL stack ([Qwen3.8-Flash-Next-rocKE.md](../../../../ATOM/recipes/Qwen3.8-Flash-Next-rocKE.md)
in ATOM) in the same job. The package holds:
- this recipe;
- patches against pinned commits;
- the server environment of every arm;
- a single script, `repro.sh`, with check / gen / gate / quick / full modes.

The trees it builds are the exact trees the measured jobs ran: a check compares them file by file with those jobs'
stage manifests.

This tree states ratios only. The absolute expected numbers, per node and with job ids, are in the private store:
`hackathon-prep/private-perf/qwen38-true-mono/repro/EXPECTED.md`. The measurements and their history are in
[../quick/README.md](../quick/README.md): "Consolidated stack 2", "Round boundary" and "Consolidated stack".

## What the stack is

Each piece is a default-off knob. With the knobs unset, the server is the hybrid. The pieces, bottom up:

| layer | piece | numerics class | where measured |
|---|---|---|---|
| layer kernels (FlyDSL mono, ATOM) | the hybrid's schedule levers (NUMA dual-copy mailboxes, holds), plus the in-situ lookback picks for every CONC-1 cell (`picksH`: delta / qsa / delta_ple at r1, r2s, r3s, r4s, with the P3 HC spread and the P10 hold drops where they won) | bitwise | quick README "3a", "Follow-up" |
| | P13 N4a `qsa:qdense=3` (dense QSA under budget) | tolerance class (T2 / T4 gated) | "Stack with qdense" |
| stage functions linked into the layer (rocKE, `ATOM_Q38_EXTERN_STAGES`) | fused MoE M3 / M4 / M5 and gated delta B3 (+ prologue): the hybrid's, regenerated by the stack's recipe | bitwise | the hybrid recipe |
| | **WP2a**: HC stages A1 / A2 / M1 / M2 (`stages2/hc.py`, `WP2a-fix/picks_final.json`: 12 cells) | bitwise | [../wave2/WP2a.md](../wave2/WP2a.md) |
| | **WP2b**: GEMV stages B1 / C1 / Q1 / P1 (`stages2/gemv.py`, `WP2b/picks_all.json` as staged 2026-11-02 04:27 UTC, 48 cells) | bitwise | [../wave2/WP2b.md](../wave2/WP2b.md) |
| | **WP2c**: QSA Q5-Q7 at every narrow width; Q3 / Q4 everywhere except r4s, where the FlyDSL Q3 / Q4 stay; `numa=1,qdense=3,prepoll=q5+q6+q7,q5_kpre=True`; `qsa.r1:hold.q7=64` (`qsa2/`) | bitwise | [../wave2/WP2c.md](../wave2/WP2c.md) |
| head / tail | rocKE `lm_head` (fused argmax); N8 draft vocab (96K rows of a frequency-ranked subset, `n8/ranking.pt`) plus the tuned subset head; the P9 verify tail (`truemono_tail.py`, `ATOM_Q38_TRUE_MONO=tail`) at K=3; the fused-argmax tail at K=0 | verify side bitwise; draft side judged on acceptance | "Consolidated stack" |
| round boundary (K=3) | lean draft round (`ATOM_Q38_TM_TAIL_DRAFT=lean`) + `ATOM_Q38_RB_DEVMETA=1` with `_TRIM=1` and `_ADV=1`: round t+1 is prepared without the host waiting for round t's verdict | bitwise (the check mode compares 60 tensors per step) | "Round boundary" |

`c2s3b_n8_64k` is `c2s3b` with `ATOM_Q38_DRAFT_VOCAB_N=65536`. It wins on round time on both nodes; on tok/s per
request it is neutral, because acceptance is 0.5 % lower. It is the K=3 base that later levers stack on.

## What to check out

| piece | where | version |
|---|---|---|
| ATOM | `github.com/proaditya/ATOM`, branch `qwen38-megakernels` | `352229d6d4a756cf58146bc5cb37519dd5d5aefd` + `patches/01-*.patch` (the hybrid rocKE work, as the hybrid recipe lists it) + `patches/02-*.patch` (the true-mono work). Hybrid arm: 01 only. |
| rocKE | `github.com/ROCm/rocm-libraries`, branch `users/vanantha/rocke-monokernels`, dir `dnn-providers/hip-kernel-provider/rocke` | `4478c174a136aa5d3ff76e70b0a53da2dc64c91e` + `patches/03-*.patch` (4 files edited after that commit: `ccl/wire.py`, `large_batch/moe_lb2.py`, `tune/catalog.py`, `tests/test_qwen38_tune.py`; none of them is on the served path) |
| hybrid rocKE (the bar) | same branch | `89209ac76656aede13abd8bb4e5e6ec371ed6a1f`, unchanged (the cluster stage `rocke-tmbase`) |
| branch base | `origin/develop` | merge-base `9b4fee7d99071ca2e40ae74dfa043919fb35d90a` |
| container | `rocm/pytorch@sha256:bbdaba66029f905321be3bbc95206a2d9a56a2bc5300877d146ad24d40db9a12` (ROCm 10.0, torch 2.13), with ATOM's repro venv (`scripts/qwen38_flash_mono/repro.sh setup`: aiter `c53e9b3f`, FlyDSL `0.3.4.1`) | as in the hybrid recipe |
| checkpoint | `Qwen3.8-Flash-Next-MXFP4` | `scripts/qwen38_flash_mxfp4/Qwen3.8-Flash-Next-MXFP4.sha256` |
| harness | `hackathon-prep/qwen38/truemono-quick/` + `truemono/{C2,CS,RB,WP2a-fix,WP2b}`, frozen as the stage `h-qk-c2-f` | sha256 `ce005780d12f7bad...`. A copy ships as `hackathon-prep/qwen38/truemono-repro/frozen/h-qk-c2-f.tar`. The local harness has moved on since (RB t2 work, a newer WP2b `picks_all.json`), so use the frozen copy. |

The later commits on the branch (`f0cd03418e9`, `7d863a231eb`) also carry newer, unverified work: W11 N1 / N2, the
wide builds, Rust levers, and changes to `stages2/hc.py` and `persist/`. Do not serve the stack from the branch
head. The patches pin the measured trees.

How the trees were obtained, read-only:
```bash
git -C ATOM -c core.autocrlf=false archive 352229d6 | tar -x -C atom          # + patch 01 (+ 02)
git -C rocm-libraries -c core.autocrlf=false archive 4478c174a13 dnn-providers/hip-kernel-provider/rocke | tar -x -C rk
(cd atom && git apply --check -p1 01.patch && patch -p1 < 01.patch)          # the same for 02 and 03
```
`repro.sh check` does exactly this and compares every file with the measured stages' manifests (`manifests/`). On
2026-11-02 the result was:
- hybrid ATOM: 1924 files, 0 differences;
- ATOM-truemono: 1941 files, 0 differences;
- rocKE: 2675 files, 0 differences;
- hybrid rocKE `89209ac`: 2400 files, 0 differences.

That count excludes the repo's three symlinks (`.clinerules`, `CLAUDE.md`, `.cursor/rules/ai-rules.mdc`), `.ruff_cache/`
and the one non-qwen file below. The stage packer keeps CRLF in files without a text extension (`.rs`, `.html`, ...)
where the Windows work tree had it. The compare therefore ignores CR. Python never loads those files.

### The Rust crate and the rust-compiler branch

`platform/rust/rocke-codeobj` (the direct code-object compiler) is part of the rocKE tree at `4478c174a13`. The stack
does **not** use it. Every arm compiles through rocKE's Python backend and the container's COMGR. The direct path is
opt-in, through `ROCKE_CODEOBJ_DIRECT_LIB` pointing at a built `librocke_codeobj_ffi.so` (`runtime/code_object.py`),
and no arm sets it.

The branch `users/vanantha/rocke/rust-compiler` (head `23c165f8b3f`, forked from develop at `9b4fee7d990`) is
optional as well: nothing in this package builds or loads it.

### Non-qwen changes (left out of the patches)

- ATOM `atom/entrypoints/openai/serving_anthropic.py`: system messages inside `messages` for the Anthropic endpoint.
  It is in both measured ATOM stages, but it is not on the decode path.
- copy-rocm-lib working tree: whatever `repro.sh files` lists under "other uncommitted changes". These are other
  lanes' live edits (large-batch `ple_lb.py`, `head_lb.py`, docs). They are not part of the stack.
  `patches/pending/copy-rocm-lib-worktree-vs-HEAD.patch` snapshots them.

`FILES-ATOM.txt` and `FILES-copy-rocm-lib.txt` list the files for `git add`, per patch.

## The knobs, per arm

The resolved server environment of each arm is in `knobs/<arm>.env`. It is taken from the measured jobs' `arms.txt`
(`qk-c2-e4-n01` for the K=3 arms, `qk-c2-f1-n01` for `c2s0b`), with the paths templated as `$FN`, `$H` and `$RK`. The
harness builds these environments itself (`arms.d/c2.sh`: `arm_c2s3b`, `arm_c2s3b_n8_64k`, `arm_c2s0b`; `arms.sh`:
`arm_hybrid`). The files are for reading and for linting.

| knob | hybrid | c2s3b (K=3) | c2s3b_n8_64k | c2s0b (K=0) |
|---|---|---|---|---|
| `ATOM_Q38_SCHED` | `best` | the recipe's `sched.c2.txt` (`numa=1` + the picksH cell groups + `qsa:qdense=3` + `qsa.r1:hold.q7=64`) | = c2s3b | = c2s3b |
| `ATOM_Q38_ROCKE_HEAD` | 1 | 1 | 1 | 1 |
| `ATOM_Q38_EXTERN_MISSING` | `mono` | `mono` | `mono` | `mono` |
| `ATOM_Q38_EXTERN_STAGES` | m3 / m4 / m5 / b3 / b3pre (`gen_e2ecomp2`) | per kind: m3-m5, b3, b3pre, b1, c1, q1, p1 (`k/{kind}/`), a1 / a2 / m1 / m2 (`hc/{kind}`), q3 / q4 (`qsa2_34`), q5-q7 (`qsa2`) | = c2s3b | = c2s3b |
| `ATOM_Q38_DRAFT_VOCAB` / `_N` / `_SPEC` | - | `n8/ranking.pt` / 98304 / `{"korder":"lane","ctas":256,"xpad":64,"waves_per_eu":1,"fsync":"counter"}` | 65536 | - |
| `ATOM_Q38_TRUE_MONO` / `_LAYER` / `_STATS` | - | `tail` / `kernels.gfx950.qwen38_mono.truemono_tail:layer_passthrough` / 100000 | = c2s3b | = c2s3b (K=0 tail) |
| `ATOM_Q38_TM_TAIL_DRAFT` | - | `lean` | `lean` | - |
| `ATOM_Q38_RB_DEVMETA` / `_TRIM` / `_ADV` | - | 1 / 1 / 1 | 1 / 1 / 1 | - |
| `PYTHONPATH` | hybrid ATOM + rocKE `89209ac` | ATOM-truemono + rocKE (patched `4478c174a13`) | = c2s3b | = c2s3b |

The functions directory `$FN` comes from the recipe `gen_c2Ab` (`arms.d/c2.sh`). It runs four steps:
1. tunedHqd: `lookback/gen_tuned.py` with `lookback/picksH.json` and `--kind-extra qsa:qdense=3`;
2. WP2b: `WP2b/wp2b_gen.py` with `picks_all.json`;
3. WP2a: `WP2a-fix/wp2a_gen.py` with `picks_final.json`, plus the picks' HC schedule groups;
4. WP2c: `gen_w8c57`, with `numa=1,qdense=3,prepoll=q5+q6+q7,q5_kpre=True` and `qsa.r1:hold.q7=64`.

It writes `sched.ref.txt` (the stack's schedule before the stage ports), `sched.c2.txt` (the served one),
`expect_k<K>.txt` (the build lines the server log must show) and `manifest.sha256`.

The job env must carry `WP2B_KNOBS_JSON=picks_all.json WP2B_JSON_ONLY=1`. Without them the GEMV functions are the
WP2b parity defaults, which are about 4 % slower at delta r4s (alola rule 236). It must also carry `CS_GEN_PP=1`
(see the pitfalls).

`ATOM_Q38_RB_DEVMETA` modes:
- `1` (served): the device-side next-round metadata;
- `check`: every eligible step is prepared twice and compared bit for bit; `_CHECK_EVERY=N` checks every Nth step,
  so production steps run in between;
- `_FUSE` (default 1): v2, one kernel; `0` = v1;
- `_TRIM=1`: skips the repeated GDN padding fills;
- `_ADV=1`: advances the accept-dependent uploads on the device. `_ADV=2` is **not gated**: its first version failed
  at CONC 2, so do not use it;
- `_MUT`: the must-fail mutants.

## Reproducing

All modes run from this directory. The cluster modes use the alola q38 tooling: `hackathon-prep/tools/q38/`, the
`rocke` partition, and the q38 container seeded by `q38/seed.sh`. Settings come from the environment: `DEV`,
`ATOM_REPO`, `ROCKE_REPO`, `HARNESS`, `CL_HOST` (default `vanantha@ctr2-alola-login-01`; login-03's DNS is dead),
`NODE` (`Q38_NODE` pin), `GPUS`, `PASSES`, `SEED`, `TAG`, `OUT`, `ARMS_K3`, `ARMS_K0`. `bash repro.sh help` lists
them.

```bash
bash repro.sh check                 # local, CPU, ~4 min: patches apply, trees == measured stages, files, knob lint
bash repro.sh stage rp-a            # build the patched trees, upload stages {h-qk,rocke-qk,atom-truemono,hb-atom,rp-h}-rp-a
bash repro.sh gen rp-a              # 1-GPU job: regenerate the stage functions, compare with the measured manifests
bash repro.sh gate rp-a             # 1-GPU job: gen + the c2 gates (~1 h)
PASSES=2 bash repro.sh quick rp-a   # 1-GPU job: K=3 and K=0 arms + A-vs-A twins (~70 min at 2 passes)
bash repro.sh full rp-a             # 2 quick jobs (n01-09, n07-17; 4 passes) + the T4 accuracy job
bash repro.sh wait <tag>            # wait, then fetch the log and results to $OUT/<tag>
bash repro.sh t4 rp-t4-rp-a         # T4 verdict of the accuracy job (acc_stats.py + t4_gate.py on the login node)
```

| mode | what it does | pass criterion |
|---|---|---|
| `check` | Read-only `git archive` of the three pinned commits into a new dir, then `git apply --check` + `patch`. Compares every file with `manifests/*.manifest` (the measured stages, CR ignored). Checks the frozen harness tar's sha and its files (recipes, picks, lookback, gates, arm definitions). Knob lint: every `ATOM_Q38_*` knob of every arm is read by the patched trees; the tail provider module resolves; N8 sizes; no K=3-only knob at K=0. | every line PASS (27 checks on 2026-11-02) |
| `stage` | Builds the same trees and packs them as the q38 packer does, then uploads them with a sha check. The harness stage is the frozen `h-qk-c2-f`. The hybrid rocKE stays the cluster's `rocke-tmbase` (`quick.sh` hard-codes that name); its sha is checked against the recorded one. | `SYNC_OK` for all five stages; the staged trees compare equal |
| `gen` | `gen_check.sh` runs `gen_c2Ab` and `gen_e2ecomp2` in the container, as the gate's layer step does, and diffs the generated files against the measured job's manifests (`qk-c2-e4-n01`). `*.bc` files embed the stage path, so they are left out. | `REPRO_GEN PASS`: 655 / 66 manifest lines, 0 differences |
| `gate` | `gen`, then the frozen `C2/c2_gates.sh`, unchanged, on recipe `c2Ab` with all three port sets, as job `qk-c2-gate3` ran it. Its three gates are below this table. | `C2_LAYER_GATE PASS`, `RB GATE PASS`, `CS_GATE PASS` |
| `quick` | `truemono-quick/quick.sh`, unchanged, in one job on one GPU. Per K: an untimed prime server, then `$PASSES` seeded rotated passes. One fresh server per arm-run, 2 warm-up requests, 32 gsm8k prompts (DFlash seed 42), CONC 1, greedy, reasoning xhigh, a 2048-token cap. Arms: K=3 `hybrid hybrid_b c2s3b c2s3b_b c2s3b_n8_64k`; K=0 `hybrid hybrid_b c2s0b c2s0b_b`. The server-log checks are below this table. | the ratios below |
| `full` | `quick` with 4 passes on each rocke node (pinned), plus the T4 accuracy job. T4 runs GSM8K 1,319 5-shot chat (CONC 1), HumanEval 164 (CONC 1, low) and RULER mix v1 8K + 32K (7 tasks, 50 per task, 4096 cap, CONC 4) for `hybrid` and `c2s3b`, 3 passes each, as in job `qk-c2-t4b`. `t4` then gates it with `t4_gate.py` against the P11 / W9 hybrid pool. | ratios in the band on both nodes; T4 PASS on all five suites |

The three gates that `gate` runs:
- **The stacked layer gate.** The served layer is bitwise equal to the previous stack's layer and to plain mono in
  every cell: delta / delta_ple / qsa at r1 and r4s, qsa r4, and qsa at L 1024 and 32,768. The must-fail mutants
  `mix_avg`, C1 `put_bit` and `q7_gate` must differ.
- **The RB gate.** Devmeta check mode 64 / 64 OK, the deep checks equal, K=1, K=0 unaffected, and the ADV mutant
  caught.
- **The CS tail gate.** Check mode 64 / 64, every deep check, transitions, and the accept / fids / lean-dpos / K=0
  mutants caught.

The server-log checks that `quick` runs on every arm-run:
- mono decode is on, there is no `mono off`, and there are 0 tracebacks;
- the rocKE head is on, and no `numa off for` line appears;
- every expected build line of `expect_k<K>.txt` appears;
- the arm's expected lines appear (`draft vocab subset on: 98304|65536 of`, `tm tail r4s: draft chain bound`,
  `tm tail r1: rocKE fused argmax head`);
- the arm's rejected lines are absent (`tm tail ... off`, `deepcheck ... DIFF`, `rb devmeta: off`, ...).

### Expected ratios (same-job hybrid; below 1 is faster for time, above 1 is faster for tok/s)

Metrics come from `summarize.py`:
- `med` = decode tok/s per request, the median over requests;
- `rnd_med` = ms per verify round, the median over requests.

Decide on the medians: neighbour bursts move the means. At K=3, `rnd_med` resolves about 0.3 %. Per-request tok/s
also carries a 2-4 % run-to-run acceptance noise (the served path is not token-deterministic).

| ratio | n01-09 | n07-17 | jobs (4 passes each) |
|---|---|---|---|
| K=3 `c2s3b` / hybrid, tok/s per request (median) | 1.128 / 1.141 / 1.134 | 1.136 / 1.118 / 1.122 | e1 / f1 / e4 (n01), e2 / f2 / e4 (n07) |
| K=3 `c2s3b` / hybrid, ms per round (median) | 0.889 / 0.889 / 0.888 | 0.887 / 0.888 / 0.889 | same |
| K=3 `c2s3b_n8_64k` / hybrid, tok/s / round | 1.120 / 0.884 | 1.124 / 0.887 | e4 |
| K=0 `c2s0b` / hybrid, tok/s per request (median) | 1.136 / 1.136 | 1.139 / 1.139 | e1 / f1, e2 / f2 |
| A-vs-A twins (hybrid_b / hybrid, c2s3b_b / c2s3b) | tok/s 0.99-1.01, round 0.997-1.003 | same | all |
| GSM8K / HumanEval / RULER v1 8K / 32K, `c2s3b` - hybrid pool | -0.003 / +0.016 / 0.000 / -0.006, inside T4 tolerances 0.012-0.057 | - | t4b |

The pass band for `quick` at 2 passes is:
- K=3 `c2s3b` / hybrid: tok/s per request 1.11-1.15, and ms per round 0.88-0.895;
- K=0 `c2s0b` / hybrid: tok/s 1.12-1.15;
- the twins within about ±1.5 % tok/s and ±0.5 % round.

Against the stack before it (`full3_l_rba`), `c2s3b` is 0.968 of the round (8 / 8 pairs) and 1.040 of the tok/s.

### Validation of this package (2026-11-02, stages `*-rp-a` built by `repro.sh stage` from pristine exports + patches)

| mode | job | result |
|---|---|---|
| `check` (laptop) | - | 27 PASS, 0 FAIL. Trees equal the measured stages file for file (see "What to check out"). |
| `gen` | `rp-gengate-rp-a2` (n01-09); re-run alone as `rp-gen-rp-a3` (n07-17) | `gen_c2Ab`: 608 / 608 and `gen_e2ecomp2`: 66 / 66 generated files byte-identical to the measured job's (`.bc` excluded) |
| `gate` | `rp-gengate-rp-a2` | `C2_LAYER_GATE PASS` (10 cells bitwise and deterministic, `hc_mix_avg` / `c1_put_bit` mutants differ); `RB GATE PASS` (c3la 64 / 64 OK, d3la 1,728 deep checks equal, c1a 56 / 56, d0 unaffected, ADV mutant 16 / 16 DIFF); `CS_GATE PASS` (29 CPU tests, g3l / g0, 4 must-fail mutants caught). The job's rc was 1 only because of a bug in `gen_check.sh`'s own exit status; that is fixed, and `rp-gen-rp-a3` re-checks it. |
| `quick` (1 GPU, 2 passes, n07-17) | `rp-quick-rp-a2` | K=3 `c2s3b` / hybrid: **tok/s per request (median) 1.131 [1.122-1.141], ms per round (median) 0.889 [0.889-0.889]**. Twin `c2s3b_b`: 1.135 / 0.889. `c2s3b_n8_64k`: 1.135 / 0.886. hybrid_b: 0.996 / 1.000. K=0 `c2s0b` / hybrid: **1.142 [1.140-1.143]**; twin 1.138; hybrid_b 1.004. Every server-log check passed. |
| "Reproduce on your own MI355X" step 1, from fresh public clones (2026-11-06, alola CPU job 68293691) | `hackathon-prep/qwen38/truemono-repro/sa-check/check_fresh.sh` | `git apply` of 01, 02 (ATOM) and 03 (rocm-libraries) all clean. The four trees equal the measured stages file for file: 1941 / 1924 / 2675 / 2400 files, 0 differences. `kit/SHA256SUMS` OK. Steps 2-3 (`docker run` + `kit/in_container.sh`) were not run on a GPU in this round. Their content is the measured jobs' harness and server env, byte for byte. |
| `full` (both nodes, 4 passes, T4) | not run | Its pieces are the measured c2 jobs (`qk-c2-e1/e2/f1/f2/e4`, `qk-c2-t4b`). This package's own `full` and `t4` modes were not exercised (GPU budget). |

## Known pitfalls

- **Container COMGR `waves=8` miscompile; ship prebuilt code objects.** The q38 container's COMGR (ROCm 10) builds
  some 512-thread (`waves=8`) large-batch QSA kernels wrong: their output is wrong and nondeterministic. Scratch
  == 0 does not make such a build safe (alola rules 209 / 217 / 219 / 222). None of these kernels is in this stack:
  every served stage function here passes the bitwise layer gate in the container. Any lever that adds such kernels
  must do two things:
  - build its code objects in a venv job (torch 2.12 COMGR) and load them strictly by kernel name in the container
    job (`*_HSACO_DIR` + `*_STRICT=1`: a missing object is refused, never compiled);
  - check every timed object on random data.
- **n07-17 `/tmp` wipes and reseeding.** n07-17 dropped out at about 05:45 UTC two days running (alola rules 198 / 223).
  Running jobs were requeued, and the requeue overwrote their `.out`. The node then also lost its seeded
  `/tmp/vanantha-q38` (image, checkpoint, venv), and container jobs died in 1-2 s. What to do:
  - Reseed with `q38/seed.sh` (about 14 min). Never run two seeds of one node at once.
  - Compare `q38/seeded/<node>` against the drop-out time before you submit.
  - Pass `XDG_RUNTIME_DIR=/tmp/vanantha-xdg ENROOT_RUNTIME_PATH=/tmp/vanantha-enroot/runtime`. `repro.sh` does this
    for every job.
  - Avoid 05:30-06:30 UTC on n07-17 for long jobs.
  - The n07-17 GPU at bus e5 (GRES idx 6) is bad. Every quick job starts with a 16 GiB randn health fill, which
    fails on it with exit 125.
- **The `gen_e2ecomp2` PYTHONPATH issue.** `q38_job.sh` puts the FIRST `rocke*` stage on `PYTHONPATH`, which is
  `rocke-tmbase`, and `e2ecomp2/gen.sh` does not override it. The candidate rocKE's MoE generator then imports the old
  platform and fails (`IRBuilder` has no `_lever_scope`; "gen: moe: 0 .ll"). `CS_GEN_PP=1` makes `arms.d/cs.sh` put the
  generating stage's own `platform/python:library` first. Every job of this package sets it.
- **The `hz` `s_nop` knob.** gfx9 needs 5 wait states between a VALU write of an SGPR and a VMEM instruction that
  reads it. LLVM pads its own instructions but never looks inside inline asm. Inline-asm VMEM (atomics, polls, stores)
  or `v_mbcnt` on an `s` operand right after such a write therefore produces silent, layout-dependent faults or wrong
  counts (alola rules 176 / 182 / 185 / 191 / 227). The fix is `s_nop 4` at the start of the asm.
  - In rocKE it is the knob `hz`, default on, in the large-batch kernels (`large_batch/*`, plus `insitu.pad_asm_vmem`).
  - The served CONC-1 stage functions of this stack carry no such asm path. They pass the bitwise gates.
  - Never turn `hz` off. Scan new asm-bearing kernels with `p-l4/fix/asm_hazard.py` / `lbf_hazard`.
- **Port collisions.** Another process on the node can hold the job-id-derived server port (`errno 98`, job
  `qk-c2-e4-n07` lost two arm-runs to it). `quick.sh` takes a free port from the OS. Other harnesses must set
  `Q38_PORT`.
- **Never `pkill` inside `srun --overlap`.** The container shares the host PID namespace, so it kills other jobs'
  processes on the node (alola rule 229). Kill by PID only.
- **Stale stage names.** Re-syncing a stage name after a job destroys that job's exact bytes. Every configuration
  gets its own suffix, and `repro.sh stage` writes new stages only.

## Pending add-ons (not in this stack; each lands on `c2s3b_n8_64k` / `c2s0b` once verified)

- **The rb agent's device-advanced round.** Arms `c2s3b_t2*`, jobs `rb-e2e5` / `rb-e2e6`: a whole round prepared
  and advanced on the device. ATOM-truemono already holds 9 files edited after the measured stage (`model_runner.py`,
  `gdn_attn.py`, `draft_runner.py`, `extern_stages.py`, `numa.py`, `sched.py`, `rb_devmeta.py`, `h2d.py` and the RB
  test), snapshotted in `patches/pending/atom-truemono-worktree-after-c2f.patch`. Unverified; do not apply.
- **W11 N1** (`w11x3`, the HC chain restructure; tolerance class). Its rocKE code (`stages2/hc_n1.py`) is in branch
  commits after `4478c174a13`.
- **N2** (`w17n3`, the out_proj K-split into the B3 / Q7 epilogues). It is REJECTED on base `c2s3b`: in situ every
  arm loses 2-7 %. It stays a default-off knob ([../w11/WP17.md](../w11/WP17.md)).
- **The re-climbed schedule groups** (`c2u3` / `c2u0`, `C2/c2t_sched_s2.txt`). They pass every gate but are neutral
  end to end. They are kept as the knob `C2T_SCHED`.
