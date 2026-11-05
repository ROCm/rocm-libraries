# DeepSeek-V4.1-Flash B0 baseline: exact repro

Everything runs on the alola `rocke` partition (gfx950, 8 GPUs per node), seeded node `smci355-ccs-aus-n01-09`.
Harness: `hackathon-prep/dsv41/` (local) mirrored to `~/rocke-sdpa-claude/dsv41/` in the Austin home
(`/home_aus/AMD/vanantha/...` on the login node, `/home/AMD/vanantha/...` on the nodes). Login: `ctr2-alola-login-01`.

## Pinned inputs

| what | pin |
|---|---|
| ATOM | `origin/main` = `df86c48298f926c98ee71972ace8bda237bc9958` (2026-10-16), shipped as stage `atom-main-df86c482.tar` (`git archive`, sha256 `bceac218...`), prepended on `PYTHONPATH` as `/atom-main` |
| container | `rocm/atom-dev:nightly_202610171506` = `docker://rocm/atom-dev@sha256:d03d841f9f46324e30385e50338748dc96432de8f995a192925b905a8b1d56ac` (its own `/app/ATOM` is the same commit `df86c482`) |
| aiter | the image's `/app/aiter-test` = `2cc8c750461c035f0333711dfb3c02571161aa95` (2026-10-16). The recipe's pin `2039d2b9...` (2026-09-13) is an ancestor of it; the baseline uses the image's aiter (what ATOM main CI runs) |
| torch / triton / ROCm | 2.10.0+rocm7.2.4 / 3.8.0 / ROCm 7.2.4 user space (image) |
| checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash` at the recipe's revision `dba1be0a40aa45a94ad051997016db3960a90277` (ungated, MIT, 88 files, 475 GiB) |
| prompts | GSM8K test set (`openai/grade-school-math` `test.jsonl`, sha256 `3730d312...`), a seed-0 permutation; the first N questions per cell |

## 1. Seed a node (CPU-only job, ~15 min, no GPU)

```bash
# local: ATOM main stage tar (read-only git archive)
cd hackathon-prep/dsv41
git -C ../../ATOM -c core.autocrlf=false -c core.eol=lf archive --format=tar \
  --add-virtual-file=.atom_commit:"$(git -C ../../ATOM rev-parse origin/main) origin/main ..." \
  -o stages/atom-main-df86c482.tar origin/main
sha256sum stages/atom-main-df86c482.tar | cut -d' ' -f1 > stages/atom-main-df86c482.tar.sha256
# copy stages/, jobs/, bench/, data/gsm8k_test.jsonl to ~/rocke-sdpa-claude/dsv41/ (Austin home), then on login-01:
cp jobs/seed.sh jobs/seed.run1.sh   # never edit a script a job is running
sbatch -p rocke -A miopen -w smci355-ccs-aus-n01-09 -c 48 --mem=128G -t 03:00:00 --chdir=/tmp \
  -J rk-dsv41-seed-n01 -o /home/AMD/vanantha/rocke-sdpa-claude/dsv41/logs/seed-n01-%j.out \
  /home_aus/AMD/vanantha/rocke-sdpa-claude/dsv41/jobs/seed.run1.sh
```

`seed.sh` imports the image with enroot into `/tmp/vanantha-dsv41/images/`, unpacks the ATOM stage (sha-checked) to
`/tmp/vanantha-dsv41/atom-main-df86c482/`, downloads the checkpoint with `jobs/hf_fetch.py` (32 parallel 1 GiB range
streams, then sha256 of every LFS file against the HF API and git-sha1 of the small files) into
`/tmp/vanantha-dsv41/models/DeepSeek-V4.1-Flash/` (`.hf_fetch/VERIFIED`, `SHA256SUMS`), prints the container inventory,
and writes `~/rocke-sdpa-claude/dsv41/seeded/<node>`. Re-running resumes / skips verified parts. No durable copy is kept
on any home: re-seed from HF (~10 min download + 1 min verify).

## 2. Baseline jobs (container, fresh server per arm-run)

`jobs/dsv41_job.sh` (raw sbatch body) checks the seed, copies `bench/` and `data/` into the job dir and runs
`bench/run_arms.sh` in the image with `srun --container-writable` and mounts `/atom-main`, `/model`, `/job`,
`/tmp/vanantha-dsv41`. Results land in `~/rocke-sdpa-claude/dsv41/results/<tag>/`.

```bash
D=/home_aus/AMD/vanantha/rocke-sdpa-claude/dsv41; cd $D && cp jobs/dsv41_job.sh jobs/dsv41_job.<tag>.sh
# TP2 (2 GPUs)
sbatch -p rocke -A miopen -w smci355-ccs-aus-n01-09 --gres=gpu:2 -c 32 --mem=640G -t 04:00:00 --chdir=/tmp \
  -J rk-dsv41-b0tp2a -o /home/AMD/vanantha/rocke-sdpa-claude/dsv41/logs/b0tp2a-%j.out \
  --export=ALL,TAG=b0tp2a,ARMS='tp2-noep-ds1-m0-i8 tp2-noep-ds1-m1-i8 tp2-noep-ds0-m0-i8 tp2-noep-ds0-m1-i8 tp2-noep-ds1-m1-i4 tp2-noep-ds1-m0-i8-b',CELLS='1:16 2:16 4:32 8:32 16:64',ACC_ARMS='tp2-noep-ds1-m0-i8 tp2-noep-ds1-m1-i8',ACC=3:100:8 \
  $D/jobs/dsv41_job.b0tp2a.sh
# TP4 (4 GPUs)
sbatch -p rocke -A miopen -w smci355-ccs-aus-n01-09 --gres=gpu:4 -c 48 --mem=900G -t 03:30:00 --chdir=/tmp \
  -J rk-dsv41-b0tp4a -o /home/AMD/vanantha/rocke-sdpa-claude/dsv41/logs/b0tp4a-%j.out \
  --export=ALL,TAG=b0tp4a,ARMS='tp4-noep-ds1-m0-i8 tp4-noep-ds1-m1-i8 tp4-noep-ds0-m0-i8 tp4-noep-ds0-m1-i8 tp4-noep-ds1-m1-i4 tp4-noep-ds1-m0-i4 tp4-ep-ds1-m0-i8 tp4-ep-ds1-m1-i8 tp4-ep-ds0-m0-i8 tp4-noep-ds1-m1-i8-b tp4-noep-ds1-m0-i8-b',CELLS='1:16 2:16 4:32 8:32 16:64 64:128',ACC_ARMS='tp4-noep-ds1-m0-i8 tp4-noep-ds1-m1-i8 tp4-ep-ds1-m0-i8',ACC=3:100:8,TRACE_ARMS='tp4-noep-ds1-m0-i8 tp4-noep-ds1-m1-i8 tp4-ep-ds1-m0-i8' \
  $D/jobs/dsv41_job.b0tp4a.sh
# kernel traces only (4 GPUs; TP2 arms use the first two)
... --gres=gpu:4 ... --export=ALL,TAG=b0trace,ARMS='',TRACE_ARMS='tp4-noep-ds1-m0-i8 tp4-noep-ds1-m1-i8 tp4-ep-ds1-m0-i8 tp4-noep-ds0-m0-i8 tp4-noep-ds0-m1-i8 tp2-noep-ds1-m0-i8 tp2-noep-ds1-m1-i8' ...
```

Arm names: `<tp>-<ep|noep>-<ds1|ds0>-<m1|m0>-<i8|i4>[-b]`: tensor parallel size; `--enable-expert-parallel` or not;
DSpark (`--method dspark --num-speculative-tokens 5`) or none; `ATOM_MONO_ENABLE=1/0`; `--index-cache-dtype fp8/fp4`;
`-b` = an A-vs-A twin of the same arm (own server). Every server (the AgentX recipe's flags, except the level and the
fixed acceptance length):

```
ATOM_MONO_ENABLE=<0|1> OMP_NUM_THREADS=4 ATOM_NUMA_BIND=0 ATOM_DISABLE_MMAP=true AITER_LOG_LEVEL=WARNING \
AITER_REUSE_IDENTICAL_COMM_GROUPS=1 PYTHONPATH=/atom-main \
python -u -m atom.entrypoints.openai_server --model /model --trust-remote-code --host 127.0.0.1 --server-port <free> \
  --tensor-parallel-size <2|4> [--enable-expert-parallel] --kv_cache_dtype bf16 --index-cache-dtype <fp8|fp4> \
  --gpu-memory-utilization 0.9 --max-num-seqs 64 --max-model-len 32768 --max-num-batched-tokens 16384 \
  --attn-prefill-chunk-size 16384 --enable_prefix_caching --block-size 16 --state-checkpoint-interval-tokens 8192 \
  --level 0 --cudagraph-mode FULL --cudagraph-capture-sizes [1,2,3,4,5,6,7,8,16,32,48,64] --tool-call-parser dsml_v41 \
  [--method dspark --num-speculative-tokens 5]
```

`--level 0` because `--level 3` stalled for 25+ minutes inside inductor (see BASELINE.md); level 0 + FULL graphs is the
recipe's development / acceptance mode and the mode its mono table was measured in.

Per arm-run (`run_arms.sh`): health fill on every GPU; server start (ready = `/v1/models` answers; the log is watched for
"proc died"); the server log's `V4.1 mono decode|draft on|off` lines are recorded (proof of the path served); then per
cell `conc:prompts` `bench/dsv41_bench.py` (streaming chat completions, greedy, `max_tokens` 1024, untimed warm-up of
max(2, conc) other prompts at the cell's concurrency, then the timed prompts at that concurrency): per request TTFT,
completion tokens (server usage) and decode tok/s = (tokens - 1) / (last chunk - first chunk); acceptance length =
1 + accepted / verify steps from `/debug/mtp_stats` deltas. ACC arms then run
`lm_eval --model local-chat-completions --tasks gsm8k --num_fewshot 3 --limit 100 --apply_chat_template
--fewshot_as_multiturn` (8 concurrent, `max_gen_toks=4096`, temperature 0). TRACE arms start a separate server with
`--torch-profiler-dir`, run one request, and `bench/trace_decode.py` sends one long-generation prompt (`ignore_eos`, 1024 tokens) and
brackets 48 decode chunks (= DSpark steps) after the first 32 with `/start_profile` / `/stop_profile`; `bench/analyze_trace.py` sums every rank's GPU kernels
by name and family. Compile caches (`HOME`, triton, aiter's `flydsl_cache`) persist per node in
`/tmp/vanantha-dsv41/cache`.

## 3. Summaries

```bash
# traces: re-run the family map locally on the fetched kineto files
python hackathon-prep/dsv41/bench/analyze_trace.py <dir with rank traces> trace_<arm>.window.json trace_<arm>.breakdown2.json
python hackathon-prep/dsv41/bench/summarize.py <results dirs> --ref <arm> --out-abs <private>.md --out-ratio <ratio>.md
```

Absolute tables go to `hackathon-prep/private-perf/dsv41-mono/`; only the ratio tables are copied here.
