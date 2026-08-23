# Auto-Engine — Resume & Operate Guide

Autonomous FP32 TN GEMM tuning engine for gfx1100. Targets: 4096x4096x4096, 2000x4000x3000,
4000x2000x3000 (TN = transA=T transB=N, library Cijk_Alik_Bljk).

## What it does
Iteratively proposes kernel configs (MacroTile/ThreadTile/WorkGroup/DepthU/WGM/StaggerU/LDS-layout
params), builds each via Tensile, and confirms with NOISE-ROBUST paired-interleaved cold-cache
benching. Only promotes a config if median delta >5% AND every paired rep agrees in sign (noise
floor on this box is ~3%). Never leaves a trial deployed — restores committed TN after every iter.

## Files (all under auto_engine/)
- `engine.py` — the whole engine. `python3 engine.py iter` runs ONE iteration. `baseline` inits.
- `run.sh [max_iters]` — flock'd loop calling engine.py iter forever. Launch with nohup.
- `report.sh` — read-only status (run anytime, no lock, no GPU).
- `config.json` — shapes, paths, tuning constants, valid param ranges.
- `ledger.csv` — APPEND-ONLY ground truth: every config attempt + result. Authoritative.
- `state.json` — atomic cache: iter, per-shape best/phase/baseline, running stage. Rederivable from ledger.
- `tried.idx` — config_hash dedup set (rederivable from ledger).
- `research_diary.md` — human narrative of iterations/hypotheses/insights.
- `pristine_tn.yaml` — snapshot of committed TN (the deploy anchor; NEVER overwritten by tuning).
- `best/<shape>/lib` + `logic.yaml` — champion device lib + logic per shape.
- `baseline/lib` — lib built from pristine (fallback baseline anchor).

## How to RESUME (fresh orchestrator / after kill)
1. `bash report.sh` — see where things stand.
2. Check no stale lock: `ps aux | grep engine.py`. If dead but lock file exists, it's fine (flock
   releases on process death). If `state.json.running` is non-null, the last iter was interrupted —
   engine.py is idempotent (tried.idx + ledger prevent redo); just relaunch.
3. `nohup bash run.sh > run_console.log 2>&1 &` — resumes the loop. It reloads state.json, recomputes
   best-per-shape from ledger.csv if needed, and skips already-tried configs.

## How to REPORT progress anytime
`bash report.sh` — per-shape best GF + %vs-baseline, phase, last improvement, current running stage,
attempt-status counts, promotions, diary tail.

## How to STOP
`pkill -f run.sh; pkill -f engine.py` then `pkill -f hipblaslt-bench`. Deployed TN auto-restores to
pristine at each iter end, so stopping is safe. If killed mid-build, run:
`cp pristine_tn.yaml <deployed_tn path from config.json>` then rebuild once to be safe.

## How to DEPLOY the accumulated bests (USER-GATED — do not auto-run)
Each `best/<shape>/logic.yaml` holds that shape's winning entry. To deploy: merge all per-shape
best logics onto pristine TN (TensileMergeLibrary --force_merge 1), rebuild, back up the deployed
file, copy in. A `deploy.sh` should be written for this when the user approves. NEVER commit/push
without user OK.

## Safety invariants
- One GPU job at a time (flock). - Committed TN never altered during tuning. - Every iter ends with
deployed TN == pristine. - ledger.csv is the source of truth; state.json is a cache.
