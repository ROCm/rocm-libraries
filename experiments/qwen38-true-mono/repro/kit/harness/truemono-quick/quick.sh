#!/usr/bin/env bash
# truemono-quick: the quick end-to-end tok/s check (PLAN decision 14). ONE job, one GPU, q38 container
# (q38_job.sh --mode container). For each K in $KS: a prime server (untimed, fills the job-local compile caches),
# then $PASSES passes over $ARMS, the arm order a seeded permutation rotated by the pass index; one fresh server per
# arm-run, one server at a time. Each arm-run: 2 untimed gsm8k warmup requests (they also JIT the extern-linked
# builds at the served shapes, which are never disk-cached), then the timed single_request_bench (gsm8k, $PROMPTS
# prompts = the DFlash seed-42 selection, the same as quick_tps.sh, CONC 1, greedy, reasoning xhigh, $TOKENS cap).
# summarize.py: per arm and K tok/s per request, token-weighted and e2e tok/s, ms per round / per token, acceptance,
# gsm8k correct, server-log checks; ratios to the `hybrid` arm paired by pass.
#
# Stages (sync.sh): h-qk-<sfx> (this dir + common/ + e2ecomp2/ + sched/ + harness/), rocke-tmbase (pinned hybrid rocKE), rocke-qk-<sfx>
# (copy-rocm-lib rocKE), atom-truemono-<sfx> (at /atom) and hb-atom-<sfx> (the pinned hybrid ATOM, byte copy of
# atom-rkrepro), all from truemono/stage_atom.sh naming.
# Env: CELLS / CELLS_K<k> (extra cells, see cell_run), SPEED_C1 (1), ARMS ("mono hybrid hyb_tm"), ARMS_K0 / ARMS_K3 (per-K lists, default ARMS), KS ("3 0"), PASSES (2),
#      PROMPTS (32), TOKENS (2048), PRIME (1), SEED (0), PRE_TESTS (ATOM-truemono test files that must pass first),
#      QK_SFX (stage suffix, "qk"), ARM_FILES (extra arm files), Q38_H (harness stage name).
# Arms: arms.sh (mono, hybrid, hyb_tm) + arms.d/*.sh (n8*, p10*, tuned*); each a bash function of env vars + stage
# names, with optional server-log checks (ARM_EXPECT[_K0|_K3], ARM_REJECT, <FN>/expect_k<K>.txt).
# The decision metrics are the per-request MEDIANS (summarize.py `med`, `rnd_med`): neighbour bursts move means.
set -uo pipefail
SFX=${QK_SFX:-qk}
H=/job/stages/${Q38_H:-h-qk-$SFX}
HB=/job/stages/hb-atom-$SFX
TM=/atom
RK_BASE=rocke-tmbase
RK_TM=rocke-qk-$SFX
# a free port (the job-id-derived default collided with another process on the node: job qk-acc3, errno 98)
[ -n "${Q38_PORT:-}" ] || Q38_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])")
export Q38_PORT
source "$H/common/server.sh"
source "$H/truemono-quick/arms.sh"
O=$Q38_OUT/q
mkdir -p "$O"
read -ra ks <<< "${KS:-3 0}"
# per-K arm lists: ARMS_K<k> (default ARMS); every arm of any list is resolved once
declare -A KARMS
for k in "${ks[@]}"; do v=ARMS_K$k; KARMS[$k]=${!v:-${ARMS:-mono hybrid hyb_tm}}; done
read -ra names <<< "$(for k in "${ks[@]}"; do echo ${KARMS[$k]}; done | tr ' ' '\n' | awk 'NF && !seen[$0]++' | tr '\n' ' ')"
PROMPTS=${PROMPTS:-32}
TOKENS=${TOKENS:-2048}
PASSES=${PASSES:-2}
echo "${CELLS:-} ${CELLS_K0:-} ${CELLS_K1:-} ${CELLS_K2:-} ${CELLS_K3:-}" | tr ' ' '\n' | awk 'NF && !seen[$0]++' > "$O/cells.txt"
[ "${SPEED_C1:-1}" = 1 ] || touch "$O/speed_c1_off"
echo "truemono-quick arms=${names[*]} ($(for k in "${ks[@]}"; do echo -n "K$k: ${KARMS[$k]}; "; done)) K=${ks[*]} passes=$PASSES prompts=$PROMPTS tokens=$TOKENS node=$(hostname) $(date -u)"
python -c "import torch; p=torch.cuda.get_device_properties(0); print('gpu', p.name, getattr(p,'pci_bus_id','?'))" 2>&1 | tail -1
echo "neighbours: $(squeue -h -w "$(hostname -s)" -o '%u:%j' 2> /dev/null | tr '\n' ' ')"
# health precheck (alola rule 146): a 16 GiB randn fill must pass before any server
timeout -s KILL 300 python -c "import torch; x=torch.empty(4*1024**3, device='cuda'); x.normal_(); torch.cuda.synchronize(); print('health ok', float(x[:4].sum()))" \
  || { echo "GPU HEALTH CHECK FAILED"; exit 125; }
# PIP_EXTRA: packages the cells need beyond the image (RULER: "wonderwords nltk"), job-local (alola rule 132)
if [ -n "${PIP_EXTRA:-}" ]; then
  # shellcheck disable=SC2086
  pip install -q --target /job/pysite $PIP_EXTRA > "$O/pip_extra.log" 2>&1 && echo "pip extra: ok ($PIP_EXTRA)" \
    || { echo "pip extra: FAIL"; tail -n 5 "$O/pip_extra.log"; }
  export PYTHONPATH=/job/pysite${PYTHONPATH:+:$PYTHONPATH} NLTK_DATA=/job/nltk_data
  python -c "import nltk; [nltk.download(p, download_dir='/job/nltk_data', quiet=True) for p in ('punkt', 'punkt_tab')]" > /dev/null 2>&1 \
    && echo "nltk data: ok" || echo "nltk data: not downloaded"
fi
# PRE_TESTS: ATOM-truemono test files (GPU tests included) that must pass before any arm runs (a lever's gate)
if [ -n "${PRE_TESTS:-}" ]; then
  # shellcheck disable=SC2086
  (cd $TM && PYTHONPATH=$TM:/job/stages/$RK_TM/platform/python:/job/stages/$RK_TM/library timeout 1800 \
    python -m pytest -q -p no:cacheprovider $PRE_TESTS) > "$O/pre_tests.log" 2>&1 \
    && echo "pre-tests: PASS $(tail -n 1 "$O/pre_tests.log")" || { echo "pre-tests: FAIL"; tail -n 30 "$O/pre_tests.log"; exit 4; }
fi

# ---- arms: resolve every definition once (and generate functions) ----
declare -A A_ATOM A_ROCKE A_ENV A_HEAD A_EXPECT A_EXPECT0 A_EXPECT3 A_REJECT A_FN GENDONE
rc=0
for n in "${names[@]}"; do
  declare -F "arm_$n" > /dev/null || { echo "unknown arm $n"; exit 2; }
  ARM_ATOM="" ARM_ROCKE="" ARM_GEN="" ARM_ENV="" ARM_HEAD=0 ARM_EXPECT="" ARM_EXPECT_K0="" ARM_EXPECT_K3="" ARM_REJECT="" ARM_PYPATH=""
  "arm_$n"
  FN=""
  if [ -n "$ARM_GEN" ]; then
    FN=/job/fns/$ARM_GEN-$ARM_ROCKE
    if [ -z "${GENDONE[$FN]:-}" ]; then
      echo "--- gen $ARM_GEN with $ARM_ROCKE -> $FN $(date -u +%H:%M:%S)"
      "gen_$ARM_GEN" "$ARM_ROCKE" "$FN" > "$O/gen-$ARM_GEN-$ARM_ROCKE.log" 2>&1 || { tail -20 "$O/gen-$ARM_GEN-$ARM_ROCKE.log"; echo "gen failed: $n"; exit 3; }
      tail -3 "$O/gen-$ARM_GEN-$ARM_ROCKE.log"
      cp "$FN/manifest.sha256" "$O/manifest-$ARM_GEN-$ARM_ROCKE.sha256" 2> /dev/null
      GENDONE[$FN]=1
    fi
  fi
  R=/job/stages/$ARM_ROCKE
  A_ATOM[$n]=$ARM_ATOM
  A_ROCKE[$n]=$ARM_ROCKE
  A_ENV[$n]="$(eval "echo \"$ARM_ENV\"") PYTHONPATH=$ARM_ATOM:$R/platform/python:$R/library${ARM_PYPATH:+:$ARM_PYPATH}"
  A_HEAD[$n]=$ARM_HEAD
  A_EXPECT[$n]=$ARM_EXPECT
  A_EXPECT0[$n]=$ARM_EXPECT_K0
  A_EXPECT3[$n]=$ARM_EXPECT_K3
  A_FN[$n]=$FN
  A_REJECT[$n]=$ARM_REJECT
  echo "  arm $n: atom=$ARM_ATOM rocke=$ARM_ROCKE env=${A_ENV[$n]} head=$ARM_HEAD expect=${ARM_EXPECT:--} expect_k0=${ARM_EXPECT_K0:--} expect_k3=${ARM_EXPECT_K3:--} reject=${ARM_REJECT:--}" | tee -a "$O/arms.txt"
done

check_log() { # check_log <server log> <arm> <K>: one line of checks; nonzero when a hard check fails
  local L=$1 a=$2 k=$3 ok=1 tb on off head numaoff miss tml exp=- kexp=- rej=- pat
  tb=$(grep -a -c Traceback "$L"); on=$(grep -a -c 'Qwen3.8 mono decode on' "$L")
  off=$(grep -a -c -E 'Qwen3.8 mono (decode|draft) off' "$L"); head=$(grep -a -c 'rocKE lm_head on' "$L")
  numaoff=$(grep -a -c 'numa off for' "$L"); miss=$(grep -a -c 'EXTERN_STAGES\] no ' "$L"); tml=$(grep -a -c 'Qwen3.8 true mono' "$L")
  [ "$tb" = 0 ] && [ "$on" -ge 1 ] && [ "$off" = 0 ] && [ "$numaoff" = 0 ] || ok=0
  if [ "${A_HEAD[$a]}" = 1 ]; then [ "$head" -ge 1 ] || ok=0; else [ "$head" = 0 ] || ok=0; fi
  if [ -n "${A_EXPECT[$a]}" ]; then exp=$(grep -a -c -E "${A_EXPECT[$a]}" "$L"); [ "$exp" -ge 1 ] || ok=0; fi
  # K-specific checks hold for the cells they name: K=0 (r1 builds) and K=3 (r4s builds); K=1 / 2 have neither
  pat=""
  [ "$k" = 0 ] && pat=${A_EXPECT0[$a]}
  [ "$k" = 3 ] && pat=${A_EXPECT3[$a]}
  if [ -n "$pat" ]; then kexp=$(grep -a -c -E "$pat" "$L"); [ "$kexp" -ge 1 ] || ok=0; fi
  # a functions recipe may write <FN>/expect_k<K>.txt: every line (a build line, fixed string) must be in the log
  local ef=${A_FN[$a]:+${A_FN[$a]}/expect_k$k.txt} line nf=0
  if [ -n "$ef" ] && [ -f "$ef" ]; then
    while IFS= read -r line; do [ -z "$line" ] || grep -a -q -F -- "$line" "$L" || nf=$((nf + 1)); done < "$ef"
    kexp="$kexp,file-missing=$nf"
    [ $nf = 0 ] || ok=0
  fi
  if [ -n "${A_REJECT[$a]}" ]; then rej=$(grep -a -c -E "${A_REJECT[$a]}" "$L"); [ "$rej" = 0 ] || ok=0; fi
  echo "checks ok=$ok tracebacks=$tb mono-on=$on mono-off=$off rocke-head=$head numa-off=$numaoff extern-missing=$miss true-mono-lines=$tml expect=$exp expect_k=$kexp reject=$rej"
  [ $ok = 1 ]
}

PORT=$(q38_port)
mtp() { curl -sf "http://127.0.0.1:$PORT/debug/mtp_stats" > "$1" 2> /dev/null || echo '{}' > "$1"; }
# Extra cells (CELLS, run on every arm-run's server after the default C1 speed run; SPEED_C1=0 skips that one).
# Each writes $O/<run>__<cell>.json (or .lmeval/, .ruler/):
#   speed:<set>:<conc>:<prompts>[:<max_tokens>]   single_request_bench; CONC > 1 first runs an untimed mt-bench
#                                                 warmup at that concurrency (new batch shapes JIT; matrix.sh's rule)
#   lmeval:gsm8k:<conc>[:<limit>]                 lm_eval GSM8K 5-shot chat (truemono-baseline cell 14's settings)
#   acc:<task>:<conc>[:<reasoning>]               accuracy_eval (humaneval, gpqa, mmlupro)
#   ruler:<len,len>:<conc>:<limit>[:<task+task>[:<max_gen_toks>]]  lm_eval RULER tasks (needs Q38_MML > the longest; the
#                                                 reasoning model spends tokens before its answer: 256 truncates many)
cell_run() { # cell_run <tag> <spec> <K>
  local tag=$1 spec=$2 f
  IFS=: read -ra f <<< "$spec"
  case ${f[0]} in
    speed)
      if [ "${f[2]}" -gt 1 ]; then
        q38_speed "$O/$tag.warm.json" mt-bench $((3 * f[2])) 128 --warmup 0 --concurrency "${f[2]}" > "$O/$tag.warm.log" 2>&1 \
          || { tail -n 5 "$O/$tag.warm.log"; return 1; }
      fi
      q38_speed "$O/$tag.json" "${f[1]}" "${f[3]}" "${f[4]:-2048}" --warmup 0 --concurrency "${f[2]}" > "$O/$tag.log" 2>&1
      ;;
    lmeval)
      local lim=()
      [ "${f[3]:-0}" = 0 ] || lim=(--limit "${f[3]}")
      mtp "$O/$tag.mtp0.json"
      lm_eval --model local-chat-completions --tasks gsm8k --num_fewshot 5 "${lim[@]}" \
        --apply_chat_template --fewshot_as_multiturn \
        --model_args "model=/model,base_url=http://127.0.0.1:$PORT/v1/chat/completions,num_concurrent=${f[2]},max_retries=3,timeout=900,tokenized_requests=False,trust_remote_code=True" \
        --gen_kwargs 'max_gen_toks=4096,until=<|im_end|>' --output_path "$O/$tag.lmeval" --log_samples > "$O/$tag.log" 2>&1
      local r=$?
      mtp "$O/$tag.mtp1.json"
      grep -E "^\|(Tasks|gsm8k| )" "$O/$tag.log" | head -n 6
      return $r
      ;;
    acc)
      python -m atom.benchmarks.qwen38_flash_mono.accuracy_eval --base-url "http://127.0.0.1:$PORT" \
        --model /model --task "${f[1]}" --concurrency "${f[2]}" --reasoning-effort "${f[3]:-low}" \
        --limit 0 --out "$O/$tag.json" > "$O/$tag.log" 2>&1
      local r=$?
      python -c "import json,sys; s=json.load(open(sys.argv[1]))['summary']; print('     ', sys.argv[2], s.get('accuracy'), s.get('correct'), s.get('n'))" "$O/$tag.json" "${f[1]}" 2> /dev/null
      return $r
      ;;
    ruler)
      # one lm_eval per length: --limit takes the first N docs of the task's dataset, which holds every length in
      # order, so a multi-length run with a limit evaluates only the first length (job qk-acc2: 32K never ran)
      local tasks=${f[4]:-niah_single_1+niah_single_2+niah_multikey_1+niah_multiquery+ruler_vt} L r=0
      for L in ${f[1]//,/ }; do
        lm_eval --model local-chat-completions --tasks "${tasks//+/,}" --limit "${f[3]}" --apply_chat_template           --metadata "{\"max_seq_lengths\":[$L],\"tokenizer\":\"/model\",\"pretrained\":\"/model\"}"           --model_args "model=/model,base_url=http://127.0.0.1:$PORT/v1/chat/completions,num_concurrent=${f[2]},max_retries=3,timeout=1800,tokenized_requests=False,trust_remote_code=True,tokenizer=/model"           --gen_kwargs "max_gen_toks=${f[5]:-256}" --output_path "$O/$tag.ruler/L$L" --log_samples > "$O/$tag.L$L.log" 2>&1 || r=1
        grep -E "^\|" "$O/$tag.L$L.log" | grep -v -- "-1.000" | head -n 12
      done
      return $r
      ;;
    *) echo "unknown cell $spec"; return 2 ;;
  esac
}

run_arm() { # run_arm <K> <run tag> <arm> [prime]
  local k=$1 run=$2 a=$3 prime=${4:-} which t0
  t0=$(date +%s)
  # shellcheck disable=SC2086
  which=$(cd "${A_ATOM[$a]}" && env ${A_ENV[$a]} python -c 'import os, atom.models.qwen4_exp_mono as m; print(os.path.dirname(m.__file__))' 2>&1 | tail -1)
  Q38_ATOM_DIR=${A_ATOM[$a]} Q38_SERVER_ENV="${A_ENV[$a]}" q38_server_start "$O/$run.server.log" mono "$k" \
    > "$O/$run.start.log" 2>&1 || { tail -5 "$O/$run.start.log"; echo "$run: SERVER FAILED"; return 1; }
  local ready=$(($(date +%s) - t0))
  q38_speed "$O/$run.warm.json" gsm8k 2 64 --warmup 0 > "$O/$run.warm.log" 2>&1
  local cfail=0 spec
  if [ -z "$prime" ] && [ "${SPEED_C1:-1}" = 1 ]; then
    q38_speed "$O/$run.json" gsm8k "$PROMPTS" "$TOKENS" --warmup 0 > "$O/$run.log" 2>&1 || { tail -5 "$O/$run.log"; q38_server_stop; return 1; }
  fi
  if [ -z "$prime" ]; then
    local cv=CELLS_K$k
    for spec in ${!cv:-${CELLS:-}}; do
      local ct0
      ct0=$(date +%s)
      if cell_run "${run}__$(echo "$spec" | tr ':,+' '---')" "$spec" "$k"; then
        echo "    cell $spec: ok ($(($(date +%s) - ct0)) s)"
      else
        echo "    cell $spec: FAIL ($(($(date +%s) - ct0)) s)"; cfail=1
      fi
    done
  fi
  q38_server_stop
  local c
  c=$(check_log "$O/$run.server.log" "$a" "$k")
  local cr=$?
  grep -a -o '\[ATOM_Q38_SCHED\] build [^|]*' "$O/$run.server.log" | sort -u > "$O/$run.builds.txt"
  echo "$run: ready ${ready}s total $(($(date +%s) - t0))s package=$which $c" | tee -a "$O/runs.txt"
  [ -n "$prime" ] || python - "$O/$run.json" << 'EOF'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
f = lambda k, fmt="{:.1f}": fmt.format(s[k]) if s.get(k) is not None else "-"
print(f"    req {f('decode_tok_s_per_request_mean')} tw {f('decode_tok_s_token_weighted')} e2e {f('end_to_end_tok_s')} "
      f"ms/round {f('ms_per_round', '{:.3f}')} acc {f('acceptance', '{:.3f}')} correct {s.get('correct')}/{s.get('checked')} at_cap {s.get('at_cap')}")
EOF
  [ $cfail = 0 ] || cr=1
  return $cr
}

for k in "${ks[@]}"; do
  read -ra kn <<< "${KARMS[$k]}"
  echo "${kn[*]}" > "$O/arms_k$k.txt"
  if [ "${PRIME:-1}" = 1 ]; then
    echo "--- k${k} prime (untimed ${kn[0]}) $(date -u +%H:%M:%S)"
    run_arm "$k" "k${k}_0_${kn[0]}" "${kn[0]}" prime || echo "prime failed (not counted)"
  fi
  for p in $(seq 1 "$PASSES"); do
    read -ra order <<< "$(python -c "import random,sys; a=sys.argv[3:]; random.Random(1000*int(sys.argv[1])+7+int('${SEED:-0}')).shuffle(a); p=(int(sys.argv[2])-1)%len(a); print(*(a[p:]+a[:p]))" "$k" "$p" "${kn[@]}")"
    echo "k$k pass $p order: ${order[*]}" | tee -a "$O/order.txt"
    for a in "${order[@]}"; do
      echo "--- k${k}_${p}_$a $(date -u +%H:%M:%S)"
      run_arm "$k" "k${k}_${p}_$a" "$a" || rc=1
    done
  done
done
python "$H/truemono-quick/summarize.py" "$O" "${KS:-3 0}" "$PASSES" "${names[*]}" || rc=1
echo "neighbours at end: $(squeue -h -w "$(hostname -s)" -o '%u:%j' 2> /dev/null | tr '\n' ' ')"
exit $rc
