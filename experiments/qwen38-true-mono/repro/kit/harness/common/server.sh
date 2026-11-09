# Sourced inside the q38 container (q38_job.sh --mode container): start / stop an ATOM server
# serving Qwen3.8-Flash-Next from /model, as repro.sh's server_cmd does, on a job-unique port.
#   q38_server_start <log> <decode mono|native> <K 0..3> [extra server args...]
#   q38_server_stop
#   q38_port            the port in use (Q38_PORT, else derived from SLURM_JOB_ID / $$)
# Env knobs: Q38_MAX_SEQS (4), Q38_MML (10240), Q38_GMU (0.9), Q38_SERVER_ENV (extra K=V, space separated),
#   Q38_SERVER_PREFIX (a launcher in front of python, e.g. "rocprofv3 --kernel-trace ... --").

# Job-local compile caches (common/caches.sh; Q38_SHARED_CACHE=1 for the node's shared FlyDSL /
# Triton / inductor caches)
source "$(dirname "${BASH_SOURCE[0]}")/caches.sh"

q38_port() { echo "${Q38_PORT:=$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}"; }

q38_server_start() {
  local log=$1 decode=$2 k=$3
  shift 3
  local envs=(ATOM_RUNNER_INLINE_SEND=1)
  rm -rf "${HOME:-/work/home}/.cache/atom"  # ATOM's compile cache must not outlive a source change
  if [ "$decode" = mono ]; then envs+=(ATOM_MONO_ENABLE=1); else envs+=(ATOM_MONO_ENABLE=0); fi
  local args=(--model /model --trust-remote-code -tp 1 --server-port "$(q38_port)"
    --max-num-seqs "${Q38_MAX_SEQS:-4}" --max-model-len "${Q38_MML:-10240}"
    --gpu-memory-utilization "${Q38_GMU:-0.9}" --hf-overrides '{"_multimodal_config": null}')
  if [ "$k" != 0 ]; then
    envs+=(ATOM_ENABLE_REPLAYSSM=0)
    args+=(--method mtp --num-speculative-tokens "$k")
  fi
  # shellcheck disable=SC2206
  envs+=(${Q38_SERVER_ENV:-})
  echo "q38: server ${envs[*]} python -m atom.entrypoints.openai_server ${args[*]} $*"
  # Q38_SERVER_PREFIX: a launcher in front of python (e.g. rocprofv3 ... --)
  # shellcheck disable=SC2086
  (cd "${Q38_ATOM_DIR:-/atom}" && exec setsid env "${envs[@]}" ${Q38_SERVER_PREFIX:-} python -m atom.entrypoints.openai_server "${args[@]}" "$@") > "$log" 2>&1 &
  Q38_SERVER_PID=$!
  local t=0
  until curl -sf "http://127.0.0.1:$(q38_port)/v1/models" > /dev/null; do
    if ! kill -0 "$Q38_SERVER_PID" 2> /dev/null; then
      tail -n 40 "$log"
      echo "q38: server exited"
      return 1
    fi
    if [ $t -ge 5400 ]; then
      tail -n 40 "$log"
      echo "q38: server not ready after 90 min"
      return 1
    fi
    sleep 10
    t=$((t + 10))
  done
  grep -ao 'Qwen3.8 mono [a-z]* o[nf]*[^[]*' "$log" | sort -u | sed 's/^/q38: /' || true
  echo "q38: server ready after ${t}s on port $(q38_port)"
}

# SIGINT to the server's main process first (a graceful shutdown lets a profiler flush), then the group
q38_server_stop_graceful() {
  [ -n "${Q38_SERVER_PID:-}" ] || return 0
  local main
  # the python server itself, not a launcher in front of it (rocprofv3's command line holds it too)
  main=$(pgrep -g "$Q38_SERVER_PID" -f "^[^ ]*python[0-9.]* -m atom.entrypoints.openai_server" | head -1)
  echo "q38: graceful stop: SIGINT to ${main:-group leader $Q38_SERVER_PID}"
  kill -INT "${main:-$Q38_SERVER_PID}" 2> /dev/null || true
  for _ in $(seq 1 ${Q38_STOP_WAIT:-300}); do pgrep -g "$Q38_SERVER_PID" > /dev/null || break; sleep 1; done
  pgrep -g "$Q38_SERVER_PID" > /dev/null && echo "q38: graceful stop timed out; killing the group"
  q38_server_stop
}

q38_server_stop() {
  [ -n "${Q38_SERVER_PID:-}" ] || return 0
  # setsid: the server leads its own process group (engine core and model runner children too)
  kill -- -"$Q38_SERVER_PID" 2> /dev/null || kill "$Q38_SERVER_PID" 2> /dev/null || true
  for _ in $(seq 1 60); do kill -0 "$Q38_SERVER_PID" 2> /dev/null || break; sleep 2; done
  kill -9 -- -"$Q38_SERVER_PID" 2> /dev/null || true
  sleep 5
  Q38_SERVER_PID=
}

# single_request_bench on the running server: q38_speed <out.json> <dataset> <prompts> <max_tokens> [extra...]
q38_speed() {
  local out=$1 set=$2 n=$3 mt=$4
  shift 4
  python -m atom.benchmarks.qwen38_flash_mono.single_request_bench --base-url "http://127.0.0.1:$(q38_port)" \
    --model /model --dataset "$set" --num-prompts "$n" --max-tokens "$mt" \
    --chat-template-kwargs '{"reasoning_effort": "xhigh"}' --out "$out" "$@"
}
