#!/usr/bin/env bash
# Run every Tensile config in gfx1250_96cu/ through run_tensile_with_agt.py on this DPX node, isolated and
# concurrent:
#   isolated    one GPU at a time (0, then 1) while the package's other partition idles
#   concurrent  GPUs 0 and 1 released together (both partitions of the package)
# Each config's runs go back to back, so its isolated and concurrent numbers come from the same stretch of time,
# and all of them share one --out, so each config's Tensile build is made once and reused.
#
# Usage: ./sweep.sh [run_tensile_with_agt.py options, passed to every run, e.g. --runs 3]
#   CONFIGS="bbs_tn_maf f8_tn_mab" ./sweep.sh     only these configs (default: every *.yaml in CFG_DIR)
#   MODES=concurrent ./sweep.sh                   isolated, concurrent, or both (default "isolated concurrent")
#   ISOLATED_GPUS=0 ./sweep.sh                    GPUs to run isolated, one at a time (default "0 1")
# Detached: setsid nohup ./sweep.sh > sweep.out 2>&1 < /dev/null &
#
# Outputs: the runner's own run dirs in agt_runs/<config>.gpu<list>.<time>/, plus in sweep_logs/<time>/ a log per
# run, status.tsv (config, mode, gpus, rc, run dir), summary.csv (every run's summary.csv rows with config and
# mode prepended) and summary.md (every run's table).
set -u
D=$(cd "$(dirname "$0")" && pwd)
H=/home/minsukim/projects/multi-repos/1_rocm-libraries_latest/projects/hipblaslt
CFG_DIR=/home/minsukim/projects/multi-repos/4_rocm-libraries_jimmy/projects/hipblaslt/tensilelite/gfx1250_96cu
RUNNER=$CFG_DIR/run_tensile_with_agt.py
AGT=/home/minsukim/projects/test/AGT4.1.171.9/agt_internal
# The build's venv, not the shell's python3: the runner launches Tensile with its own interpreter, and
# ~/projects/.venv is Python 3.9, which cannot import the cpython-312 rocisa the build made.
PY=$H/build/venv/bin/python
OUT=$D/agt_runs
LOGS=$D/sweep_logs/$(date +%Y%m%d-%H%M%S)

[[ -x "$PY" ]] || { echo "no $PY: build hipBLASLt in $H first (~/build_hipblaslt.sh)"; exit 1; }
if [[ -n "${CONFIGS:-}" ]]; then
    read -ra configs <<< "$CONFIGS"
else
    configs=()
    for f in "$CFG_DIR"/*.yaml; do configs+=("$(basename "$f" .yaml)"); done
fi
for c in "${configs[@]}"; do
    [[ -f "$CFG_DIR/$c.yaml" ]] || { echo "no config $CFG_DIR/$c.yaml"; exit 1; }
done
read -ra modes <<< "${MODES:-isolated concurrent}"
read -ra isolated_gpus <<< "${ISOLATED_GPUS:-0 1}"
mkdir -p "$LOGS"
printf 'config\tmode\tgpus\trc\trundir\n' > "$LOGS/status.tsv"
say() { echo "$(date +%T) $*"; }

failed=0
interrupted=0
# Ctrl-C reaches the runner too, which then finishes its batch, stops agt_internal and writes its summary; the
# sweep stops after that run. tee ignores it, so the runner never writes into a closed pipe while it cleans up.
trap 'interrupted=1' INT TERM
for c in "${configs[@]}"; do
    runs=()
    for m in "${modes[@]}"; do
        case $m in
            isolated)   for g in "${isolated_gpus[@]}"; do runs+=("isolated $g"); done ;;
            concurrent) runs+=("concurrent 0,1") ;;
            *)          echo "unknown mode '$m' (isolated or concurrent)"; exit 1 ;;
        esac
    done
    for r in "${runs[@]}"; do
        read -r m gpus <<< "$r"
        log="$LOGS/$c.$m.gpu${gpus//,/_}.log"
        say "$c $m on GPU(s) $gpus"
        "$PY" "$RUNNER" "$CFG_DIR/$c.yaml" "$gpus" "$H" "$AGT" --out "$OUT" "$@" < /dev/null 2>&1 |
            ( trap '' INT TERM; exec tee "$log" )
        rc=${PIPESTATUS[0]}
        rundir=$(sed -n 's/^[0-9:]* outputs in //p' "$log" | tail -1)
        printf '%s\t%s\t%s\t%s\t%s\n' "$c" "$m" "$gpus" "$rc" "$rundir" >> "$LOGS/status.tsv"
        (( rc == 130 )) && interrupted=1
        (( interrupted )) && break 2
        if (( rc != 0 )); then failed=$((failed + 1)); say "$c $m on GPU(s) $gpus failed (rc $rc), continuing"; fi
    done
done

# One CSV and one markdown file over every run that got as far as a summary.
header=0
while IFS=$'\t' read -r c m gpus rc rundir; do
    [[ -n "$rundir" && -f "$rundir/summary.csv" ]] || continue
    if (( ! header )); then
        printf 'config,mode,%s\n' "$(head -1 "$rundir/summary.csv")" > "$LOGS/summary.csv"
        header=1
    fi
    tail -n +2 "$rundir/summary.csv" | sed "s/^/$c,$m,/" >> "$LOGS/summary.csv"
    { printf '## %s, %s\n\n' "$c" "$m"; cat "$rundir/summary.md"; echo; } >> "$LOGS/summary.md"
done < <(tail -n +2 "$LOGS/status.tsv")

say "sweep done: $(( $(wc -l < "$LOGS/status.tsv") - 1 )) run(s), $failed failed$( (( interrupted )) && echo ', interrupted')"
say "status, summary.csv and summary.md in $LOGS"
(( failed == 0 && ! interrupted ))
