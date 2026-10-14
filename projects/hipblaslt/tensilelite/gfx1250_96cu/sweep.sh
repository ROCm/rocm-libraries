#!/usr/bin/env bash
# Run every Tensile config in gfx1250_96cu/ through run_tensile_with_agt.py, in any of these modes:
#   isolated    each GPU in ISOLATED_GPUS alone, one at a time, while its package's other partition idles
#   concurrent  each package's GPUs in GPUS released together (both partitions of a DPX package), one package at a
#               time; a package with only one GPU in GPUS has no concurrent run
#   all         every GPU in GPUS released together (GPUS=all: the whole node)
# A GPU set runs once per config, under the first mode in MODES that names it. Each config's runs go back to back, so
# its numbers in every mode come from the same stretch of time, and all of them share one --out, so each config's
# Tensile build is made once and reused.
#
# Usage: sweep.sh [run_tensile_with_agt.py options, passed to every run, e.g. --runs 3]
#   GPUS=all sweep.sh                             GPUs, as the runner takes them: 0,1 | 0-3 | all (default 0,1, the
#                                                 first package)
#   ISOLATED_GPUS=0,2,4,6 sweep.sh                GPUs to run isolated, in the same form (default: GPUS)
#   MODES="isolated concurrent all" sweep.sh      modes to run (default "isolated concurrent")
#   CONFIGS="bbs_tn_maf f8_tn_mab" sweep.sh       only these configs (default: every *.yaml in CFG_DIR)
#   AGT=/path/to/agt_internal sweep.sh            AGT binary (default /opt/agt/agt_internal); it runs through
#                                                 `sudo -n`, so it and /usr/bin/true need a NOPASSWD sudoers rule
# Detached: setsid nohup <this dir>/sweep.sh > sweep.out 2>&1 < /dev/null &    stop it with kill -TERM -- -<its pid>
#
# Outputs, in the current directory, whose path must have no spaces or colons (the runner's release gate is preloaded
# from there): the runner's run dirs in agt_runs/<config>.gpu<list>.<time>/ and its builds in agt_runs/build/, plus
# in sweep_logs/<time>/ a log per run, status.tsv (config, mode, gpus, rc, run dir), summary.csv (every run's
# summary.csv rows with config and mode prepended) and summary.md (every run's table).
set -u
H=/home/minsukim/projects/multi-repos/1_rocm-libraries_latest/projects/hipblaslt
CFG_DIR=/home/minsukim/projects/multi-repos/2_rocm-libraries_jimmy/projects/hipblaslt/tensilelite/gfx1250_96cu
RUNNER=$CFG_DIR/run_tensile_with_agt.py
# A root-owned copy, so the NOPASSWD rule for it does not hand root to whatever is at a path this user can write.
AGT=${AGT:-/opt/agt/agt_internal}
# The build's venv, not the shell's python3: the runner launches Tensile with its own interpreter, and
# ~/projects/.venv is Python 3.9, which cannot import the cpython-312 rocisa the build made.
PY=$H/build/venv/bin/python
OUT=$PWD/agt_runs
LOGS=$PWD/sweep_logs/$(date +%Y%m%d-%H%M%S)

[[ -x "$PY" ]] || { echo "no $PY: build hipBLASLt in $H first (~/build_hipblaslt.sh)"; exit 1; }
[[ -f "$AGT" && -x "$AGT" ]] || { echo "no $AGT (set AGT to the agt_internal binary)"; exit 1; }
# The runner starts agt_internal through `sudo -n`, from a session with no terminal, so a cached `sudo -v` does not
# carry over: without a NOPASSWD rule every run would fail at its start.
sudo -n true 2> /dev/null ||
    { echo "sudo -n asks for a password here; agt_internal needs a NOPASSWD sudoers rule for $AGT and /usr/bin/true"
      exit 1; }
# LD_PRELOAD, which carries the gate the runner builds in $OUT, splits at spaces and colons and cannot escape them.
[[ $OUT != *[[:space:]:]* ]] || { echo "run sweep.sh from a directory whose path has no spaces or colons"; exit 1; }
if [[ -n "${CONFIGS:-}" ]]; then
    read -ra configs <<< "$CONFIGS"
else
    configs=()
    for f in "$CFG_DIR"/*.yaml; do configs+=("$(basename "$f" .yaml)"); done
fi
(( ${#configs[@]} )) || { echo "no configs to run"; exit 1; }
for c in "${configs[@]}"; do
    [[ -f "$CFG_DIR/$c.yaml" ]] || { echo "no config $CFG_DIR/$c.yaml"; exit 1; }
done

# Every config's runs, one "mode gpus" line each, from the runner's own GPU lists and packages (a package is the AGT
# instance the runner logs a GPU under). -I keeps the current directory off the import path.
plan=$("$PY" -I -B - "$RUNNER" "$AGT" "${GPUS:-0,1}" "${ISOLATED_GPUS:-${GPUS:-0,1}}" "${MODES:-isolated concurrent}" \
       << 'EOF'
import os, sys
sys.path.insert(0, os.path.dirname(sys.argv[1]))
import run_tensile_with_agt as r
gpus = r.kfd_gpus()
if not gpus:
    r.fail("no GPUs in the KFD topology")
agt = os.path.abspath(sys.argv[2])
instances, listing = r.agt_gpu_instances(agt)
if not instances:
    r.fail(f"`sudo -n {agt} -i` listed no GPUs:\n{listing[-2000:]}")
selected, isolated = (r.parse_gpus(",".join(a.replace(",", " ").split()), len(gpus)) for a in sys.argv[3:5])
lay = r.layout(gpus, instances, sorted(set(selected + isolated)))
planned = {}
for mode in sys.argv[5].split():
    if mode == "isolated":
        sets = [[d] for d in isolated]
    elif mode == "concurrent":
        sets = [[d for d in selected if lay[d]["inst"] == i] for i in dict.fromkeys(lay[d]["inst"] for d in selected)]
    elif mode == "all":
        sets = [selected]
    else:
        r.fail(f"unknown mode {mode!r} (isolated, concurrent or all)")
    for s in sets:
        if mode == "isolated" or len(s) > 1:
            planned.setdefault(",".join(map(str, s)), mode)
if not planned:
    r.fail(f"nothing to run: MODES {sys.argv[5]!r} on GPUS {selected}, ISOLATED_GPUS {isolated}")
for gpu_set, mode in planned.items():
    print(mode, gpu_set)
EOF
) || exit 1
mapfile -t runs <<< "$plan"

mkdir -p "$LOGS" || exit 1
printf 'config\tmode\tgpus\trc\trundir\n' > "$LOGS/status.tsv"
say() { echo "$(date +%T) $*"; }
list=$(printf '%s; ' "${runs[@]}")
say "${#configs[@]} config(s) x ${#runs[@]} run(s): ${list%; }"

failed=0
interrupted=0
# Ctrl-C reaches the runner too, which then finishes its batch, stops agt_internal and writes its summary; the
# sweep stops after that run. tee ignores it, so the runner never writes into a closed pipe while it cleans up.
trap 'interrupted=1' INT TERM
for c in "${configs[@]}"; do
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
    { printf '## %s, %s on GPU(s) %s\n\n' "$c" "$m" "$gpus"; cat "$rundir/summary.md"; echo; } >> "$LOGS/summary.md"
done < <(tail -n +2 "$LOGS/status.tsv")

say "sweep done: $(( $(wc -l < "$LOGS/status.tsv") - 1 )) run(s), $failed failed$( (( interrupted )) && echo ', interrupted')"
say "status, summary.csv and summary.md in $LOGS"
(( failed == 0 && ! interrupted ))
