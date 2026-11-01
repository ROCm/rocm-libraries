#!/bin/bash
# RT mutant builds (alola CPUONLY&MARKHAM): one librocke_codeobj_ffi.so per hardware-gate mutant of rt_mutants.py,
# for the GPU job (rt_gpu.sh step mut:...) whose numerics must fail on each.
#   sbatch ... --export=ALL,TAG=<tag>,RT_STAGE=<tree stage> rt_mut.sh   (rt_mutants.py beside this script)
# Outputs: q38-rt/out/$TAG/mut_*.so.
set -uo pipefail
B=$HOME/rocke-sdpa-claude/q38-rt
TAG=${TAG:?}
OUT=$B/out/$TAG
J=/tmp/vanantha-rtm-${SLURM_JOB_ID:-$$}
export RUSTUP_HOME=$HOME/rocke-sdpa-claude/d950-rust/rustup
export CARGO_HOME=$J/cargo PATH=$RUSTUP_HOME/toolchains/1.99.1-x86_64-unknown-linux-gnu/bin:$PATH CARGO_NET_OFFLINE=true
mkdir -p "$J/tree" "$OUT"
echo "== node $(hostname) job ${SLURM_JOB_ID:-none} $(date -u +%FT%TZ)"
tar -xf "$B/stages/${RT_STAGE:?}.tar" -C "$J/tree" || { echo "unpack failed"; exit 125; }
PY=python3
[ -x "$HOME/ckdsl-venv/bin/python" ] && PY=$HOME/ckdsl-venv/bin/python
$PY "$B/jobs/rt_mutants.py" build "$J/tree" "$J/target" "$OUT"
rc=$?
ls -la "$OUT"
rm -rf "$J"
echo "== rc=$rc $(date -u +%FT%TZ)"
exit $rc
