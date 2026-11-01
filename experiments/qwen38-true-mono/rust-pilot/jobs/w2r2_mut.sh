#!/bin/bash
# R2 mutant builds (alola CPUONLY&MARKHAM): one librocke_codeobj_ffi.so per hardware-gate mutant of r2_mutants.py,
# for the GPU job (w2r2_mutgpu.sh) whose numerics must fail on each.
#   sbatch ... --export=ALL,TAG=<tag> w2r2_mut.sh
# Inputs (Markham home): $B/stages/w2-r2-rust.tar, $B/stages/w2-r2-tools.tar. Outputs: $B/out/$TAG/mut_*.so.
set -uo pipefail
B=$HOME/rocke-sdpa-claude/q38-w2r2
TAG=${TAG:?}
OUT=$B/out/$TAG
J=/tmp/vanantha-w2r2m-${SLURM_JOB_ID:-$$}
export RUSTUP_HOME=$HOME/rocke-sdpa-claude/d950-rust/rustup
export CARGO_HOME=$J/cargo PATH=$RUSTUP_HOME/toolchains/1.99.1-x86_64-unknown-linux-gnu/bin:$PATH CARGO_NET_OFFLINE=true
mkdir -p "$J"/{tree,tools} "$OUT"
echo "== node $(hostname) job ${SLURM_JOB_ID:-none} $(date -u +%FT%TZ)"
tar -xf "$B/stages/w2-r2-rust.tar" -C "$J/tree" && tar -xf "$B/stages/w2-r2-tools.tar" -C "$J/tools" || { echo "unpack failed"; exit 125; }
PY=python3
[ -x "$HOME/ckdsl-venv/bin/python" ] && PY=$HOME/ckdsl-venv/bin/python
$PY "$J/tools/r2_mutants.py" build "$J/tree" "$J/target" "$OUT"
rc=$?
ls -la "$OUT"
rm -rf "$J"
echo "== rc=$rc $(date -u +%FT%TZ)"
exit $rc
