#!/bin/bash
# R1 CPU job (alola CPUONLY&MARKHAM): build librocke_codeobj_ffi.so and the gate examples from the
# overlaid rust tree with the private offline toolchain, run the crate gates, keep the .so.
#   sbatch ... --export=ALL,TAG=<tag> w2r1_cpu.sh
# Inputs (Markham home): $B/stages/w2-r1-rust.tar (+ .sha256). Outputs: $B/out/$TAG/ (logs, so/, bin/).
set -uo pipefail
B=$HOME/rocke-sdpa-claude/q38-w2r1
TAG=${TAG:?}
OUT=$B/out/$TAG
J=/tmp/vanantha-w2r1-${SLURM_JOB_ID:-$$}
export RUSTUP_HOME=$HOME/rocke-sdpa-claude/d950-rust/rustup
TC=$RUSTUP_HOME/toolchains/1.99.1-x86_64-unknown-linux-gnu/bin
export CARGO_HOME=$J/cargo CARGO_TARGET_DIR=$J/target PATH=$TC:$PATH CARGO_NET_OFFLINE=true
mkdir -p "$J/tree" "$OUT/so" "$OUT/bin" "$OUT/logs"
echo "== node $(hostname) job ${SLURM_JOB_ID:-none} cpus ${SLURM_CPUS_ON_NODE:-?} $(date -u +%FT%TZ)"
st=$B/stages/w2-r1-rust.tar
[ "$(sha256sum "$st" | cut -d' ' -f1)" = "$(cat "$st.sha256")" ] || { echo "stage sha mismatch"; exit 125; }
tar -xf "$st" -C "$J/tree" || { echo "unpack failed"; exit 125; }
C=$J/tree/platform/rust/rocke-codeobj
cd "$C" || exit 126
cargo --version; rustc --version
rc=0
step() { # name cmd...
    local name=$1; shift
    local t0=$SECONDS
    "$@" > "$OUT/logs/$name.txt" 2>&1; local r=$?
    echo "== step $name rc=$r ($((SECONDS - t0)) s): $(grep -E '^test result|passed|failed|error' "$OUT/logs/$name.txt" | tail -n 2 | tr '\n' ' ')"
    [ $r = 0 ] || rc=1
    return $r
}
step build cargo build --release --locked --offline --lib
step build_examples cargo build --release --locked --offline --examples
step cargo_test_lib cargo test --release --locked --offline --lib
step cargo_test_all cargo test --release --locked --offline
# Single-threaded test run: the compiler is thread-count invariant (ROCKE_DIRECT_THREADS=1).
ROCKE_DIRECT_THREADS=1 ROCKE_DIRECT_PARSE_THREADS=1 step cargo_test_1t cargo test --release --locked --offline --lib
PY=python3
for cand in "$HOME/ckdsl-venv/bin/python" "$HOME/rocke-sdpa-claude/venv/bin/python"; do [ -x "$cand" ] && { PY=$cand; break; }; done
echo "== python $PY $($PY -V 2>&1)"
cd "$J/tree/platform" || exit 126
step isa_tables_check env PYTHONPATH=python $PY tools/isa_tables.py check
step isa_tables_check_gfx950 env PYTHONPATH=python $PY tools/isa_tables.py check --isa gfx950
if $PY -c 'import pytest' 2> /dev/null; then
    step pytest_tools env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 $PY -m pytest -p no:cacheprovider -q tests/tools/test_isa_tables.py tests/tools/test_isa_xcheck.py tests/tools/test_isa_burs.py
else
    echo "== step pytest_tools SKIPPED (no pytest for $PY)"
fi
cp "$CARGO_TARGET_DIR/release/librocke_codeobj_ffi.so" "$OUT/so/" && (cd "$OUT/so" && sha256sum librocke_codeobj_ffi.so | tee librocke_codeobj_ffi.so.sha256)
for ex in hazard_corpus target_compile bench950 isa_xcheck; do cp "$CARGO_TARGET_DIR/release/examples/$ex" "$OUT/bin/" 2> /dev/null; done
ls -la "$OUT/so" "$OUT/bin"
rm -rf "$J"
echo "== rc=$rc $(date -u +%FT%TZ)"
exit $rc
