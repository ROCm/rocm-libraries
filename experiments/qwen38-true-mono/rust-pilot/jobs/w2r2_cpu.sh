#!/bin/bash
# R2 CPU job (alola CPUONLY&MARKHAM): build librocke_codeobj_ffi.so and the gate examples from the R2 rust tree with
# the private offline toolchain, run the crate gates, and the no-regression census against the R1 crate: every
# gfx950 census IR and gfx1151 corpus IR compiles to the same status and, when both compile, the same object byte
# for byte (differences only where R1 refused with ROCKE_DIRECT_UNSUPPORTED and R2 now lowers, listed).
#   sbatch ... --export=ALL,TAG=<tag> w2r2_cpu.sh
# Inputs (Markham home): $B/stages/w2-r2-rust.tar (+ .sha256), $R1/stages/w2-r1-rust.tar, $R1/up/{ir950,corpus1151}.tgz.
# Outputs: $B/out/$TAG/ (logs, so/, bin/, census_compare.txt).
set -uo pipefail
B=$HOME/rocke-sdpa-claude/q38-w2r2
R1=$HOME/rocke-sdpa-claude/q38-w2r1
TAG=${TAG:?}
OUT=$B/out/$TAG
J=/tmp/vanantha-w2r2-${SLURM_JOB_ID:-$$}
NP=${SLURM_CPUS_ON_NODE:-16}
export RUSTUP_HOME=$HOME/rocke-sdpa-claude/d950-rust/rustup
TC=$RUSTUP_HOME/toolchains/1.99.1-x86_64-unknown-linux-gnu/bin
export CARGO_HOME=$J/cargo PATH=$TC:$PATH CARGO_NET_OFFLINE=true
mkdir -p "$J"/{tree,a,ir,o} "$OUT/so" "$OUT/bin" "$OUT/logs"
echo "== node $(hostname) job ${SLURM_JOB_ID:-none} cpus $NP $(date -u +%FT%TZ)"
st=$B/stages/w2-r2-rust.tar
[ "$(sha256sum "$st" | cut -d' ' -f1)" = "$(cat "$st.sha256")" ] || { echo "stage sha mismatch"; exit 125; }
tar -xf "$st" -C "$J/tree" && tar -xf "$R1/stages/w2-r1-rust.tar" -C "$J/a" platform/rust/rocke-codeobj \
    && tar -xzf "$R1/up/ir950.tgz" -C "$J/ir" && tar -xzf "$R1/up/corpus1151.tgz" -C "$J/ir" || { echo "unpack failed"; exit 125; }
C=$J/tree/platform/rust/rocke-codeobj
CA=$J/a/platform/rust/rocke-codeobj
cargo --version; rustc --version
rc=0
step() { # name cmd...
    local name=$1; shift
    local t0=$SECONDS
    "$@" > "$OUT/logs/$name.txt" 2>&1; local r=$?
    echo "== step $name rc=$r ($((SECONDS - t0)) s): $(grep -E '^test result|passed|failed|error|OK|FAIL' "$OUT/logs/$name.txt" | tail -n 2 | tr '\n' ' ')"
    [ $r = 0 ] || rc=1
    return $r
}
cd "$C" || exit 126
export CARGO_TARGET_DIR=$J/tb
step build cargo build --release --locked --offline --lib
step build_examples cargo build --release --locked --offline --examples
nw=$(cat "$OUT/logs/build.txt" "$OUT/logs/build_examples.txt" | grep -c '^warning'); echo "== build warnings: $nw"; [ "$nw" = 0 ] || rc=1
step cargo_test_lib cargo test --release --locked --offline --lib
step cargo_test_all cargo test --release --locked --offline
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
for ex in hazard_corpus target_compile bench950 isa_xcheck mir_dump; do cp "$CARGO_TARGET_DIR/release/examples/$ex" "$OUT/bin/" 2> /dev/null; done
# ---- census: R1 crate (a) vs R2 crate (b)
(cd "$CA" && CARGO_TARGET_DIR=$J/ta cargo build --release --locked --offline --example target_compile > "$OUT/logs/build_a.txt" 2>&1) \
    || { echo "build a failed"; exit 1; }
run() { # side gfx list outdir
    local tc=$J/t$1/release/examples/target_compile
    mkdir -p "$4"
    xargs -a "$3" -n 40 -P "$NP" "$tc" "$2" "$4" 2> /dev/null | sort > "$4.status"
}
ls "$J"/ir/ir950/*.ir > "$J/l950"; ls "$J"/ir/corpus/*.ir > "$J/l1151"
echo "== census inputs $(wc -l < "$J/l950") gfx950, $(wc -l < "$J/l1151") gfx1151"
for s in a b; do
    run $s gfx950 "$J/l950" "$J/o/${s}950"
    run $s gfx1151 "$J/l1151" "$J/o/${s}1151"
done
python3 - "$J/o" "$OUT" << 'EOF'
import hashlib, os, sys
o, out = sys.argv[1], sys.argv[2]
def load(side, gfx):
    st = {}
    for ln in open(f"{o}/{side}{gfx}.status"):
        parts = ln.rstrip("\n").split("\t")
        if len(parts) >= 2:
            st[parts[0]] = (parts[1], "\t".join(parts[2:]))
    return st
def sha(side, gfx, stem):
    p = f"{o}/{side}{gfx}/{stem}.hsaco"
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
lines, bad = [], 0
# R2's refusal messages (all ROCKE_DIRECT_UNSUPPORTED)
REFUSED_R2 = ('"=s,0" pin of', 'tile.inline_asm other than an empty or comment-only')
for gfx in ("950", "1151"):
    a, b = load("a", gfx), load("b", gfx)
    n_ok = n_same = n_err_same = 0
    newly, refined = [], []
    for stem in sorted(set(a) | set(b)):
        ra, rb = a.get(stem), b.get(stem)
        if ra is None or rb is None:
            bad += 1; lines.append(f"gfx{gfx} {stem}: missing in {'a' if ra is None else 'b'}"); continue
        if ra[0] == "ok" and rb[0] == "ok":
            n_ok += 1
            if sha("a", gfx, stem) == sha("b", gfx, stem) and ra[1] == rb[1]:
                n_same += 1
            else:
                bad += 1; lines.append(f"gfx{gfx} {stem}: OBJECT DIFFERS {ra[1]} | {rb[1]}")
        elif ra == rb:
            n_err_same += 1
        elif ra[0] != "ok" and rb[0] == "ok" and any(m in ra[1] for m in ("not lowered by the direct backend", "kernarg preload", "inline_asm other than")):
            newly.append(f"gfx{gfx} {stem}: {ra[1][:160]}  ->  now compiles")
        elif ra[0] != "ok" and rb[0] != "ok" and "inline_asm other than" in ra[1] and any(m in rb[1] for m in REFUSED_R2):
            # an asm refusal (ROCKE_DIRECT_UNSUPPORTED) that R2 refuses more precisely, still unsupported
            refined.append(f"gfx{gfx} {stem}: {ra[1][:120]}  ->  {rb[1][:120]}")
        else:
            bad += 1; lines.append(f"gfx{gfx} {stem}: STATUS {ra} -> {rb}")
    lines.append(f"gfx{gfx}: {len(a)} kernels, {n_ok} compiled by both ({n_same} byte-identical), {n_err_same} same error, "
                 f"{len(newly)} newly lowered, {len(refined)} refusals reworded")
    lines += ["  newly: " + x for x in newly] + ["  reworded: " + x for x in refined]
lines.append(f"CENSUS {'OK' if not bad else 'FAIL'} ({bad} unexpected differences)")
open(f"{out}/census_compare.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines[-40:]))
sys.exit(1 if bad else 0)
EOF
[ $? = 0 ] || rc=1
cp "$J"/o/*.status "$OUT/"
rm -rf "$J"
echo "== rc=$rc $(date -u +%FT%TZ)"
exit $rc
