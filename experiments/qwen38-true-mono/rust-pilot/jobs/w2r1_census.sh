#!/bin/bash
# R1 no-regression gate (alola CPUONLY&MARKHAM): compile the gfx950 coverage census (2,029 IRs of
# stage3-after-g950n) and the gfx1151 18-kernel corpus with target_compile built from (a) the rust
# branch tip crate and (b) the R1 crate; every kernel must get the same status and, when compiled,
# the same object byte for byte. Differences are allowed only where (a) failed with a non-fallback
# error and (b) refuses with ROCKE_DIRECT_UNSUPPORTED (listed). Also bench_corpus on the gfx1151
# corpus (objects equal the corpus's).
#   sbatch ... --export=ALL,TAG=<tag> w2r1_census.sh
set -uo pipefail
B=$HOME/rocke-sdpa-claude/q38-w2r1
TAG=${TAG:?}
OUT=$B/out/$TAG
J=/tmp/vanantha-w2r1c-${SLURM_JOB_ID:-$$}
NP=${SLURM_CPUS_ON_NODE:-16}
export RUSTUP_HOME=$HOME/rocke-sdpa-claude/d950-rust/rustup
export CARGO_HOME=$J/cargo PATH=$RUSTUP_HOME/toolchains/1.99.1-x86_64-unknown-linux-gnu/bin:$PATH CARGO_NET_OFFLINE=true
mkdir -p "$J"/{a,b,ir,o} "$OUT"
echo "== node $(hostname) job ${SLURM_JOB_ID:-none} cpus $NP $(date -u +%FT%TZ)"
tar -xzf "$B/up/tip-crate.tgz" -C "$J/a" && tar -xf "$B/stages/w2-r1-rust.tar" -C "$J/b" platform/rust/rocke-codeobj \
    && tar -xzf "$B/up/ir950.tgz" -C "$J/ir" && tar -xzf "$B/up/corpus1151.tgz" -C "$J/ir" || { echo "unpack failed"; exit 125; }
CA=$J/a/rocke-codeobj CB=$J/b/platform/rust/rocke-codeobj
for s in a b; do
    c=$CA; [ $s = b ] && c=$CB
    (cd "$c" && CARGO_TARGET_DIR=$J/t$s cargo build --release --locked --offline --example target_compile --example bench_corpus > "$OUT/build_$s.txt" 2>&1) \
        || { echo "build $s failed"; tail -20 "$OUT/build_$s.txt"; exit 1; }
done
echo "== built both $(date -u +%T)"
run() { # side gfx list outdir
    local tc=$J/t$1/release/examples/target_compile
    xargs -a "$3" -n 40 -P "$NP" "$tc" "$2" "$4" 2> /dev/null | sort > "$4.status"
}
ls "$J"/ir/ir950/*.ir > "$J/l950"; ls "$J"/ir/corpus/*.ir > "$J/l1151"
echo "== inputs $(wc -l < "$J/l950") gfx950, $(wc -l < "$J/l1151") gfx1151"
for s in a b; do
    run $s gfx950 "$J/l950" "$J/o/${s}950"
    run $s gfx1151 "$J/l1151" "$J/o/${s}1151"
done
echo "== compiled $(date -u +%T)"
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
REFUSALS = ("is not lowered by the direct backend", 'tile.inline_asm other than the empty "=v,0" barrier',
            "kernarg preload (inreg)", "kernel symbol", "a pointer to ", "unsupported element type",
            "kernel attribute", "only global pointers", "unsupported kernel argument type")
for gfx in ("950", "1151"):
    a, b = load("a", gfx), load("b", gfx)
    n_ok = n_same = n_err_same = 0
    allowed = []
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
        elif ra[0] != "ok" and rb[0] != "ok" and any(m in rb[1] for m in REFUSALS):
            # target_compile prints the message without the kind: the R1 refusal messages are all
            # ROCKE_DIRECT_UNSUPPORTED (an unknown op was ROCKE_DIRECT_PARSE before; the asm
            # refusal moved from isel to validation, both UNSUPPORTED).
            allowed.append(f"gfx{gfx} {stem}: {ra[1][:200]}  ->  {rb[1][:200]}")
        else:
            bad += 1; lines.append(f"gfx{gfx} {stem}: STATUS {ra} -> {rb}")
    lines.append(f"gfx{gfx}: {len(a)} kernels, {n_ok} compiled by both ({n_same} byte-identical), {n_err_same} same error, {len(allowed)} now refused as unsupported")
    lines += ["  allowed: " + x for x in allowed]
lines.append(f"CENSUS {'OK' if not bad else 'FAIL'} ({bad} unexpected differences)")
open(f"{out}/census_compare.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines[-40:]))
sys.exit(1 if bad else 0)
EOF
rc=$?
for s in a b; do
    (cd "$J/ir/corpus" && "$J/t$s/release/examples/bench_corpus" "$J/ir/corpus" 1 > "$OUT/bench_corpus_$s.txt" 2>&1)
    echo "== bench_corpus $s rc=$? $(tail -n 2 "$OUT/bench_corpus_$s.txt" | tr '\n' ' ')"
done
cp "$J"/o/*.status "$OUT/"
rm -rf "$J"
echo "== rc=$rc $(date -u +%FT%TZ)"
exit $rc
