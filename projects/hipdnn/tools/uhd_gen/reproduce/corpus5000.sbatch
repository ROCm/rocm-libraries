#!/bin/bash
#SBATCH --job-name=uhd-corpus5000
#SBATCH --partition=defq
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
#
# Build one 5000-graph corpus PER ENGINE, filtered to the facets that engine can serve.
#
# Per engine, not one shared corpus, and the reason is measured: AITER's first gfx950 L1
# trained on the 68 graphs it could serve out of a 1000-graph corpus drawn from rocKE's
# geometries, and came out systematically under-predicting by -285 TFLOPS. Rebuilt against
# a corpus of what AITER actually serves (mask-restricted, bf16, d128), the same pipeline
# produced -80 TFLOPS bias and took engine selection from 78.9% to 94.2% correct.
#
# A corpus an engine mostly declines is not a smaller training set; it is a biased one --
# the declines are not random, they are one end of the shape distribution.
set -uo pipefail

SRC=/tmp/uhd/src
OUT=${UHD_OUT:-/exchange/corpus-5000}
# sbatch's --output lands in the job's OWN view of the submit directory, which inside a
# --container-remap-root container is not the host home that /exchange maps to: the file
# is written and then thrown away with the container. Everything is teed into /exchange
# from the first line instead, because a job whose failure is invisible costs a whole
# run to diagnose (this one: 18 minutes to learn nothing).
mkdir -p "$OUT"
exec > >(tee "$OUT/job-${SLURM_JOB_ID:-manual}.log") 2>&1
BRANCH=${UHD_BRANCH:-users/jascampb/uhd-integration-test-branch}
COUNT=${UHD_COUNT:-5000}
SEED=${UHD_SEED:-0}
step() { echo; echo "===== $* ====="; date -Is; }

step "toolchain"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq git python3.12-venv python3-dev 2>&1 | tail -1
python3 -m pip install --quiet --break-system-packages pandas numpy flatbuffers 2>&1 | tail -1

step "source: $BRANCH"
rm -rf /tmp/uhd; mkdir -p /tmp/uhd "$OUT"
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 200 --single-branch --branch "$BRANCH" \
    https://github.com/ROCm/rocm-libraries.git "$SRC" > /tmp/uhd/clone.log 2>&1 \
    || { echo "FATAL: clone failed"; tail -5 /tmp/uhd/clone.log; exit 1; }
for ref in ${UHD_PREREQ_REFS:-}; do
    git -C "$SRC" fetch --unshallow origin "$ref" >> /tmp/uhd/clone.log 2>&1 \
        || git -C "$SRC" fetch origin "$ref" >> /tmp/uhd/clone.log 2>&1 \
        || { echo "FATAL: prerequisite ref $ref did not fetch"; tail -5 /tmp/uhd/clone.log; exit 1; }
done
if [ -n "${UHD_BUNDLE:-}" ]; then
    git -C "$SRC" fetch "$UHD_BUNDLE" "${UHD_COMMIT:-HEAD}" >> /tmp/uhd/clone.log 2>&1 \
        && git -C "$SRC" checkout --detach FETCH_HEAD >> /tmp/uhd/clone.log 2>&1 \
        || { echo "FATAL: bundle did not apply"; tail -8 /tmp/uhd/clone.log; exit 1; }
fi
git -C "$SRC" log --oneline -1

TOOLS="$SRC/projects/hipdnn/tools"
PACKS="$SRC/dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors/rocKE"

# name|extra corpus_build arguments
#
# rocKE draws from its own pack's geometries: its kernels are compiled per exact shape
# (Gfx950AttentionDenseNative.cpp matches num_query_heads/seqlen_q/seqlen_kv/batch
# exactly), so a graph outside the pack's 513 shapes has no candidate at all.
#
# AITER's gfx950 forward table is two kernels, bf16/d128, and NO causal mask -- a general
# SDPA corpus is ~90% causal, so nine measurements in ten would be spent on a decline.
CORPORA=$(cat <<'ROWS'
# --kernel-share 1.0: rocKE's kernels are compiled per exact shape, so a graph that is
# not one of the pack's own geometries has no candidate and contributes a decline.
# At the default mix (kernel 0.60, model 0.15, sweep 0.25) a 5000-graph request
# yielded 905 servable graphs and spent 4095 measurements' worth of wall time on
# shapes the engine refuses. The pack is the ceiling either way -- 664 geometries on
# gfx942, 847 on gfx950, doubled by the two causal anchors -- so asking for 5000 gets
# the ceiling, and now every drawn graph is one the engine can actually run.
rocke-gfx942|--kdp-root PACKS/gfx942_attention_dense --min-candidates 2 --head-dim 64 --head-dim 128 --kernel-share 1.0 --model-share 0 --sweep-share 0
rocke-gfx950|--kdp-root PACKS/gfx950_attention_dense --min-candidates 2 --head-dim 64 --head-dim 128 --kernel-share 1.0 --model-share 0 --sweep-share 0
aiter-gfx942|--dtype bf16 --head-dim 128 --causal 0
aiter-gfx950|--dtype bf16 --head-dim 128 --causal 0
flydsl-gfx950|--dtype bf16 --dtype fp16 --head-dim 128 --min-candidates 1
ROWS
)

cd "$TOOLS" || exit 1
rc_all=0
while IFS='|' read -r name args; do
    [ -n "$name" ] || continue
    step "corpus $name"
    args=${args//PACKS/$PACKS}
    dest="$OUT/$name"
    rm -rf "$dest"
    # shellcheck disable=SC2086
    python3 -m corpus_build --out "$dest" --count "$COUNT" --seed "$SEED" $args \
        > "$OUT/$name.log" 2>&1
    rc=$?
    echo "rc=$rc"
    tail -4 "$OUT/$name.log"
    if [ $rc -ne 0 ]; then rc_all=1; continue; fi
    python3 - "$dest/manifest.json" <<'PY'
import json, sys, collections
manifest = json.load(open(sys.argv[1]))
graphs = manifest["graphs"] if isinstance(manifest, dict) else manifest
print(f"  graphs: {len(graphs)}")
for facet in ("dtype", "head_dim", "causal", "phase"):
    counts = collections.Counter(str(g.get(facet)) for g in graphs)
    print(f"  {facet}: {dict(sorted(counts.items()))}")
PY
done <<< "$CORPORA"

step "staged"
du -sh "$OUT"/*/ 2>/dev/null
exit $rc_all
