#!/usr/bin/env bash
# Qwen3.8-Flash-Next true rocKE monokernel stack ("consolidated stack 2"): one entry point for the reproduction.
# Recipe and expected ratios: REPRO.md (this directory). Ratios only in this tree.
#
#   bash repro.sh check                  local, CPU: pristine exports + patches apply, trees equal the measured
#                                        stages, files present, knob lint
#   bash repro.sh stage <sfx>            build the patched trees locally and upload them as q38 stages
#                                        (atom-truemono-<sfx>, hb-atom-<sfx>, rocke-qk-<sfx>, h-qk-<sfx>, rp-h-<sfx>)
#   bash repro.sh gen <sfx>              cluster job: regenerate the stack's stage functions (gen_c2Ab, gen_e2ecomp2)
#                                        and compare them with the measured jobs' manifests
#   bash repro.sh gate <sfx>             cluster job: gen + the c2 gates (stacked layer bitwise + mutants, RB, CS tail)
#   bash repro.sh quick <sfx>            cluster job: one node, hybrid / hybrid_b / c2s3b / c2s3b_b / c2s3b_n8_64k at K=3,
#                                        hybrid / hybrid_b / c2s0b / c2s0b_b at K=0, 32 gsm8k prompts, $PASSES passes
#   bash repro.sh full <sfx>             cluster jobs: quick on both rocke nodes with 4 passes + the T4 accuracy job
#   bash repro.sh wait <tag>             wait for a job's "== rc=" line; fetch its log and results to $OUT/<tag>
#   bash repro.sh t4 <tag>               T4 verdict for a finished accuracy job (login node: acc_stats.py + t4_gate.py)
#   bash repro.sh files                  refresh FILES-*.txt and patches/pending/* from the live trees (read-only git)
#
# Settings (env): DEV (/c/Users/vanantha/Documents/dev), ATOM_REPO ($DEV/ATOM), ROCKE_REPO ($DEV/copy-rocm-lib),
#   ATOM_TM ($DEV/ATOM-truemono), HARNESS ($DEV/hackathon-prep), RP_H ($HARNESS/qwen38/truemono-repro),
#   WORK ($RP_H/work), OUT ($RP_H/results), PY (venv python), CL_HOST (vanantha@ctr2-alola-login-01),
#   NODE (Q38_NODE pin; default: none), GPUS (1), PASSES (2 for quick, 4 for full), SEED (0), TAG (rp-<mode>-<sfx>),
#   Q38_MAX_GPUS (14), KS ("3 0"), ARMS_K3 / ARMS_K0 (the arm lists below).
# Hard rules this script keeps: read-only git only (archive / diff / cat-file / merge-base); never deletes files
# (every build goes to a new timestamped directory); no absolute performance numbers are printed into this tree.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV=${DEV:-/c/Users/vanantha/Documents/dev}
ATOM_REPO=${ATOM_REPO:-$DEV/ATOM}
ROCKE_REPO=${ROCKE_REPO:-$DEV/copy-rocm-lib}
ATOM_TM=${ATOM_TM:-$DEV/ATOM-truemono}
HARNESS=${HARNESS:-$DEV/hackathon-prep}
RP_H=${RP_H:-$HARNESS/qwen38/truemono-repro}
WORK=${WORK:-$RP_H/work}
OUT=${OUT:-$RP_H/results}
export CL_HOST=${CL_HOST:-vanantha@ctr2-alola-login-01}
if [ -z "${PY:-}" ]; then
  if [ -x "$DEV/.venv-ckdsl/Scripts/python.exe" ]; then PY=$DEV/.venv-ckdsl/Scripts/python.exe; else PY=$(command -v python3 || command -v python); fi
fi
AUS=/home_aus/AMD/vanantha/rocke-sdpa-claude/q38

# ---- the pinned versions (REPRO.md "What to check out")
ATOM_BASE=352229d6d4a756cf58146bc5cb37519dd5d5aefd       # proaditya/ATOM qwen38-megakernels
ROCKE_C2=4478c174a13                                      # copy-rocm-lib users/vanantha/rocke-monokernels, + patch 03
ROCKE_HYB=89209ac7665                                     # the pinned hybrid rocKE (stage rocke-tmbase)
ROCKE_DEVELOP_BASE=9b4fee7d99071ca2e40ae74dfa043919fb35d90a  # merge-base of the branch with origin/develop
RK_SUB=dnn-providers/hip-kernel-provider/rocke
H_FROZEN_SHA=ce005780d12f7bad3b715867461c1fb8db2cf446e9c0d17bbc5e1ea8164a3ddc  # harness stage h-qk-c2-f
RK_TMBASE_SHA=f6f644d4f0a1c8805c4876791556898d9b941c78d18e4f7c4fe5533c8a3fb6d5  # cluster stage rocke-tmbase.tar
P=$HERE/patches
ARMS_K3=${ARMS_K3:-hybrid hybrid_b c2s3b c2s3b_b c2s3b_n8_64k}
ARMS_K0=${ARMS_K0:-hybrid hybrid_b c2s0b c2s0b_b}

win() { if command -v cygpath > /dev/null 2>&1; then cygpath -m "$1"; else echo "$1"; fi; }
pyt() { local t=$1; shift; "$PY" "$(win "$HERE/tools/$t")" "$@"; }
ok=0 bad=0
res() { # res PASS|FAIL|WARN <what>
  echo "$1  $2"
  case $1 in PASS) ok=$((ok + 1)) ;; FAIL) bad=$((bad + 1)) ;; esac
}

# ---- trees <dir>: pristine exports + patches -> <dir>/{atom-hybrid,atom-truemono,rocke,rocke-tmbase}
apply_patch() { # apply_patch <tree> <patch>: git apply --check (read-only), then GNU patch writes the files
  local t=$1 p=$2
  (cd "$t" && git apply --check -p1 "$(win "$p")") || return 1
  (cd "$t" && patch -p1 --quiet --forward --no-backup-if-mismatch < "$p") || return 1
}
build_trees() {
  local d=$1 r=0
  mkdir -p "$d/atom-hybrid" "$d/rocke-c2" "$d/rocke-hyb"
  git -C "$ATOM_REPO" -c core.autocrlf=false archive "$ATOM_BASE" | tar -x -C "$d/atom-hybrid" 2> /dev/null
  [ -f "$d/atom-hybrid/atom/models/qwen4_exp_mono/runner.py" ] && res PASS "export ATOM $ATOM_BASE (git archive)" || { res FAIL "export ATOM $ATOM_BASE"; return 1; }
  apply_patch "$d/atom-hybrid" "$P/01-atom-hybrid-rocke-on-352229d6.patch" && res PASS "patch 01 applies to pristine ATOM (hybrid tree)" || { res FAIL "patch 01"; r=1; }
  cp -r "$d/atom-hybrid" "$d/atom-truemono"
  apply_patch "$d/atom-truemono" "$P/02-atom-truemono-c2-on-01.patch" && res PASS "patch 02 applies on top (ATOM-truemono tree)" || { res FAIL "patch 02"; r=1; }
  # tar cannot create the repo's symlinks on Windows (.clinerules, CLAUDE.md): harmless, filtered in the compare
  git -C "$ROCKE_REPO" -c core.autocrlf=false archive "$ROCKE_C2" "$RK_SUB" | tar -x -C "$d/rocke-c2" 2> /dev/null
  git -C "$ROCKE_REPO" -c core.autocrlf=false archive "$ROCKE_HYB" "$RK_SUB" | tar -x -C "$d/rocke-hyb" 2> /dev/null
  [ -d "$d/rocke-c2/$RK_SUB/library/kernels/gfx950/qwen38_mono" ] && res PASS "export rocKE $ROCKE_C2 + $ROCKE_HYB (git archive)" || { res FAIL "export rocKE"; return 1; }
  apply_patch "$d/rocke-c2" "$P/03-rocke-c2f-on-4478c174a13.patch" && res PASS "patch 03 applies to pristine rocKE $ROCKE_C2" || { res FAIL "patch 03"; r=1; }
  return $r
}
cmp_manifest() { # cmp_manifest <tree> <manifest> <label> [ignore globs...]
  local t=$1 m=$2 l=$3; shift 3
  local ig=() g
  for g in '.ruff_cache/*' '*/.ruff_cache/*' .clinerules CLAUDE.md .cursor/rules/ai-rules.mdc "$@"; do ig+=(--ignore "$g"); done
  local o
  o=$(pyt tree_vs_manifest.py "$(win "$t")" "$(win "$m")" "${ig[@]}" 2>&1)
  if [ $? = 0 ]; then res PASS "$l: $(echo "$o" | tail -1)"; else res FAIL "$l: $(echo "$o" | tail -1)"; echo "$o" | head -20 | sed 's/^/      /'; fi
}

mode_check() {
  local d=$WORK/check-$(date +%Y%m%d-%H%M%S)-$$
  echo "== repro check $(date -u) -> $d"
  command -v git > /dev/null && command -v patch > /dev/null && [ -x "$PY" ] && res PASS "tools: git, patch, python ($PY)" || res FAIL "tools (git, GNU patch, python)"
  # checkouts (read-only git)
  git -C "$ATOM_REPO" cat-file -e "$ATOM_BASE^{commit}" 2> /dev/null && res PASS "ATOM commit $ATOM_BASE present ($ATOM_REPO)" || res FAIL "ATOM commit $ATOM_BASE missing in $ATOM_REPO"
  local c
  for c in "$ROCKE_C2" "$ROCKE_HYB"; do
    git -C "$ROCKE_REPO" merge-base --is-ancestor "$c" users/vanantha/rocke-monokernels 2> /dev/null \
      && res PASS "copy-rocm-lib $c is on users/vanantha/rocke-monokernels" || res FAIL "copy-rocm-lib $c not an ancestor of users/vanantha/rocke-monokernels"
  done
  [ "$(git -C "$ROCKE_REPO" merge-base users/vanantha/rocke-monokernels origin/develop 2> /dev/null)" = "$ROCKE_DEVELOP_BASE" ] \
    && res PASS "branch base on develop = ${ROCKE_DEVELOP_BASE:0:11}" || res WARN "branch merge-base with origin/develop moved (recorded ${ROCKE_DEVELOP_BASE:0:11})"
  # files of the package
  local f nmiss=0
  for f in REPRO.md FILES-ATOM.txt FILES-copy-rocm-lib.txt patches/01-atom-hybrid-rocke-on-352229d6.patch \
    patches/02-atom-truemono-c2-on-01.patch patches/03-rocke-c2f-on-4478c174a13.patch knobs/hybrid.env knobs/c2s3b.env \
    knobs/c2s3b_n8_64k.env knobs/c2s0b.env manifests/atom-truemono-c2-f.manifest manifests/hb-atom-c2-f.manifest \
    manifests/rocke-qk-c2-f.manifest manifests/rocke-tmbase.manifest tools/tree_vs_manifest.py tools/tree_diff.py tools/mkpatch.py; do
    [ -s "$HERE/$f" ] || { res FAIL "missing $f"; nmiss=$((nmiss + 1)); }
  done
  [ $nmiss = 0 ] && res PASS "package files present"
  grep -l 'GIT binary patch' "$P"/*.patch > /dev/null 2>&1 && res FAIL "binary hunks in a patch" || res PASS "patches are text-only"
  if grep -n $'\r' "$P"/*.patch > /dev/null 2>&1; then res FAIL "CR in a patch"; else res PASS "patches are LF"; fi
  # trees: apply + compare with the measured stages
  build_trees "$d" || true
  cmp_manifest "$d/atom-hybrid" "$HERE/manifests/hb-atom-c2-f.manifest" "hybrid ATOM tree == stage hb-atom-c2-f (= atom-rkrepro)" \
    atom/entrypoints/openai/serving_anthropic.py
  cmp_manifest "$d/atom-truemono" "$HERE/manifests/atom-truemono-c2-f.manifest" "ATOM-truemono tree == stage atom-truemono-c2-f" \
    atom/entrypoints/openai/serving_anthropic.py
  cmp_manifest "$d/rocke-c2/$RK_SUB" "$HERE/manifests/rocke-qk-c2-f.manifest" "rocKE tree == stage rocke-qk-c2-f"
  cmp_manifest "$d/rocke-hyb/$RK_SUB" "$HERE/manifests/rocke-tmbase.manifest" "hybrid rocKE $ROCKE_HYB == stage rocke-tmbase"
  # harness (frozen h stage of the measured jobs)
  local h=$RP_H/frozen/h-qk-c2-f.tar hx=$d/h
  if [ -f "$h" ] && [ "$(sha256sum "$h" | cut -d' ' -f1)" = "$H_FROZEN_SHA" ]; then
    res PASS "frozen harness h-qk-c2-f.tar sha ${H_FROZEN_SHA:0:16}"
    mkdir -p "$hx" && tar -xf "$h" -C "$hx"
    for f in truemono-quick/quick.sh truemono-quick/arms.d/c2.sh truemono-quick/arms.d/cs.sh truemono-quick/arms.d/tuned.sh \
      truemono-quick/lookback/picksH.json truemono-quick/lookback/gen_tuned.py truemono-quick/n8/ranking.pt \
      truemono/WP2a-fix/picks_final.json truemono/WP2a-fix/wp2a_gen.py truemono/WP2b/picks_all.json truemono/WP2b/wp2b_gen.py \
      truemono/RB/arms_rb.sh truemono/RB/rb_gate.sh truemono/CS/cs_gate.sh truemono/C2/c2_gates.sh truemono/C2/c2_layer_gate.py \
      e2ecomp2/gen.sh common/server.sh; do
      [ -s "$hx/$f" ] || res FAIL "harness file missing: $f"
    done
    res PASS "harness files (recipes, picks, lookback, gates) present"
    local a
    for a in hybrid hybrid_b c2s3b c2s3b_b c2s3b_n8_64k c2s0b c2s0b_b; do
      grep -q -E "^arm_$a\(\)" "$hx"/truemono-quick/arms.sh "$hx"/truemono-quick/arms.d/*.sh || res FAIL "arm $a not defined"
    done
    grep -q '^gen_c2Ab()' "$hx/truemono-quick/arms.d/c2.sh" && res PASS "arms + recipe gen_c2Ab defined" || res FAIL "gen_c2Ab missing"
  else
    res FAIL "frozen harness tar missing or sha differs ($h)"
  fi
  # knob lint: every ATOM_Q38_* knob of every arm is read by the patched trees; module / file references resolve
  local e k n miss
  for e in "$HERE"/knobs/*.env; do
    miss=""
    n=0
    while IFS= read -r k; do
      n=$((n + 1))
      grep -r -q --include='*.py' -F "\"$k\"" "$d/atom-truemono/atom" "$d/atom-hybrid/atom" "$d/rocke-c2/$RK_SUB/library" 2> /dev/null \
        || grep -r -q --include='*.py' -F "'$k'" "$d/atom-truemono/atom" "$d/rocke-c2/$RK_SUB/library" 2> /dev/null \
        || grep -r -q --include='*.py' -E "$k([^A-Z_]|$)" "$d/atom-truemono/atom" "$d/rocke-c2/$RK_SUB/library" 2> /dev/null || miss="$miss $k"
    done < <(grep -o '^ATOM_Q38_[A-Z0-9_]*' "$e")
    [ -z "$miss" ] && res PASS "knob lint $(basename "$e"): $n ATOM_Q38_* knobs, each read by the trees" || res FAIL "knob lint $(basename "$e"): unread:$miss"
  done
  grep -q 'kernels.gfx950.qwen38_mono.truemono_tail:layer_passthrough' "$HERE/knobs/c2s3b.env" \
    && [ -f "$d/rocke-c2/$RK_SUB/library/kernels/gfx950/qwen38_mono/truemono_tail.py" ] \
    && grep -q 'def layer_passthrough' "$d/rocke-c2/$RK_SUB/library/kernels/gfx950/qwen38_mono/truemono_tail.py" \
    && res PASS "tail provider module + layer_passthrough present" || res FAIL "tail provider module"
  grep -q 'ATOM_Q38_DRAFT_VOCAB_N=65536' "$HERE/knobs/c2s3b_n8_64k.env" && grep -q 'ATOM_Q38_DRAFT_VOCAB_N=98304' "$HERE/knobs/c2s3b.env" \
    && res PASS "N8 subsets: c2s3b 96K, c2s3b_n8_64k 64K" || res FAIL "N8 subset sizes"
  if grep -h -E '^ATOM_Q38_(TM_TAIL_DRAFT|RB_DEVMETA)' "$HERE/knobs/c2s0b.env" > /dev/null; then res FAIL "c2s0b carries K=3-only knobs"; else res PASS "c2s0b (K=0) has no K=3-only knobs"; fi
  echo "== check: $ok PASS, $bad FAIL (trees in $d)"
  [ $bad = 0 ]
}

stage_tar_upload() { # stage_tar_upload <local tar> <stage name>: .part + sha check, then rename (the q38_sync.sh pattern)
  local t=$1 n=$2 sum got
  sum=$(sha256sum "$t" | cut -d' ' -f1)
  got=$(ssh -o BatchMode=yes "$CL_HOST" "cat > $AUS/stages/.$n.tar.part; sha256sum $AUS/stages/.$n.tar.part | cut -d' ' -f1" < "$t")
  [ "$got" = "$sum" ] || { echo "upload of $n failed (sha)"; return 1; }
  ssh -o BatchMode=yes "$CL_HOST" "mv -f $AUS/stages/.$n.tar.part $AUS/stages/$n.tar && echo $sum > $AUS/stages/$n.tar.sha256" && echo "SYNC_OK $n sha=${sum:0:16}"
}
mode_stage() {
  local sfx=${1:?stage <sfx>}
  [[ "$sfx" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad sfx"; return 2; }
  local d=$WORK/build/$(date +%Y%m%d-%H%M%S)-$sfx
  echo "== repro stage $sfx -> $d (CL_HOST=$CL_HOST)"
  build_trees "$d" || return 1
  # pack like q38_sync.sh / sync_pack.py (pack_stage.py also handles Windows long paths), then upload with a sha check
  local L=$d/stages p a src
  mkdir -p "$L"
  for p in "atom-truemono-$sfx $d/atom-truemono" "hb-atom-$sfx $d/atom-hybrid" "rocke-qk-$sfx $d/rocke-c2/$RK_SUB" "rp-h-$sfx $RP_H/job"; do
    read -r a src <<< "$p"
    pyt pack_stage.py "$(win "$src")" "$(win "$L/$a.tar")" "$(win "$L/stage-$a.manifest")" || return 1
    stage_tar_upload "$L/$a.tar" "$a" || return 1
  done
  stage_tar_upload "$RP_H/frozen/h-qk-c2-f.tar" "h-qk-$sfx" || return 1
  # the pinned hybrid rocKE stays the cluster's rocke-tmbase (quick.sh hard-codes the name); check it is the recorded one
  local tb
  tb=$(ssh -o BatchMode=yes "$CL_HOST" "cat $AUS/stages/rocke-tmbase.tar.sha256")
  [ "$tb" = "$RK_TMBASE_SHA" ] && echo "rocke-tmbase: recorded sha ${tb:0:16} (= copy-rocm-lib $ROCKE_HYB, see check)" \
    || { echo "rocke-tmbase sha moved: $tb (expected $RK_TMBASE_SHA); stage $ROCKE_HYB as rocke-tmbase first"; return 1; }
  # the uploaded trees are the ones check compares (EOL-tolerant: the measured stages kept CRLF in some non-.py files)
  cmp_manifest "$d/atom-hybrid" "$HERE/manifests/hb-atom-c2-f.manifest" "staged hb-atom-$sfx == hb-atom-c2-f" atom/entrypoints/openai/serving_anthropic.py
  cmp_manifest "$d/atom-truemono" "$HERE/manifests/atom-truemono-c2-f.manifest" "staged atom-truemono-$sfx == atom-truemono-c2-f" atom/entrypoints/openai/serving_anthropic.py
  cmp_manifest "$d/rocke-c2/$RK_SUB" "$HERE/manifests/rocke-qk-c2-f.manifest" "staged rocke-qk-$sfx == rocke-qk-c2-f"
  echo "stages: h-qk-$sfx rocke-tmbase rocke-qk-$sfx atom-truemono-$sfx hb-atom-$sfx rp-h-$sfx"
}

submit() { # submit <tag> <gpus> <time> <stage list> -- <env K=V ...> -- <cmd>
  local tag=$1 g=$2 t=$3 st=$4; shift 4
  local args=(--tag "$tag" --gpus "$g" --time "$t" --cpus $((32 * g)) --mem $((192 * g)) --mode container)
  local s e
  for s in $st; do args+=(--stage "$s"); done
  [ "$1" = -- ] && shift
  while [ $# -gt 0 ] && [ "$1" != -- ]; do args+=(--env "$1"); shift; done
  shift
  mkdir -p "$OUT"
  echo "== submit $tag ($g GPU, $t${NODE:+, node $NODE})"
  Q38_NODE=${NODE:-} Q38_MAX_GPUS=${Q38_MAX_GPUS:-14} bash "$HARNESS/qwen38/truemono-lb/submit.sh" "$OUT/$tag.submit.log" "${args[@]}" -- "$@"
}
COMMON_ENV() { # the env every measured c2 job passed (alola rules 198 / 236, COORD-consolidated.md)
  local sfx=$1
  echo "QK_SFX=$sfx XDG_RUNTIME_DIR=/tmp/vanantha-xdg ENROOT_RUNTIME_PATH=/tmp/vanantha-enroot/runtime CS_GEN_PP=1 WP2B_KNOBS_JSON=picks_all.json WP2B_JSON_ONLY=1"
}
STAGES() { echo "h-qk-$1 rocke-tmbase rocke-qk-$1 atom-truemono-$1 hb-atom-$1 rp-h-$1"; }
ARMFILES() { echo "ARM_FILES=/job/stages/h-qk-$1/truemono/RB/arms_rb.sh /job/stages/h-qk-$1/truemono/WP2a-fix/arms_wp2a.sh /job/stages/h-qk-$1/truemono/WP2b/arms_wp2b.sh"; }

mode_gen() {
  local sfx=${1:?gen <sfx>} steps=${2:-gen}
  local tag=${TAG:-rp-${steps// /}-$sfx}
  # shellcheck disable=SC2046
  submit "$tag" "${GPUS:-1}" 02:30:00 "$(STAGES "$sfx")" -- $(COMMON_ENV "$sfx") "RP_STEPS=$steps" -- "bash /job/stages/rp-h-$sfx/gate_job.sh"
}
mode_quick() {
  local sfx=${1:?quick <sfx>} p=${PASSES:-2}
  local tag=${TAG:-rp-quick-$sfx}
  # shellcheck disable=SC2046
  submit "$tag" "${GPUS:-1}" "${QTIME:-04:00:00}" "$(STAGES "$sfx")" -- $(COMMON_ENV "$sfx") "$(ARMFILES "$sfx")" \
    "KS=${KS:-3 0}" "ARMS_K3=$ARMS_K3" "ARMS_K0=$ARMS_K0" "PASSES=$p" "SEED=${SEED:-0}" \
    -- "bash /job/stages/h-qk-$sfx/truemono-quick/quick.sh"
}
mode_full() {
  local sfx=${1:?full <sfx>}
  PASSES=${PASSES:-4} NODE=smci355-ccs-aus-n01-09 TAG=rp-full-n01-$sfx SEED=${SEED:-131} mode_quick "$sfx"
  PASSES=${PASSES:-4} NODE=smci355-ccs-aus-n07-17 TAG=rp-full-n07-$sfx SEED=${SEED2:-137} mode_quick "$sfx"
  # T4 accuracy (P11 rev 3): GSM8K 1,319 5-shot (CONC 1), HumanEval (CONC 1), RULER mix v1 8K + 32K (CONC 4); the arm and
  # the hybrid, 3 passes each, as job qk-c2-t4b ran it
  local tag=rp-t4-$sfx
  # shellcheck disable=SC2046
  TAG=$tag submit "$tag" 1 12:00:00 "$(STAGES "$sfx")" -- $(COMMON_ENV "$sfx") "$(ARMFILES "$sfx")" \
    KS=3 "ARMS_K3=${T4_ARMS:-hybrid c2s3b}" PASSES=3 SPEED_C1=0 PRIME=0 SEED=93 Q38_MML=262141 "PIP_EXTRA=wonderwords nltk" \
    "CELLS_K3=lmeval:gsm8k:1 acc:humaneval:1:low ruler:8192,32768:4:50:niah_single_1+niah_single_2+niah_multikey_1+niah_multikey_2+niah_multivalue+niah_multiquery+ruler_vt:4096" \
    -- "bash /job/stages/h-qk-$sfx/truemono-quick/quick.sh"
}
mode_wait() {
  local tag=${1:?wait <tag>} f
  echo "== waiting for $tag (log $AUS/logs/$tag-<id>.out)"
  while :; do
    f=$(ssh -o BatchMode=yes "$CL_HOST" "ls -t $AUS/logs/$tag-*.out 2> /dev/null | head -1" 2> /dev/null)
    if [ -n "$f" ] && ssh -o BatchMode=yes "$CL_HOST" "grep -q '^== rc=' $f" 2> /dev/null; then break; fi
    sleep "${POLL:-120}"
  done
  mkdir -p "$OUT/$tag"
  ssh -o BatchMode=yes "$CL_HOST" "cat $f" > "$OUT/$tag/job.out"
  ssh -o BatchMode=yes "$CL_HOST" "cd $AUS/results/$tag && tar -cf - --exclude='*.server.log' --exclude='*.lmeval' --exclude='*.ruler' ." | tar -xf - -C "$OUT/$tag"
  grep -E '^(== rc=|REPRO_|C2_LAYER_GATE|RB GATE|CS_GATE|c2:)' "$OUT/$tag/job.out"
  [ -f "$OUT/$tag/q/ratios.json" ] && sed -n '/== ratios to hybrid/,$p' "$OUT/$tag/job.out" | head -60
}
mode_t4() {
  local tag=${1:?t4 <tag>}
  ssh -o BatchMode=yes "$CL_HOST" "cd $AUS && python3 c2tools/acc_stats.py results/$tag/q --json c2tools/$tag-acc.json > c2tools/$tag-acc.txt && \
    python3 c2tools/t4_gate.py c2tools/$tag-acc.json w9n4a/pool_k3c1.json w9n4a/pool_ruler.json --arm ${T4_ARM:-c2s3b}"
}
mode_files() {
  # FILES-*.txt: what the stack adds / changes, per repository, for `git add` (paths relative to the repo root)
  local f=$HERE/FILES-ATOM.txt g=$HERE/FILES-copy-rocm-lib.txt
  {
    echo "# ATOM (proaditya/ATOM, branch qwen38-megakernels @ ${ATOM_BASE:0:8}): files of the reproduced stack"
    echo "# = patch 01 (the hybrid rocKE work, as in recipes/Qwen3.8-Flash-Next-rocKE.md) + patch 02 (the true-mono work)."
    echo "# Status letter: A = added, M = modified. Not listed (non-qwen, left out of the patches): atom/entrypoints/openai/serving_anthropic.py"
    for p in 01-atom-hybrid-rocke-on-352229d6 02-atom-truemono-c2-on-01; do
      echo "## patch $p"
      awk '/^diff --git /{f=$4; sub(/^b\//,"",f)} /^new file mode/{print "A " f; f=""} /^index /{if (f!="") {print "M " f; f=""}}' "$P/$p.patch"
    done
  } > "$f"
  {
    echo "# copy-rocm-lib (users/vanantha/rocke-monokernels): files for the reproduction."
    echo "# The stack's rocKE code is COMMITTED (4478c174a13 = the measured tree minus the 4 files of patch 03)."
    echo "## patch 03 (measured tree vs $ROCKE_C2; later commits on the branch already contain other versions of these files)"
    awk '/^diff --git /{f=$4; sub(/^b\//,"",f)} /^index /{print "M " f}' "$P/03-rocke-c2f-on-4478c174a13.patch"
    echo "## this reproduction package (new, untracked)"
    (cd "$ROCKE_REPO" && git ls-files --others --exclude-standard -- experiments/qwen38-true-mono/repro | sed 's/^/A /')
    echo "## other uncommitted changes in the working tree right now (not part of the stack; patches/pending/)"
    (cd "$ROCKE_REPO" && git status --porcelain --untracked-files=all | grep -v 'experiments/qwen38-true-mono/repro/' | sed 's/^ M /M /; s/^?? /A /')
  } > "$g"
  local pd=$P/pending
  mkdir -p "$pd"
  (cd "$ROCKE_REPO" && git diff HEAD | tr -d '\r') > "$pd/copy-rocm-lib-worktree-vs-HEAD.patch"
  (cd "$ROCKE_REPO" && git ls-files --others --exclude-standard | grep -v '^experiments/qwen38-true-mono/repro/' | while read -r u; do
    git diff --no-index /dev/null "$u" | tr -d '\r'; done) >> "$pd/copy-rocm-lib-worktree-vs-HEAD.patch"
  echo "wrote $f ($(grep -c '^[AM] ' "$f") files), $g ($(grep -c '^[AM] ' "$g") entries), $pd/copy-rocm-lib-worktree-vs-HEAD.patch"
}

cmd=${1:-help}
shift || true
case $cmd in
  check) mode_check ;;
  stage) mode_stage "$@" ;;
  gen) mode_gen "${1:-}" gen ;;
  gate) mode_gen "${1:-}" "gen gate" ;;
  quick) mode_quick "$@" ;;
  full) mode_full "$@" ;;
  wait) mode_wait "$@" ;;
  t4) mode_t4 "$@" ;;
  files) mode_files ;;
  *) sed -n '2,30p' "$0" ;;
esac
