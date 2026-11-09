# 3a: the lookback picks (lookback/run.py PICK lines, collected in lookback/picks*.json) served on the candidate base
# hyb_tm: per-kind function dirs (the hybrid's set with each pick's variant rewritten) and sched.BEST with each pick's
# group (lookback/gen_tuned.py). PICKS (default picks.json) selects the file; one arm per file below.
gen_tuned() { # gen_tuned <rocke stage> <out dir> [picks file]
  local base=/job/fns/e2ecomp2-$1
  [ -f "$base/manifest.sha256" ] || gen_e2ecomp2 "$1" "$base" || return 1
  (cd "$H/truemono-quick/lookback" && PYTHONPATH=/atom:/job/stages/$1/platform/python:/job/stages/$1/library \
    ROCKE_BACKEND=python ROCKE_LLVM_FLAVOR=llvm23 python gen_tuned.py --base "$base" ${GT_EXTRA:+--kind-extra "$GT_EXTRA"} \
    --picks "$H/truemono-quick/lookback/${3:-picks.json}" --out "$2") || return 1
  (cd "$2" && find k -type f | sort | xargs sha256sum > manifest.sha256)
}
gen_tunedB() { gen_tuned "$1" "$2" picksB.json; }
TUNED_ENV='ATOM_Q38_SCHED=$(cat $FN/sched.txt) ATOM_Q38_ROCKE_HEAD=1 ATOM_Q38_EXTERN_MISSING=mono ATOM_Q38_EXTERN_STAGES=m3=$FN/k/{kind}/moe/m3_{variant}.ll,m4=$FN/k/{kind}/moe/m4_{variant}.ll,m5=$FN/k/{kind}/moe/m5_{variant}.ll,b3=$FN/k/{kind}/b3,b3pre=$FN/k/{kind}/b3'
arm_tuned() { arm_hyb_tm; ARM_GEN=tuned; ARM_ENV=$TUNED_ENV; }
arm_tunedB() { arm_hyb_tm; ARM_GEN=tunedB; ARM_ENV=$TUNED_ENV; }
# stacks: the picks + N8 (the subset size the N8 arms pick)
arm_tuned_n8_64k() { arm_tuned; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=65536"; ARM_EXPECT='draft vocab subset on: 65536 of'; }
arm_tuned_n8_96k() { arm_tuned; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=98304"; ARM_EXPECT='draft vocab subset on: 98304 of'; }
arm_tunedB_n8_96k() { arm_tunedB; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=98304"; ARM_EXPECT='draft vocab subset on: 98304 of'; }
# picksD: picksC with the delta_ple cells' confirmed "lookback pick + P10 hold drop" (qk-lbD2)
gen_tunedD() { gen_tuned "$1" "$2" picksD.json; }
arm_tunedD() { arm_hyb_tm; ARM_GEN=tunedD; ARM_ENV=$TUNED_ENV; }
arm_tunedD_n8_96k() { arm_tunedD; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=98304"; ARM_EXPECT='draft vocab subset on: 98304 of'; }
# + the subset head's lookback pick (qk-lbD2 head.fused.m1.n98304: korder lane, xpad 64, fsync counter; draft-side only,
# acceptance-gated: the fused argmax gate compares argmax indices, the logits order differs from the mono's)
N8_HEAD_SPEC='{\"korder\":\"lane\",\"ctas\":256,\"xpad\":64,\"waves_per_eu\":1,\"fsync\":\"counter\"}'
arm_tunedD_n8_96k_hd() { arm_tunedD_n8_96k; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB_SPEC=$N8_HEAD_SPEC"; ARM_EXPECT='draft vocab subset on: 98304 of .*n98304_k2560_r4_bf16_fused_kolane'; }
# the final stack of qk-stack3 (K >= 1; at K=0 serve tunedD: N8 / the subset head act only on draft steps)
arm_final() { arm_tunedD_n8_96k_hd; }
# picksE: picksD with the delta_ple cells re-searched at budget 256 (qk-lbE; each beat picksD's pick + P10 drop in
# the same confirmation rounds); qsa.r1 at 256 tied the qk-lbB pick, which stays
gen_tunedE() { gen_tuned "$1" "$2" picksE.json; }
arm_tunedE() { arm_hyb_tm; ARM_GEN=tunedE; ARM_ENV=$TUNED_ENV; }
arm_finalE() { arm_tunedE; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=98304 ATOM_Q38_DRAFT_VOCAB_SPEC=$N8_HEAD_SPEC"; ARM_EXPECT='draft vocab subset on: 98304 of .*n98304_k2560_r4_bf16_fused_kolane'; }
# picksF: picksE + the P3 HC spread (spread.a1=10) on delta.r1, qsa.r1, delta_ple.r1, confirmed in situ on top of
# each pick (qk-lbF); qsa.r4s (spread=2) and delta.r4s (spread.a1=2) lost on top of their picks and stay as before
gen_tunedF() { gen_tuned "$1" "$2" picksF.json; }
arm_tunedF() { arm_hyb_tm; ARM_GEN=tunedF; ARM_ENV=$TUNED_ENV; }
arm_finalF() { arm_tunedF; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=98304 ATOM_Q38_DRAFT_VOCAB_SPEC=$N8_HEAD_SPEC"; ARM_EXPECT='draft vocab subset on: 98304 of .*n98304_k2560_r4_bf16_fused_kolane'; }
# picksG: picksF with delta.r1 and delta_ple.r1 re-searched at 256 with the spread knobs in the pool (qk-lbG: 0.928 /
# 0.932 vs picksF's 0.932 / 0.941 in the same rounds); qsa.r1's re-search tied picksF (0.966 vs 0.967) and stays
gen_tunedG() { gen_tuned "$1" "$2" picksG.json; }
arm_tunedG() { arm_hyb_tm; ARM_GEN=tunedG; ARM_ENV=$TUNED_ENV; }
arm_finalG() { arm_tunedG; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=98304 ATOM_Q38_DRAFT_VOCAB_SPEC=$N8_HEAD_SPEC"; ARM_EXPECT='draft vocab subset on: 98304 of .*n98304_k2560_r4_bf16_fused_kolane'; }
# picksH: picksG + the K=1 / K=2 verify cells r2s / r3s (qk-lbH, qk-lbI; budget 256, delta_ple 128, order restarts 2)
gen_tunedH() { gen_tuned "$1" "$2" picksH.json; }
arm_tunedH() { arm_hyb_tm; ARM_GEN=tunedH; ARM_ENV=$TUNED_ENV; }
arm_finalH() { arm_tunedH; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=98304 ATOM_Q38_DRAFT_VOCAB_SPEC=$N8_HEAD_SPEC"; ARM_EXPECT='draft vocab subset on: 98304 of .*n98304_k2560_r4_bf16_fused_kolane'; }
# + P13 N4a (W5): ATOM_Q38_SCHED kind group `qsa:qdense=3` on every narrow ungrouped QSA build (tolerance class, near
# bitwise; accuracy-gated). The recipe appends the group and puts qdense=3 into the expected qsa build lines.
gen_tunedGqd() { GT_EXTRA="qsa:qdense=3" gen_tuned "$1" "$2" picksG.json; }
gen_tunedHqd() { GT_EXTRA="qsa:qdense=3" gen_tuned "$1" "$2" picksH.json; }
arm_tunedGqd() { arm_hyb_tm; ARM_GEN=tunedGqd; ARM_ENV=$TUNED_ENV; }
arm_finalGqd() { arm_tunedGqd; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=98304 ATOM_Q38_DRAFT_VOCAB_SPEC=$N8_HEAD_SPEC"; ARM_EXPECT='draft vocab subset on: 98304 of .*n98304_k2560_r4_bf16_fused_kolane'; }
arm_tunedHqd() { arm_hyb_tm; ARM_GEN=tunedHqd; ARM_ENV=$TUNED_ENV; }
arm_finalHqd() { arm_tunedHqd; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=98304 ATOM_Q38_DRAFT_VOCAB_SPEC=$N8_HEAD_SPEC"; ARM_EXPECT='draft vocab subset on: 98304 of .*n98304_k2560_r4_bf16_fused_kolane'; }
# picksI: picksH + the grouped / wide K=3 builds (schedule only; qk-lbJ search, qk-lbK confirmation): delta.r8g4,
# qsa.r8g4, delta.r16g4 (qsa.r16g4 NEUTRAL)
gen_tunedIqd() { GT_EXTRA="qsa:qdense=3" gen_tuned "$1" "$2" picksI.json; }
arm_tunedIqd() { arm_hyb_tm; ARM_GEN=tunedIqd; ARM_ENV=$TUNED_ENV; }
arm_finalIqd() { arm_tunedIqd; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=98304 ATOM_Q38_DRAFT_VOCAB_SPEC=$N8_HEAD_SPEC"; ARM_EXPECT='draft vocab subset on: 98304 of .*n98304_k2560_r4_bf16_fused_kolane'; }
