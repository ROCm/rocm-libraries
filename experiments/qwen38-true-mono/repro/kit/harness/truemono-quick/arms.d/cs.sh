# Consolidated stack (agent "cs", COORD-consolidated.md): the WP4h tail provider over probe P9's round-tail kernel
# (copy-rocm-lib kernels/gfx950/qwen38_mono/truemono_tail.py) on top of the best stacks. ATOM-truemono's
# ATOM_Q38_TRUE_MONO=tail mode with the hybrid's layer launchers (layer_passthrough) and the provider (the hook's
# default module): K=3 one request (r4s) = P9's verify chain in one launch, K=0 one request (r1) = the rocKE fused-
# argmax head; every other width is the hybrid's. Gated by truemono/CS/cs_gate.sh (check mode + deep check + mutants).
CS_TAIL_ENV='ATOM_Q38_TRUE_MONO=tail ATOM_Q38_TRUE_MONO_LAYER=kernels.gfx950.qwen38_mono.truemono_tail:layer_passthrough ATOM_Q38_TRUE_MONO_STATS=100000'
CS_TAIL_REJECT='tm tail r(4s|1) off|draft chain off|differs from the hybrid|deepcheck [a-z0-9.]+ DIFF|Traceback'
_cs_tail() { ARM_ENV="$ARM_ENV $CS_TAIL_ENV${1:+ $1}"; ARM_EXPECT_K3='tm tail r4s: bound'; ARM_EXPECT_K0='tm tail r1: rocKE fused argmax head'; ARM_REJECT=$CS_TAIL_REJECT; }
arm_hyb_tm_tail() { arm_hyb_tm; _cs_tail; }
arm_finalG_tail() { arm_finalG; _cs_tail; }
arm_tunedG_tail() { arm_tunedG; _cs_tail; }
arm_finalGqd_tail() { arm_finalGqd; _cs_tail; }
arm_tunedGqd_tail() { arm_tunedGqd; _cs_tail; }
# + the draft chain (ATOM_Q38_TM_TAIL_DRAFT=p9): every draft step's tail as a P9 draft launch (N8 subset head + map)
arm_finalG_tailD() { arm_finalG; _cs_tail ATOM_Q38_TM_TAIL_DRAFT=p9; ARM_EXPECT='draft vocab subset on: 98304 of'; ARM_EXPECT_K3='tm tail r4s: draft chain bound'; }
arm_finalGqd_tailD() { arm_finalGqd; _cs_tail ATOM_Q38_TM_TAIL_DRAFT=p9; ARM_EXPECT='draft vocab subset on: 98304 of'; ARM_EXPECT_K3='tm tail r4s: draft chain bound'; }
arm_hyb_tm_tailD() { arm_hyb_tm; _cs_tail ATOM_Q38_TM_TAIL_DRAFT=p9; ARM_EXPECT_K3='tm tail r4s: draft chain bound'; }
# P9 lever arms on the tail (P9Spec knob overrides, JSON in ATOM_Q38_TM_TAIL_P9)
arm_finalG_tail_hd0() { arm_finalG; _cs_tail 'ATOM_Q38_TM_TAIL_P9={\"hdelay\":0}'; }
arm_finalG_tail_xbt4() { arm_finalG; _cs_tail 'ATOM_Q38_TM_TAIL_P9={\"xbt\":4}'; }
# CS_GEN_PP=1 (my jobs only): gen_e2ecomp2 with the generating rocKE stage's own platform first on PYTHONPATH. Without
# it gen.sh imports `rocke` from q38_job.sh's first rocke* stage (rocke-tmbase), and copy-rocm-lib's moe_fused.py
# since 2026-10-31 19:10 PDT (lever scopes, IRBuilder._lever_scope) fails there (job cs-gate1 68236631).
if [ "${CS_GEN_PP:-0}" = 1 ]; then
  gen_e2ecomp2() { PYTHONPATH=/job/stages/$1/platform/python:/job/stages/$1/library${PYTHONPATH:+:$PYTHONPATH} Q38_ROCKE=$1 bash "$H/e2ecomp2/gen.sh" "$2"; }
fi
# ---- the consolidated full stack (picksH + qdense + N8 96K + subset-head pick + the tail with the draft chain) and
# PLAN decision 15's re-tests ON IT (each "rejected on base X" lever, X = the full stack)
_cs_full3() { # _cs_full3 <gen recipe> [N8 rows] [extra env]: K=3 arm on the full stack
  arm_hyb_tm; ARM_GEN=$1
  ARM_ENV="$TUNED_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=${2:-98304} ATOM_Q38_DRAFT_VOCAB_SPEC=$N8_HEAD_SPEC $CS_TAIL_ENV${3:+ $3}"
  ARM_EXPECT="draft vocab subset on: ${2:-98304} of"; ARM_EXPECT_K3='tm tail r4s: bound'; ARM_REJECT=$CS_TAIL_REJECT
  case "${3:-}" in *ATOM_Q38_TM_TAIL_DRAFT=p9* | *ATOM_Q38_TM_TAIL_DRAFT=lean*) ARM_EXPECT_K3='tm tail r4s: draft chain bound' ;; esac
}
_cs_full0() { # _cs_full0 <gen recipe> [extra env]: K=0 arm on the full stack
  arm_hyb_tm; ARM_GEN=$1; ARM_ENV="$TUNED_ENV $CS_TAIL_ENV${2:+ $2}"; ARM_EXPECT_K0='tm tail r1: rocKE fused argmax head'; ARM_REJECT=$CS_TAIL_REJECT
}
gen_csHqd_p10() { GT_EXTRA="qsa:qdense=3" gen_tuned "$1" "$2" picksCS_p10r4s.json; }
gen_csHqd_spr4s() { GT_EXTRA="qsa:qdense=3" gen_tuned "$1" "$2" picksCS_spr4s.json; }
gen_csHqd_n3b() { GT_EXTRA="qsa:qdense=3" gen_tuned "$1" "$2" picksCS_n3b.json; }
gen_csH_n3b() { gen_tuned "$1" "$2" picksCS_n3b.json; }
# 2026-11-01 (job qk-cs-stk2-n01): the draft chain lost on base finalHqd + verify tail (round 1.0027, 0/4), so the
# full stack is the VERIFY tail only; the chain stays a default-off knob (arm full3_d)
arm_full3() { _cs_full3 tunedHqd; }                          # the full stack, K=3 (= full3_v)
arm_full3_d() { _cs_full3 tunedHqd 98304 ATOM_Q38_TM_TAIL_DRAFT=p9; }
arm_full0() { _cs_full0 tunedHqd; }                          # the full stack, K=0
arm_full3_noqd() { _cs_full3 tunedH; }                       # full stack without qdense (if qdense is not gated)
arm_full0_noqd() { _cs_full0 tunedH; }
# decision-15 re-tests on base full3 / full0
arm_full3_n8_64k() { _cs_full3 tunedHqd 65536; }
arm_full3_n8_32k() { _cs_full3 tunedHqd 32768; }
arm_full3_n8_16k() { _cs_full3 tunedHqd 16384; }
arm_full3_p10() { _cs_full3 csHqd_p10; }                     # P10 r4s hold drops
arm_full3_spr4s() { _cs_full3 csHqd_spr4s; }                 # r4s spread variants
arm_full3_n3b() { _cs_full3 csHqd_n3b; }                     # P15 N3b (delta.r4s)
arm_full0_n3b() { _cs_full0 csHqd_n3b; }                     # P15 N3b (delta.r1)
# the verify head is P9's at CONC 1 K=3 (the hybrid's verify-head picks no longer run there): its own head levers
arm_full3_hd0() { _cs_full3 tunedHqd 98304 'ATOM_Q38_TM_TAIL_P9={\"hdelay\":0}'; }
arm_full3_hd16() { _cs_full3 tunedHqd 98304 'ATOM_Q38_TM_TAIL_P9={\"hdelay\":16}'; }
arm_full3_dg3() { _cs_full3 tunedHqd 98304 'ATOM_Q38_TM_TAIL_P9={\"head_dg\":3}'; }
arm_full3_dg1() { _cs_full3 tunedHqd 98304 'ATOM_Q38_TM_TAIL_P9={\"head_dg\":1}'; }
arm_full3_xbt4() { _cs_full3 tunedHqd 98304 'ATOM_Q38_TM_TAIL_P9={\"xbt\":4}'; }
arm_full3_d_hd8() { _cs_full3 tunedHqd 98304 'ATOM_Q38_TM_TAIL_DRAFT=p9 ATOM_Q38_TM_TAIL_P9D={\"hdelay\":8}'; }
arm_full3_d_n8_64k() { _cs_full3 tunedHqd 65536 ATOM_Q38_TM_TAIL_DRAFT=p9; }
arm_full3_d_n8_32k() { _cs_full3 tunedHqd 32768 ATOM_Q38_TM_TAIL_DRAFT=p9; }
# the K=0 tail's argmax head with the verify-head pick levels that are bitwise (cache policy / issue schedule)
arm_full0_hsq() { _cs_full0 tunedHqd 'ATOM_Q38_ROCKE_HEAD_SPEC_ARGMAX={\"sched\":\"q\"}'; }
arm_full0_hsc0() { _cs_full0 tunedHqd 'ATOM_Q38_ROCKE_HEAD_SPEC_ARGMAX={\"cpol\":\"sc0\"}'; }
arm_full3_v() { arm_finalHqd; _cs_tail; }                    # finalHqd + the verify tail only (ATOM's drafter)
# short combination lookback with the tail in place (job qk-cs-rt3-n01: no single re-test won; the near-neutral ones
# combined): verify-launch hdelay 0 with N8 64K / head_dg 3
arm_full3_hd0_n8_64k() { _cs_full3 tunedHqd 65536 'ATOM_Q38_TM_TAIL_P9={\"hdelay\":0}'; }
arm_full3_hd0_dg3() { _cs_full3 tunedHqd 98304 'ATOM_Q38_TM_TAIL_P9={\"hdelay\":0,\"head_dg\":3}'; }
# the lean draft round (ATOM_Q38_TM_TAIL_DRAFT=lean): the draft steps as GPU launches only, no host-side ATOM drafter
# (the round is host-bound around the tail: cs_round / cs_gaps traces of job cs-trace1)
arm_full3_l() { _cs_full3 tunedHqd 98304 ATOM_Q38_TM_TAIL_DRAFT=lean; }
# diagnostics: a host busy-wait after the verify launch (does host time show up in the round?)
arm_full3_h100() { _cs_full3 tunedHqd 98304 ATOM_Q38_TM_TAIL_HOSTUS=100; }
arm_full3_h300() { _cs_full3 tunedHqd 98304 ATOM_Q38_TM_TAIL_HOSTUS=300; }
arm_full3_l_h300() { _cs_full3 tunedHqd 98304 'ATOM_Q38_TM_TAIL_DRAFT=lean ATOM_Q38_TM_TAIL_HOSTUS=300'; }
