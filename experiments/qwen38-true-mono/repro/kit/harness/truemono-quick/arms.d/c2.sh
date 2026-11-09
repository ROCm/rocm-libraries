# CONSOLIDATED PUSH 2 (agent "c2", 2026-11-02; copy-rocm-lib experiments/qwen38-true-mono/quick/README.md "Consolidated
# stack 2"). New file of the c2 lane; nothing else of this harness is changed. Every verified deployable win stacked in
# one served arm: the round-boundary base (lean draft round + ATOM_Q38_RB_DEVMETA with TRIM + ADV, RB/arms_rb.sh) +
# WP2c's served QSA set (rocKE Q5-Q7 everywhere, rocKE Q3 / Q4 except r4s, prepoll + q5_kpre, qsa.r1 hold.q7=64:
# arms.d/w8c.sh / w8c2.sh, fix-round e3b / e0 picks) + WP2a's HC picks (WP2a-fix/arms_wp2a.sh, picks_final.json) +
# P16's fold.b1 on delta.r1 (K=0) [+ WP2b's GEMV functions (WP2b/arms_wp2b.sh, picks_all.json) in the *b arms].
# Needs ARM_FILES with truemono/RB/arms_rb.sh, truemono/WP2a-fix/arms_wp2a.sh and truemono/WP2b/arms_wp2b.sh (their
# functions are called lazily, at arm resolution). Every lever is default-off: unset, the server is the hybrid.
#
# Recipes (gen_<recipe> <rocke stage> <out dir>): the stack's tunedHqd (picksH + qsa:qdense=3), then optional WP2b,
# then optional WP2a (wp2a_gen.py: HC functions + the picks' HC schedule groups), then the WP2c functions + schedule
# overrides (gen_w8c57 with its own tunedHqd call stubbed out). <out>/sched.ref.txt = the stack's schedule before any
# of them (the reference of truemono/C2/c2_layer_gate.py).
#   c2P   qsa2 only (= full0_w8c / full3_w8c57 of the WP2c fix round)
#   c2F   qsa2 + fold.b1
#   c2A   qsa2 + WP2a + fold.b1                  (the stack)
#   c2Ab  qsa2 + WP2a + WP2b (no fold: WP2b's rocKE B1 replaces the FlyDSL B1 that fold.b1 changes)
#   c2Q   WP2a only (no qsa2, no fold), attribution
C2_W8C_KNOBS=${C2_W8C_KNOBS:-numa=1,qdense=3,prepoll=q5+q6+q7,q5_kpre=True}
C2_SCHED_HOLD='qsa.r1:hold.q7=64'
C2_SCHED_FOLD='delta.r1:fold.b1=1'
_c2_gen() { # _c2_gen <rocke> <out> <wp2b 0|1> <wp2a 0|1|port> <qsa2 0|1|5> <sched overrides "cell:k=v;..."> [picks]
  # qsa2 5 = all five rocKE QSA functions at r4s too (gen_w8c; serve with _w8c_env); [picks] default picksH.json
  local rk=$1 out=$2 b=$3 a=$4 q=$5 so=$6 pk=${7:-picksH.json}
  if [ "$b" = 1 ]; then gen_wp2bHqd "$rk" "$out" || return 1; else GT_EXTRA="qsa:qdense=3" gen_tuned "$rk" "$out" "$pk" || return 1; fi
  cp "$out/sched.txt" "$out/sched.ref.txt"
  if [ "$a" = 1 ]; then _wp2a_gen "$rk" "$out" || return 1; fi
  if [ "$a" = port ]; then _wp2a_gen "$rk" "$out" --port || return 1; fi
  if [ "$q" = 1 ] || [ "$q" = 5 ]; then
    (
      gen_tunedHqd() { :; }  # gen_w8c's own base call: the stack is already in <out>
      W8C_KNOBS=$C2_W8C_KNOBS W8C_VKNOBS="" W8C_SCHED=$so
      export W8C_KNOBS W8C_VKNOBS W8C_SCHED
      if [ "$q" = 5 ]; then gen_w8c "$rk" "$out"; else gen_w8c57 "$rk" "$out"; fi
    ) || return 1
  elif [ -n "$so" ]; then
    echo "c2: schedule overrides without qsa2 are not supported" >&2; return 1
  fi
  cp "$out/sched.txt" "$out/sched.c2.txt"
  (cd "$out" && find k $(ls -d hc qsa2 qsa2_34 2> /dev/null) -type f | sort | xargs sha256sum > manifest.sha256)
}
gen_c2P() { _c2_gen "$1" "$2" 0 0 1 "$C2_SCHED_HOLD"; }
gen_c2F() { _c2_gen "$1" "$2" 0 0 1 "$C2_SCHED_HOLD;$C2_SCHED_FOLD"; }
gen_c2A() { _c2_gen "$1" "$2" 0 1 1 "$C2_SCHED_HOLD;$C2_SCHED_FOLD"; }
gen_c2Ab() { _c2_gen "$1" "$2" 1 1 1 "$C2_SCHED_HOLD"; }
gen_c2Q() { _c2_gen "$1" "$2" 0 1 0 ""; }
_c2_x() { # _c2_x <qsa2 0|1|5> <wp2a 0|1> <wp2b 0|1>: the externs of the recipe on top of the stack's
  [ "$1" = 1 ] && _w8c57_env
  [ "$1" = 5 ] && _w8c_env
  [ "$2" = 1 ] && _wp2a_x
  [ "$3" = 1 ] && _wp2b_x
  return 0
}
C2_RBA='ATOM_Q38_RB_DEVMETA_TRIM=1 ATOM_Q38_RB_DEVMETA_ADV=1'
_c2_3() { _cs_full3 "$1" 98304 ATOM_Q38_TM_TAIL_DRAFT=lean; _c2_x "$2" "$3" "$4"; _rb_on "$C2_RBA"; }  # K=3: lean + rba
_c2_0() { _cs_full0 "$1"; _c2_x "$2" "$3" "$4"; }                                                       # K=0: the K=0 tail
# K=3 (CONC 1; the base full3_l_rba = previous best, arms_rb.sh)
arm_c2w3() { _c2_3 c2P 1 0 0; }       # full3_l_rba + WP2c (attribution)
arm_c2s3() { _c2_3 c2A 1 1 0; }       # THE STACK at K=3 (fold.b1 is a delta.r1 lever: not built at K=3)
arm_c2s3_b() { arm_c2s3; }            # A-vs-A twin
arm_c2s3b() { _c2_3 c2Ab 1 1 1; }     # + WP2b
arm_c2q3() { _c2_3 c2Q 0 1 0; }       # full3_l_rba + WP2a only (attribution)
arm_full3_l_rba_b() { arm_full3_l_rba; }  # A-vs-A twin of the previous best
# K=0 (CONC 1; the K=0 tail; previous best full0_w8c = c2p0 with the fix round's knobs)
arm_c2p0() { _c2_0 c2P 1 0 0; }       # = full0_w8c (fix round e0): previous best K=0
arm_c2p0_b() { arm_c2p0; }            # A-vs-A twin
arm_c2f0() { _c2_0 c2F 1 0 0; }       # + fold.b1 (attribution)
arm_c2s0() { _c2_0 c2A 1 1 0; }       # THE STACK at K=0: + WP2a + fold.b1
arm_c2s0_b() { arm_c2s0; }
arm_c2s0b() { _c2_0 c2Ab 1 1 1; }     # + WP2a + WP2b (no fold)
# accuracy / T2 arms: the K=0 layers without the tail (ATOM_Q38_TF needs the plain sampler, W9 N4a-acc)
arm_c2t0() { arm_hyb_tm; ARM_GEN=c2A; ARM_ENV=$TUNED_ENV; _c2_x 1 1 0; }
arm_c2t0ref() { arm_tunedHqd; }       # the previous full stack's K=0 layers, no tail (T2 reference)
arm_c2t0p4() { arm_c2t0; ARM_ENV="$ARM_ENV ATOM_Q38_PERTURB=all:2^-4"; ARM_EXPECT='q38 perturb on: scope=all scale=2\^-4 '; }
# ---- PLAN decision-15 re-tests ON the stack (base c2s3 / c2s0: every earlier rejection was "on base X")
gen_c2Anf() { _c2_gen "$1" "$2" 0 1 1 "$C2_SCHED_HOLD"; }                                      # stack without fold.b1
gen_c2A_b1h() { _c2_gen "$1" "$2" 0 1 1 "$C2_SCHED_HOLD;delta.r1:fold.b1=1,hold.b1=32"; }      # P16 b1h (re-tuned hold)
gen_c2A_c1() { _c2_gen "$1" "$2" 0 1 1 "$C2_SCHED_HOLD;delta.r1:fold.b1=1,fold.c1=1"; }        # P16 fold.c1 (lost on picksH)
gen_c2A_n3b() { _c2_gen "$1" "$2" 0 1 1 "$C2_SCHED_HOLD;$C2_SCHED_FOLD" picksCS_n3b.json; }   # P15 N3b (lost on full3 / full0)
gen_c2A5() { _c2_gen "$1" "$2" 0 1 5 "$C2_SCHED_HOLD;$C2_SCHED_FOLD"; }                        # WP2c all five at r4s (pick57 better in situ)
gen_c2Aport() { _c2_gen "$1" "$2" 0 port 1 "$C2_SCHED_HOLD;$C2_SCHED_FOLD"; }                  # WP2a port defaults, not the picks
_c2_3x() { _cs_full3 "$1" "${3:-98304}" "ATOM_Q38_TM_TAIL_DRAFT=${4:-lean}${5:+ $5}"; _c2_x ${2}; _rb_on "$C2_RBA"; }
arm_c2s3_n8_64k() { _c2_3x c2A "1 1 0" 65536; }
arm_c2s3_hd8() { _c2_3x c2A "1 1 0" 98304 lean 'ATOM_Q38_TM_TAIL_P9={\"hdelay\":8}'; }
arm_c2s3_dg3() { _c2_3x c2A "1 1 0" 98304 lean 'ATOM_Q38_TM_TAIL_P9={\"head_dg\":3}'; }
arm_c2s3_p9d() { _c2_3x c2A "1 1 0" 98304 p9; }                                    # P9 draft chain instead of lean
arm_c2s3_v() { _cs_full3 c2A; _c2_x 1 1 0; _rb_on "$C2_RBA"; }                    # verify tail only (ATOM's drafter)
arm_c2s3_w5() { _c2_3x c2A5 "5 1 0"; }
arm_c2s3_port() { _c2_3x c2Aport "1 1 0"; }
arm_c2s3_n3b() { _c2_3x c2A_n3b "1 1 0"; }
arm_c2s3_trim() { _cs_full3 c2A 98304 ATOM_Q38_TM_TAIL_DRAFT=lean; _c2_x 1 1 0; _rb_on ATOM_Q38_RB_DEVMETA_TRIM=1; }  # rbt (no ADV)
arm_c2a0() { _c2_0 c2Anf 1 1 0; }                                                 # stack without fold.b1 (fold on this base)
arm_c2s0_b1h() { _c2_0 c2A_b1h 1 1 0; }
arm_c2s0_c1() { _c2_0 c2A_c1 1 1 0; }
arm_c2s0_n3b() { _c2_0 c2A_n3b 1 1 0; }
arm_c2s0_port() { _c2_0 c2Aport 1 1 0; }
arm_c2s0_hsq() { _cs_full0 c2A 'ATOM_Q38_ROCKE_HEAD_SPEC_ARGMAX={\"sched\":\"q\"}'; _c2_x 1 1 0; }
arm_c2s0_hsc0() { _cs_full0 c2A 'ATOM_Q38_ROCKE_HEAD_SPEC_ARGMAX={\"cpol\":\"sc0\"}'; _c2_x 1 1 0; }
# T2 (K=0 teacher forcing, plain sampler: no tail) of the WP2b variant of the stack, and its planted must-fail
arm_c2t0b() { arm_hyb_tm; ARM_GEN=c2Ab; ARM_ENV=$TUNED_ENV; _c2_x 1 1 1; }
arm_c2t0bp4() { arm_c2t0b; ARM_ENV="$ARM_ENV ATOM_Q38_PERTURB=all:2^-4"; ARM_EXPECT='q38 perturb on: scope=all scale=2\^-4 '; }
# K=0 re-tests on base c2s0b (the WP2b variant: fold.b1 is not built there, WP2b's rocKE B1 replaces FlyDSL's)
gen_c2Ab_c1() { _c2_gen "$1" "$2" 1 1 1 "$C2_SCHED_HOLD;delta.r1:fold.c1=1"; }
gen_c2Ab5() { _c2_gen "$1" "$2" 1 1 5 "$C2_SCHED_HOLD"; }
gen_c2Abport() { _c2_gen "$1" "$2" 1 port 1 "$C2_SCHED_HOLD"; }
arm_c2s0b_b() { arm_c2s0b; }
arm_c2s3b_b() { arm_c2s3b; }
arm_c2s0b_c1() { _c2_0 c2Ab_c1 1 1 1; }
arm_c2s0b_port() { _c2_0 c2Abport 1 1 1; }
arm_c2s0b_hsq() { _cs_full0 c2Ab 'ATOM_Q38_ROCKE_HEAD_SPEC_ARGMAX={\"sched\":\"q\"}'; _c2_x 1 1 1; }
arm_c2s0b_hsc0() { _cs_full0 c2Ab 'ATOM_Q38_ROCKE_HEAD_SPEC_ARGMAX={\"cpol\":\"sc0\"}'; _c2_x 1 1 1; }
arm_c2s3b_n8_64k() { _c2_3x c2Ab "1 1 1" 65536; }
arm_c2s3b_hd8() { _c2_3x c2Ab "1 1 1" 98304 lean 'ATOM_Q38_TM_TAIL_P9={\"hdelay\":8}'; }
arm_c2s3b_dg3() { _c2_3x c2Ab "1 1 1" 98304 lean 'ATOM_Q38_TM_TAIL_P9={\"head_dg\":3}'; }
arm_c2s3b_p9d() { _c2_3x c2Ab "1 1 1" 98304 p9; }
arm_c2s3b_v() { _cs_full3 c2Ab; _c2_x 1 1 1; _rb_on "$C2_RBA"; }
arm_c2s3b_w5() { _c2_3x c2Ab5 "5 1 1"; }
arm_c2s3b_port() { _c2_3x c2Abport "1 1 1"; }
arm_c2s3b_trim() { _cs_full3 c2Ab 98304 ATOM_Q38_TM_TAIL_DRAFT=lean; _c2_x 1 1 1; _rb_on ATOM_Q38_RB_DEVMETA_TRIM=1; }
# WP2b everywhere except delta.r1's B1 (FlyDSL B1 there, so fold.b1 applies): the two K=0 levers that touch delta.r1 B1,
# combined (decision 13: both kept). The unused function is moved aside inside the job's functions dir, never deleted.
gen_c2AbF() {
  _c2_gen "$1" "$2" 1 1 1 "$C2_SCHED_HOLD;$C2_SCHED_FOLD" || return 1
  mkdir -p "$2/unused" && mv "$2"/k/delta/gemv/b1_delta_r1.* "$2/unused/" || return 1
  grep -v -F "b1 r1: rocke_q38_b1_r1 from $2/k/delta/gemv/b1_delta_r1.ll" "$2/expect_k0.txt" > "$2/expect_k0.tmp" && mv "$2/expect_k0.tmp" "$2/expect_k0.txt"
  (cd "$2" && find k hc qsa2 qsa2_34 -type f | sort | xargs sha256sum > manifest.sha256)
}
arm_c2s0bf() { _c2_0 c2AbF 1 1 1; }
arm_c2s3bf() { _c2_3 c2AbF 1 1 1; }   # the same recipe at K=3 (delta.r1 is not built there: equal to c2s3b by construction)
# ---- last mile (2026-11-02 afternoon): the schedule re-climbed with the stacked functions in place
# (truemono/C2/c2_sched_tune.py picks; C2T_SCHED = their gen_w8c-style overrides "cell:k=v|off,...;cell:...", passed
# per job) on top of c2Ab; c2u3 / c2u0 the K=3 / K=0 arms, _b twins, _n8 = N8 64K draft vocab.
gen_c2T() { _c2_gen "$1" "$2" 1 1 1 "$C2_SCHED_HOLD${C2T_SCHED:+;$C2T_SCHED}"; }
arm_c2u3() { _c2_3 c2T 1 1 1; }
arm_c2u3_b() { arm_c2u3; }
arm_c2u3_n8() { _c2_3x c2T "1 1 1" 65536; }
arm_c2u0() { _c2_0 c2T 1 1 1; }
arm_c2u0_b() { arm_c2u0; }
arm_c2s3b_n8() { _c2_3x c2Ab "1 1 1" 65536; }
# decision-15 re-tests ON the final base c2u3 / c2u0 (c2T + C2T_SCHED)
gen_c2T5() { _c2_gen "$1" "$2" 1 1 5 "$C2_SCHED_HOLD${C2T_SCHED:+;$C2T_SCHED}"; }
gen_c2Tport() { _c2_gen "$1" "$2" 1 port 1 "$C2_SCHED_HOLD${C2T_SCHED:+;$C2T_SCHED}"; }
arm_c2u3_hd8() { _c2_3x c2T "1 1 1" 98304 lean 'ATOM_Q38_TM_TAIL_P9={\"hdelay\":8}'; }
arm_c2u3_hd16() { _c2_3x c2T "1 1 1" 98304 lean 'ATOM_Q38_TM_TAIL_P9={\"hdelay\":16}'; }
arm_c2u3_dg3() { _c2_3x c2T "1 1 1" 98304 lean 'ATOM_Q38_TM_TAIL_P9={\"head_dg\":3}'; }
arm_c2u3_p9d() { _c2_3x c2T "1 1 1" 98304 p9; }
arm_c2u3_v() { _cs_full3 c2T; _c2_x 1 1 1; _rb_on "$C2_RBA"; }
arm_c2u3_trim() { _cs_full3 c2T 98304 ATOM_Q38_TM_TAIL_DRAFT=lean; _c2_x 1 1 1; _rb_on ATOM_Q38_RB_DEVMETA_TRIM=1; }
arm_c2u3_w5() { _c2_3x c2T5 "5 1 1"; }
arm_c2u3_port() { _c2_3x c2Tport "1 1 1"; }
arm_c2u0_hsq() { _cs_full0 c2T 'ATOM_Q38_ROCKE_HEAD_SPEC_ARGMAX={\"sched\":\"q\"}'; _c2_x 1 1 1; }
arm_c2u0_hsc0() { _cs_full0 c2T 'ATOM_Q38_ROCKE_HEAD_SPEC_ARGMAX={\"cpol\":\"sc0\"}'; _c2_x 1 1 1; }
