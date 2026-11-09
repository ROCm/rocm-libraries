# WP2b arms for truemono-quick (ARM_FILES=$H/truemono/WP2b/arms_wp2b.sh): the consolidated full stack (cs.sh full3 /
# full0, picksH + qdense + the tail) with the rocKE GEMV-family functions (B1 / C1 / Q1 / P1,
# qwen38_mono/stages2/gemv.py) linked over the FlyDSL stages. Functions: wp2b_gen.py over the stack's own functions
# (its numa lever, its pause, B1 hosting the B3 prologue where the stack's B3 has pre=dma). Knobs: WP2B_KNOBS (one set
# for every stage) or WP2B_KNOBS_JSON (per cell / stage, the tuner's picks; a file under the h stage).
WP2B_VARIANTS=${WP2B_VARIANTS:-r1,r2,r3,r4,r2s,r3s,r4s,r4g2,r6g2,r8g2,r6g3,r9g3,r12g3,r8g4,r12g4,r16g4}
gen_wp2bHqd() { # gen_wp2bHqd <rocke stage> <out dir>
  gen_tunedHqd "$1" "$2" || return 1
  (cd /job && PYTHONPATH=/atom:/job/stages/$1/platform/python:/job/stages/$1/library ROCKE_BACKEND=python \
    ROCKE_LLVM_FLAVOR=llvm23 python "$H/truemono/WP2b/wp2b_gen.py" --stk "$2" --out "$2" --variants "$WP2B_VARIANTS" \
    ${WP2B_KNOBS:+--knobs "$WP2B_KNOBS"} ${WP2B_KNOBS_JSON:+--knobs-json "$H/truemono/WP2b/$WP2B_KNOBS_JSON"} ${WP2B_JSON_ONLY:+--json-only}) || return 1
  (cd "$2" && find k -type f | sort | xargs sha256sum > manifest.sha256)
  # server-log checks (quick.sh <FN>/expect_k<K>.txt): every served build links its rocKE GEMV functions
  local k v kind st
  for k in 0 3; do
    v=r1; [ $k = 3 ] && v=r4s
    for kind in delta delta_ple qsa; do
      for st in b1 c1 q1 p1; do
        [ -f "$2/k/$kind/gemv/${st}_${kind}_$v.ll" ] || continue
        echo "[ATOM_Q38_EXTERN_STAGES] $st $v: rocke_q38_${st}_$v from $2/k/$kind/gemv/${st}_${kind}_$v.ll" >> "$2/expect_k$k.txt"
      done
    done
  done
}
# the extern entries appended to the stack's ATOM_Q38_EXTERN_STAGES (after its b3pre entry)
WP2B_X=',b1=$FN/k/{kind}/gemv/b1_{kind}_{variant}.ll,c1=$FN/k/{kind}/gemv/c1_{kind}_{variant}.ll,q1=$FN/k/{kind}/gemv/q1_{kind}_{variant}.ll,p1=$FN/k/{kind}/gemv/p1_{kind}_{variant}.ll'
_wp2b_x() { ARM_ENV=$(printf '%s' "$ARM_ENV" | sed 's#b3pre=$FN/k/{kind}/b3#&'"$WP2B_X"'#'); }
arm_full3_wp2b() { _cs_full3 wp2bHqd; _wp2b_x; }
arm_full0_wp2b() { _cs_full0 wp2bHqd; _wp2b_x; }
