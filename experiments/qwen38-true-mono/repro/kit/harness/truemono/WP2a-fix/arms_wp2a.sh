# WP2a fix-round arms for truemono-quick (ARM_FILES=$H/truemono/WP2a-fix/arms_wp2a.sh): the consolidated full stack
# (cs.sh full3 / full0: picksH + qdense + the verify tail) with the rocKE HC functions (A1 / A2 / M1 / M2,
# qwen38_mono/stages2/hc.py) of the WP2a picks linked over the FlyDSL HC stages, the picks' HC schedule levers in the
# cells' groups (WP2a-fix/wp2a_gen.py). Optional combination with the WP2b GEMV functions (arms_wp2b.sh, sourced
# when present: gen_wp2bHqd).
WP2A_PICKS=${WP2A_PICKS:-picks_final.json}
_wp2a_gen() { # _wp2a_gen <rocke stage> <out dir> [--port]
  (cd /job && PYTHONPATH=/atom:/job/stages/$1/platform/python:/job/stages/$1/library ROCKE_BACKEND=python \
    ROCKE_LLVM_FLAVOR=llvm23 python "$H/truemono/WP2a-fix/wp2a_gen.py" --stk "$2" \
    --picks "$H/truemono/WP2a-fix/$WP2A_PICKS" ${WP2A_CELLS:+--cells "$WP2A_CELLS"} ${3:-}) || return 1
  (cd "$2" && find k hc -type f | sort | xargs sha256sum > manifest.sha256)
}
gen_wp2aHqd() { gen_tunedHqd "$1" "$2" || return 1; _wp2a_gen "$1" "$2"; }
gen_wp2aPortHqd() { gen_tunedHqd "$1" "$2" || return 1; _wp2a_gen "$1" "$2" --port; }
gen_wp2abHqd() { gen_wp2bHqd "$1" "$2" || return 1; _wp2a_gen "$1" "$2"; }
# the extern entries appended to the stack's ATOM_Q38_EXTERN_STAGES (after its b3pre entry)
WP2A_X=',a1=$FN/hc/{kind},a2=$FN/hc/{kind},m1=$FN/hc/{kind},m2=$FN/hc/{kind}'
_wp2a_x() { ARM_ENV=$(printf '%s' "$ARM_ENV" | sed 's#b3pre=$FN/k/{kind}/b3#&'"$WP2A_X"'#'); }
arm_full3_wp2a() { _cs_full3 wp2aHqd; _wp2a_x; }
arm_full0_wp2a() { _cs_full0 wp2aHqd; _wp2a_x; }
arm_full3_wp2aport() { _cs_full3 wp2aPortHqd; _wp2a_x; }
arm_full0_wp2aport() { _cs_full0 wp2aPortHqd; _wp2a_x; }
arm_full3_wp2ab() { _cs_full3 wp2abHqd; _wp2b_x; _wp2a_x; }
arm_full0_wp2ab() { _cs_full0 wp2abHqd; _wp2b_x; _wp2a_x; }
