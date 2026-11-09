# W8 WP2c fix round (copy-rocm-lib experiments/qwen38-true-mono/wave2/WP2c.md "Fix round"): arms for the DESIGN 9.1
# E2E win rule and the r4s Q3 / Q4 placement follow-up. New file of the W8 lane; w8c.sh and every other file of this
# harness are unchanged.
#   hybrid_b / full3_b / full0_b   A-vs-A twins (the same config as hybrid / full3 / full0 under another arm name:
#                                  one more server per pass, so a pair of identical arms measures the per-cell noise)
#   full3_w8c57                    full3_w8c with FlyDSL's Q3 / Q4 at r4s (the rocKE Q3 / Q4 at every other narrow
#                                  width, the rocKE Q5-Q7 everywhere): ATOM_Q38_EXTERN_MISSING=mono keeps the FlyDSL
#                                  stage for a variant whose file is absent; the expect file asserts it
arm_hybrid_b() { arm_hybrid; }
arm_full3_b() { arm_full3; }
arm_full0_b() { arm_full0; }
gen_w8c57() { # gen_w8c57 <rocke stage> <out dir>: gen_w8c, then qsa2_34 = qsa2's Q3 / Q4 files without r4s
  gen_w8c "$1" "$2" || return 1
  mkdir -p "$2/qsa2_34"
  local f
  for f in "$2"/qsa2/q3_* "$2"/qsa2/q4_*; do
    case $f in *_r4s.*) ;; *) cp "$f" "$2/qsa2_34/" ;; esac
  done
  grep -v -E '^q[34] r4s: ' "$2/expect_k3.txt" > "$2/expect_k3.tmp" && mv "$2/expect_k3.tmp" "$2/expect_k3.txt"
  echo "no q3 function for build variant r4s" >> "$2/expect_k3.txt"
  echo "no q4 function for build variant r4s" >> "$2/expect_k3.txt"
  (cd "$2" && find k qsa2 qsa2_34 -type f | sort | xargs sha256sum > manifest.sha256)
}
_w8c57_env() {
  ARM_ENV=$(printf '%s' "$ARM_ENV" | sed 's#b3pre=\$FN/k/{kind}/b3#&,q3=$FN/qsa2_34,q4=$FN/qsa2_34,q5=$FN/qsa2,q6=$FN/qsa2,q7=$FN/qsa2#')
}
arm_full3_w8c57() { _cs_full3 w8c57; _w8c57_env; }
