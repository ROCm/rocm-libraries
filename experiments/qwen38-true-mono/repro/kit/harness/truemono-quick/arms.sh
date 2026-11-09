# truemono-quick arm definitions (sourced by quick.sh inside the q38 container).
#
# An arm is a bash function `arm_<name>` that sets:
#   ARM_ATOM    ATOM tree the server runs from (cwd + first on PYTHONPATH): $HB (pinned hybrid) or $TM (ATOM-truemono)
#   ARM_ROCKE   rocKE stage name (under /job/stages) on PYTHONPATH: $RK_BASE (pinned hybrid rocKE) or $RK_TM (copy-rocm-lib)
#   ARM_GEN     functions recipe ("" = none). quick.sh runs `gen_<recipe> <rocke stage> <out dir>` once per
#               (recipe, rocke) and sets $FN to that dir before it evaluates ARM_ENV
#   ARM_ENV     server env, space-separated K=V (no spaces inside values); single-quote it so $FN expands late
#   ARM_HEAD    1 if the rocKE lm_head must be on (server-log check), else 0
#   ARM_EXPECT  optional: a grep -E pattern that must appear in the server log (the lever really engaged)
#   ARM_PYPATH  optional: extra PYTHONPATH entries (after the ATOM tree and rocKE)
# Add an arm: drop a file defining arm_<name> into arms.d/ (sourced below; sync.sh packs it), or point ARM_FILES at
# files in a stage. Arm names: [A-Za-z0-9_]+.

# The hybrid ALL env of truemono-baseline/matrix.sh and quick_tps.sh (the recipe's serving line).
HYB_ENV='ATOM_Q38_SCHED=best ATOM_Q38_ROCKE_HEAD=1 ATOM_Q38_EXTERN_MISSING=mono ATOM_Q38_EXTERN_STAGES=m3=$FN/moe/m3_{variant}.ll,m4=$FN/moe/m4_{variant}.ll,m5=$FN/moe/m5_{variant}.ll,b3=$FN/b3,b3pre=$FN/b3'

# plain FlyDSL mono, pinned tree, no knobs
arm_mono() { ARM_ATOM=$HB; ARM_ROCKE=$RK_BASE; ARM_GEN=""; ARM_ENV=""; ARM_HEAD=0; }
# the bar: the pinned hybrid ALL stack (pinned ATOM tree + pinned rocKE, e2ecomp2 functions)
arm_hybrid() { ARM_ATOM=$HB; ARM_ROCKE=$RK_BASE; ARM_GEN=e2ecomp2; ARM_ENV=$HYB_ENV; ARM_HEAD=1; }
# the candidate base: the same ALL env on ATOM-truemono + copy-rocm-lib's current rocKE (every candidate arm starts here)
arm_hyb_tm() { ARM_ATOM=$TM; ARM_ROCKE=$RK_TM; ARM_GEN=e2ecomp2; ARM_ENV=$HYB_ENV; ARM_HEAD=1; }

# functions recipes: gen_<recipe> <rocke stage> <out dir>
gen_e2ecomp2() { Q38_ROCKE=$1 bash "$H/e2ecomp2/gen.sh" "$2"; }

for f in "$H"/truemono-quick/arms.d/*.sh ${ARM_FILES:-}; do
  # shellcheck disable=SC1090
  [ -f "$f" ] && source "$f"
done
