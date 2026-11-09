# Generation recipes of the stacks C, D, E and F (see REPRO.md). Source this file AFTER the frozen harness's arm files
# (`$ARM_FILES`, so that `_c2_gen`, `C2_SCHED_HOLD` and `C2_W8C_KNOBS` exist), with `$STACK_TOOLS` pointing at ./tools.
#
#   gen_<recipe> <rocKE stage dir name> <output functions dir>
#
# What each knob string means:
#   HC knobs (every HC stage of the chosen kinds): invmath=rsq32 (f32 rsq inverse RMS), sigm=hw (hardware exp + rcp sigmoid)
#   QSA2 knobs (global to the Q3-Q7 stage functions): sigm=hw, q5_mm=mfma, q7_mm=mfma, q7_vskip=True
#   MoE router (picks path, `moe.router_mma` in the cell's canonical): router_mma=1 on the named cells
#   Schedule string (ATOM_Q38_SCHED side, not rocKE code): the qsa.r4s levers of $STACK_D_SCHED
#
# Besides generating, each recipe checks that the knobs reached the sidecars and appends the served-function hash
# lines the server log must show (`expect_k3.txt`).

STACK_TOOLS=${STACK_TOOLS:?set STACK_TOOLS to the tools/ dir next to this file}
STACK_D_SCHED=${STACK_D_SCHED:-qsa.r4s:hold.q5=off,hold.q2=96,dyn.q7=off}   # hold.q5 removed, hold.q2=96, dyn.q7 removed

# The functions the K=3 server links per changed family
STACK_K3_HC_VARIANTS="delta:r4s delta_ple:r4s qsa:r1,r4s"
STACK_K3_Q7_VARIANTS="r1 r4 r4s"
_fm_expect() { # _fm_expect <out> <stage> <variant> <file relative to out>
  echo "[ATOM_Q38_EXTERN_STAGES] $2 $3: rocke_q38_$2_$3 from $1/$4 ($(sha256sum "$1/$4" | cut -c1-16))" >> "$1/expect_k3.txt"
}

# _fm_gen <rocke stage> <out dir> <hold.m2 value> <HC knobs ("" = none)> <qsa2 extra knobs ("" = none)> [HC kinds ("" = every cell)] [MoE "cell:knob=v,...;..." ("" = none)] [extra sched overrides ("" = none)]
_fm_gen() {
  local rk=$1 out=$2 hold=$3 hc=$4 q7=$5 kinds=${6:-} moe=${7:-} so="$C2_SCHED_HOLD"
  [ "$hold" = 64 ] || so="$so;delta.r4s:hold.m2=$hold"
  [ -z "${8:-}" ] || so="$so;$8"
  (
    if [ -n "$hc" ]; then
      python "$STACK_TOOLS/mk_picks_fm.py" "$H/truemono/WP2a-fix/picks_final.json" /job/picks_fm.json "$hc" "$kinds" || exit 1
      export WP2A_PICKS_ABS=/job/picks_fm.json WP2A_GEN_WRAP=$STACK_TOOLS/wp2a_gen_fm.py WP2A_GEN_PY=$H/truemono/WP2a-fix/wp2a_gen.py
    fi
    if [ -n "$moe" ]; then
      mkdir -p "$(dirname "$out")" || exit 1
      python "$STACK_TOOLS/mk_picks_moe.py" "$H/truemono-quick/lookback/picksH.json" "$out.picksH.json" "$moe" || exit 1
      PK_REL=$(realpath -m --relative-to="$H/truemono-quick/lookback" "$out.picksH.json")
      gen_tunedHqd() { GT_EXTRA="qsa:qdense=3" gen_tuned "$1" "$2" "$PK_REL"; }
    fi
    C2_W8C_KNOBS="$C2_W8C_KNOBS${q7:+,$q7}"
    _c2_gen "$rk" "$out" 1 1 1 "$so" || exit 1
    local kd n want st v kv cell tok
    if [ -n "$hc" ]; then  # every sidecar of a chosen kind carries the first knob, none of the other kinds
      for kd in delta delta_ple qsa; do
        n=$(grep -l -F "${hc%%,*}" "$out"/hc/$kd/*.json | wc -l); want=0
        if [ -z "$kinds" ] || [[ ",$kinds," == *",$kd,"* ]]; then want=$(ls "$out"/hc/$kd/*.json | wc -l); fi
        echo "stack: $n HC sidecars of $kd carry ${hc%%,*} (want $want)"; [ "$n" = "$want" ] || exit 1
        [ "$want" = 0 ] && continue
        for v in $(echo " $STACK_K3_HC_VARIANTS " | grep -o " $kd:[^ ]*" | cut -d: -f2 | tr ',' ' '); do
          for st in a1 a2 m1 m2; do _fm_expect "$out" $st $v hc/$kd/${st}_$v.ll; done
        done
      done
    fi
    n=$(grep -l "\"sigm\": \"hw\"" "$out"/qsa2/q7_*.json | wc -l); want=0; [[ ",$q7," != *",sigm=hw,"* ]] || want=$(ls "$out"/qsa2/q7_*.json | wc -l)
    echo "stack: $n Q7 sidecars carry sigm=hw (want $want)"; [ "$n" = "$want" ] || exit 1
    for st in q5 q7; do  # Q-MMA: every Q5 / Q7 sidecar records it or none does
      n=$(grep -l "\"${st}_mm\": \"mfma\"" "$out"/qsa2/${st}_*.json | wc -l); want=0; [[ ",$q7," != *",${st}_mm=mfma,"* ]] || want=$(ls "$out"/qsa2/${st}_*.json | wc -l)
      echo "stack: $n ${st^^} sidecars carry ${st}_mm=mfma (want $want)"; [ "$n" = "$want" ] || exit 1
    done
    n=$(grep -l '"q7_vskip": true' "$out"/qsa2/q7_*.json | wc -l); want=0; [[ ",$q7," != *",q7_vskip=True,"* ]] || want=$(ls "$out"/qsa2/q7_*.json | wc -l)
    echo "stack: $n Q7 sidecars carry q7_vskip=True (want $want)"; [ "$n" = "$want" ] || exit 1
    if [ -n "$q7" ]; then for v in $STACK_K3_Q7_VARIANTS; do _fm_expect "$out" q7 $v qsa2/q7_$v.bc; done; fi
    if [[ ",$q7," == *",q5_mm=mfma,"* ]]; then for v in $STACK_K3_Q7_VARIANTS; do _fm_expect "$out" q5 $v qsa2/q5_$v.bc; done; fi
    if [ -n "$moe" ]; then  # the changed cell's M3 sidecar names knob + value, and no other M3 sidecar does
      for cell in $(echo "$moe" | tr ';' ' '); do
        kd=${cell%%.*}; v=$(echo "${cell%%:*}" | cut -d. -f2)
        for kv in $(echo "${cell#*:}" | tr ',' ' '); do
          tok="${kv%%=*}"; tok="${tok//_/}${kv#*=}"
          grep -q -E "[:_]${tok}[_\"]" "$out/k/$kd/moe/m3_$v.json" || { echo "stack: ${cell%%:*} M3 lacks $tok"; exit 1; }
        done
        _fm_expect "$out" m3 "$v" "k/$kd/moe/m3_$v.ll"
      done
    fi
    echo "stack: $(grep -c -F '[ATOM_Q38_EXTERN_STAGES]' "$out/expect_k3.txt") served-function hash lines in expect_k3.txt"
  )
}

HC_FM="invmath=rsq32,sigm=hw"
QSA_FM="sigm=hw,q5_mm=mfma,q7_mm=mfma,q7_vskip=True"

# C = fast math + Q-MMA + V-skip
gen_stackC() { _fm_gen "$1" "$2" 64 "$HC_FM" "$QSA_FM" delta,qsa ""; }
# D = C + the qsa.r4s schedule levers (schedule string only: the generated functions are C's)
gen_stackD() { _fm_gen "$1" "$2" 64 "$HC_FM" "$QSA_FM" delta,qsa "" "$STACK_D_SCHED"; }
# E = D + the MFMA router on delta.r4s
gen_stackE() { _fm_gen "$1" "$2" 64 "$HC_FM" "$QSA_FM" delta,qsa "delta.r4s:router_mma=1" "$STACK_D_SCHED"; }
# F = E + the MFMA router on qsa.r4s and qsa.r1
gen_stackF() { _fm_gen "$1" "$2" 64 "$HC_FM" "$QSA_FM" delta,qsa "delta.r4s:router_mma=1;qsa.r4s:router_mma=1;qsa.r1:router_mma=1" "$STACK_D_SCHED"; }
