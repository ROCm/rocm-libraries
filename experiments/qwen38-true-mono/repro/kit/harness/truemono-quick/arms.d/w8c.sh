# W8 WP2c (copy-rocm-lib experiments/qwen38-true-mono/wave2/WP2c.md): the rocKE QSA Q3-Q7 stage functions
# (copy-rocm-lib kernels/gfx950/qwen38_mono/qsa2) linked into every narrow QSA build of the full consolidated stack
# (cs.sh: full3 = finalHqd + the verify tail, full0 = tunedHqd + the K=0 tail). Grouped / wide QSA widths keep the
# FlyDSL stages (ATOM_Q38_EXTERN_MISSING=mono). New file of the W8 lane; nothing else of this harness is changed.
#   W8C_KNOBS  the functions' QsaKnobs (default the stack's numa=1,qdense=3; the WP2c climb's pick is passed per job)
#   W8C_VKNOBS per-variant overrides on top of W8C_KNOBS, "r1:k=v,...;r4s:k=v,..." (the climb picks per width)
#   W8C_SCHED  schedule-lever overrides of the stack's sched.txt, "cell:lever=value,...;cell:..." (e.g.
#              "qsa.r1:hold.q7=64"; value "off" removes the lever): the climb's re-tuned holds; the build lines in
#              expect_k*.txt follow
#   full3_w8c / full0_w8c   the full stack + the functions (exact knobs: bitwise the stack's Q3-Q7)
gen_w8c() { # gen_w8c <rocke stage> <out dir>: the full stack's recipe, then the QSA functions and their expect lines
  gen_tunedHqd "$1" "$2" || return 1
  (cd /atom && PYTHONPATH=/atom:/job/stages/$1/platform/python:/job/stages/$1/library ROCKE_BACKEND=python \
    ROCKE_LLVM_FLAVOR=llvm23 python -m kernels.gfx950.qwen38_mono.qsa2.gen --out "$2/qsa2" \
    --knobs "${W8C_KNOBS:-numa=1,qdense=3}" --variant-knobs "${W8C_VKNOBS:-}" > "$2/qsa2.gen.json") || return 1
  if [ -n "${W8C_SCHED:-}" ]; then
    python - "$2" "$W8C_SCHED" << 'PY' || return 1
import re, sys
d, spec = sys.argv[1], sys.argv[2]
over = {}
for grp in filter(None, spec.split(";")):
    cell, _, body = grp.partition(":")
    over[cell] = dict(kv.split("=", 1) for kv in body.split(",") if kv)
sched = open(f"{d}/sched.txt").read().strip()
parts = [p for p in sched.split(";") if p]
for cell, lev in over.items():
    idx = next((i for i, p in enumerate(parts) if p.startswith(cell + ":")), None)
    cur = dict(kv.split("=", 1) for kv in parts[idx].partition(":")[2].split(",") if kv) if idx is not None else {}
    for k, v in lev.items():
        if v == "off":
            cur.pop(k, None)
        else:
            cur[k] = v
    grp = cell + ":" + ",".join(f"{k}={v}" for k, v in sorted(cur.items()))
    if idx is None:
        parts.append(grp)
    else:
        parts[idx] = grp
open(f"{d}/sched.txt", "w").write(";".join(parts))
# the build lines of the overridden cells: "[ATOM_Q38_SCHED] build <kind> <variant>: k=v ... | extern"
for k in "0123":
    f = f"{d}/expect_k{k}.txt"
    try:
        lines = open(f).read().splitlines()
    except FileNotFoundError:
        continue
    out = []
    for ln in lines:
        m = re.match(r"\[ATOM_Q38_SCHED\] build (\S+) (\S+): (.*?)( \| extern)?$", ln)
        if m and f"{m.group(1)}.{m.group(2)}" in over:
            lev = dict(kv.split("=", 1) for kv in m.group(3).split())
            for kk, v in over[f"{m.group(1)}.{m.group(2)}"].items():
                if v == "off":
                    lev.pop(kk, None)
                else:
                    lev[kk] = v
            ln = f"[ATOM_Q38_SCHED] build {m.group(1)} {m.group(2)}: " + " ".join(f"{a}={b}" for a, b in sorted(lev.items())) + (m.group(4) or "")
        out.append(ln)
    open(f, "w").write("".join(x + "\n" for x in out))
print("w8c sched overrides:", over)
PY
  fi
  for s in q3 q4 q5 q6 q7; do
    echo "$s r4s: rocke_q38_${s}_r4s from" >> "$2/expect_k3.txt"
    echo "$s r1: rocke_q38_${s}_r1 from" >> "$2/expect_k0.txt"
  done
  (cd "$2" && find k qsa2 -type f | sort | xargs sha256sum > manifest.sha256)
}
_w8c_env() { # the stack's extern list + the five QSA functions
  ARM_ENV=$(printf '%s' "$ARM_ENV" | sed 's#b3pre=\$FN/k/{kind}/b3#&,q3=$FN/qsa2,q4=$FN/qsa2,q5=$FN/qsa2,q6=$FN/qsa2,q7=$FN/qsa2#')
}
arm_full3_w8c() { _cs_full3 w8c; _w8c_env; }
arm_full0_w8c() { _cs_full0 w8c; _w8c_env; }
