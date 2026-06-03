#!/usr/bin/env bash
# Reproduce the MicroPython ck-dsl-provider spike environment from a fresh checkout.
#
# Everything this builds (the MicroPython clone, its unix/mpy-cross/embed builds,
# and the generated ckbundle/) is intentionally git-ignored — this script is the
# committed recipe that regenerates them. Run from anywhere:  spike/setup.sh
set -euo pipefail

SPIKE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPY="$SPIKE/micropython"
MPY_COMMIT="44a569b637"   # pinned: the commit this spike was verified against

echo "== spike dir: $SPIKE"

# 1. Clone MicroPython (pinned commit) if absent.
if [ ! -d "$MPY/.git" ]; then
  echo "== cloning MicroPython @ $MPY_COMMIT"
  git clone https://github.com/micropython/micropython "$MPY"
  git -C "$MPY" checkout "$MPY_COMMIT"
else
  echo "== MicroPython already present ($(git -C "$MPY" rev-parse --short HEAD))"
fi

# 2. Unix port (used by the G1a/G1b codegen + comgr-ffi spikes; has ffi/uctypes).
echo "== building MicroPython unix port"
make -C "$MPY/ports/unix" submodules >/dev/null
make -C "$MPY/ports/unix" -j"$(nproc)"
UNIX_BIN="$MPY/ports/unix/build-standard/micropython"
echo "   unix: $UNIX_BIN"

# 3. mpy-cross (needed to freeze modules for the embed port / Phase 2).
echo "== building mpy-cross"
make -C "$MPY/mpy-cross" -j"$(nproc)" >/dev/null
echo "   mpy-cross: $MPY/mpy-cross/build/mpy-cross"

# 4. Toolchain check: the bare embed example (proves the embed port builds + runs).
echo "== building + running the stock embed example (toolchain check)"
( cd "$MPY/examples/embedding" && make -f micropython_embed.mk >/dev/null && make >/dev/null && ./embed | head -1 )

# 5. Generate the transformed ck_dsl bundle (field()/star-unpacking/open/os transforms
#    + trimmed __init__s) consumed by the run_g1*/run_conv harnesses.
echo "== building ck_dsl bundle"
python3 "$SPIKE/mp1/build_bundle.py"

cat <<EOF

== setup complete. Verify the spike:
   # G1a (elementwise codegen IR == CPython):
   $UNIX_BIN $SPIKE/mp1/run_g1.py /tmp/g1_mp.ll
   # G1b (conv -> HSACO end-to-end in MicroPython):
   LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib $UNIX_BIN -X heapsize=1024M $SPIKE/mp1/run_g1b.py
EOF
