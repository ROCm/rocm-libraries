#!/usr/bin/env bash
# Build the mp2 embed integration binary: generate the conv codegen frozen tree,
# the embed package + frozen content, then compile with main.c. (Outputs git-ignored.)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
python3 "$HERE/../mp1/build_bundle.py"
python3 build_frozen.py
make -f gen.mk micropython-embed-package
make -f gen.mk build-embed/frozen_content.c
mapfile -t SRCS < <(find micropython_embed -name '*.c')
SRCS+=(build-embed/frozen_content.c embed_port.c main.c)
SRCS+=(../micropython/extmod/modre.c comgr_compile.c modcomgr.c)
cc -I. -Imicropython_embed -Imicropython_embed/port -Ibuild-embed -I../micropython \
   -DMICROPY_MODULE_FROZEN_MPY=1 -DMICROPY_MODULE_FROZEN_STR=1 \
   -I/opt/rocm-7.2.4/include -Wall -Og -fno-common -o embed_ckdsl "${SRCS[@]}" -lm -L/opt/rocm-7.2.4/lib -lamd_comgr
echo "== run =="
LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib ./embed_ckdsl
