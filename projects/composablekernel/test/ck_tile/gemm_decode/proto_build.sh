#!/usr/bin/env bash
# Standalone build for the LDS-split-K prototype (design doc gemm_decode §15.J).
# No CMake / no CK headers: it is intentionally self-contained. Run from the
# composablekernel root.
set -euo pipefail
# Anchor on this script's location -> composablekernel root (../../.. from here).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p build/bin
hipcc --offload-arch=gfx950 -O3 -std=c++17 \
    test/ck_tile/gemm_decode/proto_lds_splitk.cpp \
    -o build/bin/proto_lds_splitk
echo "built build/bin/proto_lds_splitk"
