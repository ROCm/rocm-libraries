#!/bin/bash
# Build two variants of the synthetic plugin .so plus a generic dlopen host:
#
#   libsynth_plugin_private.so  - linked against libMIOpen_private.so.1
#                                  (matches the real plugin's link wiring
#                                   when MIOPEN_ENABLE_HIPDNN_WRAPPER=ON)
#   libsynth_plugin_public.so   - linked against libMIOpen.so.1 from the
#                                  flagon build (matches the real plugin's
#                                  wiring when MIOpen_private target is
#                                  absent; also matches the pre-built
#                                  TheRock plugin .so currently on disk)
#
# Both .so files have CXX_VISIBILITY_PRESET hidden equivalent (-fvisibility=hidden)
# and explicit default visibility on the entry point, matching the real plugin.

set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
REPO=/data/nhanna/repos/rocm-libraries/projects/miopen
FLAG_LIB=$REPO/build-flagon/lib
INC="-I$REPO/include -I$REPO/build-flagon/include -I/opt/rocm/include"
CC=/opt/rocm/llvm/bin/clang++
CXXFLAGS="-xc++ -O2 -Wall -Wextra -D__HIP_PLATFORM_AMD__=1 -fvisibility=hidden"
ENTRY="-fvisibility-inlines-hidden"

cd "$DIR"

# Force the entry point to default visibility so dlsym() can find it.
cat > _entry_wrap.c <<'EOF'
__attribute__((visibility("default"))) extern int plugin_run(void);
EOF

build_variant() {
    local variant=$1   # private | public
    local liblink=$2   # MIOpen_private | MIOpen
    local out=$DIR/libsynth_plugin_${variant}.so

    "$CC" $CXXFLAGS $INC -fPIC -shared \
        -Wl,--version-script=/dev/stdin \
        -o "$out" \
        synthetic_plugin.c \
        -L"$FLAG_LIB" -l"$liblink" \
        -Wl,-rpath,"$FLAG_LIB" <<EOF
{ global: plugin_run; local: *; };
EOF
    echo "built $out"
}

build_variant private MIOpen_private
build_variant public  MIOpen

# Generic dlopen host (does not link MIOpen itself; only the loaded .so does).
$CC -O2 -Wall -Wextra -xc -o "$DIR/host" host.c -ldl
echo "built $DIR/host"
