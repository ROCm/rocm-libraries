#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Installs a ROCm release into the image with AMD's runfile installer, the way it is installed on a workstation that
# builds HIP code:
#   bash rocm-installer-VERSION.run deps=install gfx=LIBRARY_TARGETS rocm compo=core,core-dev gpu-access=user assumeyes
# (the installer's components: core is the runtime, the libraries and the compiler; core-dev their development files,
# the HIP headers include/hip/* among them. The installer's default is core alone, with which hipcc cannot compile
# <hip/hip_runtime.h>.) Then it checks that the installed ROCm reports the release asked for, that its clang is the
# pinned compiler, and that hipcc compiles <hip/hip_runtime.h> for the review's GPU targets (rk doctor checks the
# same), and records what it installed in /etc/rkmon/rocm.json (rk doctor and the setup summary read it).
#
# Usage (as root, while the image is built: the Dockerfile's rocm stage, which rkmon build gives each argument from
# rkmon_lib/build_plan.py):
#   install_rocm.sh VERSION LIBRARY_TARGETS HIP_CHECK_TARGETS CLANG_COMMIT SHA256
#   VERSION  the ROCm release, such as 10.1.0, or one build of it, such as 10.1.0-5: the installed ROCm must report the
#            release (its .info/version), or the build stops (exit 3)
#   LIBRARY_TARGETS  the GPU targets for the installer's libraries, comma-separated (the compiler covers every target
#            either way). Leave blank for the installer's own.
#   HIP_CHECK_TARGETS  the GPU targets hipcc must compile <hip/hip_runtime.h> for, comma-separated: those the review's
#            GPU tests run on
#   CLANG_COMMIT  the llvm-project commit the installed clang must name in `clang --version` (the compiler the review
#            gates are tuned on); a clang that names another stops the build (exit 4). Leave blank to take any clang.
#   SHA256   the installer's SHA-256: a mismatch installs nothing and stops the build (exit 3). Leave blank to skip
#            the check.
# The installer's full download URL is the build secret rocm_package_url: rocm.package_url, or the public installer of
# VERSION that rkmon build finds in AMD's folder. This script reads it itself (docker/param.sh) and downloads it
# through docker/fetch.sh, so it is on no command line, and neither it nor its file's name is ever printed or recorded:
# it may name an internal host. Nothing of the build host goes into the image: the installer is downloaded, run, and
# deleted. The installer's deps=install puts the system packages ROCm needs into this stage, which is the final image's
# parent, so the image carries them; the checks below run on exactly the files the image keeps. It installs them with apt,
# whose errors name the build's package mirror: the installer's lines show it by its secret's name (docker/masked.sh).
# SECRETS_DIR moves the build secrets (the tests).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)  # the installer runs in its own folder
version=${1:-}
gfx=${2:-}
hip_targets=$(printf '%s' "${3:-}" | tr ',' ' ' | sed 's/^ *//; s/ *$//; s/  */ /g')
want_commit=${4:-}
want_sha=$(printf '%s' "${5:-}" | tr 'A-F' 'a-f')
case "$version" in
    ""|*[!0-9.-]*|-*|*-*-*) echo "install_rocm.sh: ROCM_VERSION ($version) is no release such as 10.1.0 (or build" \
                              "such as 10.1.0-5): set rocm.version in your build settings" >&2; exit 2 ;;
esac
release=${version%%-*}
case "$want_sha" in
    "") ;;
    *[!0-9a-f]*) bad_sha=1 ;;
    *) [ ${#want_sha} -eq 64 ] && bad_sha=0 || bad_sha=1 ;;
esac
if [ "${bad_sha:-0}" -eq 1 ]; then
    echo "install_rocm.sh: ROCM_PACKAGE_SHA256 must be the installer's SHA-256 (64 hex digits): set" \
         "rocm.package_sha256 in your build settings; nothing installed" >&2
    exit 2
fi
for t in $hip_targets; do
    case "$t" in
        *[!0-9a-z]*) bad_targets=1 ;;
    esac
done
if [ -z "$hip_targets" ] || [ "${bad_targets:-0}" -eq 1 ]; then
    echo "install_rocm.sh: HIP_CHECK_TARGETS (${3:-}) is no list of GPU targets such as gfx942,gfx950: nothing" \
         "installed" >&2
    exit 2
fi
url=$(sh "$HERE/param.sh" rocm_package_url)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
echo "install_rocm.sh: downloading the installer of ROCm $version"
if ! why=$(printf '%s\n' "$url" | sh "$HERE/fetch.sh" "$work/installer.run" 2>&1); then
    echo "install_rocm.sh: the installer did not download ($why)" >&2
    exit 2
fi
got_sha=$(sha256sum "$work/installer.run" | cut -d' ' -f1)
if [ -n "$want_sha" ]; then
    if [ "$want_sha" != "$got_sha" ]; then
        echo "install_rocm.sh: the installer's SHA-256 is $got_sha, not rocm.package_sha256 ($want_sha): nothing" \
             "installed" >&2
        exit 3
    fi
    echo "install_rocm.sh: the installer matches rocm.package_sha256"
fi
compo=core,core-dev
args="deps=install rocm compo=$compo gpu-access=user assumeyes"
[ -n "$gfx" ] && args="deps=install gfx=$gfx rocm compo=$compo gpu-access=user assumeyes"
before=$(ls -d /opt/rocm/core-* 2>/dev/null || true)  # none yet: ls fails, and set -e would stop
# shellcheck disable=SC2086
(cd "$work" && sh "$HERE/masked.sh" package_mirror -- bash ./installer.run $args)
# The release's core folder: core-<major.minor> of its version, as the installer lays it out (another release's folder
# may be there too); else the newest folder it added; else the newest there is.
core=/opt/rocm/core-$(printf '%s' "$release" | cut -d. -f1,2)
if [ ! -d "$core" ]; then
    core=""
    for d in $(ls -d /opt/rocm/core-* 2>/dev/null | sort -V); do
        printf '%s\n' "$before" | grep -qxF "$d" || core=$d
    done
    [ -n "$core" ] || core=$(ls -d /opt/rocm/core-* 2>/dev/null | sort -V | tail -1)
fi
[ -n "$core" ] || { echo "install_rocm.sh: the installer left no /opt/rocm/core-*" >&2; exit 3; }
reported=$(sed -n '1{s/[[:space:]]*$//;p;}' "$core/.info/version" 2>/dev/null || true)
if [ "$reported" != "$release" ]; then
    echo "install_rocm.sh: the installed ROCm reports ${reported:-no version} ($core/.info/version), not rocm.version's" \
         "release $release: set rocm.version to the release rocm.package_url installs" >&2
    exit 3
fi
echo "install_rocm.sh: ROCm reports $reported"
llvm_bin="$core/lib/llvm/bin"
[ -x "$llvm_bin/clang" ] || { echo "install_rocm.sh: no clang at $llvm_bin" >&2; exit 3; }
clang=$("$llvm_bin/clang" --version | head -1)
if [ -n "$want_commit" ]; then
    case "$clang" in
        *"$want_commit"*) echo "install_rocm.sh: its clang is the pinned compiler ($want_commit)" ;;
        *) echo "install_rocm.sh: the installer's clang is \"$clang\", not the pinned compiler ($want_commit): the" \
                "review gates would differ. Set rocm.version (or rocm.package_url) to the ROCm build whose clang" \
                "that is" >&2
           exit 4 ;;
    esac
fi
# hipcc and the HIP headers: rocKE's tests that compile HIP need them, for each target the review's GPU tests run on
# (HIP_CHECK_TARGETS). Without them those tests would skip in the review checks, so a release that lacks them stops
# the build (exit 3).
hipcc="$core/bin/hipcc"
[ -x "$hipcc" ] || { echo "install_rocm.sh: the installer left no hipcc at $hipcc" >&2; exit 3; }
printf '#include <hip/hip_runtime.h>\n__global__ void rkmon_probe() {}\n' > "$work/probe.hip"
for t in $hip_targets; do
    if ! "$hipcc" --offload-arch="$t" -c "$work/probe.hip" -o "$work/probe-$t.o" > "$work/probe.log" 2>&1 \
        || [ ! -s "$work/probe-$t.o" ]; then
        echo "install_rocm.sh: $hipcc cannot compile <hip/hip_runtime.h> for $t (the installer's $compo" \
             "components should bring the HIP headers): $(tail -3 "$work/probe.log" | tr '\n' ' ')" >&2
        exit 3
    fi
done
echo "install_rocm.sh: $hipcc compiles <hip/hip_runtime.h> for $hip_targets"
mkdir -p /etc/rkmon
os=$(. /etc/os-release 2>/dev/null && echo "${ID:-}-${VERSION_ID:-}")
python3 - "$version" "$core" "$llvm_bin" "$clang" "$gfx" "$os" "$compo" "$hipcc" "$got_sha" <<'EOF'
import datetime, json, sys
version, core, llvm_bin, clang, gfx, os_, compo, hipcc, sha = sys.argv[1:10]
rec = {"version": version, "core": core, "llvm_bin": llvm_bin, "clang": clang, "gfx": gfx, "os": os_,
       "components": compo, "hipcc": hipcc, "sha256": sha,
       "installed_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
with open("/etc/rkmon/rocm.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=2)
    f.write("\n")
print("install_rocm.sh: installed", json.dumps(rec))
EOF
