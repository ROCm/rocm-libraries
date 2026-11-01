# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Sourced by split/run_split.sh, per_config/run_per_config.sh and per_config/run_per_config_build.sh.
# Every location the measurement scripts read, write or delete can be overridden from the
# environment; the defaults are the paths the committed baselines were measured with.
#   TL           TensileLite source tree pytest runs from (e.g. a git worktree of another commit)
#   CLIENT       prebuilt tensilelite-client
#   ART          build artifact dir (split build phase writes it, the run phases read it). The
#                default is the measured baseline, which check_rm_target refuses to delete, so
#                split/run_split.sh needs ART set to a new dir.
#   TENSILE_CACHE      Tensile's cache root, cleared by clear_caches. TENSILE_HELPER_CACHE_DIR and
#                      TENSILE_HELPER_PCH_DIR are always exported as $TENSILE_CACHE/helper_cache and
#                      $TENSILE_CACHE/helper_pch (inherited values are ignored), so clearing
#                      TENSILE_CACHE clears the helper-kernel cache and HIP-header PCH Tensile really
#                      uses. TENSILE_DISABLE_HELPER_PCH=1 is passed through (compiles without a PCH).
#   TENSILELITE_CACHE  ~/.tensilelite, cleared by clear_caches as before (current TensileLite
#                      sources neither read nor write it).
# OUT, LIST and SCRATCH are set by the scripts themselves (their defaults depend on the script);
# each script then calls check_rm_target on every directory it deletes.
TL=${TL:-/home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite}
CLIENT=${CLIENT:-/tmp/tl_client_msgpack/tensilelite/client/tensilelite-client}
ART=${ART:-/tmp/tl_split_gfx950/artifacts}
TENSILE_CACHE=${TENSILE_CACHE:-/home/alvasile/.tensile}
TENSILELITE_CACHE=${TENSILELITE_CACHE:-/home/alvasile/.tensilelite}
export TENSILE_HELPER_CACHE_DIR=$TENSILE_CACHE/helper_cache
export TENSILE_HELPER_PCH_DIR=$TENSILE_CACHE/helper_pch
MEASURE_DIR=$(dirname "$(realpath -- "${BASH_SOURCE[0]}")")
# Measured baseline client build and artifacts; never deleted.
BASELINE_DIRS=(/tmp/tl_client_msgpack /tmp/tl_split_gfx950/artifacts)

# check_rm_target VAR...: exit unless every $VAR is safe to `rm -rf`: an absolute path at least two
# levels deep that is not $HOME or a directory containing it, and that neither is, contains, nor is
# inside $TL, $OUT, this script's directory, a baseline dir, $CLIENT's directory or $ART (unless
# VAR is ART).
check_rm_target() {
  local var val real keep keeps
  for var in "$@"; do
    val=${!var}
    case $val in
      /*) ;;
      *) echo "$var='$val' must be an absolute path" >&2; exit 1 ;;
    esac
    real=$(realpath -m -- "$val")
    case $real in
      /*/*) ;;
      *) echo "$var='$val' is too broad to delete" >&2; exit 1 ;;
    esac
    if [ -n "$HOME" ]; then
      keep=$(realpath -m -- "$HOME")
      case $keep/ in
        "$real"/*) echo "$var='$val' is or contains $keep; refusing to delete it" >&2; exit 1 ;;
      esac
    fi
    keeps=("$TL" "$OUT" "$MEASURE_DIR" "${BASELINE_DIRS[@]}" "$(dirname -- "$CLIENT")")
    [ "$var" = ART ] || keeps+=("$ART")
    for keep in "${keeps[@]}"; do
      [ -n "$keep" ] || continue
      keep=$(realpath -m -- "$keep")
      case $keep/ in
        "$real"/*) echo "$var='$val' is or contains $keep; refusing to delete it" >&2; exit 1 ;;
      esac
      case $real/ in
        "$keep"/*) echo "$var='$val' is inside $keep; refusing to delete it" >&2; exit 1 ;;
      esac
    done
  done
}

clear_caches() { rm -rf -- "$TENSILE_CACHE" "$TENSILELITE_CACHE"; }
