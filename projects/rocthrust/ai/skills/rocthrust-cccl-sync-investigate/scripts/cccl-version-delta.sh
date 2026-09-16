#!/usr/bin/env bash
# cccl-version-delta.sh
#
# Step 0 helper: work out the CCCL version delta that an upcoming rocThrust
# sync must cover. Unlike RCCL/NCCL, rocThrust has no single authoritative
# "last synced" file (no equivalent of version.mk). Instead this script
# surfaces FOUR signals of varying strength and reports whether they agree:
#
#   1. Signal A (corroboration/fallback) — _THRUST_REQUIRED_LIBCXX_VERSION_MAJOR/MINOR
#      in projects/rocthrust/thrust/detail/config/libcxx.h — the libcu++/
#      libhipcxx version rocThrust says it requires. ASSUMPTION (not
#      guaranteed): this has historically tracked the overall CCCL release
#      major.minor, since libcu++/CUB/Thrust are now versioned together
#      upstream. Only used as the CURRENT_TAG fallback when Signal D is
#      unavailable.
#   2. Signal B (corroboration) — CHANGELOG.md prose — e.g. "CCCL 2.8.x
#      compatibility is deprecated... brought forward to CCCL 3.0.x". Known
#      to drift stale (same caveat as the RCCL/NCCL CHANGELOG check).
#   3. Signal C (corroboration) — a curated fingerprint check
#      (version-fingerprints.tsv, sibling of this script's directory). Greps
#      rocThrust's local tree for specific code patterns known to have been
#      introduced in specific CCCL tags. Local-only, no network needed.
#      Exists because Signals A and B can both agree and still be wrong:
#      dogfooding this script found rocThrust's develop already contained a
#      v2.8.1 fix verbatim while Signals A/B both reported v2.8.0 as current
#      — and Signal D later revealed the true baseline was v2.8.5.
#   4. Signal D (primary) — THRUST_VERSION in
#      projects/rocthrust/thrust/version.h. An exact, first-party, self-
#      declared version macro: Thrust's own doc comment calls this file "the
#      only Thrust header that is guaranteed to change with every thrust
#      release." Decoded per the documented formula (major = value/100000,
#      minor = value/100%1000, patch = value%100) into an exact tag. This is
#      the strongest signal available and drives CURRENT_TAG when present.
#
# It does NOT need the `cccl` git remote — the upstream tag list comes from
# the GitHub API (via `gh` if available, else `curl`), and Signals C/D only
# read the local rocThrust tree. Deep per-commit/per-file analysis happens
# later in the skill once the `cccl` remote is fetched.
#
# Usage: cccl-version-delta.sh --repo <path-to-rocm-libraries> \
#                              [--base <branch>] [--to <tag>] [--from <tag>]
#
#   --repo <path>   Absolute path to the rocm-libraries working tree (required).
#   --base <branch> Branch to measure against (default: origin/develop).
#   --to <tag>      Stop the pending range at this tag (default: latest release).
#   --from <tag>    Override the detected "current" tag (rarely needed).
#
# Output: a human-readable report on stdout. The final block is shell-eval'able
# (CURRENT_TAG=..., NEXT_TAG=..., PENDING_TAGS=..., VERSION_SIGNAL_AGREEMENT=...,
# SIGNAL_C_FLOOR_TAG=..., SIGNAL_C_STATUS=..., THRUST_VERSION_RAW=...,
# THRUST_VERSION_TAG=..., SIGNAL_D_STATUS=...) for downstream steps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINGERPRINTS_FILE="$SCRIPT_DIR/../version-fingerprints.tsv"

ROCTHRUST_REPO=""
BASE="origin/develop"
TO_TAG=""
FROM_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)   ROCTHRUST_REPO="$2"; shift 2 ;;
    --repo=*) ROCTHRUST_REPO="${1#--repo=}"; shift ;;
    --base)   BASE="$2"; shift 2 ;;
    --base=*) BASE="${1#--base=}"; shift ;;
    --to)     TO_TAG="$2"; shift 2 ;;
    --to=*)   TO_TAG="${1#--to=}"; shift ;;
    --from)   FROM_OVERRIDE="$2"; shift 2 ;;
    --from=*) FROM_OVERRIDE="${1#--from=}"; shift ;;
    -h|--help) sed -n '2,51p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown arg: $1" >&2; exit 64 ;;
  esac
done

if [[ -z "$ROCTHRUST_REPO" ]]; then
  echo "ERROR: --repo <path-to-rocm-libraries> is required" >&2
  exit 64
fi

cd "$ROCTHRUST_REPO"
ROCTHRUST_PATH="projects/rocthrust"
LIBCXX_H="$ROCTHRUST_PATH/thrust/detail/config/libcxx.h"
CHANGELOG="$ROCTHRUST_PATH/CHANGELOG.md"
VERSION_H="$ROCTHRUST_PATH/thrust/version.h"

# ── 1. Signal D: THRUST_VERSION macro from version.h (primary signal) ───────
version_h="$(git show "$BASE:$VERSION_H" 2>/dev/null || true)"
THRUST_VERSION_RAW="$(grep -oE '#define THRUST_VERSION +[0-9]+' <<<"$version_h" | awk '{print $3}')"
THRUST_VERSION_TAG=""
if [[ -n "$THRUST_VERSION_RAW" ]]; then
  thrust_major=$(( THRUST_VERSION_RAW / 100000 ))
  thrust_minor=$(( (THRUST_VERSION_RAW / 100) % 1000 ))
  thrust_patch=$(( THRUST_VERSION_RAW % 100 ))
  THRUST_VERSION_TAG="v${thrust_major}.${thrust_minor}.${thrust_patch}"
fi

# ── 2. Signal A: required libcu++/libhipcxx version from libcxx.h ───────────
libcxx_h="$(git show "$BASE:$LIBCXX_H" 2>/dev/null || true)"
if [[ -z "$libcxx_h" ]]; then
  echo "ERROR: could not read $LIBCXX_H from $BASE" >&2
  exit 1
fi
LIBCXX_MAJOR="$(grep -oE '_THRUST_REQUIRED_LIBCXX_VERSION_MAJOR +[0-9]+' <<<"$libcxx_h" | awk '{print $2}')"
LIBCXX_MINOR="$(grep -oE '_THRUST_REQUIRED_LIBCXX_VERSION_MINOR +[0-9]+' <<<"$libcxx_h" | awk '{print $2}')"

if [[ -n "$FROM_OVERRIDE" ]]; then
  CURRENT_TAG="$FROM_OVERRIDE"
elif [[ -n "$THRUST_VERSION_TAG" ]]; then
  CURRENT_TAG="$THRUST_VERSION_TAG"
elif [[ -n "$LIBCXX_MAJOR" && -n "$LIBCXX_MINOR" ]]; then
  CURRENT_TAG="v${LIBCXX_MAJOR}.${LIBCXX_MINOR}.0"
else
  CURRENT_TAG=""
fi
libcxx_h_commit="$(git log -1 --format='%h %s' "$BASE" -- "$LIBCXX_H" 2>/dev/null || true)"

signal_d_status="no THRUST_VERSION macro found"
if [[ -n "$THRUST_VERSION_TAG" ]]; then
  thrust_mm="${thrust_major}.${thrust_minor}"
  if [[ -n "$LIBCXX_MAJOR" && -n "$LIBCXX_MINOR" ]]; then
    libcxx_mm_for_d="${LIBCXX_MAJOR}.${LIBCXX_MINOR}"
    if [[ "$thrust_mm" == "$libcxx_mm_for_d" ]]; then
      signal_d_status="ok (version.h ${THRUST_VERSION_TAG} agrees with libcxx.h major.minor ${libcxx_mm_for_d})"
    else
      signal_d_status="DRIFT (version.h says ${THRUST_VERSION_TAG}, but libcxx.h/CHANGELOG suggest ${libcxx_mm_for_d})"
    fi
  else
    signal_d_status="ok (version.h ${THRUST_VERSION_TAG}; libcxx.h major.minor unavailable to cross-check)"
  fi
fi
SIGNAL_D_STATUS="$signal_d_status"

# ── 3. Signal B: CHANGELOG.md prose (corroboration only, known to drift) ────
changelog_heading="$(git show "$BASE:$CHANGELOG" 2>/dev/null | grep -m1 -E '^## ' || true)"
changelog_cccl_mention="$(git show "$BASE:$CHANGELOG" 2>/dev/null \
  | grep -m1 -iE 'CCCL [0-9]+\.[0-9]+' || true)"
changelog_ver="$(grep -oE 'CCCL [0-9]+\.[0-9]+' <<<"$changelog_cccl_mention" \
  | head -1 | grep -oE '[0-9]+\.[0-9]+' || true)"

version_signal_agreement="unknown"
if [[ -n "$LIBCXX_MAJOR" && -n "$changelog_ver" ]]; then
  libcxx_mm="${LIBCXX_MAJOR}.${LIBCXX_MINOR}"
  if [[ "$libcxx_mm" == "$changelog_ver" ]]; then
    version_signal_agreement="ok (both signals say ${libcxx_mm})"
  else
    version_signal_agreement="DRIFT (libcxx.h says ${libcxx_mm}, CHANGELOG mentions ${changelog_ver})"
  fi
fi

# ── 4. Signal C: curated local-code fingerprint check ────────────────────────
declare -a fp_lines=()
if [[ -f "$FINGERPRINTS_FILE" ]]; then
  while IFS=$'\t' read -r fp_tag fp_path fp_pattern fp_note; do
    [[ -z "$fp_tag" || "$fp_tag" == \#* ]] && continue
    content="$(git show "$BASE:$fp_path" 2>/dev/null || true)"
    if [[ -n "$content" ]] && grep -qF "$fp_pattern" <<<"$content"; then
      fp_lines+=("$fp_tag	$fp_path	MATCH	$fp_note")
    else
      fp_lines+=("$fp_tag	$fp_path	no match	$fp_note")
    fi
  done < "$FINGERPRINTS_FILE"
fi

SIGNAL_C_FLOOR_TAG=""
matched_fp_tags=()
for line in "${fp_lines[@]+"${fp_lines[@]}"}"; do
  IFS=$'\t' read -r fp_tag fp_path fp_result _ <<<"$line"
  [[ "$fp_result" == "MATCH" ]] || continue
  matched_fp_tags+=("$fp_tag")
done
if [[ ${#matched_fp_tags[@]} -gt 0 ]]; then
  SIGNAL_C_FLOOR_TAG="$(printf '%s\n' "${matched_fp_tags[@]}" | sort -V | tail -1)"
fi

signal_c_status="no fingerprints matched"
if [[ -n "$SIGNAL_C_FLOOR_TAG" ]]; then
  if [[ -z "$CURRENT_TAG" ]] || [[ "$(printf '%s\n%s\n' "$CURRENT_TAG" "$SIGNAL_C_FLOOR_TAG" | sort -V | tail -1)" == "$SIGNAL_C_FLOOR_TAG" && "$CURRENT_TAG" != "$SIGNAL_C_FLOOR_TAG" ]]; then
    drift_tags="$(printf '%s ' "${matched_fp_tags[@]}")"
    signal_c_status="DRIFT (local code already contains fingerprint(s) for: ${drift_tags% })"
  else
    signal_c_status="ok (no contradiction)"
  fi
fi
SIGNAL_C_STATUS="$signal_c_status"

# If Signal C found evidence of a fingerprint newer than Signal D's declared
# version, surface that too — it means local patches exist ahead of the
# version macro itself.
if [[ -n "$SIGNAL_C_FLOOR_TAG" && -n "$THRUST_VERSION_TAG" ]]; then
  if [[ "$(printf '%s\n%s\n' "$THRUST_VERSION_TAG" "$SIGNAL_C_FLOOR_TAG" | sort -V | tail -1)" == "$SIGNAL_C_FLOOR_TAG" && "$THRUST_VERSION_TAG" != "$SIGNAL_C_FLOOR_TAG" ]]; then
    SIGNAL_D_STATUS="${SIGNAL_D_STATUS}; NOTE: Signal C floor (${SIGNAL_C_FLOOR_TAG}) is newer than version.h — check for local patches ahead of the declared version"
  fi
fi

# ── 5. Upstream CCCL tag list (GitHub API; gh preferred, curl fallback) ──────
get_cccl_tags() {
  if command -v gh >/dev/null 2>&1; then
    gh api --paginate repos/NVIDIA/cccl/tags -q '.[].name' 2>/dev/null && return 0
  fi
  local page out
  for page in 1 2 3 4 5; do
    out="$(curl -fsSL "https://api.github.com/repos/NVIDIA/cccl/tags?per_page=100&page=${page}" 2>/dev/null || true)"
    [[ -z "$out" || "$out" == "[]" ]] && break
    jq -r '.[].name' <<<"$out" 2>/dev/null || true
  done
}

# Exclude pre-release / toolkit-pinned suffixes (-rc, -dev, -ctkNN, etc.) —
# only bare vMAJOR.MINOR.PATCH releases are sync candidates.
ALL_TAGS="$(get_cccl_tags | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V -u || true)"
if [[ -z "$ALL_TAGS" ]]; then
  echo "ERROR: could not retrieve CCCL tags (need gh auth or network to api.github.com)" >&2
  exit 1
fi
LATEST_TAG="$(tail -1 <<<"$ALL_TAGS")"
[[ -z "$TO_TAG" ]] && TO_TAG="$LATEST_TAG"

# ── Pending range ─────────────────────────────────────────────────────────────
# If CURRENT_TAG isn't an exact tag match (likely, since it's derived from a
# major.minor guess, not an exact patch), fall back to "everything <= TO_TAG
# with major.minor greater than the detected current major.minor".
PENDING_TAGS=""
if [[ -n "$CURRENT_TAG" ]]; then
  if grep -qxF "$CURRENT_TAG" <<<"$ALL_TAGS"; then
    PENDING_TAGS="$(awk -v cur="$CURRENT_TAG" -v to="$TO_TAG" '
      { tags[NR]=$0 }
      END {
        started=0
        for (i=1;i<=NR;i++) {
          if (tags[i]==cur) { started=1; continue }
          if (started) { print tags[i]; if (tags[i]==to) break }
        }
      }' <<<"$ALL_TAGS" | tr '\n' ' ' | sed 's/ *$//')"
  else
    PENDING_TAGS="$(awk -v curmm="${LIBCXX_MAJOR:-0}.${LIBCXX_MINOR:-0}" -v to="$TO_TAG" '
      function mm(v) { split(v, a, "."); return a[1]"."a[2] }
      { tags[NR]=$0 }
      END {
        for (i=1;i<=NR;i++) {
          t=tags[i]; gsub(/^v/,"",t)
          if (mm(t) > curmm) { print tags[i] }
          if (tags[i]==to) break
        }
      }' <<<"$ALL_TAGS" | tr '\n' ' ' | sed 's/ *$//')"
  fi
fi
NEXT_TAG="$(awk '{print $1}' <<<"$PENDING_TAGS")"

# ── Report ───────────────────────────────────────────────────────────────────
echo "==================== CCCL → rocThrust version delta ===================="
echo "Base branch                     : $BASE"
echo
echo "Signal A: required libcu++/libhipcxx version (corroboration, from $LIBCXX_H)"
echo "  _THRUST_REQUIRED_LIBCXX_VERSION_MAJOR/MINOR : ${LIBCXX_MAJOR:-<none>}.${LIBCXX_MINOR:-<none>}"
echo "  last-touched commit                          : ${libcxx_h_commit:-<none>}"
echo "  ASSUMPTION: this tracks the overall CCCL release major.minor. This has"
echo "  held historically (libcu++/CUB/Thrust are versioned together upstream"
echo "  as of CCCL), but is NOT guaranteed. Only used to derive CURRENT_TAG when"
echo "  Signal D is unavailable — confirm with a human before relying on it as"
echo "  the sync target."
echo
echo "Signal B: CHANGELOG.md prose (corroboration only — known to drift stale)"
echo "  top heading            : ${changelog_heading:-<none>}"
echo "  CCCL mention            : ${changelog_cccl_mention:-<none>}"
echo
echo "Signal C: local-code fingerprint check (version-fingerprints.tsv, no network needed)"
if [[ ${#fp_lines[@]} -eq 0 ]]; then
  echo "  <no fingerprints file found or file is empty>"
else
  for line in "${fp_lines[@]}"; do
    IFS=$'\t' read -r fp_tag fp_path fp_result fp_note <<<"$line"
    echo "  [$fp_result]  $fp_tag  $fp_path"
    echo "            $fp_note"
  done
fi
echo "  Signal C status         : $SIGNAL_C_STATUS"
echo
echo "Signal D: THRUST_VERSION macro (primary signal when present, from $VERSION_H)"
echo "  THRUST_VERSION (raw)    : ${THRUST_VERSION_RAW:-<none>}"
echo "  decoded tag             : ${THRUST_VERSION_TAG:-<none>}"
echo "  Thrust's own doc comment calls this file 'the only Thrust header that is"
echo "  guaranteed to change with every thrust release' — an exact, first-party"
echo "  signal. Drives CURRENT_TAG when present; Signals A/B/C above are"
echo "  corroboration/fallback only."
echo "  Signal D status         : $SIGNAL_D_STATUS"
echo
echo "Signal A/B agreement     : $version_signal_agreement"
echo
echo "Derived current tag guess : ${CURRENT_TAG:-<could not derive>}"
echo "Latest CCCL release      : $LATEST_TAG"
echo "Target (--to)            : $TO_TAG"
echo
if [[ -z "$PENDING_TAGS" ]]; then
  echo "RESULT: no pending CCCL releases detected (or current version could not"
  echo "be derived — STOP and confirm the current/target versions with a human"
  echo "before proceeding)."
else
  echo "Candidate pending CCCL releases (oldest-first — STOP and confirm with a"
  echo "human before treating this as authoritative; unlike RCCL/NCCL there is"
  echo "no exact version.mk-style ground truth here):"
  n=0; for t in $PENDING_TAGS; do n=$((n+1)); echo "  $n. $t"; done
fi
echo
echo "# ---- eval-able summary (for downstream steps) ----"
echo "CURRENT_TAG='${CURRENT_TAG:-}'"
echo "NEXT_TAG='${NEXT_TAG:-}'"
echo "TO_TAG='$TO_TAG'"
echo "PENDING_TAGS='$PENDING_TAGS'"
echo "VERSION_SIGNAL_AGREEMENT='$version_signal_agreement'"
echo "SIGNAL_C_FLOOR_TAG='$SIGNAL_C_FLOOR_TAG'"
echo "SIGNAL_C_STATUS='$SIGNAL_C_STATUS'"
echo "THRUST_VERSION_RAW='${THRUST_VERSION_RAW:-}'"
echo "THRUST_VERSION_TAG='${THRUST_VERSION_TAG:-}'"
echo "SIGNAL_D_STATUS='$SIGNAL_D_STATUS'"
