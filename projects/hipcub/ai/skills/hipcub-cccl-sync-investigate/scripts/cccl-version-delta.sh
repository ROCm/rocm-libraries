#!/usr/bin/env bash
# cccl-version-delta.sh
#
# Step 0 helper: work out the CCCL version delta that an upcoming hipCUB
# sync must cover. hipCUB declares the CUB version it tracks in CMake, but
# the declaration and the code can drift apart, so this script surfaces
# FOUR signals of varying strength and reports whether they agree:
#
#   1. Signal D (primary) — HIPCUB_CCCL_VERSION_MAJOR/MINOR/PATCH in
#      projects/hipccl/hipccl3/hipcub/CMakeLists.txt. hipCUB's own exact, self-declared
#      "CCCL-compatible version", exported to users as HIPCUB_CCCL_VERSION
#      by hipcub_version.hpp.in (e.g. 300003 = 3.0.3). Drives CURRENT_TAG
#      when present. Missing on trees that predate it (and on branches that
#      reverted the CCCL 3.0 port, e.g. origin/develop after PR #10464).
#   2. Signal A (corroboration/fallback) — CCCL_MINIMUM_VERSION in
#      projects/hipccl/hipccl3/hipcub/cmake/Dependencies.cmake. The CCCL release the CUB
#      (nvcc) backend finds or downloads. A *minimum*, not an alignment
#      claim: it can sit below Signal D (3.0.0 vs 3.0.3) without anything
#      being wrong. Only drives CURRENT_TAG when Signal D is unavailable.
#   3. Signal B (corroboration) — CHANGELOG.md prose, e.g. "Feature parity
#      with CCCL/CUB 3.0.0." Known to drift stale.
#   4. Signal C (corroboration) — a curated fingerprint check
#      (version-fingerprints.tsv, sibling of this script's directory). Greps
#      hipCUB's tree for code patterns known to come from specific CCCL
#      tags. Local-only, no network needed.
#
# HIPCUB_VERSION / VERSION_STRING (e.g. 5.0.0) is hipCUB's product version,
# not a CCCL signal. Don't read it as one.
#
# It does NOT need the `cccl` git remote — the upstream tag list comes from
# the GitHub API (via `gh` if available, else `curl`), and Signals A-D only
# read the local tree.
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
# (CURRENT_TAG=..., NEXT_TAG=..., TO_TAG=..., PENDING_TAGS=...,
# HIPCUB_CCCL_VERSION_TAG=..., CCCL_MINIMUM_VERSION=..., SIGNAL_D_STATUS=...,
# VERSION_SIGNAL_AGREEMENT=..., SIGNAL_C_FLOOR_TAG=..., SIGNAL_C_STATUS=...,
# ANCESTRY_STATUS=...) for downstream steps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINGERPRINTS_FILE="$SCRIPT_DIR/../version-fingerprints.tsv"

HIPCUB_REPO=""
BASE="origin/develop"
TO_TAG=""
FROM_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)   HIPCUB_REPO="$2"; shift 2 ;;
    --repo=*) HIPCUB_REPO="${1#--repo=}"; shift ;;
    --base)   BASE="$2"; shift 2 ;;
    --base=*) BASE="${1#--base=}"; shift ;;
    --to)     TO_TAG="$2"; shift 2 ;;
    --to=*)   TO_TAG="${1#--to=}"; shift ;;
    --from)   FROM_OVERRIDE="$2"; shift 2 ;;
    --from=*) FROM_OVERRIDE="${1#--from=}"; shift ;;
    -h|--help) sed -n '2,48p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown arg: $1" >&2; exit 64 ;;
  esac
done

if [[ -z "$HIPCUB_REPO" ]]; then
  echo "ERROR: --repo <path-to-rocm-libraries> is required" >&2
  exit 64
fi

cd "$HIPCUB_REPO"
HIPCUB_PATH="projects/hipccl/hipccl3/hipcub"
ROOT_CMAKE="$HIPCUB_PATH/CMakeLists.txt"
DEPS_CMAKE="$HIPCUB_PATH/cmake/Dependencies.cmake"
CHANGELOG="$HIPCUB_PATH/CHANGELOG.md"

# ── 1. Signal D: HIPCUB_CCCL_VERSION_* from CMakeLists.txt (primary) ────────
root_cmake="$(git show "$BASE:$ROOT_CMAKE" 2>/dev/null || true)"
if [[ -z "$root_cmake" ]]; then
  echo "ERROR: could not read $ROOT_CMAKE from $BASE" >&2
  exit 1
fi
cmake_var() {
  # arg: variable name; prints the value of `set(<name> <value>)`, if any
  grep -oE "set\\($1 +[0-9]+\\)" <<<"$root_cmake" | grep -oE '[0-9]+\)$' | tr -d ')' || true
}
D_MAJOR="$(cmake_var HIPCUB_CCCL_VERSION_MAJOR)"
D_MINOR="$(cmake_var HIPCUB_CCCL_VERSION_MINOR)"
D_PATCH="$(cmake_var HIPCUB_CCCL_VERSION_PATCH)"
HIPCUB_CCCL_VERSION_TAG=""
if [[ -n "$D_MAJOR" && -n "$D_MINOR" && -n "$D_PATCH" ]]; then
  HIPCUB_CCCL_VERSION_TAG="v${D_MAJOR}.${D_MINOR}.${D_PATCH}"
fi

# ── 2. Signal A: CCCL_MINIMUM_VERSION from cmake/Dependencies.cmake ─────────
deps_cmake="$(git show "$BASE:$DEPS_CMAKE" 2>/dev/null || true)"
CCCL_MINIMUM_VERSION="$(grep -oE 'set\(CCCL_MINIMUM_VERSION +[0-9]+\.[0-9]+\.[0-9]+' <<<"$deps_cmake" \
  | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
deps_commit="$(git log -1 --format='%h %s' "$BASE" -- "$DEPS_CMAKE" 2>/dev/null || true)"

if [[ -n "$FROM_OVERRIDE" ]]; then
  CURRENT_TAG="$FROM_OVERRIDE"
elif [[ -n "$HIPCUB_CCCL_VERSION_TAG" ]]; then
  CURRENT_TAG="$HIPCUB_CCCL_VERSION_TAG"
elif [[ -n "$CCCL_MINIMUM_VERSION" ]]; then
  CURRENT_TAG="v${CCCL_MINIMUM_VERSION}"
else
  CURRENT_TAG=""
fi

signal_d_status="no HIPCUB_CCCL_VERSION_* found"
if [[ -n "$HIPCUB_CCCL_VERSION_TAG" ]]; then
  if [[ -n "$CCCL_MINIMUM_VERSION" ]]; then
    d_mm="${D_MAJOR}.${D_MINOR}"
    a_mm="$(cut -d. -f1-2 <<<"$CCCL_MINIMUM_VERSION")"
    if [[ "$d_mm" == "$a_mm" ]]; then
      signal_d_status="ok (CMakeLists.txt ${HIPCUB_CCCL_VERSION_TAG}; CCCL_MINIMUM_VERSION ${CCCL_MINIMUM_VERSION} is the same minor line)"
    else
      signal_d_status="DRIFT (CMakeLists.txt says ${HIPCUB_CCCL_VERSION_TAG}, but CCCL_MINIMUM_VERSION is ${CCCL_MINIMUM_VERSION})"
    fi
  else
    signal_d_status="ok (CMakeLists.txt ${HIPCUB_CCCL_VERSION_TAG}; CCCL_MINIMUM_VERSION unavailable to cross-check)"
  fi
fi
SIGNAL_D_STATUS="$signal_d_status"

# ── 3. Signal B: CHANGELOG.md prose (corroboration only, known to drift) ────
changelog_heading="$(git show "$BASE:$CHANGELOG" 2>/dev/null | grep -m1 -E '^## ' || true)"
changelog_cccl_mention="$(git show "$BASE:$CHANGELOG" 2>/dev/null \
  | grep -m1 -iE 'CCCL(/CUB)? [0-9]+\.[0-9]+' || true)"
changelog_ver="$(grep -oE 'CCCL(/CUB)? [0-9]+\.[0-9]+(\.[0-9]+)?' <<<"$changelog_cccl_mention" \
  | head -1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' || true)"

version_signal_agreement="unknown"
if [[ -n "$CCCL_MINIMUM_VERSION" && -n "$changelog_ver" ]]; then
  a_mm="$(cut -d. -f1-2 <<<"$CCCL_MINIMUM_VERSION")"
  b_mm="$(cut -d. -f1-2 <<<"$changelog_ver")"
  if [[ "$a_mm" == "$b_mm" ]]; then
    version_signal_agreement="ok (both signals say ${a_mm})"
  else
    version_signal_agreement="DRIFT (CCCL_MINIMUM_VERSION says ${CCCL_MINIMUM_VERSION}, CHANGELOG mentions ${changelog_ver})"
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

# Only bare vMAJOR.MINOR.PATCH releases are sync candidates.
ALL_TAGS="$(get_cccl_tags | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V -u || true)"
if [[ -z "$ALL_TAGS" ]]; then
  echo "ERROR: could not retrieve CCCL tags (need gh auth or network to api.github.com)" >&2
  exit 1
fi
LATEST_TAG="$(tail -1 <<<"$ALL_TAGS")"
[[ -z "$TO_TAG" ]] && TO_TAG="$LATEST_TAG"

# ── Pending range ─────────────────────────────────────────────────────────────
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
    cur_mm="$(sed 's/^v//' <<<"$CURRENT_TAG" | cut -d. -f1-2)"
    PENDING_TAGS="$(awk -v curmm="$cur_mm" -v to="$TO_TAG" '
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

# ── Ancestry: is CURRENT_TAG an ancestor of TO_TAG? ──────────────────────────
# CCCL cuts every release, including X.Y.0, on its branch/X.Y.x maintenance
# branch, so a tag from an older minor line is usually NOT an ancestor of a
# newer one (v3.0.0 and v3.0.3 are not ancestors of v3.1.4). Then
# `git log CURRENT..TO` also lists main-line commits whose backports are
# already in CURRENT_TAG. hipcub-commit-list.sh marks those in its BASELINE
# column. Needs the `cccl` remote fetched; reports "unknown" otherwise.
ANCESTRY_STATUS="unknown (tags not fetched locally)"
if [[ -n "$CURRENT_TAG" ]] && git rev-parse -q --verify "$CURRENT_TAG^{commit}" >/dev/null \
   && git rev-parse -q --verify "$TO_TAG^{commit}" >/dev/null; then
  if git merge-base --is-ancestor "$CURRENT_TAG" "$TO_TAG"; then
    ANCESTRY_STATUS="ancestor ($CURRENT_TAG is an ancestor of $TO_TAG)"
  else
    ANCESTRY_STATUS="NOT ancestor ($CURRENT_TAG is on a maintenance branch; $CURRENT_TAG..$TO_TAG includes commits already backported into $CURRENT_TAG)"
  fi
fi

# ── Report ───────────────────────────────────────────────────────────────────
echo "==================== CCCL → hipCUB version delta ===================="
echo "Base branch                     : $BASE"
echo
echo "Signal A: CCCL_MINIMUM_VERSION (corroboration/fallback, from $DEPS_CMAKE)"
echo "  CCCL_MINIMUM_VERSION    : ${CCCL_MINIMUM_VERSION:-<none>}"
echo "  last-touched commit     : ${deps_commit:-<none>}"
echo "  The CCCL release the CUB (nvcc) backend builds against. A minimum, not"
echo "  an alignment claim; only drives CURRENT_TAG when Signal D is missing."
echo
echo "Signal B: CHANGELOG.md prose (corroboration only — known to drift stale)"
echo "  top heading             : ${changelog_heading:-<none>}"
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
echo "Signal D: HIPCUB_CCCL_VERSION_* (primary signal when present, from $ROOT_CMAKE)"
echo "  MAJOR/MINOR/PATCH       : ${D_MAJOR:-<none>}.${D_MINOR:-<none>}.${D_PATCH:-<none>}"
echo "  decoded tag             : ${HIPCUB_CCCL_VERSION_TAG:-<none>}"
echo "  hipCUB's own declared CCCL-compatible version (exported as"
echo "  HIPCUB_CCCL_VERSION). Drives CURRENT_TAG when present."
echo "  Signal D status         : $SIGNAL_D_STATUS"
echo
echo "Signal A/B agreement     : $version_signal_agreement"
echo
echo "Derived current tag guess : ${CURRENT_TAG:-<could not derive>}"
echo "Latest CCCL release      : $LATEST_TAG"
echo "Target (--to)            : $TO_TAG"
echo "Tag ancestry             : $ANCESTRY_STATUS"
echo
if [[ -z "$PENDING_TAGS" ]]; then
  echo "RESULT: no pending CCCL releases detected (or current version could not"
  echo "be derived — STOP and confirm the current/target versions with a human"
  echo "before proceeding)."
else
  echo "Candidate pending CCCL releases (oldest-first — STOP and confirm with a"
  echo "human before treating this as authoritative):"
  n=0; for t in $PENDING_TAGS; do n=$((n+1)); echo "  $n. $t"; done
fi
echo
echo "# ---- eval-able summary (for downstream steps) ----"
echo "CURRENT_TAG='${CURRENT_TAG:-}'"
echo "NEXT_TAG='${NEXT_TAG:-}'"
echo "TO_TAG='$TO_TAG'"
echo "PENDING_TAGS='$PENDING_TAGS'"
echo "HIPCUB_CCCL_VERSION_TAG='${HIPCUB_CCCL_VERSION_TAG:-}'"
echo "CCCL_MINIMUM_VERSION='${CCCL_MINIMUM_VERSION:-}'"
echo "SIGNAL_D_STATUS='$SIGNAL_D_STATUS'"
echo "VERSION_SIGNAL_AGREEMENT='$version_signal_agreement'"
echo "SIGNAL_C_FLOOR_TAG='$SIGNAL_C_FLOOR_TAG'"
echo "SIGNAL_C_STATUS='$SIGNAL_C_STATUS'"
echo "ANCESTRY_STATUS='$ANCESTRY_STATUS'"
