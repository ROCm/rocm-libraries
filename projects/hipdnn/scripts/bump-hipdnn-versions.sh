#!/usr/bin/env bash
#
# check-version-bumps.sh - report/apply hipDNN per-component version bumps.
# Run with --help for full usage.

set -euo pipefail

usage() {
  cat <<'EOF'
check-version-bumps.sh

For each hipDNN component (identified by a version.json file), report whether
its source has drifted since the last release branch was cut, and whether its
version.json still needs to be bumped.

Classification per component:
  OK (clean)    - no CODE changed since the release ref (cmake-only churn is OK)
  OK (bumped)   - code changed, but version.json already changed too
  NEEDS BUMP    - code changed and version.json is unchanged since release

CMake project files (CMakeLists.txt, CMakePresets.json, CMakeUserPresets.json,
*.cmake / *.cmake.in, and anything under a cmake/ directory) are build plumbing,
not shipped source, so by default they do NOT by themselves trigger a bump. Pass
--cmake-bumps to make cmake changes count toward the bump decision too. The
CHANGES column classifies each component's drift regardless of that flag:
  code        - only non-cmake files changed
  cmake-only  - only cmake project files changed
  mixed       - both code and cmake files changed
  -           - nothing changed
The SRC-CHG column counts the files that drive the decision: code only by
default, or code+cmake with --cmake-bumps.

A component's "scope" is its version.json directory, EXCLUDING any nested
subtree that has its own version.json (so hip-kernel-provider and its nested
rocke/.../version.json are scored independently).

Usage:
  check-version-bumps.sh [-C|--repo <path>] [--release <ref>] [--target <ref>]
                         [--diff <component>] [--cmake-bumps] [--check]
                         [--apply] [--major[=paths]] [--minor[=paths]] [--patch[=paths]]

Defaults:
  -C, --repo <path>   Worktree folder to operate in (default: current directory)
  --release  highest origin/release/therock-* branch (version-sorted)
  --target   Endpoint to compare the release against (default HEAD). Use your
             mainline ref (e.g. origin/develop) to measure drift since the
             release was cut, rather than HEAD which also includes your
             feature-branch commits.

Inspecting drift:
  --diff <component>  Print the git diff of the exact files that count as drift
             for one component (its scope dir, excluding version.json and any
             nested component subtree), then exit. <component> is a scope dir,
             e.g. projects/hipdnn/backend. Honors --release and --target.
  --cmake-bumps  Count CMake project-file changes toward the bump decision too
             (by default they are excluded as build plumbing).
  --check    Exit non-zero (1) if any component still needs a bump; exit 0 only
             when all components are OK. For CI gating. Honors --cmake-bumps.

Bumping (writes to working-tree version.json files; report-only otherwise):
  --apply    Write bumps. Default level is MINOR for every NEEDS BUMP component
             unless overridden by the level flags below. Implied when any level
             flag is given.
  --major [paths]   Bump the listed paths (comma-separated) at MAJOR level.
  --minor [paths]   Bump the listed paths at MINOR level.
  --patch [paths]   Bump the listed paths at PATCH level.

  A level flag with NO path list becomes the DEFAULT level for every path not
  explicitly listed under another level flag. Only ONE of the three may omit
  its list. Paths match a component by its scope dir (e.g. projects/hipdnn/backend)
  or an ancestor of it.

  Increment rules (lower fields reset to zero):
    major: X.Y.Z -> (X+1).0.0    minor: X.Y.Z -> X.(Y+1).0    patch: X.Y.Z -> X.Y.(Z+1)

  NOTE: the semver level is a human judgement the script cannot make for you.

Examples:
  check-version-bumps.sh
  check-version-bumps.sh --target origin/develop
  check-version-bumps.sh --target origin/develop --diff projects/hipdnn/backend
  check-version-bumps.sh --target origin/develop --apply
  check-version-bumps.sh --target origin/develop --major dnn-providers/hip-kernel-provider --patch projects/hipdnn/flatbuffers_sdk
  check-version-bumps.sh --target origin/develop --patch --minor projects/hipdnn/backend,projects/hipdnn/frontend
EOF
}

REPO="$(pwd)"
RELEASE_REF=""
TARGET_REF="HEAD"
APPLY=false
DIFF_COMPONENT=""
CMAKE_BUMPS=false
CHECK=false

declare -a MAJOR_PATHS=() MINOR_PATHS=() PATCH_PATHS=()
MAJOR_GIVEN=false; MINOR_GIVEN=false; PATCH_GIVEN=false
MAJOR_LISTLESS=false; MINOR_LISTLESS=false; PATCH_LISTLESS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -C|--repo) REPO="$2"; shift 2 ;;
    --release) RELEASE_REF="$2"; shift 2 ;;
    --target)  TARGET_REF="$2"; shift 2 ;;
    --diff)    DIFF_COMPONENT="$2"; shift 2 ;;
    --cmake-bumps) CMAKE_BUMPS=true; shift ;;
    --check)   CHECK=true; shift ;;
    --apply)   APPLY=true; shift ;;
    --major)
      APPLY=true; MAJOR_GIVEN=true
      if [[ $# -ge 2 && "$2" != -* ]]; then IFS=',' read -ra MAJOR_PATHS <<<"$2"; shift 2
      else MAJOR_LISTLESS=true; shift; fi ;;
    --minor)
      APPLY=true; MINOR_GIVEN=true
      if [[ $# -ge 2 && "$2" != -* ]]; then IFS=',' read -ra MINOR_PATHS <<<"$2"; shift 2
      else MINOR_LISTLESS=true; shift; fi ;;
    --patch)
      APPLY=true; PATCH_GIVEN=true
      if [[ $# -ge 2 && "$2" != -* ]]; then IFS=',' read -ra PATCH_PATHS <<<"$2"; shift 2
      else PATCH_LISTLESS=true; shift; fi ;;
    -h|--help)
      usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Determine the default bump level (from the single listless level flag, if any).
DEFAULT_LEVEL="minor"
listless_count=0
$MAJOR_LISTLESS && { DEFAULT_LEVEL="major"; listless_count=$((listless_count + 1)); }
$MINOR_LISTLESS && { DEFAULT_LEVEL="minor"; listless_count=$((listless_count + 1)); }
$PATCH_LISTLESS && { DEFAULT_LEVEL="patch"; listless_count=$((listless_count + 1)); }
if [[ "$listless_count" -gt 1 ]]; then
  echo "ERROR: only one of --major/--minor/--patch may omit its path list." >&2
  exit 2
fi

git_c() { git -C "$REPO" "$@"; }

# Validate that the repo path is a git repository.
if ! git_c rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: not a git repository: $REPO" >&2
  exit 1
fi

# --- Resolve release ref -----------------------------------------------------
if [[ -z "$RELEASE_REF" ]]; then
  RELEASE_REF=$(git_c for-each-ref --format='%(refname:short)' \
      'refs/remotes/origin/release/therock-*' \
    | sort -V | tail -1)
  if [[ -z "$RELEASE_REF" ]]; then
    echo "ERROR: no origin/release/therock-* branches found. Fetch them first." >&2
    exit 1
  fi
fi

# Validate refs exist.
for ref in "$RELEASE_REF" "$TARGET_REF"; do
  if ! git_c rev-parse --verify --quiet "$ref^{commit}" >/dev/null; then
    echo "ERROR: ref not found: $ref" >&2
    exit 1
  fi
done

echo "Repo:    $REPO"
echo "Release: $RELEASE_REF ($(git_c rev-parse --short "$RELEASE_REF"))"
echo "Target:  $TARGET_REF ($(git_c rev-parse --short "$TARGET_REF"))"
echo

# --- Discover component version.json files -----------------------------------
mapfile -t VERSION_FILES < <(
  git_c ls-files 'projects/hipdnn/**/version.json' 'dnn-providers/**/version.json' \
    | sort
)

if [[ ${#VERSION_FILES[@]} -eq 0 ]]; then
  echo "No hipDNN version.json files found under projects/hipdnn or dnn-providers." >&2
  exit 1
fi

# Directories that own a version.json (used to compute nested exclusions).
declare -a SCOPE_DIRS=()
for vf in "${VERSION_FILES[@]}"; do
  SCOPE_DIRS+=("$(dirname "$vf")")
done

# Extract the version value from a version.json at a given ref (key-name agnostic).
get_version() {
  local ref="$1" file="$2"
  { git_c show "$ref:$file" 2>/dev/null \
    | grep -oE '"[0-9]+\.[0-9]+\.[0-9]+[^"]*"' || true; } | head -1 | tr -d '"'
}

# Read the version value from a working-tree version.json.
get_worktree_version() {
  { grep -oE '"[0-9]+\.[0-9]+\.[0-9]+[^"]*"' "$REPO/$1" || true; } | head -1 | tr -d '"'
}

# Print the git pathspec for a component's scope: its dir, excluding its own
# version.json and any nested component subtree (one line per pathspec element).
# Usage: build_pathspec <version.json-path>
build_pathspec() {
  local vf="$1" dir other
  dir="$(dirname "$vf")"
  printf '%s\n' "$dir" ":(exclude)$vf"
  for other in "${SCOPE_DIRS[@]}"; do
    if [[ "$other" != "$dir" && "$other" == "$dir"/* ]]; then
      printf '%s\n' ":(exclude)$other/**"
    fi
  done
}

# True if a path is a CMake project file (build plumbing, not shipped source):
# CMakeLists.txt, CMakePresets.json, CMakeUserPresets.json, *.cmake / *.cmake.in,
# or anything living in a cmake/ directory.
is_cmake_file() {
  local path="$1" base
  base="$(basename "$path")"
  case "$base" in
    CMakeLists.txt|CMakePresets.json|CMakeUserPresets.json) return 0 ;;
    *.cmake|*.cmake.in) return 0 ;;
  esac
  [[ "$path" == cmake/* || "$path" == */cmake/* ]] && return 0
  return 1
}

# Compute the next semver given a current X.Y.Z (any suffix is dropped) and level.
bump_version() {
  local current="$1" level="$2"
  local core="${current%%-*}"          # strip any -prerelease suffix
  IFS='.' read -r maj min pat <<<"$core"
  case "$level" in
    patch) pat=$((pat + 1)) ;;
    minor) min=$((min + 1)); pat=0 ;;
    major) maj=$((maj + 1)); min=0; pat=0 ;;
  esac
  echo "$maj.$min.$pat"
}

# Rewrite the version value in a working-tree version.json, preserving its key.
write_version() {
  local file="$1" old="$2" new="$3"
  # Replace the first occurrence of the old quoted version with the new one.
  sed -i "0,/\"$old\"/s//\"$new\"/" "$REPO/$file"
}

# True if component scope dir matches (equals or is nested under) any listed path.
scope_matches() {
  local dir="$1"; shift
  local p
  for p in "$@"; do
    p="${p%/}"
    [[ "$dir" == "$p" || "$dir" == "$p"/* ]] && return 0
  done
  return 1
}

# Resolve the bump level for a component's scope dir from the level path lists,
# falling back to DEFAULT_LEVEL when it matches none.
resolve_level() {
  local dir="$1"
  if [[ ${#MAJOR_PATHS[@]} -gt 0 ]] && scope_matches "$dir" "${MAJOR_PATHS[@]}"; then echo major; return; fi
  if [[ ${#MINOR_PATHS[@]} -gt 0 ]] && scope_matches "$dir" "${MINOR_PATHS[@]}"; then echo minor; return; fi
  if [[ ${#PATCH_PATHS[@]} -gt 0 ]] && scope_matches "$dir" "${PATCH_PATHS[@]}"; then echo patch; return; fi
  echo "$DEFAULT_LEVEL"
}

# --- --diff: show the drift diff for one component, then exit ----------------
if [[ -n "$DIFF_COMPONENT" ]]; then
  want="${DIFF_COMPONENT%/}"
  match_vf=""
  for vf in "${VERSION_FILES[@]}"; do
    d="$(dirname "$vf")"
    if [[ "$d" == "$want" ]]; then match_vf="$vf"; break; fi
  done
  if [[ -z "$match_vf" ]]; then
    echo "ERROR: no component with scope dir '$want'." >&2
    echo "Known components:" >&2
    for vf in "${VERSION_FILES[@]}"; do echo "  $(dirname "$vf")" >&2; done
    exit 1
  fi
  mapfile -t pathspec < <(build_pathspec "$match_vf")
  echo "Diff for $want ($RELEASE_REF..$TARGET_REF):"
  echo
  git_c diff "$RELEASE_REF..$TARGET_REF" -- "${pathspec[@]}"
  exit 0
fi

# --- Per-component drift analysis -------------------------------------------
# Populates the global arrays BUMP_FILES / BUMP_LEVELS (NEEDS BUMP components)
# and sets NEEDS_BUMP_COUNT. Reads current versions from the working tree so a
# re-run after --apply reflects the just-written bumps.
declare -a BUMP_FILES=() BUMP_LEVELS=()
declare -A BUMPED_BY_SCRIPT=()   # version.json paths this run wrote via --apply
NEEDS_BUMP_COUNT=0

# run_analysis [post] - pass "post" to render the after-apply header labels.
run_analysis() {
  local mode="${1:-pre}"
  local old_hdr="VERSION" new_hdr="PROPOSED"
  if [[ "$mode" == "post" ]]; then old_hdr="OLD VER"; new_hdr="NEW VER"; fi

  NEEDS_BUMP_COUNT=0
  BUMP_FILES=(); BUMP_LEVELS=()

  printf '%-58s %-9s %-9s %-10s %-11s %-9s  %s\n' \
    "COMPONENT (scope dir)" "$old_hdr" "$new_hdr" "SRC-CHG" "CHANGES" "VER-CHGD" "STATUS"
  printf '%-58s %-9s %-9s %-10s %-11s %-9s  %s\n' \
    "$(printf '%.0s-' {1..58})" "$(printf '%.0s-' {1..9})" "$(printf '%.0s-' {1..9})" \
    "$(printf '%.0s-' {1..10})" "$(printf '%.0s-' {1..11})" \
    "$(printf '%.0s-' {1..9})" "$(printf '%.0s-' {1..11})"

  local vf dir pathspec f rel_version cur_version
  local code_count cmake_count bump_count changes src ver status proposed level
  for vf in "${VERSION_FILES[@]}"; do
    dir="$(dirname "$vf")"
    mapfile -t pathspec < <(build_pathspec "$vf")

    # Partition changed files (release..target) into cmake plumbing vs code.
    # Only code changes drive the bump decision.
    code_count=0; cmake_count=0
    while IFS= read -r f; do
      [[ -z "$f" ]] && continue
      if is_cmake_file "$f"; then cmake_count=$((cmake_count + 1))
      else code_count=$((code_count + 1)); fi
    done < <(git_c diff --name-only "$RELEASE_REF..$TARGET_REF" -- "${pathspec[@]}")

    # CHANGES classification.
    if [[ "$code_count" -gt 0 && "$cmake_count" -gt 0 ]]; then changes="mixed"
    elif [[ "$code_count" -gt 0 ]]; then changes="code"
    elif [[ "$cmake_count" -gt 0 ]]; then changes="cmake-only"
    else changes="-"; fi

    # Files that drive the bump decision: code only, or code+cmake with --cmake-bumps.
    if $CMAKE_BUMPS; then bump_count=$((code_count + cmake_count)); else bump_count=$code_count; fi

    # Version bumped since release? Compare working-tree value to release value.
    rel_version="$(get_version "$RELEASE_REF" "$vf")"
    cur_version="$(get_worktree_version "$vf")"
    cur_version="${cur_version:-?}"

    if [[ "$rel_version" != "$cur_version" ]]; then ver="yes"; else ver="no"; fi

    proposed="$cur_version"
    if [[ "$bump_count" -eq 0 ]]; then
      # No drift that counts toward a bump -> no bump needed.
      src="no"; status="OK (clean)"
    else
      src="yes ($bump_count)"
      if [[ "$ver" == "yes" ]]; then
        # (bumped) marks rows this script wrote this run; pre-existing bumps show OK.
        if [[ -n "${BUMPED_BY_SCRIPT[$vf]:-}" ]]; then status="OK (bumped)"; else status="OK"; fi
      else
        status="NEEDS BUMP"
        level="$(resolve_level "$dir")"
        proposed="$(bump_version "$cur_version" "$level")"
        NEEDS_BUMP_COUNT=$((NEEDS_BUMP_COUNT + 1))
        BUMP_FILES+=("$vf")
        BUMP_LEVELS+=("$level")
      fi
    fi

    printf '%-58s %-9s %-9s %-10s %-11s %-9s  %s\n' \
      "$dir" "$cur_version" "$proposed" "$src" "$changes" "$ver" "$status"
  done
}

run_analysis

echo
if [[ "$NEEDS_BUMP_COUNT" -eq 0 ]]; then
  echo "All components OK. No version bumps required."
  exit 0
fi

echo "$NEEDS_BUMP_COUNT component(s) NEED a version bump."

# --- Apply bumps (opt-in) ----------------------------------------------------
if ! $APPLY; then
  echo "(dry run - pass --apply, or --major/--minor/--patch, to write these bumps)"
  $CHECK && exit 1
  exit 0
fi

echo
echo "Applying bumps to working-tree version.json files (default level: $DEFAULT_LEVEL):"
for i in "${!BUMP_FILES[@]}"; do
  vf="${BUMP_FILES[$i]}"
  level="${BUMP_LEVELS[$i]}"
  dir="$(dirname "$vf")"
  old="$(get_worktree_version "$vf")"
  if [[ -z "$old" ]]; then
    echo "  SKIP   $vf (could not parse current version)"
    continue
  fi
  new="$(bump_version "$old" "$level")"
  write_version "$vf" "$old" "$new"
  BUMPED_BY_SCRIPT["$vf"]=1
  printf '  %-6s %-52s %s -> %s\n' "$level" "$dir" "$old" "$new"
done

echo
echo "Re-running analysis after applying bumps:"
echo
run_analysis post
echo
if [[ "$NEEDS_BUMP_COUNT" -eq 0 ]]; then
  echo "All components OK after bumping."
else
  echo "WARNING: $NEEDS_BUMP_COUNT component(s) still show NEEDS BUMP."
fi
echo "Review with: git -C $REPO diff -- '**/version.json'"
if $CHECK && [[ "$NEEDS_BUMP_COUNT" -ne 0 ]]; then exit 1; fi
