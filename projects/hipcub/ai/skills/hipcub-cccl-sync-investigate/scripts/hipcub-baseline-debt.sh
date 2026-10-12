#!/usr/bin/env bash
# hipcub-baseline-debt.sh
#
# Lists places where hipCUB (at --base) is behind upstream CUB *already at
# the baseline tag*: API debt earlier syncs left behind. No commit in
# CURRENT_TAG..TO_TAG mentions these, so the per-commit resolve loop never
# meets them. In the 3.0.3 -> 3.1.4 sync the human port (rocm-libraries
# PR #13083) cleaned up several of these and the AI-driven run missed all
# of them.
#
# Checks:
#   1. public names hipCUB declares in its thread/ block/ warp/ headers that
#      no longer exist anywhere in upstream cub/cub/ at the baseline tag
#      (e.g. removed deprecated operators Equality, Sum, Max, ...);
#   2. legacy `ARCH` template parameters (upstream dropped LEGACY_PTX_ARCH);
#   3. `namespace internal` APIs hipCUB still has when upstream has none in
#      the same header;
#   4. CCCL-namespace macros (_CCCL_*, CCCL_*) that hipCUB defines itself;
#   5. public headers added upstream in --from..--to that
#      backend/cub/hipcub.hpp neither re-exports nor wraps.
#
# Every hit is a candidate for the human to confirm, not a verdict.
#
# Usage: hipcub-baseline-debt.sh --repo <path> --base <ref> --from <CURRENT_TAG> [--to <TO_TAG>]

set -euo pipefail

REPO=""; BASE=""; FROM=""; TO=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --base) BASE="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 64 ;;
  esac
done
if [[ -z "$REPO" || -z "$BASE" || -z "$FROM" ]]; then
  echo "Usage: $0 --repo <path> --base <ref> --from <CURRENT_TAG> [--to <TO_TAG>]" >&2
  exit 64
fi
TO="${TO:-$FROM}"
cd "$REPO"

INC="projects/hipccl/hipccl3/hipcub/hipcub/include/hipcub"
ROC="$INC/backend/rocprim"

echo "=== 1. hipCUB public names absent from upstream cub/cub/ at $FROM ==="
# Capitalized struct/class names in the rocPRIM backend's thread/, block/,
# warp/ and iterator/ headers. There's no parser: detail helpers and
# hipCUB-only extensions show up too. Read each hit before acting.
found=0
while IFS= read -r line; do
  file="${line%%:*}"; name="${line##*:}"
  # A declaration, not just the word: Sum/Max/Min survive as member names.
  if ! git grep -qE "(struct|class)\\s+([A-Z_]+(\\([^)]*\\))?\\s+)?$name\\b|using\\s+$name\\s*=" \
         "$FROM" -- cub/cub/ 2>/dev/null; then
    found=1
    printf '%s\t%s\n' "$name" "$file"
  fi
done < <(git grep -E '^\s*(struct|class)\s' "$BASE" -- \
           "$ROC/thread/" "$ROC/block/" "$ROC/warp/" "$ROC/iterator/" 2>/dev/null \
           | sed -E "s|^$BASE:||" \
           | sed -nE 's/^([^:]+):\s*(struct|class)\s+(HIPCUB_DEPRECATED[A-Z_]*(\("[^"]*"\))?\s+)?([A-Z][A-Za-z0-9_]*)\s*(final\s*)?(\{|:[^:]|<|$).*/\1:\5/p' \
           | sort -u)
if [[ "$found" -eq 0 ]]; then echo "(none)"; fi

echo
echo "=== 2. Legacy ARCH template parameters ==="
up_arch="$(git grep -l -E 'LEGACY_PTX_ARCH|int ARCH\b' "$FROM" -- cub/cub/block cub/cub/warp 2>/dev/null | wc -l || true)"
echo "upstream block/warp files with LEGACY_PTX_ARCH/int ARCH at $FROM: $up_arch"
hits="$(git grep -n -E 'int\s+ARCH\s*=|HIPCUB_ARCH\s*[,>]' "$BASE" -- "$INC" 2>/dev/null | sed "s|^$BASE:||" || true)"
if [[ -n "$hits" && "$up_arch" -eq 0 ]]; then
  echo "hipCUB still has ARCH template parameters upstream removed:"
  head -30 <<< "$hits" | sed 's/^/\t/'
  n="$(wc -l <<< "$hits")"; [[ "$n" -gt 30 ]] && echo "	... $((n - 30)) more"
else
  echo "(none)"
fi

echo
echo "=== 3. namespace internal in hipCUB vs upstream ==="
found=0
while IFS= read -r f; do
  rel="${f#"$ROC/"}"; up="cub/cub/${rel%.hpp}.cuh"
  if ! git show "$FROM:$up" 2>/dev/null | grep -q 'namespace internal'; then
    found=1
    printf '%s\t(upstream %s has no namespace internal)\n' "$f" "$up"
  fi
done < <(git grep -l 'namespace internal' "$BASE" -- "$ROC" 2>/dev/null | sed "s|^$BASE:||")
if [[ "$found" -eq 0 ]]; then echo "(none)"; fi

echo
echo "=== 4. CCCL-namespace macros defined by hipCUB ==="
hits="$(git grep -n -E '#\s*define\s+_?CCCL_[A-Z0-9_]+' "$BASE" -- "$INC" 2>/dev/null | sed "s|^$BASE:||" || true)"
if [[ -n "$hits" ]]; then
  echo "$hits" | sed 's/^/\t/'
  echo "(rename to HIPCUB_* unless there's a reason to keep; see porting-categories.md 7)"
else
  echo "(none)"
fi

echo
echo "=== 5. Public headers added upstream in $FROM..$TO not reachable through backend/cub/ ==="
reexport="$(git show "$BASE:$INC/backend/cub/hipcub.hpp" 2>/dev/null || true)"
found=0
while IFS= read -r h; do
  rel="${h#cub/cub/}"
  case "$rel" in
    detail/* | agent/* | device/dispatch/* | */specializations/* | */impl/* | cub.cuh | config.cuh | version.cuh) continue ;;
  esac
  grep -qF "<cub/$rel>" <<< "$reexport" && continue
  git cat-file -e "$BASE:$INC/backend/cub/${rel%.cuh}.hpp" 2>/dev/null && continue
  found=1
  echo "cub/$rel"
done < <(comm -13 <(git ls-tree -r --name-only "$FROM" -- cub/cub/ | grep -E '\.cuh$' | sort) \
                  <(git ls-tree -r --name-only "$TO" -- cub/cub/ | grep -E '\.cuh$' | sort))
if [[ "$found" -eq 0 ]]; then echo "(none)"; fi
