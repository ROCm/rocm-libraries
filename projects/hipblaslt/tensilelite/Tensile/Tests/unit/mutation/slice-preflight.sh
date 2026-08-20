#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# slice-preflight.sh — per-slice environment / base-pin verification.
#
# Records the reproducibility state required BEFORE a mutation slice starts and
# refuses to proceed if that state is unsafe. It captures, into a per-slice
# env.json:
#   - source commit SHA + branch (git -C <src> rev-parse ...)
#   - canonical container name, image, image id, and (best-effort) image digest
#   - mutmut version from inside the container, if the container is running
#   - tracked-source-clean state (TRACKED source only, not worktree-clean)
#   - the exact dirty tracked files if not clean
#   - UTC timestamp, slice id, module, derived slug
#
# It FAILS non-zero (and writes NO env.json) if:
#   - the source dir is missing;
#   - TRACKED source under <src> is dirty (untracked sync/ and work/ artifacts
#     are NOT treated as dirty source);
#   - docker is unavailable, or the named container does not exist.
#
# Safety: read-only against source. It never edits
# tracked source, never runs mutmut/tests, and never starts or stops containers.
# `docker exec` for the mutmut version is best-effort: a stopped container yields
# a null version, not a failure.
#
# Usage:
#   slice-preflight.sh --slice <id> --module <path-or-name> \
#       [--out <dir>] [--container tl-mut] [--src projects/hipblaslt/tensilelite]
#
# Output artifact (on success only):
#   work/mutation/slices/<slice>-<slug>/env.json
#
# Library mode: source with MUTMUT_PREFLIGHT_LIB_ONLY=1 to expose the pure helpers
# (derive_slug, emit_json) without running main().

set -u

die() { printf 'slice-preflight: ERROR: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------- pure helpers
# derive_slug — module path-or-name -> lowercase alnum slug (dirs and .py stripped).
#   Tensile/LibraryIO.py -> libraryio ; CommonUtilities -> commonutilities
derive_slug() {
  local m="$1" base
  base="${m##*/}"        # strip directory
  base="${base%.py}"     # strip .py extension
  base="$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')"
  base="$(printf '%s' "$base" | tr -c 'a-z0-9' '-')"  # non-alnum -> '-'
  base="${base#-}"; base="${base%-}"                   # trim leading/trailing '-'
  printf '%s' "$base"
}

# emit_json — write env.json via a tiny embedded Python emitter (safe escaping of
# filenames/digests is error-prone in pure shell, so an emitter is justified per
# the issue note). All values arrive via env vars; the dirty file list is passed
# newline-separated in PF_DIRTY_FILES. Writes to $1.
emit_json() {
  local out_file="$1"
  PF_OUT_FILE="$out_file" python3 - <<'PY'
import json, os

def val(name):
    v = os.environ.get(name, "")
    return v if v != "" else None

dirty_raw = os.environ.get("PF_DIRTY_FILES", "")
dirty = [ln for ln in dirty_raw.splitlines() if ln.strip()]

clean_env = os.environ.get("PF_TRACKED_CLEAN", "")
tracked_clean = (clean_env == "true")

def mv(name):
    v = os.environ.get(name, "")
    if v == "true":
        return True
    if v == "false":
        return False
    return None

data = {
    "schema": "slice-preflight/1",
    "timestamp_utc": val("PF_TIMESTAMP"),
    "slice": val("PF_SLICE"),
    "module": val("PF_MODULE"),
    "slug": val("PF_SLUG"),
    "source_dir": val("PF_SRC"),
    "source_sha": val("PF_SHA"),
    "source_branch": val("PF_BRANCH"),
    "tracked_source_clean": tracked_clean,
    "dirty_tracked_files": dirty,
    "container_name": val("PF_CONTAINER"),
    "container_exists": mv("PF_CONTAINER_EXISTS"),
    "container_status": val("PF_CONTAINER_STATUS"),
    "container_image": val("PF_IMAGE"),
    "container_image_id": val("PF_IMAGE_ID"),
    "container_image_digest": val("PF_IMAGE_DIGEST"),
    "mutmut_version": val("PF_MUTMUT_VERSION"),
    "generator": "projects/hipblaslt/tensilelite/Tensile/Tests/unit/mutation/slice-preflight.sh",
}

out = os.environ["PF_OUT_FILE"]
tmp = out + ".tmp"
try:
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)  # atomic; never truncates a prior valid env.json on failure
except Exception:
    try:
        os.unlink(tmp)
    except OSError:
        pass
    raise
PY
}

# Library-only mode: expose helpers for the docker-free selftest, skip main().
if [[ "${MUTMUT_PREFLIGHT_LIB_ONLY:-0}" == "1" ]]; then
  return 0 2>/dev/null || exit 0
fi

# ------------------------------------------------------------------- arg parse
SLICE=""; MODULE=""; OUT=""; CONTAINER="tl-mut"; SRC="projects/hipblaslt/tensilelite"
need_val() { [[ $# -ge 2 ]] || die "$1 requires a value"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --slice)     need_val "$@"; SLICE="$2"; shift 2 ;;
    --module)    need_val "$@"; MODULE="$2"; shift 2 ;;
    --out)       need_val "$@"; OUT="$2"; shift 2 ;;
    --container) need_val "$@"; CONTAINER="$2"; shift 2 ;;
    --src)       need_val "$@"; SRC="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$SLICE" ]]  || die "--slice is required"
[[ -n "$MODULE" ]] || die "--module is required"

# --------------------------------------------------------------- resolve root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
fi
cd "$ROOT" || die "cannot cd to repo root: $ROOT"

SLUG="$(derive_slug "$MODULE")"
[[ -n "$SLUG" ]] || die "could not derive a slug from --module '$MODULE'"

if [[ -z "$OUT" ]]; then
  OUT="work/mutation/slices/${SLICE}-${SLUG}"
fi

# ------------------------------------------------------------ source dir check
[[ -d "$SRC" ]] || die "source dir missing: $SRC (cwd=$ROOT)"

# --------------------------------------------------------------- docker checks
command -v docker >/dev/null 2>&1 || die "docker is unavailable (binary not found)"
docker version >/dev/null 2>&1     || die "docker is unavailable (daemon not reachable)"
docker inspect --type container "$CONTAINER" >/dev/null 2>&1 \
  || die "container does not exist: $CONTAINER"

# ------------------------------------------------------ tracked-source-clean
# Scope to the source subtree AND drop untracked ('??') entries: a new untracked
# file (e.g. a new characterization test) is NOT dirty tracked source. This is
# tracked-source-clean, not worktree-clean, so sync/ and work/ artifacts (both
# outside $SRC, and untracked) never count.
DIRTY_FILES=""
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  status="${line:0:2}"
  [[ "$status" == "??" ]] && continue          # untracked -> not dirty source
  DIRTY_FILES+="${line:3}"$'\n'
done < <(git status --porcelain -- "$SRC" 2>/dev/null || true)

if [[ -n "$DIRTY_FILES" ]]; then
  printf 'slice-preflight: ERROR: tracked source is dirty under %s:\n' "$SRC" >&2
  printf '%s' "$DIRTY_FILES" | sed 's/^/  - /' >&2
  exit 1
fi
TRACKED_CLEAN="true"

# --------------------------------------------------------- gather source facts
SHA="$(git -C "$SRC" rev-parse HEAD 2>/dev/null || true)"
BRANCH="$(git -C "$SRC" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"

# ------------------------------------------------------ gather container facts
CONTAINER_STATUS="$(docker inspect --type container "$CONTAINER" --format '{{.State.Status}}' 2>/dev/null || true)"
IMAGE="$(docker inspect --type container "$CONTAINER" --format '{{.Config.Image}}' 2>/dev/null || true)"
IMAGE_ID="$(docker inspect --type container "$CONTAINER" --format '{{.Image}}' 2>/dev/null || true)"
IMAGE_DIGEST=""
if [[ -n "$IMAGE_ID" ]]; then
  IMAGE_DIGEST="$(docker image inspect "$IMAGE_ID" --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' 2>/dev/null || true)"
fi

# mutmut version: best-effort. A stopped container -> null (do NOT start it, do
# NOT fail). Take the last whitespace token of the output (mutmut prints e.g.
# "mutmut version 3.6.0" or just the version).
MUTMUT_VERSION=""
if [[ "$CONTAINER_STATUS" == "running" ]]; then
  raw_ver="$(docker exec "$CONTAINER" sh -lc 'mutmut version 2>/dev/null || mutmut --version 2>/dev/null' 2>/dev/null || true)"
  raw_ver="$(printf '%s' "$raw_ver" | tr -d '\r' | tail -n1)"
  MUTMUT_VERSION="${raw_ver##* }"
fi

# ---------------------------------------------------------------- emit artifact
TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
mkdir -p "$OUT" || die "cannot create output dir: $OUT"
OUT_FILE="$OUT/env.json"

PF_TIMESTAMP="$TIMESTAMP" \
PF_SLICE="$SLICE" \
PF_MODULE="$MODULE" \
PF_SLUG="$SLUG" \
PF_SRC="$SRC" \
PF_SHA="$SHA" \
PF_BRANCH="$BRANCH" \
PF_TRACKED_CLEAN="$TRACKED_CLEAN" \
PF_DIRTY_FILES="" \
PF_CONTAINER="$CONTAINER" \
PF_CONTAINER_EXISTS="true" \
PF_CONTAINER_STATUS="$CONTAINER_STATUS" \
PF_IMAGE="$IMAGE" \
PF_IMAGE_ID="$IMAGE_ID" \
PF_IMAGE_DIGEST="$IMAGE_DIGEST" \
PF_MUTMUT_VERSION="$MUTMUT_VERSION" \
  emit_json "$OUT_FILE" || die "failed to write env.json"

printf 'slice-preflight: OK slice=%s module=%s slug=%s\n' "$SLICE" "$MODULE" "$SLUG"
printf 'slice-preflight: wrote %s\n' "$OUT_FILE"
printf 'slice-preflight: sha=%s branch=%s container=%s(%s) mutmut=%s\n' \
  "$SHA" "$BRANCH" "$CONTAINER" "$CONTAINER_STATUS" "${MUTMUT_VERSION:-null}"
