#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Hermetic self-tests for slice-preflight.sh. These tests use a temporary
# Git repository and a fake Docker client; they never contact a Docker daemon.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUT="$HERE/../slice-preflight.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

failures=0
ok() { printf 'ok - %s\n' "$1"; }
bad() { printf 'not ok - %s\n' "$1" >&2; failures=$((failures + 1)); }

expect_success() {
  local name="$1"
  shift
  if "$@" >"$TMP/stdout" 2>"$TMP/stderr"; then
    ok "$name"
  else
    sed 's/^/  /' "$TMP/stderr" >&2
    bad "$name"
  fi
}

expect_failure() {
  local name="$1" pattern="$2"
  shift 2
  if "$@" >"$TMP/stdout" 2>"$TMP/stderr"; then
    bad "$name (unexpected success)"
  elif grep -Fq -- "$pattern" "$TMP/stderr"; then
    ok "$name"
  else
    sed 's/^/  /' "$TMP/stderr" >&2
    bad "$name (missing error: $pattern)"
  fi
}

REPO="$TMP/repo"
SCRIPT_REL="projects/hipblaslt/skills/tensilelite-mutation-rerun/scripts/slice-preflight.sh"
SRC_REL="projects/hipblaslt/tensilelite"
mkdir -p "$REPO/$(dirname "$SCRIPT_REL")" "$REPO/$SRC_REL/Tensile"
cp "$SUT" "$REPO/$SCRIPT_REL"
printf 'baseline\n' >"$REPO/$SRC_REL/Tensile/source.py"
git -C "$REPO" init -q
git -C "$REPO" config user.email mutation-selftest@example.invalid
git -C "$REPO" config user.name mutation-selftest
git -C "$REPO" add .
git -C "$REPO" commit -qm baseline

FAKE_BIN="$TMP/fake-bin"
mkdir -p "$FAKE_BIN"
cat >"$FAKE_BIN/docker" <<'EOF'
#!/usr/bin/env bash
set -u

case "${1:-}" in
  version)
    exit "${FAKE_DOCKER_VERSION_RC:-0}"
    ;;
  inspect)
    [[ "${FAKE_CONTAINER_EXISTS:-true}" == "true" ]] || exit 1
    case "$*" in
      *'{{.State.Status}}'*) printf '%s\n' "${FAKE_CONTAINER_STATUS:-running}" ;;
      *'{{.Config.Image}}'*) printf '%s\n' 'hipblaslt-mutation' ;;
      *'{{.Image}}'*) printf '%s\n' 'sha256:image-id' ;;
    esac
    ;;
  image)
    printf '%s\n' 'rocm/dev-ubuntu-22.04@sha256:digest'
    ;;
  exec)
    printf '%s\n' '3.6.0'
    ;;
  *)
    exit 2
    ;;
esac
EOF
chmod +x "$FAKE_BIN/docker"

run_preflight() {
  env PATH="$FAKE_BIN:$PATH" "$REPO/$SCRIPT_REL" \
    --slice 1 \
    --module Tensile/source.py \
    --container tl-mut \
    --src "$SRC_REL" \
    --out "$1"
}

OUT="$TMP/out-success"
expect_success "clean source and running container" run_preflight "$OUT"
python3 - "$OUT/env.json" <<'PY' || bad "successful artifact contents"
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["tracked_source_clean"] is True
assert data["container_exists"] is True
assert data["container_status"] == "running"
assert data["container_image"] == "hipblaslt-mutation"
assert data["container_image_id"] == "sha256:image-id"
assert data["container_image_digest"].endswith("@sha256:digest")
assert data["mutmut_version"] == "3.6.0"
assert data["slug"] == "source"
PY
[[ $failures -ne 0 ]] || ok "successful artifact contents"

SYMLINK_OUT="$TMP/out-symlink-temp"
SENTINEL="$TMP/preflight-sentinel"
mkdir -p "$SYMLINK_OUT"
printf 'preserve sentinel\n' > "$SENTINEL"
ln -s "$SENTINEL" "$SYMLINK_OUT/env.json.tmp"
expect_success "atomic output ignores a hostile legacy temp symlink" \
  run_preflight "$SYMLINK_OUT"
[[ "$(<"$SENTINEL")" == "preserve sentinel" ]] \
  && ok "legacy temp symlink target is preserved" \
  || bad "legacy temp symlink target is preserved"
python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$SYMLINK_OUT/env.json" \
  && ok "atomic output remains valid JSON" \
  || bad "atomic output remains valid JSON"

printf 'untracked\n' >"$REPO/$SRC_REL/Tensile/untracked.py"
expect_success "untracked campaign files do not dirty tracked source" \
  run_preflight "$TMP/out-untracked"

printf 'edited\n' >"$REPO/$SRC_REL/Tensile/source.py"
expect_failure "dirty tracked source is rejected" "tracked source is dirty" \
  run_preflight "$TMP/out-dirty"
[[ ! -e "$TMP/out-dirty/env.json" ]] \
  && ok "dirty failure writes no artifact" \
  || bad "dirty failure writes no artifact"
printf 'baseline\n' >"$REPO/$SRC_REL/Tensile/source.py"

git -C "$REPO" update-index --assume-unchanged "$SRC_REL/Tensile/source.py"
printf 'hidden edit\n' >"$REPO/$SRC_REL/Tensile/source.py"
expect_failure "assume-unchanged source is rejected" \
  "skip-worktree or assume-unchanged" run_preflight "$TMP/out-assume-unchanged"
git -C "$REPO" update-index --no-assume-unchanged "$SRC_REL/Tensile/source.py"
printf 'baseline\n' >"$REPO/$SRC_REL/Tensile/source.py"

git -C "$REPO" update-index --skip-worktree "$SRC_REL/Tensile/source.py"
printf 'hidden edit\n' >"$REPO/$SRC_REL/Tensile/source.py"
expect_failure "skip-worktree source is rejected" \
  "skip-worktree or assume-unchanged" run_preflight "$TMP/out-skip-worktree"
git -C "$REPO" update-index --no-skip-worktree "$SRC_REL/Tensile/source.py"
printf 'baseline\n' >"$REPO/$SRC_REL/Tensile/source.py"

expect_failure "unreachable Docker daemon is rejected" "daemon not reachable" \
  env FAKE_DOCKER_VERSION_RC=1 PATH="$FAKE_BIN:$PATH" \
  "$REPO/$SCRIPT_REL" --slice 1 --module source.py --container tl-mut \
  --src "$SRC_REL" --out "$TMP/out-no-daemon"

expect_failure "missing container is rejected" "container does not exist" \
  env FAKE_CONTAINER_EXISTS=false PATH="$FAKE_BIN:$PATH" \
  "$REPO/$SCRIPT_REL" --slice 1 --module source.py --container missing \
  --src "$SRC_REL" --out "$TMP/out-no-container"

OUTSIDE_SRC="$TMP/outside-source"
mkdir -p "$OUTSIDE_SRC"
expect_failure "source outside the script worktree is rejected" \
  "source dir must be inside the current Git worktree" \
  env PATH="$FAKE_BIN:$PATH" "$REPO/$SCRIPT_REL" \
  --slice 1 --module source.py --container tl-mut \
  --src "$OUTSIDE_SRC" --out "$TMP/out-outside-source"

ln -s "$OUTSIDE_SRC" "$REPO/external-source"
expect_failure "source symlink escaping the worktree is rejected" \
  "source dir must be inside the current Git worktree" \
  env PATH="$FAKE_BIN:$PATH" "$REPO/$SCRIPT_REL" \
  --slice 1 --module source.py --container tl-mut \
  --src external-source --out "$TMP/out-symlink-source"

NO_DOCKER_BIN="$TMP/no-docker-bin"
mkdir -p "$NO_DOCKER_BIN"
for command_name in dirname git readlink tr; do
  ln -s "$(command -v "$command_name")" "$NO_DOCKER_BIN/$command_name"
done
expect_failure "missing Docker binary is rejected" "binary not found" \
  env PATH="$NO_DOCKER_BIN" /bin/bash "$REPO/$SCRIPT_REL" \
  --slice 1 --module source.py --container tl-mut \
  --src "$SRC_REL" --out "$TMP/out-no-docker"

STOPPED_OUT="$TMP/out-stopped"
expect_success "stopped container is recorded without executing mutmut" \
  env FAKE_CONTAINER_STATUS=exited PATH="$FAKE_BIN:$PATH" \
  "$REPO/$SCRIPT_REL" --slice 1 --module source.py --container tl-mut \
  --src "$SRC_REL" --out "$STOPPED_OUT"
python3 - "$STOPPED_OUT/env.json" <<'PY' || bad "stopped-container artifact contents"
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["container_status"] == "exited"
assert data["mutmut_version"] is None
PY
[[ $failures -ne 0 ]] || ok "stopped-container artifact contents"

if [[ $failures -ne 0 ]]; then
  printf '%d preflight self-test(s) failed\n' "$failures" >&2
  exit 1
fi
printf 'all preflight self-tests passed\n'
