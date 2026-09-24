#!/usr/bin/env bash
# Bench helper for gfx1250 MXF4 subtile + rocprofv3 thread trace.
#
#   ./run.sh tensile          # generate + run the Tensile client (fills logs/<stamp>/tensile)
#   ./run.sh att              # ATT on the latest ClientParameters.ini (builds first if needed)
#   ./run.sh pmc              # PMC / kernel-trace on the latest client config
#   ./run.sh att --tensile    # wrap the whole Tensile process (compile + client)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILELITE="$(cd "$ROOT/../.." && pwd)"
CONFIGS="$ROOT/configs"
LOGS="$ROOT/logs"
TENSILE_YAML="$CONFIGS/tensile_mxf4_att.yaml"
ATT_YAML="$CONFIGS/rocprof_att.yaml"
PMC_YAML="$CONFIGS/rocprof_pmc.yaml"

ROCPROFV3="${ROCPROFV3:-$(command -v rocprofv3 || true)}"

# Tensile needs joblib/msgpack/etc. The system python3 does not have them, so prefer
# the tensilelite venv. Create it with:
#   python3 -m venv --system-site-packages .venv
#   .venv/bin/python -m pip install -r requirements.txt
if [[ -z "${PYTHON:-}" && -x "$TENSILELITE/.venv/bin/python" ]]; then
  PYTHON="$TENSILELITE/.venv/bin/python"
fi
PYTHON="${PYTHON:-python3}"

# rocisa is built by `invoke build-client` but not pip-installed; import it from the build tree.
ROCISA_BUILD="$TENSILELITE/build_tmp/tensilelite/rocisa"
export PYTHONPATH="${TENSILELITE}:${ROCISA_BUILD}${PYTHONPATH:+:$PYTHONPATH}"

# Physical GPU 2 only. Tensile Device: 0 and rocprof att_gpu_index 0 then mean that card.
# Set exactly one filter: ROCR_VISIBLE_DEVICES already renumbers the survivor to 0, so
# also setting HIP_VISIBLE_DEVICES=2 would filter the filtered list and yield hipErrorNoDevice.
GPU_ID="${GPU_ID:-2}"
export HIP_VISIBLE_DEVICES="$GPU_ID"
unset ROCR_VISIBLE_DEVICES
echo "==> HIP_VISIBLE_DEVICES=$GPU_ID"

usage() {
  sed -n '2,8p' "$0" | sed 's/^# \?//'
  echo
  echo "Env: GPU_ID (default 2), ROCPROFV3, TENSILE_CLIENT, PYTHON, PYTHONPATH"
  echo "Logs: $LOGS/<stamp>/"
  exit 1
}

stamp_dir() {
  local d="$LOGS/$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$d"
  echo "$d"
}

latest_tensile_out() {
  local latest
  latest="$(ls -1dt "$LOGS"/*/tensile 2>/dev/null | head -1 || true)"
  if [[ -z "$latest" ]]; then
    echo ""
    return
  fi
  printf '%s\n' "$latest"
}

find_client_ini() {
  local out="$1"
  find "$out" -name 'ClientParameters.ini' -print 2>/dev/null | head -1
}

find_client_bin() {
  if [[ -n "${TENSILE_CLIENT:-}" && -x "$TENSILE_CLIENT" ]]; then
    printf '%s\n' "$TENSILE_CLIENT"
    return
  fi
  local cand
  for cand in \
      "$TENSILELITE/build_tmp/tensilelite/client/tensilelite-client" \
      "$TENSILELITE/build/tensilelite-client/tensilelite-client" \
      "$(command -v tensilelite-client || true)"; do
    if [[ -n "$cand" && -x "$cand" ]]; then
      printf '%s\n' "$cand"
      return
    fi
  done
  find "$TENSILELITE" -name tensilelite-client -type f -executable 2>/dev/null | head -1
}

check_python() {
  if ! "$PYTHON" -c 'import joblib, rocisa' >/dev/null 2>&1; then
    echo "Tensile python deps missing for: $PYTHON" >&2
    echo "Set up once with:" >&2
    echo "  cd $TENSILELITE" >&2
    echo "  python3 -m venv --system-site-packages .venv" >&2
    echo "  .venv/bin/python -m pip install -r requirements.txt" >&2
    echo "(rocisa comes from $ROCISA_BUILD; run 'invoke build-client' if it is absent)" >&2
    exit 1
  fi
}

run_tensile() {
  local dest="$1"
  mkdir -p "$dest"
  check_python
  local client extra=()
  client="$(find_client_bin)"
  if [[ -n "$client" ]]; then
    extra=(--prebuilt-client "$client")
    echo "==> Tensile  client=$client"
  fi
  echo "==> Tensile  python=$PYTHON"
  echo "==> Tensile  yaml=$TENSILE_YAML  out=$dest"
  "$PYTHON" "$TENSILELITE/Tensile/bin/Tensile" "${extra[@]}" "$TENSILE_YAML" "$dest"
}

need_rocprof() {
  if [[ -z "$ROCPROFV3" || ! -x "$ROCPROFV3" ]]; then
    echo "rocprofv3 not found. Set ROCPROFV3 or put /opt/rocm/bin on PATH." >&2
    exit 1
  fi
}

profile_client() {
  local rocprof_input="$1"
  local label="$2"
  local wrap_tensile="${3:-0}"
  local dest ini client

  need_rocprof
  dest="$(stamp_dir)"
  echo "==> log dir  $dest"

  if [[ "$wrap_tensile" == "1" ]]; then
    echo "==> rocprofv3 $label wrapping Tensile (compile + client)"
    "$ROCPROFV3" \
      --input "$rocprof_input" \
      --output-directory "$dest/rocprof" \
      --output-file "$label" \
      -- \
      "$PYTHON" "$TENSILELITE/Tensile/bin/Tensile" "$TENSILE_YAML" "$dest/tensile"
    echo "==> done  $dest"
    return
  fi

  local tensile_out
  tensile_out="$(latest_tensile_out)"
  if [[ -z "$tensile_out" ]]; then
    echo "==> no prior Tensile output; running tensile first"
    run_tensile "$dest/tensile"
    tensile_out="$dest/tensile"
  else
    echo "==> reusing Tensile out  $tensile_out"
    ln -sfn "$tensile_out" "$dest/tensile.latest"
  fi

  ini="$(find_client_ini "$tensile_out")"
  client="$(find_client_bin)"
  if [[ -z "$ini" ]]; then
    echo "ClientParameters.ini not found under $tensile_out" >&2
    exit 1
  fi
  if [[ -z "$client" ]]; then
    echo "tensilelite-client not found. Build with: cd $TENSILELITE && invoke build-client" >&2
    echo "Or set TENSILE_CLIENT=/path/to/tensilelite-client" >&2
    exit 1
  fi

  echo "==> rocprofv3 $label"
  echo "    client=$client"
  echo "    ini=$ini"
  "$ROCPROFV3" \
    --input "$rocprof_input" \
    --output-directory "$dest/rocprof" \
    --output-file "$label" \
    -- \
    "$client" --config-file "$ini"
  echo "==> done  $dest"
}

cmd="${1:-}"
shift || true
case "$cmd" in
  tensile)
    dest="$(stamp_dir)"
    run_tensile "$dest/tensile"
    echo "==> done  $dest"
    ;;
  att)
    wrap=0
    if [[ "${1:-}" == "--tensile" ]]; then
      wrap=1
    fi
    profile_client "$ATT_YAML" "att" "$wrap"
    ;;
  pmc)
    wrap=0
    if [[ "${1:-}" == "--tensile" ]]; then
      wrap=1
    fi
    profile_client "$PMC_YAML" "pmc" "$wrap"
    ;;
  *)
    usage
    ;;
esac
