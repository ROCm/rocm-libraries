#!/usr/bin/env bash
################################################################################
#
# test.sh -- environment wrapper for rocisa_stinkytofu_adaptor tests.
#
# Auto-detects the built ``stinkytofu`` / ``rocisa`` binding directory by
# walking up from this script to the ``tensilelite/`` directory and
# globbing for ``*build*/tensilelite/rocisa`` (the layout cmake creates),
# then exports PYTHONPATH so both the parent unittest runner and the
# per-path subprocesses can ``import stinkytofu`` / ``import rocisa`` /
# ``import rocisa_stinkytofu_adaptor`` without further configuration.
#
# Usage
# -----
#   ./test.sh                                # discover & run every test in this dir
#   ./test.sh test_emission_consistency      # run a single file (suffix .py optional)
#   ./test.sh test_emission_consistency -v   # ... with extra unittest args
#   ./test.sh -v                             # discover & run with -v
#   ./test.sh --help                         # this message
#
# Environment overrides
# ---------------------
#   PYTHONPATH        if already set, the discovered binding root is *prepended*,
#                     not replaced, so callers can layer paths.
#   STINKY_BUILD_DIR  if set, skip glob discovery and use this directory
#                     (must contain ``stinkytofu/__init__.py`` and
#                     ``rocisa/__init__.py``).
#
# Exit codes
# ----------
#   0    tests passed (or all skipped)
#   1    test failures
#   2    binding directory could not be found / validated
#
################################################################################

set -euo pipefail

usage() {
    sed -n '4,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------------------
# Find ``tensilelite/`` ancestor of this script (max 6 levels up).
# ----------------------------------------------------------------------------
tensilelite=""
cur="${SCRIPT_DIR}"
for _ in 1 2 3 4 5 6; do
    if [[ "$(basename "${cur}")" == "tensilelite" ]]; then
        tensilelite="${cur}"
        break
    fi
    parent="$(dirname "${cur}")"
    if [[ "${parent}" == "${cur}" ]]; then
        break
    fi
    cur="${parent}"
done

if [[ -z "${tensilelite}" ]]; then
    echo "ERROR: could not locate a 'tensilelite' ancestor of ${SCRIPT_DIR}" >&2
    exit 2
fi

# ----------------------------------------------------------------------------
# Locate the build's binding root (a directory containing both stinkytofu
# and rocisa Python packages). Override via ``STINKY_BUILD_DIR``.
# ----------------------------------------------------------------------------
binding_root=""
if [[ -n "${STINKY_BUILD_DIR:-}" ]]; then
    binding_root="${STINKY_BUILD_DIR}"
else
    shopt -s nullglob
    candidates=(
        "${tensilelite}"/*build*/tensilelite/rocisa
        "${tensilelite}"/build*/tensilelite/rocisa
    )
    shopt -u nullglob
    for cand in "${candidates[@]}"; do
        if [[ -f "${cand}/stinkytofu/__init__.py" \
           && -f "${cand}/rocisa/__init__.py" ]]; then
            binding_root="${cand}"
            break
        fi
    done
fi

if [[ -z "${binding_root}" ]]; then
    echo "ERROR: no <build>/tensilelite/rocisa with stinkytofu+rocisa was" >&2
    echo "       found under ${tensilelite}." >&2
    echo "       Build the bindings first, or point STINKY_BUILD_DIR at one." >&2
    exit 2
fi

if [[ ! -f "${binding_root}/stinkytofu/__init__.py" \
   || ! -f "${binding_root}/rocisa/__init__.py" ]]; then
    echo "ERROR: STINKY_BUILD_DIR=${binding_root} does not look like a binding root" >&2
    echo "       (missing stinkytofu/__init__.py or rocisa/__init__.py)." >&2
    exit 2
fi

# ----------------------------------------------------------------------------
# Export PYTHONPATH for parent unittest runner and inherited subprocesses.
# ``tensilelite`` is needed for ``import rocisa_stinkytofu_adaptor``;
# ``binding_root`` for ``import stinkytofu`` / ``import rocisa``.
# ----------------------------------------------------------------------------
new_paths="${binding_root}:${tensilelite}"
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${new_paths}:${PYTHONPATH}"
else
    export PYTHONPATH="${new_paths}"
fi

cd "${SCRIPT_DIR}"

# ----------------------------------------------------------------------------
# Dispatch:
#   - no args, or first arg starts with '-'  -> unittest discover
#   - first arg looks like a test file       -> run that file directly
# ----------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
    exec python3 -m unittest discover -s .
fi

first="$1"
case "${first}" in
    -*)
        # leading flag -> treat the whole arg list as flags to discover
        exec python3 -m unittest discover -s . "$@"
        ;;
    *)
        shift
        first="${first%.py}"
        if [[ ! -f "${first}.py" ]]; then
            echo "ERROR: no such test file '${first}.py' in ${SCRIPT_DIR}" >&2
            echo "Available:" >&2
            for f in "${SCRIPT_DIR}"/test_*.py; do
                echo "  $(basename "${f%.py}")" >&2
            done
            exit 2
        fi
        exec python3 "${first}.py" "$@"
        ;;
esac
