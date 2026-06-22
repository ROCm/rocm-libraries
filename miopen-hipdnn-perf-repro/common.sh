#!/usr/bin/env bash
# Shared path-discovery helpers for the miopen-hipdnn-perf-repro scripts.
#
# Portable by design: no hard-coded build-directory names and no assumption
# about where this folder lives. The only requirement is that the build
# artifacts can be found — either because this folder sits inside a
# rocm-libraries checkout that has been built, or because the relevant paths are
# provided via environment variables (see below).
#
# Environment overrides (all optional):
#   ROCM_LIBRARIES_ROOT   - path to the rocm-libraries checkout (else: git toplevel)
#   ROCM_PATH             - ROCm install (default /opt/rocm)
#   HIPCC                 - hipcc binary (default $ROCM_PATH/bin/hipcc)
#   HIPDNN_BACKEND_INCLUDE- dir containing hipdnn_backend.h
#   HIPDNN_EXPORT_HEADER  - path to the generated hipdnn_backend_export.h
#   HIPDNN_BACKEND_LIB    - path to libhipdnn_backend.so
#   HIPDNN_PLUGIN_LIB     - path to libmiopen_plugin.so
#   MIOPEN_LIB            - path to libMIOpen.so
#   MIOPEN_DRIVER         - path to the MIOpenDriver binary

# Resolve the rocm-libraries repo root. Arg 1: a directory to search from
# (typically the script's own dir, so resolution is independent of the CWD).
repro_repo_root() {
    if [[ -n "${ROCM_LIBRARIES_ROOT:-}" ]]; then
        if [[ -d "$ROCM_LIBRARIES_ROOT" ]]; then
            ( cd "$ROCM_LIBRARIES_ROOT" && pwd )
            return 0
        fi
        echo "ERROR: ROCM_LIBRARIES_ROOT='$ROCM_LIBRARIES_ROOT' is not a directory" >&2
        return 1
    fi
    local d
    if d="$(git -C "$1" rev-parse --show-toplevel 2>/dev/null)"; then
        printf '%s\n' "$d"
        return 0
    fi
    echo "ERROR: could not locate the rocm-libraries repo root." >&2
    echo "       Run from inside the checkout, or set ROCM_LIBRARIES_ROOT." >&2
    return 1
}

# Print the newest file or symlink named <name> anywhere under <root>.
# Prints nothing (and returns nonzero) if there is no match.
repro_find_newest() {  # <root> <name>
    local hit
    hit="$(find "$1" -path '*/.git' -prune -o \
                \( -type f -o -type l \) -name "$2" \
                -printf '%T@\t%p\n' 2>/dev/null | sort -rn | head -n1 | cut -f2-)"
    [[ -n "$hit" ]] || return 1
    printf '%s\n' "$hit"
}

# Resolve one artifact: use the override env var if set, else auto-discover the
# newest match by name. On failure, print a helpful error naming the override.
#   repro_resolve <repo_root> <filename> <OVERRIDE_ENV_VAR_NAME>
repro_resolve() {  # <root> <name> <override_var>
    local root="$1" name="$2" var="$3"
    local override="${!var:-}"
    if [[ -n "$override" ]]; then
        if [[ -e "$override" ]]; then
            printf '%s\n' "$override"
            return 0
        fi
        echo "ERROR: $var='$override' does not exist" >&2
        return 1
    fi
    local hit
    if hit="$(repro_find_newest "$root" "$name")"; then
        printf '%s\n' "$hit"
        return 0
    fi
    echo "ERROR: could not find '$name' under $root" >&2
    echo "       Build rocm-libraries (hipDNN + miopen-provider + MIOpen) first," >&2
    echo "       or set $var to its path." >&2
    return 1
}
