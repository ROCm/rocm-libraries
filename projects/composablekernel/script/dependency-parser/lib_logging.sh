#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Shared logging helper for the smart-build scripts.
#
# start_tee_log <logfile>
#   Stream the calling script's stdout+stderr to <logfile> as well as the
#   console, for CI artifact archiving.
#
#   A backgrounded tee draining a FIFO (whose PID is waited on at exit) is used
#   instead of `exec > >(tee)` so the log is fully flushed before the script
#   exits - the bare process-substitution form does not wait for tee and can
#   drop the tail (including the final verdict/pass-fail banner).
#
#   When _SMART_BUILD_NESTED is set, setup is skipped: the parent already tees a
#   combined log, so the child's output flows into it in order (and the child's
#   own log file is intentionally not produced).
#
#   Must be called from the top level of a script (not a subshell): it redirects
#   the current shell's fds and installs an EXIT trap. It replaces any existing
#   EXIT trap, so do not use it in scripts that need their own EXIT handler.
start_tee_log() {
    local logfile="$1"
    [ -n "${_SMART_BUILD_NESTED:-}" ] && return 0

    local fifo
    fifo="$(mktemp -u)"
    mkfifo "${fifo}"
    tee "${logfile}" < "${fifo}" &
    local tee_pid=$!
    exec > "${fifo}" 2>&1
    rm -f "${fifo}"
    # Bake the tee PID into the trap now (the local goes out of scope at exit).
    trap '_rc=$?; exec 1>&- 2>&-; wait '"${tee_pid}"' 2>/dev/null || true; exit ${_rc}' EXIT
}
