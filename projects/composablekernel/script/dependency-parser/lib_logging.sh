#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Shared logging helper for the smart-build scripts.
#
# start_tee_log <logfile>
#   Stream the calling script's stdout+stderr to <logfile> as well as the
#   console, for CI artifact archiving.
#
#   A backgrounded tee draining a FIFO (whose PID is waited on at exit) flushes
#   the log fully before the script exits - including the final verdict/pass-fail
#   banner. (A bare `exec > >(tee)` is not awaited and can drop the tail.)
#
#   When _SMART_BUILD_NESTED is set, setup is skipped: the parent already tees a
#   combined log, so the child's output flows into it in order (the child's own
#   log file is then redundant).
#
#   Call from a script's top level: it redirects the current shell's fds and owns
#   the EXIT trap (replacing any existing one). Use it where the script lets this
#   helper own EXIT.
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
