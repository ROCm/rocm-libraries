#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Runbook step 1c: prove the three things stages 8 and 9 need, ON THE MACHINE THAT
# WILL RUN THEM. Exits 0 only if all three hold.
#
#   1. a device of the requested arch is present
#   2. this is the host you think it is, and the install tree is visible
#   3. the sweep root is writable FROM HERE
#
# Run it wherever the work will run -- directly on a workstation, or as the payload
# of a one-shot allocation on a scheduled cluster:
#
#   srun -p <partition> -A <account> --gpus=1 device_probe.sh gfx950 /shared/sweeps
#   ssh <host> 'bash -s' -- < device_probe.sh gfx950 /shared/sweeps
#
# A script, not a snippet, because the runbook's worked example demonstrated these
# commands on a LOGIN node -- which cannot satisfy the gate it precedes, having no
# device. Submitting this file is the gate.
set -uo pipefail

arch=${1:-${ARCH:-}}
sweep_root=${2:-${SWEEP_ROOT:-}}
install_tree=${3:-${INSTALL:-}}

if [ -z "$arch" ]; then
    echo "usage: $(basename "$0") <gfxNNN> [sweep_root] [install_tree]" >&2
    echo "       (or set ARCH / SWEEP_ROOT / INSTALL)" >&2
    exit 2
fi

fail=0
note() { printf '%-7s %s\n' "$1" "$2"; }

echo "host: $(hostname)"

# 1. A device of the requested arch.
if ! command -v rocminfo >/dev/null 2>&1; then
    note FAIL "rocminfo not on PATH -- cannot confirm a device from here"
    fail=1
else
    # Capture status separately: a rocminfo that prints a plausible agent line and
    # then FAILS (no KFD node, permission denied on /dev/kfd) was reported as a
    # present device, because only its stdout was inspected.
    rocminfo_out=$(rocminfo 2>/dev/null); rocminfo_status=$?
    found=$(printf '%s' "$rocminfo_out" | grep -oE 'gfx[0-9a-f]+' | sort -u | tr '\n' ' ')
    if [ "$rocminfo_status" -ne 0 ]; then
        note FAIL "rocminfo exited $rocminfo_status -- the runtime is not healthy here, whatever it printed"
        fail=1
    elif [ -z "$found" ]; then
        note FAIL "rocminfo lists no GPU agent (a login node does this)"
        fail=1
    # -qxF: whole token, FIXED string, one candidate per line. `grep -qw "$arch"`
    # treated the argument as a regex, so `gfx9.a` or `gfx.*` "matched" gfx90a and
    # the probe echoed the pattern back as the discovered device -- the arch gate
    # passing on hardware it was meant to reject.
    elif ! printf '%s\n' $found | grep -qxF "$arch"; then
        note FAIL "wanted $arch, found: $found -- packs arch-prune, so no other GPU will do"
        fail=1
    else
        note OK "device $arch present"
    fi
fi

# 2. The install tree is visible from HERE.
if [ -n "$install_tree" ]; then
    if [ -d "$install_tree" ]; then
        note OK "install tree visible: $install_tree"
    else
        note FAIL "install tree not visible from this host: $install_tree"
        fail=1
    fi
else
    note SKIP "no install tree given (pass one once step 7 has produced it)"
fi

# 3. The sweep root is writable from HERE, not merely from your shell.
if [ -n "$sweep_root" ]; then
    # PROBE, do not create. mkdir -p made a typo'd path pass by bringing it into
    # existence -- the one thing a probe must never do is manufacture the
    # condition it reports on. An absent root is a FAIL the human resolves.
    if [ ! -d "$sweep_root" ]; then
        note FAIL "sweep root does not exist on this host: $sweep_root (create it deliberately, or fix the path)"
        fail=1
    else
        probe=$sweep_root/.device_probe.$$
        if touch "$probe" 2>/dev/null; then
            rm -f "$probe"
            note OK "sweep root writable: $sweep_root"
        else
            note FAIL "sweep root NOT writable from this host: $sweep_root"
            fail=1
        fi
    fi
else
    note FAIL "no sweep root given -- stages 8 and 9 need one the device machine can write"
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "step 1c NOT satisfied on $(hostname) -- resolve before relying on stage 8"
    exit 1
fi
echo "step 1c satisfied on $(hostname)"
