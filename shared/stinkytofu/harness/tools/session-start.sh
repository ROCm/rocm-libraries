#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Claude Code SessionStart hook: print the harness routers (and the database index, if set) so the
# agent has them from the first turn. Point your own hook at this file; see ../SETUP.md.
# Prints nothing when the session is not inside the clone this script belongs to.

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mine=$(git -C "$here" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || exit 0
top=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cur=$(git -C "$top" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || exit 0
[ "$mine" = "$cur" ] || exit 0

for router in projects/hipblaslt/harness/HARNESS.md shared/stinkytofu/harness/HARNESS.md; do
  [ -f "$top/$router" ] && { echo "=== $router ==="; cat "$top/$router"; }
done

if [ -n "${HARNESS_DB:-}" ] && [ "$HARNESS_DB" != none ] && [ -f "$HARNESS_DB/index.md" ]; then
  echo "=== harness database index ==="
  cat "$HARNESS_DB/index.md"
fi
exit 0
