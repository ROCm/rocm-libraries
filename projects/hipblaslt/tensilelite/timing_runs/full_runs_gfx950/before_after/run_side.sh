#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# One side ("before" or "after") of a before/after timing of the common tests. Start one instance
# per side, at the same time, each with its own source tree, client, cores and GPU, so that both
# sides see the same host load. Each side runs:
#   1. split/run_split.sh: the CI-like build phase then run phase (uninstrumented, -n 4), and
#      records the configs whose run passed (<side>/split_run_passed.txt);
#   2. waits until every side in SIDES has finished step 1, then writes the shared config list
#      (<side>/configs.txt): the union of the configs that passed in any side's split run phase;
#   3. per_config/run_per_config_build.sh over that list, instrumented, caches cleared once before
#      the first config, artifacts kept;
#   4. per_config/run_per_config.sh on those artifacts, instrumented, caches not cleared.
# Compare the two sides with before_after/compare_aggregates.py.
# Usage: bash run_side.sh <side>        (<side> is one of SIDES)
# Env:   ROOT        work directory shared by the sides (required); <side>'s output goes to
#                    $ROOT/<side>/
#        TL, CLIENT  this side's TensileLite tree and prebuilt tensilelite-client (required)
#        SIDES       side names, default "before after"
#        GPU_TARGET, CORES, GPU, ANALYZE_DIR, DIST, CLEAR_RUN_CACHE: see ../measure_env.sh and
#                    ../split/run_split.sh. Give each side its own CORES and GPU.
set -u
S=${1:?usage: run_side.sh <side>}
: "${ROOT:?set ROOT}" "${TL:?set TL}" "${CLIENT:?set CLIENT}"
SIDES=${SIDES:-before after}
case " $SIDES " in
  *" $S "*) ;;
  *) echo "side '$S' is not in SIDES='$SIDES'" >&2; exit 1 ;;
esac
SCRIPTS=$(dirname "$(dirname "$(realpath -- "${BASH_SOURCE[0]}")")")
export TL CLIENT
R=$ROOT/$S
mkdir -p "$R"
export TMPDIR=$R/tmp
mkdir -p "$TMPDIR"
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/driver.log"; }
# Host conditions at each step, for judging how comparable the two sides are on a shared host.
snap() {
  {
    echo "== $1 $(date -u +%FT%TZ)"
    uptime
    rocm-smi -d "${GPU:-0}" --showuse --showmemuse --showpids 2>/dev/null | grep -Ev '^=+$|^$'
  } >> "$R/conditions.log"
}

# 1. CI-like split: wall time of each phase and the configs that pass
if ! grep -qs ALLDONE "$R/split/times.txt"; then
  snap split_start; log split start
  OUT=$R/split ART=$R/split_art TENSILE_CACHE=$R/cache_split/tensile \
    TENSILELITE_CACHE=$R/cache_split/tensilelite bash "$SCRIPTS/split/run_split.sh"
  snap split_end; log split done
fi
python - "$R/split/run.xml" > "$R/split_run_passed.txt" <<'EOF'
import re
import sys
import xml.etree.ElementTree as ET

for tc in ET.parse(sys.argv[1]).iter("testcase"):
    m = re.search(r"test_config_run\[(.*)\]", tc.get("name", ""))
    if m and not any(c.tag in ("failure", "error", "skipped") for c in tc):
        print(m.group(1))
EOF
touch "$R/split.done"

# 2. shared config list: union of the configs whose run passed in any side's split run phase
for s in $SIDES; do
  while [ ! -f "$ROOT/$s/split.done" ]; do sleep 30; done
done
for s in $SIDES; do cat "$ROOT/$s/split_run_passed.txt"; done | sort -u > "$R/configs.txt"
log "configs: $(wc -l < "$R/configs.txt")"

# 3. per-config build, instrumented; caches cleared once at the start, artifacts kept
export ART=$R/art TENSILE_CACHE=$R/cache/tensile TENSILELITE_CACHE=$R/cache/tensilelite
snap build_start; log build start
OUT=$R/build SCRATCH=$R/scratch CLEAR_CACHE=once KEEP_ARTIFACTS=1 \
  bash "$SCRIPTS/per_config/run_per_config_build.sh" "$R/configs.txt"
snap build_end; log build done

# 4. per-config run on those artifacts, caches not cleared
snap run_start; log run start
OUT=$R/run CLEAR_CACHE=never bash "$SCRIPTS/per_config/run_per_config.sh" "$R/configs.txt"
snap run_end; log run done
