#!/bin/bash
# usage: mk_timing_ini.sh NAME LOGDIR
# Writes logs/t-NAME/ClientParameters.ini: LOGDIR's ini with the baseline timing settings
# (100 warmups, 100 enqueues, 4096x4096x65536) so alt_timing.py compares like with like.
set -e
BENCH="$(cd "$(dirname "$0")" && pwd)"
name="$1"; src="$2"
[[ "$src" = /* ]] || src="$BENCH/$src"
ini=$(find "$src" -name ClientParameters.ini | head -1)
dst="$BENCH/logs/t-$name"
mkdir -p "$dst"
sed -e "s/^num-warmups=.*/num-warmups=100/" -e "s/^num-enqueues-per-sync=.*/num-enqueues-per-sync=100/" \
    -e "s/^problem-size=.*/problem-size=4096,4096,1,65536/" "$ini" > "$dst/ClientParameters.ini"
echo "logs/t-$name"
