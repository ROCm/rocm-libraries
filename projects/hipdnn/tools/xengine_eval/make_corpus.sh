#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Generate evaluation graphs with hipdnn_corpus_gen for the operations where several engines
# can compete, one output subdirectory per operation: <out dir>/<op>/graphs/*.fb.
#
# Usage: make_corpus.sh <corpus_gen binary> <out dir> <count per op> <seed>
#
# The arch is whatever GPU the engines are asked on (corpus_gen has no arch flag); run this on a
# node of the arch under evaluation. Environment (all optional):
#   CORPUS_GEN_ENGINE        engine every problem is offered to (--engine-name), e.g. MIOPEN_ENGINE.
#                            Unset: --without-engine, i.e. every problem the operation declarations
#                            express (no GPU needed). That is a superset of what any engine serves;
#                            the harness reports engine applicability per graph and the report only
#                            counts graphs where >= 2 engines can run.
#   CORPUS_GEN_ALSO_ENGINES  space-separated further engines that must also serve every problem
#                            (--also-engine-name each); needs CORPUS_GEN_ENGINE.
#   CORPUS_GEN_PLUGIN_DIRS   colon-separated engine plugin dirs (--plugin-dir each); needed with
#                            CORPUS_GEN_ENGINE.
#   CORPUS_GEN_OPS           space-separated operations to generate instead of the default list.
# Feed <out dir> to: xengine_eval.py run --graphs <out dir> ...
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <corpus_gen binary> <out dir> <count per op> <seed>" >&2
    exit 64
fi
bin=$1
out=$2
count=$3
seed=$4

if [[ ! -x $bin ]]; then
    echo "error: corpus_gen binary '$bin' is not executable" >&2
    exit 1
fi
[[ $count =~ ^[0-9]+$ && $count -gt 0 ]] || { echo "error: count must be a positive integer, got '$count'" >&2; exit 64; }
[[ $seed =~ ^[0-9]+$ ]] || { echo "error: seed must be a non-negative integer, got '$seed'" >&2; exit 64; }

# Operations where engines compete (names are the *.opmeta.json "operation" values).
# CORPUS_GEN_OPS (space-separated) generates only those, e.g. "sdpa_fwd sdpa_bwd".
ops=(
    conv_fwd conv_bwd_data conv_bwd_weights
    sdpa_fwd sdpa_bwd
    batchnorm_fwd_training batchnorm_inference
    layernorm_fwd rmsnorm_fwd
)
if [[ -n ${CORPUS_GEN_OPS:-} ]]; then
    read -r -a ops <<<"$CORPUS_GEN_OPS"
fi

engine_args=()
if [[ -n ${CORPUS_GEN_ENGINE:-} ]]; then
    engine_args+=(--engine-name "$CORPUS_GEN_ENGINE")
    for e in ${CORPUS_GEN_ALSO_ENGINES:-}; do
        engine_args+=(--also-engine-name "$e")
    done
    if [[ -n ${CORPUS_GEN_PLUGIN_DIRS:-} ]]; then
        IFS=: read -r -a plugin_dirs <<<"$CORPUS_GEN_PLUGIN_DIRS"
        for d in "${plugin_dirs[@]}"; do
            engine_args+=(--plugin-dir "$d")
        done
    fi
else
    if [[ -n ${CORPUS_GEN_ALSO_ENGINES:-} ]]; then
        echo "error: CORPUS_GEN_ALSO_ENGINES needs CORPUS_GEN_ENGINE" >&2
        exit 64
    fi
    engine_args+=(--without-engine)
fi

mkdir -p "$out"
counts_file="$out/counts.tsv"
printf 'operation\tgraphs\tstatus\n' >"$counts_file"
total=0

for op in "${ops[@]}"; do
    dir="$out/$op"
    log="$out/$op.log"
    rc=0
    "$bin" --operation "$op" --output "$dir" --count "$count" --seed "$seed" \
        "${engine_args[@]}" >"$log" 2>&1 || rc=$?

    if [[ $rc -eq 1 ]] && grep -q "no declaration for operation" "$log"; then
        echo "skip $op: not known to this corpus_gen build"
        printf '%s\t0\tunknown-operation\n' "$op" >>"$counts_file"
        rm -rf "$dir" "$log"
        continue
    fi

    # 0 ok; 2 empty corpus (no engine serves any proposed problem); 3 fewer than --count.
    case $rc in
        0) status=ok ;;
        2) status=empty ;;
        3) status=short ;;
        *)
            echo "error: corpus_gen failed for $op (exit $rc); see $log" >&2
            tail -n 20 "$log" >&2
            exit "$rc"
            ;;
    esac

    n=0
    if [[ -d $dir/graphs ]]; then
        n=$(find "$dir/graphs" -maxdepth 1 -name '*.fb' | wc -l)
    fi
    n=$((n + 0))
    total=$((total + n))
    printf '%s\t%d\t%s\n' "$op" "$n" "$status" >>"$counts_file"
    echo "$op: $n graphs ($status)"
done

echo "total: $total graphs in $out (counts: $counts_file)"
if [[ $total -eq 0 ]]; then
    echo "error: no graphs generated" >&2
    exit 2
fi
