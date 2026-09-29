#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# ALMIOPEN-2812 POC driver: synthetic trees, the DescriptorSet dump, timing and the gated
# test counts, for one build of one commit. Linux, inside the ROCm container the build was
# configured in. Paths come from the environment:
#   POC_WT     rocm-libraries worktree          POC_BUILD  its gfx950 ingestor build dir
#   POC_OUT    results root (per-label subdirs)  POC_TREES  scratch for synthetic trees (tmpfs)
#   POC_PY     python with msgpack/zstandard (for pytest and kpack)
#
#   poc.sh trees                 regenerate n0/n840/n20000/n100000 from the build's packed tree
#   poc.sh dump  <label>         canonical dump + sha256 of every tree the invariant covers
#   poc.sh time  <label> [sizes] best-of-N timing rows (default: n0 n840 n20000 n100000)
#   poc.sh tests <label>         gtest (filtered) and pytest counts
set -uo pipefail

: "${POC_WT:?}" "${POC_BUILD:?}" "${POC_OUT:?}" "${POC_TREES:?}" "${POC_PY:?}"
HERE="$POC_WT/projects/hipdnn/tools/ingestor_load_poc"
ENG="$POC_BUILD/lib/hipdnn_plugins/engines"
PLUGIN="$ENG/libhip_kernel_provider.so"
PACKED="$ENG/arch_content/hip-kernel-provider"
KDP_REL=gfx950/rocKE/gfx950_attention_dense/gfx950_attention_dense.kdp.json
FILTER='*Descriptor*:*Ingestor*:*Kpack*'

tool() { find "$POC_BUILD" -name "$1" -type f -perm -u+x -print -quit; }

cmd_trees() {
    rm -rf "$POC_TREES"; mkdir -p "$POC_TREES/n0"
    for n in 840 20000 100000; do
        "$POC_PY" "$HERE/gen_synthetic_kdp.py" "$PACKED" "$n" "$POC_TREES/n$n" || return 1
    done
    # The generator must reproduce the packer at N=840, or the scale-ups measure a
    # different format than the one shipped.
    cmp "$PACKED/$KDP_REL" "$POC_TREES/n840/$KDP_REL" || { echo "FAIL: n840 != packer output"; return 1; }
    echo "n840 KDP byte-identical to the packer's"
    du -sh "$POC_TREES"/*
}

dump_roots() {
    local k="$POC_WT/dnn-providers/hip-kernel-provider"
    echo "shipped $PACKED"
    echo "test_arch_content $ENG/test_arch_content"
    echo "n840 $POC_TREES/n840"
    echo "n20000 $POC_TREES/n20000"
    for d in unit shared integration archive_fixture; do
        echo "src_$d $k/src/engines/kernel_ingestor_engine/test_descriptors/$d"
    done
    echo "src_descriptors $k/src/engines/kernel_ingestor_engine/descriptors"
    echo "pkg_fixtures $k/descriptor-packaging/tests/fixtures"
    echo "pkg_examples $k/descriptor-packaging/examples/descriptors"
    echo "ig_fixtures $POC_WT/projects/hipdnn/tools/IngestorGenerator/tests/fixtures"
}

cmd_dump() {
    local out="$POC_OUT/$1/dump" dumper
    dumper=$(tool hipdnn_poc_dump_descriptor_sets)
    [ -n "$dumper" ] || { echo "FAIL: dump tool not built"; return 1; }
    rm -rf "$out"; mkdir -p "$out"
    while read -r name root; do
        [ -d "$root" ] || { echo "FAIL: missing tree $name at $root"; return 1; }
        HIPDNN_LOG_LEVEL=off "$dumper" "$root" > "$out/$name.jsonl" || { echo "FAIL: dump $name"; return 1; }
        printf '%s  %s  %s\n' "$(sha256sum < "$out/$name.jsonl" | cut -c1-64)" "$name" "$(tail -1 "$out/$name.jsonl")"
    done < <(dump_roots) | tee "$out/SHA256SUMS"
}

cmd_time() {
    local label="$1"; shift
    local sizes="${*:-n0 n840 n20000 n100000}" out="$POC_OUT/$label/timing" timer reps
    timer=$(tool hipdnn_poc_plugin_load_timing)
    [ -n "$timer" ] || { echo "FAIL: timing tool not built"; return 1; }
    mkdir -p "$out"; : > "$out/raw.jsonl"
    echo "host $(hostname) nproc $(nproc) load $(cut -d' ' -f1-3 /proc/loadavg) $(date -Is)" | tee "$out/host.txt"
    for tree in $sizes; do
        [ "$tree" = n100000 ] && reps=2 || reps=5
        for rep in $(seq "$reps"); do
            line=$(HIPDNN_LOG_LEVEL=off HIPDNN_DESCRIPTOR_DIR="$POC_TREES/$tree" "$timer" "$PLUGIN") \
                || { echo "FAIL: $tree rep $rep"; return 1; }
            echo "{\"tree\": \"$tree\", \"rep\": $rep, ${line#\{}" >> "$out/raw.jsonl"
        done
    done
    echo "load after $(cut -d' ' -f1-3 /proc/loadavg)" >> "$out/host.txt"
    "$POC_PY" "$HERE/summarize_timing.py" "$out/raw.jsonl" > "$out/rows.json"
}

# The root build dir registers no tests unless ROCM_LIBS_ENABLE_ROOT_CTEST was on at its
# first configure, so each entry is looked up in its own project's ctest dir. The provider
# binary is registered once per tier; the quick tier supplies the environment and the
# filter replaces its selection.
cmd_tests() {
    local out="$POC_OUT/$1/tests" hkp="$POC_BUILD/dnn-providers/hip-kernel-provider"
    mkdir -p "$out"; : > "$out/counts.jsonl"
    local runs=(
        "$POC_BUILD/projects/hipdnn|hipdnn_plugin_sdk_tests|$FILTER"
        "$hkp|hip_kernel_provider_tests_quick_suite|$FILTER"
        "$hkp|hip-kernel-provider-hkp-pack-quick|"
        "$hkp|hip-kernel-provider-hkp-pack|"
    )
    local run dir name filter
    for run in "${runs[@]}"; do
        IFS='|' read -r dir name filter <<< "$run"
        "$POC_PY" "$HERE/run_gtest.py" "$dir" "$name" "$filter" "$out/$name.log" | tee -a "$out/counts.jsonl"
    done
    local g="$POC_WT/projects/hipdnn/tools/IngestorGenerator"
    (cd "$g" && "$POC_PY" -m pytest -q -p no:cacheprovider tests > "$out/pytest_ig.log" 2>&1)
    echo "{\"pytest\": \"IngestorGenerator\", \"rc\": $?, \"summary\": \"$(tail -1 "$out/pytest_ig.log")\"}" | tee -a "$out/counts.jsonl"
}

case "${1:-}" in
    trees) cmd_trees ;;
    dump) cmd_dump "${2:?label}" ;;
    time) shift; cmd_time "$@" ;;
    tests) cmd_tests "${2:?label}" ;;
    *) sed -n '5,16p' "$0"; exit 2 ;;
esac
