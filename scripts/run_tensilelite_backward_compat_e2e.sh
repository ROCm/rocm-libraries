#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
rocm_path="${ROCM_PATH:-/opt/rocm}"
arch="${ARCH:-gfx942}"
tensile_root="$repo_root/projects/hipblaslt/tensilelite"
logic_file="${LOGIC_FILE:-$repo_root/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/aquavanjaram/gfx942/Equality/aquavanjaram_Cijk_Alik_Bljk_SB_UserArgs.yaml}"
build_root="${BUILD_ROOT:-$repo_root/build/tensilelite-backward-compat}"
run_dir="${RUN_DIR:-$build_root/run-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
jobs="${JOBS:-8}"
harness_source="$script_dir/tensilelite_backward_compat_e2e.cpp"
harness_binary="$run_dir/tensilelite_backward_compat_e2e"
generated_root="$run_dir/generated"
generated_no_compress_root="$run_dir/generated-no-compress"
adapted_library="$run_dir/packaging-adapted-library"

mkdir -p "$run_dir"
exec > >(tee "$run_dir/run.log") 2>&1

log_command()
{
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
}

run()
{
    log_command "$@"
    "$@"
}

run_capture_status()
{
    local output_file="$1"
    shift
    log_command "$@"
    set +e
    "$@" >"$output_file" 2>&1
    local status=$?
    set -e
    sed -n '1,240p' "$output_file"
    return "$status"
}

echo "run_dir=$run_dir"
echo "repo_root=$repo_root"
echo "rocm_path=$rocm_path"
echo "architecture=$arch"
echo "logic_file=$logic_file"

if [[ "$arch" != "gfx942" ]]; then
    echo "This checked-in default logic and numerical problem are for gfx942; set LOGIC_FILE and update the harness before using another architecture." >&2
    exit 2
fi

logic_relative="${logic_file#"$repo_root/"}"
run git -C "$repo_root" ls-files --error-unmatch "$logic_relative"
run git -C "$repo_root" diff --quiet -- "$logic_relative"
run git -C "$repo_root" diff --cached --quiet -- "$logic_relative"

producer_revision="$(git -C "$repo_root" rev-parse HEAD)"
logic_revision="$(git -C "$repo_root" log -1 --format=%H -- "$logic_relative")"
installed_library="$(readlink -f "$rocm_path/lib/libhipblaslt.so.1")"
installed_version="$(awk '/HIPBLASLT_VERSION_MAJOR/{major=$3} /HIPBLASLT_VERSION_MINOR/{minor=$3} /HIPBLASLT_VERSION_PATCH/{patch=$3} END{print major "." minor "." patch}' "$rocm_path/include/hipblaslt-version.h")"
installed_tweak="$(awk '/HIPBLASLT_VERSION_TWEAK/{print $3}' "$rocm_path/include/hipblaslt-version.h")"

echo "producer_revision=$producer_revision"
echo "logic_revision=$logic_revision"
echo "installed_hipblaslt_version=$installed_version"
echo "installed_hipblaslt_tweak=$installed_tweak"
echo "installed_hipblaslt_library=$installed_library"

agent_arches="$("$rocm_path/bin/rocm_agent_enumerator" | sort -u | tr '\n' ' ')"
echo "enumerated_gpu_architectures=$agent_arches"
if ! grep -qw "$arch" <<<"$agent_arches"; then
    echo "The requested architecture $arch was not enumerated." >&2
    exit 3
fi

python="${TENSILE_PYTHON:-}"
if [[ -z "$python" ]]; then
    if [[ -x "$tensile_root/.tox/py3/bin/python" ]]; then
        python="$tensile_root/.tox/py3/bin/python"
    else
        python="$(command -v python3)"
    fi
fi

rocisa_pythonpath="${ROCISA_PYTHONPATH:-}"
if [[ -z "$rocisa_pythonpath" && -d "$tensile_root/build_tmp/tensilelite/rocisa/rocisa" ]]; then
    rocisa_pythonpath="$tensile_root/build_tmp/tensilelite/rocisa"
fi

source_pythonpath="$tensile_root"
if [[ -n "$rocisa_pythonpath" ]]; then
    source_pythonpath="$rocisa_pythonpath:$source_pythonpath"
fi

echo "tensile_python=$python"
echo "source_pythonpath=$source_pythonpath"
if ! PYTHONPATH="$source_pythonpath${PYTHONPATH:+:$PYTHONPATH}" "$python" -c '
import pathlib
import Tensile
import rocisa
from rocisa.instruction import VCndMaskB16
print(f"Tensile_source={pathlib.Path(Tensile.__file__).resolve()}")
print(f"rocisa_source={pathlib.Path(rocisa.__file__).resolve()}")
print(f"rocisa_probe={VCndMaskB16.__name__}")
'; then
    echo "Current-source rocisa is missing or stale. Build it with 'cd projects/hipblaslt/tensilelite && invoke rocisa', or set ROCISA_PYTHONPATH to a current from-source rocisa package root." >&2
    exit 4
fi

run "$rocm_path/bin/hipcc" \
    -std=c++17 \
    -O2 \
    "$harness_source" \
    -I"$rocm_path/include" \
    -L"$rocm_path/lib" \
    -Wl,-rpath,"$rocm_path/lib" \
    -lhipblaslt \
    -lamdhip64 \
    -ldl \
    -o "$harness_binary"

run ldd "$harness_binary"
resolved_harness_library="$(LD_LIBRARY_PATH="$rocm_path/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$harness_binary" | awk '/libhipblaslt.so.1/{print $3; exit}')"
if [[ "$(readlink -f "$resolved_harness_library")" != "$installed_library" ]]; then
    echo "Harness resolved the wrong hipBLASLt: $resolved_harness_library" >&2
    exit 5
fi

baseline_library="$rocm_path/lib/hipblaslt/library"
echo "baseline_library=$baseline_library"
if run_capture_status "$run_dir/baseline.log" \
    env \
    "LD_LIBRARY_PATH=$rocm_path/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "HIPBLASLT_TENSILE_LIBPATH=$baseline_library" \
    HIPBLASLT_LOG_LEVEL=4 \
    TENSILE_DB=0x2B060 \
    "$harness_binary"; then
    baseline_status=0
else
    baseline_status=$?
    echo "verdict=INCONCLUSIVE_BASELINE_FAILED status=$baseline_status" | tee "$run_dir/verdict.txt"
    echo "The installed library baseline must pass before generated-library compatibility can be judged." >&2
    exit 6
fi
if ! rg -q 'PASS numerical_validation' "$run_dir/baseline.log"; then
    echo "The baseline process exited successfully without the numerical PASS marker." >&2
    exit 7
fi

logic_dir="$(dirname "$logic_file")"
logic_stem="$(basename "$logic_file" .yaml)"
export PATH="$rocm_path/bin:$rocm_path/llvm/bin:$PATH"
export PYTHONPATH="$source_pythonpath${PYTHONPATH:+:$PYTHONPATH}"
run "$python" -m Tensile.TensileCreateLibrary \
    "$logic_dir" \
    "$generated_root" \
    HIP \
    --architecture "$arch" \
    --logic-filter "$logic_stem" \
    --library-format msgpack \
    --no-enumerate \
    --jobs "$jobs" \
    --verbose 2 \
    --cxx-compiler "$rocm_path/bin/amdclang++" \
    --c-compiler "$rocm_path/bin/amdclang" \
    --assembler "$rocm_path/bin/amdclang++" \
    --offload-bundler "$rocm_path/llvm/bin/clang-offload-bundler"

generated_library="$generated_root/library/$arch"
if [[ ! -d "$generated_library" ]]; then
    echo "Expected generated library directory not found: $generated_library" >&2
    find "$generated_root" -maxdepth 4 -type f -print
    exit 8
fi

echo "generated_library=$generated_library"
run find "$generated_library" -maxdepth 1 -type f -printf '%f\t%s bytes\n'
run sha256sum "$generated_library"/*
run "$python" -c '
import pathlib
import sys
import zlib

import msgpack

root = pathlib.Path(sys.argv[1])
solution = None
source = None
for candidate in sorted(root.glob("*.dat.zlib")):
    payload = msgpack.unpackb(zlib.decompress(candidate.read_bytes()), raw=False, strict_map_key=False)
    if isinstance(payload, dict) and payload.get("solutions"):
        solution = payload["solutions"][0]
        source = candidate
        break
if solution is None:
    raise SystemExit("No solution-bearing MessagePack file found")

def predicates(node):
    if not isinstance(node, dict):
        return []
    result = [node.get("type")] if node.get("type") else []
    value = node.get("value")
    if isinstance(value, list):
        for child in value:
            result.extend(predicates(child))
    return result

def find_predicate(node, wanted):
    if not isinstance(node, dict):
        return None
    if node.get("type") == wanted:
        return node
    value = node.get("value")
    if isinstance(value, list):
        for child in value:
            found = find_predicate(child, wanted)
            if found is not None:
                return found
    return None

problem_type = solution["problemType"]
types_equal = find_predicate(solution["problemPredicate"], "TypesEqual")
print(f"metadata_source={source.name}")
print("metadata_compute_fields=" + ",".join(
    key for key in ("computeInputType", "computeInputTypeA", "computeInputTypeB")
    if key in problem_type))
print("metadata_types_equal_length={}".format(len(types_equal["value"]) if types_equal else "missing"))
print("metadata_problem_predicates=" + ",".join(sorted(set(predicates(solution["problemPredicate"])))))
print("metadata_task_predicates=" + ",".join(sorted(set(predicates(solution["taskPredicate"])))))
print("metadata_kernargs_version={}".format(solution["internalArgsSupport"]["version"]))
' "$generated_library"

generated_solution_code_object="$(find "$generated_library" -maxdepth 1 -type f -name '*.co' -print -quit)"
if [[ -z "$generated_solution_code_object" ]]; then
    echo "No generated solution code object was found." >&2
    exit 9
fi

raw_status=0
if run_capture_status "$run_dir/generated-raw.log" \
    env \
    "LD_LIBRARY_PATH=$rocm_path/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "HIPBLASLT_TENSILE_LIBPATH=$generated_library" \
    HIPBLASLT_LOG_LEVEL=4 \
    TENSILE_DB=0x2B060 \
    "$harness_binary"; then
    raw_status=0
else
    raw_status=$?
fi

if [[ "$raw_status" -eq 0 ]] && rg -q 'PASS numerical_validation' "$run_dir/generated-raw.log"; then
    if ! rg -Fq "loaded code object $generated_solution_code_object" "$run_dir/generated-raw.log"; then
        echo "The numerical run passed, but the log did not prove that a generated code object was loaded." >&2
        exit 10
    fi
    echo "verdict=COMPATIBLE_AS_IS" | tee "$run_dir/verdict.txt"
    exit 0
fi

run "$python" -m Tensile.TensileCreateLibrary \
    "$logic_dir" \
    "$generated_no_compress_root" \
    HIP \
    --architecture "$arch" \
    --logic-filter "$logic_stem" \
    --library-format msgpack \
    --no-compress \
    --no-enumerate \
    --jobs "$jobs" \
    --verbose 2 \
    --cxx-compiler "$rocm_path/bin/amdclang++" \
    --c-compiler "$rocm_path/bin/amdclang" \
    --assembler "$rocm_path/bin/amdclang++" \
    --offload-bundler "$rocm_path/llvm/bin/clang-offload-bundler"

generated_no_compress_library="$generated_no_compress_root/library/$arch"
if [[ ! -d "$generated_no_compress_library" ]]; then
    echo "Expected --no-compress library directory not found: $generated_no_compress_library" >&2
    exit 11
fi

echo "generated_no_compress_library=$generated_no_compress_library"
run find "$generated_no_compress_library" -maxdepth 1 -type f -printf '%f\t%s bytes\n'
run sha256sum "$generated_no_compress_library"/*

no_compress_solution_code_object="$(find "$generated_no_compress_library" -maxdepth 1 -type f -name '*.co' -print -quit)"
no_compress_status=0
if run_capture_status "$run_dir/generated-no-compress.log" \
    env \
    "LD_LIBRARY_PATH=$rocm_path/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "HIPBLASLT_TENSILE_LIBPATH=$generated_no_compress_library" \
    HIPBLASLT_LOG_LEVEL=4 \
    TENSILE_DB=0x2B060 \
    "$harness_binary"; then
    no_compress_status=0
else
    no_compress_status=$?
fi

if [[ "$no_compress_status" -eq 0 ]] \
    && rg -q 'PASS numerical_validation' "$run_dir/generated-no-compress.log"; then
    if ! rg -Fq "loaded code object $no_compress_solution_code_object" "$run_dir/generated-no-compress.log"; then
        echo "The --no-compress run passed, but no generated solution code-object load was observed." >&2
        exit 12
    fi
    echo "verdict=COMPATIBLE_WITH_NO_COMPRESS" | tee "$run_dir/verdict.txt"
    exit 0
fi

run cp -a "$generated_library" "$adapted_library"
run "$python" -c '
import pathlib
import sys
import zlib

root = pathlib.Path(sys.argv[1])
for source in sorted(root.glob("*.zlib")):
    destination = source.with_suffix("")
    destination.write_bytes(zlib.decompress(source.read_bytes()))
    print(f"decompressed {source.name} -> {destination.name}")
' "$adapted_library"

mapping_source="$adapted_library/TensileLiteLibrary_lazy_${arch}_Mapping.dat"
mapping_destination="$adapted_library/TensileLiteLibrary_lazy_Mapping.dat"
if [[ -f "$mapping_source" ]]; then
    run cp "$mapping_source" "$mapping_destination"
fi

adapted_status=0
if run_capture_status "$run_dir/generated-packaging-adapted.log" \
    env \
    "LD_LIBRARY_PATH=$rocm_path/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "HIPBLASLT_TENSILE_LIBPATH=$adapted_library" \
    HIPBLASLT_LOG_LEVEL=4 \
    TENSILE_DB=0x2B060 \
    "$harness_binary"; then
    adapted_status=0
else
    adapted_status=$?
fi

if [[ "$adapted_status" -eq 0 ]] \
    && rg -q 'PASS numerical_validation' "$run_dir/generated-packaging-adapted.log"; then
    adapted_solution_code_object="$adapted_library/$(basename "$generated_solution_code_object")"
    if ! rg -Fq "loaded code object $adapted_solution_code_object" "$run_dir/generated-packaging-adapted.log"; then
        echo "The adapted numerical run passed, but no adapted code-object load was observed." >&2
        exit 11
    fi
    echo "verdict=PACKAGING_INCOMPATIBLE_METADATA_AND_KERNEL_COMPATIBLE" | tee "$run_dir/verdict.txt"
    exit 0
fi

{
    echo "verdict=INCOMPATIBLE"
    echo "raw_status=$raw_status"
    echo "no_compress_status=$no_compress_status"
    echo "packaging_adapted_status=$adapted_status"
} | tee "$run_dir/verdict.txt"

echo "The current producer's library did not load and run accurately through installed hipBLASLt, even after a separate packaging-only probe."
