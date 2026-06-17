"""
Convert Google Benchmark JSON files to primbench CSV format.

Reads Google Benchmark JSON output files from an input directory and converts
them to primbench CSV format, writing results to an output directory.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, cast
import re

seen_binary_search_params: Set[str] = set()

# These nested tuples generate a lookup table of scales
# that benchmark_device_histogram forgot to output
histogram_even_scales = (
    (12345,) * 4
    + (1234,) * 4
    + (5,) * 4
    + (16,)
    + (1,)
    + (1234,) * 8
    + (5,) * 4
    + (16,)
    + (1,)
    + (16,)
    + (1,)
) * 4
histogram_even_index = 0

histogram_multi_even_scales = (
    (1234,) * 4 + (5,) * 4 + (16,) + (1,) + (1234,) * 4 + (16,) + (1,) + (16,) + (1,)
) * 4
histogram_multi_even_index = 0

device_scan_deterministic_skipped_indices = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10)
device_scan_deterministic_index = 0

device_scan_skipped_indices = (8, 10)
device_scan_index = 0


def replace(params: Dict[str, Any], key: str, entries: Dict[str, str]):
    """Replaces for example an 'offset_type' its value 'int' with 'i32'."""
    for old, new in entries.items():
        if params.get(key) == old:
            params[key] = new


def transform(params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply algorithm-specific transformations to benchmark parameters."""

    # These params don't exist in hipcub
    lvl = params.get("lvl", "")
    algo = params.get("algo", "")
    name = lvl + "_" + algo

    for key in (
        "key_type",
        "value_type",
        "offset_type",
        "item_type",
        "size_type",
        "data_type",
        "flag_type",
        "input_type",
        "output_type",
    ):
        replace(
            params,
            key,
            {
                "char": "i8",
                "common::custom_type<1024,float,float>": "huge<1024,f32,f32>",
                "common::custom_type<2048,float,float>": "huge<2048,f32,f32>",
                "common::custom_type<char,double>": "custom<i8,f64>",
                "common::custom_type<char,short>": "custom<i8,i16>",
                "common::custom_type<double,double>": "custom<f64,f64>",
                "common::custom_type<float,float>": "custom<f32,f32>",
                "common::custom_type<float,int16_t>": "custom<f32,i16>",
                "common::custom_type<int,double>": "custom<i32,f64>",
                "common::custom_type<int,int>": "custom<i32,i32>",
                "common::custom_type<int64_t,double>": "custom<i64,f64>",
                "common::custom_type_copyable<char,double>": "copyable<i8,f64>",
                "common::custom_type_copyable<double,double>": "copyable<f64,f64>",
                "custom_128": "custom<i64,i64>",
                "custom_char_double": "custom<i8,f64>",
                "custom_double2": "custom<f64,f64>",
                "custom_float2": "custom<f32,f32>",
                "custom_int2": "custom<i32,i32>",
                "custom_int_double": "custom<i32,f64>",
                "custom_int_type": "custom<i32,i32>",
                "custom_longlong_double": "custom<i64,f64>",
                "custom_type<int,double>": "custom<i32,f64>",
                "double": "f64",
                "empty_type": "empty",
                "float": "f32",
                "int": "i32",
                "int16_t": "i16",
                "int32_t": "i32",
                "int64_t": "i64",
                "int8_t": "i8",
                "long long": "i64",
                "rocprim::half": "half",
                "rocprim::int128_t": "i128",
                "rocprim::uint128_t": "u128",
                "short": "i16",
                "uint32_t": "u32",
                "uint64_t": "u64",
                "uint8_t": "u8",
                "uint8_t": "u8",
                "unsigned char": "u8",
                "unsigned int": "u32",
                "unsigned long long": "u64",
            },
        )

    if name in ("device_adjacent_difference", "device_adjacent_difference_inplace"):
        if "is_left" in params:
            params["left"] = params.pop("is_left")
        params["inplace"] = algo.endswith("_inplace")

    if name == "block_radix_rank":
        params["cfg"]["method"] = params["cfg"]["method"].removeprefix(
            "rocprim::block_radix_rank_algorithm::"
        )

    if name == "block_run_length_decode":
        params["cfg"]["bs"] = params["cfg"].pop("block_size")

    # These were all in benchmark_config_dispatch
    if algo in (
        "default_stream",
        "per_thread_stream",
        "explicit_stream",
        "async_stream",
        "empty_kernel",
    ):
        params["method"] = algo

    if name == "device_adjacent_find":
        params["first_adj_pos"] = float(params["first_adj_pos"])

    # These were all in benchmark_device_batch_memcpy
    if algo in ("batch_memcpy", "batch_copy"):
        params["subalgo"] = algo

    # These were all in benchmark_device_binary_search
    if algo in ("binary_search", "lower_bound", "upper_bound"):
        params["key_type"] = params.pop("value_type")
        params["subalgo"] = algo
        params["needles_percent"] = 10

        global seen_binary_search_params
        s = str(params)
        params["sorted_needles"] = s not in seen_binary_search_params
        seen_binary_search_params.add(s)

    if name == "device_find_end":
        params["repeating"] = params.pop("value_pattern") == "repeating"

    if name == "device_find_first_of":
        params["first_occurrence"] = f"{float(params['first_occurrence']):g}"

    # These were all in benchmark_device_histogram
    if algo in (
        "histogram_even",
        "multi_histogram_even",
        "histogram_range",
        "multi_histogram_range",
    ):
        params["subalgo"] = algo.replace("histogram_", "")

    # benchmark_device_histogram forgot to output scale
    if algo == "histogram_even":
        global histogram_even_index
        params["scale"] = histogram_even_scales[histogram_even_index]
        histogram_even_index += 1
    if algo == "multi_histogram_even":
        global histogram_multi_even_index
        params["scale"] = histogram_multi_even_scales[histogram_multi_even_index]
        histogram_multi_even_index += 1

    if name == "device_memory" and params["subalgo"] == "copy":
        params["cfg"] = {"bs": 1, "ipt": 1}
        params["operation"] = "no_operation"

    if name in (
        "device_nth_element",
        "device_partial_sort_copy",
        "device_partial_sort",
    ):
        params["small_n"] = params.pop("nth") == "small"

    if algo == "partition_two_way":
        params["subalgo"] = f"two_way_{params['subalgo']}"
    if algo == "partition_three_way":
        params["subalgo"] = "three_way"

    if name == "device_run_length_encode" and "subalgo" in params:
        del params["subalgo"]

    if name == "device_search":
        params["repeating"] = params.pop("value_pattern") == "repeating"

    if algo in ("transform", "transform_pointer"):
        params["is_binary"] = params.pop("op") == "binary"

    if algo in ("read_predicate_it", "write_predicate_it", "transform_it"):
        params["subalgo"] = algo.removesuffix("_it")
        params["percent"] = params.pop("p").removeprefix("p")

    # segmented_radix_sort_keys always output "value_type: empty"
    if algo == "segmented_radix_sort" and params["value_type"] == "empty":
        del params["value_type"]

    if name == "warp_sort":
        if params["value_type"] == "empty":
            del params["value_type"]

    return params


def strip_prefixes(s: str) -> str:
    """Remove rocprim:: and common:: prefixes from string."""
    prefixes = ["rocprim::", "common::"]
    for prefix in prefixes:
        while prefix in s:
            s = s.replace(prefix, "")
    return s


def serialize(value: Any) -> str:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, Any], value)
        items: List[str] = []
        for k, v in mapping.items():
            items.append(f"{k}: {serialize(v)}")
        return "{ " + ", ".join(items) + " }"

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seq = cast(Sequence[Any], value)
        return "[ " + ", ".join(serialize(v) for v in seq) + " ]"

    if isinstance(value, bool):
        return str(value).lower()

    if value is None:
        return "null"

    return strip_prefixes(str(value))


def sort_dict_alphabetically(d: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a new dict with keys sorted alphabetically, recursively."""
    sorted_dict: Dict[str, Any] = {}
    for key in sorted(d, key=str):
        value = d[key]
        if isinstance(value, Mapping):
            sorted_dict[key] = sort_dict_alphabetically(cast(Mapping[str, Any], value))
        else:
            sorted_dict[key] = value
    return sorted_dict


def parse_benchmark_name(name: str) -> str:
    """Extract and format benchmark parameters from JSON-formatted name."""
    name = name.removesuffix("/manual_time")
    name = name.removesuffix("/iterations:100")

    params = json.loads(name)
    params = transform(params)

    # Alphabetically sort keys
    params = sort_dict_alphabetically(params)

    blacklist = {"lvl", "algo"}

    parts: List[str] = []
    for key, value in params.items():
        # Skip blacklisted keys
        if key in blacklist:
            continue

        # Skip default configs
        if key == "cfg" and value == "default_config":
            continue

        parts.append(f"{key}: {serialize(value)}")

    return ", ".join(parts)


def convert_rocrand_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for idx, bench in enumerate(data["benchmarks"]):
        name = bench["name"]  # device_kernel<lfsr113,uniform-uint>/manual_time

        if name.startswith("device_kernel<"):
            is_device_api = True
            inner = name[len("device_kernel<") :]  # lfsr113,uniform-uint>/manual_time
        elif name.startswith("device_generate<"):
            is_device_api = False
            inner = name[
                len("device_generate<") :
            ]  # lfsr113,default,uniform-uint>/manual_time
        else:
            continue

        inner = inner.split(">", 1)[0]  # lfsr113,uniform-uint
        parts = inner.split(",")  # ['lfsr113', 'uniform-uint']

        ordering = None
        if len(parts) == 2:
            assert is_device_api
            engine, distribution_raw = parts  # lfsr113, uniform-uint
        elif len(parts) == 3:
            assert not is_device_api
            engine, ordering, distribution_raw = parts  # lfsr113, default, uniform-uint
        else:
            continue

        poisson_lambda = None

        if distribution_raw.startswith("uniform-"):
            distribution_name = "uniform"
            type_raw = distribution_raw[len("uniform-") :]  # uint
        elif distribution_raw.startswith("normal-"):
            distribution_name = "normal"
            type_raw = distribution_raw[len("normal-") :]  # float
        elif distribution_raw.startswith("log-normal-"):
            distribution_name = "log_normal"
            type_raw = distribution_raw[len("log-normal-") :]  # float
        elif distribution_raw.startswith(
            "discrete-poisson("
        ) or distribution_raw.startswith("poisson("):
            distribution_name = (
                "discrete_poisson"
                if distribution_raw.startswith("discrete-poisson(")
                else "poisson"
            )
            type_raw = "uint"
            if "lambda=" in distribution_raw:
                raw = distribution_raw.split("lambda=")[1].rstrip(")")  # 10.0
                lam = float(raw)
                poisson_lambda = str(int(lam)) if lam.is_integer() else str(lam)  # 10
        elif distribution_raw == "discrete-custom":
            distribution_name = "discrete_custom"
            type_raw = "uint"
        else:
            continue

        type_map = {
            "uchar": "u8",
            "ushort": "u16",
            "uint": "u32",
            "long-long": "u64",  # It not being "ulong-long" was a bug in old benchmark
            "ullong": "u64",
            "float": "f32",
            "double": "f64",
            "half": "half",
        }

        type_name = type_map[type_raw]  # u32

        name_parts: List[str] = []

        # Device API benchmarks do not provide a config.
        # Assume 256 blocks and 256 threads, even though they may have been overriden.
        if is_device_api:
            name_parts.append("cfg: { blocks: 256, threads: 256 }")

        name_parts.append(f"distribution: {distribution_name}")
        name_parts.append(f"engine: {engine}")

        if ordering is not None:
            if "sobol" in engine:
                name_parts.append(f"ordering: quasi_default")
            else:
                name_parts.append(f"ordering: {ordering}")
        if poisson_lambda is not None:
            name_parts.append(f"poisson_lambda: {poisson_lambda}")

        name_parts.append(f"type: {type_name}")

        result_name = ", ".join(name_parts)

        bytes_per_second = bench["bytes_per_second"]
        gib_per_second = bytes_per_second / (1024.0 * 1024.0 * 1024.0)

        results.append(
            {
                "index": idx,
                "name": result_name,
                "bytes_per_second": bytes_per_second,
                "gib_per_second": gib_per_second,
                "items_per_second": bench["items_per_second"],
                "noise_timeout": 0,
                "noise_percent": 0,
            }
        )

    return results


def convert_rocprim_json(
    data: Dict[str, Any], noise_threshold: float
) -> List[Dict[str, Any]]:
    """Load Google Benchmark JSON and convert to primbench format."""
    times_seen_adjacent_find_i32 = 0
    times_seen_adjacent_find_i16 = 0

    seen_specializations: Set[str] = set()

    results: List[Dict[str, Any]] = []
    for idx, bench in enumerate(data["benchmarks"]):
        name = parse_benchmark_name(bench["name"])

        # device_adjacent_find registers i16 and i32 specializations
        # in a group of three: 0.1, 0.5, and 0.9
        # It accidentally registered this group twice, so skip that 2nd group
        if "adjacent_find" in bench["name"]:
            if name.endswith("input_type: i16"):
                times_seen_adjacent_find_i16 += 1
                if times_seen_adjacent_find_i16 > 3:
                    continue
            if name.endswith("input_type: i32"):
                times_seen_adjacent_find_i32 += 1
                if times_seen_adjacent_find_i32 > 3:
                    continue

        # These accidentally benchmarked some specializations several times
        if (
            "find_first_of" in bench["name"] or "reduce_by_key" in bench["name"]
        ) and bench["name"] in seen_specializations:
            continue

        # The only way to tell device_scan_by_key_deterministic
        # apart from device_scan_by_key is the executable/JSON file name
        # The executable is less likely to have been renamed, so use that
        if data["context"]["executable"].endswith(
            "/benchmark_device_scan_by_key_deterministic"
        ):
            # benchmark_device_scan_by_key_deterministic accidentally benchmarked
            # specializations that had Deterministic=False, so skip those
            if "key_type: i32," not in name or "max_segment_length: 1024" in name:
                continue

        if data["context"]["executable"].endswith(
            "/benchmark_device_scan_deterministic"
        ):
            global device_scan_deterministic_index
            skipped = (
                device_scan_deterministic_index
                in device_scan_deterministic_skipped_indices
            )
            device_scan_deterministic_index += 1
            if skipped:
                continue

        if data["context"]["executable"].endswith("/benchmark_device_scan"):
            global device_scan_index
            skipped = device_scan_index in device_scan_skipped_indices
            device_scan_index += 1
            if skipped:
                continue

        seen_specializations.add(bench["name"])

        bytes_per_second = bench["bytes_per_second"]
        gib_per_second = bytes_per_second / (1024.0 * 1024.0 * 1024.0)

        result: Dict[str, Any] = {
            "index": idx,
            "name": name,
            "bytes_per_second": bytes_per_second,
            "gib_per_second": gib_per_second,
            "items_per_second": bench["items_per_second"],
            "noise_timeout": 1 if (bench["cv"] * 100) > noise_threshold else 0,
            "noise_percent": bench["cv"] * 100,
        }
        results.append(result)

    # This asserts that scale was added to all even and multi_even specializations
    if "histogram_even" in data["benchmarks"][0]["name"]:
        assert len(histogram_even_scales) == histogram_even_index
        assert len(histogram_multi_even_scales) == histogram_multi_even_index

    return results


def split_top_level(s):
    parts = []
    current = []
    depth = 0

    for ch in s:
        if ch in ("<", "[", "("):
            depth += 1
        elif ch in (">", "]", ")"):
            depth -= 1

        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)

    if current:
        parts.append("".join(current))

    return parts


def fix_to_json(s):
    s = s.strip("{}")
    result = {}

    for part in split_top_level(s):
        if ":::" in part:
            key, value = part.split(":::", 1)
            value = "::" + value
        elif ":" in part:
            key, value = part.split(":", 1)
        else:
            continue

        key = key.strip()
        value = value.strip()

        if value.isdigit():
            value = int(value)

        if isinstance(value, str):
            value = re.sub(
                r"::hipcub::([A-Z0-9_]+)",
                lambda m: m.group(1).lower(),
                value
            )

            value = re.sub(
                r"hipcub::([A-Z0-9_]+)",
                lambda m: m.group(1).lower(),
                value
            )

        result[key] = value

    return result

def convert_hipcub_json(
    data: Dict[str, Any], noise_threshold: float
) -> List[Dict[str, Any]]:
    """Load Google Benchmark JSON and convert to hipcub format."""
    times_seen_adjacent_find_i32 = 0
    times_seen_adjacent_find_i16 = 0

    seen_specializations: Set[str] = set()

    # Hack for device_for
    device_for_f32_found = False
    device_for_f64_found = False

    results: List[Dict[str, Any]] = []
    for idx, bench in enumerate(data["benchmarks"]):
        name_str = bench["name"]

        is_block_adjacent_difference = "block_adjacent_difference" in name_str
        is_block_discontinuity = "block_discontinuity" in name_str
        is_block_radix_rank = "block_radix_rank" in name_str
        is_block_reduce = "block_reduce" in name_str
        is_block_scan = "block_scan" in name_str
        is_block_shuffle = "block_shuffle" in name_str
        is_device_for = "for_each" in name_str
        is_device_histogram = "device_histogram" in name_str or "device_multi_histogram" in name_str
        is_device_memory = "device_memory" in name_str
        is_device_merge_sort = "device_merge_sort" in name_str
        is_device_merge = "device_merge" in name_str and not is_device_merge_sort
        is_device_partition = "device_parition" in name_str # "partition" is misspelled in the old gbench code
        is_device_radix_sort = "device_radix_sort" in name_str
        is_device_reduce_by_key = "device_reduce_by_key" in name_str
        is_device_reduce = "device_reduce" in name_str and not is_device_reduce_by_key
        is_device_run_length_encode = "device_run_length_encode" in name_str or "run_length_encode_non_trivial_runs" in name_str
        is_device_scan = "device_inclusive_scan" in name_str or "device_exclusive_scan" in name_str
        is_device_segmented_radix_sort = "device_segmented_radix_sort_keys" in name_str or "device_segmented_radix_sort_pairs" in name_str
        is_device_segmented_reduce = "device_segmented_reduce" in name_str
        is_device_segmented_sort = "device_segmented_sort" in name_str
        is_device_select = "device_select" in name_str
        is_device_spmv = "device_spmv" in name_str
        is_warp_exchange = "warp_exchange" in name_str
        is_warp_merge_sort = "warp_merge_sort" in name_str
        is_warp_reduce = "warp_reduce" in name_str

        if is_warp_merge_sort:
            # The original gbench code had these two swapped, so we fix it here
            if "segmented_sort" in name_str:
                name_str = re.sub(r":segmented_sort", ":sort", name_str)
            else:
                name_str = re.sub(r":sort", ":segmented_sort", name_str)

            # Turn the subalgo into the segmented+pair bools
            warp_merge_is_segmented = "segmented_sort" in name_str
            warp_merge_is_pairs = "values" in name_str

        if is_warp_reduce:
            # Turn the subalgo into the segmented+pair bools
            warp_reduce_is_segmented = "segmented_reduce" in name_str

        if is_warp_exchange:
            if "warp_exchange_striped_to_blocked" in name_str:
                name_str = re.sub(r"<", "<op:striped_to_blocked_op,", name_str, count=1)
            elif "warp_exchange_blocked_to_striped" in name_str:
                name_str = re.sub(r"<", "<op:blocked_to_striped_op,", name_str, count=1)
            elif "warp_exchange_scatter_to_striped" in name_str:    
                name_str = re.sub(r"<", "<subalgo:scatter_to_striped,", name_str, count=1)
        
        if is_block_adjacent_difference:
            name_str = re.sub(r"subtract_left<", "subtract_left,", name_str, count=1)
            name_str = re.sub(r"subtract_right<", "subtract_right,", name_str, count=1)
            name_str = re.sub(r"subtract_left_partial_tile<", "subtract_left_partial_tile,", name_str, count=1)
            name_str = re.sub(r"subtract_right_partial_tile<", "subtract_right_partial_tile,", name_str, count=1)
        
        if is_block_discontinuity:
            name_str = re.sub(r"flag_heads<", "flag_heads,", name_str, count=1)
            name_str = re.sub(r"flag_tails<", "flag_tails,", name_str, count=1)
            name_str = re.sub(r"flag_heads_and_tails<", "flag_heads_and_tails,", name_str, count=1)

        if is_block_radix_rank:
            name_str = name_str.replace("kind", "sub_algorithm_name");

            name_str = name_str.replace("RadixRankAlgorithm::RADIX_RANK_BASIC", "basic")
            name_str = name_str.replace("RadixRankAlgorithm::RADIX_RANK_MATCH", "match")
            name_str = name_str.replace("RadixRankAlgorithm::RADIX_RANK_MEMOIZE", "memoize")

        if is_block_reduce:
            # Lowercase the enums
            name_str = name_str.replace("BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY", "block_reduce_raking_commutative_only")
            name_str = name_str.replace("BLOCK_REDUCE_RAKING", "block_reduce_raking")
            name_str = name_str.replace("BLOCK_REDUCE_WARP_REDUCTIONS", "block_reduce_warp_reductions")

        if is_block_scan:
            # Lowercase the enums
            name_str = name_str.replace("BLOCK_SCAN_RAKING_MEMOIZE", "block_scan_raking_memoize")
            name_str = name_str.replace("BLOCK_SCAN_RAKING", "block_scan_raking")
            name_str = name_str.replace("BLOCK_SCAN_WARP_SCANS", "block_scan_warp_scans")

        if is_device_histogram:
            # Get rid of the ()
            name_str = name_str.replace("(", "")
            name_str = name_str.replace(">.entropy_percent", ",entropy_percent")
            name_str = name_str.replace(">.bin_count", ",bin_count")
            name_str = name_str.replace(" bins)", ">")
            name_str = name_str.replace("%", "")

            subalgo = "even" if "device_histogram_even" in name_str else ""
            subalgo = "multi_even" if "device_multi_histogram_even" in name_str else subalgo
            subalgo = "range" if "device_histogram_range" in name_str else subalgo
            subalgo = "multi_range" if "device_multi_histogram_range" in name_str else subalgo

            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",subalgo:{subalgo}" + name_str[pos:] 

        if is_device_reduce_by_key or is_device_run_length_encode or is_device_partition or is_device_segmented_radix_sort or is_device_segmented_reduce or is_device_segmented_sort or is_device_select:
            inside = re.search(r'\((.*?)\)', name_str).group(1)
            name_str = re.sub(r'\.?\(.*?\)', '', name_str)

            pos = name_str.rfind('>')
            name_str = name_str[:pos] + f",{inside}" + name_str[pos:]

        if is_device_select:
            name_str = re.sub(r"probability:\s*([0-9]*\.?[0-9]+)f", r"probability: \1", name_str)

            subalgo = "flagged" if "device_select_flagged" in name_str else ""
            subalgo = "flagged_if" if "device_select_flagged_if" in name_str else subalgo
            subalgo = "unique" if "device_select_unique" in name_str else subalgo
            subalgo = "unique_by_key" if "device_select_unique_by_key" in name_str else subalgo
            subalgo = "if" if "device_select_if" in name_str else subalgo

            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",subalgo:{subalgo}" + name_str[pos:] 

            if "device_select_unique_by_key" in name_str:
                name_str = name_str.replace("Key data_type", "key_data_type")

        if is_device_segmented_reduce or is_device_segmented_sort:
            name_str = re.sub(r"number_of_segments:~(\d+)\s+segments", r"desired_segments: \1", name_str)

        if is_device_segmented_radix_sort:
            name_str = re.sub(r"segments:~(\d+)\s+segments", r"desired_segments: \1", name_str)

            keys = "device_segmented_radix_sort_keys" in name_str
            subalgo = "sort_keys" if keys else "sort_pairs"

            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",subalgo:{subalgo}" + name_str[pos:] 

        if is_device_segmented_sort:
            keys = "device_segmented_sort_keys" in name_str
            subalgo = "sort_keys" if keys else "sort_pairs"

            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",subalgo:{subalgo}" + name_str[pos:] 

        if is_device_run_length_encode:
            nontrivial = "run_length_encode_non_trivial_runs" in name_str
            subalgo = "non_trivial_runs" if nontrivial else "encode"

            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",subalgo:{subalgo}" + name_str[pos:] 

            name_str = name_str.replace("run_length_encode_non_trivial_runs", "device_run_length_encode")

        if is_device_spmv:
            name_str = re.sub(r'e-(\d)f', r'e-0\1f', name_str)

        if is_device_partition:
            # The subalgo is included in the name, so properly put "subalgo: [subalgo]" before the last >
            subalgo = None
            if "flagged" in name_str:
                subalgo = "flagged"
            elif "predicate" in name_str:
                subalgo = "predicate"
            elif "three_way" in name_str:
                subalgo = "three_way"

            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",subalgo:{subalgo}" + name_str[pos:] 

        if is_device_reduce or is_device_segmented_reduce:
            name_str = name_str.replace("hipcub::ArgMin", "argmin")
            name_str = name_str.replace("argMin", "argmin")

        if is_device_radix_sort:
            descending = "descending" in name_str

            subalgo = "sort_keys"
            if "value_data_type" in name_str:
                subalgo = "sort_pairs"

            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",descending:{descending}, subalgo:{subalgo}" + name_str[pos:] 

        if is_device_merge:
            if "value_data_type" in name_str:
                name_str = name_str.replace("<", "<subalgo: merge_pairs, ", 1)
            else:
                name_str = name_str.replace("<", "<subalgo: merge_keys, ", 1)  

        if is_device_merge_sort:
            if "value_data_type" in name_str:
                name_str = name_str.replace("<", "<subalgo: sort_pairs, ", 1)
            else:
                name_str = name_str.replace("<", "<subalgo: sort_keys, ", 1)

        if is_device_scan:
            exclusive = False
            if "device_exclusive_scan" in name_str:
                exclusive = True

            subalgo = "scan"
            if "by_key" in name_str:
                subalgo = "scan_by_key"

            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",exclusive:{exclusive}, subalgo:{subalgo}" + name_str[pos:] 

        if is_device_memory and "device_memory_memcpy" in name_str:
            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",subalgo: memcpy" + name_str[pos:] 

        name_str = re.sub(r'^[^<]*<', '', name_str)
        name_str = name_str.removesuffix("/manual_time")
        name_str = re.sub(r"/iterations:\d+$", "", name_str)
        name_str = re.sub(r'>$','', name_str)

        name_str = name_str.replace("unsigned int", "u32")
        name_str = name_str.replace("uint8_t", "u8")
        name_str = name_str.replace("int8_t", "i8")
        name_str = name_str.replace("unsigned short", "u16")
        name_str = name_str.replace("uint16_t", "u16")
        name_str = name_str.replace("int16_t", "i16")
        name_str = name_str.replace("uint32_t", "u32")
        name_str = name_str.replace("int32_t", "i32")
        name_str = name_str.replace("uint64_t", "u64")
        name_str = name_str.replace("std::int64_t", "i64")
        name_str = name_str.replace("int64_t", "i64")
        name_str = name_str.replace("unsigned long long", "u64")
        name_str = name_str.replace("custom_int_t", "custom<i32>")
        name_str = name_str.replace("custom_int_double", "custom<i32,f64>")
        name_str = name_str.replace("long long", "i64")
        name_str = re.sub(r'\bint\b', 'i32', name_str)
        name_str = re.sub(r'\b__half\b', 'f16', name_str)
        name_str = re.sub(r'\bshort\b', 'i16', name_str)
        name_str = re.sub(r'\bfloat\b', 'f32', name_str)
        name_str = re.sub(r'\bdouble\b', 'f64', name_str)
        name_str = name_str.replace("sub_algorithm_name", "subalgo")
        name_str = name_str.replace(">.", ",")

        name_str = name_str.replace("Datatype", "data_type")

        name_str = "{" + name_str + "}"

        if is_device_for:       
            # f32 and f64 are duplicated in the old gbench code
            if not device_for_f32_found and "f32" in name_str:
                device_for_f32_found = True
                continue

            if not device_for_f64_found and "f64" in name_str:
                device_for_f64_found = True
                continue

        if is_device_memory:
            # Substitute size: megabytes<i32>(x) with the computed value of x * 1024 * 1024
            name_str = re.sub(r'megabytes<(?:i32|int)>\((\d+)\)', lambda m: str(int(m.group(1)) * 1024 * 1024), name_str)
            name_str = name_str.replace("operation", "kernel_op")
            name_str = name_str.replace("method", "subalgo")

        name_str = name_str.replace("custom_double2", "custom<f64,f64>")
        name_str = name_str.replace("custom_float2", "custom<f32,f32>")
        name_str = name_str.replace("custom_char_double", "custom<i8,f64>")
        name_str = name_str.replace("custom_double_char", "custom<f64,i8>")

        name_json = fix_to_json(name_str)

        if is_warp_merge_sort:
            name_json["segmented"] = warp_merge_is_segmented
            name_json["pairs"] = warp_merge_is_pairs
            del name_json["subalgo"]

        if is_warp_reduce:
            name_json["segmented"] = warp_reduce_is_segmented
            del name_json["subalgo"]

        # In some cases method_name is just the actual algorithm name, and in others, it's a part of the subalgorithm name
        if "method_name" in name_json:
            if is_block_scan:
                # Merge subalgo and method_name into one
                name_json["subalgo"] = name_json.pop("method_name") + "(" + name_json["subalgo"] + ")"
            else:    
                name_json.pop("method_name")

        if is_block_shuffle:
            # If the subalgo is either "offset" or "rotate", then we have to include a dummy value for the graph matchmaking
            if name_json["subalgo"] == "offset" or name_json["subalgo"] == "rotate":
                name_json["items_per_thread"] = 1

        name = parse_benchmark_name(json.dumps(name_json))

        # Fix boolean capitalization
        name = re.sub(r"\bTrue\b", "true", name)
        name = re.sub(r"\bFalse\b", "false", name)

        seen_specializations.add(bench["name"])

        bytes_per_second = bench["bytes_per_second"]
        gib_per_second = bytes_per_second / (1024.0 * 1024.0 * 1024.0)

        result: Dict[str, Any] = {
            "index": idx,
            "name": name,
            "bytes_per_second": bytes_per_second,
            "gib_per_second": gib_per_second,
            "items_per_second": bench["items_per_second"],
            "noise_timeout": 1 if (bench.get("cv", 0) * 100) > noise_threshold else 0,
            "noise_percent": bench.get("cv", 0) * 100,
        }
        results.append(result)

    return results

def convert_rocthrust_json(
    data: Dict[str, Any], noise_threshold: float
) -> List[Dict[str, Any]]:
    """Load rocThrust Google Benchmark JSON and convert to primbench format.

    rocThrust encodes benchmark parameters directly as a JSON object in the
    benchmark name field, e.g.:
      {"algo":"adjacent_difference","subalgo":"basic","input_type":"int8_t",
       "elements":"1 << 16"}/min_time:0.400/manual_time

    Type names use rocThrust conventions (float32_t, float64_t,
    bench_utils::large_data, etc.) that are mapped to primbench conventions
    (f32, f64, large_data, etc.).  The noise metric is 'gpu_noise' (a
    fraction), rather than the 'cv' used by rocPRIM and hipCUB.
    """
    TYPE_MAP: Dict[str, str] = {
        "bench_utils::large_data": "large_data",
        "double": "f64",
        "float": "f32",
        "float32_t": "f32",
        "float64_t": "f64",
        "int128_t": "i128",
        "int16_t": "i16",
        "int32_t": "i32",
        "int64_t": "i64",
        "int8_t": "i8",
        "uint16_t": "u16",
        "uint32_t": "u32",
        "uint64_t": "u64",
        "uint8_t": "u8",
    }

    TYPE_FIELDS = {
        "input_type",
        "key_type",
        "output_type",
        "value_type",
    }

    results: List[Dict[str, Any]] = []

    for idx, bench in enumerate(data["benchmarks"]):
        # Skip benchmarks that failed at runtime (e.g. hipErrorOutOfMemory)
        if bench.get("error_occurred"):
            continue

        raw_name = bench["name"]

        # Strip benchmark timing suffixes, e.g. "/min_time:0.400/manual_time"
        raw_name = re.sub(r"/min_time:[0-9.]+/manual_time$", "", raw_name)
        raw_name = raw_name.removesuffix("/manual_time")
        raw_name = re.sub(r"/iterations:\d+$", "", raw_name)

        # The name is a JSON object encoding all benchmark parameters
        raw_name = re.sub(r'""([^"]+)""', r'"\1"', raw_name)
        params: Dict[str, Any] = json.loads(raw_name)

        # Apply type name transformations to fields that carry type names
        for field in TYPE_FIELDS:
            if field in params and isinstance(params[field], str):
                params[field] = TYPE_MAP.get(params[field], params[field])

        # Sort alphabetically (recursive, consistent with other converters)
        params = sort_dict_alphabetically(params)

        # Build primbench name; exclude 'algo' because it is encoded in the
        # CSV filename (mirrors the rocPRIM convention of excluding 'algo'/'lvl')
        blacklist = {"algo"}
        parts: List[str] = []
        for key, value in params.items():
            if key in blacklist:
                continue
            parts.append(f"{key}: {serialize(value)}")
        name = ", ".join(parts)

        bytes_per_second = bench["bytes_per_second"]
        gib_per_second = bytes_per_second / (1024.0 * 1024.0 * 1024.0)

        # rocThrust reports noise as 'gpu_noise' (a fraction 0-1); treat
        # absent / null values as zero noise
        gpu_noise: float = bench.get("gpu_noise") or 0.0
        noise_percent = gpu_noise * 100

        result: Dict[str, Any] = {
            "index": idx,
            "name": name,
            "bytes_per_second": bytes_per_second,
            "gib_per_second": gib_per_second,
            "items_per_second": bench["items_per_second"],
            "noise_timeout": 1 if noise_percent > noise_threshold else 0,
            "noise_percent": noise_percent,
        }
        results.append(result)

    return results


def write_csv_output(results: List[Dict[str, Any]], output_file: Path) -> None:
    """Write results to primbench CSV format."""
    # Sort results alphabetically by name
    results = sorted(results, key=lambda x: x["name"])

    # Re-index after sorting
    for idx, result in enumerate(results):
        result["index"] = idx

    fieldnames = [
        "index",
        "name",
        "bytes_per_second",
        "gib_per_second",
        "items_per_second",
        "noise_timeout",
        "noise_percent",
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            # Format floating point values with C++17 std::ofstream double precision
            row["bytes_per_second"] = f"{row['bytes_per_second']:.5e}"
            row["gib_per_second"] = f"{row['gib_per_second']:g}"
            row["items_per_second"] = f"{row['items_per_second']:.5e}"
            row["noise_percent"] = f"{row['noise_percent']:.6f}"
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Google Benchmark JSON to primbench CSV format"
    )
    parser.add_argument(
        "--project", choices=["rocprim", "rocrand", "hipcub", "rocthrust"], required=True, help="Project name"
    )
    parser.add_argument(
        "--noise-threshold-percentage",
        type=float,
        required=True,
        help="The noise threshold percentage, past which benchmark specializations "
        "are considered to be too noisy",
    )
    parser.add_argument(
        "input_dir", type=Path, help="Directory containing Google Benchmark JSON files"
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for primbench CSV files"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each JSON file in input directory
    for json_path in args.input_dir.glob("*.json"):
        print(f"Converting {json_path.name}...")
        with open(json_path, "r") as f:
            data = json.load(f)

        if args.project == "rocprim":
            results = convert_rocprim_json(data, args.noise_threshold_percentage)
        elif args.project == "rocrand":
            results = convert_rocrand_json(data)
        elif args.project == "hipcub":
            results = convert_hipcub_json(data, args.noise_threshold_percentage)
        elif args.project == "rocthrust":
            results = convert_rocthrust_json(data, args.noise_threshold_percentage)
        else:
            raise ValueError(f"Missing convert function for {args.project}")

        # Output file has same stem as input, but with .csv extension
        output_file = args.output_dir / f"{json_path.stem}.csv"
        write_csv_output(results, output_file)
        print(f"Converted {json_path.name} -> {output_file.name}")


if __name__ == "__main__":
    main()